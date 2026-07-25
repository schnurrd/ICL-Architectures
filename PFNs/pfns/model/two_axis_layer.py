from __future__ import annotations

import typing as tp

import torch
from torch import nn

from pfns.model.layer_norm import LayerNorm
from pfns.model.mlp import MLP
from pfns.model.multi_head_attention import MultiHeadAttention
from pfns.model.row_attention import build_row_attention


class TwoAxisLayer(nn.Module):
    """TabPFN-v2-style per-feature layer with a recurrent row (item) mixer."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        recompute_sublayers: bool = False,
        layer_norm_with_elementwise_affine: bool = False,
        zero_init: bool = True,
        save_peak_mem_factor: int | None = None,
        attention_init_gain: float = 1.0,
        d_k: int | None = None,
        d_v: int | None = None,
        row_attention: str = "deltanet",
        row_attention_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert d_model % nhead == 0 or (d_k is not None and d_v is not None)

        if d_k is None:
            d_k = d_model // nhead
        if d_v is None:
            d_v = d_model // nhead

        self.self_attn_between_features = MultiHeadAttention(
            input_size=d_model,
            output_size=d_model,
            d_k=d_k,
            d_v=d_v,
            nhead=nhead,
            device=device,
            dtype=dtype,
            initialize_output_to_zero=zero_init,
            recompute=recompute_sublayers,
            init_gain=attention_init_gain,
        )

        self.item_attn = build_row_attention(
            row_attention,
            d_model=d_model,
            num_heads=nhead,
            zero_init=zero_init,
            **(row_attention_kwargs or {}),
        )

        if dim_feedforward is None:
            dim_feedforward = 2 * d_model

        self.mlp = MLP(
            size=d_model,
            hidden_size=dim_feedforward,
            activation=activation,
            device=device,
            dtype=dtype,
            initialize_output_to_zero=zero_init,
            recompute=recompute_sublayers,
        )

        self.layer_norms = nn.ModuleList(
            [
                LayerNorm(
                    d_model,  # type: ignore
                    layer_norm_eps,
                    elementwise_affine=layer_norm_with_elementwise_affine,
                    **factory_kwargs,
                )
                for _ in range(3)
            ],
        )

        self.save_peak_mem_factor = save_peak_mem_factor
        # The MLP peaks at ~8x the attention sublayers' memory, matching
        # `PerFeatureLayer.forward`'s mlp_save_peak_mem_factor.
        self.mlp_save_peak_mem_factor = (
            save_peak_mem_factor * 8 if save_peak_mem_factor is not None else None
        )

    def _attn_between_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn_between_features(
            x,
            save_peak_mem_factor=self.save_peak_mem_factor,
            add_input=True,
            allow_inplace=True,
        )

    def _apply_norm(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return self.layer_norms[idx](
            x,
            allow_inplace=True,
            save_peak_mem_factor=self.save_peak_mem_factor,
        )

    def _apply_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(
            x,
            save_peak_mem_factor=self.mlp_save_peak_mem_factor,
            add_input=True,
            allow_inplace=True,
        )

    def _run(
        self,
        x: torch.Tensor,
        row_attention: tp.Callable[[torch.Tensor], tp.Any],
    ) -> tp.Any:
        """column attention -> row attention -> MLP, each post-normed.

        `row_attention` returns either the row-mixer output or an
        (output, state) pair; the state is passed straight back to the caller.
        """
        assert x.dim() == 4, (
            "x must be of shape (batch_size, num_items, num_feature_blocks, d_model)"
        )
        x = self._apply_norm(self._attn_between_features(x), 0)

        result = row_attention(x)
        attn_out, state = result if isinstance(result, tuple) else (result, None)
        x = self._apply_norm(x + attn_out, 1)

        out = self._apply_norm(self._apply_mlp(x), 2)
        return out if state is None else (out, state)

    def forward(
        self,
        state: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cache_trainset_representation: bool = False,
        **_: object,
    ) -> torch.Tensor:
        """Pass the input through the encoder layer.

        `single_eval_pos` splits the train context from the test tokens; the
        row mixer keeps them separated in training as well as at inference
        (see `RowAttentionBase.forward`). `cache_trainset_representation` is
        unsupported -- use `incontext_fit`/`incontext_predict` for caching.
        """
        assert not cache_trainset_representation, (
            "TwoAxisLayer does not support cache_trainset_representation "
            "in forward(); use incontext_fit/incontext_predict instead."
        )
        return self._run(
            state, lambda h: self.item_attn(h, single_eval_pos=single_eval_pos)
        )

    def incontext_fit(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._run(x, self.item_attn.incontext_fit)

    def incontext_predict(
        self, x: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._run(x, lambda h: self.item_attn.incontext_predict(h, state))

    def empty_trainset_representation_cache(self) -> None:
        self.item_attn.empty_trainset_representation_cache()
