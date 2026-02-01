#  Copyright (c) Prior Labs GmbH 2025.

# TODO: Seems like there's a lot in this file that is over-parametrized for regular
# usage. Could likely just remove it.
from __future__ import annotations

from functools import partial
from typing import Any, ClassVar

import torch

from pfns.model.layer_norm import LayerNorm
from pfns.model.mlp import MLP
from pfns.model.multi_head_attention import MultiHeadAttention
from torch import nn
from torch.nn.modules.transformer import Module, Tensor


class PerFeatureLayer(Module):
    """Transformer encoder layer that processes each feature block separately.

    This layer consists of multi-head attention between features, multi-head
    attention between items, and feedforward neural networks (MLPs).

    It supports various configurations and optimization options.

    """

    __constants__: ClassVar = ["batch_first"]

    def __init__(  # noqa: PLR0913
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        recompute_sublayers: bool = False,
        second_mlp: bool = False,
        layer_norm_with_elementwise_affine: bool = False,
        zero_init: bool = True,
        save_peak_mem_factor: int | None = None,
        attention_between_features: bool = True,
        multiquery_item_attention: bool = False,
        multiquery_item_attention_for_test_set: bool = False,
        attention_init_gain: float = 1.0,
        d_k: int | None = None,
        d_v: int | None = None,
        precomputed_kv: None | torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None,
        item_attention_mask_mode: str | None = None,
    ) -> None:
        """
        Args:
            d_model: The dimensionality of the input and output embeddings.
            nhead: The number of attention heads.
            dim_feedforward:
                The dimensionality of the feedforward network.
                Default is None (2 * d_model).
            activation: The activation function to use in the MLPs.
            layer_norm_eps: The epsilon value for layer normalization.
            device: The device to use for the layer parameters.
            dtype: The data type to use for the layer parameters.
            recompute_sublayers: Whether to recompute attention during backpropagation.
            second_mlp: Whether to include a second MLP in the layer. `self.second_mlp` will be put between the first (between features) and second (between items) attention layers.
            layer_norm_with_elementwise_affine:
                Whether to use elementwise affine parameters in layer normalization.
            zero_init: Whether to initialize the output of the MLPs to zero.
            save_peak_mem_factor:
                The factor to save peak memory, only effective with post-norm.
            attention_between_features: Whether to apply attention between feature blocks.
            multiquery_item_attention: Whether to use multiquery attention for items.
            multiquery_item_attention_for_test_set:
                Whether to use multiquery attention for the test set.
            attention_init_gain: The gain value for initializing attention parameters.
            d_k:
                The dimensionality of the query and key vectors.
                Default is (d_model // nhead).
            d_v:
                The dimensionality of the value vectors. Default is (d_model // nhead).
            precomputed_kv: Precomputed key-value pairs for attention.
            item_attention_mask_mode:
                Optional mask mode applied to attention between items.
                Supported: "test_to_train_only", "causal_train_only".
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert d_model % nhead == 0 or (d_k is not None and d_v is not None)
        if multiquery_item_attention_for_test_set and multiquery_item_attention:
            raise ValueError(
                "Cannot use both multiquery_item_attention_for_test_set"
                "and multiquery_item_attention",
            )

        if d_k is None:
            d_k = d_model // nhead

        if d_v is None:
            d_v = d_model // nhead

        self.self_attn_between_features: MultiHeadAttention | None = None
        if attention_between_features:
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

        if isinstance(precomputed_kv, tuple):
            precomputed_k, precomputed_v = precomputed_kv
            precomputed_kv = None
        else:
            precomputed_k = precomputed_v = None

        self.self_attn_between_items = MultiHeadAttention(
            input_size=d_model,
            output_size=d_model,
            d_k=d_k,
            d_v=d_v,
            nhead=nhead,
            device=device,
            dtype=dtype,
            share_kv_across_n_heads=nhead if multiquery_item_attention else 1,
            initialize_output_to_zero=zero_init,
            recompute=recompute_sublayers,
            precomputed_k=precomputed_k,
            precomputed_v=precomputed_v,
            precomputed_kv=precomputed_kv,
            init_gain=attention_init_gain,
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
                for _ in range(4 if second_mlp else 3)
            ],
        )

        self.second_mlp: MLP | None = None
        if second_mlp:
            assert (
                attention_between_features
            ), "`second_mlp` requires `attention_between_features` to be enabled."
            self.second_mlp = MLP(
                size=d_model,
                hidden_size=dim_feedforward,
                activation=activation,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_sublayers,
            )

        self.recompute_attn = recompute_sublayers
        self.save_peak_mem_factor = save_peak_mem_factor
        self.multiquery_item_attention_for_test_set = (
            multiquery_item_attention_for_test_set
        )
        self.item_attention_mask_mode = item_attention_mask_mode

    def _build_item_attention_mask(
        self,
        *,
        mode: str,
        seq_len_q: int,
        seq_len_kv: int,
        train_len: int,
        device: torch.device,
        dtype: torch.dtype,
        q_offset: int = 0,
        k_offset: int = 0,
    ) -> torch.Tensor:
        if seq_len_q == 0 or seq_len_kv == 0:
            raise ValueError("Sequence length must be non-zero for attention masking.")
        assert train_len > 0, "train_len must be > 0 for item attention masking."
        q_pos = torch.arange(q_offset, q_offset + seq_len_q, device=device)
        k_pos = torch.arange(k_offset, k_offset + seq_len_kv, device=device)
        if mode == "test_to_train_only":
            mask = torch.full(
                (seq_len_q, seq_len_kv),
                float("-inf"),
                device=device,
                dtype=dtype,
            )
            train_q = q_pos < train_len
            if train_q.any():
                mask[train_q] = torch.where(
                    k_pos.unsqueeze(0) == q_pos[train_q].unsqueeze(1),
                    torch.zeros(1, device=device, dtype=dtype),
                    mask[train_q],
                )
            if (~train_q).any():
                mask[~train_q] = torch.where(
                    k_pos.unsqueeze(0) < train_len,
                    torch.zeros(1, device=device, dtype=dtype),
                    mask[~train_q],
                )
        elif mode == "causal_train_only":
            mask = torch.full(
                (seq_len_q, seq_len_kv),
                float("-inf"),
                device=device,
                dtype=dtype,
            )
            train_q = q_pos < train_len
            if train_q.any():
                train_k = k_pos.unsqueeze(0) < train_len
                causal_k = k_pos.unsqueeze(0) <= q_pos[train_q].unsqueeze(1)
                mask[train_q] = torch.where(
                    train_k & causal_k,
                    torch.zeros(1, device=device, dtype=dtype),
                    mask[train_q],
                )
            if (~train_q).any():
                mask[~train_q] = torch.where(
                    k_pos.unsqueeze(0) < train_len,
                    torch.zeros(1, device=device, dtype=dtype),
                    mask[~train_q],
                )
        else:
            raise ValueError(f"Unknown item_attention_mask_mode: {mode}")

        return mask

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("save_peak_mem_factor", None)
        super().__setstate__(state)

    def forward(  # noqa: C901
        self,
        state: Tensor,
        single_eval_pos: int | None = None,
        *,
        cache_trainset_representation: bool = False,
        att_src: Tensor | None = None,
    ) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            state:
                The transformer state passed as input to the layer of shape
                (batch_size, num_items, num_feature_blocks, d_model).
            single_eval_pos:
                The position from which on everything is treated as test
                set.
            cache_trainset_representation:
                Whether to cache the trainset representation.
                If single_eval_pos is set (> 0 and not None), create a cache of the
                trainset KV. This may require a lot of memory. Otherwise, use
                cached KV representations for inference.
            att_src:
                The tensor to attend to from the final layer of the encoder.
                It has a shape of
                (batch_size, num_train_items, num_feature_blocks, d_model).
                This does not work with multiquery_item_attention_for_test_set and
                cache_trainset_representation at this point.

        Returns:
            The transformer state passed through the encoder layer.
        """
        assert (
            len(state.shape) == 4
        ), "src must be of shape (batch_size, num_items, num feature blocks, d_model)"
        if single_eval_pos is None:
            single_eval_pos = 0

        save_peak_mem_factor = self.save_peak_mem_factor
        if cache_trainset_representation and not single_eval_pos:
            assert self.self_attn_between_items.has_cached_kv, "To use the cache, you must first fill it. See the `cache_trainset_representation` argument docstring."
            save_peak_mem_factor = None

        if att_src is not None:
            assert (
                not self.multiquery_item_attention_for_test_set
            ), "Not implemented yet."
            assert not cache_trainset_representation, "Not implemented yet."
            assert not single_eval_pos, (
                "single_eval_pos should not be set, as the train representation"
                " is in att_src"
            )

        if self.self_attn_between_features is None:
            assert not cache_trainset_representation, "`cache_trainset_representation` is not supported without `attention_between_features`. It should be easy to implement, but we didn't need it yet."
            assert state.shape[2] == 1, (
                f"One group architecture expects one feature group, "
                f"but got {state.shape[2]} feature groups."
            )

        def attn_between_features(x: torch.Tensor) -> torch.Tensor:
            assert self.self_attn_between_features is not None
            return self.self_attn_between_features(
                x,
                save_peak_mem_factor=save_peak_mem_factor,
                add_input=True,
                allow_inplace=True,
            )

        def attn_between_items(x: torch.Tensor) -> torch.Tensor:
            # we need to transpose as self attention always treats
            # dim -2 as the sequence dimension
            def build_attention_mask(
                *,
                seq_len_q: int,
                seq_len_kv: int,
                q_offset: int = 0,
                k_offset: int = 0,
            ) -> torch.Tensor | None:
                if self.item_attention_mask_mode is None:
                    return None
                return self._build_item_attention_mask(
                    mode=self.item_attention_mask_mode,
                    seq_len_q=seq_len_q,
                    seq_len_kv=seq_len_kv,
                    train_len=single_eval_pos,
                    device=x.device,
                    dtype=x.dtype,
                    q_offset=q_offset,
                    k_offset=k_offset,
                )

            if self.multiquery_item_attention_for_test_set:
                if single_eval_pos < x.shape[1]:
                    test_len = x.shape[1] - single_eval_pos
                    test_kv_len = single_eval_pos if single_eval_pos else test_len
                    test_attention_mask = build_attention_mask(
                        seq_len_q=test_len,
                        seq_len_kv=test_kv_len,
                        q_offset=single_eval_pos,
                        k_offset=0,
                    )
                    new_x_test = self.self_attn_between_items(
                        x[:, single_eval_pos:].transpose(1, 2),
                        (
                            x[:, :single_eval_pos].transpose(1, 2)
                            if single_eval_pos
                            else None
                        ),
                        save_peak_mem_factor=save_peak_mem_factor,
                        cache_kv=False,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=not single_eval_pos,
                        reuse_first_head_kv=True,
                        attn_mask=test_attention_mask,
                    ).transpose(1, 2)
                else:
                    new_x_test = None

                if single_eval_pos:
                    train_attention_mask = build_attention_mask(
                        seq_len_q=single_eval_pos,
                        seq_len_kv=single_eval_pos,
                        q_offset=0,
                        k_offset=0,
                    )
                    new_x_train = self.self_attn_between_items(
                        x[:, :single_eval_pos].transpose(1, 2),
                        x[:, :single_eval_pos].transpose(1, 2),
                        save_peak_mem_factor=save_peak_mem_factor,
                        cache_kv=cache_trainset_representation,
                        only_cache_first_head_kv=True,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=False,
                        attn_mask=train_attention_mask,
                    ).transpose(1, 2)
                else:
                    new_x_train = None

                return torch.cat(
                    [x_ for x_ in [new_x_train, new_x_test] if x_ is not None],
                    dim=1,
                )

            attention_src_x = None
            if att_src is not None:
                attention_src_x = att_src.transpose(1, 2)
            elif single_eval_pos:
                attention_src_x = x[:, :single_eval_pos].transpose(1, 2)

            seq_len_q = x.shape[1]
            seq_len_kv = (
                attention_src_x.shape[2] if attention_src_x is not None else seq_len_q
            )
            attention_mask = build_attention_mask(
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                q_offset=0,
                k_offset=0,
            )
            return self.self_attn_between_items(
                x.transpose(1, 2),
                attention_src_x,
                save_peak_mem_factor=save_peak_mem_factor,
                cache_kv=cache_trainset_representation and single_eval_pos,
                add_input=True,
                allow_inplace=True,
                use_cached_kv=cache_trainset_representation and not single_eval_pos,
                attn_mask=attention_mask,
            ).transpose(1, 2)

        # the mlp tends to require 8 times more memory at its peak, that is why we use 8 here
        # todo: this depends on the hidden size, though, and should generally be a function of the hidden size
        mlp_save_peak_mem_factor = (
            save_peak_mem_factor * 8 if save_peak_mem_factor is not None else None
        )

        sublayers = []
        if self.self_attn_between_features is not None:
            sublayers.append(attn_between_features)
        else:
            assert state.shape[2] == 1, (
                "If there is no attention between features, the number of feature"
                " blocks must be 1."
            )

        sublayers += [
            attn_between_items,
            partial(
                self.mlp.__call__,
                save_peak_mem_factor=(
                    mlp_save_peak_mem_factor
                    if (
                        mlp_save_peak_mem_factor is not None
                        and state.numel() // state.shape[-1] // mlp_save_peak_mem_factor
                    )
                    > 32
                    else None
                ),
                add_input=True,
                allow_inplace=True,
            ),
        ]

        if self.second_mlp is not None:
            sublayers.insert(
                1,
                partial(
                    self.second_mlp.__call__,
                    save_peak_mem_factor=mlp_save_peak_mem_factor,
                    add_input=True,
                    allow_inplace=True,
                ),
            )

        for sublayer, layer_norm in zip(sublayers, self.layer_norms):
            state = sublayer(state)
            state = layer_norm(
                state,
                allow_inplace=True,
                save_peak_mem_factor=save_peak_mem_factor,
            )

        return state

    def empty_trainset_representation_cache(self) -> None:
        """Empty the trainset representation cache."""
        self.self_attn_between_items.empty_kv_cache()

        # TODO: This could be None but this just ignored that fact here.
        assert self.self_attn_between_features is not None
        self.self_attn_between_features.empty_kv_cache()  # not necessary, just in case
