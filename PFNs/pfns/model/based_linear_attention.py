from __future__ import annotations

import torch
from torch import nn

from pfns.model.linear_attention import LinearAttention
from pfns.model.rebased_feature_map import BasedFeatureMap, RebasedFeatureMap


class BasedLinearAttention(LinearAttention):
    """LinearAttention variant that swaps in based/rebased feature maps."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        *,
        dim_mlp_hidden: int | None = None,
        feature_dim: int | None = None,
        dense: bool = False,
        feature_map: str = "rebased",
        use_gamma: bool = True,
        use_beta: bool = True,
        normalize: bool = True,
        causal: bool = False,
        causal_train_only: bool = False,
        causal_chunk_size: int | None = None,
        normalize_q_sum: bool = False,
        normalize_k_sum: bool = False,
        qk_norm: bool | str | None = None,
        use_k_sum_normalization: bool = True,
        use_query_scale: bool = False,
        use_attention_norm: bool = True,
        use_output_norm: bool = True,
        use_mlp_norm: bool = True,
        norm_type: str = "rmsnorm",
        fuse_swiglu: bool = False,
        state_renormalization: str | None = None,
        learnable_state_renorm_scale: bool = True,
        state_renormalization_target_norm: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        resolved_mlp_hidden_dim = (
            mlp_hidden_dim if mlp_hidden_dim is not None else dim_mlp_hidden
        )
        if resolved_mlp_hidden_dim is None:
            raise ValueError(
                "BasedLinearAttention requires `mlp_hidden_dim` or "
                "`dim_mlp_hidden`."
            )

        self.feature_map_name = feature_map.strip().lower().replace("-", "_")
        self.rebased_dense = bool(dense)
        self.rebased_use_gamma = bool(use_gamma)
        self.rebased_use_beta = bool(use_beta)
        self.rebased_normalize = bool(normalize)

        per_head_feature_dim = (
            int(feature_dim) if feature_dim is not None else d_model // num_heads
        )
        key_dim = num_heads * per_head_feature_dim
        if key_dim % d_model != 0:
            expand_k = key_dim / d_model
        else:
            expand_k = key_dim // d_model

        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            expand_k=expand_k,
            mlp_hidden_dim=resolved_mlp_hidden_dim,
            use_mlp_norm=use_mlp_norm,
            feature_map=self.feature_map_name,
            causal=causal,
            causal_train_only=causal_train_only,
            causal_chunk_size=causal_chunk_size,
            normalize_q_sum=normalize_q_sum,
            normalize_k_sum=normalize_k_sum,
            qk_norm=qk_norm,
            use_k_sum_normalization=use_k_sum_normalization,
            use_attention_norm=use_attention_norm,
            use_output_norm=use_output_norm,
            norm_type=norm_type,
            fuse_swiglu=fuse_swiglu,
            state_renormalization=state_renormalization,
            learnable_state_renorm_scale=learnable_state_renorm_scale,
            state_renormalization_target_norm=state_renormalization_target_norm,
            eps=eps,
        )
        if self.key_dim != key_dim:
            raise ValueError(
                "Computed key dimension does not match requested based "
                f"feature_dim. Expected {key_dim}, got {self.key_dim}."
            )
        self.feature_dim = self.head_k_dim
        self.use_query_scale = use_query_scale

    def _build_feature_maps(
        self,
        feature_map: str,
        head_k_dim: int,
    ) -> tuple[nn.Module, nn.Module]:
        if feature_map == "rebased":
            return (
                RebasedFeatureMap(
                    head_dim=head_k_dim,
                    use_gamma=self.rebased_use_gamma,
                    use_beta=self.rebased_use_beta,
                    normalize=self.rebased_normalize,
                    dense=self.rebased_dense,
                ),
                RebasedFeatureMap(
                    head_dim=head_k_dim,
                    use_gamma=self.rebased_use_gamma,
                    use_beta=self.rebased_use_beta,
                    normalize=self.rebased_normalize,
                    dense=self.rebased_dense,
                ),
            )
        if feature_map == "based":
            return (
                BasedFeatureMap(dense=self.rebased_dense),
                BasedFeatureMap(dense=self.rebased_dense),
            )
        raise ValueError(
            f"Unsupported feature_map: {feature_map!r}. "
            "Expected one of: 'rebased', 'based'."
        )

    def _apply_feature_map(
        self,
        x: torch.Tensor,
        feature_map: nn.Module,
    ) -> torch.Tensor:
        return self._apply_qk_norm_to_tensor(feature_map(x))
