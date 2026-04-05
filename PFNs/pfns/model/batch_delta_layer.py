from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn


class BatchDeltaLayer(nn.Module):
    """Multi-head full-support fast-weight solver layer.

    This layer fits a head-wise ridge-regularized linear fast model on the full
    support set and applies it to all tokens. It is intended as a non-causal
    upper layer for TabPFN-style support/query learning.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_state: int,
        num_solver_steps: int = 1,
        support_target_mode: Literal["label_only", "hidden_plus_label"] = "hidden_plus_label",
        target_bilinear_rank: int = 0,
        fast_weight_rank: int = 0,
        base_fast_weight_context_rank: int = 0,
        incontext_opt_rank: int = 0,
        incontext_opt_steps: int = 0,
        incontext_opt_lr: float = 1e-2,
        incontext_opt_weight_decay: float = 0.0,
        ridge_lambda_init: float = 1e-2,
        learnable_ridge_lambda: bool = True,
        qk_l2_normalize: bool = True,
        layer_norm_eps: float = 1e-5,
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if d_state <= 0:
            raise ValueError("d_state must be > 0.")
        if num_solver_steps <= 0:
            raise ValueError("num_solver_steps must be > 0.")
        if support_target_mode not in {"label_only", "hidden_plus_label"}:
            raise ValueError(
                "support_target_mode must be 'label_only' or 'hidden_plus_label'."
            )
        if target_bilinear_rank < 0:
            raise ValueError("target_bilinear_rank must be >= 0.")
        if fast_weight_rank < 0:
            raise ValueError("fast_weight_rank must be >= 0.")
        if fast_weight_rank > d_state:
            raise ValueError("fast_weight_rank must be <= d_state.")
        if base_fast_weight_context_rank < 0:
            raise ValueError("base_fast_weight_context_rank must be >= 0.")
        if incontext_opt_rank < 0:
            raise ValueError("incontext_opt_rank must be >= 0.")
        if incontext_opt_steps < 0:
            raise ValueError("incontext_opt_steps must be >= 0.")
        if incontext_opt_lr <= 0:
            raise ValueError("incontext_opt_lr must be > 0.")
        if incontext_opt_weight_decay < 0:
            raise ValueError("incontext_opt_weight_decay must be >= 0.")
        if ridge_lambda_init <= 0:
            raise ValueError("ridge_lambda_init must be > 0.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_state = d_state
        self.num_solver_steps = num_solver_steps
        self.support_target_mode = support_target_mode
        self.target_bilinear_rank = target_bilinear_rank
        self.fast_weight_rank = fast_weight_rank
        self.base_fast_weight_context_rank = base_fast_weight_context_rank
        self.incontext_opt_rank = incontext_opt_rank
        self.incontext_opt_steps = incontext_opt_steps
        self.incontext_opt_lr = incontext_opt_lr
        self.incontext_opt_weight_decay = incontext_opt_weight_decay
        self.qk_l2_normalize = qk_l2_normalize

        self.input_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.target_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.label_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.q_proj = nn.Linear(d_model, n_heads * d_state, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_state, bias=False)
        self.target_proj = nn.Linear(d_model, d_model, bias=False)
        self.label_proj = nn.Linear(d_model, d_model, bias=False)
        if target_bilinear_rank > 0:
            self.hidden_bilinear_proj = nn.Linear(
                d_model,
                target_bilinear_rank,
                bias=False,
            )
            self.label_bilinear_proj = nn.Linear(
                d_model,
                target_bilinear_rank,
                bias=False,
            )
            self.bilinear_out_proj = nn.Linear(
                target_bilinear_rank,
                d_model,
                bias=False,
            )
        else:
            self.hidden_bilinear_proj = None
            self.label_bilinear_proj = None
            self.bilinear_out_proj = None
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_net = nn.Linear(d_model, d_model)
        self.base_fast_weight = nn.Parameter(
            torch.empty(n_heads, self.d_head, d_state)
        )
        if fast_weight_rank > 0:
            self.fast_weight_input_basis = nn.Parameter(
                torch.empty(n_heads, d_state, fast_weight_rank)
            )
        else:
            self.fast_weight_input_basis = None
        if base_fast_weight_context_rank > 0:
            self.base_fast_weight_context_left = nn.Parameter(
                torch.empty(n_heads, self.d_head, base_fast_weight_context_rank)
            )
            self.base_fast_weight_context_right = nn.Parameter(
                torch.empty(n_heads, d_state, base_fast_weight_context_rank)
            )
            self.base_fast_weight_context_proj = nn.Linear(
                d_model,
                n_heads * base_fast_weight_context_rank,
            )
        else:
            self.base_fast_weight_context_left = None
            self.base_fast_weight_context_right = None
            self.base_fast_weight_context_proj = None
        if incontext_opt_rank > 0:
            self.incontext_opt_left = nn.Parameter(
                torch.empty(n_heads, self.d_head, incontext_opt_rank)
            )
            self.incontext_opt_right = nn.Parameter(
                torch.empty(n_heads, d_state, incontext_opt_rank)
            )
        else:
            self.incontext_opt_left = None
            self.incontext_opt_right = None
        self.output_residual_scale = nn.Parameter(
            torch.full((d_model,), residual_scale_init)
        )

        if learnable_ridge_lambda:
            init_unconstrained = math.log(math.expm1(ridge_lambda_init))
            self._ridge_lambda_unconstrained = nn.Parameter(
                torch.tensor(init_unconstrained, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "_ridge_lambda_unconstrained",
                torch.tensor(math.log(math.expm1(ridge_lambda_init)), dtype=torch.float32),
                persistent=True,
            )

        self.item_attention_mask_mode = None
        self._cached_w_stars: list[torch.Tensor] | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.target_proj.weight)
        nn.init.xavier_uniform_(self.label_proj.weight)
        if self.hidden_bilinear_proj is not None:
            nn.init.xavier_uniform_(self.hidden_bilinear_proj.weight)
            nn.init.xavier_uniform_(self.label_bilinear_proj.weight)
            nn.init.xavier_uniform_(self.bilinear_out_proj.weight, gain=0.1)
        if self.fast_weight_input_basis is not None:
            nn.init.xavier_uniform_(self.fast_weight_input_basis)
        if self.base_fast_weight_context_left is not None:
            nn.init.xavier_uniform_(self.base_fast_weight_context_left)
            nn.init.xavier_uniform_(self.base_fast_weight_context_right)
            nn.init.zeros_(self.base_fast_weight_context_proj.weight)
            nn.init.zeros_(self.base_fast_weight_context_proj.bias)
        if self.incontext_opt_left is not None:
            nn.init.xavier_uniform_(self.incontext_opt_left)
            nn.init.xavier_uniform_(self.incontext_opt_right)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.normal_(self.base_fast_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.constant_(self.gate_net.bias, -2.0)

    def empty_trainset_representation_cache(self) -> None:
        self._cached_w_stars = None

    def get_trainset_representation_cache(
        self,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor | None]:
        return {"w_stars": self._cached_w_stars}

    def load_trainset_representation_cache(
        self,
        cache_state: dict[str, list[torch.Tensor] | torch.Tensor | None],
    ) -> None:
        cached = cache_state.get("w_stars")
        if cached is None:
            legacy = cache_state.get("w_star")
            self._cached_w_stars = [legacy] if torch.is_tensor(legacy) else None
            return
        if torch.is_tensor(cached):
            self._cached_w_stars = [cached]
            return
        self._cached_w_stars = list(cached)

    def _ridge_lambda(self, device: torch.device) -> torch.Tensor:
        unconstrained = self._ridge_lambda_unconstrained.to(device=device)
        return torch.nn.functional.softplus(unconstrained) + 1e-8

    @staticmethod
    def _solve_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return dtype

    def _reshape_state_proj(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.d_state)

    def _reshape_head_output(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.d_head)

    def _build_support_targets(
        self,
        target_h: torch.Tensor,
        support_y: torch.Tensor,
    ) -> torch.Tensor:
        normalized_labels = self.label_norm(support_y)
        support_targets = self.label_proj(normalized_labels)
        if self.support_target_mode == "hidden_plus_label":
            support_targets = support_targets + self.target_proj(target_h)
        if (
            self.support_target_mode == "hidden_plus_label"
            and self.hidden_bilinear_proj is not None
        ):
            bilinear_hidden = self.hidden_bilinear_proj(target_h)
            bilinear_label = self.label_bilinear_proj(normalized_labels)
            bilinear = self.bilinear_out_proj(
                F.silu(bilinear_hidden * bilinear_label)
            )
            support_targets = support_targets + bilinear
        return self._reshape_head_output(support_targets)

    def _solver_key_features(
        self,
        k_support: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.fast_weight_input_basis is None:
            return k_support, None
        key_basis = self.fast_weight_input_basis.to(device=device, dtype=dtype)
        return torch.einsum("bshk,hkr->bshr", k_support, key_basis), key_basis

    def _base_fast_weight_with_context(
        self,
        pooled_support_context: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        base_fast_weight = self.base_fast_weight.to(device=device, dtype=dtype).unsqueeze(0)
        if self.base_fast_weight_context_proj is None:
            return base_fast_weight
        context_scales = self.base_fast_weight_context_proj(
            pooled_support_context
        ).view(
            pooled_support_context.shape[0],
            self.n_heads,
            self.base_fast_weight_context_rank,
        )
        context_scales = context_scales.to(dtype=dtype)
        context_left = self.base_fast_weight_context_left.to(device=device, dtype=dtype)
        context_right = self.base_fast_weight_context_right.to(device=device, dtype=dtype)
        context_delta = torch.einsum(
            "bhr,hdr,hkr->bhdk",
            context_scales,
            context_left,
            context_right,
        )
        return base_fast_weight + context_delta

    def supports_incontext_fast_weight_optimization(self) -> bool:
        return (
            self.incontext_opt_steps > 0
            and self.incontext_opt_left is not None
            and self.incontext_opt_right is not None
        )

    def init_incontext_fast_weight_coefficients(
        self,
        *,
        batch_size: int,
        num_cached_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[nn.Parameter]:
        if not self.supports_incontext_fast_weight_optimization():
            return []
        return [
            nn.Parameter(
                torch.zeros(
                    batch_size,
                    self.n_heads,
                    self.incontext_opt_rank,
                    device=device,
                    dtype=dtype,
                )
            )
            for _ in range(num_cached_steps)
        ]

    def compose_incontext_fast_weight(
        self,
        cached_w_star: torch.Tensor,
        coefficient: torch.Tensor,
    ) -> torch.Tensor:
        if not self.supports_incontext_fast_weight_optimization():
            return cached_w_star
        opt_left = self.incontext_opt_left.to(
            device=cached_w_star.device,
            dtype=cached_w_star.dtype,
        )
        opt_right = self.incontext_opt_right.to(
            device=cached_w_star.device,
            dtype=cached_w_star.dtype,
        )
        delta = torch.einsum(
            "bhr,hdr,hkr->bhdk",
            coefficient,
            opt_left,
            opt_right,
        )
        return cached_w_star + delta

    def _fit_fast_weight(
        self,
        support_h: torch.Tensor,
        support_label_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        solve_dtype = self._solve_dtype(support_h.dtype)
        autocast_context = (
            torch.autocast(device_type=support_h.device.type, enabled=False)
            if support_h.device.type in {"cuda", "cpu"}
            else nullcontext()
        )

        with autocast_context:
            support_h_fp = support_h.to(dtype=solve_dtype)
            support_y_fp = support_label_embeddings.to(dtype=solve_dtype)

            normed_h = self.input_norm(support_h_fp)
            target_h = self.target_norm(support_h_fp)

            k_support = self._reshape_state_proj(self.k_proj(normed_h))
            if self.qk_l2_normalize:
                k_support = F.normalize(k_support, p=2, dim=-1, eps=1e-6)
            support_targets = self._build_support_targets(target_h, support_y_fp)
            solver_keys, key_basis = self._solver_key_features(
                k_support,
                dtype=solve_dtype,
                device=support_h.device,
            )
            pooled_support_context = target_h.mean(dim=1)
            base_fast_weight = self._base_fast_weight_with_context(
                pooled_support_context,
                device=support_h.device,
                dtype=solve_dtype,
            )
            base_support = torch.einsum(
                "bshk,bhdk->bshd",
                k_support,
                base_fast_weight,
            )
            residual_targets = support_targets - base_support

            k_support_h = solver_keys.permute(0, 2, 1, 3)
            residual_targets_h = residual_targets.permute(0, 2, 1, 3)

            gram = torch.matmul(k_support_h.transpose(-1, -2), k_support_h)
            rhs = torch.matmul(
                k_support_h.transpose(-1, -2),
                residual_targets_h,
            )
            ridge_lambda = self._ridge_lambda(support_h.device).to(dtype=solve_dtype)
            solver_dim = gram.shape[-1]
            eye = torch.eye(
                solver_dim,
                device=gram.device,
                dtype=gram.dtype,
            ).view(1, 1, solver_dim, solver_dim)
            delta_w_t = torch.linalg.solve(gram + ridge_lambda * eye, rhs)
            if key_basis is None:
                delta_w = delta_w_t.transpose(-1, -2)
            else:
                delta_w = torch.einsum(
                    "bhdr,hkr->bhdk",
                    delta_w_t.transpose(-1, -2),
                    key_basis,
                )
            return base_fast_weight + delta_w

    def _apply_fast_weight(
        self,
        h: torch.Tensor,
        w_star: torch.Tensor,
    ) -> torch.Tensor:
        q = self._reshape_state_proj(self.q_proj(self.input_norm(h))).to(dtype=w_star.dtype)
        if self.qk_l2_normalize:
            q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        fast_out = torch.einsum("bshk,bhdk->bshd", q, w_star).to(dtype=h.dtype)
        merged = fast_out.reshape(h.shape[0], h.shape[1], self.d_model)
        adapted = self.output_residual_scale * self.out_proj(merged)
        gate = torch.sigmoid(self.gate_net(h))
        return h + gate * adapted

    def _apply_solver_steps(
        self,
        h: torch.Tensor,
        *,
        single_eval_pos: int,
        support_label_embeddings: torch.Tensor,
        cache_trainset_representation: bool,
    ) -> torch.Tensor:
        support_y = support_label_embeddings[:, :single_eval_pos].to(dtype=h.dtype)
        current_h = h
        cached_w_stars: list[torch.Tensor] = []
        for _ in range(self.num_solver_steps):
            support_h = current_h[:, :single_eval_pos]
            w_star = self._fit_fast_weight(support_h, support_y)
            if cache_trainset_representation:
                cached_w_stars.append(w_star.detach())
            current_h = self._apply_fast_weight(current_h, w_star)
        if cache_trainset_representation:
            self._cached_w_stars = cached_w_stars
        return current_h

    def _apply_cached_solver_steps(self, h: torch.Tensor) -> torch.Tensor:
        if self._cached_w_stars is None:
            raise ValueError(
                "No cached support fast weights found. Call incontext_fit first."
            )
        current_h = h
        for w_star in self._cached_w_stars:
            current_h = self._apply_fast_weight(current_h, w_star)
        return current_h

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        optional_suffixes = [
            "output_residual_scale",
            "label_norm.weight",
            "label_norm.bias",
            "hidden_bilinear_proj.weight",
            "label_bilinear_proj.weight",
            "bilinear_out_proj.weight",
            "fast_weight_input_basis",
            "base_fast_weight_context_left",
            "base_fast_weight_context_right",
            "base_fast_weight_context_proj.weight",
            "base_fast_weight_context_proj.bias",
            "incontext_opt_left",
            "incontext_opt_right",
        ]
        for suffix in optional_suffixes:
            key = prefix + suffix
            while key in missing_keys:
                missing_keys.remove(key)

    def forward(
        self,
        state: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cache_trainset_representation: bool = False,
        support_label_embeddings: torch.Tensor | None = None,
        rope_pairwise_positions: bool = False,
        **_: Any,
    ) -> torch.Tensor:
        if rope_pairwise_positions:
            raise ValueError("BatchDeltaLayer does not support interleaved x/y pairs.")
        if state.ndim != 4:
            raise ValueError(
                "BatchDeltaLayer expects state of shape (batch, seq, token, d_model)."
            )
        if state.shape[2] != 1:
            raise ValueError(
                "BatchDeltaLayer requires exactly one token per item. "
                "Use attention_between_features=False with one feature group."
            )

        seq_len = state.shape[1]
        if single_eval_pos is None:
            single_eval_pos = 0
        if not 0 <= single_eval_pos <= seq_len:
            raise ValueError(
                f"single_eval_pos must satisfy 0 <= single_eval_pos <= {seq_len}, got {single_eval_pos}."
            )

        h = state[:, :, 0, :]

        if cache_trainset_representation and single_eval_pos == 0:
            return self._apply_cached_solver_steps(h).unsqueeze(2)

        if single_eval_pos == 0:
            return state

        if support_label_embeddings is None:
            raise ValueError(
                "support_label_embeddings must be provided when fitting the BatchDelta solver."
            )
        if support_label_embeddings.ndim != 3:
            raise ValueError(
                "support_label_embeddings must have shape (batch, seq, d_model)."
            )
        if support_label_embeddings.shape[:2] != h.shape[:2]:
            raise ValueError(
                "BatchDeltaLayer requires non-interleaved inputs without prepended style "
                "tokens so label embeddings align with item states."
            )

        return self._apply_solver_steps(
            h,
            single_eval_pos=single_eval_pos,
            support_label_embeddings=support_label_embeddings,
            cache_trainset_representation=cache_trainset_representation,
        ).unsqueeze(2)
