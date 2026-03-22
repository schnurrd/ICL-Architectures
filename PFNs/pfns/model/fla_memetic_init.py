from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from fla.layers.gla import GatedLinearAttention


# Large positive bias => logsigmoid(bias) ~ 0 => minimal decay / open recurrent gate
_MEMETIC_OPEN_GATE_BIAS = 6.0

# Keep Q and K very strongly correlated at init
_MEMETIC_QK_PERTURB_STD = 1e-3


def _zero_linear_(linear: nn.Linear, *, bias_value: float = 0.0) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.fill_(bias_value)


def _sample_semi_orthogonal(
    shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return a semi orthogonal matrix:
    - if rows <= cols: orthonormal rows
    - if rows > cols: orthonormal columns
    """
    rows, cols = shape
    basis = torch.randn(rows, cols, device=device, dtype=torch.float32)

    if rows <= cols:
        q, _ = torch.linalg.qr(basis.T, mode="reduced")
        return q.T.to(dtype=dtype)

    q, _ = torch.linalg.qr(basis, mode="reduced")
    return q.to(dtype=dtype)


def _repeat_head_blocks(
    weight: torch.Tensor,
    *,
    num_heads: int,
    repeat_factor: int,
    head_dim: int,
) -> torch.Tensor:
    return (
        weight.view(num_heads, head_dim, weight.shape[1])
        .repeat_interleave(repeat_factor, dim=0)
        .reshape(num_heads * repeat_factor * head_dim, weight.shape[1])
    )


def _expand_grouped_weight(
    weight: torch.Tensor,
    *,
    num_heads: int,
    repeat_factor: int,
    head_dim: int,
) -> torch.Tensor:
    if repeat_factor == 1:
        return weight
    return _repeat_head_blocks(
        weight,
        num_heads=num_heads,
        repeat_factor=repeat_factor,
        head_dim=head_dim,
    )


def _set_memetic_query_key_(
    attn: GatedLinearAttention,
    *,
    perturb_std: float = _MEMETIC_QK_PERTURB_STD,
) -> None:
    """
    Initialize Q/K from one shared semi-orthogonal key map.
    For grouped-KV attention, query heads follow the repeated key structure used
    in the GLA forward pass.
    """
    with torch.no_grad():
        k_weight = _sample_semi_orthogonal(
            attn.k_proj.weight.shape,
            device=attn.k_proj.weight.device,
            dtype=attn.k_proj.weight.dtype,
        )
        q_weight = _expand_grouped_weight(
            k_weight.clone(),
            num_heads=attn.num_kv_heads,
            repeat_factor=attn.num_kv_groups,
            head_dim=attn.head_k_dim,
        )

        if perturb_std > 0.0:
            q_weight = F.normalize(q_weight + perturb_std * torch.randn_like(q_weight), dim=-1)
            k_weight = F.normalize(k_weight + perturb_std * torch.randn_like(k_weight), dim=-1)

        attn.q_proj.weight.copy_(q_weight)
        attn.k_proj.weight.copy_(k_weight)

        if attn.q_proj.bias is not None:
            attn.q_proj.bias.zero_()
        if attn.k_proj.bias is not None:
            attn.k_proj.bias.zero_()


def _set_memetic_value_output_(attn: GatedLinearAttention) -> None:
    """
    Initialize V/O so o_proj reconstructs the repeated grouped-V map as well as
    the parameterization allows. For the standard case this reduces to transpose.
    """
    with torch.no_grad():
        v_weight = _sample_semi_orthogonal(
            attn.v_proj.weight.shape,
            device=attn.v_proj.weight.device,
            dtype=attn.v_proj.weight.dtype,
        )
        expanded_v_weight = _expand_grouped_weight(
            v_weight,
            num_heads=attn.num_kv_heads,
            repeat_factor=attn.num_kv_groups,
            head_dim=attn.head_v_dim,
        )
        if attn.num_kv_groups == 1 and v_weight.shape[0] >= v_weight.shape[1]:
            o_weight = v_weight.T
        else:
            o_weight = torch.linalg.pinv(expanded_v_weight.to(dtype=torch.float32)).to(
                dtype=attn.o_proj.weight.dtype
            )

        attn.v_proj.weight.copy_(v_weight)
        attn.o_proj.weight.copy_(o_weight)

        if attn.v_proj.bias is not None:
            attn.v_proj.bias.zero_()
        if attn.o_proj.bias is not None:
            attn.o_proj.bias.zero_()


def _set_final_constant_gate_(module: nn.Module, *, final_bias_value: float) -> None:
    """
    Zero only the final linear layer so the gate output starts as a constant.
    This is sufficient for the current GLA gate MLP, which has no hidden activation.
    """
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linears:
        raise ValueError("Expected at least one nn.Linear inside gate module.")

    _zero_linear_(linears[-1], bias_value=final_bias_value)


def _apply_memetic_gla_init(
    attn: GatedLinearAttention,
    *,
    qk_perturb_std: float = _MEMETIC_QK_PERTURB_STD,
    allow_short_conv: bool = False,
) -> None:
    if getattr(attn, "use_short_conv", False) and not allow_short_conv:
        raise ValueError(
            "Memetic GLA init assumes use_short_conv=False. "
            "Set allow_short_conv=True only if you explicitly want to apply it anyway."
        )

    _set_memetic_query_key_(attn, perturb_std=qk_perturb_std)
    _set_memetic_value_output_(attn)
    _set_final_constant_gate_(attn.gk_proj, final_bias_value=_MEMETIC_OPEN_GATE_BIAS)


def apply_memetic_fla_init(
    model: nn.Module,
    *,
    layer_indices: Iterable[int] | None = None,
    qk_perturb_std: float = _MEMETIC_QK_PERTURB_STD,
    allow_short_conv: bool = False,
) -> None:
    """
    Apply memetic init to selected GatedLinearAttention layers.

    layer_indices:
        None      -> all GLA layers
        [i, j, k] -> only those GLA layers in traversal order
    """
    gla_layers = [module for module in model.modules() if isinstance(module, GatedLinearAttention)]
    if not gla_layers:
        raise ValueError(
            f"Expected at least one {GatedLinearAttention.__name__} layer in "
            f"{type(model).__name__}, but found none."
        )

    selected = None if layer_indices is None else set(layer_indices)
    applied_any = False

    for gla_idx, module in enumerate(gla_layers):
        if selected is None or gla_idx in selected:
            _apply_memetic_gla_init(
                module,
                qk_perturb_std=qk_perturb_std,
                allow_short_conv=allow_short_conv,
            )
            applied_any = True

    if not applied_any:
        raise ValueError(
            f"No GatedLinearAttention layer matched layer_indices={sorted(selected)}."
        )
