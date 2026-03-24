from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.gla import GatedLinearAttention


# Large positive bias => logsigmoid(bias) ~ 0 => minimal decay / open recurrent gate
_MIMETIC_OPEN_GATE_BIAS = 6.0

# Keep Q and K very strongly correlated at init
_MIMETIC_QK_PERTURB_STD = 1e-3

# Small positive delta-rule decay => recurrent gate starts nearly open
_MIMETIC_DELTA_DECAY = 1e-3

def _zero_linear_(linear: nn.Linear, *, bias_value: float = 0.0) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.fill_(bias_value)


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"Expected positive value for inverse softplus, got {value}.")
    return value + math.log(-math.expm1(-value))


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
    expected_rows = num_heads * head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            "Expected weight.shape[0] == num_heads * head_dim, "
            f"got {weight.shape[0]} != {num_heads} * {head_dim}."
        )
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


def _set_mimetic_query_key_(
    attn: GatedLinearAttention | GatedDeltaNet,
    *,
    perturb_std: float = _MIMETIC_QK_PERTURB_STD,
) -> None:
    """
    Initialize Q/K from one shared semi-orthogonal key map.
    For grouped-KV attention, query heads follow the repeated key structure used
    in the GLA forward pass.
    """
    with torch.no_grad():
        num_key_heads = getattr(attn, "num_kv_heads", getattr(attn, "num_heads"))
        key_repeat_factor = getattr(attn, "num_kv_groups", 1)
        k_weight = _sample_semi_orthogonal(
            attn.k_proj.weight.shape,
            device=attn.k_proj.weight.device,
            dtype=attn.k_proj.weight.dtype,
        )
        q_weight = _expand_grouped_weight(
            k_weight.clone(),
            num_heads=num_key_heads,
            repeat_factor=key_repeat_factor,
            head_dim=attn.head_k_dim,
        )

        if perturb_std > 0.0:
            k_weight = F.normalize(k_weight + perturb_std * torch.randn_like(k_weight), dim=-1)

        attn.q_proj.weight.copy_(q_weight)
        attn.k_proj.weight.copy_(k_weight)

        if attn.q_proj.bias is not None:
            attn.q_proj.bias.zero_()
        if attn.k_proj.bias is not None:
            attn.k_proj.bias.zero_()


def _set_mimetic_value_output_(attn: GatedLinearAttention | GatedDeltaNet) -> None:
    """
    Initialize V/O so o_proj reconstructs the repeated grouped-V map as well as
    the parameterization allows. For the standard case this reduces to transpose.
    """
    with torch.no_grad():
        num_value_heads = getattr(
            attn,
            "num_v_heads",
            getattr(attn, "num_kv_heads", getattr(attn, "num_heads")),
        )
        value_repeat_factor = getattr(attn, "num_kv_groups", 1)
        v_weight = _sample_semi_orthogonal(
            attn.v_proj.weight.shape,
            device=attn.v_proj.weight.device,
            dtype=attn.v_proj.weight.dtype,
        )
        expanded_v_weight = _expand_grouped_weight(
            v_weight,
            num_heads=num_value_heads,
            repeat_factor=value_repeat_factor,
            head_dim=attn.head_v_dim,
        )
        if value_repeat_factor == 1 and v_weight.shape[0] >= v_weight.shape[1]:
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


def _set_causal_identity_short_conv_(conv: nn.Conv1d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[:, 0, -1].fill_(1.0)
        if conv.bias is not None:
            conv.bias.zero_()


def _set_mimetic_gated_deltanet_beta_(attn: GatedDeltaNet) -> None:
    with torch.no_grad():
        q_blocks = attn.q_proj.weight.view(attn.num_heads, attn.head_k_dim, attn.hidden_size)
        beta_rows = q_blocks.abs().mean(dim=1)
        if attn.num_v_heads > attn.num_heads:
            if attn.num_v_heads % attn.num_heads != 0:
                raise ValueError(
                    f"Cannot expand num_heads={attn.num_heads} to num_v_heads={attn.num_v_heads}."
                )
            beta_rows = beta_rows.repeat_interleave(attn.num_v_heads // attn.num_heads, dim=0)
        elif attn.num_heads > attn.num_v_heads:
            if attn.num_heads % attn.num_v_heads != 0:
                raise ValueError(
                    f"Cannot reduce num_heads={attn.num_heads} to num_v_heads={attn.num_v_heads}."
                )
            beta_rows = beta_rows.view(attn.num_v_heads, -1, attn.hidden_size).mean(dim=1)
        beta_rows = F.normalize(beta_rows, dim=-1)
        attn.b_proj.weight.copy_(beta_rows * 4.0)


def _apply_mimetic_gated_deltanet_init(
    attn: GatedDeltaNet,
    *,
    qk_perturb_std: float = _MIMETIC_QK_PERTURB_STD,
) -> None:
    if getattr(attn, "use_short_conv", False):
        for conv in (attn.q_conv1d, attn.k_conv1d, attn.v_conv1d):
            _set_causal_identity_short_conv_(conv)

    _set_mimetic_query_key_(attn, perturb_std=qk_perturb_std)
    _set_mimetic_value_output_(attn)
    with torch.no_grad():
        _zero_linear_(attn.a_proj)
        attn.A_log.zero_()
        attn.dt_bias.fill_(_inverse_softplus(_MIMETIC_DELTA_DECAY))
    _set_mimetic_gated_deltanet_beta_(attn)


def apply_mimetic_fla_init(
    model: nn.Module,
    *,
    layer_indices: Iterable[int] | None = None,
    qk_perturb_std: float = _MIMETIC_QK_PERTURB_STD,
    allow_short_conv: bool = False,
) -> None:
    """
    Apply mimetic init to selected supported FLA layers.

    layer_indices:
        None      -> all supported layers
        [i, j, k] -> only those supported layers in traversal order
    """
    supported_layers = [
        module
        for module in model.modules()
        if isinstance(module, (GatedLinearAttention, GatedDeltaNet))
    ]
    if not supported_layers:
        raise ValueError(
            "Expected at least one supported mimetic-init layer "
            f"({GatedLinearAttention.__name__} or {GatedDeltaNet.__name__}) in "
            f"{type(model).__name__}, but found none."
        )

    selected = None if layer_indices is None else set(layer_indices)
    applied_any = False

    for layer_idx, module in enumerate(supported_layers):
        if selected is not None and layer_idx not in selected:
            continue

        if isinstance(module, GatedLinearAttention):
            if getattr(module, "use_short_conv", False) and not allow_short_conv:
                raise ValueError(
                    "Mimetic GLA init assumes use_short_conv=False. "
                    "Set allow_short_conv=True only if you explicitly want to apply it anyway."
                )
            _set_mimetic_query_key_(module, perturb_std=qk_perturb_std)
            _set_mimetic_value_output_(module)
            _set_final_constant_gate_(module.gk_proj, final_bias_value=_MIMETIC_OPEN_GATE_BIAS)
        else:
            _apply_mimetic_gated_deltanet_init(
                module,
                qk_perturb_std=qk_perturb_std,
            )
        applied_any = True

    if not applied_any:
        raise ValueError(
            f"No supported mimetic-init layer matched layer_indices={sorted(selected)}."
        )
