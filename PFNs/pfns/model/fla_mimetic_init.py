from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.gla import GatedLinearAttention


# Large positive bias => logsigmoid(bias) ~ 0 => minimal decay / open recurrent gate
_MIMETIC_OPEN_GATE_BIAS = 6.0
_MIMETIC_A_LOG = -8.0
_MIMETIC_DT_BIAS = float(torch.log(torch.expm1(torch.tensor(1.0, dtype=torch.float32))))

def _zero_linear_(linear: nn.Linear, *, bias_value: float = 0.0) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.fill_(bias_value)


def _set_block_identity_(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        diag = min(linear.weight.shape)
        linear.weight[:diag, :diag] = torch.eye(
            diag,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        if linear.bias is not None:
            linear.bias.zero_()


def _set_encoder_decoder_identity_(encoder: nn.Linear, decoder: nn.Linear) -> None:
    with torch.no_grad():
        encoder.weight.zero_()
        decoder.weight.zero_()
        if encoder.bias is not None:
            encoder.bias.zero_()
        if decoder.bias is not None:
            decoder.bias.zero_()

        enc_out, enc_in = encoder.weight.shape
        dec_out, dec_in = decoder.weight.shape
        if enc_out != dec_in or enc_in != dec_out:
            raise ValueError("Encoder/decoder shapes must compose back to the input size.")
        if enc_out < enc_in:
            raise ValueError(
                "Cannot build identity composition when encoder out_features < in_features "
                f"(got {enc_out} < {enc_in})."
            )

        counts = torch.zeros(enc_in, device=encoder.weight.device, dtype=torch.int64)
        for row in range(enc_out):
            col = row % enc_in
            encoder.weight[row, col] = 1.0
            counts[col] += 1

        for col in range(dec_out):
            matching_rows = torch.arange(
                col,
                dec_in,
                dec_out,
                device=decoder.weight.device,
            )
            decoder.weight[col, matching_rows] = 1.0 / float(counts[col].item())


def _zero_gate_(module: nn.Module, *, final_bias_value: float = 0.0) -> None:
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    for idx, linear in enumerate(linears):
        _zero_linear_(
            linear,
            bias_value=final_bias_value if idx == len(linears) - 1 else 0.0,
        )


def _set_causal_identity_short_conv_(conv: nn.Conv1d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[:, 0, -1].fill_(1.0)
        if conv.bias is not None:
            conv.bias.zero_()


def _maybe_perturb_query_key_(
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    *,
    perturb_std: float,
) -> None:
    if perturb_std <= 0.0:
        return
    with torch.no_grad():
        q_proj.weight.add_(torch.randn_like(q_proj.weight) * perturb_std)
        k_proj.weight.add_(torch.randn_like(k_proj.weight) * perturb_std)


def _require_attrs(module: nn.Module, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise ValueError(
            f"{type(module).__name__} is missing required attributes for mimetic init: {missing}"
        )


def apply_mimetic_fla_init(
    model: nn.Module,
    *,
    layer_indices: Iterable[int] | None = None,
    qk_perturb_std: float = 0.0,
    allow_short_conv: bool = True,
) -> None:
    """
    Apply mimetic init to selected supported FLA layers.

    Args:
        model: Model containing supported FLA attention modules.
        layer_indices: Supported-layer indices to initialize. None initializes all.
        qk_perturb_std: Optional Gaussian stddev added to q/k weights after identity init.
        allow_short_conv: Whether to apply causal-identity init to short-conv paths.
    """
    if qk_perturb_std < 0.0:
        raise ValueError(f"qk_perturb_std must be >= 0, got {qk_perturb_std}.")

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

    selected: set[int]
    if layer_indices is None:
        selected = set(range(len(supported_layers)))
    elif isinstance(layer_indices, str):
        raise ValueError(
            "layer_indices must be an iterable of ints or None; "
            f"got {layer_indices!r}."
        )
    else:
        selected = {int(idx) for idx in layer_indices}
    applied_any = False

    for layer_idx, module in enumerate(supported_layers):
        if layer_idx not in selected:
            continue

        if isinstance(module, GatedLinearAttention):
            _require_attrs(module, ("q_proj", "k_proj", "v_proj", "o_proj", "gk_proj"))
            _set_block_identity_(module.q_proj)
            _set_block_identity_(module.k_proj)
            _maybe_perturb_query_key_(
                module.q_proj,
                module.k_proj,
                perturb_std=qk_perturb_std,
            )
            _set_encoder_decoder_identity_(module.v_proj, module.o_proj)
            _zero_gate_(module.gk_proj, final_bias_value=_MIMETIC_OPEN_GATE_BIAS)
        else:
            _require_attrs(module, ("q_proj", "k_proj", "v_proj", "o_proj", "a_proj", "A_log", "dt_bias"))
            if allow_short_conv and getattr(module, "use_short_conv", False):
                _require_attrs(module, ("q_conv1d", "k_conv1d", "v_conv1d"))
                for conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
                    _set_causal_identity_short_conv_(conv)
            _set_block_identity_(module.q_proj)
            _set_block_identity_(module.k_proj)
            _maybe_perturb_query_key_(
                module.q_proj,
                module.k_proj,
                perturb_std=qk_perturb_std,
            )
            _set_encoder_decoder_identity_(module.v_proj, module.o_proj)
            _zero_gate_(module.a_proj)
            with torch.no_grad():
                module.A_log.fill_(_MIMETIC_A_LOG)
                module.dt_bias.fill_(_MIMETIC_DT_BIAS)
        applied_any = True

    if not applied_any:
        raise ValueError(
            f"No supported mimetic-init layer matched layer_indices={sorted(selected)}."
        )
