"""Bidirectional wrappers for FLA backbones."""
from __future__ import annotations

import copy
import typing as tp
from dataclasses import dataclass

import torch
from torch import nn

from pfns.model.fla_cache_utils import copy_cache, repeat_cache, shallow_copy

BIDIRECTIONAL_FLA_SEQUENCE_MODES = {"Comb_ST"}
BIDIRECTIONAL_STATE_FUSIONS = {
    "linear_output_two_cache",
    "mean_output_two_cache",
    "mean_output_mean_cache",
}


def _uses_mean_hidden_fusion(state_fusion: str) -> bool:
    return state_fusion in {"mean_output_two_cache", "mean_output_mean_cache"}


def _uses_fused_prediction_cache(state_fusion: str) -> bool:
    return state_fusion in {"mean_output_mean_cache"}


def _uses_linear_output_fusion(state_fusion: str) -> bool:
    return state_fusion in {"linear_output_two_cache"}


def _get_fla_layers(fla_model: nn.Module) -> nn.ModuleList:
    layers = getattr(fla_model, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("FLA model does not expose layers as an nn.ModuleList.")
    return layers


@dataclass
class BidirectionalFLACache:
    forward_cache: tp.Any | None
    backward_cache: tp.Any | None

    def __fla_copy_cache__(self) -> "BidirectionalFLACache":
        return BidirectionalFLACache(
            forward_cache=copy_cache(self.forward_cache),
            backward_cache=copy_cache(self.backward_cache),
        )

    def __fla_repeat_cache__(self, repeat: int) -> "BidirectionalFLACache":
        return BidirectionalFLACache(
            forward_cache=repeat_cache(self.forward_cache, repeat),
            backward_cache=repeat_cache(self.backward_cache, repeat),
        )


@dataclass
class FusedBidirectionalFLACache:
    cache: tp.Any | None
    state_fusion: str = "mean_output_mean_cache"

    def __fla_copy_cache__(self) -> "FusedBidirectionalFLACache":
        return FusedBidirectionalFLACache(
            cache=copy_cache(self.cache),
            state_fusion=self.state_fusion,
        )

    def __fla_repeat_cache__(self, repeat: int) -> "FusedBidirectionalFLACache":
        return FusedBidirectionalFLACache(
            cache=repeat_cache(self.cache, repeat),
            state_fusion=self.state_fusion,
        )


def validate_bidirectional_cache(cache_params: tp.Any) -> None:
    if not isinstance(cache_params, (BidirectionalFLACache, FusedBidirectionalFLACache)):
        raise TypeError(
            "BidirectionalFLABackbone requires a BidirectionalFLACache or "
            "FusedBidirectionalFLACache for prediction."
        )


def _fuse_prediction_value(
    forward_value: tp.Any,
    backward_value: tp.Any,
) -> tp.Any:
    if forward_value is backward_value:
        return forward_value
    if forward_value is None or backward_value is None:
        assert forward_value is None and backward_value is None, (
            "Forward and backward cache values must either both be present or both be None."
        )
        return None
    if torch.is_tensor(forward_value):
        assert torch.is_tensor(backward_value) and forward_value.shape == backward_value.shape, (
            "Forward and backward cache tensors must have matching shapes for state fusion."
        )
        return (forward_value + backward_value) / 2
    if isinstance(forward_value, (tuple, list)):
        assert type(forward_value) is type(backward_value) and len(forward_value) == len(backward_value), (
            "Forward and backward cache sequences must have matching structure for state fusion."
        )
        fused_items = [
            _fuse_prediction_value(f_item, b_item)
            for f_item, b_item in zip(forward_value, backward_value, strict=True)
        ]
        return tuple(fused_items) if isinstance(forward_value, tuple) else fused_items
    if isinstance(forward_value, dict):
        assert isinstance(backward_value, dict) and forward_value.keys() == backward_value.keys(), (
            "Forward and backward cache dicts must have matching keys for state fusion."
        )
        return {
            key: _fuse_prediction_value(forward_value[key], backward_value[key])
            for key in forward_value
        }
    if hasattr(forward_value, "__dict__"):
        assert hasattr(backward_value, "__dict__")
        assert forward_value.__dict__.keys() == backward_value.__dict__.keys(), (
            "Forward and backward cache objects must have matching attributes for state fusion."
        )
        fused_value = shallow_copy(forward_value)
        for attr_name in forward_value.__dict__:
            setattr(
                fused_value,
                attr_name,
                _fuse_prediction_value(
                    getattr(forward_value, attr_name),
                    getattr(backward_value, attr_name),
                ),
            )
        return fused_value
    if forward_value == backward_value:
        return forward_value
    raise AssertionError(
        "Unsupported cache value type for state fusion. "
        "Expected tensor/None/list/tuple/dict leaves, got "
        f"{type(forward_value)!r}"
    )


def fuse_bidirectional_cache(
    cache_params: BidirectionalFLACache,
    *,
    state_fusion: str,
) -> tp.Any:
    if not _uses_fused_prediction_cache(state_fusion):
        raise ValueError(f"Unsupported state_fusion mode: {state_fusion!r}")
    return _fuse_prediction_value(
        cache_params.forward_cache,
        cache_params.backward_cache,
    )


def prepare_bidirectional_cache(
    cache_params: BidirectionalFLACache | FusedBidirectionalFLACache,
    *,
    chunk_len: int,
    use_custom_recurrent: bool,
    supports_custom_recurrent: bool,
) -> tp.Any:
    def maybe_repeat(cache_value: tp.Any) -> tp.Any:
        if use_custom_recurrent and supports_custom_recurrent:
            return cache_value
        return repeat_cache(cache_value, chunk_len)

    if isinstance(cache_params, FusedBidirectionalFLACache):
        return maybe_repeat(cache_params.cache)
    return (
        maybe_repeat(cache_params.forward_cache),
        maybe_repeat(cache_params.backward_cache),
    )


def run_bidirectional_layers(
    hidden: torch.Tensor,
    *,
    layers: tp.Iterable[nn.Module],
    chunk_cache: tp.Any,
) -> torch.Tensor:
    for layer in layers:
        if not isinstance(layer, BidirectionalFLALayer):
            raise TypeError(
                "Expected BidirectionalFLALayer in bidirectional FLA backbone."
            )
        if isinstance(chunk_cache, tuple):
            forward_cache, backward_cache = chunk_cache
            forward_output = layer.forward_layer(
                hidden,
                use_cache=False,
                output_attentions=False,
                past_key_values=forward_cache,
            )
            backward_output = layer.backward_layer(
                hidden,
                use_cache=False,
                output_attentions=False,
                past_key_values=backward_cache,
            )
            hidden = layer._fuse_hidden_states(
                layer._extract_hidden_states(forward_output),
                layer._extract_hidden_states(backward_output),
            )
        else:
            fused_output = layer.forward_layer(
                hidden,
                use_cache=False,
                output_attentions=False,
                past_key_values=chunk_cache,
            )
            hidden = layer._fuse_single_hidden_state(
                layer._extract_hidden_states(fused_output)
            )
    return hidden


class BidirectionalFLALayer(nn.Module):
    """Wrap an FLA layer with forward/reverse passes fused back to one state.

    Note: this wrapper is heuristic. It fuses full forward/backward layer outputs
    and does not implement a theory-backed correction for possible self-term
    double counting in bidirectional linear-attention-style models (see Lions paper). 
    A principled fix would likely need model-specific changes inside each FLA layer/
    kernel, rather than a generic post-hoc wrapper correction.
    """

    def __init__(
        self,
        layer: nn.Module,
        *,
        hidden_size: int,
        bidirectional_share_weights: bool = True,
        state_fusion: str = "mean_output_mean_cache",
    ) -> None:
        super().__init__()
        self.bidirectional_share_weights = bool(bidirectional_share_weights)
        self.state_fusion = state_fusion
        self.forward_layer = layer
        self.backward_layer = (
            layer
            if self.bidirectional_share_weights
            else copy.deepcopy(layer)
        )
        self.fusion_out = (
            nn.Linear(hidden_size * 2, hidden_size)
            if _uses_linear_output_fusion(self.state_fusion)
            else None
        )

    def _prepare_branch_kwargs(
        self,
        kwargs: dict[str, tp.Any],
        *,
        reverse: bool,
    ) -> dict[str, tp.Any]:
        branch_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key != "past_key_values"
        }
        branch_kwargs["output_attentions"] = False
        if reverse:
            for key in ("attention_mask", "cache_position"):
                value = branch_kwargs.get(key)
                if torch.is_tensor(value) and value.ndim >= 1:
                    branch_kwargs[key] = value.flip(-1)
        return branch_kwargs

    @staticmethod
    def _cache_from_kwargs(kwargs: dict[str, tp.Any]) -> tp.Any | None:
        return kwargs.get("past_key_values")

    @classmethod
    def _split_cache_for_branches(
        cls,
        cache_value: tp.Any | None,
    ) -> tuple[tp.Any | None, tp.Any | None]:
        if isinstance(cache_value, BidirectionalFLACache):
            return cache_value.forward_cache, cache_value.backward_cache
        return copy_cache(cache_value), copy_cache(cache_value)

    @staticmethod
    def _extract_hidden_states(output: tp.Any) -> torch.Tensor:
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return output[0]
        last_hidden_state = getattr(output, "last_hidden_state", None)
        if torch.is_tensor(last_hidden_state):
            return last_hidden_state
        raise TypeError(
            "Bidirectional FLA expects layer outputs with a hidden-state tensor."
        )

    @staticmethod
    def _extract_cache(output: tp.Any) -> tp.Any | None:
        if isinstance(output, tuple):
            return output[2] if len(output) >= 3 else None
        if not hasattr(output, "past_key_values"):
            raise TypeError(
                "Bidirectional FLA expects layer outputs with past_key_values."
            )
        return output.past_key_values

    @staticmethod
    def _rebuild_output_like(
        reference_output: tp.Any,
        hidden_states: torch.Tensor,
        *,
        cache_value: tp.Any | None = None,
        override_cache: bool = False,
    ) -> tp.Any:
        if isinstance(reference_output, tuple):
            output_items = list(reference_output)
            if not output_items:
                raise TypeError(
                    "Bidirectional FLA expects non-empty tuple outputs when tuple outputs are used."
                )
            output_items[0] = hidden_states
            if override_cache and len(output_items) >= 3:
                output_items[2] = cache_value
            return tuple(output_items)
        if not hasattr(reference_output, "__dict__"):
            raise TypeError(
                "Bidirectional FLA expects layer outputs with object attributes."
            )
        rebuilt_output = copy.copy(reference_output)
        rebuilt_output.last_hidden_state = hidden_states
        if override_cache:
            rebuilt_output.past_key_values = cache_value
        return rebuilt_output

    def _fuse_hidden_states(
        self,
        forward_hidden: torch.Tensor,
        backward_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if _uses_mean_hidden_fusion(self.state_fusion):
            return (forward_hidden + backward_hidden) / 2
        assert _uses_linear_output_fusion(self.state_fusion), (
            f"Unsupported state_fusion mode: {self.state_fusion!r}"
        )
        assert self.fusion_out is not None
        fusion_input = torch.cat([forward_hidden, backward_hidden], dim=-1)
        return self.fusion_out(fusion_input)

    def _fuse_single_hidden_state(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._fuse_hidden_states(hidden, hidden)

    def forward(self, hidden_states: torch.Tensor, **kwargs: tp.Any) -> tp.Any:
        use_cache = bool(kwargs.get("use_cache", False))
        forward_kwargs = self._prepare_branch_kwargs(kwargs, reverse=False)
        reversed_hidden_states = hidden_states.flip(1)
        backward_kwargs = self._prepare_branch_kwargs(kwargs, reverse=True)

        forward_cache: tp.Any | None = None
        backward_cache: tp.Any | None = None
        if use_cache:
            cache_value = self._cache_from_kwargs(kwargs)
            forward_cache, backward_cache = self._split_cache_for_branches(cache_value)
            forward_kwargs["past_key_values"] = forward_cache
            backward_kwargs["past_key_values"] = backward_cache

        forward_output = self.forward_layer(hidden_states, **forward_kwargs)
        backward_output = self.backward_layer(reversed_hidden_states, **backward_kwargs)
        fused_hidden = self._fuse_hidden_states(
            self._extract_hidden_states(forward_output),
            self._extract_hidden_states(backward_output).flip(1),
        )
        if use_cache:
            cache_output = BidirectionalFLACache(
                forward_cache=self._extract_cache(forward_output),
                backward_cache=self._extract_cache(backward_output),
            )
            return self._rebuild_output_like(
                forward_output,
                fused_hidden,
                cache_value=cache_output,
                override_cache=True,
            )
        return self._rebuild_output_like(forward_output, fused_hidden)


def _make_fla_model_bidirectional(
    fla_model: nn.Module,
    *,
    hidden_size: int,
    bidirectional_share_weights: bool = True,
    state_fusion: str = "mean_output_mean_cache",
) -> nn.Module:
    layers = _get_fla_layers(fla_model)
    fla_model.layers = nn.ModuleList(
        [
            BidirectionalFLALayer(
                layer,
                hidden_size=hidden_size,
                bidirectional_share_weights=bidirectional_share_weights,
                state_fusion=state_fusion,
            )
            for layer in layers
        ]
    )
    return fla_model
