from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import torch


def _shallow_copy(obj: tp.Any) -> tp.Any:
    out = obj.__class__.__new__(obj.__class__)
    out.__dict__.update(obj.__dict__)
    return out


def _map_nested(value: tp.Any, tensor_fn: tp.Callable[[torch.Tensor], torch.Tensor]) -> tp.Any:
    if torch.is_tensor(value):
        return tensor_fn(value)
    if isinstance(value, tuple):
        return tuple(_map_nested(item, tensor_fn) for item in value)
    if isinstance(value, list):
        return [_map_nested(item, tensor_fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_nested(item, tensor_fn) for key, item in value.items()}
    return value


def _first_tensor(value: tp.Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            if (tensor := _first_tensor(item)) is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            if (tensor := _first_tensor(item)) is not None:
                return tensor
    return None


def _transform_cache(
    cache_params: tp.Any,
    tensor_fn: tp.Callable[[torch.Tensor, int], torch.Tensor],
) -> tp.Any:
    if cache_params is None:
        return None
    if torch.is_tensor(cache_params):
        return tensor_fn(cache_params, 0)
    conv_states = getattr(cache_params, "conv_states", None)
    ssm_states = getattr(cache_params, "ssm_states", None)
    if conv_states is not None and ssm_states is not None:
        cache_params = _shallow_copy(cache_params)
        cache_params.conv_states = tensor_fn(conv_states, 1)
        cache_params.ssm_states = tensor_fn(ssm_states, 1)
        return cache_params
    if hasattr(cache_params, "layers"):
        cache_params = _shallow_copy(cache_params)
        cache_params.layers = [_transform_layer(layer, tensor_fn) for layer in cache_params.layers]
        return cache_params
    return _shallow_copy(cache_params)


def _transform_layer(
    layer: tp.Any,
    tensor_fn: tp.Callable[[torch.Tensor, int], torch.Tensor],
) -> tp.Any:
    layer = _shallow_copy(layer)
    state = getattr(layer, "state", None)
    if isinstance(state, dict):
        layer.state = _map_nested(state, lambda tensor: tensor_fn(tensor, 0))
    return layer


def freeze_cache(cache_params: tp.Any) -> tp.Any:
    return _transform_cache(cache_params, lambda tensor, _: tensor.detach().clone())


def cache_batch_size(cache_params: tp.Any) -> int:
    if cache_params is None:
        raise ValueError("cache_params must not be None.")
    if torch.is_tensor(cache_params):
        return int(cache_params.size(0))
    if hasattr(cache_params, "conv_states") and hasattr(cache_params, "ssm_states"):
        return int(cache_params.conv_states.size(1))
    if hasattr(cache_params, "layers"):
        for layer in cache_params.layers:
            if (tensor := _first_tensor(getattr(layer, "state", None))) is not None:
                return int(tensor.size(0))
        raise ValueError("Unable to infer batch size from FLA cache state.")
    raise ValueError("Unsupported cache_params structure for state passing.")


def select_cache_entries(cache_params: tp.Any, indices: torch.Tensor) -> tp.Any:
    return _transform_cache(
        cache_params,
        lambda tensor, dim: tensor.index_select(dim, indices.to(tensor.device)),
    )


def zero_cache_entries(cache_params: tp.Any, zero_mask: torch.Tensor) -> tp.Any:
    zero_indices = zero_mask.nonzero(as_tuple=False).flatten()
    return _transform_cache(
        cache_params,
        lambda tensor, dim: tensor.clone().index_fill_(dim, zero_indices.to(tensor.device), 0),
    )


def aligned_indices(prev_batch_size: int, batch_size: int, *, device: torch.device) -> torch.Tensor:
    return torch.arange(batch_size, device=device) % prev_batch_size


@dataclass
class FLAStatePassing:
    dropout_prob: float = 0.1
    previous_cache: tp.Any | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be in [0, 1].")

    def reset(self) -> None:
        self.previous_cache = None

    def remember(self, cache_params: tp.Any | None) -> None:
        self.previous_cache = freeze_cache(cache_params)

    def sample_initial_cache(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> tp.Any | None:
        previous_cache = self.previous_cache
        if previous_cache is None or batch_size <= 0:
            return None
        prev_batch_size = cache_batch_size(previous_cache)
        indices = aligned_indices(prev_batch_size, batch_size, device=device)
        cache = select_cache_entries(previous_cache, indices)
        return (
            cache
            if self.dropout_prob <= 0.0
            else zero_cache_entries(cache, torch.rand(batch_size, device=device) < self.dropout_prob)
        )

    def remember_split_cache(
        self,
        train_cache: tp.Any | None,
        test_x: torch.Tensor,
        *,
        run_fla: tp.Callable[..., tuple[torch.Tensor, tp.Any | None]],
        copy_cache: tp.Callable[[tp.Any], tp.Any],
    ) -> None:
        final_cache = train_cache
        if test_x.numel() > 0:
            _, final_cache = run_fla(
                test_x,
                cache_params=copy_cache(final_cache),
                return_cache=True,
            )
        self.remember(final_cache)
