from __future__ import annotations

import typing as tp

import torch


def shallow_copy(obj: tp.Any) -> tp.Any:
    obj_copy = obj.__class__.__new__(obj.__class__)
    obj_copy.__dict__.update(obj.__dict__)
    return obj_copy


def repeat_state(state: dict[str, tp.Any], repeat: int, *, dim: int) -> dict[str, tp.Any]:
    def repeat_value(value: tp.Any) -> tp.Any:
        if torch.is_tensor(value):
            if repeat == 1:
                return value.clone()
            return value.repeat_interleave(repeat, dim=dim)
        if isinstance(value, tuple):
            return tuple(repeat_value(item) for item in value)
        return value

    return {key: repeat_value(value) for key, value in state.items()}


def copy_state(state: dict[str, tp.Any]) -> dict[str, tp.Any]:
    def copy_value(value: tp.Any) -> tp.Any:
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, tuple):
            return tuple(copy_value(item) for item in value)
        return value

    return {key: copy_value(value) for key, value in state.items()}


def repeat_cache(cache_params: tp.Any, repeat: int) -> tp.Any:
    if cache_params is None:
        return None
    if torch.is_tensor(cache_params):
        if repeat == 1:
            return cache_params.clone()
        return cache_params.repeat_interleave(repeat, dim=0)
    if hasattr(cache_params, "conv_states") and hasattr(cache_params, "ssm_states"):
        cache_params_copy = shallow_copy(cache_params)
        if repeat == 1:
            cache_params_copy.conv_states = cache_params.conv_states.clone()
            cache_params_copy.ssm_states = cache_params.ssm_states.clone()
        else:
            cache_params_copy.conv_states = cache_params.conv_states.repeat_interleave(repeat, dim=1)
            cache_params_copy.ssm_states = cache_params.ssm_states.repeat_interleave(repeat, dim=1)
        return cache_params_copy
    if hasattr(cache_params, "layers"):
        cache_params_copy = shallow_copy(cache_params)
        new_layers = []
        for layer in cache_params.layers:
            layer_copy = shallow_copy(layer)
            state = getattr(layer, "state", None)
            if isinstance(state, dict):
                layer_copy.state = repeat_state(state, repeat, dim=0)
            else:
                raise ValueError("Unsupported layer state structure for repetition.")
            new_layers.append(layer_copy)
        cache_params_copy.layers = new_layers
        return cache_params_copy
    raise ValueError("Unsupported cache_params structure for repetition.")


def copy_cache(cache_params: tp.Any) -> tp.Any:
    if cache_params is None:
        return None
    if torch.is_tensor(cache_params):
        return cache_params.clone()
    if hasattr(cache_params, "conv_states") and hasattr(cache_params, "ssm_states"):
        cache_params_copy = shallow_copy(cache_params)
        cache_params_copy.conv_states = cache_params.conv_states.clone()
        cache_params_copy.ssm_states = cache_params.ssm_states.clone()
        return cache_params_copy
    if hasattr(cache_params, "layers"):
        cache_params_copy = shallow_copy(cache_params)
        new_layers = []
        for layer in cache_params.layers:
            layer_copy = shallow_copy(layer)
            state = getattr(layer, "state", None)
            if isinstance(state, dict):
                layer_copy.state = copy_state(state)
            else:
                raise ValueError("Unsupported layer state structure for copy.")
            new_layers.append(layer_copy)
        cache_params_copy.layers = new_layers
        return cache_params_copy
    if hasattr(cache_params, "states"):
        cache_params_copy = shallow_copy(cache_params)
        cache_params_copy.states = [
            copy_state(state) if isinstance(state, dict) else state
            for state in cache_params.states
        ]
        return cache_params_copy
    return shallow_copy(cache_params)
