"""Configuration classes for different model backbones.

This module provides base classes and implementations for configuring
different backbones that can be used within the ModelConfig.
"""
from __future__ import annotations

import os
import typing as tp
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fla.models import GLAConfig, GLAModel
from fla.models import Mamba2Config, Mamba2Model
from fla.models import KDAConfig, KDAModel
from fla.models import DeltaNetConfig, DeltaNetModel
from fla.models import GatedDeltaNetConfig, GatedDeltaNetModel
from fla.models import LinearAttentionConfig, LinearAttentionModel

from pfns import base_config
from pfns.model.fla_patches import (
    _maybe_patch_gla_with_stateless_recurrent,
    _maybe_patch_kda_with_stateless_recurrent,
    _maybe_patch_deltanet_with_stateless_recurrent,
    _maybe_patch_gated_deltanet_with_stateless_recurrent,
    _maybe_patch_mamba2_with_stateless_recurrent,
    _maybe_patch_linear_attn_with_stateless_recurrent,
    _maybe_patch_shortconv_forward_pytorch,
)
from pfns.model.layer import PerFeatureLayer
from pfns.model.linear_attention import LinearAttention
from pfns.model.mode_normalization import (
    CANONICAL_SEQUENCE_MODES,
    resolve_sequence_mode,
)
from pfns.model.rebased_linear_attention import RebasedLinearAttention
from pfns.model.tabular_model import LayerStack
# Registry mapping model types to their config and model classes
FLA_MODEL_REGISTRY = {
    "gla": (GLAConfig, GLAModel),
    "mamba2": (Mamba2Config, Mamba2Model),
    "kda": (KDAConfig, KDAModel),
    "deltanet": (DeltaNetConfig, DeltaNetModel),
    "gated_deltanet": (GatedDeltaNetConfig, GatedDeltaNetModel),
    "linear_attn": (LinearAttentionConfig, LinearAttentionModel),
}
FLA_SEQUENCE_MODES = set(CANONICAL_SEQUENCE_MODES)


def _resolve_fla_sequence_mode(sequence_mode: str) -> str:
    return resolve_sequence_mode(sequence_mode)


class Backbone(nn.Module, ABC):
    """Abstract base class for backbone implementations.
    
    This provides a common interface for different backbone architectures.
    Backbones should implement forward() to process embedded sequences.
    """
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        single_eval_pos: int | None = None,
        half_layers: bool = False,
        cache_trainset_representation: bool = False,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        """Process embedded input sequence.
        
        Args:
            x: Embedded input tensor, shape depends on architecture
               For PFN-style: (batch, seq, num_tokens, embed_dim)
            single_eval_pos: Position marking end of training context
            half_layers: Whether to use only half the layers
            cache_trainset_representation: Whether caching is enabled
            **kwargs: Additional architecture-specific arguments
            
        Returns:
            Processed tensor, typically same shape as input
        """
        pass

    def incontext_fit(
        self,
        x: torch.Tensor,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, tp.Any]:
        """Process training context and return cached state."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement incontext_fit."
        )

    def incontext_predict(
        self,
        x: torch.Tensor,
        cached_state: tp.Any,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        """Process test tokens using cached state."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement incontext_predict."
        )


@dataclass(frozen=True)
class BackboneConfig(base_config.BaseConfig, ABC):
    """Base class for backbone configurations.
    
    A backbone is the core neural network architecture that processes
    the embedded inputs. Different backbones can be swapped without
    changing encoders or decoders.
    """

    @abstractmethod
    def create_backbone(
        self,
        ninp: int,
        attention_between_features: bool,
        **kwargs: tp.Any,
    ) -> Backbone:
        """Create the backbone module.
        
        Args:
            ninp: Input/embedding dimension
            attention_between_features: Whether to apply attention between features
                (relevant for PFN-style architectures, may be ignored by others)
            **kwargs: Additional arguments passed to the backbone
            
        Returns:
            The initialized backbone module
        """
        pass
    
    @property
    def nhid(self) -> int:
        """Hidden dimension for decoder initialization."""
        return 512


@dataclass(frozen=True)
class TransformerBackboneConfig(BackboneConfig):
    """Configuration for a Transformer-based backbone.
    
    This is the standard transformer encoder stack used in the original
    TabPFN architecture.
    """
    
    nhid: int = 200
    nlayers: int = 6
    nhead: int = 2
    activation: tp.Literal["gelu", "relu"] = "gelu"
    recompute_layer: bool = False
    min_num_layers_layer_dropout: tp.Optional[int] = None
    layer_kwargs: tp.Dict[str, base_config.BaseTypes] | None = None

    def create_backbone(
        self,
        ninp: int,
        attention_between_features: bool,
        **kwargs: tp.Any,
    ) -> Backbone:
        """Create the transformer backbone.
        
        Args:
            ninp: Input/embedding dimension (emsize)
            attention_between_features: Whether to apply attention between features
            **kwargs: Additional arguments (currently unused)
            
        Returns:
            A TransformerBackbone wrapping LayerStack
        """
        
        def layer_creator():
            return PerFeatureLayer(
                d_model=ninp,
                nhead=self.nhead,
                dim_feedforward=self.nhid,
                activation=self.activation,
                zero_init=True,
                precomputed_kv=None,
                attention_between_features=attention_between_features,
                **(self.layer_kwargs or {}),
            )

        layer_stack = LayerStack(
            layer_creator=layer_creator,
            num_layers=self.nlayers,
            recompute_each_layer=self.recompute_layer,
            min_num_layers_layer_dropout=self.min_num_layers_layer_dropout,
        )
        
        return TransformerBackbone(layer_stack)


class TransformerBackbone(Backbone):
    """Wrapper for LayerStack to conform to Backbone interface."""
    
    def __init__(self, layer_stack: nn.Module):
        super().__init__()
        self.layer_stack = layer_stack

    @property
    def layers(self):
        return self.layer_stack.layers

    @staticmethod
    def _assert_item_mask_disabled_for_incontext(layers: tp.Iterable[nn.Module]) -> None:
        for i, layer in enumerate(layers):
            mask_mode = getattr(layer, "item_attention_mask_mode", None)
            assert mask_mode is None, (
                "item_attention_mask_mode must be None when using "
                "incontext_fit/incontext_predict. "
                f"Found {mask_mode!r} on layer index {i}."
            )
        
    def forward(
        self,
        x: torch.Tensor,
        *,
        single_eval_pos: int | None = None,
        half_layers: bool = False,
        cache_trainset_representation: bool = False,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        return self.layer_stack(
            x,
            single_eval_pos=single_eval_pos,
            half_layers=half_layers,
            cache_trainset_representation=cache_trainset_representation,
            **kwargs,
        )

    @staticmethod
    def _extract_item_attention_cache(layers: tp.Iterable[nn.Module]) -> dict[str, tp.Any]:
        cache_layers: list[dict[str, torch.Tensor | None]] = []
        for layer in layers:
            attn = getattr(layer, "self_attn_between_items", None)
            if attn is None:
                cache_layers.append({"k": None, "v": None, "kv": None})
                continue
            cache_layers.append(
                {
                    "k": getattr(attn, "_k_cache", None), # None
                    "v": getattr(attn, "_v_cache", None), # None
                    "kv": getattr(attn, "_kv_cache", None), # only one used
                }
            )
        return {"layers": cache_layers}

    @staticmethod
    def _clear_attention_cache(layers: tp.Iterable[nn.Module]) -> None:
        for layer in layers:
            if hasattr(layer, "empty_trainset_representation_cache"):
                layer.empty_trainset_representation_cache()
            
    @staticmethod
    def _take_item_attention_cache(layers: tp.Iterable[nn.Module]) -> dict[str, tp.Any]:
        # Transfer cache ownership from layers into a plain Python state object.
        cache_state = TransformerBackbone._extract_item_attention_cache(layers)
        TransformerBackbone._clear_attention_cache(layers)
        return cache_state

    @staticmethod
    def _load_item_attention_cache(
        layers: tp.Iterable[nn.Module],
        cache_state: dict[str, tp.Any],
    ) -> None:
        layer_states = cache_state.get("layers", [])
        for layer, state in zip(layers, layer_states):
            attn = getattr(layer, "self_attn_between_items", None)
            if attn is None:
                continue
            attn._k_cache = state.get("k")
            attn._v_cache = state.get("v")
            attn._kv_cache = state.get("kv")

    def incontext_fit(
        self,
        x: torch.Tensor,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, tp.Any]:
        self._assert_item_mask_disabled_for_incontext(self.layers)
        train_len = x.shape[1]
        for layer in self.layers:
            if hasattr(layer, "empty_trainset_representation_cache"):
                layer.empty_trainset_representation_cache()
        out = self.forward(
            x,
            single_eval_pos=train_len,
            half_layers=False,
            cache_trainset_representation=True,
            **kwargs,
        )
        cache_state = self._take_item_attention_cache(self.layers)
        return out, cache_state

    def incontext_predict(
        self,
        x: torch.Tensor,
        cached_state: tp.Any,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        self._assert_item_mask_disabled_for_incontext(self.layers)
        self._load_item_attention_cache(self.layers, cached_state)
        output= self.forward(
            x,
            single_eval_pos=0,
            half_layers=False,
            cache_trainset_representation=True,
            **kwargs,
        )
        self._clear_attention_cache(self.layers) # keep cache ownership in external state
        return output


@dataclass(frozen=True)
class FLABackboneConfig(BackboneConfig):
    """Configuration for Flash Linear Attention (FLA) based backbones."""

    model_type: tp.Literal[
        "gla", "mamba2", "kda", "deltanet", "gated_deltanet", "linear_attn"
    ] = "linear_attn"
    config_kwargs: dict[str, tp.Any] | None = None
    sequence_mode: tp.Literal["Comb_ST", "Int_ST", "Comb_MT", "Int_MT"] = "Comb_ST"
    cache_chunk_size: int | None = None
    # Backward-compatibility only: older checkpoints may serialize this field.
    # It is ignored by FLABackbone and has no effect on training/inference.
    deltanet_state_reg_weight: float | None = None

    def __post_init__(self):
        if self.model_type not in FLA_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. Available: {list(FLA_MODEL_REGISTRY)}"
            )
        object.__setattr__(
            self,
            "sequence_mode",
            _resolve_fla_sequence_mode(self.sequence_mode),
        )

    def create_backbone(self, ninp: int, attention_between_features: bool, **kwargs: tp.Any) -> "Backbone":
        ConfigClass, ModelClass = FLA_MODEL_REGISTRY[self.model_type]

        assert attention_between_features is False, (
            "FLA backbones currently do not support attention between features"
        )

        if self.config_kwargs is None:
            raise ValueError("FLABackboneConfig requires config_kwargs to build the FLA config.")

        assert ninp is None or ninp == self.config_kwargs.get("hidden_size", ninp), (
            "FLA backbone ninp must match config_kwargs hidden_size"
        )

        config = ConfigClass(**self.config_kwargs)
        fla_model = ModelClass(config)

        return FLABackbone(
            fla_model=fla_model,
            sequence_mode=self.sequence_mode,
            cache_chunk_size=self.cache_chunk_size,
        )


class FLABackbone(Backbone):
    """Wrapper for FLA models to conform to Backbone interface."""

    _CUSTOM_RECURRENT_MODELS: tuple[type[nn.Module], ...] = (
        GLAModel,
        KDAModel,
        DeltaNetModel,
        GatedDeltaNetModel,
        Mamba2Model,
        LinearAttentionModel,
    )

    def __init__(
        self,
        fla_model: nn.Module,
        sequence_mode: str = "Comb_ST",
        cache_chunk_size: int | None = None,
    ):
        super().__init__()
        self.fla = fla_model.model if hasattr(fla_model, "model") else fla_model
        self.sequence_mode = _resolve_fla_sequence_mode(sequence_mode)
        self.cache_chunk_size = cache_chunk_size

    @staticmethod
    def _summarize_cache(cache_params: tp.Any) -> str:
        tensor_count = 0
        total_elems = 0
        total_bytes = 0
        max_tensor = ("", 0, 0)

        def _walk(obj: tp.Any, prefix: str) -> None:
            nonlocal tensor_count, total_elems, total_bytes, max_tensor
            if torch.is_tensor(obj):
                tensor_count += 1
                elems = obj.numel()
                bytes_ = obj.element_size() * elems
                total_elems += elems
                total_bytes += bytes_
                if bytes_ > max_tensor[2]:
                    max_tensor = (prefix, elems, bytes_)
                return
            if isinstance(obj, dict):
                for key, value in obj.items():
                    _walk(value, f"{prefix}.{key}" if prefix else str(key))
                return
            if isinstance(obj, (list, tuple)):
                for idx, value in enumerate(obj):
                    _walk(value, f"{prefix}[{idx}]")
                return
            if hasattr(obj, "__dict__"):
                for key, value in obj.__dict__.items():
                    _walk(value, f"{prefix}.{key}" if prefix else key)

        _walk(cache_params, "cache")
        mb = total_bytes / (1024 ** 2)
        max_name, max_elems, max_bytes = max_tensor
        max_mb = max_bytes / (1024 ** 2)
        return (
            f"cache tensors={tensor_count}, total={mb:.2f}MB, "
            f"largest={max_name} ({max_elems} elems, {max_mb:.2f}MB)"
        )

    @staticmethod
    def _shallow_copy(obj: tp.Any) -> tp.Any:
        obj_copy = obj.__class__.__new__(obj.__class__)
        obj_copy.__dict__.update(obj.__dict__)
        return obj_copy

    @staticmethod
    def _repeat_state(state: dict[str, tp.Any], repeat: int, *, dim: int) -> dict[str, tp.Any]:
        def _repeat_value(value: tp.Any) -> tp.Any:
            if torch.is_tensor(value):
                if repeat == 1:
                    return value.clone()
                return value.repeat_interleave(repeat, dim=dim)
            if isinstance(value, tuple):
                return tuple(_repeat_value(item) for item in value)
            return value

        return {key: _repeat_value(value) for key, value in state.items()}

    @staticmethod
    def _copy_state(state: dict[str, tp.Any]) -> dict[str, tp.Any]:
        def _copy_value(value: tp.Any) -> tp.Any:
            if torch.is_tensor(value):
                return value.clone()
            if isinstance(value, tuple):
                return tuple(_copy_value(item) for item in value)
            return value

        return {key: _copy_value(value) for key, value in state.items()}

    @staticmethod
    def _repeat_cache(cache_params: tp.Any, repeat: int) -> tp.Any:
        if cache_params is None:
            return None
        if torch.is_tensor(cache_params):
            if repeat == 1:
                return cache_params.clone()
            return cache_params.repeat_interleave(repeat, dim=0)
        if hasattr(cache_params, "conv_states") and hasattr(cache_params, "ssm_states"):  # Mamba2 style
            cache_params_copy = FLABackbone._shallow_copy(cache_params)
            if repeat == 1:
                cache_params_copy.conv_states = cache_params.conv_states.clone()
                cache_params_copy.ssm_states = cache_params.ssm_states.clone()
            else:
                cache_params_copy.conv_states = cache_params.conv_states.repeat_interleave(repeat, dim=1)
                cache_params_copy.ssm_states = cache_params.ssm_states.repeat_interleave(repeat, dim=1)
            return cache_params_copy
        if hasattr(cache_params, "layers"):  # GLA, KDA, (Gated) Deltanet style
            cache_params_copy = FLABackbone._shallow_copy(cache_params)
            new_layers = []
            for layer in cache_params.layers:
                layer_copy = FLABackbone._shallow_copy(layer)
                state = getattr(layer, "state", None)
                if isinstance(state, dict):
                    layer_copy.state = FLABackbone._repeat_state(state, repeat, dim=0)
                else:
                    raise ValueError("Unsupported layer state structure for repetition.")
                new_layers.append(layer_copy)
            cache_params_copy.layers = new_layers
            return cache_params_copy
        raise ValueError("Unsupported cache_params structure for repetition.")

    @staticmethod
    def _copy_cache(cache_params: tp.Any) -> tp.Any:
        if cache_params is None:
            return None
        if torch.is_tensor(cache_params):
            return cache_params.clone()
        if hasattr(cache_params, "conv_states") and hasattr(cache_params, "ssm_states"):  # Mamba2 style
            cache_params_copy = FLABackbone._shallow_copy(cache_params)
            cache_params_copy.conv_states = cache_params.conv_states.clone()
            cache_params_copy.ssm_states = cache_params.ssm_states.clone()
            return cache_params_copy
        if hasattr(cache_params, "layers"):  # GLA, KDA, (Gated) Deltanet style
            cache_params_copy = FLABackbone._shallow_copy(cache_params)
            new_layers = []
            for layer in cache_params.layers:
                layer_copy = FLABackbone._shallow_copy(layer)
                state = getattr(layer, "state", None)
                if isinstance(state, dict):
                    layer_copy.state = FLABackbone._copy_state(state)
                else:
                    raise ValueError("Unsupported layer state structure for copy.")
                new_layers.append(layer_copy)
            cache_params_copy.layers = new_layers
            return cache_params_copy
        if hasattr(cache_params, "states"):
            cache_params_copy = FLABackbone._shallow_copy(cache_params)
            cache_params_copy.states = [
                FLABackbone._copy_state(state) if isinstance(state, dict) else state
                for state in cache_params.states
            ]
            return cache_params_copy
        return FLABackbone._shallow_copy(cache_params)

    @staticmethod
    def _prepare_fla_input(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int] | None]:
        if x.dim() != 4:
            return x, None
        batch_size, seq_len, num_tokens, embed_dim = x.shape
        assert num_tokens == 1, (
            "Currently we only support num_tokens=1 for FLA backbones, "
            f"got num_tokens={num_tokens}"
        )
        x_batched = (
            x.transpose(1, 2)
            .reshape(batch_size * num_tokens, seq_len, embed_dim)
        )
        return x_batched, (batch_size, num_tokens, seq_len, embed_dim)

    @staticmethod
    def _unprepare_fla_output(
        out: torch.Tensor,
        shape_info: tuple[int, int, int, int] | None,
    ) -> torch.Tensor:
        if shape_info is None:
            return out
        batch_size, num_tokens, seq_len, embed_dim = shape_info
        return (
            out.reshape(batch_size, num_tokens, seq_len, embed_dim)
            .transpose(1, 2)
        )

    def incontext_fit(
        self,
        train_x: torch.Tensor,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, tp.Any]:
        """Run the FLA model on the training context and return cached state.

        Accepts either flattened input (B, S, E) or unflattened PFN input
        (B, S, N, E). However we currently only support N=1 for unflattened input.
        """
        x_batched, shape_info = self._prepare_fla_input(train_x)
        train_out, cache_params = self._run_fla(x_batched, return_cache=True)
        cached_state = {
            "cache_params": cache_params,
            "cache_position_start": x_batched.size(1),
        }
        out = self._unprepare_fla_output(train_out, shape_info)
        return out, cached_state

    def incontext_predict(
        self,
        test_x: torch.Tensor,
        cached_state: tp.Any,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        """Run the FLA model on test inputs using cached past key values in parallel."""
        cache_position_start = cached_state.get("cache_position_start", None)
        cache_params = cached_state["cache_params"]

        x_batched, shape_info = self._prepare_fla_input(test_x)
        output = self._run_test_with_cache(
            x_batched,
            cache_params,
            cache_position_start=cache_position_start,
        )
        return self._unprepare_fla_output(output, shape_info)

    def _run_fla(
        self,
        x: torch.Tensor,
        *,
        cache_params: tp.Any | None = None,
        cache_position_start: int | None = None,
        return_cache: bool = True,
        use_custom_recurrent: bool = False,
        use_custom_shortconv: bool = False,
    ) -> tuple[torch.Tensor, tp.Any | None]:
        use_cache = return_cache or (
            cache_params is not None and isinstance(self.fla, Mamba2Model)
        )
        kwargs: dict[str, tp.Any] = {"inputs_embeds": x, "use_cache": use_cache}
        if cache_params is not None:
            if isinstance(self.fla, self._CUSTOM_RECURRENT_MODELS):
                kwargs["past_key_values"] = cache_params
            else:
                raise ValueError("Unsupported FLA model type for cache_params.")
        try:
            with ExitStack() as stack:
                for ctx in self._patch_contexts(
                    use_custom_recurrent=use_custom_recurrent,
                    use_custom_shortconv=use_custom_shortconv,
                ):
                    stack.enter_context(ctx)
                out = self.fla(**kwargs)
        except TypeError as exc:
            raise TypeError(
                "FLA model does not support cache usage; required for independent evaluation."
            ) from exc

        if hasattr(out, "last_hidden_state"):
            last_hidden_state = out.last_hidden_state
        else:
            raise RuntimeError("FLA model output does not contain last_hidden_state.")

        cache_params = None
        if return_cache:
            if hasattr(out, "past_key_values"):
                cache_params = out.past_key_values
            elif hasattr(out, "cache_params"):
                cache_params = out.cache_params
            else:
                raise RuntimeError("FLA model output does not contain past_key_values or cache_params.")
            # Store cache_position_start as fallback for direct _run_fla calls (bypassing incontext_fit)
            if cache_params is not None and not hasattr(cache_params, "_cache_position_start"):
                cache_params._cache_position_start = x.size(1)
        return last_hidden_state, cache_params

    def _patch_contexts(
        self, 
        use_custom_recurrent: bool,
        use_custom_shortconv : bool = False,
    ) -> tp.Iterable[tp.ContextManager[tp.Any]]:
        """
        Get context managers for patching FLA model behavior. 
        If use_custom_recurrent is True, we also patch the shortconv forward.
        """
        model = self.fla
        contexts: list[tp.ContextManager[tp.Any]] = []
        contexts.append(_maybe_patch_shortconv_forward_pytorch(use_custom_shortconv or use_custom_recurrent))

        patch_registry: tuple[
            tuple[
                type[nn.Module],
                tp.Callable[..., tp.ContextManager[tp.Any]],
            ],
            ...,
        ] = (
            (GLAModel, _maybe_patch_gla_with_stateless_recurrent),
            (KDAModel, _maybe_patch_kda_with_stateless_recurrent),
            (DeltaNetModel, _maybe_patch_deltanet_with_stateless_recurrent),
            (GatedDeltaNetModel, _maybe_patch_gated_deltanet_with_stateless_recurrent),
            (Mamba2Model, _maybe_patch_mamba2_with_stateless_recurrent),
            (LinearAttentionModel, _maybe_patch_linear_attn_with_stateless_recurrent),
        )
        for model_type, ctx_factory in patch_registry:
            if isinstance(model, model_type):
                contexts.append(ctx_factory(use_custom_recurrent))
        return contexts

    def _supports_custom_recurrent(self) -> bool:
        return isinstance(self.fla, self._CUSTOM_RECURRENT_MODELS)

    def _run_test_with_cache(
        self,
        test_x: torch.Tensor,
        cache_params: tp.Any,
        cache_position_start: int | None = None,
        use_custom_recurrent: bool = True,
        use_custom_shortconv: bool = True,
    ) -> torch.Tensor:
        """
        Run the FLA model on test inputs using cached past key values in parallel.
        """
        if test_x.numel() == 0:
            return test_x

        assert cache_params is not None, "Cache parameters must be provided for test-time evaluation."
        if cache_position_start is None:
            cache_position_start = getattr(cache_params, "_cache_position_start", None)

        batch_size, seq_len, embed_dim = test_x.shape
        supports_custom_recurrent = self._supports_custom_recurrent()

        def _run_parallel_chunk(chunk_x: torch.Tensor) -> torch.Tensor:
            chunk_len = chunk_x.size(1)
            if not use_custom_recurrent or not supports_custom_recurrent:
                expanded_cache = self._repeat_cache(cache_params, chunk_len)
            else:
                expanded_cache = cache_params
            chunk_flat = chunk_x.contiguous().view(batch_size * chunk_len, 1, embed_dim)
            output, _ = self._run_fla(
                chunk_flat,
                cache_params=expanded_cache,
                cache_position_start=cache_position_start,
                return_cache=False,
                use_custom_recurrent=use_custom_recurrent,
                use_custom_shortconv=use_custom_shortconv,
            )
            output = output.view(batch_size, chunk_len, embed_dim)
            return output

        if self.cache_chunk_size is None or seq_len <= self.cache_chunk_size:
            return _run_parallel_chunk(test_x)

        outputs = []
        for chunk_start in range(0, seq_len, self.cache_chunk_size):
            chunk_end = min(chunk_start + self.cache_chunk_size, seq_len)
            chunk_x = test_x[:, chunk_start:chunk_end]
            outputs.append(_run_parallel_chunk(chunk_x))
        return torch.cat(outputs, dim=1)
    
    def _run_test_with_cache_naive(
        self,
        test_x: torch.Tensor,
        cache_params: tp.Any | None,
        cache_position_start: int | None = None,
        use_custom_recurrent: bool = False,
        use_custom_shortconv: bool = False,
    ) -> torch.Tensor:
        """
        Sequentially processes the test sequence one token at a time.
        """
        if test_x.numel() == 0:
            return test_x
        if cache_position_start is None and cache_params is not None:
            cache_position_start = getattr(cache_params, "_cache_position_start", None)

        output_tokens = []
        seq_len = test_x.size(1)
    
        for t in range(seq_len):
            current_input = test_x[:, t : t + 1, :]  # shape (batch, 1, dim)
            output, _ = self._run_fla(
                current_input,
                cache_params=self._copy_cache(cache_params),
                cache_position_start=cache_position_start,
                return_cache=False,
                use_custom_recurrent=use_custom_recurrent,
                use_custom_shortconv=use_custom_shortconv,
            )
            output_tokens.append(output)
        output = torch.cat(output_tokens, dim=1)
            
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        *,
        single_eval_pos: int | None = None,
        half_layers: bool = False,
        cache_trainset_representation: bool = False,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        """Forward pass through FLA model.
        
        Args:
            x: Input tensor of shape (batch, seq, num_tokens, embed)
            single_eval_pos: Position marking end of training context
            half_layers: Whether to use only half the layers
            cache_trainset_representation: Whether caching is enabled
            **kwargs: Additional arguments
            
        Returns:
            Output tensor of shape (batch, seq, num_tokens, embed)
        """
        assert half_layers is False, "half_layers not supported in FLA backbone"
        assert cache_trainset_representation is False, (
            "cache_trainset_representation not supported in FLA backbone"
        )
        assert single_eval_pos is not None, "single_eval_pos must be provided for FLA backbone"

        batch_size, seq_len, num_tokens, embed_dim = x.shape
        # Input x is usually [Batch, SeqLen, NumTokens, EmSize]
        # FLA expects [Batch, SeqLen, EmSize] -> so we flatten NumTokens into Batch
        x_batched = x.transpose(1, 2).reshape(batch_size * num_tokens, seq_len, embed_dim)

        train_len = min(single_eval_pos, seq_len)
    
        if self.sequence_mode in {"Comb_ST", "Int_ST"} or not self.training:
            train_x = x_batched[:, :train_len]
            test_x = x_batched[:, train_len:]

            train_out, state = self.incontext_fit(train_x)
            test_out = self.incontext_predict(test_x, state)
            attn_out = torch.cat([train_out, test_out], dim=1)
        else:
            attn_out, _ = self._run_fla(x_batched, return_cache=False)
        
        out = attn_out.reshape(batch_size, num_tokens, seq_len, embed_dim).transpose(1, 2)
        return out


@dataclass(frozen=True)
class LinearAttentionBackboneConfig(BackboneConfig):
    """Configuration for a Linear Attention backbone."""
    nlayers: int = 6
    nhead: int = 2
    mlp_hidden_dim: int = 200
    dropout: float = 0.0
    activation: tp.Literal["gelu", "relu", "swish", "silu"] = "silu"
    recompute_layer: bool = False
    recompute_every_n_layers: int = 1
    layer_kwargs: tp.Dict[str, base_config.BaseTypes] | None = None

    def create_backbone(
        self,
        ninp: int,
        attention_between_features: bool,
        **kwargs: tp.Any,
    ) -> Backbone:
        layers = nn.ModuleList([
            LinearAttention(
                d_model=ninp,
                num_heads=self.nhead,
                dim_mlp_hidden=self.mlp_hidden_dim,
                dropout=self.dropout,
                activation=self.activation,
                attention_between_features=attention_between_features,
                **(self.layer_kwargs or {}),
            )
            for _ in range(self.nlayers)
        ])
        return LinearAttentionBackbone(
            layers,
            recompute_each_layer=self.recompute_layer,
            recompute_every_n_layers=self.recompute_every_n_layers,
        )


class LinearAttentionBackbone(Backbone):
    """Stack of LinearAttention layers as a backbone."""
    def __init__(
        self,
        layers: nn.ModuleList,
        *,
        recompute_each_layer: bool = False,
        recompute_every_n_layers: int | None = 1,
    ):
        super().__init__()
        self.layers = layers
        self.recompute_each_layer = bool(recompute_each_layer)
        self.recompute_every_n_layers = (
            None if recompute_every_n_layers is None else int(recompute_every_n_layers)
        )
        if self.recompute_every_n_layers is not None and self.recompute_every_n_layers <= 0:
            raise ValueError("recompute_every_n_layers must be >= 1")

    def forward(
        self,
        x: torch.Tensor,
        *,
        single_eval_pos: int | None = None,
        half_layers: bool = False,
        cache_trainset_representation: bool = False,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        # x: (batch, seq, num_tokens, embed_dim)
        out = x
        assert half_layers is False, "half_layers not supported in LinearAttention backbone"
        assert (
            cache_trainset_representation is False
        ), "cache_trainset_representation not supported in LinearAttention backbone"
        for idx, layer in enumerate(self.layers):
            should_recompute = (
                self.recompute_each_layer
                and out.requires_grad
                and self.recompute_every_n_layers is not None
                and (idx % self.recompute_every_n_layers == 0)
            )
            if should_recompute:
                out = checkpoint(
                    layer,
                    out,
                    single_eval_pos=single_eval_pos,
                    use_reentrant=False,
                    **kwargs,
                )
            else:
                out = layer(out, single_eval_pos=single_eval_pos, **kwargs)
        return out

    def incontext_fit(
        self,
        x: torch.Tensor,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, tp.Any]:
        out = x
        layer_states: list[dict[str, torch.Tensor]] = []
        for layer in self.layers:
            out, state = layer.incontext_fit(out)
            layer_states.append(state)
        cached_state = {"layer_states": layer_states}
        return out, cached_state

    def incontext_predict(
        self,
        x: torch.Tensor,
        cached_state: tp.Any,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        out = x
        layer_states = cached_state.get("layer_states", [])
        for layer, state in zip(self.layers, layer_states):
            out = layer.incontext_predict(out, state)
        return out


@dataclass(frozen=True)
class RebasedBackboneConfig(BackboneConfig):
    nlayers: int = 6
    mlp_hidden_dim: int = 200
    num_heads: int = 2
    activation: str = "silu"
    dropout: float = 0.1
    recompute_layer: bool = False
    recompute_every_n_layers: int | None = None
    layer_kwargs: tp.Dict[str, base_config.BaseTypes] | None = None


    def create_backbone(
        self,
        ninp: int,
        attention_between_features: bool,
        **kwargs: tp.Any,
    ) -> Backbone:
        assert attention_between_features is False, (
            "RebasedBackbone currently does not support attention between features"
        )

        layers = nn.ModuleList(
            [
                RebasedLinearAttention(
                    d_model=ninp,
                    num_heads=self.num_heads,
                    dim_mlp_hidden=self.mlp_hidden_dim,
                    dropout=self.dropout,
                    activation=self.activation,
                    **(self.layer_kwargs or {}),
                )
                for _ in range(self.nlayers)
            ]
        )
        return LinearAttentionBackbone(
            layers,
            recompute_each_layer=self.recompute_layer,
            recompute_every_n_layers=self.recompute_every_n_layers,
        )
