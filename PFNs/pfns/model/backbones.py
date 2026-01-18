"""Configuration classes for different model backbones.

This module provides base classes and implementations for configuring
different backbones that can be used within the ModelConfig.
"""
from __future__ import annotations

import os
import typing as tp
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn

from fla.models import GLAConfig, GLAModel
from fla.models import RetNetConfig, RetNetModel
from fla.models import Mamba2Config, Mamba2Model
from fla.models import KDAConfig, KDAModel
from fla.models import DeltaNetConfig, DeltaNetModel
from fla.models import GatedDeltaNetConfig, GatedDeltaNetModel

from pfns import base_config
from pfns.model.layer import PerFeatureLayer
from pfns.model.linear_attention import LinearAttention
from pfns.model.rebased_linear_attention import RebasedLinearAttention
from pfns.model.tabular_model import LayerStack
# Registry mapping model types to their config and model classes
FLA_MODEL_REGISTRY = {
    "gla": (GLAConfig, GLAModel),
    "retnet": (RetNetConfig, RetNetModel),
    "mamba2": (Mamba2Config, Mamba2Model),
    "kda": (KDAConfig, KDAModel),
    "deltanet": (DeltaNetConfig, DeltaNetModel),
    "gated_deltanet": (GatedDeltaNetConfig, GatedDeltaNetModel),
}


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


@dataclass(frozen=True)
class FLABackboneConfig(BackboneConfig):
    """Configuration for Flash Linear Attention (FLA) based backbones."""

    model_type: tp.Literal["gla", "retnet", "mamba2", "kda", "deltanet", "gated_deltanet"] = "gla"
    config_kwargs: dict[str, tp.Any] | None = None
    sequence_mode: tp.Literal["cached", "causal", "teacher_forcing"] = "cached"
    cache_chunk_size: int | None = None

    def __post_init__(self):
        if self.model_type not in FLA_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. Available: {list(FLA_MODEL_REGISTRY)}"
            )
        if self.sequence_mode not in {"cached", "causal", "teacher_forcing"}:
            raise ValueError(f"Unknown sequence_mode: {self.sequence_mode}")

    def create_backbone(self, ninp: int, attention_between_features: bool, **kwargs: tp.Any) -> "Backbone":
        ConfigClass, ModelClass = FLA_MODEL_REGISTRY[self.model_type]

        assert attention_between_features is False, (
            "FLA backbones currently do not support attention between features"
        )

        if self.config_kwargs is None:
            raise ValueError("FLABackboneConfig requires config_kwargs to build the FLA config.")
        
        config = ConfigClass(**self.config_kwargs)
        fla_model = ModelClass(config)

        return FLABackbone(
            fla_model=fla_model,
            sequence_mode=self.sequence_mode,
            cache_chunk_size=self.cache_chunk_size,
        )


class FLABackbone(Backbone):
    """Wrapper for FLA models to conform to Backbone interface."""

    def __init__(
        self,
        fla_model: nn.Module,
        sequence_mode: tp.Literal["cached", "causal", "teacher_forcing"] = "cached",
        cache_chunk_size: int | None = None,
    ):
        super().__init__()
        self.fla = fla_model.model if hasattr(fla_model, "model") else fla_model
        self.sequence_mode = sequence_mode
        self.cache_chunk_size = cache_chunk_size
        self._debug_cache = os.getenv("FLA_CACHE_DEBUG", "0") not in {"", "0", "false", "False"}
        self._cache_debugged = False

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

    def incontext_fit(
        self,
        train_x: torch.Tensor,
    ) -> tuple[torch.Tensor, tp.Any]:
        """Run the FLA model on the training context and return cached state."""
        train_out, cache_params = self._run_fla(train_x, return_cache=True)
        cached_state = {'cache_params': cache_params, 'cache_position_start': train_x.size(1)}
        return train_out, cached_state

    def incontext_predict(
        self,
        test_x: torch.Tensor,
        cached_state: tp.Any,
    ) -> torch.Tensor:
        """Run the FLA model on test inputs using cached past key values in parallel."""
        cache_position_start = cached_state.get('cache_position_start', None)
        cache_params = cached_state['cache_params']

        output = self._run_test_with_cache(
            test_x,
            cache_params,
            cache_position_start=cache_position_start,
        )
        
        return output

    def _run_fla(
        self,
        x: torch.Tensor,
        *,
        cache_params: tp.Any | None = None,
        cache_position_start: int | None = None,
        return_cache: bool = True,
    ) -> tuple[torch.Tensor, tp.Any | None]:
        kwargs: dict[str, tp.Any] = {"inputs_embeds": x, "use_cache": cache_params is not None or return_cache} 
        if cache_params is not None:
            if isinstance(self.fla, Mamba2Model):
                kwargs["cache_params"] = cache_params
                if cache_position_start is None:
                    raise ValueError(
                        "cache_position_start is required for Mamba2 when cache_params is provided."
                    )
                kwargs["cache_position"] = torch.arange(
                    cache_position_start,
                    cache_position_start + x.size(1),
                    device=x.device,
                )
            elif isinstance(self.fla, (GLAModel, KDAModel, DeltaNetModel, GatedDeltaNetModel)):
                kwargs["past_key_values"] = cache_params
            else:
                raise ValueError("Unsupported FLA model type for cache_params.")
        try:
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
            if self._debug_cache and cache_params is not None and not self._cache_debugged:
                print(f"[FLA cache] {self._summarize_cache(cache_params)}")
                self._cache_debugged = True
        return last_hidden_state, cache_params

    def _run_test_with_cache(
        self,
        test_x: torch.Tensor,
        cache_params: tp.Any,
        cache_position_start: int | None = None,
    ) -> torch.Tensor:
        """
        Run the FLA model on test inputs using cached past key values in parallel.
        """
        if test_x.numel() == 0:
            return test_x

        assert cache_params is not None, "Cache parameters must be provided for test-time evaluation."

        batch_size, seq_len, embed_dim = test_x.shape

        def _shallow_copy(obj: tp.Any) -> tp.Any:
            obj_copy = obj.__class__.__new__(obj.__class__)
            obj_copy.__dict__.update(obj.__dict__)
            return obj_copy

        def _repeat_state(state: dict[str, tp.Any], repeat: int, *, dim: int) -> dict[str, tp.Any]:
            def _repeat_value(value: tp.Any) -> tp.Any:
                if torch.is_tensor(value):
                    return value.repeat_interleave(repeat, dim=dim)
                if isinstance(value, tuple):
                    return tuple(_repeat_value(item) for item in value)
                return value

            return {key: _repeat_value(value) for key, value in state.items()}

        def _repeat_cache(cache_params: tp.Any, repeat: int) -> tp.Any:
            if torch.is_tensor(cache_params):
                return cache_params.repeat_interleave(repeat, dim=0)
            if hasattr(cache_params, "conv_states") and hasattr(cache_params, "ssm_states"): # Mamba2 style
                cache_params_copy = _shallow_copy(cache_params)
                cache_params_copy.conv_states = cache_params.conv_states.repeat_interleave(repeat, dim=1)
                cache_params_copy.ssm_states = cache_params.ssm_states.repeat_interleave(repeat, dim=1)
                return cache_params_copy
            if hasattr(cache_params, "layers"):  # GLA, KDA, (Gated) Deltanet style
                cache_params_copy = _shallow_copy(cache_params)
                new_layers = []
                for layer in cache_params.layers:
                    layer_copy = _shallow_copy(layer)
                    state = getattr(layer, "state", None)
                    if isinstance(state, dict):
                        layer_copy.state = _repeat_state(state, repeat, dim=0)
                    else:
                        raise ValueError("Unsupported layer state structure for repetition.")
                    new_layers.append(layer_copy)
                cache_params_copy.layers = new_layers
                return cache_params_copy
            raise ValueError("Unsupported cache_params structure for repetition.")

        def _run_parallel_chunk(chunk_x: torch.Tensor) -> torch.Tensor:
            chunk_len = chunk_x.size(1)
            
            expanded_cache = _repeat_cache(cache_params, chunk_len)
            chunk_flat = chunk_x.contiguous().view(batch_size * chunk_len, 1, embed_dim)
            output, _ = self._run_fla(
                chunk_flat,
                cache_params=expanded_cache,
                cache_position_start=cache_position_start,
                return_cache=False,
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
    ) -> torch.Tensor:
        """
        Sequentially processes the test sequence one token at a time.
        """
        if test_x.numel() == 0:
            return test_x

        output_tokens = []
        seq_len = test_x.size(1)
        for t in range(seq_len):
            current_input = test_x[:, t : t + 1, :]  # shape (batch, 1, dim)
            output, cache_params = self._run_fla(
                current_input,
                cache_params=cache_params,
                cache_position_start=cache_position_start,
                return_cache=True,
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
    
        if self.sequence_mode == "cached" or not self.training: # during eval, always use cached mode
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
    nhid: int = 200
    dropout: float = 0.1
    activation: tp.Literal["gelu", "relu", "swish", "silu"] = "gelu"
    feature_attention_softmax: bool = False

    def create_backbone(
        self,
        ninp: int,
        attention_between_features: bool,
        **kwargs: tp.Any,
    ) -> Backbone:
        layers = nn.ModuleList([
            LinearAttention(
                d_model=ninp,
                nhead=self.nhead,
                dim_feedforward=self.nhid,
                dropout=self.dropout,
                activation=self.activation,
                attention_between_features=attention_between_features,
                feature_attention_softmax=self.feature_attention_softmax,
            )
            for _ in range(self.nlayers)
        ])
        return LinearAttentionBackbone(layers)


class LinearAttentionBackbone(Backbone):
    """Stack of LinearAttention layers as a backbone."""
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

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
        for layer in self.layers:
            out = layer(out, single_eval_pos=single_eval_pos)
        return out


@dataclass(frozen=True)
class RebasedBackboneConfig(BackboneConfig):
    nlayers: int = 6
    nhid: int | None = None
    num_heads: int = 4
    feature_dim: int = 16
    activation: str = "silu"
    dropout: float = 0.1
    use_gamma: bool = True
    use_beta: bool = True
    normalize: bool = True
    eps: float = 1e-5

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
                    dim_feedforward=self.nhid,
                    num_heads=self.num_heads,
                    feature_dim=self.feature_dim,
                    dropout=self.dropout,
                    activation=self.activation,
                    use_gamma=self.use_gamma,
                    use_beta=self.use_beta,
                    normalize=self.normalize,
                    eps=self.eps,
                )
                for _ in range(self.nlayers)
            ]
        )
        return LinearAttentionBackbone(layers)
