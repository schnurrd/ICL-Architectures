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
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fla.models import GLAConfig, GLAModel
from fla.models import Mamba2Config, Mamba2Model
from fla.models import KDAConfig, KDAModel
from fla.models import DeltaNetConfig, DeltaNetModel
from fla.models import GatedDeltaNetConfig, GatedDeltaNetModel
from fla.models import LinearAttentionConfig, LinearAttentionModel
from fla.models import MesaNetConfig, MesaNetModel
from fla.models.utils import Cache as FLACache

from pfns import base_config
from pfns.model.fla_mimetic_init import MimeticInitMode, apply_mimetic_fla_init
from pfns.model.fla_patches import (
    DELTANET_BETA_DECAY_MODES,
    _maybe_patch_gla_with_stateless_recurrent,
    _maybe_patch_kda_with_stateless_recurrent,
    _maybe_patch_deltanet_with_stateless_recurrent,
    _maybe_patch_gated_deltanet_with_stateless_recurrent,
    _maybe_patch_mesanet_with_stateless_recurrent,
    _maybe_patch_mamba2_with_stateless_recurrent,
    _maybe_patch_linear_attn_with_stateless_recurrent,
    _maybe_patch_shortconv_forward_pytorch,
)
from pfns.model.fla_state_passing import (
    FLAStatePassing,
    prepare_deltanet_cache_for_fla,
)
from pfns.model.attention_utils import build_norm
from pfns.model.layer import PerFeatureLayer
from pfns.model.linear_attention import (
    LinearAttention,
    init_linear_attention_weights_like_fla,
)
from pfns.model.mode_normalization import (
    CANONICAL_SEQUENCE_MODES,
    resolve_sequence_mode,
)

from pfns.model.bidirectional_fla import (  # noqa: E402
    BIDIRECTIONAL_FLA_SEQUENCE_MODES,
    BIDIRECTIONAL_STATE_FUSIONS,
    BidirectionalFLACache,
    FusedBidirectionalFLACache,
    _get_fla_layers,
    _make_fla_model_bidirectional,
    _uses_fused_prediction_cache,
    fuse_bidirectional_cache,
    prepare_bidirectional_cache,
    run_bidirectional_layers,
    validate_bidirectional_cache,
)
from pfns.model.fla_cache_utils import (
    copy_cache as _copy_fla_cache,
    copy_state as _copy_fla_state,
    repeat_cache as _repeat_fla_cache,
    repeat_state as _repeat_fla_state,
    shallow_copy as _shallow_copy_fla,
)

from pfns.model.based_linear_attention import BasedLinearAttention
from pfns.model.tabular_model import LayerStack
# Registry mapping model types to their config and model classes
FLA_MODEL_REGISTRY = {
    "gla": (GLAConfig, GLAModel),
    "mamba2": (Mamba2Config, Mamba2Model),
    "kda": (KDAConfig, KDAModel),
    "deltanet": (DeltaNetConfig, DeltaNetModel),
    "gated_deltanet": (GatedDeltaNetConfig, GatedDeltaNetModel),
    "linear_attn": (LinearAttentionConfig, LinearAttentionModel),
    "mesanet": (MesaNetConfig, MesaNetModel),
}

FLA_SEQUENCE_MODES = set(CANONICAL_SEQUENCE_MODES)
FLA_SPLIT_SEQUENCE_MODES = {"Comb_ST", "Int_ST"}
FINAL_STATE_READOUT_FLA_MODELS = {
    "linear_attn",
    "gla",
    "kda",
    "deltanet",
    "gated_deltanet",
}
FINAL_STATE_READOUT_FLA_MODEL_CLASSES = {
    FLA_MODEL_REGISTRY[model_type][1]
    for model_type in FINAL_STATE_READOUT_FLA_MODELS
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
        "gla", "mamba2", "kda", "deltanet", "gated_deltanet", "linear_attn", "mesanet"
    ] = "linear_attn"
    config_kwargs: dict[str, tp.Any] | None = None
    sequence_mode: tp.Literal["Comb_ST", "Int_ST", "Comb_MT", "Int_MT"] = "Comb_ST"
    cache_chunk_size: int | None = None
    bidirectional: bool = False
    bidirectional_share_weights: bool = True
    bidirectional_state_fusion: str = "mean_output_mean_cache"
    state_passing: bool = False
    state_passing_dropout: float = 0.1
    state_weaving: bool = False
    include_self_term: bool = True
    final_state_readout: bool = False
    deltanet_beta_decay: tp.Literal[
        "none",
        "inverse",
        "sqrt_inverse",
        "sqrt_length_inverse",
        "online_inverse",
        "online_sqrt_inverse",
        "nlms",
        "nlms_inverse",
        "nlms_sqrt_inverse",
    ] = "none"
    deltanet_beta_decay_t0: int = 1000
    mimetic_init: bool = False
    mimetic_init_layer_indices: tuple[int, ...] | list[int] | None = None
    mimetic_init_mode: MimeticInitMode = "gate_only"

    def __post_init__(self):
        if self.model_type not in FLA_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. Available: {list(FLA_MODEL_REGISTRY)}"
            )
        object.__setattr__(
            self,
            "sequence_mode",
            resolve_sequence_mode(self.sequence_mode),
        )
        if self.bidirectional and self.sequence_mode not in BIDIRECTIONAL_FLA_SEQUENCE_MODES:
            raise ValueError(
                "Bidirectional FLA currently supports only sequence_mode "
                f"in {sorted(BIDIRECTIONAL_FLA_SEQUENCE_MODES)}, "
                f"got {self.sequence_mode!r}."
            )
        if self.bidirectional:
            supported_bidirectional_models = {"linear_attn", "gla", "deltanet"}
            if self.model_type not in supported_bidirectional_models:
                raise ValueError(
                    "Bidirectional FLA supports only model_type in "
                    f"{sorted(supported_bidirectional_models)}, got {self.model_type!r}."
                )
            if self.bidirectional_state_fusion not in BIDIRECTIONAL_STATE_FUSIONS:
                raise ValueError(
                    "bidirectional_state_fusion must be one of "
                    f"{sorted(BIDIRECTIONAL_STATE_FUSIONS)}."
                )
            if (
                not self.bidirectional_share_weights
                and _uses_fused_prediction_cache(self.bidirectional_state_fusion)
            ):
                raise ValueError(
                    "bidirectional_share_weights=False is not supported with fused "
                    "bidirectional prediction caches."
                )
        if self.bidirectional and self.state_passing:
            raise ValueError("Bidirectional FLA does not support state_passing.")
        if self.final_state_readout:
            if self.model_type not in FINAL_STATE_READOUT_FLA_MODELS:
                raise ValueError(
                    "final_state_readout currently supports only model_type in "
                    f"{sorted(FINAL_STATE_READOUT_FLA_MODELS)}."
                )
            if self.bidirectional:
                raise ValueError("final_state_readout does not support bidirectional FLA.")
            if self.state_weaving:
                raise ValueError("final_state_readout does not support state_weaving.")
        if self.deltanet_beta_decay not in DELTANET_BETA_DECAY_MODES:
            raise ValueError(
                "deltanet_beta_decay must be one of "
                f"{sorted(DELTANET_BETA_DECAY_MODES)}, got {self.deltanet_beta_decay!r}."
            )
        if self.deltanet_beta_decay != "none":
            if self.model_type != "deltanet":
                raise ValueError(
                    "deltanet_beta_decay is supported only for model_type='deltanet'."
                )
            if self.bidirectional:
                raise ValueError("deltanet_beta_decay does not support bidirectional FLA.")
            if self.state_passing:
                raise ValueError("deltanet_beta_decay does not support state_passing.")
            if self.state_weaving:
                raise ValueError("deltanet_beta_decay does not support state_weaving.")
            if self.deltanet_beta_decay_t0 <= 0:
                raise ValueError("deltanet_beta_decay_t0 must be > 0.")
        if self.state_weaving:
            if self.sequence_mode != "Comb_ST":
                raise ValueError("state_weaving currently supports only sequence_mode='Comb_ST'.")
            if self.bidirectional:
                raise ValueError("Bidirectional FLA does not support state_weaving.")
            if self.state_passing:
                raise ValueError("state_weaving does not support state_passing.")
        if not 0.0 <= self.state_passing_dropout <= 1.0:
            raise ValueError("state_passing_dropout must be in [0, 1].")

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
        if self.mimetic_init:
            layer_indices = self.mimetic_init_layer_indices
            apply_mimetic_fla_init(
                fla_model,
                layer_indices=layer_indices,
                mode=self.mimetic_init_mode,
            )
        if self.bidirectional:
            wrapped_model = fla_model.model if hasattr(fla_model, "model") else fla_model
            _make_fla_model_bidirectional(
                wrapped_model,
                hidden_size=int(config.hidden_size),
                bidirectional_share_weights=self.bidirectional_share_weights,
                state_fusion=self.bidirectional_state_fusion,
            )

        backbone_cls = BidirectionalFLABackbone if self.bidirectional else FLABackbone
        backbone_kwargs = dict(
            fla_model=fla_model,
            sequence_mode=self.sequence_mode,
            cache_chunk_size=self.cache_chunk_size,
            state_passing=self.state_passing,
            state_passing_dropout=self.state_passing_dropout,
            include_self_term=self.include_self_term,
        )
        if self.bidirectional:
            backbone_kwargs["state_fusion"] = self.bidirectional_state_fusion
        else:
            backbone_kwargs["state_weaving"] = self.state_weaving
            backbone_kwargs["final_state_readout"] = self.final_state_readout
            backbone_kwargs["deltanet_beta_decay"] = self.deltanet_beta_decay
            backbone_kwargs["deltanet_beta_decay_t0"] = self.deltanet_beta_decay_t0

        return backbone_cls(**backbone_kwargs)


class FLABackbone(Backbone):
    """Wrapper for FLA models to conform to Backbone interface."""

    _CUSTOM_RECURRENT_MODELS: tuple[type[nn.Module], ...] = (
        GLAModel,
        KDAModel,
        DeltaNetModel,
        GatedDeltaNetModel,
        Mamba2Model,
        LinearAttentionModel,
        MesaNetModel,
    )

    def __init__(
        self,
        fla_model: nn.Module,
        sequence_mode: str = "Comb_ST",
        cache_chunk_size: int | None = None,
        state_passing: bool = False,
        state_passing_dropout: float = 0.1,
        state_weaving: bool = False,
        include_self_term: bool = True,
        final_state_readout: bool = False,
        deltanet_beta_decay: str = "none",
        deltanet_beta_decay_t0: int = 1000,
    ):
        super().__init__()
        self.fla = fla_model.model if hasattr(fla_model, "model") else fla_model
        assert not (
            state_passing and isinstance(self.fla, Mamba2Model)
        ), "Mamba2 does not support state_passing."
        self.sequence_mode = resolve_sequence_mode(sequence_mode)
        self.cache_chunk_size = cache_chunk_size
        self.include_self_term = bool(include_self_term)
        self.final_state_readout = bool(final_state_readout)
        self.deltanet_beta_decay = str(deltanet_beta_decay)
        self.deltanet_beta_decay_t0 = int(deltanet_beta_decay_t0)
        self.state_weaving = bool(state_weaving)
        self.state_passing = (
            FLAStatePassing(dropout_prob=state_passing_dropout)
            if state_passing
            else None
        )
        self.state_weaving_initial_states = nn.ParameterList()
        if self.state_weaving:
            for layer in self.layers:
                attn = getattr(layer, "attn", None)
                if not all(hasattr(attn, name) for name in ("num_heads", "head_k_dim", "head_v_dim")):
                    raise ValueError(
                        "state_weaving requires every FLA layer to expose generic "
                        "recurrent state dimensions on layer.attn."
                    )
                if getattr(attn, "use_short_conv", False) and not all(
                    hasattr(attn, name) for name in ("q_conv1d", "k_conv1d", "v_conv1d")
                ):
                    raise ValueError(
                        "state_weaving requires short-conv layers to expose q/k/v "
                        "conv states. This model needs a custom conv-state adapter."
                    )
                head_k_dim = int(attn.head_k_dim)
                self.state_weaving_initial_states.append(
                    nn.Parameter(
                        torch.randn(
                            int(attn.num_heads),
                            head_k_dim,
                            int(attn.head_v_dim),
                        )
                        / head_k_dim
                    )
                )

    @property
    def layers(self) -> nn.ModuleList:
        return _get_fla_layers(self.fla)

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
        return _shallow_copy_fla(obj)

    @staticmethod
    def _repeat_state(state: dict[str, tp.Any], repeat: int, *, dim: int) -> dict[str, tp.Any]:
        return _repeat_fla_state(state, repeat, dim=dim)

    @staticmethod
    def _copy_state(state: dict[str, tp.Any]) -> dict[str, tp.Any]:
        return _copy_fla_state(state)

    @staticmethod
    def _repeat_cache(cache_params: tp.Any, repeat: int) -> tp.Any:
        return _repeat_fla_cache(cache_params, repeat)

    @staticmethod
    def _copy_cache(cache_params: tp.Any) -> tp.Any:
        return _copy_fla_cache(cache_params)

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

    @staticmethod
    def _unpack_fla_output(
        out: tp.Any,
        *,
        return_cache: bool,
        model_name: str = "FLA model",
    ) -> tuple[torch.Tensor, tp.Any | None]:
        if hasattr(out, "last_hidden_state"):
            last_hidden_state = out.last_hidden_state
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            last_hidden_state = out[0]
        else:
            raise RuntimeError(f"{model_name} output does not contain last_hidden_state.")

        cache_params = None
        if return_cache:
            if hasattr(out, "past_key_values"):
                cache_params = out.past_key_values
            elif isinstance(out, (tuple, list)) and len(out) > 1:
                cache_params = out[1]
            else:
                raise RuntimeError(f"{model_name} output does not contain past_key_values.")
        return last_hidden_state, cache_params

    def incontext_fit(
        self,
        train_x: torch.Tensor,
        *,
        initial_cache_params: tp.Any | None = None,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, tp.Any]:
        """Run the FLA model on the training context and return cached state.

        Accepts either flattened input (B, S, E) or unflattened PFN input
        (B, S, N, E). However we currently only support N=1 for unflattened input.
        """
        x_batched, shape_info = self._prepare_fla_input(train_x)
        if self.state_weaving:
            if initial_cache_params is not None:
                raise ValueError("state_weaving does not support initial_cache_params.")
            train_out, cache_params = self._run_state_weaving(x_batched)
        else:
            train_out, cache_params = self._run_fla(
                x_batched,
                cache_params=initial_cache_params,
                return_cache=True,
            )
        cached_state = {"cache_params": cache_params}
        out = self._unprepare_fla_output(train_out, shape_info)
        return out, cached_state

    def _run_state_weaving(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tp.Any]:
        hidden_states = x
        cache_params = FLACache()
        previous_recurrent_state: torch.Tensor | None = None

        for layer, learned_state in zip(self.layers, self.state_weaving_initial_states):
            initial_state = learned_state.unsqueeze(0).expand(x.size(0), -1, -1, -1)
            if previous_recurrent_state is not None:
                initial_state = initial_state + previous_recurrent_state
            initial_state = initial_state.contiguous()

            layer_cache = FLACache()
            conv_state = None
            attn = layer.attn
            if getattr(attn, "use_short_conv", False):
                conv_state = tuple(
                    x.new_zeros(x.size(0), conv.weight.size(0), conv.weight.size(-1))
                    for conv in (attn.q_conv1d, attn.k_conv1d, attn.v_conv1d)
                )
            layer_cache.update(
                recurrent_state=initial_state,
                conv_state=conv_state,
                layer_idx=int(layer.layer_idx),
                offset=0,
            )
            hidden_states, _, layer_cache = layer(
                hidden_states,
                past_key_values=layer_cache,
                use_cache=True,
            )
            layer_state = layer_cache.layers[int(layer.layer_idx)].state
            previous_recurrent_state = layer_state["recurrent_state"]
            cache_params.update(
                recurrent_state=previous_recurrent_state,
                conv_state=layer_state.get("conv_state"),
                layer_idx=int(layer.layer_idx),
                offset=x.size(1),
            )

        hidden_states = self.fla.norm(hidden_states)
        return hidden_states, cache_params

    def incontext_predict(
        self,
        test_x: torch.Tensor,
        cached_state: tp.Any,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        """Run the FLA model on test inputs using cached past key values in parallel."""
        cache_params = cached_state["cache_params"]

        x_batched, shape_info = self._prepare_fla_input(test_x)
        output = self._run_test_with_cache(
            x_batched,
            cache_params,
        )
        return self._unprepare_fla_output(output, shape_info)

    def _run_fla(
        self,
        x: torch.Tensor,
        *,
        cache_params: tp.Any | None = None,
        return_cache: bool = True,
        use_custom_recurrent: bool = False,
        use_custom_shortconv: bool = False,
        deltanet_beta_decay_start: int = 0,
    ) -> tuple[torch.Tensor, tp.Any | None]:
        if (
            cache_params is not None
            and isinstance(self.fla, MesaNetModel)
            and not use_custom_recurrent
        ):
            return self._run_mesanet_with_initial_cache(
                x,
                cache_params=cache_params,
                return_cache=return_cache,
            )

        if cache_params is not None and return_cache and use_custom_recurrent:
            raise ValueError(
                "Custom stateless recurrent FLA patches do not support returning "
                "updated caches. Use return_cache=False, or use native cache "
                "processing for cache-updating decode."
            )

        use_cache = return_cache or (
            cache_params is not None and isinstance(self.fla, Mamba2Model)
        )
        kwargs: dict[str, tp.Any] = {
            "inputs_embeds": x,
            "use_cache": use_cache,
            "return_dict": True,
        }
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
                    deltanet_beta_decay_start=deltanet_beta_decay_start,
                ):
                    stack.enter_context(ctx)
                out = self.fla(**kwargs)
        except TypeError as exc:
            raise TypeError(
                "FLA model does not support cache usage; required for independent evaluation."
            ) from exc

        return self._unpack_fla_output(out, return_cache=return_cache)

    def _run_mesanet_with_initial_cache(
        self,
        x: torch.Tensor,
        *,
        cache_params: tp.Any,
        return_cache: bool,
    ) -> tuple[torch.Tensor, tp.Any | None]:
        if x.numel() == 0:
            return x, (self._copy_cache(cache_params) if return_cache else None)

        current_cache = self._copy_cache(cache_params)
        outputs = []
        for t in range(x.size(1)):
            step_x = x[:, t : t + 1, :].transpose(0, 1).contiguous()
            out = self.fla(
                inputs_embeds=step_x,
                past_key_values=current_cache,
                use_cache=True,
                return_dict=True,
            )
            last_hidden_state, current_cache = self._unpack_fla_output(
                out,
                return_cache=True,
                model_name="MesaNet",
            )
            outputs.append(last_hidden_state.transpose(0, 1))

        return torch.cat(outputs, dim=1), (current_cache if return_cache else None)

    def _patch_contexts(
        self, 
        use_custom_recurrent: bool,
        use_custom_shortconv : bool = False,
        deltanet_beta_decay_start: int = 0,
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
            (MesaNetModel, _maybe_patch_mesanet_with_stateless_recurrent),
            (Mamba2Model, _maybe_patch_mamba2_with_stateless_recurrent),
            (LinearAttentionModel, _maybe_patch_linear_attn_with_stateless_recurrent),
        )
        for model_type, ctx_factory in patch_registry:
            if isinstance(model, model_type):
                supports_final_state_readout = (
                    model_type in FINAL_STATE_READOUT_FLA_MODEL_CLASSES
                )
                final_state_readout_active = (
                    self.final_state_readout and supports_final_state_readout
                )
                # Native and custom cached final-state readout both use a pure
                # q @ state projection, so token-local update terms are skipped.
                include_self_term = self.include_self_term and not final_state_readout_active
                extra_kwargs: dict[str, tp.Any] = {}
                if supports_final_state_readout:
                    extra_kwargs["final_state_readout"] = final_state_readout_active
                if model_type is DeltaNetModel:
                    extra_kwargs["beta_decay"] = self.deltanet_beta_decay
                    extra_kwargs["beta_decay_t0"] = self.deltanet_beta_decay_t0
                    extra_kwargs["beta_decay_start"] = deltanet_beta_decay_start
                contexts.append(
                    ctx_factory(
                        use_custom_recurrent,
                        include_self_term=include_self_term,
                        **extra_kwargs,
                    )
                )
        return contexts

    def _supports_custom_recurrent(self) -> bool:
        return isinstance(self.fla, self._CUSTOM_RECURRENT_MODELS)

    @staticmethod
    def _cache_seq_length(cache_params: tp.Any) -> int:
        if hasattr(cache_params, "get_seq_length"):
            return int(cache_params.get_seq_length(0))
        return 0

    def _run_test_with_cache(
        self,
        test_x: torch.Tensor,
        cache_params: tp.Any,
        use_custom_recurrent: bool = True,
        use_custom_shortconv: bool = True,
    ) -> torch.Tensor:
        """
        Run the FLA model on test inputs using cached past key values in parallel.
        """
        if test_x.numel() == 0:
            return test_x

        assert cache_params is not None, "Cache parameters must be provided for test-time evaluation."

        batch_size, seq_len, embed_dim = test_x.shape
        supports_custom_recurrent = self._supports_custom_recurrent()
        if (
            self.deltanet_beta_decay != "none"
            and isinstance(self.fla, DeltaNetModel)
            and (not use_custom_recurrent or not supports_custom_recurrent)
        ):
            return self._run_test_with_cache_naive(
                test_x,
                cache_params,
                use_custom_recurrent=use_custom_recurrent,
                use_custom_shortconv=use_custom_shortconv,
            )

        cache_seq_length = self._cache_seq_length(cache_params)

        def _run_parallel_chunk(chunk_x: torch.Tensor, *, chunk_start: int) -> torch.Tensor:
            chunk_len = chunk_x.size(1)
            if not use_custom_recurrent or not supports_custom_recurrent:
                expanded_cache = self._repeat_cache(cache_params, chunk_len)
            else:
                expanded_cache = cache_params
            chunk_flat = chunk_x.contiguous().view(batch_size * chunk_len, 1, embed_dim)
            output, _ = self._run_fla(
                chunk_flat,
                cache_params=expanded_cache,
                return_cache=False,
                use_custom_recurrent=use_custom_recurrent,
                use_custom_shortconv=use_custom_shortconv,
                deltanet_beta_decay_start=cache_seq_length + chunk_start,
            )
            output = output.view(batch_size, chunk_len, embed_dim)
            return output

        effective_cache_chunk_size = self.cache_chunk_size
        if (
            effective_cache_chunk_size is None
            and use_custom_recurrent
            and isinstance(self.fla, MesaNetModel)
            and seq_len > 128
        ):
            effective_cache_chunk_size = 128

        if effective_cache_chunk_size is None or seq_len <= effective_cache_chunk_size:
            return _run_parallel_chunk(test_x, chunk_start=0)

        outputs = []
        for chunk_start in range(0, seq_len, effective_cache_chunk_size):
            chunk_end = min(chunk_start + effective_cache_chunk_size, seq_len)
            chunk_x = test_x[:, chunk_start:chunk_end]
            outputs.append(_run_parallel_chunk(chunk_x, chunk_start=chunk_start))
        return torch.cat(outputs, dim=1)
    
    def _run_test_with_cache_naive(
        self,
        test_x: torch.Tensor,
        cache_params: tp.Any | None,
        use_custom_recurrent: bool = False,
        use_custom_shortconv: bool = False,
    ) -> torch.Tensor:
        """
        Sequentially processes the test sequence one token at a time.
        """
        if test_x.numel() == 0:
            return test_x

        output_tokens = []
        seq_len = test_x.size(1)
        cache_seq_length = self._cache_seq_length(cache_params)
    
        for t in range(seq_len):
            current_input = test_x[:, t : t + 1, :]  # shape (batch, 1, dim)
            output, _ = self._run_fla(
                current_input,
                cache_params=self._copy_cache(cache_params),
                return_cache=False,
                use_custom_recurrent=use_custom_recurrent,
                use_custom_shortconv=use_custom_shortconv,
                deltanet_beta_decay_start=cache_seq_length + t,
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

        x_batched, shape_info = self._prepare_fla_input(x)
        seq_len = x_batched.size(1)
        train_len = min(single_eval_pos, seq_len)
        state_passing = self.state_passing if self.training else None
        initial_cache_params = (
            None
            if state_passing is None
            else state_passing.sample_initial_cache(
                x_batched.size(0),
                device=x_batched.device,
            )
        )
        if initial_cache_params is not None and isinstance(self.fla, DeltaNetModel):
            initial_cache_params = prepare_deltanet_cache_for_fla(initial_cache_params)

        use_split_path = self.sequence_mode in FLA_SPLIT_SEQUENCE_MODES or not self.training
        if use_split_path:
            train_out, state = self.incontext_fit(
                x_batched[:, :train_len],
                initial_cache_params=initial_cache_params,
            )
            test_out = self.incontext_predict(x_batched[:, train_len:], state)
            attn_out = torch.cat([train_out, test_out], dim=1)
            if state_passing is not None:
                state_passing.remember(state["cache_params"])
        elif self.state_weaving:
            if initial_cache_params is not None:
                raise ValueError("state_weaving does not support initial_cache_params.")
            attn_out, _ = self._run_state_weaving(x_batched)
        else:
            attn_out, cache_params = self._run_fla(
                x_batched,
                cache_params=initial_cache_params,
                return_cache=state_passing is not None,
            )
            if state_passing is not None:
                state_passing.remember(cache_params)

        return self._unprepare_fla_output(attn_out, shape_info)

class BidirectionalFLABackbone(FLABackbone):
    def __init__(
        self,
        fla_model: nn.Module,
        sequence_mode: str = "Comb_ST",
        cache_chunk_size: int | None = None,
        state_passing: bool = False,
        state_passing_dropout: float = 0.1,
        include_self_term: bool = True,
        state_fusion: str = "mean_output_mean_cache",
    ):
        super().__init__(
            fla_model=fla_model,
            sequence_mode=sequence_mode,
            cache_chunk_size=cache_chunk_size,
            state_passing=state_passing,
            state_passing_dropout=state_passing_dropout,
            include_self_term=include_self_term,
        )
        self.state_fusion = state_fusion

    def incontext_fit(
        self,
        train_x: torch.Tensor,
        *,
        initial_cache_params: tp.Any | None = None,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, tp.Any]:
        x_batched, shape_info = self._prepare_fla_input(train_x)
        if initial_cache_params is not None:
            raise ValueError("Bidirectional FLA does not support initial_cache_params.")

        train_out, cache_params = self._run_fla(x_batched, return_cache=True)
        if not isinstance(cache_params, BidirectionalFLACache):
            raise TypeError(
                "Bidirectional FLA train cache build must return a BidirectionalFLACache."
            )
        returned_cache: tp.Any = cache_params
        if _uses_fused_prediction_cache(self.state_fusion):
            returned_cache = FusedBidirectionalFLACache(
                cache=fuse_bidirectional_cache(cache_params, state_fusion=self.state_fusion),
                state_fusion=self.state_fusion,
            )
        out = self._unprepare_fla_output(train_out, shape_info)
        return out, {"cache_params": returned_cache}

    def _run_test_with_cache(
        self,
        test_x: torch.Tensor,
        cache_params: tp.Any,
        use_custom_recurrent: bool = True,
        use_custom_shortconv: bool = True,
    ) -> torch.Tensor:
        if test_x.numel() == 0:
            return test_x
        validate_bidirectional_cache(cache_params)

        batch_size, seq_len, embed_dim = test_x.shape
        supports_custom_recurrent = self._supports_custom_recurrent()

        def _run_chunk(chunk_x: torch.Tensor) -> torch.Tensor:
            chunk_len = chunk_x.size(1)
            hidden = chunk_x.contiguous().view(batch_size * chunk_len, 1, embed_dim)
            chunk_cache = prepare_bidirectional_cache(
                cache_params,
                chunk_len=chunk_len,
                use_custom_recurrent=use_custom_recurrent,
                supports_custom_recurrent=supports_custom_recurrent,
            )

            with ExitStack() as stack:
                for ctx in self._patch_contexts(
                    use_custom_recurrent=use_custom_recurrent,
                    use_custom_shortconv=use_custom_shortconv,
                ):
                    stack.enter_context(ctx)
                hidden = run_bidirectional_layers(
                    hidden,
                    layers=self.layers,
                    chunk_cache=chunk_cache,
                )
            if hasattr(self.fla, "norm"):
                hidden = self.fla.norm(hidden)
            return hidden.view(batch_size, chunk_len, embed_dim)

        if self.cache_chunk_size is None or seq_len <= self.cache_chunk_size:
            return _run_chunk(test_x)

        outputs = []
        for chunk_start in range(0, seq_len, self.cache_chunk_size):
            chunk_end = min(chunk_start + self.cache_chunk_size, seq_len)
            outputs.append(_run_chunk(test_x[:, chunk_start:chunk_end]))
        return torch.cat(outputs, dim=1)

    def _run_test_with_cache_naive(
        self,
        test_x: torch.Tensor,
        cache_params: tp.Any | None,
        use_custom_recurrent: bool = False,
        use_custom_shortconv: bool = False,
    ) -> torch.Tensor:
        validate_bidirectional_cache(cache_params)
        if test_x.numel() == 0:
            return test_x

        output_tokens = []
        for t in range(test_x.size(1)):
            current_input = test_x[:, t : t + 1, :]
            output_tokens.append(
                self._run_test_with_cache(
                    current_input,
                    self._copy_cache(cache_params),
                    use_custom_recurrent=use_custom_recurrent,
                    use_custom_shortconv=use_custom_shortconv,
                )
            )
        return torch.cat(output_tokens, dim=1)


@dataclass(frozen=True)
class LinearAttentionBackboneConfig(BackboneConfig):
    """Configuration for a Linear Attention backbone."""
    nlayers: int = 6
    nhead: int = 2
    mlp_hidden_dim: int = 200
    use_query_scale: bool = True
    recompute_layer: bool = False
    recompute_every_n_layers: int = 1
    use_final_norm: bool = False
    initializer_range: float = 0.02
    layer_kwargs: tp.Dict[str, base_config.BaseTypes] | None = None


    def create_backbone(
        self,
        ninp: int,
        attention_between_features: bool,
        **kwargs: tp.Any,
    ) -> Backbone:
        if attention_between_features:
            raise NotImplementedError(
                "LinearAttentionBackbone no longer supports attention_between_features."
            )
        layer_kwargs = dict(self.layer_kwargs or {})
        linear_attention_kwargs = {
            "d_model": ninp,
            "num_heads": self.nhead,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "use_query_scale": self.use_query_scale,
            **layer_kwargs,
        }
        layers = nn.ModuleList([
            LinearAttention(
                **linear_attention_kwargs,
            )
            for _ in range(self.nlayers)
        ])
        layers.apply(
            lambda module: init_linear_attention_weights_like_fla(
                module,
                initializer_range=self.initializer_range,
            )
        )
        final_norm = build_norm(
            ninp,
            enabled=self.use_final_norm,
            norm_type=str(layer_kwargs.get("norm_type", "rmsnorm")),
        )
        return LinearAttentionBackbone(
            layers,
            final_norm=final_norm,
            recompute_each_layer=self.recompute_layer,
            recompute_every_n_layers=self.recompute_every_n_layers,
        )


class LinearAttentionBackbone(Backbone):
    """Stack of LinearAttention layers as a backbone."""
    def __init__(
        self,
        layers: nn.ModuleList,
        *,
        final_norm: nn.Module | None = None,
        recompute_each_layer: bool = False,
        recompute_every_n_layers: int | None = 1,
    ):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm if final_norm is not None else nn.Identity()
        self.recompute_each_layer = bool(recompute_each_layer)
        self.recompute_every_n_layers = (
            None if recompute_every_n_layers is None else int(recompute_every_n_layers)
        )
        if self.recompute_every_n_layers is not None and self.recompute_every_n_layers <= 0:
            raise ValueError("recompute_every_n_layers must be >= 1")

    @staticmethod
    def _pack_recurrent_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
        if state.get("k_sum") is None:
            return state["kv_state"]
        return torch.cat([state["kv_state"], state["k_sum"].unsqueeze(-1)], dim=-1)

    @staticmethod
    def _unpack_recurrent_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        recurrent_state = state["recurrent_state"]
        if "k_sum" in state:
            return {
                "kv_state": recurrent_state,
                "k_sum": state["k_sum"],
            }
        return {
            "kv_state": recurrent_state[..., :-1],
            "k_sum": recurrent_state[..., -1],
        }

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
        return self.final_norm(out)

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
        cached_state = {
            "cache_params": SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        state={
                            "recurrent_state": state["kv_state"],
                            "k_sum": state["k_sum"],
                        }
                    )
                    for state in layer_states
                ]
            )
        }
        return self.final_norm(out), cached_state

    def incontext_predict(
        self,
        x: torch.Tensor,
        cached_state: tp.Any,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        out = x
        cache_params = cached_state.get("cache_params")
        if cache_params is not None:
            layer_states = [
                self._unpack_recurrent_state(layer.state)
                for layer in cache_params.layers
            ]
        else:
            layer_states = cached_state.get("layer_states", [])
        for layer, state in zip(self.layers, layer_states):
            out = layer.incontext_predict(out, state)
        return self.final_norm(out)


@dataclass(frozen=True)
class RebasedBackboneConfig(BackboneConfig):
    nlayers: int = 6
    mlp_hidden_dim: int = 200
    num_heads: int = 2
    recompute_layer: bool = False
    recompute_every_n_layers: int | None = 1
    use_final_norm: bool = False
    initializer_range: float = 0.02
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
                BasedLinearAttention(
                    d_model=ninp,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    **(self.layer_kwargs or {}),
                )
                for _ in range(self.nlayers)
            ]
        )
        layers.apply(
            lambda module: init_linear_attention_weights_like_fla(
                module,
                initializer_range=self.initializer_range,
            )
        )
        final_norm = build_norm(
            ninp,
            enabled=self.use_final_norm,
            norm_type=str((self.layer_kwargs or {}).get("norm_type", "rmsnorm")),
        )
        return LinearAttentionBackbone(
            layers,
            final_norm=final_norm,
            recompute_each_layer=self.recompute_layer,
            recompute_every_n_layers=self.recompute_every_n_layers,
        )
