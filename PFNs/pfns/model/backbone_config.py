"""Configuration classes for different model backbones.

This module provides base classes and implementations for configuring
different transformer backbones that can be used within the ModelConfig.
"""
from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pfns import base_config
from pfns.model.layer import PerFeatureLayer
from pfns.model.transformer import LayerStack
from pfns.model.linear_attention import LinearAttention
import torch
from torch import nn

from fla.models import GLAConfig, GLAModel
from fla.models import RetNetConfig, RetNetModel
    
# Registry mapping model types to their config and model classes
FLA_MODEL_REGISTRY = {
    "gla": (GLAConfig, GLAModel),
    "retnet": (RetNetConfig, RetNetModel),
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

    model_type: tp.Literal["gla", "retnet"] = "gla"
    nlayers: int = 6
    nhead: int = 4
    intermediate_size: int | None = None
    dropout: float = 0.1
    activation: tp.Literal["gelu", "relu", "swish", "silu"] = "gelu"
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.model_type not in FLA_MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type: {self.model_type}. Available: {list(FLA_MODEL_REGISTRY)}")

    def create_backbone(self, ninp: int, attention_between_features: bool, **kwargs: tp.Any) -> "Backbone":
        d_ff = self.intermediate_size or (4 * ninp)
        ConfigClass, ModelClass = FLA_MODEL_REGISTRY[self.model_type]

        assert attention_between_features is False, (
            "FLA backbones currently do not support attention between features"
        )

        config = ConfigClass(
            hidden_size=ninp,
            num_hidden_layers=self.nlayers,
            num_heads=self.nhead,
            intermediate_size=d_ff,
            hidden_act=self.activation,
            norm_eps=self.norm_eps,
            use_cache=True,
        )
        fla_model = ModelClass(config)

        return FLABackbone(
            fla_model=fla_model,
            d_model=ninp,
            d_ff=d_ff,
            dropout=self.dropout,
            activation=self.activation,
        )


class FLABackbone(Backbone):
    """Wrapper for FLA models to conform to Backbone interface."""

    def __init__(
        self,
        fla_model: nn.Module,
        *,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.fla = fla_model.model if hasattr(fla_model, "model") else fla_model

    def _run_fla(
        self,
        x: torch.Tensor,
        *,
        past_key_values: tp.Any | None = None,
    ) -> tuple[torch.Tensor, tp.Any | None]:
        kwargs = {"inputs_embeds": x, "use_cache": True}
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        try:
            out = self.fla(**kwargs)
        except TypeError as exc:
            raise TypeError(
                "FLA model does not support cache usage; required for independent evaluation."
            ) from exc

        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            hidden = out

        past = None
        if hasattr(out, "past_key_values"):
            past = out.past_key_values
        elif isinstance(out, (tuple, list)) and len(out) > 1:
            past = out[1]

        return hidden, past

    def _run_test_with_cache(
        self,
        test_x: torch.Tensor,
        past_key_values: tp.Any | None,
    ) -> torch.Tensor:
        """
        Run the FLA model on test inputs using cached past key values in parallel.
        """
        if test_x.numel() == 0:
            return test_x

        if past_key_values is None:
            output, _ = self._run_fla(test_x)
            return output

        batch_size, seq_len, embed_dim = test_x.shape

        def _repeat_past(past: tp.Any, repeat: int) -> tp.Any:
            if torch.is_tensor(past):
                return past.repeat_interleave(repeat, dim=0)
            elif hasattr(past, "layers"):
                for layer in past.layers:
                    state = getattr(layer, "state", None)
                    if not isinstance(state, dict):
                        continue
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.repeat_interleave(repeat, dim=0)
                return past
            else:
                raise ValueError("Unsupported past_key_values structure for repetition.")

        expanded_past = _repeat_past(past_key_values, seq_len)
        test_x_flat = test_x.contiguous().view(batch_size * seq_len, 1, embed_dim)
        output, _ = self._run_fla(test_x_flat, past_key_values=expanded_past)
        output = output.view(batch_size, seq_len, embed_dim)
            
        return output
    
    def _run_test_with_cache_naive(
        self,
        test_x: torch.Tensor,
        past_key_values: tp.Any | None,
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
            output, _ = self._run_fla(
                current_input, past_key_values=past_key_values
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
        train_x = x_batched[:, :train_len]
        test_x = x_batched[:, train_len:]

        train_out, past = self._run_fla(train_x)
        if past is None:
            raise RuntimeError(
                "FLA model returned no past_key_values; cache is required for independent evaluation."
            )
            
        test_out = self._run_test_with_cache(test_x, past)
        attn_out = torch.cat([train_out, test_out], dim=1)

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
        if LinearAttention is None:
            raise ImportError("LinearAttention module not found. Please implement or install it in pfns.model.linear_attention.")

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
