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


class SimpleFFNBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act in ("swish", "silu"):
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(attn_out)
        y = x = self.norm1(x)
        y = self.ff(y)
        return self.norm2(x + y)


@dataclass(frozen=True)
class FLABackboneConfig(BackboneConfig):
    """Configuration for Flash Linear Attention (FLA) based backbones."""
    
    model_type: tp.Literal["gla", "retnet"] = "gla"
    nlayers: int = 6
    nhead: int = 4
    intermediate_size: int | None = None  # defaults to 4*ninp if None
    dropout: float = 0.1
    activation: tp.Literal["gelu", "relu", "swish", "silu"] = "gelu"
    norm_eps: float = 1e-5 # 1e-6 lead to NaNs in the output of the GLA model
    use_cache: bool = False

    
    def __post_init__(self):
        if self.model_type not in FLA_MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type: {self.model_type}. Available: {list(FLA_MODEL_REGISTRY)}")

    def create_backbone(self, ninp: int, attention_between_features: bool, **kwargs: tp.Any) -> "Backbone":
        d_ff = self.intermediate_size or (4 * ninp)
        ConfigClass, ModelClass = FLA_MODEL_REGISTRY[self.model_type]
        
        assert attention_between_features == True, "FLA backbones currently do not support attention between features"

        config = ConfigClass(
            hidden_size=ninp,
            num_hidden_layers=self.nlayers,
            num_heads=self.nhead,
            intermediate_size=d_ff,
            hidden_act=self.activation,
            norm_eps=self.norm_eps,
            use_cache=self.use_cache,
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
        self.post = SimpleFFNBlock(d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)
    
    def _run_fla(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run the FLA model on input x.
        
        Args:
            x: Input tensor of shape (batch * num_tokens, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch * num_tokens, seq_len, embed_dim)
        """
        out = self.fla(inputs_embeds=x)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    
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
        assert cache_trainset_representation is False, "cache_trainset_representation not supported in FLA backbone"
        
        batch_size, seq_len, num_tokens, embed_dim = x.shape

        x_batched = x.transpose(1, 2).reshape(batch_size * num_tokens, seq_len, embed_dim)
        
        attn_out = self._run_fla(x_batched)
        out = self.post(x_batched, attn_out)
        
        out = out.reshape(batch_size, num_tokens, seq_len, embed_dim).transpose(1, 2)
        return out