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


class DropPath(nn.Module):
    """Stochastic depth regularization used by TabFlex-style blocks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(
            mask_shape, device=x.device, dtype=x.dtype
        )
        binary_mask = random_tensor.floor()
        return x.div(keep_prob) * binary_mask


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
    nhid: int = 512
    nlayers: int = 6
    nhead: int = 4
    intermediate_size: int | None = None  # Defaults to 4 * hidden_size
    activation: str = "swish"
    norm_eps: float = 1e-6
    use_cache: bool = False
    mix_tokens: bool = True  # Whether to mix information between feature tokens
    layout: tp.Literal["separate_tokens", "interleaved"] = "separate_tokens"
    token_mixer_type: tp.Literal["attention", "mlp", "none"] = "mlp"
    token_mixer_layers: int = 1
    token_mixer_dropout: float = 0.1
    token_mixer_mlp_factor: float = 2.0
    drop_path: float = 0.05
    feature_layer_norm: bool = True
    use_torch_compile: bool = False
    
    def __post_init__(self):
        if self.model_type not in FLA_MODEL_REGISTRY:
            available = list(FLA_MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown FLA model_type: {self.model_type}. "
                f"Available options: {available}"
            )
        if self.layout != "separate_tokens":
            raise ValueError("Interleaved layout is disabled; use 'separate_tokens'.")
        if self.token_mixer_type not in ("attention", "mlp", "none"):
            raise ValueError("token_mixer_type must be 'attention', 'mlp', or 'none'.")
        if self.token_mixer_layers < 0:
            raise ValueError("token_mixer_layers must be non-negative.")
        if not 0.0 <= self.drop_path < 1.0:
            raise ValueError("drop_path must be in [0, 1).")
    
    def create_backbone(
        self,
        ninp: int,
        attention_between_features: bool,
        **kwargs: tp.Any,
    ) -> Backbone:
        """Create the FLA backbone.
        
        Args:
            ninp: Input/embedding dimension (hidden_size for FLA)
            attention_between_features: Whether to apply attention between features.
                If False, processes each token independently.
            **kwargs: Additional arguments
            
        Returns:
            An FLABackbone wrapping the chosen FLA model
        """
        intermediate_size = self.intermediate_size or (ninp * 4)
        
        ConfigClass, ModelClass = FLA_MODEL_REGISTRY[self.model_type]
        
        config = ConfigClass(
            hidden_size=ninp,
            num_hidden_layers=self.nlayers,
            num_heads=self.nhead,
            intermediate_size=intermediate_size,
            hidden_act=self.activation,
            norm_eps=self.norm_eps,
            use_cache=self.use_cache,
        )
        model = ModelClass(config)
        
        return FLABackbone(
            model, 
            mix_tokens=self.mix_tokens and attention_between_features,
            layout=self.layout,
            token_mixer_type=self.token_mixer_type,
            token_mixer_layers=self.token_mixer_layers,
            token_mixer_dropout=self.token_mixer_dropout,
            token_mixer_mlp_factor=self.token_mixer_mlp_factor,
            drop_path=self.drop_path,
            feature_layer_norm=self.feature_layer_norm,
        )


class FLABackbone(Backbone):
    """Wrapper for FLA models to conform to Backbone interface.
    
    Processes each feature token independently through the FLA encoder and
    optionally applies column mixing. Interleaved layout is disabled for
    stability.
    """
    
    def __init__(
        self, 
        fla_model: nn.Module, 
        mix_tokens: bool = False,
        *,
        layout: str = "separate_tokens",
        token_mixer_type: str = "attention",
        token_mixer_layers: int = 2,
        token_mixer_dropout: float = 0.1,
        token_mixer_mlp_factor: float = 4.0,
        drop_path: float = 0.0,
        feature_layer_norm: bool = True,
    ):
        super().__init__()
        self.fla_model = fla_model
        self.mix_tokens = mix_tokens
        self.layout = layout
        self.token_mixer_type = "none" if not mix_tokens else token_mixer_type
        self.token_mixer_layers = token_mixer_layers
        self.feature_layer_norm = feature_layer_norm
        embed_dim = fla_model.config.hidden_size
        
        self.fla_encoder = fla_model.model if hasattr(fla_model, 'model') else fla_model
         
        self.token_mixers: nn.ModuleList | None = None
        self.token_mixer_norms: nn.ModuleList | None = None
        self.drop_paths: nn.ModuleList | None = None
        if self.token_mixer_type != "none" and token_mixer_layers > 0:
            token_mixers = []
            token_mixer_norms = []
            drop_paths = []
            drop_values = torch.linspace(0.0, drop_path, token_mixer_layers)
            for drop_value in drop_values:
                if self.token_mixer_type == "attention":
                    mixer = nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=fla_model.config.num_heads,
                        batch_first=True,
                        dropout=token_mixer_dropout,
                    )
                elif self.token_mixer_type == "mlp":
                    hidden = int(embed_dim * token_mixer_mlp_factor)
                    mixer = nn.Sequential(
                        nn.Linear(embed_dim, hidden),
                        nn.GELU(),
                        nn.Dropout(token_mixer_dropout),
                        nn.Linear(hidden, embed_dim),
                        nn.Dropout(token_mixer_dropout),
                    )
                else:
                    raise ValueError(f"Unsupported token_mixer_type: {self.token_mixer_type}")

                token_mixers.append(mixer)
                token_mixer_norms.append(
                    nn.LayerNorm(embed_dim) if feature_layer_norm else nn.Identity()
                )
                drop_paths.append(DropPath(float(drop_value)))

            self.token_mixers = nn.ModuleList(token_mixers)
            self.token_mixer_norms = nn.ModuleList(token_mixer_norms)
            self.drop_paths = nn.ModuleList(drop_paths)
        
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
        
        if self.layout == "separate_tokens":
            x_batched = x.transpose(1, 2).reshape(batch_size * num_tokens, seq_len, embed_dim)
            output_batched = self.fla_encoder(inputs_embeds=x_batched).last_hidden_state
            output = output_batched.reshape(batch_size, num_tokens, seq_len, embed_dim).transpose(1, 2)
        else:
            raise ValueError("Interleaved layout is disabled; use 'separate_tokens'.")

        if (
            self.token_mixers is not None
            and self.token_mixer_norms is not None
            and self.drop_paths is not None
        ):
            output = self._mix_tokens(output)
        
        return output

    def _mix_tokens(self, output: torch.Tensor) -> torch.Tensor:
        """Column-wise mixing of feature tokens."""
        batch_size, seq_len, num_tokens, embed_dim = output.shape
        tokens = output.reshape(batch_size * seq_len, num_tokens, embed_dim)

        for mixer, norm, drop_path in zip(
            self.token_mixers, self.token_mixer_norms, self.drop_paths
        ):
            residual = tokens
            tokens = norm(tokens)
            if self.token_mixer_type == "attention":
                tokens, _ = mixer(tokens, tokens, tokens, need_weights=False)
            else:
                tokens = mixer(tokens)
            tokens = residual + drop_path(tokens)

        return tokens.reshape(batch_size, seq_len, num_tokens, embed_dim)
