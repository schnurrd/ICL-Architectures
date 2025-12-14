import typing as tp
from dataclasses import dataclass

from pfns import base_config
from pfns.model import encoders, transformer
from pfns.model.backbone_config import BackboneConfig, TransformerBackboneConfig
from pfns.model.bar_distribution import BarDistribution
from pfns.model.criterions import BarDistributionConfig, CrossEntropyConfig
from pfns.model.encoders import StyleEncoderConfig

from torch import nn


@dataclass(frozen=True)
class ModelConfig(base_config.BaseConfig):
    criterion: CrossEntropyConfig | BarDistributionConfig
    backbone: BackboneConfig = TransformerBackboneConfig()
    encoder: tp.Optional[encoders.EncoderConfig] = (
        None  # todo add back in as config, currently only supporting standard encoder
    )
    y_encoder: tp.Optional[encoders.EncoderConfig] = (
        None  # todo add back in as config, currently only supporting standard encoder
    )
    style_encoder: tp.Optional[StyleEncoderConfig] = None
    y_style_encoder: tp.Optional[StyleEncoderConfig] = None
    decoder_dict: tp.Dict[str, base_config.BaseTypes] | None = None
    emsize: int = 200
    features_per_group: int = 1 # number of features grouped together as one token
    attention_between_features: bool = True
    feature_positional_embedding: (
        tp.Literal[
            "normal_rand_vec",
            "uni_rand_vec",
            "learned",
            "subspace",
        ]
        | None
    ) = None,
    model_extra_args: tp.Dict[str, base_config.BaseTypes] | None = None
    
    # Legacy parameters for backward compatibility
    nhid: tp.Optional[int] = None
    nlayers: tp.Optional[int] = None
    nhead: tp.Optional[int] = None
    seed: int = 0

    def create_model(self) -> transformer.TabularModel:
        if self.nhid is not None or self.nlayers is not None or self.nhead is not None:
            backbone = TransformerBackboneConfig(
                nhid=self.nhid if self.nhid is not None else 200,
                nlayers=self.nlayers if self.nlayers is not None else 6,
                nhead=self.nhead if self.nhead is not None else 2,
            )
        else:
            backbone = self.backbone
        
        # Resolve criterion
        criterion = self.criterion.get_criterion()

        # Determine n_out based on the resolved criterion
        if isinstance(criterion, BarDistribution):
            n_out = criterion.num_bars
        elif isinstance(criterion, nn.CrossEntropyLoss):
            n_out = criterion.weight.shape[0]
        else:
            raise ValueError(f"Criterion {criterion} not supported")

        decoder_dict = (
            self.decoder_dict if self.decoder_dict else {"standard": (None, n_out)}
        )

        if self.encoder is not None:
            encoder = self.encoder.create_encoder(
                features=self.features_per_group, emsize=self.emsize
            )
        else:
            encoder = None

        if self.y_encoder is not None:
            y_encoder = self.y_encoder.create_encoder(features=1, emsize=self.emsize)
        else:
            y_encoder = None

        if self.style_encoder is not None:
            style_encoder = self.style_encoder.create_encoder(self.emsize)
        else:
            style_encoder = None

        if self.y_style_encoder is not None:
            y_style_encoder = self.y_style_encoder.create_encoder(self.emsize)
        else:
            y_style_encoder = None

        transformer_layers = backbone.create_backbone(
            ninp=self.emsize,
            attention_between_features=self.attention_between_features,
        )
        
        nhid = getattr(backbone, 'nhid', self.emsize * 4)

        model = transformer.TabularModel(
            encoder=encoder,
            transformer_layers=transformer_layers,
            y_encoder=y_encoder,
            features_per_group=self.features_per_group,
            decoder_dict=decoder_dict,
            ninp=self.emsize,
            nhid=nhid,
            attention_between_features=self.attention_between_features,
            style_encoder=style_encoder,
            y_style_encoder=y_style_encoder,
            batch_first=True,  # model is batch_first by default now
            feature_positional_embedding=self.feature_positional_embedding,
            seed=self.seed,  # Seed is important for reproducibility of feature positional embeddings
            **(self.model_extra_args or {}),
        )
        model.criterion = criterion
        return model


# Backwards compatibility alias for models saved with the old name
TransformerConfig = ModelConfig
