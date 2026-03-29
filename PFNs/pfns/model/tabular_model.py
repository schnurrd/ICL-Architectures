from __future__ import annotations

import random
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import einops
import torch

from pfns.model.encoders import (
    get_linear_x_encoder,
    get_linear_y_encoder,
    SequentialEncoder,
)
from pfns.model.layer import PerFeatureLayer
from torch import nn
from torch.utils.checkpoint import checkpoint

DEFAULT_EMSIZE = 128


def _iter_tensors(obj: Any) -> list[torch.Tensor]:
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, dict):
        tensors: list[torch.Tensor] = []
        for v in obj.values():
            tensors.extend(_iter_tensors(v))
        return tensors
    if isinstance(obj, (list, tuple)):
        tensors = []
        for v in obj:
            tensors.extend(_iter_tensors(v))
        return tensors
    if hasattr(obj, "__dict__"):
        return _iter_tensors(vars(obj))
    return []


def estimate_state_size_bytes(state: Any) -> int:
    total = 0
    for t in _iter_tensors(state):
        total += t.numel() * t.element_size()
    return total


@dataclass
class InContextState:
    backbone_state: Any

    def size_bytes(self) -> int:
        return estimate_state_size_bytes(self.backbone_state)


class TabularModel(nn.Module):
    """A Transformer model processes a token per feature and sample.

    This model extends the standard Transformer architecture to operate on a
    per-feature basis.
    It allows for processing each feature separately while still leveraging the
    power of self-attention.

    The model consists of an encoder, decoder, and optional components such
    as a feature positional embedding and a separate decoder for each feature.
    """

    # TODO: Feel like this could be simplified a lot from this part downwards
    def __init__(  # noqa: C901, D417, PLR0913
        self,
        *,
        transformer_layers: nn.Module,
        encoder: nn.Module | None = None,
        ninp: int = DEFAULT_EMSIZE,
        nhid: int = DEFAULT_EMSIZE * 4,
        y_encoder: nn.Module | None = None,
        decoder_dict: (
            dict[str, tuple[Callable[[int, int, int], nn.Module] | None, int]] | None
        ) = None,
        features_per_group: int = 1,
        feature_positional_embedding: (
            Literal[
                "normal_rand_vec",
                "uni_rand_vec",
                "learned",
                "subspace",
            ]
            | None
        ) = None,
        cache_trainset_representation: bool = False,
        seed: int | None = 0,
        style_encoder: nn.Module | None = None,
        y_style_encoder: nn.Module | None = None,
        attention_between_features: bool = True,
        batch_first: bool = True,
        interleave_x_y_pairs: bool = False,
    ):
        """Initializes the TabularModel (formerly PerFeatureTransformer).

        Args:
            transformer_layers: Pre-built backbone module (REQUIRED).
                This should be a Backbone instance (e.g., TransformerBackbone, MambaBackbone)
                created via BackboneConfig.create_backbone().
                The backbone processes embedded sequences through its forward() method.
            encoder:
                Pass a nn.Module that takes in a batch of sequences of inputs and
                returns something of the shape (seq_len, batch_size, ninp)
            ninp: Input dimension, also called the embedding dimension
            nhid: Hidden dimension in the MLP decoders
            y_encoder:
                A nn.Module that takes in a batch of sequences of outputs and
                returns something of the shape (seq_len, batch_size, ninp)
            decoder_dict: A mapping from output keys to a tuple of a decoder model and the number of output neurons.
                The number of output neurons for 10-way classification is 10 for example, and for regression with a bar distribution
                with 1000 buckets, it is 1000.
                If the decoder model is None, an MLP decoder is used, if one wants to specify it, it has the signature:
                    decoder_model(
                        ninp,
                        nhid,
                        decoder_n_out,
                    )
            features_per_group:
                If > 1, the features will be grouped into groups of this
                size and the attention is across groups.
            feature_positional_embedding:
                There is a risk that our models confuse
                features with each other. This positional embedding is added to the
                features to help the model distinguish them.
                We recommend setting this to "subspace".
            cache_trainset_representation: Whether to cache the training set representation for faster inference.
            seed: The seed to use for the random embeddings that identify features.
            style_encoder: A nn.Module that per dataset takes in a single style vector (batch_size, -1)
                or one style vector per feature (batch_size, num_features, -1) and returns a style embedding of the shape (batch_size, ninp)
            y_style_encoder: A nn.Module that per dataset takes in a single style vector (batch_size, -1) and returns a style embedding of the shape (batch_size, ninp)
            attention_between_features: If True, apply attention between feature groups. If False, use the old PFN architecture, see https://github.com/automl/TransformersCanDoBayesianInference
            batch_first: If True, then the input and output tensors are provided
                as (batch, seq, feature). Default is True. If False,
                (seq, batch, feature).
            interleave_x_y_pairs: If True, then the training set part of the sequence is interleaved
                token-wise as (x1, y1, x2, y2, ...). The test set part is also
                interleaved; missing test targets are encoded from NaN placeholders.
        """
        if decoder_dict is None:
            decoder_dict = {"standard": (None, 1)}

        super().__init__()

        if encoder is None:
            print("Using linear x encoder, as no encoder was provided.")
            encoder = get_linear_x_encoder(ninp, features_per_group)

        if y_encoder is None:
            print("Using linear y encoder, as no y_encoder was provided.")
            y_encoder = get_linear_y_encoder(ninp)

        self.encoder = encoder
        self.y_encoder = y_encoder
        self.ninp = ninp
        self.nhid = nhid
        self.features_per_group = features_per_group
        self.cache_trainset_representation = cache_trainset_representation
        self.cached_embeddings: torch.Tensor | None = None
        self.attention_between_features = attention_between_features
        self.batch_first = batch_first
        self.interleave_x_y_pairs = interleave_x_y_pairs

        assert transformer_layers is not None, "Must provide pre-built transformer_layers for TabularModel."
        self.transformer_layers = transformer_layers
        self.backbone = transformer_layers
        
        # Register hook for backward compatibility with checkpoint formats.
        # Handles old: transformer_layers.layers.X <-> new: transformer_layers.layer_stack.layers.X
        self._register_load_state_dict_pre_hook(self._remap_old_checkpoint_keys)
        
        initialized_decoder_dict = {}
        for decoder_key in decoder_dict:
            decoder_model, decoder_n_out = decoder_dict[decoder_key]
            if decoder_model is None:
                initialized_decoder_dict[decoder_key] = nn.Sequential(
                    nn.Linear(ninp, nhid),
                    nn.GELU(),
                    nn.Linear(nhid, decoder_n_out),
                )
            else:
                initialized_decoder_dict[decoder_key] = decoder_model(
                    ninp,
                    nhid,
                    decoder_n_out,
                )
        self.decoder_dict = nn.ModuleDict(initialized_decoder_dict)

        self.feature_positional_embedding = feature_positional_embedding
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        self.cached_feature_positional_embeddings: torch.Tensor | None = None

        self.seed = seed
        
        self.style_encoder = style_encoder
        if y_style_encoder is not None:
            assert attention_between_features, "Attention between features must be True when using a y_style_encoder, otherwise only use a style_encoder."
        self.y_style_encoder = y_style_encoder

    def _remap_old_checkpoint_keys(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Pre-hook to remap checkpoint keys for backward compatibility."""
        expects_new = hasattr(self.transformer_layers, "layer_stack")
        expects_old = hasattr(self.transformer_layers, "layers") and not expects_new

        # Support both legacy LayerStack naming conventions on both module aliases.
        for module_name in ("transformer_layers", "backbone"):
            old_prefix = prefix + f"{module_name}.layers."
            new_prefix = prefix + f"{module_name}.layer_stack.layers."

            has_old = any(k.startswith(old_prefix) for k in state_dict.keys())
            has_new = any(k.startswith(new_prefix) for k in state_dict.keys())

            if has_old and expects_new:
                keys_to_remap = [k for k in state_dict.keys() if k.startswith(old_prefix)]
                for old_key in keys_to_remap:
                    new_key = old_key.replace(old_prefix, new_prefix, 1)
                    state_dict[new_key] = state_dict.pop(old_key)
            elif has_new and expects_old:
                keys_to_remap = [k for k in state_dict.keys() if k.startswith(new_prefix)]
                for new_key in keys_to_remap:
                    old_key = new_key.replace(new_prefix, old_prefix, 1)
                    state_dict[old_key] = state_dict.pop(new_key)

        # Keep transformer/backbone alias checkpoints interoperable.
        transformer_prefix = prefix + "transformer_layers."
        backbone_prefix = prefix + "backbone."
        has_transformer = any(k.startswith(transformer_prefix) for k in state_dict.keys())
        has_backbone = any(k.startswith(backbone_prefix) for k in state_dict.keys())

        if has_transformer and not has_backbone:
            for key in [k for k in state_dict.keys() if k.startswith(transformer_prefix)]:
                alias_key = key.replace(transformer_prefix, backbone_prefix, 1)
                state_dict[alias_key] = state_dict[key]
        elif has_backbone and not has_transformer:
            for key in [k for k in state_dict.keys() if k.startswith(backbone_prefix)]:
                alias_key = key.replace(backbone_prefix, transformer_prefix, 1)
                state_dict[alias_key] = state_dict[key]

    def _prepare_batch_first_inputs(
        self,
        x: torch.Tensor | None,
        y: torch.Tensor | None,
        test_x: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        
        # Prepare batch-first versions of x, y, test_x for _forward
        # and clone all to be sure not to change outside data
        x_bf = x.clone() if x is not None else None
        y_bf = y.clone() if y is not None else None
        test_x_bf = test_x.clone() if test_x is not None else None

        if not self.batch_first:
            if x_bf is not None:
                x_bf = x_bf.transpose(0, 1)
            if y_bf is not None:
                # Ensure y_bf is a tensor before transposing. _forward will handle dict conversion.
                if y_bf.numel() > 0:
                    y_bf = y_bf.transpose(0, 1)
            if test_x_bf is not None:
                test_x_bf = test_x_bf.transpose(0, 1)

        # Now x_bf, y_bf, test_x_bf are batch-first (or None)
        return x_bf, y_bf, test_x_bf

    def _is_per_feature_transformer(self) -> bool:
        """Returns whether the backbone is the PerFeature transformer stack."""
        layers = None
        if isinstance(self.transformer_layers, LayerStack):
            layers = self.transformer_layers.layers
        else:
            layer_stack = getattr(self.transformer_layers, "layer_stack", None)
            if layer_stack is not None:
                layers = getattr(layer_stack, "layers", None)
            if layers is None:
                layers = getattr(self.transformer_layers, "layers", None)
        if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
            return all(isinstance(layer, PerFeatureLayer) for layer in layers)
        return False

    def incontext_fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        style: torch.Tensor | None = None,
        y_style: torch.Tensor | None = None,
        categorical_inds: list[int] | None = None,
    ) -> InContextState:
        """Run the training context through the model and return cached state."""
        assert style is None and y_style is None, (
            "incontext_fit currently does not support style/y_style. "
            "Please use style=None and y_style=None."
        )
        x_bf, y_bf, _ = self._prepare_batch_first_inputs(x, y, None)
        assert x_bf is not None and y_bf is not None
        single_eval_pos = y_bf.shape[1]

        embedded_input, _, should_interleave, _, backbone_kwargs = self._build_embedded_input(
            x_bf,
            y_bf,
            single_eval_pos=single_eval_pos,
            style=style,
            y_style=y_style,
            categorical_inds=categorical_inds,
            # Keep encoder preprocessing state (e.g. NaN replacement stats)
            # for the matching incontext_predict call.
            cache_trainset_representation=True,
        )

        _, backbone_state = self.transformer_layers.incontext_fit(
            embedded_input,
            rope_pairwise_positions=should_interleave,
            **backbone_kwargs,
        )
        return InContextState(backbone_state=backbone_state)

    def incontext_predict(
        self,
        state: InContextState,
        test_x: torch.Tensor,
        *,
        style: torch.Tensor | None = None,
        y_style: torch.Tensor | None = None,
        categorical_inds: list[int] | None = None,
        only_return_standard_out: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Run the test inputs using a cached state from incontext_fit."""
        assert style is None and y_style is None, (
            "incontext_predict currently does not support style/y_style. "
            "Please use style=None and y_style=None."
        )

        x_bf, _, _ = self._prepare_batch_first_inputs(test_x, None, None)
        assert x_bf is not None

        embedded_input, current_context_len, should_interleave, Int_MT_mode, backbone_kwargs = self._build_embedded_input(
            x_bf,
            None,
            # single_eval_pos=None signals pure test-time transform while
            # reusing train-time cached encoder preprocessing state.
            single_eval_pos=None,
            style=style,
            y_style=y_style,
            categorical_inds=categorical_inds,
            cache_trainset_representation=True,
        )

        encoder_out = self.transformer_layers.incontext_predict(
            embedded_input,
            state.backbone_state,
            rope_pairwise_positions=should_interleave,
            **backbone_kwargs,
        )

        output_decoded = self._decode_from_encoder_out(
            encoder_out,
            current_context_len,
            should_interleave,
            Int_MT_mode,
        )
        
        if not self.batch_first:
            for key, value in output_decoded.items():
                output_decoded[key] = value.transpose(0, 1)
        
        if only_return_standard_out:
            return output_decoded["standard"]
        return output_decoded

    def forward(
        self,
        x: torch.Tensor | None,
        y: torch.Tensor | None,
        test_x: torch.Tensor | None = None,
        style: torch.Tensor | None = None,
        y_style: torch.Tensor | None = None,
        only_return_standard_out: bool = True,
        single_eval_pos: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:  # noqa: D417
        """
        x can either contain both the train and test part, or the test part can be passed as test_x.

        Args:
            x: The input data for the training set, or both the train and test part if test_x is None.
                Shape: (batch_size, seq_len_train | seq_len_train + seq_len_test, num_features) if batch_first=True,
                else (seq_len_train | seq_len_train + seq_len_test, batch_size, num_features).
                When predicting from cached trainset representations, x can be None or contain the test set.
            y: The target data for the training set, where num targets is typically 1. In which case the last dimension can be omitted.
                Shape: (batch_size, seq_len_train | seq_len_train + seq_len_test, num_targets) if batch_first=True,
                else (seq_len_train | seq_len_train + seq_len_test, batch_size, num_targets).
                If y is None, we perform predictions for the test set using cached trainset representations.
            test_x: The input data for the test set.
                Shape: (batch_size, seq_len_test, num_features) if batch_first=True,
                else (seq_len_test, batch_size, num_features).
                When predicting from cached trainset representations, test_x can be None (using x instead) or contain the test set.
            style: (batch_size, style_dim) or (batch_size, num_features, style_dim) The style vector. Assumed batch-first.
            y_style: (batch_size, style_dim) The style vector for the y data. Assumed batch-first.
            only_return_standard_out: If True, only the standard output is returned.
            single_eval_pos: The position in the sequence where the training data ends and the test data begins.
            **kwargs: Keyword arguments passed to the `_forward` method:
                - `categorical_inds`: The indices of categorical features. A single list of indices for the whole batch:
                    these are shared between the datasets within a batch.
                - `half_layers`: Whether to use the first half of the layers.
        """

        x_bf, y_bf, test_x_bf = self._prepare_batch_first_inputs(x, y, test_x)
        # Now x_bf, y_bf, test_x_bf are batch-first (or None)

        # Determine single_eval_pos based on the original y shape
        if y_bf is not None and single_eval_pos is None:
            single_eval_pos = y_bf.shape[1]

        # Handle cache_trainset_representation and combining x, test_x
        if self.cache_trainset_representation and y is None:
            assert (
                (test_x is None) != (x is None)
            ), "Provide the test inputs only via test_x or x, not both, when cache_trainset_representation is True"
            if test_x is not None:
                x_bf = test_x_bf
        else:
            assert (
                x_bf is not None
            ), "x must be provided when not predicting from cached trainset representations"
            assert (
                y is not None
            ), "y must be provided when not predicting from cached trainset representations"

            if test_x_bf is not None:
                # x_bf and test_x_bf are batch-first. Concatenate along sequence dim (1).
                assert (
                    x_bf.shape[1] == single_eval_pos
                ), f"Batch-first x sequence length {x_bf.shape[1]} must match single_eval_pos {single_eval_pos} for concatenation"
                x_bf = torch.cat((x_bf, test_x_bf), dim=1)

        # Call _forward with batch-first tensors
        output_decoded = self._forward(
            x_bf,
            y_bf,
            single_eval_pos=single_eval_pos,  # This is the length of the training part of the sequence
            style=style,  # style is assumed batch-first from input
            y_style=y_style,  # y_style is assumed batch-first from input
            **kwargs,  # contains only_return_standard_out, categorical_inds, half_layers
        )

        # If original input was sequence-first, transpose outputs back
        if not self.batch_first:
            for key, value in output_decoded.items():
                output_decoded[key] = value.transpose(0, 1)
        if only_return_standard_out:
            output_decoded = output_decoded["standard"]

        return output_decoded


    def _build_embedded_input(
        self,
        x: torch.Tensor | dict,
        y: torch.Tensor | dict | None,
        *,
        single_eval_pos: int | None,
        style: torch.Tensor | None,
        y_style: torch.Tensor | None,
        categorical_inds: list[int] | None,
        cache_trainset_representation: bool,
    ) -> tuple[torch.Tensor, int, bool, bool, dict[str, torch.Tensor]]:
        current_context_len = single_eval_pos or 0

        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:  # x is a tensor
            x = {"main": x}
        # x is now a dict of batch-first tensors: x[k] is (batch_size, seq_len, features)

        _batch_size, _seq_len, _num_features_orig_main = x["main"].shape

        if (
            y is None
        ):  # Should only happen if cache_trainset_representation and not single_eval_pos
            y_main_ref = x["main"]
            y = {
                "main": torch.zeros(
                    _batch_size,
                    0,
                    device=y_main_ref.device,
                    dtype=y_main_ref.dtype,
                )
            }  # 0 sequence length
        elif isinstance(y, torch.Tensor):  # y is a tensor
            y = {"main": y}
        # y is now a dict of batch-first tensors: y[k] is (batch_size, seq_len_y, targets)

        # Pad features of x to be multiple of features_per_group
        for k in x:
            # x[k] is (batch_size, seq_len, num_features_k)
            num_features_k = x[k].shape[2]
            missing_to_next = (
                self.features_per_group - (num_features_k % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            x[k].shape[0],  # batch_size
                            x[k].shape[1],  # seq_len
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,  # Pad along feature dimension
                )
                if style is not None and style.ndim == 3 and k == "main":
                    style = torch.cat(
                        (
                            style,
                            torch.zeros(
                                style.shape[0],  # batch_size
                                missing_to_next,  # Padding for feature dimension
                                style.shape[2],  # style_dim
                                device=style.device,
                                dtype=style.dtype,
                            ),
                        ),
                        dim=1,  # Pad along style's feature dimension (dim 1)
                    )

        # Splits up the input into subgroups (batch-first)
        # x[k] from (batch_size, seq_len, num_features_padded) to (batch_size, seq_len, num_groups, features_per_group)
        for k in x:
            x[k] = einops.rearrange(
                x[k],
                "b s (f n) -> b s f n",
                n=self.features_per_group,
            )

        num_groups_main = x["main"].shape[2]  # Number of feature groups in x["main"]

        if style is not None:
            if style.ndim == 3:  # (batch_size, num_features_style_padded, style_dim)
                batched_style = einops.rearrange(
                    style,
                    "b (f n) s_dim -> (b f) n s_dim",
                    n=self.features_per_group,
                )
            else:  # style.ndim == 2, (batch_size, style_dim)
                assert style.ndim == 2
                batched_style = einops.repeat(
                    style, "b s_dim -> (b f) s_dim", f=num_groups_main
                )
        else:
            batched_style = None

        # We have to re-work categoricals based on the subgroup they fall into.
        categorical_inds_to_use: list[list[int]] | None = None
        if categorical_inds is not None:
            new_categorical_inds = []
            n_subgroups = x["main"].shape[2]

            for subgroup in range(n_subgroups):
                subgroup_lower = subgroup * self.features_per_group
                subgroup_upper = (subgroup + 1) * self.features_per_group
                subgroup_indices = [
                    i - subgroup_lower
                    for i in categorical_inds
                    if subgroup_lower <= i < subgroup_upper
                ]
                new_categorical_inds.append(subgroup_indices)

            categorical_inds_to_use = new_categorical_inds

        for k in y:
            # y[k] is (batch_size, current_seq_len_y, num_targets_y)
            if y[k].ndim == 2:  # (B,S) or (B,T)
                y[k] = y[k].unsqueeze(-1)  # B S -> B S 1

            # Pad y sequence length if shorter than x's sequence length (_seq_len)
            if y[k].shape[1] < _seq_len:  # _seq_len is full sequence length from x
                # current_context_len is the length of the training part of y
                assert (
                    y[k].shape[1]
                    == current_context_len  # y should only contain train part if shorter
                    or y[k].shape[1] == _seq_len  # Should not happen if already shorter
                ), f"y[{k}] seq len {y[k].shape[1]} vs train_seq_len {current_context_len} vs x_seq_len {_seq_len}"

                # Only pad if y is for training part or not main y (auxiliary targets might be full length)
                if k != "main" or y[k].shape[1] == current_context_len:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],  # batch_size
                                _seq_len - y[k].shape[1],  # seq_len difference
                                y[k].shape[2],  # num_targets_y
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,  # Pad along sequence dimension (dim 1 for batch-first)
                    )
        # Now y[k] is (batch_size, _seq_len, num_targets_y)

        # Making sure no label leakage ever happens for y["main"] (batch-first indexing)
        # current_context_len is the length of the training data part
        if "main" in y and y["main"].shape[1] > current_context_len:
            if not (
                (getattr(self.transformer_layers, "sequence_mode", None) == "Int_MT"
                and self.transformer_layers.training)
            ):
                y["main"][:, current_context_len:] = torch.nan

        # Prepare y for y_encoder (transpose to sequence-first if y_encoder expects it)
        y_for_y_encoder = {}
        for k_enc, v_enc in y.items():
            y_for_y_encoder[k_enc] = v_enc.transpose(0, 1)  # B S T -> S B T

        embedded_y = self.y_encoder(
            y_for_y_encoder,
            single_eval_pos=current_context_len,  # Length of training part for y_encoder
            cache_trainset_representation=cache_trainset_representation,
        ).transpose(0, 1)

        del y, y_for_y_encoder
        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )

        extra_encoders_args = {}
        if categorical_inds_to_use is not None and isinstance(
            self.encoder,
            SequentialEncoder,
        ):
            extra_encoders_args["categorical_inds"] = categorical_inds_to_use

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")

        embedded_x = einops.rearrange(
            self.encoder(
                x,
                single_eval_pos=current_context_len,
                cache_trainset_representation=cache_trainset_representation,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )  # b s f 1 -> b s f e
        del x
        
        if torch.isnan(embedded_x).any():
            raise ValueError(
                f"{torch.isnan(embedded_x).any()=}, make sure to add nan handlers"
                " to the xs that are not fully provided (test set missing)",
            )

        embedded_x, embedded_y = self.add_embeddings(
            embedded_x,  # (b s num_groups e)
            embedded_y,  # (b s e)
            num_features=_num_features_orig_main,
            seq_len=_seq_len,
            cache_embeddings=(cache_trainset_representation and single_eval_pos is not None),
            use_cached_embeddings=(cache_trainset_representation and single_eval_pos is None),
        )
        
        if torch.isnan(embedded_x).any() or torch.isnan(embedded_y).any():
            raise ValueError(
                f"There should be no NaNs in the embedded x and y after adding positional or style embeddings."
                "Check that your embedding layers do not produce NaNs for the given inputs."
                f"Your embedded x and y at this point are the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_y).any()=}",
            )

        Int_MT_mode = (
            getattr(self.transformer_layers, "sequence_mode", None) == "Int_MT"
        )
        should_interleave = Int_MT_mode or self.interleave_x_y_pairs

        if should_interleave and self.attention_between_features:
            raise ValueError(
                "Teacher forcing or interleaved x/y pairs requires attention_between_features=False."
            )
        if should_interleave and (style is not None or y_style is not None):
            raise ValueError(
                "Teacher forcing or interleaved x/y pairs does not support style/y_style embeddings yet."
            )

        if self.attention_between_features:
            # b s f e + b s 1 e -> b s f+1 e
            embedded_input = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)
        else:
            assert (
                embedded_x.shape[2] == 1
            ), f"Only 1 feature per group supported for attention_between_features=False, got {embedded_x.shape=}."

            if should_interleave:
                if self._is_per_feature_transformer() or (Int_MT_mode and self.transformer_layers.training):
                    embedded_y_tokens = embedded_y.unsqueeze(2)
                    embedded_input = torch.stack(
                        (embedded_x, embedded_y_tokens), dim=2
                    ).reshape(embedded_x.shape[0], -1, 1, embedded_x.shape[-1])
                else:
                    embedded_y_tokens = embedded_y[:, :current_context_len].unsqueeze(2)
                    embedded_input = torch.stack(
                        (embedded_x[:, :current_context_len], embedded_y_tokens), dim=2
                    ).reshape(embedded_x.shape[0], -1, 1, embedded_x.shape[-1])
                    embedded_input = torch.cat(
                        (embedded_input, embedded_x[:, current_context_len:]),
                        dim=1,
                    )
                current_context_len *= 2  # Each x token is followed by a y token
            else:
                # add them together in this case, like for the original PFNs
                # b s 1 e + b s 1 e -> b s 1 e
                embedded_input = embedded_x + embedded_y.unsqueeze(2)

        backbone_kwargs = self._build_backbone_projection_kwargs(
            embedded_x,
            embedded_y,
            should_interleave=should_interleave,
            style=style,
            y_style=y_style,
        )

        if style is not None:
            embedded_style = self.style_encoder(
                batched_style
            )  # (batch num_groups) style_dim | (batch num_groups) num_features style_dim -> (batch num_groups) emsize
            embedded_style = einops.rearrange(
                embedded_style, "(b f) e -> b 1 f e", b=_batch_size
            )  # (batch num_groups) emsize -> batch 1 num_groups emsize
        else:
            embedded_style = None

        if y_style is not None:
            embedded_y_style = self.y_style_encoder(
                y_style
            )  # batch style_dim -> batch emsize
            embedded_y_style = einops.rearrange(
                embedded_y_style, "b e -> b 1 1 e"
            )  # batch emsize -> batch 1 1 emsize
        else:
            embedded_y_style = None

        if embedded_style is not None or embedded_y_style is not None:
            if embedded_style is None:
                embedded_style = torch.zeros(
                    _batch_size,
                    1,  # Style is a single token in sequence dim
                    num_groups_main,
                    embedded_input.shape[3],  # emsize
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )

            if embedded_y_style is None:
                embedded_y_style = torch.zeros(
                    _batch_size,
                    1,
                    1,  # for the y-token
                    embedded_input.shape[3],  # emsize
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )

            full_embedded_style = torch.cat((embedded_style, embedded_y_style), dim=2)

            embedded_input = torch.cat(
                (full_embedded_style, embedded_input),
                dim=1,  # Concatenate along sequence dimension
            )
            current_context_len += 1  # Context length for attention now includes style

        if torch.isnan(embedded_input).any():
            raise ValueError(
                f"There should be no NaNs in the encoded x and y."
                "Check that you do not feed NaNs or use a NaN-handling enocder."
                "Your embedded x and y returned the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_y).any()=}",
            )
        del embedded_y, embedded_x

        return (
            embedded_input,
            current_context_len,
            should_interleave,
            Int_MT_mode,
            backbone_kwargs,
        )

    def _build_backbone_projection_kwargs(
        self,
        embedded_x: torch.Tensor,
        embedded_y: torch.Tensor,
        *,
        should_interleave: bool,
        style: torch.Tensor | None,
        y_style: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if (
            not getattr(self.transformer_layers, "supports_deltanet_input_projections", False)
            or should_interleave
            or self.attention_between_features
            or style is not None
            or y_style is not None
        ):
            return {}

        embedded_y_tokens = embedded_y.unsqueeze(2)
        return {
            "deltanet_qk_input": embedded_x,
            "deltanet_v_input": embedded_y_tokens,
            "deltanet_beta_input": embedded_x + embedded_y_tokens,
        }

    def _decode_from_encoder_out(
        self,
        encoder_out: torch.Tensor,
        current_context_len: int,
        should_interleave: bool,
        Int_MT_mode: bool,
    ) -> dict[str, torch.Tensor]:
        if should_interleave:
            if self._is_per_feature_transformer() or (
                Int_MT_mode and self.transformer_layers.training
            ):
                encoder_out = encoder_out[:, ::2]  # remove interleaved y tokens
            else:
                encoder_out = torch.cat(
                    (
                        encoder_out[:, :current_context_len:2],
                        encoder_out[:, current_context_len:],
                    ),
                    dim=1,
                )
            current_context_len = current_context_len // 2

        # current_context_len now marks the end of the training/style part in the sequence dimension
        # encoder_out is (batch, seq_with_style, num_tokens_incl_y, embed_dim)
        # We want the output for the y-token (last token in the feature/token dimension)
        # for the test sequence part (after current_context_len).

        test_encoder_out = encoder_out[
            :, current_context_len:, -1
        ]  # (batch, seq_test, embed_dim)
        train_encoder_out = encoder_out[
            :, :current_context_len, -1
        ]  # (batch, seq_train_and_style, embed_dim)

        # No transposition needed here as _forward returns batch-first

        output_decoded = (
            {k: v(test_encoder_out) for k, v in self.decoder_dict.items()}
            if self.decoder_dict is not None
            else {}
        )

        output_decoded["train_embeddings"] = train_encoder_out
        output_decoded["test_embeddings"] = test_encoder_out  # Already batch-first

        return output_decoded

    def _forward(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor | dict,  # Expected to be batch-first
        y: torch.Tensor | dict | None,  # Expected to be batch-first
        *,
        single_eval_pos: int
        | None = None,  # Length of the training part of the sequence
        style: torch.Tensor | None = None,  # Assumed batch-first
        y_style: torch.Tensor | None = None,  # Assumed batch-first
        categorical_inds: list[int] | None = None,
        half_layers: bool = False,
    ) -> Any | dict[str, torch.Tensor]:
        """The core forward pass of the model. Assumes batch-first inputs for x and y."""
        # Assertions and initial setup
        if self.cache_trainset_representation:
            if not single_eval_pos:  # none or 0
                assert (
                    y is None
                ), "_forward expects y=None if single_eval_pos is 0/None and caching"
        else:
            assert (
                y is not None
            ), "_forward expects y if not caching for pure inference or during training"
            assert (
                single_eval_pos is not None
            ), "_forward expects single_eval_pos if not caching for pure inference or during training"

        embedded_input, current_context_len, should_interleave, Int_MT_mode, backbone_kwargs = self._build_embedded_input(
            x,
            y,
            single_eval_pos=single_eval_pos,
            style=style,
            y_style=y_style,
            categorical_inds=categorical_inds,
            cache_trainset_representation=self.cache_trainset_representation,
        )

        encoder_out = self.transformer_layers(
            embedded_input,  # (b s_effective (num_groups+1_for_y) e)
            single_eval_pos=current_context_len,
            half_layers=half_layers,
            cache_trainset_representation=self.cache_trainset_representation,
            rope_pairwise_positions=should_interleave,
            **backbone_kwargs,
        )  # b s (num_groups+1_for_y) e -> b s (num_groups+1_for_y) e

        del embedded_input

        return self._decode_from_encoder_out(
            encoder_out,
            current_context_len,
            should_interleave,
            Int_MT_mode,
        )

    def add_embeddings(  # noqa: C901, PLR0912
        self,
        x: torch.Tensor,  # (b s num_groups e)
        y: torch.Tensor,  # (b s e)
        *,
        num_features: int,  # Original number of features (before grouping)
        seq_len: int,  # Sequence length
        cache_embeddings: bool = False,
        use_cached_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_cached_embeddings and self.cached_embeddings is not None:
            x += self.cached_embeddings[None, None]
            return x, y

        positional_embedding_rng = torch.Generator(device=x.device).manual_seed(self.seed)
        
        if self.feature_positional_embedding == "normal_rand_vec":
            embs = torch.randn(
                (x.shape[2], x.shape[3]),  # (num_groups, emsize)
                device=x.device,
                dtype=x.dtype,
                generator=positional_embedding_rng,
            )
            x += embs[None, None]  # Broadcast across batch and seq_len
        elif self.feature_positional_embedding == "uni_rand_vec":
            embs = (
                torch.rand(
                    (x.shape[2], x.shape[3]),  # (num_groups, emsize)
                    device=x.device,
                    dtype=x.dtype,
                    generator=positional_embedding_rng,
                )
                * 2
                - 1
            )
            x += embs[None, None]
        elif self.feature_positional_embedding == "learned":
            w = self.feature_positional_embedding_embeddings.weight
            embs = w[
                torch.randint(
                    0,
                    w.shape[0],
                    (x.shape[2],),  # num_groups indices
                    generator=positional_embedding_rng,
                )
            ]  # (num_groups, emsize)
            x += embs[None, None]
        elif self.feature_positional_embedding == "subspace":
            # x.shape[2] is num_groups, x.shape[3] is emsize
            # Generate (num_groups, emsize // 4) random vectors
            rand_vecs_for_subspace = torch.randn(
                (x.shape[2], x.shape[3] // 4),
                device=x.device,
                dtype=x.dtype,
                generator=positional_embedding_rng,
            )
            # print(f"beginning of rand_vecs_for_subspace:")
            # print(rand_vecs_for_subspace[:, 0])
            embs = self.feature_positional_embedding_embeddings(
                rand_vecs_for_subspace
            )  # (num_groups, emsize)
            x += embs[None, None]
        elif self.feature_positional_embedding is None:
            embs = None
        else:
            raise ValueError(f"Unknown {self.feature_positional_embedding=}")

        self.cached_embeddings = None
        if cache_embeddings and embs is not None:
            self.cached_embeddings = embs

        return x, y

    def empty_trainset_representation_cache(self) -> None:
        for layer in self.transformer_layers.layers:
            layer.empty_trainset_representation_cache()

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """Sets the save_peak_mem_factor for all layers.

        This factor controls how much memory is saved during the forward pass
        in inference mode.

        Setting this factor > 1 will cause the model to save more memory during
        the forward pass in inference mode.

        A value of 8 is good for a 4x larger width in the fully-connected layers.
        and yields a situation were we need around
        `2*num_features*num_items*emsize*2` bytes of memory

        for a forward pass (using mixed precision).

        WARNING: It should only be used with post-norm.

        Args:
            factor: The save_peak_mem_factor to set. Recommended value is 8.
        """
        for layer in self.transformer_layers.layers:
            assert hasattr(
                layer,
                "save_peak_mem_factor",
            ), "Layer does not have save_peak_mem_factor"
            layer.save_peak_mem_factor = factor  # type: ignore


### Utility functions


class LayerStack(nn.Module):
    """Same as nn.Sequential, but with support for passing keyword arguments
    to layers and stacks the same layer multiple times, which is passed as creater function.

    This is used as transformer encoder and decoder.
    """

    def __init__(
        self,
        *,
        layer_creator: Callable[[], nn.Module],
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        """
        Args:
            layer_creator: A function that returns the layer as a nn.Module.
            num_layers: The number of layers to stack.
            recompute_each_layer: If True, the layers will be recomputed on each
                forward pass in training. This is useful to save memory.
            min_num_layers_layer_dropout: If this is set, it enables to drop the last
                layers randomly during training up to this number.
        """
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        half_layers: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if half_layers:
            assert (
                self.min_num_layers_layer_dropout == self.num_layers
            ), "half_layers only works without layer dropout"
            n_layers = self.num_layers // 2
        elif self.training and self.min_num_layers_layer_dropout < self.num_layers:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()
        else:
            n_layers = self.num_layers

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)  # type: ignore
            else:
                x = layer(x, **kwargs)

        return x


### Utility functions


@contextmanager
def isolate_torch_rng(seed: int, device: torch.device) -> Generator[None, None, None]:
    """
    Use the specified seed within the context manager (`with isolate_torch_rng(...)`)
    and return to the original state after the context manager exits.
    """
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)


# Backward compatibility alias for old checkpoints and code
TableTransformer = TabularModel
