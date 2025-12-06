import os
import pathlib
import random
import itertools

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import (
    LabelEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pfns.scripts.tabpfn_model_builder import load_model_only_inference
from pfns.utils import (
    NOP,
    normalize_by_used_features_f,
    normalize_data,
    remove_outliers,
)


# =============================================================================
# Ensemble Configuration
# =============================================================================


@dataclass
class EnsembleConfig:
    """Configuration for a single ensemble member."""

    class_shift: int = 0
    feature_shift: int = 0
    transform_type: Literal[
        "none", "power", "power_all", "quantile", "quantile_all", "robust", "robust_all"
    ] = "none"
    max_features: int = 100


# =============================================================================
# Model Architecture Interface
# =============================================================================


class ModelBackbone(Protocol):
    """Protocol defining the interface for model backbones."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input features [seq_len, batch_size, features]
            y: Target labels [seq_len, batch_size]
            style: Optional style encoding

        Returns:
            Logits [seq_len, batch_size, num_classes]
        """
        ...

    def to(self, device: str) -> "ModelBackbone":
        """Move model to device."""
        ...

    def eval(self) -> "ModelBackbone":
        """Set model to evaluation mode."""
        ...


class TransformerBackbone:
    """Wrapper for the existing Transformer model to match the protocol."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass delegating to wrapped model."""
        x_bf = x.transpose(0, 1)
        y_bf = y.transpose(0, 1).float()
        style_bf = style.repeat(x_bf.shape[0], 1) if style is not None else None

        output = self.model(x=x_bf, y=y_bf, style=style_bf)
        return output.transpose(0, 1)

    def to(self, device: str):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self


# =============================================================================
# Inference Engine
# =============================================================================


class InferenceEngine:
    """Manages preprocessing, ensemble prediction, and aggregation."""

    def __init__(
        self,
        backbone: ModelBackbone,
        ensemble_configs: list[EnsembleConfig],
        num_classes: int,
        device: str = "cpu",
        softmax_temperature: float = 0.8,
        batch_size_inference: int = 32,
        average_logits: bool = True,
        extend_features: bool = True,
        fp16_inference: bool = False,
        no_grad: bool = True,
    ):
        self.backbone = backbone
        self.ensemble_configs = ensemble_configs
        self.num_classes = num_classes
        self.device = device
        self.softmax_temperature = softmax_temperature
        self.batch_size_inference = batch_size_inference
        self.average_logits = average_logits
        self.extend_features = extend_features
        self.fp16_inference = fp16_inference
        self.no_grad = no_grad

    def _get_sklearn_transformer(self, transform_type: str):
        """Get sklearn transformer based on config."""
        if transform_type == "none":
            return None
        elif transform_type in ["power", "power_all"]:
            return PowerTransformer(standardize=True)
        elif transform_type in ["quantile", "quantile_all"]:
            return QuantileTransformer(output_distribution="normal")
        elif transform_type in ["robust", "robust_all"]:
            return RobustScaler(unit_variance=True)
        return None

    def _preprocess_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        eval_position: int,
        transform_type: str,
        max_features: int,
    ) -> torch.Tensor:
        """Preprocess input data for one ensemble member."""
        X = normalize_data(X, normalize_positions=eval_position)

        # Remove constant features
        X = X[:, 0, :]
        sel = [
            len(torch.unique(X[0 : y.shape[0], col])) > 1 for col in range(X.shape[1])
        ]
        X = X[:, sel]

        # Apply sklearn transforms
        if transform_type != "none":
            import warnings

            X_np = X.cpu().numpy()
            feats = (
                set(range(X_np.shape[1]))
                if "all" in transform_type
                else set(range(X_np.shape[1]))
            )

            warnings.simplefilter("error")
            for col in feats:
                try:
                    transformer = self._get_sklearn_transformer(transform_type)
                    if transformer is not None:
                        transformer.fit(X_np[0:eval_position, col : col + 1])
                        trans = transformer.transform(X_np[:, col : col + 1])
                        X_np[:, col : col + 1] = trans
                except Exception:
                    pass
            warnings.simplefilter("default")
            X = torch.tensor(X_np).float()

        X = X.unsqueeze(1)
        X = remove_outliers(X, normalize_positions=eval_position)
        X = normalize_by_used_features_f(
            X, X.shape[-1], max_features, normalize_with_sqrt=False
        )

        # Subsample features if needed
        if X.shape[2] > max_features:
            X = X[
                :, :, sorted(np.random.choice(X.shape[2], max_features, replace=False))
            ]

        return X.to(self.device)

    def predict(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        eval_position: int,
        style: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Run inference with ensemble.

        Args:
            X: Input features [n_samples, batch_size, n_features]
            y: Target labels [n_samples, batch_size]
            eval_position: Split between train and test
            style: Optional style encoding
            return_logits: If True, return logits instead of probabilities

        Returns:
            Predictions [batch_size, n_test_samples, n_classes]
        """
        self.backbone.to(self.device)
        self.backbone.eval()

        y_train = y[:eval_position]

        inputs_list, labels_list = self._prepare_ensemble_inputs(
            X, y_train, eval_position
        )

        inputs_batched = torch.split(inputs_list, self.batch_size_inference, dim=1)
        labels_batched = torch.split(labels_list, self.batch_size_inference, dim=1)

        all_outputs = []
        for batch_input, batch_label in zip(inputs_batched, labels_batched):
            batch_output = self._forward_batch(batch_input, batch_label, style)
            all_outputs.append(batch_output)

        outputs = torch.cat(all_outputs, dim=1)

        final_output = self._aggregate_ensemble_outputs(outputs, return_logits)

        return final_output

    def _prepare_ensemble_inputs(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        eval_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess inputs for all ensemble members."""
        inputs = []
        labels = []

        # Cache preprocessed data by transform type
        preprocessed_cache = {}

        for config in self.ensemble_configs:
            transform_type = config.transform_type
            if transform_type in preprocessed_cache:
                X_processed = preprocessed_cache[transform_type].clone()
            else:
                X_processed = self._preprocess_data(
                    X.clone(), y, eval_position, transform_type, config.max_features
                )
                if self.no_grad:
                    X_processed = X_processed.detach()
                preprocessed_cache[transform_type] = X_processed

            y_shifted = ((y + config.class_shift) % self.num_classes).float()

            # Apply feature shift
            if config.feature_shift > 0:
                X_processed = torch.cat(
                    [
                        X_processed[..., config.feature_shift :],
                        X_processed[..., : config.feature_shift],
                    ],
                    dim=-1,
                )

            # Extend features to max_features if needed
            if self.extend_features:
                if X_processed.shape[2] < config.max_features:
                    padding = torch.zeros(
                        (
                            X_processed.shape[0],
                            X_processed.shape[1],
                            config.max_features - X_processed.shape[2],
                        )
                    ).to(self.device)
                    X_processed = torch.cat([X_processed, padding], dim=-1)

            inputs.append(X_processed)
            labels.append(y_shifted)

        return torch.cat(inputs, dim=1), torch.cat(labels, dim=1)

    def _forward_batch(
        self,
        batch_input: torch.Tensor,
        batch_label: torch.Tensor,
        style: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass for a batch."""
        import warnings
        from torch.utils.checkpoint import checkpoint

        inference_mode_ctx = torch.inference_mode() if self.no_grad else NOP()

        with inference_mode_ctx:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="None of the inputs have requires_grad=True",
                )
                warnings.filterwarnings(
                    "ignore",
                    message="torch.cuda.amp.autocast only affects CUDA ops",
                )

                if self.device == "cpu":
                    output = checkpoint(
                        self._forward_fn,
                        batch_input,
                        batch_label,
                        style,
                        use_reentrant=False,
                    )
                else:
                    with torch.amp.autocast("cuda", enabled=self.fp16_inference):
                        output = checkpoint(
                            self._forward_fn,
                            batch_input,
                            batch_label,
                            style,
                            use_reentrant=False,
                        )

        return output

    def _forward_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        style: torch.Tensor | None,
    ) -> torch.Tensor:
        """Actual forward function through the backbone."""
        output = self.backbone.forward(x=x, y=y, style=style)

        output = output[:, :, 0 : self.num_classes]

        output = output / self.softmax_temperature

        return output

    def _aggregate_ensemble_outputs(
        self,
        outputs: torch.Tensor,
        return_logits: bool,
    ) -> torch.Tensor:
        """Aggregate outputs from all ensemble members."""
        # Reshape to separate ensemble members
        # outputs shape: [n_samples, n_ensemble * batch, n_classes]
        n_ensemble = len(self.ensemble_configs)
        batch_size = outputs.shape[1] // n_ensemble

        ensemble_outputs = []
        for i, config in enumerate(self.ensemble_configs):
            output_i = outputs[:, i * batch_size : (i + 1) * batch_size, :]

            # Reverse class shift
            if config.class_shift > 0:
                output_i = torch.cat(
                    [
                        output_i[..., config.class_shift :],
                        output_i[..., : config.class_shift],
                    ],
                    dim=-1,
                )

            if not self.average_logits and not return_logits:
                output_i = torch.nn.functional.softmax(output_i, dim=-1)

            ensemble_outputs.append(output_i)

        # Average across ensemble
        aggregated = torch.stack(ensemble_outputs).mean(dim=0)

        if self.average_logits and not return_logits:
            aggregated = torch.nn.functional.softmax(aggregated, dim=-1)

        return aggregated.transpose(0, 1)


# =============================================================================
# Scikit-learn Compatible Classifier Interface
# =============================================================================

default_base_path = pathlib.Path(__file__).parent.parent.resolve()


class TabPFNClassifier(BaseEstimator, ClassifierMixin):
    """Refactored TabPFN Classifier with separated preprocessing and model backbone."""

    models_in_memory = {}

    def __init__(
        self,
        device: str = "cpu",
        base_path: Path = default_base_path,
        model_string: str = "",
        N_ensemble_configurations: int = 3,
        no_preprocess_mode: bool = False,
        multiclass_decoder: Literal["permutation", "none"] = "permutation",
        feature_shift_decoder: bool = True,
        only_inference: bool = True,
        seed: int = 0,
        no_grad: bool = True,
        batch_size_inference: int = 32,
        subsample_features: bool = False,
        backbone_type: str = "transformer",  # New: allows swapping backbones
    ):
        """
        Initializes the classifier and loads the model.
        Depending on the arguments, the model is either loaded from memory, from a file, or downloaded from the
        repository if no model is found.

        Can also be used to compute gradients with respect to the inputs X_train and X_test. Therefore no_grad has to be
        set to False and no_preprocessing_mode must be True. Furthermore, X_train and X_test need to be given as
        torch.Tensors and their requires_grad parameter must be set to True.

        :param device: If the model should run on cuda or cpu.
        :param base_path: Base path of the directory, from which the folders like models_diff can be accessed.
        :param model_string: Name of the model. Used first to check if the model is already in memory, and if not,
               tries to load a model with that name from the models_diff directory. It looks for files named as
               follows: "prior_diff_real_checkpoint" + model_string + "_n_0_epoch_e.cpkt", where e can be a number
               between 100 and 0, and is checked in a descending order.
        :param N_ensemble_configurations: The number of ensemble configurations used for the prediction. Thereby the
               accuracy, but also the running time, increases with this number.
        :param no_preprocess_mode: Specifies whether preprocessing is to be performed.
        :param multiclass_decoder: If set to permutation, randomly shifts the classes for each ensemble configuration.
        :param feature_shift_decoder: If set to true shifts the features for each ensemble configuration according to a
               random permutation.
        :param only_inference: Indicates if the model should be loaded to only restore inference capabilities or also
               training capabilities. Note that the training capabilities are currently not being fully restored.
        :param seed: Seed that is used for the prediction. Allows for a deterministic behavior of the predictions.
        :param batch_size_inference: This parameter is a trade-off between performance and memory consumption.
               The computation done with different values for batch_size_inference is the same,
               but it is split into smaller/larger batches.
        :param no_grad: If set to false, allows for the computation of gradients with respect to X_train and X_test.
               For this to correctly function no_preprocessing_mode must be set to true.
        :param subsample_features: If set to true and the number of features in the dataset exceeds self.max_features (100),
                the features are subsampled to self.max_features.
        :param backbone_type: Type of model backbone to use ('transformer', 'mamba', etc.)
        """
        self.device = device
        self.base_path = base_path
        self.model_string = model_string
        self.N_ensemble_configurations = N_ensemble_configurations
        self.no_preprocess_mode = no_preprocess_mode
        self.multiclass_decoder = multiclass_decoder
        self.feature_shift_decoder = feature_shift_decoder
        self.only_inference = only_inference
        self.seed = seed
        self.no_grad = no_grad
        self.batch_size_inference = batch_size_inference
        self.subsample_features = subsample_features
        self.backbone_type = backbone_type

        model_key = model_string + "|" + str(device)
        if model_key in self.models_in_memory:
            model, c, results_file = self.models_in_memory[model_key]
        else:
            model, c, results_file = load_model_workflow(
                name=model_string,
                base_path=base_path,
                device=device,
            )
            self.models_in_memory[model_key] = (model, c, results_file)
            if len(self.models_in_memory) == 2:
                print(
                    "Multiple models in memory. This might lead to memory issues. Consider calling remove_models_from_memory()"
                )

        self.model = model
        self.c = c

        self.max_num_features = c.batch_shape_sampler.max_num_features
        self.max_num_classes = c.model.criterion.num_classes

        assert (
            self.no_preprocess_mode if not self.no_grad else True
        ), "If no_grad is false, no_preprocess_mode must be true, because otherwise no gradient can be computed."

        self.backbone = self._create_backbone(model, backbone_type)

    def _create_backbone(
        self, model: torch.nn.Module, backbone_type: str
    ) -> ModelBackbone:
        """Factory method to create different backbone types."""
        if backbone_type == "transformer":
            return TransformerBackbone(model)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )
        self.classes_ = cls
        return np.asarray(y, dtype=np.float64, order="C")

    def fit(self, X, y, overwrite_warning=False):
        """
        Validates the training set and stores it.

        If clf.no_grad (default is True):
        X, y should be of type np.array
        else:
        X should be of type torch.Tensors (y can be np.array or torch.Tensor)
        """
        if self.no_grad:
            # Check that X and y have correct shape
            X, y = check_X_y(X, y, ensure_all_finite=False)
        # Store the classes seen during fit
        y = self._validate_targets(y)
        self.label_encoder = LabelEncoder()  # encodes y to 0,...,n_classes-1
        y = self.label_encoder.fit_transform(y)

        self.X_ = X
        self.y_ = y

        if X.shape[1] > self.max_num_features:
            if self.subsample_features:
                print(
                    "WARNING: The number of features for this classifier is restricted to ",
                    self.max_num_features,
                    " and will be subsampled.",
                )
            else:
                raise ValueError(
                    "The number of features for this classifier is restricted to ",
                    self.max_num_features,
                )
        if len(np.unique(y)) > self.max_num_classes:
            raise ValueError(
                "The number of classes for this classifier is restricted to ",
                self.max_num_classes,
            )
        if X.shape[0] > 1024 and not overwrite_warning:
            raise ValueError(
                "⚠️ WARNING: TabPFN is not made for datasets with a trainingsize > 1024. Prediction might take a while, be less reliable. We advise not to run datasets > 10k samples, which might lead to your machine crashing (due to quadratic memory scaling of TabPFN). Please confirm you want to run by passing overwrite_warning=True to the fit function."
            )

        # Return the classifier
        return self

    def predict_proba(self, X, return_logits=False):
        """
        Predict the probabilities for the input X depending on the training set previously passed in the method fit.

        If no_grad is true in the classifier the function takes X as a numpy.ndarray. If no_grad is false X must be a
        torch tensor and is not fully checked.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        if self.no_grad:
            X = check_array(X, ensure_all_finite=False)
            X_full = np.concatenate([self.X_, X], axis=0)
            X_full = torch.tensor(X_full, device=self.device).float().unsqueeze(1)
        else:
            assert torch.is_tensor(self.X_) & torch.is_tensor(X), (
                "If no_grad is false, this function expects X as "
                "a tensor to calculate a gradient"
            )
            X_full = torch.cat((self.X_, X), dim=0).float().unsqueeze(1).to(self.device)

            if int(torch.isnan(X_full).sum()):
                print(
                    "X contains nans and the gradient implementation is not designed to handel nans."
                )

        y_full = np.concatenate([self.y_, np.zeros(shape=X.shape[0])], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)

        eval_pos = self.X_.shape[0]
        num_classes = len(self.classes_)

        preprocess_transforms = (
            ["none"] if self.no_preprocess_mode else ["none", "power_all"]
        )

        # Generate ensemble configurations
        if self.seed is not None:
            torch.manual_seed(self.seed)

        feature_shifts = (
            torch.randperm(X_full.shape[2]).tolist()
            if self.feature_shift_decoder
            else [0]
        )
        class_shifts = (
            torch.randperm(num_classes).tolist()
            if self.multiclass_decoder == "permutation"
            else [0]
        )

        combinations = list(
            itertools.product(class_shifts, feature_shifts, preprocess_transforms)
        )
        rng = random.Random(self.seed)
        rng.shuffle(combinations)
        combinations = combinations[: self.N_ensemble_configurations]

        ensemble_configs = [
            EnsembleConfig(
                class_shift=cs,
                feature_shift=fs,
                transform_type=transform,
                max_features=self.max_num_features,
            )
            for cs, fs, transform in combinations
        ]

        engine = InferenceEngine(
            backbone=self.backbone,
            ensemble_configs=ensemble_configs,
            num_classes=num_classes,
            device=self.device,
            softmax_temperature=0.8,
            batch_size_inference=self.batch_size_inference,
            average_logits=True,
            extend_features=True,
            fp16_inference=False,
            no_grad=self.no_grad,
        )

        prediction = engine.predict(
            X=X_full,
            y=y_full,
            eval_position=eval_pos,
            style=None,
            return_logits=return_logits,
        )

        prediction_ = prediction.squeeze(0)
        return prediction_.detach().cpu().numpy() if self.no_grad else prediction_

    def predict(self, X, return_winning_probability=False):
        """Predict class labels."""
        p = self.predict_proba(X)
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=np.intp))

        if return_winning_probability:
            return y, p.max(axis=-1)
        return y

    def remove_models_from_memory(self):
        """Clear cached models."""
        self.models_in_memory = {}


def load_model_workflow(name, base_path, device="cpu"):
    """Load model from checkpoint."""
    model_path = os.path.join(base_path, name)
    results_file = os.path.join(base_path, f"results_{name}.pkl")

    if name is None:
        raise Exception("No checkpoint found at " + str(model_path))

    model, config = load_model_only_inference(base_path, name, device)
    return model, config, results_file
