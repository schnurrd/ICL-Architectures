import os
import pathlib
import random
import itertools

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

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

from pfns.model.backbones import FLABackbone
from pfns.utils import (
    NOP,
    normalize_by_used_features_f,
    normalize_data,
    remove_outliers,
    strip_compiled_state_dict_prefix,
)
from pfns.base_config import BaseConfig
from pfns.train import MainConfig, resolve_autocast_dtype
from pfns.run_logger import download_model_from_wandb


# =============================================================================
# Ensemble Configuration
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuration for a single ensemble member."""

    class_shift: int = 0
    feature_shift: int = 0
    sample_permutation: torch.Tensor | None = None
    transform_type: Literal[
        "none", "power", "power_all", "quantile", "quantile_all", "robust", "robust_all"
    ] = "none"
    max_features: int = 100


# =============================================================================
# Inference Engine
# =============================================================================


class InferenceEngine:
    """Manages preprocessing, ensemble prediction, and aggregation."""

    def __init__(
        self,
        model: torch.nn.Module,
        ensemble_configs: list[EnsembleConfig],
        num_classes: int,
        device: str = "cpu",
        softmax_temperature: float = 0.8,
        batch_size_inference: int = 32,
        average_logits: bool = True,
        extend_features: bool = True,
        autocast_dtype: torch.dtype | None = None,
        no_grad: bool = True,
        categorical_feats: tuple[int, ...] = (),
        seed: Optional[int] = None,
    ):
        self.model = model
        self.ensemble_configs = ensemble_configs
        self.num_classes = num_classes
        self.device = device
        self.softmax_temperature = softmax_temperature
        self.batch_size_inference = batch_size_inference
        self.average_logits = average_logits
        self.extend_features = extend_features
        self.autocast_dtype = autocast_dtype
        self.no_grad = no_grad
        self.categorical_feats = categorical_feats
        self.seed = seed
        self._numpy_rng = np.random.default_rng(seed)

    def _get_sklearn_transformer(self, transform_type: str):
        """Get sklearn transformer based on config."""
        if transform_type == "none":
            return None
        elif transform_type in ["power", "power_all"]:
            return PowerTransformer(standardize=True)
        elif transform_type in ["quantile", "quantile_all"]:
            return QuantileTransformer(
                output_distribution="normal",
                n_quantiles=100,
                random_state=self.seed,
            )
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
    ) -> tuple[torch.Tensor, list[int]]:
        """Preprocess input data for one ensemble member and track categorical indices."""
        categorical_inds = list(self.categorical_feats) if self.categorical_feats else []
        active_indices = list(range(X.shape[2]))

        # ToDo: It would make more sense to switch the order between the constant feature removal and the max feature subsampling.
        if X.shape[2] > max_features:
            selected = sorted(
                self._numpy_rng.choice(X.shape[2], max_features, replace=False).tolist()
            )
            X = X[:, :, selected]
            active_indices = [active_indices[i] for i in selected]
            print(
                f"Warning: Subsampling features to {max_features} for preprocessing."
            )

        X = normalize_data(X, normalize_positions=eval_position)  # probably redundant

        # Remove constant features
        X = X[:, 0, :]
        sel = [
            len(torch.unique(X[0 : y.shape[0], col])) > 1 for col in range(X.shape[1])
        ]
        X = X[:, sel]
        active_indices = [idx for idx, keep in zip(active_indices, sel) if keep]
        categorical_inds = [
            new_idx
            for new_idx, orig_idx in enumerate(active_indices)
            if orig_idx in self.categorical_feats
        ]

        # Apply sklearn transforms
        if transform_type != "none":
            import warnings

            X_np = X.cpu().numpy()
            feats = (
                set(range(X_np.shape[1]))
                if "all" in transform_type
                else set(range(X_np.shape[1])) - set(categorical_inds)
            )

            warnings.simplefilter("error")
            for col in feats:
                try:
                    transformer = self._get_sklearn_transformer(transform_type)
                    if transformer is not None:
                        transformer.fit(X_np[0:eval_position, col : col + 1])
                        trans = transformer.transform(X_np[:, col : col + 1])
                        X_np[:, col : col + 1] = trans
                except Exception as e:
                    print(
                        f"Warning: Preprocessing transform {transform_type} failed on column {col}. Skipping transform. Error: {e}"
                    )
            warnings.simplefilter("default")
            X = torch.tensor(X_np).float()

        X = X.unsqueeze(1)
        return X.to(self.device), categorical_inds

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
        self.model.to(self.device)
        self.model.eval()

        y_train = y[:eval_position]

        inputs_list, labels_list, categorical_inds_list = self._prepare_ensemble_inputs(
            X, y_train, eval_position
        )

        all_outputs = []
        for inp, lbl, cat_inds in zip(inputs_list, labels_list, categorical_inds_list):
            batch_output = self._forward_batch(inp, lbl, style, cat_inds)
            all_outputs.append(batch_output)

        final_output = self._aggregate_ensemble_outputs(all_outputs, return_logits)

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
        categorical_inds_list = []

        # Cache preprocessed data by transform type
        preprocessed_cache = {}

        for config in self.ensemble_configs:
            transform_type = config.transform_type
            if transform_type in preprocessed_cache:
                X_processed, cached_cat_inds = preprocessed_cache[transform_type]
                X_processed = X_processed.clone()
                cat_inds_processed = list(cached_cat_inds)
            else:
                X_processed, cat_inds_processed = self._preprocess_data(
                    X.clone(), y, eval_position, transform_type, config.max_features
                )
                if self.no_grad:
                    X_processed = X_processed.detach()
                preprocessed_cache[transform_type] = (
                    X_processed,
                    list(cat_inds_processed),
                )

            y_shifted = ((y + config.class_shift) % self.num_classes).float()

            # Apply sample permutation on the training portion only
            if config.sample_permutation is not None and eval_position > 1:
                perm = config.sample_permutation
                if perm.shape[0] == eval_position:
                    if perm.device != X_processed.device:
                        perm = perm.to(X_processed.device)
                    X_processed = torch.cat(
                        [X_processed[:eval_position][perm], X_processed[eval_position:]],
                        dim=0,
                    )
                    y_shifted = torch.cat(
                        [y_shifted[:eval_position][perm], y_shifted[eval_position:]],
                        dim=0,
                    )

            # Apply feature shift
            if config.feature_shift > 0:
                X_processed = torch.cat(
                    [
                        X_processed[..., config.feature_shift :],
                        X_processed[..., : config.feature_shift],
                    ],
                    dim=-1,
                )
                cat_inds_processed = [
                    (idx - config.feature_shift) % X_processed.shape[2]
                    for idx in cat_inds_processed
                ]

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
            categorical_inds_list.append(cat_inds_processed)

        return inputs, labels, categorical_inds_list

    def _forward_batch(
        self,
        batch_input: torch.Tensor,
        batch_label: torch.Tensor,
        style: torch.Tensor | None,
        categorical_inds: list[int] | None,
    ) -> torch.Tensor:
        """Forward pass for a batch."""
        import warnings
        from torch.utils.checkpoint import checkpoint

        inference_mode_ctx = torch.inference_mode() if self.no_grad else NOP()
        autocast_dtype = self.autocast_dtype
        autocast_enabled = autocast_dtype is not None

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

                outputs = []
                for split_input, split_label in zip(
                    torch.split(batch_input, self.batch_size_inference, dim=1),
                    torch.split(batch_label, self.batch_size_inference, dim=1),
                ):
                    if self.device == "cpu":
                        out = checkpoint(
                            self._forward_fn,
                            split_input,
                            split_label,
                            style,
                            categorical_inds,
                            use_reentrant=False,
                        )
                    else:
                        with torch.amp.autocast(
                            "cuda",
                            enabled=autocast_enabled,
                            dtype=autocast_dtype,
                        ):
                            out = checkpoint(
                                self._forward_fn,
                                split_input,
                                split_label,
                                style,
                                categorical_inds,
                                use_reentrant=False,
                            )
                    outputs.append(out)

        return torch.cat(outputs, dim=1)

    def _forward_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        style: torch.Tensor | None,
        categorical_inds: list[int] | None,
    ) -> torch.Tensor:
        """Actual forward function through the model."""
        
        x_bf = x.transpose(0, 1)
        y_bf = y.transpose(0, 1).float()
        style_bf = style.repeat(x_bf.shape[0], 1) if style is not None else None

        output = self.model.forward(
            x=x_bf,
            y=y_bf,
            style=style_bf,
            categorical_inds=categorical_inds,
        ).transpose(0, 1)

        output = output[:, :, 0 : self.num_classes]

        output = output / self.softmax_temperature

        return output

    def _aggregate_ensemble_outputs(
        self,
        outputs: list[torch.Tensor],
        return_logits: bool,
    ) -> torch.Tensor:
        """Aggregate outputs from all ensemble members."""
        ensemble_outputs = []
        for output_i, config in zip(outputs, self.ensemble_configs):
            # Reverse class shift
            if torch.isnan(output_i).any():
                print("Warning: NaNs detected in ensemble output.")
            if torch.isinf(output_i).any():
                print("Warning: Infs detected in ensemble output.")
            
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

        aggregated = torch.stack(ensemble_outputs).mean(dim=0)

        if self.average_logits and not return_logits:
            aggregated = torch.nn.functional.softmax(aggregated, dim=-1)

        return aggregated.transpose(0, 1)


# =============================================================================
# Classifier Interface
# =============================================================================

default_base_path = pathlib.Path(__file__).parent.parent.resolve()


class TabPFNClassifier(BaseEstimator, ClassifierMixin):
    """Refactored TabPFN Classifier with separated preprocessing and model."""

    models_in_memory = {}
    name = "OurModel"

    def __init__(
        self,
        device: str = "cpu",
        base_path: Path | None = None,
        model_string: str = "checkpoint.pt",
        wandb_run_id: str | None = None,
        N_ensemble_configurations: int = 3,
        no_preprocess_mode: bool = False,
        multiclass_decoder: Literal["permutation", "none"] = "permutation",
        feature_shift_decoder: bool = True,
        sample_order_permutation: bool = False,
        only_inference: bool = True,
        seed: int = 0,
        no_grad: bool = True,
        batch_size_inference: int = 32,
        subsample_features: bool = False,
        preprocess_transforms: list[str] = None,
        fla_cache_chunk_size: int | None = None,
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
        :param wandb_run_id: If provided, downloads the model from the specified wandb run (entity/project/run_id or just run_id)
               and loads it. This overrides model_string.
        :param N_ensemble_configurations: The number of ensemble configurations used for the prediction. Thereby the
               accuracy, but also the running time, increases with this number.
        :param no_preprocess_mode: Specifies whether preprocessing is to be performed.
        :param multiclass_decoder: If set to permutation, randomly shifts the classes for each ensemble configuration.
        :param feature_shift_decoder: If set to true shifts the features for each ensemble configuration according to a
               random permutation.
        :param sample_order_permutation: If set to true permutes the training sample order for each ensemble configuration.
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
        :param preprocess_transforms: List of preprocessing transforms to consider during inference.
        :param fla_cache_chunk_size: If set and the model uses an FLA backbone, chunk size for cache-backed inference.
        """
        if wandb_run_id is not None:
            destination_path = (
                os.path.join(str(base_path), model_string)
                if base_path is not None
                else None
            )
            model_path = download_model_from_wandb(
                wandb_run_id,
                destination_path=destination_path,
            )
            base_path = os.path.dirname(model_path)
            model_string = os.path.basename(model_path)

        self.device = device
        self.base_path = Path(base_path) if base_path is not None else default_base_path
        self.model_string = model_string
        self.wandb_run_id = wandb_run_id
        self.N_ensemble_configurations = N_ensemble_configurations
        self.no_preprocess_mode = no_preprocess_mode
        self.multiclass_decoder = multiclass_decoder
        self.feature_shift_decoder = feature_shift_decoder
        self.sample_order_permutation = sample_order_permutation
        self.only_inference = only_inference
        self.seed = seed
        self.no_grad = no_grad
        self.batch_size_inference = batch_size_inference
        self.subsample_features = subsample_features
        self.preprocess_transforms = preprocess_transforms
        self.fla_cache_chunk_size = fla_cache_chunk_size
        self.categorical_feats: tuple[int, ...] = ()

        model_key = (
            f"{self.base_path.resolve()}|{self.model_string}|{self.device}|"
            f"{self.wandb_run_id or ''}"
        )

        if model_key in self.models_in_memory:
            model, config, results_file = self.models_in_memory[model_key]
        else:
            model, config, results_file = load_model_workflow(
                name=self.model_string,
                base_path=str(self.base_path),
                device=self.device,
            )
            self.models_in_memory[model_key] = (model, config, results_file)
            if len(self.models_in_memory) == 2:
                print(
                    "Multiple models in memory. This might lead to memory issues. Consider calling remove_models_from_memory()"
                )

        self.model = model
        self.config = config
        if self.fla_cache_chunk_size is not None:
            backbone = getattr(self.model, "transformer_layers", None)
            if isinstance(backbone, FLABackbone):
                backbone.cache_chunk_size = self.fla_cache_chunk_size
            else:
                print(
                    "Warning: fla_cache_chunk_size was provided but the model does not use an FLA backbone."
                )

        self.max_num_features = config.batch_shape_sampler.max_num_features
        self.max_num_classes = config.model.criterion.num_classes

        assert (
            self.no_preprocess_mode if not self.no_grad else True
        ), "If no_grad is false, no_preprocess_mode must be true, because otherwise no gradient can be computed."


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
        return np.asarray(y, dtype=np.int64, order="C")

    def fit(self, X, y, categorical_feats=None, overwrite_warning=True):
        """
        Validates the training set and stores it.

        If clf.no_grad (default is True):
        X, y should be of type np.array
        else:
        X should be of type torch.Tensors (y can be np.array or torch.Tensor)
        """
        if categorical_feats is not None:
            self.categorical_feats = tuple(categorical_feats)
        if self.no_grad:
            # Check that X and y have correct shape
            X, y = check_X_y(X, y, ensure_all_finite=False)
        # Store the classes seen during fit
        y = self._validate_targets(y)

        self.X_ = X
        self.y_ = y

        if X.shape[1] > self.max_num_features:
            if self.subsample_features:
                print(
                    "WARNING: The number of features for this classifier is restricted to ",
                    self.max_num_features,
                    " and will be subsampled.",
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

        y_full = np.concatenate([self.y_, np.full(shape=X.shape[0], fill_value=np.nan)], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)

        eval_pos = self.X_.shape[0]
        num_classes = len(self.classes_)

        if self.no_preprocess_mode:
            preprocess_transforms = ["none"]
        elif self.preprocess_transforms is not None:
            preprocess_transforms = self.preprocess_transforms
        else:
            preprocess_transforms = [
                "power",
                "quantile",
                "robust",
                "none",
            ]

        # Generate ensemble configurations
        if self.seed is not None:
            torch.manual_seed(self.seed)

        feature_shifts = (
            torch.randperm(X_full.shape[2])
            if self.feature_shift_decoder
            else [0]
        )
        class_shifts = (
            torch.randperm(num_classes)
            if self.multiclass_decoder == "permutation"
            else [0]
        )

        class_feature_pairs = list(itertools.product(class_shifts, feature_shifts))
        rng = random.Random(self.seed)
        rng.shuffle(class_feature_pairs)

        if self.sample_order_permutation and eval_pos > 1:
            sample_permutations = [
                torch.randperm(eval_pos, device=self.device)
                for _ in range(self.N_ensemble_configurations)
            ]
        else:
            sample_permutations = [None]

        combinations = []
        if class_feature_pairs and self.N_ensemble_configurations > 0:
            combinations = [
                (
                    *class_feature_pairs[i % len(class_feature_pairs)],
                    sample_permutations[i % len(sample_permutations)],
                    preprocess_transforms[i % len(preprocess_transforms)],
                )
                for i in range(self.N_ensemble_configurations)
            ]

        ensemble_configs = [
            EnsembleConfig(
                class_shift=cs,
                feature_shift=fs,
                sample_permutation=sp,
                transform_type=transform,
                max_features=self.max_num_features,
            )
            for cs, fs, sp, transform in combinations
        ]

        autocast_dtype = (
            resolve_autocast_dtype(
                self.device,
                self.config.train_mixed_precision_dtype,
            )
            if self.config.train_mixed_precision
            else None
        )
        engine = InferenceEngine(
            model=self.model,
            ensemble_configs=ensemble_configs,
            num_classes=num_classes,
            device=self.device,
            softmax_temperature=0.8,
            batch_size_inference=self.batch_size_inference,
            average_logits=True,
            extend_features=True,
            autocast_dtype=autocast_dtype,
            no_grad=self.no_grad,
            categorical_feats=self.categorical_feats,
            seed=self.seed,
        )

        prediction = engine.predict(
            X=X_full,
            y=y_full,
            eval_position=eval_pos,
            style=None,
            return_logits=return_logits,
        )

        prediction_ = prediction.squeeze(0)
        return prediction_.detach().float().cpu().numpy() if self.no_grad else prediction_

    def predict(self, X, return_winning_probability=False, return_prediction_probs=False):
        """Predict class labels."""
        p = self.predict_proba(X)
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=np.intp))

        out = [y]
        if return_winning_probability:
            out.append(p.max(axis=-1))
        if return_prediction_probs:
            out.append(p)
        return out[0] if len(out) == 1 else tuple(out)

    def remove_models_from_memory(self):
        """Clear cached models."""
        self.models_in_memory.clear()

def load_model_workflow(name, base_path, device="cpu"):
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """
    
    model_path = os.path.join(base_path, name)
    results_file = os.path.join(base_path, f"results_{name}.pkl")

    if name is None:
        raise Exception("No checkpoint found at " + str(model_path))

    checkpoint = torch.load(os.path.join(base_path, name), map_location="cpu")

    if "config" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing the serialized training config under key 'config'."
        )
    
    config: BaseConfig = MainConfig.from_dict(checkpoint["config"])
    
    model = config.model.create_model()
    model_state = checkpoint["model_state_dict"]
    model_state, _ = strip_compiled_state_dict_prefix(model_state)

    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    
    return model, config, results_file
