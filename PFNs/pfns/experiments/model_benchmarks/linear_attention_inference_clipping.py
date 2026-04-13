from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from pfns.model.linear_attention import LinearAttention
from pfns.training_utils import (
    categorical_mask_to_inds,
    is_autocast_dtype_enabled,
    move_style_and_check_shape,
    move_y_style_and_check_shape,
)

from .analysis import nested_metric_table_to_long_df
from .benchmark_batch_generators import (
    _set_data_generation_seed,
    create_seq_len_batch_generator,
)
from .evaluation import evaluate_models_over_seqlens
from .hashing import experiment_payload_hash
from .model_registry import (
    functional_model_config,
    get_autocast_models_from_registry,
    get_forward_models_from_registry,
)
from .models import load_models_for_benchmark


@dataclass(frozen=True)
class InferenceClipModelBundle:
    models: dict[str, Any]
    configs: dict[str, dict[str, Any]]
    autocast_models: dict[str, torch.dtype]
    forward_models: dict[str, Any]


def normalize_inference_clip_experiments(
    clip_experiments: Mapping[str, Mapping[str, Any] | None] | None,
) -> dict[str, dict[str, Any]]:
    if not clip_experiments:
        return {"baseline": {}}
    return {
        str(clip_name): {
            str(key): value
            for key, value in (clip_kwargs or {}).items()
            if value is not None
        }
        for clip_name, clip_kwargs in clip_experiments.items()
    }

def experiment_cache_path(
    *,
    cache_dir: str | Path,
    experiment: Mapping[str, Any],
    model_configs: Mapping[str, Mapping[str, Any]],
    device: str | None = None,
    extra_payload: Any | None = None,
) -> Path:
    experiment_payload: dict[str, Any] = {
        "experiment": dict(experiment),
        "models_to_compare": {
            str(model_name): functional_model_config(dict(model_config))
            for model_name, model_config in model_configs.items()
        },
    }
    if device is not None:
        experiment_payload["device"] = str(device)
    if extra_payload is not None:
        experiment_payload["extra_payload"] = extra_payload
    cache_key = experiment_payload_hash(experiment_payload=experiment_payload)
    return Path(cache_dir) / f"{experiment['name']}_{cache_key}.pkl"


def materialize_seq_len_batch(
    *,
    experiment: Mapping[str, Any],
    seqlen: int,
    device: str,
    rep: int = 0,
    task_kwargs: Mapping[str, Any] | None = None,
):
    if rep < 0:
        raise ValueError("rep must be >= 0")
    _set_data_generation_seed(int(experiment["data_generation_seed"]))
    generator = create_seq_len_batch_generator(
        task_variant="tabular_prior",
        num_batches=rep + 1,
        smallest_seqlen=int(seqlen),
        largest_seqlen=int(seqlen),
        num_features=int(experiment["num_features"]),
        num_classes=int(experiment["num_classes"]),
        number_of_test_samples=int(experiment["num_test_samples"]),
        default_device=device,
        task_kwargs=dict(task_kwargs or {}),
    )
    for current_rep, (batch, _) in enumerate(generator):
        if current_rep == rep:
            return batch
    raise RuntimeError(f"Unable to materialize repetition {rep} for seqlen={seqlen}.")


def _batch_inputs(
    batch: Any,
    *,
    seqlen: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, Any, Any, list[int]]:
    x = batch.x[:, :seqlen]
    y = batch.y[:, :seqlen]
    return (
        x.to(device),
        y.to(device),
        move_style_and_check_shape(batch.style, x, device),
        move_y_style_and_check_shape(batch.y_style, y, device),
        categorical_mask_to_inds(batch.categorical_mask),
    )


def prepare_embedded_train_input(
    model: Any,
    batch: Any,
    *,
    seqlen: int,
    device: str,
) -> torch.Tensor:
    x, y, style, y_style, categorical_inds = _batch_inputs(batch, seqlen=seqlen, device=device)
    x_bf, y_bf, _ = model._prepare_batch_first_inputs(x, y, None)
    embedded, _, _, _ = model._build_embedded_input(
        x_bf,
        y_bf,
        single_eval_pos=int(y_bf.shape[1]),
        style=style,
        y_style=y_style,
        categorical_inds=categorical_inds,
        cache_trainset_representation=True,
    )
    return embedded


def run_model_autocast(
    fn: Any,
    *,
    model_name: str,
    autocast_models: Mapping[str, torch.dtype],
    device: str,
) -> Any:
    autocast_dtype = autocast_models.get(model_name)
    device_type = torch.device(device).type
    with torch.inference_mode():
        with torch.autocast(
            device_type=device_type,
            enabled=(device_type == "cuda") and is_autocast_dtype_enabled(autocast_dtype),
            dtype=autocast_dtype or torch.float32,
        ):
            return fn()


def _reference_seqlen(
    clip_kwargs: Mapping[str, Any],
    *,
    default_reference_seqlen: int,
) -> int:
    return int(clip_kwargs.get("state_ratio_reference_seqlen", default_reference_seqlen))


def _patch_linear_attention_inference(
    model: Any,
    clip_kwargs: Mapping[str, Any],
    *,
    default_reference_seqlen: int,
):
    patched_model = deepcopy(model)
    backbone = getattr(patched_model, "transformer_layers", None)
    if backbone is None or not hasattr(backbone, "layers"):
        return patched_model
    for layer in backbone.layers:
        if not isinstance(layer, LinearAttention):
            continue
        layer.hidden_state_frobenius_norm_max = clip_kwargs.get("state_max")
        layer.hidden_state_frobenius_norm_apply = clip_kwargs.get("state_apply", "pre_attention")
        layer.hidden_state_frobenius_norm_target = clip_kwargs.get("state_target", "joint")
        layer.hidden_state_frobenius_norm_length_normalization = clip_kwargs.get("state_len_norm", "none")
        layer.hidden_state_frobenius_norm_constant_after_length = clip_kwargs.get("state_constant_after_length")
        layer.hidden_state_frobenius_norm_eval_only = clip_kwargs.get("state_eval_only", False)
        layer.hidden_state_kv_over_ksum_reference = clip_kwargs.get("state_ratio_reference")
        layer.hidden_state_kv_over_ksum_reference_seqlen = (
            _reference_seqlen(
                clip_kwargs,
                default_reference_seqlen=default_reference_seqlen,
            )
            if clip_kwargs.get("state_ratio_reference") == "auto"
            else None
        )
        layer.hidden_state_kv_over_ksum_reference_apply = clip_kwargs.get(
            "state_ratio_reference_apply",
            "pre_prediction",
        )
        layer.hidden_state_kv_over_ksum_reference_eval_only = clip_kwargs.get(
            "state_ratio_reference_eval_only",
            False,
        )
        layer.attention_output_norm_max = clip_kwargs.get("output_max")
        layer.attention_output_norm_length_normalization = clip_kwargs.get("output_len_norm", "none")
    patched_model.eval()
    return patched_model


def build_inference_clip_model_bundle(
    model_configs: Mapping[str, Mapping[str, Any]],
    *,
    device: str,
    experiment: Mapping[str, Any],
    clip_experiments: Mapping[str, Mapping[str, Any] | None] | None,
    default_reference_seqlen: int = 1000,
) -> InferenceClipModelBundle:
    base_models, base_configs = load_models_for_benchmark(dict(model_configs), device=device)
    base_autocast_models = get_autocast_models_from_registry(dict(model_configs), device=device)
    base_forward_models = get_forward_models_from_registry(dict(model_configs))
    normalized_clip_experiments = normalize_inference_clip_experiments(clip_experiments)

    models: dict[str, Any] = {}
    configs: dict[str, dict[str, Any]] = {}
    autocast_models: dict[str, torch.dtype] = {}
    forward_models: dict[str, Any] = {}
    for raw_name, model in base_models.items():
        model_name = str(raw_name)
        for clip_name, clip_kwargs in normalized_clip_experiments.items():
            clipped_model_name = (
                model_name if clip_name == "baseline" else f"{model_name} | {clip_name}"
            )
            models[clipped_model_name] = _patch_linear_attention_inference(
                model,
                clip_kwargs,
                default_reference_seqlen=default_reference_seqlen,
            )
            configs[clipped_model_name] = base_configs[model_name]
            if model_name in base_autocast_models:
                autocast_models[clipped_model_name] = base_autocast_models[model_name]
            if model_name in base_forward_models:
                forward_models[clipped_model_name] = base_forward_models[model_name]
    return InferenceClipModelBundle(
        models=models,
        configs=configs,
        autocast_models=autocast_models,
        forward_models=forward_models,
    )


def inference_clip_bundle_from_results(
    results: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    metric_df = nested_metric_table_to_long_df(results["metric_table"])
    timing_df = nested_metric_table_to_long_df(results["timing_table"])
    memory_df = nested_metric_table_to_long_df(results["memory_table"])
    return {
        "results": dict(results),
        "metric_df": metric_df,
        "timing_df": timing_df,
        "memory_df": memory_df,
        "bundle_metadata": {
            "schema_version": results.get("schema_version"),
            "experiment": dict(experiment),
            "run_metadata": results.get("metadata", {}),
            "row_counts": {
                "metric": int(len(metric_df)),
                "timing": int(len(timing_df)),
                "memory": int(len(memory_df)),
            },
            "per_model_bundle_paths": {},
        },
    }


def _validate_inference_clip_bundle(bundle: Any, *, cache_path: Path) -> dict[str, Any]:
    required_keys = {"results", "metric_df", "timing_df", "memory_df", "bundle_metadata"}
    missing_keys = (
        sorted(required_keys - set(bundle))
        if isinstance(bundle, dict)
        else sorted(required_keys)
    )
    if missing_keys:
        raise RuntimeError(
            f"Inference-only clip cache at {cache_path} is missing keys: {missing_keys}. "
            "Delete it or set cache_overwrite = True."
        )
    return dict(bundle)


def load_or_run_inference_clip_benchmark(
    *,
    cache_path: str | Path,
    model_configs: Mapping[str, Mapping[str, Any]],
    experiment: Mapping[str, Any],
    clip_experiments: Mapping[str, Mapping[str, Any] | None] | None,
    device: str,
    task_kwargs: Mapping[str, Any] | None = None,
    precomputed_batches: list[Any] | None = None,
    cache_overwrite: bool = False,
    default_reference_seqlen: int = 1000,
    progress_desc: str = "Inference-only clip benchmark",
) -> dict[str, Any]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not cache_overwrite:
        bundle = _validate_inference_clip_bundle(
            pd.read_pickle(cache_path),
            cache_path=cache_path,
        )
        print(f"Loaded inference-only clip benchmark from cache: {cache_path}")
        return bundle

    runtime = build_inference_clip_model_bundle(
        model_configs,
        device=device,
        experiment=experiment,
        clip_experiments=clip_experiments,
        default_reference_seqlen=default_reference_seqlen,
    )
    results = evaluate_models_over_seqlens(
        models=runtime.models,
        configs=runtime.configs,
        seqlen_list=experiment["seqlen_list"],
        num_features=experiment["num_features"],
        num_classes=experiment["num_classes"],
        number_of_test_samples=experiment["num_test_samples"],
        number_of_repetitions=experiment["num_repetitions"],
        use_warmup_iters=experiment["use_warmup_iters"],
        print_timing=experiment["print_timing"],
        autocast_models=runtime.autocast_models,
        device=device,
        data_generation_seed=experiment["data_generation_seed"],
        progress_desc=progress_desc,
        forward_models=runtime.forward_models,
        task_kwargs=dict(task_kwargs or {}),
        precomputed_batches=precomputed_batches,
    )
    bundle = inference_clip_bundle_from_results(results, experiment=experiment)
    pd.to_pickle(bundle, cache_path)
    print(f"Saved inference-only clip benchmark to cache: {cache_path}")
    return bundle
