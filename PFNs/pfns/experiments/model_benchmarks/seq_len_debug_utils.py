from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
from types import MethodType
from typing import Any, Literal, Mapping

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from pfns.experiments.model_benchmarks.benchmark_batch_generators import (
    _set_data_generation_seed,
    create_seq_len_batch_generator,
)
from pfns.experiments.model_benchmarks.model_registry import (
    get_autocast_models_from_registry,
)
from pfns.experiments.model_benchmarks.models import load_models_for_benchmark
from pfns.experiments.model_benchmarks.plotting import (
    apply_pretraining_split_background,
    create_panel_figure,
    plot_grouped_runs_with_distribution,
    resolve_display_name_map,
)
from pfns.tensor_tree_utils import iter_named_tensors
from pfns.training_utils import (
    categorical_mask_to_inds,
    is_autocast_dtype_enabled,
    move_style_and_check_shape,
    move_y_style_and_check_shape,
)

_HIDDEN_STATE_HINTS = ("state", "cache", "kv", "ssm", "h0")
_LAYER_PATTERN = re.compile(r"(?:layers|layer_states)\[(\d+)\]")
_MATRIX_METRICS = ("frobenius_norm", "spectral_norm", "effective_rank", "stable_rank")
_METRIC_DISPLAY_NAMES = {
    "abs_max": "Absolute Max",
    "frobenius_norm": "Frobenius Norm",
    "spectral_norm": "Spectral Norm",
    "effective_rank": "Effective Rank",
    "stable_rank": "Stable Rank",
    "kv_state_norm": "KV-State Norm",
    "kv_state_norm_post_renorm": "KV-State Norm After Renormalisation",
    "k_sum_norm": "K-Sum Norm",
    "joint_hidden_state_norm": "Joint Hidden-State Norm",
    "kv_over_ksum_ratio": "KV-over-K-Sum Ratio",
    "state_cosine_to_reference": "State Cosine Similarity to Reference",
    "state_top_subspace_to_reference": "Top Singular Subspace to Reference",
    "state_renorm_scale": "State Renormalisation Scale",
    "output_norm": "Output Norm",
    "output_cosine_to_reference": "Readout Cosine to Reference",
    "state_cosine_to_previous": "State Cosine to Previous Context",
    "output_cosine_to_previous": "Readout Cosine to Previous Context",
}


def _sanitize_plot_filename(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("_") or "plot"


def _save_plot_figure(
    fig: Any,
    *,
    save_dir: str | Path | None,
    filename_stem: str,
    save_formats: tuple[str, ...],
) -> list[Path]:
    if save_dir is None:
        return []
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    safe_stem = _sanitize_plot_filename(filename_stem)
    for fmt in save_formats:
        path = output_dir / f"{safe_stem}.{fmt.lstrip('.')}"
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if fmt.lower().lstrip(".") in {"png", "jpg", "jpeg"}:
            save_kwargs["dpi"] = 300
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)
    return saved_paths


def _compute_padded_y_limits(
    y_min: float,
    y_max: float,
    *,
    log_scale: bool,
) -> tuple[float, float]:
    if log_scale:
        return (max(y_min / 1.2, 1e-12), y_max * 1.2)
    pad = max((y_max - y_min) * 0.05, 1e-6) if y_max != y_min else max(abs(y_min) * 0.05, 1e-6)
    return (y_min - pad, y_max + pad)


def _reordered_tab20_palette() -> list[str]:
    tab20 = [mcolors.to_hex(color) for color in plt.get_cmap("tab20").colors]
    reorder = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    return [tab20[idx] for idx in reorder]


@dataclass(frozen=True)
class HiddenStateTrackingConfig:
    num_classes: int
    num_features: int
    num_test_samples: int
    num_repetitions: int
    data_generation_seed: int
    seqlen_list: tuple[int, ...]
    name: str = "seq_len_hidden_state_debug"

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if self.num_features < 1:
            raise ValueError("num_features must be >= 1")
        if self.num_test_samples < 1:
            raise ValueError("num_test_samples must be >= 1")
        if self.num_repetitions < 1:
            raise ValueError("num_repetitions must be >= 1")
        if not self.seqlen_list or min(self.seqlen_list) < 1:
            raise ValueError("seqlen_list must be non-empty and all values >= 1")

    @property
    def sorted_seqlens(self) -> tuple[int, ...]:
        return tuple(sorted(self.seqlen_list))

    @property
    def smallest_seqlen(self) -> int:
        return min(self.seqlen_list)

    @property
    def largest_seqlen(self) -> int:
        return max(self.seqlen_list)

    @classmethod
    def from_mapping(cls, experiment: Mapping[str, Any]) -> "HiddenStateTrackingConfig":
        keys = (
            "num_classes",
            "num_features",
            "num_test_samples",
            "num_repetitions",
            "data_generation_seed",
            "seqlen_list",
        )
        missing = [key for key in keys if key not in experiment]
        if missing:
            raise KeyError(f"Missing experiment keys: {missing}")
        return cls(
            name=str(experiment.get("name", "seq_len_hidden_state_debug")),
            num_classes=int(experiment["num_classes"]),
            num_features=int(experiment["num_features"]),
            num_test_samples=int(experiment["num_test_samples"]),
            num_repetitions=int(experiment["num_repetitions"]),
            data_generation_seed=int(experiment["data_generation_seed"]),
            seqlen_list=tuple(int(v) for v in experiment["seqlen_list"]),
        )


def _device_runtime(
    models_to_compare: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.nn.Module], dict[str, torch.dtype]]:
    for name, cfg in models_to_compare.items():
        if cfg.get("eval_mode", "fit_predict") != "fit_predict":
            raise ValueError(f"{name} must use eval_mode='fit_predict'")
    if torch.device(device).type != "cuda":
        raise ValueError("seq_len_debug_utils assumes a CUDA device.")
    models, _ = load_models_for_benchmark(models_to_compare, device=device)
    autocast_models = get_autocast_models_from_registry(models_to_compare, device=device)
    return models, autocast_models


def _run_autocast(
    fn: Any,
    *,
    model_name: str,
    autocast_models: dict[str, torch.dtype],
) -> Any:
    autocast_dtype = autocast_models.get(model_name)
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            enabled=is_autocast_dtype_enabled(autocast_dtype),
            dtype=autocast_dtype or torch.float32,
        ):
            return fn()


def _iter_hidden_tensors(
    state: Any,
    tensor_name_patterns: list[str] | tuple[str, ...] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    named = list(iter_named_tensors(state, prefix="state"))
    if not named:
        return []
    if tensor_name_patterns:
        patterns = [str(pattern).lower() for pattern in tensor_name_patterns if str(pattern)]
        if patterns:
            return [(name, tensor) for name, tensor in named if any(p in name.lower() for p in patterns)]
    filtered = [(name, tensor) for name, tensor in named if any(hint in name.lower() for hint in _HIDDEN_STATE_HINTS)]
    return filtered or named


def _layer_idx(name: str) -> int:
    return int(match.group(1)) if (match := _LAYER_PATTERN.search(name)) else -1


def _head_matrices(tensor: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
    arr = tensor.detach().float() # can be (1, 1, num_heads, h_dim, h_dim) or (num_heads, h_dim, h_dim)
    if arr.ndim < 2:
        return []
    if arr.ndim == 2:
        return [(0, arr)]
    arr = arr.movedim(arr.ndim - 3, 0)
    if arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], -1, arr.shape[-2], arr.shape[-1]).squeeze(1)
    return [(head_idx, arr[head_idx]) for head_idx in range(int(arr.shape[0]))]


def _head_vectors(tensor: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
    arr = tensor.detach().float()  # can be (1, 1, num_heads, h_dim) or (num_heads, h_dim)
    if arr.ndim == 0:
        return []
    if arr.ndim == 1:
        return [(0, arr)]
    arr = arr.movedim(arr.ndim - 2, 0)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1, arr.shape[-1]).squeeze(1)
    return [(head_idx, arr[head_idx]) for head_idx in range(int(arr.shape[0]))]


def _k_sum_norms_by_layer(state: Any) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for name, tensor in _iter_hidden_tensors(state):
        if "k_sum" not in name.lower():
            continue
        layer_idx = _layer_idx(name)
        for head_idx, vector in _head_vectors(tensor):
            out[(int(layer_idx), int(head_idx))] = float(torch.linalg.vector_norm(vector).item())
    return out


def _effective_hidden_state_tensor(
    model: Any,
    *,
    tensor_name: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    lowered_name = tensor_name.lower()
    if "recurrent_state" not in lowered_name and "kv_state" not in lowered_name:
        return tensor
    backbone = getattr(model, "transformer_layers", None)
    if backbone is None or backbone.__class__.__name__ != "LinearAttentionBackbone":
        return tensor
    layer_idx = _layer_idx(tensor_name)
    if layer_idx < 0 or layer_idx >= len(backbone.layers):
        return tensor
    layer = backbone.layers[layer_idx]
    if getattr(layer, "state_renormalization", None) in {None, "none"}:
        return tensor
    with torch.no_grad():
        return layer._renormalize_state(tensor.detach().clone())


def _effective_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    singular_values = singular_values[singular_values > 0]
    if singular_values.numel() == 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    return float((-(probs * probs.log()).sum()).exp().item())


def _stable_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    singular_values = singular_values[singular_values > 0]
    if singular_values.numel() == 0:
        return 0.0
    spectral_norm = singular_values.max()
    if float(spectral_norm.item()) == 0.0:
        return 0.0
    frobenius_sq = singular_values.square().sum()
    return float((frobenius_sq / spectral_norm.square()).item())


def _matrix_metrics(matrix: torch.Tensor) -> dict[str, float | tuple[int, ...] | int]:
    arr = matrix.detach().float()
    shape = tuple(int(v) for v in arr.shape)
    if arr.numel() == 0:
        return {
            "shape": shape,
            "abs_max": float("nan"),
            "frobenius_norm": float("nan"),
            "spectral_norm": float("nan"),
            "effective_rank": 0.0,
            "stable_rank": 0.0,
        }
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("Matrix contains non-finite values, cannot compute metrics.")
    singular_values = torch.linalg.svdvals(arr)
    return {
        "shape": shape,
        "abs_max": float(arr.abs().max().item()),
        "frobenius_norm": float(torch.linalg.vector_norm(arr).item()),
        "spectral_norm": float(singular_values.max().item()),
        "effective_rank": _effective_rank_from_singular_values(singular_values),
        "stable_rank": _stable_rank_from_singular_values(singular_values),
    }


def _top_subspace_similarity(
    matrix: torch.Tensor,
    reference: torch.Tensor,
    *,
    rank: int = 8,
) -> float:
    u, _, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    u_ref, _, vh_ref = torch.linalg.svd(reference.float(), full_matrices=False)
    k = min(int(rank), u.shape[1], u_ref.shape[1], vh.shape[0], vh_ref.shape[0])
    if k <= 0:
        return float("nan")
    left = torch.linalg.matrix_norm(u[:, :k].T @ u_ref[:, :k], ord="fro").square() / k
    right = torch.linalg.matrix_norm(vh[:k] @ vh_ref[:k].T, ord="fro").square() / k
    return float(((left + right) / 2).item())


def _linear_attention_readouts(
    model: Any,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    style: torch.Tensor | None,
    y_style: torch.Tensor | None,
    categorical_inds: list[int] | None,
) -> dict[tuple[int, int], torch.Tensor]:
    base_model = getattr(model, "base_model", model)
    backbone = getattr(base_model, "transformer_layers", None)
    if backbone is None or backbone.__class__.__name__ != "LinearAttentionBackbone":
        return {}

    x_bf, y_bf, _ = base_model._prepare_batch_first_inputs(x, y, None)
    assert x_bf is not None and y_bf is not None
    embedded_input, _, _, _ = base_model._build_embedded_input(
        x_bf,
        y_bf,
        single_eval_pos=y_bf.shape[1],
        style=style,
        y_style=y_style,
        categorical_inds=categorical_inds,
        cache_trainset_representation=True,
    )

    readouts: dict[tuple[int, int], torch.Tensor] = {}
    out = embedded_input
    for layer_idx, layer in enumerate(backbone.layers):
        q_raw, k_raw, v = layer._project_qkv(out)
        if layer.causal or layer.causal_train_only:
            attn, _, _ = layer._causal_attention(q_raw, k_raw, v)
        else:
            attn, _, _ = layer._noncausal_attention(q_raw, k_raw, v)
        for head_idx in range(int(attn.shape[2])):
            readouts[(int(layer_idx), int(head_idx))] = (
                attn[:, :, head_idx].detach().float().cpu()
            )
        out = layer._apply_output(out, attn)
    return readouts


def _sequence_prefix_cosine(matrix: torch.Tensor, reference: torch.Tensor) -> float:
    overlap = min(int(matrix.shape[1]), int(reference.shape[1]))
    if overlap <= 0:
        return float("nan")
    return float(
        torch.nn.functional.cosine_similarity(
            matrix[:, :overlap].reshape(1, -1),
            reference[:, :overlap].reshape(1, -1),
            dim=-1,
        ).item()
    )


def run_hidden_state_tracking(
    *,
    experiment: HiddenStateTrackingConfig | Mapping[str, Any],
    models_to_compare: dict[str, Any],
    device: str,
    tensor_name_patterns: list[str] | tuple[str, ...] | None = None,
    reference_seqlen: int | None = None,
    reference_subspace_rank: int = 8,
) -> pd.DataFrame:
    cfg = experiment if isinstance(experiment, HiddenStateTrackingConfig) else HiddenStateTrackingConfig.from_mapping(experiment)
    if reference_seqlen is not None and int(reference_seqlen) not in cfg.seqlen_list:
        raise ValueError(f"reference_seqlen={reference_seqlen} must be in seqlen_list.")
    _set_data_generation_seed(cfg.data_generation_seed)
    models, autocast_models = _device_runtime(models_to_compare, device)
    rows: list[dict[str, Any]] = []

    batch_generator = create_seq_len_batch_generator(
        task_variant="tabular_prior",
        num_batches=cfg.num_repetitions,
        smallest_seqlen=cfg.smallest_seqlen,
        largest_seqlen=cfg.largest_seqlen,
        num_features=cfg.num_features,
        num_classes=cfg.num_classes,
        number_of_test_samples=cfg.num_test_samples,
        default_device=device,
        task_kwargs={},
    )
    for rep, (base_batch, _) in enumerate(tqdm(batch_generator, total=cfg.num_repetitions, desc="Hidden-state tracking")):
        categorical_inds = categorical_mask_to_inds(base_batch.categorical_mask)
        for raw_name, model in models.items():
            model_name = str(raw_name)
            state_matrices: dict[tuple[int, str, int, int], torch.Tensor] = {}
            output_tensors: dict[tuple[int, str, int, int], torch.Tensor] = {}
            state_row_indices: dict[tuple[int, str, int, int], int] = {}
            for seqlen in cfg.sorted_seqlens:
                x = base_batch.x[:, :seqlen]
                y = base_batch.y[:, :seqlen]
                fit_kwargs = {
                    "x": x.to(device),
                    "y": y.to(device),
                    "style": move_style_and_check_shape(base_batch.style, x, device),
                    "y_style": move_y_style_and_check_shape(base_batch.y_style, y, device),
                    "categorical_inds": categorical_inds,
                }
                try:
                    state = _run_autocast(
                        lambda: model.incontext_fit(**fit_kwargs),
                        model_name=model_name,
                        autocast_models=autocast_models,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
                except RuntimeError as err:
                    if "out of memory" in str(err).lower():
                        torch.cuda.empty_cache()
                        break
                    raise
                extra = {"rep": int(rep), "seqlen": int(seqlen)}
                k_sum_norms = _k_sum_norms_by_layer(state)
                try:
                    readouts = _run_autocast(
                        lambda: _linear_attention_readouts(model, **fit_kwargs),
                        model_name=model_name,
                        autocast_models=autocast_models,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    readouts = {}
                for name, tensor in _iter_hidden_tensors(state, tensor_name_patterns):
                    lowered_name = name.lower()
                    if "recurrent_state" not in lowered_name and "kv_state" not in lowered_name:
                        continue
                    effective_tensor = _effective_hidden_state_tensor(
                        model,
                        tensor_name=name,
                        tensor=tensor,
                    )
                    for head_idx, matrix in _head_matrices(effective_tensor):
                        metrics = _matrix_metrics(matrix)
                        layer_idx = _layer_idx(name)
                        kv_state_norm = float(metrics["frobenius_norm"])
                        k_sum_norm = k_sum_norms.get((int(layer_idx), int(head_idx)), float("nan"))
                        joint_hidden_state_norm = (
                            float((kv_state_norm**2 + k_sum_norm**2) ** 0.5)
                            if np.isfinite(k_sum_norm)
                            else float("nan")
                        )
                        kv_over_ksum_ratio = (
                            float(kv_state_norm / max(k_sum_norm, 1e-12))
                            if np.isfinite(k_sum_norm)
                            else float("nan")
                        )
                        row = {
                            "model": model_name,
                            **extra,
                            "tensor_name": name,
                            "shape": str(metrics["shape"]),
                            "abs_max": float(metrics["abs_max"]),
                            **{k: float(metrics[k]) for k in _MATRIX_METRICS},
                            "kv_state_norm": kv_state_norm,
                            "k_sum_norm": k_sum_norm,
                            "joint_hidden_state_norm": joint_hidden_state_norm,
                            "kv_over_ksum_ratio": kv_over_ksum_ratio,
                            "output_norm": float("nan"),
                            "layer_idx": layer_idx,
                            "head_idx": int(head_idx),
                        }
                        readout = readouts.get((int(layer_idx), int(head_idx)))
                        if readout is not None:
                            row["output_norm"] = float(torch.linalg.vector_norm(readout).item())
                        if reference_seqlen is not None:
                            row["reference_seqlen"] = int(reference_seqlen)
                            row["state_cosine_to_reference"] = float("nan") # add a placeholder
                            row["state_top_subspace_to_reference"] = float("nan")
                            row["output_cosine_to_reference"] = float("nan")
                            row["previous_seqlen"] = float("nan")
                            row["state_cosine_to_previous"] = float("nan")
                            row["output_cosine_to_previous"] = float("nan")
                            key = (int(seqlen), name, int(layer_idx), int(head_idx))
                            state_matrices[key] = matrix.detach().float().cpu()
                            if readout is not None:
                                output_tensors[key] = readout
                            state_row_indices[key] = len(rows)
                        rows.append(row)
            if reference_seqlen is not None:
                ref_by_state = {
                    key[1:]: matrix
                    for key, matrix in state_matrices.items()
                    if key[0] == int(reference_seqlen)
                } # extract reference matrices for the reference seqlen -> num_layer x num_head many matrices
                for key, matrix in state_matrices.items():
                    reference = ref_by_state.get(key[1:])
                    if reference is None:
                        continue
                    cosine = torch.nn.functional.cosine_similarity(
                        matrix.reshape(1, -1),
                        reference.reshape(1, -1),
                        dim=-1,
                    ).item()
                    rows[state_row_indices[key]]["state_cosine_to_reference"] = float(cosine)
                    rows[state_row_indices[key]]["state_top_subspace_to_reference"] = _top_subspace_similarity(
                        matrix,
                        reference,
                        rank=reference_subspace_rank,
                    )
                ref_outputs_by_state = {
                    key[1:]: output
                    for key, output in output_tensors.items()
                    if key[0] == int(reference_seqlen)
                }
                for key, output in output_tensors.items():
                    reference_output = ref_outputs_by_state.get(key[1:])
                    if reference_output is None:
                        continue
                    rows[state_row_indices[key]]["output_cosine_to_reference"] = _sequence_prefix_cosine(
                        output,
                        reference_output,
                    )
                sorted_seqlens = cfg.sorted_seqlens
                previous_by_seqlen = {
                    int(seqlen): int(sorted_seqlens[idx - 1])
                    for idx, seqlen in enumerate(sorted_seqlens)
                    if idx > 0
                }
                for key, matrix in state_matrices.items():
                    previous_seqlen = previous_by_seqlen.get(key[0])
                    if previous_seqlen is None:
                        continue
                    previous_matrix = state_matrices.get((previous_seqlen, *key[1:]))
                    if previous_matrix is None:
                        continue
                    row = rows[state_row_indices[key]]
                    row["previous_seqlen"] = int(previous_seqlen)
                    row["state_cosine_to_previous"] = float(
                        torch.nn.functional.cosine_similarity(
                            matrix.reshape(1, -1),
                            previous_matrix.reshape(1, -1),
                            dim=-1,
                        ).item()
                    )
                for key, output in output_tensors.items():
                    previous_seqlen = previous_by_seqlen.get(key[0])
                    if previous_seqlen is None:
                        continue
                    previous_output = output_tensors.get((previous_seqlen, *key[1:]))
                    if previous_output is None:
                        continue
                    row = rows[state_row_indices[key]]
                    row["previous_seqlen"] = int(previous_seqlen)
                    row["output_cosine_to_previous"] = _sequence_prefix_cosine(
                        output,
                        previous_output,
                    )
    return pd.DataFrame(rows)


def summarize_hidden_state_by_seqlen(hidden_state_df: pd.DataFrame) -> pd.DataFrame:
    df = hidden_state_df.copy()
    for col, default in (
        ("layer_idx", -1),
        ("head_idx", -1),
        *((key, float("nan")) for key in _MATRIX_METRICS),
        ("kv_state_norm", float("nan")),
        ("k_sum_norm", float("nan")),
        ("joint_hidden_state_norm", float("nan")),
        ("kv_over_ksum_ratio", float("nan")),
        ("state_cosine_to_reference", float("nan")),
        ("state_top_subspace_to_reference", float("nan")),
        ("output_norm", float("nan")),
        ("output_cosine_to_reference", float("nan")),
        ("state_cosine_to_previous", float("nan")),
        ("output_cosine_to_previous", float("nan")),
    ):
        if col not in df.columns:
            df[col] = default
    if hidden_state_df.empty:
        cols = [
            "model", "tensor_name", "layer_idx", "head_idx", "seqlen",
            "abs_max_mean",
            "frobenius_norm_mean", "spectral_norm_mean", "effective_rank_mean", "stable_rank_mean", "n",
            "kv_state_norm_mean", "k_sum_norm_mean", "joint_hidden_state_norm_mean", "kv_over_ksum_ratio_mean",
            "state_cosine_to_reference_mean",
            "state_top_subspace_to_reference_mean",
            "output_norm_mean",
            "output_cosine_to_reference_mean",
            "state_cosine_to_previous_mean",
            "output_cosine_to_previous_mean",
        ]
        return pd.DataFrame(columns=cols)
    group_cols = [c for c in ("model", "tensor_name", "layer_idx", "head_idx", "seqlen") if c in df.columns]
    out = (
        df.groupby(group_cols, observed=True)
        .agg(
            abs_max_mean=("abs_max", "mean"),
            frobenius_norm_mean=("frobenius_norm", "mean"),
            spectral_norm_mean=("spectral_norm", "mean"),
            effective_rank_mean=("effective_rank", "mean"),
            stable_rank_mean=("stable_rank", "mean"),
            kv_state_norm_mean=("kv_state_norm", "mean"),
            k_sum_norm_mean=("k_sum_norm", "mean"),
            joint_hidden_state_norm_mean=("joint_hidden_state_norm", "mean"),
            kv_over_ksum_ratio_mean=("kv_over_ksum_ratio", "mean"),
            state_cosine_to_reference_mean=("state_cosine_to_reference", "mean"),
            state_top_subspace_to_reference_mean=("state_top_subspace_to_reference", "mean"),
            output_norm_mean=("output_norm", "mean"),
            output_cosine_to_reference_mean=("output_cosine_to_reference", "mean"),
            state_cosine_to_previous_mean=("state_cosine_to_previous", "mean"),
            output_cosine_to_previous_mean=("output_cosine_to_previous", "mean"),
            n=("rep", "nunique"),
        )
        .reset_index()
    )
    order = [c for c in ("model", "layer_idx", "head_idx", "tensor_name", "seqlen") if c in out.columns]
    return out.sort_values(order)


def collect_state_renorm_scales(
    *,
    models_to_compare: dict[str, Any],
    device: str,
) -> pd.DataFrame:
    models, _ = _device_runtime(models_to_compare, device)
    rows = []
    for model_name, model in models.items():
        backbone = getattr(model, "transformer_layers", None)
        layers = getattr(backbone, "layers", [])
        for layer_idx, layer in enumerate(layers):
            log_scale = getattr(layer, "state_renorm_log_scale", None)
            if log_scale is None:
                continue
            scale = log_scale.detach().float().exp().cpu()
            for head_idx, value in enumerate(scale):
                rows.append(
                    {
                        "model": str(model_name),
                        "layer_idx": int(layer_idx),
                        "head_idx": int(head_idx),
                        "state_renorm_scale": float(value.item()),
                    }
                )
    return pd.DataFrame(rows)


def _plot_recurrent_metric(
    plot_df: pd.DataFrame,
    *,
    metric: str,
    title_prefix: str,
    group_key: str,
    line_key: str,
    group_name: str,
    line_name: str,
    model: str | None = None,
    training_context_length: int | None = None,
    log_x: bool = True,
    log_y: bool = False,
    plot_mode: Literal["individual_runs", "violin"] = "individual_runs",
    run_alpha: float = 0.35,
    distribution_alpha: float | None = 0.3,
    distribution_width_frac: float = 0.4,
    show_title: bool = True,
    show_suptitle: bool = True,
    include_group_in_title: bool = True,
    legend_title: str | None = None,
    save_dir: str | Path | None = None,
    filename_prefix: str | None = None,
    save_formats: tuple[str, ...] = ("pdf",),
) -> None:
    if plot_mode not in {"individual_runs", "violin"}:
        raise ValueError("plot_mode must be 'individual_runs' or 'violin'.")
    if not 0.0 < run_alpha <= 1.0:
        raise ValueError("run_alpha must be in the interval (0, 1].")
    if distribution_alpha is not None and not 0.0 < distribution_alpha <= 1.0:
        raise ValueError("distribution_alpha must be in the interval (0, 1].")
    if not 0.0 < distribution_width_frac:
        raise ValueError("distribution_width_frac must be positive.")

    model_values = sorted(plot_df["model"].astype(str).unique().tolist())
    display_name_map = resolve_display_name_map(plot_df)
    metric_label = _METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())
    split = model is None and len(model_values) > 1
    for group_value, group_df in plot_df.groupby(group_key, observed=True):
        panel_models = model_values if split else [model_values[0]]
        fig, axes = create_panel_figure(
            panel_count=len(panel_models),
            figsize=(6.4 * len(panel_models), 5),
            sharey=split,
        )
        visible_axes: list[Any] = []
        shared_x_left: float | None = None
        shared_x_right: float | None = None
        shared_y_limits: tuple[float, float] | None = None
        if plot_mode == "violin":
            finite_group_values = group_df[metric].to_numpy(dtype=float, copy=False)
            finite_group_values = finite_group_values[np.isfinite(finite_group_values)]
            if log_y:
                finite_group_values = finite_group_values[finite_group_values > 0.0]
            if finite_group_values.size > 0:
                y_min = float(finite_group_values.min())
                y_max = float(finite_group_values.max())
                shared_y_limits = _compute_padded_y_limits(y_min, y_max, log_scale=log_y)
        for idx, model_name in enumerate(panel_models):
            ax = axes[idx if split else 0]
            sub = group_df if not split else group_df[group_df["model"] == model_name]
            summary_fn = "median" if plot_mode == "violin" else "mean"
            agg = (
                sub.groupby([line_key, "seqlen"], observed=True)[metric]
                .agg(summary_fn)
                .reset_index()
                .sort_values([line_key, "seqlen"])
            )
            if agg.empty:
                ax.set_visible(False)
                continue
            visible_axes.append(ax)
            line_values = agg[line_key].dropna().unique().tolist()
            tab20_reordered = _reordered_tab20_palette()
            colors = {value: tab20_reordered[i % len(tab20_reordered)] for i, value in enumerate(line_values)}
            for line_idx, line_value in enumerate(line_values):
                value = int(line_value)
                label = f"{line_name.title()} {value}" if value >= 0 else f"{line_name.title()} unknown"
                color = colors[line_value]
                raw_line_df = sub[sub[line_key] == line_value]
                plot_grouped_runs_with_distribution(
                    ax=ax,
                    sub=raw_line_df,
                    x_col="seqlen",
                    value_col=metric,
                    rep_col="rep",
                    model_label=label,
                    marker="o",
                    linestyle="-",
                    color=color,
                    run_alpha=run_alpha,
                    distribution_alpha=distribution_alpha,
                    distribution_width_frac=distribution_width_frac,
                    show_run_lines=plot_mode == "individual_runs",
                    log_x=log_x,
                    distribution_style="half_violin" if plot_mode == "violin" else "none",
                    model_index=line_idx,
                    model_count=len(line_values),
                    summary_stat=summary_fn,
                    line_width=1.3,
                    marker_size=5.5,
                    distribution_zorder=2,
                    summary_zorder=3,
                )
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            curr_x_left, curr_x_right = ax.get_xlim()
            shared_x_left = curr_x_left if shared_x_left is None else min(shared_x_left, curr_x_left)
            shared_x_right = curr_x_right if shared_x_right is None else max(shared_x_right, curr_x_right)
            if training_context_length is not None:
                apply_pretraining_split_background(
                    ax,
                    boundary=float(training_context_length),
                    boundary_label="Train context",
                )
            ax.set_xlabel("Sequence length")
            ax.set_ylabel(metric_label if idx == 0 else "")
            panel_title = (
                (
                    display_name_map.get(str(model_name), str(model_name))
                    if show_suptitle
                    else (
                        display_name_map.get(str(model_name), str(model_name))
                        if not include_group_in_title
                        else (
                            f"{display_name_map.get(str(model_name), str(model_name))}\n"
                            f"{group_name.title()} {int(group_value)}"
                        )
                    )
                )
                if split
                else (
                    title_prefix
                    if not include_group_in_title
                    else f"{title_prefix} | {group_name}[{int(group_value)}]"
                )
            )
            ax.set_title(panel_title if show_title else "")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8, ncol=2, title=legend_title)
        for ax in visible_axes:
            if shared_x_left is not None and shared_x_right is not None:
                ax.set_xlim(shared_x_left, shared_x_right)
            if shared_y_limits is not None:
                ax.set_ylim(*shared_y_limits)
            if training_context_length is not None:
                apply_pretraining_split_background(
                    ax,
                    boundary=float(training_context_length),
                    boundary_label="Train context",
                )
        if split and show_suptitle:
            suptitle = (
                title_prefix
                if not include_group_in_title
                else f"{title_prefix} | {group_name}[{int(group_value)}]"
            )
            fig.suptitle(suptitle)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            fig.tight_layout()
        _save_plot_figure(
            fig,
            save_dir=save_dir,
            filename_stem=f"{filename_prefix or metric}_{group_name}_{int(group_value)}",
            save_formats=save_formats,
        )


def plot_recurrent_metric_per_head(
    df: pd.DataFrame,
    *,
    metric: str,
    title_prefix: str,
    model: str | None = None,
    training_context_length: int | None = None,
    log_x: bool = True,
    log_y: bool = False,
    plot_mode: Literal["individual_runs", "violin"] = "individual_runs",
    run_alpha: float = 0.35,
    distribution_alpha: float | None = 0.3,
    distribution_width_frac: float = 0.4,
    show_title: bool = True,
    show_suptitle: bool = True,
    include_group_in_title: bool = True,
    legend_title: str | None = None,
    save_dir: str | Path | None = None,
    filename_prefix: str | None = None,
    save_formats: tuple[str, ...] = ("pdf",),
) -> None:
    if df.empty:
        print(f"No rows to plot for: {title_prefix}")
        return
    missing = sorted({"head_idx", "layer_idx", "seqlen", metric} - set(df.columns))
    if missing:
        print(f"Skipping {title_prefix}: missing columns {missing}")
        return
    plot_df = df.copy()
    if model is not None:
        plot_df = plot_df[plot_df["model"] == model]
    metric_values = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df[np.isfinite(metric_values)]
    if plot_df.empty:
        print(f"No finite rows to plot for: {title_prefix}")
        return
    _plot_recurrent_metric(
        plot_df,
        metric=metric,
        title_prefix=title_prefix,
        group_key="head_idx",
        line_key="layer_idx",
        group_name="head",
        line_name="layer",
        model=model,
        training_context_length=training_context_length,
        log_x=log_x,
        log_y=log_y,
        plot_mode=plot_mode,
        run_alpha=run_alpha,
        distribution_alpha=distribution_alpha,
        distribution_width_frac=distribution_width_frac,
        show_title=show_title,
        show_suptitle=show_suptitle,
        include_group_in_title=include_group_in_title,
        legend_title=legend_title,
        save_dir=save_dir,
        filename_prefix=filename_prefix,
        save_formats=save_formats,
    )


def plot_avg_metric_per_layer_per_head(
    df: pd.DataFrame,
    *,
    metric: str,
    title_prefix: str,
    model: str | None = None,
    show_title: bool = True,
    show_suptitle: bool = True,
    legend_title: str | None = None,
    save_dir: str | Path | None = None,
    filename_prefix: str | None = None,
    save_formats: tuple[str, ...] = ("pdf",),
) -> None:
    if df.empty:
        print(f"No rows to plot for: {title_prefix}")
        return
    missing = sorted({"model", "layer_idx", "head_idx", metric} - set(df.columns))
    if missing:
        print(f"Skipping {title_prefix}: missing columns {missing}")
        return

    plot_df = df.copy()
    if model is not None:
        plot_df = plot_df[plot_df["model"] == model]
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df["layer_idx"] = pd.to_numeric(plot_df["layer_idx"], errors="coerce")
    plot_df["head_idx"] = pd.to_numeric(plot_df["head_idx"], errors="coerce")
    plot_df = plot_df[
        np.isfinite(plot_df[metric])
        & np.isfinite(plot_df["layer_idx"])
        & np.isfinite(plot_df["head_idx"])
    ].copy()
    plot_df = plot_df[(plot_df["layer_idx"] >= 0) & (plot_df["head_idx"] >= 0)].copy()
    if plot_df.empty:
        print(f"Skipping {title_prefix}: no finite rows.")
        return

    display_name_map = resolve_display_name_map(plot_df)
    avg_metric_df = (
        plot_df.groupby(["model", "head_idx", "layer_idx"], observed=True)[metric]
        .mean()
        .reset_index()
        .sort_values(["model", "head_idx", "layer_idx"])
    )
    if avg_metric_df.empty:
        print(f"Skipping {title_prefix}: no rows after aggregation.")
        return

    model_values = sorted(avg_metric_df["model"].astype(str).unique().tolist())
    split = model is None and len(model_values) > 1
    panel_models = model_values if split else [model_values[0]]
    fig, axes = create_panel_figure(
        panel_count=len(panel_models),
        figsize=(6.2 * len(panel_models), 4.8),
        sharey=split,
    )
    head_values = sorted(avg_metric_df["head_idx"].astype(int).unique().tolist())
    tab20_reordered = _reordered_tab20_palette()
    colors = {head_idx: tab20_reordered[i % len(tab20_reordered)] for i, head_idx in enumerate(head_values)}
    metric_label = _METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())
    ylabel = f"Average {metric_label} across sequence length"

    for idx, model_name in enumerate(panel_models):
        ax = axes[idx if split else 0]
        sub = avg_metric_df if not split else avg_metric_df[avg_metric_df["model"] == model_name]
        if sub.empty:
            ax.set_visible(False)
            continue
        for head_idx, head_df in sub.groupby("head_idx", observed=True):
            ax.plot(
                head_df["layer_idx"].astype(int),
                head_df[metric],
                marker="o",
                linewidth=1.7,
                color=colors[int(head_idx)],
                label=f"head[{int(head_idx)}]",
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel if idx == 0 else "")
        panel_title = (
            display_name_map.get(str(model_name), str(model_name))
            if split
            else title_prefix
        )
        ax.set_title(panel_title if show_title else "")
        ax.set_xticks(sorted(sub["layer_idx"].astype(int).unique().tolist()))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, title=legend_title)

    if split and show_suptitle:
        fig.suptitle(f"{title_prefix} (averaged across sequence lengths)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    _save_plot_figure(
        fig,
        save_dir=save_dir,
        filename_stem=filename_prefix or f"{metric}_avg_per_layer_per_head",
        save_formats=save_formats,
    )


def plot_metric_layer_seqlen_heatmap(
    df: pd.DataFrame,
    *,
    metric: str,
    title_prefix: str,
    model: str | None = None,
    training_context_length: int | None = None,
    cmap: str = "viridis",
    color_scale: Literal["linear", "log_reference_gap"] = "linear",
    reference_value: float = 1.0,
    show_title: bool = True,
    show_suptitle: bool = True,
    font_size: float = 12.0,
    save_dir: str | Path | None = None,
    filename_prefix: str | None = None,
    save_formats: tuple[str, ...] = ("pdf",),
) -> None:
    if df.empty:
        print(f"No rows to plot for: {title_prefix}")
        return
    missing = sorted({"model", "layer_idx", "seqlen", metric} - set(df.columns))
    if missing:
        print(f"Skipping {title_prefix}: missing columns {missing}")
        return

    plot_df = df.copy()
    if model is not None:
        plot_df = plot_df[plot_df["model"] == model]
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df[metric]) & (plot_df["layer_idx"] >= 0)]
    if plot_df.empty:
        print(f"No finite rows to plot for: {title_prefix}")
        return

    display_name_map = resolve_display_name_map(plot_df)
    heatmap_df = (
        plot_df.groupby(["model", "layer_idx", "seqlen"], observed=True)[metric]
        .mean()
        .reset_index()
    )
    model_values = sorted(heatmap_df["model"].astype(str).unique().tolist())
    split = model is None and len(model_values) > 1
    panel_models = model_values if split else [model_values[0]]
    fig, axes = create_panel_figure(
        panel_count=len(panel_models),
        figsize=(6.2 * len(panel_models), 4.8),
        sharey=True,
    )
    fig.subplots_adjust(right=0.925, wspace=0.08)

    if color_scale not in {"linear", "log_reference_gap"}:
        raise ValueError("color_scale must be 'linear' or 'log_reference_gap'.")

    metric_label = _METRIC_DISPLAY_NAMES.get(metric, metric)
    heatmap_value_col = "_heatmap_value"
    metric_values = heatmap_df[metric].astype(float)
    heatmap_cmap = cmap
    cbar_label = metric_label
    heatmap_df[heatmap_value_col] = metric_values
    image_kwargs: dict[str, Any] = {
        "vmin": float(metric_values.min()),
        "vmax": float(metric_values.max()),
    }

    if color_scale == "log_reference_gap":
        gap_values = (reference_value - metric_values).clip(lower=0.0)
        positive_values = gap_values[gap_values > 0].to_numpy()
        if positive_values.size:
            min_positive = float(np.nanmin(positive_values))
            max_positive = float(np.nanmax(positive_values))
            floor = max(
                min_positive / 2.0,
                max_positive / 1_000_000.0,
                np.finfo(float).tiny,
            )
            heatmap_df[heatmap_value_col] = gap_values.clip(lower=floor)
            image_kwargs = {
                "norm": mcolors.LogNorm(vmin=floor, vmax=max(max_positive, floor * 10.0))
            }
            if isinstance(cmap, str) and not cmap.endswith("_r"):
                heatmap_cmap = f"{cmap}_r"
            cbar_label = f"{reference_value:g} - {metric_label} (log scale)"

    for idx, model_name in enumerate(panel_models):
        ax = axes[idx if split else 0]
        sub = heatmap_df if not split else heatmap_df[heatmap_df["model"] == model_name]
        table = (
            sub.pivot(index="layer_idx", columns="seqlen", values=heatmap_value_col)
            .sort_index()
            .sort_index(axis=1)
        )
        image = ax.imshow(
            table.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap=heatmap_cmap,
            **image_kwargs,
        )
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(
            [str(int(v)) for v in table.columns],
            rotation=45,
            ha="right",
            fontsize=font_size,
        )
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([str(int(v) + 1) for v in table.index], fontsize=font_size)
        if training_context_length in set(int(v) for v in table.columns):
            ax.axvline(
                list(table.columns).index(training_context_length),
                color="white",
                linestyle="--",
                linewidth=1.2,
            )
        ax.set_xlabel("Sequence length", fontsize=font_size + 1)
        ax.set_ylabel("Layer" if idx == 0 else "", fontsize=font_size + 1)
        panel_title = (
            display_name_map.get(str(model_name), str(model_name))
            if split
            else title_prefix
        )
        ax.set_title(panel_title if show_title else "", fontsize=font_size + 2)
    if split and show_suptitle:
        fig.suptitle(title_prefix, fontsize=font_size + 4)
        fig.tight_layout(rect=(0, 0, 0.925, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 0.925, 1))
    cbar_ax = fig.add_axes([0.932, 0.18, 0.014, 0.66])
    cbar = fig.colorbar(image, cax=cbar_ax, label=cbar_label)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label(cbar_label, fontsize=font_size + 1)
    _save_plot_figure(
        fig,
        save_dir=save_dir,
        filename_stem=filename_prefix or f"{metric}_layer_seqlen_heatmap",
        save_formats=save_formats,
    )


def plot_recurrent_metric_per_layer(
    df: pd.DataFrame,
    *,
    metric: str,
    title_prefix: str,
    model: str | None = None,
    layer_indices: list[int] | tuple[int, ...] | None = None,
    head_indices: list[int] | tuple[int, ...] | None = None,
    training_context_length: int | None = None,
    log_x: bool = True,
    log_y: bool = False,
    plot_mode: Literal["individual_runs", "violin"] = "individual_runs",
    run_alpha: float = 0.35,
    distribution_alpha: float | None = 0.3,
    distribution_width_frac: float = 0.4,
    show_title: bool = True,
    show_suptitle: bool = True,
    include_group_in_title: bool = True,
    legend_title: str | None = None,
    save_dir: str | Path | None = None,
    filename_prefix: str | None = None,
    save_formats: tuple[str, ...] = ("pdf",),
) -> None:
    if df.empty:
        print(f"No rows to plot for: {title_prefix}")
        return
    missing = sorted({"head_idx", "layer_idx", "seqlen", metric} - set(df.columns))
    if missing:
        print(f"Skipping {title_prefix}: missing columns {missing}")
        return
    plot_df = df.copy()
    if model is not None:
        plot_df = plot_df[plot_df["model"] == model]
    if layer_indices is not None:
        plot_df = plot_df[plot_df["layer_idx"].isin({int(v) for v in layer_indices})]
    if head_indices is not None:
        plot_df = plot_df[plot_df["head_idx"].isin({int(v) for v in head_indices})]
    metric_values = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df[np.isfinite(metric_values)]
    if plot_df.empty:
        print(f"No finite rows to plot for: {title_prefix}")
        return
    _plot_recurrent_metric(
        plot_df,
        metric=metric,
        title_prefix=title_prefix,
        group_key="layer_idx",
        line_key="head_idx",
        group_name="layer",
        line_name="head",
        model=model,
        training_context_length=training_context_length,
        log_x=log_x,
        log_y=log_y,
        plot_mode=plot_mode,
        run_alpha=run_alpha,
        distribution_alpha=distribution_alpha,
        distribution_width_frac=distribution_width_frac,
        show_title=show_title,
        show_suptitle=show_suptitle,
        include_group_in_title=include_group_in_title,
        legend_title=legend_title,
        save_dir=save_dir,
        filename_prefix=filename_prefix,
        save_formats=save_formats,
    )
