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
    get_models_from_names,
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
    "k_sum_norm": "K-Sum Norm",
    "joint_hidden_state_norm": "Joint Hidden-State Norm",
    "kv_over_ksum_ratio": "KV-over-K-Sum Ratio",
    "output_norm": "Output Norm",
}


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

def _prepare_embedded_train_input(
    model: Any,
    batch: Any,
    *,
    seqlen: int,
    device: str,
) -> torch.Tensor:
    x = batch.x[:, :seqlen]
    y = batch.y[:, :seqlen]
    style = move_style_and_check_shape(batch.style, x, device)
    y_style = move_y_style_and_check_shape(batch.y_style, y, device)
    categorical_inds = categorical_mask_to_inds(batch.categorical_mask)
    x_bf, y_bf, _ = model._prepare_batch_first_inputs(x.to(device), y.to(device), None)
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


def run_hidden_state_tracking(
    *,
    experiment: HiddenStateTrackingConfig | Mapping[str, Any],
    models_to_compare: dict[str, Any],
    device: str,
    tensor_name_patterns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    cfg = experiment if isinstance(experiment, HiddenStateTrackingConfig) else HiddenStateTrackingConfig.from_mapping(experiment)
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
                        rows.append(
                            {
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
                                "layer_idx": layer_idx,
                                "head_idx": int(head_idx),
                            }
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
    ):
        if col not in df.columns:
            df[col] = default
    if hidden_state_df.empty:
        cols = [
            "model", "tensor_name", "layer_idx", "head_idx", "seqlen",
            "abs_max_mean",
            "frobenius_norm_mean", "spectral_norm_mean", "effective_rank_mean", "stable_rank_mean", "n",
            "kv_state_norm_mean", "k_sum_norm_mean", "joint_hidden_state_norm_mean", "kv_over_ksum_ratio_mean",
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
            n=("rep", "nunique"),
        )
        .reset_index()
    )
    order = [c for c in ("model", "layer_idx", "head_idx", "tensor_name", "seqlen") if c in out.columns]
    return out.sort_values(order)

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
            violin_values: list[np.ndarray] = []
            for line_idx, line_value in enumerate(line_values):
                value = int(line_value)
                label = f"{line_name}[{value}]" if value >= 0 else f"{line_name}[unknown]"
                color = colors[line_value]
                raw_line_df = sub[sub[line_key] == line_value]
                if plot_mode == "violin":
                    finite_line_values = pd.to_numeric(
                        raw_line_df[metric],
                        errors="coerce",
                    ).to_numpy(dtype=float, copy=False)
                    finite_line_values = finite_line_values[np.isfinite(finite_line_values)]
                    if log_y:
                        finite_line_values = finite_line_values[finite_line_values > 0.0]
                    if finite_line_values.size > 0:
                        violin_values.append(finite_line_values)
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
            if plot_mode == "violin" and violin_values:
                all_violin_values = np.concatenate(
                    [values.astype(float, copy=False) for values in violin_values if values.size > 0]
                )
                finite_violin_values = all_violin_values[np.isfinite(all_violin_values)]
                if finite_violin_values.size > 0:
                    y_min = float(finite_violin_values.min())
                    y_max = float(finite_violin_values.max())
                    ax.set_ylim(*_compute_padded_y_limits(y_min, y_max, log_scale=log_y))
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
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel(metric_label if idx == 0 else "")
            ax.set_title(
                display_name_map.get(str(model_name), str(model_name))
                if split
                else f"{title_prefix} | {group_name}[{int(group_value)}]"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8, ncol=2)
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
        if split:
            fig.suptitle(f"{title_prefix} | {group_name}[{int(group_value)}]")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            fig.tight_layout()


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
    )


def plot_avg_metric_per_layer_per_head(
    df: pd.DataFrame,
    *,
    metric: str,
    title_prefix: str,
    model: str | None = None,
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
    display_name_map = resolve_display_name_map(avg_metric_df)
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
    ylabel = f"Average {metric_label} Across Seq Len"

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
        ax.set_title(
            display_name_map.get(str(model_name), str(model_name))
            if split
            else title_prefix
        )
        ax.set_xticks(sorted(sub["layer_idx"].astype(int).unique().tolist()))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    if split:
        fig.suptitle(f"{title_prefix} (averaged across sequence lengths)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()


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
    )

def _finite_mean_std(values: torch.Tensor) -> tuple[float, float]:
    values = values.reshape(-1).float()
    values = values.masked_fill(~torch.isfinite(values), torch.nan)
    if torch.isnan(values).all():
        return float("nan"), float("nan")
    mean = torch.nanmean(values)
    variance = torch.nanmean((values - mean).square())
    return float(mean.item()), float(variance.sqrt().item())


def _metric_summary(values_by_name: Mapping[str, torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, values in values_by_name.items():
        mean, std = _finite_mean_std(values)
        out[name] = mean
        out[f"{name}_std"] = std
    return out


def _prepare_position_metric_attention(
    layer: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    q_mapped, k_mapped = layer._apply_query_key_feature_maps(q, k)
    if layer.causal or layer.causal_train_only:
        attn, _, _ = layer._causal_attention(q, k, v)
        return k_mapped, attn, None, None

    final_kv_state = torch.einsum("bshf,bshd->bhfd", k_mapped, v)
    final_k_sum = k_mapped.sum(dim=1)
    attn = layer._read_from_kv_state(
        q_mapped,
        final_kv_state,
        final_k_sum if layer.use_k_sum_normalization else None,
    )
    final_kv_norm = torch.linalg.vector_norm(final_kv_state.float().flatten(start_dim=-2), dim=-1)
    final_k_sum_norm = torch.linalg.vector_norm(final_k_sum.float(), dim=-1)
    return k_mapped, attn, final_kv_norm, final_k_sum_norm


def _track_linear_attention_layer_position_metrics(
    *,
    model_name: str,
    layer_idx: int,
    k: torch.Tensor,
    v: torch.Tensor,
    attn: torch.Tensor,
    final_kv_norm: torch.Tensor | None,
    final_k_sum_norm: torch.Tensor | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    running_kv_state = torch.zeros(
        (k.shape[0], k.shape[2], k.shape[3], v.shape[3]),
        device=k.device,
        dtype=torch.float32,
    )
    running_k_sum = torch.zeros(
        (k.shape[0], k.shape[2], k.shape[3]),
        device=k.device,
        dtype=torch.float32,
    )
    for token_idx in range(int(k.shape[1])):
        position = int(token_idx + 1)
        k_t = k[:, token_idx].float()
        v_t = v[:, token_idx].float()
        delta_kv_t = torch.einsum("bhf,bhd->bhfd", k_t, v_t)
        running_kv_state = running_kv_state + delta_kv_t
        running_k_sum = running_k_sum + k_t
        if final_kv_norm is None or final_k_sum_norm is None:
            k_vals = torch.linalg.vector_norm(running_k_sum, dim=-1)
            kv_vals = torch.linalg.vector_norm(running_kv_state.flatten(start_dim=-2), dim=-1)
        else:
            k_vals = final_k_sum_norm
            kv_vals = final_kv_norm

        stats = _metric_summary(
            {
                "k_sum_norm": k_vals,
                "kv_state_norm": kv_vals,
                "kv_over_ksum_ratio": kv_vals / k_vals.clamp_min(1e-12),
                "output_norm": torch.linalg.vector_norm(attn[:, token_idx].float(), dim=-1),
            }
        )
        rows.append({"model": model_name, "layer_idx": int(layer_idx), "position": position, **stats})
    return rows


def _track_custom_linear_attention_position_metrics(
    model: Any,
    embedded: torch.Tensor,
    *,
    model_name: str,
) -> pd.DataFrame:
    backbone = getattr(model, "transformer_layers", None)
    if backbone is None or backbone.__class__.__name__ != "LinearAttentionBackbone":
        raise TypeError(f"{model_name} is not backed by the custom LinearAttentionBackbone.")

    out = embedded
    rows: list[dict[str, object]] = []
    for layer_idx, layer in enumerate(backbone.layers):
        q, k, v = layer._project_qkv(out)
        k_mapped, attn, final_kv_norm, final_k_sum_norm = _prepare_position_metric_attention(layer, q, k, v)
        rows.extend(
            _track_linear_attention_layer_position_metrics(
                model_name=model_name,
                layer_idx=int(layer_idx),
                k=k_mapped,
                v=v,
                attn=attn,
                final_kv_norm=final_kv_norm,
                final_k_sum_norm=final_k_sum_norm,
            )
        )
        out = layer._apply_output(out, attn)
        del q, k, v, k_mapped, attn
        if final_kv_norm is not None:
            del final_kv_norm, final_k_sum_norm
    return pd.DataFrame(rows)


def run_position_metric_tracking(
    *,
    experiment: Mapping[str, Any],
    device: str,
    training_context_length: int | None = None,
    partial_cache_path: Path | None = None,
) -> pd.DataFrame:
    model_configs = get_models_from_names(list(experiment["model_names"]))
    models, autocast_models = _device_runtime(model_configs, device)

    rows: list[dict[str, object]] = []
    completed_pairs: set[tuple[str, int]] = set()
    if partial_cache_path is not None and partial_cache_path.exists():
        partial_df = pd.read_pickle(partial_cache_path)
        if not partial_df.empty:
            rows.extend(partial_df.to_dict(orient="records"))
            completed_pairs = {
                (str(model_name), int(rep_idx))
                for model_name, rep_idx in partial_df[["model", "rep"]].drop_duplicates().itertuples(index=False, name=None)
            }
            print(f"Loaded partial position_metric_df from cache: {partial_cache_path}")

    seqlen = int(experiment["seqlen"])
    _ = training_context_length
    _set_data_generation_seed(int(experiment["data_generation_seed"]))
    batch_generator = create_seq_len_batch_generator(
        task_variant="tabular_prior",
        num_batches=int(experiment["num_repetitions"]),
        smallest_seqlen=seqlen,
        largest_seqlen=seqlen,
        num_features=int(experiment["num_features"]),
        num_classes=int(experiment["num_classes"]),
        number_of_test_samples=int(experiment["num_test_samples"]),
        default_device=device,
        task_kwargs={},
    )
    for rep, (base_batch, _) in enumerate(
        tqdm(batch_generator, total=int(experiment["num_repetitions"]), desc="Position-metric tracking")
    ):
        rep_rows: list[dict[str, object]] = []
        for raw_name, model in models.items():
            model_name = str(raw_name)
            if (model_name, int(rep)) in completed_pairs:
                print(f"Skipping cached position metrics for model={model_name}, rep={rep}")
                continue
            embedded = _run_autocast(
                lambda: _prepare_embedded_train_input(model, base_batch, seqlen=seqlen, device=device),
                model_name=model_name,
                autocast_models=autocast_models,
            )
            metric_df = _run_autocast(
                lambda: _track_custom_linear_attention_position_metrics(model, embedded, model_name=model_name),
                model_name=model_name,
                autocast_models=autocast_models,
            )
            metric_df["rep"] = int(rep)
            metric_df["seqlen"] = seqlen
            rep_rows.extend(metric_df.to_dict(orient="records"))
            completed_pairs.add((model_name, int(rep)))
            del embedded, metric_df
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if rep_rows:
            rows.extend(rep_rows)
            if partial_cache_path is not None:
                pd.DataFrame(rows).to_pickle(partial_cache_path)
    return pd.DataFrame(rows)


def auto_select_position_metric_layers(
    layer_values: list[int] | tuple[int, ...],
    *,
    max_layers: int = 3,
) -> list[int]:
    layer_values = sorted(int(v) for v in layer_values)
    if len(layer_values) <= max_layers:
        return layer_values
    candidate_indices = [0, len(layer_values) // 2, len(layer_values) - 1]
    return [layer_values[idx] for idx in sorted(set(candidate_indices))]


def plot_position_metric_per_layer(
    df: pd.DataFrame,
    *,
    metric: str,
    title_prefix: str,
    model_order: list[str] | tuple[str, ...] | None = None,
    layer_indices: list[int] | tuple[int, ...] | None = None,
    log_x: bool = True,
    y_scale: Literal["linear", "log", "symlog"] = "linear",
    y_eps: float = 1e-12,
    show_std_shading: bool = False,
    symlog_linthresh: float = 1e-3,
    seqlen: int | None = None,
) -> None:
    if df.empty:
        print(f"No rows to plot for: {title_prefix}")
        return
    metric_std = f"{metric}_std"
    missing = sorted({"model", "layer_idx", "position", metric, metric_std} - set(df.columns))
    if missing:
        print(f"Skipping {title_prefix}: missing columns {missing}")
        return

    plot_df = df.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df[metric_std] = pd.to_numeric(plot_df[metric_std], errors="coerce").fillna(0.0)
    plot_df["layer_idx"] = pd.to_numeric(plot_df["layer_idx"], errors="coerce")
    plot_df["position"] = pd.to_numeric(plot_df["position"], errors="coerce")
    plot_df = plot_df[
        np.isfinite(plot_df[metric])
        & np.isfinite(plot_df["layer_idx"])
        & np.isfinite(plot_df["position"])
    ].copy()
    if layer_indices is not None:
        plot_df = plot_df[plot_df["layer_idx"].isin({int(v) for v in layer_indices})]
    if plot_df.empty:
        print(f"No finite rows to plot for: {title_prefix}")
        return

    panel_models = (
        sorted(plot_df["model"].astype(str).unique().tolist())
        if model_order is None
        else [str(model_name) for model_name in model_order]
    )
    display_name_map = resolve_display_name_map(plot_df)
    fig, axes = create_panel_figure(panel_count=len(panel_models), figsize=(7 * len(panel_models), 4), sharey=False)
    for idx, model_name in enumerate(panel_models):
        ax = axes[idx]
        sub = plot_df[plot_df["model"].astype(str) == str(model_name)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        summary = (
            sub.groupby(["layer_idx", "position"], observed=True)
            .agg(mean=(metric, "mean"), std=(metric_std, "mean"))
            .reset_index()
            .sort_values(["layer_idx", "position"])
        )
        for layer_idx, layer_df in summary.groupby("layer_idx", observed=True):
            x = layer_df["position"].to_numpy(dtype=float, copy=False)
            y = layer_df["mean"].to_numpy(dtype=float, copy=False)
            y_std = layer_df["std"].to_numpy(dtype=float, copy=False)
            if y_scale == "log":
                y_line = np.clip(y, y_eps, None)
                y_lo = np.clip(y - y_std, y_eps, None)
                y_hi = np.clip(y + y_std, y_eps, None)
            else:
                y_line = y
                y_lo = y - y_std
                y_hi = y + y_std
            ax.plot(x, y_line, label=f"L{int(layer_idx)}", linewidth=1.5)
            if show_std_shading:
                ax.fill_between(x, y_lo, y_hi, alpha=0.15)
        if log_x:
            ax.set_xscale("log")
        if y_scale == "log":
            ax.set_yscale("log")
        elif y_scale == "symlog":
            ax.set_yscale("symlog", linthresh=symlog_linthresh)
        ax.set_title(display_name_map.get(str(model_name), str(model_name)), loc="center", x=0.5, pad=12)
        ax.set_xlabel("Token position")
        ax.set_ylabel(_METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title()))
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)

    title = f"{title_prefix} (seqlen={int(seqlen)})" if seqlen is not None else title_prefix
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
