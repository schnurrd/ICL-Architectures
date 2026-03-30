from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

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
from pfns.tensor_tree_utils import iter_named_tensors
from pfns.training_utils import (
    categorical_mask_to_inds,
    is_autocast_dtype_enabled,
    move_style_and_check_shape,
    move_y_style_and_check_shape,
)

_HIDDEN_STATE_HINTS = ("state", "cache", "kv", "ssm", "h0")
_LAYER_PATTERN = re.compile(r"(?:layers|layer_states)\[(\d+)\]")
_MATRIX_METRICS = ("frobenius_norm", "spectral_norm", "effective_rank")


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


def _is_matrix_state(name: str) -> bool:
    lowered = name.lower()
    return "recurrent_state" in lowered or "kv_state" in lowered


def _head_matrices(tensor: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
    arr = tensor.detach().float() # can be (1, 1, num_heads, h_dim, h_dim) or (num_heads, h_dim, h_dim)
    if arr.ndim < 2:
        return []
    if arr.ndim == 2:
        return [(0, arr)]
    arr = arr.movedim(arr.ndim - 3, 0)
    if arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], -1, arr.shape[-2], arr.shape[-1]).mean(dim=1)
    return [(head_idx, arr[head_idx]) for head_idx in range(int(arr.shape[0]))]


def _effective_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    singular_values = singular_values[singular_values > 0]
    if singular_values.numel() == 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    return float((-(probs * probs.log()).sum()).exp().item())


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
                for name, tensor in _iter_hidden_tensors(state, tensor_name_patterns):
                    if not _is_matrix_state(name):
                        continue
                    for head_idx, matrix in _head_matrices(tensor):
                        metrics = _matrix_metrics(matrix)
                        rows.append(
                            {
                                "model": model_name,
                                **extra,
                                "tensor_name": name,
                                "shape": str(metrics["shape"]),
                                "abs_max": float(metrics["abs_max"]),
                                **{k: float(metrics[k]) for k in _MATRIX_METRICS},
                                "layer_idx": _layer_idx(name),
                                "head_idx": int(head_idx),
                            }
                        )
    return pd.DataFrame(rows)


def _rows_from_cache_trajectory(*, model_name: str, token_idx: int, cache_params: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, tensor in _iter_hidden_tensors(cache_params):
        if not _is_matrix_state(name):
            continue
        for head_idx, matrix in _head_matrices(tensor):
            metrics = _matrix_metrics(matrix)
            rows.append(
                {
                    "model": model_name,
                    "token_idx": int(token_idx),
                    "tensor_name": name,
                    "layer_idx": _layer_idx(name),
                    "head_idx": int(head_idx),
                    "abs_max": float(metrics["abs_max"]),
                    **{k: float(metrics[k]) for k in _MATRIX_METRICS},
                }
            )
    return rows


def _fla_recurrent_state_trajectory(
    backbone: Any,
    embedded: torch.Tensor,
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    prepared, _ = backbone._prepare_fla_input(embedded)
    cache = None
    rows: list[dict[str, Any]] = []
    for token_idx in tqdm(range(int(prepared.shape[1])), total=int(prepared.shape[1]), desc=f"Recurrent-state trajectory ({model_name})"):
        token = prepared[:, token_idx : token_idx + 1]
        _, cache = backbone._run_fla(
            token,
            cache_params=cache,
            cache_position_start=token_idx if cache is not None else None,
            return_cache=True,
        )
        rows.extend(_rows_from_cache_trajectory(model_name=model_name, token_idx=token_idx + 1, cache_params=cache))
    return rows


def run_recurrent_state_trajectory_tracking(
    *,
    experiment: HiddenStateTrackingConfig | Mapping[str, Any],
    models_to_compare: dict[str, Any],
    device: str,
    seqlen: int,
    rep: int = 0,
) -> pd.DataFrame:
    cfg = experiment if isinstance(experiment, HiddenStateTrackingConfig) else HiddenStateTrackingConfig.from_mapping(experiment)
    if seqlen < 1 or rep < 0:
        raise ValueError("seqlen must be >= 1 and rep must be >= 0")
    _set_data_generation_seed(cfg.data_generation_seed)
    models, autocast_models = _device_runtime(models_to_compare, device)

    base_batch = None
    generator = create_seq_len_batch_generator(
        task_variant="tabular_prior",
        num_batches=rep + 1,
        smallest_seqlen=seqlen,
        largest_seqlen=seqlen,
        num_features=cfg.num_features,
        num_classes=cfg.num_classes,
        number_of_test_samples=cfg.num_test_samples,
        default_device=device,
        task_kwargs={},
    )
    for current_rep, (candidate, _) in enumerate(generator):
        if current_rep == rep:
            base_batch = candidate
            break
    if base_batch is None:
        raise RuntimeError(f"Unable to materialize repetition {rep} for seqlen={seqlen}.")

    categorical_inds = categorical_mask_to_inds(base_batch.categorical_mask)
    rows: list[dict[str, Any]] = []
    for raw_name, model in models.items():
        model_name = str(raw_name)
        x = base_batch.x[:, :seqlen]
        y = base_batch.y[:, :seqlen]
        x_device = x.to(device)
        y_device = y.to(device)
        style = move_style_and_check_shape(base_batch.style, x, device)
        y_style = move_y_style_and_check_shape(base_batch.y_style, y, device)
        if not hasattr(model, "_prepare_batch_first_inputs") or not hasattr(model, "_build_embedded_input"):
            raise TypeError(
                "Trajectory tracking requires a TabularModel-like interface with "
                "_prepare_batch_first_inputs and _build_embedded_input."
            )
        backbone = getattr(model, "transformer_layers", None)
        if backbone is None or not hasattr(backbone, "_run_fla") or not hasattr(backbone, "_prepare_fla_input"):
            raise TypeError("Trajectory tracking currently supports FLA backbones only.")

        x_bf, y_bf, _ = model._prepare_batch_first_inputs(x_device, y_device, None)
        assert x_bf is not None and y_bf is not None
        embedded, _, _, _ = model._build_embedded_input(
            x_bf,
            y_bf,
            single_eval_pos=int(y_bf.shape[1]),
            style=style,
            y_style=y_style,
            categorical_inds=categorical_inds,
            cache_trainset_representation=True,
        )

        try:
            rows.extend(
                _run_autocast(
                    lambda: _fla_recurrent_state_trajectory(backbone, embedded, model_name=model_name),
                    model_name=model_name,
                    autocast_models=autocast_models,
                )
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        except RuntimeError as err:
            if "out of memory" in str(err).lower():
                torch.cuda.empty_cache()
            raise
    return pd.DataFrame(rows)


def summarize_hidden_state_by_seqlen(hidden_state_df: pd.DataFrame) -> pd.DataFrame:
    df = hidden_state_df.copy()
    for col, default in (
        ("layer_idx", -1),
        ("head_idx", -1),
        *((key, float("nan")) for key in _MATRIX_METRICS),
    ):
        if col not in df.columns:
            df[col] = default
    if hidden_state_df.empty:
        cols = [
            "model", "tensor_name", "layer_idx", "head_idx", "seqlen",
            "abs_max_mean",
            "frobenius_norm_mean", "spectral_norm_mean", "effective_rank_mean", "n",
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
            n=("rep", "nunique"),
        )
        .reset_index()
    )
    order = [c for c in ("model", "layer_idx", "head_idx", "tensor_name", "seqlen") if c in out.columns]
    return out.sort_values(order)


def _short_tensor_label(name: str) -> str:
    for prefix in ("state.backbone_state.", "state."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    layer = _layer_idx(name)
    if match := re.search(r"(?:layers|layer_states)\[\d+\]\.(.+)", name):
        name = match.group(1)
    name = name.replace("::", " ").replace("_", " ")
    return f"L{layer} {name}" if layer >= 0 else name

def plot_hidden_state_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    tensor_names: list[str] | None = None,
    model: str | None = None,
    log_x: bool = True,
) -> None:
    if df.empty:
        print(f"No rows to plot for: {title}")
        return
    plot_df = df.copy()
    if model is not None:
        plot_df = plot_df[plot_df["model"] == model]
    if tensor_names is not None:
        plot_df = plot_df[plot_df["tensor_name"].isin(tensor_names)]
    if plot_df.empty:
        print(f"No matching rows for: {title}")
        return

    model_values = sorted(plot_df["model"].astype(str).unique().tolist())
    split = model is None and len(model_values) > 1
    panel_models = model_values if split else [model_values[0]]
    fig, axes = plt.subplots(1, len(panel_models), figsize=(6.4 * len(panel_models), 5), sharey=split, squeeze=False)
    palette = list(mcolors.TABLEAU_COLORS.values())
    for idx, model_name in enumerate(panel_models):
        ax = axes[0, idx if split else 0]
        sub = plot_df if not split else plot_df[plot_df["model"] == model_name]
        agg = sub.groupby(["tensor_name", "seqlen"], observed=True)[metric].mean().reset_index().sort_values(["tensor_name", "seqlen"])
        names = agg["tensor_name"].astype(str).unique().tolist()
        base = np.array(mcolors.to_rgb(palette[idx % len(palette)]))
        shades = np.linspace(0.55, 0.05, num=max(len(names), 1))
        colors = {name: tuple((1 - a) * base + a * np.ones(3)) for name, a in zip(names, shades, strict=False)}
        for name, group in agg.groupby("tensor_name", observed=True):
            ax.plot(group["seqlen"], group[metric], marker="o", linewidth=1.6, markersize=3.5, color=colors[str(name)], label=_short_tensor_label(str(name)))
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel("Sequence length")
        ax.set_ylabel(metric)
        ax.set_title(str(model_name) if split else title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
    if split:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()


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
) -> None:
    model_values = sorted(plot_df["model"].astype(str).unique().tolist())
    split = model is None and len(model_values) > 1
    for group_value, group_df in plot_df.groupby(group_key, observed=True):
        panel_models = model_values if split else [model_values[0]]
        fig, axes = plt.subplots(1, len(panel_models), figsize=(6.4 * len(panel_models), 5), sharey=split, squeeze=False)
        for idx, model_name in enumerate(panel_models):
            ax = axes[0, idx if split else 0]
            sub = group_df if not split else group_df[group_df["model"] == model_name]
            agg = sub.groupby([line_key, "seqlen"], observed=True)[metric].mean().reset_index().sort_values([line_key, "seqlen"])
            if agg.empty:
                ax.set_visible(False)
                continue
            for line_value, line_df in agg.groupby(line_key, observed=True):
                value = int(line_value)
                label = f"{line_name}[{value}]" if value >= 0 else f"{line_name}[unknown]"
                ax.plot(line_df["seqlen"], line_df[metric], marker="o", linewidth=1.3, label=label)
            if training_context_length is not None:
                ax.axvline(
                    int(training_context_length),
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label="Train context",
                )
            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("Sequence length")
            ax.set_ylabel(metric)
            ax.set_title(str(model_name) if split else f"{title_prefix} | {group_name}[{int(group_value)}]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8, ncol=2)
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
    log_x: bool = True,
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
    if plot_df.empty:
        print(f"No matching rows to plot for: {title_prefix}")
        return
    _plot_recurrent_metric(plot_df, metric=metric, title_prefix=title_prefix, group_key="head_idx", line_key="layer_idx", group_name="head", line_name="layer", model=model, log_x=log_x)


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
    if plot_df.empty:
        print(f"No matching rows to plot for: {title_prefix}")
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
    )


def plot_recurrent_effective_rank_trajectory(
    df: pd.DataFrame,
    *,
    title_prefix: str = "Recurrent-state effective rank over token positions",
    model: str | None = None,
    layer_indices: list[int] | tuple[int, ...] | None = None,
    head_indices: list[int] | tuple[int, ...] | None = None,
    training_context_length: int | None = None,
    boundary_positions: list[int] | tuple[int, ...] | None = None,
    max_heads: int | None = 4,
    show_legend: bool = True,
) -> None:
    if df.empty:
        print(f"No rows to plot for: {title_prefix}")
        return
    missing = sorted({"model", "token_idx", "layer_idx", "head_idx", "effective_rank"} - set(df.columns))
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
    if plot_df.empty:
        print(f"No matching rows to plot for: {title_prefix}")
        return
    model_values = [model] if model is not None else sorted(plot_df["model"].astype(str).unique().tolist())
    layer_values = sorted((int(v) for v in plot_df["layer_idx"].dropna().unique()), reverse=True)
    head_values = sorted(int(v) for v in plot_df["head_idx"].dropna().unique())
    if max_heads is not None:
        head_values = head_values[: int(max_heads)]
        plot_df = plot_df[plot_df["head_idx"].isin(head_values)]
    if plot_df.empty or not layer_values or not head_values:
        print(f"No matching rows to plot for: {title_prefix}")
        return

    fig, axes = plt.subplots(
        len(layer_values),
        len(model_values),
        figsize=(4.2 * len(model_values), 2.4 * len(layer_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = {head_idx: f"C{idx}" for idx, head_idx in enumerate(head_values)}
    for row_idx, layer_idx in enumerate(layer_values):
        for col_idx, model_name in enumerate(model_values):
            ax = axes[row_idx, col_idx]
            sub = plot_df[(plot_df["model"] == model_name) & (plot_df["layer_idx"] == layer_idx)].sort_values(["head_idx", "token_idx"])
            if sub.empty:
                ax.set_visible(False)
                continue
            for head_idx, group in sub.groupby("head_idx", observed=True):
                ax.plot(group["token_idx"], group["effective_rank"], linewidth=1.1, alpha=0.9, color=colors[int(head_idx)], label=f"head[{int(head_idx)}]")
            for idx, position in enumerate(boundary_positions or ()):
                ax.axvline(
                    int(position),
                    color="gray",
                    linestyle="-",
                    linewidth=0.8,
                    alpha=0.35,
                    label="Boundary" if idx == 0 else None,
                )
            if training_context_length is not None:
                ax.axvline(
                    int(training_context_length),
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label="Train context",
                )
            ax.grid(True, alpha=0.25)
            if len(layer_values) == 1:
                ax.set_title(str(model_name) if len(model_values) > 1 else title_prefix)
            elif row_idx == 0:
                ax.set_title(str(model_name))
            if col_idx == 0:
                ax.set_ylabel(f"Layer {int(layer_idx)}\nEffective rank")
            if row_idx == len(layer_values) - 1:
                ax.set_xlabel("Sequence length")

    if len(layer_values) > 1 or len(model_values) > 1:
        fig.suptitle(title_prefix, y=0.975)
    if show_legend and head_values:
        from matplotlib.lines import Line2D

        fig.legend(
            handles=[Line2D([0], [0], color=colors[h], linewidth=1.2, label=f"head[{h}]") for h in head_values],
            loc="upper center",
            ncol=min(len(head_values), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
