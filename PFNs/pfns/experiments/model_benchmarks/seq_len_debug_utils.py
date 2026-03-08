from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from pfns.experiments.model_benchmarks.benchmark_batch_generators import (
    create_seq_len_batch_generator,
)
from pfns.experiments.model_benchmarks.model_registry import (
    get_autocast_models_from_registry,
)
from pfns.experiments.model_benchmarks.models import load_models_for_benchmark
from pfns.tensor_tree_utils import iter_named_tensors
from pfns.training_utils import (
    categorical_mask_to_inds,
    move_style_and_check_shape,
    move_y_style_and_check_shape,
)

DEFAULT_HIDDEN_STATE_NAME_HINTS = ("state", "cache", "kv", "ssm", "h0")
_LAYER_INDEX_PATTERN = re.compile(r"layers\[(\d+)\]")


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
        if not self.seqlen_list:
            raise ValueError("seqlen_list must be non-empty")
        if min(self.seqlen_list) < 1:
            raise ValueError("All values in seqlen_list must be >= 1")

    @property
    def sorted_seqlens(self) -> tuple[int, ...]:
        return tuple(sorted(self.seqlen_list))

    @property
    def smallest_seqlen(self) -> int:
        return int(min(self.seqlen_list))

    @property
    def largest_seqlen(self) -> int:
        return int(max(self.seqlen_list))

    @classmethod
    def from_mapping(cls, experiment: Mapping[str, Any]) -> "HiddenStateTrackingConfig":
        required_keys = (
            "num_classes",
            "num_features",
            "num_test_samples",
            "num_repetitions",
            "data_generation_seed",
            "seqlen_list",
        )
        missing = [key for key in required_keys if key not in experiment]
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


def _resolve_tracking_config(
    experiment: HiddenStateTrackingConfig | Mapping[str, Any],
) -> HiddenStateTrackingConfig:
    if isinstance(experiment, HiddenStateTrackingConfig):
        return experiment
    return HiddenStateTrackingConfig.from_mapping(experiment)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _iter_named_tensors(
    obj: Any,
    *,
    prefix: str = "state",
    visited: set[int] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    return list(iter_named_tensors(obj, prefix=prefix, visited=visited))


def _looks_like_hidden_state(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in DEFAULT_HIDDEN_STATE_NAME_HINTS)


def select_hidden_state_tensors(
    state: Any,
    *,
    tensor_name_patterns: list[str] | tuple[str, ...] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    named = _iter_named_tensors(state, prefix="state")
    if not named:
        return []

    if tensor_name_patterns is not None:
        patterns = [str(pattern).lower() for pattern in tensor_name_patterns if str(pattern)]
        if patterns:
            return [
                (name, tensor)
                for name, tensor in named
                if any(pattern in name.lower() for pattern in patterns)
            ]

    filtered = [(name, tensor) for name, tensor in named if _looks_like_hidden_state(name)]
    return filtered if filtered else named


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    arr = tensor.detach().float()
    numel = int(arr.numel())
    shape = tuple(int(v) for v in arr.shape)
    if numel == 0:
        return {
            "shape": shape,
            "numel": 0,
            "l2_norm": float("nan"),
            "abs_max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "finite_frac": float("nan"),
        }

    finite_mask = torch.isfinite(arr)
    finite_count = int(finite_mask.sum().item())
    if finite_count > 0:
        arr_finite = arr[finite_mask]
        l2_norm = float(torch.linalg.vector_norm(arr_finite).item())
        abs_max = float(arr_finite.abs().max().item())
        mean = float(arr_finite.mean().item())
        std = float(arr_finite.std(unbiased=False).item())
    else:
        l2_norm = float("nan")
        abs_max = float("nan")
        mean = float("nan")
        std = float("nan")
    return {
        "shape": shape,
        "numel": numel,
        "l2_norm": l2_norm,
        "abs_max": abs_max,
        "mean": mean,
        "std": std,
        "finite_frac": float(finite_count / numel),
    }


def _parse_layer_idx_from_name(tensor_name: str) -> int | None:
    match = _LAYER_INDEX_PATTERN.search(tensor_name)
    return int(match.group(1)) if match else None


def _iter_recurrent_state_head_matrices(
    recurrent_state: torch.Tensor,
) -> list[tuple[int, torch.Tensor]]:
    arr = recurrent_state.detach().float()
    if arr.ndim < 2:
        return []
    if arr.ndim == 2:
        return [(0, arr)]

    # Treat the third-to-last dim as head dim: [*, H, M, N]
    head_dim = arr.ndim - 3
    arr = arr.movedim(head_dim, 0)  # [H, *, M, N]
    head_count = int(arr.shape[0])
    if arr.ndim > 3:
        arr = arr.reshape(head_count, -1, arr.shape[-2], arr.shape[-1]).mean(dim=1)
    return [(head_idx, arr[head_idx]) for head_idx in range(head_count)]


def _matrix_head_metrics(matrix: torch.Tensor) -> dict[str, float]:
    matrix_f = matrix.detach().float()
    if matrix_f.numel() == 0:
        return {
            "fro_norm": float("nan"),
            "spectral_norm": float("nan"),
            "cond_proxy": float("nan"),
        }
    if not bool(torch.isfinite(matrix_f).all()):
        return {
            "fro_norm": float("nan"),
            "spectral_norm": float("nan"),
            "cond_proxy": float("nan"),
        }
    singular_values = torch.linalg.svdvals(matrix_f)
    if singular_values.numel() == 0:
        return {
            "fro_norm": float("nan"),
            "spectral_norm": float("nan"),
            "cond_proxy": float("nan"),
        }
    max_sv = float(singular_values.max().item())
    min_sv = float(singular_values.min().item())
    eps = 1e-12
    return {
        "fro_norm": float(torch.linalg.norm(matrix_f, ord="fro").item()),
        "spectral_norm": max_sv,
        "cond_proxy": float(max_sv / max(min_sv, eps)),
    }


def _build_model_eval_kwargs(
    *,
    base_batch: Any,
    seqlen: int,
    device: str,
    categorical_inds: Any,
) -> dict[str, Any]:
    train_x = base_batch.x[:, :seqlen]
    train_y = base_batch.y[:, :seqlen]

    x_train = train_x.to(device)
    y_train = train_y.to(device)
    style = move_style_and_check_shape(base_batch.style, train_x, device)
    y_style = move_y_style_and_check_shape(base_batch.y_style, train_y, device)

    fit_kwargs = {
        "x": x_train,
        "y": y_train,
        "style": style,
        "y_style": y_style,
        "categorical_inds": categorical_inds,
    }
    return fit_kwargs


def _evaluate_model_state(
    *,
    model_name: str,
    model: torch.nn.Module,
    fit_kwargs: dict[str, Any],
    autocast_models: dict[str, torch.dtype],
    device_type: str,
    is_cuda: bool,
) -> Any:
    with torch.inference_mode():
        with torch.autocast(
            device_type=device_type,
            enabled=is_cuda and model_name in autocast_models,
            dtype=autocast_models.get(model_name, torch.float32),
        ):
            return model.incontext_fit(**fit_kwargs)


def _rows_from_state(
    *,
    model_name: str,
    rep: int,
    seqlen: int,
    state: Any,
    tensor_name_patterns: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    named_tensors = select_hidden_state_tensors(
        state,
        tensor_name_patterns=tensor_name_patterns,
    )
    for tensor_name, tensor in named_tensors:
        stats = _tensor_stats(tensor)
        rows.append(
            {
                "model": model_name,
                "rep": int(rep),
                "seqlen": int(seqlen),
                "tensor_name": str(tensor_name),
                "shape": str(stats["shape"]),
                "numel": int(stats["numel"]),
                "l2_norm": float(stats["l2_norm"]),
                "abs_max": float(stats["abs_max"]),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "finite_frac": float(stats["finite_frac"]),
                "fro_norm": float("nan"),
                "spectral_norm": float("nan"),
                "cond_proxy": float("nan"),
                "state_scope": "tensor",
                "layer_idx": -1,
                "head_idx": -1,
            }
        )

        if "recurrent_state" not in tensor_name.lower():
            continue
        layer_idx = _parse_layer_idx_from_name(tensor_name)
        for head_idx, matrix in _iter_recurrent_state_head_matrices(tensor):
            matrix_stats = _tensor_stats(matrix)
            head_metrics = _matrix_head_metrics(matrix)
            rows.append(
                {
                    "model": model_name,
                    "rep": int(rep),
                    "seqlen": int(seqlen),
                    "tensor_name": f"{tensor_name}::head[{head_idx}]",
                    "shape": str(matrix_stats["shape"]),
                    "numel": int(matrix_stats["numel"]),
                    "l2_norm": float(matrix_stats["l2_norm"]),
                    "abs_max": float(matrix_stats["abs_max"]),
                    "mean": float(matrix_stats["mean"]),
                    "std": float(matrix_stats["std"]),
                    "finite_frac": float(matrix_stats["finite_frac"]),
                    "fro_norm": float(head_metrics["fro_norm"]),
                    "spectral_norm": float(head_metrics["spectral_norm"]),
                    "cond_proxy": float(head_metrics["cond_proxy"]),
                    "state_scope": "recurrent_state_head",
                    "layer_idx": int(layer_idx) if layer_idx is not None else -1,
                    "head_idx": int(head_idx),
                }
            )
    return rows


def run_hidden_state_tracking(
    *,
    experiment: HiddenStateTrackingConfig | Mapping[str, Any],
    models_to_compare: dict[str, Any],
    device: str,
    tensor_name_patterns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    cfg = _resolve_tracking_config(experiment)

    autocast_models = get_autocast_models_from_registry(models_to_compare, device=device)
    unsupported_eval_modes = {
        str(model_name): model_cfg.get("eval_mode")
        for model_name, model_cfg in models_to_compare.items()
        if model_cfg.get("eval_mode", "fit_predict") != "fit_predict"
    }
    if unsupported_eval_modes:
        raise ValueError(
            "seq_len_debug_utils only supports fit_predict models. "
            f"Found unsupported eval_mode entries: {unsupported_eval_modes}"
        )

    device_type = torch.device(device).type
    is_cuda = device_type == "cuda"

    rows: list[dict[str, Any]] = []
    seed_everything(cfg.data_generation_seed)
    models, _ = load_models_for_benchmark(models_to_compare, device=device)

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

    for rep, (base_batch, _) in enumerate(
        tqdm(batch_generator, total=cfg.num_repetitions, desc="Hidden-state tracking"),
        start=0,
    ):
        categorical_inds = categorical_mask_to_inds(base_batch.categorical_mask)
        for raw_model_name, model in models.items():
            model_name = str(raw_model_name)

            for seqlen in cfg.sorted_seqlens:
                fit_kwargs = _build_model_eval_kwargs(
                    base_batch=base_batch,
                    seqlen=seqlen,
                    device=device,
                    categorical_inds=categorical_inds,
                )

                try:
                    state = _evaluate_model_state(
                        model_name=model_name,
                        model=model,
                        fit_kwargs=fit_kwargs,
                        autocast_models=autocast_models,
                        device_type=device_type,
                        is_cuda=is_cuda,
                    )
                except torch.cuda.OutOfMemoryError:
                    if is_cuda:
                        torch.cuda.empty_cache()
                    break
                except RuntimeError as err:
                    if is_cuda and "out of memory" in str(err).lower():
                        torch.cuda.empty_cache()
                        break
                    raise

                rows.extend(
                    _rows_from_state(
                        model_name=model_name,
                        rep=rep,
                        seqlen=seqlen,
                        state=state,
                        tensor_name_patterns=tensor_name_patterns,
                    )
                )

    return pd.DataFrame(rows)


def summarize_hidden_state_by_seqlen(hidden_state_df: pd.DataFrame) -> pd.DataFrame:
    df = hidden_state_df.copy()
    for col, default in (
        ("state_scope", "tensor"),
        ("layer_idx", -1),
        ("head_idx", -1),
        ("fro_norm", float("nan")),
        ("spectral_norm", float("nan")),
        ("cond_proxy", float("nan")),
    ):
        if col not in df.columns:
            df[col] = default

    if hidden_state_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "tensor_name",
                "state_scope",
                "layer_idx",
                "head_idx",
                "seqlen",
                "numel",
                "l2_norm_mean",
                "l2_norm_std",
                "abs_max_mean",
                "mean_mean",
                "std_mean",
                "finite_frac_mean",
                "fro_norm_mean",
                "spectral_norm_mean",
                "cond_proxy_mean",
                "n",
            ]
        )

    group_cols = [
        col
        for col in ("model", "tensor_name", "state_scope", "layer_idx", "head_idx", "seqlen")
        if col in df.columns
    ]
    grouped = df.groupby(group_cols, observed=True)
    out = grouped.agg(
        numel=("numel", "max"),
        l2_norm_mean=("l2_norm", "mean"),
        l2_norm_std=("l2_norm", "std"),
        abs_max_mean=("abs_max", "mean"),
        mean_mean=("mean", "mean"),
        std_mean=("std", "mean"),
        finite_frac_mean=("finite_frac", "mean"),
        fro_norm_mean=("fro_norm", "mean"),
        spectral_norm_mean=("spectral_norm", "mean"),
        cond_proxy_mean=("cond_proxy", "mean"),
        n=("rep", "nunique"),
    )
    out = out.reset_index()
    sort_cols = [
        col
        for col in ("model", "state_scope", "layer_idx", "head_idx", "tensor_name", "seqlen")
        if col in out.columns
    ]
    return out.sort_values(sort_cols)


def plot_hidden_state_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    tensor_names: list[str] | None = None,
    model: str | None = None,
    log_x: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "plot_hidden_state_metric requires matplotlib. Install it in your notebook environment."
        ) from err

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

    agg = (
        plot_df.groupby(["tensor_name", "seqlen"], observed=True)[metric]
        .mean()
        .reset_index()
        .sort_values(["tensor_name", "seqlen"])
    )

    plt.figure(figsize=(10, 5))
    for tensor_name, group in agg.groupby("tensor_name", observed=True):
        plt.plot(group["seqlen"], group[metric], marker="o", label=str(tensor_name))

    if log_x:
        plt.xscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()


def plot_recurrent_metric_per_head(
    df: pd.DataFrame,
    *,
    metric: str,
    title_prefix: str,
    model: str | None = None,
    log_x: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "plot_recurrent_metric_per_head requires matplotlib. Install it in your notebook environment."
        ) from err

    if df.empty:
        print(f"No rows to plot for: {title_prefix}")
        return
    required_cols = {"state_scope", "head_idx", "layer_idx", "seqlen", metric}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        print(f"Skipping {title_prefix}: missing columns {missing}")
        return

    plot_df = df[df["state_scope"] == "recurrent_state_head"].copy()
    if model is not None:
        plot_df = plot_df[plot_df["model"] == model]
    if plot_df.empty:
        print(f"No recurrent_state_head rows to plot for: {title_prefix}")
        return

    head_values = sorted(int(v) for v in plot_df["head_idx"].dropna().unique())
    for head_idx in head_values:
        head_df = plot_df[plot_df["head_idx"] == head_idx]
        if head_df.empty:
            continue
        agg = (
            head_df.groupby(["layer_idx", "seqlen"], observed=True)[metric]
            .mean()
            .reset_index()
            .sort_values(["layer_idx", "seqlen"])
        )
        if agg.empty:
            continue

        plt.figure(figsize=(10, 5))
        for layer_idx, group in agg.groupby("layer_idx", observed=True):
            layer_label = f"layer[{int(layer_idx)}]" if int(layer_idx) >= 0 else "layer[unknown]"
            plt.plot(group["seqlen"], group[metric], marker="o", label=layer_label)

        if log_x:
            plt.xscale("log")
        plt.xlabel("Sequence length")
        plt.ylabel(metric)
        plt.title(f"{title_prefix} | head[{head_idx}]")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
