"""I/O helpers for experiment result bundles."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import wandb

from .analysis import long_df_to_nested_metric_table, nested_metric_table_to_long_df
from .constants import SCHEMA_VERSION


def make_bundle_path(root_dir: str | Path, experiment_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(root_dir) / f"{experiment_name}_{timestamp}"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value, key=lambda x: str(x))]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__module__") and value.__module__.startswith(("torch", "numpy")):
        return str(value)
    return value


def sanitize_wandb_artifact_component(value: str) -> str:
    """Normalize artifact-name parts to a conservative W&B-safe token."""
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    return token.strip("_") or "unnamed"


def make_model_artifact_name(
    *,
    base_artifact_name: str,
    model_name: str,
    model_hash: str,
) -> str:
    """Build deterministic per-model artifact names."""
    return (
        f"{sanitize_wandb_artifact_component(base_artifact_name)}_"
        f"{sanitize_wandb_artifact_component(model_name)}_"
        f"{sanitize_wandb_artifact_component(model_hash)}"
    )


def run_metadata_matches(
    run_metadata: dict[str, Any],
    *,
    expected: dict[str, Any],
    keys: tuple[str, ...],
) -> bool:
    """Return True when all selected metadata keys match expected values exactly."""
    return all(run_metadata.get(key) == expected.get(key) for key in keys)


def merge_model_results(
    results_by_model: dict[str, dict[str, Any]],
    *,
    merged_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge single-model benchmark outputs into one canonical results dict."""
    if not results_by_model:
        raise ValueError("results_by_model cannot be empty.")

    merged_metric: dict[str, dict[str, dict[int, list[float]]]] = {}
    merged_timing: dict[str, dict[str, dict[int, list[float]]]] = {}
    merged_memory: dict[str, dict[str, dict[int, list[float]]]] = {}
    merged_oom: dict[str, list[int]] = {}
    schema_version = SCHEMA_VERSION
    fallback_metadata: dict[str, Any] | None = None

    for model_name, result in results_by_model.items():
        raw_schema = result.get("schema_version", SCHEMA_VERSION)
        schema_version = int(float(raw_schema if raw_schema is not None else SCHEMA_VERSION))
        fallback_metadata = fallback_metadata or result.get("metadata", {})

        metric_table = result.get("metric_table", {})
        timing_table = result.get("timing_table", {})
        memory_table = result.get("memory_table", {})
        oom_errors = result.get("oom_errors", {})

        if model_name not in metric_table or model_name not in timing_table:
            raise KeyError(
                f"Model '{model_name}' not found in result tables. "
                "Expected one-model result dict per model."
            )

        merged_metric[model_name] = metric_table[model_name]
        merged_timing[model_name] = timing_table[model_name]
        merged_memory[model_name] = memory_table.get(model_name, {})

        raw_oom = oom_errors.get(model_name, [])
        merged_oom[model_name] = sorted({int(x) for x in raw_oom})

    return {
        "schema_version": schema_version,
        "metric_table": merged_metric,
        "timing_table": merged_timing,
        "memory_table": merged_memory,
        "oom_errors": merged_oom,
        "metadata": merged_metadata if merged_metadata is not None else (fallback_metadata or {}),
    }


def save_results_bundle(
    results: dict[str, Any],
    bundle_dir: str | Path,
    *,
    files: dict[str, str] | None = None,
    experiment: dict[str, Any] | None = None,
    include_raw_torch: bool = True,
    schema_version: int = SCHEMA_VERSION,
) -> Path:
    """Persist benchmark results using the canonical bundle layout documented above."""
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    metric_df = nested_metric_table_to_long_df(results.get("metric_table", {}))
    timing_df = nested_metric_table_to_long_df(results.get("timing_table", {}))
    memory_df = nested_metric_table_to_long_df(results.get("memory_table", {}))

    metric_df.to_csv(bundle / files["metric"], index=False)
    timing_df.to_csv(bundle / files["timing"], index=False)
    memory_df.to_csv(bundle / files["memory"], index=False)

    with open(bundle / files["oom"], "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results.get("oom_errors", {})), f, indent=2, sort_keys=True)

    metadata = {
        "schema_version": int(schema_version),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": experiment or {},
        "run_metadata": results.get("metadata", {}),
        "files": files,
        "row_counts": {
            "metric": int(len(metric_df)),
            "timing": int(len(timing_df)),
            "memory": int(len(memory_df)),
        },
    }
    with open(bundle / files["metadata"], "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metadata), f, indent=2, sort_keys=True)

    if include_raw_torch:
        torch.save(results, bundle / files["raw"])

    return bundle


def save_dataframe_bundle(
    *,
    dataframes: dict[str, pd.DataFrame | None],
    bundle_dir: str | Path,
    experiment: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
    files: dict[str, str] | None = None,
    schema_version: int = 1,
) -> Path:
    """Persist arbitrary named dataframes as a simple CSV+metadata bundle."""
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    dataframe_files = files or {name: f"{name}.csv" for name in dataframes}
    for name in dataframes:
        if name not in dataframe_files:
            raise KeyError(f"Missing filename mapping for dataframe key: {name}")

    row_counts: dict[str, int] = {}
    for name, file_name in dataframe_files.items():
        df = dataframes.get(name)
        if df is None:
            df = pd.DataFrame()
        df.to_csv(bundle / file_name, index=False)
        row_counts[name] = int(len(df))

    metadata = {
        "schema_version": int(schema_version),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": experiment or {},
        "run_metadata": run_metadata or {},
        "files": {"metadata": "metadata.json", **dataframe_files},
        "row_counts": row_counts,
    }
    with open(bundle / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metadata), f, indent=2, sort_keys=True)

    return bundle


def load_dataframe_bundle(
    bundle_dir: str | Path,
    *,
    expected_keys: tuple[str, ...] | list[str] | None = None,
    empty_on_missing: bool = True,
) -> dict[str, Any]:
    """Load a dataframe bundle created by :func:`save_dataframe_bundle`."""
    bundle = Path(bundle_dir)
    with open(bundle / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    files = dict(metadata.get("files", {}))
    files.pop("metadata", None)

    keys = list(expected_keys) if expected_keys is not None else list(files.keys())
    dataframes: dict[str, pd.DataFrame] = {}
    for key in keys:
        file_name = files.get(key, f"{key}.csv")
        path = bundle / file_name
        if not path.exists():
            if empty_on_missing:
                dataframes[key] = pd.DataFrame()
                continue
            raise FileNotFoundError(f"Missing dataframe file for key '{key}': {path}")
        try:
            dataframes[key] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            dataframes[key] = pd.DataFrame()

    return {
        "bundle_path": bundle,
        "bundle_metadata": metadata,
        "dataframes": dataframes,
        "metadata": metadata.get("run_metadata", {}),
    }


def download_results_bundle_from_wandb(
    *,
    artifact_name: str,
    entity: str,
    project: str,
    download_root: str | Path = ".",
    required_files: list[str] | tuple[str, ...] | None = None,
    artifact_alias: str = "latest",
) -> Path | None:
    """Download a bundle artifact and check for required files.

    Returns ``None`` when the artifact is unavailable or cannot be read, so callers
    can treat this as a cache miss and recompute results.
    """

    reference = f"{entity}/{project}/{artifact_name}:{artifact_alias}"
    root = Path(download_root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_dir = Path(wandb.Api().artifact(reference).download(root=str(root)))
    except Exception as err:
        message = str(err).lower()
        cache_miss_markers = (
            "not found",
            "does not exist",
            "unable to fetch files for artifact",
        )
        if any(marker in message for marker in cache_miss_markers):
            return None
        raise

    missing_files = [
        name for name in required_files if not (downloaded_dir / name).exists()
    ]
    if missing_files:
        # Treat incomplete artifacts as cache misses and rerun the model.
        return None
    return downloaded_dir


def upload_results_bundle_to_wandb(
    bundle_dir: str | Path,
    *,
    artifact_name: str,
    entity: str | None = None,
    project: str | None = None,
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    artifact_alias: str = "latest",
    run_mode: str = "online",
    job_type: str = "seq_len_bundle_upload",
    artifact_type: str = "dataset",
) -> str:
    """Upload a bundle directory to W&B and return the artifact reference."""
    bundle = Path(bundle_dir)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle}")
    resolved_entity = entity or "icl_arch"
    resolved_project = project or "seq_len_exp"

    with wandb.init(
        project=resolved_project,
        entity=resolved_entity,
        mode=run_mode,
        name=run_name,
        job_type=job_type,
    ) as run:
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=_to_jsonable(metadata or {}),
        )
        artifact.add_dir(str(bundle))
        run.log_artifact(artifact, aliases=[artifact_alias])

    return f"{resolved_entity}/{resolved_project}/{artifact_name}:{artifact_alias}"


def load_results_bundle(
    bundle_dir: str | Path,
    *,
    files: dict[str, str] | None = None,
    load_raw_torch: bool = False,
) -> dict[str, Any]:
    """Load a canonical results bundle and return dataframe + nested-table views."""
    bundle = Path(bundle_dir)

    with open(bundle / files["metadata"], "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(bundle / files["oom"], "r", encoding="utf-8") as f:
        oom_errors = json.load(f)

    metric_df = pd.read_csv(bundle / files["metric"])
    timing_df = pd.read_csv(bundle / files["timing"])
    memory_df = pd.read_csv(bundle / files["memory"])

    out = {
        "bundle_path": bundle,
        "bundle_metadata": metadata,
        "metric_df": metric_df,
        "timing_df": timing_df,
        "memory_df": memory_df,
        "oom_errors": oom_errors,
        "metric_table": long_df_to_nested_metric_table(metric_df),
        "timing_table": long_df_to_nested_metric_table(timing_df),
        "memory_table": long_df_to_nested_metric_table(memory_df),
        "metadata": metadata.get("run_metadata", {}),
    }

    raw = bundle / files["raw"]
    if load_raw_torch and raw.exists():
        out["raw_results"] = torch.load(raw, map_location="cpu")

    return out
