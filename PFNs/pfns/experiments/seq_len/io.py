"""I/O helpers for sequence-length experiment result bundles.

Canonical bundle layout written by :func:`save_results_bundle`:

`<bundle_dir>/`
- `metadata.json`:
  - `schema_version` (int, from `SCHEMA_VERSION`)
  - `created_at_utc` (UTC ISO-8601 timestamp ending in `Z`)
  - `experiment` (JSON-serializable experiment config)
  - `run_metadata` (JSON-serializable run metadata copied from `results["metadata"]`)
  - `files` (filename mapping used for this bundle)
  - `row_counts` (`metric`/`timing`/`memory` CSV row counts)
- `metric.csv`   (long format columns: `model, metric, seqlen, rep, value`)
- `timing.csv`   (long format columns: `model, metric, seqlen, rep, value`)
- `memory.csv`   (long format columns: `model, metric, seqlen, rep, value`)
- `oom_errors.json` (JSON-serializable copy of `results["oom_errors"]`)
- `raw_results.pt` (optional; only written when `include_raw_torch=True`)

Notes:
- The three CSV files are generated from nested per-model/per-metric tables and
  currently use `seqlen` as the x-axis column name.
- `load_results_bundle` always requires JSON + CSV files above; `raw_results.pt`
  is optional and loaded only when `load_raw_torch=True`.
"""

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

BUNDLE_FILES: dict[str, str] = {
    "metadata": "metadata.json",
    "oom": "oom_errors.json",
    "metric": "metric.csv",
    "timing": "timing.csv",
    "memory": "memory.csv",
    "raw": "raw_results.pt",
}
WANDB_ENTITY = "icl_arch"
WANDB_PROJECT = "seq_len_exp"
WANDB_ARTIFACT_ALIAS = "latest"
WANDB_RUN_MODE = "online"


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
    experiment: dict[str, Any] | None = None,
    include_raw_torch: bool = True,
) -> Path:
    """Persist benchmark results using the canonical bundle layout documented above."""
    files = BUNDLE_FILES
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
        "schema_version": SCHEMA_VERSION,
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


def download_results_bundle_from_wandb(
    *,
    artifact_name: str,
    download_root: str | Path = ".",
) -> Path | None:
    """Download a bundle artifact from the default icl_arch/seq_len_exp target.

    Returns ``None`` when the artifact is unavailable or cannot be read, so callers
    can treat this as a cache miss and recompute results.
    """
    reference = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_name}:{WANDB_ARTIFACT_ALIAS}"
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

    required_files = [
        BUNDLE_FILES["metadata"],
        BUNDLE_FILES["oom"],
        BUNDLE_FILES["metric"],
        BUNDLE_FILES["timing"],
        BUNDLE_FILES["memory"],
    ]
    missing_files = [name for name in required_files if not (downloaded_dir / name).exists()]
    if missing_files:
        # Treat incomplete artifacts as cache misses and rerun the model.
        return None
    return downloaded_dir


def upload_results_bundle_to_wandb(
    bundle_dir: str | Path,
    *,
    artifact_name: str,
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Upload a bundle directory to the default icl_arch/seq_len_exp target."""
    bundle = Path(bundle_dir)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle}")

    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_RUN_MODE,
        name=run_name,
        job_type="seq_len_bundle_upload",
    ) as run:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            metadata=_to_jsonable(metadata or {}),
        )
        artifact.add_dir(str(bundle))
        run.log_artifact(artifact, aliases=[WANDB_ARTIFACT_ALIAS])

    return f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_name}:{WANDB_ARTIFACT_ALIAS}"


def load_results_bundle(bundle_dir: str | Path, *, load_raw_torch: bool = False) -> dict[str, Any]:
    """Load a canonical results bundle and return dataframe + nested-table views."""
    files = BUNDLE_FILES
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
