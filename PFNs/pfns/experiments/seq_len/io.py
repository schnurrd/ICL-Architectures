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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

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
    return value


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
