"""Assemble the hidden-state oracle comparison on real (TabArena) datasets.

The synthetic version of this experiment lives in the sequence-length sweep: a
frozen DeltaNet is compared against the same checkpoint whose per-layer recurrent
states have been optimised directly on the in-context set. This module does the
same comparison on real data, and pulls the non-causal reference models from
whatever real-world bundles are already on disk instead of rerunning them.

Bundles are matched on the experiment keys that decide whether two runs are
comparable. ``benchmark`` is deliberately not one of them: it only selects
*which* datasets a run covered, and the tables here restrict to the datasets
every arm has in common anyway. Everything else -- fold count, ensemble size,
preprocessing, sample caps -- must agree, or rows from different bundles would
not be measuring the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from pfns.experiments.model_benchmarks.io import (
    REAL_WORLD_REQUIRED_FILES,
    load_dataframe_bundle,
)
from pfns.experiments.model_benchmarks.paired_stats import (
    holm_correct,
    paired_test,
)

#: Experiment settings two bundles must share before their rows are comparable.
COMPARABILITY_KEYS: tuple[str, ...] = (
    "max_samples",
    "max_features",
    "max_classes",
    "n_splits",
    "batch_size_inference",
    "n_ensemble_configurations",
    "preprocess_transforms",
    "sample_order_permutation",
)

#: The default arms, in report order. Keys are registry model names.
DEFAULT_ARMS: dict[str, str] = {
    "equal_params:Transformer_Comb_ST": "Softmax Non-Causal",
    "Linear_Attention_Non_Causal": "Linear Non-Causal",
    "oracles:DeltaNet_Comb_ST": "DeltaNet (causal)",
    "oracles:Oracle_Hidden_State_DeltaNet_Comb_ST_Matched": "DeltaNet + Hidden-State Oracle",
}

#: The comparison the experiment exists to make.
ORACLE_ARM = "oracles:Oracle_Hidden_State_DeltaNet_Comb_ST_Matched"
BASELINE_ARM = "oracles:DeltaNet_Comb_ST"

METRICS: tuple[str, ...] = ("roc_auc", "accuracy", "log_loss")


@dataclass(frozen=True)
class ArmBundle:
    """One model's per-split results, with the bundle it came from."""

    model: str
    bundle_dir: Path
    results: pd.DataFrame
    experiment: dict[str, Any]


def _comparability_signature(experiment: Mapping[str, Any]) -> tuple:
    def freeze(value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    return tuple(freeze(experiment.get(key)) for key in COMPARABILITY_KEYS)


def find_arm_bundles(
    model_names: Iterable[str],
    *,
    output_root: str | Path,
    subdir: str = "real_world",
) -> dict[str, ArmBundle]:
    """Newest bundle on disk for each requested model.

    Bundles are keyed by the model column inside ``results.csv`` rather than by
    directory name, because a directory name is a sanitised model name and the
    sanitisation is not invertible (``oracles:X`` and ``oracles_X`` collide).
    """
    wanted = set(model_names)
    search_root = Path(output_root) / subdir
    found: dict[str, ArmBundle] = {}
    if not search_root.exists():
        return found

    for metadata_path in sorted(
        search_root.rglob("metadata.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        bundle_dir = metadata_path.parent
        if not all((bundle_dir / name).exists() for name in REAL_WORLD_REQUIRED_FILES):
            continue
        try:
            bundle = load_dataframe_bundle(bundle_dir, expected_keys=("results",))
        except Exception:
            continue

        results = bundle["dataframes"]["results"]
        if results is None or results.empty or "model" not in results.columns:
            continue

        # `bundle_metadata["experiment"]` is the nested experiment dict;
        # `metadata` is the run metadata, which repeats the same keys flat. Both
        # are read so a bundle missing either one still yields a signature
        # instead of silently comparing empty dicts to each other.
        experiment = {
            **dict(bundle["metadata"]),
            **dict(bundle["bundle_metadata"].get("experiment", {})),
        }
        for model in wanted - set(found):
            rows = results[results["model"].astype(str) == model]
            if rows.empty:
                continue
            found[model] = ArmBundle(
                model=model,
                bundle_dir=bundle_dir,
                results=rows.copy(),
                experiment=experiment,
            )
        if wanted <= set(found):
            break

    return found


def check_comparable(arms: Mapping[str, ArmBundle]) -> list[str]:
    """Return a warning per arm whose experiment settings differ from the rest."""
    if len(arms) < 2:
        return []
    signatures = {name: _comparability_signature(arm.experiment) for name, arm in arms.items()}
    counts = pd.Series(list(signatures.values())).value_counts()
    majority = counts.index[0]
    warnings: list[str] = []
    for name, signature in signatures.items():
        if signature == majority:
            continue
        differing = [
            f"{key}={arms[name].experiment.get(key)!r} (others {dict(zip(COMPARABILITY_KEYS, majority))[key]!r})"
            for key, value, expected in zip(COMPARABILITY_KEYS, signature, majority)
            if value != expected
        ]
        warnings.append(f"{name}: {', '.join(differing)}")
    return warnings


def build_split_table(
    arms: Mapping[str, ArmBundle],
    *,
    datasets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Long per-(model, dataset, split) table over the datasets all arms share."""
    frames = [arm.results.assign(model=name) for name, arm in arms.items()]
    if not frames:
        return pd.DataFrame(columns=["model", "dataset", "split", *METRICS])
    table = pd.concat(frames, ignore_index=True)

    shared = set.intersection(
        *(set(arm.results["dataset"].astype(str)) for arm in arms.values())
    )
    if datasets is not None:
        shared &= set(datasets)
    return table[table["dataset"].astype(str).isin(shared)].reset_index(drop=True)


def build_metric_table(split_table: pd.DataFrame, *, metric: str = "roc_auc") -> pd.DataFrame:
    """Per-dataset mean and std of ``metric`` across folds, one column per arm."""
    if split_table.empty:
        return pd.DataFrame()
    grouped = (
        split_table.groupby(["dataset", "model"])[metric].agg(["mean", "std", "count"]).reset_index()
    )
    rows = split_table.groupby("dataset")["dataset_num_rows"].max()
    table = grouped.pivot(index="dataset", columns="model", values="mean")
    table.insert(0, "rows", rows.reindex(table.index).astype("Int64"))
    return table.sort_values("rows", ascending=False)


def build_delta_table(
    split_table: pd.DataFrame,
    *,
    arm: str = ORACLE_ARM,
    baseline: str = BASELINE_ARM,
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """Per-dataset paired ``arm - baseline`` differences over the shared folds."""
    wide = split_table.pivot_table(
        index=["dataset", "split"], columns="model", values=metric
    )
    if arm not in wide.columns or baseline not in wide.columns:
        return pd.DataFrame()

    records = []
    for dataset, group in wide.groupby(level="dataset"):
        paired = group[[baseline, arm]].dropna()
        if paired.empty:
            continue
        deltas = (paired[arm] - paired[baseline]).to_numpy()
        records.append(
            {
                "dataset": dataset,
                "rows": int(
                    split_table.loc[split_table["dataset"] == dataset, "dataset_num_rows"].max()
                ),
                "n_folds": int(len(deltas)),
                "baseline": float(paired[baseline].mean()),
                "oracle": float(paired[arm].mean()),
                "delta": float(deltas.mean()),
                "delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan"),
            }
        )
    frame = pd.DataFrame(records)
    return frame.sort_values("rows", ascending=False).reset_index(drop=True) if not frame.empty else frame


def paired_stats(
    split_table: pd.DataFrame,
    *,
    arm: str = ORACLE_ARM,
    baseline: str = BASELINE_ARM,
    metric: str = "roc_auc",
) -> dict[str, Any]:
    """Paired test of ``arm - baseline`` at two pairing units.

    ``by_fold`` pairs every (dataset, fold); it has more pairs but folds within a
    dataset are not independent, so it overstates the evidence on its own.
    ``by_dataset`` pairs the five dataset means, which is the unit a claim about
    "real data" is actually made over.
    """
    wide = split_table.pivot_table(
        index=["dataset", "split"], columns="model", values=metric
    )
    if arm not in wide.columns or baseline not in wide.columns:
        return {}
    paired = wide[[baseline, arm]].dropna()
    if paired.empty:
        return {}

    by_dataset = paired.groupby(level="dataset").mean()
    return {
        "by_fold": paired_test(paired[arm].to_numpy(), paired[baseline].to_numpy()).as_record(),
        "by_dataset": paired_test(
            by_dataset[arm].to_numpy(), by_dataset[baseline].to_numpy()
        ).as_record(),
    }


def arm_vs_arm_stats(
    split_table: pd.DataFrame,
    *,
    reference: str,
    others: Iterable[str],
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """Holm-corrected paired tests of every ``other`` arm against ``reference``."""
    wide = split_table.pivot_table(
        index=["dataset", "split"], columns="model", values=metric
    )
    records = []
    for other in others:
        if other not in wide.columns or reference not in wide.columns:
            continue
        paired = wide[[reference, other]].dropna()
        if paired.empty:
            continue
        test = paired_test(paired[other].to_numpy(), paired[reference].to_numpy())
        records.append({"arm": other, "reference": reference, **test.as_record()})

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    adjusted, rejected = holm_correct(frame["p_value"].tolist())
    frame["p_holm"] = adjusted
    frame["significant"] = rejected
    return frame


def format_report(
    *,
    arms: Mapping[str, ArmBundle],
    split_table: pd.DataFrame,
    labels: Mapping[str, str],
    metric: str = "roc_auc",
) -> str:
    """Render the full text report."""
    lines: list[str] = []
    lines.append(f"Hidden-state oracle on real data -- metric: {metric}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Arms and source bundles")
    lines.append("-" * 78)
    for name, arm in arms.items():
        lines.append(f"  {labels.get(name, name):32s} {name}")
        lines.append(f"  {'':32s} {arm.bundle_dir}")
    warnings = check_comparable(arms)
    if warnings:
        lines.append("")
        lines.append("  WARNING: arms differ in experiment settings:")
        lines.extend(f"    {w}" for w in warnings)

    metric_table = build_metric_table(split_table, metric=metric)
    lines.append("")
    lines.append(f"Per-dataset {metric} (mean over folds)")
    lines.append("-" * 78)
    if metric_table.empty:
        lines.append("  (no shared datasets)")
    else:
        renamed = metric_table.rename(columns=dict(labels))
        lines.append(renamed.to_string(float_format=lambda v: f"{v:.4f}"))
        mean_row = renamed.drop(columns="rows").mean()
        lines.append("")
        lines.append("  mean over datasets:")
        for column, value in mean_row.items():
            lines.append(f"    {column:34s} {value:.4f}")

    delta_table = build_delta_table(split_table, metric=metric)
    if not delta_table.empty:
        lines.append("")
        lines.append(f"Oracle - DeltaNet, paired within fold ({metric})")
        lines.append("-" * 78)
        lines.append(delta_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        stats = paired_stats(split_table, metric=metric)
        for unit, record in stats.items():
            lines.append("")
            lines.append(
                f"  paired over {unit.replace('by_', '')}s: n={record['n_pairs']} "
                f"mean={record['mean_delta']:+.4f} "
                f"CI=[{record['ci_low']:+.4f}, {record['ci_high']:+.4f}] "
                f"p={record['p_value']:.4f} r={record['rank_biserial']:+.2f}"
            )

    others = [name for name in arms if name != BASELINE_ARM]
    comparison = arm_vs_arm_stats(
        split_table, reference=BASELINE_ARM, others=others, metric=metric
    )
    if not comparison.empty:
        lines.append("")
        lines.append(f"Every arm against the causal DeltaNet baseline ({metric})")
        lines.append("-" * 78)
        display = comparison.assign(arm=[labels.get(a, a) for a in comparison["arm"]])
        lines.append(
            display[
                ["arm", "n_pairs", "mean_delta", "ci_low", "ci_high", "p_value", "p_holm", "significant"]
            ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )

    return "\n".join(lines)
