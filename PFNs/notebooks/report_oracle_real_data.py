#!/usr/bin/env python3
"""Report the hidden-state oracle comparison on the largest TabArena datasets.

Reads the real-world bundles written by `run_real_world_experiments.py` -- the
oracle and DeltaNet arms from the `tabarena_largest_5` run, the non-causal
reference arms from whatever compatible TabArena bundle is already on disk --
and prints the per-dataset table plus the paired oracle-minus-baseline test.

Usage (from the repo root; the scripts live under PFNs/):
    python PFNs/notebooks/report_oracle_real_data.py
    python PFNs/notebooks/report_oracle_real_data.py --metric accuracy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pfns.experiments.model_benchmarks.real_world_benchmarks import (
    get_real_world_benchmark_dataset_ids,
)
from pfns.experiments.model_benchmarks.real_world_oracle_report import (
    DEFAULT_ARMS,
    METRICS,
    build_split_table,
    find_arm_bundles,
    format_report,
)
from pfns.utils import build_exp_outputs_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=build_exp_outputs_path(__file__, "real_world_eval"),
        help="Root holding the real_world/ bundle directories.",
    )
    parser.add_argument(
        "--metric",
        default="roc_auc",
        choices=list(METRICS),
        help="Metric to report. The paper's headline metric is roc_auc.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_ARMS),
        help="Registry model names to include, in report order.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to also write the report to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = find_arm_bundles(args.models, output_root=args.output_root)

    missing = [name for name in args.models if name not in arms]
    if missing:
        print(f"No bundle found for: {', '.join(missing)}", file=sys.stderr)
    if not arms:
        raise SystemExit(f"No real-world bundles under {args.output_root}.")

    # Report order follows --models, not bundle discovery order.
    ordered = {name: arms[name] for name in args.models if name in arms}
    split_table = build_split_table(ordered)
    if split_table.empty:
        raise SystemExit("Arms share no datasets; nothing to compare.")

    report = format_report(
        arms=ordered,
        split_table=split_table,
        labels=DEFAULT_ARMS,
        metric=args.metric,
    )
    print(report)
    print()
    print(f"TabArena largest-5 dataset ids: {get_real_world_benchmark_dataset_ids('tabarena_largest_5')}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(report + "\n")
        print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
