from __future__ import annotations

SCHEMA_VERSION = "1.0"

DEFAULT_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#B8860B",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#000000",
    "#999999",
    "#882255",
    "#44AA99",
    "#332288",
]
DEFAULT_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "X", "h", "8"]
DEFAULT_LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 1)),
    (0, (1, 1)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (5, 10)),
    "-",
    "--",
]
DEFAULT_BUCKET_BINS = [0, 500, 1000, 2000, 5000, 10000, float("inf")]
DEFAULT_BUCKET_LABELS = ["<=500", "501-1K", "1K-2K", "2K-5K", "5K-10K", "10K+"]

METRIC_NAMES = ("acc", "ce", "roc_auc")
TIMING_NAMES = ("forward_time_ms", "fit_time_ms", "predict_time_ms")
MEMORY_NAMES = ("peak_allocated_mb", "peak_reserved_mb", "context_size_mb")
