import os
import warnings
import numpy as np
import openml
import pandas as pd
import torch
import logging
import hashlib
import json

from pathlib import Path
from typing import List

logging.getLogger("openml.datasets.functions").setLevel(logging.ERROR)

openml.config.set_root_cache_directory(
    os.environ.get("OPENML_CACHE_DIRECTORY", str(Path(__file__).parent / "openml"))
)

def _local_cache_dir() -> Path:
    d = Path(
        os.environ.get(
            "OPENML_LOCAL_CACHE_DIRECTORY",
            str(Path(__file__).parent / "openml_local_cache"),
        )
    )
    d.mkdir(parents=True, exist_ok=True)
    return d

def _dataset_cache_path(did: int) -> Path:
    return _local_cache_dir() / f"openml_{did}_raw.npz"

def _openml_list_cache_path(dids: List[int]) -> Path:
    h = hashlib.sha1(str(tuple(_normalize_dids(dids))).encode()).hexdigest()[:16]
    return _local_cache_dir() / f"openml_list_{h}.parquet"

def _suite_cache_path(prefix: str, suite_id: int, descriptor: str) -> Path:
    return _local_cache_dir() / f"{prefix}_{int(suite_id)}_{descriptor}.json"

def _normalize_dids(dids) -> list[int]:
    return sorted({int(did) for did in dids})

def _load_cached_dids(cache_file: Path) -> list[int]:
    with cache_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _normalize_dids(payload.get("dids", []))

def _save_cached_dids(cache_file: Path, dids: list[int]) -> None:
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump({"dids": _normalize_dids(dids)}, f, indent=2)

def load_openml_list_cached(dids: List[int]) -> pd.DataFrame:
    cache_file = _openml_list_cache_path(dids)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    df = openml.datasets.list_datasets(dids, output_format="dataframe")
    df.to_parquet(cache_file, index=False)
    return df


def get_benchmark_suite_dids(
    *,
    suite_id: int = 457,
    min_samples: int | None = None,
    max_samples: int | None = None,
    max_features: int | None = None,
    refresh_cache: bool = False,
) -> list[int]:
    """Resolve classification dataset IDs from an OpenML benchmark suite."""
    min_samples, min_part = (None, "all") if min_samples is None else (int(min_samples), str(int(min_samples)))
    max_samples, max_part = (None, "all") if max_samples is None else (int(max_samples), str(int(max_samples)))
    max_features, feat_part = (None, "all") if max_features is None else (int(max_features), str(int(max_features)))
    if (
        min_samples is not None
        and max_samples is not None
        and min_samples > max_samples
    ):
        raise ValueError("min_samples must be <= max_samples when both are set.")
    if max_features is not None and max_features <= 0:
        raise ValueError("max_features must be > 0 when set.")
    
    cache_file = _suite_cache_path(
        "benchmark_suite",
        suite_id,
        f"dids_{min_part}_{max_part}_{feat_part}",
    )
    if cache_file.exists() and not refresh_cache:
        return _load_cached_dids(cache_file)

    suite_cache_file = _suite_cache_path("openml_suite", suite_id, "dids")
    if suite_cache_file.exists() and not refresh_cache:
        dids = _load_cached_dids(suite_cache_file)
    else:
        suite = openml.study.get_suite(int(suite_id))
        dids = _normalize_dids(getattr(suite, "data", []))
        if not dids:
            raise RuntimeError(
                f"OpenML suite {suite_id} did not expose dataset IDs via `suite.data`."
            )
        _save_cached_dids(suite_cache_file, dids)

    suite_df = load_openml_list_cached(dids).copy()

    required_cols = ["did", "NumberOfClasses", "NumberOfInstances", "NumberOfFeatures"]
    missing = [col for col in required_cols if col not in suite_df.columns]
    if missing:
        cols = ", ".join(f"'{col}'" for col in missing)
        raise RuntimeError(f"Missing expected column(s) {cols} in OpenML suite metadata.")
    for col in required_cols:
        suite_df[col] = pd.to_numeric(suite_df[col], errors="coerce")

    filtered = suite_df[suite_df["NumberOfClasses"] > 0]
    if min_samples is not None:
        filtered = filtered[filtered["NumberOfInstances"] >= min_samples]
    if max_samples is not None:
        filtered = filtered[filtered["NumberOfInstances"] <= max_samples]
    if max_features is not None:
        filtered = filtered[filtered["NumberOfFeatures"] <= max_features]

    filtered_dids = _normalize_dids(filtered["did"].dropna().tolist())
    _save_cached_dids(cache_file, filtered_dids)
    return filtered_dids

def _cat_idx_from_indicator(categorical_indicator, n_features: int) -> List[int]:
    ci = np.asarray(categorical_indicator)
    if ci.dtype == bool:
        if ci.shape[0] != n_features:
            raise ValueError("categorical_indicator mask has wrong length")
        return np.where(ci)[0].astype(np.int64).tolist()
    return ci.astype(np.int64).tolist()

def _encode_labels(y: np.ndarray):
    _, y_mapped = np.unique(y, return_inverse=True)
    return y_mapped.astype(np.int64)

def get_openml_classification(did, seed=42):
    cache_file = _dataset_cache_path(int(did))

    X = y = cat_idx = attribute_names = None

    if cache_file.exists():
        try:
            data = np.load(cache_file, allow_pickle=False)
            X, y = data["X"], data["y"]
            cat_idx = data["cat_idx"].astype(np.int64).tolist()
            attribute_names = data["attribute_names"].astype(str).tolist()
        except Exception as e:
            print(f"Can't load cached dataset for did={did}, rebuilding cache. Exception: {e}")
            cache_file.unlink(missing_ok=True)
            X = y = cat_idx = attribute_names = None

    if X is None:
        dataset = openml.datasets.get_dataset(int(did))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_df, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe",
                target=dataset.default_target_attribute,
            )

        cat_idx = _cat_idx_from_indicator(
            categorical_indicator,
            n_features=X_df.shape[1],
        )
        for i, col in enumerate(X_df.columns):
            if i in cat_idx:
                codes = X_df[col].astype("category").cat.codes.to_numpy()
                col_values = codes.astype(np.float32, copy=False)
                col_values[codes == -1] = np.nan
                X_df[col] = col_values
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        X = X_df.to_numpy(dtype=np.float32, copy=False)
        y = np.asarray(y)
        # Ensure cached targets are never object/string arrays.
        if y.dtype.kind in {"O", "U", "S"}:
            y = _encode_labels(y)
        elif np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64, copy=False)
        else:
            y = y.astype(np.float64, copy=False)

        np.savez_compressed(
            cache_file,
            X=X,
            y=y,
            cat_idx=np.asarray(cat_idx, dtype=np.int64),
            attribute_names=np.asarray(list(attribute_names), dtype=str),
        )

    y_enc = _encode_labels(np.asarray(y))

    # shuffle dataset
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_enc))
    X = np.asarray(X)[order]
    y_enc = y_enc[order]
    
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y_enc, dtype=torch.int64)

    return X_t, y_t, cat_idx, attribute_names

def load_openml_list(
    dids,
    filter_for_nan=False,
    num_feats=100,
    min_samples=100,
    max_samples=400,
    max_num_classes=10,
    return_capped=True,
    verbose: bool = True,
):
    if min_samples > max_samples:
        raise ValueError("min_samples must be <= max_samples")

    datasets = []
    openml_list = load_openml_list_cached(dids)
    if verbose:
        print(f"Number of datasets: {len(openml_list)}")

    if filter_for_nan:
        openml_list = openml_list[
            openml_list["NumberOfInstancesWithMissingValues"] == 0
        ]
        if verbose:
            print(
                f"Number of datasets after Nan and feature number filtering: {len(openml_list)}"
            )

    for ds in openml_list.index:
        modifications = {
            "samples_capped": False,
            "classes_capped": False,
            "feats_capped": False,
        }
        entry = openml_list.loc[ds]

        if verbose:
            print("Loading", entry["name"], entry.did, "..")

        if entry["NumberOfClasses"] == 0.0:
            raise RuntimeError("Regression not supported")
        X, y, categorical_feats, attribute_names = get_openml_classification(
            int(entry.did),
        )
        if X is None:
            print("Warning: Could not load dataset, skipping.")
            continue

        num_classes = int(torch.unique(y).numel())
        if num_classes > max_num_classes:
            if not return_capped:
                print("Too many classes")
                continue
            y_np = y.cpu().numpy()
            vals, counts = np.unique(y_np, return_counts=True)
            top_vals = np.sort(vals[np.argsort(-counts)[:max_num_classes]])
            keep_vals_t = torch.as_tensor(top_vals, device=y.device, dtype=y.dtype)
            keep = torch.isin(y, keep_vals_t)
            X = X[keep]
            y = y[keep]
            modifications["classes_capped"] = True

        if X.shape[0] > max_samples:
            if not return_capped:
                print("Too many samples")
                continue
            X = X[0:max_samples, :]
            y = y[0:max_samples]
            modifications["samples_capped"] = True

        if X.shape[0] < min_samples:
            print("Too few samples left")
            continue

        if X.shape[1] > num_feats:
            if not return_capped:
                print("Too many features")
                continue
            X = X[:, 0:num_feats]
            categorical_feats = [c for c in categorical_feats if c < num_feats]
            attribute_names = attribute_names[:num_feats]
            modifications["feats_capped"] = True

        datasets.append(
            [
                entry["name"],
                X,
                y,
                categorical_feats,
                attribute_names,
                modifications,
            ]
        )

    return datasets, openml_list


# Classification
valid_dids_classification = [13, 59, 4, 15, 40710, 43, 1498]
test_dids_classification = [
    973,
    1596,
    40981,
    1468,
    40984,
    40975,
    41163,
    41147,
    1111,
    41164,
    1169,
    1486,
    41143,
    1461,
    41167,
    40668,
    41146,
    41169,
    41027,
    23517,
    41165,
    41161,
    41159,
    41138,
    1590,
    41166,
    1464,
    41168,
    41150,
    1489,
    41142,
    3,
    12,
    31,
    54,
    1067,
]
valid_large_classification = [
    943,
    23512,
    49,
    838,
    1131,
    767,
    1142,
    748,
    1112,
    1541,
    384,
    912,
    1503,
    796,
    20,
    30,
    903,
    4541,
    961,
    805,
    1000,
    4135,
    1442,
    816,
    1130,
    906,
    1511,
    184,
    181,
    137,
    1452,
    1481,
    949,
    449,
    50,
    913,
    1071,
    831,
    843,
    9,
    896,
    1532,
    311,
    39,
    451,
    463,
    382,
    778,
    474,
    737,
    1162,
    1538,
    820,
    188,
    452,
    1156,
    37,
    957,
    911,
    1508,
    1054,
    745,
    1220,
    763,
    900,
    25,
    387,
    38,
    757,
    1507,
    396,
    4153,
    806,
    779,
    746,
    1037,
    871,
    717,
    1480,
    1010,
    1016,
    981,
    1547,
    1002,
    1126,
    1459,
    846,
    837,
    1042,
    273,
    1524,
    375,
    1018,
    1531,
    1458,
    6332,
    1546,
    1129,
    679,
    389,
]

open_cc_dids = [
    11,
    14,
    15,
    16,
    18,
    22,
    23,
    29,
    31,
    37,
    50,
    54,
    188,
    458,
    469,
    1049,
    1050,
    1063,
    1068,
    1510,
    1494,
    1480,
    1462,
    1464,
    6332,
    23381,
    40966,
    40982,
    40994,
    40975,
]
# Filtered by N_samples < 2000, N feats < 100, N classes < 10

open_cc_valid_dids = [
    13,
    25,
    35,
    40,
    41,
    43,
    48,
    49,
    51,
    53,
    55,
    56,
    59,
    61,
    187,
    285,
    329,
    333,
    334,
    335,
    336,
    337,
    338,
    377,
    446,
    450,
    451,
    452,
    460,
    463,
    464,
    466,
    470,
    475,
    481,
    679,
    694,
    717,
    721,
    724,
    733,
    738,
    745,
    747,
    748,
    750,
    753,
    756,
    757,
    764,
    765,
    767,
    774,
    778,
    786,
    788,
    795,
    796,
    798,
    801,
    802,
    810,
    811,
    814,
    820,
    825,
    826,
    827,
    831,
    839,
    840,
    841,
    844,
    852,
    853,
    854,
    860,
    880,
    886,
    895,
    900,
    906,
    907,
    908,
    909,
    915,
    925,
    930,
    931,
    934,
    939,
    940,
    941,
    949,
    966,
    968,
    984,
    987,
    996,
    1048,
    1054,
    1071,
    1073,
    1100,
    1115,
    1412,
    1442,
    1443,
    1444,
    1446,
    1447,
    1448,
    1451,
    1453,
    1488,
    1490,
    1495,
    1498,
    1499,
    1506,
    1508,
    1511,
    1512,
    1520,
    1523,
    4153,
    23499,
    40496,
    40646,
    40663,
    40669,
    40680,
    40682,
    40686,
    40690,
    40693,
    40705,
    40706,
    40710,
    40711,
    40981,
    41430,
    41538,
    41919,
    41976,
    42172,
    42261,
    42544,
    42585,
    42638,
]

grinzstjan_categorical_regression = [
    44054,
    44055,
    44056,
    44057,
    44059,
    44061,
    44062,
    44063,
    44064,
    44065,
    44066,
    44068,
    44069,
]

grinzstjan_numerical_classification = [
    44089,
    44090,
    44091,
    44120,
    44121,
    44122,
    44123,
    44124,
    44125,
    44126,
    44127,
    44128,
    44129,
    44130,
    44131,
]

grinzstjan_categorical_classification = [
    44156,
    44157,
    44159,
    44160,
    44161,
    44162,
    44186,
]
