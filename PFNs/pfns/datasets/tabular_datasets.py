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
    suite_id: int = 457, # tabarena - v0.1 suite
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
        try:
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            
            def _encode_if_category(col: pd.Series, is_categorical: bool) -> pd.Series:
                # similar to https://github.com/openml/openml-python/blob/449f2cb9274a6a4d566748c6f1fdc4b3899482ba/openml/datasets/dataset.py#L654
                if isinstance(col.dtype, pd.CategoricalDtype) or is_categorical:
                    cat = col.astype("category")
                    codes = cat.cat.codes.astype(np.float32)
                    codes[codes == -1] = np.nan
                    return codes

                numeric = pd.to_numeric(col, errors="coerce")
                non_missing = col.notna()
                # If all non-missing values can be converted to numeric, use the numeric version. Otherwise, treat as categorical.
                if numeric[non_missing].notna().all():
                    return numeric.astype(np.float32)

                cat = col.astype("category")
                codes = cat.cat.codes.astype(np.float32)
                codes[codes == -1] = np.nan
                return codes
        
            columns = {
                column_name: _encode_if_category(X.loc[:, column_name], categorical_indicator[i])
                for i, column_name in enumerate(X.columns)
            }
            X = pd.DataFrame(columns).to_numpy(dtype=np.float32)
            y = _encode_if_category(y, True).to_numpy(dtype=np.int64)
        except Exception as e:
            print(f"Failed to load dataset for did={did} from OpenML. Exception: {e}")
            return None, None, None, None

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            print("Not a NP Array, skipping")
            return None, None, None, None
        
        cat_idx = _cat_idx_from_indicator(categorical_indicator, n_features=X.shape[1])

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
    num_feats=20,
    min_samples=100,
    max_samples=1000,
    max_num_classes=10,
    return_capped=True,
    random_state: int = 42,
    verbose: bool = True,
):
    if min_samples > max_samples:
        raise ValueError("min_samples must be <= max_samples")
    if int(num_feats) <= 0:
        raise ValueError("num_feats must be > 0")

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
        dataset_id = int(entry.did)

        if verbose:
            print("Loading", entry["name"], dataset_id, "..")

        if entry["NumberOfClasses"] == 0.0:
            raise RuntimeError("Regression not supported")
        X, y, categorical_feats, attribute_names = get_openml_classification(
            dataset_id,
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
            sample_rng = np.random.default_rng(
                np.random.SeedSequence([int(random_state), dataset_id, 0])
            )
            sample_idx = np.sort(
                sample_rng.choice(int(X.shape[0]), size=int(max_samples), replace=False)
            )
            sample_idx_t = torch.as_tensor(sample_idx, dtype=torch.long)
            X = X[sample_idx_t]
            y = y[sample_idx_t]
            modifications["samples_capped"] = True

        if X.shape[0] < min_samples:
            print("Too few samples left")
            continue

        if X.shape[1] > num_feats:
            if not return_capped:
                print("Too many features")
                continue
            feature_rng = np.random.default_rng(
                np.random.SeedSequence([int(random_state), dataset_id, 1])
            )
            selected_cols = np.sort(
                feature_rng.choice(int(X.shape[1]), size=int(num_feats), replace=False)
            )
            selected_cols_list = selected_cols.tolist()
            selected_lookup = {
                old_idx: new_idx for new_idx, old_idx in enumerate(selected_cols_list)
            }
            X = X[:, selected_cols_list]
            categorical_feats = [
                selected_lookup[idx] for idx in categorical_feats if idx in selected_lookup
            ]
            attribute_names = [attribute_names[idx] for idx in selected_cols_list]
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
