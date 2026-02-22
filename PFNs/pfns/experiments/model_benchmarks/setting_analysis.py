from __future__ import annotations

from typing import Any, Iterable
import re

import pandas as pd

CANONICAL_SETTING_PATTERN = re.compile(
    r"^(?P<model_type>.+)_(?P<setting>Comb_MT|Comb_ST|Int_MT|Int_ST)$"
)

# If True, higher is better; if False, lower is better.
SETTING_METRIC_DIRECTION: dict[str, bool] = {
    "accuracy": True,
    "roc_auc": True,
    "log_loss": False,
    "ece": False,
}

SETTING_METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
    "log_loss": "CE",
    "ece": "ECE",
}

def parse_setting_model_name(model_name: str) -> dict[str, str] | None:
    match = CANONICAL_SETTING_PATTERN.match(str(model_name))
    if match is None:
        return None
    return {
        "model": str(model_name),
        "model_type": match.group("model_type"),
        "setting": match.group("setting"),
    }


def extract_setting_model_meta(model_names: Iterable[str]) -> pd.DataFrame:
    rows = [
        parsed
        for model_name in model_names
        if (parsed := parse_setting_model_name(model_name)) is not None
    ]
    if not rows:
        raise RuntimeError("No canonical Comb/Int setting models were found.")
    return pd.DataFrame(rows)


def build_setting_presence(
    *, model_type_setting_df: pd.DataFrame, target_settings: Iterable[str]
) -> pd.DataFrame:
    target_settings = list(dict.fromkeys(target_settings))
    return (
        model_type_setting_df[["model_type", "setting"]]
        .drop_duplicates()
        .assign(present=True)
        .pivot_table(
            index="model_type",
            columns="setting",
            values="present",
            aggfunc="max",
            fill_value=False,
            observed=True,
        )
        .reindex(columns=target_settings, fill_value=False)
        .sort_index()
    )


def ci95_halfwidth(values: pd.Series) -> float:
    n = int(values.shape[0])
    if n <= 1:
        return 0.0
    std = float(values.std(ddof=1))
    sem = float(std / (n ** 0.5))
    return float(1.96 * sem)


def summarize_diff(diff: pd.Series) -> dict[str, float | int | bool] | None:
    n = int(diff.shape[0])
    if n == 0:
        return None

    mean_gain = float(diff.mean())
    std_gain = float(diff.std(ddof=1)) if n > 1 else 0.0 # sample standard deviation of gains
    sem_gain = float(std_gain / (n ** 0.5)) if n > 1 else 0.0 # standard errror of mean gain
    ci95 = ci95_halfwidth(diff)
    ci95_low = mean_gain - ci95
    ci95_high = mean_gain + ci95

    return {
        "mean_gain": mean_gain,
        "std_gain": std_gain,
        "sem_gain": sem_gain,
        "ci95": ci95,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "n_pairs": n,
        "ci95_excludes_zero": (ci95_low > 0.0) or (ci95_high < 0.0),
    }


def get_setting_preprocess(
    *, results_df: pd.DataFrame, target_settings: Iterable[str]
) -> dict[str, Any]:
    if results_df is None or results_df.empty:
        raise RuntimeError("No results dataframe available for setting preprocessing.")
    if "model" not in results_df.columns:
        raise RuntimeError("Expected a 'model' column in results_df for setting preprocessing.")

    target_settings = tuple(dict.fromkeys(target_settings))
    
    model_meta = extract_setting_model_meta(
        sorted(results_df["model"].astype(str).unique())
    )
    setting_results = results_df.merge(model_meta, on="model", how="inner")
    setting_results = setting_results[
        setting_results["setting"].isin(target_settings)
    ].copy()

    presence = build_setting_presence(
        model_type_setting_df=setting_results[["model_type", "setting"]],
        target_settings=target_settings,
    )

    eligible_model_types = presence.index[presence.all(axis=1)].tolist()
    if not eligible_model_types:
        raise RuntimeError(
            "No model type has all requested settings in current results for paired setting analysis."
        )

    filtered_results = setting_results[
        setting_results["model_type"].isin(eligible_model_types)
    ].copy()

    payload = {
        "target_settings": list(target_settings),
        "model_meta": model_meta,
        "setting_results": setting_results,
        "filtered_results": filtered_results,
        "presence": presence,
        "eligible_model_types": eligible_model_types,
    }
    return payload
