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

def get_setting_preprocess(
    *, results_df: pd.DataFrame, target_settings: Iterable[str]
) -> dict[str, Any]:
    if results_df is None or results_df.empty:
        raise RuntimeError("No results dataframe available for setting preprocessing.")
    if "model" not in results_df.columns:
        raise RuntimeError("Expected a 'model' column in results_df for setting preprocessing.")

    target_settings = tuple(dict.fromkeys(target_settings))

    model_meta_rows: list[dict[str, str]] = []
    for model_name in sorted(results_df["model"].astype(str).unique()):
        match = CANONICAL_SETTING_PATTERN.match(model_name)
        if match is None:
            continue
        model_meta_rows.append(
            {
                "model": model_name,
                "model_type": match.group("model_type"),
                "setting": match.group("setting"),
            }
        )
    if not model_meta_rows:
        raise RuntimeError("No canonical Comb/Int setting models were found.")
    model_meta = pd.DataFrame(model_meta_rows)

    setting_results = results_df.merge(model_meta, on="model", how="inner")
    setting_results = setting_results[
        setting_results["setting"].isin(target_settings)
    ].copy()

    presence = (
        setting_results[["model_type", "setting"]]
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

    eligible_model_types = presence.index[presence.all(axis=1)].tolist()
    if not eligible_model_types:
        raise RuntimeError(
            "No model type has all requested settings in current results for paired setting analysis."
        )

    filtered_results = setting_results[
        setting_results["model_type"].isin(eligible_model_types)
    ].copy()

    return {
        "target_settings": list(target_settings),
        "model_meta": model_meta,
        "setting_results": setting_results,
        "filtered_results": filtered_results,
        "presence": presence,
        "eligible_model_types": eligible_model_types,
    }
