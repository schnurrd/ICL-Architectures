from __future__ import annotations

import hashlib
import json
from typing import Any

from pfns.experiments.model_benchmarks.model_registry import NON_FUNCTIONAL_CONFIG_KEYS

def model_identity_from_config(
    *,
    model_name: str,
    model_config: dict[str, Any],
) -> str:
    """Return a stable model identity, preferring checkpoint/run identifiers over display names."""
    if model_config.get("wandb_run_id"):
        return str(model_config["wandb_run_id"])
    if model_config.get("baseline_name"):
        return f"baseline:{model_config['baseline_name']}"
    return model_name


def single_model_hash(
    *,
    model_name: str,
    model_config: dict[str, Any],
    experiment_payload: dict[str, Any],
    hash_length: int = 16,
) -> str:
    """Build a deterministic short hash keyed by model identity + config + experiment payload."""
    filtered_model_config = {
        key: value
        for key, value in model_config.items()
        if key not in NON_FUNCTIONAL_CONFIG_KEYS
    }
    model_identity = model_identity_from_config(
        model_name=model_name,
        model_config=filtered_model_config,
    )
    payload = {
        "experiment": experiment_payload,
        "model_identity": model_identity,
        "model_config": filtered_model_config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:hash_length]
