from __future__ import annotations

import hashlib
import json
from typing import Any

from pfns.experiments.model_benchmarks.model_registry import functional_model_config


def experiment_payload_hash(
    *,
    experiment_payload: dict[str, Any],
    hash_length: int = 16,
) -> str:
    """Build a deterministic short hash for an experiment payload."""
    return hashlib.sha256(
        json.dumps(experiment_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:hash_length]


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
    filtered_model_config = functional_model_config(model_config)
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
