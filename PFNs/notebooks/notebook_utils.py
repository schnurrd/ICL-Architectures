from __future__ import annotations

import hashlib
import json
from typing import Any

def single_model_hash(
    *,
    model_name: str,
    model_config: dict[str, Any],
    experiment_payload: dict[str, Any],
    hash_length: int = 16,
) -> str:
    payload = {
        "experiment": experiment_payload,
        "model_name": model_name,
        "model_config": model_config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:hash_length]
