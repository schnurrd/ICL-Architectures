from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_REAL_WORLD_PRESET = "openml"

_REAL_WORLD_PRESET_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "openml": {
        "experiment": {
            "name": "real_world_openml_comparison",
            "benchmark": "opencc",
            "max_samples": 1_000,
            "max_features": 20,
            "max_classes": 10,
            "n_splits": 5,
            "batch_size_inference": 32,
            "n_ensemble_configurations": 10,
            "preprocess_transforms": ["none", "power"],
            "sample_order_permutation": True,
            "fla_cache_chunk_size": None,
        },
        "wandb": {
            "artifact_project": "real_world_eval_artifacts",
        },
    },
    "tabarena": {
        "experiment": {
            "name": "real_world_tabarena_comparison",
            "benchmark": "tabarena_full",
            "max_samples": 1_000_000,
            "max_features": 20,
            "max_classes": 10,
            "n_splits": 5,
            "batch_size_inference": 10,
            "n_ensemble_configurations": 10,
            "preprocess_transforms": ["none", "power_all"],
            "sample_order_permutation": True,
            "fla_cache_chunk_size": None,
        },
        "wandb": {
            "artifact_project": "real_world_tabarena_full_eval_artifacts",
        },
    },
}

REAL_WORLD_PRESET_ALIASES = {
    "openml": "openml",
    "real_world_openml_comparison": "openml",
    "tabarena": "tabarena",
    "real_world_tabarena_comparison": "tabarena",
}
REAL_WORLD_PRESET_CHOICES = tuple(sorted(_REAL_WORLD_PRESET_CONFIGS))


def normalize_real_world_preset_name(preset: str) -> str:
    normalized = REAL_WORLD_PRESET_ALIASES.get(str(preset).strip())
    if normalized is None:
        valid_names = ", ".join(sorted(REAL_WORLD_PRESET_ALIASES))
        raise ValueError(
            f"Unknown real-world preset '{preset}'. Expected one of: {valid_names}."
        )
    return normalized


def get_real_world_preset(preset: str) -> dict[str, dict[str, Any]]:
    preset_name = normalize_real_world_preset_name(preset)
    preset_config = _REAL_WORLD_PRESET_CONFIGS[preset_name]
    return {
        "name": preset_name,
        "experiment": deepcopy(preset_config["experiment"]),
        "wandb": deepcopy(preset_config["wandb"]),
    }
