from __future__ import annotations

from pfns.priors.prior import AdhocPriorConfig
from pfns.priors.tabpfn_prior_adapter import TabPFNPriorConfig

TABPFN_PRIOR_DEFAULTS = {
    "prior_type": "mlp",
    "max_num_classes": 10,
    "max_num_features": 20,
    "flexible": True,
    "differentiable": True,
    "return_categorical_mask": True,
    "nan_handling": True,
}

ASSOCIATIVE_RECALL_SETTINGS = {
    "task_variant": "associative_recall",
    "min_single_eval_pos": 128,
    "fixed_num_features": 10,
    "fixed_num_classes": 10,
}


def build_prior_for_task(
    *,
    task_variant: str,
    prior_device: str,
    max_num_classes: int | None = None,
    max_num_features: int | None = None,
):
    resolved_max_num_classes = (
        int(TABPFN_PRIOR_DEFAULTS["max_num_classes"])
        if max_num_classes is None
        else int(max_num_classes)
    )
    resolved_max_num_features = (
        int(TABPFN_PRIOR_DEFAULTS["max_num_features"])
        if max_num_features is None
        else int(max_num_features)
    )

    if task_variant == "tabular_prior":
        tabpfn_kwargs = dict(TABPFN_PRIOR_DEFAULTS)
        tabpfn_kwargs["max_num_classes"] = resolved_max_num_classes
        tabpfn_kwargs["max_num_features"] = resolved_max_num_features
        tabpfn_kwargs["device"] = prior_device
        return TabPFNPriorConfig(**tabpfn_kwargs)

    if task_variant == ASSOCIATIVE_RECALL_SETTINGS["task_variant"]:
        return AdhocPriorConfig(
            prior_names="associative_recall",
            prior_kwargs={
                "max_num_classes": resolved_max_num_classes,
                "batch_device": prior_device,
                "fixed_num_features": ASSOCIATIVE_RECALL_SETTINGS[
                    "fixed_num_features"
                ],
                "fixed_num_classes": ASSOCIATIVE_RECALL_SETTINGS[
                    "fixed_num_classes"
                ],
            },
        )

    raise ValueError(
        f"Unknown task_variant {task_variant!r}. "
        "Expected one of: 'tabular_prior', 'associative_recall'."
    )
