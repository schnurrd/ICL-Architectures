"""
Simple Evaluation Framework for PFN Models
"""

from .baselines import RandomForestBaseline, XGBoostBaseline, get_baselines
from .evaluate import (
    evaluate_model,
    evaluate_on_openml,
    compare_models
)

__all__ = [
    "RandomForestBaseline",
    "XGBoostBaseline",
    "get_baselines",
    "evaluate_model",
    "evaluate_on_openml",
    "compare_models",
]
