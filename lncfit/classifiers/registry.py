"""Registry of pluggable classifier wrappers (teddy/model/registry.py pattern)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lncfit.classifiers.base import ClassifierModel

CLASSIFIER_REGISTRY: dict[str, type["ClassifierModel"]] = {}


def register_classifier(name: str):
    """Class decorator that registers a classifier wrapper under *name*."""

    def decorator(cls):
        if name in CLASSIFIER_REGISTRY:
            raise ValueError(f"Classifier {name!r} is already registered.")
        CLASSIFIER_REGISTRY[name] = cls
        return cls

    return decorator


def build_classifier(name: str, **params) -> "ClassifierModel":
    """Instantiate a registered classifier by name, forwarding hyperparameters."""
    if name not in CLASSIFIER_REGISTRY:
        available = ", ".join(sorted(CLASSIFIER_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available: {available}")
    return CLASSIFIER_REGISTRY[name](**params)


def available_classifiers() -> list[str]:
    """Sorted list of registered classifier names."""
    return sorted(CLASSIFIER_REGISTRY)
