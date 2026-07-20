"""Pluggable lncRNA-hit classifier layer.

Import a wrapper by name and run it through the shared train/eval pipeline:

    from lncfit.classifiers import build_classifier, available_classifiers
    model = build_classifier("xgboost", max_depth=4)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)

New architectures register themselves via the ``@register_classifier("name")``
decorator; importing the concrete-wrapper modules below triggers that
registration (same trigger-on-import pattern as teddy/model/__init__.py).
"""
from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import (
    CLASSIFIER_REGISTRY,
    available_classifiers,
    build_classifier,
    register_classifier,
)

# Import concrete wrappers to trigger their @register_classifier decorators.
from lncfit.classifiers.null import NullClassifier
from lncfit.classifiers.logreg import LogRegClassifier
from lncfit.classifiers.xgboost_clf import XGBoostClassifier
from lncfit.classifiers.randomforest import RandomForestClassifier
from lncfit.classifiers.knn import KNNClassifier

__all__ = [
    "ClassifierModel",
    "CLASSIFIER_REGISTRY",
    "available_classifiers",
    "build_classifier",
    "register_classifier",
    "NullClassifier",
    "LogRegClassifier",
    "XGBoostClassifier",
    "RandomForestClassifier",
    "KNNClassifier",
]
