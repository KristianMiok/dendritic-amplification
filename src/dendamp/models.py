"""SDM fitting — Random Forest and XGBoost with frozen hyperparameters.

Discipline carried over from m21: tune ONCE per entity on the clean benchmark via nested CV,
then FREEZE for every displacement run, so measured degradation reflects contamination alone
and not adaptive re-tuning.

Maxent is intentionally excluded (slow; elapid importance was not comparable in m21).

Contract:
    tune(algo, X, y) -> params            # run once per entity on benchmark
    fit(algo, X, y, params) -> model
    predict(model, X) -> suitability      # continuous score for Tier-3 comparison
"""

from __future__ import annotations


def tune(algo, X, y):
    raise NotImplementedError("Nested CV on benchmark; return frozen hyperparameters.")


def fit(algo, X, y, params):
    raise NotImplementedError("RandomForestClassifier / XGBClassifier with given params.")


def predict(model, X):
    raise NotImplementedError("Return continuous suitability (predict_proba positive class).")
