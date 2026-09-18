"""Rail Corrugation candidate models for the Stage 2 baseline comparison.

Why class weights instead of resampling (e.g. SMOTE) for this first baseline:
the training set has only 14 Side I and 24 Side II examples. Resampling
before splitting would leak synthetic copies of the same minority examples
across train/validation folds; resampling correctly *inside* every fold adds
real complexity and a new dependency for a first pass. `class_weight="balanced"`
achieves the same goal (penalise the model more for missing a minority class)
with no extra dependency and no leakage risk, so it's what every weighted
model below uses. If resampling is tried later, it must be added inside each
training fold (e.g. via an imblearn Pipeline), never before cross-validation.
"""

from __future__ import annotations

from typing import Dict

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Fixed across every model and every script run so results are reproducible.
RANDOM_STATE = 42


def build_candidate_models() -> Dict[str, object]:
    """Return the Stage 2 baseline model comparison set, keyed by name.

    - DummyMostFrequent: the "can't possibly do worse than this" floor. If a
      real model can't beat it on macro F1, something is wrong.
    - LogisticRegression: a simple linear baseline. Wrapped in a Pipeline
      with StandardScaler, which is REQUIRED here (unlike the tree models)
      because logistic regression's regularisation is scale-sensitive -- the
      scaler is fit only on each training fold during cross-validation, so
      no information from the held-out fold leaks into it.
    - RandomForest / ExtraTrees: tree ensembles that don't need feature
      scaling and naturally handle the mix of very different feature scales
      (RMS in m/s^2 vs. frequency in Hz vs. ratios in [0, 1]) our features
      contain.

    Every model uses class_weight="balanced" (see module docstring) and the
    same fixed RANDOM_STATE, so re-running this script gives identical
    results.
    """
    return {
        "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_rail_model(estimator, feature_frame, labels):
    """Fit one candidate estimator on the full given feature matrix/labels.

    This is a thin, explicit wrapper (not a hidden pipeline) kept for reuse
    once a baseline is selected -- e.g. to fit the chosen model on all 272
    training files before it is ever used for real test-file predictions.
    Stage 2 does not call this to produce test predictions (see project
    instructions); it only evaluates candidates via cross-validation.
    """
    if feature_frame is None or labels is None:
        raise ValueError("Rail feature matrix and labels are required before training.")
    estimator.fit(feature_frame, labels)
    return estimator


__all__ = ["RANDOM_STATE", "build_candidate_models", "train_rail_model"]
