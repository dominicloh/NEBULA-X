"""Door candidate classifier models.

Mirrors src/rail_corrugation/model.py's pattern: a plain function returning
a dict of ready-to-evaluate scikit-learn estimators, all class-weighted
(Door's 80 Normal / 30 Abnormal resistance split is imbalanced, though less
severely than Rail Corrugation's), all using the same fixed random seed for
reproducibility.

This module only CONSTRUCTS models -- it never trains one and never claims
one is "best" before real evaluation (see train.py for that).
"""

from __future__ import annotations

from typing import Dict

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.door.config import RANDOM_STATE


def build_candidate_models() -> Dict[str, object]:
    """Return the Door baseline model comparison set, keyed by name.

    - DummyMostFrequent: the floor. A real model that can't beat this on
      macro F1 has a bug somewhere.
    - LogisticRegression: simple linear baseline, wrapped with
      StandardScaler (required -- logistic regression is scale-sensitive,
      and Door's features mix very different units: milliseconds,
      millivolts, raw counts).
    - RandomForest: a tree ensemble that doesn't need feature scaling.
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
    }


def train_door_model(estimator, feature_frame, labels):
    """Fit one candidate estimator on the full given feature matrix/labels.

    A thin, explicit wrapper (mirrors src/rail_corrugation/model.py's
    train_rail_model) -- kept for reuse once a baseline is selected, not
    called to fabricate results anywhere in this scaffold.
    """
    if feature_frame is None or labels is None:
        raise ValueError("Door feature table and labels are required before training.")
    estimator.fit(feature_frame, labels)
    return estimator


# Backward-compatible aliases -- earlier scaffold code imported these names.
def build_door_baseline_model(feature_frame, labels, model_config: dict | None = None):
    """Deprecated name -- kept for backward compatibility. Builds and fits
    the LogisticRegression candidate specifically. Prefer
    `build_candidate_models()` + `train_door_model()` in new code.
    """
    if feature_frame is None or labels is None:
        raise ValueError("Door feature table and labels are required before model training.")
    estimator = build_candidate_models()["LogisticRegression"]
    return train_door_model(estimator, feature_frame, labels)


def evaluate_door_model(predictions, truth, eval_config: dict | None = None):
    """Deprecated placeholder -- the real evaluation lives in train.py,
    which reports macro F1 and per-class precision/recall/F1 using
    cross-validation. This function is kept only so old imports don't
    break; it does not compute the official IoU-weighted F1 (that needs
    real predicted segments, produced by predict.py, not just labels).
    """
    raise NotImplementedError(
        "evaluate_door_model() is a compatibility placeholder. Use train.py for classification "
        "metrics (macro F1, precision/recall/F1), and remember the OFFICIAL Door metric is "
        "IoU-weighted F1 over predicted segments, which additionally depends on segmentation."
    )


def save_door_pipeline(model_path, pipeline):
    """Persist a trained Door pipeline to disk (joblib), mirroring
    src/rail_corrugation/train.py's train_final_model save step.
    """
    import joblib
    from pathlib import Path

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def load_door_pipeline(model_path):
    """Load a trained Door pipeline from disk. Raises a clear error if it
    doesn't exist yet -- never returns a fake/default model.
    """
    import joblib
    from pathlib import Path

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Door model artifact not found at {path}. Train and save one first with: "
            "python src/door/train.py --finalize"
        )
    return joblib.load(path)


__all__ = [
    "build_candidate_models",
    "train_door_model",
    "build_door_baseline_model",
    "evaluate_door_model",
    "save_door_pipeline",
    "load_door_pipeline",
]
