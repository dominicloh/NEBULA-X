"""Anomaly-detection model scaffolding for bogie temperature analysis.

The initial approach is Isolation Forest, but the project should compare against a
simple statistical baseline when time allows. The model output and maintenance
severity are conceptually related but not identical.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover
    IsolationForest = None


def train_isolation_forest(feature_frame, config):
    """Train an Isolation Forest using engineered features.

    This should only be used after the actual feature columns have been defined in
    the config. The function intentionally avoids placeholder training data.
    """
    if not isinstance(config, dict) or not config:
        raise ValueError(
            "Model configuration is missing. Supply the feature-selection settings "
            "before training an anomaly model."
        )

    if IsolationForest is None:
        raise ImportError("scikit-learn is required to train the Isolation Forest model.")

    feature_columns = config.get("feature_columns")
    if not feature_columns:
        raise ValueError(
            "No feature columns were supplied in the model config. Provide the "
            "selected engineered features before training."
        )

    missing = [column for column in feature_columns if column not in feature_frame.columns]
    if missing:
        raise ValueError("Model feature columns missing: " + ", ".join(missing))

    model = IsolationForest(
        contamination=config.get("contamination", 0.05),
        random_state=config.get("random_state", 42),
        n_estimators=config.get("n_estimators", 200),
    )
    model.fit(feature_frame[feature_columns].astype(float))
    return model


def predict_anomalies(model, feature_frame):
    """Return model predictions and anomaly scores for the provided feature frame."""
    if model is None:
        raise ValueError("Model instance is required before generating predictions.")

    if "__feature_columns__" in feature_frame.columns:
        feature_columns = list(feature_frame["__feature_columns__"].dropna().unique())
    else:
        feature_columns = [col for col in feature_frame.columns if col not in ["__source_file__", "timestamp"]]

    if not feature_columns:
        raise ValueError("No feature columns are available for anomaly prediction.")

    score = model.decision_function(feature_frame[feature_columns].astype(float))
    prediction = model.predict(feature_frame[feature_columns].astype(float))

    result = feature_frame.copy()
    result["anomaly_score"] = score
    result["anomaly_label"] = prediction
    result["is_anomaly"] = result["anomaly_label"] == -1
    return result


def calculate_severity(results, config):
    """Calculate a severity prioritisation score for abnormal observations.

    Severity is not an official railway safety threshold. It is a prioritisation aid
    for maintenance planning and engineering investigation.

    The severity may combine:
    - anomaly score magnitude
    - temperature deviation
    - rate of change
    - persistence over multiple abnormal readings
    """
    if not isinstance(config, dict) or not config:
        raise ValueError(
            "Severity configuration is missing. Provide the required thresholds and fields."
        )

    severity_frame = results.copy()
    anomaly_score = severity_frame.get("anomaly_score", 0.0)
    deviation = severity_frame.get("deviation_from_baseline", 0.0)
    rate_of_change = severity_frame.get("rate_of_change", 0.0)
    persistence = severity_frame.get("persistence_score", 0.0)

    severity_values = (
        (-anomaly_score) * config.get("score_weight", 1.0)
        + abs(deviation) * config.get("deviation_weight", 1.0)
        + abs(rate_of_change) * config.get("rate_change_weight", 1.0)
        + persistence * config.get("persistence_weight", 1.0)
    )

    severity_frame["severity_score"] = severity_values

    thresholds = config.get("severity_thresholds", {"critical": 10, "warning": 5, "minor": 2})
    severity_frame["severity_category"] = "normal"
    severity_frame.loc[severity_frame["severity_score"] >= thresholds.get("critical", 10), "severity_category"] = "critical"
    severity_frame.loc[
        (severity_frame["severity_score"] >= thresholds.get("warning", 5))
        & (severity_frame["severity_score"] < thresholds.get("critical", 10)),
        "severity_category",
    ] = "warning"
    severity_frame.loc[
        (severity_frame["severity_score"] >= thresholds.get("minor", 2))
        & (severity_frame["severity_score"] < thresholds.get("warning", 5)),
        "severity_category",
    ] = "minor"

    return severity_frame
