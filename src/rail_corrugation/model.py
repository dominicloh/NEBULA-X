"""Rail Corrugation classification model scaffold."""

from __future__ import annotations


def train_rail_model(feature_frame, labels, config: dict | None = None):
    """Train a file-level Rail Corrugation classifier after the schema and labels are confirmed."""
    if config is None:
        raise RuntimeError(
            "Rail Corrugation model configuration is missing. Read the Rail Corrugation Info Kit before training the classifier."
        )
    if feature_frame is None or labels is None:
        raise ValueError("Rail feature matrix and labels are required before training.")
    raise RuntimeError("Rail Corrugation model is not ready until the file-level schema and labels are confirmed.")


__all__ = ["train_rail_model"]
