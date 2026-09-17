"""Door binary-classification model scaffold.

Competition scoring uses IoU-weighted F1 over predicted and true temporal segments,
so the final evaluation must consider segmentation quality and class labels together.
"""

from __future__ import annotations

from pathlib import Path


def build_door_baseline_model(feature_frame, labels, config: dict | None = None):
    """Create a baseline Door classifier interface with a clear fail-fast guard."""
    if config is None:
        raise RuntimeError(
            "Door model configuration is missing. Read the Door Info Kit before implementing the baseline classifier."
        )

    if feature_frame is None or labels is None:
        raise ValueError("Door feature table and labels are required before model training.")

    raise RuntimeError(
        "Door baseline model is not ready until the Info Kit confirms the signal schema, labels, and evaluation setup."
    )


def evaluate_door_model(_predictions, _truth, config: dict | None = None):
    """Evaluate Door predictions using temporal IoU-weighted F1 as the official metric."""
    if config is None:
        raise RuntimeError(
            "Door evaluation configuration is missing. Final scoring must use IoU-weighted F1 across predicted and true temporal segments."
        )
    raise RuntimeError("Door model evaluation is not ready until segmentation targets and labels are confirmed.")


def save_door_pipeline(model_path, _pipeline):
    """Persist a trained Door preprocessing + model pipeline."""
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raise RuntimeError("Door model pipeline is not yet trained. Save only after the Info Kit validation and training steps are complete.")


def load_door_pipeline(model_path):
    """Load a trained Door pipeline from disk."""
    raise FileNotFoundError(f"Door model pipeline not found at {model_path}.")


__all__ = [
    "build_door_baseline_model",
    "evaluate_door_model",
    "save_door_pipeline",
    "load_door_pipeline",
]
