"""Door inference interface.

This module validates and intentionally stops before generating fake results when the
segmentation or model is unavailable.
"""

from __future__ import annotations

from .preprocess import load_door_stream


def predict_door_segments(file_path, config: dict | None = None):
    """Load a continuous Door stream and predict one label for each detected segment."""
    stream = load_door_stream(file_path)
    if stream.empty:
        raise ValueError("Door stream is empty; no rows were loaded for prediction.")
    _ = stream
    if config is None:
        raise RuntimeError(
            "Door inference is not ready. No model or segmentation configuration is available yet."
        )

    raise RuntimeError(
        "Door model not ready: segmentation and classification are not configured until the Door Info Kit is reviewed and the pipeline is trained."
    )


__all__ = ["predict_door_segments"]
