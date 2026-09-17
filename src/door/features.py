"""Door per-segment feature interface.

The feature list below is intentionally candidate-only and must be confirmed against
available signal types and the Door Info Kit before being treated as final.
"""

from __future__ import annotations

import pandas as pd


def extract_door_segment_features(_segments, _stream, config: dict | None = None):
    """Extract segment-level features once the Door schema and rules are confirmed."""
    if config is None:
        raise RuntimeError(
            "Door feature extraction is not configured yet. Confirm the signal columns and feature definitions from the Door Info Kit."
        )

    if not isinstance(_segments, pd.DataFrame):
        raise TypeError("Door segment features expect a pandas DataFrame of candidate cycles.")

    raise RuntimeError(
        "Door feature engineering is not ready until the Info Kit confirms the exact signal types and cycle-level features."
    )


def build_door_feature_table(segments, stream, config: dict | None = None):
    """Compatibility wrapper for segment feature generation."""
    return extract_door_segment_features(segments, stream, config=config)


__all__ = ["extract_door_segment_features", "build_door_feature_table"]
