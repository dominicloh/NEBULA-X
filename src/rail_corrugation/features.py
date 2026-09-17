"""Rail Corrugation feature engineering scaffold."""

from __future__ import annotations

import pandas as pd


def extract_rail_features(frame, config: dict | None = None):
    """Create file-level features for Rail Corrugation classification."""
    if config is None:
        raise RuntimeError(
            "Rail Corrugation feature extraction is not configured until the Info Kit confirms the exact vibration and shock schema."
        )
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Rail feature extraction expects a pandas DataFrame.")
    raise RuntimeError("Rail Corrugation feature engineering is not ready until the schema and labels are confirmed.")


__all__ = ["extract_rail_features"]
