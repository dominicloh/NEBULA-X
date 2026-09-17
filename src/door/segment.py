"""Door segmentation scaffold.

This module intentionally refuses to invent exact cycle boundaries until the Info Kit
confirms the signal rules, timestamp format and expected cycle behaviour.
"""

from __future__ import annotations

import pandas as pd


def detect_door_segments(stream, config: dict | None = None):
    """Detect door operating cycles in the continuous stream.

    This is intentionally not implemented with fake segmentation boundaries. Once the
    Info Kit confirms the required rules, this function should return a DataFrame with
    start_time, end_time and raw segment metadata.
    """
    if config is None:
        raise RuntimeError(
            "Door segmentation is not configured yet. Read the Door Info Kit and provide the required cycle rules before training or inference."
        )

    if not isinstance(stream, pd.DataFrame):
        raise TypeError("Door segmentation requires a pandas DataFrame stream.")

    raise RuntimeError(
        "Door segmentation model not ready: the actual segmentation rule set and signal thresholds are not yet confirmed from the Door Info Kit."
    )


def segment_door_stream(stream, config: dict | None = None):
    """Backward-compatible wrapper for Door segmentation entry points."""
    return detect_door_segments(stream, config=config)


__all__ = ["detect_door_segments", "segment_door_stream"]
