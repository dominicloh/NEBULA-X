"""Door preprocessing scaffold.

This module intentionally does not hard-code the final Door schema until the
Info Kit confirms the exact column names, timestamps and cycle rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_door_stream(path, required_columns: Iterable[str] | None = None):
    """Load the continuous Door stream while preserving timestamps and checking validity."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Door stream file not found: {file_path}")

    stream = pd.read_csv(file_path)
    if required_columns is not None:
        required = list(required_columns)
        missing = [column for column in required if column not in stream.columns]
        if missing:
            raise ValueError(
                "Door stream is missing required columns: " + ", ".join(missing)
                + ". Confirm the exact schema from the Door Info Kit."
            )

    if "timestamp" in stream.columns:
        stream["timestamp"] = pd.to_datetime(stream["timestamp"], errors="coerce")
        if stream["timestamp"].isna().any():
            raise ValueError("Door stream contains invalid timestamp values; confirm the exact timestamp format from the Door Info Kit.")
        stream = stream.sort_values("timestamp").reset_index(drop=True)

    duplicate_timestamps = stream.duplicated(subset=["timestamp"]).sum() if "timestamp" in stream.columns else 0
    if duplicate_timestamps:
        raise ValueError(f"Door stream contains {duplicate_timestamps} duplicate timestamps; check the continuous stream for repeated samples.")

    missing_values = stream.isna().sum().to_dict()
    if any(value > 0 for value in missing_values.values()):
        raise ValueError(
            "Door stream contains missing values in one or more signal columns: "
            + ", ".join(f"{key}={value}" for key, value in missing_values.items() if value > 0)
        )

    return stream


def validate_door_columns(stream, required_columns):
    """Validate required Door columns using configuration without hard-coding the schema."""
    required = list(required_columns)
    missing = [column for column in required if column not in stream.columns]
    if missing:
        raise ValueError("Missing Door columns: " + ", ".join(missing))
    return True


def sort_door_stream(stream):
    """Sort a Door stream chronologically if a timestamp column is available."""
    if "timestamp" in stream.columns:
        stream = stream.sort_values("timestamp").reset_index(drop=True)
    return stream


def check_duplicate_door_timestamps(stream):
    """Return the number of duplicate timestamps in the Door stream."""
    if "timestamp" not in stream.columns:
        return 0
    return int(stream.duplicated(subset=["timestamp"]).sum())


def check_missing_door_values(stream):
    """Return a dictionary of missing values by column for the Door stream."""
    return stream.isna().sum().to_dict()


__all__ = [
    "load_door_stream",
    "validate_door_columns",
    "sort_door_stream",
    "check_duplicate_door_timestamps",
    "check_missing_door_values",
]
