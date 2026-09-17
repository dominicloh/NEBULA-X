"""Shared schema validation utilities for Door and Rail Corrugation outputs."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

VALID_DOOR_LABELS = ("Normal", "Abnormal resistance")
VALID_RAIL_LABELS = ("Normal", "Side I", "Side II")


def _validate_labels(frame: pd.DataFrame, column: str, allowed_labels: Iterable[str], subsystem: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"{subsystem} output is missing the '{column}' column.")

    invalid = sorted(set(frame[column].dropna().astype(str)) - set(allowed_labels))
    if invalid:
        raise ValueError(
            f"{subsystem} output contains unsupported labels in '{column}': {invalid}. "
            f"Allowed values: {list(allowed_labels)}"
        )


def validate_door_predictions(df: pd.DataFrame) -> None:
    """Validate Door predictions against the required segment output contract."""
    expected_columns = ["start_time", "end_time", "prediction"]
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Door output is missing required columns: " + ", ".join(missing_columns)
        )

    actual_columns = list(df.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            "Door output column names/order are invalid. Expected: "
            + ", ".join(expected_columns)
            + "; found: "
            + ", ".join(actual_columns)
        )

    if df.empty:
        raise ValueError("Door output requires at least one predicted segment for a final submission.")

    if df.isnull().any().any():
        raise ValueError("Door output contains null values in required columns.")

    if "start_time" not in df.columns or "end_time" not in df.columns:
        raise ValueError("Door output must contain both start_time and end_time values.")

    _validate_labels(df, "prediction", VALID_DOOR_LABELS, "Door")


def validate_rail_predictions(df: pd.DataFrame) -> None:
    """Validate Rail Corrugation predictions against the required file-level contract."""
    expected_columns = ["file_id", "prediction"]
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Rail output is missing required columns: " + ", ".join(missing_columns)
        )

    actual_columns = list(df.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            "Rail output column names/order are invalid. Expected: "
            + ", ".join(expected_columns)
            + "; found: "
            + ", ".join(actual_columns)
        )

    if df.empty:
        raise ValueError("Rail output requires at least one prediction row for a final submission.")

    if df.isnull().any().any():
        raise ValueError("Rail output contains null values in required columns.")

    _validate_labels(df, "prediction", VALID_RAIL_LABELS, "Rail Corrugation")
