"""Door preprocessing: loading, Datetime parsing, and structural validation.

The exact column names, dtypes and Datetime format below are all confirmed
from the real Train.csv/Test.csv files (see planning/door_handoff.md and
src/door/config.py) -- this module no longer guesses the schema.

This module deliberately does NOT silently clean bad data (no filling
missing values, no dropping bad rows without telling you). If something
looks wrong, it raises a clear error naming the problem, per the Door
handoff guide's "don't hide data-quality problems" rule.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402

# The column pandas adds internally once a Datetime string has been parsed.
# Kept separate from the raw "Datetime" column so the original string is
# never silently overwritten.
PARSED_DATETIME_COLUMN = "_datetime_parsed"


def parse_door_datetime(value: str) -> pd.Timestamp:
    """Parse one Door timestamp string into a pandas Timestamp.

    Confirmed format: Year-Month-Date-Hour-Minute-Second-Millisecond,
    hyphen-separated, NOT zero-padded -- e.g. "2023-7-5-0-0-3-760".
    `pandas.to_datetime` cannot parse this format on its own, so we split
    and build the Timestamp by hand. Raises ValueError with the offending
    value if it doesn't have exactly 7 hyphen-separated integer fields.
    """
    parts = value.split("-")
    if len(parts) != 7:
        raise ValueError(
            f"Door timestamp {value!r} does not have the expected 7 hyphen-separated fields "
            "(Year-Month-Date-Hour-Minute-Second-Millisecond)."
        )
    try:
        year, month, day, hour, minute, second, millisecond = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Door timestamp {value!r} contains a non-integer field.") from exc

    return pd.Timestamp(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
        microsecond=millisecond * 1000,
    )


def parse_door_datetime_series(series: pd.Series) -> pd.Series:
    """Parse a whole column of Door timestamp strings at once."""
    try:
        return series.map(parse_door_datetime)
    except ValueError as exc:
        raise ValueError(f"Could not parse one or more Door timestamps in column {series.name!r}: {exc}") from exc


def validate_door_columns(frame: pd.DataFrame, required_columns: Iterable[str] = config.EXPECTED_COLUMNS) -> None:
    """Confirm every required Door column is present. Raises ValueError, listing
    exactly which columns are missing, rather than failing on the first one.
    """
    required = list(required_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Door data is missing required column(s): " + ", ".join(missing) + ". "
            f"Expected: {list(required)}. Found: {list(frame.columns)}."
        )


def load_door_csv(source) -> pd.DataFrame:
    """Load one Door CSV (Train.csv, Test.csv, or an uploaded file-like object).

    - Validates that all 17 confirmed columns are present.
    - Parses the "Datetime" column into `PARSED_DATETIME_COLUMN`
      (the original "Datetime" string column is kept untouched).
    - Sorts chronologically by parsed Datetime (Train.csv/Test.csv are
      already sorted, but an uploaded file might not be).
    - Raises a clear error on missing values, duplicate rows, or duplicate
      timestamps -- it does NOT silently drop or fix them.

    Works from a path (str/Path) or a file-like object with a `.name`
    attribute (e.g. a Streamlit UploadedFile), the same convention used by
    src/rail_corrugation/predict.py.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Door CSV file not found: {path}")
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(source)

    if frame.empty:
        raise ValueError("Door CSV has no data rows.")

    validate_door_columns(frame)

    if frame.isna().any().any():
        n_missing = int(frame.isna().sum().sum())
        raise ValueError(
            f"Door data contains {n_missing} missing value(s). The official files have none -- "
            "investigate before proceeding rather than filling them in."
        )

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        raise ValueError(f"Door data contains {duplicate_rows} fully duplicate row(s).")

    frame[PARSED_DATETIME_COLUMN] = parse_door_datetime_series(frame[config.DATETIME_COLUMN])

    duplicate_timestamps = int(frame[PARSED_DATETIME_COLUMN].duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(f"Door data contains {duplicate_timestamps} duplicate Datetime value(s).")

    frame = frame.sort_values(PARSED_DATETIME_COLUMN).reset_index(drop=True)
    return frame


# Backward-compatible alias: earlier scaffold code called this
# `load_door_stream` and assumed a lowercase "timestamp" column, which the
# confirmed schema does not have (the real column is "Datetime"). Existing
# imports (e.g. src/door/__init__.py) keep working via this alias.
def load_door_stream(path, required_columns: Iterable[str] | None = None):
    """Deprecated name for `load_door_csv` -- kept for backward compatibility.

    `required_columns` is accepted but ignored beyond the confirmed 17
    columns `load_door_csv` already checks; pass a source through
    `load_door_csv` directly in new code.
    """
    return load_door_csv(path)


def sort_door_stream(stream: pd.DataFrame) -> pd.DataFrame:
    """Sort a Door stream chronologically. Prefers the already-parsed
    column if present (from load_door_csv), otherwise parses on the fly.
    """
    if PARSED_DATETIME_COLUMN in stream.columns:
        return stream.sort_values(PARSED_DATETIME_COLUMN).reset_index(drop=True)
    if config.DATETIME_COLUMN in stream.columns:
        parsed = parse_door_datetime_series(stream[config.DATETIME_COLUMN])
        return stream.assign(**{PARSED_DATETIME_COLUMN: parsed}).sort_values(PARSED_DATETIME_COLUMN).reset_index(drop=True)
    return stream


def check_duplicate_door_timestamps(stream: pd.DataFrame) -> int:
    """Return the number of duplicate Datetime values in a Door stream."""
    if PARSED_DATETIME_COLUMN in stream.columns:
        return int(stream[PARSED_DATETIME_COLUMN].duplicated().sum())
    if config.DATETIME_COLUMN not in stream.columns:
        return 0
    return int(parse_door_datetime_series(stream[config.DATETIME_COLUMN]).duplicated().sum())


def check_missing_door_values(stream: pd.DataFrame) -> dict:
    """Return a dictionary of missing-value counts by column."""
    return stream.isna().sum().to_dict()


__all__ = [
    "PARSED_DATETIME_COLUMN",
    "parse_door_datetime",
    "parse_door_datetime_series",
    "validate_door_columns",
    "load_door_csv",
    "load_door_stream",
    "sort_door_stream",
    "check_duplicate_door_timestamps",
    "check_missing_door_values",
]
