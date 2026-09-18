"""Door cycle segmentation using a frozen Train-derived timestamp-gap rule.

This module never loads ground-truth answers. It splits only at consecutive
timestamp gaps above the configured threshold and returns the original
timestamp text at each detected block's first and last sample.
"""

from __future__ import annotations

import math

import pandas as pd

from src.door import config as door_config
from src.door.preprocess import PARSED_DATETIME_COLUMN, parse_door_datetime_series

# Classification assigns status later. These columns flow directly into
# extract_cycle_features() and retain the organiser's native time format.
SEGMENT_COLUMNS = ("segment_id", "start_time", "end_time")


def detect_cycles(dataframe: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Return one chronologically ordered, non-overlapping row per cycle.

    ``config`` is an internal testing/experimentation override for
    ``gap_threshold_ms``. Normal prediction calls omit it and use the frozen
    value in door/config.py.
    The input order is never changed or repaired here.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("detect_cycles expects a pandas DataFrame (see src.door.preprocess.load_door_csv).")
    if dataframe.empty:
        raise ValueError("detect_cycles requires at least two Door readings.")
    if door_config.DATETIME_COLUMN not in dataframe.columns:
        raise ValueError(f"detect_cycles requires the original {door_config.DATETIME_COLUMN!r} column.")
    if PARSED_DATETIME_COLUMN not in dataframe.columns:
        raise ValueError(
            f"detect_cycles expects a DataFrame with a {PARSED_DATETIME_COLUMN!r} column -- "
            "load it with src.door.preprocess.load_door_csv first."
        )
    parameters = door_config.SEGMENTATION_CONFIG if config is None else config
    if not isinstance(parameters, dict) or set(parameters) != {"gap_threshold_ms"}:
        raise ValueError("Segmentation config must contain only gap_threshold_ms.")
    threshold_ms = parameters["gap_threshold_ms"]
    if (
        isinstance(threshold_ms, bool)
        or not isinstance(threshold_ms, (int, float))
        or not math.isfinite(threshold_ms)
        or threshold_ms <= 0
    ):
        raise ValueError("gap_threshold_ms must be a finite positive number.")

    timestamps = dataframe[PARSED_DATETIME_COLUMN].reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise ValueError(f"{PARSED_DATETIME_COLUMN!r} must have a datetime dtype.")
    if timestamps.isna().any():
        raise ValueError(f"{PARSED_DATETIME_COLUMN!r} contains missing or invalid timestamps.")
    raw = dataframe[door_config.DATETIME_COLUMN].reset_index(drop=True)
    try:
        reparsed = parse_door_datetime_series(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed original Door Datetime: {exc}") from exc
    if not reparsed.equals(timestamps):
        raise ValueError("Parsed timestamps do not match the original Datetime text.")
    differences = timestamps.diff()
    if differences.iloc[1:].le(pd.Timedelta(0)).any():
        raise ValueError("Door timestamps must be strictly increasing in source row order; duplicate or unsorted input found.")

    threshold = pd.Timedelta(milliseconds=threshold_ms)
    cuts = [int(i) for i in differences.index[differences.gt(threshold)]]
    starts = [0, *cuts]
    ends = [i - 1 for i in cuts] + [len(dataframe) - 1]
    if any(start >= end for start, end in zip(starts, ends)):
        raise ValueError("A timestamp gap leaves a block with fewer than two readings; cycle boundaries are ambiguous.")
    return pd.DataFrame(
        [
            (f"detected_seg_{number:03d}", raw.iloc[start], raw.iloc[end])
            for number, (start, end) in enumerate(zip(starts, ends), start=1)
        ],
        columns=list(SEGMENT_COLUMNS),
    )


# Backward-compatible aliases -- earlier scaffold code imported these names.
def detect_door_segments(stream: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Deprecated name for `detect_cycles` -- kept for backward compatibility."""
    return detect_cycles(stream, config=config)


def segment_door_stream(stream: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Deprecated name for `detect_cycles` -- kept for backward compatibility."""
    return detect_cycles(stream, config=config)


def validate_segment_table(segments: pd.DataFrame, dataframe: pd.DataFrame | None = None) -> list[str]:
    """Check a table of segments (e.g. Train_Segments_Answer.csv, or your
    own detect_cycles() output) for structural problems.

    Returns a list of human-readable problem descriptions (empty = valid).
    Used by both inspect_data.py (read-only reporting) and train.py
    (refusing to train on a broken segment table) so this logic lives in
    exactly one place.

    Checks: start before end, no duplicate segment_id, no overlapping
    consecutive segments, and (if `dataframe` is given) every boundary
    falls inside the dataframe's own time range.
    """
    problems: list[str] = []
    required = {"start_time", "end_time"}
    missing = required - set(segments.columns)
    if missing:
        return [f"Segment table is missing required column(s): {sorted(missing)}"]

    from src.door.preprocess import PARSED_DATETIME_COLUMN, parse_door_datetime

    def _as_timestamp(value):
        return value if isinstance(value, pd.Timestamp) else parse_door_datetime(str(value))

    starts = segments["start_time"].map(_as_timestamp)
    ends = segments["end_time"].map(_as_timestamp)

    invalid_order = int((starts >= ends).sum())
    if invalid_order:
        problems.append(f"{invalid_order} segment(s) have start_time >= end_time.")

    if "segment_id" in segments.columns:
        duplicate_ids = int(segments["segment_id"].duplicated().sum())
        if duplicate_ids:
            problems.append(f"{duplicate_ids} duplicate segment_id value(s) found.")

    order = starts.sort_values().index
    sorted_starts = starts.loc[order].reset_index(drop=True)
    sorted_ends = ends.loc[order].reset_index(drop=True)
    overlaps = int((sorted_ends.iloc[:-1].reset_index(drop=True) > sorted_starts.iloc[1:].reset_index(drop=True)).sum())
    if overlaps:
        problems.append(f"{overlaps} pair(s) of consecutive segments overlap in time.")

    if dataframe is not None and PARSED_DATETIME_COLUMN in dataframe.columns:
        data_min = dataframe[PARSED_DATETIME_COLUMN].min()
        data_max = dataframe[PARSED_DATETIME_COLUMN].max()
        outside = int(((starts < data_min) | (ends > data_max)).sum())
        if outside:
            problems.append(f"{outside} segment(s) fall outside the given stream's own time range ({data_min} to {data_max}).")

    return problems


__all__ = [
    "SEGMENT_COLUMNS",
    "detect_cycles",
    "detect_door_segments",
    "segment_door_stream",
    "validate_segment_table",
]
