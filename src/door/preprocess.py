"""Door preprocessing: loading, Datetime parsing, structural validation, and
the Stage 1 cleaning/audit report.

The exact column names, dtypes and Datetime format below are all confirmed
from the real Train.csv/Test.csv files (see planning/door_handoff.md and
src/door/config.py) -- this module no longer guesses the schema.

`load_door_csv` deliberately does NOT silently clean bad data (no filling
missing values, no dropping bad rows without telling you) -- it raises a
clear error naming the problem, per the Door handoff guide's "don't hide
data-quality problems" rule. `validate_and_clean` (used by
notebooks/clean_door_data.py) instead returns a full audit report so every
issue is visible at once, but shares the same timestamp parser and the same
canonical loader rather than a second, independent implementation of either.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import IO, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402

# The column pandas adds internally once a Datetime string has been parsed.
# Kept separate from the raw "Datetime" column so the original string is
# never silently overwritten.
PARSED_DATETIME_COLUMN = "_datetime_parsed"

# Temporary, internal-only column used by `validate_and_clean` to track each
# input row's original position (0-indexed) through cleaning, so row-order
# preservation can be measured directly from positions rather than inferred
# from timestamp values. Always dropped before any DataFrame is returned or
# exported.
SOURCE_ROW_POSITION_COLUMN = "_source_row_position"


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


def parse_door_datetime_series_lenient(series: pd.Series) -> pd.Series:
    """Parse a column of Door timestamps, returning NaT for values that don't
    match the confirmed format instead of raising.

    Uses the exact same field-splitting logic as `parse_door_datetime` (just
    catching its ValueError per value) so there is only ever one Door
    timestamp parser to maintain -- this is the lenient/reporting entry
    point for callers (e.g. validate_and_clean) that need to identify which
    rows are malformed rather than fail on the first bad one.
    """

    def _try_parse(value: object) -> pd.Timestamp:
        try:
            return parse_door_datetime(value)
        except (TypeError, ValueError):
            return pd.NaT

    return series.map(_try_parse)


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


# ---------------------------------------------------------------------------
# Stage 1 cleaning/audit report (moved here from notebooks/clean_door_data.py
# -- that file is now only a thin CLI wrapper around `validate_and_clean` and
# `process_file`). Unlike `load_door_csv` above, `validate_and_clean` never
# raises on a data-quality problem: every issue is recorded in the returned
# report's "fail_reasons" instead, so a caller gets a full audit rather than
# stopping at the first problem. It reuses `parse_door_datetime` (via
# `parse_door_datetime_series` / `parse_door_datetime_series_lenient`) and
# `load_door_csv` -- there is exactly one Door timestamp parser and one
# canonical loader in this codebase, never a second, independent copy.
# ---------------------------------------------------------------------------

# All non-Datetime columns are numeric -- this is exactly config.SIGNAL_COLUMNS
# (config.EXPECTED_COLUMNS minus config.DATETIME_COLUMN), reused rather than
# re-listed so the two can never drift apart.
NUMERIC_COLUMNS = config.SIGNAL_COLUMNS

# Binary on/off signal columns, checked for values outside {0, 1}.
STATE_COLUMNS = (
    "Close command",
    "Open command",
    "DCSR",
    "DCSL",
    "DLSR",
    "DLSL",
    "Door Opened",
    "Door Locked",
    "Door is opening",
    "Door is closing",
)

# Continuous sensor columns, reported with summary statistics.
SENSOR_COLUMNS = (
    "Motor current(mA)",
    "Motor Voltage(10mV)",
    "Motor electrodynamic force",
    "Door leaf position",
)

# Pandas/Excel-style export index columns (e.g. "Unnamed: 0") sometimes left
# behind by a prior `to_csv()` without `index=False`; safe to drop.
EXPORT_COLUMN_PATTERN = re.compile(r"^Unnamed:\s*\d+(?:\.\d+)?$", re.I)


def check_row_order_preserved(source_positions: Iterable[int]) -> bool:
    """True iff `source_positions` (each surviving row's position in the
    original input, in current row order) is still non-decreasing -- i.e.
    whatever produced this sequence did not change the rows' relative order.

    A pure, directly-testable wrapper around the monotonicity check used by
    `validate_and_clean`'s SOURCE_ROW_POSITION_COLUMN checkpoint, so the
    detection logic itself can be verified with a synthetic reordering
    without needing a full CSV that happens to trigger one end-to-end.
    """
    return bool(pd.Series(list(source_positions)).is_monotonic_increasing)


def _read_door_source_as_text(source: str | Path | IO[str] | IO[bytes]) -> pd.DataFrame:
    """Read a Door CSV source as all-string dtype.

    Lenient validation needs to see each cell's raw text (to tell "1", " ",
    "" and "abc" apart) before any numeric/timestamp coercion happens.
    Accepts the same source types as `load_door_csv` (path or file-like),
    with the same clear FileNotFoundError for a missing path.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Door CSV file not found: {path}")
        return pd.read_csv(path, dtype="string", low_memory=False, keep_default_na=False)
    return pd.read_csv(source, dtype="string", low_memory=False, keep_default_na=False)


def validate_and_clean(
    data: pd.DataFrame | str | Path | IO[str] | IO[bytes],
) -> tuple[pd.DataFrame | None, dict]:
    """Validate and clean one Door CSV, returning ``(cleaned_dataframe, report)``.

    Accepts a path, a file-like object, or an already-loaded DataFrame; never
    mutates the input. ``report["pass"]`` is True (and ``cleaned`` is not
    None) only when the required schema is present and no unresolved
    row-level validity issue remains. Gaps, constant columns, and unknown
    extras are warnings, not failures, because they may be genuine
    properties of the stream.
    """
    is_dataframe = isinstance(data, pd.DataFrame)
    frame = data.copy(deep=True) if is_dataframe else _read_door_source_as_text(data)

    report: dict = {
        "pass": False,
        "ready_for_segmentation": False,
        "rows_before": len(frame),
        "rows_after": None,
        "columns_before": list(frame.columns),
        "columns_after": [],
        "missing_required_columns": [],
        "extra_columns": [],
        "removed_export_columns": [],
        "missing_values": {},
        "infinite_values": {},
        "datatype_issues": {},
        "invalid_state_values": {},
        "duplicate_rows_removed": 0,
        "row_order_preserved": None,
        "timestamps_monotonic": None,
        "duplicate_timestamps": {"count_before_cleaning": 0, "count": 0, "row_indices": []},
        "timestamp_errors": {"count": 0, "row_indices": [], "out_of_order_count": 0},
        "constant_columns": [],
        "time_gap_distribution": {
            "consecutive_valid_intervals": 0,
            "interval_counts_ms": {},
            "quantiles_ms": {},
        },
        "sampling_gaps": {
            "count": 0,
            "median_interval_ms": None,
            "threshold_ms": None,
            "gaps": [],
        },
        "irregular_sampling_intervals": 0,
        "numeric_ranges": {},
        "sensor_statistics": {},
        "warnings": [],
        "changes_made": [],
        "fail_reasons": [],
    }

    original_names = list(frame.columns)
    stripped_names = [str(c).strip() for c in original_names]
    if stripped_names != original_names:
        frame.columns = stripped_names
        report["changes_made"].append("Trimmed whitespace from column names")
    if frame.columns.duplicated().any():
        report["fail_reasons"].append("Duplicate column names after normalization")
        report["columns_after"] = list(frame.columns)
        report["rows_after"] = len(frame)
        return None, report

    export_columns = [c for c in frame if EXPORT_COLUMN_PATTERN.fullmatch(c)]
    if export_columns:
        frame = frame.drop(columns=export_columns)
        report["removed_export_columns"] = export_columns
        report["changes_made"].append(f"Removed export index columns: {export_columns}")
    report["missing_required_columns"] = [c for c in config.EXPECTED_COLUMNS if c not in frame]
    report["extra_columns"] = [c for c in frame if c not in config.EXPECTED_COLUMNS]
    if report["extra_columns"]:
        report["warnings"].append(f"Unknown extra columns retained: {report['extra_columns']}")
    if report["missing_required_columns"]:
        report["fail_reasons"].append(f"Missing required columns: {report['missing_required_columns']}")
        report["rows_after"] = len(frame)
        report["columns_after"] = list(frame.columns)
        return None, report
    if PARSED_DATETIME_COLUMN in frame:
        report["fail_reasons"].append(
            f"Input column {PARSED_DATETIME_COLUMN!r} conflicts with the parsed timestamp output"
        )
        report["rows_after"] = len(frame)
        report["columns_after"] = list(frame.columns)
        return None, report
    if SOURCE_ROW_POSITION_COLUMN in frame:
        report["fail_reasons"].append(
            f"Input column {SOURCE_ROW_POSITION_COLUMN!r} conflicts with the internal row-position tracking field"
        )
        report["rows_after"] = len(frame)
        report["columns_after"] = list(frame.columns)
        return None, report

    raw_parsed = parse_door_datetime_series_lenient(frame[config.DATETIME_COLUMN])
    report["duplicate_timestamps"]["count_before_cleaning"] = int(
        (raw_parsed.notna() & raw_parsed.duplicated(keep=False)).sum()
    )

    # Tag each surviving row with its original position so duplicate-row
    # removal's effect on row order can be measured directly from positions,
    # not inferred from Datetime values.
    frame[SOURCE_ROW_POSITION_COLUMN] = np.arange(len(frame))
    data_columns = [c for c in frame.columns if c != SOURCE_ROW_POSITION_COLUMN]
    exact_duplicates = int(frame.duplicated(subset=data_columns).sum())
    if exact_duplicates:
        frame = frame.drop_duplicates(subset=data_columns).reset_index(drop=True)
        report["changes_made"].append(f"Removed {exact_duplicates} exact duplicate rows")
    report["duplicate_rows_removed"] = exact_duplicates

    # Genuinely measured from the temporary position field -- NOT derived
    # from timestamp ordering -- whether duplicate-row removal (the only
    # row-removing step in this function) changed the relative order of the
    # surviving rows. Dropped immediately after so it never reaches numeric
    # processing, reporting, or the cleaned output.
    report["row_order_preserved"] = check_row_order_preserved(frame[SOURCE_ROW_POSITION_COLUMN])
    frame = frame.drop(columns=[SOURCE_ROW_POSITION_COLUMN])

    for column in NUMERIC_COLUMNS:
        raw = frame[column]
        text = raw.astype("string").str.strip()
        blank = text.isna() | text.eq("").fillna(False)
        converted = pd.to_numeric(text.mask(blank), errors="coerce")
        nonnumeric = (~blank & converted.isna()).fillna(False)
        if nonnumeric.any():
            count = int(nonnumeric.sum())
            report["datatype_issues"][column] = {
                "count": count,
                "row_indices": frame.index[nonnumeric].tolist()[:20],
            }
            report["fail_reasons"].append(f"Nonnumeric values in {column}: {count}")
        infinity = np.isinf(converted.to_numpy(dtype=float, na_value=np.nan))
        report["infinite_values"][column] = int(infinity.sum())
        if infinity.any():
            converted = converted.mask(infinity)
            report["changes_made"].append(f"Converted {int(infinity.sum())} infinite values in {column} to missing")
            report["fail_reasons"].append(f"Infinite values in {column}: {int(infinity.sum())}")
        frame[column] = converted

    for column in STATE_COLUMNS:
        invalid = frame[column].notna() & ~frame[column].isin([0, 1])
        if invalid.any():
            values = frame.loc[invalid, column].drop_duplicates().astype(str).tolist()[:20]
            report["invalid_state_values"][column] = {
                "count": int(invalid.sum()),
                "values": values,
            }
            report["fail_reasons"].append(f"State column {column} contains values outside 0/1")

    parsed = parse_door_datetime_series_lenient(frame[config.DATETIME_COLUMN])
    bad_time = parsed.isna()
    report["timestamp_errors"]["count"] = int(bad_time.sum())
    report["timestamp_errors"]["row_indices"] = frame.index[bad_time].tolist()[:20]
    if bad_time.any():
        report["fail_reasons"].append(f"Invalid timestamps: {int(bad_time.sum())}")
    duplicate_time = parsed.notna() & parsed.duplicated(keep=False)
    report["duplicate_timestamps"] = {
        "count_before_cleaning": report["duplicate_timestamps"]["count_before_cleaning"],
        "count": int(duplicate_time.sum()),
        "row_indices": frame.index[duplicate_time].tolist()[:20],
    }
    if duplicate_time.any():
        report["fail_reasons"].append(f"Duplicate timestamps: {int(duplicate_time.sum())} rows")
    # diff() on the full stream prevents a missing timestamp from creating a
    # synthetic interval between two rows that were not originally adjacent.
    intervals = parsed.diff().dt.total_seconds().mul(1000)
    backward = intervals.lt(0)
    report["timestamp_errors"]["out_of_order_count"] = int(backward.sum())
    if backward.any():
        report["fail_reasons"].append(f"Timestamps out of order: {int(backward.sum())} backward steps")
    positive_intervals = intervals[intervals.gt(0)]
    valid_intervals = intervals.dropna()
    report["time_gap_distribution"] = {
        "consecutive_valid_intervals": len(valid_intervals),
        "interval_counts_ms": {
            str(round(float(ms), 3)): int(count)
            for ms, count in valid_intervals.round(3).value_counts().sort_index().items()
        },
        "quantiles_ms": {
            label: round(float(valid_intervals.quantile(q)), 3)
            for label, q in (("min", 0), ("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("max", 1))
        }
        if not valid_intervals.empty
        else {},
    }
    if not positive_intervals.empty:
        median_ms = float(positive_intervals.median())
        threshold_ms = 5 * median_ms
        gap_mask = intervals.gt(threshold_ms)
        report["sampling_gaps"] = {
            "count": int(gap_mask.sum()),
            "median_interval_ms": median_ms,
            "threshold_ms": threshold_ms,
            "gaps": [
                {
                    "previous_row_index": int(i) - 1,
                    "row_index": int(i),
                    "previous_datetime": str(frame.iloc[int(i) - 1][config.DATETIME_COLUMN]),
                    "datetime": str(frame.iloc[int(i)][config.DATETIME_COLUMN]),
                    "gap_ms": round(float(v), 3),
                }
                for i, v in intervals[gap_mask].items()
            ],
        }
        irregular = positive_intervals.sub(median_ms).abs().gt(0.2 * median_ms)
        report["irregular_sampling_intervals"] = int(irregular.sum())
        if gap_mask.any():
            report["warnings"].append(f"{int(gap_mask.sum())} gaps exceed five times the median sampling interval")
        if irregular.any():
            report["warnings"].append(f"{int(irregular.sum())} positive intervals differ from the median by over 20%")

    # Timestamp ORDERING is reported separately from row-position preservation
    # (see `row_order_preserved` above, computed purely from positions) --
    # this is the timestamp-value side, e.g. an upload whose rows are already
    # in their original order but whose Datetime values are not chronological.
    # Left as None (unknown) rather than guessed when malformed timestamps
    # make the ordering undefined.
    report["timestamps_monotonic"] = None if bad_time.any() else bool(parsed.is_monotonic_increasing)

    report["missing_values"] = {c: int(n) for c, n in frame.isna().sum().items() if n}
    if report["missing_values"]:
        report["fail_reasons"].append("Missing values remain; no imputation applied")
    report["constant_columns"] = [
        c for c in frame if c not in (config.DATETIME_COLUMN, PARSED_DATETIME_COLUMN) and frame[c].nunique(dropna=True) <= 1
    ]
    if report["constant_columns"]:
        report["warnings"].append(f"Constant columns retained: {report['constant_columns']}")
    for column in NUMERIC_COLUMNS:
        series = frame[column]
        report["numeric_ranges"][column] = {
            "min": _number_or_none(series.min()),
            "max": _number_or_none(series.max()),
        }
    for column in SENSOR_COLUMNS:
        series = frame[column]
        report["sensor_statistics"][column] = {
            "min": _number_or_none(series.min()),
            "max": _number_or_none(series.max()),
            "mean": _number_or_none(series.mean()),
            "std": _number_or_none(series.std()),
            "missing_count": int(series.isna().sum()),
            "infinite_count": report["infinite_values"][column],
        }
    report["rows_after"] = len(frame)
    report["columns_after"] = list(frame.columns)
    report["pass"] = not report["fail_reasons"]
    report["ready_for_segmentation"] = report["pass"]

    if not report["pass"]:
        return None, report

    # Build the cleaned output by reusing the canonical loader wherever
    # possible, rather than a second implementation of "parse + sort".
    # `load_door_csv` can only take a fresh path/file-like read from the
    # original source (it re-validates and re-parses on its own), so it is
    # only equivalent to what we just validated when nothing upstream of it
    # needed to change: no renamed/dropped columns, no removed duplicate
    # rows, and not an in-memory DataFrame to begin with.
    unchanged_structure = (
        not is_dataframe
        and exact_duplicates == 0
        and not export_columns
        and stripped_names == original_names
        and not report["extra_columns"]
    )
    if unchanged_structure:
        if hasattr(data, "seek"):
            data.seek(0)
        cleaned = load_door_csv(data)
    else:
        cleaned = frame.copy()
        cleaned[PARSED_DATETIME_COLUMN] = parse_door_datetime_series(cleaned[config.DATETIME_COLUMN])
        cleaned = cleaned.sort_values(PARSED_DATETIME_COLUMN, kind="stable").reset_index(drop=True)

    report["changes_made"].append("Added parsed timestamp; original Datetime text preserved")
    return cleaned, report


def _number_or_none(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def process_file(source: Path, output_dir: Path) -> dict:
    """Run `validate_and_clean` on one Door CSV file and write its cleaned
    output + JSON audit report into `output_dir`. Returns the report dict.
    """
    cleaned, report = validate_and_clean(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{source.stem}_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if cleaned is not None:
        output = output_dir / f"{source.stem}_cleaned.csv"
        cleaned.to_csv(output, index=False)
    print(f"{source.name}: {'PASS' if report['pass'] else 'FAIL'}; {report['rows_before']} -> {report['rows_after']} rows")
    return report


__all__ = [
    "PARSED_DATETIME_COLUMN",
    "SOURCE_ROW_POSITION_COLUMN",
    "parse_door_datetime",
    "parse_door_datetime_series",
    "parse_door_datetime_series_lenient",
    "validate_door_columns",
    "load_door_csv",
    "load_door_stream",
    "sort_door_stream",
    "check_duplicate_door_timestamps",
    "check_missing_door_values",
    "check_row_order_preserved",
    "NUMERIC_COLUMNS",
    "STATE_COLUMNS",
    "SENSOR_COLUMNS",
    "validate_and_clean",
    "process_file",
]
