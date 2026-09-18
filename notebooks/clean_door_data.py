"""Stage 1 only: validate and clean a continuous Door sensor CSV.

Run ``python notebooks/clean_door_data.py`` to process the repository's Train
and Test files. ``validate_and_clean`` also accepts a DataFrame or uploaded
file-like object and returns ``(cleaned_dataframe, report)``. Ground-truth
segment labels are deliberately outside this module.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "Datetime",
    "Motor current(mA)",
    "Motor Voltage(10mV)",
    "Motor electrodynamic force",
    "Door opening time(.1s)",
    "Door closing time(.1s)",
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
    "Door leaf position",
)
STATE_COLUMNS = (
    "Close command", "Open command", "DCSR", "DCSL", "DLSR", "DLSL",
    "Door Opened", "Door Locked", "Door is opening", "Door is closing",
)
SENSOR_COLUMNS = (
    "Motor current(mA)", "Motor Voltage(10mV)",
    "Motor electrodynamic force", "Door leaf position",
)
NUMERIC_COLUMNS = tuple(c for c in REQUIRED_COLUMNS if c != "Datetime")
TIMESTAMP_PATTERN = (
    r"^(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})-"
    r"(\d{1,2})-(\d{1,2})-(\d{1,3})$"
)
EXPORT_COLUMN_PATTERN = re.compile(r"^Unnamed:\s*\d+(?:\.\d+)?$", re.I)
PARSED_COLUMN = "timestamp"


def parse_door_datetime(values: pd.Series) -> pd.Series:
    """Parse seven-part timestamps; the last part is milliseconds."""
    parts = values.astype("string").str.extract(TIMESTAMP_PATTERN)
    parts = parts.apply(pd.to_numeric, errors="coerce")
    malformed = parts.isna().any(axis=1)
    # pandas' date assembler may reject nullable integer columns before it can
    # apply errors="coerce". Fill temporarily, then restore invalid rows to NaT.
    safe = parts.fillna(1).astype("int64")
    seconds = pd.to_datetime(
        dict(
            year=safe[0], month=safe[1], day=safe[2],
            hour=safe[3], minute=safe[4], second=safe[5],
        ),
        errors="coerce",
    )
    return (seconds + pd.to_timedelta(safe[6], unit="ms")).mask(malformed)


def _read_input(data: pd.DataFrame | str | Path | IO[str] | IO[bytes]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy(deep=True)
    # Read as text to expose nonnumeric cells before converting them.
    return pd.read_csv(data, dtype="string", low_memory=False, keep_default_na=False)


def validate_and_clean(
    data: pd.DataFrame | str | Path | IO[str] | IO[bytes],
) -> tuple[pd.DataFrame | None, dict]:
    """Return cleaned data and an audit report; never alter the input object.

    ``pass`` means the required schema is present and no unresolved row-level
    validity issue remains. Gaps, constant columns, and unknown extras are
    warnings because they may be genuine properties of the stream.
    """
    frame = _read_input(data)
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
        "rows_reordered": 0,
        "row_order_preserved": True,
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
    report["missing_required_columns"] = [c for c in REQUIRED_COLUMNS if c not in frame]
    report["extra_columns"] = [c for c in frame if c not in REQUIRED_COLUMNS]
    if report["extra_columns"]:
        report["warnings"].append(
            f"Unknown extra columns retained: {report['extra_columns']}"
        )
    if report["missing_required_columns"]:
        report["fail_reasons"].append(
            f"Missing required columns: {report['missing_required_columns']}"
        )
        report["rows_after"] = len(frame)
        report["columns_after"] = list(frame.columns)
        return None, report
    if PARSED_COLUMN in frame:
        report["fail_reasons"].append(
            f"Input column {PARSED_COLUMN!r} conflicts with the parsed timestamp output"
        )
        report["rows_after"] = len(frame)
        report["columns_after"] = list(frame.columns)
        return None, report

    raw_parsed = parse_door_datetime(frame["Datetime"])
    report["duplicate_timestamps"]["count_before_cleaning"] = int(
        (raw_parsed.notna() & raw_parsed.duplicated(keep=False)).sum()
    )
    exact_duplicates = int(frame.duplicated().sum())
    if exact_duplicates:
        frame = frame.drop_duplicates().reset_index(drop=True)
        report["changes_made"].append(f"Removed {exact_duplicates} exact duplicate rows")
    report["duplicate_rows_removed"] = exact_duplicates

    for column in NUMERIC_COLUMNS:
        raw = frame[column]
        text = raw.astype("string").str.strip()
        blank = text.isna() | text.eq("").fillna(False)
        converted = pd.to_numeric(text.mask(blank), errors="coerce")
        nonnumeric = (~blank & converted.isna()).fillna(False)
        if nonnumeric.any():
            count = int(nonnumeric.sum())
            report["datatype_issues"][column] = {
                "count": count, "row_indices": frame.index[nonnumeric].tolist()[:20]
            }
            report["fail_reasons"].append(f"Nonnumeric values in {column}: {count}")
        infinity = np.isinf(converted.to_numpy(dtype=float, na_value=np.nan))
        report["infinite_values"][column] = int(infinity.sum())
        if infinity.any():
            converted = converted.mask(infinity)
            report["changes_made"].append(
                f"Converted {int(infinity.sum())} infinite values in {column} to missing"
            )
            report["fail_reasons"].append(
                f"Infinite values in {column}: {int(infinity.sum())}"
            )
        frame[column] = converted

    for column in STATE_COLUMNS:
        invalid = frame[column].notna() & ~frame[column].isin([0, 1])
        if invalid.any():
            values = frame.loc[invalid, column].drop_duplicates().astype(str).tolist()[:20]
            report["invalid_state_values"][column] = {
                "count": int(invalid.sum()), "values": values,
            }
            report["fail_reasons"].append(
                f"State column {column} contains values outside 0/1"
            )

    parsed = parse_door_datetime(frame["Datetime"])
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
        report["fail_reasons"].append(
            f"Duplicate timestamps: {int(duplicate_time.sum())} rows"
        )
    # diff() on the full stream prevents a missing timestamp from creating a
    # synthetic interval between two rows that were not originally adjacent.
    intervals = parsed.diff().dt.total_seconds().mul(1000)
    backward = intervals.lt(0)
    report["timestamp_errors"]["out_of_order_count"] = int(backward.sum())
    if backward.any():
        report["fail_reasons"].append(
            f"Timestamps out of order: {int(backward.sum())} backward steps"
        )
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
            for label, q in (("min", 0), ("p50", 0.5), ("p90", 0.9),
                             ("p95", 0.95), ("p99", 0.99), ("max", 1))
        } if not valid_intervals.empty else {},
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
                    "previous_datetime": str(frame.iloc[int(i) - 1]["Datetime"]),
                    "datetime": str(frame.iloc[int(i)]["Datetime"]),
                    "gap_ms": round(float(v), 3),
                }
                for i, v in intervals[gap_mask].items()
            ],
        }
        irregular = positive_intervals.sub(median_ms).abs().gt(0.2 * median_ms)
        report["irregular_sampling_intervals"] = int(irregular.sum())
        if gap_mask.any():
            report["warnings"].append(
                f"{int(gap_mask.sum())} gaps exceed five times the median sampling interval"
            )
        if irregular.any():
            report["warnings"].append(
                f"{int(irregular.sum())} positive intervals differ from the median by over 20%"
            )
    frame.insert(1, PARSED_COLUMN, parsed)
    report["changes_made"].append(
        "Added parsed timestamp; original Datetime text and row order preserved"
    )

    report["missing_values"] = {
        c: int(n) for c, n in frame.isna().sum().items() if n
    }
    if report["missing_values"]:
        report["fail_reasons"].append("Missing values remain; no imputation applied")
    report["constant_columns"] = [
        c for c in frame if c not in ("Datetime", PARSED_COLUMN)
        and frame[c].nunique(dropna=True) <= 1
    ]
    if report["constant_columns"]:
        report["warnings"].append(
            f"Constant columns retained: {report['constant_columns']}"
        )
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
    return frame, report


def _number_or_none(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def process_file(source: Path, output_dir: Path) -> dict:
    cleaned, report = validate_and_clean(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{source.stem}_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if cleaned is not None:
        output = output_dir / f"{source.stem}_cleaned.csv"
        cleaned.to_csv(output, index=False)
    print(f"{source.name}: {'PASS' if report['pass'] else 'FAIL'}; "
          f"{report['rows_before']} -> {report['rows_after']} rows")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_source = root / "organiser-materials" / "PS3" / "02_Datasets" / "Door"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, default=[default_source / "Train.csv", default_source / "Test.csv"])
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed" / "door_stage1")
    args = parser.parse_args()
    for source in args.files:
        process_file(source, args.output_dir)


if __name__ == "__main__":
    main()
