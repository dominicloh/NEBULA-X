#!/usr/bin/env python3
"""Door dataset inspector.

Read-only inspection of the official Door dataset (Train.csv, Test.csv,
Train_Segments_Answer.csv). Never modifies the input CSVs and never trains
a model -- it only reports facts about the data. Confirmed facts this
script computes are recorded in planning/door_handoff.md.

Usage:
    python src/door/inspect_data.py [dataset_dir]

Example (dataset_dir defaults to the organiser materials sibling folder):
    python src/door/inspect_data.py
    python src/door/inspect_data.py "../organiser-materials/PS3/02_Datasets/Door"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402
from src.door.preprocess import PARSED_DATETIME_COLUMN, load_door_csv  # noqa: E402
from src.door.segment import validate_segment_table  # noqa: E402


def find_dataset_files(dataset_root: Path) -> dict:
    """Locate Train.csv, Test.csv and Train_Segments_Answer.csv. Fails with
    a clear message listing exactly what's missing, rather than a generic
    FileNotFoundError from deep inside pandas.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Door dataset directory does not exist: {dataset_root}\n"
            "Pass the correct path as the command-line argument, e.g.:\n"
            '  python src/door/inspect_data.py "../organiser-materials/PS3/02_Datasets/Door"'
        )

    paths = {
        "train": dataset_root / config.TRAIN_FILENAME,
        "test": dataset_root / config.TEST_FILENAME,
        "answers": dataset_root / config.TRAIN_SEGMENTS_ANSWER_FILENAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Door dataset directory is missing expected file(s):\n  " + "\n  ".join(missing))
    return paths


def report_schema(name: str, frame: pd.DataFrame) -> None:
    print(f"\n--- {name} ---")
    print(f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print("Columns and dtypes:")
    for column, dtype in frame.dtypes.items():
        print(f"  {column}  ({dtype})")


def report_data_quality(name: str, frame: pd.DataFrame) -> None:
    n_missing = int(frame.isna().sum().sum())
    n_duplicate_rows = int(frame.duplicated().sum())
    numeric_frame = frame.select_dtypes(include="number")
    n_infinite = 0
    if not numeric_frame.empty:
        n_infinite = int(np.isinf(numeric_frame.to_numpy(dtype=float, na_value=0.0)).sum())
    print(f"\n{name} data quality:")
    print(f"  Missing values: {n_missing}")
    print(f"  Duplicate rows: {n_duplicate_rows}")
    print(f"  Infinite values: {n_infinite}")


def report_datetime_ordering(name: str, frame: pd.DataFrame) -> pd.Series:
    """Check strict ordering and report the gap pattern described in
    planning/door_handoff.md (mostly ~20ms steps, with large gaps between
    dense blocks of activity).
    """
    is_sorted = frame[PARSED_DATETIME_COLUMN].is_monotonic_increasing
    duplicate_timestamps = int(frame[PARSED_DATETIME_COLUMN].duplicated().sum())
    print(f"\n{name} Datetime ordering:")
    print(f"  Strictly increasing: {is_sorted}")
    print(f"  Duplicate Datetime values: {duplicate_timestamps}")

    gaps_ms = frame[PARSED_DATETIME_COLUMN].diff().dt.total_seconds() * 1000
    gaps_ms = gaps_ms.dropna()
    small_gaps = gaps_ms[gaps_ms <= config.CANDIDATE_GAP_THRESHOLD_MS]
    large_gaps = gaps_ms[gaps_ms > config.CANDIDATE_GAP_THRESHOLD_MS]
    print(f"  Row-to-row gaps <= {config.CANDIDATE_GAP_THRESHOLD_MS}ms: {len(small_gaps)} (most common: {small_gaps.mode().tolist()[:3]} ms)")
    print(f"  Row-to-row gaps > {config.CANDIDATE_GAP_THRESHOLD_MS}ms (candidate cycle boundaries): {len(large_gaps)}")
    print(
        "  NOTE: this OBSERVED gap pattern is not officially documented -- see "
        "planning/door_handoff.md Section 2 before relying on it for segmentation."
    )
    return gaps_ms


def report_segment_answers(answers: pd.DataFrame, train_stream: pd.DataFrame) -> None:
    print("\n--- Train_Segments_Answer.csv ---")
    print(f"Total labelled segments: {len(answers)}")

    print("\nClass distribution (status):")
    for label, count in answers["status"].value_counts().items():
        print(f"  {label:22s}: {count:4d}  ({100 * count / len(answers):5.1f}%)")

    print("\noperation distribution (informational only -- not predicted):")
    print(answers["operation"].value_counts().to_string())

    from src.door.preprocess import parse_door_datetime

    starts = answers["start_time"].map(parse_door_datetime)
    ends = answers["end_time"].map(parse_door_datetime)
    durations = (ends - starts).dt.total_seconds()
    print("\nSegment duration (seconds):")
    print(f"  min={durations.min():.2f}  mean={durations.mean():.2f}  max={durations.max():.2f}")
    print("n_rows per segment:")
    print(f"  min={answers['n_rows'].min()}  mean={answers['n_rows'].mean():.1f}  max={answers['n_rows'].max()}")

    sorted_starts = starts.sort_values().reset_index(drop=True)
    gaps_between_segments = (sorted_starts.shift(-1) - ends.loc[starts.sort_values().index].reset_index(drop=True)).dt.total_seconds()
    gaps_between_segments = gaps_between_segments.dropna()
    print("Gap between consecutive segments (seconds):")
    print(f"  min={gaps_between_segments.min():.2f}  mean={gaps_between_segments.mean():.2f}  max={gaps_between_segments.max():.2f}")

    print("\nValidity checks:")
    problems = validate_segment_table(answers, train_stream)
    if problems:
        print("  PROBLEMS FOUND:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("  No invalid or overlapping segments found. All boundaries fall inside Train.csv's own time range.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only inspection of the official Door dataset (no training).")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(config.DEFAULT_DATASET_DIR),
        help=f"Door dataset directory. Default: {config.DEFAULT_DATASET_DIR}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_dir).expanduser().resolve()

    print("=" * 78)
    print("DOOR -- DATA INSPECTION (read-only, no model training)")
    print("=" * 78)
    print(f"Dataset root: {dataset_root}")

    paths = find_dataset_files(dataset_root)
    print(f"Train:   {paths['train']}")
    print(f"Test:    {paths['test']}")
    print(f"Answers: {paths['answers']}")

    train = load_door_csv(paths["train"])
    test = load_door_csv(paths["test"])
    answers = pd.read_csv(paths["answers"])

    report_schema("Train.csv", train)
    report_schema("Test.csv", test)
    report_schema("Train_Segments_Answer.csv", answers)

    print("\n--- Train vs. Test schema comparison ---")
    train_cols = [c for c in train.columns if c != PARSED_DATETIME_COLUMN]
    test_cols = [c for c in test.columns if c != PARSED_DATETIME_COLUMN]
    if train_cols == test_cols:
        print("Train and Test have identical columns (confirmed).")
    else:
        print(f"WARNING: schema mismatch. Train-only: {set(train_cols) - set(test_cols)}; Test-only: {set(test_cols) - set(train_cols)}")

    report_data_quality("Train.csv", train)
    report_data_quality("Test.csv", test)

    report_datetime_ordering("Train.csv", train)
    report_datetime_ordering("Test.csv", test)

    missing_answer_columns = set(config.SEGMENT_ANSWER_COLUMNS) - set(answers.columns)
    if missing_answer_columns:
        raise ValueError(f"Train_Segments_Answer.csv is missing expected column(s): {sorted(missing_answer_columns)}")
    report_segment_answers(answers, train)

    print("\n" + "=" * 78)
    print("Inspection complete. No files were modified. No model was trained.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
