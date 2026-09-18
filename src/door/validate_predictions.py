#!/usr/bin/env python3
"""Door official prediction CSV validator.

Checks a `door_predictions.csv` against the official schema (Info Kit
Section 3 + `04_Example_Submission/door_predictions.csv`). Reuses the
shared `validate_door_predictions` (columns, no nulls, allowed labels) from
`src/common/validation.py`, then adds Door-specific checks that validator
doesn't cover: start-before-end ordering and duplicate segments.

This validator does NOT require the hidden ground-truth labels -- it only
checks the structure and content of the file you're about to submit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.validation import validate_door_predictions  # noqa: E402
from src.door import config  # noqa: E402
from src.door.preprocess import parse_door_datetime  # noqa: E402

DEFAULT_PREDICTIONS_PATH = config.PREDICTIONS_OUTPUT_PATH


def _parse_timestamp_column(predictions: pd.DataFrame, column: str) -> pd.Series:
    """Parse a start_time/end_time column, accepting either the native Door
    format or a standard ISO timestamp (the Info Kit allows both).
    """

    def _parse_one(value: str):
        try:
            return parse_door_datetime(str(value))
        except ValueError:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                raise ValueError(
                    f"Could not parse {column} value {value!r} as either the native Door format "
                    "(Year-Month-Date-Hour-Minute-Second-Millisecond) or a standard ISO timestamp."
                )
            return parsed

    return predictions[column].map(_parse_one)


def validate_door_predictions_file(predictions_path: Path, verbose: bool = True) -> None:
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Door predictions file not found: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    if predictions.columns[0].startswith("Unnamed"):
        raise ValueError(
            f"Door predictions file has an unnamed index column ({predictions.columns[0]!r}) as its "
            "first column -- likely written with to_csv(index=True). Re-save with index=False."
        )

    # 1. Generic schema: exact columns/order, no nulls, only allowed labels.
    validate_door_predictions(predictions)
    if verbose:
        print(f"[OK] Columns are exactly {list(predictions.columns)}, no missing values, only allowed labels.")

    # 2. start_time must be before end_time for every predicted segment.
    starts = _parse_timestamp_column(predictions, "start_time")
    ends = _parse_timestamp_column(predictions, "end_time")
    invalid_order = int((starts >= ends).sum())
    if invalid_order:
        raise ValueError(f"{invalid_order} predicted segment(s) have start_time >= end_time.")
    if verbose:
        print("[OK] Every predicted segment has start_time before end_time.")

    # 3. No duplicate segments (identical start_time + end_time predicted more than once).
    duplicate_mask = predictions.duplicated(subset=["start_time", "end_time"], keep=False)
    n_duplicates = int(duplicate_mask.sum())
    if n_duplicates:
        raise ValueError(f"{n_duplicates} row(s) share a duplicate (start_time, end_time) pair.")
    if verbose:
        print("[OK] No duplicate (start_time, end_time) segments.")

    # 4. Filename check (a warning, not a hard failure -- draft files during
    #    development may reasonably be named something else).
    if predictions_path.name != config.OFFICIAL_OUTPUT_FILENAME and verbose:
        print(
            f"WARNING: file is named {predictions_path.name!r}, not the official "
            f"{config.OFFICIAL_OUTPUT_FILENAME!r}. Rename it before final submission."
        )

    if verbose:
        print(f"\nAll Door submission checks passed for: {predictions_path}")
        print("NOTE: this validator does not and cannot check IoU-weighted F1 (the official metric) --")
        print("that requires the hidden ground-truth segments, which the team does not have.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the official Door door_predictions.csv.")
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS_PATH),
        help=f"Path to door_predictions.csv. Default: {DEFAULT_PREDICTIONS_PATH}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_door_predictions_file(Path(args.predictions))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nVALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)


__all__ = ["validate_door_predictions_file"]
