#!/usr/bin/env python3
"""Validate Door submission schema and exact boundaries against the frozen detector."""

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
from src.door.preprocess import load_door_csv, parse_door_datetime  # noqa: E402
from src.door.segment import detect_cycles  # noqa: E402


def validate_door_predictions_frame(predictions: pd.DataFrame, expected_segments: pd.DataFrame | None = None) -> None:
    """Reject malformed, unordered, overlapping, or unexpected cycle predictions."""
    if any(str(column).startswith("Unnamed") for column in predictions.columns):
        raise ValueError("Door predictions contain an unnamed index column; save with index=False.")
    validate_door_predictions(predictions)
    try:
        starts = predictions["start_time"].map(lambda value: parse_door_datetime(str(value)))
        ends = predictions["end_time"].map(lambda value: parse_door_datetime(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Door predictions contain invalid official timestamp text: {exc}") from exc
    if (starts >= ends).any():
        raise ValueError("Door prediction start_time must precede end_time.")
    if not starts.is_monotonic_increasing or not ends.is_monotonic_increasing:
        raise ValueError("Door predictions must be in chronological order.")
    if (starts.iloc[1:].reset_index(drop=True) <= ends.iloc[:-1].reset_index(drop=True)).any():
        raise ValueError("Door prediction cycles overlap or share a boundary.")
    if predictions.duplicated(["start_time", "end_time"]).any():
        raise ValueError("Door predictions contain duplicate cycles.")
    if expected_segments is not None:
        if len(predictions) != len(expected_segments):
            raise ValueError(f"Door prediction count differs from detected cycles: {len(predictions)} versus {len(expected_segments)}.")
        for column in ("start_time", "end_time"):
            if predictions[column].astype(str).tolist() != expected_segments[column].astype(str).tolist():
                raise ValueError(f"Door prediction {column} values differ from detect_cycles boundaries.")


def validate_door_predictions_file(predictions_path: Path, verbose: bool = True,
                                   test_source: Path | None = None) -> None:
    """Validate the official file against Test.csv without hidden labels."""
    predictions_path = Path(predictions_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Door predictions file not found: {predictions_path}")
    predictions = pd.read_csv(predictions_path, dtype=str)
    validate_door_predictions_frame(predictions)
    if predictions_path.name != config.OFFICIAL_OUTPUT_FILENAME:
        raise ValueError(f"Official Door output filename must be {config.OFFICIAL_OUTPUT_FILENAME}.")
    if test_source is None:
        test_source = config.DEFAULT_DATASET_DIR / config.TEST_FILENAME
        if not test_source.is_file():
            test_source = PROJECT_ROOT / "organiser-materials" / "PS3" / "02_Datasets" / "Door" / config.TEST_FILENAME
    expected = detect_cycles(load_door_csv(test_source))
    validate_door_predictions_frame(predictions, expected_segments=expected)
    if verbose:
        print(f"[OK] {len(predictions)} ordered, non-overlapping Door predictions with exact detected Test boundaries.")
        print(f"[OK] Official filename, columns, timestamp text, labels, and no unnamed index: {predictions_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=config.PREDICTIONS_OUTPUT_PATH)
    parser.add_argument("--test", type=Path, help="Optional explicit Test.csv path for local validation")
    args = parser.parse_args()
    validate_door_predictions_file(args.predictions, test_source=args.test)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)


__all__ = ["validate_door_predictions_frame", "validate_door_predictions_file"]
