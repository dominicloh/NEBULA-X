"""Evaluate Door cycle boundaries on Train, then optionally inspect Test.

Only this evaluation module reads Train_Segments_Answer.csv. The detection
function itself uses sensor-stream timestamps alone. Reports contain aggregate
metrics, never raw rows or boundary timestamps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402
from src.door.preprocess import PARSED_DATETIME_COLUMN, load_door_csv, parse_door_datetime_series  # noqa: E402
from src.door.segment import detect_cycles, validate_segment_table  # noqa: E402


def evaluate_train_segments(detected: pd.DataFrame, official: pd.DataFrame) -> dict:
    """Match positive-IoU intervals one-to-one and summarize boundary error."""
    for name, table in (("detected", detected), ("official", official)):
        missing = {"start_time", "end_time"} - set(table.columns)
        if missing:
            raise ValueError(f"{name} segments missing columns: {sorted(missing)}")
        if table.empty:
            raise ValueError(f"{name} segments are empty")

    def bounds(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        starts = parse_door_datetime_series(table["start_time"].astype(str)).reset_index(drop=True)
        ends = parse_door_datetime_series(table["end_time"].astype(str)).reset_index(drop=True)
        if starts.isna().any() or ends.isna().any() or starts.ge(ends).any():
            raise ValueError("Segment boundaries must be valid and start before end")
        return starts, ends

    detected_start, detected_end = bounds(detected)
    official_start, official_end = bounds(official)
    iou = np.zeros((len(detected), len(official)), dtype=float)
    for i in range(len(detected)):
        for j in range(len(official)):
            intersection = max(
                0.0,
                (min(detected_end[i], official_end[j]) - max(detected_start[i], official_start[j])).total_seconds(),
            )
            if intersection:
                detected_duration = (detected_end[i] - detected_start[i]).total_seconds()
                official_duration = (official_end[j] - official_start[j]).total_seconds()
                iou[i, j] = intersection / (detected_duration + official_duration - intersection)

    predicted_indices, official_indices = linear_sum_assignment(iou, maximize=True)
    matches = [
        (int(i), int(j), float(iou[i, j]))
        for i, j in zip(predicted_indices, official_indices)
        if iou[i, j] > 0
    ]
    scores = np.array([score for _, _, score in matches], dtype=float)
    start_errors = [
        abs((detected_start[i] - official_start[j]).total_seconds() * 1000)
        for i, j, _ in matches
    ]
    end_errors = [
        abs((detected_end[i] - official_end[j]).total_seconds() * 1000)
        for i, j, _ in matches
    ]
    chronological_pair_overlap = None
    if len(detected) == len(official):
        d_order = detected_start.argsort().to_numpy()
        o_order = official_start.argsort().to_numpy()
        chronological_pair_overlap = bool(
            all(iou[int(i), int(j)] > 0 for i, j in zip(d_order, o_order))
        )
    return {
        "official_cycles": len(official),
        "detected_cycles": len(detected),
        "matched_cycles": len(matches),
        "missing_official_cycles": len(official) - len(matches),
        "extra_detected_cycles": len(detected) - len(matches),
        "mean_iou": float(scores.mean()) if len(scores) else None,
        "median_iou": float(np.median(scores)) if len(scores) else None,
        "minimum_iou": float(scores.min()) if len(scores) else None,
        "iou_at_least_0_50": int(np.sum(scores >= 0.50)),
        "iou_at_least_0_75": int(np.sum(scores >= 0.75)),
        "iou_at_least_0_90": int(np.sum(scores >= 0.90)),
        "mean_absolute_start_boundary_error_ms": float(np.mean(start_errors)) if start_errors else None,
        "mean_absolute_end_boundary_error_ms": float(np.mean(end_errors)) if end_errors else None,
        "chronological_pair_overlap": chronological_pair_overlap,
    }


def summarize_test_segments(detected: pd.DataFrame, stream: pd.DataFrame) -> dict:
    """Check Test boundaries without using or inventing Test labels."""
    problems = validate_segment_table(detected, stream)
    starts = parse_door_datetime_series(detected["start_time"].astype(str))
    ends = parse_door_datetime_series(detected["end_time"].astype(str))
    if not starts.is_monotonic_increasing:
        problems.append("Detected Test cycles are not chronological")
    if starts.min() < stream[PARSED_DATETIME_COLUMN].min() or ends.max() > stream[PARSED_DATETIME_COLUMN].max():
        problems.append("Detected Test boundaries fall outside the Test stream")
    return {"detected_cycles": len(detected), "boundaries_valid": not problems, "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--test", type=Path, help="Run only after Train validation passes")
    parser.add_argument("--output", type=Path, help="Optional aggregate-only JSON report")
    args = parser.parse_args()

    train = load_door_csv(args.train)
    detected = detect_cycles(train)
    official = pd.read_csv(args.answers, dtype={"start_time": str, "end_time": str})
    report = {"train": evaluate_train_segments(detected, official)}
    train_metrics = report["train"]
    validated = (
        train_metrics["matched_cycles"] == train_metrics["official_cycles"]
        == train_metrics["detected_cycles"]
        and train_metrics["mean_iou"] is not None
        and train_metrics["mean_iou"] > 0.90
        and train_metrics["chronological_pair_overlap"] is True
    )
    report["train_validated_for_test_run"] = validated
    if args.test:
        if not validated:
            raise ValueError("Train segmentation has not met the all-matched, mean-IoU > 0.90 validation gate; Test was not run")
        test = load_door_csv(args.test)
        report["test"] = summarize_test_segments(detect_cycles(test), test)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
