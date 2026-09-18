"""Synthetic Door segmentation and one-to-one evaluation tests."""

from __future__ import annotations

import sys
import ast
import inspect
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.door.evaluate_segmentation import evaluate_train_segments, summarize_test_segments  # noqa: E402
from src.door import config as door_config  # noqa: E402
from src.door.predict import predict_door_file  # noqa: E402
from src.door.preprocess import PARSED_DATETIME_COLUMN, load_door_csv, parse_door_datetime_series  # noqa: E402
from src.door.segment import detect_cycles, validate_segment_table  # noqa: E402


def stream(times: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"Datetime": times, "Motor current(mA)": range(len(times))})
    frame[PARSED_DATETIME_COLUMN] = parse_door_datetime_series(frame["Datetime"])
    return frame


def seconds(*values: int) -> list[str]:
    return [f"2023-7-5-0-0-{value}-0" for value in values]


def csv_stream(times: list[str]) -> io.StringIO:
    frame = pd.DataFrame({column: [0] * len(times) for column in door_config.EXPECTED_COLUMNS})
    frame["Datetime"] = times
    return io.StringIO(frame.to_csv(index=False))


class TestDoorLoader(unittest.TestCase):
    def test_rejects_unsorted_source_rows(self):
        with self.assertRaisesRegex(ValueError, "source row order; unsorted input"):
            load_door_csv(csv_stream(seconds(1, 0, 2)))

    def test_preserves_valid_source_row_order_and_timestamp_text(self):
        times = ["2023-7-5-0-0-0-0", "2023-7-5-0-0-0-20", "2023-7-5-0-0-1-0"]
        loaded = load_door_csv(csv_stream(times))
        self.assertEqual(loaded["Datetime"].tolist(), times)
        self.assertEqual(loaded.index.tolist(), list(range(len(times))))
        self.assertEqual(loaded[PARSED_DATETIME_COLUMN].tolist(), parse_door_datetime_series(pd.Series(times)).tolist())


class TestSegmentation(unittest.TestCase):
    def test_single_cycle_and_no_large_gaps(self):
        frame = stream(["2023-7-5-0-0-0-0", "2023-7-5-0-0-0-20", "2023-7-5-0-0-0-40"])
        result = detect_cycles(frame, {"gap_threshold_ms": 100})
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "start_time"], frame.loc[0, "Datetime"])
        self.assertEqual(result.loc[0, "end_time"], frame.loc[2, "Datetime"])
        self.assertEqual(list(result.columns), ["segment_id", "start_time", "end_time"])

    def test_multiple_cycles_exact_boundaries_chronology_and_no_overlap(self):
        times = ["2023-7-5-0-0-0-0", "2023-7-5-0-0-0-20",
                 "2023-7-5-0-0-1-0", "2023-7-5-0-0-1-20",
                 "2023-7-5-0-0-2-0", "2023-7-5-0-0-2-20"]
        frame = stream(times)
        result = detect_cycles(frame, {"gap_threshold_ms": 100})
        self.assertEqual(result["start_time"].tolist(), [times[0], times[2], times[4]])
        self.assertEqual(result["end_time"].tolist(), [times[1], times[3], times[5]])
        self.assertEqual(validate_segment_table(result, frame), [])
        self.assertEqual(detect_cycles(frame, {"gap_threshold_ms": 100}).to_json(), result.to_json())
        self.assertEqual(frame["Datetime"].tolist(), times)

    def test_strict_threshold_boundary(self):
        frame = stream(["2023-7-5-0-0-0-0", "2023-7-5-0-0-0-20",
                        "2023-7-5-0-0-0-120", "2023-7-5-0-0-0-140"])
        self.assertEqual(len(detect_cycles(frame)), 1)
        self.assertEqual(len(detect_cycles(frame, {"gap_threshold_ms": 99})), 2)

    def test_malformed_original_timestamp(self):
        frame = stream(seconds(0, 1))
        frame.loc[0, "Datetime"] = "bad"
        with self.assertRaisesRegex(ValueError, "Malformed"):
            detect_cycles(frame, {"gap_threshold_ms": 100})

    def test_missing_parsed_timestamp(self):
        frame = stream(seconds(0, 1)).drop(columns=PARSED_DATETIME_COLUMN)
        with self.assertRaisesRegex(ValueError, PARSED_DATETIME_COLUMN):
            detect_cycles(frame, {"gap_threshold_ms": 100})

    def test_stale_parsed_timestamp(self):
        frame = stream(seconds(0, 1))
        frame.loc[1, PARSED_DATETIME_COLUMN] = frame.loc[0, PARSED_DATETIME_COLUMN]
        with self.assertRaisesRegex(ValueError, "do not match"):
            detect_cycles(frame, {"gap_threshold_ms": 100})

    def test_unsorted_and_duplicate_timestamps(self):
        frame = stream(seconds(1, 0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            detect_cycles(frame, {"gap_threshold_ms": 100})
        frame = stream(seconds(0, 0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            detect_cycles(frame, {"gap_threshold_ms": 100})

    def test_empty_and_single_sample(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            detect_cycles(stream([]), {"gap_threshold_ms": 100})
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            detect_cycles(stream(seconds(0)), {"gap_threshold_ms": 100})

    def test_no_answer_file_dependency(self):
        frame = stream(["2023-7-5-0-0-0-0", "2023-7-5-0-0-0-20"])
        with patch("builtins.open", side_effect=AssertionError("unexpected file access")):
            self.assertEqual(len(detect_cycles(frame, {"gap_threshold_ms": 100})), 1)

    def test_invalid_threshold(self):
        frame = stream(seconds(0, 1))
        for value in (0, -1, float("nan"), float("inf"), float("-inf"), "100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                detect_cycles(frame, {"gap_threshold_ms": value})

    def test_prediction_does_not_expose_or_pass_threshold_override(self):
        self.assertEqual(list(inspect.signature(predict_door_file).parameters), ["source", "model_path"])
        tree = ast.parse(inspect.getsource(predict_door_file))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "detect_cycles"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 1)
        self.assertEqual(calls[0].keywords, [])

    def test_streamlit_does_not_pass_threshold_override(self):
        tree = ast.parse((ROOT / "app" / "door_view.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "detect_cycles"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 1)
        self.assertEqual(calls[0].keywords, [])


class TestEvaluation(unittest.TestCase):
    def test_official_train_and_test_metrics_remain_unchanged(self):
        dataset_dir = door_config.DEFAULT_DATASET_DIR
        if not (dataset_dir / "Train.csv").is_file():
            dataset_dir = ROOT / "organiser-materials" / "PS3" / "02_Datasets" / "Door"
        if not all((dataset_dir / name).is_file() for name in
                   ("Train.csv", "Test.csv", "Train_Segments_Answer.csv")):
            self.skipTest("official Door datasets are not available in this checkout")

        train = load_door_csv(dataset_dir / "Train.csv")
        test = load_door_csv(dataset_dir / "Test.csv")
        detected_train = detect_cycles(train)
        official = pd.read_csv(dataset_dir / "Train_Segments_Answer.csv", dtype=str)
        metrics = evaluate_train_segments(detected_train, official)
        self.assertEqual((metrics["official_cycles"], metrics["detected_cycles"], metrics["matched_cycles"]),
                         (110, 110, 110))
        self.assertEqual((metrics["missing_official_cycles"], metrics["extra_detected_cycles"]), (0, 0))
        self.assertEqual((metrics["mean_iou"], metrics["median_iou"], metrics["minimum_iou"]), (1.0, 1.0, 1.0))
        self.assertEqual((metrics["mean_absolute_start_boundary_error_ms"],
                          metrics["mean_absolute_end_boundary_error_ms"]), (0.0, 0.0))
        self.assertEqual(summarize_test_segments(detect_cycles(test), test),
                         {"detected_cycles": 38, "boundaries_valid": True, "problems": []})

    def test_one_to_one_matching_prevents_double_credit(self):
        detected = pd.DataFrame({"start_time": seconds(0), "end_time": seconds(3)})
        official = pd.DataFrame({"start_time": seconds(0, 2), "end_time": seconds(1, 3)})
        result = evaluate_train_segments(detected, official)
        self.assertEqual(result["matched_cycles"], 1)
        self.assertEqual(result["missing_official_cycles"], 1)
        self.assertEqual(result["extra_detected_cycles"], 0)

    def test_extra_detected_cycle(self):
        detected = pd.DataFrame({"start_time": seconds(0, 2), "end_time": seconds(1, 3)})
        official = pd.DataFrame({"start_time": seconds(0), "end_time": seconds(1)})
        result = evaluate_train_segments(detected, official)
        self.assertEqual(result["matched_cycles"], 1)
        self.assertEqual(result["extra_detected_cycles"], 1)
        self.assertEqual(result["mean_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
