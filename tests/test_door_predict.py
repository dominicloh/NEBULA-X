"""Synthetic submission validation plus an explicit local real-data smoke test."""

import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.door_fixture import synthetic_stream  # noqa: E402
from src.door import config  # noqa: E402
from src.door.predict import predict_door_file  # noqa: E402
from src.door.validate_predictions import validate_door_predictions_file, validate_door_predictions_frame  # noqa: E402


class TestDoorPredict(unittest.TestCase):
    def test_missing_model_and_no_external_boundary_arguments(self):
        self.assertEqual(list(inspect.signature(predict_door_file).parameters), ["source", "model_path"])
        with self.assertRaisesRegex(FileNotFoundError, "not ready"):
            predict_door_file("unused.csv", model_path="missing-model.joblib")

    def test_schema_boundary_and_index_rejection(self):
        times = synthetic_stream()["Datetime"].tolist()
        expected = pd.DataFrame({"start_time": [times[0], times[2]], "end_time": [times[1], times[3]]})
        valid = expected.assign(prediction=["Normal", "Abnormal resistance"])
        validate_door_predictions_frame(valid, expected)
        for broken in (valid.assign(prediction=["bogus", "Normal"]),
                       valid.assign(**{"Unnamed: 0": [0, 1]}),
                       valid.assign(end_time=[times[1], times[2]])):
            with self.assertRaises(ValueError):
                validate_door_predictions_frame(broken, expected)
        with self.assertRaises(ValueError):
            validate_door_predictions_frame(valid.iloc[[1, 0]].reset_index(drop=True), expected)

    def test_real_test_has_38_automatic_predictions_and_official_schema(self):
        dataset_dir = config.DEFAULT_DATASET_DIR
        if not (dataset_dir / "Test.csv").is_file():
            dataset_dir = ROOT / "organiser-materials" / "PS3" / "02_Datasets" / "Door"
        if not (dataset_dir / "Test.csv").is_file() or not config.MODEL_ARTIFACT_PATH.is_file():
            self.skipTest("local official Test.csv or final Door model unavailable")
        import src.door.predict as predictor
        from src.door.segment import detect_cycles
        with patch.object(predictor, "detect_cycles", wraps=detect_cycles) as detector:
            official, detailed = predict_door_file(dataset_dir / "Test.csv")
        detector.assert_called_once()
        self.assertEqual(official.shape, (38, 3))
        self.assertEqual(official.columns.tolist(), list(config.OFFICIAL_OUTPUT_COLUMNS))
        self.assertEqual(len(detailed), 38)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "door_predictions.csv"
            official.to_csv(path, index=False)
            validate_door_predictions_file(path, verbose=False, test_source=dataset_dir / "Test.csv")


if __name__ == "__main__":
    unittest.main()
