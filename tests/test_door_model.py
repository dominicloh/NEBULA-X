"""Door model CV, matching, score, and artifact tests."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.door_fixture import synthetic_stream  # noqa: E402
from src.door import config  # noqa: E402
from src.door.features import FEATURE_NAMES  # noqa: E402
from src.door.model import build_candidate_models, load_door_pipeline, save_door_pipeline  # noqa: E402
from src.door.train import load_training_data, official_iou_weighted_f1  # noqa: E402


class TestDoorModel(unittest.TestCase):
    def test_learned_scaler_is_inside_cv_estimator(self):
        models = build_candidate_models()
        logistic = models["LogisticRegression"]
        self.assertIsInstance(logistic, Pipeline)
        self.assertEqual(list(logistic.named_steps), ["scaler", "clf"])
        self.assertFalse(hasattr(logistic.named_steps["scaler"], "mean_"))
        self.assertEqual(models["RandomForest"].random_state, 42)

    def test_artifact_round_trip_and_reproducible_predictions(self):
        X = pd.DataFrame(np.arange(8 * len(FEATURE_NAMES)).reshape(8, -1), columns=FEATURE_NAMES)
        y = np.array(["Normal", "Abnormal resistance"] * 4)
        first = build_candidate_models()["RandomForest"].fit(X, y)
        second = build_candidate_models()["RandomForest"].fit(X, y)
        self.assertEqual(first.predict(X).tolist(), second.predict(X).tolist())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "door.joblib"
            save_door_pipeline(path, {"pipeline": first, "feature_names": list(FEATURE_NAMES)})
            loaded = load_door_pipeline(path)
            self.assertEqual(loaded["feature_names"], list(FEATURE_NAMES))
            self.assertEqual(loaded["pipeline"].predict(X).tolist(), first.predict(X).tolist())

    def test_detected_train_boundaries_must_match_answers_exactly(self):
        frame = synthetic_stream().drop(columns="_datetime_parsed")
        times = frame["Datetime"].tolist()
        answers = pd.DataFrame({
            "segment_id": ["a", "b"], "start_time": [times[0], times[2]],
            "end_time": [times[1], times[3]], "operation": ["Open", "Close"],
            "status": list(config.VALID_DOOR_LABELS), "n_rows": [2, 2],
        })
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame.to_csv(folder / "Train.csv", index=False)
            answers.to_csv(folder / "Train_Segments_Answer.csv", index=False)
            X, labels = load_training_data(folder, verbose=False)
            self.assertEqual(X.shape, (2, 29))
            self.assertEqual(labels.tolist(), list(config.VALID_DOOR_LABELS))
            answers.loc[0, "end_time"] = times[2]
            answers.to_csv(folder / "Train_Segments_Answer.csv", index=False)
            with self.assertRaises(ValueError):
                load_training_data(folder, verbose=False)

    def test_official_iou_score_rejects_wrong_label_and_rewards_overlap(self):
        times = synthetic_stream()["Datetime"].tolist()
        truth = pd.DataFrame({"start_time": [times[0]], "end_time": [times[1]], "status": ["Normal"]})
        found = pd.DataFrame({"start_time": [times[0]], "end_time": [times[1]], "prediction": ["Normal"]})
        self.assertEqual(official_iou_weighted_f1(found, truth), 1.0)
        found.loc[0, "prediction"] = "Abnormal resistance"
        self.assertEqual(official_iou_weighted_f1(found, truth), 0.0)


if __name__ == "__main__":
    unittest.main()
