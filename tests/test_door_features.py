"""Synthetic feature extraction and leakage guards."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.door_fixture import synthetic_stream  # noqa: E402
from src.door.features import FEATURE_NAMES, extract_cycle_features  # noqa: E402
from src.door.segment import detect_cycles  # noqa: E402


class TestDoorFeatures(unittest.TestCase):
    def test_one_row_per_cycle_deterministic_order_and_no_leakage(self):
        stream = synthetic_stream()
        segments = detect_cycles(stream)
        first = extract_cycle_features(stream, segments)
        second = extract_cycle_features(stream, segments)
        self.assertEqual(first.shape, (2, 29))
        self.assertEqual(first.columns.tolist(), list(FEATURE_NAMES))
        self.assertTrue(first.equals(second))
        self.assertTrue(np.isfinite(first.to_numpy()).all())
        self.assertFalse(any(word in name.lower() for name in FEATURE_NAMES
                             for word in ("datetime", "timestamp", "status", "label", "segment_id", "n_rows")))
        modified = segments.assign(status=["Normal", "Abnormal resistance"], answer_row=[99, 100])
        self.assertTrue(extract_cycle_features(stream, modified).equals(first))

    def test_rejects_missing_and_infinite_sensor_values(self):
        for value in (np.nan, np.inf, -np.inf):
            stream = synthetic_stream()
            segments = detect_cycles(stream)
            stream["Motor current(mA)"] = stream["Motor current(mA)"].astype(float)
            stream.loc[0, "Motor current(mA)"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "missing or infinite"):
                extract_cycle_features(stream, segments)

    def test_rejects_missing_sensor_column(self):
        stream = synthetic_stream()
        segments = detect_cycles(stream)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            extract_cycle_features(stream.drop(columns="Door leaf position"), segments)


if __name__ == "__main__":
    unittest.main()
