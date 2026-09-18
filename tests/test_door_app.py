"""Door Streamlit upload and readiness checks without a browser session."""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.door_view as door_view  # noqa: E402
import app.rail_view as rail_view  # noqa: E402
from tests.door_fixture import synthetic_stream  # noqa: E402
from src.door import config  # noqa: E402


class TestDoorApp(unittest.TestCase):
    def test_imports_and_no_upload(self):
        self.assertTrue(callable(rail_view.render_rail_page))
        fake = MagicMock()
        fake.file_uploader.return_value = None
        with patch.object(door_view, "st", fake):
            door_view.render_door_page()
        fake.info.assert_called_once()

    def test_malformed_upload_rejected(self):
        fake = MagicMock()
        fake.file_uploader.return_value = io.BytesIO(b"bad,col\n1,2\n")
        with patch.object(door_view, "st", fake):
            door_view.render_door_page()
        fake.error.assert_called_once()
        fake.download_button.assert_not_called()

    def test_valid_upload_predicts_and_offers_validated_download(self):
        if not config.MODEL_ARTIFACT_PATH.is_file():
            self.skipTest("final Door model unavailable")
        fake = MagicMock()
        source = io.BytesIO(synthetic_stream()[list(config.EXPECTED_COLUMNS)].to_csv(index=False).encode())
        source.name = "synthetic.csv"
        fake.file_uploader.return_value = source
        fake.columns.return_value = [MagicMock() for _ in range(4)]
        fake.selectbox.return_value = 0
        with patch.object(door_view, "st", fake):
            door_view.render_door_page()
        fake.error.assert_not_called()
        self.assertEqual(fake.download_button.call_count, 2)
        official_call = fake.download_button.call_args_list[0]
        self.assertEqual(official_call.kwargs["file_name"], "door_predictions.csv")
        self.assertEqual(official_call.kwargs["data"].decode().splitlines()[0], "start_time,end_time,prediction")

    def test_readiness_requires_valid_prediction(self):
        stream = synthetic_stream()
        with patch.object(door_view, "predict_door_file", side_effect=ValueError("invalid model")):
            status = door_view._check_pipeline_status(stream)
        self.assertTrue(status["segmentation_ready"])
        self.assertFalse(status["pipeline_ready"])


if __name__ == "__main__":
    unittest.main()
