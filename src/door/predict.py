#!/usr/bin/env python3
"""Door inference: full pipeline structure for turning a continuous Door
stream into the official `door_predictions.csv`.

This module never fabricates a prediction. `predict_door_file()` checks,
in order:
  1. Does a trained classifier artifact exist? If not: clear "Door model is
     not ready" error. Never falls back to a random/default label.
  2. Segment the stream with the frozen Train-validated `detect_cycles` rule.
  3. Only once both exist: extract features for each detected cycle,
     predict a label, and build both the official output and a detailed
     evidence table for the app.

Usage (once a separately validated classifier artifact exists):
    python src/door/predict.py <path to Test.csv> [--model-path PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402
from src.door.features import extract_cycle_features  # noqa: E402
from src.door.model import load_door_pipeline  # noqa: E402
from src.door.preprocess import load_door_csv  # noqa: E402
from src.door.segment import detect_cycles  # noqa: E402
from src.door.validate_predictions import validate_door_predictions_file, validate_door_predictions_frame  # noqa: E402
from src.export_results import export_official_predictions  # noqa: E402


def predict_door_file(source, model_path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Door pipeline on one continuous stream (e.g. Test.csv).

    Returns (official_df, detailed_df):
      - official_df: exactly `config.OFFICIAL_OUTPUT_COLUMNS`
        (start_time, end_time, prediction) -- the competition submission shape.
      - detailed_df: official_df plus per-class confidence, for the
        Streamlit app's evidence view. NOT the official submission format.

    Raises a clear, specific error rather than ever returning a
    random/default label:
      - FileNotFoundError("Door model is not ready...") if no trained
        artifact exists at `model_path`.
      - ValueError if the stream cannot be safely segmented.

    GUARANTEE: this function has no parameter for externally-supplied
    segment boundaries and never reads Train_Segments_Answer.csv (or any
    other ground-truth file). The only source of segment boundaries for a
    real prediction is `detect_cycles(stream)` below -- there is exactly one
    call to it, and its return value is used directly. This is deliberate:
    `train.py` is allowed to use the official ground-truth segments for
    classifier evaluation (see its own docstring for why that's legitimate),
    but a live prediction on `source` must always reflect this function's
    OWN segmentation, never a shortcut through known answers. If this
    function is ever changed to accept a `segments=` argument or to read
    config.TRAIN_SEGMENTS_ANSWER_FILENAME, that guarantee is broken --
    tests/test_door_scaffold.py checks for exactly that regression.
    """
    model_path = Path(model_path) if model_path is not None else config.MODEL_ARTIFACT_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Door model is not ready: no trained artifact found at {model_path}. "
            "Train and save one first with: python src/door/train.py --finalize"
        )
    artifact = load_door_pipeline(model_path)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("feature_names"), list):
        raise ValueError("Door model artifact lacks its ordered feature schema.")
    if artifact.get("segmentation_threshold_ms") != config.SEGMENTATION_CONFIG["gap_threshold_ms"]:
        raise ValueError("Door model artifact segmentation threshold differs from frozen config.")
    if set(artifact.get("class_labels", [])) != set(config.VALID_DOOR_LABELS):
        raise ValueError("Door model artifact has unexpected class labels.")

    stream = load_door_csv(source)
    # Boundaries come only from the sensor stream, never the answer file.
    segments = detect_cycles(stream)

    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    class_labels = artifact["class_labels"]

    features = extract_cycle_features(stream, segments)
    if list(features.columns) != feature_names:
        raise ValueError("Door feature schema/order differs from the frozen model artifact.")

    predictions = pipeline.predict(features)

    official_rows = {
        "start_time": segments["start_time"].tolist(),
        "end_time": segments["end_time"].tolist(),
        "prediction": list(predictions),
    }
    official_df = pd.DataFrame(official_rows, columns=list(config.OFFICIAL_OUTPUT_COLUMNS))
    validate_door_predictions_frame(official_df, expected_segments=segments)

    detailed_df = official_df.copy()
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        pipeline_classes = list(pipeline.classes_)
        for label in class_labels:
            key = f"confidence_{label.lower().replace(' ', '_')}"
            if label in pipeline_classes:
                detailed_df[key] = probabilities[:, pipeline_classes.index(label)]

    return official_df, detailed_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Door pipeline on a continuous stream (e.g. Test.csv).")
    parser.add_argument("source", help="Path to the Door CSV to process (e.g. the official Test.csv).")
    parser.add_argument("--model-path", default=str(config.MODEL_ARTIFACT_PATH), help="Path to the trained Door model artifact.")
    parser.add_argument("--output", default=str(config.PREDICTIONS_OUTPUT_PATH), help="Where to write the official door_predictions.csv.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official_df, detailed_df = predict_door_file(args.source, model_path=args.model_path)

    output_path = Path(args.output)
    config_dict = {
        "required_columns": list(config.OFFICIAL_OUTPUT_COLUMNS),
        "output_columns": list(config.OFFICIAL_OUTPUT_COLUMNS),
        "subsystem": "door",
    }
    export_official_predictions(official_df, output_path, config_dict)
    validate_door_predictions_file(output_path, verbose=False, test_source=Path(args.source))
    diagnostics_dir = config.PROJECT_ROOT / "output" / "door" / "classification"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(diagnostics_dir / "test_diagnostics.csv", index=False)
    (diagnostics_dir / "test_prediction_summary.json").write_text(
        json.dumps({"detected_cycles": len(official_df),
                    "class_counts": official_df["prediction"].value_counts().to_dict(),
                    "note": "Test labels are hidden; this is a prediction distribution, not accuracy."}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(official_df)} predicted segment(s) to: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except NotImplementedError as exc:
        print(f"\nNOT READY YET: {exc}", file=sys.stderr)
        raise SystemExit(1)


__all__ = ["predict_door_file"]
