#!/usr/bin/env python3
"""Door inference: full pipeline structure for turning a continuous Door
stream into the official `door_predictions.csv`.

This module never fabricates a prediction. `predict_door_file()` checks,
in order:
  1. Does a trained classifier artifact exist? If not: clear "Door model is
     not ready" error. Never falls back to a random/default label.
  2. Can the stream be segmented? This calls `segment.py::detect_cycles`,
     which currently always raises NotImplementedError -- that error is
     allowed to surface as-is (it already explains what to implement).
  3. Only once both exist: extract features for each detected cycle,
     predict a label, and build both the official output and a detailed
     evidence table for the app.

Usage (once detect_cycles() and a trained model both exist):
    python src/door/predict.py <path to Test.csv> [--model-path PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
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
      - NotImplementedError (from detect_cycles) if segmentation isn't
        implemented yet.

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

    stream = load_door_csv(source)
    # This is expected to raise NotImplementedError until detect_cycles()
    # is implemented -- that error already explains what to do next, so we
    # deliberately do not catch or reword it here.
    segments = detect_cycles(stream)

    pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact
    feature_names = artifact["feature_names"] if isinstance(artifact, dict) else None
    class_labels = artifact["class_labels"] if isinstance(artifact, dict) else list(config.VALID_DOOR_LABELS)

    features = extract_cycle_features(stream, segments)
    if feature_names is not None:
        features = features.reindex(columns=feature_names)

    predictions = pipeline.predict(features)

    official_rows = {
        "start_time": segments["start_time"].tolist(),
        "end_time": segments["end_time"].tolist(),
        "prediction": list(predictions),
    }
    official_df = pd.DataFrame(official_rows, columns=list(config.OFFICIAL_OUTPUT_COLUMNS))

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
    official_df, _detailed_df = predict_door_file(args.source, model_path=args.model_path)

    output_path = Path(args.output)
    config_dict = {
        "required_columns": list(config.OFFICIAL_OUTPUT_COLUMNS),
        "output_columns": list(config.OFFICIAL_OUTPUT_COLUMNS),
        "subsystem": "door",
    }
    export_official_predictions(official_df, output_path, config_dict)
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
