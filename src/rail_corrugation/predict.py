#!/usr/bin/env python3
"""Rail Corrugation Stage 3: reusable inference interface.

This module loads the ALREADY-FITTED pipeline saved by
`train.py --finalize` (see `models/rail_corrugation_model.joblib`) and uses
it to predict Normal / Side I / Side II for new Rail Corrugation CSV files.
It never fits or refits anything -- training happens exactly once, in
train.py, on the 272 labelled Train files.

Two callers share this one code path, so predictions are identical no
matter where they come from:
  - The CLI at the bottom of this file (used to generate the official
    `predictions/rail_predictions.csv` for all 68 Test files).
  - The Streamlit app, which can call `predict_rail_files()` directly with
    a list of uploaded file objects for the interactive dashboard.

WHY the returned DataFrame has more columns than the official submission
------------------------------------------------------------------------
`predict_rail_files()` returns `file_id`, `prediction`, one confidence
column per class, and `top_confidence` -- useful for a dashboard, where a
user benefits from seeing how confident the model was. The official
competition CSV must contain ONLY `file_id` and `prediction` (confirmed from
`04_Example_Submission/rail_predictions.csv`), so `export_official_predictions_csv`
below always drops the confidence columns before writing that file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.validation import VALID_RAIL_LABELS  # noqa: E402
from src.export_results import export_official_predictions  # noqa: E402
from src.rail_corrugation.features import SAMPLING_FREQUENCY_HZ, extract_rail_features  # noqa: E402
from src.rail_corrugation.inspect_data import list_csv_files  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "rail_corrugation_model.joblib"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "predictions" / "rail_predictions.csv"

# The official schema, confirmed from organiser-materials/PS3/04_Example_Submission/
# rail_predictions.csv and the PS3 spec's Deliverables table: exactly these two
# columns, in this order, no index column, no confidence column.
OFFICIAL_RAIL_COLUMNS = ["file_id", "prediction"]


def load_model_artifact(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Load the fitted pipeline + metadata saved by train.py --finalize.

    Never fits anything -- this only deserialises an already-trained model.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Rail Corrugation model artifact not found at: {model_path}\n"
            "Train and save it first with:\n"
            "  python src/rail_corrugation/train.py --finalize"
        )
    artifact = joblib.load(model_path)
    required_keys = {"pipeline", "feature_names", "class_labels"}
    missing_keys = required_keys - set(artifact)
    if missing_keys:
        raise ValueError(
            f"Rail Corrugation model artifact at {model_path} is missing expected key(s): "
            f"{sorted(missing_keys)}. It may have been saved by an older/incompatible version "
            "of train.py."
        )
    return artifact


def _resolve_input_sources(inputs) -> list:
    """Normalise the accepted `inputs` shapes into a flat list of "sources",
    each either a Path (for a CSV on disk) or a file-like object (e.g. a
    Streamlit UploadedFile) that pandas can read directly.

    Accepts: a single CSV path, a directory path (all *.csv inside, natural
    sorted), a list/tuple mixing paths and/or file-like objects, or a single
    file-like object.
    """
    if isinstance(inputs, (str, Path)):
        path = Path(inputs)
        if path.is_dir():
            return list(list_csv_files(path))  # natural-sorted, raises if empty
        if path.is_file():
            return [path]
        raise FileNotFoundError(f"Rail Corrugation input path does not exist: {path}")

    if isinstance(inputs, (list, tuple)):
        if not inputs:
            raise ValueError("No Rail Corrugation input files were provided.")
        return list(inputs)

    # A single file-like object (e.g. one Streamlit UploadedFile).
    return [inputs]


def _load_named_frame(source) -> tuple:
    """Read one input source into (file_id, DataFrame). Loads exactly one
    file at a time -- callers should discard the frame before the next call
    so large batches never hold every file in memory together.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Rail Corrugation input file not found: {path}")
        file_id = path.name
    else:
        # File-like object: must expose a filename via '.name' (Streamlit's
        # UploadedFile does). Strip any path separators a browser might send.
        raw_name = getattr(source, "name", None)
        if not raw_name:
            raise ValueError(
                "Uploaded Rail Corrugation file objects must have a non-empty '.name' attribute "
                "so the original filename can be preserved as file_id."
            )
        file_id = Path(raw_name).name
        path = source

    if not file_id.lower().endswith(".csv"):
        raise ValueError(f"Rail Corrugation expects a .csv file; received: {file_id!r}")

    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pandas raises several distinct error types on bad CSVs
        raise ValueError(f"Rail Corrugation file {file_id!r} could not be read as CSV: {exc}") from exc

    return file_id, frame


def predict_rail_files(inputs, model_path: Path = DEFAULT_MODEL_PATH, artifact: dict | None = None) -> pd.DataFrame:
    """Predict Normal / Side I / Side II for one CSV, many CSVs, or a directory of CSVs.

    `inputs` may be:
      - a path (str/Path) to one CSV file
      - a path (str/Path) to a directory of CSV files
      - a list/tuple of paths and/or file-like objects (e.g. Streamlit
        UploadedFile instances, which have a '.name' and are directly
        readable by pandas)
      - a single file-like object

    Returns a DataFrame with one row per input file:
      file_id, prediction, confidence_normal, confidence_side_i,
      confidence_side_ii, top_confidence

    This dashboard-ready shape is NOT the official submission format --
    use `export_official_predictions_csv` to write the official
    file_id/prediction-only CSV.

    Malformed files are rejected immediately with a clear, per-file error
    message (naming the offending file) rather than silently skipped, so a
    batch export can never come back with quietly-missing predictions.
    """
    artifact = artifact or load_model_artifact(model_path)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    class_labels = artifact["class_labels"]
    sampling_frequency_hz = artifact.get("sampling_frequency_hz", SAMPLING_FREQUENCY_HZ)

    sources = _resolve_input_sources(inputs)

    rows = []
    seen_file_ids = set()
    for source in sources:
        file_id, frame = _load_named_frame(source)

        if file_id in seen_file_ids:
            raise ValueError(f"Duplicate file supplied for prediction: {file_id!r}")
        seen_file_ids.add(file_id)

        try:
            feature_row = extract_rail_features(frame, sampling_frequency_hz)
        except (ValueError, TypeError) as exc:
            # extract_rail_features already validates the column schema and
            # raises a clear message; we just attach which file it was.
            raise ValueError(f"Rail Corrugation file {file_id!r} was rejected: {exc}") from exc
        finally:
            del frame  # one file's raw data in memory at a time

        # Reorder/align to the EXACT feature order the model was trained on.
        # A silent column mismatch here would scramble every prediction.
        feature_row = feature_row.reindex(feature_names)
        if feature_row.isna().any():
            missing = feature_row[feature_row.isna()].index.tolist()
            raise ValueError(
                f"Rail Corrugation file {file_id!r} did not produce all features the model "
                f"expects. Missing: {missing}"
            )
        # Keep this as a one-row DataFrame with the named feature columns
        # (not a bare numpy array) -- the pipeline's StandardScaler was fit
        # on a DataFrame with these exact column names, so passing named
        # columns back avoids sklearn's "X does not have valid feature
        # names" warning and keeps column identity explicit end to end.
        feature_frame = feature_row.to_frame().T
        feature_frame.index = [0]

        predicted_label = pipeline.predict(feature_frame)[0]
        if predicted_label not in class_labels:
            raise ValueError(
                f"Model produced label {predicted_label!r} for {file_id!r}, which is outside "
                f"the allowed set {class_labels}."
            )

        row = {"file_id": file_id, "prediction": predicted_label}

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(feature_frame)[0]
            # Map back to class labels using the pipeline's own learned
            # class order (classes_), not our constant, in case they differ.
            proba_by_class = dict(zip(pipeline.classes_, probabilities))
            for label in class_labels:
                key = f"confidence_{label.lower().replace(' ', '_')}"
                row[key] = float(proba_by_class.get(label, float("nan")))
            row["top_confidence"] = float(max(probabilities))

        rows.append(row)

    return pd.DataFrame(rows)


def export_official_predictions_csv(results: pd.DataFrame, output_path: Path) -> Path:
    """Write the OFFICIAL rail_predictions.csv: exactly file_id,prediction,
    no index column, no confidence columns, validated against the shared
    Rail schema rules before being written to disk for real.
    """
    config = {
        "required_columns": OFFICIAL_RAIL_COLUMNS,
        "output_columns": OFFICIAL_RAIL_COLUMNS,
        "subsystem": "rail",
    }
    return export_official_predictions(results, output_path, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Rail Corrugation class (Normal/Side I/Side II) for one or more CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more Rail Corrugation CSV files, or a single directory containing CSV files.",
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to the fitted model artifact.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_PREDICTIONS_PATH),
        help=f"Where to write the official file_id,prediction CSV. Default: {DEFAULT_PREDICTIONS_PATH}",
    )
    parser.add_argument(
        "--with-confidence-csv",
        default=None,
        help="Optional extra path to also save file_id/prediction/per-class confidence (NOT the official schema).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # A single directory argument (nargs="+" gives a 1-item list) should be
    # treated as "predict everything in this directory", not "predict a file
    # named after the directory".
    inputs = args.inputs[0] if len(args.inputs) == 1 else args.inputs

    artifact = load_model_artifact(Path(args.model_path))
    print(f"Loaded model: {artifact.get('model_name', '?')} (trained on {artifact.get('n_training_files', '?')} files)")

    results = predict_rail_files(inputs, artifact=artifact)
    print(f"\nPredicted {len(results)} file(s).")
    print("Class distribution:")
    print(results["prediction"].value_counts().reindex(VALID_RAIL_LABELS, fill_value=0).to_string())

    output_path = Path(args.output)
    export_official_predictions_csv(results, output_path)
    print(f"\nSaved official predictions to: {output_path}")

    if args.with_confidence_csv:
        confidence_path = Path(args.with_confidence_csv)
        confidence_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(confidence_path, index=False)
        print(f"Saved dashboard-ready predictions (with confidence) to: {confidence_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


__all__ = [
    "DEFAULT_MODEL_PATH",
    "OFFICIAL_RAIL_COLUMNS",
    "load_model_artifact",
    "predict_rail_files",
    "export_official_predictions_csv",
]
