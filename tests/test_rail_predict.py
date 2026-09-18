#!/usr/bin/env python3
"""Rail Corrugation Stage 3 smoke test.

A small, dependency-free check (no pytest -- not in requirements.txt, and
this is simple enough not to need it) that the saved model artifact and the
prediction interface actually work together, using a few files whose true
label we already know from Train_Labels.csv (one per class). This is a
smoke test, not a performance evaluation: it does not measure accuracy, and
it must NEVER be pointed at the unlabelled Test/ files (see Stage 3 rule:
don't evaluate performance using the unlabelled test files).

Checks:
  1. The saved model artifact loads without error and has the expected keys.
  2. Feature order stays consistent: the feature extractor's own column
     order matches the column order the artifact was trained on.
  3. Every predicted label is one of the three allowed labels.
  4. The results DataFrame has the expected columns/dtypes and no nulls.

Run directly:
    python tests/test_rail_predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.validation import VALID_RAIL_LABELS  # noqa: E402
from src.rail_corrugation.features import extract_rail_features  # noqa: E402
from src.rail_corrugation.inspect_data import DEFAULT_DATASET_DIR  # noqa: E402
from src.rail_corrugation.predict import DEFAULT_MODEL_PATH, load_model_artifact, predict_rail_files  # noqa: E402

# Three Train files (never Test/ files) with known labels confirmed directly
# from Train_Labels.csv -- one example per class.
KNOWN_TRAIN_FILES = {
    "Train1.csv": "Normal",
    "Train2.csv": "Side II",
    "Train62.csv": "Side I",
}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def main() -> int:
    train_dir = Path(DEFAULT_DATASET_DIR) / "Train"
    missing_files = [name for name in KNOWN_TRAIN_FILES if not (train_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Smoke test needs these known Train files, not found under {train_dir}: {missing_files}"
        )

    # --- Check 1: model artifact loads correctly -----------------------
    artifact = load_model_artifact(DEFAULT_MODEL_PATH)
    _check("pipeline" in artifact and hasattr(artifact["pipeline"], "predict"), "model artifact loads and has a fitted pipeline")
    _check(isinstance(artifact["feature_names"], list) and len(artifact["feature_names"]) > 0, "model artifact stores a non-empty feature name list")
    _check(list(artifact["class_labels"]) == list(VALID_RAIL_LABELS), "model artifact's class labels match the official 3 classes")

    # --- Check 2: feature order stays consistent ------------------------
    sample_path = train_dir / "Train1.csv"
    sample_frame = pd.read_csv(sample_path)
    fresh_feature_order = list(extract_rail_features(sample_frame).index)
    _check(
        fresh_feature_order == artifact["feature_names"],
        "extract_rail_features() column order matches the order the saved model was trained on",
    )

    # --- Checks 3 & 4: predict on known files, check labels + schema ---
    file_paths = [train_dir / name for name in KNOWN_TRAIN_FILES]
    results = predict_rail_files(file_paths, artifact=artifact)

    expected_columns = {"file_id", "prediction", "confidence_normal", "confidence_side_i", "confidence_side_ii", "top_confidence"}
    _check(expected_columns.issubset(set(results.columns)), f"results DataFrame has the expected columns: {sorted(expected_columns)}")
    _check(len(results) == len(KNOWN_TRAIN_FILES), f"one prediction row per input file ({len(KNOWN_TRAIN_FILES)} expected)")
    _check(results[["file_id", "prediction"]].isna().sum().sum() == 0, "no missing file_id/prediction values")
    _check(set(results["file_id"]) == set(KNOWN_TRAIN_FILES), "file_id values exactly match the input filenames")

    invalid_labels = sorted(set(results["prediction"]) - set(VALID_RAIL_LABELS))
    _check(not invalid_labels, f"every predicted label is one of {VALID_RAIL_LABELS} (found invalid: {invalid_labels})")

    # Informational only (NOT a hard assertion): the model won't necessarily
    # get every training file right (Stage 2's own CV showed imperfect
    # recall, especially for Side I), so we report agreement with the known
    # label rather than requiring it.
    results_by_file = results.set_index("file_id")["prediction"]
    print("\nPredicted vs. known label (informational; training-file recall is not 100%, see model_experiment_log.md):")
    n_matching = 0
    for filename, true_label in KNOWN_TRAIN_FILES.items():
        predicted_label = results_by_file[filename]
        matched = "MATCH" if predicted_label == true_label else "different"
        n_matching += predicted_label == true_label
        print(f"  {filename}: true={true_label!r} predicted={predicted_label!r} ({matched})")
    print(f"  {n_matching}/{len(KNOWN_TRAIN_FILES)} matched the known label.")

    print("\nAll Stage 3 smoke test checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
