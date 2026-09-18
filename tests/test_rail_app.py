#!/usr/bin/env python3
"""Rail Corrugation Stage 4 app-integration checks.

Complements `tests/test_rail_predict.py` (Stage 3's model-artifact smoke
test -- model loads, feature order matches, labels valid) with checks
specific to the Streamlit app layer (`app/rail_view.py`) added in Stage 4,
plus a guard confirming the Rail integration didn't break Door.

No pytest (not a project dependency) -- plain assertions, like the rest of
this project's tests.

Run directly:
    python tests/test_rail_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.rail_view as rail_view  # noqa: E402
from src.common.validation import validate_rail_predictions  # noqa: E402
from src.rail_corrugation.inspect_data import DEFAULT_DATASET_DIR  # noqa: E402
from src.rail_corrugation.predict import load_model_artifact, predict_rail_files  # noqa: E402


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def check_valid_single_and_multi_file_prediction(artifact: dict, train_dir: Path) -> pd.DataFrame:
    files = ["Train1.csv", "Train2.csv", "Train62.csv"]

    single = predict_rail_files(train_dir / files[0], artifact=artifact)
    _check(len(single) == 1, "valid single-file prediction returns exactly 1 row")

    multi = predict_rail_files([train_dir / name for name in files], artifact=artifact)
    _check(len(multi) == len(files), "valid multi-file prediction returns one row per file")
    _check(set(multi["file_id"]) == set(files), "multi-file prediction preserves each filename as file_id")
    return multi


def check_official_output_schema(multi_results: pd.DataFrame) -> None:
    official = multi_results[["file_id", "prediction"]].copy()
    _check(list(official.columns) == ["file_id", "prediction"], "official output has exactly file_id,prediction, in order")
    validate_rail_predictions(official)
    print("[PASS] official output passes the shared Rail schema validator (labels, no nulls)")


def check_confidence_sums_to_one(multi_results: pd.DataFrame) -> None:
    sums = multi_results[["confidence_normal", "confidence_side_i", "confidence_side_ii"]].sum(axis=1)
    _check(bool((sums.sub(1.0).abs() < 1e-6).all()), "class confidence probabilities sum to ~1.0 for every row")


def check_model_loads_without_retraining(artifact: dict) -> None:
    pipeline = artifact["pipeline"]
    _check(hasattr(pipeline, "predict"), "model artifact has a usable fitted pipeline")
    classifier = pipeline.named_steps["clf"] if hasattr(pipeline, "named_steps") else pipeline
    # A fitted LogisticRegression exposes learned attributes ending in "_"
    # (e.g. classes_, coef_). Their presence confirms the artifact was
    # loaded pre-fitted, not constructed fresh/retrained by this test.
    _check(hasattr(classifier, "classes_") and hasattr(classifier, "coef_"), "loaded classifier is already fitted (classes_/coef_ present) -- confirms no retraining occurred")


def check_invalid_uploads_are_rejected(train_dir: Path) -> None:
    frame = pd.read_csv(train_dir / "Train1.csv")

    short_frame = frame.iloc[:100].copy()
    problems = rail_view.validate_uploaded_frame(short_frame)
    _check(any("rows" in p for p in problems), "invalid row count (100 rows) is rejected with a row-count message")

    wrong_columns = frame.rename(columns={frame.columns[10]: "not_a_real_column"})
    problems = rail_view.validate_uploaded_frame(wrong_columns)
    _check(len(problems) > 0, "an unrecognised/renamed column is rejected")

    missing_values_frame = frame.copy()
    missing_values_frame.iloc[0, 1] = None
    problems = rail_view.validate_uploaded_frame(missing_values_frame)
    _check(any("missing" in p.lower() for p in problems), "missing values are rejected with a missing-value message")

    _check(not rail_view.validate_uploaded_frame(frame), "sanity check: the original untouched file is still reported valid")


def check_door_not_broken() -> None:
    import src.door as door_pkg  # noqa: F401 -- import success is the check

    _check(hasattr(door_pkg, "predict_door_segments"), "src.door package still imports and exposes predict_door_segments")

    import app.streamlit_app as streamlit_app

    _check(hasattr(streamlit_app, "_door_upload_and_analysis"), "app.streamlit_app still exposes the Door UI function")
    _check(streamlit_app.render_rail_page is not None, "app.streamlit_app successfully wired up the Rail page")


def main() -> int:
    train_dir = Path(DEFAULT_DATASET_DIR) / "Train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Cannot find the organiser Train directory at: {train_dir}")

    artifact = load_model_artifact()

    multi_results = check_valid_single_and_multi_file_prediction(artifact, train_dir)
    check_official_output_schema(multi_results)
    check_confidence_sums_to_one(multi_results)
    check_model_loads_without_retraining(artifact)
    check_invalid_uploads_are_rejected(train_dir)
    check_door_not_broken()

    print("\nAll Rail app-integration checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        print(f"\nTEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
