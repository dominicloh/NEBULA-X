#!/usr/bin/env python3
"""Door Stage 5 scaffold checks.

No pytest (not a project dependency) -- plain assertions, matching the
style already used by tests/test_rail_predict.py and tests/test_rail_app.py.

These tests check the remaining Door interfaces after segmentation was added.
They do not train or require a classifier artifact.

Run directly:
    python tests/test_door_scaffold.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def check_door_modules_import() -> None:
    import src.door  # noqa: F401
    import src.door.config  # noqa: F401
    import src.door.features  # noqa: F401
    import src.door.inspect_data  # noqa: F401
    import src.door.model  # noqa: F401
    import src.door.predict  # noqa: F401
    import src.door.preprocess  # noqa: F401
    import src.door.segment  # noqa: F401
    import src.door.train  # noqa: F401
    import src.door.validate_predictions  # noqa: F401

    _check(True, "all src.door modules import successfully")


def check_valid_csv_loading(train_path: Path) -> None:
    from src.door.preprocess import load_door_csv

    stream = load_door_csv(train_path)
    _check(len(stream) > 0, "valid Door CSV loads with at least one row")
    _check("_datetime_parsed" in stream.columns, "load_door_csv adds a parsed Datetime column")
    _check(stream["_datetime_parsed"].is_monotonic_increasing, "loaded stream is sorted chronologically")


def check_malformed_csv_rejected() -> None:
    from src.door.preprocess import load_door_csv

    malformed = io.StringIO("not,a,real,door,file\n1,2,3,4,5\n")
    try:
        load_door_csv(malformed)
        raise AssertionError("malformed CSV should have been rejected")
    except ValueError:
        _check(True, "malformed/wrong-schema CSV is rejected with a clear ValueError")


def check_missing_column_rejected(train_path: Path) -> None:
    from src.door.preprocess import load_door_csv, validate_door_columns

    frame = pd.read_csv(train_path)
    frame = frame.drop(columns=["Door leaf position"])
    try:
        validate_door_columns(frame)
        raise AssertionError("frame missing a required column should have been rejected")
    except ValueError as exc:
        _check("Door leaf position" in str(exc), "missing-column error names the missing column")

    # Also check the full load path rejects it via a temp in-memory CSV.
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    try:
        load_door_csv(buffer)
        raise AssertionError("load_door_csv should reject a frame missing a required column")
    except ValueError:
        _check(True, "load_door_csv rejects a CSV missing a required column")


def check_unfinished_model_produces_clear_error(train_path: Path) -> None:
    from src.door.predict import predict_door_file

    try:
        predict_door_file(train_path, model_path=Path("this/path/does/not/exist.joblib"))
        raise AssertionError("predict_door_file should have raised: no model artifact exists")
    except FileNotFoundError as exc:
        _check("not ready" in str(exc).lower(), "predict_door_file raises a clear 'Door model is not ready' error when no artifact exists")


def check_predict_never_uses_ground_truth_segments(train_path: Path) -> None:
    """Prediction must take boundaries only from detect_cycles, never answers."""
    import inspect

    from src.door import config
    from src.door.predict import predict_door_file

    signature = inspect.signature(predict_door_file)
    _check(
        "segments" not in signature.parameters and "answers" not in signature.parameters,
        "predict_door_file has no parameter for externally-supplied segments/answers",
    )

    # Check actual CODE usage, not comments/docstrings (predict_door_file's
    # own docstring deliberately names Train_Segments_Answer.csv to explain
    # this very guarantee, so a naive "is the filename anywhere in the
    # source" check would false-positive on its own documentation). Strip
    # module/function docstrings out of the AST first, then check what's left.
    import ast

    class _DocstringRemover(ast.NodeTransformer):
        @staticmethod
        def _strip_leading_docstring(node):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
            return node

        def visit_Module(self, node):
            self.generic_visit(node)
            return self._strip_leading_docstring(node)

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            return self._strip_leading_docstring(node)

    tree = ast.parse(inspect.getsource(sys.modules["src.door.predict"]))
    cleaned_tree = ast.fix_missing_locations(_DocstringRemover().visit(tree))
    code_only = ast.unparse(cleaned_tree)
    _check(
        "Train_Segments_Answer" not in code_only and "TRAIN_SEGMENTS_ANSWER_FILENAME" not in code_only,
        "src/door/predict.py's executable code (excluding docstrings) never references Train_Segments_Answer.csv",
    )

    _check("segments = detect_cycles(stream)" in code_only,
           "predict_door_file obtains its segment boundaries from detect_cycles")


def check_segmentation_implemented(train_path: Path) -> None:
    from src.door.preprocess import load_door_csv
    from src.door.segment import detect_cycles, validate_segment_table

    stream = load_door_csv(train_path)
    segments = detect_cycles(stream)
    _check(len(segments) > 0, "detect_cycles returns candidate cycles")
    _check(not validate_segment_table(segments, stream), "detected cycles have valid ordered boundaries")


def check_validator_rejects_wrong_columns() -> None:
    from src.door.validate_predictions import validate_door_predictions_file

    buffer = io.StringIO()
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(buffer, index=False)
    buffer.seek(0)
    # validate_door_predictions_file expects a Path; write to a temp file instead.
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write(buffer.getvalue())
        tmp_path = Path(tmp.name)
    try:
        validate_door_predictions_file(tmp_path, verbose=False)
        raise AssertionError("validator should reject a file with the wrong columns")
    except ValueError:
        _check(True, "validate_door_predictions_file rejects wrong/missing columns")
    finally:
        tmp_path.unlink(missing_ok=True)


def check_validator_rejects_invalid_labels() -> None:
    from src.door.validate_predictions import validate_door_predictions_file

    import tempfile

    frame = pd.DataFrame(
        {
            "start_time": ["2023-7-5-0-0-0-0"],
            "end_time": ["2023-7-5-0-0-3-0"],
            "prediction": ["Completely Jammed"],
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        frame.to_csv(tmp, index=False)
        tmp_path = Path(tmp.name)
    try:
        validate_door_predictions_file(tmp_path, verbose=False)
        raise AssertionError("validator should reject an invalid label")
    except ValueError as exc:
        _check("label" in str(exc).lower(), "validate_door_predictions_file rejects an invalid label")
    finally:
        tmp_path.unlink(missing_ok=True)


def check_pipeline_status_not_ready_despite_trained_artifact(train_path: Path) -> None:
    """The page still requires a separate valid classifier artifact."""
    from src.door.preprocess import load_door_csv

    import app.door_view as door_view

    stream = load_door_csv(train_path)
    status = door_view._check_pipeline_status(stream)

    _check(status["segmentation_ready"] is True, "pipeline status recognizes validated segmentation")
    _check(
        status["pipeline_ready"] == (status["segmentation_ready"] and status["classifier_ready"]),
        "pipeline readiness requires both segmentation and a loadable classifier",
    )


def check_no_cv_leakage_in_candidate_models() -> None:
    """Guards against cross-validation leakage in src/door/train.py.

    Reviewed manually: `evaluate_candidate_models()` passes the raw feature
    matrix X straight into `cross_validate`/`cross_val_predict` with no
    scaling/imputation/selection applied to X beforehand. Both sklearn
    functions clone the given estimator once per fold and call `.fit()`
    only on that fold's training rows, so whatever preprocessing lives
    INSIDE the estimator is refit per fold automatically.

    This test encodes the two facts that make that true, so a future edit
    that breaks either one fails loudly instead of silently leaking:
      1. The LogisticRegression candidate is a Pipeline with the scaler
         INSIDE it (not a bare classifier fit on pre-scaled features).
      2. train.py actually uses cross_validate/cross_val_predict (fold-safe)
         rather than a manual single fit/score pattern.
    """
    import inspect

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from src.door.model import build_candidate_models

    models = build_candidate_models()
    logistic = models["LogisticRegression"]
    _check(isinstance(logistic, Pipeline), "LogisticRegression candidate is an sklearn Pipeline, not a bare classifier")
    step_types = [type(step) for _, step in logistic.steps]
    _check(StandardScaler in step_types, "LogisticRegression Pipeline has a StandardScaler step INSIDE it (refit per CV fold)")

    import src.door.train as door_train

    train_source = inspect.getsource(door_train)
    _check(
        "cross_validate(" in train_source and "cross_val_predict(" in train_source,
        "train.py evaluates models via cross_validate/cross_val_predict (fold-safe), not a manual fit-once pattern",
    )


def check_render_door_page_exists() -> None:
    from app.door_view import render_door_page

    _check(callable(render_door_page), "app.door_view.render_door_page exists and is callable")


def check_door_import_does_not_break_rail() -> None:
    import app.door_view  # noqa: F401
    import app.rail_view as rail_view
    import app.streamlit_app as streamlit_app

    _check(callable(rail_view.render_rail_page), "app.rail_view.render_rail_page still importable after Door import")
    _check(streamlit_app.render_rail_page is not None, "app.streamlit_app still wires up Rail after Door import")
    _check(streamlit_app.render_door_page is not None, "app.streamlit_app wires up Door too")


def main() -> int:
    from src.door import config

    train_path = config.DEFAULT_DATASET_DIR / config.TRAIN_FILENAME
    if not train_path.is_file():
        # Local smoke-test checkout keeps ignored organiser data inside the
        # repository; production's configured sibling path stays unchanged.
        train_path = PROJECT_ROOT / "organiser-materials" / "PS3" / "02_Datasets" / "Door" / config.TRAIN_FILENAME
    if not train_path.is_file():
        raise FileNotFoundError(f"Cannot find the organiser Door Train.csv at: {train_path}")

    check_door_modules_import()
    check_valid_csv_loading(train_path)
    check_malformed_csv_rejected()
    check_missing_column_rejected(train_path)
    check_unfinished_model_produces_clear_error(train_path)
    check_predict_never_uses_ground_truth_segments(train_path)
    check_segmentation_implemented(train_path)
    check_validator_rejects_wrong_columns()
    check_validator_rejects_invalid_labels()
    check_no_cv_leakage_in_candidate_models()
    check_pipeline_status_not_ready_despite_trained_artifact(train_path)
    check_render_door_page_exists()
    check_door_import_does_not_break_rail()

    print("\nAll Door scaffold checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        print(f"\nTEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
