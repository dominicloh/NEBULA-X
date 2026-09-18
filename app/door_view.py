"""Door upload, classification, evidence, and official submission download."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import streamlit as st
except ImportError:  # permits non-UI module imports
    class _MissingStreamlit:
        def __getattr__(self, name):
            raise RuntimeError("streamlit is required to render the Door page.")
    st = _MissingStreamlit()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402
from src.door.model import load_door_pipeline  # noqa: E402
from src.door.predict import predict_door_file  # noqa: E402
from src.door.preprocess import load_door_csv, parse_door_datetime  # noqa: E402
from src.door.segment import detect_cycles  # noqa: E402
from src.door.validate_predictions import validate_door_predictions_frame  # noqa: E402

REVIEW_CONFIDENCE_CUTOFF = 0.70  # Prototype triage cue, never a safety threshold.


def _check_pipeline_status(stream: pd.DataFrame) -> dict:
    """Mark ready only after automatic segmentation, model inference, and validation."""
    status = {"segmentation_ready": False, "classifier_ready": False, "pipeline_ready": False}
    try:
        segments = detect_cycles(stream)
        status["segmentation_ready"] = True
        status["segmentation_message"] = f"{len(segments)} cycles detected"
    except (ValueError, NotImplementedError) as exc:
        status["segmentation_message"] = str(exc)
        status["classifier_message"] = "Prediction unavailable"
        return status
    try:
        load_door_pipeline(config.MODEL_ARTIFACT_PATH)
        # Use the same public path as the upload workflow, with source columns
        # only. No labels, answer boundaries, or threshold override enter here.
        source = io.StringIO(stream[list(config.EXPECTED_COLUMNS)].to_csv(index=False))
        official, _ = predict_door_file(source, model_path=config.MODEL_ARTIFACT_PATH)
        validate_door_predictions_frame(official, expected_segments=segments)
        status["classifier_ready"] = True
        status["pipeline_ready"] = True
        status["classifier_message"] = "Frozen model loaded; predictions validated"
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        status["classifier_message"] = str(exc)
    return status


def _cycle_evidence(stream: pd.DataFrame, start_text: str, end_text: str) -> pd.DataFrame:
    start, end = parse_door_datetime(start_text), parse_door_datetime(end_text)
    return stream.loc[stream["_datetime_parsed"].between(start, end)]


def render_door_page() -> None:
    st.subheader("Door")
    st.write("Upload one continuous Door CSV to detect and classify each operating cycle.")
    uploaded = st.file_uploader("Continuous Door CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload Train.csv, Test.csv, or another valid continuous Door recording.")
        return
    try:
        uploaded.seek(0)
        stream = load_door_csv(uploaded)
        uploaded.seek(0)
        official, detailed = predict_door_file(uploaded, model_path=config.MODEL_ARTIFACT_PATH)
        segments = detect_cycles(stream)
        validate_door_predictions_frame(official, expected_segments=segments)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Door upload or prediction rejected: {exc}")
        return

    st.success("Door pipeline READY: detected cycles, frozen model prediction, and output validation passed.")
    confidence_columns = [column for column in detailed.columns if column.startswith("confidence_")]
    detailed = detailed.copy()
    detailed["model_confidence"] = detailed[confidence_columns].max(axis=1) if confidence_columns else float("nan")
    detailed["needs_review"] = detailed["model_confidence"].lt(REVIEW_CONFIDENCE_CUTOFF)
    count_a, count_b, count_c, count_d = st.columns(4)
    count_a.metric("Cycles detected", len(official))
    count_b.metric("Normal", int((official["prediction"] == "Normal").sum()))
    count_c.metric("Abnormal resistance", int((official["prediction"] == "Abnormal resistance").sum()))
    count_d.metric("Needs Review", int(detailed["needs_review"].sum()))
    st.caption(f"Needs Review uses prototype model confidence below {REVIEW_CONFIDENCE_CUTOFF:.0%}; confidence is not guaranteed correctness or a safety threshold.")

    st.markdown("#### Chronological review table")
    st.dataframe(detailed, use_container_width=True)
    selected = st.selectbox("Inspect cycle", range(len(detailed)),
                            format_func=lambda index: f"Cycle {index + 1}: {detailed.iloc[index]['prediction']}")
    row = detailed.iloc[selected]
    window = _cycle_evidence(stream, row["start_time"], row["end_time"])
    duration = (parse_door_datetime(row["end_time"]) - parse_door_datetime(row["start_time"])).total_seconds()
    st.write(f"Start: {row['start_time']}  |  End: {row['end_time']}  |  Duration: {duration:.2f} s")
    st.write(f"Prediction: {row['prediction']}  |  Model confidence: {row['model_confidence']:.1%}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for axis, column in zip(axes, ("Motor current(mA)", "Door leaf position")):
        axis.plot(window["_datetime_parsed"], window[column], linewidth=0.8)
        axis.set_ylabel(column)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    artifact = load_door_pipeline(config.MODEL_ARTIFACT_PATH)
    pipeline = artifact["pipeline"]
    if hasattr(pipeline, "feature_importances_"):
        importance = pd.Series(pipeline.feature_importances_, index=artifact["feature_names"])
        st.write("Model-wide feature importance (top five):")
        st.dataframe(importance.sort_values(ascending=False).head(5).rename("importance"))
    st.caption("Signal plots and model importance support review; they do not independently prove a mechanical root cause.")

    st.download_button("Download validated door_predictions.csv", data=official.to_csv(index=False).encode("utf-8"),
                       file_name=config.OFFICIAL_OUTPUT_FILENAME, mime="text/csv")
    st.download_button("Download engineering review table", data=detailed.to_csv(index=False).encode("utf-8"),
                       file_name="door_engineering_review.csv", mime="text/csv")


__all__ = ["render_door_page", "_check_pipeline_status", "_cycle_evidence"]
