"""Door page for the shared NEBULA X Streamlit app.

Kept separate from `app/rail_view.py` and `app/streamlit_app.py` (per
TEAM_WORKFLOW.md: "keep the Door and Rail pipelines separate at the module
level"), so the Door teammate can extend this file freely without touching
Rail's working code, and vice versa.

This page NEVER shows an invented prediction. Door's pipeline has two
stages -- segmentation (`src/door/segment.py::detect_cycles`) and
classification (`src/door/model.py`) -- and segmentation is not implemented
yet (see planning/door_handoff.md). Every section below either does real
work (upload validation, signal preview, a basic chart) or honestly reports
"not ready yet", never a placeholder table dressed up as a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover - lets non-UI code (e.g. tests) import this module
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
from src.door.preprocess import load_door_csv  # noqa: E402
from src.door.segment import detect_cycles  # noqa: E402

# Cap how many rows the preview chart draws, purely for rendering speed on a
# large uploaded stream -- this NEVER affects segmentation/prediction
# (which always runs against the full uploaded data), only the chart.
MAX_CHART_ROWS = 4000


def _check_pipeline_status(stream: pd.DataFrame) -> dict:
    """Actually attempt each pipeline stage and report what really
    happened -- never a hard-coded status string.

    IMPORTANT: `segmentation_ready` is set ONLY when `detect_cycles()`
    actually returns without raising. A trained classifier artifact
    existing on disk (`models/door_model.joblib`) does NOT make
    segmentation ready, and must never be read as "Door is ready" by
    itself -- see `pipeline_ready` below, which requires BOTH.
    """
    status: dict = {"segmentation_ready": False, "classifier_ready": False}

    try:
        detect_cycles(stream)
        status["segmentation_ready"] = True
        status["segmentation_message"] = "READY"
    except NotImplementedError:
        status["segmentation_message"] = "NOT IMPLEMENTED YET (see src/door/segment.py::detect_cycles)"
    except Exception as exc:  # pragma: no cover - defensive: report, don't crash the page
        status["segmentation_message"] = f"ERROR: {exc}"

    # A trained-classifier artifact existing on disk is necessary but NOT
    # sufficient for "Door is ready" -- it says nothing about segmentation.
    if config.MODEL_ARTIFACT_PATH.is_file():
        try:
            artifact = load_door_pipeline(config.MODEL_ARTIFACT_PATH)
            status["classifier_ready"] = True
            status["classifier_message"] = f"TRAINED (artifact at {config.MODEL_ARTIFACT_PATH.name})"
            if isinstance(artifact, dict):
                status["classifier_note"] = artifact.get("validation_summary", {}).get("note")
        except Exception as exc:  # pragma: no cover - defensive: a corrupt artifact is "not ready", not a crash
            status["classifier_message"] = f"ARTIFACT FOUND BUT COULD NOT BE LOADED: {exc}"
    else:
        status["classifier_message"] = "NOT TRAINED YET (run: python src/door/train.py --finalize)"

    # The single source of truth for "is Door ready end-to-end" -- requires
    # BOTH stages, computed here rather than left for a reader to infer by
    # combining two separate metrics themselves.
    status["pipeline_ready"] = status["segmentation_ready"] and status["classifier_ready"]
    return status


def _render_signal_preview(stream: pd.DataFrame) -> None:
    st.markdown("#### Uploaded signal preview")
    st.write(f"Rows: {len(stream):,} | Columns: {stream.shape[1]}")
    st.dataframe(stream.head(10), use_container_width=True)

    chart_columns = ["Door leaf position", "Motor current(mA)"]
    available_columns = [c for c in chart_columns if c in stream.columns]
    if not available_columns:
        return

    chart_frame = stream.head(MAX_CHART_ROWS)
    if len(stream) > MAX_CHART_ROWS:
        st.caption(f"Chart shows the first {MAX_CHART_ROWS:,} of {len(stream):,} rows (display only -- prediction always uses the full stream).")

    fig, axes = plt.subplots(len(available_columns), 1, figsize=(8, 2.4 * len(available_columns)), sharex=True)
    if len(available_columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, available_columns):
        ax.plot(chart_frame["_datetime_parsed"], chart_frame[column], linewidth=0.6, color="#2563eb")
        ax.set_ylabel(column)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_pipeline_status(stream: pd.DataFrame) -> dict:
    st.markdown("#### Pipeline status")
    status = _check_pipeline_status(stream)

    # One unambiguous top-level banner FIRST -- never let the two granular
    # metrics below be the only signal of overall readiness. A trained
    # classifier artifact existing is not enough on its own.
    if status["pipeline_ready"]:
        st.success("Door pipeline status: READY (segmentation + trained classifier both available).")
    else:
        st.error("Door pipeline status: NOT READY. A prediction cannot be produced yet.")

    col1, col2 = st.columns(2)
    col1.metric("Segmentation", "Ready" if status["segmentation_ready"] else "Not ready")
    col1.caption(status["segmentation_message"])
    col2.metric("Classifier", "Trained" if status["classifier_ready"] else "Not trained")
    col2.caption(status["classifier_message"])

    if status["classifier_ready"] and status.get("classifier_note"):
        st.caption(f"Classifier validation note: {status['classifier_note']}")

    if not status["pipeline_ready"]:
        st.warning(
            "Door model is not ready. Both segmentation and a trained classifier are required "
            "before real predictions can be produced -- a trained classifier artifact existing on "
            "disk does NOT mean Door is ready by itself. See planning/door_handoff.md for what's left."
        )
    return status


def render_door_page() -> None:
    st.subheader("Door")
    st.markdown(
        "Finds each **door-open/close cycle** in a continuous sensor stream and classifies it as "
        "**Normal** or **Abnormal resistance**. Unlike Rail Corrugation, Door's input is one long "
        "continuous recording, not one file per example -- the model must first find where each "
        "cycle starts and ends, then classify it."
    )
    st.caption(
        "This page is a working base, not a finished pipeline -- see planning/door_handoff.md for "
        "exactly what is implemented and what the Door teammate still needs to build."
    )

    uploaded_file = st.file_uploader(
        "Upload a continuous Door stream (e.g. Test.csv)",
        type=["csv"],
        help="Door uses a single continuous stream, not one file per cycle -- upload the whole recording.",
    )
    if uploaded_file is None:
        st.info("Upload a Door CSV stream to validate it and preview its signals.")
        return

    try:
        stream = load_door_csv(uploaded_file)
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Upload rejected: {exc}")
        return

    st.success(f"'{uploaded_file.name}' passed structural validation (columns, dtypes, no missing/duplicate rows).")

    _render_signal_preview(stream)
    _render_pipeline_status(stream)

    st.markdown("#### Detected cycles")
    st.info("Will appear here once segmentation (`detect_cycles`) is implemented. No invented cycles are shown.")

    st.markdown("#### Prediction confidence / evidence")
    st.info("Will appear here once both segmentation and a trained classifier are ready.")

    st.markdown("#### Download")
    try:
        official_df, _detailed_df = predict_door_file(uploaded_file, model_path=config.MODEL_ARTIFACT_PATH)
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        st.warning(f"Cannot produce door_predictions.csv yet: {exc}")
    else:
        st.download_button(
            "Download official door_predictions.csv",
            data=official_df.to_csv(index=False).encode("utf-8"),
            file_name=config.OFFICIAL_OUTPUT_FILENAME,
            mime="text/csv",
        )

    st.divider()
    st.caption(
        "This prototype supports engineering review. A prediction does not independently confirm "
        "a door fault or replace railway inspection and safety procedures."
    )


__all__ = ["render_door_page"]
