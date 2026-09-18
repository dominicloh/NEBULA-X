"""Shared Streamlit app for the two official PS3 subsystems.

Door still intentionally stays safe until the Door segmentation pipeline is
configured from its Info Kit (see `_door_upload_and_analysis` below) --
untouched by the Rail Corrugation integration in this file. Rail Corrugation
now has a frozen, validated model (see planning/model_experiment_log.md);
its UI/processing logic lives entirely in `app/rail_view.py` to keep this
shared file's diff minimal and avoid conflicts with Door work-in-progress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency for non-UI execution
    class _MissingStreamlit:
        def __getattr__(self, name):
            raise RuntimeError("streamlit is required to run the app.")

    st = _MissingStreamlit()

# Ensure "app.rail_view" and "src...." imports resolve regardless of the
# working directory the app is launched from (e.g. `streamlit run
# app/streamlit_app.py` from the project root, per the README).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imported at module load, but guarded: a problem loading the Rail module
# (e.g. a missing model artifact) must never take down the Door page.
try:
    from app.rail_view import render_rail_page

    _RAIL_VIEW_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - defensive guard, see _rail_upload_and_analysis
    render_rail_page = None
    _RAIL_VIEW_IMPORT_ERROR = exc

VALID_DOOR_LABELS = ("Normal", "Abnormal resistance")
VALID_RAIL_LABELS = ("Normal", "Side I", "Side II")

if hasattr(st, "set_page_config"):
    st.set_page_config(page_title="NEBULA X — Train Condition Monitoring", layout="wide")


def _render_shared_intro():
    st.title("NEBULA X — Train Condition Monitoring")
    st.caption(
        "Our system provides one accessible condition-monitoring app for two rail subsystems. It analyses door motor-cycle signals to detect and classify normal or abnormal-resistance operating cycles, and analyses axle-box vibration and shock files to classify rail condition as Normal, Side I or Side II corrugation. Users can upload data, review the prediction and supporting signal evidence, and download competition-ready prediction files."
    )
    st.markdown("**Primary user:** Rail condition-monitoring or reliability engineer")
    st.markdown(
        "**Main user need:** Help rail engineers process subsystem sensor data quickly, identify which recordings require attention, understand the evidence behind each result and download consistent outputs for further investigation."
    )
    st.markdown(
        "**Value proposition:** Turn complex rail sensor time series into clear, explainable and downloadable condition-monitoring results."
    )


def _door_upload_and_analysis():
    st.subheader("Door subsystem")
    uploaded_file = st.file_uploader(
        "Upload continuous Door stream",
        type=["csv"],
        help="Door uses a single continuous test stream named Test.csv as defined by the Door Info Kit.",
    )
    if uploaded_file is None:
        st.info("Upload a Door stream to inspect the continuous signal and prepare for segmentation.")
        return

    st.write(f"Uploaded file: {uploaded_file.name}")
    try:
        stream = pd.read_csv(uploaded_file)
        required_fields = ["timestamp"]
        missing = [field for field in required_fields if field not in stream.columns]
        if missing:
            raise ValueError(
                "Door validation failed: required fields are missing. Confirm the exact schema from the Door Info Kit."
            )
        st.success("Door stream loaded successfully; schema check passed for the uploaded file.")
        st.write(f"Rows: {len(stream)} | Columns: {list(stream.columns)}")
        st.write(stream.head())
    except (ValueError, TypeError, FileNotFoundError) as exc:  # pragma: no cover - UI-level guard
        st.error(f"Door validation failed: {exc}")
        return

    st.warning(
        "Door segmentation and classification are not yet active until the Door Info Kit confirms the exact cycle rules, timestamp format and labels. This app stops safely before generating fake segment boundaries or labels."
    )

    if st.button("Analyse Door stream"):
        st.error("Model not ready: Door segmentation/classification is pending the Info Kit and trained pipeline.")


def _rail_upload_and_analysis():
    # All Rail UI/processing logic lives in app/rail_view.py (separate module
    # per TEAM_WORKFLOW.md, reusing the frozen model + src/rail_corrugation
    # pipeline from Stage 3). If that module ever fails to import (e.g. the
    # model artifact is missing on a fresh checkout), Door stays unaffected.
    if render_rail_page is None:
        st.error(f"Rail Corrugation page failed to load: {_RAIL_VIEW_IMPORT_ERROR}")
        st.caption("Door is unaffected by this. See app/rail_view.py and models/rail_corrugation_model.joblib.")
        return
    render_rail_page()


def _render_methodology_and_disclaimer():
    st.subheader("Methodology and limitations")
    st.markdown(
        "- Door pipeline: temporal segment detection and binary classification using continuous motor-cycle signals."
    )
    st.markdown(
        "- Rail Corrugation pipeline: file-level classification of axle-box vibration and shock data across Normal, Side I and Side II classes."
    )
    st.markdown(
        "- The app provides an explainable investigation workflow, but it does not confirm an exact mechanical root cause from signal data alone."
    )
    st.caption(
        "Disclaimer: This system supports engineering review and prioritisation. It does not replace formal engineering diagnosis or maintenance decisions."
    )


def main():
    _render_shared_intro()

    subsystem = st.selectbox("Select subsystem", ["Door", "Rail Corrugation"])
    if subsystem == "Door":
        _door_upload_and_analysis()
    else:
        _rail_upload_and_analysis()

    _render_methodology_and_disclaimer()


if __name__ == "__main__":
    main()
