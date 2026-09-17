"""Shared Streamlit app for the two official PS3 subsystems.

This app intentionally stays safe until the Door segmentation pipeline and Rail model
are configured from their respective Info Kits. It presents the expected workflow and
stops with clear model-not-ready messages rather than generating fake outputs.
"""

from __future__ import annotations

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency for non-UI execution
    class _MissingStreamlit:
        def __getattr__(self, name):
            raise RuntimeError("streamlit is required to run the app.")

    st = _MissingStreamlit()

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
    st.subheader("Rail Corrugation subsystem")
    uploaded_files = st.file_uploader(
        "Upload Rail Corrugation files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or many axle-box vibration and shock CSV files. Results are file-level classifications.",
    )
    if not uploaded_files:
        st.info("Upload one or more Rail Corrugation CSV files to classify them as Normal, Side I or Side II.")
        return

    st.write(f"Uploaded {len(uploaded_files)} file(s): {[file.name for file in uploaded_files]}")
    table = pd.DataFrame({"file_id": [file.name for file in uploaded_files], "prediction": ["[TO CONFIRM FROM RAIL INFO KIT]" for _ in uploaded_files]})
    st.dataframe(table)
    st.warning(
        "Rail classification is not yet active until the file schema, feature set and trained model are confirmed."
    )

    if st.button("Analyse Rail files"):
        st.error("Model not ready: Rail Corrugation classification is pending the Info Kit and trained model.")


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
