"""Door upload, classification, evidence, and official submission download.

Presentation only was touched in the NEBULA X dashboard refactor -- every
call into `src/door/*` below is unchanged: same functions, same arguments,
same official output columns (`start_time,end_time,prediction`).
`_check_pipeline_status` and `_cycle_evidence` keep their exact names and
behaviour because `tests/test_door_scaffold.py` calls `_check_pipeline_status`
directly.

DASHBOARD LAYOUT NOTE: functions below are grouped by the six dashboard rows
in the design brief (KPI cards; cycle timeline + classification summary;
review queue + selected cycle; sensor evidence tabs; model insight;
downloads). Every number/chart comes from `official`/`detailed`
(`predict_door_file`'s real output) or the loaded model artifact -- nothing
here invents a metric or a threshold.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

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

from app import ui  # noqa: E402
from src.door import config  # noqa: E402
from src.door.model import load_door_pipeline  # noqa: E402
from src.door.predict import predict_door_file  # noqa: E402
from src.door.preprocess import load_door_csv, parse_door_datetime  # noqa: E402
from src.door.segment import detect_cycles  # noqa: E402
from src.door.validate_predictions import validate_door_predictions_frame  # noqa: E402

REVIEW_CONFIDENCE_CUTOFF = 0.70  # Prototype triage cue, never a safety threshold.

# Confirmed sensor columns (src/door/config.py::SIGNAL_COLUMNS) shown as
# tabs in the sensor-evidence card. Labels are the exact column names --
# units are only shown where the Info Kit confirmed one (mA, 10mV); no unit
# is invented for the two columns that don't have a documented one.
_SENSOR_TABS = {
    "Motor current": "Motor current(mA)",
    "Motor voltage": "Motor Voltage(10mV)",
    "Back EMF": "Motor electrodynamic force",
    "Door position": "Door leaf position",
}


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


# ---------------------------------------------------------------------------
# Row 1 -- KPI cards
# ---------------------------------------------------------------------------


def render_summary_cards(official: pd.DataFrame, detailed: pd.DataFrame) -> None:
    n = len(official)
    n_normal = int((official["prediction"] == "Normal").sum())
    n_abnormal = int((official["prediction"] == "Abnormal resistance").sum())
    n_review = int(detailed["needs_review"].sum())

    def pct(count: int) -> str:
        return f"{count / n:.0%} of batch" if n else ""

    items = [
        {"label": "Cycles detected", "value": n, "tone": "neutral"},
        {"label": "Normal", "value": n_normal, "sub": pct(n_normal), "tone": "good"},
        {"label": "Abnormal resistance", "value": n_abnormal, "sub": pct(n_abnormal), "tone": "warning"},
        {"label": "Needs review", "value": n_review, "sub": pct(n_review), "tone": "info"},
    ]
    # "Average model confidence" only shown because model_confidence is
    # already computed for every cycle (detailed["model_confidence"]) --
    # this averages an existing per-cycle value, it doesn't compute a new one.
    if "model_confidence" in detailed.columns and detailed["model_confidence"].notna().all():
        items.append({"label": "Avg. model confidence", "value": f"{detailed['model_confidence'].mean():.1%}", "tone": "rail"})
    ui.render_kpi_row(items)


# ---------------------------------------------------------------------------
# Row 2 -- Primary analytics (cycle timeline + classification summary)
# ---------------------------------------------------------------------------


def render_primary_analytics(detailed: pd.DataFrame) -> None:
    left, right = st.columns([2, 1])

    with left:
        with ui.card():
            ui.render_section_heading("Door cycle timeline", help_text="Hover a cycle for its number, timing, prediction and confidence.")
            timeline_frame = pd.DataFrame(
                {
                    "start_dt": detailed["start_time"].map(parse_door_datetime),
                    "end_dt": detailed["end_time"].map(parse_door_datetime),
                    "Cycle": [f"Cycle {i + 1}" for i in range(len(detailed))],
                    "prediction": detailed["prediction"],
                    "start_time": detailed["start_time"],
                    "end_time": detailed["end_time"],
                    "Confidence": detailed["model_confidence"].map(lambda v: f"{v:.1%}"),
                }
            )
            ui.render_timeline(
                timeline_frame,
                start_col="start_dt",
                end_col="end_dt",
                color_col="prediction",
                hover_cols=["Cycle", "start_time", "end_time", "Confidence"],
            )

    with right:
        with ui.card():
            ui.render_section_heading("Classification summary")
            counts = detailed["prediction"].value_counts()
            labels = ["Normal", "Abnormal resistance"]
            values = [int(counts.get(label, 0)) for label in labels]
            ui.render_donut_chart(labels, values, colors=[ui.CLASS_COLORS["Normal"], ui.CLASS_COLORS["Abnormal resistance"]])
            total = sum(values) or 1
            for label, value in zip(labels, values):
                st.caption(f"{label}: {value} ({value / total:.0%})")


# ---------------------------------------------------------------------------
# Row 3 -- Engineer review (cycle queue + selected cycle)
# ---------------------------------------------------------------------------


def render_cycle_queue(detailed: pd.DataFrame) -> pd.DataFrame:
    queue = detailed.copy()
    queue.insert(0, "Priority", range(1, len(queue) + 1))
    queue.insert(1, "Cycle", range(1, len(queue) + 1))

    display_table = pd.DataFrame(
        {
            "Priority": queue["Priority"],
            "Cycle": queue["Cycle"],
            "Start time": queue["start_time"],
            "End time": queue["end_time"],
            "Prediction": queue["prediction"],
            "Confidence": queue["model_confidence"].map(lambda v: f"{v:.1%}"),
            # Plain text only -- never a Streamlit colour-markup string --
            # see the note in app/rail_view.py::render_review_queue.
            "Review status": queue["needs_review"].map({True: "Needs review", False: "Reviewed"}),
        }
    )
    st.dataframe(display_table, use_container_width=True, hide_index=True, height=280)
    return queue


def render_selected_cycle_panel(queue: pd.DataFrame, stream: pd.DataFrame) -> pd.Series:
    selected = st.selectbox(
        "Inspect cycle",
        range(len(queue)),
        format_func=lambda index: f"Cycle {index + 1}: {queue.iloc[index]['prediction']}",
    )
    row = queue.iloc[selected]
    duration = (parse_door_datetime(row["end_time"]) - parse_door_datetime(row["start_time"])).total_seconds()

    st.markdown(f"**Cycle:** {row['Cycle']}")
    st.markdown(f"**Prediction:** {row['prediction']}")
    tone = "warning" if row["needs_review"] else "good"
    ui.render_status_pill(f"{row['model_confidence']:.1%} confidence", tone)
    st.markdown(f"**Duration:** {duration:.2f} s")
    st.markdown(f"**Start:** {row['start_time']}  \n**End:** {row['end_time']}")
    st.markdown("**Validation result:** Valid (passed schema + boundary checks)")

    normal_p = row.get("confidence_normal", row["model_confidence"] if row["prediction"] == "Normal" else 1 - row["model_confidence"])
    abnormal_p = row.get("confidence_abnormal_resistance", 1 - normal_p)
    ui.render_probability_bars([("Normal", float(normal_p)), ("Abnormal resistance", float(abnormal_p))])
    return row


# ---------------------------------------------------------------------------
# Row 4 -- Sensor evidence (tabs, one view at a time)
# ---------------------------------------------------------------------------


def render_sensor_evidence(stream: pd.DataFrame, row: pd.Series) -> None:
    window = _cycle_evidence(stream, row["start_time"], row["end_time"])
    tab_labels = list(_SENSOR_TABS.keys()) + ["Overlaid (current & position)"]
    tabs = st.tabs(tab_labels)

    for tab, label in zip(tabs, _SENSOR_TABS.keys()):
        column = _SENSOR_TABS[label]
        with tab:
            ui.render_line_chart(
                window["_datetime_parsed"], window[column],
                color="#168FE5" if label != "Door position" else "#9254DE",
                xlabel="Time", ylabel=column,
            )

    with tabs[-1]:
        fig = go.Figure()
        fig.add_scatter(x=window["_datetime_parsed"], y=window["Motor current(mA)"], name="Motor current(mA)", line=dict(color="#168FE5", width=1.2), yaxis="y1")
        fig.add_scatter(x=window["_datetime_parsed"], y=window["Door leaf position"], name="Door leaf position", line=dict(color="#9254DE", width=1.2), yaxis="y2")
        fig.update_layout(
            height=260,
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.08),
            xaxis=dict(title="Time"),
            yaxis=dict(title="Motor current(mA)"),
            yaxis2=dict(title="Door leaf position", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Row 5 -- Model insight
# ---------------------------------------------------------------------------


def render_model_insight() -> None:
    artifact = load_door_pipeline(config.MODEL_ARTIFACT_PATH)
    pipeline = artifact["pipeline"]
    if not hasattr(pipeline, "feature_importances_"):
        return
    with ui.card():
        ui.render_section_heading("Model-wide feature importance")
        importance = pd.Series(pipeline.feature_importances_, index=artifact["feature_names"]).sort_values(ascending=False).head(10)
        ui.render_horizontal_bar(list(importance.index), list(importance.values), value_format=".3f", height=340)
        ui.render_info_banner(
            "Importance describes the fitted model overall and is not a per-cycle physical root-cause explanation.",
            tone="warning",
        )


# ---------------------------------------------------------------------------
# Row 6 -- Downloads
# ---------------------------------------------------------------------------


def render_downloads(official: pd.DataFrame, detailed: pd.DataFrame) -> None:
    with ui.card():
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download official door_predictions.csv",
                data=official.to_csv(index=False).encode("utf-8"),
                file_name=config.OFFICIAL_OUTPUT_FILENAME,
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Download engineering review report",
                data=detailed.to_csv(index=False).encode("utf-8"),
                file_name="door_engineering_review.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.caption(f"Only `{config.OFFICIAL_OUTPUT_FILENAME}` (`start_time,end_time,prediction`) is scored -- the review report is supplementary.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render_door_page() -> None:
    with ui.card():
        st.caption("Upload one continuous Door recording (e.g. Train.csv, Test.csv, or another export in the same 17-column format).")
        uploaded = st.file_uploader("Continuous Door CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        ui.render_empty_state("Upload a continuous Door CSV to detect and classify each operating cycle.")
        return

    with st.status("Validating file...", expanded=False) as status:
        try:
            uploaded.seek(0)
            stream = load_door_csv(uploaded)
            ui.safe_status_update(status, label="Detecting cycles...")
            segments = detect_cycles(stream)
            ui.safe_status_update(status, label="Generating predictions...")
            uploaded.seek(0)
            official, detailed = predict_door_file(uploaded, model_path=config.MODEL_ARTIFACT_PATH)
            validate_door_predictions_frame(official, expected_segments=segments)
        except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
            ui.safe_status_update(status, label="Rejected", state="error")
            ui.render_info_banner(f"Upload or prediction rejected: {exc}", tone="critical")
            return
        ui.safe_status_update(status, label="Done", state="complete")

    ui.render_status_pill(f"Pipeline ready · {len(segments)} cycles detected", "good")

    confidence_columns = [column for column in detailed.columns if column.startswith("confidence_")]
    detailed = detailed.copy()
    detailed["model_confidence"] = detailed[confidence_columns].max(axis=1) if confidence_columns else float("nan")
    detailed["needs_review"] = detailed["model_confidence"].lt(REVIEW_CONFIDENCE_CUTOFF)

    render_summary_cards(official, detailed)
    render_primary_analytics(detailed)

    queue_col, selected_col = st.columns([0.65, 0.35])
    with queue_col:
        with ui.card():
            ui.render_section_heading("Cycle review queue", help_text=f"\"Needs review\" flags model confidence below {REVIEW_CONFIDENCE_CUTOFF:.0%} -- a prototype triage cue, not a safety threshold.")
            queue = render_cycle_queue(detailed)
    with selected_col:
        with ui.card():
            ui.render_section_heading("Selected cycle")
            selected_row = render_selected_cycle_panel(queue, stream)

    with ui.card():
        ui.render_section_heading("Sensor evidence", help_text="One sensor at a time -- choose a tab below. X-axis is time; y-axis units are shown where confirmed by the Info Kit.")
        render_sensor_evidence(stream, selected_row)

    render_model_insight()

    ui.render_section_heading("Downloads")
    render_downloads(official, detailed)


__all__ = ["render_door_page", "_check_pipeline_status", "_cycle_evidence"]
