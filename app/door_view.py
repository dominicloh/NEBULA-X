"""Door upload, classification, evidence, and official submission download.

Presentation only was touched in the NEBULA X dashboard refactor -- every
call into `src/door/*` below is unchanged: same functions, same arguments,
same official output columns (`start_time,end_time,prediction`).
`_check_pipeline_status` and `_cycle_evidence` keep their exact names and
behaviour because `tests/test_door_scaffold.py` calls `_check_pipeline_status`
directly.

DASHBOARD LAYOUT NOTE: functions below are grouped by the six dashboard rows
in the design brief (KPI cards; cycle timeline + classification summary;
review queue + selected cycle; graphical display tabs; model insight;
downloads). Every number/chart comes from `official`/`detailed`
(`predict_door_file`'s real output) or the loaded model artifact -- nothing
here invents a metric or a threshold.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
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
        with ui.card("door_timeline_card"):
            ui.render_section_heading("Door Cycle Timeline", help_text="Hover a cycle for its number, timing, prediction and confidence.")
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
        with ui.card("door_summary_card"):
            ui.render_section_heading("Classification summary")
            counts = detailed["prediction"].value_counts()
            labels = ["Normal", "Abnormal resistance"]
            values = [int(counts.get(label, 0)) for label in labels]
            # No caption duplicating counts/percentages below the chart --
            # the donut's outside labels (label, count, percentage) already
            # show this.
            ui.render_donut_chart(labels, values, colors=[ui.CLASS_COLORS["Normal"], ui.CLASS_COLORS["Abnormal resistance"]])


# ---------------------------------------------------------------------------
# Row 3 -- Engineer review (cycle queue + selected cycle)
# ---------------------------------------------------------------------------


def _build_cycle_queue(detailed: pd.DataFrame) -> pd.DataFrame:
    """Add Priority/Cycle columns in the existing chronological order --
    this is the one, unfiltered source of truth for the queue. Never
    mutates `detailed`; returns a fresh copy so filtering below can never
    touch the underlying prediction DataFrame.
    """
    queue = detailed.copy()
    queue.insert(0, "Priority", range(1, len(queue) + 1))
    queue.insert(1, "Cycle", range(1, len(queue) + 1))
    return queue


_STATUS_FILTER_OPTIONS = ("All cycles", "Needs review", "No review flag")
_PREDICTION_FILTER_OPTIONS = ("All predictions", "Normal", "Abnormal resistance")


def _reset_door_filters() -> None:
    """`on_click` callback for the "Show all cycles" button -- resets filter
    widget state before the next rerun, which is the crash-free way to
    change a `key`-bound widget's value (mutating st.session_state directly
    after the widget has already rendered this run raises in Streamlit).
    """
    st.session_state["door_status_filter"] = "All cycles"
    st.session_state["door_prediction_filter"] = "All predictions"
    st.session_state["door_cycle_filter"] = "All"


def _apply_queue_filters(queue: pd.DataFrame, status_filter: str, prediction_filter: str, cycle_filter: str) -> pd.DataFrame:
    """Filter the DISPLAY copy only -- uses the existing `needs_review`
    boolean and `prediction` values exactly as already computed; never
    recalculates a threshold and never touches `queue` in place.

    "No review flag" (needs_review=False) intentionally does NOT mean a
    human has reviewed the cycle -- the app has no such record. It only
    means the model's own confidence didn't trigger the review flag.
    """
    filtered = queue
    if status_filter == "Needs review":
        filtered = filtered[filtered["needs_review"]]
    elif status_filter == "No review flag":
        filtered = filtered[~filtered["needs_review"]]
    # "All cycles" -> no status filtering.

    if prediction_filter != "All predictions":
        filtered = filtered[filtered["prediction"] == prediction_filter]

    if cycle_filter != "All":
        filtered = filtered[filtered["Cycle"] == int(cycle_filter)]

    return filtered.copy()


def render_cycle_filters(queue: pd.DataFrame) -> pd.DataFrame:
    """Renders the filter row + a concise result count, and returns the
    FILTERED queue (a copy) for the table and selected-cycle panel below.
    Filtering only changes what's displayed here -- KPI totals and the
    classification summary are rendered from `official`/`detailed` earlier
    in `render_door_page` and never see this filtered view.
    """
    # Default: "Needs review" when at least one cycle needs it, else "All
    # cycles" -- seeded once (like app.ui.subsystem_selector's pattern) so a
    # later manual filter choice on the same result isn't overridden every rerun.
    if "door_status_filter" not in st.session_state:
        st.session_state["door_status_filter"] = "Needs review" if bool(queue["needs_review"].any()) else "All cycles"
    st.session_state.setdefault("door_prediction_filter", "All predictions")
    st.session_state.setdefault("door_cycle_filter", "All")

    ui.render_legend([("Normal", "good"), ("Needs review", "warning"), ("Abnormal resistance", "critical")])

    col1, col2, col3, col4 = st.columns([0.9, 1.1, 1.1, 1])
    with col1:
        cycle_options = ["All"] + [str(c) for c in queue["Cycle"].tolist()]
        cycle_filter = st.selectbox("Cycle", cycle_options, key="door_cycle_filter")
    with col2:
        prediction_filter = st.selectbox("Prediction", _PREDICTION_FILTER_OPTIONS, key="door_prediction_filter")
    with col3:
        status_filter = st.selectbox("Review status", _STATUS_FILTER_OPTIONS, key="door_status_filter")
    with col4:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)  # align with the selectboxes above
        st.button("Show all cycles", on_click=_reset_door_filters, use_container_width=True)

    filtered = _apply_queue_filters(queue, status_filter, prediction_filter, cycle_filter)
    st.caption(f"Showing {len(filtered)} of {len(queue)} cycles")
    return filtered


# Static demo content for the "Cases" popover -- deliberately NOT presented
# as retrieved-by-similarity or as a confirmed cause: see _render_maintenance_popover.
#
# "last_updated" is a fixed literal, not datetime.now() -- it records when
# this demonstration guide ENTRY was last edited, not when a fault occurred,
# and must stay identical across every rerun rather than drift with today's
# date.
_MAINTENANCE_REFERENCE_CASES = (
    {
        "title": "Door-track obstruction",
        "reference_id": "DEMO-DOR-001",
        "last_updated": "2026-09-19",
        "possible_sign": "Increased motor effort during part of the movement.",
        "previous_action": "Inspected the track and removed debris.",
        "outcome": "Normal movement restored.",
    },
    {
        "title": "Roller or guide misalignment",
        "reference_id": "DEMO-DOR-002",
        "last_updated": "2026-09-19",
        "possible_sign": "Resistance repeatedly appearing near the same door position.",
        "previous_action": "Checked roller alignment and mechanical wear.",
        "outcome": "Guide adjusted and cycle retested.",
    },
    {
        "title": "Increased drive friction",
        "reference_id": "DEMO-DOR-003",
        "last_updated": "2026-09-19",
        "possible_sign": "Higher motor loading across a larger part of the cycle.",
        "previous_action": "Inspected the drive mechanism, seals and moving components.",
        "outcome": "Friction source identified and corrective maintenance performed.",
    },
)


def _render_engineer_notes(cycle: int) -> None:
    """Prototype, session-only note-taking for one cycle -- stored in
    st.session_state keyed by the cycle number, never written to disk or
    mixed with the model's own DataFrames.
    """
    notes_store = st.session_state.setdefault("door_engineer_notes", {})

    with st.expander("Add engineer note", expanded=False, key=f"door_notes_expander_{cycle}"):
        st.caption("Prototype notes are stored for this browser session only.")
        with st.form(key=f"door_note_form_{cycle}", clear_on_submit=True):
            engineer = st.text_input("Engineer name", key=f"door_note_engineer_{cycle}")
            observation = st.text_area("Observation", key=f"door_note_observation_{cycle}")
            action = st.text_area("Action taken", key=f"door_note_action_{cycle}")
            outcome = st.text_area("Outcome", key=f"door_note_outcome_{cycle}")
            submitted = st.form_submit_button("Save note")
        if submitted and any(field.strip() for field in (engineer, observation, action, outcome)):
            # Timestamp is generated here, at the moment "Save note" is
            # clicked -- never on an ordinary rerun -- so it reflects when
            # the note was actually saved, not when the page happened to redraw.
            notes_store.setdefault(cycle, []).append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "engineer": engineer.strip(),
                    "observation": observation.strip(),
                    "action": action.strip(),
                    "outcome": outcome.strip(),
                }
            )

        saved_notes = notes_store.get(cycle, [])
        if saved_notes:
            st.markdown(f"**Saved notes for Cycle {cycle} (this session)**")
            for i, note in enumerate(saved_notes, start=1):
                st.markdown(
                    f"**#{i} · {note['engineer'] or 'Unnamed engineer'} · {note.get('timestamp', '—')}**  \n"
                    f"Observation: {note['observation'] or '—'}  \n"
                    f"Action taken: {note['action'] or '—'}  \n"
                    f"Outcome: {note['outcome'] or '—'}"
                )


def _render_maintenance_popover(cycle: int) -> None:
    """Popover content for one 'Abnormal resistance' cycle's Cases control.

    These three cards are fixed demonstration content, not real organiser
    maintenance records and not the output of any similarity search or
    model lookup -- they never claim to explain *this* cycle's prediction,
    only to suggest generically similar situations worth investigating.
    """
    st.markdown("**Reference maintenance cases**")
    ui.render_status_pill("Demo guidance", "neutral")
    st.caption(
        "Possible situations with similar symptoms. Use these as investigation "
        "guidance, not confirmed diagnoses."
    )
    st.caption("Demonstration reference cases -- not genuine organiser maintenance records.")
    for case in _MAINTENANCE_REFERENCE_CASES:
        st.markdown(
            f"""
            <div class="nebula-reference-case">
              <div class="nebula-reference-case__title">{case['title']}</div>
              <div class="nebula-reference-case__meta">Reference ID: {case['reference_id']} · Last updated: {case['last_updated']}</div>
              <div class="nebula-reference-case__body">
                <strong>Possible sign:</strong> {case['possible_sign']}<br>
                <strong>Previous action:</strong> {case['previous_action']}<br>
                <strong>Outcome:</strong> {case['outcome']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()
    _render_engineer_notes(cycle)


_QUEUE_COLUMN_WEIGHTS = (0.55, 0.55, 1.3, 1.3, 1.5, 0.9, 1.4, 0.5)
_QUEUE_COLUMN_HEADERS = ("Priority", "Cycle", "Start time", "End time", "Prediction", "Confidence", "Review status", "Cases")


def render_cycle_queue_table(filtered_queue: pd.DataFrame) -> None:
    """A presentation-only, row-by-row rendering of `filtered_queue` using
    containers/columns instead of `st.dataframe` -- `st.dataframe` cells
    can't hold an interactive widget, and the "Cases" column below needs a
    real `st.popover` on every "Abnormal resistance" row. Every value here
    is read straight from `filtered_queue`, already computed/filtered
    upstream; this function only draws it, in the same row order.
    """
    if filtered_queue.empty:
        ui.render_empty_state("No cycles match the selected filters.")
        return

    header_cols = st.columns(_QUEUE_COLUMN_WEIGHTS)
    for header_col, label in zip(header_cols, _QUEUE_COLUMN_HEADERS):
        header_col.markdown(f"<span class='nebula-kpi__label'>{label}</span>", unsafe_allow_html=True)

    with st.container(height=280, border=False):
        for _, row in filtered_queue.iterrows():
            cycle = int(row["Cycle"])
            cols = st.columns(_QUEUE_COLUMN_WEIGHTS)
            cols[0].markdown(str(row["Priority"]))
            cols[1].markdown(str(cycle))
            cols[2].markdown(str(row["start_time"]))
            cols[3].markdown(str(row["end_time"]))
            with cols[4]:
                prediction_tone = "critical" if row["prediction"] == "Abnormal resistance" else "good"
                ui.render_status_pill(row["prediction"], prediction_tone)
            cols[5].markdown(f"{row['model_confidence']:.1%}")
            with cols[6]:
                needs_review = bool(row["needs_review"])
                # "No review flag" (not "Reviewed"): the app has no record of
                # a human actually reviewing the cycle -- this only means the
                # model's own confidence didn't trigger the review flag.
                status_text = "Needs review" if needs_review else "No review flag"
                ui.render_status_pill(status_text, "warning" if needs_review else "good")
            with cols[7]:
                if row["prediction"] == "Abnormal resistance":
                    with st.popover("⋯", help="View three reference cases", key=f"door_cases_popover_{cycle}"):
                        _render_maintenance_popover(cycle)
                else:
                    st.markdown("<span class='nebula-muted'>—</span>", unsafe_allow_html=True)


def render_selected_cycle_panel(filtered_queue: pd.DataFrame, stream: pd.DataFrame) -> pd.Series | None:
    if filtered_queue.empty:
        ui.render_empty_state("No cycle selected -- adjust filters to inspect a cycle.")
        return None

    available_cycles = filtered_queue["Cycle"].tolist()
    # If the previously selected cycle was filtered out, safely fall back to
    # the first visible one -- set BEFORE the widget renders, so this can
    # never raise or show an empty selection.
    if st.session_state.get("door_selected_cycle") not in available_cycles:
        st.session_state["door_selected_cycle"] = available_cycles[0]

    selected_cycle = st.selectbox(
        "Inspect cycle",
        available_cycles,
        key="door_selected_cycle",
        format_func=lambda c: f"Cycle {c}: {filtered_queue.loc[filtered_queue['Cycle'] == c, 'prediction'].iloc[0]}",
    )
    row = filtered_queue.loc[filtered_queue["Cycle"] == selected_cycle].iloc[0]
    duration = (parse_door_datetime(row["end_time"]) - parse_door_datetime(row["start_time"])).total_seconds()

    st.markdown(f"**Cycle:** {row['Cycle']}")
    prediction_tone = "critical" if row["prediction"] == "Abnormal resistance" else "good"
    ui.render_status_pill(row["prediction"], prediction_tone)
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
# Row 4 -- Graphical Display (tabs, one view at a time)
# ---------------------------------------------------------------------------


def render_graphical_display(stream: pd.DataFrame, row: pd.Series) -> None:
    window = _cycle_evidence(stream, row["start_time"], row["end_time"])
    tab_labels = list(_SENSOR_TABS.keys()) + ["Overlay (current & position)"]
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




# ---------------------------------------------------------------------------
# Row 6 -- Downloads
# ---------------------------------------------------------------------------


def render_downloads(official: pd.DataFrame, detailed: pd.DataFrame) -> None:
    with ui.card("door_downloads_card"):
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
    with ui.card("door_upload_card"):
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

    # KPIs and the classification summary are computed from `official`/
    # `detailed` directly -- the queue filters introduced below only affect
    # what's displayed in the queue table and the selected-cycle panel.
    render_summary_cards(official, detailed)
    render_primary_analytics(detailed)

    queue_full = _build_cycle_queue(detailed)

    queue_col, selected_col = st.columns([0.65, 0.35])
    with queue_col:
        with ui.card("door_queue_card"):
            ui.render_section_heading("Cycle Review Queue", help_text=f"\"Needs review\" flags model confidence below {REVIEW_CONFIDENCE_CUTOFF:.0%} -- a prototype triage cue, not a safety threshold.")
            filtered_queue = render_cycle_filters(queue_full)
            render_cycle_queue_table(filtered_queue)
    with selected_col:
        with ui.card("door_selected_card"):
            ui.render_section_heading("Selected cycle")
            selected_row = render_selected_cycle_panel(filtered_queue, stream)

    if selected_row is not None:
        with ui.card("door_evidence_card"):
            ui.render_section_heading("Graphical Display", help_text="One sensor at a time -- choose a tab below. X-axis is time; y-axis units are shown where confirmed by the Info Kit.")
            st.markdown("<div class='nebula-caption'>View the sensor signals used to support the model’s prediction.</div>", unsafe_allow_html=True)
            render_graphical_display(stream, selected_row)


    ui.render_section_heading("Downloads", level="section")
    render_downloads(official, detailed)


__all__ = ["render_door_page", "_check_pipeline_status", "_cycle_evidence"]
