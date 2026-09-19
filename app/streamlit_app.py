"""Shared Streamlit app for the two official PS3 subsystems.

Rail Corrugation has a frozen, validated model (see
planning/model_experiment_log.md); Door has a frozen, Train-validated
segmentation + classification pipeline (see planning/door_handoff.md).
Each subsystem's UI/processing logic lives in its own module
(`app/rail_view.py`, `app/door_view.py`) so this shared file stays a thin,
low-conflict router between them, plus shared page chrome from `app/ui.py`.

This file is presentation/navigation only -- it never loads a model, reads
a sensor file, or computes a prediction itself. It is deliberately short:
per the dashboard's information-architecture rules, one-line task
descriptions replace paragraphs, and methodology/limitations live in a
collapsed expander at the bottom, not the main workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency for non-UI execution
    class _MissingStreamlit:
        def __getattr__(self, name):
            raise RuntimeError("streamlit is required to run the app.")

    st = _MissingStreamlit()

# Ensure "app.ui"/"app.rail_view"/"app.door_view" and "src...." imports
# resolve regardless of the working directory the app is launched from
# (e.g. `streamlit run app/streamlit_app.py` from the project root, per the README).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import ui  # noqa: E402

# Imported at module load, but guarded: a problem loading either subsystem's
# module (e.g. a missing model artifact) must never take down the other's page.
try:
    from app.rail_view import render_rail_page

    _RAIL_VIEW_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - defensive guard, see _rail_upload_and_analysis
    render_rail_page = None
    _RAIL_VIEW_IMPORT_ERROR = exc

try:
    from app.door_view import render_door_page

    _DOOR_VIEW_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - defensive guard, see _door_upload_and_analysis
    render_door_page = None
    _DOOR_VIEW_IMPORT_ERROR = exc

VALID_DOOR_LABELS = ("Normal", "Abnormal resistance")
VALID_RAIL_LABELS = ("Normal", "Side I", "Side II")

# One-line task descriptions only -- detailed schema/model info lives on
# each subsystem's own page (already shown live there) or in the bottom
# methodology expander, not repeated here as a paragraph.
TASK_LINES = {
    "Door Monitoring": "Detects each door-open/close cycle in a continuous stream and classifies it as Normal or Abnormal resistance.",
    "Rail Corrugation": "Classifies each one-second axle-box recording as Normal, Side I, or Side II corrugation.",
}

if hasattr(st, "set_page_config"):
    st.set_page_config(page_title="RailClarity | Train Condition Monitoring", layout="wide")

ui.inject_css()


def _door_upload_and_analysis():
    # All Door UI/processing logic lives in app/door_view.py (separate
    # module per TEAM_WORKFLOW.md, reusing src/door/*). If that module ever
    # fails to import, Rail Corrugation stays unaffected.
    if render_door_page is None:
        ui.render_info_banner(f"Door page failed to load: {_DOOR_VIEW_IMPORT_ERROR}", tone="critical")
        st.caption("Rail Corrugation is unaffected by this. See app/door_view.py and planning/door_handoff.md.")
        return
    render_door_page()


def _rail_upload_and_analysis():
    # All Rail UI/processing logic lives in app/rail_view.py (separate module
    # per TEAM_WORKFLOW.md, reusing the frozen model + src/rail_corrugation
    # pipeline from Stage 3). If that module ever fails to import (e.g. the
    # model artifact is missing on a fresh checkout), Door stays unaffected.
    if render_rail_page is None:
        ui.render_info_banner(f"Rail Corrugation page failed to load: {_RAIL_VIEW_IMPORT_ERROR}", tone="critical")
        st.caption("Door is unaffected by this. See app/rail_view.py and models/rail_corrugation_model.joblib.")
        return
    render_rail_page()


def _render_methodology_and_disclaimer():
    with st.expander("Methodology and limitations"):
        st.markdown(
            "- **Door**: temporal segment detection (`src/door/segment.py::detect_cycles`) followed by binary "
            "classification of each detected cycle. Official metric: IoU-weighted F1 over predicted segments."
        )
        st.markdown(
            "- **Rail Corrugation**: file-level three-class classification from axle-box vibration/shock features "
            "(`src/rail_corrugation/features.py`). Official metric: macro F1."
        )
        st.markdown(
            "- Neither pipeline is retrained by this app -- both load a frozen artifact from `models/`."
        )
        st.markdown(
            "- The app supports engineering review; it does not independently confirm a mechanical root cause."
        )
        st.markdown(
            "- Model confidence reflects the model's own probability estimate for its prediction -- it is not a "
            "guarantee of correctness and not a railway safety threshold."
        )
        st.markdown(
            "- Results describe the uploaded batch only, not the health of a whole train or fleet."
        )

    ui.render_footer(
        "This system supports engineering review and prioritisation. It does not replace formal "
        "engineering diagnosis, inspection procedures, or maintenance decisions."
    )


def main():
    ui.render_header()

    subsystem = ui.subsystem_selector(["Door Monitoring", "Rail Corrugation"])
    ui.render_task_line(TASK_LINES[subsystem])

    if subsystem == "Door Monitoring":
        _door_upload_and_analysis()
    else:
        _rail_upload_and_analysis()

    _render_methodology_and_disclaimer()


if __name__ == "__main__":
    main()
