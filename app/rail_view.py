"""Rail Corrugation page for the shared NEBULA X Streamlit app.

Kept as a separate module (per TEAM_WORKFLOW.md: "keep the Door and Rail
pipelines separate at the module level") so Rail-specific UI/validation code
never has to touch `app/streamlit_app.py` or `src/door/*`, reducing the
chance of merge conflicts with the Door owner's work.

This module NEVER retrains the model -- it only loads the frozen artifact at
`models/rail_corrugation_model.joblib` (via `st.cache_resource`, so it's
loaded once per server process, not once per upload) and reuses the exact
same feature extractor and prediction function that produced the official
`predictions/rail_predictions.csv` (Stage 3):
  - `src/rail_corrugation/features.py` for feature extraction
  - `src/rail_corrugation/predict.py` for loading the model + predicting
  - `src/common/validation.py` for the official output schema check
No second feature-extraction or prediction implementation lives here.

DASHBOARD LAYOUT NOTE: the functions below are grouped by the six dashboard
rows described in the design brief (KPI cards; batch overview + review
status; review queue + selected file; model explanation + signal profile;
signal evidence tabs; downloads). Every number/chart they draw comes from
`results_df` (built from real `predict_rail_files` output) or from feature
values `extract_rail_features` already computed -- nothing here invents a
metric.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover - lets non-UI code (e.g. tests) import this module
    class _MissingStreamlit:
        def __getattr__(self, name):
            raise RuntimeError("streamlit is required to render the Rail Corrugation page.")

    st = _MissingStreamlit()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import ui  # noqa: E402
from src.common.validation import VALID_RAIL_LABELS, validate_rail_predictions  # noqa: E402
from src.rail_corrugation.features import (  # noqa: E402
    SAMPLING_FREQUENCY_HZ,
    SPEED_COLUMN_NAME,
    _rotating_speed_features,  # reused deliberately -- see render_signal_evidence()
    extract_rail_features,
    validate_rail_columns,
)
from src.rail_corrugation.predict import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    load_model_artifact,
    predict_rail_files,
)

# ---------------------------------------------------------------------------
# Constants (unchanged from the previous UI pass -- logic, not presentation)
# ---------------------------------------------------------------------------

# Data-quality gate for UPLOADED files, separate from feature extraction
# itself. The frozen 74-feature extractor (features.py) will happily run on
# any correctly-shaped numeric frame; these checks exist so the app never
# silently predicts on a file that isn't really a valid 1-second, 10,000 Hz
# Rail Corrugation recording as confirmed by the Info Kit.
EXPECTED_ROW_COUNT = 10_000
EXPECTED_COLUMN_COUNT = 129

# Interface-only confidence bands (NOT a railway safety standard -- see the
# disclaimer shown next to every use of these labels in the UI).
HIGH_CONFIDENCE_THRESHOLD = 0.80
MODERATE_CONFIDENCE_THRESHOLD = 0.60

CONFIDENCE_THRESHOLDS_NOTE = (
    "High/Moderate/Needs review are prototype interface categories for this tool only -- not railway safety limits."
)

PRODUCT_MESSAGE = "From raw axle-box signals to a prioritised, explainable engineering review queue."

RAIL_DISCLAIMER = (
    "This prototype supports engineering review. A prediction does not independently "
    "confirm rail corrugation or replace railway inspection and safety procedures."
)

FEATURE_CONTRIBUTION_DISCLAIMER = (
    "Feature contributions explain what influenced the model output. They do not "
    "independently identify the physical root cause of a rail defect."
)

# ---------------------------------------------------------------------------
# Readable feature descriptions -- unchanged
# ---------------------------------------------------------------------------

_SPECIAL_FEATURE_DESCRIPTIONS = {
    "side_energy_diff": "Side I minus Side II overall vibration+shock energy (RMS-based)",
    "side_energy_ratio": "Side I divided by Side II overall energy (safe-epsilon ratio)",
    "max_side_energy": "The higher of the two sides' overall energy",
    "vibration_rms_diff": "Side I minus Side II vibration RMS difference",
    "shock_rms_diff": "Side I minus Side II shock RMS difference",
    "spectral_energy_diff": "Side I minus Side II spectral-energy difference",
    "s1_car_rms_max": "Highest per-car RMS on Side I (worst single car)",
    "s1_car_rms_std": "How much Side I's RMS varies from car to car",
    "s2_car_rms_max": "Highest per-car RMS on Side II (worst single car)",
    "s2_car_rms_std": "How much Side II's RMS varies from car to car",
    "rotating_speed_mean": "Average level of the raw speed-sensor toggle signal",
    "rotating_speed_std": "Variability of the raw speed-sensor toggle signal",
    "rotating_speed_toggle_count": "Number of speed-sensor tooth pulses counted in the file",
    "rotating_speed_estimated_kmh": "Estimated train speed (km/h) from the speed-sensor pulses",
}

_GROUP_PREFIXES = {
    "vib_s1": ("Vibration", "Side I"),
    "vib_s2": ("Vibration", "Side II"),
    "shock_s1": ("Shock", "Side I"),
    "shock_s2": ("Shock", "Side II"),
}

_STAT_DESCRIPTIONS = {
    "mean": "average level",
    "std": "variability (standard deviation)",
    "rms": "RMS (overall energy)",
    "abs_peak": "peak (largest absolute value)",
    "peak_to_peak": "peak-to-peak range",
    "kurtosis": "kurtosis (signal \"spikiness\")",
    "crest_factor": "crest factor (peak vs. RMS -- flags sharp shocks)",
    "spectral_energy": "spectral energy",
    "dominant_frequency": "dominant frequency",
    "spectral_centroid": "spectral centroid (frequency \"center of mass\")",
    "band_energy_low_ratio": "share of energy below 500 Hz",
    "band_energy_mid_ratio": "share of energy 500-2000 Hz",
    "band_energy_high_ratio": "share of energy above 2000 Hz",
}


def describe_feature_name(name: str) -> str:
    """Translate an engineered feature name into a short, readable phrase."""
    if name in _SPECIAL_FEATURE_DESCRIPTIONS:
        return _SPECIAL_FEATURE_DESCRIPTIONS[name]

    for prefix, (signal_type, side) in _GROUP_PREFIXES.items():
        if not name.startswith(prefix + "_"):
            continue
        remainder = name[len(prefix) + 1 :]
        if remainder.endswith("_avg"):
            stat_key, scope = remainder[: -len("_avg")], "averaged across 4 positions x 8 cars"
        elif remainder.endswith("_std"):
            stat_key, scope = remainder[: -len("_std")], "consistency across the 8 cars"
        else:
            stat_key, scope = remainder, ""
        stat_label = _STAT_DESCRIPTIONS.get(stat_key, stat_key.replace("_", " "))
        return f"{signal_type} {stat_label} on {side} ({scope})" if scope else f"{signal_type} {stat_label} on {side}"

    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Model loading -- cached, never retrained -- unchanged
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the frozen Rail Corrugation model...")
def _load_cached_rail_artifact(model_path_str: str) -> dict:
    """Load the fitted pipeline once per server process. Streamlit's
    cache_resource shares this across every user session/upload, so the
    model is never re-fit inside the app.
    """
    return load_model_artifact(Path(model_path_str))


def load_cached_rail_artifact() -> dict:
    return _load_cached_rail_artifact(str(DEFAULT_MODEL_PATH))


# ---------------------------------------------------------------------------
# Confidence categories -- unchanged
# ---------------------------------------------------------------------------


def confidence_category(top_confidence: float) -> str:
    if top_confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "High confidence"
    if top_confidence >= MODERATE_CONFIDENCE_THRESHOLD:
        return "Moderate confidence"
    return "Needs Review"


_STATUS_TONE = {
    "High confidence": "good",
    "Moderate confidence": "warning",
    "Needs Review": "info",
}


# ---------------------------------------------------------------------------
# Upload expansion: plain CSVs and/or one ZIP of CSVs, in memory only --
# unchanged
# ---------------------------------------------------------------------------


def _basename(name: str) -> str:
    """Strip any path parts a browser/zip entry might include -- we only
    ever use the bare filename as file_id, and never write anything to disk.
    """
    return Path(name.replace("\\", "/")).name


def expand_uploaded_files(uploaded_files) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Turn Streamlit's uploaded file objects into a flat list of
    (file_id, raw_csv_bytes), expanding any .zip archives entirely in
    memory (no extraction to disk, so a path-traversal-style entry name is
    harmless -- it's never used as a filesystem path).

    Returns (sources, whole_archive_problems). Per-CSV problems (encrypted
    entries, non-csv entries) are also folded into whole_archive_problems
    since they aren't tied to a file_id that ever became a source.
    """
    sources: list[tuple[str, bytes]] = []
    problems: list[str] = []

    for uploaded in uploaded_files:
        name = uploaded.name
        if name.lower().endswith(".zip"):
            try:
                archive = zipfile.ZipFile(io.BytesIO(uploaded.getvalue()))
            except zipfile.BadZipFile:
                problems.append(f"'{name}': not a valid ZIP archive (malformed or corrupted) -- skipped.")
                continue

            csv_entries = [info for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".csv")]
            if not csv_entries:
                problems.append(f"'{name}': ZIP archive contains no .csv files -- skipped.")
                continue

            for info in csv_entries:
                entry_id = _basename(info.filename)
                try:
                    raw_bytes = archive.read(info)
                except RuntimeError as exc:
                    # zipfile raises RuntimeError for password-protected entries.
                    problems.append(f"'{name}' -> '{entry_id}': appears encrypted/password-protected -- skipped ({exc}).")
                    continue
                sources.append((entry_id, raw_bytes))
        else:
            sources.append((_basename(name), uploaded.getvalue()))

    return sources, problems


def check_filenames(sources: list[tuple[str, bytes]]) -> tuple[list[tuple[str, bytes]], dict[str, list[str]]]:
    """Reject empty and duplicate filenames before any CSV parsing happens.

    A duplicate name is rejected outright (all occurrences), rather than
    silently keeping "the first one", so a prediction can never be
    attributed to an ambiguous source file.
    """
    counts: dict[str, int] = {}
    for file_id, _ in sources:
        counts[file_id] = counts.get(file_id, 0) + 1

    kept: list[tuple[str, bytes]] = []
    rejected: dict[str, list[str]] = {}
    for file_id, raw_bytes in sources:
        if not file_id or not file_id.strip():
            rejected.setdefault("(blank filename)", []).append("Uploaded file has an empty or blank filename.")
            continue
        if counts[file_id] > 1:
            if file_id not in rejected:
                rejected[file_id] = [
                    f"Filename '{file_id}' was uploaded more than once ({counts[file_id]} times); "
                    "rejected to avoid an ambiguous prediction."
                ]
            continue
        kept.append((file_id, raw_bytes))
    return kept, rejected


# ---------------------------------------------------------------------------
# Per-file validation -- unchanged
# ---------------------------------------------------------------------------


def _try_read_csv(raw_bytes: bytes) -> tuple[pd.DataFrame | None, str | None]:
    try:
        frame = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:  # pandas raises several distinct error types on bad CSVs
        return None, f"Could not be read as a CSV file: {exc}"
    if frame.empty:
        return None, "CSV file has no data rows."
    return frame, None


def validate_uploaded_frame(frame: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list = valid).

    Reuses `validate_rail_columns` (the same column-schema check the frozen
    feature extractor itself uses) instead of re-implementing column-name
    checks, and adds the extra upload-time data-quality checks the model
    doesn't need to make for itself (exact row count, numeric dtypes,
    missing/infinite values).
    """
    problems: list[str] = []

    if frame.shape[1] != EXPECTED_COLUMN_COUNT:
        problems.append(f"Expected {EXPECTED_COLUMN_COUNT} columns, found {frame.shape[1]}.")

    try:
        validate_rail_columns(frame.columns)
    except ValueError as exc:
        problems.append(str(exc))

    if frame.shape[0] != EXPECTED_ROW_COUNT:
        problems.append(
            f"Expected {EXPECTED_ROW_COUNT:,} rows (1 second at {SAMPLING_FREQUENCY_HZ:,} Hz), found {frame.shape[0]:,}."
        )

    non_numeric_columns = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric_columns:
        shown = non_numeric_columns[:5]
        suffix = " ..." if len(non_numeric_columns) > 5 else ""
        problems.append(f"Non-numeric value(s) found in column(s): {shown}{suffix}")

    if frame.isna().any().any():
        problems.append(f"Contains {int(frame.isna().sum().sum())} missing value(s).")

    numeric_frame = frame.select_dtypes(include="number")
    if not numeric_frame.empty:
        finite_check = numeric_frame.to_numpy(dtype=float, na_value=0.0)
        if np.isinf(finite_check).any():
            problems.append("Contains infinite value(s).")

    return problems


# ---------------------------------------------------------------------------
# Explainability -- unchanged
# ---------------------------------------------------------------------------


def compute_feature_contributions(pipeline, feature_row: pd.Series, predicted_label: str) -> pd.Series | None:
    """standardised_feature_value x coefficient_for_predicted_class.

    Only defined for a linear (scaler + Logistic Regression) pipeline, which
    is what the frozen model actually is. Returns None if the deployed model
    is ever swapped for something non-linear, so the UI can degrade
    gracefully instead of crashing.
    """
    named_steps = getattr(pipeline, "named_steps", None)
    if not named_steps or "scaler" not in named_steps or "clf" not in named_steps:
        return None
    classifier = named_steps["clf"]
    if not hasattr(classifier, "coef_") or predicted_label not in list(classifier.classes_):
        return None

    scaler = named_steps["scaler"]
    standardised = scaler.transform(feature_row.to_frame().T)[0]
    class_index = list(classifier.classes_).index(predicted_label)
    contributions = standardised * classifier.coef_[class_index]
    return pd.Series(contributions, index=feature_row.index).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Row 5 -- Signal evidence (tabs: Waveform / Frequency spectrum / Side comparison)
# ---------------------------------------------------------------------------


def _rail_signal(frame: pd.DataFrame, car: int, position: int, signal_type: str) -> tuple[np.ndarray, str, str] | None:
    column_name = f"{signal_type} of bearing in position {position} of car {car}"
    if column_name not in frame.columns:
        return None
    side = "Side I" if position in (1, 3, 5, 7) else "Side II"
    return frame[column_name].to_numpy(dtype=float), column_name, side


def render_signal_evidence(frame: pd.DataFrame, feature_row: pd.Series, file_id: str) -> None:
    """One evidence view at a time (tabs), per the dashboard layout rules --
    reuses the exact same signal columns/FFT math as before, just charted
    with Plotly (for hover) instead of static matplotlib images.
    """
    col1, col2, col3 = st.columns(3)
    car = col1.selectbox("Car number", list(range(1, 9)), key=f"rail_car_{file_id}")
    position = col2.selectbox("Bearing position", list(range(1, 9)), key=f"rail_pos_{file_id}")
    signal_type = col3.selectbox("Signal type", ["Vibration", "Shock"], key=f"rail_sigtype_{file_id}")

    resolved = _rail_signal(frame, car, position, signal_type)
    if resolved is None:
        st.error(f"Column not found in this file for car {car}, position {position}, {signal_type}.")
        return
    signal, column_name, side = resolved
    st.caption(f"**{column_name}** ({side})")

    tab_waveform, tab_spectrum, tab_side = st.tabs(["Waveform", "Frequency spectrum", "Side comparison"])

    with tab_waveform:
        time_seconds = np.arange(len(signal)) / SAMPLING_FREQUENCY_HZ
        ui.render_line_chart(time_seconds, signal, color="#168FE5", xlabel="Time (s)", ylabel="Acceleration (m/s²)")

    with tab_spectrum:
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLING_FREQUENCY_HZ)
        spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
        ui.render_line_chart(freqs, spectrum, color="#9254DE", xlabel="Frequency (Hz)", ylabel="Magnitude")

    with tab_side:
        # Reuses the already-computed feature values (no new computation) --
        # a direct visual of the Side I vs Side II comparison the model
        # itself was given as input features.
        pairs = [
            ("Vibration RMS", "vib_s1_rms_avg", "vib_s2_rms_avg"),
            ("Shock RMS", "shock_s1_rms_avg", "shock_s2_rms_avg"),
            ("Spectral energy", "vib_s1_spectral_energy_avg", "vib_s2_spectral_energy_avg"),
        ]
        labels, side_i_values, side_ii_values = [], [], []
        for label, key_i, key_ii in pairs:
            if key_i in feature_row.index and key_ii in feature_row.index:
                labels.append(label)
                side_i_values.append(float(feature_row[key_i]))
                side_ii_values.append(float(feature_row[key_ii]))
        if labels:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_bar(name="Side I", x=labels, y=side_i_values, marker_color=ui.CLASS_COLORS["Side I"])
            fig.add_bar(name="Side II", x=labels, y=side_ii_values, marker_color=ui.CLASS_COLORS["Side II"])
            fig.update_layout(barmode="group", height=260, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"side_energy_diff (Side I − Side II, RMS-based): {feature_row.get('side_energy_diff', float('nan')):+.4f}")

    if SPEED_COLUMN_NAME in frame.columns:
        speed_info = _rotating_speed_features(frame[SPEED_COLUMN_NAME].to_numpy(dtype=float), SAMPLING_FREQUENCY_HZ)
        st.caption(
            f"Estimated train speed: {speed_info['rotating_speed_estimated_kmh']:.1f} km/h "
            "(from speed-sensor pulses -- not part of the official prediction CSV)."
        )


# ---------------------------------------------------------------------------
# Row 1 -- KPI cards
# ---------------------------------------------------------------------------


def render_summary_cards(results_df: pd.DataFrame) -> None:
    n = len(results_df)
    n_normal = int((results_df["prediction"] == "Normal").sum())
    n_side_i = int((results_df["prediction"] == "Side I").sum())
    n_side_ii = int((results_df["prediction"] == "Side II").sum())
    n_review = int((results_df["confidence_category"] == "Needs Review").sum())

    def pct(count: int) -> str:
        return f"{count / n:.0%} of batch" if n else ""

    ui.render_kpi_row(
        [
            {"label": "Files analysed", "value": n, "tone": "neutral"},
            {"label": "Normal", "value": n_normal, "sub": pct(n_normal), "tone": "good"},
            {"label": "Side I", "value": n_side_i, "sub": pct(n_side_i), "tone": "rail"},
            {"label": "Side II", "value": n_side_ii, "sub": pct(n_side_ii), "tone": "info"},
            {"label": "Needs review", "value": n_review, "sub": pct(n_review), "tone": "warning"},
        ]
    )


# ---------------------------------------------------------------------------
# Row 2 -- Primary analytics (batch overview + review status)
# ---------------------------------------------------------------------------


def render_primary_analytics(results_df: pd.DataFrame) -> None:
    left, right = st.columns([2, 1])

    with left:
        with ui.card():
            ui.render_section_heading(
                "Batch classification overview",
                help_text="Counts (or, for a single file, class probabilities) come directly from the model's predictions.",
            )
            if len(results_df) == 1:
                row = results_df.iloc[0]
                ui.render_class_bar_chart(
                    ["Normal", "Side I", "Side II"],
                    [row["confidence_normal"], row["confidence_side_i"], row["confidence_side_ii"]],
                )
                st.caption("Only one file was analysed -- showing its class probabilities instead of a batch count.")
            else:
                counts = results_df["prediction"].value_counts()
                labels = ["Normal", "Side I", "Side II"]
                values = [int(counts.get(label, 0)) for label in labels]
                ui.render_class_bar_chart(labels, values)

    with right:
        with ui.card():
            ui.render_section_heading("Review status", help_text=CONFIDENCE_THRESHOLDS_NOTE)
            if len(results_df) == 1:
                row = results_df.iloc[0]
                st.markdown(f"**Predicted class:** {row['prediction']}")
                st.markdown(f"**Model confidence:** {row['top_confidence']:.1%}")
                st.markdown(f"**Attention score:** {row['attention_score']:.1%}")
                ui.render_probability_bars(
                    [
                        ("Normal", row["confidence_normal"]),
                        ("Side I", row["confidence_side_i"]),
                        ("Side II", row["confidence_side_ii"]),
                    ]
                )
            else:
                category_counts = results_df["confidence_category"].value_counts()
                labels = ["High confidence", "Moderate confidence", "Needs Review"]
                values = [int(category_counts.get(label, 0)) for label in labels]
                colors = [ui.TONE_COLORS["good"], ui.TONE_COLORS["warning"], ui.TONE_COLORS["info"]]
                ui.render_donut_chart(labels, values, colors=colors)


# ---------------------------------------------------------------------------
# Row 3 -- Engineer review (queue + selected file)
# ---------------------------------------------------------------------------


def render_review_queue(results_df: pd.DataFrame) -> pd.DataFrame:
    queue = results_df.sort_values("attention_score", ascending=False).reset_index(drop=True)
    queue.insert(0, "Priority", queue.index + 1)

    filter_choice = st.selectbox("Filter", ["All", "Normal", "Side I", "Side II", "Needs Review"], key="rail_queue_filter")
    if filter_choice == "Needs Review":
        display_queue = queue[queue["confidence_category"] == "Needs Review"]
    elif filter_choice != "All":
        display_queue = queue[queue["prediction"] == filter_choice]
    else:
        display_queue = queue

    display_table = pd.DataFrame(
        {
            "Priority": display_queue["Priority"],
            "Filename": display_queue["file_id"],
            "Prediction": display_queue["prediction"],
            "Model confidence": display_queue["top_confidence"].map(lambda v: f"{v:.1%}"),
            "Attention score": display_queue["attention_score"].map(lambda v: f"{v:.1%}"),
            # Plain text only (e.g. "High confidence") -- a plain st.dataframe
            # cell renders markdown/colour syntax like ":green[...]" literally
            # as text, so status colour lives in the legend pill row instead.
            "Review status": display_queue["confidence_category"],
        }
    )
    st.dataframe(display_table, use_container_width=True, hide_index=True, height=280)
    st.caption(f"{len(display_queue)} of {len(queue)} file(s) shown. Attention score = 1 − P(Normal).")
    return queue


def render_selected_file_panel(queue: pd.DataFrame, frames_by_file_id: dict, artifact: dict) -> tuple[str, pd.Series]:
    """Returns (selected_file_id, feature_row) so callers below (model
    explanation, signal evidence) can reuse the same selection/features
    without recomputing the selectbox or re-extracting features twice.
    """
    selected_file_id = st.selectbox("Choose a file to inspect", queue["file_id"].tolist(), key="rail_selected_file")
    row = queue.loc[queue["file_id"] == selected_file_id].iloc[0]

    st.markdown(f"**Filename:** {row['file_id']}")
    st.markdown(f"**Prediction:** {row['prediction']}")
    ui.render_status_pill(f"{row['top_confidence']:.1%} · {row['confidence_category']}", _STATUS_TONE.get(row["confidence_category"], "neutral"))
    st.markdown(f"**Attention score:** {row['attention_score']:.1%}")
    st.markdown("**Validation status:** Valid (passed all upload checks)")
    ui.render_probability_bars(
        [
            ("Normal", row["confidence_normal"]),
            ("Side I", row["confidence_side_i"]),
            ("Side II", row["confidence_side_ii"]),
        ]
    )

    # Recomputed on demand for just this one selected file, using the exact
    # same shared extractor already used for prediction -- not a second
    # implementation, just a second (cheap) call for the one file being
    # inspected, so we don't carry every file's raw feature vector in memory.
    feature_row = extract_rail_features(frames_by_file_id[selected_file_id]).reindex(artifact["feature_names"])
    return selected_file_id, feature_row


# ---------------------------------------------------------------------------
# Row 4 -- Model explanation (why this prediction + signal profile)
# ---------------------------------------------------------------------------


def render_model_explanation(row: pd.Series, feature_row: pd.Series, artifact: dict) -> None:
    left, right = st.columns(2)

    with left:
        with ui.card():
            ui.render_section_heading("Why this prediction", help_text=FEATURE_CONTRIBUTION_DISCLAIMER)
            contributions = compute_feature_contributions(artifact["pipeline"], feature_row, row["prediction"])
            if contributions is None:
                st.info("Feature-contribution explanation is only available for the deployed linear (Logistic Regression) model.")
            else:
                top = contributions.head(6)
                readable_labels = [describe_feature_name(name) for name in top.index]
                ui.render_horizontal_bar(readable_labels, list(top.values), hover_text=list(top.index))
                with st.expander("Exact feature values"):
                    st.dataframe(
                        pd.DataFrame({"Feature": top.index, "Readable name": readable_labels, "Contribution": top.values}).style.format({"Contribution": "{:+.3f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
            st.caption(FEATURE_CONTRIBUTION_DISCLAIMER)

    with right:
        with ui.card():
            ui.render_section_heading("Signal profile", help_text="Summarises values already computed by the feature extractor for this file.")
            speed_kmh = feature_row.get("rotating_speed_estimated_kmh")
            vibration_energy = (feature_row.get("vib_s1_spectral_energy_avg", 0) + feature_row.get("vib_s2_spectral_energy_avg", 0)) / 2
            shock_energy = (feature_row.get("shock_s1_spectral_energy_avg", 0) + feature_row.get("shock_s2_spectral_energy_avg", 0)) / 2
            dominant_category = "Vibration" if vibration_energy >= shock_energy else "Shock"
            side_diff = feature_row.get("side_energy_diff", 0.0)
            dominant_side = "Side I" if side_diff > 0 else ("Side II" if side_diff < 0 else "Balanced")

            ui.render_kpi_row(
                [
                    {"label": "Est. rotating speed", "value": f"{speed_kmh:.1f} km/h" if speed_kmh is not None else "n/a", "tone": "info"},
                    {"label": "Dominant signal", "value": dominant_category, "tone": "rail"},
                    {"label": "Higher-energy side", "value": dominant_side, "sub": f"Δ {side_diff:+.3f}", "tone": "neutral"},
                ]
            )
            st.caption(f"Validation: {EXPECTED_COLUMN_COUNT}/{EXPECTED_COLUMN_COUNT} columns, {EXPECTED_ROW_COUNT:,}/{EXPECTED_ROW_COUNT:,} rows, no missing/infinite values.")


# ---------------------------------------------------------------------------
# Row 6 -- Downloads
# ---------------------------------------------------------------------------


def render_downloads(results_df: pd.DataFrame) -> None:
    official_df = results_df[["file_id", "prediction"]].copy()
    validate_rail_predictions(official_df)  # reuse the shared schema check before offering the file

    review_df = results_df[
        [
            "file_id",
            "prediction",
            "confidence_normal",
            "confidence_side_i",
            "confidence_side_ii",
            "top_confidence",
            "confidence_category",
            "attention_score",
        ]
    ].copy()

    with ui.card():
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download official rail_predictions.csv",
                data=official_df.to_csv(index=False).encode("utf-8"),
                file_name="rail_predictions.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Download engineering review report",
                data=review_df.to_csv(index=False).encode("utf-8"),
                file_name="rail_engineering_review.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.caption("Only `rail_predictions.csv` (`file_id,prediction`) is scored -- the review report is supplementary.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render_rail_page() -> None:
    try:
        artifact = load_cached_rail_artifact()
    except FileNotFoundError as exc:
        ui.render_info_banner(f"Rail model not available: {exc}", tone="critical")
        return

    with ui.card():
        st.caption("Upload one or more axle-box vibration/shock CSV files, or a ZIP archive of CSV files.")
        uploaded_files = st.file_uploader(
            "Rail Corrugation file(s)",
            type=["csv", "zip"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    if not uploaded_files:
        ui.render_empty_state("Upload one or more Rail Corrugation CSV files (or a ZIP of CSV files) to begin.")
        return

    # --- Validating/processing state (real stages, no fake percentages) --
    with st.status("Validating files...", expanded=False) as status:
        sources, archive_problems = expand_uploaded_files(uploaded_files)
        sources, filename_problems = check_filenames(sources)

        validation_rows = []
        valid_frames: dict[str, pd.DataFrame] = {}
        for file_id, raw_bytes in sources:
            frame, read_error = _try_read_csv(raw_bytes)
            if read_error:
                validation_rows.append({"file_id": file_id, "valid": False, "problems": [read_error]})
                continue
            problems = validate_uploaded_frame(frame)
            if problems:
                validation_rows.append({"file_id": file_id, "valid": False, "problems": problems})
                continue
            validation_rows.append({"file_id": file_id, "valid": True, "problems": []})
            valid_frames[file_id] = frame
        for file_id, problems in filename_problems.items():
            validation_rows.append({"file_id": file_id, "valid": False, "problems": problems})

        n_received = len(validation_rows)
        n_valid = sum(1 for r in validation_rows if r["valid"])
        n_rejected = n_received - n_valid

        result_rows = []
        prediction_errors: list[tuple[str, str]] = []
        if valid_frames:
            ui.safe_status_update(status, label=f"Generating predictions (0/{len(valid_frames)})...")
            for index, (file_id, frame) in enumerate(valid_frames.items(), start=1):
                try:
                    single_result = predict_rail_files([(file_id, frame)], artifact=artifact)
                except (ValueError, TypeError) as exc:
                    prediction_errors.append((file_id, str(exc)))
                    continue
                row_dict = single_result.iloc[0].to_dict()
                row_dict["attention_score"] = 1.0 - row_dict["confidence_normal"]
                row_dict["confidence_category"] = confidence_category(row_dict["top_confidence"])
                result_rows.append(row_dict)
                ui.safe_status_update(status, label=f"Generating predictions ({index}/{len(valid_frames)})...")
        ui.safe_status_update(status, label="Done", state="complete")

    # --- Validation summary (KPI-style, not a paragraph) ------------------
    ui.render_kpi_row(
        [
            {"label": "Files received", "value": n_received, "tone": "neutral"},
            {"label": "Files valid", "value": n_valid, "tone": "good"},
            {"label": "Files rejected", "value": n_rejected, "tone": "warning" if n_rejected else "neutral"},
            {"label": "Model ready", "value": "Yes", "tone": "rail"},
        ]
    )

    for message in archive_problems:
        ui.render_info_banner(message, tone="warning")

    rejected_rows = [r for r in validation_rows if not r["valid"]]
    if rejected_rows:
        with st.expander(f"Rejected files ({len(rejected_rows)}) -- click for details", expanded=False):
            for row in rejected_rows:
                st.markdown(f"**{row['file_id']}**")
                for problem in row["problems"]:
                    st.markdown(f"- {problem}")

    if not valid_frames:
        ui.render_info_banner("No valid files to predict. Fix the issues above and re-upload.", tone="warning")
        return

    if prediction_errors:
        with st.expander(f"Files that passed validation but failed prediction ({len(prediction_errors)})", expanded=False):
            for file_id, message in prediction_errors:
                st.markdown(f"**{file_id}**: {message}")

    if not result_rows:
        ui.render_info_banner("No predictions could be produced.", tone="warning")
        return

    results_df = pd.DataFrame(result_rows)

    # --- Results dashboard grid -------------------------------------------
    render_summary_cards(results_df)
    render_primary_analytics(results_df)

    queue_col, selected_col = st.columns([0.65, 0.35])
    with queue_col:
        with ui.card():
            ui.render_section_heading("Engineer review queue")
            queue = render_review_queue(results_df)
    with selected_col:
        with ui.card():
            ui.render_section_heading("Selected file")
            selected_file_id, feature_row = render_selected_file_panel(queue, valid_frames, artifact)

    selected_row = queue.loc[queue["file_id"] == selected_file_id].iloc[0]
    render_model_explanation(selected_row, feature_row, artifact)

    with ui.card():
        ui.render_section_heading("Signal evidence", help_text="One evidence view at a time -- choose a tab below.")
        render_signal_evidence(valid_frames[selected_file_id], feature_row, selected_file_id)

    ui.render_section_heading("Downloads", level="section")
    render_downloads(results_df)

    ui.render_footer(RAIL_DISCLAIMER)


__all__ = ["render_rail_page"]
