"""Rail Corrugation page for the shared NEBULA X Streamlit app.

Kept as a separate module (per TEAM_WORKFLOW.md: "keep the Door and Rail
pipelines separate at the module level") so Rail-specific UI/validation code
never has to touch `app/streamlit_app.py` or `src/door/*`, reducing the
chance of merge conflicts with the Door owner's work.

`app/streamlit_app.py` should only need:

    from app.rail_view import render_rail_page
    ...
    render_rail_page()

This module NEVER retrains the model -- it only loads the frozen artifact at
`models/rail_corrugation_model.joblib` (via `st.cache_resource`, so it's
loaded once per server process, not once per upload) and reuses the exact
same feature extractor and prediction function that produced the official
`predictions/rail_predictions.csv` (Stage 3):
  - `src/rail_corrugation/features.py` for feature extraction
  - `src/rail_corrugation/predict.py` for loading the model + predicting
  - `src/common/validation.py` for the official output schema check
No second feature-extraction or prediction implementation lives here.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for a server-side app
import matplotlib.pyplot as plt
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
# Constants
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
    "Confidence categories (High / Moderate / Needs Review) are **prototype interface "
    "thresholds** chosen for this tool only -- they are **not railway safety limits**."
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
# Readable feature descriptions (Part D)
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
# Model loading (Part A.4/A.5) -- cached, never retrained
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
# Confidence categories (Part C)
# ---------------------------------------------------------------------------


def confidence_category(top_confidence: float) -> str:
    if top_confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "High confidence"
    if top_confidence >= MODERATE_CONFIDENCE_THRESHOLD:
        return "Moderate confidence"
    return "Needs Review"


# ---------------------------------------------------------------------------
# Upload expansion: plain CSVs and/or one ZIP of CSVs, in memory only
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
            # Only record the explanation once per duplicated name, even
            # though it applies to every occurrence.
            if file_id not in rejected:
                rejected[file_id] = [
                    f"Filename '{file_id}' was uploaded more than once ({counts[file_id]} times); "
                    "rejected to avoid an ambiguous prediction."
                ]
            continue
        kept.append((file_id, raw_bytes))
    return kept, rejected


# ---------------------------------------------------------------------------
# Per-file validation (Part B)
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
# Explainability (Part D)
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
# Signal evidence (Part E)
# ---------------------------------------------------------------------------


def render_signal_evidence(frame: pd.DataFrame, file_id: str) -> None:
    with st.expander("Inspect signal evidence (optional)"):
        st.caption(
            "Raw sensor evidence for this file. The chart below always uses the full "
            "10,000-sample recording (no downsampling needed at this size) -- and this "
            "view never changes what the model already predicted above."
        )

        col1, col2, col3 = st.columns(3)
        car = col1.selectbox("Car number", list(range(1, 9)), key=f"rail_car_{file_id}")
        position = col2.selectbox("Bearing position", list(range(1, 9)), key=f"rail_pos_{file_id}")
        signal_type = col3.selectbox("Signal type", ["Vibration", "Shock"], key=f"rail_sigtype_{file_id}")

        column_name = f"{signal_type} of bearing in position {position} of car {car}"
        if column_name not in frame.columns:
            st.error(f"Column not found in this file: {column_name}")
            return

        signal = frame[column_name].to_numpy(dtype=float)
        side = "Side I" if position in (1, 3, 5, 7) else "Side II"
        st.caption(f"**{column_name}** ({side})")

        time_seconds = np.arange(len(signal)) / SAMPLING_FREQUENCY_HZ
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.plot(time_seconds, signal, linewidth=0.5, color="#2563eb")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_title("Raw waveform (1 second)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        freqs = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLING_FREQUENCY_HZ)
        spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
        fig2, ax2 = plt.subplots(figsize=(8, 2.6))
        ax2.plot(freqs, spectrum, linewidth=0.6, color="#7c3aed")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_title("Frequency spectrum")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        if SPEED_COLUMN_NAME in frame.columns:
            speed_info = _rotating_speed_features(frame[SPEED_COLUMN_NAME].to_numpy(dtype=float), SAMPLING_FREQUENCY_HZ)
            st.metric("Estimated train speed", f"{speed_info['rotating_speed_estimated_kmh']:.1f} km/h")
            st.caption(
                "\"Estimated train speed\" is calculated from the raw speed-sensor pulses "
                "(90-tooth wheel sensor, 0.85 m wheel diameter, per the Info Kit) -- it is not "
                "a direct measurement. Shown for context only: it is not part of the official "
                "prediction CSV and does not change the frozen model's features."
            )


# ---------------------------------------------------------------------------
# Review queue (Part C)
# ---------------------------------------------------------------------------


def render_summary_cards(results_df: pd.DataFrame) -> None:
    st.markdown("#### Prediction summary")
    cols = st.columns(5)
    cols[0].metric("Files analysed", len(results_df))
    cols[1].metric("Normal", int((results_df["prediction"] == "Normal").sum()))
    cols[2].metric("Side I", int((results_df["prediction"] == "Side I").sum()))
    cols[3].metric("Side II", int((results_df["prediction"] == "Side II").sum()))
    cols[4].metric("Needs Review", int((results_df["confidence_category"] == "Needs Review").sum()))


_STATUS_BADGE = {
    "High confidence": ":green[High confidence]",
    "Moderate confidence": ":orange[Moderate confidence]",
    "Needs Review": ":red[Needs Review]",
}


def render_review_queue(results_df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("#### Engineer Review Queue")
    st.caption(CONFIDENCE_THRESHOLDS_NOTE)

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
            "Prediction confidence": display_queue["top_confidence"].map(lambda v: f"{v:.1%}"),
            "Attention score": display_queue["attention_score"].map(lambda v: f"{v:.1%}"),
            "Review status": display_queue["confidence_category"].map(lambda v: _STATUS_BADGE.get(v, v)),
        }
    )
    st.dataframe(display_table, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(display_queue)} of {len(queue)} file(s). Attention score = 1 - P(Normal).")
    return queue


# ---------------------------------------------------------------------------
# Selected-file explanation (Part D)
# ---------------------------------------------------------------------------


def render_selected_file_explanation(
    queue: pd.DataFrame,
    frames_by_file_id: dict,
    artifact: dict,
) -> None:
    st.markdown("#### Selected file")
    selected_file_id = st.selectbox("Choose a file to inspect", queue["file_id"].tolist(), key="rail_selected_file")
    row = queue.loc[queue["file_id"] == selected_file_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Filename:** {row['file_id']}")
        st.markdown(f"**Prediction:** {row['prediction']}")
        st.markdown(f"**Confidence:** {row['top_confidence']:.1%} ({row['confidence_category']})")
        st.markdown("**Data validation result:** Valid (passed all upload checks)")
    with col2:
        st.markdown(f"**Normal probability:** {row['confidence_normal']:.1%}")
        st.markdown(f"**Side I probability:** {row['confidence_side_i']:.1%}")
        st.markdown(f"**Side II probability:** {row['confidence_side_ii']:.1%}")
        st.markdown(f"**Attention score:** {row['attention_score']:.1%}")

    st.markdown("##### Why the model predicted this")
    # Recomputed on demand for just this one selected file, using the exact
    # same shared extractor prediction already used -- not a second
    # implementation, just a second (cheap) call for the one file being
    # inspected, so we don't need to carry every file's raw feature vector
    # around in memory for the whole session.
    feature_row = extract_rail_features(frames_by_file_id[selected_file_id]).reindex(artifact["feature_names"])
    contributions = compute_feature_contributions(artifact["pipeline"], feature_row, row["prediction"])

    if contributions is None:
        st.info("Feature-contribution explanation is only available for the deployed linear (Logistic Regression) model.")
    else:
        top_contributions = contributions.head(6)
        explanation_table = pd.DataFrame(
            {
                "Feature": [describe_feature_name(name) for name in top_contributions.index],
                "Contribution": top_contributions.values,
            }
        )
        st.dataframe(
            explanation_table.style.format({"Contribution": "{:+.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Contribution = standardised feature value x coefficient for the predicted class "
            f"({row['prediction']}). Positive values pushed the model toward this prediction."
        )

    st.warning(FEATURE_CONTRIBUTION_DISCLAIMER)

    render_signal_evidence(frames_by_file_id[selected_file_id], selected_file_id)


# ---------------------------------------------------------------------------
# Downloads (Part F)
# ---------------------------------------------------------------------------


def render_downloads(results_df: pd.DataFrame) -> None:
    st.markdown("#### Downloads")

    official_df = results_df[["file_id", "prediction"]].copy()
    validate_rail_predictions(official_df)  # reuse the shared schema check before offering the file
    st.download_button(
        "Download official rail_predictions.csv",
        data=official_df.to_csv(index=False).encode("utf-8"),
        file_name="rail_predictions.csv",
        mime="text/csv",
        help="Exact competition schema: file_id,prediction only, one row per valid uploaded file.",
    )

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
    st.download_button(
        "Download engineering review report (rail_engineering_review.csv)",
        data=review_df.to_csv(index=False).encode("utf-8"),
        file_name="rail_engineering_review.csv",
        mime="text/csv",
        help="Extra detail for engineering review only.",
    )
    st.caption(
        "`rail_engineering_review.csv` is a supplementary report for engineering review only -- "
        "it is **not** the official competition submission file. Only `rail_predictions.csv` "
        "(`file_id,prediction`) is scored."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render_rail_page() -> None:
    st.subheader("Rail Corrugation")
    st.caption(PRODUCT_MESSAGE)
    st.markdown(
        "Classifies each **one-second axle-box vibration/shock recording** as **Normal**, "
        "**Side I**, or **Side II** corrugation. This tool supports **engineering review and "
        "prioritisation** -- it does not, by itself, confirm a physical rail defect."
    )

    try:
        artifact = load_cached_rail_artifact()
    except FileNotFoundError as exc:
        st.error(f"Rail model not available: {exc}")
        return
    st.caption(
        f"Model loaded: {artifact.get('model_name', '?')} · "
        f"trained on {artifact.get('n_training_files', '?')} labelled files · "
        "loaded from disk, never retrained in this app."
    )
    validation_summary = artifact.get("validation_summary") or {}
    if validation_summary:
        with st.expander("Validation estimate (from training-data cross-validation only)"):
            st.write(
                f"Mean macro F1 (repeated 5-fold CV): "
                f"{validation_summary.get('mean_macro_f1_repeated_cv', 'n/a')}"
            )
            st.write(f"Out-of-fold macro F1 (fixed 5-fold): {validation_summary.get('oof_macro_f1_fixed_5fold', 'n/a')}")
            st.caption(validation_summary.get("note", ""))

    uploaded_files = st.file_uploader(
        "Upload Rail Corrugation file(s)",
        type=["csv", "zip"],
        accept_multiple_files=True,
        help="Upload one or more axle-box vibration/shock CSV files, or a single ZIP archive of CSV files.",
    )
    if not uploaded_files:
        st.info("Upload one or more Rail Corrugation CSV files (or a ZIP of CSV files) to begin.")
        return

    # --- Expand + validate (Part B) -------------------------------------
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

    st.markdown("#### Validation summary")
    cols = st.columns(4)
    cols[0].metric("Files received", n_received)
    cols[1].metric("Files valid", n_valid)
    cols[2].metric("Files rejected", n_rejected)
    cols[3].metric("Model ready", "Yes")

    for message in archive_problems:
        st.warning(message)

    rejected_rows = [r for r in validation_rows if not r["valid"]]
    if rejected_rows:
        with st.expander(f"Rejected files ({len(rejected_rows)}) -- click for details", expanded=False):
            for row in rejected_rows:
                st.markdown(f"**{row['file_id']}**")
                for problem in row["problems"]:
                    st.markdown(f"- {problem}")

    if not valid_frames:
        st.warning("No valid files to predict. Fix the issues above and re-upload.")
        return

    # --- Predict, one valid file at a time (Part A.6) --------------------
    # Reuses predict_rail_files exactly (Stage 3's own function) via the
    # (file_id, DataFrame) input shape it accepts. Looping one file at a
    # time means one unexpected failure can never take down the rest of an
    # already-validated batch.
    result_rows = []
    prediction_errors: list[tuple[str, str]] = []

    for file_id, frame in valid_frames.items():
        try:
            single_result = predict_rail_files([(file_id, frame)], artifact=artifact)
        except (ValueError, TypeError) as exc:
            prediction_errors.append((file_id, str(exc)))
            continue
        row_dict = single_result.iloc[0].to_dict()
        row_dict["attention_score"] = 1.0 - row_dict["confidence_normal"]
        row_dict["confidence_category"] = confidence_category(row_dict["top_confidence"])
        result_rows.append(row_dict)

    if prediction_errors:
        with st.expander(f"Files that passed validation but failed prediction ({len(prediction_errors)})", expanded=False):
            for file_id, message in prediction_errors:
                st.markdown(f"**{file_id}**: {message}")

    if not result_rows:
        st.warning("No predictions could be produced.")
        return

    results_df = pd.DataFrame(result_rows)

    render_summary_cards(results_df)
    queue = render_review_queue(results_df)
    render_selected_file_explanation(queue, valid_frames, artifact)
    render_downloads(results_df)

    st.divider()
    st.caption(RAIL_DISCLAIMER)


__all__ = ["render_rail_page"]
