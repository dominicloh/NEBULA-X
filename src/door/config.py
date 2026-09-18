"""Door subsystem configuration -- confirmed values and paths ONLY.

Every value in this file was checked directly against the official files in
`organiser-materials/PS3/02_Datasets/Door/` and
`organiser-materials/PS3/03_References/Door/` (see planning/door_handoff.md
for how each fact was confirmed). Anything NOT yet confirmed is left unset
(`None`) or clearly marked as an observation/placeholder -- never guessed
into a hard constant.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The organiser materials repo is a sibling folder of this team repo, per
# the confirmed Windows folder layout (same convention as
# src/rail_corrugation/inspect_data.py's DEFAULT_DATASET_DIR).
DEFAULT_DATASET_DIR = PROJECT_ROOT.parent / "organiser-materials" / "PS3" / "02_Datasets" / "Door"

TRAIN_FILENAME = "Train.csv"
TEST_FILENAME = "Test.csv"
TRAIN_SEGMENTS_ANSWER_FILENAME = "Train_Segments_Answer.csv"

# --- Confirmed from Train.csv / Test.csv's own header row (both match) -----
DATETIME_COLUMN = "Datetime"
SIGNAL_COLUMNS = (
    "Motor current(mA)",
    "Motor Voltage(10mV)",
    "Motor electrodynamic force",
    "Door opening time(.1s)",
    "Door closing time(.1s)",
    "Close command",
    "Open command",
    "DCSR",
    "DCSL",
    "DLSR",
    "DLSL",
    "Door Opened",
    "Door Locked",
    "Door is opening",
    "Door is closing",
    "Door leaf position",
)
EXPECTED_COLUMNS = (DATETIME_COLUMN,) + SIGNAL_COLUMNS
EXPECTED_COLUMN_COUNT = len(EXPECTED_COLUMNS)  # 17, confirmed

# --- Confirmed from Train_Segments_Answer.csv's own header row -------------
SEGMENT_ANSWER_COLUMNS = ("segment_id", "start_time", "end_time", "operation", "status", "n_rows")

# Confirmed from Train_Segments_Answer.csv's own "operation" column values.
# Informational only -- not something the model predicts.
VALID_OPERATIONS = ("Open", "Close")

# --- Confirmed official output contract -------------------------------------
# Source: Door_Subsystem_Info_Kit.md Section 3, and
# organiser-materials/PS3/04_Example_Submission/door_predictions.csv
OFFICIAL_OUTPUT_FILENAME = "door_predictions.csv"
OFFICIAL_OUTPUT_COLUMNS = ("start_time", "end_time", "prediction")

# Confirmed exact label spelling (Info Kit + Train_Segments_Answer.csv's own
# "status" column values).
VALID_DOOR_LABELS = ("Normal", "Abnormal resistance")

# --- Observed (NOT officially documented) -----------------------------------
# Row-to-row Datetime gaps within a dense block are ~20ms in both Train.csv
# and Test.csv (measured directly from the files -- see
# planning/door_handoff.md Section 2). This is NOT stated anywhere in the
# Info Kit as an official sampling rate. Re-confirm with inspect_data.py
# rather than trusting this blindly if the data ever changes.
OBSERVED_DENSE_SAMPLE_INTERVAL_MS = 20

# A "large" gap between consecutive Datetime values that likely marks a
# boundary between cycles rather than noise within one. This threshold
# (100ms = 5x the observed dense interval) is a starting point for
# `segment.py`'s segmentation approach, not an organiser-confirmed value --
# tune and justify it against Train_Segments_Answer.csv yourself.
CANDIDATE_GAP_THRESHOLD_MS = 100

# --- Explicitly unset -- do not invent these ---------------------------------
# The Info Kit deliberately leaves the segmentation rule as an open design
# decision ("think about what actually changes at a cycle boundary"). Do not
# assign this a value until segment.py's approach is implemented and
# validated against Train_Segments_Answer.csv.
SEGMENTATION_CONFIG = None

# --- Where a trained artifact should live once training is implemented -----
# Mirrors models/rail_corrugation_model.joblib's convention.
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "models" / "door_model.joblib"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "output" / "door" / "baseline"
PREDICTIONS_OUTPUT_PATH = PROJECT_ROOT / "predictions" / OFFICIAL_OUTPUT_FILENAME

# Fixed everywhere a random seed is needed, for reproducibility (same
# convention as src/rail_corrugation/model.py::RANDOM_STATE).
RANDOM_STATE = 42


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_DATASET_DIR",
    "TRAIN_FILENAME",
    "TEST_FILENAME",
    "TRAIN_SEGMENTS_ANSWER_FILENAME",
    "DATETIME_COLUMN",
    "SIGNAL_COLUMNS",
    "EXPECTED_COLUMNS",
    "EXPECTED_COLUMN_COUNT",
    "SEGMENT_ANSWER_COLUMNS",
    "VALID_OPERATIONS",
    "OFFICIAL_OUTPUT_FILENAME",
    "OFFICIAL_OUTPUT_COLUMNS",
    "VALID_DOOR_LABELS",
    "OBSERVED_DENSE_SAMPLE_INTERVAL_MS",
    "CANDIDATE_GAP_THRESHOLD_MS",
    "SEGMENTATION_CONFIG",
    "MODEL_ARTIFACT_PATH",
    "MODEL_OUTPUT_DIR",
    "PREDICTIONS_OUTPUT_PATH",
    "RANDOM_STATE",
]
