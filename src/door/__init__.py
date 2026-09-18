"""Door subsystem package.

Segmentation (`detect_cycles`) is not implemented yet -- see
planning/door_handoff.md. Everything else (loading, feature extraction,
model comparison, validation, read-only inspection) is functional now and
does not depend on segmentation being finished.
"""

from .config import OFFICIAL_OUTPUT_COLUMNS, VALID_DOOR_LABELS
from .features import build_door_feature_table, extract_cycle_features
from .model import (
    build_candidate_models,
    build_door_baseline_model,
    evaluate_door_model,
    load_door_pipeline,
    save_door_pipeline,
    train_door_model,
)
from .predict import predict_door_file
from .preprocess import load_door_csv, load_door_stream, parse_door_datetime
from .segment import detect_cycles, detect_door_segments, segment_door_stream, validate_segment_table

__all__ = [
    "VALID_DOOR_LABELS",
    "OFFICIAL_OUTPUT_COLUMNS",
    "load_door_csv",
    "load_door_stream",
    "parse_door_datetime",
    "detect_cycles",
    "detect_door_segments",
    "segment_door_stream",
    "validate_segment_table",
    "extract_cycle_features",
    "build_door_feature_table",
    "build_candidate_models",
    "train_door_model",
    "build_door_baseline_model",
    "evaluate_door_model",
    "save_door_pipeline",
    "load_door_pipeline",
    "predict_door_file",
]
