"""Door subsystem package scaffold."""

from .features import (
    build_door_feature_table,
    extract_door_segment_features,
)
from .model import (
    build_door_baseline_model,
    evaluate_door_model,
    load_door_pipeline,
    save_door_pipeline,
)
from .predict import predict_door_segments
from .preprocess import load_door_stream
from .segment import detect_door_segments, segment_door_stream

__all__ = [
    "load_door_stream",
    "detect_door_segments",
    "segment_door_stream",
    "extract_door_segment_features",
    "build_door_feature_table",
    "build_door_baseline_model",
    "evaluate_door_model",
    "save_door_pipeline",
    "load_door_pipeline",
    "predict_door_segments",
]
