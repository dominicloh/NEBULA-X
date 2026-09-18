"""Door per-cycle feature extraction.

WHY this works even though segmentation isn't implemented yet
----------------------------------------------------------------
`extract_cycle_features()` only needs a table of segment boundaries
(start_time, end_time) -- it doesn't care whether those boundaries came
from `segment.py::detect_cycles` (not implemented yet) or from the
*official* `Train_Segments_Answer.csv`. That means you can already train
and evaluate a classifier (see `train.py`) using the ground-truth segments,
while segmentation is still being built. This is not a shortcut around the
real problem -- producing `door_predictions.csv` from `Test.csv` still
needs `detect_cycles` finished -- it just lets Stage 2 (classification)
start immediately instead of waiting on Stage 1 (segmentation).

Each detected/labelled cycle becomes exactly ONE feature row, the same
"one example = one row" principle used in
`src/rail_corrugation/features.py`, because the label (Normal / Abnormal
resistance) is defined per cycle, not per raw sensor reading.

Only confirmed signal columns (src/door/config.py::SIGNAL_COLUMNS) are
used -- nothing here depends on an unconfirmed column or unit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402
from src.door.preprocess import PARSED_DATETIME_COLUMN, parse_door_datetime  # noqa: E402

# Small constant so a slope/rate feature never divides by zero on a
# zero-duration segment.
_EPSILON = 1e-9


def _ensure_parsed_timestamp(value):
    """Accept either an already-parsed pandas Timestamp or a raw Door
    timestamp string (e.g. from Train_Segments_Answer.csv) and return a
    Timestamp either way.
    """
    if isinstance(value, pd.Timestamp):
        return value
    return parse_door_datetime(str(value))


def _compute_column_stats(values: np.ndarray, duration_seconds: float) -> dict:
    """Compact, deterministic stats for one signal column within one cycle.

    Works the same way for a continuous-looking signal (e.g. motor current)
    and a 0/1 flag column (e.g. Close command) -- for a flag column, "mean"
    is simply the fraction of the cycle spent at 1, which is still
    meaningful, even though "slope"/"energy" are less informative there.
    """
    mean_val = float(values.mean())
    std_val = float(values.std())
    min_val = float(values.min())
    max_val = float(values.max())
    peak_to_peak = max_val - min_val
    energy = float(np.sum(np.square(values)))
    slope = float((values[-1] - values[0]) / max(duration_seconds, _EPSILON))
    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "peak_to_peak": peak_to_peak,
        "energy": energy,
        "slope": slope,
    }


_STAT_NAMES = ("mean", "std", "min", "max", "peak_to_peak", "energy", "slope")


def extract_cycle_features(dataframe: pd.DataFrame, segments: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Turn one or many detected/labelled cycles into feature row(s).

    Parameters
    ----------
    dataframe:
        A Door stream loaded by `src.door.preprocess.load_door_csv` (must
        have the parsed `_datetime_parsed` column).
    segments:
        A DataFrame with at least `start_time` and `end_time` columns
        (either raw Door-format strings or already-parsed Timestamps) --
        this can be `Train_Segments_Answer.csv` (loaded with `pandas.read_csv`)
        or the output of `segment.py::detect_cycles` once implemented. If a
        `segment_id` column is present it becomes the output's index
        (never a feature value) -- otherwise segments are numbered 0..N-1.

    Returns
    -------
    A DataFrame with one feature row per segment (index = segment_id or
    position), or a single Series if `segments` has exactly one row.

    Raises a clear ValueError if a segment's window contains zero rows
    (e.g. because the boundaries don't actually fall inside `dataframe`) --
    it never silently returns a fake/empty feature row.
    """
    if PARSED_DATETIME_COLUMN not in dataframe.columns:
        raise ValueError(
            f"extract_cycle_features expects a DataFrame with a {PARSED_DATETIME_COLUMN!r} column -- "
            "load it with src.door.preprocess.load_door_csv first."
        )
    required_segment_columns = {"start_time", "end_time"}
    missing = required_segment_columns - set(segments.columns)
    if missing:
        raise ValueError(f"'segments' is missing required column(s): {sorted(missing)}")

    has_segment_id = "segment_id" in segments.columns
    rows: dict = {}

    for position, segment_row in segments.reset_index(drop=True).iterrows():
        start = _ensure_parsed_timestamp(segment_row["start_time"])
        end = _ensure_parsed_timestamp(segment_row["end_time"])
        if start >= end:
            raise ValueError(f"Segment has start_time >= end_time: start={start}, end={end}")

        window = dataframe[(dataframe[PARSED_DATETIME_COLUMN] >= start) & (dataframe[PARSED_DATETIME_COLUMN] <= end)]
        if window.empty:
            raise ValueError(
                f"No rows found between {start} and {end} -- this segment's boundaries don't fall "
                "inside the given Door stream. Fake/empty features are never returned."
            )

        duration_seconds = (end - start).total_seconds()
        features: dict = {"duration_seconds": duration_seconds, "n_rows": int(len(window))}
        for column in config.SIGNAL_COLUMNS:
            if column not in window.columns:
                continue  # defensive: only compute features for columns that actually exist
            stats = _compute_column_stats(window[column].to_numpy(dtype=float), duration_seconds)
            safe_name = column.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
            for stat_name in _STAT_NAMES:
                features[f"{safe_name}_{stat_name}"] = stats[stat_name]

        key = segment_row["segment_id"] if has_segment_id else position
        rows[key] = pd.Series(features)

    result = pd.DataFrame.from_dict(rows, orient="index")
    if len(result) == 1:
        return result.iloc[0]
    return result


# Backward-compatible alias -- earlier scaffold code imported this name.
def build_door_feature_table(segments: pd.DataFrame, stream: pd.DataFrame, feature_config: dict | None = None) -> pd.DataFrame:
    """Deprecated argument order/name for `extract_cycle_features` -- kept
    for backward compatibility. `feature_config` is unused (feature
    definitions are fixed and deterministic; see module docstring). Named
    `feature_config`, not `config`, so it never shadows the
    `src.door.config` module imported at the top of this file.
    """
    return extract_cycle_features(stream, segments)


__all__ = ["extract_cycle_features", "build_door_feature_table"]
