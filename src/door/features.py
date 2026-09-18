"""Deterministic features for each automatically detected Door cycle.

The four sensor channels are motor current, motor voltage, motor back EMF,
and door leaf position. For each channel, mean, standard deviation, minimum,
maximum, range, squared-signal energy, and first-to-last slope describe its
level, variation, extremes, total magnitude, and overall direction. Cycle
duration is elapsed time between its first and last original readings.
Absolute time, cycle position, answer columns, and constant state fields are
never model inputs. All features are available during live prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.door.preprocess import PARSED_DATETIME_COLUMN, parse_door_datetime

SENSOR_COLUMNS = (
    "Motor current(mA)",
    "Motor Voltage(10mV)",
    "Motor electrodynamic force",
    "Door leaf position",
)
STAT_NAMES = ("mean", "std", "min", "max", "range", "energy", "slope")


def _name(column: str) -> str:
    return column.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")


FEATURE_NAMES = ("duration_seconds",) + tuple(
    f"{_name(column)}_{stat}" for column in SENSOR_COLUMNS for stat in STAT_NAMES
)


def extract_cycle_features(dataframe: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Return one finite feature row per cycle, always in ``FEATURE_NAMES`` order."""
    required = {PARSED_DATETIME_COLUMN, *SENSOR_COLUMNS}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Door feature stream is missing columns: {sorted(missing)}")
    if not {"start_time", "end_time"}.issubset(segments.columns) or segments.empty:
        raise ValueError("Door segments require nonempty start_time and end_time columns.")
    if dataframe[PARSED_DATETIME_COLUMN].isna().any():
        raise ValueError("Door feature stream has invalid timestamps.")

    rows = []
    ids = []
    timestamps = dataframe[PARSED_DATETIME_COLUMN]
    for position, segment in segments.reset_index(drop=True).iterrows():
        start = parse_door_datetime(str(segment["start_time"]))
        end = parse_door_datetime(str(segment["end_time"]))
        if start >= end:
            raise ValueError("Door segment start_time must precede end_time.")
        window = dataframe.loc[timestamps.between(start, end)]
        if len(window) < 2 or window[PARSED_DATETIME_COLUMN].iloc[[0, -1]].tolist() != [start, end]:
            raise ValueError("Door segment boundaries must identify at least two exact source readings.")
        duration = (end - start).total_seconds()
        values = [duration]
        for column in SENSOR_COLUMNS:
            try:
                signal = window[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Door sensor {column!r} contains nonnumeric values.") from exc
            if not np.isfinite(signal).all():
                raise ValueError(f"Door sensor {column!r} contains missing or infinite values.")
            values.extend((
                float(signal.mean()), float(signal.std()), float(signal.min()),
                float(signal.max()), float(np.ptp(signal)),
                float(np.square(signal).sum()), float((signal[-1] - signal[0]) / duration),
            ))
        rows.append(values)
        ids.append(segment["segment_id"] if "segment_id" in segments.columns else position)
    result = pd.DataFrame(rows, index=ids, columns=FEATURE_NAMES, dtype=float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Door feature extraction produced missing or infinite values.")
    return result


def build_door_feature_table(segments: pd.DataFrame, stream: pd.DataFrame, feature_config: dict | None = None) -> pd.DataFrame:
    """Compatibility alias for the original argument order."""
    return extract_cycle_features(stream, segments)


__all__ = ["FEATURE_NAMES", "SENSOR_COLUMNS", "STAT_NAMES", "extract_cycle_features", "build_door_feature_table"]
