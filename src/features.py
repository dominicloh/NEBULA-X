"""Feature engineering for bogie temperature anomaly detection.

This module intentionally uses configuration-driven column names. It does not
assume a specific train or temperature schema before the official format is known.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def build_features(df, config):
    """Build temperature and trend features from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing telemetry records.
    config : dict
        Required configuration describing the exact field names used in the data.
        At minimum, it should provide the names for timestamps and temperature values,
        as well as any grouping columns such as train or component identifiers.

    Returns
    -------
    pandas.DataFrame
        Dataframe with engineered features.

    Notes
    -----
    Planned features include:
    - Current temperature
    - Rolling mean
    - Rolling standard deviation
    - Temperature difference
    - Rate of change
    - Deviation from component baseline
    - Paired-component difference, if possible
    - Persistence or consecutive anomaly duration

    The function should fail clearly if the required schema is not supplied.
    """
    if not isinstance(config, dict) or not config:
        raise ValueError(
            "Feature configuration is missing. Provide the required dataset schema "
            "before building model features."
        )

    required_keys = ["timestamp_column", "value_column"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(
            "Missing configuration keys for feature engineering: " + ", ".join(missing)
        )

    feature_frame = df.copy()
    timestamp_column = config["timestamp_column"]
    value_column = config["value_column"]

    if timestamp_column not in feature_frame.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in dataframe.")
    if value_column not in feature_frame.columns:
        raise ValueError(f"Value column '{value_column}' not found in dataframe.")

    if pd.api.types.is_datetime64_any_dtype(feature_frame[timestamp_column]):
        feature_frame = feature_frame.sort_values(timestamp_column).reset_index(drop=True)
    else:
        feature_frame[timestamp_column] = pd.to_datetime(feature_frame[timestamp_column], errors="coerce")
        feature_frame = feature_frame.sort_values(timestamp_column).reset_index(drop=True)

    group_columns = config.get("group_columns", [])
    if isinstance(group_columns, str):
        group_columns = [group_columns]

    if group_columns:
        for group_column in group_columns:
            if group_column not in feature_frame.columns:
                raise ValueError(f"Group column '{group_column}' not found in dataframe.")

    rolling_window = int(config.get("rolling_window", 5))
    if rolling_window <= 1:
        rolling_window = 5

    feature_frame["current_temperature"] = feature_frame[value_column].astype(float)
    feature_frame["rolling_mean"] = (
        feature_frame.groupby(group_columns)["current_temperature"].transform(
            lambda s: s.rolling(window=rolling_window, min_periods=1).mean()
        )
        if group_columns
        else feature_frame["current_temperature"].rolling(window=rolling_window, min_periods=1).mean()
    )
    feature_frame["rolling_std"] = (
        feature_frame.groupby(group_columns)["current_temperature"].transform(
            lambda s: s.rolling(window=rolling_window, min_periods=1).std(ddof=0)
        )
        if group_columns
        else feature_frame["current_temperature"].rolling(window=rolling_window, min_periods=1).std(ddof=0)
    )
    feature_frame["temperature_difference"] = feature_frame["current_temperature"].diff().fillna(0.0)
    feature_frame["rate_of_change"] = feature_frame["temperature_difference"] / (
        feature_frame[timestamp_column].diff().dt.total_seconds().fillna(1.0) / 60.0
    )
    feature_frame["rate_of_change"] = feature_frame["rate_of_change"].replace([float("inf"), float("-inf")], 0.0)

    baseline_columns = config.get("baseline_columns", [])
    if baseline_columns:
        for baseline_column in baseline_columns:
            if baseline_column not in feature_frame.columns:
                raise ValueError(f"Baseline column '{baseline_column}' not found in dataframe.")

    if group_columns:
        feature_frame["component_baseline"] = (
            feature_frame.groupby(group_columns)["current_temperature"].transform("mean")
        )
        feature_frame["deviation_from_baseline"] = (
            feature_frame["current_temperature"] - feature_frame["component_baseline"]
        )
    else:
        feature_frame["component_baseline"] = feature_frame["current_temperature"].mean()
        feature_frame["deviation_from_baseline"] = (
            feature_frame["current_temperature"] - feature_frame["component_baseline"]
        )

    # Optional paired-component difference: if two component groups are available, the
    # real schema can define a pair and calculate the relative behaviour between them.
    if "paired_component_column" in config:
        paired_column = config["paired_component_column"]
        if paired_column in feature_frame.columns:
            feature_frame["paired_component_difference"] = feature_frame[value_column].astype(float) - feature_frame[paired_column].astype(float)

    # Persistence is a model output concept; here we prepare a placeholder field that
    # the final logic can update once the exact event grouping rules are known.
    feature_frame["persistence_score"] = 0.0

    return feature_frame
