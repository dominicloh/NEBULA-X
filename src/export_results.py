"""Export trained model output and dashboard-ready summary data.

The official prediction CSV format must match the required Door or Rail Corrugation
schema exactly. No assumptions are made about unconfirmed final dataset details.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.validation import validate_door_predictions, validate_rail_predictions


def export_official_predictions(predictions, output_path, config):
    """Export the official prediction CSV for a configured subsystem schema."""
    if not isinstance(config, dict) or not config:
        raise ValueError(
            "Prediction export configuration is missing. Provide the official CSV "
            "schema before exporting predictions."
        )

    required_keys = ["required_columns", "output_columns", "subsystem"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(
            "Missing export schema information: " + ", ".join(missing)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(predictions, pd.DataFrame):
        export_frame = predictions.copy()
    else:
        export_frame = pd.DataFrame(predictions)

    required_columns = config["required_columns"]
    missing_columns = [column for column in required_columns if column not in export_frame.columns]
    if missing_columns:
        raise ValueError(
            "Official predictions are missing required columns: " + ", ".join(missing_columns)
        )

    final_columns = config.get("output_columns", list(export_frame.columns))
    export_frame = export_frame.loc[:, [column for column in final_columns if column in export_frame.columns]]
    export_frame.to_csv(output_path, index=False)

    subsystem = config["subsystem"]
    if subsystem == "door":
        validate_door_predictions(export_frame)
    elif subsystem == "rail":
        validate_rail_predictions(export_frame)
    else:
        raise ValueError(f"Unsupported subsystem for export validation: {subsystem}")

    return output_path


def export_dashboard_json(summary, alerts, timeseries, output_path):
    """Export JSON in a dashboard-ready format with Python-safe data types."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "generated_at": None,
            "model": "Not yet trained",
            "data_status": "Placeholder",
        },
        "summary": summary or {
            "trains_requiring_attention": 0,
            "critical_anomalies": 0,
            "developing_warnings": 0,
            "highest_priority_component": None,
        },
        "alerts": alerts or [],
        "timeseries": timeseries or [],
    }

    def convert_value(value):
        if value is None:
            return None
        if isinstance(value, (dict, list, str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except TypeError:
                pass
        return str(value)

    cleaned_payload = json.loads(json.dumps(payload, default=convert_value))
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(cleaned_payload, handle, indent=2)
        handle.write("\n")

    return output_path
