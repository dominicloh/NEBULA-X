"""Rail Corrugation preprocessing scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_rail_files(paths):
    """Load Rail Corrugation files in a manner that preserves original filenames."""
    file_list = [Path(item) for item in paths] if isinstance(paths, (list, tuple, set)) else [Path(paths)]
    loaded = []
    for file_path in file_list:
        if not file_path.exists():
            raise FileNotFoundError(f"Rail input file not found: {file_path}")
        if file_path.suffix.lower() != ".csv":
            raise ValueError(f"Rail Corrugation expects CSV files; received: {file_path.name}")
        frame = pd.read_csv(file_path)
        frame["file_id"] = file_path.name
        loaded.append(frame)
    return loaded


def validate_rail_schema(frame, required_columns: Iterable[str] | None = None):
    """Validate a loaded Rail dataframe against the known file-level schema contract."""
    if required_columns is not None:
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError("Rail schema missing required columns: " + ", ".join(missing))
    return True


__all__ = ["load_rail_files", "validate_rail_schema"]
