"""Rail Corrugation inference interface."""

from __future__ import annotations

from pathlib import Path


def predict_rail_files(file_paths, config: dict | None = None):
    """Predict Rail Corrugation class for each input file."""
    if config is None:
        raise RuntimeError(
            "Rail Corrugation inference is not ready. The file-level model and schema are not configured yet."
        )
    file_list = [Path(item) for item in file_paths] if isinstance(file_paths, (list, tuple, set)) else [Path(file_paths)]
    if not file_list:
        raise ValueError("No Rail Corrugation files were supplied for prediction.")
    raise RuntimeError("Rail Corrugation model not ready until the Info Kit and training pipeline are confirmed.")


__all__ = ["predict_rail_files"]
