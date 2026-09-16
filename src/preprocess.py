"""Utilities for loading, inspecting and validating hackathon CSV inputs.

This module intentionally avoids assuming the final dataset schema. The real field
names, timestamp format, and required columns must be confirmed from the official
problem specification before the pipeline is considered final.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def load_csv_files(input_path):
    """Load one CSV file or all CSV files in a directory into a single DataFrame.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Either a single CSV file path or a directory containing CSV files.

    Returns
    -------
    tuple
        A tuple containing:
        - combined pandas.DataFrame
        - list of source file names used for the combined dataset

    Notes
    -----
    The loader is intentionally generic. It does not assume a specific schema such
    as train IDs, timestamps, or temperature fields.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.csv"))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not files:
        raise FileNotFoundError(
            f"No CSV files were found in the input location: {input_path}. "
            "Add the organiser-provided CSV files only after checking the data instructions."
        )

    frames = []
    for csv_file in files:
        df = pd.read_csv(csv_file)
        df = df.copy()
        df["__source_file__"] = csv_file.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    source_files = [file.name for file in files]
    return combined, source_files


def inspect_dataframe(df):
    """Return a concise summary of the dataframe for inspection and debugging."""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return summary


def validate_required_columns(df, required_columns):
    """Check whether the dataframe contains all required columns.

    Parameters
    ----------
    df : pandas.DataFrame
    required_columns : iterable of str
        Column names required by the pipeline.

    Returns
    -------
    list of str
        Missing columns, if any.
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + ". Update the required schema once the official specification is read."
        )
    return missing


def clean_dataframe(df, config):
    """Apply generic cleaning steps after the real schema is known.

    Parameters
    ----------
    df : pandas.DataFrame
    config : dict
        Optional cleaning configuration. The real schema should define the needed
        values and transformations.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe.

    TODO:
    - Convert timestamps into a consistent datetime format.
    - Sort records chronologically within each relevant grouping.
    - Remove duplicate rows after confirming whether duplicates are legitimate.
    - Handle missing values according to field-specific rules once the schema is known.
    - Standardise unit conventions and component naming if required.
    """
    cleaned = df.copy()

    if config is None:
        config = {}

    # No schema assumptions are made here. The real pipeline must define columns and
    # cleaning behaviour after the organiser specification is reviewed.
    cleaned = cleaned.reset_index(drop=True)
    return cleaned
