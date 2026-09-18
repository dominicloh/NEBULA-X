#!/usr/bin/env python3
"""Rail Corrugation dataset inspector.

Stage 1 of the Rail Corrugation pipeline: a **read-only** inspection of the
official organiser dataset (Train/ folder, Test/ folder, Train_Labels.csv).
It never modifies the input CSVs and never trains a model — it only reports
facts about the data so the team can plan features and a train/validation
split from confirmed information, not assumptions.

Files are read one at a time (never all held in memory together), because the
raw files are large (10,000 rows per file) and there are hundreds of them.

Usage:
    python src/rail_corrugation/inspect_data.py [dataset_dir]

Example (dataset_dir defaults to the organiser materials sibling folder):
    python src/rail_corrugation/inspect_data.py
    python src/rail_corrugation/inspect_data.py "../organiser-materials/PS3/02_Datasets/Rail_Corrugation"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make "from src...." imports work when this script is run directly
# (python src/rail_corrugation/inspect_data.py), matching the convention
# already used by scripts/validate_predictions.py in this repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.validation import VALID_RAIL_LABELS  # noqa: E402  (Normal, Side I, Side II)

# Default dataset location: the organiser materials repo is a sibling folder
# of this team repo, per the project's known Windows folder layout. Still
# fully overridable via the command-line argument.
DEFAULT_DATASET_DIR = PROJECT_ROOT.parent / "organiser-materials" / "PS3" / "02_Datasets" / "Rail_Corrugation"


# ---------------------------------------------------------------------------
# Natural sort: "Train2.csv" must sort before "Train10.csv"
# ---------------------------------------------------------------------------
_NUMBER_RUN = re.compile(r"(\d+)")


def natural_sort_key(path: Path):
    """Split a filename into text/number chunks so numeric runs sort by value."""
    chunks = _NUMBER_RUN.split(path.stem)
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in chunks]


def natural_sorted(paths):
    return sorted(paths, key=natural_sort_key)


# ---------------------------------------------------------------------------
# Discovery — fail loudly and clearly if the expected layout isn't there
# ---------------------------------------------------------------------------
def find_dataset_layout(dataset_root: Path):
    """Locate Train/, Test/, and Train_Labels.csv under the dataset root."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Rail Corrugation dataset directory does not exist: {dataset_root}\n"
            "Pass the correct path as the command-line argument, e.g.:\n"
            '  python src/rail_corrugation/inspect_data.py "../organiser-materials/PS3/02_Datasets/Rail_Corrugation"'
        )

    train_dir = dataset_root / "Train"
    test_dir = dataset_root / "Test"
    labels_path = dataset_root / "Train_Labels.csv"

    missing = []
    if not train_dir.is_dir():
        missing.append(f"Train directory not found at: {train_dir}")
    if not test_dir.is_dir():
        missing.append(f"Test directory not found at: {test_dir}")
    if not labels_path.is_file():
        missing.append(f"Train_Labels.csv not found at: {labels_path}")
    if missing:
        raise FileNotFoundError(
            "Rail Corrugation dataset directory is missing expected items:\n  "
            + "\n  ".join(missing)
            + f"\n\nChecked dataset root: {dataset_root}"
        )
    return train_dir, test_dir, labels_path


def list_csv_files(directory: Path) -> list[Path]:
    files = natural_sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")
    return files


# ---------------------------------------------------------------------------
# Per-file inspection (one file loaded at a time, then discarded)
# ---------------------------------------------------------------------------
def describe_file(path: Path) -> dict:
    """Read one CSV and compute lightweight summary stats. Never writes back."""
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])

    n_inf = 0
    if not numeric.empty:
        n_inf = int(np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).sum())

    constant_cols = [
        column for column in numeric.columns if numeric[column].nunique(dropna=True) <= 1
    ]

    info = {
        "file": path.name,
        "n_rows": int(frame.shape[0]),
        "n_cols": int(frame.shape[1]),
        "columns": tuple(frame.columns),
        "dtypes": tuple(str(dtype) for dtype in frame.dtypes),
        "n_missing": int(frame.isna().sum().sum()),
        "n_inf": n_inf,
        "n_dup_rows": int(frame.duplicated().sum()),
        "constant_cols": tuple(constant_cols),
    }
    # Explicitly drop references before the next file is loaded, so only one
    # file's data is ever resident in memory at a time.
    del frame, numeric
    return info


def summarize_split(name: str, files: list[Path]) -> list[dict]:
    """Describe every file in a split (Train or Test), one at a time, with progress output."""
    print(f"\nInspecting {name} split: {len(files)} file(s)...")
    records = []
    for index, path in enumerate(files, start=1):
        record = describe_file(path)
        records.append(record)
        if index % 50 == 0 or index == len(files):
            print(f"  ...processed {index}/{len(files)} files")
    return records


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def report_schema_consistency(label: str, records: list[dict]) -> tuple:
    """Print whether every file in a split shares the same column schema."""
    schemas = {r["columns"] for r in records}
    if len(schemas) == 1:
        print(f"  {label}: all {len(records)} file(s) share one identical column schema ({len(records[0]['columns'])} columns).")
    else:
        print(f"  {label}: found {len(schemas)} DIFFERENT column schemas across {len(records)} file(s)!")
        reference = records[0]["columns"]
        for record in records:
            if record["columns"] != reference:
                print(f"    - {record['file']} differs from {records[0]['file']}")
    return schemas


def report_row_stats(label: str, records: list[dict]) -> None:
    row_counts = [r["n_rows"] for r in records]
    print(
        f"  {label} rows per file: min={min(row_counts)}, "
        f"median={int(np.median(row_counts))}, max={max(row_counts)}"
    )
    if min(row_counts) == max(row_counts):
        print(f"  {label}: all files have equal length ({row_counts[0]} rows).")
    else:
        print(f"  {label}: file lengths are NOT all equal.")


def report_data_quality(label: str, records: list[dict]) -> None:
    total_missing = sum(r["n_missing"] for r in records)
    total_inf = sum(r["n_inf"] for r in records)
    total_dup_rows = sum(r["n_dup_rows"] for r in records)
    files_with_missing = [r["file"] for r in records if r["n_missing"] > 0]
    files_with_inf = [r["file"] for r in records if r["n_inf"] > 0]
    files_with_dup = [r["file"] for r in records if r["n_dup_rows"] > 0]

    print(f"  {label} missing values (total cells): {total_missing} across {len(files_with_missing)} file(s)")
    print(f"  {label} infinite values (total cells): {total_inf} across {len(files_with_inf)} file(s)")
    print(f"  {label} duplicate rows (total rows):   {total_dup_rows} across {len(files_with_dup)} file(s)")

    # Columns that are constant (single unique value) in EVERY file of the split.
    if records:
        always_constant = set(records[0]["constant_cols"])
        for record in records[1:]:
            always_constant &= set(record["constant_cols"])
        if always_constant:
            print(f"  {label} columns constant in ALL files: {sorted(always_constant)}")
        else:
            print(f"  {label}: no column is constant across every file.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only inspection of the official Rail Corrugation dataset (no training)."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(DEFAULT_DATASET_DIR),
        help=(
            "Path to the Rail_Corrugation dataset directory "
            "(must contain Train/, Test/, Train_Labels.csv). "
            f"Default: {DEFAULT_DATASET_DIR}"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_dir).expanduser().resolve()

    print("=" * 78)
    print("RAIL CORRUGATION - DATA INSPECTION (read-only, no model training)")
    print("=" * 78)
    print(f"Dataset root: {dataset_root}")

    train_dir, test_dir, labels_path = find_dataset_layout(dataset_root)

    train_files = list_csv_files(train_dir)
    test_files = list_csv_files(test_dir)

    print(f"Train directory: {train_dir}  -> {len(train_files)} CSV file(s)")
    print(f"Test directory:  {test_dir}  -> {len(test_files)} CSV file(s)")
    print(f"Labels file:     {labels_path}")

    # --- Labels -------------------------------------------------------
    labels = pd.read_csv(labels_path)
    required_label_columns = {"filename", "label"}
    missing_label_columns = required_label_columns - set(labels.columns)
    if missing_label_columns:
        raise ValueError(
            f"Train_Labels.csv is missing required column(s): {sorted(missing_label_columns)}. "
            f"Found columns: {list(labels.columns)}"
        )

    print(f"\n--- Labels ({labels_path.name}) ---")
    print(f"Label rows: {len(labels)}")

    duplicate_label_rows = labels[labels.duplicated(subset=["filename"], keep=False)]
    if not duplicate_label_rows.empty:
        dup_names = sorted(duplicate_label_rows["filename"].unique())
        print(f"WARNING: {len(dup_names)} filename(s) have more than one label row: {dup_names}")
    else:
        print("Every labelled filename has exactly one label row.")

    unexpected_labels = sorted(set(labels["label"].dropna().unique()) - set(VALID_RAIL_LABELS))
    if unexpected_labels:
        print(f"WARNING: unexpected label value(s) found (not in {VALID_RAIL_LABELS}): {unexpected_labels}")
    else:
        print(f"All label values are within the expected set: {VALID_RAIL_LABELS}")

    print("\nClass distribution (Train_Labels.csv):")
    class_counts = labels["label"].value_counts(dropna=False)
    for label_name, count in class_counts.items():
        pct = 100.0 * count / len(labels)
        print(f"  {label_name!s:10s}: {count:4d}  ({pct:5.1f}%)")

    # --- Filename coverage: labels vs. actual Train files --------------
    train_filenames = {p.name for p in train_files}
    label_filenames = set(labels["filename"])

    missing_labels_for_files = sorted(train_filenames - label_filenames)
    labels_without_files = sorted(label_filenames - train_filenames)

    print("\n--- Train file <-> label coverage ---")
    if missing_labels_for_files:
        print(f"WARNING: {len(missing_labels_for_files)} Train file(s) have NO label row: {missing_labels_for_files}")
    else:
        print("Every Train file has a label row.")
    if labels_without_files:
        print(f"WARNING: {len(labels_without_files)} label row(s) reference a Train file that does not exist: {labels_without_files}")
    else:
        print("Every label row refers to an existing Train file (no extra/orphaned labels).")

    # --- Per-file schema & quality inspection (one file at a time) -----
    train_records = summarize_split("Train", train_files)
    test_records = summarize_split("Test", test_files)

    print("\n--- Schema consistency ---")
    train_schemas = report_schema_consistency("Train", train_records)
    test_schemas = report_schema_consistency("Test", test_records)

    print("\n--- Train vs. Test schema comparison ---")
    if train_schemas == test_schemas:
        print("Train and Test share the exact same column schema.")
    else:
        only_in_train = train_schemas - test_schemas
        only_in_test = test_schemas - train_schemas
        print("WARNING: Train and Test column schemas differ.")
        if only_in_train:
            print(f"  Schema(s) only seen in Train: {only_in_train}")
        if only_in_test:
            print(f"  Schema(s) only seen in Test: {only_in_test}")

    print("\n--- Row counts per file ---")
    report_row_stats("Train", train_records)
    report_row_stats("Test", test_records)

    print("\n--- Column names & dtypes (first Train file, for reference) ---")
    first = train_records[0]
    for column, dtype in zip(first["columns"], first["dtypes"]):
        print(f"  {column}  ({dtype})")

    print("\n--- Data quality ---")
    report_data_quality("Train", train_records)
    report_data_quality("Test", test_records)

    print("\n" + "=" * 78)
    print("Inspection complete. No files were modified. No model was trained.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
