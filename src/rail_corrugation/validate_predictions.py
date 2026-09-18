#!/usr/bin/env python3
"""Rail Corrugation Stage 3: official prediction CSV validator.

`scripts/validate_predictions.py` already checks the generic Door/Rail
schema rules (columns, no nulls, allowed labels) via
`src/common/validation.py`. This script reuses that check and adds the
Rail-specific checks that require knowing the actual official Test/
directory: exact row count, every test filename present exactly once, and
no unexpected filenames -- none of which the generic validator can check on
its own since it has no dataset to compare against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.validation import validate_rail_predictions  # noqa: E402
from src.rail_corrugation.inspect_data import DEFAULT_DATASET_DIR, find_dataset_layout, list_csv_files  # noqa: E402

DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "predictions" / "rail_predictions.csv"


def validate_rail_predictions_file(predictions_path: Path, dataset_root: Path, verbose: bool = True) -> dict:
    """Run every Stage 3 check against one rail_predictions.csv.

    Raises ValueError/FileNotFoundError with a clear message on the first
    problem found. Returns a small dict of confirmed facts on success (row
    count, expected file count) for callers that want to report them.
    """
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Rail predictions file not found: {predictions_path}")

    # Read with the default header inference; an accidental extra index
    # column written by a stray `to_csv(index=True)` would show up as a
    # leading "Unnamed: 0" column and get caught by the exact-columns check
    # inside validate_rail_predictions below, but we check it explicitly
    # here first for a clearer, Rail-specific error message.
    predictions = pd.read_csv(predictions_path)
    if predictions.columns[0].startswith("Unnamed"):
        raise ValueError(
            f"Rail predictions file has an unnamed index column ({predictions.columns[0]!r}) as its "
            "first column -- likely written with to_csv(index=True). Re-save with index=False."
        )

    # 1. Generic schema: exact columns/order, no nulls, only allowed labels.
    validate_rail_predictions(predictions)
    if verbose:
        print(f"[OK] Columns are exactly {list(predictions.columns)}, no missing values, only allowed labels.")

    # 2. No duplicate file_id.
    duplicate_ids = sorted(predictions.loc[predictions["file_id"].duplicated(keep=False), "file_id"].unique())
    if duplicate_ids:
        raise ValueError(f"Rail predictions contain duplicate file_id value(s): {duplicate_ids}")
    if verbose:
        print("[OK] No duplicate file_id values.")

    # 3. Row count and filenames must exactly match the official Test/ files.
    _train_dir, test_dir, _labels_path = find_dataset_layout(dataset_root)
    expected_files = {path.name for path in list_csv_files(test_dir)}
    actual_files = set(predictions["file_id"])

    if len(predictions) != len(expected_files):
        raise ValueError(
            f"Rail predictions has {len(predictions)} row(s); expected exactly {len(expected_files)} "
            f"(one per file in {test_dir})."
        )
    if verbose:
        print(f"[OK] Row count matches the {len(expected_files)} official Test files.")

    missing_files = sorted(expected_files - actual_files)
    if missing_files:
        raise ValueError(f"Rail predictions is missing {len(missing_files)} required test filename(s): {missing_files}")
    if verbose:
        print("[OK] Every official test filename appears exactly once.")

    unexpected_files = sorted(actual_files - expected_files)
    if unexpected_files:
        raise ValueError(
            f"Rail predictions contains {len(unexpected_files)} unexpected filename(s) not in Test/: {unexpected_files}"
        )
    if verbose:
        print("[OK] No unexpected filenames.")

    if verbose:
        print(f"\nAll Rail Corrugation submission checks passed for: {predictions_path}")

    return {"n_rows": len(predictions), "n_expected_test_files": len(expected_files)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the official Rail Corrugation rail_predictions.csv.")
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS_PATH),
        help=f"Path to rail_predictions.csv. Default: {DEFAULT_PREDICTIONS_PATH}",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help=f"Rail_Corrugation dataset directory (for the official Test/ file list). Default: {DEFAULT_DATASET_DIR}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_rail_predictions_file(Path(args.predictions), Path(args.dataset_dir).expanduser().resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nVALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)


__all__ = ["validate_rail_predictions_file"]
