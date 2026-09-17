#!/usr/bin/env python3
"""Package Door and Rail prediction CSV files into a zipped submission bundle."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.validation import validate_door_predictions, validate_rail_predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package both official prediction CSV files into predictions.zip at the repository root.",
    )
    parser.add_argument("--door", default="predictions/door_predictions.csv", help="Path to the Door prediction CSV.")
    parser.add_argument("--rail", default="predictions/rail_predictions.csv", help="Path to the Rail Corrugation prediction CSV.")
    parser.add_argument("--output", default="predictions.zip", help="Path to the output ZIP archive.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output archive if it already exists.")
    return parser.parse_args()


def _validate_inputs(door_path: Path, rail_path: Path) -> None:
    if not door_path.exists():
        raise FileNotFoundError(f"Door prediction file not found: {door_path}")
    if not rail_path.exists():
        raise FileNotFoundError(f"Rail prediction file not found: {rail_path}")

    validate_door_predictions(pd.read_csv(door_path))
    validate_rail_predictions(pd.read_csv(rail_path))


def main() -> int:
    args = _parse_args()
    door_path = Path(args.door)
    rail_path = Path(args.rail)
    output_path = Path(args.output)

    _validate_inputs(door_path, rail_path)

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output archive already exists at {output_path}. Use --force to overwrite it."
        )

    if output_path.exists() and args.force:
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(door_path, arcname="door_predictions.csv")
        archive.write(rail_path, arcname="rail_predictions.csv")

    with zipfile.ZipFile(output_path, "r") as archive:
        names = sorted(archive.namelist())
        expected = ["door_predictions.csv", "rail_predictions.csv"]
        if names != expected:
            raise ValueError(f"ZIP contents are invalid: {names}. Expected exactly {expected} at the archive root.")

    print(f"Created {output_path} with {expected} at the archive root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
