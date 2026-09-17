#!/usr/bin/env python3
"""Validate export files for the Door and Rail Corrugation subsystems."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.validation import validate_door_predictions, validate_rail_predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Door or Rail Corrugation prediction exports.",
    )
    parser.add_argument("--subsystem", required=True, choices=["door", "rail"], help="Subsystem to validate.")
    parser.add_argument("--input", required=True, help="Path to the prediction CSV to validate.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {input_path}")

    frame = pd.read_csv(input_path)
    if args.subsystem == "door":
        validate_door_predictions(frame)
        print(f"Door validation passed for {input_path.name}.")
    else:
        validate_rail_predictions(frame)
        print(f"Rail Corrugation validation passed for {input_path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
