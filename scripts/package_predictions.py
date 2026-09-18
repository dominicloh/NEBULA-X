#!/usr/bin/env python3
"""Package Door and Rail prediction CSV files into a zipped submission bundle.

Only the subsystems actually attempted (i.e. whose prediction file exists
and validates) go into the archive, each placed directly at the ZIP root
with no internal folders. Malformed prediction files are refused with a
clear error (via the shared schema validators), never silently packaged.

By default this refuses to build an archive unless BOTH Door and Rail
Corrugation prediction files are present and valid -- a partial archive is
not a real submission. Pass --allow-incomplete to explicitly build a
Rail-only (or Door-only) TEST archive while Door (or Rail) isn't ready yet;
that archive's filename is forced to include "INCOMPLETE" so it can never be
mistaken for the final `predictions.zip`.
"""

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

# subsystem key -> (CLI arg default path, arcname inside the zip, validator)
_SUBSYSTEMS = {
    "door": ("predictions/door_predictions.csv", "door_predictions.csv", validate_door_predictions),
    "rail": ("predictions/rail_predictions.csv", "rail_predictions.csv", validate_rail_predictions),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package attempted subsystem prediction CSV files into a submission ZIP archive.",
    )
    parser.add_argument("--door", default=_SUBSYSTEMS["door"][0], help="Path to the Door prediction CSV.")
    parser.add_argument("--rail", default=_SUBSYSTEMS["rail"][0], help="Path to the Rail Corrugation prediction CSV.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output ZIP archive. Default: 'predictions.zip' when every subsystem is present, "
        "or 'predictions_INCOMPLETE.zip' when --allow-incomplete is used to build a partial test archive.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output archive if it already exists.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow building a TEST archive when one subsystem's prediction file is missing. "
        "The resulting archive is NOT the final submission -- its filename must contain 'INCOMPLETE'.",
    )
    return parser.parse_args()


def _check_subsystems(door_path: Path, rail_path: Path) -> tuple[dict, list]:
    """Validate whichever prediction files actually exist.

    Returns (available, missing): `available` maps arcname -> source Path for
    every subsystem whose file exists AND passes its schema validator (a
    malformed file is a hard error here, never silently skipped); `missing`
    lists (subsystem_name, path) for files that simply aren't there yet.
    """
    paths = {"door": door_path, "rail": rail_path}
    available: dict[str, Path] = {}
    missing: list[tuple[str, Path]] = []

    for key, (_default, arcname, validator) in _SUBSYSTEMS.items():
        path = paths[key]
        if not path.exists():
            missing.append((key, path))
            continue
        # A present-but-malformed file is refused outright -- it is never
        # treated as "missing" (which would let it be silently dropped from
        # an --allow-incomplete archive instead of being fixed).
        validator(pd.read_csv(path))
        available[arcname] = path

    return available, missing


def main() -> int:
    args = _parse_args()
    door_path = Path(args.door)
    rail_path = Path(args.rail)

    available, missing = _check_subsystems(door_path, rail_path)

    if not available:
        raise FileNotFoundError(
            "No valid prediction files found (checked "
            f"door={door_path}, rail={rail_path}). Nothing to package."
        )

    for subsystem_key, path in missing:
        subsystem_label = "Door" if subsystem_key == "door" else "Rail Corrugation"
        print(f"NOTE: {subsystem_label} prediction file not found at {path} -- it will be OMITTED, not fabricated.")

    is_complete = not missing
    if not is_complete and not args.allow_incomplete:
        missing_labels = ", ".join("Door" if key == "door" else "Rail Corrugation" for key, _ in missing)
        raise FileNotFoundError(
            f"Refusing to create the final predictions.zip: {missing_labels} prediction file is missing. "
            "This is not yet a complete submission. Pass --allow-incomplete to build a clearly-labelled "
            "INCOMPLETE test archive with only the available subsystem(s) instead."
        )

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = Path("predictions.zip" if is_complete else "predictions_INCOMPLETE.zip")

    if not is_complete and "incomplete" not in output_path.name.lower():
        raise ValueError(
            f"Refusing to name a partial archive {output_path.name!r}: an incomplete archive's filename must "
            "contain 'INCOMPLETE' so it can never be mistaken for the final predictions.zip. "
            "Omit --output to use the default, or choose a name containing 'INCOMPLETE'."
        )

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output archive already exists at {output_path}. Use --force to overwrite it.")
    if output_path.exists() and args.force:
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname, source_path in available.items():
            # arcname has no directory component, so every file lands
            # directly at the archive root with no internal folders.
            archive.write(source_path, arcname=arcname)

    with zipfile.ZipFile(output_path, "r") as archive:
        names = sorted(archive.namelist())
        expected = sorted(available.keys())
        if names != expected:
            raise ValueError(f"ZIP contents are invalid: {names}. Expected exactly {expected} at the archive root.")

    status = "COMPLETE" if is_complete else "INCOMPLETE TEST ARCHIVE -- not a final submission"
    print(f"Created {output_path} [{status}] with {expected} at the archive root.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
