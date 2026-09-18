#!/usr/bin/env python3
"""Rail Corrugation exploratory plotting (EDA) - Stage 1 support script.

Plots a small number of example Train files per class (Normal / Side I /
Side II) so the team can visually sanity-check the confirmed signal channels
before building features. This script does NOT train a model and does NOT
modify any input CSV -- it only reads files and writes PNGs.

Usage:
    python src/rail_corrugation/plot_samples.py [dataset_dir] [--per-class N] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNG files directly, no GUI window needed
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rail_corrugation.inspect_data import DEFAULT_DATASET_DIR, find_dataset_layout

# Confirmed by the Rail Corrugation Info Kit (organiser-materials/PS3/03_References/
# Rail_Corrugation/Rail_Corrugation_Info_Kit.md, section 2.1): each file is a 1-second
# recording sampled at 10,000 Hz.
SAMPLING_FREQUENCY_HZ = 10_000

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "rail_corrugation" / "eda"


def pick_examples(labels: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """Pick up to `per_class` labelled files for each class, in file order."""
    picked = [group.head(per_class) for _, group in labels.groupby("label")]
    return pd.concat(picked, ignore_index=True)


def find_side_columns(columns, car: int, positions: list[int]) -> list[str]:
    """Return the Vibration column name for each requested axle-box position.

    Matches the exact header text confirmed in the dataset's own first row
    ("Vibration of bearing in position N of car C") instead of a hard-coded
    column index, so this keeps working even if column order ever changes.
    """
    found = []
    for position in positions:
        needle = f"Vibration of bearing in position {position} of car {car}"
        if needle in columns:
            found.append(needle)
    return found


def plot_one_file(csv_path: Path, label: str, output_dir: Path) -> Path:
    """Load one file, plot its confirmed channels, save a PNG, then free memory."""
    frame = pd.read_csv(csv_path)
    time_seconds = frame.index / SAMPLING_FREQUENCY_HZ

    speed_col = frame.columns[0]  # confirmed: "Rotating speed" (column 1)

    # Per the Info Kit: odd axle-box positions (1,3,5,7) = Side I rail,
    # even positions (2,4,6,8) = Side II rail. Car 1 is plotted as one
    # representative car -- all 8 cars share this same position layout.
    side_i_cols = find_side_columns(frame.columns, car=1, positions=[1, 3, 5, 7])
    side_ii_cols = find_side_columns(frame.columns, car=1, positions=[2, 4, 6, 8])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(time_seconds, frame[speed_col], color="black", linewidth=0.8)
    axes[0].set_ylabel(speed_col)

    for col in side_i_cols:
        short_label = col.replace("Vibration of bearing in ", "")
        axes[1].plot(time_seconds, frame[col], linewidth=0.6, label=short_label)
    axes[1].set_ylabel("Side I vibration (m/s^2)\n(Car 1, positions 1/3/5/7)")
    if side_i_cols:
        axes[1].legend(fontsize=6, loc="upper right")

    for col in side_ii_cols:
        short_label = col.replace("Vibration of bearing in ", "")
        axes[2].plot(time_seconds, frame[col], linewidth=0.6, label=short_label)
    axes[2].set_ylabel("Side II vibration (m/s^2)\n(Car 1, positions 2/4/6/8)")
    if side_ii_cols:
        axes[2].legend(fontsize=6, loc="upper right")
    axes[2].set_xlabel("Time (s)")

    fig.suptitle(f"{csv_path.name}  -  label: {label}")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_")
    out_path = output_dir / f"{csv_path.stem}_{safe_label}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)  # release the figure immediately; files are large, one at a time

    del frame  # release the dataframe before moving on to the next file
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a few example Rail Corrugation Train files per class (Normal / Side I / Side II)."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(DEFAULT_DATASET_DIR),
        help=f"Rail_Corrugation dataset directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=2,
        help="Number of example files to plot per class (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save PNG plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    train_dir, _test_dir, labels_path = find_dataset_layout(dataset_root)
    labels = pd.read_csv(labels_path)
    if "filename" not in labels.columns or "label" not in labels.columns:
        raise ValueError(
            f"Train_Labels.csv must have 'filename' and 'label' columns; found: {list(labels.columns)}"
        )

    examples = pick_examples(labels, args.per_class)
    print(f"Plotting {len(examples)} example file(s) (up to {args.per_class} per class)...")

    saved_paths = []
    for _, row in examples.iterrows():
        csv_path = train_dir / row["filename"]
        if not csv_path.is_file():
            print(f"  SKIP: {row['filename']} is listed in labels but was not found in {train_dir}")
            continue
        out_path = plot_one_file(csv_path, row["label"], output_dir)
        print(f"  saved: {out_path}")
        saved_paths.append(out_path)

    print(f"\nDone. {len(saved_paths)} plot(s) saved under: {output_dir}")
    print("(These PNGs are exploratory outputs only -- not part of the git-tracked submission.)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
