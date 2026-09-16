"""Command-line entry point for the bogie-temperature analysis pipeline.

This script deliberately stops before training or exporting final official
predictions unless the dataset schema and required config are known.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess import inspect_dataframe, load_csv_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect, prepare and evaluate bogie-temperature telemetry."
    )
    parser.add_argument(
        "--input",
        default="data/input",
        help="Directory or CSV file containing the organiser-provided telemetry data.",
    )
    parser.add_argument(
        "--dashboard-output",
        default="docs/data/dashboard_data.json",
        help="Path for the dashboard JSON output.",
    )
    parser.add_argument(
        "--predictions-output",
        default="predictions/sample_predictions.csv",
        help="Path for the formal predictions CSV output.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON or YAML configuration file describing the real dataset schema.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    print("NEBULA X bogie-temperature pipeline scaffold")
    print("This script is inspection-first and will stop before training a model.")

    try:
        df, source_files = load_csv_files(input_path)
    except FileNotFoundError as exc:
        print(f"No input data found: {exc}")
        print("Place organiser-provided CSV files in data/input after checking the official instructions.")
        return 1

    print(f"Loaded {len(source_files)} CSV file(s): {source_files}")
    summary = inspect_dataframe(df)
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    print("Missing values:")
    for column, count in summary["missing_values"].items():
        print(f"  - {column}: {count}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")

    if args.config is None:
        print("\nThe real dataset schema is not configured yet.")
        print("Read Problem_Statement_3_Specifications.md and define the required columns before model training.")
        print("No final ML logic or official prediction export should be created from this state.")
        print("Stopping safely before invalid output is produced.")
        return 0

    # A real implementation would load the config file, validate the schema, clean the
    # data, engineer features, train a model, and export official results here.
    print("A complete schema-driven pipeline is not active until the config file is supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
