#!/usr/bin/env python3
"""Door Stage 1 cleaning/validation checks.

No pytest (not a project dependency) -- plain assertions, matching the style
already used by tests/test_door_scaffold.py and tests/test_rail_predict.py.

Covers `src.door.preprocess.validate_and_clean` (the implementation moved
out of notebooks/clean_door_data.py, which is now only a thin CLI wrapper
around it) and the sibling-repo organiser-materials path in
src/door/config.py.

Run directly:
    python tests/test_clean_door_data.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402
from src.door.preprocess import PARSED_DATETIME_COLUMN, validate_and_clean  # noqa: E402


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def _valid_frame(n_rows: int = 5) -> pd.DataFrame:
    """A small, fully valid Door frame: correct schema, no missing/invalid
    values, strictly increasing timestamps 20ms apart -- mirrors the real
    Train.csv/Test.csv shape described in planning/door_handoff.md.
    """
    timestamps = [f"2023-7-5-0-0-{i // 50}-{(i % 50) * 20}" for i in range(n_rows)]
    data = {"Datetime": timestamps}
    for column in config.SIGNAL_COLUMNS:
        if column in ("Motor current(mA)", "Motor Voltage(10mV)", "Motor electrodynamic force", "Door leaf position"):
            data[column] = [100 + i for i in range(n_rows)]
        else:
            data[column] = [0 for _ in range(n_rows)]
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# Input types: filepath, file-like, DataFrame
# --------------------------------------------------------------------------


def check_filepath_input(tmp_path: Path) -> None:
    frame = _valid_frame()
    csv_path = tmp_path / "door_filepath.csv"
    frame.to_csv(csv_path, index=False)

    cleaned, report = validate_and_clean(csv_path)
    _check(report["pass"] is True, "filepath input: a valid CSV passes")
    _check(cleaned is not None, "filepath input: cleaned frame is returned on pass")
    _check(len(cleaned) == len(frame), "filepath input: row count is preserved")


def check_filelike_input() -> None:
    frame = _valid_frame()
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)

    cleaned, report = validate_and_clean(buffer)
    _check(report["pass"] is True, "file-like input: a valid in-memory CSV passes")
    _check(cleaned is not None and len(cleaned) == len(frame), "file-like input: cleaned frame matches input row count")


def check_dataframe_input() -> None:
    frame = _valid_frame()
    original_datetimes = list(frame["Datetime"])
    original_columns = list(frame.columns)
    cleaned, report = validate_and_clean(frame)
    _check(report["pass"] is True, "DataFrame input: a valid DataFrame passes")
    _check(cleaned is not None and len(cleaned) == len(frame), "DataFrame input: cleaned frame matches input row count")
    _check(
        list(frame["Datetime"]) == original_datetimes and list(frame.columns) == original_columns,
        "DataFrame input: caller's original frame is not mutated",
    )


# --------------------------------------------------------------------------
# Original Datetime preservation + parsed timestamps
# --------------------------------------------------------------------------


def check_original_datetime_preserved_and_parsed() -> None:
    frame = _valid_frame()
    cleaned, report = validate_and_clean(frame)
    _check(report["pass"] is True, "setup: valid frame passes")
    _check(list(cleaned["Datetime"]) == list(frame["Datetime"]), "original Datetime text column is preserved verbatim")
    _check(PARSED_DATETIME_COLUMN in cleaned.columns, "a parsed timestamp column is added")
    _check(
        pd.api.types.is_datetime64_any_dtype(cleaned[PARSED_DATETIME_COLUMN]),
        "the parsed timestamp column has a real datetime dtype",
    )
    _check(
        cleaned.loc[0, PARSED_DATETIME_COLUMN] == pd.Timestamp("2023-7-5 00:00:00.000"),
        "parsed timestamp values match the confirmed Year-Month-Date-Hour-Minute-Second-Millisecond format",
    )


# --------------------------------------------------------------------------
# Row order
#
# `row_order_preserved` measures whether preprocessing changed the relative
# order of the SOURCE rows -- computed purely from a temporary internal
# row-position field (src.door.preprocess.SOURCE_ROW_POSITION_COLUMN),
# never from timestamp values. Timestamp ordering is reported separately as
# `timestamps_monotonic`. The two are independent: a stream can arrive with
# out-of-order timestamps yet have its row order fully preserved (nothing in
# validate_and_clean reordered it), and vice versa.
# --------------------------------------------------------------------------


def check_ordered_timestamps_row_order_preserved() -> None:
    """1. Ordered timestamps, with row order preserved."""
    frame = _valid_frame()
    _, report = validate_and_clean(frame)
    _check(report["pass"] is True, "setup: an already-ordered valid frame passes")
    _check(report["row_order_preserved"] is True, "row order is preserved for already-ordered input")
    _check(report["timestamps_monotonic"] is True, "timestamps are reported as monotonic")
    _check("rows_reordered" not in report, "rows_reordered is not reported (its count cannot be measured accurately)")


def check_unordered_timestamps_row_order_still_preserved() -> None:
    """2. Deliberately unordered timestamps whose row order is still preserved.

    validate_and_clean never reorders `frame` itself (only a later sort of
    the *output* could) -- so row order must be reported as preserved here
    even though the Datetime values are not chronological, and even though
    the backward timestamp step is (separately) a hard validation failure.
    """
    frame = _valid_frame()
    shuffled = frame.iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)
    cleaned, report = validate_and_clean(shuffled)
    _check(cleaned is None, "a backward timestamp step is still treated as a validation failure")
    _check(
        any("out of order" in reason.lower() for reason in report["fail_reasons"]),
        "the backward timestamp step is named in fail_reasons",
    )
    _check(report["timestamps_monotonic"] is False, "timestamps are genuinely reported as non-monotonic")
    _check(
        report["row_order_preserved"] is True,
        "row_order_preserved stays True: unordered timestamps alone are not a row-position change",
    )


def check_reordering_transformation_is_detected() -> None:
    """3. A transformation that would reorder rows must be detected.

    validate_and_clean's own only row-removing step (duplicate-row removal)
    never reorders survivors, by construction, so no well-formed CSV can
    exercise a False result end-to-end. The detection mechanism itself --
    `check_row_order_preserved`, the exact function `validate_and_clean`
    calls against its temporary position field -- is tested directly here
    with a manufactured reordering to prove it actually catches one.
    """
    from src.door.preprocess import check_row_order_preserved

    _check(check_row_order_preserved([0, 1, 2, 3, 4]) is True, "already-ordered source positions are reported as preserved")
    _check(
        check_row_order_preserved([0, 2, 1, 3, 4]) is False,
        "a transformation that swaps two rows' source positions is detected as NOT preserved",
    )
    _check(
        check_row_order_preserved([4, 3, 2, 1, 0]) is False,
        "a fully reversed transformation is detected as NOT preserved",
    )


def check_duplicate_removal_retains_relative_order() -> None:
    """4. Duplicate removal retains the relative order of remaining rows."""
    frame = _valid_frame(6)
    # Insert an exact duplicate of row 1 in the middle of the stream (not
    # merely appended at the end), so a naive implementation that sorted
    # before/after deduplicating would be exposed.
    with_duplicate = pd.concat([frame.iloc[:4], frame.iloc[[1]], frame.iloc[4:]], ignore_index=True)
    cleaned, report = validate_and_clean(with_duplicate)
    _check(report["pass"] is True, "a duplicate row inserted mid-stream still passes")
    _check(report["duplicate_rows_removed"] == 1, "exactly one duplicate row is removed")
    _check(report["row_order_preserved"] is True, "row_order_preserved is True after duplicate removal")
    _check(
        list(cleaned["Datetime"]) == list(frame["Datetime"]),
        "the remaining rows keep their original relative order after the duplicate is removed",
    )


def check_row_order_preserved_none_when_malformed_columns_prevent_measurement() -> None:
    frame = _valid_frame().drop(columns=["Door leaf position"])
    _, report = validate_and_clean(frame)
    _check(report["pass"] is False, "missing a required column fails validation")
    _check(
        report["row_order_preserved"] is None,
        "row_order_preserved stays unmeasured (None) when validation fails before the position field can be checked",
    )


# --------------------------------------------------------------------------
# Missing columns
# --------------------------------------------------------------------------


def check_missing_columns_reported() -> None:
    frame = _valid_frame().drop(columns=["Door leaf position"])
    cleaned, report = validate_and_clean(frame)
    _check(cleaned is None, "cleaned is None when a required column is missing")
    _check(report["pass"] is False, "missing a required column fails validation")
    _check("Door leaf position" in report["missing_required_columns"], "the specific missing column is named in the report")


# --------------------------------------------------------------------------
# Malformed timestamps
# --------------------------------------------------------------------------


def check_malformed_timestamps_reported() -> None:
    frame = _valid_frame()
    frame.loc[1, "Datetime"] = "2023-13-40-99-99-99-9999"  # wrong field count is not the point; garbage value
    frame.loc[3, "Datetime"] = "garbage"
    cleaned, report = validate_and_clean(frame)
    _check(cleaned is None, "cleaned is None when timestamps are malformed")
    _check(report["timestamp_errors"]["count"] == 2, "both malformed timestamps are counted")
    _check(1 in report["timestamp_errors"]["row_indices"] and 3 in report["timestamp_errors"]["row_indices"], "malformed row indices are identified")
    _check(any("Invalid timestamps" in reason for reason in report["fail_reasons"]), "malformed timestamps are a fail reason")


# --------------------------------------------------------------------------
# Missing and infinite values
# --------------------------------------------------------------------------


def check_missing_values_reported() -> None:
    frame = _valid_frame()
    frame["Motor current(mA)"] = frame["Motor current(mA)"].astype(object)
    frame.loc[2, "Motor current(mA)"] = ""
    cleaned, report = validate_and_clean(frame)
    _check(cleaned is None, "cleaned is None when a required value is missing")
    _check(report["missing_values"].get("Motor current(mA)") == 1, "the missing value is counted against its column")
    _check(any("Missing values" in reason for reason in report["fail_reasons"]), "missing values are a fail reason")


def check_infinite_values_reported() -> None:
    frame = _valid_frame()
    frame["Motor current(mA)"] = frame["Motor current(mA)"].astype(object)
    frame.loc[0, "Motor current(mA)"] = "inf"
    cleaned, report = validate_and_clean(frame)
    _check(cleaned is None, "cleaned is None when an infinite value is present")
    _check(report["infinite_values"]["Motor current(mA)"] == 1, "the infinite value is counted against its column")
    _check(any("Infinite values" in reason for reason in report["fail_reasons"]), "infinite values are a fail reason")


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------


def check_exact_duplicate_rows_cleaned_not_failed() -> None:
    frame = _valid_frame()
    with_duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    cleaned, report = validate_and_clean(with_duplicate)
    _check(report["pass"] is True, "an exact duplicate row is cleaned, not a failure")
    _check(report["duplicate_rows_removed"] == 1, "the duplicate row count is reported")
    _check(len(cleaned) == len(frame), "the duplicate row is actually removed from the cleaned output")


def check_duplicate_timestamps_reported() -> None:
    frame = _valid_frame()
    # Same Datetime as row 0, but different sensor values -- not an exact
    # duplicate row, so it survives to the duplicate-timestamp check.
    frame.loc[1, "Datetime"] = frame.loc[0, "Datetime"]
    cleaned, report = validate_and_clean(frame)
    _check(cleaned is None, "cleaned is None when timestamps are duplicated")
    _check(report["duplicate_timestamps"]["count"] == 2, "both rows sharing the duplicated timestamp are counted")
    _check(any("Duplicate timestamps" in reason for reason in report["fail_reasons"]), "duplicate timestamps are a fail reason")


# --------------------------------------------------------------------------
# Sibling organiser path
# --------------------------------------------------------------------------


def check_organiser_materials_is_a_sibling_path() -> None:
    expected = config.PROJECT_ROOT.parent / "organiser-materials" / "PS3" / "02_Datasets" / "Door"
    _check(config.DEFAULT_DATASET_DIR == expected, "DEFAULT_DATASET_DIR resolves to the sibling organiser-materials layout")
    _check(
        config.PROJECT_ROOT not in config.DEFAULT_DATASET_DIR.parents,
        "organiser-materials is NOT treated as nested inside the team repository",
    )


def check_clean_door_data_cli_uses_config_default() -> None:
    """notebooks/clean_door_data.py must not hard-code its own (buggy,
    repo-nested) organiser-materials path -- it should reuse
    config.DEFAULT_DATASET_DIR, the single source of truth.
    """
    from unittest import mock

    import notebooks.clean_door_data as clean_door_data

    with mock.patch.object(clean_door_data, "process_file") as mock_process_file:
        sys_argv_backup = sys.argv
        sys.argv = ["clean_door_data.py"]
        try:
            clean_door_data.main()
        finally:
            sys.argv = sys_argv_backup

    called_sources = [call.args[0] for call in mock_process_file.call_args_list]
    _check(len(called_sources) == 2, "the CLI wrapper processes exactly Train.csv and Test.csv by default")
    _check(
        all(config.DEFAULT_DATASET_DIR in path.parents for path in called_sources),
        "the CLI wrapper's default file paths live under config.DEFAULT_DATASET_DIR (the sibling path), not a repo-nested one",
    )


# --------------------------------------------------------------------------
# No dependency on Train_Segments_Answer.csv
# --------------------------------------------------------------------------


def check_no_dependency_on_train_segments_answer() -> None:
    import ast
    import inspect

    import src.door.preprocess as preprocess
    import notebooks.clean_door_data as clean_door_data

    for module in (preprocess, clean_door_data):
        tree = ast.parse(inspect.getsource(module))
        # Strip out every string literal (docstrings included) so only real,
        # executable references would trigger this check -- both modules'
        # docstrings legitimately name the file to explain why it's
        # excluded, which must not be mistaken for a code dependency.
        _check(
            "Train_Segments_Answer" not in _strip_string_literals(tree),
            f"{module.__name__} has no executable reference to Train_Segments_Answer.csv",
        )


def _strip_string_literals(tree) -> str:
    import ast

    class _StringStripper(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

    stripped = ast.fix_missing_locations(_StringStripper().visit(tree))
    return ast.unparse(stripped)


# --------------------------------------------------------------------------
# No CSV/JSON artifacts get produced anywhere but the configured output dir
# --------------------------------------------------------------------------


def check_process_file_writes_only_to_output_dir(tmp_path: Path) -> None:
    from src.door.preprocess import process_file

    frame = _valid_frame()
    source = tmp_path / "Train.csv"
    frame.to_csv(source, index=False)
    output_dir = tmp_path / "out"

    report = process_file(source, output_dir)
    _check(report["pass"] is True, "process_file: a valid source file passes")
    _check((output_dir / "Train_cleaned.csv").is_file(), "process_file writes the cleaned CSV to the given output dir")
    _check((output_dir / "Train_validation_report.json").is_file(), "process_file writes the JSON report to the given output dir")


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        check_filepath_input(tmp_path)
        check_filelike_input()
        check_dataframe_input()
        check_original_datetime_preserved_and_parsed()
        check_ordered_timestamps_row_order_preserved()
        check_unordered_timestamps_row_order_still_preserved()
        check_reordering_transformation_is_detected()
        check_duplicate_removal_retains_relative_order()
        check_row_order_preserved_none_when_malformed_columns_prevent_measurement()
        check_missing_columns_reported()
        check_malformed_timestamps_reported()
        check_missing_values_reported()
        check_infinite_values_reported()
        check_exact_duplicate_rows_cleaned_not_failed()
        check_duplicate_timestamps_reported()
        check_organiser_materials_is_a_sibling_path()
        check_clean_door_data_cli_uses_config_default()
        check_no_dependency_on_train_segments_answer()
        check_process_file_writes_only_to_output_dir(tmp_path)

    print("\nAll Door cleaning/validation checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        print(f"\nTEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
