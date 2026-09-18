"""Door cycle segmentation -- THIS IS THE MAIN FILE THE DOOR TEAMMATE MUST FINISH.

`detect_cycles()` deliberately raises NotImplementedError below. Do not
invent a segmentation rule here without justifying it against
`Train_Segments_Answer.csv` first -- see planning/door_handoff.md Sections
4 and 6 for the confirmed facts and a safe first approach.

Suggested beginner workflow (see planning/door_handoff.md for the full
detail behind each step):

    1. Plot the door-position signal (and a few others) for a known cycle
       from Train_Segments_Answer.csv, so you know what a real cycle looks
       like before writing any detection code.
    2. Identify candidate opening/closing transitions -- planning/door_handoff.md
       Section 2 found that large gaps in the "Datetime" column (>100ms,
       vs. the usual ~20ms) line up almost exactly with the boundaries
       BETWEEN cycles in Train.csv. That is a measured hint, not a proven
       rule -- verify it yourself with inspect_data.py before relying on it.
    3. From each candidate window, derive a proposed start_time/end_time
       (you will likely need to refine the exact edges using the signal
       columns themselves -- a big gap tells you roughly WHERE a boundary
       is, not necessarily the exact millisecond).
    4. Compare your candidates against Train_Segments_Answer.csv: how many
       of the 110 true segments did you find? How close are your start/end
       times? Did you invent any extra segments that aren't real?
    5. Tune your approach using ONLY Train.csv + Train_Segments_Answer.csv.
       Never look at Test.csv while tuning.
    6. Measure temporal overlap (IoU) between your candidates and the true
       segments -- this is exactly what the official IoU-weighted F1 metric
       rewards (see the Door Info Kit, Section 4, for the exact formula).
"""

from __future__ import annotations

import pandas as pd

from src.door.preprocess import PARSED_DATETIME_COLUMN

# NOTE: this module does not import src.door.config at the top level,
# because detect_cycles()'s own `config` parameter (below) would shadow it --
# a classic beginner trap. If your implementation of detect_cycles needs
# values from src/door/config.py (e.g. CANDIDATE_GAP_THRESHOLD_MS), import
# it inside the function with a different name, e.g.:
#     from src.door import config as door_config
#     threshold = door_config.CANDIDATE_GAP_THRESHOLD_MS

# The exact columns detect_cycles() must return, once implemented. This
# matches the official Train_Segments_Answer.csv schema (minus "status",
# which classification -- not segmentation -- assigns) so the same table
# shape flows straight into extract_cycle_features().
SEGMENT_COLUMNS = ("segment_id", "start_time", "end_time")


def detect_cycles(dataframe: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Detect candidate door-open/close cycles in a continuous Door stream.

    Parameters
    ----------
    dataframe:
        A Door stream loaded by `src.door.preprocess.load_door_csv` (must
        already have the parsed "_datetime_parsed" column).
    config:
        Segmentation parameters (e.g. the gap threshold to use). Left as
        `None` deliberately -- `src/door/config.py::SEGMENTATION_CONFIG` is
        also `None` until this function is implemented and validated, so a
        caller can't accidentally "configure" a segmentation rule that
        doesn't exist yet.

    Returns
    -------
    A DataFrame with (at least) columns `segment_id`, `start_time`,
    `end_time` -- one row per detected candidate cycle -- once implemented.

    Currently always raises NotImplementedError: inventing a segmentation
    rule without validating it against Train_Segments_Answer.csv would risk
    silently producing wrong cycle boundaries that look like real output.
    See the module docstring above and planning/door_handoff.md for how to
    implement this for real.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("detect_cycles expects a pandas DataFrame (see src.door.preprocess.load_door_csv).")
    if PARSED_DATETIME_COLUMN not in dataframe.columns:
        raise ValueError(
            f"detect_cycles expects a DataFrame with a {PARSED_DATETIME_COLUMN!r} column -- "
            "load it with src.door.preprocess.load_door_csv first."
        )

    raise NotImplementedError(
        "Door segmentation (detect_cycles) is not implemented yet. This is the main piece the "
        "Door teammate needs to build -- see planning/door_handoff.md Sections 4 and 6 for the "
        "confirmed data facts (e.g. large Datetime gaps lining up with cycle boundaries in "
        "Train.csv) and a suggested safe first approach. Validate any approach against "
        "Train_Segments_Answer.csv before ever running it on Test.csv."
    )


# Backward-compatible aliases -- earlier scaffold code imported these names.
def detect_door_segments(stream: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Deprecated name for `detect_cycles` -- kept for backward compatibility."""
    return detect_cycles(stream, config=config)


def segment_door_stream(stream: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Deprecated name for `detect_cycles` -- kept for backward compatibility."""
    return detect_cycles(stream, config=config)


def validate_segment_table(segments: pd.DataFrame, dataframe: pd.DataFrame | None = None) -> list[str]:
    """Check a table of segments (e.g. Train_Segments_Answer.csv, or your
    own detect_cycles() output) for structural problems.

    Returns a list of human-readable problem descriptions (empty = valid).
    Used by both inspect_data.py (read-only reporting) and train.py
    (refusing to train on a broken segment table) so this logic lives in
    exactly one place.

    Checks: start before end, no duplicate segment_id, no overlapping
    consecutive segments, and (if `dataframe` is given) every boundary
    falls inside the dataframe's own time range.
    """
    problems: list[str] = []
    required = {"start_time", "end_time"}
    missing = required - set(segments.columns)
    if missing:
        return [f"Segment table is missing required column(s): {sorted(missing)}"]

    from src.door.preprocess import PARSED_DATETIME_COLUMN, parse_door_datetime

    def _as_timestamp(value):
        return value if isinstance(value, pd.Timestamp) else parse_door_datetime(str(value))

    starts = segments["start_time"].map(_as_timestamp)
    ends = segments["end_time"].map(_as_timestamp)

    invalid_order = int((starts >= ends).sum())
    if invalid_order:
        problems.append(f"{invalid_order} segment(s) have start_time >= end_time.")

    if "segment_id" in segments.columns:
        duplicate_ids = int(segments["segment_id"].duplicated().sum())
        if duplicate_ids:
            problems.append(f"{duplicate_ids} duplicate segment_id value(s) found.")

    order = starts.sort_values().index
    sorted_starts = starts.loc[order].reset_index(drop=True)
    sorted_ends = ends.loc[order].reset_index(drop=True)
    overlaps = int((sorted_ends.iloc[:-1].reset_index(drop=True) > sorted_starts.iloc[1:].reset_index(drop=True)).sum())
    if overlaps:
        problems.append(f"{overlaps} pair(s) of consecutive segments overlap in time.")

    if dataframe is not None and PARSED_DATETIME_COLUMN in dataframe.columns:
        data_min = dataframe[PARSED_DATETIME_COLUMN].min()
        data_max = dataframe[PARSED_DATETIME_COLUMN].max()
        outside = int(((starts < data_min) | (ends > data_max)).sum())
        if outside:
            problems.append(f"{outside} segment(s) fall outside the given stream's own time range ({data_min} to {data_max}).")

    return problems


__all__ = [
    "SEGMENT_COLUMNS",
    "detect_cycles",
    "detect_door_segments",
    "segment_door_stream",
    "validate_segment_table",
]
