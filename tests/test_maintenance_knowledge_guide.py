#!/usr/bin/env python3
"""Maintenance Knowledge Guide reference-case data checks (Door + Rail).

No pytest (not a project dependency) -- plain assertions, matching the style
already used by tests/test_door_scaffold.py and tests/test_rail_predict.py.

Checks the STATIC demonstration case data behind the "Cases" popovers in
app/door_view.py and app/rail_view.py: every one of the 9 reference cases
(3 Door + 3 Rail Side I + 3 Rail Side II) has a well-formed, unique
Reference ID and a "last_updated" date in ISO YYYY-MM-DD format. This is
presentation-only content -- it never touches src/door/, src/rail_corrugation/,
models/, predictions/ or data/, and these checks don't either.

Run directly:
    python tests/test_maintenance_knowledge_guide.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def _check_case_fields(case: dict, group_name: str) -> None:
    for field in ("title", "reference_id", "last_updated", "possible_sign", "previous_action", "outcome"):
        value = case.get(field)
        _check(isinstance(value, str) and value.strip() != "", f"{group_name} case has a non-empty '{field}' field ({case.get('reference_id', '?')})")

    reference_id = case["reference_id"]
    _check(isinstance(reference_id, str) and reference_id.startswith("DEMO-"), f"{group_name} reference_id '{reference_id}' is a demonstration ID (starts with 'DEMO-')")

    last_updated = case["last_updated"]
    _check(isinstance(last_updated, str) and bool(_ISO_DATE_RE.match(last_updated)), f"{group_name} last_updated '{last_updated}' matches YYYY-MM-DD")
    # Also confirms it's a real calendar date, not just digit-shaped.
    datetime.strptime(last_updated, "%Y-%m-%d")
    _check(True, f"{group_name} last_updated '{last_updated}' ({reference_id}) parses as a real ISO calendar date")

    _check("handled_by" not in case, f"{group_name} case ({reference_id}) has no leftover 'handled_by' field")


def check_door_reference_cases() -> None:
    import app.door_view as door_view

    cases = door_view._MAINTENANCE_REFERENCE_CASES
    _check(len(cases) == 3, f"Door has exactly 3 reference cases, found {len(cases)}")
    for case in cases:
        _check_case_fields(case, "Door")

    ids = [case["reference_id"] for case in cases]
    _check(ids == ["DEMO-DOR-001", "DEMO-DOR-002", "DEMO-DOR-003"], f"Door reference IDs are exactly DEMO-DOR-001..003 in order, got {ids}")
    _check(all(case["last_updated"] == "2026-09-19" for case in cases), "every Door case's last_updated is 2026-09-19")


def check_rail_side_i_reference_cases() -> None:
    import app.rail_view as rail_view

    cases = rail_view._RAIL_SIDE_I_CASES
    _check(len(cases) == 3, f"Rail Side I has exactly 3 reference cases, found {len(cases)}")
    for case in cases:
        _check_case_fields(case, "Rail Side I")

    ids = [case["reference_id"] for case in cases]
    _check(ids == ["DEMO-RC-S1-001", "DEMO-RC-S1-002", "DEMO-RC-S1-003"], f"Rail Side I reference IDs are exactly DEMO-RC-S1-001..003 in order, got {ids}")
    _check(all(case["last_updated"] == "2026-09-19" for case in cases), "every Rail Side I case's last_updated is 2026-09-19")


def check_rail_side_ii_reference_cases() -> None:
    import app.rail_view as rail_view

    cases = rail_view._RAIL_SIDE_II_CASES
    _check(len(cases) == 3, f"Rail Side II has exactly 3 reference cases, found {len(cases)}")
    for case in cases:
        _check_case_fields(case, "Rail Side II")

    ids = [case["reference_id"] for case in cases]
    _check(ids == ["DEMO-RC-S2-001", "DEMO-RC-S2-002", "DEMO-RC-S2-003"], f"Rail Side II reference IDs are exactly DEMO-RC-S2-001..003 in order, got {ids}")
    _check(all(case["last_updated"] == "2026-09-19" for case in cases), "every Rail Side II case's last_updated is 2026-09-19")


def check_all_nine_reference_ids_are_globally_unique() -> None:
    import app.door_view as door_view
    import app.rail_view as rail_view

    all_ids = (
        [case["reference_id"] for case in door_view._MAINTENANCE_REFERENCE_CASES]
        + [case["reference_id"] for case in rail_view._RAIL_SIDE_I_CASES]
        + [case["reference_id"] for case in rail_view._RAIL_SIDE_II_CASES]
    )
    _check(len(all_ids) == 9, f"exactly 9 reference cases exist across Door + Rail Side I + Rail Side II, found {len(all_ids)}")
    _check(len(set(all_ids)) == 9, f"all 9 reference IDs are globally unique, got {all_ids}")


def check_side_i_and_side_ii_case_sets_never_overlap() -> None:
    import app.rail_view as rail_view

    side_i_ids = {case["reference_id"] for case in rail_view._RAIL_SIDE_I_CASES}
    side_ii_ids = {case["reference_id"] for case in rail_view._RAIL_SIDE_II_CASES}
    _check(side_i_ids.isdisjoint(side_ii_ids), "Rail Side I and Side II reference-case ID sets never overlap")
    _check(rail_view._RAIL_REFERENCE_CASES_BY_PREDICTION["Side I"][1] is rail_view._RAIL_SIDE_I_CASES, "the 'Side I' popover is wired to the Side I case tuple, not Side II's")
    _check(rail_view._RAIL_REFERENCE_CASES_BY_PREDICTION["Side II"][1] is rail_view._RAIL_SIDE_II_CASES, "the 'Side II' popover is wired to the Side II case tuple, not Side I's")


def main() -> int:
    check_door_reference_cases()
    check_rail_side_i_reference_cases()
    check_rail_side_ii_reference_cases()
    check_all_nine_reference_ids_are_globally_unique()
    check_side_i_and_side_ii_case_sets_never_overlap()

    print("\nAll Maintenance Knowledge Guide reference-case checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\nTEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
