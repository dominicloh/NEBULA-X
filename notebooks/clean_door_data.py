"""Stage 1 only: validate and clean a continuous Door sensor CSV.

Thin command-line wrapper around the real implementation in
``src/door/preprocess.py`` (``validate_and_clean`` / ``process_file``) --
this file holds no cleaning/validation logic of its own.

Run ``python notebooks/clean_door_data.py`` to process the repository's Train
and Test files (the organiser materials repo, a SIBLING of this repository,
not a subfolder of it -- see ``src/door/config.py::DEFAULT_DATASET_DIR``).
``validate_and_clean`` also accepts a DataFrame or uploaded file-like object
and returns ``(cleaned_dataframe, report)``. Ground-truth segment labels
(``Train_Segments_Answer.csv``) are deliberately outside this module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.door import config  # noqa: E402
from src.door.preprocess import process_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=[
            config.DEFAULT_DATASET_DIR / config.TRAIN_FILENAME,
            config.DEFAULT_DATASET_DIR / config.TEST_FILENAME,
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "door_stage1")
    args = parser.parse_args()
    for source in args.files:
        process_file(source, args.output_dir)


if __name__ == "__main__":
    main()
