"""Small synthetic Door streams; no organiser rows or labels are copied here."""

from __future__ import annotations

import pandas as pd

from src.door import config
from src.door.preprocess import PARSED_DATETIME_COLUMN, parse_door_datetime_series


def synthetic_stream() -> pd.DataFrame:
    times = ["2023-7-5-0-0-0-0", "2023-7-5-0-0-0-20",
             "2023-7-5-0-0-1-0", "2023-7-5-0-0-1-20"]
    data = {column: [0, 0, 0, 0] for column in config.EXPECTED_COLUMNS}
    data["Datetime"] = times
    data["Motor current(mA)"] = [10, 20, 30, 40]
    data["Motor Voltage(10mV)"] = [3, 4, 5, 6]
    data["Motor electrodynamic force"] = [1, 2, 3, 4]
    data["Door leaf position"] = [0, 1, 0, 2]
    frame = pd.DataFrame(data)
    frame[PARSED_DATETIME_COLUMN] = parse_door_datetime_series(frame["Datetime"])
    return frame
