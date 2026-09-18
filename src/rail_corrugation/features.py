"""Rail Corrugation feature engineering.

WHY each CSV becomes exactly one feature row
---------------------------------------------
The competition task is *file-level* classification: one Rail Corrugation CSV
(one 1-second, 10,000-row recording) gets exactly one label (Normal / Side I /
Side II). A model can only be trained and scored at that same granularity, so
every function here reduces one whole file down to a single fixed-length
feature vector -- never a per-row prediction.

WHY feature extraction preserves Side I vs. Side II information
-----------------------------------------------------------------
The Info Kit confirms that axle-box positions 1, 3, 5, 7 are on the Side I
rail and positions 2, 4, 6, 8 are on the Side II rail, and that Side I and
Side II must be judged *independently from the same recording*. If we
averaged all 64 axle-box channels together we would destroy exactly the
signal the labels depend on. So every aggregation step below groups channels
by side (and by vibration vs. shock) *before* combining them, and we add
explicit Side I vs. Side II comparison features (difference, ratio, max) so
the model can directly see "is one side worse than the other?" rather than
having to re-derive it from 64 separate columns.

Column-name parsing is validated, not guessed: `validate_rail_columns` raises
a clear error if any expected column is missing or doesn't match the exact
header text confirmed in the dataset, rather than silently skipping it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Make "from src...." imports work when this module's functions are called
# from a script run directly (matches the convention used elsewhere in this
# package, e.g. inspect_data.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.validation import VALID_RAIL_LABELS  # noqa: E402  (Normal, Side I, Side II)

# Confirmed by the Rail Corrugation Info Kit: 10,000 Hz sampling frequency,
# 1 second per file (=> 10,000 rows/file). Exposed as a module constant so
# train.py and the Streamlit app use the exact same value as this module.
SAMPLING_FREQUENCY_HZ = 10_000

# Confirmed by the Info Kit (section 2.1): a toothed speed-sensor wheel with
# 90 teeth, wheel diameter 0.85 m. Used only to turn the raw 0/1 toggle
# "Rotating speed" column into an estimated train speed feature.
SPEED_SENSOR_TEETH = 90
WHEEL_DIAMETER_M = 0.85

# Small constant added to denominators so a ratio/division feature never
# raises ZeroDivisionError or produces inf/NaN on a near-silent channel.
_EPSILON = 1e-9

SPEED_COLUMN_NAME = "Rotating speed"

# Exact header text confirmed from the dataset's own first row, e.g.:
#   "Vibration of bearing in position 1 of car 1"
#   "Shock of bearing in position 3 of car 2"
_SIGNAL_COLUMN_PATTERN = re.compile(
    r"^(?P<signal_type>Vibration|Shock) of bearing in position (?P<position>\d+) of car (?P<car>\d+)$"
)

# Per the Info Kit: odd axle-box positions are on the Side I rail, even
# positions are on the Side II rail. Same rule for every car.
_SIDE_I_POSITIONS = frozenset({1, 3, 5, 7})
_SIDE_II_POSITIONS = frozenset({2, 4, 6, 8})

EXPECTED_CARS = tuple(range(1, 9))
EXPECTED_POSITIONS = tuple(range(1, 9))
EXPECTED_SIGNAL_TYPES = ("Vibration", "Shock")


def parse_signal_column(column_name: str) -> dict:
    """Parse one vibration/shock column name into its structured meaning.

    Returns a dict with: signal_type ("Vibration"/"Shock"), car (int),
    position (int, 1-8), and side ("Side I"/"Side II"). Raises ValueError if
    the column name does not match the exact format confirmed in the
    dataset -- we never guess a mapping for a column we don't recognise.
    """
    match = _SIGNAL_COLUMN_PATTERN.match(column_name)
    if match is None:
        raise ValueError(
            f"Could not parse Rail Corrugation column name: {column_name!r}. "
            "Expected the exact format 'Vibration of bearing in position N of car C' "
            "or 'Shock of bearing in position N of car C', as confirmed in the Info Kit."
        )
    position = int(match.group("position"))
    car = int(match.group("car"))
    if position in _SIDE_I_POSITIONS:
        side = "Side I"
    elif position in _SIDE_II_POSITIONS:
        side = "Side II"
    else:
        # Positions are only ever 1-8 per the Info Kit; anything else means
        # either a data-format change or a bug in the regex above.
        raise ValueError(
            f"Column {column_name!r} has axle-box position {position}, which is outside "
            "the confirmed 1-8 range (positions 1/3/5/7 = Side I, 2/4/6/8 = Side II)."
        )
    return {
        "column": column_name,
        "signal_type": match.group("signal_type"),
        "car": car,
        "position": position,
        "side": side,
    }


def validate_rail_columns(columns: Iterable[str]) -> dict:
    """Validate a Rail Corrugation file's columns against the confirmed schema.

    Confirms: column 1 is exactly "Rotating speed", and all 128 remaining
    columns parse as Vibration/Shock columns for cars 1-8 and positions 1-8,
    with none missing and none unexpected. Fails clearly (ValueError) rather
    than silently proceeding with a partial or reordered schema.

    Returns a dict with the speed column name and a list of parsed channel
    dicts (see `parse_signal_column`) for the remaining 128 columns.
    """
    columns = list(columns)
    if not columns or columns[0] != SPEED_COLUMN_NAME:
        found = repr(columns[0]) if columns else "(no columns)"
        raise ValueError(
            f"Expected column 1 to be {SPEED_COLUMN_NAME!r}, but found: {found}"
        )

    signal_columns = columns[1:]
    channels = [parse_signal_column(column) for column in signal_columns]

    expected_count = len(EXPECTED_CARS) * len(EXPECTED_POSITIONS) * len(EXPECTED_SIGNAL_TYPES)
    if len(channels) != expected_count:
        raise ValueError(
            f"Expected {expected_count} vibration/shock columns "
            f"({len(EXPECTED_CARS)} cars x {len(EXPECTED_POSITIONS)} positions x "
            f"{len(EXPECTED_SIGNAL_TYPES)} signal types), but found {len(channels)}."
        )

    seen = {(c["signal_type"], c["car"], c["position"]) for c in channels}
    expected = {
        (signal_type, car, position)
        for signal_type in EXPECTED_SIGNAL_TYPES
        for car in EXPECTED_CARS
        for position in EXPECTED_POSITIONS
    }
    missing = expected - seen
    if missing:
        raise ValueError(
            f"Rail Corrugation schema is missing {len(missing)} expected channel(s): "
            f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}"
        )

    return {"speed_column": columns[0], "channels": channels}


# ---------------------------------------------------------------------------
# Per-channel signal statistics (computed once per axle-box channel, then
# aggregated into compact side/signal-type groups -- see `extract_rail_features`)
# ---------------------------------------------------------------------------
def _band_energy_ratios(power_spectrum: np.ndarray, freqs: np.ndarray) -> tuple:
    """Split the (DC-excluded) power spectrum into 3 broad bands and return
    each band's share of total spectral energy (so the ratios are scale-free
    and comparable across channels of different amplitude).

    Band edges (0-500Hz / 500-2000Hz / 2000-5000Hz, with 5,000 Hz = Nyquist
    for the confirmed 10,000 Hz sampling rate) are a plain, generic 3-way
    split -- the Info Kit does not specify official corrugation frequency
    bands, so this is a modelling choice, not an organiser-confirmed fact.
    """
    total_energy = power_spectrum.sum()
    if total_energy <= _EPSILON:
        return (0.0, 0.0, 0.0)
    low = power_spectrum[freqs < 500].sum() / total_energy
    mid = power_spectrum[(freqs >= 500) & (freqs < 2000)].sum() / total_energy
    high = power_spectrum[freqs >= 2000].sum() / total_energy
    return (float(low), float(mid), float(high))


def compute_channel_stats(signal: np.ndarray, sampling_frequency_hz: int) -> dict:
    """Compute a compact battery of time- and frequency-domain stats for one
    1-D signal (one axle-box channel from one file).
    """
    signal = np.asarray(signal, dtype=float)
    n_samples = signal.shape[0]

    mean_val = float(signal.mean())
    std_val = float(signal.std())
    rms_val = float(np.sqrt(np.mean(np.square(signal))))
    abs_peak = float(np.max(np.abs(signal)))
    peak_to_peak = float(signal.max() - signal.min())

    # Excess kurtosis (0 for a perfect Gaussian), computed directly with numpy
    # so we don't need to add scipy as a new project dependency.
    if std_val > _EPSILON:
        kurtosis = float(np.mean((signal - mean_val) ** 4) / (std_val ** 4) - 3.0)
    else:
        kurtosis = 0.0

    crest_factor = abs_peak / (rms_val + _EPSILON)

    # Frequency-domain: real FFT, power spectrum, DC bin excluded so a
    # nonzero mean level doesn't dominate "spectral energy"/"dominant
    # frequency" (we already capture the mean level separately above).
    fft_values = np.fft.rfft(signal - mean_val)
    power_spectrum = np.abs(fft_values) ** 2
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency_hz)

    spectral_energy = float(power_spectrum.sum())
    if spectral_energy > _EPSILON:
        dominant_frequency = float(freqs[np.argmax(power_spectrum)])
        spectral_centroid = float(np.sum(freqs * power_spectrum) / spectral_energy)
    else:
        dominant_frequency = 0.0
        spectral_centroid = 0.0

    band_low, band_mid, band_high = _band_energy_ratios(power_spectrum, freqs)

    return {
        "mean": mean_val,
        "std": std_val,
        "rms": rms_val,
        "abs_peak": abs_peak,
        "peak_to_peak": peak_to_peak,
        "kurtosis": kurtosis,
        "crest_factor": crest_factor,
        "spectral_energy": spectral_energy,
        "dominant_frequency": dominant_frequency,
        "spectral_centroid": spectral_centroid,
        "band_energy_low_ratio": band_low,
        "band_energy_mid_ratio": band_mid,
        "band_energy_high_ratio": band_high,
    }


_CHANNEL_STAT_NAMES = (
    "mean", "std", "rms", "abs_peak", "peak_to_peak", "kurtosis", "crest_factor",
    "spectral_energy", "dominant_frequency", "spectral_centroid",
    "band_energy_low_ratio", "band_energy_mid_ratio", "band_energy_high_ratio",
)
# Stats we also summarise with a cross-channel std (in addition to the mean),
# to capture how consistently a fault shows up across the 8 cars on one side.
_SPREAD_STAT_NAMES = ("rms", "spectral_energy")

_GROUP_KEYS = (
    ("Vibration", "Side I"),
    ("Vibration", "Side II"),
    ("Shock", "Side I"),
    ("Shock", "Side II"),
)
_GROUP_LABELS = {
    ("Vibration", "Side I"): "vib_s1",
    ("Vibration", "Side II"): "vib_s2",
    ("Shock", "Side I"): "shock_s1",
    ("Shock", "Side II"): "shock_s2",
}


def _rotating_speed_features(speed_signal: np.ndarray, sampling_frequency_hz: int) -> dict:
    """Sensible summaries of the raw 'Rotating speed' toggle column.

    Per the Info Kit, this column is the raw 0/1 output of a 90-tooth speed
    sensor, not a pre-computed speed value. We summarise its level/spread
    directly, and also estimate an actual train speed (km/h) the way the
    Info Kit describes: count tooth-passing (rising-edge) transitions over
    the file's known 1-second duration, convert to wheel revolutions using
    the confirmed 90-tooth count, then to distance using the confirmed
    0.85 m wheel diameter.
    """
    speed_signal = np.asarray(speed_signal, dtype=float)
    duration_seconds = speed_signal.shape[0] / sampling_frequency_hz

    mean_val = float(speed_signal.mean())
    std_val = float(speed_signal.std())

    # Rising edges = the sensor toggling from 0 up to 1, i.e. one tooth
    # passing the detector.
    rising_edges = int(np.sum(np.diff(speed_signal) > 0))
    teeth_per_second = rising_edges / duration_seconds if duration_seconds > 0 else 0.0
    wheel_revs_per_second = teeth_per_second / SPEED_SENSOR_TEETH
    wheel_circumference_m = np.pi * WHEEL_DIAMETER_M
    estimated_speed_kmh = wheel_revs_per_second * wheel_circumference_m * 3.6

    return {
        "rotating_speed_mean": mean_val,
        "rotating_speed_std": std_val,
        "rotating_speed_toggle_count": float(rising_edges),
        "rotating_speed_estimated_kmh": float(estimated_speed_kmh),
    }


def extract_rail_features(frame: pd.DataFrame, sampling_frequency_hz: int = SAMPLING_FREQUENCY_HZ) -> pd.Series:
    """Reduce one Rail Corrugation CSV (already loaded as a DataFrame) to a
    single, fixed-length feature row.

    This is the one function both the training pipeline (train.py, reading
    files from disk) and the Streamlit app (reading an uploaded file into a
    DataFrame) should call, so features are computed identically in both
    places. It is deterministic: the same input frame always produces the
    same output Series, with no randomness and no dependence on the
    filename or on the order files happen to be processed in -- the
    filename/order are never read by this function at all.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("extract_rail_features expects a pandas DataFrame.")

    schema = validate_rail_columns(frame.columns)

    # --- Step 1: per-channel stats for all 128 vibration/shock columns -----
    channel_records = []
    for channel in schema["channels"]:
        stats = compute_channel_stats(frame[channel["column"]].to_numpy(), sampling_frequency_hz)
        channel_records.append({**channel, **stats})
    channel_df = pd.DataFrame(channel_records)

    features: dict = {}

    # --- Step 2: aggregate into 4 compact groups (signal type x side) ------
    group_summary: dict = {}
    for signal_type, side in _GROUP_KEYS:
        group_label = _GROUP_LABELS[(signal_type, side)]
        group_rows = channel_df[(channel_df["signal_type"] == signal_type) & (channel_df["side"] == side)]

        group_stats = {}
        for stat_name in _CHANNEL_STAT_NAMES:
            avg_value = float(group_rows[stat_name].mean())
            features[f"{group_label}_{stat_name}_avg"] = avg_value
            group_stats[stat_name] = avg_value
        # Cross-channel spread within the group: are all 8 cars on this side
        # agreeing, or is only one car driving the group average?
        for stat_name in _SPREAD_STAT_NAMES:
            features[f"{group_label}_{stat_name}_std"] = float(group_rows[stat_name].std())

        group_summary[(signal_type, side)] = group_stats

    # --- Step 3: car-level summary, "where useful" (kept compact) ----------
    # For each side, combine vibration+shock RMS per car (mean of its 4
    # positions on that side), then summarise across the 8 cars with max
    # (worst car) and std (car-to-car consistency) rather than emitting one
    # column per car (which would be 8x more features for little benefit).
    for side in ("Side I", "Side II"):
        side_rows = channel_df[channel_df["side"] == side]
        per_car_rms = side_rows.groupby("car")["rms"].mean()
        side_key = "s1" if side == "Side I" else "s2"
        features[f"{side_key}_car_rms_max"] = float(per_car_rms.max())
        features[f"{side_key}_car_rms_std"] = float(per_car_rms.std())

    # --- Step 4: direct Side I vs. Side II comparison features --------------
    # These are the features most directly tied to what the labels mean: is
    # one side showing more energy/vibration/shock than the other?
    side_i_rms = (group_summary[("Vibration", "Side I")]["rms"] + group_summary[("Shock", "Side I")]["rms"]) / 2.0
    side_ii_rms = (group_summary[("Vibration", "Side II")]["rms"] + group_summary[("Shock", "Side II")]["rms"]) / 2.0
    side_i_spectral_energy = (
        group_summary[("Vibration", "Side I")]["spectral_energy"] + group_summary[("Shock", "Side I")]["spectral_energy"]
    ) / 2.0
    side_ii_spectral_energy = (
        group_summary[("Vibration", "Side II")]["spectral_energy"] + group_summary[("Shock", "Side II")]["spectral_energy"]
    ) / 2.0

    features["side_energy_diff"] = side_i_rms - side_ii_rms
    features["side_energy_ratio"] = side_i_rms / (side_ii_rms + _EPSILON)
    features["max_side_energy"] = max(side_i_rms, side_ii_rms)
    features["vibration_rms_diff"] = (
        group_summary[("Vibration", "Side I")]["rms"] - group_summary[("Vibration", "Side II")]["rms"]
    )
    features["shock_rms_diff"] = (
        group_summary[("Shock", "Side I")]["rms"] - group_summary[("Shock", "Side II")]["rms"]
    )
    features["spectral_energy_diff"] = side_i_spectral_energy - side_ii_spectral_energy

    # --- Step 5: Rotating speed summaries -----------------------------------
    features.update(_rotating_speed_features(frame[schema["speed_column"]].to_numpy(), sampling_frequency_hz))

    # NOTE: nothing derived from the filename, file index, or the order files
    # are processed in is ever added to `features` -- those must never leak
    # into the model as predictive signal.
    return pd.Series(features)


def extract_features_for_files(
    file_paths: Iterable[Path],
    sampling_frequency_hz: int = SAMPLING_FREQUENCY_HZ,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extract features for many files, one file loaded at a time.

    Returns a DataFrame indexed by filename (not a feature column -- see the
    note in `extract_rail_features`), with one row per file.
    """
    file_paths = list(file_paths)
    rows = {}
    for index, path in enumerate(file_paths, start=1):
        path = Path(path)
        frame = pd.read_csv(path)
        rows[path.name] = extract_rail_features(frame, sampling_frequency_hz)
        del frame  # one file resident in memory at a time
        if verbose and (index % 50 == 0 or index == len(file_paths)):
            print(f"  ...extracted features for {index}/{len(file_paths)} files")
    return pd.DataFrame.from_dict(rows, orient="index")


__all__ = [
    "SAMPLING_FREQUENCY_HZ",
    "VALID_RAIL_LABELS",
    "parse_signal_column",
    "validate_rail_columns",
    "compute_channel_stats",
    "extract_rail_features",
    "extract_features_for_files",
]
