"""BBL / CSV flight-log parser for BlackTune.

Three ingestion paths:
1. ``parse_csv_log``  -- parse a blackbox_decode CSV into a FlightLog
2. ``parse_headers``  -- extract H-line key:value pairs from raw BBL/BFL files
3. ``load_bbl_orangebox`` -- full BBL parsing via the orangebox library

``load_log`` auto-detects the format from the file extension.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Optional

import numpy as np

from blacktune.models import AxisData, FilterSettings, FlightLog, PIDValues

# ---------------------------------------------------------------------------
# CSV parsing (from blackbox_decode output)
# ---------------------------------------------------------------------------

def parse_csv_log(csv_path: str) -> FlightLog:
    """Parse a blackbox_decode CSV into a :class:`FlightLog`.

    Expected columns include ``loopIteration``, ``time (us)``,
    ``axisP[0..2]``, ``axisI[0..2]``, ``axisD[0..2]``, ``axisF[0..2]``,
    ``rcCommand[0..3]``, ``setpoint[0..3]``, ``gyroADC[0..2]``,
    ``motor[0..3]``, and optionally ``gyroUnfilt[0..2]``,
    ``eRPM[0..3]``, ``debug[0..7]``.

    Sample rate is estimated from the median time delta.
    """
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    n = len(rows)
    if n == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # helper: read a column into a float array, default to zeros
    def _col(name: str) -> np.ndarray:
        try:
            return np.array([float(r[name]) for r in rows], dtype=np.float64)
        except (KeyError, ValueError):
            return np.zeros(n, dtype=np.float64)

    # Time array (microseconds -> seconds)
    time_us = _col("time (us)")
    time_s = time_us / 1_000_000.0

    # Estimate sample rate from median dt
    if n > 1:
        dts = np.diff(time_us)
        median_dt = float(np.median(dts))
        sample_rate = int(round(1_000_000.0 / median_dt)) if median_dt > 0 else 1000
    else:
        sample_rate = 1000

    duration_s = float(time_s[-1] - time_s[0]) if n > 1 else 0.0

    # Check for optional columns
    has_unfilt = "gyroUnfilt[0]" in fieldnames
    has_erpm = "eRPM[0]" in fieldnames

    # Build AxisData for roll (index 0), pitch (index 1), yaw (index 2)
    axes = {}
    for idx, name in enumerate(("roll", "pitch", "yaw")):
        gyro_unfilt = _col(f"gyroUnfilt[{idx}]") if has_unfilt else None
        axes[name] = AxisData(
            name=name,
            gyro=_col(f"gyroADC[{idx}]"),
            setpoint=_col(f"setpoint[{idx}]"),
            p_term=_col(f"axisP[{idx}]"),
            i_term=_col(f"axisI[{idx}]"),
            d_term=_col(f"axisD[{idx}]"),
            time=time_s.copy(),
            f_term=_col(f"axisF[{idx}]"),
            gyro_unfiltered=gyro_unfilt,
        )

    # Throttle: rcCommand[3] is the raw stick value (typically ~1000-2000).
    throttle = _col("rcCommand[3]")

    # Motors: shape (4, N)
    motors = np.vstack([_col(f"motor[{m}]") for m in range(4)])

    # eRPM: shape (4, N) or None
    erpm: Optional[np.ndarray] = None
    if has_erpm:
        erpm = np.vstack([_col(f"eRPM[{m}]") for m in range(4)])

    # CSVs don't carry header info so PIDs default to zeros and firmware is
    # "unknown".  Callers that also have the BBL can merge header data later.
    default_pids = PIDValues(roll_p=0, roll_i=0, roll_d=0)

    return FlightLog(
        roll=axes["roll"],
        pitch=axes["pitch"],
        yaw=axes["yaw"],
        throttle=throttle,
        motors=motors,
        sample_rate=sample_rate,
        duration_s=duration_s,
        firmware="unknown",
        current_pids=default_pids,
        current_filters=None,
        erpm=erpm,
    )


# ---------------------------------------------------------------------------
# BBL header parsing
# ---------------------------------------------------------------------------

def parse_headers(bbl_path: str) -> Dict[str, str]:
    """Parse BBL file header lines (``H key:value`` format) into a dict.

    BBL headers are ASCII lines at the start of the file, each beginning
    with ``H ``.  Parsing stops at the first line that does not start with
    ``H `` (i.e. the binary frame data).

    Returns a dict mapping header key -> value (both strings).
    """
    headers: Dict[str, str] = {}
    with open(bbl_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line.startswith("H "):
                # First non-header line => done
                break
            # Strip the "H " prefix, then split on the FIRST colon
            payload = line[2:]
            colon_idx = payload.find(":")
            if colon_idx < 0:
                continue
            key = payload[:colon_idx].strip()
            value = payload[colon_idx + 1:].strip()
            headers[key] = value
    return headers


# ---------------------------------------------------------------------------
# PID extraction from headers
# ---------------------------------------------------------------------------

def pids_from_headers(headers: Dict[str, str]) -> PIDValues:
    """Extract :class:`PIDValues` from parsed BBL headers.

    Betaflight header keys:
    - ``rollPID``, ``pitchPID``, ``yawPID``: ``"P,I,D"``
    - ``dMax`` or ``d_max``: ``"roll,pitch,yaw"``
    - ``feedforward`` or ``f``: ``"roll,pitch,yaw"``
    """
    def _split3(key: str, *alt_keys: str) -> tuple[float, float, float]:
        """Return (a, b, c) from a ``"a,b,c"`` header value, or (0,0,0)."""
        for k in (key, *alt_keys):
            if k in headers:
                parts = headers[k].split(",")
                if len(parts) >= 3:
                    return float(parts[0]), float(parts[1]), float(parts[2])
        return 0.0, 0.0, 0.0

    roll_p, roll_i, roll_d = _split3("rollPID")
    pitch_p, pitch_i, pitch_d = _split3("pitchPID")
    yaw_p, yaw_i, yaw_d = _split3("yawPID")
    dmax_r, dmax_p, dmax_y = _split3("dMax", "d_max")
    ff_r, ff_p, ff_y = _split3("feedforward", "f")

    return PIDValues(
        roll_p=roll_p,
        roll_i=roll_i,
        roll_d=roll_d,
        roll_d_max=dmax_r,
        roll_f=ff_r,
        pitch_p=pitch_p,
        pitch_i=pitch_i,
        pitch_d=pitch_d,
        pitch_d_max=dmax_p,
        pitch_f=ff_p,
        yaw_p=yaw_p,
        yaw_i=yaw_i,
        yaw_d=yaw_d,
        yaw_f=ff_y,
    )


# ---------------------------------------------------------------------------
# Filter extraction from headers
# ---------------------------------------------------------------------------

def filters_from_headers(headers: Dict[str, str]) -> FilterSettings:
    """Extract :class:`FilterSettings` from parsed BBL headers.

    Missing keys fall back to the :class:`FilterSettings` dataclass defaults.
    """
    def _str(key: str, default: str) -> str:
        return headers.get(key, default)

    def _float(key: str, default: float) -> float:
        try:
            return float(headers[key])
        except (KeyError, ValueError):
            return default

    def _int(key: str, default: int) -> int:
        try:
            return int(headers[key])
        except (KeyError, ValueError):
            return default

    # Instantiate with defaults so any missing header gracefully falls back.
    defaults = FilterSettings()

    return FilterSettings(
        gyro_lpf1_type=_str("gyro_lpf1_type", defaults.gyro_lpf1_type),
        gyro_lpf1_hz=_float("gyro_lpf1_static_hz", defaults.gyro_lpf1_hz),
        gyro_lpf1_dyn_min_hz=_float("gyro_lpf1_dyn_min_hz", defaults.gyro_lpf1_dyn_min_hz),
        gyro_lpf1_dyn_max_hz=_float("gyro_lpf1_dyn_max_hz", defaults.gyro_lpf1_dyn_max_hz),
        gyro_lpf2_type=_str("gyro_lpf2_type", defaults.gyro_lpf2_type),
        gyro_lpf2_hz=_float("gyro_lpf2_static_hz", defaults.gyro_lpf2_hz),
        dterm_lpf1_type=_str("dterm_lpf1_type", defaults.dterm_lpf1_type),
        dterm_lpf1_hz=_float("dterm_lpf1_static_hz", defaults.dterm_lpf1_hz),
        dterm_lpf1_dyn_min_hz=_float("dterm_lpf1_dyn_min_hz", defaults.dterm_lpf1_dyn_min_hz),
        dterm_lpf1_dyn_max_hz=_float("dterm_lpf1_dyn_max_hz", defaults.dterm_lpf1_dyn_max_hz),
        dterm_lpf2_type=_str("dterm_lpf2_type", defaults.dterm_lpf2_type),
        dterm_lpf2_hz=_float("dterm_lpf2_static_hz", defaults.dterm_lpf2_hz),
        dyn_notch_count=_int("dyn_notch_count", defaults.dyn_notch_count),
        dyn_notch_q=_int("dyn_notch_q", defaults.dyn_notch_q),
        dyn_notch_min_hz=_float("dyn_notch_min_hz", defaults.dyn_notch_min_hz),
        dyn_notch_max_hz=_float("dyn_notch_max_hz", defaults.dyn_notch_max_hz),
        rpm_harmonics=_int("rpm_filter_harmonics", defaults.rpm_harmonics),
        rpm_min_hz=_float("rpm_filter_min_hz", defaults.rpm_min_hz),
        rpm_q=_int("rpm_filter_q", defaults.rpm_q),
    )


# ---------------------------------------------------------------------------
# Orangebox (pure-Python BBL parser)
# ---------------------------------------------------------------------------

def load_bbl_orangebox(bbl_path: str) -> FlightLog:
    """Parse a BBL file using orangebox (pure Python).

    Uses ``orangebox.Parser`` to read headers and iterate frames.  Falls back
    to header-only extraction if frame iteration fails.
    """
    from orangebox import Parser as OBParser

    parser = OBParser.load(bbl_path)

    # -- headers -> PID / filter / firmware --
    raw_headers: Dict[str, str] = {}
    for k, v in parser.headers.items():
        raw_headers[str(k)] = str(v)

    pids = pids_from_headers(raw_headers)
    filters = filters_from_headers(raw_headers)
    firmware = raw_headers.get("Firmware revision", "unknown")

    # -- frame data --
    field_names = list(parser.field_names)
    frames = list(parser.frames())

    n = len(frames)
    if n == 0:
        raise ValueError(f"No frames found in BBL file: {bbl_path}")

    # helper: extract a column by field name
    def _field(name: str) -> np.ndarray:
        if name in field_names:
            return np.array([f.data.get(name, 0) for f in frames], dtype=np.float64)
        return np.zeros(n, dtype=np.float64)

    time_us = _field("time")
    time_s = time_us / 1_000_000.0

    # sample rate from median dt
    if n > 1:
        dts = np.diff(time_us)
        dts = dts[dts > 0]
        median_dt = float(np.median(dts)) if len(dts) > 0 else 125.0
        sample_rate = int(round(1_000_000.0 / median_dt))
    else:
        sample_rate = 1000

    duration_s = float(time_s[-1] - time_s[0]) if n > 1 else 0.0

    # Check for optional fields
    has_unfilt = "gyroUnfilt[0]" in field_names
    has_erpm = "eRPM[0]" in field_names

    axes = {}
    for idx, name in enumerate(("roll", "pitch", "yaw")):
        gyro_unfilt = _field(f"gyroUnfilt[{idx}]") if has_unfilt else None
        axes[name] = AxisData(
            name=name,
            gyro=_field(f"gyroADC[{idx}]"),
            setpoint=_field(f"setpoint[{idx}]"),
            p_term=_field(f"axisP[{idx}]"),
            i_term=_field(f"axisI[{idx}]"),
            d_term=_field(f"axisD[{idx}]"),
            time=time_s.copy(),
            f_term=_field(f"axisF[{idx}]"),
            gyro_unfiltered=gyro_unfilt,
        )

    throttle = _field("rcCommand[3]")
    motors = np.vstack([_field(f"motor[{m}]") for m in range(4)])
    erpm: Optional[np.ndarray] = None
    if has_erpm:
        erpm = np.vstack([_field(f"eRPM[{m}]") for m in range(4)])

    return FlightLog(
        roll=axes["roll"],
        pitch=axes["pitch"],
        yaw=axes["yaw"],
        throttle=throttle,
        motors=motors,
        sample_rate=sample_rate,
        duration_s=duration_s,
        firmware=firmware,
        current_pids=pids,
        current_filters=filters,
        erpm=erpm,
    )


# ---------------------------------------------------------------------------
# Unified loader (auto-detect)
# ---------------------------------------------------------------------------

def load_log(path: str) -> FlightLog:
    """Auto-detect format and load a flight log.

    - ``.csv`` -> :func:`parse_csv_log`
    - ``.bbl`` / ``.bfl`` -> :func:`load_bbl_orangebox`
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return parse_csv_log(path)
    if ext in (".bbl", ".bfl"):
        return load_bbl_orangebox(path)
    raise ValueError(
        f"Unsupported file extension '{ext}'. Expected .csv, .bbl, or .bfl"
    )
