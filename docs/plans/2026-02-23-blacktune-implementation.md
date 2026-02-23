# BlackTune Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a desktop app that reads Betaflight blackbox logs, analyzes flight dynamics, and auto-suggests PID + filter settings based on quad profile.

**Architecture:** Python desktop app (PyQt6 + pyqtgraph). BBL files decoded via bundled `blackbox_decode` -> CSV -> NumPy arrays. Analysis engine computes FFT noise, step response (Wiener deconvolution), issue detection, motor heat estimation. Rule-based optimizer suggests PIDs and filters scaled by cell count / prop size. Dark-themed UI with 4 tabs.

**Tech Stack:** Python 3.11+, PyQt6, pyqtgraph, NumPy, SciPy, pandas

---

## Task 1: Project Scaffolding

**Files:**
- Create: `blacktune/main.py`
- Create: `blacktune/__init__.py`
- Create: `blacktune/requirements.txt`
- Create: `tests/__init__.py`

**Step 1: Create directory structure**

```bash
cd C:/Users/zacle/blacktune
mkdir -p blacktune tests
```

**Step 2: Create requirements.txt**

```
PyQt6>=6.6.0
pyqtgraph>=0.13.3
numpy>=1.26.0
scipy>=1.12.0
pandas>=2.1.0
pytest>=8.0.0
```

**Step 3: Create venv and install deps**

```bash
cd C:/Users/zacle/blacktune
python -m venv venv
venv/Scripts/pip install -r blacktune/requirements.txt
```

**Step 4: Create main.py entry point**

```python
"""BlackTune - FPV PID Autotuner."""
import sys
from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BlackTune")
    # Window created in Task 11
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

**Step 5: Create __init__.py**

```python
"""BlackTune - FPV PID Autotuner."""
__version__ = "0.1.0"
```

**Step 6: Verify install**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -c "import PyQt6; import pyqtgraph; import numpy; import scipy; print('OK')"`
Expected: `OK`

**Step 7: Commit**

```bash
git init
git add -A
git commit -m "feat: project scaffolding with venv and deps"
```

---

## Task 2: Data Models

**Files:**
- Create: `blacktune/models.py`
- Create: `tests/test_models.py`

**Step 1: Write failing tests**

```python
# tests/test_models.py
import numpy as np
from blacktune.models import (
    FlightLog, QuadProfile, PIDValues, FilterSettings,
    AxisData, AnalysisResult, TuneRecommendation,
)


def test_quad_profile_defaults():
    qp = QuadProfile(cell_count=6, prop_size=5.0)
    assert qp.cell_count == 6
    assert qp.prop_size == 5.0
    assert qp.frame_size is None
    assert qp.flying_style == "freestyle"


def test_pid_values():
    pids = PIDValues(roll_p=45, roll_i=80, roll_d=35,
                     pitch_p=47, pitch_i=84, pitch_d=38,
                     yaw_p=45, yaw_i=90, yaw_d=0)
    assert pids.roll_p == 45
    assert pids.as_dict()["roll_p"] == 45


def test_filter_settings():
    fs = FilterSettings(
        gyro_lpf1_type="PT1", gyro_lpf1_hz=250,
        gyro_lpf2_type="PT1", gyro_lpf2_hz=500,
        dterm_lpf1_type="PT1", dterm_lpf1_hz=150,
        dterm_lpf2_type="PT1", dterm_lpf2_hz=150,
        dyn_notch_count=4, dyn_notch_q=3.5,
        dyn_notch_min_hz=100, dyn_notch_max_hz=600,
        rpm_harmonics=3, rpm_min_hz=100, rpm_q=500,
    )
    assert fs.gyro_lpf1_hz == 250
    assert fs.rpm_harmonics == 3


def test_axis_data():
    t = np.linspace(0, 1, 1000)
    gyro = np.sin(2 * np.pi * 50 * t)
    ad = AxisData(name="roll", gyro=gyro, setpoint=gyro * 0.9,
                  p_term=gyro * 0.5, i_term=gyro * 0.1,
                  d_term=gyro * 0.05, time=t)
    assert ad.name == "roll"
    assert len(ad.gyro) == 1000


def test_flight_log():
    t = np.linspace(0, 1, 1000)
    zero = np.zeros(1000)
    ad = AxisData(name="roll", gyro=zero, setpoint=zero,
                  p_term=zero, i_term=zero, d_term=zero, time=t)
    log = FlightLog(
        roll=ad, pitch=ad, yaw=ad,
        throttle=zero, motors=np.zeros((4, 1000)),
        sample_rate=1000, duration_s=1.0,
        firmware="Betaflight 4.4.0",
        current_pids=PIDValues(45, 80, 35, 47, 84, 38, 45, 90, 0),
        current_filters=None,
    )
    assert log.sample_rate == 1000
    assert log.firmware == "Betaflight 4.4.0"
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement models.py**

```python
"""Data models for BlackTune."""
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np


@dataclass
class PIDValues:
    roll_p: float
    roll_i: float
    roll_d: float
    pitch_p: float
    pitch_i: float
    pitch_d: float
    yaw_p: float
    yaw_i: float
    yaw_d: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilterSettings:
    gyro_lpf1_type: str = "PT1"
    gyro_lpf1_hz: float = 250
    gyro_lpf2_type: str = "PT1"
    gyro_lpf2_hz: float = 500
    dterm_lpf1_type: str = "PT1"
    dterm_lpf1_hz: float = 150
    dterm_lpf2_type: str = "PT1"
    dterm_lpf2_hz: float = 150
    dyn_notch_count: int = 4
    dyn_notch_q: float = 3.5
    dyn_notch_min_hz: float = 100
    dyn_notch_max_hz: float = 600
    rpm_harmonics: int = 3
    rpm_min_hz: float = 100
    rpm_q: float = 500


@dataclass
class AxisData:
    name: str
    gyro: np.ndarray
    setpoint: np.ndarray
    p_term: np.ndarray
    i_term: np.ndarray
    d_term: np.ndarray
    time: np.ndarray
    gyro_unfiltered: Optional[np.ndarray] = None


@dataclass
class QuadProfile:
    cell_count: int
    prop_size: float
    frame_size: Optional[str] = None
    flying_style: str = "freestyle"


@dataclass
class FlightLog:
    roll: AxisData
    pitch: AxisData
    yaw: AxisData
    throttle: np.ndarray
    motors: np.ndarray  # shape (4, N)
    sample_rate: int
    duration_s: float
    firmware: str
    current_pids: PIDValues
    current_filters: Optional[FilterSettings] = None


@dataclass
class Issue:
    axis: str
    category: str  # "P_HIGH", "P_LOW", "D_HIGH", etc.
    severity: str  # "red", "yellow", "green"
    message: str
    detail: str


@dataclass
class StepResponseMetrics:
    rise_time_ms: float
    overshoot_pct: float
    settling_time_ms: float
    peak_time_ms: float
    steady_state_error: float


@dataclass
class AnalysisResult:
    step_response: dict  # axis -> StepResponseMetrics
    noise_peaks: dict    # axis -> list of (freq_hz, amplitude_db)
    issues: list         # list of Issue
    motor_heat_index: dict  # motor_idx -> float (0-1)
    d_term_rms: dict     # axis -> float
    filter_delay_ms: Optional[float] = None


@dataclass
class TuneRecommendation:
    suggested_pids: PIDValues
    suggested_filters: FilterSettings
    confidence: float  # 0.0 - 1.0
    explanations: dict  # param_name -> str reason
    cli_commands: str   # Betaflight CLI set commands
```

**Step 4: Run tests**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add blacktune/models.py tests/test_models.py
git commit -m "feat: add data models for flight log, PIDs, filters, analysis"
```

---

## Task 3: BBL Parser (blackbox_decode wrapper)

**Files:**
- Create: `blacktune/parser.py`
- Create: `tests/test_parser.py`
- Create: `tests/fixtures/` (test data dir)

**Step 1: Download blackbox_decode**

Download from https://github.com/betaflight/blackbox-tools/releases -- get `blackbox-tools-*-windows-amd64.zip`, extract `blackbox_decode.exe` to `blacktune/bin/blackbox_decode.exe`.

```bash
mkdir -p C:/Users/zacle/blacktune/blacktune/bin
# Manual download or:
# curl -L -o bb-tools.zip https://github.com/betaflight/blackbox-tools/releases/latest/download/blackbox-tools-0.4.5-windows-amd64.zip
# unzip bb-tools.zip -d bb-tools-tmp
# cp bb-tools-tmp/blackbox_decode.exe blacktune/bin/
```

**Step 2: Write failing test for CSV parsing**

```python
# tests/test_parser.py
import numpy as np
import tempfile
import os
from pathlib import Path
from blacktune.parser import parse_csv_log, parse_header_from_csv


SAMPLE_HEADER = """loopIteration,time (us),axisP[0],axisP[1],axisP[2],axisI[0],axisI[1],axisI[2],axisD[0],axisD[1],axisD[2],axisF[0],axisF[1],axisF[2],rcCommand[0],rcCommand[1],rcCommand[2],rcCommand[3],setpoint[0],setpoint[1],setpoint[2],setpoint[3],gyroADC[0],gyroADC[1],gyroADC[2],motor[0],motor[1],motor[2],motor[3]
"""

SAMPLE_ROW = "1,1000,10,12,5,50,55,40,3,4,0,0,0,0,0,0,0,1500,0,0,0,0,1.5,2.0,0.5,1200,1200,1200,1200"


def _make_csv(rows=100, sample_rate_us=1000):
    """Generate a minimal valid decoded CSV."""
    lines = [SAMPLE_HEADER.strip()]
    for i in range(rows):
        t = i * sample_rate_us
        lines.append(
            f"{i},{t},{10+i%5},{12+i%5},{5},{50},{55},{40},{3},{4},{0},"
            f"{0},{0},{0},{0},{0},{0},{1500},"
            f"{0.0},{0.0},{0.0},{0.0},"
            f"{1.5+0.1*(i%10)},{2.0+0.1*(i%10)},{0.5},"
            f"{1200},{1200},{1200},{1200}"
        )
    return "\n".join(lines)


def test_parse_csv_log():
    csv_text = _make_csv(rows=200, sample_rate_us=1000)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_text)
        f.flush()
        path = f.name
    try:
        log = parse_csv_log(path)
        assert log.sample_rate > 0
        assert len(log.roll.gyro) == 200
        assert len(log.roll.p_term) == 200
        assert log.motors.shape == (4, 200)
    finally:
        os.unlink(path)
```

**Step 3: Run test to verify it fails**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_parser.py -v`
Expected: FAIL (ImportError)

**Step 4: Implement parser.py**

```python
"""BBL/CSV parser for Betaflight blackbox logs."""
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from blacktune.models import (
    AxisData, FilterSettings, FlightLog, PIDValues,
)

# Path to bundled blackbox_decode
_BIN_DIR = Path(__file__).parent / "bin"
_DECODER = _BIN_DIR / "blackbox_decode.exe"


def decode_bbl(bbl_path: str, output_dir: Optional[str] = None) -> list[str]:
    """Run blackbox_decode on a .bbl/.bfl file, return list of CSV paths."""
    bbl_path = Path(bbl_path)
    if not bbl_path.exists():
        raise FileNotFoundError(f"BBL file not found: {bbl_path}")

    if not _DECODER.exists():
        raise FileNotFoundError(
            f"blackbox_decode not found at {_DECODER}. "
            "Download from https://github.com/betaflight/blackbox-tools/releases"
        )

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="blacktune_")

    cmd = [str(_DECODER), str(bbl_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir,
                            timeout=60)

    if result.returncode != 0:
        raise RuntimeError(f"blackbox_decode failed: {result.stderr}")

    # blackbox_decode creates files like: logname.01.csv, logname.02.csv, etc.
    csv_files = sorted(Path(bbl_path.parent).glob(f"{bbl_path.stem}*.csv"))
    if not csv_files:
        # Check output_dir too
        csv_files = sorted(Path(output_dir).glob("*.csv"))
    return [str(f) for f in csv_files]


def parse_csv_log(csv_path: str) -> FlightLog:
    """Parse a decoded blackbox CSV into a FlightLog."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Clean column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Time array
    if "time (us)" in df.columns:
        time_us = df["time (us)"].to_numpy(dtype=np.float64)
    elif "time" in df.columns:
        time_us = df["time"].to_numpy(dtype=np.float64)
    else:
        raise ValueError("No time column found in CSV")

    time_s = (time_us - time_us[0]) / 1e6
    n = len(time_s)

    # Estimate sample rate from median time delta
    dt_us = np.median(np.diff(time_us))
    sample_rate = int(round(1e6 / dt_us))
    duration_s = time_s[-1] - time_s[0]

    def _col(name, default=None):
        if name in df.columns:
            return df[name].to_numpy(dtype=np.float64)
        if default is not None:
            return np.full(n, default, dtype=np.float64)
        return np.zeros(n, dtype=np.float64)

    # Build axis data
    def _axis(idx, name):
        return AxisData(
            name=name,
            gyro=_col(f"gyroADC[{idx}]"),
            setpoint=_col(f"setpoint[{idx}]"),
            p_term=_col(f"axisP[{idx}]"),
            i_term=_col(f"axisI[{idx}]"),
            d_term=_col(f"axisD[{idx}]"),
            time=time_s,
            gyro_unfiltered=_col(f"gyroUnfilt[{idx}]") if f"gyroUnfilt[{idx}]" in df.columns else None,
        )

    roll = _axis(0, "roll")
    pitch = _axis(1, "pitch")
    yaw = _axis(2, "yaw")

    # Throttle
    throttle = _col("rcCommand[3]")

    # Motors (4xN)
    motors = np.stack([_col(f"motor[{i}]") for i in range(4)])

    # Parse PIDs from header comments if available (blackbox_decode puts them
    # in a separate .txt or the first lines of the CSV as comments).
    # For now use placeholder -- Task 3b will parse headers.
    pids = PIDValues(
        roll_p=0, roll_i=0, roll_d=0,
        pitch_p=0, pitch_i=0, pitch_d=0,
        yaw_p=0, yaw_i=0, yaw_d=0,
    )

    firmware = ""

    return FlightLog(
        roll=roll, pitch=pitch, yaw=yaw,
        throttle=throttle, motors=motors,
        sample_rate=sample_rate,
        duration_s=duration_s,
        firmware=firmware,
        current_pids=pids,
        current_filters=None,
    )


def parse_headers(bbl_path: str) -> dict:
    """Parse BBL header lines to extract PID/filter/firmware settings.

    BBL files start with ASCII header lines like:
      H Product:Blackbox flight data recorder by Nicholas Sherlock
      H Firmware type:Betaflight
      H Firmware revision:4.4.0
      H rollPID:45,80,35
      H pitchPID:47,84,38
      H yawPID:45,90,0
      H gyro_lpf1_static_hz:250
      ...
    """
    headers = {}
    with open(bbl_path, "rb") as f:
        for raw_line in f:
            try:
                line = raw_line.decode("ascii", errors="ignore").strip()
            except Exception:
                break
            if not line.startswith("H "):
                break
            # Parse "H key:value"
            match = re.match(r"H\s+(.+?):(.+)", line)
            if match:
                headers[match.group(1).strip()] = match.group(2).strip()
    return headers


def pids_from_headers(headers: dict) -> PIDValues:
    """Extract PIDValues from parsed BBL headers."""
    def _parse_pid(key):
        val = headers.get(key, "0,0,0")
        parts = [float(x) for x in val.split(",")]
        while len(parts) < 3:
            parts.append(0)
        return parts

    r = _parse_pid("rollPID")
    p = _parse_pid("pitchPID")
    y = _parse_pid("yawPID")
    return PIDValues(
        roll_p=r[0], roll_i=r[1], roll_d=r[2],
        pitch_p=p[0], pitch_i=p[1], pitch_d=p[2],
        yaw_p=y[0], yaw_i=y[1], yaw_d=y[2],
    )


def filters_from_headers(headers: dict) -> FilterSettings:
    """Extract FilterSettings from parsed BBL headers."""
    def _get(key, default, cast=float):
        return cast(headers.get(key, default))

    return FilterSettings(
        gyro_lpf1_type=headers.get("gyro_lpf1_type", "PT1"),
        gyro_lpf1_hz=_get("gyro_lpf1_static_hz", 250),
        gyro_lpf2_type=headers.get("gyro_lpf2_type", "PT1"),
        gyro_lpf2_hz=_get("gyro_lpf2_static_hz", 500),
        dterm_lpf1_type=headers.get("dterm_lpf1_type", "PT1"),
        dterm_lpf1_hz=_get("dterm_lpf1_static_hz", 150),
        dterm_lpf2_type=headers.get("dterm_lpf2_type", "PT1"),
        dterm_lpf2_hz=_get("dterm_lpf2_static_hz", 150),
        dyn_notch_count=_get("dyn_notch_count", 4, int),
        dyn_notch_q=_get("dyn_notch_q", 3.5),
        dyn_notch_min_hz=_get("dyn_notch_min_hz", 100),
        dyn_notch_max_hz=_get("dyn_notch_max_hz", 600),
        rpm_harmonics=_get("dshot_bidir", 0, int),  # 0 = off
        rpm_min_hz=_get("rpm_filter_min_hz", 100),
        rpm_q=_get("rpm_filter_q", 500),
    )


def load_bbl(bbl_path: str) -> FlightLog:
    """Full pipeline: decode BBL -> parse CSV -> enrich with headers."""
    headers = parse_headers(bbl_path)
    csv_files = decode_bbl(bbl_path)
    if not csv_files:
        raise RuntimeError("No CSV files produced by blackbox_decode")

    log = parse_csv_log(csv_files[0])
    log.current_pids = pids_from_headers(headers)
    log.current_filters = filters_from_headers(headers)
    log.firmware = headers.get("Firmware revision", "")
    return log
```

**Step 5: Run tests**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_parser.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add blacktune/parser.py tests/test_parser.py
git commit -m "feat: BBL parser with blackbox_decode wrapper and CSV parsing"
```

---

## Task 4: Noise Analyzer (FFT + Spectrogram)

**Files:**
- Create: `blacktune/analyzers/noise.py`
- Create: `blacktune/analyzers/__init__.py`
- Create: `tests/test_noise.py`

**Step 1: Write failing tests**

```python
# tests/test_noise.py
import numpy as np
from blacktune.analyzers.noise import (
    compute_fft_spectrum, compute_spectrogram, find_noise_peaks,
)


def _make_signal(freq_hz, sample_rate=2000, duration=2.0, noise_amp=0.1):
    """Sine wave at freq_hz plus noise."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    signal = np.sin(2 * np.pi * freq_hz * t) + noise_amp * np.random.randn(len(t))
    return t, signal


def test_fft_finds_peak():
    t, sig = _make_signal(200, sample_rate=2000)
    freqs, psd_db = compute_fft_spectrum(sig, sample_rate=2000)
    # Peak should be near 200 Hz
    peak_idx = np.argmax(psd_db)
    assert abs(freqs[peak_idx] - 200) < 5


def test_spectrogram_shape():
    t, sig = _make_signal(200, sample_rate=2000, duration=5.0)
    throttle = np.linspace(0, 100, len(sig))
    freq_bins, throttle_bins, spec_db = compute_spectrogram(
        sig, throttle, sample_rate=2000, n_throttle_bins=10
    )
    assert len(throttle_bins) == 10
    assert len(freq_bins) > 0
    assert spec_db.shape == (len(freq_bins), len(throttle_bins))


def test_find_noise_peaks():
    t, sig = _make_signal(300, sample_rate=2000)
    # Add a second harmonic
    sig += 0.5 * np.sin(2 * np.pi * 600 * t)
    freqs, psd_db = compute_fft_spectrum(sig, sample_rate=2000)
    peaks = find_noise_peaks(freqs, psd_db, min_prominence_db=10)
    peak_freqs = [p[0] for p in peaks]
    assert any(abs(f - 300) < 10 for f in peak_freqs)
    assert any(abs(f - 600) < 10 for f in peak_freqs)
```

**Step 2: Run tests to verify fail**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_noise.py -v`
Expected: FAIL

**Step 3: Implement noise analyzer**

```python
# blacktune/analyzers/__init__.py
"""Analysis engines for BlackTune."""

# blacktune/analyzers/noise.py
"""FFT noise analysis for gyro data."""
import numpy as np
from scipy.signal import welch, find_peaks


def compute_fft_spectrum(
    signal: np.ndarray,
    sample_rate: int,
    nperseg: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method.

    Returns (frequencies_hz, psd_db).
    """
    freqs, psd = welch(signal, fs=sample_rate, nperseg=nperseg,
                       window="hann", noverlap=nperseg // 2)
    # Convert to dB (avoid log of zero)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
    return freqs, psd_db


def compute_spectrogram(
    signal: np.ndarray,
    throttle: np.ndarray,
    sample_rate: int,
    n_throttle_bins: int = 20,
    nperseg: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D spectrogram binned by throttle position.

    Returns (freq_bins, throttle_bins, spec_db) where spec_db has shape
    (n_freq, n_throttle_bins).
    """
    throttle_edges = np.linspace(0, 100, n_throttle_bins + 1)
    throttle_centers = (throttle_edges[:-1] + throttle_edges[1:]) / 2

    # Normalize throttle to 0-100 if needed
    thr = throttle.copy()
    if thr.max() > 100:
        thr = (thr - thr.min()) / (thr.max() - thr.min()) * 100

    # Compute FFT for each throttle bin
    freq_bins = None
    spectra = []

    for i in range(n_throttle_bins):
        mask = (thr >= throttle_edges[i]) & (thr < throttle_edges[i + 1])
        seg = signal[mask]
        if len(seg) < nperseg:
            if freq_bins is not None:
                spectra.append(np.full(len(freq_bins), -60.0))
            else:
                spectra.append(None)
            continue
        f, psd = welch(seg, fs=sample_rate, nperseg=nperseg,
                       window="hann", noverlap=nperseg // 2)
        psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
        if freq_bins is None:
            freq_bins = f
        spectra.append(psd_db)

    # Fill any None entries
    if freq_bins is None:
        freq_bins = np.linspace(0, sample_rate / 2, nperseg // 2 + 1)
    for i, s in enumerate(spectra):
        if s is None:
            spectra[i] = np.full(len(freq_bins), -60.0)

    spec_db = np.column_stack(spectra)
    return freq_bins, throttle_centers, spec_db


def find_noise_peaks(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    min_prominence_db: float = 6.0,
    min_freq_hz: float = 50.0,
    max_freq_hz: float = 1000.0,
) -> list[tuple[float, float]]:
    """Find significant noise peaks in a spectrum.

    Returns list of (frequency_hz, amplitude_db) sorted by amplitude descending.
    """
    # Mask to frequency range
    mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    masked_freqs = freqs[mask]
    masked_psd = psd_db[mask]

    peaks, props = find_peaks(masked_psd, prominence=min_prominence_db,
                              distance=5)
    results = []
    for idx in peaks:
        results.append((float(masked_freqs[idx]), float(masked_psd[idx])))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
```

**Step 4: Run tests**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_noise.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add blacktune/analyzers/ tests/test_noise.py
git commit -m "feat: FFT noise analyzer with spectrogram and peak detection"
```

---

## Task 5: Step Response Analyzer (Wiener Deconvolution)

**Files:**
- Create: `blacktune/analyzers/step_response.py`
- Create: `tests/test_step_response.py`

**Step 1: Write failing tests**

```python
# tests/test_step_response.py
import numpy as np
from blacktune.analyzers.step_response import (
    compute_step_response, measure_step_metrics,
)


def _make_step_data(overshoot_pct=10, rise_time_ms=30, sample_rate=2000):
    """Simulate a second-order step response."""
    duration = 2.0
    t = np.arange(0, duration, 1.0 / sample_rate)
    n = len(t)

    # Create setpoint: step from 0 to 200 deg/s at t=0.5
    setpoint = np.zeros(n)
    step_idx = int(0.5 * sample_rate)
    setpoint[step_idx:] = 200.0

    # Simulate underdamped response
    zeta = 1.0 - overshoot_pct / 100.0  # approximate
    wn = 2.2 / (rise_time_ms / 1000.0)  # natural frequency
    gyro = np.zeros(n)
    for i in range(step_idx, n):
        t_step = (i - step_idx) / sample_rate
        wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else wn
        if zeta < 1:
            gyro[i] = 200.0 * (1 - np.exp(-zeta * wn * t_step) *
                       (np.cos(wd * t_step) +
                        (zeta / np.sqrt(1 - zeta**2)) * np.sin(wd * t_step)))
        else:
            gyro[i] = 200.0 * (1 - np.exp(-wn * t_step))

    # Add small noise
    gyro += np.random.randn(n) * 0.5
    return t, setpoint, gyro


def test_step_response_shape():
    t, sp, gy = _make_step_data()
    response, resp_time = compute_step_response(sp, gy, sample_rate=2000)
    assert len(response) > 0
    assert len(resp_time) == len(response)


def test_step_metrics_overshoot():
    # Create a response with clear overshoot
    resp_time = np.linspace(0, 0.5, 1000)
    response = np.ones(1000)
    response[200:400] = 1.15  # 15% overshoot
    metrics = measure_step_metrics(response, resp_time)
    assert metrics.overshoot_pct > 10
    assert metrics.overshoot_pct < 20
```

**Step 2: Run tests to verify fail**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_step_response.py -v`
Expected: FAIL

**Step 3: Implement step response analyzer**

```python
# blacktune/analyzers/step_response.py
"""Step response analysis via Wiener deconvolution."""
import numpy as np
from blacktune.models import StepResponseMetrics


def compute_step_response(
    setpoint: np.ndarray,
    gyro: np.ndarray,
    sample_rate: int,
    segment_duration_s: float = 2.0,
    response_window_s: float = 0.5,
    min_input_dps: float = 20.0,
    regularization: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute step response using Wiener deconvolution.

    Based on PIDtoolbox's PTstepcalc algorithm and Plasmatree PID-Analyzer.

    Returns (mean_step_response, time_array_seconds).
    """
    seg_len = int(segment_duration_s * sample_rate)
    resp_len = int(response_window_s * sample_rate)
    n = len(setpoint)

    if n < seg_len:
        seg_len = n

    responses = []
    window = np.hanning(seg_len)

    # Slide through data in overlapping segments
    step = seg_len // 4
    for start in range(0, n - seg_len, step):
        sp_seg = setpoint[start:start + seg_len]
        gy_seg = gyro[start:start + seg_len]

        # Skip low-input segments
        if np.max(np.abs(sp_seg)) < min_input_dps:
            continue

        # Apply window
        sp_w = sp_seg * window
        gy_w = gy_seg * window

        # FFT
        SP = np.fft.rfft(sp_w)
        GY = np.fft.rfft(gy_w)

        # Wiener deconvolution: H = GY * conj(SP) / (SP * conj(SP) + reg)
        SP_conj = np.conj(SP)
        H = (GY * SP_conj) / (SP * SP_conj + regularization)

        # Inverse FFT -> impulse response
        impulse = np.fft.irfft(H, n=seg_len)

        # Cumulative sum -> step response
        step_resp = np.cumsum(impulse[:resp_len])

        # Normalize so target = 1.0
        if np.max(np.abs(step_resp)) > 0:
            # Normalize by the value it settles to (last 20%)
            settle_val = np.mean(step_resp[int(resp_len * 0.8):])
            if abs(settle_val) > 0.01:
                step_resp = step_resp / settle_val

        responses.append(step_resp)

    if not responses:
        resp_time = np.linspace(0, response_window_s, resp_len)
        return np.ones(resp_len), resp_time

    mean_response = np.mean(responses, axis=0)
    resp_time = np.linspace(0, response_window_s, resp_len)
    return mean_response, resp_time


def measure_step_metrics(
    response: np.ndarray,
    resp_time: np.ndarray,
) -> StepResponseMetrics:
    """Extract metrics from a normalized step response (target = 1.0)."""
    target = 1.0

    # Peak
    peak_val = np.max(response)
    peak_idx = np.argmax(response)
    peak_time_ms = resp_time[peak_idx] * 1000

    # Overshoot
    overshoot_pct = max(0.0, (peak_val - target) / target * 100)

    # Rise time: time from 10% to 90% of target
    try:
        idx_10 = np.where(response >= 0.1 * target)[0][0]
        idx_90 = np.where(response >= 0.9 * target)[0][0]
        rise_time_ms = (resp_time[idx_90] - resp_time[idx_10]) * 1000
    except IndexError:
        rise_time_ms = resp_time[-1] * 1000

    # Settling time: time until response stays within 5% of target
    tolerance = 0.05 * target
    settled = np.abs(response - target) < tolerance
    settling_idx = len(response) - 1
    for i in range(len(response) - 1, -1, -1):
        if not settled[i]:
            settling_idx = min(i + 1, len(response) - 1)
            break
    settling_time_ms = resp_time[settling_idx] * 1000

    # Steady-state error
    ss_val = np.mean(response[-max(1, len(response) // 5):])
    steady_state_error = abs(ss_val - target)

    return StepResponseMetrics(
        rise_time_ms=rise_time_ms,
        overshoot_pct=overshoot_pct,
        settling_time_ms=settling_time_ms,
        peak_time_ms=peak_time_ms,
        steady_state_error=steady_state_error,
    )
```

**Step 4: Run tests**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_step_response.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add blacktune/analyzers/step_response.py tests/test_step_response.py
git commit -m "feat: step response via Wiener deconvolution with metrics"
```

---

## Task 6: Issue Detector + Motor Heat Estimation

**Files:**
- Create: `blacktune/analyzers/issues.py`
- Create: `tests/test_issues.py`

**Step 1: Write failing tests**

```python
# tests/test_issues.py
import numpy as np
from blacktune.models import StepResponseMetrics, Issue
from blacktune.analyzers.issues import (
    detect_pid_issues, estimate_motor_heat, compute_dterm_rms,
)


def test_detect_high_p():
    metrics = StepResponseMetrics(
        rise_time_ms=15, overshoot_pct=25,
        settling_time_ms=200, peak_time_ms=20,
        steady_state_error=0.01,
    )
    noise_peaks = [(300, -10)]  # one noise peak
    issues = detect_pid_issues("roll", metrics, noise_peaks, d_rms=5.0)
    categories = [i.category for i in issues]
    assert "P_HIGH" in categories


def test_detect_low_p():
    metrics = StepResponseMetrics(
        rise_time_ms=80, overshoot_pct=0,
        settling_time_ms=150, peak_time_ms=100,
        steady_state_error=0.05,
    )
    issues = detect_pid_issues("roll", metrics, [], d_rms=5.0)
    categories = [i.category for i in issues]
    assert "P_LOW" in categories


def test_motor_heat_estimation():
    # High D-term noise -> high heat
    np.random.seed(42)
    d_terms = np.random.randn(4, 2000) * 20  # high amplitude
    motors = np.random.randn(4, 2000) * 100 + 1500
    heat = estimate_motor_heat(d_terms, motors, sample_rate=2000)
    assert len(heat) == 4
    assert all(0 <= h <= 1 for h in heat.values())


def test_dterm_rms():
    signal = np.sin(np.linspace(0, 10, 1000)) * 15
    rms = compute_dterm_rms(signal)
    assert rms > 0
```

**Step 2: Run tests to verify fail**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_issues.py -v`
Expected: FAIL

**Step 3: Implement issues.py**

```python
# blacktune/analyzers/issues.py
"""Issue detection and motor heat estimation."""
import numpy as np
from scipy.signal import butter, sosfilt

from blacktune.models import Issue, StepResponseMetrics


def detect_pid_issues(
    axis: str,
    step_metrics: StepResponseMetrics,
    noise_peaks: list[tuple[float, float]],
    d_rms: float,
) -> list[Issue]:
    """Detect PID tuning issues from analysis data."""
    issues = []

    # P gain analysis
    if step_metrics.overshoot_pct > 15:
        severity = "red" if step_metrics.overshoot_pct > 25 else "yellow"
        issues.append(Issue(
            axis=axis, category="P_HIGH", severity=severity,
            message=f"{axis.title()} P gain too high",
            detail=f"Overshoot is {step_metrics.overshoot_pct:.1f}% "
                   f"(target: <10%). Reduce P to decrease overshoot.",
        ))
    elif step_metrics.rise_time_ms > 60:
        issues.append(Issue(
            axis=axis, category="P_LOW", severity="yellow",
            message=f"{axis.title()} P gain too low",
            detail=f"Rise time is {step_metrics.rise_time_ms:.1f}ms "
                   f"(target: 20-50ms). Increase P for snappier response.",
        ))

    # D gain analysis
    if d_rms > 20:
        severity = "red" if d_rms > 30 else "yellow"
        issues.append(Issue(
            axis=axis, category="D_HIGH", severity=severity,
            message=f"{axis.title()} D-term noise is high",
            detail=f"D-term RMS is {d_rms:.1f} (target: <15). "
                   f"Reduce D or tighten D-term filters to prevent motor heat.",
        ))
    elif step_metrics.overshoot_pct > 10 and step_metrics.settling_time_ms > 100:
        issues.append(Issue(
            axis=axis, category="D_LOW", severity="yellow",
            message=f"{axis.title()} D gain may be too low",
            detail=f"Overshoot ({step_metrics.overshoot_pct:.1f}%) with slow "
                   f"settling ({step_metrics.settling_time_ms:.0f}ms) suggests "
                   f"more D damping needed.",
        ))

    # I gain analysis
    if step_metrics.steady_state_error > 0.1:
        issues.append(Issue(
            axis=axis, category="I_LOW", severity="yellow",
            message=f"{axis.title()} I gain may be too low",
            detail=f"Steady-state error is {step_metrics.steady_state_error:.2f} "
                   f"(target: <0.05). Increase I for better tracking.",
        ))

    # Noise issues
    high_noise_peaks = [p for p in noise_peaks if p[1] > -10]
    if len(high_noise_peaks) > 3:
        issues.append(Issue(
            axis=axis, category="NOISE_HIGH", severity="red",
            message=f"{axis.title()} has excessive noise",
            detail=f"Found {len(high_noise_peaks)} strong noise peaks above -10dB. "
                   f"Tighten gyro filters or check motor/prop balance.",
        ))

    # Good tune detection
    if not issues:
        issues.append(Issue(
            axis=axis, category="GOOD", severity="green",
            message=f"{axis.title()} looks well tuned",
            detail=f"Rise: {step_metrics.rise_time_ms:.0f}ms, "
                   f"Overshoot: {step_metrics.overshoot_pct:.1f}%, "
                   f"D-RMS: {d_rms:.1f}",
        ))

    return issues


def compute_dterm_rms(d_term: np.ndarray) -> float:
    """Compute RMS of D-term signal."""
    return float(np.sqrt(np.mean(d_term ** 2)))


def estimate_motor_heat(
    d_terms: np.ndarray,
    motors: np.ndarray,
    sample_rate: int,
) -> dict[int, float]:
    """Estimate relative motor heat index (0-1) from D-term energy.

    d_terms: shape (4, N) -- D-term contribution per motor
             OR (3, N) for axis D-terms (will be mixed to motors)
    motors: shape (4, N) -- motor output values

    Returns dict of motor_index -> heat_index (0.0=cool, 1.0=danger).
    """
    n_motors = motors.shape[0]

    # High-pass filter motor commands above 100 Hz to isolate noise
    if sample_rate > 200:
        sos = butter(2, 100, btype="high", fs=sample_rate, output="sos")
        motor_noise = np.array([sosfilt(sos, motors[i]) for i in range(n_motors)])
    else:
        motor_noise = motors

    # RMS of high-frequency motor noise per motor
    noise_rms = np.array([np.sqrt(np.mean(motor_noise[i] ** 2))
                          for i in range(n_motors)])

    # Normalize to 0-1 range using empirical thresholds
    # Based on community data: RMS < 5 = cool, > 25 = danger
    heat_index = {}
    for i in range(n_motors):
        heat = np.clip((noise_rms[i] - 5) / 20, 0.0, 1.0)
        heat_index[i] = float(heat)

    return heat_index
```

**Step 4: Run tests**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_issues.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add blacktune/analyzers/issues.py tests/test_issues.py
git commit -m "feat: issue detection and motor heat estimation"
```

---

## Task 7: Quad Profile & PID Optimizer

**Files:**
- Create: `blacktune/optimizer.py`
- Create: `tests/test_optimizer.py`

**Step 1: Write failing tests**

```python
# tests/test_optimizer.py
import numpy as np
from blacktune.models import (
    PIDValues, FilterSettings, QuadProfile, StepResponseMetrics,
    Issue, AnalysisResult, TuneRecommendation,
)
from blacktune.optimizer import (
    get_baseline_pids, get_baseline_filters,
    optimize_pids, optimize_filters, generate_cli_commands,
)


def test_baseline_pids_5inch_6s():
    profile = QuadProfile(cell_count=6, prop_size=5.0)
    pids = get_baseline_pids(profile)
    # 6S should have lower PIDs than 4S
    profile_4s = QuadProfile(cell_count=4, prop_size=5.0)
    pids_4s = get_baseline_pids(profile_4s)
    assert pids.roll_p < pids_4s.roll_p


def test_baseline_filters():
    profile = QuadProfile(cell_count=4, prop_size=5.0)
    filters = get_baseline_filters(profile)
    assert filters.gyro_lpf1_hz > 0
    assert filters.dterm_lpf1_hz > 0


def test_optimize_reduces_p_on_overshoot():
    current = PIDValues(60, 80, 35, 62, 84, 38, 45, 90, 0)
    profile = QuadProfile(cell_count=6, prop_size=5.0)
    metrics = {
        "roll": StepResponseMetrics(20, 25, 150, 25, 0.02),
        "pitch": StepResponseMetrics(20, 25, 150, 25, 0.02),
        "yaw": StepResponseMetrics(30, 5, 100, 35, 0.03),
    }
    issues = [Issue("roll", "P_HIGH", "red", "P high", "overshoot")]
    analysis = AnalysisResult(
        step_response=metrics, noise_peaks={}, issues=issues,
        motor_heat_index={0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3},
        d_term_rms={"roll": 10, "pitch": 10, "yaw": 5},
    )
    rec = optimize_pids(current, profile, analysis)
    assert rec.roll_p < current.roll_p


def test_generate_cli():
    pids = PIDValues(45, 80, 35, 47, 84, 38, 45, 90, 0)
    filters = FilterSettings()
    cli = generate_cli_commands(pids, filters)
    assert "set p_roll" in cli.lower() or "set pid_roll" in cli.lower() or "set roll_p" in cli.lower()
```

**Step 2: Run tests to verify fail**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_optimizer.py -v`
Expected: FAIL

**Step 3: Implement optimizer.py**

```python
# blacktune/optimizer.py
"""PID and filter optimizer for BlackTune."""
from blacktune.models import (
    AnalysisResult, FilterSettings, Issue, PIDValues,
    QuadProfile, StepResponseMetrics, TuneRecommendation,
)

# Baseline PIDs for 5" 4S freestyle (Betaflight defaults as reference)
_BASE_PIDS_5IN_4S = PIDValues(
    roll_p=50, roll_i=80, roll_d=35,
    pitch_p=52, pitch_i=84, pitch_d=38,
    yaw_p=45, yaw_i=90, yaw_d=0,
)

# Voltage scaling factors (relative to 4S)
_VOLTAGE_SCALE = {3: 1.15, 4: 1.0, 5: 0.85, 6: 0.70}

# Prop size scaling factors (relative to 5")
_PROP_SCALE = {
    2.0: 0.75, 2.5: 0.80, 3.0: 0.85,
    4.0: 0.92, 5.0: 1.0, 5.1: 1.02,
    6.0: 1.10, 7.0: 1.20,
}

# Style aggressiveness multipliers
_STYLE_SCALE = {
    "freestyle": 1.0,
    "race": 1.15,
    "cinematic": 0.80,
    "long_range": 0.75,
}


def _nearest_key(d: dict, val: float) -> float:
    return min(d.keys(), key=lambda k: abs(k - val))


def get_baseline_pids(profile: QuadProfile) -> PIDValues:
    """Get baseline PID values scaled for the quad profile."""
    v_scale = _VOLTAGE_SCALE.get(profile.cell_count, 1.0)
    p_scale = _PROP_SCALE.get(
        _nearest_key(_PROP_SCALE, profile.prop_size), 1.0
    )
    s_scale = _STYLE_SCALE.get(profile.flying_style, 1.0)
    scale = v_scale * p_scale * s_scale

    base = _BASE_PIDS_5IN_4S
    # D needs less scaling for larger props (inertia helps damping)
    d_scale = v_scale * (p_scale ** 0.5) * s_scale

    return PIDValues(
        roll_p=round(base.roll_p * scale),
        roll_i=round(base.roll_i * scale),
        roll_d=round(base.roll_d * d_scale),
        pitch_p=round(base.pitch_p * scale),
        pitch_i=round(base.pitch_i * scale),
        pitch_d=round(base.pitch_d * d_scale),
        yaw_p=round(base.yaw_p * scale),
        yaw_i=round(base.yaw_i * scale),
        yaw_d=0,
    )


def get_baseline_filters(profile: QuadProfile) -> FilterSettings:
    """Get baseline filter settings for the quad profile."""
    # Larger props = lower RPM = lower noise frequencies = can filter tighter
    if profile.prop_size <= 3.0:
        gyro1, gyro2 = 300, 600
        dterm1, dterm2 = 170, 200
    elif profile.prop_size <= 5.0:
        gyro1, gyro2 = 250, 500
        dterm1, dterm2 = 150, 170
    else:
        gyro1, gyro2 = 200, 400
        dterm1, dterm2 = 120, 150

    return FilterSettings(
        gyro_lpf1_type="PT1", gyro_lpf1_hz=gyro1,
        gyro_lpf2_type="PT1", gyro_lpf2_hz=gyro2,
        dterm_lpf1_type="PT2", dterm_lpf1_hz=dterm1,
        dterm_lpf2_type="PT1", dterm_lpf2_hz=dterm2,
        dyn_notch_count=4, dyn_notch_q=3.5,
        dyn_notch_min_hz=100, dyn_notch_max_hz=600,
        rpm_harmonics=3, rpm_min_hz=100, rpm_q=500,
    )


def optimize_pids(
    current: PIDValues,
    profile: QuadProfile,
    analysis: AnalysisResult,
) -> PIDValues:
    """Optimize PIDs based on current values and analysis results."""
    baseline = get_baseline_pids(profile)
    explanations = {}

    def _adjust(axis, current_p, current_i, current_d, base_p, base_i, base_d):
        p, i, d = current_p, current_i, current_d
        metrics = analysis.step_response.get(axis)
        d_rms = analysis.d_term_rms.get(axis, 0)

        if metrics is None:
            return base_p, base_i, base_d

        # P adjustment based on overshoot
        if metrics.overshoot_pct > 20:
            reduction = min(0.25, (metrics.overshoot_pct - 10) / 100)
            p = round(p * (1 - reduction))
            explanations[f"{axis}_p"] = (
                f"Reduced P: {metrics.overshoot_pct:.0f}% overshoot -> "
                f"target <10%"
            )
        elif metrics.overshoot_pct < 3 and metrics.rise_time_ms > 50:
            increase = min(0.20, (metrics.rise_time_ms - 30) / 200)
            p = round(p * (1 + increase))
            explanations[f"{axis}_p"] = (
                f"Increased P: sluggish response "
                f"({metrics.rise_time_ms:.0f}ms rise time)"
            )

        # D adjustment based on D-term noise and overshoot
        if d_rms > 25:
            reduction = min(0.30, (d_rms - 15) / 50)
            d = round(d * (1 - reduction))
            explanations[f"{axis}_d"] = (
                f"Reduced D: high D-term noise (RMS={d_rms:.0f}), "
                f"risk of motor heat"
            )
        elif metrics.overshoot_pct > 10 and metrics.settling_time_ms > 80:
            increase = min(0.15, metrics.overshoot_pct / 200)
            d = round(d * (1 + increase))
            explanations[f"{axis}_d"] = (
                f"Increased D: needs more damping "
                f"({metrics.overshoot_pct:.0f}% overshoot, "
                f"{metrics.settling_time_ms:.0f}ms settling)"
            )

        # I adjustment based on steady-state error
        if metrics.steady_state_error > 0.1:
            i = round(i * 1.10)
            explanations[f"{axis}_i"] = (
                f"Increased I: steady-state error "
                f"{metrics.steady_state_error:.2f}"
            )

        # Safety clamps
        p = max(20, min(120, p))
        i = max(30, min(200, i))
        d = max(0, min(80, d))
        return p, i, d

    rp, ri, rd = _adjust("roll", current.roll_p, current.roll_i,
                         current.roll_d, baseline.roll_p, baseline.roll_i,
                         baseline.roll_d)
    pp, pi_, pd = _adjust("pitch", current.pitch_p, current.pitch_i,
                          current.pitch_d, baseline.pitch_p, baseline.pitch_i,
                          baseline.pitch_d)
    yp, yi, yd = _adjust("yaw", current.yaw_p, current.yaw_i,
                         current.yaw_d, baseline.yaw_p, baseline.yaw_i,
                         baseline.yaw_d)

    return PIDValues(rp, ri, rd, pp, pi_, pd, yp, yi, yd)


def optimize_filters(
    current: FilterSettings,
    profile: QuadProfile,
    analysis: AnalysisResult,
) -> FilterSettings:
    """Optimize filter settings based on noise analysis."""
    baseline = get_baseline_filters(profile)
    suggested = FilterSettings(
        gyro_lpf1_type=current.gyro_lpf1_type,
        gyro_lpf1_hz=current.gyro_lpf1_hz,
        gyro_lpf2_type=current.gyro_lpf2_type,
        gyro_lpf2_hz=current.gyro_lpf2_hz,
        dterm_lpf1_type=current.dterm_lpf1_type,
        dterm_lpf1_hz=current.dterm_lpf1_hz,
        dterm_lpf2_type=current.dterm_lpf2_type,
        dterm_lpf2_hz=current.dterm_lpf2_hz,
        dyn_notch_count=current.dyn_notch_count,
        dyn_notch_q=current.dyn_notch_q,
        dyn_notch_min_hz=current.dyn_notch_min_hz,
        dyn_notch_max_hz=current.dyn_notch_max_hz,
        rpm_harmonics=current.rpm_harmonics,
        rpm_min_hz=current.rpm_min_hz,
        rpm_q=current.rpm_q,
    )

    # Check if motors are running hot -> tighten D-term filters
    max_heat = max(analysis.motor_heat_index.values()) if analysis.motor_heat_index else 0
    if max_heat > 0.7:
        suggested.dterm_lpf1_hz = max(80, int(current.dterm_lpf1_hz * 0.85))
        suggested.dterm_lpf2_hz = max(80, int(current.dterm_lpf2_hz * 0.85))
        suggested.dterm_lpf1_type = "PT2"  # Stronger filtering

    # Check D-term RMS across axes
    max_d_rms = max(analysis.d_term_rms.values()) if analysis.d_term_rms else 0
    if max_d_rms > 20:
        suggested.dterm_lpf1_hz = max(80, int(current.dterm_lpf1_hz * 0.90))

    # Check for high noise -> tighten gyro filters
    total_issues = [i for i in analysis.issues if i.category == "NOISE_HIGH"]
    if total_issues:
        suggested.gyro_lpf1_hz = max(150, int(current.gyro_lpf1_hz * 0.85))
        suggested.gyro_lpf2_hz = max(300, int(current.gyro_lpf2_hz * 0.85))

    # Suggest RPM filter if not enabled and noise is significant
    if suggested.rpm_harmonics == 0 and max_heat > 0.5:
        suggested.rpm_harmonics = 3
        suggested.rpm_min_hz = 100

    return suggested


def generate_cli_commands(pids: PIDValues, filters: FilterSettings) -> str:
    """Generate Betaflight CLI commands to apply the suggested tune."""
    lines = [
        "# BlackTune suggested settings",
        "# Paste into Betaflight CLI tab",
        "",
        "# PID values",
        f"set p_roll = {int(pids.roll_p)}",
        f"set i_roll = {int(pids.roll_i)}",
        f"set d_roll = {int(pids.roll_d)}",
        f"set p_pitch = {int(pids.pitch_p)}",
        f"set i_pitch = {int(pids.pitch_i)}",
        f"set d_pitch = {int(pids.pitch_d)}",
        f"set p_yaw = {int(pids.yaw_p)}",
        f"set i_yaw = {int(pids.yaw_i)}",
        f"set d_yaw = {int(pids.yaw_d)}",
        "",
        "# Filter settings",
        f"set gyro_lpf1_type = {filters.gyro_lpf1_type}",
        f"set gyro_lpf1_static_hz = {int(filters.gyro_lpf1_hz)}",
        f"set gyro_lpf2_type = {filters.gyro_lpf2_type}",
        f"set gyro_lpf2_static_hz = {int(filters.gyro_lpf2_hz)}",
        f"set dterm_lpf1_type = {filters.dterm_lpf1_type}",
        f"set dterm_lpf1_static_hz = {int(filters.dterm_lpf1_hz)}",
        f"set dterm_lpf2_type = {filters.dterm_lpf2_type}",
        f"set dterm_lpf2_static_hz = {int(filters.dterm_lpf2_hz)}",
        f"set dyn_notch_count = {filters.dyn_notch_count}",
        f"set dyn_notch_q = {int(filters.dyn_notch_q * 10) / 10}",
        f"set dyn_notch_min_hz = {int(filters.dyn_notch_min_hz)}",
        f"set dyn_notch_max_hz = {int(filters.dyn_notch_max_hz)}",
        "",
        "# RPM filter",
        f"set dshot_bidir = {'ON' if filters.rpm_harmonics > 0 else 'OFF'}",
        f"set rpm_filter_harmonics = {filters.rpm_harmonics}",
        f"set rpm_filter_min_hz = {int(filters.rpm_min_hz)}",
        f"set rpm_filter_q = {int(filters.rpm_q)}",
        "",
        "save",
    ]
    return "\n".join(lines)
```

**Step 4: Run tests**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_optimizer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add blacktune/optimizer.py tests/test_optimizer.py
git commit -m "feat: PID/filter optimizer with baseline profiles and CLI export"
```

---

## Task 8: Full Analysis Pipeline

**Files:**
- Create: `blacktune/analyzer.py`
- Create: `tests/test_analyzer.py`

**Step 1: Write failing test**

```python
# tests/test_analyzer.py
import numpy as np
from blacktune.models import (
    AxisData, FlightLog, PIDValues, QuadProfile,
    AnalysisResult, TuneRecommendation,
)
from blacktune.analyzer import run_analysis, generate_recommendation


def _make_flight_log(n=4000, sr=2000):
    t = np.linspace(0, n / sr, n)
    noise = np.random.randn(n) * 2

    def _axis(name):
        sp = np.zeros(n)
        sp[1000:2000] = 200  # step input
        gyro = sp + noise
        return AxisData(name=name, gyro=gyro, setpoint=sp,
                        p_term=gyro * 0.3, i_term=gyro * 0.05,
                        d_term=noise * 5, time=t)

    return FlightLog(
        roll=_axis("roll"), pitch=_axis("pitch"), yaw=_axis("yaw"),
        throttle=np.linspace(1000, 1800, n),
        motors=np.random.randn(4, n) * 50 + 1400,
        sample_rate=sr, duration_s=n / sr,
        firmware="Betaflight 4.4",
        current_pids=PIDValues(50, 80, 35, 52, 84, 38, 45, 90, 0),
        current_filters=None,
    )


def test_run_analysis():
    log = _make_flight_log()
    result = run_analysis(log)
    assert "roll" in result.step_response
    assert "pitch" in result.step_response
    assert len(result.issues) > 0
    assert len(result.motor_heat_index) == 4


def test_generate_recommendation():
    log = _make_flight_log()
    profile = QuadProfile(cell_count=6, prop_size=5.0)
    result = run_analysis(log)
    rec = generate_recommendation(log, profile, result)
    assert rec.suggested_pids is not None
    assert "save" in rec.cli_commands.lower()
    assert rec.confidence > 0
```

**Step 2: Implement analyzer.py**

```python
# blacktune/analyzer.py
"""Top-level analysis pipeline for BlackTune."""
from blacktune.analyzers.noise import compute_fft_spectrum, find_noise_peaks
from blacktune.analyzers.step_response import (
    compute_step_response, measure_step_metrics,
)
from blacktune.analyzers.issues import (
    compute_dterm_rms, detect_pid_issues, estimate_motor_heat,
)
from blacktune.models import (
    AnalysisResult, FlightLog, QuadProfile, TuneRecommendation,
)
from blacktune.optimizer import (
    generate_cli_commands, optimize_filters, optimize_pids,
    get_baseline_filters,
)


def run_analysis(log: FlightLog) -> AnalysisResult:
    """Run the full analysis pipeline on a flight log."""
    step_metrics = {}
    noise_peaks = {}
    all_issues = []
    d_rms = {}

    for axis_data in [log.roll, log.pitch, log.yaw]:
        name = axis_data.name

        # Step response
        response, resp_time = compute_step_response(
            axis_data.setpoint, axis_data.gyro, log.sample_rate,
        )
        metrics = measure_step_metrics(response, resp_time)
        step_metrics[name] = metrics

        # Noise analysis
        freqs, psd_db = compute_fft_spectrum(axis_data.gyro, log.sample_rate)
        peaks = find_noise_peaks(freqs, psd_db)
        noise_peaks[name] = peaks

        # D-term RMS
        d_rms[name] = compute_dterm_rms(axis_data.d_term)

        # Issues
        issues = detect_pid_issues(name, metrics, peaks, d_rms[name])
        all_issues.extend(issues)

    # Motor heat estimation
    import numpy as np
    d_terms_for_motors = np.stack([
        log.roll.d_term, log.pitch.d_term,
        log.yaw.d_term, log.yaw.d_term,  # yaw duplicated for 4th motor
    ])
    motor_heat = estimate_motor_heat(d_terms_for_motors, log.motors,
                                     log.sample_rate)

    return AnalysisResult(
        step_response=step_metrics,
        noise_peaks=noise_peaks,
        issues=all_issues,
        motor_heat_index=motor_heat,
        d_term_rms=d_rms,
    )


def generate_recommendation(
    log: FlightLog,
    profile: QuadProfile,
    analysis: AnalysisResult,
) -> TuneRecommendation:
    """Generate PID + filter recommendations."""
    # PID optimization
    suggested_pids = optimize_pids(log.current_pids, profile, analysis)

    # Filter optimization
    current_filters = log.current_filters or get_baseline_filters(profile)
    suggested_filters = optimize_filters(current_filters, profile, analysis)

    # CLI commands
    cli = generate_cli_commands(suggested_pids, suggested_filters)

    # Confidence: higher if more data, lower if extreme issues
    red_count = sum(1 for i in analysis.issues if i.severity == "red")
    green_count = sum(1 for i in analysis.issues if i.severity == "green")
    total = len(analysis.issues) or 1
    confidence = max(0.3, min(0.95, 0.7 + green_count / total * 0.3
                                     - red_count / total * 0.2))

    # Collect explanations
    explanations = {}
    for issue in analysis.issues:
        if issue.severity != "green":
            explanations[f"{issue.axis}_{issue.category}"] = issue.detail

    return TuneRecommendation(
        suggested_pids=suggested_pids,
        suggested_filters=suggested_filters,
        confidence=confidence,
        explanations=explanations,
        cli_commands=cli,
    )
```

**Step 3: Run tests**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m pytest tests/test_analyzer.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add blacktune/analyzer.py tests/test_analyzer.py
git commit -m "feat: full analysis pipeline connecting all analyzers"
```

---

## Task 9: UI - Main Window Shell

**Files:**
- Modify: `blacktune/main.py`
- Create: `blacktune/ui/__init__.py`
- Create: `blacktune/ui/main_window.py`
- Create: `blacktune/ui/theme.py`

**Step 1: Create dark theme**

```python
# blacktune/ui/__init__.py
"""BlackTune UI components."""

# blacktune/ui/theme.py
"""Dark theme for BlackTune."""

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}
QTabWidget::pane {
    border: 1px solid #2d2d4e;
    background-color: #16213e;
}
QTabBar::tab {
    background-color: #1a1a2e;
    color: #8888aa;
    padding: 8px 20px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: #00d4ff;
    border-bottom: 2px solid #00d4ff;
}
QTabBar::tab:hover {
    color: #ffffff;
}
QPushButton {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #1a5276;
    border-radius: 4px;
    padding: 6px 16px;
}
QPushButton:hover {
    background-color: #1a5276;
}
QPushButton:pressed {
    background-color: #0a2647;
}
QPushButton#primary {
    background-color: #00d4ff;
    color: #1a1a2e;
    font-weight: bold;
}
QComboBox {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    padding: 4px 8px;
}
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #e0e0e0;
    selection-background-color: #0f3460;
}
QLabel {
    color: #e0e0e0;
}
QLabel#header {
    font-size: 14px;
    font-weight: bold;
    color: #00d4ff;
}
QGroupBox {
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
    color: #8888aa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QScrollBar:vertical {
    background: #1a1a2e;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #2d2d4e;
    border-radius: 5px;
}
QTableWidget {
    background-color: #16213e;
    gridline-color: #2d2d4e;
    color: #e0e0e0;
    border: none;
}
QTableWidget::item:selected {
    background-color: #0f3460;
}
QHeaderView::section {
    background-color: #1a1a2e;
    color: #8888aa;
    border: 1px solid #2d2d4e;
    padding: 4px;
}
"""
```

**Step 2: Create main window**

```python
# blacktune/ui/main_window.py
"""Main application window."""
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QStatusBar, QFileDialog, QMenuBar,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence

from blacktune.ui.theme import DARK_STYLESHEET


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BlackTune - FPV PID Autotuner")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(DARK_STYLESHEET)

        self._flight_log = None
        self._analysis_result = None
        self._recommendation = None

        self._setup_menu()
        self._setup_tabs()
        self._setup_statusbar()

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open BBL/CSV...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _setup_tabs(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Placeholder tabs -- replaced in Tasks 10-13
        self.viewer_tab = QWidget()
        self.analysis_tab = QWidget()
        self.tune_tab = QWidget()
        self.history_tab = QWidget()

        for tab, name in [
            (self.viewer_tab, "Log Viewer"),
            (self.analysis_tab, "Analysis"),
            (self.tune_tab, "Tune"),
            (self.history_tab, "History"),
        ]:
            layout = QVBoxLayout(tab)
            label = QLabel(f"{name} - Load a BBL file to begin")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self.tabs.addTab(tab, name)

    def _setup_statusbar(self):
        self.statusBar().showMessage("Ready - Open a BBL or CSV file to begin")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Blackbox Log",
            "", "Blackbox Logs (*.bbl *.bfl *.csv);;All Files (*)",
        )
        if path:
            self.statusBar().showMessage(f"Loading {path}...")
            self._load_file(path)

    def _load_file(self, path: str):
        """Load and analyze a flight log file."""
        try:
            from blacktune.parser import parse_csv_log, load_bbl

            if path.lower().endswith(".csv"):
                self._flight_log = parse_csv_log(path)
            else:
                self._flight_log = load_bbl(path)

            self.statusBar().showMessage(
                f"Loaded: {path} | "
                f"{self._flight_log.sample_rate}Hz | "
                f"{self._flight_log.duration_s:.1f}s | "
                f"{self._flight_log.firmware}"
            )
            # Analysis triggered in Task 12 integration
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")

    def set_flight_log(self, log):
        self._flight_log = log

    def set_analysis(self, result):
        self._analysis_result = result

    def set_recommendation(self, rec):
        self._recommendation = rec
```

**Step 3: Update main.py**

```python
# blacktune/main.py
"""BlackTune - FPV PID Autotuner."""
import sys
from PyQt6.QtWidgets import QApplication
from blacktune.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BlackTune")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

**Step 4: Test manually**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m blacktune.main`
Expected: Dark-themed window with 4 tabs appears

**Step 5: Commit**

```bash
git add blacktune/ui/ blacktune/main.py
git commit -m "feat: main window with dark theme and 4-tab layout"
```

---

## Task 10: UI - Log Viewer Tab

**Files:**
- Create: `blacktune/ui/viewer_tab.py`
- Modify: `blacktune/ui/main_window.py`

**Step 1: Implement viewer tab with pyqtgraph plots**

```python
# blacktune/ui/viewer_tab.py
"""Log Viewer tab with time-series plots."""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
)
import pyqtgraph as pg
import numpy as np

from blacktune.models import FlightLog

# FPV-themed colors
COLORS = {
    "roll": "#ff6b6b",
    "pitch": "#4ecdc4",
    "yaw": "#ffe66d",
    "setpoint": "#a8e6cf",
    "throttle": "#ff8b94",
    "motor0": "#ff6b6b",
    "motor1": "#4ecdc4",
    "motor2": "#ffe66d",
    "motor3": "#a8e6cf",
}


class ViewerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Axis toggles
        toggle_layout = QHBoxLayout()
        self.cb_roll = QCheckBox("Roll")
        self.cb_roll.setChecked(True)
        self.cb_pitch = QCheckBox("Pitch")
        self.cb_pitch.setChecked(True)
        self.cb_yaw = QCheckBox("Yaw")
        self.cb_yaw.setChecked(True)
        for cb in [self.cb_roll, self.cb_pitch, self.cb_yaw]:
            cb.stateChanged.connect(self._update_plots)
            toggle_layout.addWidget(cb)
        toggle_layout.addStretch()
        layout.addLayout(toggle_layout)

        # Plot widget with linked X axes
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("#16213e")
        layout.addWidget(self.plot_widget)

        # Create 4 plot rows: Gyro+Setpoint, PID terms, Motors, Throttle
        self.gyro_plot = self.plot_widget.addPlot(row=0, col=0,
                                                   title="Gyro vs Setpoint (deg/s)")
        self.pid_plot = self.plot_widget.addPlot(row=1, col=0,
                                                  title="PID Terms")
        self.motor_plot = self.plot_widget.addPlot(row=2, col=0,
                                                    title="Motor Output")
        self.throttle_plot = self.plot_widget.addPlot(row=3, col=0,
                                                       title="Throttle")

        # Link X axes for synchronized scrolling
        self.pid_plot.setXLink(self.gyro_plot)
        self.motor_plot.setXLink(self.gyro_plot)
        self.throttle_plot.setXLink(self.gyro_plot)

        # Style plots
        for plot in [self.gyro_plot, self.pid_plot, self.motor_plot,
                     self.throttle_plot]:
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.getAxis("left").setPen("#8888aa")
            plot.getAxis("bottom").setPen("#8888aa")
            plot.addLegend(offset=(10, 10))

        # Placeholder label
        self.empty_label = QLabel("Open a BBL or CSV file to view flight data")
        self.empty_label.setStyleSheet("color: #8888aa; font-size: 16px;")
        layout.addWidget(self.empty_label)

    def load_data(self, log: FlightLog):
        self._log = log
        self.empty_label.hide()
        self.plot_widget.show()
        self._update_plots()

    def _update_plots(self):
        if self._log is None:
            return

        for plot in [self.gyro_plot, self.pid_plot, self.motor_plot,
                     self.throttle_plot]:
            plot.clear()

        axes = []
        if self.cb_roll.isChecked():
            axes.append(self._log.roll)
        if self.cb_pitch.isChecked():
            axes.append(self._log.pitch)
        if self.cb_yaw.isChecked():
            axes.append(self._log.yaw)

        for ax in axes:
            color = COLORS[ax.name]
            sp_color = COLORS["setpoint"]
            # Gyro + Setpoint
            self.gyro_plot.plot(ax.time, ax.gyro, pen=pg.mkPen(color, width=1),
                                name=f"{ax.name} gyro")
            self.gyro_plot.plot(ax.time, ax.setpoint,
                                pen=pg.mkPen(sp_color, width=1, style=2),
                                name=f"{ax.name} SP")
            # PID terms
            self.pid_plot.plot(ax.time, ax.p_term,
                               pen=pg.mkPen(color, width=1),
                               name=f"{ax.name} P")
            self.pid_plot.plot(ax.time, ax.d_term,
                               pen=pg.mkPen(color, width=1, style=3),
                               name=f"{ax.name} D")

        # Motors
        for i in range(min(4, self._log.motors.shape[0])):
            self.motor_plot.plot(
                self._log.roll.time, self._log.motors[i],
                pen=pg.mkPen(COLORS[f"motor{i}"], width=1),
                name=f"Motor {i+1}",
            )

        # Throttle
        self.throttle_plot.plot(
            self._log.roll.time, self._log.throttle,
            pen=pg.mkPen(COLORS["throttle"], width=1.5),
            name="Throttle",
        )
```

**Step 2: Wire into main_window.py** -- Replace the placeholder viewer_tab with the real one in `_setup_tabs()`:

Replace the viewer_tab placeholder block with:
```python
from blacktune.ui.viewer_tab import ViewerTab
# In _setup_tabs:
self.viewer_tab = ViewerTab()
self.tabs.addTab(self.viewer_tab, "Log Viewer")
```

**Step 3: Test manually**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/python -m blacktune.main`
Expected: Window shows with Log Viewer tab, axis checkboxes, empty plot area

**Step 4: Commit**

```bash
git add blacktune/ui/viewer_tab.py blacktune/ui/main_window.py
git commit -m "feat: log viewer tab with gyro/PID/motor/throttle plots"
```

---

## Task 11: UI - Analysis Dashboard Tab

**Files:**
- Create: `blacktune/ui/analysis_tab.py`
- Modify: `blacktune/ui/main_window.py`

**Step 1: Implement analysis tab**

```python
# blacktune/ui/analysis_tab.py
"""Analysis Dashboard tab."""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QGridLayout,
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

from blacktune.models import AnalysisResult, FlightLog


class IssueCard(QFrame):
    """Color-coded issue card."""

    SEVERITY_COLORS = {
        "red": "#ff4444",
        "yellow": "#ffaa00",
        "green": "#44ff44",
    }

    def __init__(self, axis, category, severity, message, detail, parent=None):
        super().__init__(parent)
        color = self.SEVERITY_COLORS.get(severity, "#888888")
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1e2a4a;
                border-left: 4px solid {color};
                border-radius: 4px;
                padding: 8px;
                margin: 2px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        header = QLabel(f"[{axis.upper()}] {message}")
        header.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        body = QLabel(detail)
        body.setWordWrap(True)
        body.setStyleSheet("color: #cccccc; font-size: 11px;")
        layout.addWidget(body)


class MotorHeatBar(QWidget):
    """Visual motor heat indicator."""

    def __init__(self, motor_idx, heat_value, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        label = QLabel(f"Motor {motor_idx + 1}")
        label.setFixedWidth(60)
        layout.addWidget(label)

        bar = QFrame()
        if heat_value < 0.3:
            color = "#44ff44"
            text = "Cool"
        elif heat_value < 0.6:
            color = "#ffaa00"
            text = "Warm"
        elif heat_value < 0.8:
            color = "#ff6600"
            text = "Hot"
        else:
            color = "#ff4444"
            text = "DANGER"

        width = max(20, int(heat_value * 200))
        bar.setFixedSize(width, 16)
        bar.setStyleSheet(f"background-color: {color}; border-radius: 3px;")
        layout.addWidget(bar)

        val_label = QLabel(f"{text} ({heat_value:.0%})")
        val_label.setStyleSheet(f"color: {color};")
        layout.addWidget(val_label)
        layout.addStretch()


class AnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Top row: FFT plots
        pg.setConfigOptions(antialias=True)
        self.fft_widget = pg.GraphicsLayoutWidget()
        self.fft_widget.setBackground("#16213e")
        self.fft_widget.setMinimumHeight(250)
        layout.addWidget(self.fft_widget)

        self.fft_roll = self.fft_widget.addPlot(row=0, col=0, title="Roll Spectrum")
        self.fft_pitch = self.fft_widget.addPlot(row=0, col=1, title="Pitch Spectrum")
        self.fft_yaw = self.fft_widget.addPlot(row=0, col=2, title="Yaw Spectrum")

        for p in [self.fft_roll, self.fft_pitch, self.fft_yaw]:
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setLabel("bottom", "Frequency", units="Hz")
            p.setLabel("left", "Power", units="dB")

        # Middle row: Step response
        self.step_widget = pg.GraphicsLayoutWidget()
        self.step_widget.setBackground("#16213e")
        self.step_widget.setMinimumHeight(200)
        layout.addWidget(self.step_widget)

        self.step_plot = self.step_widget.addPlot(title="Step Response")
        self.step_plot.showGrid(x=True, y=True, alpha=0.2)
        self.step_plot.setLabel("bottom", "Time", units="ms")
        self.step_plot.setLabel("left", "Response")
        self.step_plot.addLegend()

        # Bottom: Issues + Motor Heat side by side
        bottom = QHBoxLayout()

        # Issues scroll area
        issues_frame = QFrame()
        issues_frame.setStyleSheet("background-color: #16213e; border-radius: 4px;")
        self.issues_layout = QVBoxLayout(issues_frame)
        self.issues_layout.addWidget(QLabel("Issues"))

        scroll = QScrollArea()
        scroll.setWidget(issues_frame)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        bottom.addWidget(scroll)

        # Motor heat
        heat_frame = QFrame()
        heat_frame.setStyleSheet("background-color: #16213e; border-radius: 4px;")
        self.heat_layout = QVBoxLayout(heat_frame)
        self.heat_layout.addWidget(QLabel("Motor Heat Estimation"))
        bottom.addWidget(heat_frame)

        layout.addLayout(bottom)

        self.empty_label = QLabel("Run analysis to see results")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #8888aa; font-size: 16px;")
        layout.addWidget(self.empty_label)

    def load_results(self, log: FlightLog, result: AnalysisResult):
        self.empty_label.hide()
        from blacktune.analyzers.noise import compute_fft_spectrum

        colors = {"roll": "#ff6b6b", "pitch": "#4ecdc4", "yaw": "#ffe66d"}
        plots = {"roll": self.fft_roll, "pitch": self.fft_pitch, "yaw": self.fft_yaw}

        # FFT plots
        for axis_data in [log.roll, log.pitch, log.yaw]:
            name = axis_data.name
            freqs, psd_db = compute_fft_spectrum(axis_data.gyro, log.sample_rate)
            plots[name].clear()
            plots[name].plot(freqs, psd_db,
                             pen=pg.mkPen(colors[name], width=1.5))
            # Mark peaks
            peaks = result.noise_peaks.get(name, [])
            for freq, amp in peaks[:5]:
                plots[name].addItem(pg.InfiniteLine(
                    pos=freq, angle=90,
                    pen=pg.mkPen("#ff444488", width=1, style=2),
                ))

        # Step response
        self.step_plot.clear()
        # Target line at 1.0
        self.step_plot.addItem(pg.InfiniteLine(
            pos=1.0, angle=0, pen=pg.mkPen("#ffffff44", width=1, style=3),
        ))

        from blacktune.analyzers.step_response import compute_step_response
        for axis_data in [log.roll, log.pitch, log.yaw]:
            name = axis_data.name
            response, resp_time = compute_step_response(
                axis_data.setpoint, axis_data.gyro, log.sample_rate,
            )
            self.step_plot.plot(resp_time * 1000, response,
                                pen=pg.mkPen(colors[name], width=2),
                                name=name.title())

        # Issue cards
        # Clear existing
        while self.issues_layout.count() > 1:
            item = self.issues_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        for issue in result.issues:
            card = IssueCard(issue.axis, issue.category, issue.severity,
                             issue.message, issue.detail)
            self.issues_layout.addWidget(card)

        # Motor heat
        while self.heat_layout.count() > 1:
            item = self.heat_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        for idx, heat in sorted(result.motor_heat_index.items()):
            bar = MotorHeatBar(idx, heat)
            self.heat_layout.addWidget(bar)
```

**Step 2: Wire into main_window.py**

**Step 3: Commit**

```bash
git add blacktune/ui/analysis_tab.py blacktune/ui/main_window.py
git commit -m "feat: analysis dashboard with FFT, step response, issues, motor heat"
```

---

## Task 12: UI - Tune Tab

**Files:**
- Create: `blacktune/ui/tune_tab.py`
- Modify: `blacktune/ui/main_window.py`

**Step 1: Implement tune tab with quad profile + PID comparison + CLI export**

```python
# blacktune/ui/tune_tab.py
"""Quad Profile & Tune tab."""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QTextEdit, QApplication, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from blacktune.models import (
    PIDValues, FilterSettings, QuadProfile, TuneRecommendation,
)


class TuneTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._recommendation = None
        self._on_profile_changed = None
        self._setup_ui()

    def set_profile_callback(self, callback):
        self._on_profile_changed = callback

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # Left panel: Quad profile
        left = QVBoxLayout()
        profile_group = QGroupBox("Quad Profile")
        profile_layout = QGridLayout(profile_group)

        profile_layout.addWidget(QLabel("Cell Count:"), 0, 0)
        self.cell_combo = QComboBox()
        self.cell_combo.addItems(["3S", "4S", "5S", "6S"])
        self.cell_combo.setCurrentIndex(1)  # Default 4S
        profile_layout.addWidget(self.cell_combo, 0, 1)

        profile_layout.addWidget(QLabel("Prop Size:"), 1, 0)
        self.prop_combo = QComboBox()
        self.prop_combo.addItems(['2"', '2.5"', '3"', '4"', '5"', '6"', '7"'])
        self.prop_combo.setCurrentIndex(4)  # Default 5"
        profile_layout.addWidget(self.prop_combo, 1, 1)

        profile_layout.addWidget(QLabel("Frame Size:"), 2, 0)
        self.frame_combo = QComboBox()
        self.frame_combo.addItems(["Auto", "Micro", '3"', '5"', '7"', '10"+'])
        profile_layout.addWidget(self.frame_combo, 2, 1)

        profile_layout.addWidget(QLabel("Flying Style:"), 3, 0)
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Freestyle", "Race", "Cinematic", "Long Range"])
        profile_layout.addWidget(self.style_combo, 3, 1)

        self.analyze_btn = QPushButton("Analyze & Tune")
        self.analyze_btn.setObjectName("primary")
        self.analyze_btn.clicked.connect(self._on_analyze)
        profile_layout.addWidget(self.analyze_btn, 4, 0, 1, 2)

        left.addWidget(profile_group)

        # Confidence indicator
        self.confidence_label = QLabel("")
        self.confidence_label.setStyleSheet("font-size: 14px; padding: 8px;")
        left.addWidget(self.confidence_label)

        left.addStretch()
        layout.addLayout(left)

        # Center: PID + Filter comparison tables
        center = QVBoxLayout()

        pid_label = QLabel("PID Values")
        pid_label.setObjectName("header")
        center.addWidget(pid_label)

        self.pid_table = QTableWidget(3, 7)
        self.pid_table.setHorizontalHeaderLabels([
            "Axis", "Current P", "Suggested P",
            "Current I", "Suggested I",
            "Current D", "Suggested D",
        ])
        self.pid_table.setVerticalHeaderLabels(["Roll", "Pitch", "Yaw"])
        self.pid_table.horizontalHeader().setStretchLastSection(True)
        center.addWidget(self.pid_table)

        filter_label = QLabel("Filter Settings")
        filter_label.setObjectName("header")
        center.addWidget(filter_label)

        self.filter_table = QTableWidget(6, 3)
        self.filter_table.setHorizontalHeaderLabels([
            "Setting", "Current", "Suggested",
        ])
        center.addWidget(self.filter_table)

        layout.addLayout(center, stretch=2)

        # Right: CLI output
        right = QVBoxLayout()
        cli_label = QLabel("Betaflight CLI Commands")
        cli_label.setObjectName("header")
        right.addWidget(cli_label)

        self.cli_text = QTextEdit()
        self.cli_text.setReadOnly(True)
        self.cli_text.setStyleSheet(
            "background-color: #0a0a1a; color: #00ff00; "
            "font-family: 'Consolas', monospace; font-size: 12px;"
        )
        right.addWidget(self.cli_text)

        self.copy_btn = QPushButton("Copy CLI Commands")
        self.copy_btn.clicked.connect(self._copy_cli)
        right.addWidget(self.copy_btn)

        layout.addLayout(right)

    def get_profile(self) -> QuadProfile:
        cell_map = {"3S": 3, "4S": 4, "5S": 5, "6S": 6}
        cell = cell_map[self.cell_combo.currentText()]
        prop = float(self.prop_combo.currentText().replace('"', ''))
        style_map = {
            "Freestyle": "freestyle", "Race": "race",
            "Cinematic": "cinematic", "Long Range": "long_range",
        }
        style = style_map[self.style_combo.currentText()]
        return QuadProfile(cell_count=cell, prop_size=prop, flying_style=style)

    def _on_analyze(self):
        if self._on_profile_changed:
            self._on_profile_changed(self.get_profile())

    def load_recommendation(self, current_pids: PIDValues,
                            rec: TuneRecommendation):
        self._recommendation = rec
        sp = rec.suggested_pids

        # PID table
        rows = [
            ("Roll", current_pids.roll_p, sp.roll_p,
             current_pids.roll_i, sp.roll_i,
             current_pids.roll_d, sp.roll_d),
            ("Pitch", current_pids.pitch_p, sp.pitch_p,
             current_pids.pitch_i, sp.pitch_i,
             current_pids.pitch_d, sp.pitch_d),
            ("Yaw", current_pids.yaw_p, sp.yaw_p,
             current_pids.yaw_i, sp.yaw_i,
             current_pids.yaw_d, sp.yaw_d),
        ]

        for row, (axis, cp, sp_p, ci, si, cd, sd) in enumerate(rows):
            self.pid_table.setItem(row, 0, QTableWidgetItem(axis))
            for col, (curr, sugg) in enumerate([(cp, sp_p), (ci, si), (cd, sd)]):
                curr_item = QTableWidgetItem(str(int(curr)))
                sugg_item = QTableWidgetItem(str(int(sugg)))
                # Color code changes
                if sugg > curr:
                    sugg_item.setForeground(QColor("#44ff44"))
                elif sugg < curr:
                    sugg_item.setForeground(QColor("#ff6b6b"))
                self.pid_table.setItem(row, 1 + col * 2, curr_item)
                self.pid_table.setItem(row, 2 + col * 2, sugg_item)

        # CLI commands
        self.cli_text.setText(rec.cli_commands)

        # Confidence
        conf = rec.confidence
        if conf > 0.8:
            color = "#44ff44"
            text = "High"
        elif conf > 0.5:
            color = "#ffaa00"
            text = "Medium"
        else:
            color = "#ff6b6b"
            text = "Low"
        self.confidence_label.setText(f"Confidence: {text} ({conf:.0%})")
        self.confidence_label.setStyleSheet(
            f"color: {color}; font-size: 14px; font-weight: bold; padding: 8px;"
        )

    def _copy_cli(self):
        if self._recommendation:
            QApplication.clipboard().setText(self._recommendation.cli_commands)
```

**Step 2: Wire into main_window.py**

**Step 3: Commit**

```bash
git add blacktune/ui/tune_tab.py blacktune/ui/main_window.py
git commit -m "feat: tune tab with quad profile, PID comparison, CLI export"
```

---

## Task 13: UI - History Tab + Full Integration

**Files:**
- Create: `blacktune/ui/history_tab.py`
- Create: `blacktune/history.py`
- Modify: `blacktune/ui/main_window.py`

**Step 1: Implement simple JSON-based history**

```python
# blacktune/history.py
"""Tune history tracking."""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

_HISTORY_FILE = Path.home() / ".blacktune" / "history.json"


def save_session(filename: str, current_pids: dict, suggested_pids: dict,
                 profile: dict, metrics: dict):
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_history()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "profile": profile,
        "current_pids": current_pids,
        "suggested_pids": suggested_pids,
        "metrics": metrics,
    }
    history.append(entry)
    _HISTORY_FILE.write_text(json.dumps(history, indent=2))


def load_history() -> list[dict]:
    if not _HISTORY_FILE.exists():
        return []
    try:
        return json.loads(_HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []
```

**Step 2: Implement history tab** (table showing past sessions)

**Step 3: Wire all tabs into main_window.py with full load/analyze/recommend pipeline**

In `_load_file()`, after loading the log:
1. Create `ViewerTab` and call `load_data(log)`
2. Run `run_analysis(log)` in a thread
3. On complete, update `AnalysisTab` with results
4. When user clicks "Analyze & Tune" in TuneTab, call `generate_recommendation()`
5. Save to history

**Step 4: Commit**

```bash
git add blacktune/history.py blacktune/ui/history_tab.py blacktune/ui/main_window.py
git commit -m "feat: history tracking and full analysis pipeline integration"
```

---

## Task 14: Drag-and-Drop Support

**Files:**
- Modify: `blacktune/ui/main_window.py`

**Step 1: Add drag-and-drop to main window**

```python
# Add to MainWindow.__init__:
self.setAcceptDrops(True)

# Add methods:
def dragEnterEvent(self, event):
    if event.mimeData().hasUrls():
        for url in event.mimeData().urls():
            if url.toLocalFile().lower().endswith(('.bbl', '.bfl', '.csv')):
                event.acceptProposedAction()
                return

def dropEvent(self, event):
    for url in event.mimeData().urls():
        path = url.toLocalFile()
        if path.lower().endswith(('.bbl', '.bfl', '.csv')):
            self._load_file(path)
            break
```

**Step 2: Commit**

```bash
git add blacktune/ui/main_window.py
git commit -m "feat: drag-and-drop BBL/CSV file loading"
```

---

## Task 15: PyInstaller Packaging

**Files:**
- Create: `blacktune.spec` or use CLI
- Create: `build.bat`

**Step 1: Create build script**

```bat
@echo off
cd C:\Users\zacle\blacktune
venv\Scripts\pyinstaller --onefile --windowed ^
    --name BlackTune ^
    --add-data "blacktune/bin/blackbox_decode.exe;blacktune/bin" ^
    blacktune/main.py
echo Build complete: dist\BlackTune.exe
```

**Step 2: Test the build**

Run: `cd C:/Users/zacle/blacktune && venv/Scripts/pip install pyinstaller && build.bat`
Expected: `dist/BlackTune.exe` created

**Step 3: Commit**

```bash
git add build.bat
git commit -m "feat: PyInstaller build script for standalone exe"
```

---

## Summary

| Task | Component | Estimated Complexity |
|------|-----------|---------------------|
| 1 | Project scaffolding | Simple |
| 2 | Data models | Simple |
| 3 | BBL parser | Medium |
| 4 | Noise analyzer (FFT) | Medium |
| 5 | Step response (Wiener deconv) | Medium-Hard |
| 6 | Issue detection + motor heat | Medium |
| 7 | PID/filter optimizer | Medium-Hard |
| 8 | Analysis pipeline | Simple (wiring) |
| 9 | UI main window | Simple |
| 10 | Log viewer tab | Medium |
| 11 | Analysis dashboard tab | Medium |
| 12 | Tune tab | Medium |
| 13 | History + integration | Medium |
| 14 | Drag-and-drop | Simple |
| 15 | PyInstaller packaging | Simple |
