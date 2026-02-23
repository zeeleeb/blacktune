# BlackTune

Betaflight blackbox log analyzer with automatic PID and filter tuning recommendations.

Load a `.bbl`, `.bfl`, or `.csv` blackbox log, analyze gyro noise and step response, and get ready-to-paste CLI commands for your flight controller.

## Features

- **Log Viewer** -- interactive plots for gyro, setpoint, PID terms, motor outputs, and throttle
- **Noise Analysis** -- FFT spectrum (Welch PSD), throttle-binned spectrograms, automatic peak detection
- **Step Response** -- Wiener deconvolution with rise time, overshoot, settling time metrics
- **Issue Detection** -- identifies P too high/low, D too high/low, I windup, noise floor, oscillation
- **Motor Heat Estimation** -- estimates motor heating from high-frequency D-term energy
- **Auto Tune** -- generates optimized PID and filter values based on your quad profile (cell count, prop size, frame size, flying style)
- **CLI Export** -- copy-paste Betaflight CLI commands with one click
- **Session History** -- tracks past analyses with timestamps

## Screenshot

The app has four tabs: Log Viewer, Analysis, Tune, and History. The dark theme is designed for readability during tuning sessions.

## Installation

### From Source

```bash
git clone https://github.com/zeeleeb/blacktune.git
cd blacktune
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
python -m blacktune.main
```

### Standalone Executable

Download `BlackTune.exe` from [Releases](https://github.com/zeeleeb/blacktune/releases). No Python required -- just run it.

> Windows SmartScreen may warn about an unsigned executable. Click "More info" > "Run anyway".

### Build the Executable Yourself

```bash
pip install pyinstaller
build.bat
```

Output: `dist/BlackTune.exe`

## Usage

1. Open BlackTune
2. Click **Open Log** or drag-and-drop a `.bbl` / `.bfl` / `.csv` file
3. Browse the **Log Viewer** tab to inspect raw flight data
4. Check the **Analysis** tab for noise spectrum, step response, detected issues, and motor heat
5. Go to the **Tune** tab, set your quad profile (cell count, prop size, frame size, flying style)
6. Click **Analyze & Tune** to generate optimized PID and filter values
7. Copy the CLI commands and paste them into the Betaflight CLI

## Supported Log Formats

| Format | Source | Parser |
|--------|--------|--------|
| `.bbl` / `.bfl` | Betaflight blackbox | [orangebox](https://github.com/thenickdude/orangebox) |
| `.csv` | Betaflight blackbox_decode | pandas |

## How It Works

**Step Response Analysis** uses Wiener deconvolution (same algorithm as PIDtoolbox) to extract the closed-loop step response from setpoint/gyro data. This reveals overshoot, oscillation, and settling behavior per axis.

**Noise Analysis** computes Welch PSD spectra and detects peaks that indicate motor noise, propwash, or electrical interference.

**PID Optimizer** starts from Betaflight defaults scaled by voltage (cell count) and inertia (prop size), then adjusts per-axis based on step response metrics and detected issues.

**Filter Optimizer** adjusts gyro and D-term lowpass frequencies and dynamic notch settings based on detected noise floor and peak locations.

## Project Structure

```
blacktune/
  main.py              # entry point
  models.py            # dataclasses (PIDValues, FilterSettings, FlightLog, etc.)
  parser.py            # BBL/CSV log parsing
  analyzer.py          # full analysis pipeline
  optimizer.py         # PID/filter optimization + CLI generation
  history.py           # session history persistence
  analyzers/
    noise.py           # FFT spectrum + spectrogram + peak detection
    step_response.py   # Wiener deconvolution + metrics
    issues.py          # issue detection + motor heat estimation
  ui/
    main_window.py     # main window with 4 tabs
    viewer_tab.py      # log viewer with linked plots
    analysis_tab.py    # FFT, step response, issues, motor heat
    tune_tab.py        # quad profile, PID/filter comparison, CLI export
    history_tab.py     # session history table
    theme.py           # dark theme stylesheet
tests/                 # 111 tests
```

## Requirements

- Python 3.10+
- PyQt6, pyqtgraph, numpy, scipy, pandas, orangebox

## License

MIT
