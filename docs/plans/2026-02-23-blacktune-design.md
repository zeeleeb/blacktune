# BlackTune -- FPV PID Autotuner Design

**Date:** 2026-02-23
**Status:** Approved

## Overview

Desktop application that reads Betaflight blackbox logs (.bbl/.bfl), analyzes flight dynamics, detects tuning issues, and suggests optimized PID + filter settings based on quad profile (cell count, prop size, frame, flying style).

## Tech Stack

- Python 3.11+, PyQt6, pyqtgraph, NumPy, SciPy
- PyInstaller for .exe distribution (Windows)

## Architecture

```
┌─────────────────────────────────────────────┐
│              Desktop App (PyQt6)             │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ Log      │ │ Analysis │ │ PID/Filter  │  │
│  │ Viewer   │ │ Dashboard│ │ Suggestions │  │
│  │ (graphs) │ │ (issues) │ │ (export)    │  │
│  └──────────┘ └──────────┘ └─────────────┘  │
├─────────────────────────────────────────────┤
│           Analysis Engine (Python)           │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ BBL      │ │ Noise &  │ │ PID/Filter  │  │
│  │ Parser   │ │ Response │ │ Optimizer   │  │
│  │          │ │ Analyzer │ │             │  │
│  └──────────┘ └──────────┘ └─────────────┘  │
└─────────────────────────────────────────────┘
```

## BBL Parser

- Decode Betaflight blackbox log format (.bbl/.bfl)
- Extract: gyro raw, gyro filtered, PID setpoint, PID output (P/I/D terms), motor output, RC commands, debug fields
- Parse header for current PID values, rates, filter settings, firmware version

## Analysis Engine

### Noise Analyzer (FFT)

- Per-axis frequency spectrum of gyro noise
- Identify motor noise peaks (motor RPM harmonics)
- Compare pre-filter vs post-filter noise
- Detect throttle-correlated noise bands
- Evaluate filter effectiveness

### Step Response Analyzer

- Setpoint vs gyro tracking per axis (roll/pitch/yaw)
- Metrics: rise time, overshoot %, settling time, steady-state error
- Oscillation detection (sustained overshoot cycles)

### Issue Detector

| Issue | Symptom |
|-------|---------|
| P too high | High-freq oscillation, hot motors |
| P too low | Sluggish, doesn't track setpoint |
| D too high | Motor heat, noise amplification |
| D too low | Overshoot, bounce-back on stops |
| I too high | Slow wobble on hover |
| I too low | Drift, poor wind rejection |
| Filter too aggressive | Latency, mushy response |
| Filter too loose | Noise bleedthrough, motor heat |

### Motor Heat Estimation

- D-term energy (sum of squared D-term output) per motor
- Throttle duty cycle and duration factoring
- Relative heat index: cool / warm / hot / danger
- Identify worst-offender axis

## Quad Profile System

| Input | Options | Impact |
|-------|---------|--------|
| Cell count | 3S, 4S, 5S, 6S | Higher voltage = lower PID baseline |
| Prop size | 2", 3", 4", 5", 6", 7"+ | Larger = more inertia, different D needs |
| Frame size | Micro, 3", 5", 7", 10"+ (optional) | Weight/drag affects I term |
| Flying style | Freestyle, Race, Cinematic, Long Range | Aggressiveness of tune |

## PID Optimizer

- Known-good baseline PIDs per quad class
- Measure actual response from blackbox data
- Ziegler-Nichols-inspired tuning adapted for multirotors
- Adjust based on detected issues
- Output with confidence level and "why" explanation

## Filter Optimizer

- D-term lowpass 1 & 2 (type: PT1/BIQUAD/PT2/PT3, cutoff Hz)
- Gyro lowpass 1 & 2 (type + cutoff)
- Dynamic notch filter settings
- RPM filter (harmonics count, min/max frequency, Q factor)
- Goal: maximize noise rejection while minimizing latency

## UI Design (4 Tabs)

### Tab 1: Log Viewer
- Drag-and-drop file loading
- Scrollable time-series: gyro, setpoint, PID terms, motor output
- Axis toggle (R/P/Y), zoom/pan, time range selection
- Throttle overlay

### Tab 2: Analysis Dashboard
- FFT noise spectrum per axis with motor peaks highlighted
- Step response with overshoot/settling markers
- Color-coded issue cards (red/yellow/green)
- Filter response curves (current vs recommended)
- Motor heat estimation display

### Tab 3: Quad Profile & Tune
- Quad setup: cell count, prop size, frame, style dropdowns
- Side-by-side current vs suggested PIDs
- Side-by-side current vs suggested filter settings
- Confidence indicator per suggestion
- "Why" tooltips per change
- "Copy CLI" button (Betaflight `set` commands)
- Export before/after diff

### Tab 4: History
- Track tune iterations over time
- Compare logs from different sessions
- Improvement/regression metrics

### General
- Dark theme
- pyqtgraph 60fps interactive plots
- Keyboard shortcuts
