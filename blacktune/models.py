"""Data models for BlackTune."""
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np


@dataclass
class PIDValues:
    """PID gains for all axes. Values are Betaflight CLI integers (0-250 for P/I/D, 0-1000 for F)."""
    roll_p: float
    roll_i: float
    roll_d: float       # This is d_min in BF
    roll_d_max: float = 0
    roll_f: float = 0   # Feedforward
    pitch_p: float = 0
    pitch_i: float = 0
    pitch_d: float = 0
    pitch_d_max: float = 0
    pitch_f: float = 0
    yaw_p: float = 0
    yaw_i: float = 0
    yaw_d: float = 0
    yaw_f: float = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilterSettings:
    """Betaflight filter configuration."""
    gyro_lpf1_type: str = "PT1"
    gyro_lpf1_hz: float = 250
    gyro_lpf1_dyn_min_hz: float = 250
    gyro_lpf1_dyn_max_hz: float = 500
    gyro_lpf2_type: str = "PT1"
    gyro_lpf2_hz: float = 500
    dterm_lpf1_type: str = "PT1"
    dterm_lpf1_hz: float = 75
    dterm_lpf1_dyn_min_hz: float = 75
    dterm_lpf1_dyn_max_hz: float = 150
    dterm_lpf2_type: str = "PT1"
    dterm_lpf2_hz: float = 150
    dyn_notch_count: int = 3
    dyn_notch_q: int = 300        # Integer! Not float 3.5
    dyn_notch_min_hz: float = 150
    dyn_notch_max_hz: float = 600
    rpm_harmonics: int = 3
    rpm_min_hz: float = 100
    rpm_q: int = 500


@dataclass
class AxisData:
    """Time-series data for one axis (roll, pitch, or yaw)."""
    name: str
    gyro: np.ndarray            # Filtered gyro (deg/s)
    setpoint: np.ndarray        # PID setpoint (deg/s)
    p_term: np.ndarray          # P output
    i_term: np.ndarray          # I output
    d_term: np.ndarray          # D output
    time: np.ndarray            # Time in seconds
    f_term: Optional[np.ndarray] = None        # Feedforward
    gyro_unfiltered: Optional[np.ndarray] = None


@dataclass
class QuadProfile:
    """Quad hardware/style profile for PID baseline selection."""
    cell_count: int             # 3-6
    prop_size: float            # inches (2.0 - 7.0+)
    frame_size: Optional[str] = None
    flying_style: str = "freestyle"  # freestyle/race/cinematic/long_range


@dataclass
class FlightLog:
    """Complete parsed flight log."""
    roll: AxisData
    pitch: AxisData
    yaw: AxisData
    throttle: np.ndarray
    motors: np.ndarray          # shape (4, N) or (num_motors, N)
    sample_rate: int            # Hz
    duration_s: float
    firmware: str
    current_pids: PIDValues
    current_filters: Optional[FilterSettings] = None
    erpm: Optional[np.ndarray] = None  # shape (4, N), motor eRPM if available


@dataclass
class Issue:
    """A detected tuning issue."""
    axis: str
    category: str               # P_HIGH, P_LOW, D_HIGH, D_LOW, I_LOW, NOISE_HIGH, FILTER_LAG, GOOD
    severity: str               # red, yellow, green
    message: str
    detail: str


@dataclass
class StepResponseMetrics:
    """Metrics from step response analysis."""
    rise_time_ms: float
    overshoot_pct: float
    settling_time_ms: float
    peak_time_ms: float
    steady_state_error: float


@dataclass
class AnalysisResult:
    """Results from the full analysis pipeline."""
    step_response: dict         # axis_name -> StepResponseMetrics
    noise_peaks: dict           # axis_name -> list of (freq_hz, amplitude_db)
    issues: list                # list of Issue
    motor_heat_index: dict      # motor_idx -> float (0.0-1.0)
    d_term_rms: dict            # axis_name -> float
    filter_delay_ms: Optional[float] = None


@dataclass
class TuneRecommendation:
    """PID + filter recommendations with explanations."""
    suggested_pids: PIDValues
    suggested_filters: FilterSettings
    confidence: float           # 0.0 - 1.0
    explanations: dict          # param_name -> str reason
    cli_commands: str           # Betaflight CLI set commands
