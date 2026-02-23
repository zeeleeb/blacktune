"""Issue detection and motor heat estimation.

Analyzes step response metrics, D-term RMS, and noise peaks to identify
specific PID tuning problems. Instead of just showing graphs, we tell the
pilot *what* is wrong and *why* -- this is what feeds the PID optimizer.

Motor heat estimation detects when D-term noise is driving high-frequency
motor commands that convert to heat instead of thrust.
"""
import numpy as np
from scipy.signal import butter, sosfilt

from blacktune.models import Issue, StepResponseMetrics


def detect_pid_issues(
    axis: str,
    step_metrics: StepResponseMetrics,
    noise_peaks: list,
    d_rms: float,
) -> list:
    """Detect PID tuning issues from analysis data.

    Rules
    -----
    - P_HIGH : overshoot > 15% (yellow), > 25% (red)
    - P_LOW  : rise_time > 60ms and overshoot < 3%
    - D_HIGH : d_rms > 20 (yellow), > 30 (red) -- motors getting hot from noise
    - D_LOW  : overshoot > 10% AND settling_time > 100ms -- needs more damping
    - I_LOW  : steady_state_error > 0.1
    - I_HIGH : overshoot present AND settling oscillation pattern
    - NOISE_HIGH : > 3 strong noise peaks above -10dB
    - FILTER_LAG : rise_time > 50ms with low overshoot (filters adding delay)
    - GOOD   : no issues detected

    Parameters
    ----------
    axis : str
        Axis name ("roll", "pitch", "yaw").
    step_metrics : StepResponseMetrics
        Metrics from step response analysis.
    noise_peaks : list of (freq_hz, amplitude_db)
        Detected noise peaks from FFT analysis.
    d_rms : float
        RMS value of the D-term signal.

    Returns
    -------
    list of Issue
        Detected issues, or a single GOOD issue if nothing is wrong.
    """
    issues: list[Issue] = []

    # ── P_HIGH: excessive overshoot ──────────────────────────────────────
    if step_metrics.overshoot_pct > 25.0:
        issues.append(Issue(
            axis=axis,
            category="P_HIGH",
            severity="red",
            message=f"P gain too high on {axis}",
            detail=(
                f"Overshoot is {step_metrics.overshoot_pct:.1f}% (threshold: 25%). "
                "The quad is overshooting its target significantly. Lower P gain "
                "or increase D to add damping."
            ),
        ))
    elif step_metrics.overshoot_pct > 15.0:
        issues.append(Issue(
            axis=axis,
            category="P_HIGH",
            severity="yellow",
            message=f"P gain slightly high on {axis}",
            detail=(
                f"Overshoot is {step_metrics.overshoot_pct:.1f}% (threshold: 15%). "
                "Some overshoot is present. Consider lowering P or raising D slightly."
            ),
        ))

    # ── P_LOW: slow response with no overshoot ───────────────────────────
    if step_metrics.rise_time_ms > 60.0 and step_metrics.overshoot_pct < 3.0:
        issues.append(Issue(
            axis=axis,
            category="P_LOW",
            severity="yellow",
            message=f"P gain too low on {axis}",
            detail=(
                f"Rise time is {step_metrics.rise_time_ms:.1f}ms with only "
                f"{step_metrics.overshoot_pct:.1f}% overshoot. The quad feels "
                "sluggish. Increase P gain for sharper response."
            ),
        ))

    # ── D_HIGH: D-term noise heating motors ──────────────────────────────
    if d_rms > 30.0:
        issues.append(Issue(
            axis=axis,
            category="D_HIGH",
            severity="red",
            message=f"D-term noise dangerously high on {axis}",
            detail=(
                f"D-term RMS is {d_rms:.1f} (danger threshold: 30). "
                "Motors are being driven by noise and will overheat. "
                "Lower D gain or increase D-term filtering immediately."
            ),
        ))
    elif d_rms > 20.0:
        issues.append(Issue(
            axis=axis,
            category="D_HIGH",
            severity="yellow",
            message=f"D-term noise elevated on {axis}",
            detail=(
                f"D-term RMS is {d_rms:.1f} (warning threshold: 20). "
                "Motor heat from D noise is getting high. Consider lowering D "
                "or adding D-term filtering."
            ),
        ))

    # ── D_LOW: overshoot with slow settling (needs more damping) ─────────
    if step_metrics.overshoot_pct > 10.0 and step_metrics.settling_time_ms > 100.0:
        issues.append(Issue(
            axis=axis,
            category="D_LOW",
            severity="yellow",
            message=f"D gain too low on {axis}",
            detail=(
                f"Overshoot {step_metrics.overshoot_pct:.1f}% with settling time "
                f"{step_metrics.settling_time_ms:.1f}ms. The quad overshoots and "
                "takes too long to settle. Increase D gain for better damping."
            ),
        ))

    # ── I_LOW: persistent steady-state error ─────────────────────────────
    if step_metrics.steady_state_error > 0.1:
        issues.append(Issue(
            axis=axis,
            category="I_LOW",
            severity="yellow",
            message=f"I gain too low on {axis}",
            detail=(
                f"Steady-state error is {step_metrics.steady_state_error:.3f} "
                "(threshold: 0.1). The quad doesn't fully reach its target. "
                "Increase I gain to eliminate the offset."
            ),
        ))

    # ── I_HIGH: overshoot with oscillatory settling ──────────────────────
    # Heuristic: overshoot present + settling time high relative to rise time
    # indicates the I-term is winding up and causing oscillation
    if (step_metrics.overshoot_pct > 5.0
            and step_metrics.settling_time_ms > 2.0 * step_metrics.rise_time_ms
            and step_metrics.settling_time_ms > 80.0):
        issues.append(Issue(
            axis=axis,
            category="I_HIGH",
            severity="yellow",
            message=f"I gain may be too high on {axis}",
            detail=(
                f"Overshoot {step_metrics.overshoot_pct:.1f}% with settling time "
                f"{step_metrics.settling_time_ms:.1f}ms (>{2.0 * step_metrics.rise_time_ms:.0f}ms). "
                "This pattern suggests I-term windup. Consider lowering I gain or "
                "reducing iterm_relax limits."
            ),
        ))

    # ── NOISE_HIGH: too many strong noise peaks ──────────────────────────
    strong_peaks = [p for p in noise_peaks if p[1] > -10.0]
    if len(strong_peaks) > 3:
        issues.append(Issue(
            axis=axis,
            category="NOISE_HIGH",
            severity="red",
            message=f"High noise level on {axis}",
            detail=(
                f"{len(strong_peaks)} strong noise peaks detected above -10dB. "
                "This indicates significant mechanical vibration or electrical noise. "
                "Check prop balance, motor bearings, and frame rigidity."
            ),
        ))

    # ── FILTER_LAG: filters adding delay ─────────────────────────────────
    if step_metrics.rise_time_ms > 50.0 and step_metrics.overshoot_pct < 3.0:
        issues.append(Issue(
            axis=axis,
            category="FILTER_LAG",
            severity="yellow",
            message=f"Filter lag detected on {axis}",
            detail=(
                f"Rise time is {step_metrics.rise_time_ms:.1f}ms with only "
                f"{step_metrics.overshoot_pct:.1f}% overshoot. Filters may be "
                "cutting too aggressively, adding latency. Consider raising filter "
                "cutoff frequencies if noise allows."
            ),
        ))

    # ── GOOD: nothing wrong ──────────────────────────────────────────────
    if len(issues) == 0:
        issues.append(Issue(
            axis=axis,
            category="GOOD",
            severity="green",
            message=f"{axis} tuning looks good",
            detail=(
                f"No significant issues detected. Rise time {step_metrics.rise_time_ms:.1f}ms, "
                f"overshoot {step_metrics.overshoot_pct:.1f}%, settling "
                f"{step_metrics.settling_time_ms:.1f}ms, D-RMS {d_rms:.1f}."
            ),
        ))

    return issues


def compute_dterm_rms(d_term: np.ndarray) -> float:
    """Compute RMS of D-term signal.

    Parameters
    ----------
    d_term : 1-D array
        D-term time-series values.

    Returns
    -------
    float
        Root mean square of the D-term signal.
    """
    d_term = np.asarray(d_term, dtype=np.float64)
    return float(np.sqrt(np.mean(d_term ** 2)))


def estimate_motor_heat(
    d_terms,
    motors: np.ndarray,
    sample_rate: int,
) -> dict:
    """Estimate relative motor heat index (0-1) from high-frequency motor noise.

    High-frequency components in motor output are caused by D-term noise passing
    through to the mixer. These rapid current changes heat motors without
    producing useful thrust. This function isolates that energy and maps it to
    a 0-1 heat danger scale.

    Algorithm
    ---------
    1. High-pass filter motor commands above 100Hz to isolate noise
    2. Compute RMS of the high-frequency component per motor
    3. Normalize to 0-1 using empirical thresholds:
       - RMS < 5 -> 0.0 (cool)
       - RMS > 25 -> 1.0 (danger)
       - Linear interpolation between

    Parameters
    ----------
    d_terms : ignored
        Not used directly -- we filter the motor outputs instead.
    motors : ndarray, shape (num_motors, N)
        Motor output values (typically 0-100% or 0-2000 raw).
    sample_rate : int
        Sampling frequency in Hz.

    Returns
    -------
    dict of {motor_index: float}
        Heat index per motor, clamped to [0.0, 1.0].
    """
    motors = np.asarray(motors, dtype=np.float64)
    num_motors = motors.shape[0]

    # Design a high-pass Butterworth filter at 100Hz
    # Nyquist frequency
    nyquist = sample_rate / 2.0
    cutoff_hz = 100.0

    # Ensure cutoff is below Nyquist (required by filter design)
    if cutoff_hz >= nyquist:
        cutoff_hz = nyquist * 0.9

    sos = butter(2, cutoff_hz / nyquist, btype="high", output="sos")

    # Empirical thresholds (based on typical Betaflight motor output ranges)
    rms_cool = 5.0
    rms_danger = 25.0

    heat_index = {}
    for i in range(num_motors):
        # High-pass filter this motor's output
        hf_signal = sosfilt(sos, motors[i])

        # RMS of the high-frequency component
        rms = float(np.sqrt(np.mean(hf_signal ** 2)))

        # Linear map to [0, 1]
        heat = (rms - rms_cool) / (rms_danger - rms_cool)
        heat = float(np.clip(heat, 0.0, 1.0))

        heat_index[i] = heat

    return heat_index
