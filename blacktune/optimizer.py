"""PID/filter optimizer with baseline profiles and CLI export.

Takes analysis results (step response, noise, D-term RMS, motor heat) and
produces actionable PID/filter changes as Betaflight CLI commands.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .models import (
    QuadProfile,
    PIDValues,
    FilterSettings,
    AnalysisResult,
    StepResponseMetrics,
    Issue,
)


# ── Constants ────────────────────────────────────────────────────────────────

# Betaflight defaults for a 5" 4S freestyle quad
_BASE_PIDS = {
    "roll":  {"p": 45, "i": 80, "d": 30, "d_max": 40, "f": 120},
    "pitch": {"p": 47, "i": 84, "d": 34, "d_max": 46, "f": 125},
    "yaw":   {"p": 45, "i": 80, "d": 0,  "d_max": 0,  "f": 120},
}

# Higher voltage = more authority per PID unit = lower PIDs needed
_VOLTAGE_SCALE = {3: 1.15, 4: 1.0, 5: 0.85, 6: 0.70}

# Prop scaling: smaller props = less inertia = need HIGHER PIDs (less authority).
# The raw table values are {2: 0.75, 3: 0.85, ...} representing physical size factor.
# We INVERT them so smaller props get higher gains.
_PROP_SIZE_FACTOR = {2: 0.75, 3: 0.85, 4: 0.92, 5: 1.0, 6: 1.10, 7: 1.20}

# Style multiplier: race = snappy, cinematic = smooth
_STYLE_SCALE = {
    "freestyle": 1.0,
    "race": 1.15,
    "cinematic": 0.80,
    "long_range": 0.75,
}

# Filter baselines by prop size bracket
_FILTER_TABLE = {
    # (gyro_lpf1, gyro_lpf2, dterm_lpf1, dterm_lpf2)
    "small":  (300, 600, 85, 170),   # <= 3"
    "medium": (250, 500, 75, 150),   # <= 5"
    "large":  (200, 400, 60, 120),   # > 5"
}

# Safety clamp ranges
_CLAMP_P = (20, 120)
_CLAMP_I = (30, 200)
_CLAMP_D = (0, 80)
_CLAMP_D_MAX = (0, 100)
_CLAMP_F = (0, 300)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _interpolate_prop_scale(prop_size: float) -> float:
    """Interpolate prop scale factor for arbitrary prop sizes.

    We INVERT the physical size factor so that smaller props yield higher PIDs.
    Physical factor: 2"->0.75, 3"->0.85, ..., 7"->1.20
    PID scale = 1.0 / physical_factor (so 3" -> 1/0.85 ~ 1.176, 7" -> 1/1.20 ~ 0.833)
    """
    sizes = sorted(_PROP_SIZE_FACTOR.keys())
    prop_clamped = max(sizes[0], min(sizes[-1], prop_size))

    # Find bracketing sizes
    for i in range(len(sizes) - 1):
        if sizes[i] <= prop_clamped <= sizes[i + 1]:
            lo, hi = sizes[i], sizes[i + 1]
            t = (prop_clamped - lo) / (hi - lo) if hi != lo else 0
            phys = _PROP_SIZE_FACTOR[lo] + t * (_PROP_SIZE_FACTOR[hi] - _PROP_SIZE_FACTOR[lo])
            return 1.0 / phys

    # Exact match at boundary
    return 1.0 / _PROP_SIZE_FACTOR[round(prop_clamped)]


def _get_voltage_scale(cell_count: int) -> float:
    """Get voltage scale factor, with linear interpolation for non-standard counts."""
    if cell_count in _VOLTAGE_SCALE:
        return _VOLTAGE_SCALE[cell_count]
    # Linear interpolation for non-standard cell counts
    cells = sorted(_VOLTAGE_SCALE.keys())
    clamped = max(cells[0], min(cells[-1], cell_count))
    for i in range(len(cells) - 1):
        if cells[i] <= clamped <= cells[i + 1]:
            lo, hi = cells[i], cells[i + 1]
            t = (clamped - lo) / (hi - lo)
            return _VOLTAGE_SCALE[lo] + t * (_VOLTAGE_SCALE[hi] - _VOLTAGE_SCALE[lo])
    return 1.0


def _get_style_scale(style: str) -> float:
    """Get style scale factor."""
    return _STYLE_SCALE.get(style.lower(), 1.0)


# ── Public API ───────────────────────────────────────────────────────────────

def get_baseline_pids(profile: QuadProfile) -> PIDValues:
    """Compute baseline PID values scaled for the quad's hardware and flying style.

    Betaflight defaults (5" 4S reference) are scaled by:
    - Voltage: higher cell count = lower PIDs
    - Prop size: smaller props = higher PIDs (less authority per motor)
    - Style: race = higher, cinematic = lower

    D and D_max use sqrt(prop_scale) because larger props have more inertia
    that naturally helps with damping (and for inverted scale, this means
    D adjustments are less aggressive).
    """
    v_scale = _get_voltage_scale(profile.cell_count)
    prop_scale = _interpolate_prop_scale(profile.prop_size)
    style_scale = _get_style_scale(profile.flying_style)

    # D-term uses less prop scaling (sqrt) because inertia helps damping
    d_prop_scale = math.sqrt(prop_scale)

    combined = v_scale * style_scale

    def _scale_axis(base: dict) -> dict:
        return {
            "p": round(base["p"] * combined * prop_scale),
            "i": round(base["i"] * combined * prop_scale),
            "d": round(base["d"] * combined * d_prop_scale),
            "d_max": round(base["d_max"] * combined * d_prop_scale),
            "f": round(base["f"] * combined * prop_scale),
        }

    roll = _scale_axis(_BASE_PIDS["roll"])
    pitch = _scale_axis(_BASE_PIDS["pitch"])
    yaw = _scale_axis(_BASE_PIDS["yaw"])

    return PIDValues(
        roll_p=roll["p"],
        roll_i=roll["i"],
        roll_d=roll["d"],
        roll_d_max=roll["d_max"],
        roll_f=roll["f"],
        pitch_p=pitch["p"],
        pitch_i=pitch["i"],
        pitch_d=pitch["d"],
        pitch_d_max=pitch["d_max"],
        pitch_f=pitch["f"],
        yaw_p=yaw["p"],
        yaw_i=yaw["i"],
        yaw_d=yaw["d"],
        yaw_f=yaw["f"],
    )


def get_baseline_filters(profile: QuadProfile) -> FilterSettings:
    """Compute baseline filter settings based on prop size.

    Smaller props spin faster, producing noise at higher frequencies,
    so filter cutoffs can be higher (less filtering needed at low frequencies).
    """
    if profile.prop_size <= 3.0:
        gyro1, gyro2, dterm1, dterm2 = _FILTER_TABLE["small"]
    elif profile.prop_size <= 5.0:
        gyro1, gyro2, dterm1, dterm2 = _FILTER_TABLE["medium"]
    else:
        gyro1, gyro2, dterm1, dterm2 = _FILTER_TABLE["large"]

    return FilterSettings(
        gyro_lpf1_type="PT1",
        gyro_lpf1_hz=gyro1,
        gyro_lpf1_dyn_min_hz=gyro1,
        gyro_lpf1_dyn_max_hz=gyro1 * 2,
        gyro_lpf2_type="PT1",
        gyro_lpf2_hz=gyro2,
        dterm_lpf1_type="PT1",
        dterm_lpf1_hz=dterm1,
        dterm_lpf1_dyn_min_hz=dterm1,
        dterm_lpf1_dyn_max_hz=dterm1 * 2,
        dterm_lpf2_type="PT1",
        dterm_lpf2_hz=dterm2,
    )


def optimize_pids(
    current: PIDValues,
    profile: QuadProfile,
    analysis: AnalysisResult,
) -> PIDValues:
    """Adjust current PIDs based on step response metrics and D-term analysis.

    If current PIDs are zero (e.g. CSV with no header data), baseline PIDs
    scaled for the quad profile are used as the starting point.

    Each axis is optimized independently. Rules:
    - Overshoot > 20%: reduce P by up to 25%
    - Overshoot < 3% AND rise_time > 50ms: increase P by up to 20%
    - D-term RMS > 25: reduce D by up to 30%
    - Overshoot > 10% AND settling > 80ms: increase D by up to 15%
    - Steady-state error > 0.1: increase I by 10%
    - Safety clamps always applied
    """
    # Fall back to profile-scaled baseline when current PIDs are zeros (CSV case)
    baseline = get_baseline_pids(profile)

    axes = ["roll", "pitch", "yaw"]
    result = {}

    for axis in axes:
        step: StepResponseMetrics = analysis.step_response.get(axis)
        d_rms: float = analysis.d_term_rms.get(axis, 0.0)

        p = getattr(current, f"{axis}_p") or getattr(baseline, f"{axis}_p")
        i = getattr(current, f"{axis}_i") or getattr(baseline, f"{axis}_i")
        d = getattr(current, f"{axis}_d") or getattr(baseline, f"{axis}_d")
        f = getattr(current, f"{axis}_f") or getattr(baseline, f"{axis}_f")

        # D_max: yaw doesn't have d_max field in PIDValues
        if axis != "yaw":
            d_max = getattr(current, f"{axis}_d_max") or getattr(baseline, f"{axis}_d_max")
        else:
            d_max = 0

        if step is not None:
            # P adjustment based on overshoot
            if step.overshoot_pct > 20:
                # Scale reduction: 20% overshoot -> small reduction, 40%+ -> 25% reduction
                reduction = min(0.25, (step.overshoot_pct - 20) / 80)
                p *= (1.0 - reduction)
            elif step.overshoot_pct < 3 and step.rise_time_ms > 50:
                # Sluggish: scale increase based on how slow
                increase = min(0.20, (step.rise_time_ms - 50) / 100)
                p *= (1.0 + increase)

            # D adjustment: high D RMS -> reduce
            if d_rms > 25:
                reduction = min(0.30, (d_rms - 25) / 25)
                d *= (1.0 - reduction)
                d_max *= (1.0 - reduction)
            # D adjustment: overshoot + slow settling -> increase D
            elif step.overshoot_pct > 10 and step.settling_time_ms > 80:
                increase = min(0.15, (step.overshoot_pct - 10) / 100 + (step.settling_time_ms - 80) / 400)
                d *= (1.0 + increase)
                d_max *= (1.0 + increase)

            # I adjustment: steady-state error
            if step.steady_state_error > 0.1:
                i *= 1.10

        # Apply safety clamps
        p = _clamp(round(p), *_CLAMP_P)
        i = _clamp(round(i), *_CLAMP_I)
        d = _clamp(round(d), *_CLAMP_D)
        d_max = _clamp(round(d_max), *_CLAMP_D_MAX)
        f = _clamp(round(f), *_CLAMP_F)

        result[f"{axis}_p"] = p
        result[f"{axis}_i"] = i
        result[f"{axis}_d"] = d
        result[f"{axis}_f"] = f
        if axis != "yaw":
            result[f"{axis}_d_max"] = d_max

    return PIDValues(
        roll_p=result["roll_p"],
        roll_i=result["roll_i"],
        roll_d=result["roll_d"],
        roll_d_max=result.get("roll_d_max", 0),
        roll_f=result["roll_f"],
        pitch_p=result["pitch_p"],
        pitch_i=result["pitch_i"],
        pitch_d=result["pitch_d"],
        pitch_d_max=result.get("pitch_d_max", 0),
        pitch_f=result["pitch_f"],
        yaw_p=result["yaw_p"],
        yaw_i=result["yaw_i"],
        yaw_d=result["yaw_d"],
        yaw_f=result["yaw_f"],
    )


def optimize_filters(
    current: FilterSettings,
    profile: QuadProfile,
    analysis: AnalysisResult,
) -> FilterSettings:
    """Adjust filter settings based on noise analysis and motor heat.

    Rules:
    - High motor heat (>0.7): tighten D-term filters by 15%, switch to PT2
    - High D-term RMS (>20): tighten D-term lowpass by 10%
    - NOISE_HIGH issues: tighten gyro filters by 15%
    - Suggest RPM filter if not enabled and heat > 0.5
    """
    gyro_lpf1_type = current.gyro_lpf1_type
    gyro_lpf1_hz = current.gyro_lpf1_hz
    gyro_lpf2_type = current.gyro_lpf2_type
    gyro_lpf2_hz = current.gyro_lpf2_hz
    dterm_lpf1_type = current.dterm_lpf1_type
    dterm_lpf1_hz = current.dterm_lpf1_hz
    dterm_lpf2_type = current.dterm_lpf2_type
    dterm_lpf2_hz = current.dterm_lpf2_hz

    # Average motor heat
    heat_values = list(analysis.motor_heat_index.values())
    avg_heat = sum(heat_values) / len(heat_values) if heat_values else 0.0

    # Average D-term RMS across axes
    d_rms_values = list(analysis.d_term_rms.values())
    avg_d_rms = sum(d_rms_values) / len(d_rms_values) if d_rms_values else 0.0

    # Check for NOISE_HIGH issues
    has_noise_high = any(
        iss.category == "NOISE_HIGH" for iss in analysis.issues
    )

    # High motor heat -> tighten D-term filters, switch to PT2
    if avg_heat > 0.7:
        dterm_lpf1_hz *= 0.85  # 15% tighter
        dterm_lpf2_hz *= 0.85
        dterm_lpf1_type = "PT2"

    # High D-term RMS -> tighten D-term lowpass
    if avg_d_rms > 20:
        dterm_lpf1_hz *= 0.90  # 10% tighter
        dterm_lpf2_hz *= 0.90

    # NOISE_HIGH -> tighten gyro filters
    if has_noise_high:
        gyro_lpf1_hz *= 0.85  # 15% tighter
        gyro_lpf2_hz *= 0.85

    # Round filter values
    gyro_lpf1_hz = round(gyro_lpf1_hz)
    gyro_lpf2_hz = round(gyro_lpf2_hz)
    dterm_lpf1_hz = round(dterm_lpf1_hz)
    dterm_lpf2_hz = round(dterm_lpf2_hz)

    # Dynamic ranges track static values
    gyro_lpf1_dyn_min = gyro_lpf1_hz
    gyro_lpf1_dyn_max = gyro_lpf1_hz * 2
    dterm_lpf1_dyn_min = dterm_lpf1_hz
    dterm_lpf1_dyn_max = dterm_lpf1_hz * 2

    return FilterSettings(
        gyro_lpf1_type=gyro_lpf1_type,
        gyro_lpf1_hz=gyro_lpf1_hz,
        gyro_lpf1_dyn_min_hz=gyro_lpf1_dyn_min,
        gyro_lpf1_dyn_max_hz=gyro_lpf1_dyn_max,
        gyro_lpf2_type=gyro_lpf2_type,
        gyro_lpf2_hz=gyro_lpf2_hz,
        dterm_lpf1_type=dterm_lpf1_type,
        dterm_lpf1_hz=dterm_lpf1_hz,
        dterm_lpf1_dyn_min_hz=dterm_lpf1_dyn_min,
        dterm_lpf1_dyn_max_hz=dterm_lpf1_dyn_max,
        dterm_lpf2_type=dterm_lpf2_type,
        dterm_lpf2_hz=dterm_lpf2_hz,
        dyn_notch_count=current.dyn_notch_count,
        dyn_notch_q=current.dyn_notch_q,
        dyn_notch_min_hz=current.dyn_notch_min_hz,
        dyn_notch_max_hz=current.dyn_notch_max_hz,
        rpm_harmonics=current.rpm_harmonics,
        rpm_min_hz=current.rpm_min_hz,
        rpm_q=current.rpm_q,
    )


def generate_cli_commands(pids: PIDValues, filters: FilterSettings) -> str:
    """Generate Betaflight CLI commands for PID and filter settings.

    Returns a multi-line string of 'set' commands ready to paste into
    the Betaflight CLI, ending with 'save'.
    """
    lines = []
    lines.append("# BlackTune PID suggestions")

    # PID values
    lines.append(f"set p_roll = {round(pids.roll_p)}")
    lines.append(f"set i_roll = {round(pids.roll_i)}")
    lines.append(f"set d_roll = {round(pids.roll_d)}")
    lines.append(f"set p_pitch = {round(pids.pitch_p)}")
    lines.append(f"set i_pitch = {round(pids.pitch_i)}")
    lines.append(f"set d_pitch = {round(pids.pitch_d)}")
    lines.append(f"set p_yaw = {round(pids.yaw_p)}")
    lines.append(f"set i_yaw = {round(pids.yaw_i)}")
    lines.append(f"set d_yaw = {round(pids.yaw_d)}")

    # D_max
    lines.append(f"set d_max_roll = {round(pids.roll_d_max)}")
    lines.append(f"set d_max_pitch = {round(pids.pitch_d_max)}")
    lines.append(f"set d_max_yaw = 0")

    # Feedforward
    lines.append(f"set f_roll = {round(pids.roll_f)}")
    lines.append(f"set f_pitch = {round(pids.pitch_f)}")
    lines.append(f"set f_yaw = {round(pids.yaw_f)}")

    # Filter settings
    lines.append("")
    lines.append("# Filter settings")
    lines.append(f"set gyro_lpf1_type = {filters.gyro_lpf1_type}")
    lines.append(f"set gyro_lpf1_static_hz = {round(filters.gyro_lpf1_hz)}")
    lines.append(f"set gyro_lpf1_dyn_min_hz = {round(filters.gyro_lpf1_dyn_min_hz)}")
    lines.append(f"set gyro_lpf1_dyn_max_hz = {round(filters.gyro_lpf1_dyn_max_hz)}")
    lines.append(f"set gyro_lpf2_type = {filters.gyro_lpf2_type}")
    lines.append(f"set gyro_lpf2_static_hz = {round(filters.gyro_lpf2_hz)}")
    lines.append(f"set dterm_lpf1_type = {filters.dterm_lpf1_type}")
    lines.append(f"set dterm_lpf1_static_hz = {round(filters.dterm_lpf1_hz)}")
    lines.append(f"set dterm_lpf1_dyn_min_hz = {round(filters.dterm_lpf1_dyn_min_hz)}")
    lines.append(f"set dterm_lpf1_dyn_max_hz = {round(filters.dterm_lpf1_dyn_max_hz)}")
    lines.append(f"set dterm_lpf2_type = {filters.dterm_lpf2_type}")
    lines.append(f"set dterm_lpf2_static_hz = {round(filters.dterm_lpf2_hz)}")

    # Dynamic notch
    lines.append(f"set dyn_notch_count = {filters.dyn_notch_count}")
    lines.append(f"set dyn_notch_q = {filters.dyn_notch_q}")
    lines.append(f"set dyn_notch_min_hz = {round(filters.dyn_notch_min_hz)}")
    lines.append(f"set dyn_notch_max_hz = {round(filters.dyn_notch_max_hz)}")

    # RPM filter
    lines.append(f"set rpm_filter_harmonics = {filters.rpm_harmonics}")
    lines.append(f"set rpm_filter_min_hz = {round(filters.rpm_min_hz)}")
    lines.append(f"set rpm_filter_q = {filters.rpm_q}")
    lines.append(f"set dshot_bidir = ON")

    lines.append("")
    lines.append("save")

    return "\n".join(lines)
