"""Generate a realistic but MESSY synthetic Betaflight blackbox CSV.

Simulates a 5" 6S freestyle quad with a BAD tune:
- P too high on roll  -> 28% overshoot on roll step inputs (P_HIGH red)
- D too low on pitch  -> 18% overshoot with 147ms settling (D_LOW + P_HIGH yellow)
- Noisy gyro          -> motor noise at ~250Hz + 3 harmonics (NOISE_HIGH red)
- Hot motors          -> high-frequency D-term noise in motor commands
- D too high on roll  -> D-term RMS > 30 (D_HIGH red)
- Sluggish yaw        -> slow rise time > 60ms (P_LOW yellow)

Strategy: directly synthesize gyro = setpoint * impulse_response (convolution)
so the Wiener deconvolution in the analyzer recovers the exact step response
shape we design.  PID terms and motor outputs are computed to be realistic
in magnitude.

Outputs: test_flight.csv (30 seconds at 2kHz)
"""
import csv
import numpy as np
import os


# ──────────────────────────────────────────────────────────────────────
# Simulation parameters
# ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 2000        # Hz
DURATION_S = 30.0         # seconds
DT = 1.0 / SAMPLE_RATE
N_SAMPLES = int(DURATION_S * SAMPLE_RATE)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_flight.csv")


# ──────────────────────────────────────────────────────────────────────
# Impulse response design
# ──────────────────────────────────────────────────────────────────────
def make_impulse_response(sample_rate, overshoot_pct, settling_ms, rise_ms,
                          steady_state_offset=0.0, length_ms=500.0):
    """Create an impulse response whose cumulative sum yields a step
    response with the specified characteristics.

    Models a second-order system:
      step(t) = target * (1 - e^{-sigma*t} * (cos(wd*t) + (sigma/wd)*sin(wd*t)))
    """
    n = int(length_ms / 1000.0 * sample_rate)
    t = np.arange(n) / sample_rate

    # Damping ratio from overshoot
    if overshoot_pct > 0.5:
        log_os = np.log(overshoot_pct / 100.0)
        zeta = -log_os / np.sqrt(np.pi**2 + log_os**2)
    else:
        zeta = 1.0

    zeta = np.clip(zeta, 0.05, 0.999)

    # Natural frequency from rise time
    rise_s = rise_ms / 1000.0
    omega_n = 1.8 / rise_s

    sigma = zeta * omega_n
    omega_d = omega_n * np.sqrt(max(1.0 - zeta**2, 1e-10))

    # Step response
    target = 1.0 - steady_state_offset
    step = target * (1.0 - np.exp(-sigma * t) * (
        np.cos(omega_d * t) + (sigma / omega_d) * np.sin(omega_d * t)
    ))

    # Impulse response = derivative of step response
    impulse = np.diff(step, prepend=0.0)
    return impulse


def generate_gyro_from_setpoint(setpoint, impulse_response, noise):
    """Convolve setpoint with impulse response, add noise."""
    gyro_clean = np.convolve(setpoint, impulse_response, mode='full')[:len(setpoint)]
    return gyro_clean + noise


# ──────────────────────────────────────────────────────────────────────
# PID term reconstruction
# ──────────────────────────────────────────────────────────────────────
def compute_pid_terms(setpoint, gyro, kp, ki, kd, kf, d_noise_amp=0.0, rng=None):
    """Compute realistic PID term arrays from setpoint/gyro.

    Output ranges are realistic for Betaflight:
    - P-term: roughly -100 to +100
    - I-term: roughly -50 to +50 (slow accumulator)
    - D-term: roughly -50 to +50 (with noise)
    - F-term: spikes at stick transitions
    """
    n = len(setpoint)
    error = setpoint - gyro

    # P-term: scale error (deg/s) down to BF PID output range
    # With 300 deg/s error and P=85: 85*300*scale should be ~80-100
    # So scale ~ 0.004
    p_scale = 0.004
    p_term = kp * error * p_scale

    # I-term: slow integrator with anti-windup
    # BF I output is typically -30 to +30
    i_scale = 0.00002
    i_term = np.zeros(n)
    i_accum = 0.0
    for idx in range(n):
        i_accum += ki * error[idx] * DT * i_scale
        i_accum = np.clip(i_accum, -50.0, 50.0)
        i_term[idx] = i_accum

    # D-term: derivative of gyro (D-on-measurement), NOT of error.
    # diff(gyro) at 2kHz has small per-sample deltas.
    # We want D-term RMS ~25-35 for "noisy" roll, ~12-18 for pitch, ~3 for yaw.
    # The direct additive noise approach is cleanest.
    d_term = np.zeros(n)
    if kd > 0:
        # Low-frequency D component from gyro changes (useful damping signal)
        d_smooth = np.zeros(n)
        d_smooth[1:] = -kd * np.diff(gyro) * 0.02  # small scale factor
        # Low-pass filter the smooth component (BF filters D-term)
        from scipy.signal import butter, sosfilt
        nyq = SAMPLE_RATE / 2.0
        cutoff = min(150.0, nyq * 0.9)
        sos = butter(2, cutoff / nyq, btype='low', output='sos')
        d_smooth = sosfilt(sos, d_smooth)
        d_term += d_smooth

    # Add D-term noise (this is the noisy component that heats motors)
    if rng is not None and d_noise_amp > 0:
        d_term += d_noise_amp * rng.normal(0, 1, size=n)

    # F-term: feedforward = derivative of setpoint, scaled to BF range
    f_term = np.zeros(n)
    f_term[1:] = kf * np.diff(setpoint) * 0.002

    return p_term, i_term, d_term, f_term


# ──────────────────────────────────────────────────────────────────────
# Stick events
# ──────────────────────────────────────────────────────────────────────
STICK_EVENTS = [
    # Roll: multiple steps to give deconvolution good data
    (1.0,  1.3,  0,  300.0),
    (2.0,  2.3,  0, -250.0),
    (4.0,  4.4,  0,  350.0),
    (5.5,  5.8,  0, -300.0),
    (8.0,  8.3,  0,  280.0),
    (12.0, 12.3, 0, -320.0),
    (15.0, 15.3, 0,  290.0),
    (17.0, 17.3, 0, -270.0),
    (26.0, 26.3, 0,  320.0),
    (27.5, 27.8, 0, -290.0),
    (29.0, 29.2, 0,  280.0),

    # Pitch: multiple steps for bounce-back detection
    (3.0,  3.3,  1,  250.0),
    (4.5,  4.8,  1, -200.0),
    (7.0,  7.3,  1,  300.0),
    (9.0,  9.3,  1, -280.0),
    (13.0, 13.3, 1,  260.0),
    (16.0, 16.3, 1, -240.0),
    (19.0, 19.3, 1,  270.0),
    (23.0, 23.3, 1,  230.0),

    # Yaw: sustained inputs to show steady-state error
    (6.0,  6.8,  2,  200.0),
    (10.0, 10.8, 2, -180.0),
    (14.0, 14.8, 2,  220.0),
    (18.0, 18.8, 2, -200.0),
    (22.0, 22.8, 2,  190.0),

    # Combined maneuvers
    (20.0, 20.3, 0,  280.0),
    (20.0, 20.3, 1,  200.0),
    (24.0, 24.4, 1, -250.0),
    (24.0, 24.5, 2, -180.0),
]


# ──────────────────────────────────────────────────────────────────────
# Throttle profile
# ──────────────────────────────────────────────────────────────────────
def generate_throttle(n):
    t = np.arange(n) / SAMPLE_RATE
    throttle = np.full(n, 1400.0)

    mask = (t >= 3.5) & (t < 5.0)
    throttle[mask] = 1400 + 500 * np.sin(np.pi / 2 * (t[mask] - 3.5) / 1.5)

    mask = (t >= 5.0) & (t < 5.5)
    throttle[mask] = 1900 - 700 * (t[mask] - 5.0) / 0.5

    mask = (t >= 5.5) & (t < 7.0)
    throttle[mask] = 1200 + 200 * (t[mask] - 5.5) / 1.5

    mask = (t >= 11.0) & (t < 13.0)
    throttle[mask] = 1400 + 500 * np.sin(np.pi / 2 * (t[mask] - 11.0) / 2.0)

    mask = (t >= 13.0) & (t < 13.5)
    throttle[mask] = 1900 - 800 * (t[mask] - 13.0) / 0.5

    mask = (t >= 13.5) & (t < 15.0)
    throttle[mask] = 1100 + 300 * (t[mask] - 13.5) / 1.5

    mask = (t >= 15.0) & (t < 18.0)
    throttle[mask] = 1400

    mask = (t >= 18.0) & (t < 20.0)
    throttle[mask] = 1400 + 400 * np.sin(np.pi / 2 * (t[mask] - 18.0) / 2.0)

    mask = (t >= 20.0) & (t < 25.0)
    throttle[mask] = 1400 + 200 * np.sin(2 * np.pi * 0.5 * t[mask])

    mask = (t >= 25.0) & (t < 28.0)
    throttle[mask] = 1400 + 500 * np.sin(np.pi / 2 * (t[mask] - 25.0) / 3.0)

    mask = (t >= 28.0) & (t < 29.0)
    throttle[mask] = 1900 - 600 * (t[mask] - 28.0)

    mask = (t >= 29.0)
    throttle[mask] = 1300

    return np.clip(throttle, 1000, 2000)


# ──────────────────────────────────────────────────────────────────────
# Setpoint generation
# ──────────────────────────────────────────────────────────────────────
def generate_setpoints(n):
    """Generate setpoint arrays using clean square steps.

    Square steps (no ramp) are essential for clean Wiener deconvolution.
    This matches what blackbox_decode shows: the setpoint after rate
    calculation is essentially a step function.
    """
    t = np.arange(n) / SAMPLE_RATE
    setpoints = [np.zeros(n), np.zeros(n), np.zeros(n)]

    for (start_s, end_s, axis, amp) in STICK_EVENTS:
        s_idx = int(start_s * SAMPLE_RATE)
        e_idx = int(end_s * SAMPLE_RATE)
        e_idx = min(e_idx, n)
        setpoints[axis][s_idx:e_idx] += amp

    return setpoints[0], setpoints[1], setpoints[2]


# ──────────────────────────────────────────────────────────────────────
# Noise models
# ──────────────────────────────────────────────────────────────────────
def motor_noise(t, throttle_frac, rng):
    """Motor vibration noise: fundamental ~250Hz + 3 harmonics.

    Unbalanced motors produce harmonics at multiples of the base frequency.
    Fundamental ~250Hz, 2nd harmonic ~500Hz, 3rd ~750Hz, 4th ~1000Hz.
    This creates >4 peaks above -10dB in the PSD -> triggers NOISE_HIGH.
    """
    base_freq = 180.0 + 150.0 * throttle_frac
    fund_amp = 1.5 + 8.0 * throttle_frac

    noise = (fund_amp * np.sin(2 * np.pi * base_freq * t)               # fundamental
             + fund_amp * 0.7 * np.sin(2 * np.pi * 2 * base_freq * t + 0.3)  # 2nd harmonic
             + fund_amp * 0.5 * np.sin(2 * np.pi * 3 * base_freq * t + 0.6)  # 3rd harmonic
             + fund_amp * 0.3 * np.sin(2 * np.pi * 4 * base_freq * t + 0.9)) # 4th harmonic

    broadband = rng.normal(0, 0.3 + 1.5 * throttle_frac, size=len(t))
    return noise + broadband


def propwash_noise(t, throttle, rng):
    """Low-freq turbulence during throttle chops."""
    throttle_frac = (throttle - 1000.0) / 1000.0
    dt_throttle = np.gradient(throttle_frac) * SAMPLE_RATE
    propwash_mask = dt_throttle < -0.5
    pw_noise = np.zeros_like(t)

    if np.any(propwash_mask):
        pw_component = (6.0 * np.sin(2 * np.pi * 45 * t)
                        + 4.0 * np.sin(2 * np.pi * 65 * t + 1.2)
                        + 3.0 * np.sin(2 * np.pi * 30 * t + 0.7))

        envelope = np.zeros_like(t)
        chop_indices = np.where(propwash_mask)[0]
        if len(chop_indices) > 0:
            gaps = np.diff(chop_indices)
            starts = [chop_indices[0]]
            for i, g in enumerate(gaps):
                if g > 10:
                    starts.append(chop_indices[i + 1])
            for s in starts:
                decay_len = int(0.5 * SAMPLE_RATE)
                end = min(s + decay_len, len(t))
                idx = np.arange(s, end)
                decay = np.exp(-3.0 * (idx - s) / SAMPLE_RATE)
                envelope[idx] = np.maximum(envelope[idx], decay)

        pw_noise = pw_component * envelope
        pw_noise += rng.normal(0, 2.0, size=len(t)) * envelope

    return pw_noise


# ──────────────────────────────────────────────────────────────────────
# Motor mixing
# ──────────────────────────────────────────────────────────────────────
def mix_motors(throttle, roll_pid, pitch_pid, yaw_pid, d_noise_per_motor):
    mix_scale = 0.5
    m0 = throttle + (+roll_pid - pitch_pid - yaw_pid) * mix_scale
    m1 = throttle + (+roll_pid + pitch_pid + yaw_pid) * mix_scale
    m2 = throttle + (-roll_pid - pitch_pid + yaw_pid) * mix_scale
    m3 = throttle + (-roll_pid + pitch_pid - yaw_pid) * mix_scale

    m0 += d_noise_per_motor[0]
    m1 += d_noise_per_motor[1]
    m2 += d_noise_per_motor[2]
    m3 += d_noise_per_motor[3]

    return (np.clip(m0, 1000, 2000), np.clip(m1, 1000, 2000),
            np.clip(m2, 1000, 2000), np.clip(m3, 1000, 2000))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(42)
    t = np.arange(N_SAMPLES) / SAMPLE_RATE

    print(f"Generating {DURATION_S}s flight log at {SAMPLE_RATE}Hz ({N_SAMPLES} samples)...")

    # 1. Throttle
    throttle = generate_throttle(N_SAMPLES)
    throttle_frac = (throttle - 1000.0) / 1000.0

    # 2. Setpoints
    sp_roll, sp_pitch, sp_yaw = generate_setpoints(N_SAMPLES)

    # 3. Design impulse responses
    # ROLL: P too high -> ~28% overshoot (red threshold = 25%)
    ir_roll = make_impulse_response(
        SAMPLE_RATE, overshoot_pct=28.0, settling_ms=120.0,
        rise_ms=12.0, steady_state_offset=0.0,
    )
    # PITCH: D too low -> 18% overshoot + ~147ms settling
    #   Triggers P_HIGH yellow (>15%) AND D_LOW yellow (>10% + settling>100ms)
    ir_pitch = make_impulse_response(
        SAMPLE_RATE, overshoot_pct=18.0, settling_ms=160.0,
        rise_ms=50.0, steady_state_offset=0.0,
    )
    # YAW: slow / sluggish response -> P_LOW yellow + FILTER_LAG yellow
    #   rise_time > 60ms with overshoot < 3%
    #   Use first-order system (overdamped, no oscillation) for guaranteed
    #   low overshoot. tau=36ms gives rise_time ~75ms.
    ir_n = int(0.5 * SAMPLE_RATE)
    ir_t = np.arange(ir_n) / SAMPLE_RATE
    tau_yaw = 0.100 / 2.197  # rise_time ~100ms (very sluggish)
    yaw_step = 1.0 - np.exp(-ir_t / tau_yaw)
    ir_yaw = np.diff(yaw_step, prepend=0.0)

    # 4. Noise signals (moderate -- strong enough for peak detection,
    #    but not so strong it swamps the deconvolution)
    noise_roll = motor_noise(t, throttle_frac, rng)
    noise_pitch = motor_noise(t, throttle_frac, rng) * 1.1
    noise_yaw = motor_noise(t, throttle_frac, rng) * 0.7

    pw_noise = propwash_noise(t, throttle, rng)
    noise_roll += pw_noise * 0.6
    noise_pitch += pw_noise * 0.8
    noise_yaw += pw_noise * 0.3

    # 5. Generate gyro
    print("  Generating roll gyro (P too high -> overshoot)...")
    gyro_roll = generate_gyro_from_setpoint(sp_roll, ir_roll, noise_roll)

    print("  Generating pitch gyro (D too low -> bounce-back)...")
    gyro_pitch = generate_gyro_from_setpoint(sp_pitch, ir_pitch, noise_pitch)

    print("  Generating yaw gyro (sluggish response -> P_LOW)...")
    gyro_yaw = generate_gyro_from_setpoint(sp_yaw, ir_yaw, noise_yaw)

    # 6. PID terms
    # Gains only affect PID term magnitudes, not the step response
    # (which comes from the convolution above).
    print("  Computing PID terms...")

    # ROLL: D-term noise RMS target ~32 (> 30 = D_HIGH red)
    p_roll, i_roll, d_roll, f_roll = compute_pid_terms(
        sp_roll, gyro_roll, kp=85, ki=65, kd=30, kf=150,
        d_noise_amp=32.0, rng=rng,
    )

    # PITCH: D-term noise RMS target ~22 (> 20 = D_HIGH yellow)
    p_pitch, i_pitch, d_pitch, f_pitch = compute_pid_terms(
        sp_pitch, gyro_pitch, kp=62, ki=60, kd=12, kf=140,
        d_noise_amp=22.0, rng=rng,
    )

    # YAW: D-term clean (D=0 on yaw)
    p_yaw, i_yaw, d_yaw, f_yaw = compute_pid_terms(
        sp_yaw, gyro_yaw, kp=55, ki=25, kd=0, kf=80,
        d_noise_amp=0.0, rng=rng,
    )

    # 7. Motor outputs
    roll_pid_total = p_roll + i_roll + d_roll + f_roll
    pitch_pid_total = p_pitch + i_pitch + d_pitch + f_pitch
    yaw_pid_total = p_yaw + i_yaw + d_yaw + f_yaw

    # Motor noise: D-term noise bleeds through mixer -> hot motors
    # Target: motor heat index ~0.5-0.8 (not all pinned at 1.0)
    # The motor heat estimator high-passes at 100Hz and looks at RMS 5-25
    d_motor_noise = []
    for motor_idx in range(4):
        # HF noise that scales with throttle
        hf = rng.normal(0, 8.0, size=N_SAMPLES) * (0.3 + 0.7 * throttle_frac)
        # Add resonant tone (motor bearing vibration)
        hf += 4.0 * np.sin(2 * np.pi * (280 + motor_idx * 15) * t) * throttle_frac
        d_motor_noise.append(hf)

    m0, m1, m2, m3 = mix_motors(
        throttle, roll_pid_total, pitch_pid_total, yaw_pid_total,
        d_motor_noise,
    )

    # 8. rcCommand
    rc_roll = sp_roll / 0.7
    rc_pitch = sp_pitch / 0.7
    rc_yaw = sp_yaw / 0.5

    # 9. Write CSV
    print(f"  Writing {OUT_PATH}...")

    header = [
        "loopIteration", "time (us)",
        "axisP[0]", "axisP[1]", "axisP[2]",
        "axisI[0]", "axisI[1]", "axisI[2]",
        "axisD[0]", "axisD[1]", "axisD[2]",
        "axisF[0]", "axisF[1]", "axisF[2]",
        "rcCommand[0]", "rcCommand[1]", "rcCommand[2]", "rcCommand[3]",
        "setpoint[0]", "setpoint[1]", "setpoint[2]", "setpoint[3]",
        "gyroADC[0]", "gyroADC[1]", "gyroADC[2]",
        "motor[0]", "motor[1]", "motor[2]", "motor[3]",
    ]

    with open(OUT_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for i in range(N_SAMPLES):
            time_us = int(i * 1_000_000 / SAMPLE_RATE)
            row = [
                i, time_us,
                f"{p_roll[i]:.2f}", f"{p_pitch[i]:.2f}", f"{p_yaw[i]:.2f}",
                f"{i_roll[i]:.2f}", f"{i_pitch[i]:.2f}", f"{i_yaw[i]:.2f}",
                f"{d_roll[i]:.2f}", f"{d_pitch[i]:.2f}", f"{d_yaw[i]:.2f}",
                f"{f_roll[i]:.2f}", f"{f_pitch[i]:.2f}", f"{f_yaw[i]:.2f}",
                f"{rc_roll[i]:.1f}", f"{rc_pitch[i]:.1f}", f"{rc_yaw[i]:.1f}",
                f"{throttle[i]:.0f}",
                f"{sp_roll[i]:.1f}", f"{sp_pitch[i]:.1f}", f"{sp_yaw[i]:.1f}",
                f"{throttle[i]:.0f}",
                f"{gyro_roll[i]:.2f}", f"{gyro_pitch[i]:.2f}", f"{gyro_yaw[i]:.2f}",
                f"{m0[i]:.0f}", f"{m1[i]:.0f}", f"{m2[i]:.0f}", f"{m3[i]:.0f}",
            ]
            writer.writerow(row)

    file_size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"  Done! {file_size_mb:.1f} MB, {N_SAMPLES} rows")
    print(f"  Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
