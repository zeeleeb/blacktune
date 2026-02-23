"""Step response analysis via Wiener deconvolution.

Computes the closed-loop step response from setpoint/gyro pairs using the
same approach as PIDtoolbox (PTstepcalc.m) and Plasmatree PID-Analyzer:
overlapping windowed segments, FFT-based Wiener deconvolution, cumulative-sum
conversion from impulse to step response, and normalization.

The resulting step response tells us how the quad actually responds to stick
inputs -- overshoot means P too high or D too low, slow rise means P too low,
ringing means poor P/D balance.
"""
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
):
    """Compute step response using Wiener deconvolution.

    Algorithm (based on PIDtoolbox PTstepcalc.m and Plasmatree PID-Analyzer):
    1. Slide through data in overlapping segments (2s windows, 25% overlap)
    2. Skip segments where max(|setpoint|) < min_input_dps
    3. Apply Hann window to each segment
    4. FFT both setpoint and gyro
    5. Wiener deconvolution: H = GY * conj(SP) / (SP * conj(SP) + regularization)
    6. IFFT -> impulse response
    7. Cumulative sum -> step response (first response_window_s samples)
    8. Normalize so target = 1.0 (normalize by mean of last 20% of response)
    9. Average all valid segment responses

    Parameters
    ----------
    setpoint : 1-D array
        PID setpoint in deg/s.
    gyro : 1-D array
        Filtered gyro in deg/s.
    sample_rate : int
        Sampling frequency in Hz.
    segment_duration_s : float
        Length of each analysis window in seconds (default 2.0).
    response_window_s : float
        How much of the step response to keep in seconds (default 0.5).
    min_input_dps : float
        Minimum max(|setpoint|) in a segment to be considered valid (default 20.0).
    regularization : float
        Wiener deconvolution regularization to prevent division by near-zero (default 1e-4).

    Returns
    -------
    mean_step_response : 1-D array
        Averaged normalized step response (target = 1.0).
    time_array_seconds : 1-D array
        Time axis in seconds for the step response.
    """
    setpoint = np.asarray(setpoint, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)

    seg_len = int(segment_duration_s * sample_rate)
    resp_len = int(response_window_s * sample_rate)
    overlap = seg_len // 4  # 25% overlap
    step_size = seg_len - overlap

    # Hann window for each segment
    window = np.hanning(seg_len)

    # Collect valid segment step responses
    valid_responses = []

    n = len(setpoint)
    start = 0
    while start + seg_len <= n:
        sp_seg = setpoint[start:start + seg_len]
        gy_seg = gyro[start:start + seg_len]

        # Skip low-activity segments
        if np.max(np.abs(sp_seg)) < min_input_dps:
            start += step_size
            continue

        # Apply Hann window
        sp_w = sp_seg * window
        gy_w = gy_seg * window

        # FFT
        SP = np.fft.rfft(sp_w)
        GY = np.fft.rfft(gy_w)

        # Wiener deconvolution: H = GY * conj(SP) / (|SP|^2 + lambda)
        SP_conj = np.conj(SP)
        H = (GY * SP_conj) / (SP * SP_conj + regularization)

        # IFFT -> impulse response
        impulse = np.fft.irfft(H, n=seg_len)

        # Cumulative sum -> step response, take first resp_len samples
        step_resp = np.cumsum(impulse[:resp_len])

        # Normalize: divide by mean of last 20% so settled value = 1.0
        tail_start = int(resp_len * 0.8)
        tail_mean = np.mean(step_resp[tail_start:])

        if abs(tail_mean) > 1e-10:
            step_resp = step_resp / tail_mean

        valid_responses.append(step_resp)
        start += step_size

    # Time array for the response window
    time_arr = np.arange(resp_len) / sample_rate

    if len(valid_responses) == 0:
        # No valid segments -- return flat 1.0
        return np.ones(resp_len), time_arr

    # Average all valid segment responses
    mean_response = np.mean(np.array(valid_responses), axis=0)

    return mean_response, time_arr


def measure_step_metrics(response: np.ndarray, resp_time: np.ndarray) -> StepResponseMetrics:
    """Extract metrics from a normalized step response (target = 1.0).

    Parameters
    ----------
    response : 1-D array
        Normalized step response where settled value should be ~1.0.
    resp_time : 1-D array
        Time axis in seconds.

    Returns
    -------
    StepResponseMetrics
        rise_time_ms : time from 10% to 90% of target (1.0)
        overshoot_pct : max(0, (peak - 1.0) / 1.0 * 100)
        settling_time_ms : time until response stays within 5% of target
        peak_time_ms : time to reach peak value
        steady_state_error : abs(mean of last 20% - 1.0)
    """
    response = np.asarray(response, dtype=np.float64)
    resp_time = np.asarray(resp_time, dtype=np.float64)

    target = 1.0

    # --- Rise time: 10% to 90% of target ---
    threshold_10 = 0.1 * target
    threshold_90 = 0.9 * target

    idx_10 = _first_crossing(response, threshold_10)
    idx_90 = _first_crossing(response, threshold_90)

    if idx_10 is not None and idx_90 is not None and idx_90 > idx_10:
        rise_time_ms = (resp_time[idx_90] - resp_time[idx_10]) * 1000.0
    else:
        # If we can't find crossings, use full window
        rise_time_ms = (resp_time[-1] - resp_time[0]) * 1000.0

    # --- Overshoot ---
    peak_val = np.max(response)
    overshoot_pct = max(0.0, (peak_val - target) / target * 100.0)

    # --- Peak time ---
    peak_idx = np.argmax(response)
    peak_time_ms = resp_time[peak_idx] * 1000.0

    # --- Settling time: last time the response leaves the 5% band ---
    settling_band = 0.05 * target
    outside_band = np.abs(response - target) > settling_band

    if np.any(outside_band):
        # Last index that is outside the band
        last_outside = np.where(outside_band)[0][-1]
        if last_outside < len(resp_time) - 1:
            settling_time_ms = resp_time[last_outside + 1] * 1000.0
        else:
            # Never settles within window
            settling_time_ms = resp_time[-1] * 1000.0
    else:
        # Already within band at t=0
        settling_time_ms = 0.0

    # --- Steady-state error ---
    tail_start = int(len(response) * 0.8)
    steady_state = np.mean(response[tail_start:])
    steady_state_error = abs(steady_state - target)

    return StepResponseMetrics(
        rise_time_ms=rise_time_ms,
        overshoot_pct=overshoot_pct,
        settling_time_ms=settling_time_ms,
        peak_time_ms=peak_time_ms,
        steady_state_error=steady_state_error,
    )


def _first_crossing(signal: np.ndarray, threshold: float) -> int | None:
    """Find the first index where signal crosses above threshold."""
    indices = np.where(signal >= threshold)[0]
    if len(indices) == 0:
        return None
    return int(indices[0])
