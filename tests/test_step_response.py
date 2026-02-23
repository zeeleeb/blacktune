"""Tests for blacktune.analyzers.step_response -- Wiener deconvolution step response."""
import numpy as np
import pytest
from blacktune.analyzers.step_response import compute_step_response, measure_step_metrics
from blacktune.models import StepResponseMetrics


SAMPLE_RATE = 2000  # Hz, typical blackbox rate


def _make_setpoint_signal(duration_s=4.0, sample_rate=SAMPLE_RATE, step_dps=100.0):
    """Generate a setpoint with square-wave steps (alternating 0 / step_dps)."""
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    # Square wave with 0.5s period (so we get many segments with activity)
    sp = np.zeros(n)
    period_samples = int(0.5 * sample_rate)
    for i in range(0, n, period_samples * 2):
        sp[i:i + period_samples] = step_dps
    return sp, t


# ---------- compute_step_response ----------


def test_step_response_returns_correct_shape():
    """Random noise signals, verify output shapes match."""
    rng = np.random.default_rng(42)
    n = SAMPLE_RATE * 4  # 4 seconds
    setpoint = rng.normal(0, 50, n)
    gyro = rng.normal(0, 50, n)

    response, resp_time = compute_step_response(setpoint, gyro, SAMPLE_RATE)

    # Both should be 1-D arrays of the same length
    assert response.ndim == 1
    assert resp_time.ndim == 1
    assert len(response) == len(resp_time)
    assert len(response) > 0

    # Time should start at 0 and be monotonically increasing
    assert resp_time[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(resp_time) > 0)


def test_step_response_perfect_tracking():
    """gyro = setpoint -> response should be close to 1.0 everywhere."""
    sp, _ = _make_setpoint_signal(duration_s=6.0)
    gyro = sp.copy()

    response, resp_time = compute_step_response(sp, gyro, SAMPLE_RATE)

    # For perfect tracking, the step response should be ~1.0 after an initial ramp
    # Check the latter half of the response is near 1.0
    half = len(response) // 2
    assert np.mean(response[half:]) == pytest.approx(1.0, abs=0.15), (
        f"Perfect tracking response should settle near 1.0, got mean={np.mean(response[half:]):.3f}"
    )


def test_step_response_with_delay():
    """gyro = delayed setpoint -> peak should come after some time."""
    sp, _ = _make_setpoint_signal(duration_s=6.0)
    delay_samples = int(0.01 * SAMPLE_RATE)  # 10ms delay
    gyro = np.zeros_like(sp)
    gyro[delay_samples:] = sp[:-delay_samples]

    response, resp_time = compute_step_response(sp, gyro, SAMPLE_RATE)

    # The response should not start at 1.0 -- it should ramp up
    assert response[0] < 0.8, (
        f"Delayed system should start below 1.0, got {response[0]:.3f}"
    )

    # But should eventually reach ~1.0
    last_quarter = response[3 * len(response) // 4:]
    assert np.mean(last_quarter) == pytest.approx(1.0, abs=0.2), (
        f"Delayed system should settle near 1.0, got mean={np.mean(last_quarter):.3f}"
    )


def test_step_response_no_valid_segments():
    """If setpoint is always below threshold, return flat 1.0 line."""
    n = SAMPLE_RATE * 4
    sp = np.ones(n) * 5.0  # Below default min_input_dps=20
    gyro = np.ones(n) * 5.0

    response, resp_time = compute_step_response(sp, gyro, SAMPLE_RATE)

    # Should return flat 1.0
    np.testing.assert_allclose(response, 1.0, atol=1e-6)


# ---------- measure_step_metrics ----------


def test_step_metrics_no_overshoot():
    """Flat response at 1.0 -> overshoot = 0."""
    n = 200
    resp_time = np.linspace(0, 0.5, n)
    response = np.ones(n)

    metrics = measure_step_metrics(response, resp_time)

    assert isinstance(metrics, StepResponseMetrics)
    assert metrics.overshoot_pct == pytest.approx(0.0, abs=0.01)
    assert metrics.steady_state_error == pytest.approx(0.0, abs=0.01)


def test_step_metrics_clear_overshoot():
    """Response that peaks at 1.15 -> overshoot ~15%."""
    n = 200
    resp_time = np.linspace(0, 0.5, n)

    # Build a response: ramp up to 1.15, then settle to 1.0
    response = np.ones(n)
    # Ramp from 0 to 1.15 in first 25% of samples
    ramp_end = n // 4
    response[:ramp_end] = np.linspace(0, 1.15, ramp_end)
    # Overshoot region: peak at 1.15, decay to 1.0
    decay_end = n // 2
    response[ramp_end:decay_end] = 1.15 * np.exp(-np.linspace(0, 3, decay_end - ramp_end) * 0.5) + \
        (1.0 - 1.15 * np.exp(-np.linspace(0, 3, decay_end - ramp_end) * 0.5)) * \
        np.linspace(0, 1, decay_end - ramp_end)
    # Make it cleaner: just do a simple overshoot profile
    response = np.ones(n)
    t_norm = np.linspace(0, 5, n)
    # Classic second-order underdamped step response
    zeta = 0.4  # damping ratio -> ~25% overshoot for reference
    wn = 15.0
    response = 1.0 - np.exp(-zeta * wn * resp_time) * (
        np.cos(wn * np.sqrt(1 - zeta**2) * resp_time)
        + (zeta / np.sqrt(1 - zeta**2)) * np.sin(wn * np.sqrt(1 - zeta**2) * resp_time)
    )

    peak_val = np.max(response)
    expected_overshoot = max(0, (peak_val - 1.0) / 1.0 * 100)

    metrics = measure_step_metrics(response, resp_time)

    assert metrics.overshoot_pct == pytest.approx(expected_overshoot, abs=1.0)
    assert metrics.overshoot_pct > 10.0, f"Expected significant overshoot, got {metrics.overshoot_pct}%"
    assert metrics.peak_time_ms > 0


def test_step_metrics_slow_rise():
    """Response that takes long to reach 0.9 -> high rise_time_ms."""
    n = 400
    resp_time = np.linspace(0, 0.5, n)

    # Very slow exponential rise -- reaches 0.9 at about 2.3*tau
    tau = 0.15  # 150ms time constant -> rise_time ~ 2.3*0.15 = 345ms
    response = 1.0 - np.exp(-resp_time / tau)

    metrics = measure_step_metrics(response, resp_time)

    # Rise time (10% to 90%) for first-order: 2.2*tau ~ 330ms
    assert metrics.rise_time_ms > 200, f"Expected slow rise, got {metrics.rise_time_ms:.1f} ms"


def test_step_metrics_steady_state_error():
    """Response that settles at 0.92 -> error ~0.08."""
    n = 200
    resp_time = np.linspace(0, 0.5, n)

    # Quick rise to 0.92, stays there
    response = 0.92 * (1.0 - np.exp(-resp_time / 0.02))

    metrics = measure_step_metrics(response, resp_time)

    assert metrics.steady_state_error == pytest.approx(0.08, abs=0.02)
