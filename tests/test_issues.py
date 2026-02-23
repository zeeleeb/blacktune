"""Tests for issue detection and motor heat estimation."""
import numpy as np
import pytest

from blacktune.models import Issue, StepResponseMetrics
from blacktune.analyzers.issues import detect_pid_issues, compute_dterm_rms, estimate_motor_heat


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_metrics(
    rise_time_ms=30.0,
    overshoot_pct=5.0,
    settling_time_ms=50.0,
    peak_time_ms=20.0,
    steady_state_error=0.02,
):
    """Build a StepResponseMetrics with sane defaults."""
    return StepResponseMetrics(
        rise_time_ms=rise_time_ms,
        overshoot_pct=overshoot_pct,
        settling_time_ms=settling_time_ms,
        peak_time_ms=peak_time_ms,
        steady_state_error=steady_state_error,
    )


def _categories(issues):
    """Extract the set of category strings from a list of Issue objects."""
    return {iss.category for iss in issues}


# ── detect_pid_issues ────────────────────────────────────────────────────────

class TestDetectPidIssues:
    """Tests for detect_pid_issues()."""

    def test_high_overshoot_yields_p_high(self):
        """Overshoot > 25% should produce a P_HIGH (red) issue."""
        metrics = _make_metrics(overshoot_pct=30.0)
        issues = detect_pid_issues("roll", metrics, noise_peaks=[], d_rms=5.0)
        cats = _categories(issues)
        assert "P_HIGH" in cats
        p_high = [i for i in issues if i.category == "P_HIGH"][0]
        assert p_high.severity == "red"
        assert p_high.axis == "roll"

    def test_moderate_overshoot_yields_yellow_p_high(self):
        """Overshoot 16-25% should produce a P_HIGH (yellow) issue."""
        metrics = _make_metrics(overshoot_pct=20.0)
        issues = detect_pid_issues("pitch", metrics, noise_peaks=[], d_rms=5.0)
        cats = _categories(issues)
        assert "P_HIGH" in cats
        p_high = [i for i in issues if i.category == "P_HIGH"][0]
        assert p_high.severity == "yellow"

    def test_slow_rise_low_overshoot_yields_p_low(self):
        """rise_time > 60ms AND overshoot < 3% -> P_LOW."""
        metrics = _make_metrics(rise_time_ms=80.0, overshoot_pct=1.0)
        issues = detect_pid_issues("roll", metrics, noise_peaks=[], d_rms=5.0)
        assert "P_LOW" in _categories(issues)

    def test_high_drms_yields_d_high(self):
        """d_rms > 30 -> D_HIGH (red)."""
        metrics = _make_metrics()
        issues = detect_pid_issues("roll", metrics, noise_peaks=[], d_rms=35.0)
        cats = _categories(issues)
        assert "D_HIGH" in cats
        d_high = [i for i in issues if i.category == "D_HIGH"][0]
        assert d_high.severity == "red"

    def test_moderate_drms_yields_yellow_d_high(self):
        """d_rms 21-30 -> D_HIGH (yellow)."""
        metrics = _make_metrics()
        issues = detect_pid_issues("roll", metrics, noise_peaks=[], d_rms=25.0)
        cats = _categories(issues)
        assert "D_HIGH" in cats
        d_high = [i for i in issues if i.category == "D_HIGH"][0]
        assert d_high.severity == "yellow"

    def test_overshoot_plus_slow_settling_yields_d_low(self):
        """overshoot > 10% AND settling_time > 100ms -> D_LOW."""
        metrics = _make_metrics(overshoot_pct=15.0, settling_time_ms=120.0)
        issues = detect_pid_issues("pitch", metrics, noise_peaks=[], d_rms=5.0)
        assert "D_LOW" in _categories(issues)

    def test_high_steady_state_error_yields_i_low(self):
        """steady_state_error > 0.1 -> I_LOW."""
        metrics = _make_metrics(steady_state_error=0.15)
        issues = detect_pid_issues("roll", metrics, noise_peaks=[], d_rms=5.0)
        assert "I_LOW" in _categories(issues)

    def test_good_metrics_yields_good(self):
        """Perfectly tuned metrics should produce only GOOD."""
        metrics = _make_metrics(
            rise_time_ms=25.0,
            overshoot_pct=4.0,
            settling_time_ms=40.0,
            peak_time_ms=15.0,
            steady_state_error=0.01,
        )
        issues = detect_pid_issues("roll", metrics, noise_peaks=[], d_rms=5.0)
        cats = _categories(issues)
        assert cats == {"GOOD"}

    def test_many_noise_peaks_yields_noise_high(self):
        """More than 3 strong peaks above -10dB -> NOISE_HIGH."""
        peaks = [
            (200.0, -5.0),
            (300.0, -8.0),
            (400.0, -3.0),
            (500.0, -7.0),
        ]
        metrics = _make_metrics()
        issues = detect_pid_issues("roll", metrics, noise_peaks=peaks, d_rms=5.0)
        assert "NOISE_HIGH" in _categories(issues)

    def test_filter_lag_detected(self):
        """rise_time > 50ms with overshoot < 3% -> FILTER_LAG."""
        metrics = _make_metrics(rise_time_ms=55.0, overshoot_pct=2.0)
        issues = detect_pid_issues("roll", metrics, noise_peaks=[], d_rms=5.0)
        assert "FILTER_LAG" in _categories(issues)


# ── compute_dterm_rms ────────────────────────────────────────────────────────

class TestComputeDtermRms:
    """Tests for compute_dterm_rms()."""

    def test_known_rms(self):
        """RMS of [3, 4] should be sqrt((9+16)/2) = sqrt(12.5) ~ 3.5355."""
        d_term = np.array([3.0, 4.0])
        result = compute_dterm_rms(d_term)
        assert pytest.approx(result, rel=1e-3) == np.sqrt(12.5)

    def test_zero_signal(self):
        """RMS of all zeros is 0."""
        d_term = np.zeros(100)
        assert compute_dterm_rms(d_term) == 0.0

    def test_constant_signal(self):
        """RMS of constant signal equals the absolute value of that constant."""
        d_term = np.full(50, 7.0)
        assert pytest.approx(compute_dterm_rms(d_term), rel=1e-6) == 7.0


# ── estimate_motor_heat ──────────────────────────────────────────────────────

class TestEstimateMotorHeat:
    """Tests for estimate_motor_heat()."""

    def test_returns_dict_with_four_entries(self):
        """Output dict should have keys 0-3, each value in [0, 1]."""
        np.random.seed(42)
        motors = np.random.randn(4, 4000) * 2.0  # low noise
        d_terms = None  # d_terms param not used directly per spec
        result = estimate_motor_heat(d_terms, motors, sample_rate=4000)
        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1, 2, 3}
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_high_noise_yields_high_heat(self):
        """Motors with strong high-frequency noise -> heat > 0.5."""
        np.random.seed(42)
        sample_rate = 4000
        n = sample_rate * 2  # 2 seconds
        t = np.arange(n) / sample_rate
        # Loud 200Hz + 300Hz sinusoidal noise on top of baseline
        noise = 40.0 * np.sin(2 * np.pi * 200 * t) + 30.0 * np.sin(2 * np.pi * 300 * t)
        motors = np.tile(noise, (4, 1))

        result = estimate_motor_heat(None, motors, sample_rate=sample_rate)
        for v in result.values():
            assert v > 0.5, f"Expected heat > 0.5, got {v}"

    def test_quiet_motors_yield_low_heat(self):
        """Motors with only low-frequency content -> heat < 0.3."""
        np.random.seed(42)
        sample_rate = 4000
        n = sample_rate * 2
        t = np.arange(n) / sample_rate
        # Pure low-frequency content (10Hz) -- below 100Hz highpass
        slow = 50.0 * np.sin(2 * np.pi * 10 * t)
        motors = np.tile(slow, (4, 1))

        result = estimate_motor_heat(None, motors, sample_rate=sample_rate)
        for v in result.values():
            assert v < 0.3, f"Expected heat < 0.3, got {v}"
