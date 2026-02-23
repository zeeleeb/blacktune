"""Tests for the full analysis pipeline (analyzer.py)."""
import numpy as np
import pytest

from blacktune.models import (
    AxisData,
    FlightLog,
    PIDValues,
    QuadProfile,
    AnalysisResult,
    TuneRecommendation,
    StepResponseMetrics,
    FilterSettings,
)
from blacktune.analyzer import run_analysis, generate_recommendation


def _make_flight_log(n=4000, sr=2000):
    """Generate a synthetic flight log for testing."""
    t = np.linspace(0, n / sr, n)
    noise = np.random.randn(n) * 2

    def _axis(name):
        sp = np.zeros(n)
        sp[500:1000] = 200    # step input 1
        sp[2000:2500] = -150  # step input 2
        gyro = sp + noise
        return AxisData(
            name=name, gyro=gyro, setpoint=sp,
            p_term=gyro * 0.3, i_term=gyro * 0.05,
            d_term=noise * 5, time=t,
        )

    return FlightLog(
        roll=_axis("roll"), pitch=_axis("pitch"), yaw=_axis("yaw"),
        throttle=np.linspace(1000, 1800, n),
        motors=np.random.randn(4, n) * 50 + 1400,
        sample_rate=sr, duration_s=n / sr,
        firmware="Betaflight 4.4",
        current_pids=PIDValues(
            50, 80, 30, roll_d_max=40, roll_f=120,
            pitch_p=52, pitch_i=84, pitch_d=34, pitch_d_max=46, pitch_f=125,
            yaw_p=45, yaw_i=80, yaw_d=0, yaw_f=120,
        ),
    )


class TestRunAnalysis:
    """Tests for run_analysis()."""

    def test_returns_analysis_result(self):
        """run_analysis returns an AnalysisResult instance."""
        log = _make_flight_log()
        result = run_analysis(log)
        assert isinstance(result, AnalysisResult)

    def test_step_response_has_all_axes(self):
        """Step response dict contains entries for roll, pitch, yaw."""
        log = _make_flight_log()
        result = run_analysis(log)
        assert "roll" in result.step_response
        assert "pitch" in result.step_response
        assert "yaw" in result.step_response
        for axis in ("roll", "pitch", "yaw"):
            assert isinstance(result.step_response[axis], StepResponseMetrics)

    def test_issues_non_empty(self):
        """run_analysis returns a non-empty issues list."""
        log = _make_flight_log()
        result = run_analysis(log)
        assert len(result.issues) > 0

    def test_motor_heat_index_has_4_entries(self):
        """motor_heat_index should have one entry per motor (4)."""
        log = _make_flight_log()
        result = run_analysis(log)
        assert len(result.motor_heat_index) == 4
        for idx in range(4):
            assert idx in result.motor_heat_index
            assert 0.0 <= result.motor_heat_index[idx] <= 1.0

    def test_d_term_rms_has_3_entries(self):
        """d_term_rms should have one entry per axis (3)."""
        log = _make_flight_log()
        result = run_analysis(log)
        assert len(result.d_term_rms) == 3
        for axis in ("roll", "pitch", "yaw"):
            assert axis in result.d_term_rms
            assert result.d_term_rms[axis] >= 0.0


class TestGenerateRecommendation:
    """Tests for generate_recommendation()."""

    def test_returns_tune_recommendation(self):
        """generate_recommendation returns a TuneRecommendation with valid fields."""
        log = _make_flight_log()
        profile = QuadProfile(cell_count=4, prop_size=5.0, flying_style="freestyle")
        analysis = run_analysis(log)
        rec = generate_recommendation(log, profile, analysis)
        assert isinstance(rec, TuneRecommendation)
        assert isinstance(rec.suggested_pids, PIDValues)
        assert isinstance(rec.suggested_filters, FilterSettings)
        assert isinstance(rec.cli_commands, str)
        assert isinstance(rec.explanations, dict)

    def test_cli_commands_contain_save(self):
        """CLI commands string should end with 'save'."""
        log = _make_flight_log()
        profile = QuadProfile(cell_count=4, prop_size=5.0, flying_style="freestyle")
        analysis = run_analysis(log)
        rec = generate_recommendation(log, profile, analysis)
        assert "save" in rec.cli_commands

    def test_confidence_in_range(self):
        """Confidence score should be between 0.3 and 0.95."""
        log = _make_flight_log()
        profile = QuadProfile(cell_count=4, prop_size=5.0, flying_style="freestyle")
        analysis = run_analysis(log)
        rec = generate_recommendation(log, profile, analysis)
        assert 0.3 <= rec.confidence <= 0.95
