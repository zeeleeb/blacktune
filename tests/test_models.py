# tests/test_models.py
import numpy as np
from blacktune.models import (
    FlightLog, QuadProfile, PIDValues, FilterSettings,
    AxisData, AnalysisResult, TuneRecommendation, Issue,
    StepResponseMetrics,
)


def test_quad_profile_defaults():
    qp = QuadProfile(cell_count=6, prop_size=5.0)
    assert qp.cell_count == 6
    assert qp.prop_size == 5.0
    assert qp.frame_size is None
    assert qp.flying_style == "freestyle"


def test_pid_values_with_feedforward():
    pids = PIDValues(
        roll_p=45, roll_i=80, roll_d=30, roll_d_max=40, roll_f=120,
        pitch_p=47, pitch_i=84, pitch_d=34, pitch_d_max=46, pitch_f=125,
        yaw_p=45, yaw_i=80, yaw_d=0, yaw_f=120,
    )
    assert pids.roll_p == 45
    assert pids.roll_f == 120
    assert pids.pitch_d_max == 46
    d = pids.as_dict()
    assert d["roll_p"] == 45
    assert d["roll_f"] == 120


def test_pid_values_minimal():
    """PIDValues with only required fields uses sensible defaults."""
    pids = PIDValues(roll_p=45, roll_i=80, roll_d=30)
    assert pids.roll_d_max == 0
    assert pids.roll_f == 0
    assert pids.pitch_p == 0
    assert pids.yaw_d == 0


def test_pid_values_as_dict_roundtrip():
    """as_dict returns all fields as a plain dict."""
    pids = PIDValues(roll_p=45, roll_i=80, roll_d=30, yaw_p=40)
    d = pids.as_dict()
    assert isinstance(d, dict)
    assert d["yaw_p"] == 40
    assert "roll_d_max" in d


def test_filter_settings_dyn_notch_q_is_int():
    fs = FilterSettings()
    assert isinstance(fs.dyn_notch_q, int)
    assert fs.dyn_notch_q == 300


def test_filter_settings_defaults():
    fs = FilterSettings()
    assert fs.gyro_lpf1_hz == 250
    assert fs.dterm_lpf1_hz == 75
    assert fs.dterm_lpf1_dyn_max_hz == 150
    assert fs.rpm_harmonics == 3
    assert fs.dyn_notch_min_hz == 150


def test_filter_settings_custom():
    fs = FilterSettings(gyro_lpf1_hz=300, dterm_lpf1_hz=100, rpm_harmonics=5)
    assert fs.gyro_lpf1_hz == 300
    assert fs.dterm_lpf1_hz == 100
    assert fs.rpm_harmonics == 5
    # Other defaults remain
    assert fs.gyro_lpf2_hz == 500


def test_axis_data():
    t = np.linspace(0, 1, 1000)
    gyro = np.sin(2 * np.pi * 50 * t)
    ad = AxisData(name="roll", gyro=gyro, setpoint=gyro * 0.9,
                  p_term=gyro * 0.5, i_term=gyro * 0.1,
                  d_term=gyro * 0.05, time=t)
    assert ad.name == "roll"
    assert len(ad.gyro) == 1000
    assert ad.f_term is None
    assert ad.gyro_unfiltered is None


def test_axis_data_with_optional_fields():
    t = np.linspace(0, 1, 500)
    gyro = np.zeros(500)
    ad = AxisData(name="pitch", gyro=gyro, setpoint=gyro,
                  p_term=gyro, i_term=gyro, d_term=gyro, time=t,
                  f_term=gyro * 0.1, gyro_unfiltered=gyro * 1.1)
    assert ad.f_term is not None
    assert ad.gyro_unfiltered is not None
    assert len(ad.f_term) == 500


def test_flight_log():
    t = np.linspace(0, 1, 1000)
    zero = np.zeros(1000)
    ad = AxisData(name="roll", gyro=zero, setpoint=zero,
                  p_term=zero, i_term=zero, d_term=zero, time=t)
    pids = PIDValues(45, 80, 30, roll_d_max=40, roll_f=120,
                     pitch_p=47, pitch_i=84, pitch_d=34, pitch_d_max=46, pitch_f=125,
                     yaw_p=45, yaw_i=80, yaw_d=0, yaw_f=120)
    log = FlightLog(
        roll=ad, pitch=ad, yaw=ad,
        throttle=zero, motors=np.zeros((4, 1000)),
        sample_rate=1000, duration_s=1.0,
        firmware="Betaflight 4.4.0",
        current_pids=pids,
    )
    assert log.sample_rate == 1000
    assert log.firmware == "Betaflight 4.4.0"
    assert log.erpm is None
    assert log.current_filters is None


def test_flight_log_with_filters_and_erpm():
    t = np.linspace(0, 1, 1000)
    zero = np.zeros(1000)
    ad = AxisData(name="roll", gyro=zero, setpoint=zero,
                  p_term=zero, i_term=zero, d_term=zero, time=t)
    pids = PIDValues(roll_p=45, roll_i=80, roll_d=30)
    fs = FilterSettings()
    erpm = np.zeros((4, 1000))
    log = FlightLog(
        roll=ad, pitch=ad, yaw=ad,
        throttle=zero, motors=np.zeros((4, 1000)),
        sample_rate=2000, duration_s=1.0,
        firmware="Betaflight 4.5.0",
        current_pids=pids,
        current_filters=fs,
        erpm=erpm,
    )
    assert log.current_filters is not None
    assert log.current_filters.gyro_lpf1_hz == 250
    assert log.erpm is not None
    assert log.erpm.shape == (4, 1000)


def test_issue():
    issue = Issue(axis="roll", category="P_HIGH", severity="red",
                  message="P too high", detail="Overshoot 25%")
    assert issue.severity == "red"


def test_issue_all_fields():
    issue = Issue(axis="yaw", category="GOOD", severity="green",
                  message="Yaw looks great", detail="No issues detected")
    assert issue.axis == "yaw"
    assert issue.category == "GOOD"
    assert issue.message == "Yaw looks great"


def test_step_response_metrics():
    m = StepResponseMetrics(rise_time_ms=25, overshoot_pct=12,
                            settling_time_ms=80, peak_time_ms=30,
                            steady_state_error=0.02)
    assert m.overshoot_pct == 12


def test_step_response_metrics_all_fields():
    m = StepResponseMetrics(rise_time_ms=15.5, overshoot_pct=5.2,
                            settling_time_ms=60.0, peak_time_ms=20.0,
                            steady_state_error=0.005)
    assert m.rise_time_ms == 15.5
    assert m.settling_time_ms == 60.0
    assert m.steady_state_error == 0.005


def test_analysis_result():
    ar = AnalysisResult(
        step_response={"roll": StepResponseMetrics(25, 12, 80, 30, 0.02)},
        noise_peaks={"roll": [(200, -15)]},
        issues=[],
        motor_heat_index={0: 0.3, 1: 0.2, 2: 0.3, 3: 0.2},
        d_term_rms={"roll": 10.5},
    )
    assert ar.filter_delay_ms is None
    assert len(ar.motor_heat_index) == 4


def test_analysis_result_with_issues():
    issue = Issue(axis="roll", category="D_HIGH", severity="yellow",
                  message="D term noisy", detail="RMS 15.2")
    ar = AnalysisResult(
        step_response={},
        noise_peaks={},
        issues=[issue],
        motor_heat_index={},
        d_term_rms={"roll": 15.2},
        filter_delay_ms=2.5,
    )
    assert len(ar.issues) == 1
    assert ar.issues[0].category == "D_HIGH"
    assert ar.filter_delay_ms == 2.5


def test_tune_recommendation():
    pids = PIDValues(roll_p=48, roll_i=85, roll_d=33)
    fs = FilterSettings(gyro_lpf1_hz=275)
    rec = TuneRecommendation(
        suggested_pids=pids,
        suggested_filters=fs,
        confidence=0.85,
        explanations={"roll_p": "Increased by 3 to reduce settling time"},
        cli_commands="set p_roll = 48\nset i_roll = 85\nset d_roll = 33",
    )
    assert rec.confidence == 0.85
    assert "roll_p" in rec.explanations
    assert "set p_roll = 48" in rec.cli_commands
    assert rec.suggested_filters.gyro_lpf1_hz == 275
