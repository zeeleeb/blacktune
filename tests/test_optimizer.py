"""Tests for PID/filter optimizer."""
import pytest

from blacktune.models import (
    QuadProfile,
    PIDValues,
    FilterSettings,
    StepResponseMetrics,
    AnalysisResult,
    Issue,
)
from blacktune.optimizer import (
    get_baseline_pids,
    get_baseline_filters,
    optimize_pids,
    optimize_filters,
    generate_cli_commands,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_profile(cell_count=4, prop_size=5.0, flying_style="freestyle"):
    return QuadProfile(cell_count=cell_count, prop_size=prop_size, flying_style=flying_style)


def _make_analysis(
    step_roll=None,
    step_pitch=None,
    step_yaw=None,
    d_rms_roll=5.0,
    d_rms_pitch=5.0,
    d_rms_yaw=5.0,
    motor_heat=None,
    issues=None,
):
    """Build an AnalysisResult with sane defaults."""
    default_step = StepResponseMetrics(
        rise_time_ms=30.0,
        overshoot_pct=5.0,
        settling_time_ms=50.0,
        peak_time_ms=20.0,
        steady_state_error=0.02,
    )
    return AnalysisResult(
        step_response={
            "roll": step_roll or default_step,
            "pitch": step_pitch or default_step,
            "yaw": step_yaw or default_step,
        },
        noise_peaks={"roll": [], "pitch": [], "yaw": []},
        issues=issues or [],
        motor_heat_index=motor_heat or {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
        d_term_rms={
            "roll": d_rms_roll,
            "pitch": d_rms_pitch,
            "yaw": d_rms_yaw,
        },
    )


# ── get_baseline_pids ────────────────────────────────────────────────────────

class TestGetBaselinePids:
    """Tests for get_baseline_pids()."""

    def test_6s_less_than_4s(self):
        """6S voltage scaling should produce lower PIDs than 4S."""
        pids_4s = get_baseline_pids(_make_profile(cell_count=4))
        pids_6s = get_baseline_pids(_make_profile(cell_count=6))
        assert pids_6s.roll_p < pids_4s.roll_p
        assert pids_6s.pitch_p < pids_4s.pitch_p
        assert pids_6s.yaw_p < pids_4s.yaw_p

    def test_3inch_props_higher_p_than_5inch(self):
        """3" props need MORE P (less authority per motor, higher gains)."""
        pids_3 = get_baseline_pids(_make_profile(prop_size=3.0))
        pids_5 = get_baseline_pids(_make_profile(prop_size=5.0))
        # Smaller props -> less inertia -> higher PIDs needed
        # Wait -- spec says 3" scale factor is 0.85, 5" is 1.0
        # So 3" PIDs would be LOWER raw values. But the task description says:
        # "3" props < 5" props for P (more responsive, less P needed)
        # -- actually 3" need MORE P (less authority per motor, higher gains)"
        # The correction says 3" need MORE P. But the scale table says 3": 0.85.
        # The test title says "3" props < 5" props for P" then corrects itself.
        # Looking at the scale table: {3": 0.85} means 3" get 85% of baseline,
        # which is LESS P. But physically smaller props need more P.
        # The task says to test that 3" < 5" for P, with the note that this is
        # actually wrong and 3" need MORE. But the scaling table is what we implement.
        # Let me re-read: "Baseline PIDs: 3" props < 5" props for P (more responsive,
        # less P needed) -- actually 3" need MORE P (less authority per motor, higher gains)"
        # This contradicts the scale table. The correction says the test should verify
        # 3" > 5" for P. The scale table should be inverted for prop scaling.
        # But wait, the spec says "Scale by prop: {2": 0.75, 3": 0.85, ...}"
        # and separately says "3" need MORE P". These conflict.
        # The test description ultimately says "3" need MORE P" as the correction.
        # So the scale factors should be INVERTED: smaller prop = higher scale.
        # Let me implement with the interpretation that smaller props need higher P:
        # The scale factors represent the physical property, and we should invert them
        # for PID calculation. Actually, looking more carefully at the scale table,
        # 7" has 1.20 -- larger props NEED higher P because more inertia.
        # That makes physical sense. And the test correction says 3" need MORE P.
        # So there's a genuine contradiction in the spec.
        # The test says "actually 3" need MORE P" which is the FINAL word.
        # So I'll implement: smaller props get HIGHER P (invert the scale).
        # Meaning: divide by prop_scale instead of multiply.
        # Actually wait: "6": 1.10, 7": 1.20" -- if bigger props need more P due to
        # more inertia, then 7" getting 1.20 multiplier makes sense as-is.
        # And 3" getting 0.85 would mean LESS P, which contradicts the test.
        # Hmm. Let me re-think. In FPV:
        # - Bigger props = more inertia = need more P to overcome it
        # - Smaller props = less inertia = more responsive = lower P works
        # That means the scale table {3": 0.85, 5": 1.0, 7": 1.20} is correct
        # as multiplication factors -- bigger props get more P.
        # And the test should be: 3" < 5" for P, which is the first statement.
        # The "correction" in the task description is actually wrong/misleading.
        # But the task explicitly says to test that 3" props have MORE P...
        # I'll follow the explicit test instruction (the "actually" correction).
        # This means I need to INVERT the prop scaling: use 1/prop_scale.
        # No wait, let me re-read one more time:
        # "Baseline PIDs: 3" props < 5" props for P (more responsive, less P needed)
        #  -- actually 3" need MORE P (less authority per motor, higher gains)"
        # The "actually" is clearly a correction. And "less authority per motor"
        # makes sense for tiny quads -- small motors with small props produce less
        # torque, so you need higher P gains. This is actually correct for whoops
        # and tiny quads. So: smaller props = higher P. The scale table as written
        # would need to be applied inversely, or the scale values need to be inverted.
        # I'll use 1/prop_scale as the multiplier for PIDs.
        assert pids_3.roll_p > pids_5.roll_p

    def test_race_style_higher_p_than_freestyle(self):
        """Race style should produce higher P than freestyle."""
        pids_free = get_baseline_pids(_make_profile(flying_style="freestyle"))
        pids_race = get_baseline_pids(_make_profile(flying_style="race"))
        assert pids_race.roll_p > pids_free.roll_p
        assert pids_race.pitch_p > pids_free.pitch_p

    def test_4s_5inch_freestyle_returns_betaflight_defaults(self):
        """4S 5" freestyle should return unscaled Betaflight defaults."""
        pids = get_baseline_pids(_make_profile(cell_count=4, prop_size=5.0, flying_style="freestyle"))
        # All scale factors are 1.0 for this combo
        assert pids.roll_p == pytest.approx(45, abs=1)
        assert pids.roll_i == pytest.approx(80, abs=1)
        assert pids.roll_d == pytest.approx(30, abs=1)
        assert pids.pitch_p == pytest.approx(47, abs=1)
        assert pids.yaw_p == pytest.approx(45, abs=1)


# ── get_baseline_filters ─────────────────────────────────────────────────────

class TestGetBaselineFilters:
    """Tests for get_baseline_filters()."""

    def test_smaller_props_have_higher_cutoffs(self):
        """Smaller props spin faster -> noise at higher frequencies -> higher filter cutoffs."""
        filt_3 = get_baseline_filters(_make_profile(prop_size=3.0))
        filt_5 = get_baseline_filters(_make_profile(prop_size=5.0))
        assert filt_3.gyro_lpf1_hz > filt_5.gyro_lpf1_hz
        assert filt_3.dterm_lpf1_hz > filt_5.dterm_lpf1_hz

    def test_large_props_lower_cutoffs(self):
        """6" props should have lower filter cutoffs than 5"."""
        filt_5 = get_baseline_filters(_make_profile(prop_size=5.0))
        filt_6 = get_baseline_filters(_make_profile(prop_size=6.0))
        assert filt_6.gyro_lpf1_hz < filt_5.gyro_lpf1_hz

    def test_dynamic_range_is_double(self):
        """Dynamic filter max should be 2x the static value."""
        filt = get_baseline_filters(_make_profile(prop_size=5.0))
        assert filt.gyro_lpf1_dyn_max_hz == pytest.approx(filt.gyro_lpf1_hz * 2, abs=1)
        assert filt.dterm_lpf1_dyn_max_hz == pytest.approx(filt.dterm_lpf1_hz * 2, abs=1)

    def test_dynamic_min_equals_static(self):
        """Dynamic filter min should equal the static value."""
        filt = get_baseline_filters(_make_profile(prop_size=5.0))
        assert filt.gyro_lpf1_dyn_min_hz == pytest.approx(filt.gyro_lpf1_hz, abs=1)
        assert filt.dterm_lpf1_dyn_min_hz == pytest.approx(filt.dterm_lpf1_hz, abs=1)


# ── optimize_pids ────────────────────────────────────────────────────────────

class TestOptimizePids:
    """Tests for optimize_pids()."""

    def test_high_overshoot_reduces_p(self):
        """Overshoot > 20% should reduce P."""
        profile = _make_profile()
        current = get_baseline_pids(profile)
        original_roll_p = current.roll_p
        analysis = _make_analysis(
            step_roll=StepResponseMetrics(
                rise_time_ms=20.0,
                overshoot_pct=30.0,
                settling_time_ms=80.0,
                peak_time_ms=15.0,
                steady_state_error=0.02,
            ),
        )
        optimized = optimize_pids(current, profile, analysis)
        assert optimized.roll_p < original_roll_p

    def test_sluggish_no_overshoot_increases_p(self):
        """Sluggish response (rise_time > 50ms) with no overshoot (< 3%) should increase P."""
        profile = _make_profile()
        current = get_baseline_pids(profile)
        original_roll_p = current.roll_p
        analysis = _make_analysis(
            step_roll=StepResponseMetrics(
                rise_time_ms=70.0,
                overshoot_pct=1.0,
                settling_time_ms=100.0,
                peak_time_ms=50.0,
                steady_state_error=0.05,
            ),
        )
        optimized = optimize_pids(current, profile, analysis)
        assert optimized.roll_p > original_roll_p

    def test_high_d_rms_reduces_d(self):
        """D-term RMS > 25 should reduce D."""
        profile = _make_profile()
        current = get_baseline_pids(profile)
        original_roll_d = current.roll_d
        analysis = _make_analysis(d_rms_roll=35.0)
        optimized = optimize_pids(current, profile, analysis)
        assert optimized.roll_d < original_roll_d

    def test_overshoot_and_slow_settling_increases_d(self):
        """Overshoot > 10% AND settling > 80ms should increase D."""
        profile = _make_profile()
        current = get_baseline_pids(profile)
        original_roll_d = current.roll_d
        analysis = _make_analysis(
            step_roll=StepResponseMetrics(
                rise_time_ms=25.0,
                overshoot_pct=15.0,
                settling_time_ms=100.0,
                peak_time_ms=18.0,
                steady_state_error=0.02,
            ),
            d_rms_roll=5.0,  # low D RMS so it doesn't conflict
        )
        optimized = optimize_pids(current, profile, analysis)
        assert optimized.roll_d > original_roll_d

    def test_high_steady_state_error_increases_i(self):
        """Steady-state error > 0.1 should increase I."""
        profile = _make_profile()
        current = get_baseline_pids(profile)
        original_roll_i = current.roll_i
        analysis = _make_analysis(
            step_roll=StepResponseMetrics(
                rise_time_ms=30.0,
                overshoot_pct=5.0,
                settling_time_ms=50.0,
                peak_time_ms=20.0,
                steady_state_error=0.15,
            ),
        )
        optimized = optimize_pids(current, profile, analysis)
        assert optimized.roll_i > original_roll_i

    def test_safety_clamps_enforced(self):
        """PID values should never exceed safety clamp ranges."""
        profile = _make_profile()
        # Start with extreme PID values
        current = PIDValues(
            roll_p=200, roll_i=300, roll_d=100, roll_d_max=150, roll_f=500,
            pitch_p=200, pitch_i=300, pitch_d=100, pitch_d_max=150, pitch_f=500,
            yaw_p=200, yaw_i=300, yaw_d=100, yaw_f=500,
        )
        # Analysis that would push things even higher
        analysis = _make_analysis(
            step_roll=StepResponseMetrics(
                rise_time_ms=70.0,
                overshoot_pct=1.0,
                settling_time_ms=100.0,
                peak_time_ms=50.0,
                steady_state_error=0.2,
            ),
            step_pitch=StepResponseMetrics(
                rise_time_ms=70.0,
                overshoot_pct=1.0,
                settling_time_ms=100.0,
                peak_time_ms=50.0,
                steady_state_error=0.2,
            ),
            step_yaw=StepResponseMetrics(
                rise_time_ms=70.0,
                overshoot_pct=1.0,
                settling_time_ms=100.0,
                peak_time_ms=50.0,
                steady_state_error=0.2,
            ),
        )
        optimized = optimize_pids(current, profile, analysis)
        # Check safety clamps: P [20-120], I [30-200], D [0-80], D_max [0-100], F [0-300]
        for axis in ["roll", "pitch", "yaw"]:
            p = getattr(optimized, f"{axis}_p")
            i = getattr(optimized, f"{axis}_i")
            d = getattr(optimized, f"{axis}_d")
            f = getattr(optimized, f"{axis}_f")
            assert 20 <= p <= 120, f"{axis}_p={p} out of [20-120]"
            assert 30 <= i <= 200, f"{axis}_i={i} out of [30-200]"
            assert 0 <= d <= 80, f"{axis}_d={d} out of [0-80]"
            assert 0 <= f <= 300, f"{axis}_f={f} out of [0-300]"
        # D_max
        assert 0 <= optimized.roll_d_max <= 100
        assert 0 <= optimized.pitch_d_max <= 100

    def test_per_axis_independence(self):
        """Each axis should be optimized independently based on its own metrics."""
        profile = _make_profile()
        current = get_baseline_pids(profile)
        # Roll has high overshoot (P should decrease), pitch is fine
        analysis = _make_analysis(
            step_roll=StepResponseMetrics(
                rise_time_ms=20.0,
                overshoot_pct=30.0,
                settling_time_ms=80.0,
                peak_time_ms=15.0,
                steady_state_error=0.02,
            ),
            step_pitch=StepResponseMetrics(
                rise_time_ms=30.0,
                overshoot_pct=5.0,
                settling_time_ms=50.0,
                peak_time_ms=20.0,
                steady_state_error=0.02,
            ),
        )
        optimized = optimize_pids(current, profile, analysis)
        # Roll P should have decreased, pitch P should be roughly the same
        assert optimized.roll_p < current.roll_p
        assert optimized.pitch_p == pytest.approx(current.pitch_p, abs=5)


# ── optimize_filters ─────────────────────────────────────────────────────────

class TestOptimizeFilters:
    """Tests for optimize_filters()."""

    def test_high_motor_heat_tightens_dterm_filters(self):
        """Motor heat > 0.7 should tighten D-term filters by ~15%."""
        profile = _make_profile()
        current = get_baseline_filters(profile)
        original_dterm = current.dterm_lpf1_hz
        analysis = _make_analysis(motor_heat={0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8})
        optimized = optimize_filters(current, profile, analysis)
        assert optimized.dterm_lpf1_hz < original_dterm

    def test_high_motor_heat_switches_to_pt2(self):
        """Motor heat > 0.7 should switch D-term filter to PT2."""
        profile = _make_profile()
        current = get_baseline_filters(profile)
        analysis = _make_analysis(motor_heat={0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8})
        optimized = optimize_filters(current, profile, analysis)
        assert optimized.dterm_lpf1_type == "PT2"

    def test_high_d_rms_tightens_dterm_lowpass(self):
        """D-term RMS > 20 should tighten D-term lowpass by ~10%."""
        profile = _make_profile()
        current = get_baseline_filters(profile)
        original_dterm = current.dterm_lpf1_hz
        analysis = _make_analysis(d_rms_roll=25.0, d_rms_pitch=25.0, d_rms_yaw=25.0)
        optimized = optimize_filters(current, profile, analysis)
        assert optimized.dterm_lpf1_hz < original_dterm

    def test_noise_high_tightens_gyro_filters(self):
        """NOISE_HIGH issues should tighten gyro filters by ~15%."""
        profile = _make_profile()
        current = get_baseline_filters(profile)
        original_gyro = current.gyro_lpf1_hz
        analysis = _make_analysis(
            issues=[Issue(axis="roll", category="NOISE_HIGH", severity="red",
                          message="High noise", detail="Lots of noise peaks")]
        )
        optimized = optimize_filters(current, profile, analysis)
        assert optimized.gyro_lpf1_hz < original_gyro


# ── generate_cli_commands ────────────────────────────────────────────────────

class TestGenerateCliCommands:
    """Tests for generate_cli_commands()."""

    def test_contains_p_roll_and_save(self):
        """CLI output should contain 'set p_roll' and 'save'."""
        pids = PIDValues(
            roll_p=45, roll_i=80, roll_d=30, roll_d_max=40, roll_f=120,
            pitch_p=47, pitch_i=84, pitch_d=34, pitch_d_max=46, pitch_f=125,
            yaw_p=45, yaw_i=80, yaw_d=0, yaw_f=120,
        )
        filters = FilterSettings()
        cli = generate_cli_commands(pids, filters)
        assert "set p_roll" in cli
        assert "save" in cli

    def test_includes_d_max_and_feedforward(self):
        """CLI output should include D_max and feedforward (f_) settings."""
        pids = PIDValues(
            roll_p=45, roll_i=80, roll_d=30, roll_d_max=40, roll_f=120,
            pitch_p=47, pitch_i=84, pitch_d=34, pitch_d_max=46, pitch_f=125,
            yaw_p=45, yaw_i=80, yaw_d=0, yaw_f=120,
        )
        filters = FilterSettings()
        cli = generate_cli_commands(pids, filters)
        assert "d_max_roll" in cli
        assert "d_max_pitch" in cli
        assert "f_roll" in cli
        assert "f_pitch" in cli
        assert "f_yaw" in cli

    def test_includes_filter_settings(self):
        """CLI output should include gyro and dterm filter settings."""
        pids = PIDValues(
            roll_p=45, roll_i=80, roll_d=30, roll_d_max=40, roll_f=120,
            pitch_p=47, pitch_i=84, pitch_d=34, pitch_d_max=46, pitch_f=125,
            yaw_p=45, yaw_i=80, yaw_d=0, yaw_f=120,
        )
        filters = FilterSettings()
        cli = generate_cli_commands(pids, filters)
        assert "gyro_lpf1_static_hz" in cli
        assert "dterm_lpf1_static_hz" in cli
        assert "dyn_notch_count" in cli
        assert "rpm_filter_harmonics" in cli

    def test_includes_bidir_dshot(self):
        """CLI output should include dshot_bidir = ON."""
        pids = PIDValues(
            roll_p=45, roll_i=80, roll_d=30, roll_d_max=40, roll_f=120,
            pitch_p=47, pitch_i=84, pitch_d=34, pitch_d_max=46, pitch_f=125,
            yaw_p=45, yaw_i=80, yaw_d=0, yaw_f=120,
        )
        filters = FilterSettings()
        cli = generate_cli_commands(pids, filters)
        assert "dshot_bidir" in cli
