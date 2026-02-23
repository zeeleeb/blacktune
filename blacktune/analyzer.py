"""Full analysis pipeline -- wires all analyzers together.

This is the main entry point for BlackTune analysis.  UI code calls
``run_analysis()`` to get metrics and ``generate_recommendation()`` to
get actionable PID/filter tuning suggestions.
"""
from __future__ import annotations

from .models import (
    FlightLog,
    QuadProfile,
    AnalysisResult,
    TuneRecommendation,
    FilterSettings,
)
from .analyzers.step_response import compute_step_response, measure_step_metrics
from .analyzers.noise import compute_fft_spectrum, find_noise_peaks
from .analyzers.issues import detect_pid_issues, compute_dterm_rms, estimate_motor_heat
from .optimizer import optimize_pids, optimize_filters, generate_cli_commands


def run_analysis(log: FlightLog) -> AnalysisResult:
    """Run the full analysis pipeline on a flight log.

    For each axis (roll, pitch, yaw):
    1. Compute step response via Wiener deconvolution
    2. Measure step response metrics
    3. Compute FFT spectrum
    4. Find noise peaks
    5. Compute D-term RMS
    6. Detect PID issues

    Then:
    7. Estimate motor heat from motor outputs

    Returns AnalysisResult with all results.
    """
    step_response = {}
    noise_peaks = {}
    d_term_rms = {}
    all_issues = []

    for axis_data in (log.roll, log.pitch, log.yaw):
        name = axis_data.name

        # 1. Compute step response via Wiener deconvolution
        resp, resp_time = compute_step_response(
            setpoint=axis_data.setpoint,
            gyro=axis_data.gyro,
            sample_rate=log.sample_rate,
        )

        # 2. Measure step response metrics
        metrics = measure_step_metrics(resp, resp_time)
        step_response[name] = metrics

        # 3. Compute FFT spectrum
        freqs, psd_db = compute_fft_spectrum(
            signal=axis_data.gyro,
            sample_rate=log.sample_rate,
        )

        # 4. Find noise peaks
        peaks = find_noise_peaks(freqs, psd_db)
        noise_peaks[name] = peaks

        # 5. Compute D-term RMS
        d_rms = compute_dterm_rms(axis_data.d_term)
        d_term_rms[name] = d_rms

        # 6. Detect PID issues
        issues = detect_pid_issues(
            axis=name,
            step_metrics=metrics,
            noise_peaks=peaks,
            d_rms=d_rms,
        )
        all_issues.extend(issues)

    # 7. Estimate motor heat from motor outputs
    motor_heat_index = estimate_motor_heat(
        d_terms=None,
        motors=log.motors,
        sample_rate=log.sample_rate,
    )

    return AnalysisResult(
        step_response=step_response,
        noise_peaks=noise_peaks,
        issues=all_issues,
        motor_heat_index=motor_heat_index,
        d_term_rms=d_term_rms,
    )


def generate_recommendation(
    log: FlightLog,
    profile: QuadProfile,
    analysis: AnalysisResult,
) -> TuneRecommendation:
    """Generate PID + filter recommendations.

    1. Optimize PIDs using optimizer.optimize_pids()
    2. Optimize filters using optimizer.optimize_filters()
    3. Generate CLI commands
    4. Calculate confidence score:
       - Higher if more axes are "GOOD"
       - Lower if more "red" severity issues
       - Range: 0.3 to 0.95
    5. Collect explanations from issues

    Returns TuneRecommendation.
    """
    # 1. Optimize PIDs
    current_pids = log.current_pids
    suggested_pids = optimize_pids(current_pids, profile, analysis)

    # 2. Optimize filters
    current_filters = log.current_filters or FilterSettings()
    suggested_filters = optimize_filters(current_filters, profile, analysis)

    # 3. Generate CLI commands
    cli_commands = generate_cli_commands(suggested_pids, suggested_filters)

    # 4. Calculate confidence score
    good_count = sum(1 for iss in analysis.issues if iss.category == "GOOD")
    red_count = sum(1 for iss in analysis.issues if iss.severity == "red")

    # Start at a base and adjust
    # 3 axes -> max 3 GOOD.  Each GOOD adds confidence, each red subtracts.
    confidence = 0.6
    confidence += good_count * 0.1       # up to +0.3 for 3 GOOD axes
    confidence -= red_count * 0.1        # each red issue reduces confidence

    # Clamp to [0.3, 0.95]
    confidence = max(0.3, min(0.95, confidence))

    # 5. Collect explanations from issues
    explanations = {}
    for issue in analysis.issues:
        key = f"{issue.axis}_{issue.category}"
        explanations[key] = issue.detail

    return TuneRecommendation(
        suggested_pids=suggested_pids,
        suggested_filters=suggested_filters,
        confidence=confidence,
        explanations=explanations,
        cli_commands=cli_commands,
    )
