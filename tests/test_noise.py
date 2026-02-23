"""Tests for blacktune.analyzers.noise -- FFT spectrum, spectrogram, peak detection."""
import numpy as np
import pytest
from blacktune.analyzers.noise import compute_fft_spectrum, compute_spectrogram, find_noise_peaks


SAMPLE_RATE = 4000  # Typical blackbox logging rate


def _sine(freq_hz, duration_s=1.0, sample_rate=SAMPLE_RATE, amplitude=1.0):
    """Helper: generate a pure sine wave."""
    t = np.arange(0, duration_s, 1.0 / sample_rate)
    return amplitude * np.sin(2 * np.pi * freq_hz * t), t


# ---------- compute_fft_spectrum ----------

def test_fft_finds_peak():
    """Create 200 Hz sine + noise, verify peak near 200 Hz."""
    sig, _ = _sine(200, duration_s=2.0)
    rng = np.random.default_rng(42)
    sig += rng.normal(0, 0.1, len(sig))

    freqs, psd_db = compute_fft_spectrum(sig, SAMPLE_RATE)

    # Frequency array should cover 0 to Nyquist
    assert freqs[0] == 0.0
    assert freqs[-1] <= SAMPLE_RATE / 2

    # Peak PSD should be near 200 Hz
    peak_idx = np.argmax(psd_db)
    assert abs(freqs[peak_idx] - 200) < 10, f"Peak at {freqs[peak_idx]} Hz, expected ~200 Hz"


def test_fft_finds_multiple_peaks():
    """Create 200 Hz + 400 Hz sines, verify both found."""
    sig200, _ = _sine(200, duration_s=2.0, amplitude=1.0)
    sig400, _ = _sine(400, duration_s=2.0, amplitude=0.8)
    rng = np.random.default_rng(42)
    combined = sig200 + sig400 + rng.normal(0, 0.05, len(sig200))

    freqs, psd_db = compute_fft_spectrum(combined, SAMPLE_RATE)
    peaks = find_noise_peaks(freqs, psd_db, min_prominence_db=3.0)

    peak_freqs = [f for f, _ in peaks]
    assert any(abs(f - 200) < 15 for f in peak_freqs), f"No peak near 200 Hz in {peak_freqs}"
    assert any(abs(f - 400) < 15 for f in peak_freqs), f"No peak near 400 Hz in {peak_freqs}"


# ---------- compute_spectrogram ----------

def test_spectrogram_shape():
    """Verify output shape matches (n_freq, n_throttle_bins)."""
    n_samples = 8000
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 1, n_samples)
    throttle = np.linspace(0, 100, n_samples)  # 0-100%

    n_throttle_bins = 20
    freq_bins, throttle_centers, spec_db = compute_spectrogram(
        signal, throttle, SAMPLE_RATE, n_throttle_bins=n_throttle_bins, nperseg=512,
    )

    assert spec_db.shape[0] == len(freq_bins), "freq axis mismatch"
    assert spec_db.shape[1] == n_throttle_bins, "throttle axis mismatch"
    assert len(throttle_centers) == n_throttle_bins


def test_spectrogram_throttle_normalization():
    """Pass RC values (1000-2000), verify it still works (auto-normalizes)."""
    n_samples = 8000
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 1, n_samples)
    throttle_rc = np.linspace(1000, 2000, n_samples)  # Raw RC

    freq_bins, throttle_centers, spec_db = compute_spectrogram(
        signal, throttle_rc, SAMPLE_RATE, n_throttle_bins=10,
    )

    # Throttle centers should be in 0-100 range after normalization
    assert throttle_centers[0] >= 0
    assert throttle_centers[-1] <= 100
    assert spec_db.shape[1] == 10


# ---------- find_noise_peaks ----------

def test_find_noise_peaks():
    """300 Hz + 600 Hz, verify both peaks found."""
    sig300, _ = _sine(300, duration_s=2.0, amplitude=1.0)
    sig600, _ = _sine(600, duration_s=2.0, amplitude=0.7)
    rng = np.random.default_rng(42)
    combined = sig300 + sig600 + rng.normal(0, 0.05, len(sig300))

    freqs, psd_db = compute_fft_spectrum(combined, SAMPLE_RATE)
    peaks = find_noise_peaks(freqs, psd_db, min_prominence_db=3.0)

    peak_freqs = [f for f, _ in peaks]
    assert any(abs(f - 300) < 15 for f in peak_freqs), f"No peak near 300 Hz in {peak_freqs}"
    assert any(abs(f - 600) < 15 for f in peak_freqs), f"No peak near 600 Hz in {peak_freqs}"

    # Should be sorted by amplitude descending
    amps = [a for _, a in peaks]
    assert amps == sorted(amps, reverse=True), "Peaks not sorted by amplitude descending"


def test_find_peaks_respects_freq_range():
    """Peak at 30 Hz should be excluded when min_freq_hz=50."""
    sig30, _ = _sine(30, duration_s=2.0, amplitude=2.0)
    sig200, _ = _sine(200, duration_s=2.0, amplitude=1.0)
    rng = np.random.default_rng(42)
    combined = sig30 + sig200 + rng.normal(0, 0.05, len(sig30))

    freqs, psd_db = compute_fft_spectrum(combined, SAMPLE_RATE)
    peaks = find_noise_peaks(freqs, psd_db, min_freq_hz=50.0, max_freq_hz=1000.0, min_prominence_db=3.0)

    peak_freqs = [f for f, _ in peaks]
    # 30 Hz peak should be excluded
    assert all(f >= 50.0 for f in peak_freqs), f"Found sub-50Hz peak in {peak_freqs}"
    # 200 Hz peak should be present
    assert any(abs(f - 200) < 15 for f in peak_freqs), f"No peak near 200 Hz in {peak_freqs}"
