"""FFT noise analysis -- power spectral density, spectrogram, and peak detection.

Uses Welch's method for PSD estimation and scipy peak finding.
All amplitudes are in dB scale (10 * log10) matching PIDtoolbox conventions.
"""
import numpy as np
from scipy.signal import welch, find_peaks


def compute_fft_spectrum(signal: np.ndarray, sample_rate: int, nperseg: int = 1024):
    """Compute power spectral density using Welch's method.

    Parameters
    ----------
    signal : 1-D array
        Time-domain signal (e.g. gyro data in deg/s).
    sample_rate : int
        Sampling frequency in Hz.
    nperseg : int
        Segment length for Welch's method (default 1024).

    Returns
    -------
    frequencies_hz : 1-D array
        Frequency bins in Hz (0 to Nyquist).
    psd_db : 1-D array
        Power spectral density in decibels (10 * log10).
    """
    # Clamp nperseg to signal length to avoid scipy error
    nperseg = min(nperseg, len(signal))

    freqs, psd = welch(signal, fs=sample_rate, nperseg=nperseg, scaling="density")

    # Convert to dB; floor at a tiny value to avoid log10(0)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))

    return freqs, psd_db


def compute_spectrogram(
    signal: np.ndarray,
    throttle: np.ndarray,
    sample_rate: int,
    n_throttle_bins: int = 20,
    nperseg: int = 512,
):
    """Compute 2-D spectrogram binned by throttle position.

    Bins the signal by throttle percentage (0-100), computes Welch PSD for
    each bin. Motor noise frequency increases with throttle (higher RPM),
    so this view is essential for FPV tuning.

    Parameters
    ----------
    signal : 1-D array
        Time-domain signal (e.g. gyro data).
    throttle : 1-D array
        Throttle values -- either 0-100 (percent) or 1000-2000 (raw RC).
    sample_rate : int
        Sampling frequency in Hz.
    n_throttle_bins : int
        Number of throttle bins (default 20).
    nperseg : int
        Segment length for Welch's method per bin (default 512).

    Returns
    -------
    freq_bins : 1-D array
        Frequency bins in Hz.
    throttle_bin_centers : 1-D array
        Center of each throttle bin (0-100 scale).
    spec_db : 2-D array, shape (n_freq, n_throttle_bins)
        Power spectral density in dB for each (freq, throttle) cell.
    """
    throttle = np.asarray(throttle, dtype=np.float64)

    # Auto-detect raw RC range (1000-2000) and normalize to 0-100
    if np.nanmax(throttle) > 200:
        throttle = (throttle - 1000.0) / 10.0  # 1000->0, 2000->100
        throttle = np.clip(throttle, 0.0, 100.0)

    # Build throttle bin edges (0 to 100)
    bin_edges = np.linspace(0.0, 100.0, n_throttle_bins + 1)
    throttle_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # We need a reference frequency vector. Compute one from the full signal
    # so the shape is consistent even if a bin is empty.
    seg = min(nperseg, len(signal))
    ref_freqs, _ = welch(signal, fs=sample_rate, nperseg=seg, scaling="density")
    n_freq = len(ref_freqs)

    spec_db = np.full((n_freq, n_throttle_bins), -200.0)  # fill with floor dB

    for i in range(n_throttle_bins):
        mask = (throttle >= bin_edges[i]) & (throttle < bin_edges[i + 1])
        # Include upper edge in last bin
        if i == n_throttle_bins - 1:
            mask |= throttle == bin_edges[i + 1]
        chunk = signal[mask]
        if len(chunk) < seg:
            # Not enough data in this bin; leave at floor
            continue
        f, psd = welch(chunk, fs=sample_rate, nperseg=seg, scaling="density")
        spec_db[:, i] = 10.0 * np.log10(np.maximum(psd, 1e-20))

    return ref_freqs, throttle_bin_centers, spec_db


def find_noise_peaks(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    min_prominence_db: float = 6.0,
    min_freq_hz: float = 50.0,
    max_freq_hz: float = 1000.0,
):
    """Find significant noise peaks in a power spectrum.

    Parameters
    ----------
    freqs : 1-D array
        Frequency bins in Hz (from compute_fft_spectrum).
    psd_db : 1-D array
        PSD in dB (from compute_fft_spectrum).
    min_prominence_db : float
        Minimum peak prominence in dB to be considered significant.
    min_freq_hz : float
        Ignore peaks below this frequency.
    max_freq_hz : float
        Ignore peaks above this frequency.

    Returns
    -------
    peaks : list of (frequency_hz, amplitude_db)
        Detected peaks sorted by amplitude descending.
    """
    # Restrict search to the requested frequency band
    freq_mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    if not np.any(freq_mask):
        return []

    # Extract the sub-spectrum
    sub_freqs = freqs[freq_mask]
    sub_psd = psd_db[freq_mask]

    indices, properties = find_peaks(sub_psd, prominence=min_prominence_db)

    if len(indices) == 0:
        return []

    results = [(float(sub_freqs[idx]), float(sub_psd[idx])) for idx in indices]

    # Sort by amplitude descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results
