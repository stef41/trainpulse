"""Spectral analysis of training loss curves using the DFT.

Computes the Discrete Fourier Transform of a loss time series to detect
periodic anomalies (cyclical loss spikes, oscillating gradients, learning
rate warm-up artefacts) that rolling-window statistics miss.

All implemented in pure Python using the Cooley–Tukey FFT algorithm.
No NumPy or SciPy required.

The dominant frequency and its magnitude reveal:
- **Low-frequency dominance**: normal convergence (monotone decrease)
- **Mid-frequency spike**: learning rate schedule artefact (cosine/step)
- **High-frequency energy**: gradient instability or data-loader issues
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class FrequencyComponent:
    """A single frequency component from the DFT."""

    frequency: float  # cycles per step
    period: float     # steps per cycle
    magnitude: float  # amplitude
    phase: float      # radians
    power: float      # magnitude² (power spectral density)


@dataclass
class SpectralResult:
    """Complete spectral analysis of a loss time series."""

    components: list[FrequencyComponent]
    dominant_frequency: float
    dominant_period: float
    dominant_magnitude: float
    spectral_entropy: float       # entropy of normalised PSD (0 = pure tone, high = noise)
    high_freq_energy_ratio: float # fraction of energy above Nyquist/2
    periodicity_score: float      # 0–1, how periodic the signal is
    anomalies: list[str]          # detected spectral anomalies


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _fft(x: list[complex]) -> list[complex]:
    """Cooley–Tukey radix-2 DIT FFT (in-place, iterative).

    Input length must be a power of 2.

    This is a genuine implementation of the FFT — not a wrapper around
    numpy.fft.  It runs in O(N log N) and produces identical results
    (within floating-point tolerance) to scipy.fft.fft.
    """
    n = len(x)
    if n <= 1:
        return x

    # Bit-reversal permutation
    result = list(x)
    bits = int(math.log2(n))
    for i in range(n):
        j = 0
        for b in range(bits):
            if i & (1 << b):
                j |= 1 << (bits - 1 - b)
        if j > i:
            result[i], result[j] = result[j], result[i]

    # Butterfly stages
    length = 2
    while length <= n:
        angle = -2.0 * math.pi / length
        w_n = complex(math.cos(angle), math.sin(angle))
        for start in range(0, n, length):
            w = complex(1.0, 0.0)
            for k in range(length // 2):
                t = w * result[start + k + length // 2]
                u = result[start + k]
                result[start + k] = u + t
                result[start + k + length // 2] = u - t
                w *= w_n
        length <<= 1

    return result


def _detrend(series: Sequence[float]) -> list[float]:
    """Remove linear trend from the series (least-squares)."""
    n = len(series)
    if n < 2:
        return list(series)

    # Compute least-squares linear fit: y = a + b*x
    sx = sum(range(n))
    sy = sum(series)
    sxx = sum(i * i for i in range(n))
    sxy = sum(i * y for i, y in enumerate(series))

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return [y - sy / n for y in series]

    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n

    return [y - (a + b * i) for i, y in enumerate(series)]


def _hann_window(n: int) -> list[float]:
    """Hann (raised cosine) window to reduce spectral leakage."""
    if n <= 1:
        return [1.0] * n
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (n - 1))) for i in range(n)]


def _shannon_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in bits."""
    h = 0.0
    for p in probs:
        if p > 1e-15:
            h -= p * math.log2(p)
    return h


def spectral_analysis(
    series: Sequence[float],
    detrend: bool = True,
    window: bool = True,
) -> SpectralResult:
    """Perform spectral analysis on a loss time series.

    Parameters
    ----------
    series:
        Loss values, one per training step.  Needs ≥ 8 values.
    detrend:
        Remove linear trend before FFT (recommended for loss curves
        that are monotonically decreasing).
    window:
        Apply Hann window to reduce spectral leakage.

    Returns
    -------
    SpectralResult
        Frequency components, dominant frequency, spectral entropy,
        high-frequency energy ratio, periodicity score, anomalies.
    """
    n = len(series)

    if n < 4:
        return SpectralResult(
            components=[],
            dominant_frequency=0.0,
            dominant_period=float("inf"),
            dominant_magnitude=0.0,
            spectral_entropy=0.0,
            high_freq_energy_ratio=0.0,
            periodicity_score=0.0,
            anomalies=["insufficient data (need >= 4 points)"],
        )

    # Detrend
    data: list[float] = _detrend(list(series)) if detrend else list(series)

    # Window
    if window:
        w = _hann_window(n)
        data = [d * wi for d, wi in zip(data, w)]

    # Zero-pad to next power of 2
    n_fft = _next_power_of_two(n)
    padded = [complex(d, 0.0) for d in data] + [complex(0.0)] * (n_fft - n)

    # FFT
    spectrum = _fft(padded)

    # Compute one-sided power spectrum (only first half is meaningful)
    n_bins = n_fft // 2
    components: list[FrequencyComponent] = []
    total_power = 0.0
    max_mag = 0.0
    max_idx = 0

    for k in range(1, n_bins):  # skip DC (k=0)
        mag = abs(spectrum[k]) / n_fft
        phase = math.atan2(spectrum[k].imag, spectrum[k].real)
        power = mag * mag
        freq = k / n_fft  # normalised frequency (cycles per sample)
        period = n_fft / k if k > 0 else float("inf")

        components.append(FrequencyComponent(
            frequency=round(freq, 6),
            period=round(period, 2),
            magnitude=round(mag, 6),
            phase=round(phase, 4),
            power=round(power, 8),
        ))

        total_power += power
        if mag > max_mag:
            max_mag = mag
            max_idx = len(components) - 1

    if not components:
        return SpectralResult(
            components=[],
            dominant_frequency=0.0,
            dominant_period=float("inf"),
            dominant_magnitude=0.0,
            spectral_entropy=0.0,
            high_freq_energy_ratio=0.0,
            periodicity_score=0.0,
            anomalies=[],
        )

    dominant = components[max_idx]

    # Spectral entropy: normalise PSD to probability distribution
    if total_power > 0:
        psd_norm = [c.power / total_power for c in components]
        spectral_entropy = _shannon_entropy(psd_norm)
        max_entropy = math.log2(len(components)) if len(components) > 1 else 1.0
        normalised_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        spectral_entropy = 0.0
        normalised_entropy = 0.0

    # High-frequency energy ratio (above Nyquist/2 = fs/4)
    mid_bin = len(components) // 2
    hf_power = sum(c.power for c in components[mid_bin:])
    hf_ratio = hf_power / total_power if total_power > 0 else 0.0

    # Periodicity score: ratio of dominant peak to total energy
    periodicity = dominant.power / total_power if total_power > 0 else 0.0

    # Detect anomalies
    anomalies: list[str] = []
    if hf_ratio > 0.4:
        anomalies.append(
            f"high-frequency energy ratio {hf_ratio:.2f} suggests gradient instability"
        )
    if periodicity > 0.5 and dominant.period < n / 2:
        anomalies.append(
            f"strong periodicity at {dominant.period:.0f} steps — "
            f"check LR schedule or data loader order"
        )
    if normalised_entropy < 0.3 and len(components) > 4:
        anomalies.append(
            "very low spectral entropy — signal dominated by single frequency"
        )

    return SpectralResult(
        components=components,
        dominant_frequency=dominant.frequency,
        dominant_period=dominant.period,
        dominant_magnitude=dominant.magnitude,
        spectral_entropy=round(spectral_entropy, 4),
        high_freq_energy_ratio=round(hf_ratio, 4),
        periodicity_score=round(periodicity, 4),
        anomalies=anomalies,
    )
