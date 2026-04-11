"""Tests for spectral analysis (FFT) of loss curves."""

from __future__ import annotations

import math

import pytest

from trainpulse.spectral import (
    SpectralResult,
    FrequencyComponent,
    spectral_analysis,
    _fft,
    _detrend,
    _hann_window,
    _next_power_of_two,
    _shannon_entropy,
)


class TestFFT:
    def test_single_value(self):
        result = _fft([complex(1.0)])
        assert len(result) == 1
        assert result[0] == pytest.approx(complex(1.0), abs=1e-6)

    def test_two_values(self):
        result = _fft([complex(1.0), complex(0.0)])
        assert len(result) == 2
        assert result[0] == pytest.approx(complex(1.0), abs=1e-6)
        assert result[1] == pytest.approx(complex(1.0), abs=1e-6)

    def test_four_values(self):
        result = _fft([complex(1), complex(0), complex(1), complex(0)])
        assert len(result) == 4
        # DC component should be 2
        assert abs(result[0]) == pytest.approx(2.0, abs=1e-6)

    def test_pure_cosine(self):
        # Generate a pure cosine at frequency 1 (8 samples)
        n = 8
        x = [complex(math.cos(2 * math.pi * i / n)) for i in range(n)]
        result = _fft(x)
        # Peak should be at bin 1 (and bin n-1 for conjugate)
        magnitudes = [abs(r) / n for r in result]
        assert magnitudes[1] > magnitudes[2]
        assert magnitudes[1] > magnitudes[3]

    def test_length_preserved(self):
        x = [complex(i) for i in range(16)]
        result = _fft(x)
        assert len(result) == 16

    def test_parseval_theorem(self):
        # Energy in time domain should equal energy in frequency domain
        x = [complex(1), complex(2), complex(3), complex(4)]
        result = _fft(x)
        time_energy = sum(abs(xi) ** 2 for xi in x)
        freq_energy = sum(abs(Xi) ** 2 for Xi in result) / len(x)
        assert time_energy == pytest.approx(freq_energy, abs=1e-6)


class TestDetrend:
    def test_constant(self):
        result = _detrend([5.0, 5.0, 5.0, 5.0])
        for r in result:
            assert abs(r) < 1e-6

    def test_linear(self):
        result = _detrend([1.0, 2.0, 3.0, 4.0])
        for r in result:
            assert abs(r) < 1e-6

    def test_preserves_oscillation(self):
        # A sine wave on top of a linear trend
        data = [i + math.sin(i) for i in range(20)]
        result = _detrend(data)
        # After detrending, std should be smaller than original
        mean_r = sum(result) / len(result)
        var_r = sum((x - mean_r) ** 2 for x in result) / len(result)
        mean_d = sum(data) / len(data)
        var_d = sum((x - mean_d) ** 2 for x in data) / len(data)
        assert var_r < var_d

    def test_single_value(self):
        result = _detrend([42.0])
        assert len(result) == 1


class TestHannWindow:
    def test_endpoints_zero(self):
        w = _hann_window(10)
        assert w[0] == pytest.approx(0.0, abs=1e-6)
        assert w[-1] == pytest.approx(0.0, abs=1e-6)

    def test_midpoint_one(self):
        w = _hann_window(11)
        assert w[5] == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self):
        w = _hann_window(8)
        for i in range(4):
            assert w[i] == pytest.approx(w[7 - i], abs=1e-6)

    def test_single(self):
        w = _hann_window(1)
        assert w == [1.0]


class TestNextPowerOfTwo:
    def test_exact_power(self):
        assert _next_power_of_two(8) == 8

    def test_non_power(self):
        assert _next_power_of_two(5) == 8

    def test_one(self):
        assert _next_power_of_two(1) == 1


class TestShannonEntropy:
    def test_uniform(self):
        # Uniform over 4 → 2 bits
        probs = [0.25, 0.25, 0.25, 0.25]
        assert _shannon_entropy(probs) == pytest.approx(2.0, abs=1e-6)

    def test_certain(self):
        # All probability on one event → 0 bits
        probs = [1.0, 0.0, 0.0]
        assert _shannon_entropy(probs) == pytest.approx(0.0, abs=1e-6)


class TestSpectralAnalysis:
    def test_returns_result(self):
        series = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3]
        result = spectral_analysis(series)
        assert isinstance(result, SpectralResult)

    def test_monotone_decrease(self):
        # Smooth loss curve — should have low high-freq energy
        series = [1.0 / (1 + 0.1 * i) for i in range(32)]
        result = spectral_analysis(series)
        assert result.high_freq_energy_ratio < 0.3

    def test_oscillating_signal(self):
        # Loss with oscillation — should detect periodicity
        series = [1.0 + 0.5 * math.sin(2 * math.pi * i / 8) for i in range(64)]
        result = spectral_analysis(series)
        assert result.periodicity_score > 0.01

    def test_noisy_signal(self):
        import random
        rng = random.Random(42)
        series = [rng.gauss(0, 1) for _ in range(32)]
        result = spectral_analysis(series)
        assert result.spectral_entropy > 0

    def test_short_series(self):
        result = spectral_analysis([1.0, 0.9])
        assert len(result.anomalies) > 0  # should flag insufficient data

    def test_components_have_fields(self):
        series = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        result = spectral_analysis(series)
        for c in result.components:
            assert isinstance(c, FrequencyComponent)
            assert c.frequency >= 0
            assert c.magnitude >= 0
            assert c.power >= 0

    def test_dominant_frequency_in_components(self):
        series = [math.sin(2 * math.pi * i / 16) for i in range(64)]
        result = spectral_analysis(series, detrend=False)
        assert result.dominant_magnitude > 0
        assert result.dominant_period > 0

    def test_spectral_entropy_range(self):
        series = [1.0 - 0.01 * i + 0.1 * math.sin(i) for i in range(32)]
        result = spectral_analysis(series)
        assert result.spectral_entropy >= 0

    def test_no_detrend(self):
        series = list(range(16))
        result = spectral_analysis(series, detrend=False)
        assert isinstance(result, SpectralResult)

    def test_no_window(self):
        series = [float(i) for i in range(16)]
        result = spectral_analysis(series, window=False)
        assert isinstance(result, SpectralResult)
