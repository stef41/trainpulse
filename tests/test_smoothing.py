"""Tests for trainpulse.smoothing module."""

from __future__ import annotations

import math

import pytest

from trainpulse.smoothing import (
    MetricSmoother,
    SmoothedSeries,
    SmoothingConfig,
    compare_methods,
    format_smoothing_report,
)


# ------------------------------------------------------------------
# SmoothingConfig / SmoothedSeries dataclasses
# ------------------------------------------------------------------


class TestDataclasses:
    def test_config_defaults(self):
        c = SmoothingConfig()
        assert c.method == "ema"
        assert c.window_size == 10
        assert c.alpha == 0.3

    def test_series_defaults(self):
        s = SmoothedSeries()
        assert s.original == []
        assert s.smoothed == []


# ------------------------------------------------------------------
# EMA
# ------------------------------------------------------------------


class TestEMA:
    def test_ema_single(self):
        s = MetricSmoother()
        assert s.ema([5.0]) == [5.0]

    def test_ema_constant(self):
        vals = [3.0] * 10
        result = MetricSmoother().ema(vals)
        assert all(pytest.approx(v, abs=1e-9) == 3.0 for v in result)

    def test_ema_direction(self):
        vals = [0.0, 10.0, 10.0, 10.0, 10.0]
        result = MetricSmoother(SmoothingConfig(alpha=0.5)).ema(vals, alpha=0.5)
        # Should increase monotonically towards 10
        for a, b in zip(result, result[1:]):
            assert b >= a

    def test_ema_empty(self):
        assert MetricSmoother().ema([]) == []

    def test_ema_custom_alpha(self):
        vals = [0.0, 1.0]
        result = MetricSmoother().ema(vals, alpha=1.0)
        assert result == [0.0, 1.0]


# ------------------------------------------------------------------
# SMA
# ------------------------------------------------------------------


class TestSMA:
    def test_sma_window_1(self):
        vals = [1.0, 2.0, 3.0]
        result = MetricSmoother().sma(vals, window=1)
        assert result == pytest.approx(vals)

    def test_sma_constant(self):
        vals = [5.0] * 8
        result = MetricSmoother().sma(vals, window=3)
        assert all(pytest.approx(v) == 5.0 for v in result)

    def test_sma_empty(self):
        assert MetricSmoother().sma([]) == []

    def test_sma_length_preserved(self):
        vals = list(range(20))
        result = MetricSmoother().sma([float(v) for v in vals], window=5)
        assert len(result) == len(vals)


# ------------------------------------------------------------------
# Gaussian
# ------------------------------------------------------------------


class TestGaussian:
    def test_gaussian_empty(self):
        assert MetricSmoother().gaussian([]) == []

    def test_gaussian_single(self):
        assert MetricSmoother().gaussian([7.0]) == pytest.approx([7.0])

    def test_gaussian_smooths(self):
        # Spike should be reduced
        vals = [1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
        result = MetricSmoother().gaussian(vals, sigma=1.0)
        assert result[3] < 100.0


# ------------------------------------------------------------------
# Median filter
# ------------------------------------------------------------------


class TestMedianFilter:
    def test_median_empty(self):
        assert MetricSmoother().median_filter([]) == []

    def test_median_removes_spike(self):
        vals = [1.0, 1.0, 1.0, 999.0, 1.0, 1.0, 1.0]
        result = MetricSmoother().median_filter(vals, window=3)
        assert result[3] == 1.0

    def test_median_length_preserved(self):
        vals = [float(i) for i in range(15)]
        assert len(MetricSmoother().median_filter(vals)) == 15


# ------------------------------------------------------------------
# Outlier detection
# ------------------------------------------------------------------


class TestOutlierDetection:
    def test_no_outliers(self):
        vals = [1.0, 1.1, 0.9, 1.0, 1.05]
        assert MetricSmoother().detect_outliers(vals, threshold=3.0) == []

    def test_detects_outlier(self):
        vals = [1.0] * 20 + [100.0]
        idx = MetricSmoother().detect_outliers(vals, threshold=2.0)
        assert 20 in idx

    def test_empty(self):
        assert MetricSmoother().detect_outliers([]) == []

    def test_single(self):
        assert MetricSmoother().detect_outliers([5.0]) == []


# ------------------------------------------------------------------
# Denoise
# ------------------------------------------------------------------


class TestDenoise:
    def test_denoise_empty(self):
        assert MetricSmoother().denoise([]) == []

    def test_denoise_reduces_noise(self):
        vals = [1.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0]
        result = MetricSmoother().denoise(vals)
        assert max(result) < 50.0


# ------------------------------------------------------------------
# smooth() dispatcher
# ------------------------------------------------------------------


class TestSmoothDispatch:
    def test_dispatch_ema(self):
        s = MetricSmoother(SmoothingConfig(method="ema"))
        series = s.smooth([1.0, 2.0, 3.0])
        assert series.method == "ema"
        assert len(series.residuals) == 3

    def test_dispatch_sma(self):
        series = MetricSmoother(SmoothingConfig(method="sma")).smooth([1.0, 2.0])
        assert series.method == "sma"

    def test_dispatch_unknown(self):
        with pytest.raises(ValueError, match="Unknown"):
            MetricSmoother(SmoothingConfig(method="bogus")).smooth([1.0])


# ------------------------------------------------------------------
# compare_methods & format_smoothing_report
# ------------------------------------------------------------------


class TestHelpers:
    def test_compare_methods_default(self):
        vals = [float(i) for i in range(10)]
        results = compare_methods(vals)
        assert set(results.keys()) == {"ema", "sma", "gaussian", "median"}

    def test_compare_methods_subset(self):
        results = compare_methods([1.0, 2.0], methods=["ema"])
        assert list(results.keys()) == ["ema"]

    def test_format_report(self):
        series = SmoothedSeries(
            original=[1.0, 2.0, 3.0],
            smoothed=[1.0, 1.5, 2.25],
            method="ema",
            residuals=[0.0, 0.5, 0.75],
        )
        report = format_smoothing_report(series)
        assert "ema" in report
        assert "Series length" in report
