"""Tests for Mann–Kendall trend test and isolation forest."""

from __future__ import annotations

import math

import pytest

from trainpulse.statistical import (
    MannKendallResult,
    mann_kendall,
    IsolationForestResult,
    AnomalyScore,
    isolation_forest,
    _sign,
    _normal_cdf,
    _c,
    _harmonic_number,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Mann–Kendall Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSign:
    def test_positive(self):
        assert _sign(3.14) == 1

    def test_negative(self):
        assert _sign(-2.7) == -1

    def test_zero(self):
        assert _sign(0.0) == 0


class TestNormalCDF:
    def test_zero(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5, abs=0.01)

    def test_large_positive(self):
        assert _normal_cdf(5.0) == pytest.approx(1.0, abs=0.001)

    def test_large_negative(self):
        assert _normal_cdf(-5.0) == pytest.approx(0.0, abs=0.001)

    def test_one_sigma(self):
        # CDF(1.0) ≈ 0.8413
        assert _normal_cdf(1.0) == pytest.approx(0.8413, abs=0.01)

    def test_two_sigma(self):
        # CDF(2.0) ≈ 0.9772
        assert _normal_cdf(2.0) == pytest.approx(0.9772, abs=0.01)

    def test_monotonic(self):
        vals = [_normal_cdf(x / 10) for x in range(-50, 51)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]


class TestMannKendallMonotone:
    def test_increasing_trend(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = mann_kendall(series)
        assert result.trend == "increasing"
        assert result.statistic_s > 0
        assert result.significant is True
        assert result.p_value < 0.05

    def test_decreasing_trend(self):
        series = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        result = mann_kendall(series)
        assert result.trend == "decreasing"
        assert result.statistic_s < 0
        assert result.significant is True

    def test_no_trend(self):
        # Oscillating data — no monotone trend
        series = [1.0, 5.0, 2.0, 4.0, 3.0, 5.0, 1.0, 4.0, 2.0, 3.0]
        result = mann_kendall(series)
        # p-value should be high (not significant)
        assert result.p_value > 0.05 or result.trend == "no_trend"

    def test_short_series(self):
        result = mann_kendall([1.0, 2.0])
        assert result.trend == "no_trend"
        assert result.n == 2


class TestMannKendallSenSlope:
    def test_perfect_linear(self):
        series = [1.0, 3.0, 5.0, 7.0, 9.0]
        result = mann_kendall(series)
        assert result.sen_slope == pytest.approx(2.0, abs=0.01)

    def test_negative_slope(self):
        series = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = mann_kendall(series)
        assert result.sen_slope < 0

    def test_zero_slope(self):
        series = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = mann_kendall(series)
        assert result.sen_slope == pytest.approx(0.0, abs=0.01)


class TestMannKendallTies:
    def test_all_tied(self):
        series = [3.0, 3.0, 3.0, 3.0, 3.0]
        result = mann_kendall(series)
        assert result.statistic_s == 0
        assert result.trend == "no_trend"

    def test_partial_ties(self):
        series = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
        result = mann_kendall(series)
        assert result.statistic_s > 0


class TestMannKendallResult:
    def test_result_fields(self):
        result = mann_kendall([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(result, MannKendallResult)
        assert isinstance(result.statistic_s, int)
        assert isinstance(result.z_score, float)
        assert isinstance(result.p_value, float)
        assert result.trend in ("increasing", "decreasing", "no_trend")
        assert 0.0 <= result.p_value <= 1.0

    def test_loss_curve(self):
        # Simulated training loss that decreases with noise
        import random
        rng = random.Random(42)
        series = [1.0 - 0.05 * i + rng.gauss(0, 0.02) for i in range(50)]
        result = mann_kendall(series)
        assert result.trend == "decreasing"
        assert result.sen_slope < 0


# ══════════════════════════════════════════════════════════════════════════════
#  Isolation Forest Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHarmonicNumber:
    def test_zero(self):
        assert _harmonic_number(0) == 0.0

    def test_one(self):
        assert _harmonic_number(1) == 1.0

    def test_positive(self):
        # H(10) ≈ 2.93
        assert _harmonic_number(10) == pytest.approx(2.93, abs=0.1)


class TestCFunction:
    def test_zero(self):
        assert _c(0) == 0.0

    def test_one(self):
        assert _c(1) == 0.0

    def test_two(self):
        assert _c(2) == 1.0

    def test_256(self):
        # c(256) ≈ 2*H(255) - 2*255/256 ≈ 10.2
        result = _c(256)
        assert 9 < result < 12


class TestIsolationForest:
    def test_returns_result(self):
        metrics = {
            "loss": [1.0 - 0.01 * i for i in range(100)],
            "grad_norm": [0.5 + 0.01 * i for i in range(100)],
        }
        result = isolation_forest(metrics, n_trees=10, seed=42)
        assert isinstance(result, IsolationForestResult)

    def test_scores_in_range(self):
        metrics = {"loss": [float(i) for i in range(50)]}
        result = isolation_forest(metrics, n_trees=10, seed=42)
        for s in result.scores:
            assert 0.0 <= s.score <= 1.0

    def test_detects_anomaly(self):
        # Normal data with one spike
        import random
        rng = random.Random(42)
        loss = [1.0 - 0.01 * i + rng.gauss(0, 0.01) for i in range(100)]
        loss[50] = 10.0  # big anomaly
        metrics = {"loss": loss}
        result = isolation_forest(metrics, n_trees=50, contamination=0.05, seed=42)
        # Step 50 should be among the anomalies
        assert 50 in result.anomaly_indices

    def test_empty_metrics(self):
        result = isolation_forest({})
        assert len(result.scores) == 0

    def test_single_metric(self):
        metrics = {"loss": [1.0, 0.9, 0.8, 0.7]}
        result = isolation_forest(metrics, n_trees=10, seed=42)
        assert len(result.scores) == 4

    def test_multiple_metrics(self):
        metrics = {
            "loss": [1.0, 0.9, 0.8, 0.7, 0.6] * 10,
            "grad_norm": [0.1, 0.2, 0.15, 0.18, 0.12] * 10,
            "lr": [0.001] * 50,
        }
        result = isolation_forest(metrics, n_trees=20, seed=42)
        assert len(result.scores) == 50

    def test_anomaly_score_fields(self):
        metrics = {"loss": [float(i) for i in range(20)]}
        result = isolation_forest(metrics, n_trees=10, seed=42)
        for s in result.scores:
            assert isinstance(s, AnomalyScore)
            assert isinstance(s.step, int)
            assert isinstance(s.score, float)
            assert isinstance(s.path_length, float)
            assert isinstance(s.is_anomaly, bool)

    def test_contamination_rate(self):
        import random
        rng = random.Random(42)
        metrics = {"loss": [rng.gauss(0, 1) for _ in range(100)]}
        result = isolation_forest(metrics, contamination=0.1, n_trees=20, seed=42)
        # Approximately 10% should be flagged as anomalies
        n_anomalies = len(result.anomaly_indices)
        assert 5 <= n_anomalies <= 15

    def test_threshold_set(self):
        metrics = {"loss": [float(i) for i in range(20)]}
        result = isolation_forest(metrics, n_trees=10, seed=42)
        assert result.threshold > 0

    def test_reproducible(self):
        metrics = {"loss": [float(i) for i in range(50)]}
        r1 = isolation_forest(metrics, n_trees=10, seed=42)
        r2 = isolation_forest(metrics, n_trees=10, seed=42)
        for s1, s2 in zip(r1.scores, r2.scores):
            assert s1.score == s2.score
