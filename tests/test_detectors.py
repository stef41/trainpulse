"""Tests for detectors module."""

import math
import pytest
from trainpulse._types import AlertSeverity
from trainpulse.detectors import (
    GradientDetector,
    LRDetector,
    LossSpikeDetector,
    NaNDetector,
    PlateauDetector,
    RollingWindow,
    StepTimeDetector,
)


class TestRollingWindow:
    def test_empty(self):
        w = RollingWindow(5)
        assert len(w) == 0
        assert w.mean == 0.0
        assert w.std == 0.0
        assert not w.is_full
        assert w.values == []

    def test_add_values(self):
        w = RollingWindow(3)
        w.add(1.0)
        w.add(2.0)
        assert len(w) == 2
        assert not w.is_full
        w.add(3.0)
        assert len(w) == 3
        assert w.is_full

    def test_rolling(self):
        w = RollingWindow(3)
        w.add(1.0)
        w.add(2.0)
        w.add(3.0)
        assert w.values == [1.0, 2.0, 3.0]
        w.add(4.0)
        assert w.values == [2.0, 3.0, 4.0]
        assert len(w) == 3

    def test_mean(self):
        w = RollingWindow(5)
        for v in [10.0, 20.0, 30.0]:
            w.add(v)
        assert w.mean == pytest.approx(20.0)

    def test_std(self):
        w = RollingWindow(3)
        w.add(1.0)
        # std of single value
        assert w.std == 0.0
        w.add(2.0)
        w.add(3.0)
        assert w.std > 0

    def test_min_size(self):
        w = RollingWindow(0)
        w.add(1.0)
        assert len(w) == 1
        assert w.is_full


class TestNaNDetector:
    def test_normal_value(self):
        d = NaNDetector()
        assert d.check(0, "loss", 1.0) is None

    def test_nan(self):
        d = NaNDetector()
        alert = d.check(5, "loss", float("nan"))
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.detector == "nan_detector"
        assert "NaN" in alert.message

    def test_inf(self):
        d = NaNDetector()
        alert = d.check(10, "grad_norm", float("inf"))
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert "Inf" in alert.message

    def test_neg_inf(self):
        d = NaNDetector()
        alert = d.check(0, "loss", float("-inf"))
        assert alert is not None
        assert "Inf" in alert.message

    def test_zero(self):
        d = NaNDetector()
        assert d.check(0, "loss", 0.0) is None

    def test_negative(self):
        d = NaNDetector()
        assert d.check(0, "loss", -1.5) is None


class TestLossSpikeDetector:
    def test_no_spike_during_warmup(self):
        d = LossSpikeDetector(threshold=5.0, window_size=5)
        # Fill window with small values
        for i in range(4):
            assert d.check(i, 1.0) is None

    def test_spike_detected(self):
        d = LossSpikeDetector(threshold=3.0, window_size=5)
        # Fill window
        for i in range(5):
            d.check(i, 1.0)
        # Spike
        alert = d.check(5, 10.0)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.detector == "loss_spike"

    def test_no_spike_normal_increase(self):
        d = LossSpikeDetector(threshold=5.0, window_size=5)
        for i in range(5):
            d.check(i, 1.0)
        # 2x is within 5x threshold
        assert d.check(5, 2.0) is None

    def test_spike_threshold_exact(self):
        d = LossSpikeDetector(threshold=2.0, window_size=3)
        for i in range(3):
            d.check(i, 1.0)
        # Exactly 2x is NOT a spike (> not >=)  
        assert d.check(3, 2.0) is None
        # But 2.1x is
        d2 = LossSpikeDetector(threshold=2.0, window_size=3)
        for i in range(3):
            d2.check(i, 1.0)
        alert = d2.check(3, 2.1)
        assert alert is not None

    def test_rolling_average_updates(self):
        d = LossSpikeDetector(threshold=5.0, window_size=3)
        for i in range(3):
            d.check(i, 1.0)
        # After adding larger values, the rolling average rises
        d.check(3, 4.0)
        d.check(4, 4.0)
        d.check(5, 4.0)
        # Now avg ≈ 4.0, so 15.0 is only 3.75x (below 5x)
        assert d.check(6, 15.0) is None


class TestGradientDetector:
    def test_normal(self):
        d = GradientDetector()
        assert d.check(0, 1.0) is None

    def test_explosion(self):
        d = GradientDetector(explosion_threshold=100.0)
        alert = d.check(5, 150.0)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.detector == "gradient"
        assert "explosion" in alert.message.lower()

    def test_vanishing(self):
        d = GradientDetector(vanish_threshold=1e-7)
        alert = d.check(10, 1e-9)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert "vanish" in alert.message.lower()

    def test_zero_gradient_no_vanish_alert(self):
        d = GradientDetector(vanish_threshold=1e-7)
        # Zero gradient: condition is 0 < grad < threshold, so 0 doesn't match
        assert d.check(0, 0.0) is None

    def test_negative_gradient_norm(self):
        d = GradientDetector(vanish_threshold=1e-7)
        # Negative shouldn't happen but shouldn't crash
        assert d.check(0, -1.0) is None

    def test_custom_thresholds(self):
        d = GradientDetector(explosion_threshold=10.0, vanish_threshold=0.01)
        assert d.check(0, 5.0) is None
        assert d.check(1, 15.0) is not None  # explosion
        assert d.check(2, 0.001) is not None  # vanishing


class TestLRDetector:
    def test_first_value(self):
        d = LRDetector()
        assert d.check(0, 0.001) is None

    def test_normal_change(self):
        d = LRDetector(change_threshold=10.0)
        d.check(0, 0.001)
        assert d.check(1, 0.0009) is None  # ~1.1x change

    def test_large_increase(self):
        d = LRDetector(change_threshold=5.0)
        d.check(0, 0.001)
        alert = d.check(1, 0.01)  # 10x increase
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.detector == "learning_rate"

    def test_large_decrease(self):
        d = LRDetector(change_threshold=5.0)
        d.check(0, 0.01)
        alert = d.check(1, 0.001)  # 10x decrease
        assert alert is not None

    def test_zero_lr(self):
        d = LRDetector(change_threshold=5.0)
        d.check(0, 0.001)
        # Zero LR should not alert (division issues)
        assert d.check(1, 0.0) is None

    def test_from_zero(self):
        d = LRDetector()
        d.check(0, 0.0)
        assert d.check(1, 0.001) is None  # prev was 0, skip

    def test_tracks_previous(self):
        d = LRDetector(change_threshold=5.0)
        d.check(0, 0.001)
        d.check(1, 0.002)
        # Now prev is 0.002, so 0.012 is 6x → alert
        assert d.check(2, 0.012) is not None


class TestPlateauDetector:
    def test_improving(self):
        d = PlateauDetector(patience=5, min_delta=0.01)
        for i in range(10):
            # Consistently decreasing loss
            assert d.check(i, 1.0 - i * 0.05) is None

    def test_plateau_detected(self):
        d = PlateauDetector(patience=5, min_delta=0.01)
        # First value sets the best
        d.check(0, 1.0)
        # No improvement for 5 steps
        for i in range(1, 6):
            result = d.check(i, 1.0)
        # 5 steps without improvement → alert on step 5
        assert result is not None
        assert result.severity == AlertSeverity.WARNING
        assert result.detector == "plateau"

    def test_plateau_only_alerts_once(self):
        d = PlateauDetector(patience=3, min_delta=0.01)
        d.check(0, 1.0)
        d.check(1, 1.0)
        d.check(2, 1.0)
        alert = d.check(3, 1.0)
        assert alert is not None
        # Should not alert again
        assert d.check(4, 1.0) is None
        assert d.check(5, 1.0) is None

    def test_improvement_resets_counter(self):
        d = PlateauDetector(patience=3, min_delta=0.01)
        d.check(0, 1.0)
        d.check(1, 1.0)
        d.check(2, 1.0)
        # Improve
        d.check(3, 0.5)
        # No improvement again
        d.check(4, 0.5)
        d.check(5, 0.5)
        # After step 3 (improve), steps_without_improvement resets to 0
        # Steps 4,5,6 → 3 steps without improvement = patience 3 → alert
        alert = d.check(6, 0.5)
        assert alert is not None

    def test_min_delta(self):
        d = PlateauDetector(patience=3, min_delta=0.1)
        d.check(0, 1.0)
        # Tiny improvements (< min_delta) don't count
        d.check(1, 0.95)  # 0.95 < 1.0 - 0.1 = 0.9? No → steps_without = 1
        d.check(2, 0.91)  # 0.91 < 0.9? No → steps_without = 2
        # Step 3: steps_without = 3 ≥ patience 3 → alert
        alert = d.check(3, 0.91)
        assert alert is not None


class TestStepTimeDetector:
    def test_no_spike_during_warmup(self):
        d = StepTimeDetector(threshold=3.0, window_size=3)
        assert d.check(0, 1.0) is None
        assert d.check(1, 1.0) is None

    def test_spike_detected(self):
        d = StepTimeDetector(threshold=3.0, window_size=3)
        for i in range(3):
            d.check(i, 1.0)
        alert = d.check(3, 5.0)  # 5x avg
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.detector == "step_time"

    def test_no_spike_within_threshold(self):
        d = StepTimeDetector(threshold=3.0, window_size=3)
        for i in range(3):
            d.check(i, 1.0)
        assert d.check(3, 2.5) is None  # 2.5x < 3.0x threshold

    def test_rolling_window_adapts(self):
        d = StepTimeDetector(threshold=3.0, window_size=3)
        for i in range(3):
            d.check(i, 1.0)
        # Add some slow steps that get absorbed into the window
        d.check(3, 2.0)
        d.check(4, 2.0)
        d.check(5, 2.0)
        # Now avg is ~2.0, so 5.5 is 2.75x (below 3x)
        assert d.check(6, 5.5) is None
