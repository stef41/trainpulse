"""Tests for the early stopping recommendation engine."""

from __future__ import annotations

import pytest

from trainpulse.early_stopping import EarlyStopping, EarlyStopResult, recommend_patience

# ──────────────────────────────────────────────────────────────────────
# EarlyStopResult dataclass
# ──────────────────────────────────────────────────────────────────────


class TestEarlyStopResult:
    def test_fields(self):
        r = EarlyStopResult(step=0, value=0.5, should_stop=False, improved=True, best_value=0.5, best_step=0)
        assert r.step == 0
        assert r.value == 0.5
        assert r.should_stop is False
        assert r.improved is True
        assert r.best_value == 0.5
        assert r.best_step == 0


# ──────────────────────────────────────────────────────────────────────
# EarlyStopping — core behaviour
# ──────────────────────────────────────────────────────────────────────


class TestEarlyStoppingMinMode:
    def test_first_step_always_improves(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        r = es.step(1.0)
        assert r.improved is True
        assert r.should_stop is False
        assert r.best_value == 1.0
        assert r.best_step == 0

    def test_monotonic_decrease_never_stops(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        for i in range(20):
            r = es.step(1.0 - i * 0.01)
        assert r.should_stop is False
        assert es.should_stop is False

    def test_patience_exhausted(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es.step(1.0)  # best
        es.step(1.1)
        es.step(1.2)
        r = es.step(1.3)  # 3rd non-improvement → stop
        assert r.should_stop is True
        assert es.should_stop is True

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es.step(1.0)
        es.step(1.1)  # +1
        es.step(1.2)  # +2
        es.step(0.5)  # improvement → reset
        assert es.steps_without_improvement == 0
        assert es.best_value == 0.5

    def test_min_delta_respected(self):
        es = EarlyStopping(patience=3, min_delta=0.1)
        es.step(1.0)
        r = es.step(0.95)  # only 0.05 better, below delta
        assert r.improved is False
        r = es.step(0.89)  # 0.11 better than best (1.0) → improved
        assert r.improved is True

    def test_best_step_tracks_correctly(self):
        es = EarlyStopping(patience=5, min_delta=0.0)
        es.step(1.0)  # step 0
        es.step(0.8)  # step 1
        es.step(0.9)  # step 2
        es.step(0.7)  # step 3
        assert es.best_step == 3
        assert es.best_value == 0.7


class TestEarlyStoppingMaxMode:
    def test_max_mode_improves_upward(self):
        es = EarlyStopping(patience=3, min_delta=0.0, mode="max")
        es.step(0.5)
        r = es.step(0.8)
        assert r.improved is True
        assert es.best_value == 0.8

    def test_max_mode_stops_on_decline(self):
        es = EarlyStopping(patience=2, min_delta=0.0, mode="max")
        es.step(0.9)
        es.step(0.8)
        r = es.step(0.7)
        assert r.should_stop is True

    def test_max_mode_min_delta(self):
        es = EarlyStopping(patience=3, min_delta=0.1, mode="max")
        es.step(1.0)
        r = es.step(1.05)  # only +0.05, below delta
        assert r.improved is False


# ──────────────────────────────────────────────────────────────────────
# EarlyStopping — edge cases & validation
# ──────────────────────────────────────────────────────────────────────


class TestEarlyStoppingEdgeCases:
    def test_patience_one(self):
        es = EarlyStopping(patience=1, min_delta=0.0)
        es.step(1.0)
        r = es.step(1.0)  # no improvement
        assert r.should_stop is True

    def test_invalid_patience_raises(self):
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            EarlyStopping(mode="average")  # type: ignore[arg-type]

    def test_properties_before_any_step(self):
        es = EarlyStopping()
        assert es.best_value is None
        assert es.best_step == 0
        assert es.should_stop is False
        assert es.steps_without_improvement == 0

    def test_step_indices_increment(self):
        es = EarlyStopping(patience=10)
        results = [es.step(float(i)) for i in range(5)]
        assert [r.step for r in results] == [0, 1, 2, 3, 4]

    def test_equal_values_are_not_improvement_min(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es.step(1.0)
        r = es.step(1.0)
        assert r.improved is False

    def test_equal_values_are_not_improvement_max(self):
        es = EarlyStopping(patience=3, min_delta=0.0, mode="max")
        es.step(1.0)
        r = es.step(1.0)
        assert r.improved is False


# ──────────────────────────────────────────────────────────────────────
# recommend_patience
# ──────────────────────────────────────────────────────────────────────


class TestRecommendPatience:
    def test_returns_default_for_short_history(self):
        assert recommend_patience([]) == 5
        assert recommend_patience([0.5]) == 5

    def test_monotonically_decreasing_loss(self):
        history = [1.0 - 0.01 * i for i in range(100)]
        p = recommend_patience(history)
        # Every step improves → gaps are all 1 → patience should be small.
        assert 3 <= p <= 10

    def test_flat_loss_gives_high_patience(self):
        history = [1.0] * 50
        p = recommend_patience(history)
        # No improvements at all → conservative patience.
        assert p >= 5

    def test_noisy_loss_with_occasional_drops(self):
        # Spiky loss that drops every ~20 steps.
        import random

        rng = random.Random(42)
        history: list[float] = []
        base = 1.0
        for i in range(100):
            if i > 0 and i % 20 == 0:
                base -= 0.1
            history.append(base + rng.uniform(-0.02, 0.02))
        p = recommend_patience(history)
        # Gaps ~20 steps → patience should be at least 10.
        assert p >= 10

    def test_result_is_always_positive_int(self):
        p = recommend_patience([10.0, 9.0, 8.0])
        assert isinstance(p, int)
        assert p >= 1
