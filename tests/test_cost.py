"""Tests for trainpulse.cost."""

import pytest

from trainpulse.cost import (
    COMMON_HARDWARE,
    CostEstimator,
    HardwareProfile,
    TrainingEstimate,
    format_cost_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h100_profile() -> HardwareProfile:
    return COMMON_HARDWARE["H100"]


def _multi_gpu() -> HardwareProfile:
    return HardwareProfile(
        name="8xA100",
        gpu_type="A100_80GB",
        gpu_count=8,
        cost_per_hour=17.68,
        memory_gb=80.0,
    )


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------

class TestHardwareProfile:
    def test_basic_fields(self):
        p = _h100_profile()
        assert p.name == "H100"
        assert p.gpu_type == "H100"
        assert p.gpu_count == 1

    def test_total_memory(self):
        p = _multi_gpu()
        assert p.total_memory_gb == pytest.approx(640.0)

    def test_single_gpu_memory(self):
        p = _h100_profile()
        assert p.total_memory_gb == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# TrainingEstimate
# ---------------------------------------------------------------------------

class TestTrainingEstimate:
    def test_cost_per_token(self):
        est = TrainingEstimate(
            total_hours=1.0,
            total_cost=3.50,
            gpu_hours=1.0,
            tokens_per_second=1000.0,
            hardware="H100",
        )
        # 1 hour * 3600 sec * 1000 tps = 3_600_000 tokens
        expected = 3.50 / 3_600_000
        assert est.cost_per_token == pytest.approx(expected)

    def test_cost_per_token_zero_hours(self):
        est = TrainingEstimate(0.0, 0.0, 0.0, 0.0, "H100")
        assert est.cost_per_token == 0.0


# ---------------------------------------------------------------------------
# CostEstimator.estimate_training
# ---------------------------------------------------------------------------

class TestEstimateTraining:
    def test_basic(self):
        est = CostEstimator(_h100_profile())
        result = est.estimate_training(total_tokens=1_000_000)
        assert result.total_hours > 0
        assert result.total_cost > 0
        assert result.hardware == "H100"

    def test_custom_tps(self):
        est = CostEstimator(_h100_profile())
        result = est.estimate_training(total_tokens=3_600_000, tokens_per_second=1000)
        # 3_600_000 / 1000 = 3600 seconds = 1 hour
        assert result.total_hours == pytest.approx(1.0)
        assert result.total_cost == pytest.approx(3.50)

    def test_multiple_epochs(self):
        est = CostEstimator(_h100_profile())
        r1 = est.estimate_training(total_tokens=1_000_000, epochs=1)
        r3 = est.estimate_training(total_tokens=1_000_000, epochs=3)
        assert r3.total_hours == pytest.approx(r1.total_hours * 3)
        assert r3.total_cost == pytest.approx(r1.total_cost * 3)

    def test_multi_gpu_scales_tps(self):
        est = CostEstimator(_multi_gpu())
        result = est.estimate_training(total_tokens=1_000_000)
        # 8 GPUs → 8× base tps
        single = CostEstimator(COMMON_HARDWARE["A100_80GB"])
        single_r = single.estimate_training(total_tokens=1_000_000)
        assert result.tokens_per_second == pytest.approx(single_r.tokens_per_second * 8)

    def test_negative_tokens_raises(self):
        est = CostEstimator(_h100_profile())
        with pytest.raises(ValueError, match="positive"):
            est.estimate_training(total_tokens=-1)

    def test_zero_tokens_raises(self):
        est = CostEstimator(_h100_profile())
        with pytest.raises(ValueError, match="positive"):
            est.estimate_training(total_tokens=0)

    def test_zero_epochs_raises(self):
        est = CostEstimator(_h100_profile())
        with pytest.raises(ValueError, match="epochs"):
            est.estimate_training(total_tokens=1000, epochs=0)

    def test_gpu_hours(self):
        p = _multi_gpu()
        est = CostEstimator(p)
        result = est.estimate_training(total_tokens=1_000_000, tokens_per_second=1000)
        # total_hours * 8 gpus
        assert result.gpu_hours == pytest.approx(result.total_hours * 8)


# ---------------------------------------------------------------------------
# CostEstimator.estimate_finetuning
# ---------------------------------------------------------------------------

class TestEstimateFinetuning:
    def test_basic(self):
        est = CostEstimator(_h100_profile())
        result = est.estimate_finetuning(
            dataset_size=10_000,
            seq_length=2048,
            batch_size=4,
            epochs=3,
        )
        assert result.total_hours > 0
        assert result.total_cost > 0

    def test_matches_training(self):
        """Finetuning estimate should equal training on dataset_size * seq_length tokens."""
        est = CostEstimator(_h100_profile())
        ft = est.estimate_finetuning(dataset_size=1000, seq_length=512, batch_size=8, epochs=2)
        tr = est.estimate_training(total_tokens=1000 * 512, epochs=2)
        assert ft.total_hours == pytest.approx(tr.total_hours)

    def test_invalid_dataset_size(self):
        est = CostEstimator(_h100_profile())
        with pytest.raises(ValueError, match="dataset_size"):
            est.estimate_finetuning(dataset_size=0, seq_length=512, batch_size=8, epochs=1)

    def test_invalid_seq_length(self):
        est = CostEstimator(_h100_profile())
        with pytest.raises(ValueError, match="seq_length"):
            est.estimate_finetuning(dataset_size=100, seq_length=-1, batch_size=8, epochs=1)

    def test_invalid_batch_size(self):
        est = CostEstimator(_h100_profile())
        with pytest.raises(ValueError, match="batch_size"):
            est.estimate_finetuning(dataset_size=100, seq_length=512, batch_size=0, epochs=1)


# ---------------------------------------------------------------------------
# compare_hardware
# ---------------------------------------------------------------------------

class TestCompareHardware:
    def test_returns_all_profiles(self):
        profiles = [COMMON_HARDWARE["H100"], COMMON_HARDWARE["T4"]]
        results = CostEstimator.compare_hardware(profiles, total_tokens=1_000_000)
        assert len(results) == 2
        assert results[0].hardware == "H100"
        assert results[1].hardware == "T4"

    def test_cheaper_gpu_takes_longer(self):
        profiles = [COMMON_HARDWARE["H100"], COMMON_HARDWARE["T4"]]
        results = CostEstimator.compare_hardware(profiles, total_tokens=10_000_000)
        h100, t4 = results
        assert h100.total_hours < t4.total_hours

    def test_empty(self):
        results = CostEstimator.compare_hardware([], total_tokens=1_000_000)
        assert results == []


# ---------------------------------------------------------------------------
# COMMON_HARDWARE
# ---------------------------------------------------------------------------

class TestCommonHardware:
    def test_known_keys(self):
        for key in ("H100", "A100_80GB", "A100_40GB", "A10G", "L4", "T4", "V100", "RTX_4090"):
            assert key in COMMON_HARDWARE

    def test_positive_costs(self):
        for profile in COMMON_HARDWARE.values():
            assert profile.cost_per_hour > 0
            assert profile.memory_gb > 0


# ---------------------------------------------------------------------------
# format_cost_report
# ---------------------------------------------------------------------------

class TestFormatCostReport:
    def test_single_estimate(self):
        est = TrainingEstimate(2.5, 8.75, 2.5, 6000, "A100_80GB")
        report = format_cost_report(est)
        assert "A100_80GB" in report
        assert "$8.75" in report

    def test_multiple_estimates(self):
        estimates = [
            TrainingEstimate(1.0, 3.50, 1.0, 12000, "H100"),
            TrainingEstimate(5.0, 2.65, 5.0, 1200, "T4"),
        ]
        report = format_cost_report(estimates)
        assert "H100" in report
        assert "T4" in report

    def test_empty(self):
        assert "No estimates" in format_cost_report([])
