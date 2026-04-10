"""Training cost estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HardwareProfile:
    """Description of a GPU hardware configuration."""

    name: str
    gpu_type: str
    gpu_count: int
    cost_per_hour: float
    memory_gb: float

    @property
    def total_memory_gb(self) -> float:
        return self.memory_gb * self.gpu_count


@dataclass
class TrainingEstimate:
    """Estimated cost and time for a training run."""

    total_hours: float
    total_cost: float
    gpu_hours: float
    tokens_per_second: float
    hardware: str

    @property
    def cost_per_token(self) -> float:
        if self.tokens_per_second <= 0 or self.total_hours <= 0:
            return 0.0
        total_tokens = self.tokens_per_second * self.total_hours * 3600
        if total_tokens == 0:
            return 0.0
        return self.total_cost / total_tokens


# ---------------------------------------------------------------------------
# Default tokens/second look-up (single-GPU baseline for 7B-class models)
# ---------------------------------------------------------------------------

_DEFAULT_TPS: Dict[str, float] = {
    "H100": 12_000.0,
    "A100_80GB": 6_000.0,
    "A100_40GB": 5_000.0,
    "A10G": 2_500.0,
    "L4": 2_000.0,
    "T4": 1_200.0,
    "V100": 2_000.0,
    "RTX_4090": 4_500.0,
}


# ---------------------------------------------------------------------------
# Pre-defined hardware profiles
# ---------------------------------------------------------------------------

COMMON_HARDWARE: Dict[str, HardwareProfile] = {
    "H100": HardwareProfile(
        name="H100",
        gpu_type="H100",
        gpu_count=1,
        cost_per_hour=3.50,
        memory_gb=80.0,
    ),
    "A100_80GB": HardwareProfile(
        name="A100_80GB",
        gpu_type="A100_80GB",
        gpu_count=1,
        cost_per_hour=2.21,
        memory_gb=80.0,
    ),
    "A100_40GB": HardwareProfile(
        name="A100_40GB",
        gpu_type="A100_40GB",
        gpu_count=1,
        cost_per_hour=1.60,
        memory_gb=40.0,
    ),
    "A10G": HardwareProfile(
        name="A10G",
        gpu_type="A10G",
        gpu_count=1,
        cost_per_hour=1.10,
        memory_gb=24.0,
    ),
    "L4": HardwareProfile(
        name="L4",
        gpu_type="L4",
        gpu_count=1,
        cost_per_hour=0.81,
        memory_gb=24.0,
    ),
    "T4": HardwareProfile(
        name="T4",
        gpu_type="T4",
        gpu_count=1,
        cost_per_hour=0.53,
        memory_gb=16.0,
    ),
    "V100": HardwareProfile(
        name="V100",
        gpu_type="V100",
        gpu_count=1,
        cost_per_hour=0.80,
        memory_gb=16.0,
    ),
    "RTX_4090": HardwareProfile(
        name="RTX_4090",
        gpu_type="RTX_4090",
        gpu_count=1,
        cost_per_hour=0.74,
        memory_gb=24.0,
    ),
}


class CostEstimator:
    """Estimate training time and cost for a given hardware profile."""

    def __init__(self, hardware: HardwareProfile) -> None:
        self.hardware = hardware

    def _resolve_tps(self, tokens_per_second: Optional[float] = None) -> float:
        if tokens_per_second is not None:
            return tokens_per_second
        base = _DEFAULT_TPS.get(self.hardware.gpu_type, 2_000.0)
        return base * self.hardware.gpu_count

    def estimate_training(
        self,
        total_tokens: int,
        tokens_per_second: Optional[float] = None,
        epochs: int = 1,
    ) -> TrainingEstimate:
        """Estimate cost for processing *total_tokens* over *epochs* epochs."""
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        tps = self._resolve_tps(tokens_per_second)
        effective_tokens = total_tokens * epochs
        total_seconds = effective_tokens / tps
        total_hours = total_seconds / 3600
        gpu_hours = total_hours * self.hardware.gpu_count
        total_cost = total_hours * self.hardware.cost_per_hour

        return TrainingEstimate(
            total_hours=total_hours,
            total_cost=total_cost,
            gpu_hours=gpu_hours,
            tokens_per_second=tps,
            hardware=self.hardware.name,
        )

    def estimate_finetuning(
        self,
        dataset_size: int,
        seq_length: int,
        batch_size: int,
        epochs: int,
    ) -> TrainingEstimate:
        """Estimate cost for fine-tuning given dataset parameters.

        *dataset_size* is the number of examples, *seq_length* the sequence
        length per example, and *batch_size* the per-device batch size.
        """
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        total_tokens = dataset_size * seq_length
        return self.estimate_training(
            total_tokens=total_tokens,
            epochs=epochs,
        )

    @staticmethod
    def compare_hardware(
        profiles: List[HardwareProfile],
        total_tokens: int,
        tokens_per_second: Optional[float] = None,
        epochs: int = 1,
    ) -> List[TrainingEstimate]:
        """Compare estimates across multiple hardware profiles."""
        results: List[TrainingEstimate] = []
        for profile in profiles:
            est = CostEstimator(profile)
            results.append(
                est.estimate_training(total_tokens, tokens_per_second, epochs)
            )
        return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_cost_report(estimates: List[TrainingEstimate] | TrainingEstimate) -> str:
    """Format one or more estimates as a human-readable report."""
    if isinstance(estimates, TrainingEstimate):
        estimates = [estimates]
    if not estimates:
        return "No estimates to report."

    lines: list[str] = []
    lines.append("Training Cost Estimate")
    lines.append("=" * 60)

    for est in estimates:
        lines.append(f"Hardware       : {est.hardware}")
        lines.append(f"Tokens/sec     : {est.tokens_per_second:,.0f}")
        lines.append(f"Total hours    : {est.total_hours:,.2f}")
        lines.append(f"GPU hours      : {est.gpu_hours:,.2f}")
        lines.append(f"Total cost     : ${est.total_cost:,.2f}")
        lines.append("-" * 60)

    return "\n".join(lines)
