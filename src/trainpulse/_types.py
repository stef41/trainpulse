"""Core types for trainpulse."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(str, Enum):
    LOSS = "loss"
    GRADIENT_NORM = "gradient_norm"
    LEARNING_RATE = "learning_rate"
    STEP_TIME = "step_time"
    MEMORY_USED = "memory_used"
    CUSTOM = "custom"


@dataclass
class MetricSnapshot:
    """A single metric recording at a given step."""

    step: int
    name: str
    value: float
    metric_type: MetricType = MetricType.CUSTOM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """An alert triggered by a detector."""

    step: int
    severity: AlertSeverity
    detector: str
    message: str
    metric_name: str = ""
    metric_value: float = 0.0

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] Step {self.step}: {self.message}"


@dataclass
class MonitorConfig:
    """Configuration for the training monitor."""

    # Loss spike detection
    loss_spike_threshold: float = 5.0  # Multiplier over rolling average
    loss_spike_window: int = 50  # Rolling window size

    # Gradient monitoring
    grad_norm_threshold: float = 100.0  # Max acceptable gradient norm
    grad_vanish_threshold: float = 1e-7  # Min acceptable gradient norm

    # NaN/Inf detection
    check_nan: bool = True

    # Learning rate monitoring
    lr_change_threshold: float = 10.0  # Max acceptable LR ratio change per step

    # Step time monitoring
    step_time_spike_threshold: float = 3.0  # Multiplier over rolling average
    step_time_window: int = 20

    # Plateau detection
    plateau_patience: int = 100  # Steps without improvement
    plateau_min_delta: float = 1e-5  # Minimum change to count as improvement

    # General
    log_interval: int = 1  # Record every N steps
    alert_callbacks: List[Callable[[Alert], None]] = field(default_factory=list)


@dataclass
class TrainingReport:
    """Summary report of training health."""

    total_steps: int
    alerts: List[Alert]
    metrics_summary: Dict[str, Dict[str, float]]  # metric_name -> {min, max, mean, last}
    health_score: float  # 0.0 (terrible) to 1.0 (perfect)

    @property
    def n_warnings(self) -> int:
        return sum(1 for a in self.alerts if a.severity == AlertSeverity.WARNING)

    @property
    def n_critical(self) -> int:
        return sum(1 for a in self.alerts if a.severity == AlertSeverity.CRITICAL)

    @property
    def is_healthy(self) -> bool:
        return self.n_critical == 0


class TrainpulseError(Exception):
    """Base exception."""
