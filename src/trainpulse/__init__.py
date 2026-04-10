"""trainpulse — lightweight training health monitor."""

__version__ = "0.2.0"

from trainpulse._types import (
    Alert,
    AlertSeverity,
    MetricSnapshot,
    MetricType,
    MonitorConfig,
    TrainingReport,
    TrainpulseError,
)
from trainpulse.callbacks import TrainingCallback
from trainpulse.early_stopping import EarlyStopping, EarlyStopResult, recommend_patience
from trainpulse.monitor import Monitor
from trainpulse.wandb_callback import WandbCallback

__all__ = [
    "Alert",
    "AlertSeverity",
    "EarlyStopping",
    "EarlyStopResult",
    "MetricSnapshot",
    "MetricType",
    "Monitor",
    "MonitorConfig",
    "TrainingCallback",
    "TrainingReport",
    "TrainpulseError",
    "WandbCallback",
    "recommend_patience",
]
