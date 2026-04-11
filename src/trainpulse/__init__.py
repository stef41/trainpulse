"""trainpulse — lightweight training health monitor."""

__version__ = "0.4.0"

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
from trainpulse.cost import (
    COMMON_HARDWARE,
    CostEstimator,
    HardwareProfile,
    TrainingEstimate,
    format_cost_report,
)
from trainpulse.early_stopping import EarlyStopping, EarlyStopResult, recommend_patience
from trainpulse.monitor import Monitor
from trainpulse.smoothing import (
    MetricSmoother,
    SmoothedSeries,
    SmoothingConfig,
    compare_methods,
    format_smoothing_report,
)
from trainpulse.wandb_callback import WandbCallback
from trainpulse.spectral import spectral_analysis, SpectralResult, FrequencyComponent
from trainpulse.statistical import (
    mann_kendall,
    MannKendallResult,
    isolation_forest,
    IsolationForestResult,
    AnomalyScore,
)

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
    # Cost estimation
    "COMMON_HARDWARE",
    "CostEstimator",
    "HardwareProfile",
    "TrainingEstimate",
    "format_cost_report",
    # Smoothing
    "MetricSmoother",
    "SmoothedSeries",
    "SmoothingConfig",
    "compare_methods",
    "format_smoothing_report",
    # Spectral analysis
    "spectral_analysis",
    "SpectralResult",
    "FrequencyComponent",
    # Statistical tests
    "mann_kendall",
    "MannKendallResult",
    "isolation_forest",
    "IsolationForestResult",
    "AnomalyScore",
]
