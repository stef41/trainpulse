"""Core training monitor."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from trainpulse._types import (
    Alert,
    AlertSeverity,
    MetricSnapshot,
    MetricType,
    MonitorConfig,
    TrainingReport,
)
from trainpulse.detectors import (
    GradientDetector,
    LossSpikeDetector,
    LRDetector,
    NaNDetector,
    PlateauDetector,
    StepTimeDetector,
)


class Monitor:
    """Lightweight training health monitor.

    Usage::

        monitor = Monitor()
        for step in range(num_steps):
            loss = train_step()
            monitor.log("loss", step, loss)
        report = monitor.report()
    """

    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        self._config = config or MonitorConfig()
        self._snapshots: Dict[str, List[MetricSnapshot]] = {}
        self._alerts: List[Alert] = []
        self._step_count = 0

        # Detectors
        self._nan_detector = NaNDetector() if self._config.check_nan else None
        self._loss_spike = LossSpikeDetector(
            threshold=self._config.loss_spike_threshold,
            window_size=self._config.loss_spike_window,
        )
        self._gradient = GradientDetector(
            explosion_threshold=self._config.grad_norm_threshold,
            vanish_threshold=self._config.grad_vanish_threshold,
        )
        self._lr_detector = LRDetector(
            change_threshold=self._config.lr_change_threshold,
        )
        self._plateau = PlateauDetector(
            patience=self._config.plateau_patience,
            min_delta=self._config.plateau_min_delta,
        )
        self._step_time = StepTimeDetector(
            threshold=self._config.step_time_spike_threshold,
            window_size=self._config.step_time_window,
        )

        # Step timer
        self._last_step_time: Optional[float] = None

    @property
    def config(self) -> MonitorConfig:
        return self._config

    @property
    def alerts(self) -> List[Alert]:
        return list(self._alerts)

    @property
    def snapshots(self) -> Dict[str, List[MetricSnapshot]]:
        return dict(self._snapshots)

    def log(self, name: str, step: int, value: float, **metadata: object) -> List[Alert]:
        """Log a metric value. Returns any alerts triggered."""
        self._step_count = max(self._step_count, step + 1)
        new_alerts: List[Alert] = []

        # NaN/Inf check
        if self._nan_detector is not None:
            alert = self._nan_detector.check(step, name, value)
            if alert is not None:
                new_alerts.append(alert)

        # Determine metric type and run appropriate detectors
        metric_type = _infer_metric_type(name)

        if metric_type == MetricType.LOSS:
            alert = self._loss_spike.check(step, value)
            if alert is not None:
                new_alerts.append(alert)
            alert = self._plateau.check(step, value)
            if alert is not None:
                new_alerts.append(alert)
        elif metric_type == MetricType.GRADIENT_NORM:
            alert = self._gradient.check(step, value)
            if alert is not None:
                new_alerts.append(alert)
        elif metric_type == MetricType.LEARNING_RATE:
            alert = self._lr_detector.check(step, value)
            if alert is not None:
                new_alerts.append(alert)
        elif metric_type == MetricType.STEP_TIME:
            alert = self._step_time.check(step, value)
            if alert is not None:
                new_alerts.append(alert)

        # Store snapshot
        snap = MetricSnapshot(
            step=step,
            name=name,
            value=value,
            metric_type=metric_type,
            metadata=dict(metadata) if metadata else {},
        )
        self._snapshots.setdefault(name, []).append(snap)

        # Dispatch alerts
        for a in new_alerts:
            self._alerts.append(a)
            for cb in self._config.alert_callbacks:
                cb(a)

        return new_alerts

    def step_start(self) -> None:
        """Mark the beginning of a training step for timing."""
        self._last_step_time = time.monotonic()

    def step_end(self, step: int) -> List[Alert]:
        """Mark the end of a training step and log the duration."""
        if self._last_step_time is None:
            return []
        elapsed = time.monotonic() - self._last_step_time
        self._last_step_time = None
        return self.log("step_time", step, elapsed)

    def report(self) -> TrainingReport:
        """Generate a training health report."""
        metrics_summary: Dict[str, Dict[str, float]] = {}
        for name, snaps in self._snapshots.items():
            vals = [s.value for s in snaps]
            finite = [v for v in vals if _is_finite(v)]
            if finite:
                metrics_summary[name] = {
                    "min": min(finite),
                    "max": max(finite),
                    "mean": sum(finite) / len(finite),
                    "last": finite[-1],
                    "count": float(len(vals)),
                }
            else:
                metrics_summary[name] = {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "last": 0.0,
                    "count": float(len(vals)),
                }

        health = _compute_health_score(self._alerts, self._step_count)

        return TrainingReport(
            total_steps=self._step_count,
            alerts=list(self._alerts),
            metrics_summary=metrics_summary,
            health_score=health,
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        self._snapshots.clear()
        self._alerts.clear()
        self._step_count = 0
        self._last_step_time = None
        # Re-initialize detectors
        self.__init__(self._config)  # type: ignore[misc]


def _infer_metric_type(name: str) -> MetricType:
    """Infer metric type from its name."""
    low = name.lower()
    if "loss" in low or "nll" in low or "perplexity" in low:
        return MetricType.LOSS
    if "grad" in low and ("norm" in low or "magnitude" in low):
        return MetricType.GRADIENT_NORM
    if low in ("lr", "learning_rate") or "lr" == low:
        return MetricType.LEARNING_RATE
    if "step_time" in low or "iteration_time" in low:
        return MetricType.STEP_TIME
    if "memory" in low or "vram" in low or "gpu_mem" in low:
        return MetricType.MEMORY_USED
    return MetricType.CUSTOM


def _is_finite(v: float) -> bool:
    import math

    return not (math.isnan(v) or math.isinf(v))


def _compute_health_score(alerts: List[Alert], total_steps: int) -> float:
    """Compute a 0-1 health score based on alerts."""
    if total_steps == 0:
        return 1.0

    penalty = 0.0
    for a in alerts:
        if a.severity == AlertSeverity.CRITICAL:
            penalty += 0.15
        elif a.severity == AlertSeverity.WARNING:
            penalty += 0.05
        elif a.severity == AlertSeverity.INFO:
            penalty += 0.01

    return max(0.0, 1.0 - penalty)
