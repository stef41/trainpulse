"""Tests for _types module."""

import pytest
from trainpulse._types import (
    Alert,
    AlertSeverity,
    MetricSnapshot,
    MetricType,
    MonitorConfig,
    TrainingReport,
    TrainpulseError,
)


class TestAlertSeverity:
    def test_values(self):
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_is_string(self):
        assert isinstance(AlertSeverity.INFO, str)
        assert AlertSeverity.WARNING == "warning"


class TestMetricType:
    def test_values(self):
        assert MetricType.LOSS.value == "loss"
        assert MetricType.GRADIENT_NORM.value == "gradient_norm"
        assert MetricType.LEARNING_RATE.value == "learning_rate"
        assert MetricType.STEP_TIME.value == "step_time"
        assert MetricType.MEMORY_USED.value == "memory_used"
        assert MetricType.CUSTOM.value == "custom"


class TestMetricSnapshot:
    def test_basic(self):
        snap = MetricSnapshot(step=10, name="loss", value=0.5)
        assert snap.step == 10
        assert snap.name == "loss"
        assert snap.value == 0.5
        assert snap.metric_type == MetricType.CUSTOM
        assert snap.metadata == {}

    def test_with_type_and_metadata(self):
        snap = MetricSnapshot(
            step=1,
            name="grad_norm",
            value=3.14,
            metric_type=MetricType.GRADIENT_NORM,
            metadata={"layer": "encoder"},
        )
        assert snap.metric_type == MetricType.GRADIENT_NORM
        assert snap.metadata == {"layer": "encoder"}


class TestAlert:
    def test_basic(self):
        alert = Alert(
            step=5,
            severity=AlertSeverity.WARNING,
            detector="loss_spike",
            message="Loss spiked to 10.0",
        )
        assert alert.step == 5
        assert alert.severity == AlertSeverity.WARNING
        assert alert.detector == "loss_spike"
        assert alert.metric_name == ""
        assert alert.metric_value == 0.0

    def test_str(self):
        alert = Alert(
            step=42,
            severity=AlertSeverity.CRITICAL,
            detector="nan",
            message="Loss is NaN",
        )
        s = str(alert)
        assert "CRITICAL" in s
        assert "42" in s
        assert "NaN" in s

    def test_with_metric_info(self):
        alert = Alert(
            step=1,
            severity=AlertSeverity.INFO,
            detector="test",
            message="test",
            metric_name="loss",
            metric_value=0.5,
        )
        assert alert.metric_name == "loss"
        assert alert.metric_value == 0.5


class TestMonitorConfig:
    def test_defaults(self):
        cfg = MonitorConfig()
        assert cfg.loss_spike_threshold == 5.0
        assert cfg.loss_spike_window == 50
        assert cfg.grad_norm_threshold == 100.0
        assert cfg.grad_vanish_threshold == 1e-7
        assert cfg.check_nan is True
        assert cfg.lr_change_threshold == 10.0
        assert cfg.step_time_spike_threshold == 3.0
        assert cfg.step_time_window == 20
        assert cfg.plateau_patience == 100
        assert cfg.plateau_min_delta == 1e-5
        assert cfg.log_interval == 1
        assert cfg.alert_callbacks == []

    def test_custom(self):
        cfg = MonitorConfig(loss_spike_threshold=3.0, grad_norm_threshold=50.0)
        assert cfg.loss_spike_threshold == 3.0
        assert cfg.grad_norm_threshold == 50.0


class TestTrainingReport:
    def _make_report(self, alerts=None, steps=100, health=0.9):
        return TrainingReport(
            total_steps=steps,
            alerts=alerts or [],
            metrics_summary={"loss": {"min": 0.1, "max": 1.0, "mean": 0.5, "last": 0.2}},
            health_score=health,
        )

    def test_healthy(self):
        report = self._make_report()
        assert report.is_healthy
        assert report.n_warnings == 0
        assert report.n_critical == 0

    def test_warnings(self):
        alerts = [
            Alert(step=1, severity=AlertSeverity.WARNING, detector="x", message="y"),
            Alert(step=2, severity=AlertSeverity.WARNING, detector="x", message="y"),
        ]
        report = self._make_report(alerts=alerts)
        assert report.n_warnings == 2
        assert report.n_critical == 0
        assert report.is_healthy  # warnings don't make it unhealthy

    def test_critical(self):
        alerts = [
            Alert(step=1, severity=AlertSeverity.CRITICAL, detector="x", message="y"),
        ]
        report = self._make_report(alerts=alerts)
        assert report.n_critical == 1
        assert not report.is_healthy

    def test_mixed_alerts(self):
        alerts = [
            Alert(step=1, severity=AlertSeverity.WARNING, detector="x", message="y"),
            Alert(step=2, severity=AlertSeverity.CRITICAL, detector="x", message="y"),
            Alert(step=3, severity=AlertSeverity.INFO, detector="x", message="y"),
        ]
        report = self._make_report(alerts=alerts)
        assert report.n_warnings == 1
        assert report.n_critical == 1
        assert not report.is_healthy


class TestTrainpulseError:
    def test_is_exception(self):
        assert issubclass(TrainpulseError, Exception)
        err = TrainpulseError("test error")
        assert str(err) == "test error"
