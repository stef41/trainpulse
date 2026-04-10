"""Tests for monitor module."""

import pytest

from trainpulse._types import Alert, AlertSeverity, MetricType, MonitorConfig
from trainpulse.monitor import Monitor, _compute_health_score, _infer_metric_type


class TestInferMetricType:
    def test_loss(self):
        assert _infer_metric_type("loss") == MetricType.LOSS
        assert _infer_metric_type("train_loss") == MetricType.LOSS
        assert _infer_metric_type("nll_loss") == MetricType.LOSS
        assert _infer_metric_type("perplexity") == MetricType.LOSS

    def test_gradient(self):
        assert _infer_metric_type("grad_norm") == MetricType.GRADIENT_NORM
        assert _infer_metric_type("gradient_norm") == MetricType.GRADIENT_NORM

    def test_lr(self):
        assert _infer_metric_type("lr") == MetricType.LEARNING_RATE
        assert _infer_metric_type("learning_rate") == MetricType.LEARNING_RATE

    def test_step_time(self):
        assert _infer_metric_type("step_time") == MetricType.STEP_TIME
        assert _infer_metric_type("iteration_time") == MetricType.STEP_TIME

    def test_memory(self):
        assert _infer_metric_type("memory_used") == MetricType.MEMORY_USED
        assert _infer_metric_type("gpu_mem") == MetricType.MEMORY_USED

    def test_custom(self):
        assert _infer_metric_type("accuracy") == MetricType.CUSTOM
        assert _infer_metric_type("f1_score") == MetricType.CUSTOM


class TestComputeHealthScore:
    def test_no_alerts(self):
        assert _compute_health_score([], 100) == 1.0

    def test_zero_steps(self):
        assert _compute_health_score([], 0) == 1.0

    def test_warnings(self):
        alerts = [
            Alert(step=1, severity=AlertSeverity.WARNING, detector="x", message="y")
        ]
        score = _compute_health_score(alerts, 100)
        assert score == pytest.approx(0.95)

    def test_critical(self):
        alerts = [
            Alert(step=1, severity=AlertSeverity.CRITICAL, detector="x", message="y")
        ]
        score = _compute_health_score(alerts, 100)
        assert score == pytest.approx(0.85)

    def test_many_alerts(self):
        alerts = [
            Alert(step=i, severity=AlertSeverity.CRITICAL, detector="x", message="y")
            for i in range(10)
        ]
        score = _compute_health_score(alerts, 100)
        # 10 * 0.15 = 1.5, clamped to 0.0
        assert score == pytest.approx(0.0)

    def test_info_alerts(self):
        alerts = [
            Alert(step=1, severity=AlertSeverity.INFO, detector="x", message="y")
        ]
        score = _compute_health_score(alerts, 100)
        assert score == pytest.approx(0.99)


class TestMonitor:
    def test_empty_report(self):
        m = Monitor()
        report = m.report()
        assert report.total_steps == 0
        assert report.alerts == []
        assert report.metrics_summary == {}
        assert report.health_score == 1.0

    def test_log_loss(self):
        m = Monitor()
        alerts = m.log("loss", 0, 1.0)
        assert alerts == []
        report = m.report()
        assert "loss" in report.metrics_summary
        assert report.metrics_summary["loss"]["last"] == 1.0

    def test_log_tracks_step_count(self):
        m = Monitor()
        m.log("loss", 0, 1.0)
        m.log("loss", 5, 0.5)
        m.log("loss", 10, 0.3)
        report = m.report()
        assert report.total_steps == 11  # max step + 1

    def test_nan_detection(self):
        m = Monitor()
        alerts = m.log("loss", 5, float("nan"))
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert "NaN" in alerts[0].message

    def test_nan_detection_disabled(self):
        cfg = MonitorConfig(check_nan=False)
        m = Monitor(cfg)
        alerts = m.log("loss", 5, float("nan"))
        # No NaN alert (but also no spike/plateau because window isn't full)
        nan_alerts = [a for a in alerts if a.detector == "nan_detector"]
        assert len(nan_alerts) == 0

    def test_loss_spike(self):
        cfg = MonitorConfig(loss_spike_threshold=3.0, loss_spike_window=5)
        m = Monitor(cfg)
        for i in range(5):
            m.log("loss", i, 1.0)
        alerts = m.log("loss", 5, 10.0)
        spike_alerts = [a for a in alerts if a.detector == "loss_spike"]
        assert len(spike_alerts) == 1

    def test_gradient_explosion(self):
        cfg = MonitorConfig(grad_norm_threshold=50.0)
        m = Monitor(cfg)
        alerts = m.log("grad_norm", 0, 100.0)
        grad_alerts = [a for a in alerts if a.detector == "gradient"]
        assert len(grad_alerts) == 1
        assert "explosion" in grad_alerts[0].message.lower()

    def test_gradient_vanishing(self):
        cfg = MonitorConfig(grad_vanish_threshold=1e-5)
        m = Monitor(cfg)
        alerts = m.log("grad_norm", 0, 1e-8)
        grad_alerts = [a for a in alerts if a.detector == "gradient"]
        assert len(grad_alerts) == 1
        assert "vanish" in grad_alerts[0].message.lower()

    def test_lr_change(self):
        cfg = MonitorConfig(lr_change_threshold=5.0)
        m = Monitor(cfg)
        m.log("learning_rate", 0, 0.001)
        alerts = m.log("learning_rate", 1, 0.1)  # 100x change
        lr_alerts = [a for a in alerts if a.detector == "learning_rate"]
        assert len(lr_alerts) == 1

    def test_alert_callback(self):
        received = []
        cfg = MonitorConfig(
            alert_callbacks=[received.append],
            check_nan=True,
        )
        m = Monitor(cfg)
        m.log("loss", 0, float("nan"))
        assert len(received) == 1
        assert received[0].severity == AlertSeverity.CRITICAL

    def test_multiple_metrics(self):
        m = Monitor()
        m.log("loss", 0, 1.0)
        m.log("grad_norm", 0, 5.0)
        m.log("learning_rate", 0, 0.001)
        report = m.report()
        assert len(report.metrics_summary) == 3
        assert "loss" in report.metrics_summary
        assert "grad_norm" in report.metrics_summary
        assert "learning_rate" in report.metrics_summary

    def test_metrics_summary_stats(self):
        m = Monitor()
        m.log("loss", 0, 1.0)
        m.log("loss", 1, 0.5)
        m.log("loss", 2, 0.3)
        report = m.report()
        summary = report.metrics_summary["loss"]
        assert summary["min"] == pytest.approx(0.3)
        assert summary["max"] == pytest.approx(1.0)
        assert summary["mean"] == pytest.approx(0.6)
        assert summary["last"] == pytest.approx(0.3)
        assert summary["count"] == 3.0

    def test_inf_values_excluded_from_summary(self):
        m = Monitor(MonitorConfig(check_nan=False))
        m.log("loss", 0, 1.0)
        m.log("loss", 1, float("inf"))
        m.log("loss", 2, 0.5)
        report = m.report()
        # Inf should be excluded from min/max/mean
        summary = report.metrics_summary["loss"]
        assert summary["min"] == pytest.approx(0.5)
        assert summary["max"] == pytest.approx(1.0)

    def test_all_nan_values_summary(self):
        m = Monitor(MonitorConfig(check_nan=False))
        m.log("loss", 0, float("nan"))
        m.log("loss", 1, float("nan"))
        report = m.report()
        summary = report.metrics_summary["loss"]
        assert summary["min"] == 0.0
        assert summary["count"] == 2.0

    def test_step_timing(self):
        m = Monitor()
        m.step_start()
        # Immediately end (will be very fast)
        alerts = m.step_end(0)
        assert alerts == []  # Window not full yet
        report = m.report()
        assert "step_time" in report.metrics_summary

    def test_step_end_without_start(self):
        m = Monitor()
        alerts = m.step_end(0)
        assert alerts == []

    def test_reset(self):
        m = Monitor()
        m.log("loss", 0, 1.0)
        m.log("loss", 1, float("nan"))
        assert len(m.alerts) > 0
        m.reset()
        report = m.report()
        assert report.total_steps == 0
        assert report.alerts == []
        assert report.metrics_summary == {}

    def test_alerts_property(self):
        m = Monitor()
        m.log("loss", 0, float("nan"))
        alerts = m.alerts
        assert len(alerts) == 1
        # Should be a copy
        alerts.clear()
        assert len(m.alerts) == 1

    def test_snapshots_property(self):
        m = Monitor()
        m.log("loss", 0, 1.0)
        snaps = m.snapshots
        assert "loss" in snaps
        assert len(snaps["loss"]) == 1

    def test_config_property(self):
        cfg = MonitorConfig(loss_spike_threshold=7.0)
        m = Monitor(cfg)
        assert m.config.loss_spike_threshold == 7.0

    def test_custom_metric_no_detector(self):
        m = Monitor()
        # Custom metrics should not trigger any detector (except NaN)
        alerts = m.log("accuracy", 0, 0.95)
        assert alerts == []

    def test_metadata_stored(self):
        m = Monitor()
        m.log("loss", 0, 1.0, epoch=1, batch_size=32)
        snaps = m.snapshots["loss"]
        assert snaps[0].metadata == {"epoch": 1, "batch_size": 32}

    def test_plateau_detection(self):
        cfg = MonitorConfig(plateau_patience=5, plateau_min_delta=0.01)
        m = Monitor(cfg)
        # Log same loss for patience+1 steps
        for i in range(6):
            m.log("loss", i, 1.0)
        report = m.report()
        plateau_alerts = [a for a in report.alerts if a.detector == "plateau"]
        assert len(plateau_alerts) == 1
