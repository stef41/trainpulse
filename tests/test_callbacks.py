"""Tests for callbacks module."""

from trainpulse._types import MonitorConfig
from trainpulse.callbacks import TrainingCallback


class TestTrainingCallback:
    def test_basic_usage(self):
        cb = TrainingCallback()
        cb.on_step_begin(0)
        cb.on_step_end(0, loss=1.0, grad_norm=5.0, lr=0.001)
        report = cb.report()
        assert report.total_steps == 1
        assert "loss" in report.metrics_summary
        assert "grad_norm" in report.metrics_summary
        assert "learning_rate" in report.metrics_summary

    def test_step_timing(self):
        cb = TrainingCallback()
        cb.on_step_begin(0)
        cb.on_step_end(0, loss=1.0)
        report = cb.report()
        assert "step_time" in report.metrics_summary

    def test_optional_metrics(self):
        cb = TrainingCallback()
        cb.on_step_end(0, loss=1.0)
        report = cb.report()
        assert "loss" in report.metrics_summary
        assert "grad_norm" not in report.metrics_summary
        assert "learning_rate" not in report.metrics_summary

    def test_extra_metrics(self):
        cb = TrainingCallback()
        cb.on_step_end(0, loss=1.0, accuracy=0.95, f1=0.9)
        report = cb.report()
        assert "accuracy" in report.metrics_summary
        assert "f1" in report.metrics_summary

    def test_nan_detection(self):
        cb = TrainingCallback()
        cb.on_step_end(0, loss=float("nan"))
        report = cb.report()
        assert report.n_critical > 0

    def test_custom_config(self):
        cfg = MonitorConfig(loss_spike_threshold=2.0, loss_spike_window=3)
        cb = TrainingCallback(config=cfg)
        assert cb.monitor.config.loss_spike_threshold == 2.0

    def test_multi_step(self):
        cb = TrainingCallback()
        for step in range(10):
            cb.on_step_begin(step)
            cb.on_step_end(step, loss=1.0 - step * 0.05)
        report = cb.report()
        assert report.total_steps == 10
        assert report.is_healthy

    def test_gradient_explosion_via_callback(self):
        cfg = MonitorConfig(grad_norm_threshold=50.0)
        cb = TrainingCallback(config=cfg)
        cb.on_step_end(0, grad_norm=100.0)
        report = cb.report()
        grad_alerts = [a for a in report.alerts if a.detector == "gradient"]
        assert len(grad_alerts) == 1

    def test_no_metrics(self):
        cb = TrainingCallback()
        cb.on_step_begin(0)
        cb.on_step_end(0)
        report = cb.report()
        # Only step_time should be recorded
        assert "step_time" in report.metrics_summary
        assert "loss" not in report.metrics_summary

    def test_monitor_access(self):
        cb = TrainingCallback()
        assert cb.monitor is not None
        cb.on_step_end(0, loss=0.5)
        assert len(cb.monitor.snapshots["loss"]) == 1
