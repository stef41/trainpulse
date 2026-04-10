"""Tests for report module."""

import json

from trainpulse._types import Alert, AlertSeverity, TrainingReport
from trainpulse.report import (
    format_report_rich,
    format_report_text,
    load_json,
    report_to_dict,
    save_json,
)


def _make_report(
    steps=100,
    health=0.85,
    alerts=None,
    metrics=None,
):
    if alerts is None:
        alerts = [
            Alert(
                step=10,
                severity=AlertSeverity.WARNING,
                detector="loss_spike",
                message="Loss spike: 5.0 (5.0x rolling avg 1.0)",
                metric_name="loss",
                metric_value=5.0,
            ),
            Alert(
                step=50,
                severity=AlertSeverity.CRITICAL,
                detector="nan_detector",
                message="loss is NaN",
                metric_name="loss",
                metric_value=float("nan"),
            ),
        ]
    if metrics is None:
        metrics = {
            "loss": {"min": 0.1, "max": 5.0, "mean": 0.8, "last": 0.2, "count": 100.0},
            "grad_norm": {"min": 0.01, "max": 50.0, "mean": 2.5, "last": 1.5, "count": 100.0},
            "learning_rate": {"min": 1e-5, "max": 1e-3, "mean": 5e-4, "last": 1e-4, "count": 100.0},
        }
    return TrainingReport(
        total_steps=steps,
        alerts=alerts,
        metrics_summary=metrics,
        health_score=health,
    )


class TestReportToDict:
    def test_basic(self):
        report = _make_report()
        d = report_to_dict(report)
        assert d["total_steps"] == 100
        assert d["health_score"] == 0.85
        assert d["is_healthy"] is False
        assert d["n_warnings"] == 1
        assert d["n_critical"] == 1
        assert len(d["alerts"]) == 2
        assert "loss" in d["metrics_summary"]

    def test_alert_fields(self):
        report = _make_report()
        d = report_to_dict(report)
        alert = d["alerts"][0]
        assert alert["step"] == 10
        assert alert["severity"] == "warning"
        assert alert["detector"] == "loss_spike"
        assert alert["metric_name"] == "loss"

    def test_empty_report(self):
        report = TrainingReport(
            total_steps=0, alerts=[], metrics_summary={}, health_score=1.0
        )
        d = report_to_dict(report)
        assert d["total_steps"] == 0
        assert d["alerts"] == []
        assert d["metrics_summary"] == {}
        assert d["health_score"] == 1.0

    def test_health_score_rounded(self):
        report = _make_report(health=0.123456789)
        d = report_to_dict(report)
        assert d["health_score"] == 0.1235


class TestSaveLoadJson:
    def test_roundtrip(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.json"
        save_json(report, path)
        assert path.exists()
        data = load_json(path)
        assert data["total_steps"] == 100
        assert len(data["alerts"]) == 2

    def test_valid_json(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.json"
        save_json(report, path)
        # NaN values in alerts should be serializable
        # Actually, NaN in metric_value for the NaN alert...
        # json.dumps handles NaN by default in Python (outputs NaN literal)
        text = path.read_text()
        # Should be parseable
        data = json.loads(text)
        assert data is not None

    def test_empty_report(self, tmp_path):
        report = TrainingReport(
            total_steps=0, alerts=[], metrics_summary={}, health_score=1.0
        )
        path = tmp_path / "empty.json"
        save_json(report, path)
        data = load_json(path)
        assert data["total_steps"] == 0


class TestFormatReportText:
    def test_contains_header(self):
        report = _make_report()
        text = format_report_text(report)
        assert "TRAINING HEALTH REPORT" in text

    def test_contains_stats(self):
        report = _make_report()
        text = format_report_text(report)
        assert "100" in text  # total steps
        assert "UNHEALTHY" in text

    def test_contains_metrics(self):
        report = _make_report()
        text = format_report_text(report)
        assert "loss" in text
        assert "grad_norm" in text

    def test_contains_alerts(self):
        report = _make_report()
        text = format_report_text(report)
        assert "ALERTS" in text
        assert "Loss spike" in text
        assert "NaN" in text

    def test_healthy_status(self):
        report = _make_report(alerts=[], health=1.0)
        text = format_report_text(report)
        assert "HEALTHY" in text

    def test_critical_prefix(self):
        report = _make_report()
        text = format_report_text(report)
        assert "!!" in text  # critical prefix

    def test_no_alerts_section_when_empty(self):
        report = _make_report(alerts=[], health=1.0)
        text = format_report_text(report)
        assert "ALERTS" not in text

    def test_no_metrics_section_when_empty(self):
        report = TrainingReport(
            total_steps=10, alerts=[], metrics_summary={}, health_score=1.0
        )
        text = format_report_text(report)
        assert "METRICS SUMMARY" not in text


class TestFormatReportRich:
    def test_returns_string(self):
        report = _make_report()
        text = format_report_rich(report)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_contains_info(self):
        report = _make_report()
        text = format_report_rich(report)
        assert "trainpulse" in text or "100" in text

    def test_falls_back_to_text(self):
        # Even if rich is available, the function should work
        report = _make_report()
        text = format_report_rich(report)
        assert len(text) > 0
