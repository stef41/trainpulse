"""Tests for CLI module."""

import json

import pytest
from click.testing import CliRunner

from trainpulse.cli import _build_cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli():
    return _build_cli()


@pytest.fixture
def sample_log(tmp_path):
    """Create a sample JSONL training log."""
    lines = []
    for step in range(50):
        entry = {
            "step": step,
            "loss": 1.0 - step * 0.01 + (0.5 if step == 25 else 0),
            "grad_norm": 2.0 + (150.0 if step == 30 else 0),
            "learning_rate": 0.001 * (1 - step / 50),
        }
        lines.append(json.dumps(entry))
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def nan_log(tmp_path):
    """Create a log with NaN values."""
    lines = []
    for step in range(10):
        entry = {"step": step, "loss": float("nan") if step == 5 else 1.0}
        lines.append(json.dumps(entry))
    path = tmp_path / "nan.jsonl"
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def empty_log(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    return path


@pytest.fixture
def saved_report(tmp_path):
    """Create a saved JSON report."""
    report_data = {
        "total_steps": 100,
        "health_score": 0.85,
        "is_healthy": False,
        "n_warnings": 1,
        "n_critical": 1,
        "alerts": [
            {
                "step": 10,
                "severity": "warning",
                "detector": "loss_spike",
                "message": "Loss spike detected",
                "metric_name": "loss",
                "metric_value": 5.0,
            }
        ],
        "metrics_summary": {
            "loss": {"min": 0.1, "max": 5.0, "mean": 0.8, "last": 0.2, "count": 100.0}
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report_data))
    return path


class TestAnalyzeCommand:
    def test_basic(self, runner, cli, sample_log):
        result = runner.invoke(cli, ["analyze", str(sample_log)])
        assert result.exit_code == 0
        assert "TRAINING HEALTH REPORT" in result.output or "trainpulse" in result.output

    def test_json_output(self, runner, cli, sample_log, tmp_path):
        out = tmp_path / "out.json"
        result = runner.invoke(cli, ["analyze", str(sample_log), "--json-out", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "total_steps" in data

    def test_nan_detection(self, runner, cli, nan_log):
        result = runner.invoke(cli, ["analyze", str(nan_log)])
        assert result.exit_code == 0
        # Should detect NaN
        assert "NaN" in result.output or "CRITICAL" in result.output or "UNHEALTHY" in result.output

    def test_empty_log(self, runner, cli, empty_log):
        result = runner.invoke(cli, ["analyze", str(empty_log)])
        assert result.exit_code == 1
        assert "No log entries" in result.output

    def test_custom_keys(self, runner, cli, tmp_path):
        lines = [
            json.dumps({"iteration": i, "train_loss": 1.0 - i * 0.01})
            for i in range(20)
        ]
        path = tmp_path / "custom.jsonl"
        path.write_text("\n".join(lines))
        result = runner.invoke(
            cli,
            ["analyze", str(path), "--step-key", "iteration", "--loss-key", "train_loss"],
        )
        assert result.exit_code == 0

    def test_nonexistent_file(self, runner, cli):
        result = runner.invoke(cli, ["analyze", "/nonexistent/file.jsonl"])
        assert result.exit_code != 0


class TestShowCommand:
    def test_basic(self, runner, cli, saved_report):
        result = runner.invoke(cli, ["show", str(saved_report)])
        assert result.exit_code == 0
        assert len(result.output) > 0

    def test_nonexistent_file(self, runner, cli):
        result = runner.invoke(cli, ["show", "/nonexistent/report.json"])
        assert result.exit_code != 0


class TestCLIHelp:
    def test_main_help(self, runner, cli):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "trainpulse" in result.output

    def test_analyze_help(self, runner, cli):
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "LOG_FILE" in result.output

    def test_show_help(self, runner, cli):
        result = runner.invoke(cli, ["show", "--help"])
        assert result.exit_code == 0
