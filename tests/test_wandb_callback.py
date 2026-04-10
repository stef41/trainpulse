"""Tests for Weights & Biases callback integration."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from trainpulse._types import Alert, AlertSeverity
from trainpulse.wandb_callback import WandbCallback


def _make_mock_wandb() -> MagicMock:
    """Create a mock wandb module with the methods we use."""
    mock = MagicMock(spec=ModuleType)
    mock.init = MagicMock(return_value=MagicMock())
    mock.log = MagicMock()
    mock.alert = MagicMock()
    mock.finish = MagicMock()
    return mock


class TestWandbCallbackInit:
    """Tests for WandbCallback construction and lazy init."""

    def test_default_params(self) -> None:
        cb = WandbCallback()
        assert cb._project == "trainpulse"
        assert cb._run_name is None
        assert cb._config is None
        assert cb._enabled is True
        assert cb._wandb is None

    def test_custom_params(self) -> None:
        cb = WandbCallback(project="myproj", run_name="run-1", config={"lr": 1e-4}, enabled=False)
        assert cb._project == "myproj"
        assert cb._run_name == "run-1"
        assert cb._config == {"lr": 1e-4}
        assert cb._enabled is False

    def test_wandb_not_imported_until_needed(self) -> None:
        cb = WandbCallback()
        assert cb._wandb is None
        assert cb._run is None

    def test_import_error_when_wandb_missing(self) -> None:
        cb = WandbCallback()
        with patch.dict(sys.modules, {"wandb": None}):
            with pytest.raises(ImportError, match="pip install wandb"):
                cb._ensure_wandb()


class TestWandbCallbackOnStep:
    """Tests for on_step metric logging."""

    def test_on_step_logs_metrics(self) -> None:
        mock_wandb = _make_mock_wandb()
        cb = WandbCallback()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            cb.on_step(10, {"loss": 0.5, "lr": 1e-4})

        mock_wandb.init.assert_called_once()
        mock_wandb.log.assert_called_once_with({"loss": 0.5, "lr": 1e-4}, step=10)

    def test_on_step_disabled_noop(self) -> None:
        cb = WandbCallback(enabled=False)
        # Should not raise or try to import wandb
        cb.on_step(1, {"loss": 1.0})
        assert cb._wandb is None

    def test_on_step_multiple_calls(self) -> None:
        mock_wandb = _make_mock_wandb()
        cb = WandbCallback()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            cb.on_step(1, {"loss": 1.0})
            cb.on_step(2, {"loss": 0.8})

        # init should only be called once
        mock_wandb.init.assert_called_once()
        assert mock_wandb.log.call_count == 2


class TestWandbCallbackOnAlert:
    """Tests for on_alert."""

    def test_on_alert_sends_wandb_alert(self) -> None:
        mock_wandb = _make_mock_wandb()
        cb = WandbCallback()
        alert = Alert(
            step=42,
            severity=AlertSeverity.CRITICAL,
            detector="loss_spike",
            message="Loss spiked 10x above rolling average",
            metric_name="loss",
            metric_value=99.0,
        )
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            cb.on_alert(alert)

        mock_wandb.alert.assert_called_once_with(
            title="[CRITICAL] loss_spike",
            text="Loss spiked 10x above rolling average",
        )

    def test_on_alert_disabled_noop(self) -> None:
        cb = WandbCallback(enabled=False)
        alert = Alert(step=1, severity=AlertSeverity.WARNING, detector="test", message="msg")
        cb.on_alert(alert)
        assert cb._wandb is None


class TestWandbCallbackOnEpochEnd:
    """Tests for on_epoch_end."""

    def test_on_epoch_end_logs_prefixed_metrics(self) -> None:
        mock_wandb = _make_mock_wandb()
        cb = WandbCallback()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            cb.on_epoch_end(3, {"loss": 0.3, "accuracy": 0.95})

        expected = {"epoch/loss": 0.3, "epoch/accuracy": 0.95, "epoch": 3}
        mock_wandb.log.assert_called_once_with(expected)

    def test_on_epoch_end_disabled_noop(self) -> None:
        cb = WandbCallback(enabled=False)
        cb.on_epoch_end(1, {"loss": 1.0})
        assert cb._wandb is None


class TestWandbCallbackFinish:
    """Tests for finish."""

    def test_finish_calls_wandb_finish(self) -> None:
        mock_wandb = _make_mock_wandb()
        cb = WandbCallback()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            cb.on_step(1, {"loss": 1.0})  # trigger init
            cb.finish()

        mock_wandb.finish.assert_called_once()

    def test_finish_without_init_is_noop(self) -> None:
        cb = WandbCallback()
        cb.finish()  # should not raise
        assert cb._wandb is None

    def test_finish_disabled_is_noop(self) -> None:
        cb = WandbCallback(enabled=False)
        cb.finish()
        assert cb._wandb is None


class TestWandbCallbackInitParams:
    """Tests for init params passed to wandb.init."""

    def test_init_passes_project_and_config(self) -> None:
        mock_wandb = _make_mock_wandb()
        cb = WandbCallback(project="test-proj", run_name="run-42", config={"lr": 0.001})
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            cb.on_step(0, {"loss": 1.0})

        mock_wandb.init.assert_called_once_with(
            project="test-proj",
            name="run-42",
            config={"lr": 0.001},
        )
