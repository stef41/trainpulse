"""Anomaly detectors for training metrics."""

from __future__ import annotations

import math

from trainpulse._types import Alert, AlertSeverity


class RollingWindow:
    """Fixed-size rolling window of float values."""

    def __init__(self, size: int) -> None:
        self._size = max(size, 1)
        self._values: list[float] = []

    def add(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self._size:
            self._values.pop(0)

    @property
    def values(self) -> list[float]:
        return list(self._values)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((v - m) ** 2 for v in self._values) / len(self._values))

    @property
    def is_full(self) -> bool:
        return len(self._values) >= self._size

    def __len__(self) -> int:
        return len(self._values)


class NaNDetector:
    """Detect NaN or Inf values in metrics."""

    def check(self, step: int, name: str, value: float) -> Alert | None:
        if math.isnan(value):
            return Alert(
                step=step,
                severity=AlertSeverity.CRITICAL,
                detector="nan_detector",
                message=f"{name} is NaN",
                metric_name=name,
                metric_value=value,
            )
        if math.isinf(value):
            return Alert(
                step=step,
                severity=AlertSeverity.CRITICAL,
                detector="nan_detector",
                message=f"{name} is Inf",
                metric_name=name,
                metric_value=value,
            )
        return None


class LossSpikeDetector:
    """Detect sudden spikes in loss values."""

    def __init__(self, threshold: float = 5.0, window_size: int = 50) -> None:
        self._threshold = threshold
        self._window = RollingWindow(window_size)

    def check(self, step: int, value: float) -> Alert | None:
        if self._window.is_full:
            avg = self._window.mean
            if avg > 0 and value > avg * self._threshold:
                alert = Alert(
                    step=step,
                    severity=AlertSeverity.WARNING,
                    detector="loss_spike",
                    message=f"Loss spike: {value:.4f} ({value/avg:.1f}x rolling avg {avg:.4f})",
                    metric_name="loss",
                    metric_value=value,
                )
                self._window.add(value)
                return alert

        self._window.add(value)
        return None


class GradientDetector:
    """Detect gradient explosion and vanishing."""

    def __init__(
        self,
        explosion_threshold: float = 100.0,
        vanish_threshold: float = 1e-7,
    ) -> None:
        self._explosion = explosion_threshold
        self._vanish = vanish_threshold

    def check(self, step: int, grad_norm: float) -> Alert | None:
        if grad_norm > self._explosion:
            return Alert(
                step=step,
                severity=AlertSeverity.CRITICAL,
                detector="gradient",
                message=f"Gradient explosion: norm={grad_norm:.4f} (threshold={self._explosion})",
                metric_name="gradient_norm",
                metric_value=grad_norm,
            )
        if 0 < grad_norm < self._vanish:
            return Alert(
                step=step,
                severity=AlertSeverity.WARNING,
                detector="gradient",
                message=f"Vanishing gradient: norm={grad_norm:.2e} (threshold={self._vanish:.2e})",
                metric_name="gradient_norm",
                metric_value=grad_norm,
            )
        return None


class LRDetector:
    """Detect suspicious learning rate changes."""

    def __init__(self, change_threshold: float = 10.0) -> None:
        self._threshold = change_threshold
        self._prev_lr: float | None = None

    def check(self, step: int, lr: float) -> Alert | None:
        if self._prev_lr is not None and self._prev_lr > 0 and lr > 0:
            ratio = max(lr / self._prev_lr, self._prev_lr / lr)
            if ratio > self._threshold:
                alert = Alert(
                    step=step,
                    severity=AlertSeverity.WARNING,
                    detector="learning_rate",
                    message=f"LR changed {ratio:.1f}x in one step ({self._prev_lr:.2e} → {lr:.2e})",
                    metric_name="learning_rate",
                    metric_value=lr,
                )
                self._prev_lr = lr
                return alert
        self._prev_lr = lr
        return None


class PlateauDetector:
    """Detect loss plateaus (no improvement for N steps)."""

    def __init__(self, patience: int = 100, min_delta: float = 1e-5) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._best_loss: float | None = None
        self._steps_without_improvement = 0
        self._alerted = False

    def check(self, step: int, loss: float) -> Alert | None:
        if self._best_loss is None:
            self._best_loss = loss
            return None

        if loss < self._best_loss - self._min_delta:
            self._best_loss = loss
            self._steps_without_improvement = 0
            self._alerted = False
            return None

        self._steps_without_improvement += 1

        if self._steps_without_improvement >= self._patience and not self._alerted:
            self._alerted = True
            return Alert(
                step=step,
                severity=AlertSeverity.WARNING,
                detector="plateau",
                message=f"Loss plateau: no improvement for {self._steps_without_improvement} steps (best={self._best_loss:.6f})",
                metric_name="loss",
                metric_value=loss,
            )
        return None


class StepTimeDetector:
    """Detect unusually slow training steps."""

    def __init__(self, threshold: float = 3.0, window_size: int = 20) -> None:
        self._threshold = threshold
        self._window = RollingWindow(window_size)

    def check(self, step: int, step_time: float) -> Alert | None:
        if self._window.is_full:
            avg = self._window.mean
            if avg > 0 and step_time > avg * self._threshold:
                alert = Alert(
                    step=step,
                    severity=AlertSeverity.WARNING,
                    detector="step_time",
                    message=f"Slow step: {step_time:.2f}s ({step_time/avg:.1f}x avg {avg:.2f}s)",
                    metric_name="step_time",
                    metric_value=step_time,
                )
                self._window.add(step_time)
                return alert

        self._window.add(step_time)
        return None
