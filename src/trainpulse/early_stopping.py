"""Early stopping recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class EarlyStopResult:
    """Result from a single early-stopping step."""

    step: int
    value: float
    should_stop: bool
    improved: bool
    best_value: float
    best_step: int


class EarlyStopping:
    """Track a metric and recommend when to stop training.

    Parameters
    ----------
    patience:
        Number of steps without improvement before recommending stop.
    min_delta:
        Minimum change from the best value to count as an improvement.
    metric:
        Name of the metric being tracked (informational).
    mode:
        ``"min"`` treats lower values as better; ``"max"`` treats higher as better.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        metric: str = "loss",
        mode: Literal["min", "max"] = "min",
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self._patience = patience
        self._min_delta = abs(min_delta)
        self._metric = metric
        self._mode = mode

        self._best_value: float | None = None
        self._best_step: int = 0
        self._current_step: int = 0
        self._steps_without_improvement: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, value: float) -> EarlyStopResult:
        """Record a new metric value and return the stop recommendation."""
        step_idx = self._current_step
        self._current_step += 1

        improved = self._is_improvement(value)
        if improved:
            self._best_value = value
            self._best_step = step_idx
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        return EarlyStopResult(
            step=step_idx,
            value=value,
            should_stop=self._steps_without_improvement >= self._patience,
            improved=improved,
            best_value=self._best_value,  # type: ignore[arg-type]
            best_step=self._best_step,
        )

    @property
    def should_stop(self) -> bool:
        """``True`` if patience has been exhausted."""
        return self._steps_without_improvement >= self._patience

    @property
    def best_value(self) -> float | None:
        """Best metric value observed so far, or ``None`` if no values logged."""
        return self._best_value

    @property
    def best_step(self) -> int:
        """Step index where the best value was recorded."""
        return self._best_step

    @property
    def steps_without_improvement(self) -> int:
        return self._steps_without_improvement

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self._mode == "min":
            return value < self._best_value - self._min_delta
        return value > self._best_value + self._min_delta


def recommend_patience(loss_history: list[float]) -> int:
    """Heuristic that recommends a patience value from a loss curve.

    The recommendation is based on the typical gap (in steps) between
    successive improvements and the relative amplitude of those
    improvements.  A noisier curve with infrequent improvements gets a
    higher patience so the run isn't killed prematurely.

    Parameters
    ----------
    loss_history:
        Sequence of loss values (one per step), assumed to be in temporal
        order.

    Returns
    -------
    int
        Recommended patience (always >= 1).
    """
    if len(loss_history) < 2:
        return 5  # sensible default

    # Find steps where a new minimum was reached.
    improvement_gaps: list[int] = []
    best = loss_history[0]
    last_improvement_step = 0

    for i in range(1, len(loss_history)):
        if loss_history[i] < best:
            gap = i - last_improvement_step
            improvement_gaps.append(gap)
            best = loss_history[i]
            last_improvement_step = i

    if not improvement_gaps:
        # No improvements at all — recommend a conservative patience equal
        # to half the history length (at least 5).
        return max(5, len(loss_history) // 2)

    mean_gap = sum(improvement_gaps) / len(improvement_gaps)
    max_gap = max(improvement_gaps)

    # Blend average and max gap so patience tolerates occasional long dry
    # spells.  The multiplier (1.5×) provides a safety margin.
    patience = int(0.5 * mean_gap + 0.5 * max_gap) + 1
    patience = int(patience * 1.5)

    # Clamp to a reasonable range.
    return max(3, min(patience, len(loss_history)))
