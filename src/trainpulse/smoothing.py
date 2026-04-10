"""Metric smoothing and noise reduction for training curves."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SmoothingConfig:
    """Configuration for metric smoothing."""

    method: str = "ema"
    window_size: int = 10
    alpha: float = 0.3


@dataclass
class SmoothedSeries:
    """Result of smoothing a metric series."""

    original: list[float] = field(default_factory=list)
    smoothed: list[float] = field(default_factory=list)
    method: str = ""
    residuals: list[float] = field(default_factory=list)


class MetricSmoother:
    """Smooth noisy training metrics with various algorithms."""

    def __init__(self, config: SmoothingConfig | None = None) -> None:
        self.config = config or SmoothingConfig()

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def smooth(self, values: list[float]) -> SmoothedSeries:
        """Apply the configured smoothing method."""
        method_name = self.config.method.lower()
        dispatch = {
            "ema": lambda v: self.ema(v),
            "sma": lambda v: self.sma(v),
            "gaussian": lambda v: self.gaussian(v),
            "median": lambda v: self.median_filter(v),
        }
        fn = dispatch.get(method_name)
        if fn is None:
            raise ValueError(f"Unknown smoothing method: {method_name!r}")
        smoothed = fn(values)
        residuals = [o - s for o, s in zip(values, smoothed)]
        return SmoothedSeries(
            original=list(values),
            smoothed=smoothed,
            method=method_name,
            residuals=residuals,
        )

    # ------------------------------------------------------------------
    # Smoothing algorithms
    # ------------------------------------------------------------------

    def ema(self, values: list[float], alpha: float | None = None) -> list[float]:
        """Exponential moving average."""
        if not values:
            return []
        a = alpha if alpha is not None else self.config.alpha
        result = [values[0]]
        for v in values[1:]:
            result.append(a * v + (1 - a) * result[-1])
        return result

    def sma(self, values: list[float], window: int | None = None) -> list[float]:
        """Simple moving average (centred, same length as input)."""
        if not values:
            return []
        w = window if window is not None else self.config.window_size
        w = max(1, w)
        n = len(values)
        result: list[float] = []
        half = w // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            result.append(sum(values[lo:hi]) / (hi - lo))
        return result

    def gaussian(self, values: list[float], sigma: float = 1.0) -> list[float]:
        """Gaussian kernel smoothing (pure-Python approximation)."""
        if not values:
            return []
        n = len(values)
        # Kernel radius: 3σ, at least 1
        radius = max(1, int(math.ceil(3 * sigma)))
        # Pre-compute kernel weights
        kernel = [math.exp(-0.5 * (x / sigma) ** 2) for x in range(-radius, radius + 1)]
        k_sum = sum(kernel)
        kernel = [k / k_sum for k in kernel]

        result: list[float] = []
        for i in range(n):
            acc = 0.0
            w_total = 0.0
            for j, kw in enumerate(kernel):
                idx = i + j - radius
                if 0 <= idx < n:
                    acc += values[idx] * kw
                    w_total += kw
            result.append(acc / w_total if w_total else values[i])
        return result

    def median_filter(self, values: list[float], window: int | None = None) -> list[float]:
        """Median filter for outlier removal."""
        if not values:
            return []
        w = window if window is not None else self.config.window_size
        w = max(1, w)
        n = len(values)
        half = w // 2
        result: list[float] = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            result.append(statistics.median(values[lo:hi]))
        return result

    # ------------------------------------------------------------------
    # Outlier detection / denoising
    # ------------------------------------------------------------------

    def detect_outliers(self, values: list[float], threshold: float = 2.0) -> list[int]:
        """Z-score based outlier detection.  Returns indices of outliers."""
        if len(values) < 2:
            return []
        mean = statistics.mean(values)
        std = statistics.pstdev(values)
        if std == 0:
            return []
        return [i for i, v in enumerate(values) if abs(v - mean) / std > threshold]

    def denoise(self, values: list[float]) -> list[float]:
        """Combined smoothing + outlier removal.

        1. Replace outliers with median-filtered values.
        2. Apply EMA smoothing on the cleaned series.
        """
        if not values:
            return []
        # Step 1: median-filter to get robust baseline
        med = self.median_filter(values)
        # Step 2: replace outlier positions with median-filtered value
        outlier_idx = set(self.detect_outliers(values))
        cleaned = [
            med[i] if i in outlier_idx else values[i]
            for i in range(len(values))
        ]
        # Step 3: final EMA pass
        return self.ema(cleaned)

    def __repr__(self) -> str:
        return f"MetricSmoother(config={self.config!r})"


# ------------------------------------------------------------------
# Standalone helpers
# ------------------------------------------------------------------


def compare_methods(
    values: list[float],
    methods: list[str] | None = None,
) -> dict[str, SmoothedSeries]:
    """Compare multiple smoothing methods on the same data."""
    if methods is None:
        methods = ["ema", "sma", "gaussian", "median"]
    results: dict[str, SmoothedSeries] = {}
    for method in methods:
        cfg = SmoothingConfig(method=method)
        smoother = MetricSmoother(cfg)
        results[method] = smoother.smooth(values)
    return results


def format_smoothing_report(series: SmoothedSeries) -> str:
    """Format a human-readable smoothing report."""
    lines: list[str] = [
        "Smoothing Report",
        "=" * 30,
        f"  Method          : {series.method}",
        f"  Series length   : {len(series.original)}",
    ]
    if series.original:
        lines.append(f"  Original range  : [{min(series.original):.4f}, {max(series.original):.4f}]")
        lines.append(f"  Smoothed range  : [{min(series.smoothed):.4f}, {max(series.smoothed):.4f}]")
    if series.residuals:
        abs_res = [abs(r) for r in series.residuals]
        lines.append(f"  Mean |residual| : {statistics.mean(abs_res):.4f}")
        lines.append(f"  Max  |residual| : {max(abs_res):.4f}")
    return "\n".join(lines)
