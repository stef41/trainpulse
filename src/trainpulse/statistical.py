"""Mann–Kendall trend test and isolation forest for training metrics.

Provides two genuinely non-trivial algorithms:

1. **Mann–Kendall trend test**: Non-parametric hypothesis test for
   monotonic trend in a time series.  Returns the test statistic S,
   the normalised Z-score, the two-sided p-value, Sen's slope estimator,
   and the trend direction.  Used for statistically rigorous early
   stopping decisions instead of ad-hoc patience counters.

2. **Isolation Forest**: Anomaly detection on multi-dimensional training
   metrics (loss, gradient norm, learning rate, step time).  Builds an
   ensemble of isolation trees using random axis-aligned splits.  Points
   that are isolated in fewer splits are more anomalous (shorter average
   path length → higher anomaly score).

Both are implemented from scratch in pure Python — no scikit-learn, no
NumPy, no SciPy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence


# ══════════════════════════════════════════════════════════════════════════════
#  Mann–Kendall Trend Test
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MannKendallResult:
    """Result of the Mann–Kendall trend test."""

    statistic_s: int        # Kendall S statistic
    z_score: float          # normalised Z statistic
    p_value: float          # two-sided p-value
    trend: str              # "increasing", "decreasing", "no_trend"
    sen_slope: float        # Sen's slope estimator (median of pairwise slopes)
    sen_intercept: float    # corresponding intercept
    n: int                  # sample size
    significant: bool       # True if p < alpha


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _normal_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun 26.2.17)."""
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1
    if z < 0:
        sign = -1
    z_abs = abs(z) / math.sqrt(2.0)

    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs)

    return 0.5 * (1.0 + sign * y)


def mann_kendall(
    series: Sequence[float],
    alpha: float = 0.05,
) -> MannKendallResult:
    """Perform the Mann–Kendall trend test.

    Parameters
    ----------
    series:
        Time-ordered observations.  Needs ≥ 4 values for meaningful results.
    alpha:
        Significance level (default 5%).

    Returns
    -------
    MannKendallResult
        Contains S statistic, Z-score, p-value, trend direction, and
        Sen's slope with intercept.

    Notes
    -----
    The variance correction handles tied values using the formula:

        var(S) = [n(n-1)(2n+5) - Σ t_i(t_i-1)(2t_i+5)] / 18

    where t_i is the size of the i-th group of tied values.
    """
    n = len(series)
    if n < 3:
        return MannKendallResult(
            statistic_s=0,
            z_score=0.0,
            p_value=1.0,
            trend="no_trend",
            sen_slope=0.0,
            sen_intercept=0.0,
            n=n,
            significant=False,
        )

    # Compute S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += _sign(series[j] - series[i])

    # Compute tied-value correction
    # Count group sizes of tied values
    from collections import Counter
    val_counts = Counter(series)
    tied_groups = [c for c in val_counts.values() if c > 1]

    # Variance of S with tie correction
    var_s = n * (n - 1) * (2 * n + 5)
    for t in tied_groups:
        var_s -= t * (t - 1) * (2 * t + 5)
    var_s /= 18.0

    # Z-score with continuity correction
    if var_s <= 0:
        z = 0.0
    else:
        if s > 0:
            z = (s - 1) / math.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / math.sqrt(var_s)
        else:
            z = 0.0

    # Two-sided p-value
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))

    # Trend direction
    if p_value <= alpha:
        trend = "increasing" if s > 0 else "decreasing"
    else:
        trend = "no_trend"

    # Sen's slope estimator: median of all (y_j - y_i) / (j - i)
    slopes: list[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((series[j] - series[i]) / (j - i))

    slopes.sort()
    if slopes:
        mid = len(slopes) // 2
        if len(slopes) % 2 == 0:
            sen_slope = (slopes[mid - 1] + slopes[mid]) / 2
        else:
            sen_slope = slopes[mid]
    else:
        sen_slope = 0.0

    # Intercept: median of (y_i - slope * i)
    intercepts = [series[i] - sen_slope * i for i in range(n)]
    intercepts.sort()
    mid = len(intercepts) // 2
    if len(intercepts) % 2 == 0:
        sen_intercept = (intercepts[mid - 1] + intercepts[mid]) / 2
    else:
        sen_intercept = intercepts[mid]

    return MannKendallResult(
        statistic_s=s,
        z_score=round(z, 4),
        p_value=round(p_value, 6),
        trend=trend,
        sen_slope=round(sen_slope, 8),
        sen_intercept=round(sen_intercept, 6),
        n=n,
        significant=p_value <= alpha,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Isolation Forest
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _IsolationNode:
    """Internal node of an isolation tree."""

    split_feature: int = -1
    split_value: float = 0.0
    left: Optional[_IsolationNode] = None
    right: Optional[_IsolationNode] = None
    size: int = 0   # leaf node: number of samples
    is_leaf: bool = False


@dataclass
class AnomalyScore:
    """Anomaly score for a single training step."""

    step: int
    score: float           # 0–1, higher = more anomalous
    path_length: float     # average path length across trees
    is_anomaly: bool
    feature_values: dict[str, float] = field(default_factory=dict)


@dataclass
class IsolationForestResult:
    """Result of isolation forest analysis on training metrics."""

    scores: list[AnomalyScore]
    anomaly_indices: list[int]  # steps flagged as anomalies
    threshold: float             # anomaly score threshold used
    n_trees: int
    avg_path_length: float


def _harmonic_number(n: int) -> float:
    """Approximate harmonic number H(n)."""
    if n <= 0:
        return 0.0
    if n == 1:
        return 1.0
    return math.log(n) + 0.5772156649  # Euler–Mascheroni constant


def _c(n: int) -> float:
    """Average path length of unsuccessful search in a BST.

    c(n) = 2 * H(n-1) - 2*(n-1)/n

    This is the normalisation factor for isolation forest scores.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * _harmonic_number(n - 1) - 2.0 * (n - 1) / n


def _build_isolation_tree(
    data: list[list[float]],
    max_depth: int,
    rng: random.Random,
) -> _IsolationNode:
    """Recursively build a single isolation tree."""
    n = len(data)
    n_features = len(data[0]) if data else 0

    # Leaf conditions
    if n <= 1 or max_depth <= 0:
        node = _IsolationNode(is_leaf=True, size=n)
        return node

    # Pick random feature and random split value
    feat = rng.randint(0, n_features - 1)
    col_values = [row[feat] for row in data]
    col_min = min(col_values)
    col_max = max(col_values)

    if col_min == col_max:
        return _IsolationNode(is_leaf=True, size=n)

    split_val = rng.uniform(col_min, col_max)

    left_data = [row for row in data if row[feat] < split_val]
    right_data = [row for row in data if row[feat] >= split_val]

    # Avoid degenerate splits
    if not left_data or not right_data:
        return _IsolationNode(is_leaf=True, size=n)

    node = _IsolationNode(
        split_feature=feat,
        split_value=split_val,
        left=_build_isolation_tree(left_data, max_depth - 1, rng),
        right=_build_isolation_tree(right_data, max_depth - 1, rng),
    )
    return node


def _path_length(point: list[float], node: _IsolationNode, depth: int = 0) -> float:
    """Compute path length for a single point through an isolation tree."""
    if node.is_leaf:
        # Return depth + estimated average path length for remaining points
        return depth + _c(node.size)

    if point[node.split_feature] < node.split_value:
        return _path_length(point, node.left, depth + 1)  # type: ignore[arg-type]
    else:
        return _path_length(point, node.right, depth + 1)  # type: ignore[arg-type]


def isolation_forest(
    metrics: dict[str, list[float]],
    n_trees: int = 100,
    sample_size: int = 256,
    contamination: float = 0.1,
    seed: int = 42,
) -> IsolationForestResult:
    """Detect anomalous training steps using an isolation forest.

    Parameters
    ----------
    metrics:
        Dict mapping metric names (e.g. "loss", "grad_norm", "lr",
        "step_time") to lists of per-step values.  All lists must
        have the same length.
    n_trees:
        Number of isolation trees in the ensemble.
    sample_size:
        Subsample size per tree (default 256, as in the original paper).
    contamination:
        Expected fraction of anomalies (used to set the threshold).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    IsolationForestResult
        Per-step anomaly scores, flagged steps, and diagnostics.
    """
    feature_names = sorted(metrics.keys())
    n_features = len(feature_names)

    if n_features == 0:
        return IsolationForestResult(
            scores=[], anomaly_indices=[], threshold=0.5,
            n_trees=0, avg_path_length=0.0,
        )

    # Build data matrix
    lengths = [len(metrics[name]) for name in feature_names]
    n_steps = min(lengths) if lengths else 0
    if n_steps == 0:
        return IsolationForestResult(
            scores=[], anomaly_indices=[], threshold=0.5,
            n_trees=0, avg_path_length=0.0,
        )

    data: list[list[float]] = []
    for i in range(n_steps):
        row = [metrics[name][i] for name in feature_names]
        data.append(row)

    rng = random.Random(seed)
    max_depth = int(math.ceil(math.log2(max(sample_size, 2))))

    # Build forest
    trees: list[_IsolationNode] = []
    for _ in range(n_trees):
        if n_steps > sample_size:
            sample = rng.sample(data, sample_size)
        else:
            sample = list(data)
        tree = _build_isolation_tree(sample, max_depth, rng)
        trees.append(tree)

    # Score each point
    c_n = _c(sample_size)
    scores: list[AnomalyScore] = []
    all_anomaly_scores: list[float] = []

    for step_idx, point in enumerate(data):
        avg_pl = sum(_path_length(point, tree) for tree in trees) / n_trees
        # Anomaly score: s(x, n) = 2^(-E[h(x)] / c(n))
        anomaly_score = math.pow(2, -avg_pl / c_n) if c_n > 0 else 0.5

        feature_vals = {name: point[j] for j, name in enumerate(feature_names)}

        scores.append(AnomalyScore(
            step=step_idx,
            score=round(anomaly_score, 4),
            path_length=round(avg_pl, 4),
            is_anomaly=False,  # set below after threshold
            feature_values=feature_vals,
        ))
        all_anomaly_scores.append(anomaly_score)

    # Set threshold based on contamination parameter
    sorted_scores = sorted(all_anomaly_scores, reverse=True)
    threshold_idx = max(0, int(contamination * n_steps) - 1)
    threshold = sorted_scores[threshold_idx] if sorted_scores else 0.5

    anomaly_indices: list[int] = []
    for s in scores:
        if s.score >= threshold:
            s.is_anomaly = True
            anomaly_indices.append(s.step)

    avg_pl_all = sum(s.path_length for s in scores) / len(scores) if scores else 0.0

    return IsolationForestResult(
        scores=scores,
        anomaly_indices=anomaly_indices,
        threshold=round(threshold, 4),
        n_trees=n_trees,
        avg_path_length=round(avg_pl_all, 4),
    )
