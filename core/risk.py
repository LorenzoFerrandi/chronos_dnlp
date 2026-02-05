from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np


def default_quantile_levels(Q: int) -> np.ndarray:
    """
    If the pipeline does not explicitly provide the quantile levels, we use a sensible convention.

    In the original repo, Q=9 is assumed => quantiles 0.1..0.9.
    """
    if Q == 9:
        return np.linspace(0.1, 0.9, 9, dtype=np.float32)

    # fallback: equally spaced quantiles in (0,1), excluding extremes
    return np.linspace(1.0 / (Q + 1), Q / (Q + 1), Q, dtype=np.float32)


def _closest_quantile_index(levels: Sequence[float], q: float) -> int:
    levels_np = np.asarray(levels, dtype=np.float32)
    return int(np.argmin(np.abs(levels_np - np.float32(q))))


def pick_quantile(y_pred_quantiles: np.ndarray, quantile_levels: Sequence[float], q: float) -> np.ndarray:
    """
    Extracts the quantile closest to the required level q.

    Args:
        y_pred_quantiles: array (N, Q)
        quantile_levels: array of Q levels
        q: livello desiderato (es. 0.1, 0.5, 0.9)

    Returns:
        array (N,)
    """
    idx = _closest_quantile_index(quantile_levels, q)
    return y_pred_quantiles[:, idx]


def interval_from_quantiles(
    y_pred_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    low_q: float,
    high_q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a [low, high] range for each asset.

    Returns:
        (lower, upper): both (N,)
    """
    low = pick_quantile(y_pred_quantiles, quantile_levels, low_q)
    high = pick_quantile(y_pred_quantiles, quantile_levels, high_q)
    lower = np.minimum(low, high)
    upper = np.maximum(low, high)
    return lower, upper


def var_violations(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """
    Per-asset VaR violations: True if the actual is below the estimated VaR threshold.

    Returns:
        bool array (N,)
    """
    var_level = pick_quantile(y_pred_quantiles, quantile_levels, alpha)
    return y_true < var_level


def mean_rate(x: np.ndarray) -> float:
    """
    Robust average (useful for hit rate, coverage rate)
    """
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def coverage_and_width(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coverage e width per-asset.

    Returns:
        covered: bool (N,)
        width: float (N,)
    """
    covered = (y_true >= lower) & (y_true <= upper)
    width = (upper - lower).astype(np.float32)
    return covered, width


def interval_nonconformity_scores(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
     Conformal score for intervals:
    - 0 if y is inside
    - distance to the nearest boundary if y is outside

    Returns:
        scores (N,)
    """
    below = lower - y_true
    above = y_true - upper
    scores = np.maximum(np.maximum(below, above), 0.0)
    return scores.astype(np.float32)


def conformal_delta(scores: np.ndarray, miscoverage_alpha: float) -> float:
    """
    Calculate the global conformal correction (delta) to apply to the intervals.

    miscoverage_alpha = 1 - target_coverage
    Example: target 0.8 => miscoverage_alpha=0.2

    Use the finite-sample correction.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    scores = scores[~np.isnan(scores)]
    n = scores.size
    if n == 0:
        return float("nan")

    k = int(np.ceil((n + 1) * (1.0 - miscoverage_alpha)))
    k = max(1, min(k, n))

    # kth order statistic
    return float(np.partition(scores, k - 1)[k - 1])


def apply_conformal_interval(lower: np.ndarray, upper: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the conformal correction to the intervals: [lower-delta, upper+delta]
    """
    return (lower - delta).astype(np.float32), (upper + delta).astype(np.float32)


@dataclass
class DailyRiskMetrics:
    """
    Metrics for each date.
    """
    date: object
    var_hit_rate: dict[float, float]                 # alpha -> rate
    interval_coverage: dict[tuple[float, float], float]  # (lo,hi) -> coverage
    interval_width: dict[tuple[float, float], float]     # (lo,hi) -> mean width


def summarize_daily_risk(daily: list[DailyRiskMetrics]) -> dict:
    """
    Aggregated summary: average across dates.
    """
    if not daily:
        return {}

    var_levels = sorted({a for d in daily for a in d.var_hit_rate.keys()})
    intervals = sorted({iv for d in daily for iv in d.interval_coverage.keys()})

    out: dict[str, float] = {}

    for a in var_levels:
        out[f"var_hit_rate_{a:g}"] = float(np.nanmean([d.var_hit_rate.get(a, np.nan) for d in daily]))

    for (lo, hi) in intervals:
        out[f"coverage_{lo:g}_{hi:g}"] = float(np.nanmean([d.interval_coverage.get((lo, hi), np.nan) for d in daily]))
        out[f"width_{lo:g}_{hi:g}"] = float(np.nanmean([d.interval_width.get((lo, hi), np.nan) for d in daily]))

    return out

def upper_violations(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """
    Per-asset upper-tail violations: True if actual is above q_{1-alpha}.
    """
    level = 1.0 - float(alpha)
    thr = pick_quantile(y_pred_quantiles, quantile_levels, level)
    return y_true > thr


def pinball_loss(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
) -> float:
    """
    Proper scoring rule for quantiles.
    Returns mean pinball loss across assets and quantiles.
    """
    y = y_true.astype(np.float32).reshape(-1, 1)  # (N,1)
    q = np.asarray(quantile_levels, dtype=np.float32).reshape(1, -1)  # (1,Q)
    f = y_pred_quantiles.astype(np.float32)  # (N,Q)

    e = y - f
    loss = np.maximum(q * e, (q - 1.0) * e)  # pinball
    return float(np.mean(loss))


def pit_values(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
) -> np.ndarray:
    """
    Approximate PIT using the empirical CDF induced by predicted quantiles:
    PIT ~ fraction of predicted quantiles below the observation.
    Returns (N,) values in [0,1].
    """
    y = y_true.astype(np.float32).reshape(-1, 1)   # (N,1)
    f = y_pred_quantiles.astype(np.float32)        # (N,Q)
    return np.mean(f <= y, axis=1).astype(np.float32)
