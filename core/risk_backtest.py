from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from tqdm import tqdm

from core.risk import (
    DailyRiskMetrics,
    apply_conformal_interval,
    conformal_delta,
    coverage_and_width,
    default_quantile_levels,
    interval_from_quantiles,
    interval_nonconformity_scores,
    mean_rate,
    summarize_daily_risk,
    var_violations,
)

#try:
#    from core.risk import upper_violations, pinball_loss, pit_values  # type: ignore
#except Exception:
#    upper_violations = None  # type: ignore
#    pinball_loss = None      # type: ignore
 #   pit_values = None        # type: ignore


def _forecast_to_numpy_quantiles(forecast) -> np.ndarray:
    """
    Normalize the output of pipeline.predict into an array of shape (N, Q).
    """
    x = forecast[0] if isinstance(forecast, (list, tuple)) else forecast

    # some wrappers might return dict-like
    if isinstance(x, dict):
        for k in ("quantiles", "forecast", "predictions"):
            if k in x:
                x = x[k]
                break

    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if x.ndim == 4:  # (B, N, Q, H)
        x = x[0]
    if x.ndim == 3:  # (N, Q, H)
        x = x[:, :, 0]
    if x.ndim != 2:
        raise ValueError(f"Unexpected forecast shape (expected (N,Q)): got {x.shape}")

    return x.astype(np.float32)


def predict_one_step_quantiles(pipeline, window_df: pd.DataFrame) -> np.ndarray:
    """
    Predicts the quantiles for the next day for all assets.

    Input:
        window_df: (context_length, N)

    Output:
        quantiles: (N, Q)
    """
    context = window_df.values.T.astype(np.float32)  # (N, context)
    forecast = pipeline.predict([{"target": context}], prediction_length=1)
    return _forecast_to_numpy_quantiles(forecast)


def run_risk_backtest(
    pipeline,
    df_returns: pd.DataFrame,
    context_length: int = 200,
    start_idx: int | None = None,
    quantile_levels: Sequence[float] | None = None,
    var_alphas: tuple[float, ...] = (0.1,),
    interval_pairs: tuple[tuple[float, float], ...] = ((0.1, 0.9),),
    # Conformal (static): calculate delta on calib_days and apply afterwards
    conformal_target_interval: tuple[float, float] | None = None,
    conformal_target_coverage: float = 0.8,
    calib_days: int = 250,
) -> dict:
    """
    Walk-forward backtest risk-aware.

    Does for each day:
    - quantile prediction (N,Q)
    - VaR hit rate at var_alphas levels
    - coverage + width for interval_pairs

    If conformal_target_interval is set:
    - use the first 'calib_days' days of backtesting to estimate the conformal delta
    - apply the delta to the intervals from the following day onward

    Returns:
    dict with:
    - daily_raw: list[DailyRiskMetrics]
    - daily_conformal: list[DailyRiskMetrics] (target interval only)
    - summary_raw, summary_conformal
    - conformal_delta

    Extra (added, non-breaking):
    - daily_raw_df: pd.DataFrame (per-day metrics for report)
    - daily_conformal_df: pd.DataFrame (per-day conformal metrics)
    - pit_raw: np.ndarray (flattened PIT values) if available
    """
    df = df_returns.dropna().copy()
    T, N = df.shape
    if start_idx is None:
        start_idx = context_length

    daily_raw: list[DailyRiskMetrics] = []
    daily_conf: list[DailyRiskMetrics] = []

    rows_raw: list[dict] = []
    rows_conf: list[dict] = []
    pit_pool_raw: list[np.ndarray] = []

    # pool for conformal delta calculation
    scores_pool: list[np.ndarray] = []
    delta: float | None = None
    miscoverage_alpha = 1.0 - float(conformal_target_coverage)

    for t in tqdm(range(start_idx, T), desc="Risk backtest"):
        window = df.iloc[t - context_length : t]
        y_true = df.iloc[t].values.astype(np.float32)

        q_pred = predict_one_step_quantiles(pipeline, window)  # (N,Q)
        Q = q_pred.shape[1]

        q_levels = (
            default_quantile_levels(Q) if quantile_levels is None
            else np.asarray(quantile_levels, dtype=np.float32)
        )
        if q_levels.shape[0] != Q:
            raise ValueError(f"quantile_levels length {len(q_levels)} != Q {Q}")

        # ---- raw VaR ----
        var_dict: dict[float, float] = {}
        for a in var_alphas:
            hits = var_violations(y_true, q_pred, q_levels, a)
            var_dict[a] = mean_rate(hits)

        # ---- raw intervals ----
        cov_dict: dict[tuple[float, float], float] = {}
        wid_dict: dict[tuple[float, float], float] = {}

        for (lo, hi) in interval_pairs:
            lower, upper = interval_from_quantiles(q_pred, q_levels, lo, hi)
            covered, width = coverage_and_width(y_true, lower, upper)
            cov_dict[(lo, hi)] = mean_rate(covered)
            wid_dict[(lo, hi)] = float(np.mean(width))

            # collect scores for conformal (target interval only) until delta is calculated
            if (
                conformal_target_interval is not None
                and (lo, hi) == conformal_target_interval
                and delta is None
            ):
                scores_pool.append(interval_nonconformity_scores(y_true, lower, upper))

        daily_raw.append(
            DailyRiskMetrics(
                date=df.index[t],
                var_hit_rate=var_dict,
                interval_coverage=cov_dict,
                interval_width=wid_dict,
            )
        )

        
        row_extra: dict = {"date": df.index[t]}

        # log first alpha and first interval for a clean report row
        a0 = var_alphas[0] if len(var_alphas) else None
        iv0 = interval_pairs[0] if len(interval_pairs) else None

        if a0 is not None:
            row_extra[f"var_hit_{a0:g}"] = var_dict.get(a0, np.nan)

            # upper-tail symmetry check (if function exists)
            if upper_violations is not None:
                try:
                    up_hits = upper_violations(y_true, q_pred, q_levels, a0)
                    row_extra[f"upper_hit_{a0:g}"] = mean_rate(up_hits)
                except Exception:
                    row_extra[f"upper_hit_{a0:g}"] = np.nan

        if iv0 is not None:
            lo0, hi0 = iv0
            row_extra[f"coverage_{lo0:g}_{hi0:g}"] = cov_dict.get((lo0, hi0), np.nan)
            row_extra[f"width_{lo0:g}_{hi0:g}"] = wid_dict.get((lo0, hi0), np.nan)

        
        if pinball_loss is not None:
            try:
                row_extra["pinball"] = float(pinball_loss(y_true, q_pred, q_levels))
            except Exception:
                row_extra["pinball"] = np.nan

        
        if pit_values is not None:
            try:
                pit = pit_values(y_true, q_pred, q_levels)  # (N,)
                pit_pool_raw.append(np.asarray(pit, dtype=np.float32).reshape(-1))
            except Exception:
                pass

        rows_raw.append(row_extra)

        # calculate delta after calib_days observations 
        if conformal_target_interval is not None and delta is None:
            if len(daily_raw) >= calib_days:
                pooled = (
                    np.concatenate(scores_pool, axis=0)
                    if scores_pool
                    else np.array([], dtype=np.float32)
                )
                delta = conformal_delta(pooled, miscoverage_alpha)

        # conformal metrics (only after delta) 
        if conformal_target_interval is not None and delta is not None:
            lo, hi = conformal_target_interval
            lower, upper = interval_from_quantiles(q_pred, q_levels, lo, hi)
            lower_c, upper_c = apply_conformal_interval(lower, upper, delta)
            covered_c, width_c = coverage_and_width(y_true, lower_c, upper_c)

            daily_conf.append(
                DailyRiskMetrics(
                    date=df.index[t],
                    var_hit_rate={},  # empty: we use conformal for intervals
                    interval_coverage={(lo, hi): mean_rate(covered_c)},
                    interval_width={(lo, hi): float(np.mean(width_c))},
                )
            )

            rows_conf.append(
                {
                    "date": df.index[t],
                    f"coverage_{lo:g}_{hi:g}": mean_rate(covered_c),
                    f"width_{lo:g}_{hi:g}": float(np.mean(width_c)),
                }
            )

    out = {
        "daily_raw": daily_raw,
        "summary_raw": summarize_daily_risk(daily_raw),
        "conformal_delta": delta,
    }
    if conformal_target_interval is not None and daily_conf:
        out["daily_conformal"] = daily_conf
        out["summary_conformal"] = summarize_daily_risk(daily_conf)

    out["daily_raw_df"] = pd.DataFrame(rows_raw)
    if rows_conf:
        out["daily_conformal_df"] = pd.DataFrame(rows_conf)

    if pit_pool_raw:
        out["pit_raw"] = np.concatenate(pit_pool_raw, axis=0)

    return out

