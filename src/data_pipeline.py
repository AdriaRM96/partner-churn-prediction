"""
data_pipeline.py

Reusable data preparation pipeline for the partner churn model.

This module is deliberately kept separate from data *generation*
(generate_dataset.py, which fabricates synthetic partner records) so that
the same feature-engineering logic here could be pointed at a real,
monthly-refreshed partner data warehouse extract without changes: swap the
CSV paths for a database query and everything downstream still works.

Expected raw inputs:
    - a "static" partner table: one row per partner with attributes that are
      already known each month (region, vertical, tier, tenure, NPS,
      engagement, support, competitor flags, and the target, if available)
    - a GTV history table: one row per partner per month of gross
      transaction value (long format: partner_id, month_index, gtv)

The pipeline computes GTV-derived features (trend, volatility, seasonality,
peer/market comparison) from the history table and joins them onto the
static table to produce a single model-ready dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TREND_WINDOW = 3      # months used for the "recent" vs "prior" trend comparison
DEFAULT_SEASONAL_LAG = 12     # months, i.e. one year, for seasonality detection
DEFAULT_SEASONALITY_THRESHOLD = 0.5  # autocorrelation cutoff to flag has_seasonality


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_partner_static(path: str | Path) -> pd.DataFrame:
    """Load the per-partner static/attribute table (one row per partner).

    Uses `keep_default_na=False` because the `region` column legitimately
    contains the literal string "NA" (North America) — pandas' default
    missing-value sentinels would otherwise silently turn that into a null
    on every read.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if "partner_id" not in df.columns:
        raise ValueError(f"Expected a 'partner_id' column in {path}")
    return df


def load_gtv_history(path: str | Path) -> pd.DataFrame:
    """Load the long-format monthly GTV history table.

    Expected columns: partner_id, month_index, gtv. `month_index` must be
    contiguous per partner and increasing in calendar order (0 = oldest
    month in the window, max = most recent), which is what every function
    below assumes when comparing "recent" vs "prior" periods or checking a
    12-month lag.
    """
    df = pd.read_csv(path)
    required = {"partner_id", "month_index", "gtv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GTV history at {path} is missing columns: {missing}")
    return df.sort_values(["partner_id", "month_index"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived feature: trend + volatility
# ---------------------------------------------------------------------------
def compute_gtv_trend_and_volatility(
    gtv_history: pd.DataFrame, trend_window: int = DEFAULT_TREND_WINDOW
) -> pd.DataFrame:
    """Compute average GTV, short-term trend, and volatility per partner.

    - avg_monthly_gtv: mean GTV over the full available history
    - gtv_trend_3m: % change of the last `trend_window` months vs the
      `trend_window` months before that (the "current momentum" signal)
    - gtv_volatility: coefficient of variation (std / mean) over the full
      history, a scale-independent measure of how erratic a partner's GTV is

    Requires at least 2 * trend_window months of history per partner.
    """
    pivot = gtv_history.pivot(index="partner_id", columns="month_index", values="gtv")
    n_months = pivot.shape[1]
    if n_months < 2 * trend_window:
        raise ValueError(
            f"Need at least {2 * trend_window} months of history to compute a "
            f"{trend_window}-month trend; got {n_months}."
        )

    values = pivot.values
    last = values[:, -trend_window:].mean(axis=1)
    prev = values[:, -2 * trend_window:-trend_window].mean(axis=1)

    gtv_trend_3m = (last - prev) / prev * 100
    gtv_mean = values.mean(axis=1)
    gtv_std = values.std(axis=1)
    gtv_volatility = gtv_std / np.where(gtv_mean == 0, 1, gtv_mean)

    return pd.DataFrame(
        {
            "partner_id": pivot.index,
            "avg_monthly_gtv": np.round(gtv_mean, 2),
            "gtv_trend_3m": np.round(gtv_trend_3m, 2),
            "gtv_volatility": np.round(gtv_volatility, 4),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived feature: seasonality
# ---------------------------------------------------------------------------
def _autocorr_at_lag(x: np.ndarray, lag: int) -> float:
    """Pearson correlation between a series and itself shifted by `lag`."""
    if len(x) <= lag:
        return 0.0
    x1, x2 = x[:-lag], x[lag:]
    if x1.std() == 0 or x2.std() == 0:
        return 0.0
    return float(np.corrcoef(x1, x2)[0, 1])


def compute_seasonality(
    gtv_history: pd.DataFrame,
    lag: int = DEFAULT_SEASONAL_LAG,
    threshold: float = DEFAULT_SEASONALITY_THRESHOLD,
) -> pd.DataFrame:
    """Flag partners whose GTV shows a genuine recurring annual pattern.

    Each partner's series is linearly detrended, then its autocorrelation at
    a `lag`-month offset (default 12, i.e. one year) is measured. This only
    produces a meaningful signal when the history spans more than `lag`
    months more than once — with less than ~2 years of data a "seasonal"
    flag can't be distinguished from a single one-off seasonal bump, so this
    should be re-validated as more history accumulates.
    """
    pivot = gtv_history.pivot(index="partner_id", columns="month_index", values="gtv")
    values = pivot.values
    n = values.shape[0]

    autocorr = np.zeros(n)
    t = np.arange(values.shape[1])
    A = np.vstack([t, np.ones_like(t)]).T
    for i in range(n):
        series = values[i]
        slope, intercept = np.linalg.lstsq(A, series, rcond=None)[0]
        detrended = series - (slope * t + intercept)
        autocorr[i] = _autocorr_at_lag(detrended, lag=lag)

    has_seasonality = (autocorr > threshold).astype(int)

    return pd.DataFrame(
        {
            "partner_id": pivot.index,
            "seasonal_autocorr_lag12": np.round(autocorr, 4),
            "has_seasonality": has_seasonality,
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived feature: peer / market comparison
# ---------------------------------------------------------------------------
def compute_peer_comparison(
    static_df: pd.DataFrame, gtv_trend_features: pd.DataFrame
) -> pd.DataFrame:
    """Compute how a partner's GTV level and growth compare to its peers.

    - gtv_vs_peers: percentile rank of avg_monthly_gtv within the partner's
      own tier + region + vertical cohort (0-1, higher = stronger relative
      performer)
    - gtv_vs_market_growth: partner's gtv_trend_3m minus the average
      gtv_trend_3m for its vertical (positive = outgrowing its market)

    `static_df` must contain partner_id, partner_tier, region, vertical.
    `gtv_trend_features` must contain partner_id, avg_monthly_gtv,
    gtv_trend_3m (the output of compute_gtv_trend_and_volatility).
    """
    merged = static_df[["partner_id", "partner_tier", "region", "vertical"]].merge(
        gtv_trend_features[["partner_id", "avg_monthly_gtv", "gtv_trend_3m"]],
        on="partner_id",
        how="inner",
    )

    merged["gtv_vs_peers"] = merged.groupby(["partner_tier", "region", "vertical"])[
        "avg_monthly_gtv"
    ].rank(pct=True)

    vertical_avg_growth = merged.groupby("vertical")["gtv_trend_3m"].transform("mean")
    merged["gtv_vs_market_growth"] = merged["gtv_trend_3m"] - vertical_avg_growth

    return pd.DataFrame(
        {
            "partner_id": merged["partner_id"],
            "gtv_vs_peers": np.round(merged["gtv_vs_peers"], 4),
            "gtv_vs_market_growth": np.round(merged["gtv_vs_market_growth"], 2),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_gtv_features(
    static_df: pd.DataFrame,
    gtv_history: pd.DataFrame,
    trend_window: int = DEFAULT_TREND_WINDOW,
    seasonal_lag: int = DEFAULT_SEASONAL_LAG,
    seasonality_threshold: float = DEFAULT_SEASONALITY_THRESHOLD,
) -> pd.DataFrame:
    """Compute the full set of GTV-derived features for every partner.

    Combines trend/volatility, seasonality, and peer/market comparison into
    a single per-partner feature table, keyed on partner_id.
    """
    trend_features = compute_gtv_trend_and_volatility(gtv_history, trend_window=trend_window)
    seasonality_features = compute_seasonality(
        gtv_history, lag=seasonal_lag, threshold=seasonality_threshold
    )
    peer_features = compute_peer_comparison(static_df, trend_features)

    gtv_features = (
        trend_features.merge(seasonality_features, on="partner_id", how="left")
        .merge(peer_features, on="partner_id", how="left")
    )
    return gtv_features


def build_model_ready_dataset(
    static_df: pd.DataFrame,
    gtv_history: pd.DataFrame,
    trend_window: int = DEFAULT_TREND_WINDOW,
    seasonal_lag: int = DEFAULT_SEASONAL_LAG,
    seasonality_threshold: float = DEFAULT_SEASONALITY_THRESHOLD,
) -> pd.DataFrame:
    """Produce the final, model-ready partner-level dataset.

    Joins the static partner attributes (region, vertical, tier, tenure,
    NPS, engagement, support, competitor signals, and the target column if
    present) with the GTV-derived features computed from the history table.

    This is the single entry point intended for reuse: given a fresh
    monthly extract of the static table and the GTV history table, calling
    this function reproduces the same feature set used for training,
    ready to be scored or used to retrain the model.
    """
    gtv_features = build_gtv_features(
        static_df,
        gtv_history,
        trend_window=trend_window,
        seasonal_lag=seasonal_lag,
        seasonality_threshold=seasonality_threshold,
    )
    model_df = static_df.merge(gtv_features, on="partner_id", how="inner")
    return model_df


def load_and_build_dataset(
    static_path: str | Path,
    gtv_history_path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper: load both raw inputs from disk and build the
    model-ready dataset in one call. This is the function a monthly batch
    job (or a notebook) would call end to end.
    """
    static_df = load_partner_static(static_path)
    gtv_history = load_gtv_history(gtv_history_path)
    return build_model_ready_dataset(static_df, gtv_history, **kwargs)
