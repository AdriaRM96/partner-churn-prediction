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

The pipeline computes GTV-derived features (trend, year-over-year trend,
seasonal deviation, volatility, seasonality, peer/market comparison, and a
cold-start cohort-ramp feature for new partners) from the history table and
joins them onto the static table to produce a single model-ready dataset.

Two feature-engineering paths:
    - "mature" partners get the full trend/seasonality/peer-comparison
      block, computed from their GTV history.
    - "new_partner" partners (< DEFAULT_COLD_START_TENURE_MONTHS tenure)
      don't have enough reliable history for those features, so that block
      is masked to a fixed neutral value for them and they instead get a
      cohort-ramp-benchmark feature. See add_scoring_path /
      mask_cold_start_features.

build_snapshot_dataset() orchestrates calling this pipeline once per
point-in-time cutoff to build a multi-snapshot (temporal) dataset, reusing
every function below completely unmodified per cutoff — see its docstring.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

DEFAULT_TREND_WINDOW = 3      # months used for the "recent" vs "prior" trend comparison
DEFAULT_SEASONAL_LAG = 12     # months, i.e. one year, for seasonality detection
DEFAULT_SEASONALITY_THRESHOLD = 0.5  # autocorrelation cutoff to flag has_seasonality
DEFAULT_YOY_LAG = 12          # months, for the year-over-year trend comparison
DEFAULT_SEASONAL_DEVIATION_HOLDOUT = 3  # months held out to evaluate the seasonal profile
DEFAULT_COLD_START_TENURE_MONTHS = 9    # below this tenure, a partner is "new_partner"

# The GTV-derived features that only make sense for a partner with enough
# history behind them. These are masked to a fixed neutral value for
# "new_partner" rows rather than left at their (unreliable, short-history)
# computed value. See mask_cold_start_features.
MATURE_GTV_FEATURE_COLUMNS = [
    "gtv_trend_3m",
    "gtv_trend_yoy_3m",
    "gtv_seasonal_deviation_pct",
    "gtv_volatility",
    "has_seasonality",
    "seasonal_autocorr_lag12",
    "gtv_vs_market_growth",
    "gtv_vs_peers",
]


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


def load_partner_snapshots(path: str | Path) -> pd.DataFrame:
    """Load the multi-snapshot model-ready table produced by
    build_snapshot_dataset / generate_dataset.py (data/partner_snapshots.csv).

    Same `keep_default_na=False` reasoning as load_partner_static — this
    table also has a `region` column containing the literal string "NA".
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    required = {"partner_id", "snapshot_role", "snapshot_cutoff_month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Partner snapshots at {path} are missing columns: {missing}")
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
    gtv_history: pd.DataFrame,
    trend_window: int = DEFAULT_TREND_WINDOW,
    yoy_lag: int = DEFAULT_YOY_LAG,
) -> pd.DataFrame:
    """Compute average GTV, short-term trend, year-over-year trend, and
    volatility per partner.

    - avg_monthly_gtv: mean GTV over the full available history
    - gtv_trend_3m: % change of the last `trend_window` months vs the
      `trend_window` months before that (the "current momentum" signal).
      CAVEAT: for a seasonal partner, this comparison can misread a normal
      post-peak seasonal decline as churn risk — e.g. a Retail partner
      sampled in January is comparing a post-holiday trough against a
      holiday peak. Use gtv_trend_yoy_3m (below) for seasonal partners
      instead; gtv_trend_3m remains the right signal for non-seasonal ones.
    - gtv_trend_yoy_3m: % change of the last `trend_window` months vs the
      SAME `trend_window` months exactly `yoy_lag` (default 12) months
      earlier. This nets out seasonality by construction — a seasonal
      partner in their normal low season shows ~0% here, not a false
      decline. NaN if there isn't yet a full year of history behind the
      comparison window.
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

    if n_months >= trend_window + yoy_lag:
        same_period_last_year = values[:, -(trend_window + yoy_lag):-yoy_lag].mean(axis=1)
        gtv_trend_yoy_3m = (last - same_period_last_year) / same_period_last_year * 100
    else:
        gtv_trend_yoy_3m = np.full(values.shape[0], np.nan)

    gtv_mean = values.mean(axis=1)
    gtv_std = values.std(axis=1)
    gtv_volatility = gtv_std / np.where(gtv_mean == 0, 1, gtv_mean)

    return pd.DataFrame(
        {
            "partner_id": pivot.index,
            "avg_monthly_gtv": np.round(gtv_mean, 2),
            "gtv_trend_3m": np.round(gtv_trend_3m, 2),
            "gtv_trend_yoy_3m": np.round(gtv_trend_yoy_3m, 2),
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


def compute_seasonal_deviation(
    gtv_history: pd.DataFrame,
    seasonality_features: pd.DataFrame,
    holdout_window: int = DEFAULT_SEASONAL_DEVIATION_HOLDOUT,
    seasonal_period: int = DEFAULT_SEASONAL_LAG,
) -> pd.DataFrame:
    """For seasonal partners, measure deviation from their OWN expected
    seasonal curve — the more rigorous replacement for reading raw
    month-over-month trend as risk.

    For each partner flagged has_seasonality=1, fits an STL decomposition
    (statsmodels.tsa.seasonal.STL — chosen over classical seasonal_decompose
    because STL's loess-based smoothing has no NaN edge gap; classical
    decomposition's centered moving average leaves the last `period/2`
    months undefined, which is exactly the region this feature needs to
    evaluate) on all months EXCEPT the most recent `holdout_window` months.
    The fitted trend + seasonal components are then extrapolated one cycle
    forward (by periodicity: the seasonal value `holdout_window` months
    past the fit window equals the seasonal value from the same calendar
    position one cycle back, held at the fit's last trend level) and
    compared against the ACTUAL average GTV over those held-out months.

    gtv_seasonal_deviation_pct = (actual - expected) / expected * 100

    A partner having a normal seasonal trough shows ~0% here (their actual
    matches their own historical pattern for this point in the cycle). A
    partner who is underperforming relative to their own normal season —
    the real risk signal — shows a large negative value. This is evaluated
    on genuinely held-out months (excluded from the STL fit), so it's an
    out-of-sample signal, not a number that's trivially near-zero because
    the same points were used to fit the curve.

    Returns 0.0 ("no deviation from an expected pattern that doesn't
    apply") for non-seasonal partners — a deliberate neutral encoding, not
    a missing value, so this column is safe to feed straight into a scaler
    without needing a separate imputation step. Falls back to NaN only when
    a seasonal partner doesn't have enough history for a reliable fit
    (needs >= holdout_window + 2 * seasonal_period months total) or its STL
    fit doesn't converge — both should be rare given this project's 27+
    month trailing-history floor at every snapshot, but any NaNs that do
    occur should be imputed downstream (e.g. filled with 0.0) before
    modeling. A count of such fallbacks is printed for visibility.
    """
    pivot = gtv_history.pivot(index="partner_id", columns="month_index", values="gtv")
    seasonal_ids = set(
        seasonality_features.loc[seasonality_features["has_seasonality"] == 1, "partner_id"]
    )

    deviations = {pid: 0.0 for pid in pivot.index if pid not in seasonal_ids}
    fallback_count = 0
    for partner_id, row in pivot.iterrows():
        if partner_id not in seasonal_ids:
            continue
        series = row.values.astype(float)
        if len(series) < holdout_window + 2 * seasonal_period:
            continue  # not enough history for a reliable fit; stays NaN

        fit_part = series[:-holdout_window]
        holdout_part = series[-holdout_window:]
        try:
            stl_result = STL(fit_part, period=seasonal_period, robust=True).fit()
            last_trend = stl_result.trend[-1]
            seasonal_cycle = stl_result.seasonal[-seasonal_period:]
            expected = np.array(
                [last_trend + seasonal_cycle[k % seasonal_period] for k in range(holdout_window)]
            )
            expected_mean = expected.mean()
            if expected_mean == 0:
                continue
            deviations[partner_id] = (holdout_part.mean() - expected_mean) / expected_mean * 100
        except Exception:
            fallback_count += 1
            continue

    if fallback_count:
        print(
            f"compute_seasonal_deviation: STL fit fell back to NaN for "
            f"{fallback_count} seasonal partner(s)."
        )

    return pd.DataFrame(
        {
            "partner_id": pivot.index,
            "gtv_seasonal_deviation_pct": [
                np.round(deviations.get(pid, np.nan), 2) for pid in pivot.index
            ],
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
# Derived feature: cold-start cohort ramp benchmark (for new partners)
# ---------------------------------------------------------------------------
def compute_new_partner_ramp_feature(
    static_df: pd.DataFrame,
    gtv_history: pd.DataFrame,
    cold_start_tenure_threshold: int = DEFAULT_COLD_START_TENURE_MONTHS,
) -> pd.DataFrame:
    """For partners with less than `cold_start_tenure_threshold` months of
    tenure, compare their own GTV ramp against the typical ramp of peers in
    the same tier/region/vertical cohort at the same relative tenure stage.

    This is the signal a trend/seasonality feature can't reliably give for
    someone who hasn't been around long enough: "is this partner ramping
    normally for someone this new" rather than "is their multi-year trend
    declining" (which needs years of history they don't have yet).

    gtv_ramp_vs_cohort = (partner's own avg ramp GTV - cohort's avg ramp
    GTV at the same tenure stage) / cohort's avg ramp GTV * 100

    Cohort grouping falls back to a coarser grouping (dropping region, then
    also dropping tier) when the fine-grained tier+region+vertical cohort
    at that tenure stage has fewer than MIN_COHORT_SIZE peers, since a real
    partner book will have thin cells for young, narrow segments.

    Returns NaN for partners at or above the tenure threshold — the feature
    doesn't apply to them (see mask_cold_start_features, which sets it to a
    fixed neutral 0.0 for those rows instead of leaving it NaN).

    SIMPLIFICATION, documented deliberately: this synthetic dataset gives
    every partner the same full-length GTV history regardless of tenure (a
    real new partner would only have a few real months on file). The "own
    ramp" here is a proxy — the trailing `months_as_partner` months of the
    partner's series, counting back from the most recent month in the
    window, not a literal join-date-anchored slice.
    """
    MIN_COHORT_SIZE = 10
    TENURE_BAND = 3

    pivot = gtv_history.pivot(index="partner_id", columns="month_index", values="gtv")
    n_months = pivot.shape[1]

    tmp = static_df[["partner_id", "partner_tier", "region", "vertical", "months_as_partner"]].copy()
    tmp = tmp.set_index("partner_id").reindex(pivot.index).reset_index()

    own_ramp_avg = []
    for partner_id, tenure in zip(tmp["partner_id"], tmp["months_as_partner"]):
        t = int(min(tenure, n_months))
        own_ramp_avg.append(pivot.loc[partner_id].values[-t:].mean())
    tmp["own_ramp_avg"] = own_ramp_avg

    cohort_avg = np.full(len(tmp), np.nan)
    tenure_arr = tmp["months_as_partner"].values
    for i in range(len(tmp)):
        band_mask = np.abs(tenure_arr - tenure_arr[i]) <= TENURE_BAND
        not_self = tmp["partner_id"] != tmp["partner_id"].iloc[i]

        fine_mask = (
            band_mask
            & not_self.values
            & (tmp["partner_tier"] == tmp["partner_tier"].iloc[i]).values
            & (tmp["region"] == tmp["region"].iloc[i]).values
            & (tmp["vertical"] == tmp["vertical"].iloc[i]).values
        )
        peers = tmp.loc[fine_mask, "own_ramp_avg"]

        if len(peers) < MIN_COHORT_SIZE:
            medium_mask = (
                band_mask
                & not_self.values
                & (tmp["partner_tier"] == tmp["partner_tier"].iloc[i]).values
                & (tmp["vertical"] == tmp["vertical"].iloc[i]).values
            )
            peers = tmp.loc[medium_mask, "own_ramp_avg"]

        if len(peers) < MIN_COHORT_SIZE:
            coarse_mask = band_mask & not_self.values & (tmp["vertical"] == tmp["vertical"].iloc[i]).values
            peers = tmp.loc[coarse_mask, "own_ramp_avg"]

        cohort_avg[i] = peers.mean() if len(peers) > 0 else np.nan

    tmp["cohort_ramp_avg"] = cohort_avg
    tmp["gtv_ramp_vs_cohort"] = np.where(
        tmp["cohort_ramp_avg"] > 0,
        (tmp["own_ramp_avg"] - tmp["cohort_ramp_avg"]) / tmp["cohort_ramp_avg"] * 100,
        np.nan,
    )
    # A partner only 3-4 months in has a noisy own-ramp average (a handful
    # of months) divided by an also-thin cohort denominator, which can
    # blow the raw percentage out to several hundred or even 1000%+ for a
    # small number of partners. Clip to a wide-but-bounded range so a
    # handful of extreme values don't distort StandardScaler for everyone
    # else — RAMP_CLIP_BOUNDS is intentionally wide (partners really can
    # ramp far above or below a cohort norm early on); this only reins in
    # the genuine long tail, not the ordinary spread of the feature.
    RAMP_CLIP_BOUNDS = (-100.0, 300.0)
    tmp["gtv_ramp_vs_cohort"] = tmp["gtv_ramp_vs_cohort"].clip(*RAMP_CLIP_BOUNDS)

    is_new_partner = tmp["months_as_partner"] < cold_start_tenure_threshold
    tmp.loc[~is_new_partner, "gtv_ramp_vs_cohort"] = np.nan

    result = tmp[["partner_id", "gtv_ramp_vs_cohort"]].copy()
    result["gtv_ramp_vs_cohort"] = result["gtv_ramp_vs_cohort"].round(2)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cold-start scoring path: tagging + feature masking
# ---------------------------------------------------------------------------
def add_scoring_path(
    model_df: pd.DataFrame,
    cold_start_tenure_threshold: int = DEFAULT_COLD_START_TENURE_MONTHS,
) -> pd.DataFrame:
    """Tag each partner with which feature path applies to them: 'mature'
    (full trend/seasonality/peer-comparison block) or 'new_partner' (that
    block masked to neutral, cohort-ramp feature active instead). See
    mask_cold_start_features for how the masking itself is applied.
    """
    model_df = model_df.copy()
    model_df["scoring_path"] = np.where(
        model_df["months_as_partner"] < cold_start_tenure_threshold, "new_partner", "mature"
    )
    return model_df


def mask_cold_start_features(
    model_df: pd.DataFrame,
    mature_mask_values: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Neutralize the features that don't apply to a partner's scoring path.

    For 'new_partner' rows: MATURE_GTV_FEATURE_COLUMNS (trend, yoy trend,
    seasonal deviation, volatility, seasonality flag/autocorr, peer/market
    comparison) are masked to a fixed neutral value, rather than left at
    their unreliable short-history computed value — the model should treat
    "this doesn't apply yet" consistently, not learn from noisy estimates
    built on a handful of months.

    For 'mature' rows: gtv_ramp_vs_cohort (which only applies to new
    partners) is masked to 0.0 — "no deviation from cohort" is a natural,
    non-data-dependent neutral value, so no reference statistic is needed
    for this direction.

    `mature_mask_values`: a {column: neutral_value} dict for the mature-
    block columns. If None, computed here from THIS DataFrame's own
    'mature' population (column medians) and returned alongside the masked
    DataFrame. For anything other than a from-scratch training fit —
    scoring a validation/test snapshot, or a real monthly production run —
    ALWAYS pass in the dict learned from the training data instead of
    leaving this None, so "neutral" means the same thing at scoring time as
    it did at training time (this mirrors fitting a SimpleImputer on train
    only and reusing it at transform time).
    """
    model_df = model_df.copy()
    if "scoring_path" not in model_df.columns:
        raise ValueError("mask_cold_start_features requires add_scoring_path to have run first")

    is_new_partner = model_df["scoring_path"] == "new_partner"
    is_mature = ~is_new_partner

    if mature_mask_values is None:
        mature_mask_values = {
            col: float(model_df.loc[is_mature, col].median())
            for col in MATURE_GTV_FEATURE_COLUMNS
        }

    for col, neutral_value in mature_mask_values.items():
        model_df.loc[is_new_partner, col] = neutral_value

    if "gtv_ramp_vs_cohort" in model_df.columns:
        model_df.loc[is_mature, "gtv_ramp_vs_cohort"] = 0.0

    return model_df, mature_mask_values


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_gtv_features(
    static_df: pd.DataFrame,
    gtv_history: pd.DataFrame,
    trend_window: int = DEFAULT_TREND_WINDOW,
    seasonal_lag: int = DEFAULT_SEASONAL_LAG,
    seasonality_threshold: float = DEFAULT_SEASONALITY_THRESHOLD,
    yoy_lag: int = DEFAULT_YOY_LAG,
    seasonal_deviation_holdout: int = DEFAULT_SEASONAL_DEVIATION_HOLDOUT,
) -> pd.DataFrame:
    """Compute the full set of GTV-derived features for every partner.

    Combines trend/volatility/yoy-trend, seasonality, seasonal deviation,
    and peer/market comparison into a single per-partner feature table,
    keyed on partner_id.
    """
    trend_features = compute_gtv_trend_and_volatility(
        gtv_history, trend_window=trend_window, yoy_lag=yoy_lag
    )
    seasonality_features = compute_seasonality(
        gtv_history, lag=seasonal_lag, threshold=seasonality_threshold
    )
    peer_features = compute_peer_comparison(static_df, trend_features)
    seasonal_deviation_features = compute_seasonal_deviation(
        gtv_history,
        seasonality_features,
        holdout_window=seasonal_deviation_holdout,
        seasonal_period=seasonal_lag,
    )

    gtv_features = (
        trend_features.merge(seasonality_features, on="partner_id", how="left")
        .merge(peer_features, on="partner_id", how="left")
        .merge(seasonal_deviation_features, on="partner_id", how="left")
    )
    return gtv_features


def build_model_ready_dataset(
    static_df: pd.DataFrame,
    gtv_history: pd.DataFrame,
    trend_window: int = DEFAULT_TREND_WINDOW,
    seasonal_lag: int = DEFAULT_SEASONAL_LAG,
    seasonality_threshold: float = DEFAULT_SEASONALITY_THRESHOLD,
    yoy_lag: int = DEFAULT_YOY_LAG,
    seasonal_deviation_holdout: int = DEFAULT_SEASONAL_DEVIATION_HOLDOUT,
    cold_start_tenure_threshold: int = DEFAULT_COLD_START_TENURE_MONTHS,
    mature_mask_values: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Produce the final, model-ready partner-level dataset.

    Joins the static partner attributes (region, vertical, tier, tenure,
    NPS, engagement, support, competitor signals, and the target column if
    present) with the GTV-derived features computed from the history table,
    tags each partner with a scoring_path ('mature' / 'new_partner'), and
    masks the features that don't apply to a partner's path to a neutral
    value (see add_scoring_path / mask_cold_start_features).

    Returns (model_df, mature_mask_values) — the second element is the
    {column: neutral_value} dict used for masking. Pass it back in on a
    later call (e.g. scoring a validation/test snapshot, or a real monthly
    production run) so "neutral" is defined consistently from training
    data rather than recomputed from whatever's being scored. Leaving it
    None computes it fresh from this DataFrame's own mature population —
    the right behavior only when this call IS the training fit.
    """
    gtv_features = build_gtv_features(
        static_df,
        gtv_history,
        trend_window=trend_window,
        seasonal_lag=seasonal_lag,
        seasonality_threshold=seasonality_threshold,
        yoy_lag=yoy_lag,
        seasonal_deviation_holdout=seasonal_deviation_holdout,
    )
    ramp_features = compute_new_partner_ramp_feature(
        static_df, gtv_history, cold_start_tenure_threshold=cold_start_tenure_threshold
    )

    model_df = (
        static_df.merge(gtv_features, on="partner_id", how="inner")
        .merge(ramp_features, on="partner_id", how="left")
    )
    model_df = add_scoring_path(model_df, cold_start_tenure_threshold=cold_start_tenure_threshold)
    model_df, mature_mask_values = mask_cold_start_features(model_df, mature_mask_values=mature_mask_values)

    return model_df, mature_mask_values


def load_and_build_dataset(
    static_path: str | Path,
    gtv_history_path: str | Path,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper: load both raw inputs from disk and build the
    model-ready dataset in one call. This is the function a monthly batch
    job (or a notebook) would call end to end for a single point-in-time
    extract. Returns (model_df, mature_mask_values) — see
    build_model_ready_dataset.
    """
    static_df = load_partner_static(static_path)
    gtv_history = load_gtv_history(gtv_history_path)
    return build_model_ready_dataset(static_df, gtv_history, **kwargs)


def build_snapshot_dataset(
    static_df_by_role: dict[str, pd.DataFrame],
    gtv_history: pd.DataFrame,
    snapshot_cutoffs: dict[str, int],
    role_order: list[str],
    **kwargs,
) -> pd.DataFrame:
    """Build a multi-snapshot (temporal) model-ready dataset by calling the
    single-snapshot pipeline once per point-in-time cutoff, with NO changes
    to any of the feature functions above.

    Why this works with zero changes to compute_gtv_trend_and_volatility /
    compute_seasonality / compute_seasonal_deviation / compute_peer_comparison
    / compute_new_partner_ramp_feature: each of them operates purely on
    whatever GTV-history matrix it's handed — they take the rightmost N
    columns *positionally*, not by absolute month_index label. Truncating
    gtv_history to `month_index <= cutoff` before calling them naturally
    redefines "rightmost" to mean "as of that cutoff." compute_peer_comparison
    groups against avg_monthly_gtv from that same truncated call, so the
    peer cohort is automatically "as of that cutoff" too — no future
    information reaches backward into an earlier snapshot at any layer.

    Parameters
    ----------
    static_df_by_role : one static-attributes DataFrame per role (e.g.
        {"train": ..., "validation": ..., "test": ...}) — each role's own
        point-in-time attributes (tenure, engagement, NPS, label, etc.),
        already computed by the caller (generate_dataset.py) for that
        cutoff.
    gtv_history : the FULL, untruncated GTV history (all months) — this
        function does the truncation per role internally.
    snapshot_cutoffs : {role: cutoff_month_index}, inclusive — history is
        truncated to month_index <= cutoff for that role.
    role_order : the order roles should be processed in. The FIRST role
        processed establishes mature_mask_values (the cold-start neutral
        values, learned once and reused for every later role) — this
        should be the "train" role, so validation/test scoring uses the
        exact same definition of "neutral" that was used at training time.

    Returns
    -------
    A single long-format DataFrame: one row per partner per role, with a
    `snapshot_role` and `snapshot_cutoff_month` column added, all roles
    stacked together.
    """
    mature_mask_values: dict | None = None
    frames = []

    for role in role_order:
        cutoff = snapshot_cutoffs[role]
        truncated_history = gtv_history[gtv_history["month_index"] <= cutoff]

        model_df, mature_mask_values = build_model_ready_dataset(
            static_df_by_role[role],
            truncated_history,
            mature_mask_values=mature_mask_values,
            **kwargs,
        )
        model_df["snapshot_role"] = role
        model_df["snapshot_cutoff_month"] = cutoff
        frames.append(model_df)

    return pd.concat(frames, ignore_index=True)
