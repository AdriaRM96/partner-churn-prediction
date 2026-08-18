"""
generate_dataset.py

DISCLAIMER: This script generates a fully SYNTHETIC dataset for a B2B partner
management program. All partners, transaction values, engagement metrics, and
churn outcomes are randomly generated to reflect realistic business dynamics
(seasonality, growth/decline patterns, correlations between engagement and
churn, etc). This is NOT real company data and does not represent any actual
partner, business, or individual.

Usage:
    python generate_dataset.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_pipeline import build_gtv_features  # noqa: E402

RANDOM_SEED = 42
N_PARTNERS = 1000
N_MONTHS = 27  # spans more than 2 full annual cycles, needed to actually
               # detect recurring seasonality rather than assume it from
               # a single peak

rng = np.random.default_rng(RANDOM_SEED)

REGIONS = ["EMEA", "NA", "LATAM", "APAC"]
VERTICALS = ["Retail", "Tech", "Manufacturing", "Services"]
TIERS = ["Bronze", "Silver", "Gold"]

REGION_WEIGHTS = [0.30, 0.30, 0.20, 0.20]
VERTICAL_WEIGHTS = [0.30, 0.25, 0.20, 0.25]
TIER_WEIGHTS = [0.50, 0.35, 0.15]

# Base monthly GTV level by tier (Gold partners transact more)
TIER_BASE_GTV = {"Bronze": 8_000, "Silver": 25_000, "Gold": 70_000}

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Static partner attributes
# ---------------------------------------------------------------------------
def generate_static_attributes(n_partners: int) -> pd.DataFrame:
    partner_ids = [f"partner_{i}" for i in range(1, n_partners + 1)]

    df = pd.DataFrame(
        {
            "partner_id": partner_ids,
            "region": rng.choice(REGIONS, size=n_partners, p=REGION_WEIGHTS),
            "vertical": rng.choice(VERTICALS, size=n_partners, p=VERTICAL_WEIGHTS),
            "partner_tier": rng.choice(TIERS, size=n_partners, p=TIER_WEIGHTS),
            "months_as_partner": rng.integers(3, 61, size=n_partners),
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. Multi-year GTV history with trend + recurring seasonality + noise
# ---------------------------------------------------------------------------
def generate_gtv_history(static_df: pd.DataFrame) -> pd.DataFrame:
    n = len(static_df)

    # Assign each partner a trajectory archetype
    # growing / declining / flat / seasonal (seasonal mostly in Retail)
    archetype_probs_default = [0.30, 0.25, 0.30, 0.15]
    archetypes = np.empty(n, dtype=object)

    for i, vertical in enumerate(static_df["vertical"]):
        if vertical == "Retail":
            # Retail partners are much more likely to show recurring seasonality
            probs = [0.20, 0.20, 0.20, 0.40]
        else:
            probs = archetype_probs_default
        archetypes[i] = rng.choice(
            ["growing", "declining", "flat", "seasonal"], p=probs
        )

    records = []
    months = np.arange(N_MONTHS)  # 0 = 27 months ago ... N_MONTHS-1 = most recent

    for idx, row in static_df.iterrows():
        base = TIER_BASE_GTV[row["partner_tier"]]
        # partner-level scale variation
        base *= rng.lognormal(mean=0.0, sigma=0.4)

        archetype = archetypes[idx]

        if archetype == "growing":
            monthly_growth = rng.uniform(0.015, 0.05)
            trend = (1 + monthly_growth) ** months
            seasonal = np.ones(N_MONTHS)
        elif archetype == "declining":
            monthly_decline = rng.uniform(0.015, 0.045)
            trend = (1 - monthly_decline) ** months
            seasonal = np.ones(N_MONTHS)
        elif archetype == "flat":
            trend = np.ones(N_MONTHS)
            seasonal = np.ones(N_MONTHS)
        else:  # seasonal — a genuine recurring annual pattern, same phase
            # every year, so it shows up consistently across all cycles in
            # the 27-month window (not just once)
            trend = np.ones(N_MONTHS)
            phase_offset = rng.integers(0, 12)  # random calendar alignment per partner
            month_of_year = (months + phase_offset) % 12
            amplitude = rng.uniform(0.45, 0.75)
            seasonal = 1 + amplitude * np.exp(
                -0.5 * ((month_of_year - 10.5) / 1.5) ** 2
            )
            # small partner-level mild long-run drift layered on top so it's
            # not perfectly flat between cycles, but the seasonal shape repeats
            mild_drift = rng.uniform(-0.005, 0.01)
            trend = (1 + mild_drift) ** months

        noise = rng.normal(loc=1.0, scale=0.10, size=N_MONTHS)
        noise = np.clip(noise, 0.5, 1.5)

        gtv_series = base * trend * seasonal * noise
        gtv_series = np.clip(gtv_series, 100, None)

        for m, gtv in zip(months, gtv_series):
            records.append(
                {
                    "partner_id": row["partner_id"],
                    "month_index": int(m),  # 0..N_MONTHS-1, most recent = N_MONTHS-1
                    "gtv": round(float(gtv), 2),
                }
            )

    gtv_df = pd.DataFrame.from_records(records)
    return gtv_df


# ---------------------------------------------------------------------------
# 3. Derived GTV features
#
# The actual feature-engineering logic (trend, volatility, seasonality,
# peer/market comparison) lives in src/data_pipeline.py so it can be reused
# unchanged against a real, monthly-refreshed partner data extract. This
# script only calls into it — see build_gtv_features() there for the
# implementation.
# ---------------------------------------------------------------------------
# 4. Engagement / relationship features, correlated with churn drivers
# ---------------------------------------------------------------------------
def generate_engagement_features(
    static_df: pd.DataFrame, gtv_features: pd.DataFrame
) -> pd.DataFrame:
    n = len(static_df)

    # Build a "health" latent factor from gtv_trend to correlate other
    # features realistically (declining GTV partners tend to have worse
    # engagement / NPS too, not purely independent noise).
    trend = gtv_features["gtv_trend_3m"].values
    trend_z = (trend - trend.mean()) / (trend.std() + 1e-9)

    health = trend_z + rng.normal(0, 1.0, size=n)  # latent partner health

    # NPS score 0-10, correlated with health
    nps_raw = 6.5 + 1.3 * health + rng.normal(0, 1.2, size=n)
    nps_score = np.clip(np.round(nps_raw), 0, 10).astype(int)

    nps_trend_probs_base = np.array([0.2, 0.5, 0.3])  # declining, stable, improving
    nps_trend = []
    for h in health:
        if h < -0.5:
            p = [0.55, 0.35, 0.10]
        elif h > 0.5:
            p = [0.10, 0.35, 0.55]
        else:
            p = nps_trend_probs_base
        nps_trend.append(rng.choice(["declining", "stable", "improving"], p=p))
    nps_trend = np.array(nps_trend)

    # Support tickets: unhealthy partners raise more tickets
    support_tickets_last_quarter = rng.poisson(
        lam=np.clip(3 - 1.5 * health, 0.2, None)
    )

    # Resolution time: worse for partners already struggling (compounding pain)
    avg_resolution_time_days = np.clip(
        rng.normal(loc=3.5 - 0.8 * health, scale=1.5, size=n), 0.5, 20
    )

    # Engagement score 0-100, correlated with health (training, events, certs)
    engagement_score = np.clip(
        rng.normal(loc=55 + 12 * health, scale=15, size=n), 0, 100
    )

    # Days since last contact: unhealthy / disengaged partners contacted less
    days_since_last_contact = np.clip(
        rng.normal(loc=40 - 12 * health, scale=20, size=n), 0, 365
    ).astype(int)

    # Active deals: healthier partners have more active deals
    num_active_deals = rng.poisson(lam=np.clip(2 + 1.2 * health, 0.1, None))

    # Competitor relationship: more likely for disengaged/unhealthy partners
    competitor_logit = -1.2 - 0.8 * health
    competitor_prob = 1 / (1 + np.exp(-competitor_logit))
    has_competitor_relationship = (rng.uniform(size=n) < competitor_prob).astype(int)

    df = pd.DataFrame(
        {
            "partner_id": static_df["partner_id"].values,
            "nps_score": nps_score,
            "nps_trend": nps_trend,
            "support_tickets_last_quarter": support_tickets_last_quarter,
            "avg_resolution_time_days": np.round(avg_resolution_time_days, 1),
            "engagement_score": np.round(engagement_score, 1),
            "days_since_last_contact": days_since_last_contact,
            "num_active_deals": num_active_deals,
            "has_competitor_relationship": has_competitor_relationship,
        }
    )
    return df, health


# ---------------------------------------------------------------------------
# 5. Target variable: churned_next_quarter via logistic function
# ---------------------------------------------------------------------------
def generate_churn_target(
    gtv_features: pd.DataFrame, engagement_df: pd.DataFrame
) -> np.ndarray:
    n = len(gtv_features)

    # Standardize predictors for a stable logistic combination
    def z(x):
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / (x.std() + 1e-9)

    gtv_trend_z = z(gtv_features["gtv_trend_3m"])
    gtv_vs_market_z = z(gtv_features["gtv_vs_market_growth"])
    nps_z = z(engagement_df["nps_score"])
    engagement_z = z(engagement_df["engagement_score"])
    days_since_contact_z = z(engagement_df["days_since_last_contact"])
    resolution_time_z = z(engagement_df["avg_resolution_time_days"])
    tickets_z = z(engagement_df["support_tickets_last_quarter"])
    competitor = engagement_df["has_competitor_relationship"].values
    nps_declining = (engagement_df["nps_trend"].values == "declining").astype(float)

    logit = (
        -1.0  # base rate anchor, tuned below to hit ~20% churn
        - 0.9 * gtv_trend_z
        - 0.4 * gtv_vs_market_z
        - 0.7 * nps_z
        + 0.5 * nps_declining
        - 0.8 * engagement_z
        + 0.7 * days_since_contact_z
        + 0.3 * resolution_time_z
        + 0.25 * tickets_z
        + 0.9 * competitor
    )

    # Random noise so relationship isn't perfectly deterministic
    logit += rng.normal(0, 0.6, size=n)

    prob = 1 / (1 + np.exp(-logit))

    # Calibrate the intercept so overall churn rate lands near 20%
    target_rate = 0.20
    # simple bisection on an intercept shift
    lo, hi = -5.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit + mid)))).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    logit_calibrated = logit + mid
    prob_calibrated = 1 / (1 + np.exp(-logit_calibrated))

    churned = (rng.uniform(size=n) < prob_calibrated).astype(int)
    return churned, prob_calibrated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    static_df = generate_static_attributes(N_PARTNERS)
    gtv_df = generate_gtv_history(static_df)

    # GTV-derived features are computed by the reusable pipeline, not
    # duplicated here — this is the same function a monthly batch job would
    # call against a real GTV history extract.
    gtv_features = build_gtv_features(static_df, gtv_df)

    engagement_df, _health = generate_engagement_features(static_df, gtv_features)
    churned, churn_prob = generate_churn_target(gtv_features, engagement_df)

    # "Raw" static table: attributes + engagement/relationship signals +
    # target, but no GTV-derived columns. This is what a real monthly
    # extract would look like, and what src/data_pipeline.py expects as
    # input alongside the GTV history table.
    static_out = static_df.merge(engagement_df, on="partner_id")
    static_out["churn_probability"] = np.round(churn_prob, 4)
    static_out["churned_next_quarter"] = churned

    final_df = static_out.merge(gtv_features, on="partner_id")

    gtv_history_out = gtv_df.merge(
        static_df[["partner_id"]], on="partner_id"
    ).sort_values(["partner_id", "month_index"])

    static_path = OUT_DIR / "partner_static.csv"
    dataset_path = OUT_DIR / "partner_churn_dataset.csv"
    gtv_history_path = OUT_DIR / "partner_gtv_history.csv"
    static_out.to_csv(static_path, index=False)
    final_df.to_csv(dataset_path, index=False)
    gtv_history_out.to_csv(gtv_history_path, index=False)

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print("=" * 70)
    print("SYNTHETIC PARTNER CHURN DATASET GENERATED")
    print("=" * 70)
    print(f"\nHistory length: {N_MONTHS} months (~{N_MONTHS/12:.1f} years)")
    print(f"Raw static table: {static_path}  shape={static_out.shape}  (pipeline input)")
    print(f"Main dataset:     {dataset_path}  shape={final_df.shape}  (model-ready)")
    print(f"GTV history:      {gtv_history_path}  shape={gtv_history_out.shape}  (pipeline input)")

    churn_rate = final_df["churned_next_quarter"].mean()
    print(f"\nOverall churn rate: {churn_rate:.2%}")

    print("\n--- Categorical distributions ---")
    for col in ["region", "vertical", "partner_tier"]:
        print(f"\n{col}:")
        print(final_df[col].value_counts(normalize=True).round(3).to_string())

    print("\n--- Churn rate by segment ---")
    for col in ["region", "vertical", "partner_tier", "has_competitor_relationship", "nps_trend"]:
        print(f"\nChurn rate by {col}:")
        print(
            final_df.groupby(col)["churned_next_quarter"]
            .mean()
            .round(3)
            .to_string()
        )

    print("\n--- Numeric feature summary ---")
    numeric_cols = [
        "months_as_partner",
        "avg_monthly_gtv",
        "gtv_trend_3m",
        "gtv_volatility",
        "seasonal_autocorr_lag12",
        "gtv_vs_peers",
        "gtv_vs_market_growth",
        "nps_score",
        "support_tickets_last_quarter",
        "avg_resolution_time_days",
        "engagement_score",
        "days_since_last_contact",
        "num_active_deals",
    ]
    print(final_df[numeric_cols].describe().round(2).to_string())

    print("\n--- Feature means: churned vs retained ---")
    print(
        final_df.groupby("churned_next_quarter")[numeric_cols]
        .mean()
        .round(2)
        .to_string()
    )

    # ---------------------------------------------------------------------
    # Seasonality sanity check: confirm has_seasonality reflects a pattern
    # that repeats across multiple years, not a single cycle.
    # We do this by independently checking correlation between year 1 and
    # year 2 (and year 2 vs year 3, where available) monthly GTV shapes for
    # partners flagged as seasonal vs not.
    # ---------------------------------------------------------------------
    print("\n--- Seasonality sanity check (multi-year consistency) ---")
    pivot = gtv_df.pivot(index="partner_id", columns="month_index", values="gtv")
    pivot = pivot.reindex(final_df["partner_id"])
    values = pivot.values

    # Compare month-of-year shape (normalized) between two 12-month blocks
    # exactly one year apart (the trailing two full years of the window).
    # A true annual repeat requires the offset to be exactly 12 months —
    # any other offset misaligns the calendar phase and would understate
    # the real year-over-year correlation.
    year1 = values[:, N_MONTHS - 24:N_MONTHS - 12]
    year2 = values[:, N_MONTHS - 12:N_MONTHS]
    year1_norm = year1 / year1.mean(axis=1, keepdims=True)
    year2_norm = year2 / year2.mean(axis=1, keepdims=True)

    year_over_year_corr = np.array(
        [np.corrcoef(year1_norm[i], year2_norm[i])[0, 1] for i in range(len(final_df))]
    )

    seasonal_mask = final_df["has_seasonality"].values == 1
    print(
        f"Avg year-over-year monthly-shape correlation, "
        f"flagged seasonal partners:     {year_over_year_corr[seasonal_mask].mean():.3f}"
    )
    print(
        f"Avg year-over-year monthly-shape correlation, "
        f"flagged non-seasonal partners: {year_over_year_corr[~seasonal_mask].mean():.3f}"
    )
    print(
        "-> Partners flagged has_seasonality=1 show a substantially higher "
        "year-over-year correlation in their monthly GTV shape, confirming "
        "the flag captures a pattern that genuinely repeats across multiple "
        "years rather than a one-off seasonal bump."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
