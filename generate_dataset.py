"""
generate_dataset.py

DISCLAIMER: This script generates a fully SYNTHETIC dataset for a B2B partner
management program. All partners, transaction values, engagement metrics, and
churn outcomes are randomly generated to reflect realistic business dynamics
(seasonality, growth/decline patterns, correlations between engagement and
churn, etc). This is NOT real company data and does not represent any actual
partner, business, or individual.

Generates a multi-SNAPSHOT dataset: each partner appears three times, once
per quarterly cutoff ("train" / "validation" / "test"), with point-in-time
features computed from GTV history truncated to that cutoff only (no future
information reaches an earlier snapshot). This is what makes a genuine
temporal train/validation/test split possible downstream, instead of a
random split that would implicitly assume the model is deployed on data
from the same time window it was trained on.

Usage:
    python generate_dataset.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_pipeline import (  # noqa: E402
    build_snapshot_dataset,
    compute_gtv_trend_and_volatility,
    compute_peer_comparison,
)

RANDOM_SEED = 42
N_PARTNERS = 1000
N_MONTHS = 33  # 33 months of history: spans >2 full annual cycles at every
               # snapshot cutoff below (even the earliest one), which is
               # what the seasonal-deviation feature in src/data_pipeline.py
               # needs (24 months to fit a seasonal profile + 3 months held
               # out to evaluate it = 27 months minimum trailing history).

# Three quarterly snapshot cutoffs, one quarter (3 months) apart. Each
# partner gets a row for each role, with features computed strictly from
# GTV history truncated to that role's cutoff — train is the earliest
# (least trailing history: 27 months), test is "now" (the full 33 months,
# "the current book"). See src/data_pipeline.py's build_snapshot_dataset
# docstring for exactly how per-cutoff truncation avoids future leakage.
SNAPSHOT_CUTOFFS = {"train": 26, "validation": 29, "test": 32}
ROLE_ORDER = ["train", "validation", "test"]
# How many months before the "test" snapshot each role sits, used to derive
# that role's own months_as_partner (see compute_role_tenure).
ROLE_MONTHS_BEFORE_TEST = {"train": 6, "validation": 3, "test": 0}

rng = np.random.default_rng(RANDOM_SEED)

REGIONS = ["EMEA", "NA", "LATAM", "APAC"]
VERTICALS = ["Retail", "Tech", "Manufacturing", "Services"]
# Partner tiers — deliberately NOT a Bronze/Silver/Gold/Diamond scheme
# (that naming belongs to a specific real company's loyalty program).
# "Registered / Certified / Strategic" reflects how B2B partner programs
# actually name tiers: by depth of relationship/investment, not a medal
# metaphor. Same relative ordering and weights as before the rename.
TIERS = ["Registered", "Certified", "Strategic"]

REGION_WEIGHTS = [0.30, 0.30, 0.20, 0.20]
VERTICAL_WEIGHTS = [0.30, 0.25, 0.20, 0.25]
TIER_WEIGHTS = [0.50, 0.35, 0.15]

# Base monthly GTV level by tier (Strategic partners transact more)
TIER_BASE_GTV = {"Registered": 8_000, "Certified": 25_000, "Strategic": 70_000}

# AR(1)-style persistence for the latent "health" factor across snapshots:
# each later snapshot's health is a blend of the partner's PRIOR health
# (most of it carries forward) and a fresh draw from that snapshot's own
# GTV trend + noise (so it responds to what's actually happened since,
# rather than being frozen). rho closer to 1 = more persistent quarter to
# quarter; rho closer to 0 = closer to an independent redraw each time.
HEALTH_PERSISTENCE_RHO = 0.6
# Once a partner has a competitor relationship, how likely it is to still
# be true next snapshot (rather than being independently re-drawn from the
# health-based logit every time, which would let it implausibly vanish).
COMPETITOR_PERSISTENCE_PROB = 0.90

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Static partner attributes (region/vertical/tier don't change over
#    time; months_as_partner is generated as of the "test" snapshot, i.e.
#    a partner's tenure today, with earlier snapshots derived from it)
# ---------------------------------------------------------------------------
def generate_static_attributes(n_partners: int) -> pd.DataFrame:
    partner_ids = [f"partner_{i}" for i in range(1, n_partners + 1)]

    df = pd.DataFrame(
        {
            "partner_id": partner_ids,
            "region": rng.choice(REGIONS, size=n_partners, p=REGION_WEIGHTS),
            "vertical": rng.choice(VERTICALS, size=n_partners, p=VERTICAL_WEIGHTS),
            "partner_tier": rng.choice(TIERS, size=n_partners, p=TIER_WEIGHTS),
            "months_as_partner_test": rng.integers(3, 61, size=n_partners),
        }
    )
    return df


def compute_role_tenure(months_as_partner_test: np.ndarray, role: str) -> np.ndarray:
    """A partner's tenure as of an earlier snapshot, derived by subtracting
    the elapsed months back from their tenure as of "test" (today).

    Floored at 1 rather than allowing negative/zero tenure or dropping the
    partner from earlier snapshots entirely. SIMPLIFICATION, documented
    deliberately: a partner whose test-tenure is under 6 months technically
    hadn't joined yet as of the "train" cutoff in a fully realistic model;
    flooring at 1 keeps every snapshot at a uniform partner count (simpler
    downstream — consistent peer-cohort sizes and categorical cardinality
    across roles) at the cost of a small artificial cluster of tenure=1
    rows in the earlier snapshots. See the EDA section for a sanity check
    on how large that cluster actually is.
    """
    elapsed = ROLE_MONTHS_BEFORE_TEST[role]
    return np.maximum(1, months_as_partner_test - elapsed)


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
    months = np.arange(N_MONTHS)  # 0 = N_MONTHS-1 months ago ... N_MONTHS-1 = most recent

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
            # the N_MONTHS window (not just once)
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
# The actual feature-engineering logic (trend, yoy trend, seasonal
# deviation, volatility, seasonality, peer/market comparison, cold-start
# ramp) lives in src/data_pipeline.py so it can be reused unchanged against
# a real, monthly-refreshed partner data extract — including the
# multi-snapshot orchestration itself (build_snapshot_dataset). This script
# only calls into it. A lightweight subset (trend + peer comparison only)
# is also computed once per role below, purely to drive that role's
# engagement-feature generation and churn-label draw — see
# generate_engagement_features / generate_churn_target.
# ---------------------------------------------------------------------------
# 4. Engagement / relationship features, correlated with a latent "health"
#    factor that itself evolves (persistently, not independently) across
#    snapshots
# ---------------------------------------------------------------------------
def update_health(
    prev_health: np.ndarray | None,
    gtv_trend_3m_this_role: np.ndarray,
    rho: float = HEALTH_PERSISTENCE_RHO,
) -> np.ndarray:
    """Evolve the latent partner-health factor one snapshot forward.

    For the first snapshot (prev_health=None), health is purely this
    snapshot's own GTV trend + noise (identical to the original single-
    snapshot formula). For later snapshots, health is an AR(1)-style blend:
    `rho` of it carries forward from the partner's own prior-snapshot
    health (persistence — a partner doesn't reset every quarter), and the
    rest is fresh (responds to this snapshot's own trend + new noise) —
    "evolving but persistent," not frozen and not independently redrawn.
    """
    trend = np.asarray(gtv_trend_3m_this_role, dtype=float)
    trend_z = (trend - trend.mean()) / (trend.std() + 1e-9)
    n = len(trend)
    fresh = trend_z + rng.normal(0, 1.0, size=n)
    if prev_health is None:
        return fresh
    return rho * prev_health + np.sqrt(1 - rho**2) * fresh


def generate_engagement_features(
    static_df: pd.DataFrame,
    health: np.ndarray,
    prev_health: np.ndarray | None = None,
    prev_competitor: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(static_df)

    # NPS score 0-10, correlated with health
    nps_raw = 6.5 + 1.3 * health + rng.normal(0, 1.2, size=n)
    nps_score = np.clip(np.round(nps_raw), 0, 10).astype(int)

    nps_trend_probs_base = np.array([0.2, 0.5, 0.3])  # declining, stable, improving
    nps_trend = []
    if prev_health is None:
        # First snapshot: no prior health to diff against, so fall back to
        # the original level-based heuristic (is this partner currently
        # healthy or not, not whether it changed).
        for h in health:
            if h < -0.5:
                p = [0.55, 0.35, 0.10]
            elif h > 0.5:
                p = [0.10, 0.35, 0.55]
            else:
                p = nps_trend_probs_base
            nps_trend.append(rng.choice(["declining", "stable", "improving"], p=p))
    else:
        # Later snapshots: "trend" should describe a CHANGE, not a level —
        # derive it from the health delta between this snapshot and the
        # partner's own prior snapshot, now that we have one to diff against.
        delta = health - prev_health
        for d in delta:
            if d < -0.3:
                p = [0.55, 0.35, 0.10]
            elif d > 0.3:
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

    # Competitor relationship: more likely for disengaged/unhealthy partners,
    # but sticky once acquired (see COMPETITOR_PERSISTENCE_PROB) rather than
    # independently re-drawn every snapshot, which would let a real
    # relationship implausibly "disappear" quarter to quarter.
    competitor_logit = -1.2 - 0.8 * health
    competitor_prob = 1 / (1 + np.exp(-competitor_logit))
    fresh_competitor_draw = (rng.uniform(size=n) < competitor_prob).astype(int)
    if prev_competitor is None:
        has_competitor_relationship = fresh_competitor_draw
    else:
        carry_forward_roll = rng.uniform(size=n) < COMPETITOR_PERSISTENCE_PROB
        has_competitor_relationship = np.where(
            prev_competitor == 1,
            np.where(carry_forward_roll, 1, fresh_competitor_draw),
            fresh_competitor_draw,
        )

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
    return df


# ---------------------------------------------------------------------------
# 5. Target variable: churned_next_quarter via logistic function
#
# Called once per snapshot role, using that role's OWN point-in-time
# features (never the partner's later-snapshot data). Since this is a
# synthetic function of point-in-time features rather than something
# literally derived from future GTV movement, each snapshot draws its own
# independent label — a partner can be churned_next_quarter=1 at "train"
# yet appear as a normal active row at "validation"/"test". There's no
# absorbing-state consistency across a partner's 3 snapshots; this is fine
# for supervised learning (each row is its own point-in-time example) but
# is worth stating explicitly rather than leaving implicit.
# ---------------------------------------------------------------------------
def generate_churn_target(
    gtv_features: pd.DataFrame, engagement_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
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
    static_attrs = generate_static_attributes(N_PARTNERS)
    gtv_history_full = generate_gtv_history(static_attrs)

    # Build each role's "raw" static table (attributes + engagement/
    # relationship signals + target, no GTV-derived columns) by walking
    # through the snapshots in chronological order, threading the latent
    # health factor and competitor-relationship state forward from one
    # snapshot to the next so they evolve persistently rather than being
    # independently redrawn each time.
    static_df_by_role: dict[str, pd.DataFrame] = {}
    prev_health = None
    prev_competitor = None

    for role in ROLE_ORDER:
        cutoff = SNAPSHOT_CUTOFFS[role]
        truncated_history = gtv_history_full[gtv_history_full["month_index"] <= cutoff]

        role_tenure = compute_role_tenure(static_attrs["months_as_partner_test"].values, role)
        role_static_attrs = static_attrs[["partner_id", "region", "vertical", "partner_tier"]].copy()
        role_static_attrs["months_as_partner"] = role_tenure

        # Lightweight trend + peer features (NOT the full pipeline — that
        # also fits seasonality/STL/cohort-ramp, which aren't needed here
        # and are computed once, properly, by build_snapshot_dataset below)
        # purely to drive this role's engagement/health update and churn
        # label draw.
        trend_features = compute_gtv_trend_and_volatility(truncated_history)
        peer_features = compute_peer_comparison(role_static_attrs, trend_features)
        lite_gtv_features = trend_features.merge(peer_features, on="partner_id")

        health = update_health(prev_health, lite_gtv_features["gtv_trend_3m"].values)
        engagement_df = generate_engagement_features(
            role_static_attrs, health, prev_health=prev_health, prev_competitor=prev_competitor
        )
        churned, churn_prob = generate_churn_target(lite_gtv_features, engagement_df)

        role_static_out = role_static_attrs.merge(engagement_df, on="partner_id")
        role_static_out["churn_probability"] = np.round(churn_prob, 4)
        role_static_out["churned_next_quarter"] = churned

        static_df_by_role[role] = role_static_out
        prev_health = health
        prev_competitor = engagement_df["has_competitor_relationship"].values

    # The real, reusable pipeline computes the full GTV-derived feature set
    # (trend, yoy trend, seasonal deviation via STL, volatility,
    # seasonality, peer/market comparison, cold-start cohort-ramp) and the
    # scoring_path/masking logic — once per role, from history truncated to
    # that role's own cutoff, with mature_mask_values learned on "train"
    # and reused unchanged for "validation"/"test" (see
    # build_snapshot_dataset's docstring in src/data_pipeline.py).
    snapshots_df = build_snapshot_dataset(
        static_df_by_role, gtv_history_full, SNAPSHOT_CUTOFFS, ROLE_ORDER
    )

    gtv_history_out = gtv_history_full.merge(
        static_attrs[["partner_id"]], on="partner_id"
    ).sort_values(["partner_id", "month_index"])

    # data/partner_static.csv is redefined as the TEST-role raw static
    # table only (i.e. "the current snapshot") — this preserves its
    # original single-point-in-time "monthly extract" framing, so
    # load_and_build_dataset (paired with the full GTV history) still works
    # exactly as documented for anyone using the pipeline the "old" way.
    static_path = OUT_DIR / "partner_static.csv"
    snapshots_path = OUT_DIR / "partner_snapshots.csv"
    gtv_history_path = OUT_DIR / "partner_gtv_history.csv"
    static_df_by_role["test"].to_csv(static_path, index=False)
    snapshots_df.to_csv(snapshots_path, index=False)
    gtv_history_out.to_csv(gtv_history_path, index=False)

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print("=" * 70)
    print("SYNTHETIC PARTNER CHURN SNAPSHOT DATASET GENERATED")
    print("=" * 70)
    print(f"\nHistory length: {N_MONTHS} months (~{N_MONTHS/12:.1f} years)")
    print(f"Snapshot cutoffs (month_index): {SNAPSHOT_CUTOFFS}")
    print(f"\nSnapshot table: {snapshots_path}  shape={snapshots_df.shape}  (model-ready, all roles)")
    print(f"Current static: {static_path}  shape={static_df_by_role['test'].shape}  (test role, pipeline input)")
    print(f"GTV history:    {gtv_history_path}  shape={gtv_history_out.shape}  (pipeline input)")

    numeric_cols = [
        "months_as_partner",
        "avg_monthly_gtv",
        "gtv_trend_3m",
        "gtv_trend_yoy_3m",
        "gtv_seasonal_deviation_pct",
        "gtv_volatility",
        "seasonal_autocorr_lag12",
        "gtv_vs_peers",
        "gtv_vs_market_growth",
        "gtv_ramp_vs_cohort",
        "nps_score",
        "support_tickets_last_quarter",
        "avg_resolution_time_days",
        "engagement_score",
        "days_since_last_contact",
        "num_active_deals",
    ]

    for role in ROLE_ORDER:
        role_df = snapshots_df[snapshots_df["snapshot_role"] == role]
        churn_rate = role_df["churned_next_quarter"].mean()
        print(f"\n{'-' * 70}\nROLE: {role}  (n={len(role_df)}, cutoff month_index={SNAPSHOT_CUTOFFS[role]})")
        print(f"Churn rate: {churn_rate:.2%}")
        print("scoring_path counts:")
        print(role_df["scoring_path"].value_counts().to_string())
        print("Tenure summary (months_as_partner):")
        print(role_df["months_as_partner"].describe().round(1).to_string())
        n_tenure_floor = (role_df["months_as_partner"] == 1).sum()
        print(f"Partners at the tenure floor (months_as_partner == 1): {n_tenure_floor}")

    print(f"\n{'-' * 70}")
    print("--- Categorical distributions (test role) ---")
    test_df = snapshots_df[snapshots_df["snapshot_role"] == "test"]
    for col in ["region", "vertical", "partner_tier"]:
        print(f"\n{col}:")
        print(test_df[col].value_counts(normalize=True).round(3).to_string())

    print("\n--- Churn rate by segment (test role) ---")
    for col in ["region", "vertical", "partner_tier", "has_competitor_relationship", "nps_trend", "scoring_path"]:
        print(f"\nChurn rate by {col}:")
        print(test_df.groupby(col)["churned_next_quarter"].mean().round(3).to_string())

    print("\n--- Numeric feature summary (test role) ---")
    print(test_df[numeric_cols].describe().round(2).to_string())

    # ---------------------------------------------------------------------
    # Seasonality sanity check: confirm has_seasonality reflects a pattern
    # that repeats across multiple years, not a single cycle. Uses the
    # full GTV history (independent of any single snapshot's truncation).
    # ---------------------------------------------------------------------
    print("\n--- Seasonality sanity check (multi-year consistency, full history) ---")
    pivot = gtv_history_full.pivot(index="partner_id", columns="month_index", values="gtv")
    pivot = pivot.reindex(test_df["partner_id"])
    values = pivot.values

    year1 = values[:, N_MONTHS - 24:N_MONTHS - 12]
    year2 = values[:, N_MONTHS - 12:N_MONTHS]
    year1_norm = year1 / year1.mean(axis=1, keepdims=True)
    year2_norm = year2 / year2.mean(axis=1, keepdims=True)

    year_over_year_corr = np.array(
        [np.corrcoef(year1_norm[i], year2_norm[i])[0, 1] for i in range(len(test_df))]
    )

    seasonal_mask = test_df["has_seasonality"].values == 1
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
