# Production Considerations

This document covers how the partner churn pipeline (`src/data_pipeline.py`) and model would actually run in production: retraining cadence, drift monitoring, unscheduled-retrain triggers, and pre-scoring data quality checks. It assumes the deployment target is a monthly partner data refresh, matching how the underlying features (GTV trend, seasonality, engagement) are defined.

This version reflects a significant methodology upgrade over an earlier pass: a fixed seasonality-trend feature, a cold-start segment for new partners, a genuine multi-snapshot temporal structure (`build_snapshot_dataset`), a cost-derived and capacity-aware risk-tiering scheme (replacing arbitrary thresholds), SHAP-based explanations, and a complementary survival-analysis view. Each section below has been updated where that upgrade changes the operational picture.

## 1. Retraining cadence

**Retrain monthly, aligned with the GTV data refresh — but retraining now means rolling the snapshot window forward, not just refitting on a growing static extract.**

The dataset is no longer a single point-in-time table. `generate_dataset.py` (and, in production, whatever job replaces it) produces three quarterly snapshots per partner via `src/data_pipeline.py`'s `build_snapshot_dataset`, each with features computed strictly from GTV history truncated to that snapshot's own cutoff. In production, each monthly cycle should:

- generate a new "current" snapshot from the latest data,
- demote the prior `test` snapshot to `validation` and the prior `validation` to `train` (or, more robustly, accumulate a longer historical partner-quarter table and re-slice the three most recent snapshots for train/validation/test each cycle),
- retrain and re-select the model on the refreshed `train`/`validation` pair, and
- report/monitor against the new `test` (the current book).

This keeps the model evaluated the way it will actually be used — trained on the past, scored on the present — every cycle, not just at the last major methodology change.

Two things to keep fixed across retrains unless deliberately revisited:
- **The 27+ month trailing-history floor at every snapshot cutoff.** `has_seasonality` needs ≥24 months (two full annual cycles) to distinguish a genuine recurring pattern from a one-off bump, and the seasonal-deviation feature (`compute_seasonal_deviation`, STL-based) needs 24 months for the profile fit plus a 3-month holdout — 27 months minimum, at *every* snapshot, not just the latest. Shortening the snapshot spacing or the total history window without re-checking this floor would silently degrade both features.
- **The 3-month trend window (`gtv_trend_3m`) and the 12-month year-over-year lag (`gtv_trend_yoy_3m`).** Changing either changes the meaning of the feature and would require re-validating it against churn outcomes, not just re-fitting the model.
- **The cold-start tenure threshold (`DEFAULT_COLD_START_TENURE_MONTHS = 9`).** This determines which partners get the mature trend/seasonality block vs. the cohort-ramp benchmark — changing it reshuffles which partners are in which feature regime and should be treated as a deliberate, re-validated decision, not a casual tuning knob.

**Full re-training** (refit all four candidate models, re-run model selection) should happen every retrain cycle — it's cheap relative to the cost of scoring against a stale model, and there's no reason to let feature/label drift accumulate between fits.

## 2. Monitoring for model and data drift

Monitor three distinct things: whether the *data* going into the pipeline is changing shape, whether the *model* is still performing, and — new this round — whether the *segmentation and thresholding logic* is still behaving sensibly.

### Data drift (input distributions)

Track, per scoring cycle, the distribution of each model feature against a recent training-period baseline:
- **Population-level shifts**: mean/median and standard deviation of `gtv_trend_3m`, `gtv_trend_yoy_3m`, `engagement_score`, `nps_score`, `days_since_last_contact` month over month. A sudden shift (e.g., average `days_since_last_contact` jumping because a CRM sync broke) is a data problem, not a market signal, and should be caught before it reaches the model.
- **Categorical mix**: share of partners by `region`, `vertical`, `partner_tier`. A sharp change usually means an upstream extract changed (e.g., a new region code appeared, or a segment was dropped from the feed) rather than that the partner base actually shifted that fast.
- **`has_seasonality` prevalence**: this should be fairly stable month to month (it's derived from ≥27 months of history, so any single new month barely moves it). A large swing is a strong signal something upstream broke, not that partner seasonality genuinely changed.
- **`scoring_path` mix** (`mature` vs. `new_partner`): should move gradually, tracking actual partner-program growth. A sudden jump usually means either a real onboarding surge (worth knowing about on its own) or a data problem upstream of the tenure field.
- **STL fallback count** from `compute_seasonal_deviation`: this should stay near zero given the 27+ month trailing-history floor. A rising count means something shortened the available history for seasonal partners upstream — investigate before trusting `gtv_seasonal_deviation_pct` for those partners that cycle.
- **Missing/null rates** per column — a rising null rate on any input feature, especially engagement or NPS fields, points to instrumentation gaps (see Section 4).

### Model/concept drift (is the model still right?)

- **Recall and precision on newly-resolved outcomes.** Each month, some partners flagged in prior cycles will have resolved (churned or retained). Track recall and precision on that resolved cohort over time. A sustained drop in recall means the model is missing more at-risk partners than it used to — the single most important metric to watch, given that a missed churn is the costliest failure mode here.
- **Calibration of the capacity-based tiers.** Confirm the actual churn rate within each tier (🟢 Low / 🟡 Medium / 🔴 High, sized by `expected_cost_of_inaction` ranking — see Section 5 of the notebook, not a fixed probability cutoff) still increases monotonically from Low to High. If "High Risk" partners start churning at a rate not meaningfully different from "Medium," the ranking has stopped being useful for prioritization even if the underlying model hasn't technically broken.
- **Stability of the cost-curve shape.** The notebook's cost-threshold analysis found that, under the current cost assumptions (flat $100 false-positive cost, false-negative cost scaled by GTV), the expected-cost curve doesn't have an interior minimum — it keeps falling as the threshold approaches zero. Re-run this check each retrain: if the curve *does* develop a stable interior minimum, that's a sign the underlying cost dynamics have shifted (e.g., partner GTV values compressed, or the assumed $100 call cost needs revisiting) and worth a manual look before continuing to use the capacity-based ranking as-is.
- **Feature importance / SHAP stability.** Re-check monthly whether the same handful of features (NPS, engagement, contact gap, competitor relationship, GTV trend) remain the top drivers across the Logistic Regression coefficients, tree-model importances, and SHAP values. A reshuffle is worth investigating — it can mean a genuine shift in what drives churn, or a sign that a feature has quietly become unreliable (see drift above).
- **Survival-model drift.** Track the median predicted survival time (time-to-churn) from the Cox Proportional Hazards model cycle over cycle, as a signal distinct from the classifier's own metrics — a sudden shift in typical predicted urgency is worth understanding even if classification metrics look stable.

## 3. What would trigger an out-of-cycle retrain

Retrain immediately, outside the normal monthly schedule, if any of the following happen:

1. **Recall on resolved outcomes drops below an agreed floor** (e.g., falls more than 10 percentage points below its trailing 3-month average). This is the clearest sign the model is no longer doing its job.
2. **A material change to the partner program itself** — a new tier structure, a pricing change, a shift in how NPS or engagement is measured or collected, or a new competitor entering the market. Any of these can change *what predicts churn*, not just the current data snapshot, and waiting for the next scheduled retrain risks scoring against stale relationships.
3. **A data pipeline change upstream** — a new CRM, a change to how `days_since_last_contact` or `support_tickets` is logged, a new GTV reporting system. Even if the retrain "succeeds," the model may be learning from redefined features rather than the same signal it was trained on; this should trigger both a retrain and a fresh validation pass, not just a retrain.
4. **A data quality check fails hard** (see Section 4) in a way that can't be resolved before the next scheduled cycle — better to skip a scoring cycle and investigate than to hand partner managers a list built on broken inputs.
5. **A sustained, unexplained shift in the overall churn rate** (e.g., churn rate doubles or halves month over month with no known business cause) — this could be a real market event worth understanding before the model quietly "adapts" to it, or a sign of upstream data corruption.
6. **The cost-model assumptions materially change** — e.g., the team's actual outreach capacity changes significantly, or there's a real update to what a missed churn costs (average partner GTV shifts a lot, or the effective cost of an outreach call changes). These feed directly into the capacity-based tiering and should be revisited deliberately, not left stale.

## 4. Data quality checks before each scoring cycle

Run these against the raw inputs before calling `build_snapshot_dataset` / `build_model_ready_dataset` / `load_and_build_dataset`, and halt the cycle (falling back to the last known-good scored list) if any hard check fails.

**Schema and completeness (hard checks — halt on failure):**
- Every expected column is present with the expected type.
- `partner_id` is unique in the static table and every `partner_id` in the GTV history has a corresponding static record (and vice versa).
- GTV history has a contiguous, complete `month_index` sequence per partner with no gaps — a gap silently corrupts `gtv_trend_3m`, `gtv_trend_yoy_3m`, `gtv_volatility`, `gtv_seasonal_deviation_pct`, and the seasonality autocorrelation, all of which assume evenly-spaced, contiguous months.
- No unexpected nulls in required fields (`region`, `vertical`, `partner_tier`, `nps_score`, `engagement_score`, `days_since_last_contact`). Note: when parsing region codes, make sure "NA" (North America) is read as a literal string, not as a missing-value sentinel — this is a known footgun in naive CSV loading that `src/data_pipeline.py`'s loaders (`load_partner_static`, `load_partner_snapshots`) already guard against (`keep_default_na=False`); any new ingestion path must do the same.
- `mature_mask_values` (the neutral values used to mask the mature GTV-trend block for `new_partner` rows) should be learned fresh from the current cycle's `train` population and NOT reused indefinitely from an old fit — but a single cycle's value shouldn't swing wildly either; a big jump usually means the mature population's typical GTV trend genuinely shifted, worth a manual look before trusting it (see Section 2).

**Range and plausibility (soft checks — flag and continue, but surface for review):**
- `nps_score` within 0–10, `engagement_score` within 0–100, GTV values non-negative.
- `days_since_last_contact` and `avg_resolution_time_days` within plausible bounds (e.g., no partner showing 10 years since last contact).
- No sudden implausible jump in an individual partner's GTV (e.g., 50x month-over-month) without a known cause (large one-off deal, currency correction) — flag for manual review rather than letting it silently distort that partner's trend, volatility, and seasonal-deviation features.

**Cold-start / cohort checks (soft checks, new this round):**
- Cohort cell sizes for `compute_new_partner_ramp_feature` (tier × region × vertical × tenure band) — a cohort shrinking below the fallback threshold silently degrades the ramp-benchmark feature for new partners in that segment. Track the rate at which the fallback (coarser grouping) triggers; a rising rate means the fine-grained cohorts are running thin.
- `scoring_path` counts by segment — confirm the `new_partner` population isn't so small in any given cycle that its features/model behavior become noisy and hard to trust.

**Volume checks (soft checks):**
- Total partner count and total GTV are within a reasonable band of the prior cycle — a large unexplained drop usually means a partial data load, not a real business change.
- The proportion of partners with a full, expected-length (≥27-month) GTV history hasn't dropped sharply — a broken feed can look like a wave of new partners, so track this ratio over time rather than as a one-off number, and cross-check against actual onboarding volume if available.

**Output sanity checks (after the pipeline runs, before publishing the scored list):**
- Overall predicted churn rate for the cycle is in a plausible range relative to history (e.g., not suddenly 2% or 80%).
- Tier sizes match the intended capacity allocation (e.g., if `HIGH_RISK_CAPACITY_PCT` is set to 10%, the High Risk tier should be close to 10% of the book by construction — a meaningful deviation signals a bug in the ranking/tiering code, not a real shift in risk).
- Spot-check a handful of SHAP explanations for plausibility each cycle — if the top contributing features for a sample of flagged partners stop making intuitive sense, that's worth investigating before the list goes out.

These checks are deliberately layered: hard checks protect against handing the team a list built on structurally broken data; soft checks catch subtler issues that still deserve a human look before the cycle's output is trusted.
