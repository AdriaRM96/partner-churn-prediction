# Production Considerations

This document covers how the partner churn pipeline (`src/data_pipeline.py`) and model would actually run in production: retraining cadence, drift monitoring, unscheduled-retrain triggers, and pre-scoring data quality checks. It assumes the deployment target is a monthly partner data refresh, matching how the underlying features (GTV trend, seasonality, engagement) are defined.

## 1. Retraining cadence

**Retrain monthly, aligned with the GTV data refresh.**

The core features (`gtv_trend_3m`, `gtv_volatility`, `has_seasonality`, `gtv_vs_peers`, `gtv_vs_market_growth`) are all derived from the trailing GTV history, and the target (`churned_next_quarter`) resolves on a quarterly lag. A monthly cadence:

- keeps the model current with the newest GTV trend and engagement data as soon as it's available
- gives three retrain cycles per quarter to absorb newly-resolved churn outcomes as they land, rather than waiting a full quarter between updates
- matches the cadence partner managers would actually receive a scored list on (see [docs/executive_summary.md](executive_summary.md))

Two things to keep fixed across retrains unless deliberately revisited:
- **The 27-month (or longer) history window for `compute_seasonality`.** Seasonality detection needs more than two full annual cycles to distinguish a genuine recurring pattern from a one-off bump (see the validation in the main notebook, Section 2.5–2.7). Shortening this window would silently degrade the seasonality flag's reliability.
- **The 3-month trend window for `gtv_trend_3m`.** Changing this changes the meaning of the feature and would require re-validating it against churn outcomes, not just re-fitting the model.

**Full re-training** (refit all four candidate models, re-run model selection) should happen every retrain cycle — it's cheap relative to the cost of scoring against a stale model, and there's no reason to let feature/label drift accumulate between fits.

## 2. Monitoring for model and data drift

Monitor two distinct things: whether the *data* going into the pipeline is changing shape, and whether the *model* is still performing.

### Data drift (input distributions)

Track, per scoring cycle, the distribution of each model feature against a recent training-period baseline:
- **Population-level shifts**: mean/median and standard deviation of `gtv_trend_3m`, `engagement_score`, `nps_score`, `days_since_last_contact` month over month. A sudden shift (e.g., average `days_since_last_contact` jumping because a CRM sync broke) is a data problem, not a market signal, and should be caught before it reaches the model.
- **Categorical mix**: share of partners by `region`, `vertical`, `partner_tier`. A sharp change usually means an upstream extract changed (e.g., a new region code appeared, or a segment was dropped from the feed) rather than that the partner base actually shifted that fast.
- **`has_seasonality` prevalence**: this should be fairly stable month to month (it's derived from 27 months of history, so any single new month barely moves it). A large swing is a strong signal something upstream broke, not that partner seasonality genuinely changed.
- **Missing/null rates** per column — a rising null rate on any input feature, especially engagement or NPS fields, points to instrumentation gaps (see Section 4).

### Model/concept drift (is the model still right?)

- **Recall and precision on newly-resolved outcomes.** Each month, some partners flagged in prior cycles will have resolved (churned or retained). Track recall and precision on that resolved cohort over time. A sustained drop in recall means the model is missing more at-risk partners than it used to — the single most important metric to watch, given that a missed churn is the costliest failure mode here.
- **Calibration of the risk bands.** Confirm the actual churn rate within each risk band (Low/Moderate/High/Critical, as used in the notebook's risk-scoring section) still increases monotonically and roughly matches the band's implied probability. If "Critical" partners start churning at, say, 40% instead of 70%+, the score is no longer trustworthy for prioritization even if the model hasn't technically "broken."
- **Feature importance stability.** Re-check monthly whether the same handful of features (GTV trend, NPS, engagement, contact gap, competitor relationship) remain the top drivers. A reshuffle is worth investigating — it can mean a genuine shift in what drives churn, or a sign that a feature has quietly become unreliable (see drift above).

## 3. What would trigger an out-of-cycle retrain

Retrain immediately, outside the normal monthly schedule, if any of the following happen:

1. **Recall on resolved outcomes drops below an agreed floor** (e.g., falls more than 10 percentage points below its trailing 3-month average). This is the clearest sign the model is no longer doing its job.
2. **A material change to the partner program itself** — a new tier structure, a pricing change, a shift in how NPS or engagement is measured or collected, or a new competitor entering the market. Any of these can change *what predicts churn*, not just the current data snapshot, and waiting for the next scheduled retrain risks scoring against stale relationships.
3. **A data pipeline change upstream** — a new CRM, a change to how `days_since_last_contact` or `support_tickets` is logged, a new GTV reporting system. Even if the retrain "succeeds," the model may be learning from redefined features rather than the same signal it was trained on; this should trigger both a retrain and a fresh validation pass, not just a retrain.
4. **A data quality check fails hard** (see Section 4) in a way that can't be resolved before the next scheduled cycle — better to skip a scoring cycle and investigate than to hand partner managers a list built on broken inputs.
5. **A sustained, unexplained shift in the overall churn rate** (e.g., churn rate doubles or halves month over month with no known business cause) — this could be a real market event worth understanding before the model quietly "adapts" to it, or a sign of upstream data corruption.

## 4. Data quality checks before each scoring cycle

Run these against the raw inputs (`partner_static` and `gtv_history` equivalents) before calling `build_model_ready_dataset` / `load_and_build_dataset`, and halt the cycle (falling back to the last known-good scored list) if any hard check fails.

**Schema and completeness (hard checks — halt on failure):**
- Every expected column is present with the expected type.
- `partner_id` is unique in the static table and every `partner_id` in the GTV history has a corresponding static record (and vice versa).
- GTV history has a contiguous, complete `month_index` sequence per partner with no gaps — a gap silently corrupts `gtv_trend_3m`, `gtv_volatility`, and the seasonality autocorrelation, all of which assume evenly-spaced, contiguous months.
- No unexpected nulls in required fields (`region`, `vertical`, `partner_tier`, `nps_score`, `engagement_score`, `days_since_last_contact`). Note: when parsing region codes, make sure "NA" (North America) is read as a literal string, not as a missing-value sentinel — this is a known footgun in naive CSV loading that `src/data_pipeline.py`'s loaders already guard against (`keep_default_na=False`); any new ingestion path must do the same.

**Range and plausibility (soft checks — flag and continue, but surface for review):**
- `nps_score` within 0–10, `engagement_score` within 0–100, GTV values non-negative.
- `days_since_last_contact` and `avg_resolution_time_days` within plausible bounds (e.g., no partner showing 10 years since last contact).
- No sudden implausible jump in an individual partner's GTV (e.g., 50x month-over-month) without a known cause (large one-off deal, currency correction) — flag for manual review rather than letting it silently distort that partner's trend and volatility features.

**Volume checks (soft checks):**
- Total partner count and total GTV are within a reasonable band of the prior cycle — a large unexplained drop usually means a partial data load, not a real business change.
- The proportion of partners with a full, expected-length GTV history hasn't dropped sharply (new partners naturally have shorter histories; a broken feed can look similar, so track this ratio over time rather than as a one-off number).

**Output sanity checks (after the pipeline runs, before publishing the scored list):**
- Overall predicted churn rate for the cycle is in a plausible range relative to history (e.g., not suddenly 2% or 80%).
- Risk band sizes are reasonable relative to team capacity — if "Critical" balloons to a third of the book, that's worth a sanity check before it lands in partner managers' inboxes.

These checks are deliberately layered: hard checks protect against handing the team a list built on structurally broken data; soft checks catch subtler issues that still deserve a human look before the cycle's output is trusted.
