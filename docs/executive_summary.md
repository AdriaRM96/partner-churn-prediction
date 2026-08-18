# Partner Churn Early-Warning: Executive Summary

*Prepared for the partner management team*

> **A note on the data:** the figures in this document come from a synthetic dataset built to reflect realistic partner behavior (growth and decline patterns, seasonal businesses, satisfaction trends, engagement levels), not from a real company or real partners. Think of this as a proof of concept for what the approach could deliver once run on our actual partner data.

## The problem

Right now, we typically find out a partner is at risk when the relationship has already gone cold — GTV has dropped for a quarter or two, nobody on our side has talked to them in months, and by the time it shows up on a scorecard, there isn't much runway left to fix it.

That's expensive. Replacing a partner means starting over: months of onboarding, enablement, and relationship-building, plus the transaction volume we lose in the meantime. If we could reliably spot the partners heading for the exit a full quarter ahead of time, we could get ahead of the problem — a check-in call, a renewed commercial conversation, a training push — instead of finding out after they've already left.

In the dataset used for this analysis, about **1 in 5 partners (20%)** churn in a given quarter. That's a large enough group that we can't treat every partner the same way, but a small enough group that a focused list is genuinely actionable.

## What actually predicts churn

We tested this by looking at everything we'd normally track about a partner — their revenue trend, satisfaction scores, how engaged they are with our programs, how often we're in touch, and whether they're also working with a competitor — and checking which of these actually separated the partners who left from the ones who stayed.

Five things stood out clearly, and they line up with what partner managers would expect from experience:

1. **A declining revenue trend.** Partners whose transaction volume with us has been shrinking over the last few months are far more likely to leave. This was, by a clear margin, the single strongest warning sign.
2. **Low or worsening satisfaction (NPS).** Partners who rate us poorly, or whose satisfaction is trending down, churn at meaningfully higher rates.
3. **Low engagement.** Partners who skip training, certifications, and program events are more likely to disengage entirely.
4. **Going quiet on our side.** The longer it's been since we last had a meaningful conversation with a partner, the more likely they are to leave — "out of sight, out of mind" is a real effect, not just a saying.
5. **A relationship with a competitor.** Partners who are also working with a competing program churn at more than **3x** the rate of partners who aren't (roughly 41% vs. 12% in this analysis).

Two things that *didn't* turn out to matter much, worth calling out because they might be assumed to: which region or industry a partner is in, and whether their business is naturally seasonal (a partner whose sales always dip in a slow month isn't at higher risk — a partner whose sales are trending down independent of the season is).

## The recommended action

**Score the partner book on a regular cadence (we'd suggest monthly) and give partner managers a prioritized, ranked list of the partners most likely to churn next quarter — with the reason each one is flagged attached.**

This isn't meant to replace a partner manager's judgment. It's meant to make sure nobody falls through the cracks simply because their decline wasn't visible yet on a standard report. The list tells the team not just *who* to call, but *why* — a partner flagged because a competitor is circling needs a very different conversation than one flagged for having gone quiet.

In this analysis, when we set the model to prioritize catching as many at-risk partners as possible (rather than being overly cautious about false alarms), it correctly identified roughly **two-thirds of the partners who actually went on to churn**, with about the same hit rate on its flags. In practice, that means: for every 3 partners flagged, roughly 2 were genuinely heading toward churn — a strong enough signal to act on, especially since the cost of an unnecessary check-in call is low compared to the cost of losing a partner we never saw coming.

## Expected impact

Think of this less as "the model prevents churn" and more as "the model buys the team time and focus":

- **Earlier warning** — flagging risk based on a trend, not waiting for a partner to already be gone.
- **Better-targeted outreach** — the team spends its limited retention time on the partners most likely to need it, with a reason attached, instead of spreading effort evenly or reactively across the whole book.
- **A defensible way to prioritize** — when there are more at-risk partners than the team can personally call in a given month, the ranked list and risk tiers give a clear, explainable basis for deciding who gets a call this week versus a lighter-touch nudge.

The direct dollar impact depends on how many of the flagged partners we're actually able to save through outreach — that's something we'd want to measure once this runs against real partner data, not assume up front.

## Next steps

1. **Pilot on real data.** Run this same approach against our actual partner records for one full quarter, without changing how the team operates yet — just to see how well the flagged list matches what partner managers already sense intuitively, and to catch any data gaps before relying on it.
2. **Build the monthly outreach list into the team's workflow.** Once validated, turn the ranked list into a recurring deliverable partner managers can act on — including the "reason flagged" for each partner, not just a risk score.
3. **Track what happens to flagged partners.** For every partner we reach out to because they were flagged, record whether they were retained or still churned. That feedback is what will let us prove the impact in step 2, and keep the model accurate as partner and market behavior shifts over time.
