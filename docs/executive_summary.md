# Partner Churn Early-Warning: Executive Summary

*Prepared for the partner management team*

> **A note on the data:** the figures in this document come from a synthetic dataset built to reflect realistic partner behavior (growth and decline patterns, seasonal businesses, satisfaction trends, engagement levels), not from a real company or real partners. Think of this as a proof of concept for what the approach could deliver once run on our actual partner data.

## The problem

Right now, we typically find out a partner is at risk when the relationship has already gone cold — GTV has dropped for a quarter or two, nobody on our side has talked to them in months, and by the time it shows up on a scorecard, there isn't much runway left to fix it.

That's expensive. Replacing a partner means starting over: months of onboarding, enablement, and relationship-building, plus the transaction volume we lose in the meantime. If we could reliably spot the partners heading for the exit a full quarter ahead of time, we could get ahead of the problem — a check-in call, a renewed commercial conversation, a training push — instead of finding out after they've already left.

In the dataset used for this analysis, about **1 in 5 partners (~20%)** churn in a given quarter. That's a large enough group that we can't treat every partner the same way, but a small enough group that a focused list is genuinely actionable.

## What actually predicts churn

We tested this by looking at everything we'd normally track about a partner — their revenue trend, satisfaction scores, how engaged they are with our programs, how often we're in touch, and whether they're also working with a competitor — and checking which of these actually separated the partners who left from the ones who stayed.

Four things stood out clearly, and they line up with what partner managers would expect from experience:

1. **Low or worsening satisfaction (NPS).** Partners who rate us poorly, or whose satisfaction is trending down, churn at meaningfully higher rates — the single strongest and most consistent signal in this analysis.
2. **Low engagement.** Partners who skip training, certifications, and program events are more likely to disengage entirely.
3. **Going quiet on our side.** The longer it's been since we last had a meaningful conversation with a partner, the more likely they are to leave — "out of sight, out of mind" is a real effect, not just a saying.
4. **A relationship with a competitor.** Partners who are also working with a competing program churn at more than **3x** the rate of partners who aren't (roughly 28% vs. 8% in this analysis).

**A declining revenue trend still matters, but with an important correction this round.** We found and fixed a real bug in how we were reading GTV trend: for a partner with a naturally seasonal business (think retail-adjacent partners peaking around the holidays), the old approach was comparing their most recent quarter to the prior quarter — which reads a completely normal post-peak dip as if the partner were declining. We now compare each partner's recent GTV to the *same period last year*, and separately check whether they're underperforming their own historical seasonal pattern. In our test, 23 partners who would have been wrongly flagged as declining under the old method turned out to be right on their normal seasonal track — a meaningful number of false alarms this fix removes.

Two things that still *don't* matter much, worth calling out because they might be assumed to: which region or industry a partner is in, and whether their business is naturally seasonal in the first place (having a seasonal pattern isn't a risk factor — falling behind your *own* normal pattern is).

**One more addition this round: new partners get their own read.** A partner with only a few months of history can't have a reliable multi-year trend — forcing one onto them was giving us a noisy, unreliable signal. Partners under about 9 months tenure are now scored on a different, simpler signal: how their early growth compares to similar partners at the same early stage, rather than a trend estimate that isn't reliable yet.

## The recommended action

**Score the partner book on a regular cadence (we'd suggest monthly) and give partner managers a prioritized, capacity-sized list of the partners most likely to churn next quarter — with the reason each one is flagged attached.**

This isn't meant to replace a partner manager's judgment. It's meant to make sure nobody falls through the cracks simply because their decline wasn't visible yet on a standard report. The list tells the team not just *who* to call, but *why* — a partner flagged because a competitor is circling needs a very different conversation than one flagged for having gone quiet, and we can now show that reasoning for each individual partner, not just as a general pattern across the whole book.

**How well does it actually work?** We tested this the way it will really be used: trained on older data, evaluated on a more recent, unseen quarter — not a random shuffle of one point in time, which would have overstated how well the model performs once deployed. Under that more honest test, the model correctly identified a bit over half of the partners who went on to churn (56%), and about 3 in 5 of its flags turned out to be right (60%). That's a real, useful signal — better than not having a system at all — but it's also an honest number we'd want to keep improving on with real data, real feedback loops, and a growing history of outcomes to learn from.

**We also rebuilt how partners get sorted into "call this week" vs. "keep an eye on it."** The old approach used a flat probability cutoff. We found that doesn't actually work well here: when we're honest about how much more a missed large partner costs versus an unnecessary call, the math says "when in doubt, reach out" almost across the board — which isn't useful if the team can't act on that many partners in one cycle. So instead, we rank every partner by *expected cost of doing nothing* (how likely they are to churn, weighted by how much business is at stake) and size the "urgent" and "worth a nudge" tiers to what the team can actually handle in a cycle — illustratively, the top 10% get a call and the next 20% get a lighter-touch check-in, though those percentages are something the team should set based on real capacity, not something we derived from the data.

## Expected impact

Think of this less as "the model prevents churn" and more as "the model buys the team time, focus, and a clear reason for every call":

- **Earlier warning** — flagging risk based on a trend, not waiting for a partner to already be gone, and now doing so without the false alarms a normal seasonal dip used to cause.
- **Better-targeted outreach** — the team spends its limited retention time on the partners most likely to need it, with a specific, per-partner reason attached (not just a general pattern), instead of spreading effort evenly or reactively across the whole book.
- **A defensible, capacity-matched way to prioritize** — the ranked list is sized to what the team can actually act on in a cycle, not an arbitrary cutoff that could flag too many or too few partners depending on how the numbers happen to fall.
- **A sense of urgency, not just risk.** Two partners can look equally "at risk" on a scorecard while one is drifting away over a year and the other is close to walking out the door. We added a complementary view that estimates *how soon*, not just *whether* — useful for deciding who gets called first when the list is longer than the team can work through immediately.

The direct dollar impact depends on how many of the flagged partners we're actually able to save through outreach — that's something we'd want to measure once this runs against real partner data, not assume up front.

## Next steps

1. **Pilot on real data.** Run this same approach against our actual partner records for one full quarter, without changing how the team operates yet — just to see how well the flagged list matches what partner managers already sense intuitively, and to catch any data gaps before relying on it.
2. **Build the monthly outreach list into the team's workflow**, sized to real outreach capacity (not the illustrative percentages used here) — including the "reason flagged" for each partner, which segment they were scored in (established partner vs. new), and an urgency read for sequencing calls when the list is long.
3. **Track what happens to flagged partners.** For every partner we reach out to because they were flagged, record whether they were retained or still churned. That feedback is what will let us prove the impact in step 2, keep the model accurate as partner and market behavior shifts over time, and improve on the ~56%/60% numbers above as we accumulate real outcomes to learn from.
