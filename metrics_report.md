# Metrics report — Team Bk

All numbers below came from actually running `rank.py` and
`scripts/ablation_study.py` against `candidates.jsonl` on 2026-06-22
(`backend: sklearn-gbr` — see "Calibration honesty" for why LightGBM isn't
the backend that actually ran). Raw output: `metrics_report_data.json`,
`ablation_results.json`. Nothing here is invented; re-run both scripts to
reproduce.

## What these numbers are measuring (read this before the tables)

We do not have access to the hidden grading labels. Every NDCG/MAP/P@10
number below is computed against `src/pseudo_labels.py`'s heuristic tiers —
relevance labels we wrote ourselves, as a deterministic function of the same
feature table the ranker scores from. That means:

- High cross-validated NDCG mostly tells us "the learned model can recover a
  function of the features that resembles the label function" — useful for
  catching a broken pipeline (if CV scores were *low*, something would be
  wrong), but it is not evidence of accuracy against the real hidden ranking.
- The numbers ARE useful for relative comparisons between our own design
  choices (rule-only vs. model-only vs. with/without honeypot detection),
  because the pseudo-label target is held fixed across each comparison — see
  "Ablation" below.

## Cross-validation (5-fold, candidate-level — not query-level)

This is a single-JD ranking problem, so there's no "held-out query" to
generalize to. The fold split is over the 300 stage-2 survivors.

| fold | NDCG@10 | NDCG@50 | MAP    | P@10 |
|------|---------|---------|--------|------|
| 0    | 0.9956  | 0.9979  | 1.0000 | 1.00 |
| 1    | 1.0000  | 1.0000  | 1.0000 | 0.50 |
| 2    | 1.0000  | 1.0000  | 1.0000 | 1.00 |
| 3    | 0.9672  | 0.9931  | 0.8813 | 0.90 |
| 4    | 1.0000  | 1.0000  | 1.0000 | 0.80 |

Fold 1's P@10 = 0.5 is the one number here worth looking at twice rather
than reading the high NDCG and moving on: it means half of that fold's top
10 predicted candidates didn't clear the tier-3 relevance threshold, even
though NDCG@10 for the same fold is 1.0 — NDCG rewards getting the *order*
of relevant items right wherever they land, P@10 cares whether the top 10
specifically clears a hard bar. With only ~60 candidates per fold, a couple
of borderline tier-2/tier-3 candidates landing in the wrong bucket swings
P@10 by 0.1 per candidate. We don't read this as a pipeline problem; we read
it as a reminder that P@10 on small folds is noisy and the average across
folds (0.84) is the more stable number to quote.

## Final run

- 18,745 candidates loaded, 1 skipped (malformed JSON on line 18746).
- Stage 1 (BM25) → 1000 survivors, 8.9s.
- Stage 2 (TF-IDF/SVD dense) → 300 survivors, fit 7.3s + filter 0.2s.
- Stage 3 (feature scoring + honeypot detection) → 300 scored, <0.1s.
- Stage 4 (cross-validate + train + ensemble) → 0.8s.
- Total wall clock: 18.2s. Budget was 5 minutes — most of the slack is
  headroom for a slower machine than our dev sandbox, not waste.
- Honeypots in stage-2 pool: 36 / 300. Honeypots in final top 100: **0**.
- Pseudo-tier distribution in stage-2 pool: tier 0 (honeypot/junk) 36,
  tier 1 (weak) 5, tier 2 (moderate) 215, tier 3 (strong) 34, tier 4
  (excellent) 10.

## Ablation

Each variant is scored against the **same fixed pseudo-label target**
(generated once by the normal full pipeline) so the comparison isolates one
design choice at a time, not a moving target.

| variant                  | NDCG@10 | NDCG@50 | MAP    | P@10 |
|---------------------------|---------|---------|--------|------|
| full ensemble (submitted) | 0.968   | 0.985   | 0.979  | 1.00 |
| rule score only           | 0.818   | 0.893   | 0.646  | 0.60 |
| learned model only        | 0.968   | 0.996   | 0.970  | 1.00 |
| flat skill weighting      | 0.868   | 0.900   | 0.667  | 0.80 |
| no honeypot detection     | 0.000   | 0.192   | 0.273  | 0.00 |

Two results worth calling out specifically:

**Honeypot detection is the single biggest lever in this whole pipeline.**
Turning it off doesn't degrade the ranking gradually — NDCG@10 goes to
*zero*, because all 10 of the top-10-by-rule-score candidates without
honeypot filtering are honeypots (`honeypots_in_top10: 10` in
`ablation_results.json`), and the pseudo-label target treats honeypots as
tier 0. All 36 stage-2 honeypots would reach the top 100 unfiltered, versus
0 with detection on. Every other ablation in this table is a matter of
degree; this one is a cliff.

**Flat skill weighting actually scores slightly *higher* against our own
pseudo-labels than category weighting** (NDCG@10 0.868 flat vs. 0.818
category-weighted rule-only; MAP 0.667 vs. 0.646). We expected category
weighting to win this comparison going in — it's the opposite of what we
expected, and we're reporting it rather than rewriting the ablation until it
agrees with us. The honest explanation, after checking: `assign_tier` only
produces 5 coarse buckets, and 215 of the 300 stage-2 survivors land in the
same tier (tier 2). NDCG against a target with that many ties is sensitive
to *which* same-tier candidates a scoring scheme happens to push slightly
higher, not to whether the scheme reflects the JD's intent — and category
weighting's whole benefit (suppressing JD-buzzword decoys) is a benefit
against decoys that mostly never reach the stage-2 pool to begin with, or
get caught by title-tier gating and honeypot detection regardless of skill
weighting. So this particular ablation doesn't actually demonstrate the
benefit we built category weighting for. The qualitative case for it still
stands — our very first baseline (plain keyword overlap, no category
weights, see `experiments/exp_log.md` §2) put a Mechanical Engineer with 6
stuffed skill tags in the top 10 on a manual spot-check, which category
weighting fixes — but that's a different, qualitative check, not this NDCG
number. We're keeping category weighting in the pipeline because of the
manual spot-check evidence and because it's a more defensible model of "what
should count as a skill match" on its own terms, not because this ablation
table proves it wins.

The `decoy_title_check` (best rule-only rank achieved by any candidate whose
title sits outside the JD's tier_3/4/5 list) came back identical under both
weighting schemes — rank 3, in both cases the same candidate,
`CAND_0008618` ("Computer Vision Engineer"). We checked this by hand rather
than trusting the number at face value (see `experiments/exp_log.md` §5 for
the full walk): this candidate has genuinely well-evidenced NLP/IR/vector-DB
skills (22-58 months duration, 12-46 endorsements, not bait-tier tags), so
skill weighting — flat or category — doesn't move them much either way.
What actually controls their rank is `title_score` (0.03, outside the tiered
title list), and `title_score` isn't touched by the skill-weighting ablation
at all. They rank 3rd under the rule score alone, but drop to 43rd under the
learned model and 39th in the final ensemble — inside our top 100, but well
outside the top 10, which we think is the right outcome for a candidate with
a mismatched title but real adjacent-domain depth, rather than the wrong
outcome for either ablation variant.

## Error analysis: where the rule score and the learned model disagree

The CAND_0008618 case above generalizes into a real, documented limitation
of the rule score (not the ensemble): because the brief's composite formula
gives `title_score` only a 0.35 sub-weight inside the 0.15-weighted
`experience_fit` term (~5% of the total), a near-zero title score cannot by
itself suppress a candidate who also scores well on `semantic` (0.30
weight) and `retrieval_skill_match` (0.20 weight) — those two terms are
exact