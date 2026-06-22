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

**Category-weighted skill scoring beats flat weighting, but the gap is
smaller than we expected going in** (NDCG@10 0.968 ensemble / 0.818 rule-only
vs. 0.868 flat — flat weighting is worse but not catastrophic). The
`decoy_title_check` (best rule-only rank achieved by any candidate whose
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
exactly half the composite and aren't gated by title. The learned reranker
doesn't inherit this limitation, because the pseudo-labels it trains on
gate tier assignment on `title_score` directly (`assign_tier` requires
`title >= 0.45` for tier 3, `title >= 0.75` for tier 4 — see
`src/pseudo_labels.py`). This is the concrete reason the final submission
uses the 50/50 rule+model ensemble rather than the rule score alone: the
rule score is the auditable backbone, but it has a known blind spot the
model corrects for on cases like this one.

## Calibration honesty

- `lightgbm` was not importable in our dev sandbox (network-restricted,
  could not install it reliably during build). Every number in this report
  was produced by the `sklearn.ensemble.GradientBoostingRegressor` fallback
  path in `src/train_ranker.py`, not LightGBM/LambdaMART. The code tries
  LightGBM first and logs which backend actually ran (`backend` field in
  `metrics_report_data.json` — currently `"sklearn-gbr"`). We did not bench
  LightGBM ourselves before submitting.
- All scores in `submission.csv` are an ensemble of an explainable rule
  score and a GBT trained on our own heuristic labels. Treat the absolute
  score values as a ranking signal, not a calibrated probability of
  anything — nothing in this pipeline was calibrated against an external
  outcome (e.g., "this candidate gets hired"), because no such label
  exists in the released data.
