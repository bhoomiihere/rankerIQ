# Redrob Candidate Ranking — Team Bk

Ranks 18,745 candidates against Redrob's Senior AI Engineer (Founding Team)
JD and outputs `submission.csv` per `docs/submission_metadata_template.yaml`'s
spec. Built for the Intelligent Candidate Discovery & Ranking Challenge.

Team: Bk (Meghna Kaushik, Bhoomi Kaushik)

## Quick start

```bash
pip install -r requirements.txt
python rank.py --candidates candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
```

Runs in ~15-20 seconds on 2 CPU cores / 4GB RAM, no network, no GPU. See
`metrics_report.md` for the actual run we used to write this README.

## What's in this repo

```
rank.py                          one-command entry point
src/
  ingest.py                      candidates.jsonl -> list[dict], skips malformed lines
  semantic.py                    TF-IDF + SVD "dense" similarity (no torch, see below)
  retrieval.py                   stage1 BM25 / stage2 dense filtering
  honeypot.py                    rule-based fake/inflated-profile detector
  features.py                    skill/title/company/behavior/trust/... scoring
  pseudo_labels.py                heuristic relevance tiers (weak supervision)
  train_ranker.py                LightGBM LambdaMART (sklearn GBR fallback)
  metrics.py                      NDCG/MAP/P@k
  reasoning.py                    template-based, evidence-grounded reasoning strings
challenge_analysis.md            reverse-engineering of the JD's hidden scoring logic
metrics_report.md                 NDCG/MAP/P@10 against our own pseudo-labels, ablation
docs/                              job description, candidate schema, metadata template
experiments/exp_log.md             dated log of what we tried, in what order, and why
weighted_job_representation.json   structured JD (required/preferred/negative signals)
submission_metadata.yaml           filled-in submission metadata (TODOs marked)
```

## Why this architecture

**Why a 4-stage funnel instead of scoring everyone with the full feature
pipeline directly.** At 18,745 candidates we could feature-score everyone
inside the time budget — we checked, it's under 2 seconds. We built the
funnel anyway because the JD explicitly frames this as a 100K+ candidate
problem, and an architecture that only works at the size of this particular
dataset isn't really solving the stated problem. BM25 (stage 1) is cheap and
catches "does this profile use JD-relevant vocabulary at all" before we pay
for anything heavier; TF-IDF/SVD (stage 2) catches semantic matches BM25's
exact-token matching misses; the full feature pipeline (stage 3, honeypot
detection + skill/title/behavior scoring) only runs on the 300 candidates
that survive both filters.

**Why TF-IDF + SVD instead of sentence-transformer embeddings.** We
considered `sentence-transformers` for stage 2 and rejected it for three
reasons, in order of how much they actually mattered: (1) the compute
constraints explicitly require CPU-only, no network, reproducible in an
unfamiliar sandbox at Stage 3 — a `torch` + transformer-weights download is
exactly the kind of dependency that fails quietly on someone else's machine;
(2) our own dev sandbox's network access to PyPI was unreliable during
build, which made "can we even get the dependency installed" a real risk
we'd already hit, not a theoretical one; (3) TF-IDF/SVD fit+transform over
the full corpus takes about 12 seconds total, comfortably inside the 5-minute
budget, and gave us interpretable output (each SVD component can be traced
back to a term loading, useful for debugging why a candidate scored the way
they did). The tradeoff: TF-IDF/SVD can't capture synonymy the way a
trained embedding model can ("recommendation engine" vs "recommender
system" get partial token overlap, not the near-identical vectors a real
embedding model would give them). We accept this because the dataset's
actual trap (see `challenge_analysis.md`) is keyword stuffing on irrelevant
titles, not synonym variation — a model that's slightly worse at synonyms
but immune to network/GPU failure modes was the better bet for this specific
brief.

**Why rule-based features plus a learned reranker, not one or the other.**
A pure rule-based score is fully explainable (every number traces to a named
field) but its weights are our guesses, frozen at whatever we set them to.
A pure learned model needs labels we don't have, and training one on labels
we invented ourselves to "predict itself" doesn't obviously buy us anything
over just using the rule score directly. We use both: the rule score is the
auditable backbone, and the learned reranker (GBT over the same feature
table) is trained on heuristic pseudo-labels purely to check whether the
features carry signal a model can recover, and to give us an ensemble
component that can pick up nonlinear interactions the fixed-weight formula
can't (e.g. "high skill match AND low title score" mattering differently
than either alone). Final score is a 50/50 blend. See `metrics_report.md`
for the honest caveats on what the cross-validated numbers do and don't
prove.

**Why honeypot detection is a separate, multiplicative penalty instead of
a feature folded into the weighted sum.** We measured this directly: the
single highest-scoring candidate by raw semantic similarity to the JD
(`CAND_0005538`, 0.897 cosine) independently trips our honeypot detector
("expert"-tier skill proficiency, a level no other candidate in the dataset
uses). If honeypot signals were just one more additively-weighted feature,
a honeypot with strong text elsewhere could still out-rank a middling real
candidate. Treating it as a multiplier on the whole composite score (not
just one sub-score) reflects what it actually means: "wrong type of
candidate," not "slightly weaker candidate."

## Assumptions

- The released `candidates.jsonl` (18,745 usable records) is representative
  of the kind of pool this pipeline would run against, even though the JD
  references a 100K+ pool. We did not invent synthetic candidates to pad
  this number; the funnel architecture is sized for the larger number, the
  test run is on the number we actually have.
- `weighted_job_representation.json`'s skill-tier and title-tier judgments
  are our own reading of the JD, verified against frequency counts in the
  data (see `challenge_analysis.md`) but not against any external ground
  truth, because none exists for us to check against.
- The composite weight formula in the brief (`0.30 semantic + 0.20
  retrieval + 0.15 experience + ...`) is taken literally as the intended
  rule-score weights, even though the actual hidden grading formula
  (`0.50 NDCG@10 + 0.30 NDCG@50 + 0.15 MAP + 0.05 P@10`) scores rankings,
  not feature weights — there's no way to causally connect the two without
  the hidden labels, so we treat the feature-weight formula as a design
  prior, not a guarantee of leaderboard performance.

## Limitations

- **No real ground truth.** Every NDCG/MAP/P@10 number in this repo is
  computed against pseudo-labels we wrote ourselves (`src/pseudo_labels.py`).
  These validate that our feature pipeline and learned model are internally
  consistent — they do not estimate hidden-leaderboard accuracy. We say this
  again in `metrics_report.md` because it's the single easiest thing to
  misread in this whole submission.
- **Single-query ranking.** This challenge has exactly one JD. LambdaMART
  and our cross-validation setup are built for a learning-to-rank problem
  that normally has many queries; here, "generalization" only means
  "generalizes across held-out candidates for this JD," not across JDs. If
  Redrob reused this pipeline for a different role, the learned reranker
  would need retraining from scratch — there's nothing query-general about
  it currently.
- **TF-IDF/SVD doesn't understand synonyms or paraphrase the way a trained
  embedding model would.** Discussed above; accepted tradeoff for
  reproducibility, not free.
- **Honeypot thresholds are tuned against this specific dataset's counts**
  (e.g., "expert" proficiency level, employer founding-year mismatches for
  the 4 employers we could verify). A differently-constructed honeypot
  set in a future release would likely need re-derived thresholds.
- **lightgbm wasn't actually available during our own development runs**
  (network-restricted dev sandbox); the numbers in `metrics_report.md` were
  produced by the sklearn `GradientBoostingRegressor` fallback path. The
  code tries LightGBM first and logs which backend ran — see
  `src/train_ranker.py`. We did not bench LightGBM ourselves before
  submitting; the fallback path is what actually got exercised.

## Unsuccessful experiments

See `experiments/exp_log.md` for the full, dated log. Short version of what
didn't make it into the final pipeline:

- **Pure skill-keyword-overlap scoring (no category weighting).** First
  thing we tried, as a baseline to measure against. It ranks several
  honeypots and decoy profiles (e.g. a Mechanical Engineer with 6 "required"
  skill tags) inside the top 50 of a small manual test set. Replaced by the
  category-and-claim-strength weighted version in `src/features.py`.
- **Folding honeypot probability into the weighted sum as one more
  feature.** Didn't reliably suppress honeypots with strong supporting text
  elsewhere. Replaced with the multiplicative penalty described above.
- **Fitting the SVD basis only on the stage-1 survivors (1000 candidates)
  instead of the full corpus.** Made the embedding space dependent on which
  1000 candidates BM25 happened to keep, which made similarity scores
  inconsistent across runs with different `--stage1-k` values. Switched to
  fitting on the full corpus once, transforming only the stage-1 survivors.

## Future improvements

- Swap in a real sentence-transformer model for stage 2 if/when network
  access in the grading sandbox is confirmed reliable — would likely help
  with synonym/paraphrase cases TF-IDF misses.
- Multi-JD pseudo-label generalization: if this pipeline needs to run
  against a second JD, the heuristic tier-assignment logic in
  `pseudo_labels.py` should move from hardcoded thresholds to something
  parameterized per-JD.
- A real held-out eval would need either organizer-released ground truth or
  a small hand-labeled sample reviewed by someone who isn't us — our own
  pseudo-labels can't substitute for that no matter how carefully tuned.

## Deploying the sandbox link

Not yet deployed — `submission_metadata.yaml` has a placeholder. Easiest
path given the zero-network-at-inference constraint: a HuggingFace Space
with a Gradio file-upload front end that calls `rank.py` on a small sample
candidates file, or a Streamlit Cloud app doing the same. Either works
within the free tier since the pipeline needs no GPU.
