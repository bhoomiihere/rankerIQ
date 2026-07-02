# Experiment log — Team Bk

Dated by when each step actually happened in this build (2026-06-22). Kept in
build order, not cleaned up after the fact, so failed attempts stay visible.

## 1. Honeypot rules — counted before scoring

Before writing `src/honeypot.py`, ran three standalone counting queries over
`candidates.jsonl` instead of guessing thresholds:

- `proficiency == "expert"` anywhere in a skill list → 38 candidates, and
  *zero* legitimate profiles use this level (max for everyone else is
  "advanced"). Strongest single tell we found.
- career_history `start_date` earlier than a verifiable real-world company
  founding year (checked CRED/Razorpay/Swiggy/Zomato only — the four
  employers in the dataset we could actually verify a founding year for)
  → 33 candidates.
  duration_months mismatched against (end_date − start_date) by >3 months
  → 7 candidates.

Sum: 78, against the brief's stated "~80 honeypots." Close enough that we
built the detector as a blend of these three rules rather than chasing the
last couple of candidates with a fourth rule — diminishing returns, and
risk of overfitting the detector to this exact dataset's count.

## 2. Baseline: pure skill-keyword overlap (rejected)

First scoring attempt, before any category weighting existed: score =
fraction of JD skills present in candidate's skill list, full stop. On a
20-candidate manual spot-check this put a Mechanical Engineer with 6
"required" skill tags (Pinecone, RAG, FAISS, Embeddings...) inside the top
10 — exactly the anti-keyword trap `challenge_analysis.md` documents. This
is what motivated `SKILL_CATEGORY_WEIGHT` in `src/features.py` (production
ML / IR skills weighted far above LLM-ops buzzwords) and title-tier gating.
Replaced, not patched — the overlap-only version doesn't survive as a
fallback path anywhere in the final code.

## 3. Honeypot probability as an additive feature (rejected)

Second attempt at honeypot handling: added `honeypot_probability` as one
more signal in the weighted sum, with a negative coefficient. Result on the
top 20 by this composite: `CAND_0005538` ("Senior AI Engineer," 0.897
cosine similarity to the JD, *also* the strongest honeypot trip in the
dataset — "expert" proficiency on three skills) still placed 4th. A strong
honeypot with strong supporting text elsewhere just doesn't get pulled down
enough by one more additive term among a dozen. Switched to a multiplicative
penalty (`honeypot_penalty = max(0, 1 - 1.6*probability)`) applied to the
whole rule score — confirmed in the current pipeline this keeps honeypot
count in the top 100 at 0 (see `metrics_report.md`).

## 4. SVD basis fit on stage-1 survivors only (rejected)

To save compute, first tried fitting the TF-IDF/SVD basis only on the 1000
BM25 survivors instead of the full 18,745-candidate corpus, then transforming
the same 1000. Ran `rank.py` twice with `--stage1-k 1000` and `--stage1-k
1500` and compared `semantic_sim` for the same candidate_id across runs —
the values moved by more than rounding error, because the SVD basis itself
changed depending on which candidates survived stage 1. That's a
reproducibility problem (the same candidate gets a different score depending
on a tuning knob that's supposed to be unrelated to them), so we now fit
once on the full corpus + query, and only pay the transform cost on
survivors. Costs ~7s more per run; worth it for stable output.

## 5. CV-without-NLP/IR negative rule — presence vs. depth (hardened, not reverted)

`negative_signal_multiplier` penalizes "Computer Vision Engineer"-titled
candidates whose skill list has CV/speech tags but no NLP/IR tags, because
the JD explicitly deprioritizes pure computer-vision backgrounds. Original
check was presence-only: `skill_names & nlp_ir` non-empty was enough to
skip the penalty.

Ran `scripts/ablation_study.py`'s `decoy_title_check` (best rule-only rank
achieved by any candidate whose title falls outside tier_3/4/5) as a sanity
check and got `CAND_0008618` at rank 3 — a "Computer Vision Engineer" with
6 matched required skills including `Embeddings` and `Semantic Search`.
Worth checking by hand whether this is the rule failing or a genuinely
strong candidate, since "decoy reaches rank 3" sounded like exactly the
trap we'd built the category weighting to avoid.

Pulled the raw skill records: `NLP` (advanced, 26mo, 43 endorsements),
`Embeddings` (advanced, 22mo, 34 endorsements), `OpenSearch` (advanced,
58mo, 12 endorsements), `Semantic Search` (intermediate, 36mo, 2
endorsements). These aren't bait-tier tags with zero depth — they're the
same duration/endorsement profile as our strongest "real" candidates. The
presence check was right for this specific candidate, but only by luck: it
would have been equally fooled by a decoy who listed `Embeddings` with
1-month duration and 0 endorsements, because presence alone doesn't
distinguish the two cases. Hardened the rule to require `claim_strength
>= 0.4` on at least one nlp_ir skill (same `claim_strength` function
already used in `skill_score`) instead of bare presence. Re-ran: this
candidate's negative-signal status is unchanged (correctly — they have real
depth), but a synthetic test case with the same tags at 1-month/0-endorsement
depth now correctly trips the penalty. No regression on the live dataset;
the rule is strictly more correct than before.

Separately, this surfaced something more interesting than a rule bug: this
candidate ranks 3rd under the **rule-only** score but drops to **43rd**
under the **model-only** score and **39th** under the final ensemble (full
ranking pipeline, `--top-k 100` run on 2026-06-22). The rule formula's exact
weights (`0.30 semantic + 0.20 retrieval + 0.15 experience_fit + ...`, taken
literally from the brief — see README "Assumptions") only let `title_score`
influence the result through `experience_fit`'s 0.35 sub-weight inside a
0.15-weighted term, an effective ~5% of the total composite. A near-zero
title score (0.03 for "Computer Vision Engineer," outside the JD's tiered
titles) can't fully suppress a candidate who also scores well on semantic
(0.43) and retrieval_skill_match — those two terms alone are 50% of the
composite and aren't gated by title at all. The learned reranker doesn't
have this limitation because the pseudo-labels it's trained on (
`src/pseudo_labels.py::assign_tier`) gate tier assignment on `title_score`
directly (tier 3 requires `title >= 0.45`, tier 4 requires `title >= 0.75`),
so the model learns the title/skill interaction the fixed-weight rule score
can't express on its own. This is the strongest argument we have for why
the ensemble (not the rule score alone) is the right thing to submit —
documented in `metrics_report.md`.

This also explains why `decoy_title_check` showed the *same* best rank (3)
under both category-weighted and flat skill weighting: the lever that
actually controls this candidate's rank is `title_score`, not skill
weighting — flat-vs-category weighting changes `retrieval_skill_match`,
which this candidate already scores reasonably on either way. Skill-weight
ablation and title-gating are answering two different questions; we'd
mentally conflated them going in.

## 6. What we didn't try

- A learned title-aware interaction term (e.g., multiplying skill_score by
  title_score before the weighted sum, rather than keeping them as separate
  additive terms) would likely fix the §5 issue at the rule-score level
  directly. Didn't implement because it means deviating from the brief's
  literal weight formula, and we wanted the rule score to stay an honest,
  auditable implementation of that formula rather than a formula-plus-an-
  unstated-correction. Logged as a future improvement instead.
- Real sentence-transformer embeddings for stage 2, blocked on the
  CPU-only / no-network constraint — see README "Future improvements."

## 7. Self-judging reasoning quality against the spec's own Stage-4 rubric (hardened)

`submission_spec.docx` Table 2 (Stage 4, manual review) lists six checks for
the `reasoning` column: specific facts, JD connection, honest concerns,
no hallucination, variation across the sampled rows, and rank consistency
("a rank-95 candidate with glowing reasoning ... indicates the reasoning was
generated independently of the ranking"). Sampled 10 rows from
`submission.csv` (ranks 1,2,3,10,25,50,51,75,90,100) and checked each
against all six by hand instead of assuming `src/reasoning.py` already
satisfied them.

Five of six held up. Rank consistency and honest concerns did not: ranks 50,
51, 75, 90, and 100 all closed with the identical "No material concerns
flagged by our rule set." despite falling well down the top 100 -- exactly
the failure mode the spec calls out, and a sign the prose wasn't actually
reflecting why a candidate was ranked where they were. Rank 100
(`CAND_0011547`, 1 matched required skill, 1 matched preferred skill) read
no differently in tone from rank 1, even though the spec's own example
submission CSV frames a rank-100 filler candidate explicitly ("Adjacent
skills only -- likely below cutoff but included as final filler...").

Pulled the actual numbers behind this before changing anything: checked
`experience_band_fit` against the real job band (5-9 years, soft curve, not
[6,8] as we'd assumed from the JD prose) -- rank 1's candidate at 4.4 years
scores 0.88 fit, genuinely not a concern, so the experience-band threshold
wasn't the gap. The real gap was skill-overlap depth: 20/100 final
candidates have <=1 matched required skill *and* <=1 matched preferred
skill, and none of them had that surfaced in their reasoning text.

Added two checks to `_concern_clause` in `src/reasoning.py`: experience-band
threshold tightened from `< 0.5` to `< 0.8` (still didn't fire for rank 1,
correctly), and a new thin-skill-overlap check (`req_n <= 1 and pref_n <=
1`) that now fires for exactly those 20 candidates. Deliberately did not
lower the bar to `<= 2` (47/100 candidates) -- that would dilute the "no
concerns" signal for genuinely reasonable matches just to look thorough.
Regenerated `submission.csv`; re-ran `validate_submission.py` (still valid,
100 rows, ranks unique, scores non-increasing). Scores and ranks are
unchanged -- this only changes the `reasoning` text, not the ranking
pipeline, so `metrics_report.md`'s numbers are unaffected.

## 8. Caught a real corrupted file before final upload (src/features.py)

Running `rank.py` fresh on a clean machine (Windows, Python 3.14, after
fixing the numpy wheel pin -- see requirements.txt) hit a `SyntaxError:
unterminated triple-quoted string literal` in `rule_based_composite`. The
function's docstring and entire body were missing -- cut off mid-sentence
at "we apply a steep penalty that g". This had been silently broken in the
repo (same truncation signature seen earlier in this build on this sandbox's
mount, see the `reasoning.py` and `build_ppt.py` incidents) without anyone
re-running the full pipeline against the affected copy after it happened, so
`submission.csv` in the repo was actually stale relative to the broken
source -- it still reflected an earlier, intact version of the function.

Reconstructed the function from what's independently documented elsewhere:
`composite_weights` in `weighted_job_representation.json` (semantic 0.30,
retrieval_skill_match 0.20, experience_fit 0.15, behavior 0.10,
evaluation_fit 0.10, trust 0.10, availability 0.05 -- sums to 1.0) for the
weighted sum, and the multiplicative honeypot penalty formula recorded in
this log's entry #3 (`max(0, 1 - 1.6*probability)`), plus `negative_
multiplier` from `negative_signal_multiplier`. Re-ran `rank.py` end to end
and diffed the new `submission.csv` against the stale-but-good one still in
the repo: all 100 candidate_ids land on the exact same rank, confirming the
reconstruction matches what was actually running before the corruption.
Re-validated with `validate_submission.py`. Flagging this here rather than
quietly re-committing, since "the file was broken and we didn't notice"
is exactly the kind of thing that should be visible in the log, not erased.

## 9. lightgbm 4.3.0 imports fine but crashes at fit time under numpy>=2.0

Second issue found running the real one-command repro on a clean Windows
machine (after fixing #8): `rank.py` got past feature scoring into stage 4
and crashed inside `lgb.LGBMRanker.fit()` with `ValueError: Unable to avoid
copy while creating an array as requested`. Reproduced the exact same crash
in our own sandbox by installing `lightgbm==4.3.0` alongside `numpy==2.2.6`
(this sandbox didn't have lightgbm installed before, so we'd never actually
exercised the "lightgbm present" code path end-to-end until now).

Root cause: lightgbm's internal `_list_to_1d_numpy` calls `np.array(data,
dtype=dtype, copy=False)` to build the LambdaMART `group` array. NumPy 2.0
changed `copy=False` from "copy only if a copy is needed" to "never copy,
raise if one is required" -- a documented breaking change, not a bug on
our end, but lightgbm 4.3.0's own code wasn't updated for it on this
particular array-construction path.

The existing fallback in `src/train_ranker.py` only caught `ImportError` at
import time -- it assumed "lightgbm available" and "lightgbm works" were
the same thing, which this case shows isn't true. Widened `train_model` to
wrap the actual `model.fit()` call in a try/except and fall back to
sklearn's `GradientBoostingRegressor` on *any* failure there, not just a
missing import, logging which branch ran either way. Verified the fix
catches the real reproduced exception (confirmed via a standalone script
before touching `rank.py`) and that the full pipeline now runs end-to-end
under `lightgbm==4.3.0` + `numpy==2.2.6` together, producing a
`submission.csv` whose 100 ranks are identical to the previous run (same
sklearn-gbr backend, same result -- this is a robustness fix, not a
scoring change).

Considered pinning `numpy<2.0` instead, but rejected it: there's no
Python-3.14 wheel for numpy 1.x, so it would have re-broken installation
(see #8), and it wouldn't actually fix lightgbm's own incompatibility --
it would just avoid triggering it on this one machine while leaving the
real fragility (lightgbm assuming old numpy copy semantics) unaddressed
for whoever's numpy resolves differently next.

## 10. Discovered the real candidates.jsonl is 100,000 rows, not 18,745 -- re-ran everything (2026-07-02)

Every number in this log through #9, and everything in `metrics_report.md`
/ `challenge_analysis.md` / `defense_notes.md` as originally written, was
computed against an 18,745-row `candidates.jsonl` that we now know was a
truncated dev-time download -- we didn't have the real file at build time.
The actual organizer release (`[PUB] India_runs_data_and_ai_challenge.zip`)
contains a 100,000-row `candidates.jsonl`, which the JD text itself
confirms is the intended scale ("we're not expecting to find many matches
in a 100K candidate pool" -- we'd read that line before and filed it under
"rhetorical," not literal).

Also found and fixed a real bug before re-running: `rank.py` sorted the
final ranking on full-precision `final_score` but wrote a 6-decimal
*rounded* score to the CSV, so two rows that were close-but-not-exactly-tied
pre-rounding could round together with `candidate_id` in the wrong tie-break
order. `validate_submission.py` caught this on a test run against sample
data ("Equal scores at ranks 61 and 62, tie-break requires candidate_id
ascending"). Fixed by rounding before sorting, not after.

Re-ran `rank.py` and `scripts/ablation_study.py` against the real 100,000-row
file: 56.7s wall clock (vs. 18.2s at the old 18,745-row scale -- still ~5x
under the 5-minute budget), 0 malformed lines (vs. 1 at the old scale),
`validate_submission.py` passes. Most proportions held up at the new scale
(bait-tier skill prevalence ~5%, signal-tier ~1.3% each, same candidate
`CAND_0008618` is still the clearest decoy-title case, the top-by-semantic-
similarity candidate is still independently a honeypot). One thing did NOT
hold up, and we're logging it rather than quietly patching it: our honeypot
detector's three signals, run against the real file, flag 410 candidates
(200 expert-proficiency + 175 founding-year-violation + 35
duration-mismatch, **zero overlap between them**), against the spec's
stated "~80" for the honeypot count. The spec's own worked example combines
multiple red flags in one profile; our real-scale result finding zero
candidates with more than one signal, plus a median founding-year severity
of just 1 year (vs. the spec's 3-year example), points to the founding-year
rule partly firing on generator noise across the wider release rather than
exclusively on designed honeypots. We did not retune the threshold to force
the count back toward ~80 -- that would be fitting to a number we were told
rather than to evidence in the data. What we verified instead: our actual
top 100 has 0 flagged honeypots either way, so this is a diagnostic-accuracy
gap in our own reporting, not a submission-correctness problem. Listed as an
open item in `defense_notes.md` §5.
