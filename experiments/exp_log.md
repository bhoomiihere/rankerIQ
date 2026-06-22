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
