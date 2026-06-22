# Defense notes, final checklist, and self-judge score — Team Bk

Written after the build, against the actual evaluation pipeline described in
`docs/submission_spec.docx` (not a generic rubric) — Sections 5 and the
reasoning-quality table in Section 3 are the source for every check below.
Where something can't be verified without the hidden ground truth, that's
stated plainly rather than assumed.

## 1. Final checklist before upload

**Ready (built and verified in this repo):**

- `submission.csv` — 100 rows + header, ranks 1-100 unique, scores
  non-increasing, every `candidate_id` real — `python validate_submission.py
  submission.csv` passes.
- `rank.py --candidates candidates.jsonl --out submission.csv` — single
  command, 18.2s wall clock on 2 CPU cores / 4GB RAM (budget: 5 min / 16GB),
  zero network calls, no GPU.
- `README.md`, `requirements.txt`, `Dockerfile`, `submission_metadata.yaml`
  (mostly filled — see blockers below), `metrics_report.md`,
  `experiments/exp_log.md`, `tests/test_metrics.py`.
- Git history: 19 commits with real, individually-justified design
  decisions and three documented self-corrections (ablation interpretation,
  CV-without-NLP/IR rule hardening, reasoning rank-consistency gap) — not a
  single dump.
- `Bk_Redrob_Submission.pptx` — built into the actual organizer template
  (`Idea Submission Template _ Redrob.pptx`), all 11 slides filled,
  rendered and visually checked slide-by-slide.

**Blocking — needs Meghna, can't be filled by the pipeline:**

- `submission_metadata.yaml`: real phone number (`primary_contact.phone`),
  confirmation of Bhoomi's actual email if it differs from
  `mkgapbvk@gmail.com`, the real `compute.platform` line for whichever
  machine actually produces the final submission run.
- `github_repo` — this repo needs to be pushed to GitHub and the URL filled
  in (`submission_metadata.yaml` + PPT slide 9). Per the original
  instruction, this build does not push on your behalf.
- `sandbox_link` — **mandatory**, flagged at Stage 1 if missing. README has
  a "Deploying the sandbox link" section with the Streamlit/HF Spaces
  options; none is deployed yet. This is the single biggest open item.
- Three-submission cap: this would be submission attempt #1 if uploaded
  now. No submissions have actually been made.

## 2. Self-judge against the real evaluation pipeline (5 stages, spec §5)

**Stage 1 — Format validation.** Pass. `validate_submission.py` checks
every rule in spec §3 (exact 100 rows, unique ranks 1-100, unique
candidate_ids that all exist in `candidates.jsonl`, non-increasing scores)
and the live run is clean.

**Stage 2 — Scoring against hidden ground truth.** Unknown by construction
— we never see it. What we *can* say honestly: our own pseudo-label CV
(NDCG@10 ~0.99 avg) only proves the pipeline is internally consistent, not
that it matches Redrob's real labels (this caveat is the first thing
`metrics_report.md` says, on purpose).

**Stage 3 — Code reproduction + honeypot check.** Pass, self-verified.
18.2s / 2 cores / well under 16GB, zero network calls (verified with network
disabled), 0 honeypots in the final top 100 against a >10%-disqualifies
threshold (36 honeypots were in the stage-2 pool of 300 — all filtered).
`Dockerfile` exists and should be tested with `docker build && docker run`
on a clean machine before the real submission, which we have not done in
this sandbox (no Docker daemon available here — flagged honestly, not
silently skipped).

**Stage 4 — Manual review.** Self-judged directly against the spec's own
six reasoning-quality checks (Table 2), see Section 3 below — this is
where we found and fixed a real gap, documented in `exp_log.md` §7.
Methodology coherence and git-history authenticity are addressed by the
README's "Assumptions/Limitations" sections and the 19-commit history with
visible failed attempts (keyword-overlap baseline, additive honeypot
penalty, SVD-on-survivors-only — all in `exp_log.md`).

**Stage 5 — Defend-your-work interview.** Can't self-judge a live
interview; Section 4 below is the prep for it — the questions we'd expect
to get and what the honest, specific answer is for each, pulled from code
and logs we actually have, not rehearsed talking points.

## 3. Self-judge: the six Stage-4 reasoning checks (spec §3, Table 2)

Sampled 10 rows (ranks 1, 2, 3, 10, 25, 50, 51, 75, 90, 100) — same sample
size the spec itself uses — and scored each check 0-100 based on what
actually held across the sample and the wider 100 rows, not just the best
case.

| # | Check | Score | Why |
|---|---|---|---|
| 1 | Specific facts | 95 | Every row cites years/title/employer, named matched skills, a real career_history snippet, notice period, response rate, last-active date. Nothing generic. |
| 2 | JD connection | 80 | Skill names are drawn from the JD's own required/preferred lists, so the connection is real but implicit — the text never explicitly says "JD asks for 5-9 years, this candidate has X." Left as-is rather than padding every row with a restated JD clause that would make 100 rows repetitive. |
| 3 | Honest concerns | 85 (was ~55 before the fix) | Originally 8/10 sampled rows used the same "no concerns" closer regardless of how thin the actual match was. Fixed by adding a thin-skill-overlap check (`src/reasoning.py`); now fires for the 20/100 candidates with <=1 matched required and <=1 matched preferred skill. Not 100 because some borderline cases (2 matched skills, weak career-history relevance) still get no concern noted — a judgment call we documented rather than hid. |
| 4 | No hallucination | 95 | Reasoning is built entirely from template fragments over fields already in the candidate record or the computed feature dict — no generation step exists that could invent a fact. Spot-checked 3 candidates' raw JSON against their reasoning string; every claim traced. |
| 5 | Variation | 85 | The substantive content (employer, skills, career snippet, concern) differs across all 10 sampled rows. The fixed closing clause ("No material concerns flagged by our rule set.") repeats verbatim across rows that have no flagged concern — that's a deliberate, honest "nothing found" signal, not padding, but it is the one part of the template that's literally identical across rows. |
| 6 | Rank consistency | 80 (was ~50 before the fix) | This was the real failure mode found: rank-90/100 candidates read as tonally identical to rank-1. Fixed for the bottom ~20 candidates (thin-skill-overlap concern now visibly present at rank 100, see `exp_log.md` §7). Ranks in the 50-89 band with 2 matched skills still read fairly neutral/positive despite being mid-pack — a partial fix, called out as partial rather than claimed as fully solved. |

**Aggregate self-judge score: (95+80+85+95+85+80)/6 ≈ 86.7 / 100** for the
reasoning-quality dimension specifically.

This is below the "iterate if total < 95" bar from the original brief, so
per that instruction we did iterate (the thin-skill-overlap fix above) —
but it's reported as 86.7, not adjusted upward to clear 95, because the
remaining gaps (check #2's implicit-only JD connection, check #6's partial
fix for the 50-89 rank band) are real and the honest move is to say so and
suggest the next iteration rather than re-score until the number clears the
bar. The other four checks (1, 4, 5, and 3 after the fix) are genuinely
strong. If there's time before the real submission, the highest-value next
step is making check #6 explicit for the 50-89 band too — see Section 5.

## 4. Anticipated defense-interview questions (Stage 5) and grounded answers

**"Walk me through what happens when I run `rank.py`."** BM25 cuts 18,745
to 1,000 in ~9s; TF-IDF+SVD (fit once on the full corpus, not just
survivors — see `exp_log.md` §4 for why we changed this) cuts 1,000 to 300
in ~7s; the 300 get full feature scoring plus honeypot detection; a 5-fold
CV diagnostic runs, then a GBT reranker trains on heuristic pseudo-labels
and produces `final_score = 0.5*rule_score + 0.5*model_score` for the
ensemble. Total ~18s.

**"Why two scores blended instead of just the better one?"** The rule
score is fully auditable (every number traces to a brief-literal formula)
but has a real, documented blind spot: `title_score` only enters through a
0.35 sub-weight inside the 0.15-weighted `experience_fit` term (~5% of the
total), so a near-zero title score can't suppress a candidate who also
scores well on `semantic` + `retrieval_skill_match` (50% of the composite,
untouched by title). We found this via `CAND_0008618` (rule-only rank 3,
model-only rank 43, ensemble rank 39) — see `exp_log.md` §5. The learned
model fixes this because its training labels gate on title directly. We
kept the rule score in the blend anyway because it's the part we can fully
explain without "the model learned it" as the answer.

**"Your ablation says flat skill weighting beats category weighting —
doesn't that undercut your whole skills taxonomy?"** No, and we say why in
`metrics_report.md` directly rather than hiding the number: the pseudo-
label target has only 5 tiers and 215/300 survivors land in one tier, so
NDCG against it is sensitive to within-tier noise, not to whether a
weighting scheme reflects the JD's intent. The actual case for category
weighting is qualitative — our first baseline (plain keyword overlap) put
a Mechanical Engineer with 6 stuffed skill tags in the top 10 on a manual
spot-check; category weighting is what fixes that, not this NDCG table.

**"How do you know your honeypot detector isn't just overfit to this exact
dataset?"** We don't claim it generalizes — we counted the three signals
(expert-tier proficiency, pre-founding employment dates, duration
mismatches) directly against this released data first (38+33+7=78, against
spec's stated ~80) instead of picking thresholds from intuition, and we
explicitly stopped chasing the last 2 candidates rather than add a fourth
rule tuned to this exact count — see `exp_log.md` §1.

**"What didn't work?"** Three things, kept visible in `exp_log.md` rather
than removed: keyword-overlap-only scoring (the original anti-keyword trap
victim), honeypot probability as an additive feature instead of a
multiplicative gate (a strong honeypot still ranked 4th), and fitting the
SVD basis on stage-1 survivors only (made scores non-reproducible across
`--stage1-k` settings).

**"Why sklearn instead of LightGBM if you designed for LightGBM?"**
`requirements.txt` pins `lightgbm`; it wasn't importable in this dev
sandbox (no network to resolve the build), so `train_ranker.py` falls back
to `sklearn.GradientBoostingRegressor` with the same train/score interface,
logged with a warning every time it fires. Every number in
`metrics_report.md` is honestly labeled `backend: sklearn-gbr` — we did not
report LightGBM numbers we never actually generated.

## 5. What we'd do next with more time (not done, stated plainly)

- Make reasoning-tone scaling explicit across the full rank range, not just
  the bottom ~20 (check #6 above is a partial fix).
- A learned title×skill interaction term at the rule-score level (logged as
  intentionally not done in `exp_log.md` §6, to keep the rule score an
  honest implementation of the brief's literal formula).
- Real sentence-transformer embeddings for stage 2, blocked by the
  CPU-only/no-network constraint in this environment.
- Deploy the sandbox link and push the GitHub repo — both outside what this
  build can do on its own.
