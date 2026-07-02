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
  submission.csv` passes, re-verified against the real 100,000-row
  `candidates.jsonl` on 2026-07-02 (earlier runs used a truncated
  18,745-row dev copy; see `experiments/exp_log.md`).
- `rank.py --candidates candidates.jsonl --out submission.csv` — single
  command, 56.7s wall clock on 2 CPU cores / 4GB RAM against the real
  100,000-row file (budget: 5 min / 16GB), zero network calls, no GPU.
- `README.md`, `requirements.txt`, `Dockerfile`, `submission_metadata.yaml`
  (mostly filled — see blockers below), `metrics_report.md`,
  `experiments/exp_log.md`, `tests/test_metrics.py`.
- Git history: 26+ commits with real, individually-justified design
  decisions and multiple documented self-corrections (ablation
  interpretation, CV-without-NLP/IR rule hardening, reasoning
  rank-consistency gap, tie-break rounding bug, full-scale data re-run) —
  not a single dump.
- `Bk_Redrob_Submission.pptx` — built into the actual organizer template
  (`Idea Submission Template _ Redrob.pptx`), all 11 slides filled,
  rendered and visually checked slide-by-slide. Not yet re-checked against
  the corrected 100K-scale numbers — see blockers below.

**Blocking — needs Meghna, can't be filled by the pipeline:**

- `submission_metadata.yaml`: real phone number (`primary_contact.phone`),
  confirmation of Bhoomi's actual email if it differs from
  `mkgapbvk@gmail.com`, the real `compute.platform` line for whichever
  machine actually produces the final submission run.
- `github_repo` — now known and pushed: `https://github.com/bhoomiihere/rankerIQ`.
  Still needs to be confirmed in `submission_metadata.yaml` + PPT slide 9
  once the latest commits (bugfix + real-data re-run + doc corrections) are
  pushed.
- `sandbox_link` — **mandatory**, flagged at Stage 1 if missing. README has
  a "Deploying the sandbox link" section with the Streamlit/HF Spaces
  options; none is deployed yet. This is still the single biggest open item.
- `Bk_Redrob_Submission.pptx` — built against the old 18,745-row numbers;
  needs a pass to match the corrected 100,000-row figures before upload.
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

**Stage 3 — Code reproduction + honeypot check.** Pass, verified in an
actual Docker clean-room on 2026-07-02: `docker build` from the committed
`Dockerfile`, then `docker run --network none --cpus 2 --memory 4g` against
the real 100,000-row `candidates.jsonl`. 85.3s wall clock (56.7s running
natively; the container is the number that matters since the grading
sandbox is containerized), zero network access (enforced by `--network
none`, not just asserted), 0 honeypots in the final top 100 against a
>10%-disqualifies threshold (179 honeypots were in the stage-2 pool of 300
— all filtered; see the "Honeypots" section of `challenge_analysis.md` for
why this count is much higher than the spec's approximate ~80 figure and
what we think is causing that, reported honestly rather than tuned away).
Two consecutive container runs produce byte-identical output. The committed
`submission.csv` is the Linux-container output specifically — macOS-native
and Linux runs differ by float noise (same 100 candidates, no rank moves
beyond 2 positions, max score delta 0.0014), and the grader reproduces on
Linux, so that's the platform our committed file should match. Full detail
in `experiments/exp_log.md` §11.

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

**"Walk me through what happens when I run `rank.py`."** BM25 cuts 100,000
to 1,000 in ~31s; TF-IDF+SVD (fit once on the full corpus, not just
survivors — see `exp_log.md` §4 for why we changed this) cuts 1,000 to 300
in ~22s; the 300 get full feature scoring plus honeypot detection; a 5-fold
CV diagnostic runs, then a GBT reranker trains on heuristic pseudo-labels
and produces `final_score = 0.5*rule_score + 0.5*model_score` for the
ensemble. Total ~57s against the real dataset, well under the 5-minute
budget.

**"Why two scores blended instead of just the better one?"** The rule
score is fully auditable (every number traces to a brief-literal formula)
but has a real, documented blind spot: `title_score` only enters through a
0.35 sub-weight inside the 0.15-weighted `experience_fit` term (~5% of the
total), so a near-zero title score can't suppress a candidate who also
scores well on `semantic` + `retrieval_skill_match` (50% of the composite,
untouched by title). We found this via `CAND_0008618` (rule-only rank 5,
model-only rank 84, ensemble rank 46) — see `exp_log.md` §5. The learned
model fixes this because its training labels gate on title directly. We
kept the rule score in the blend anyway because it's the part we can fully
explain without "the model learned it" as the answer.

**"Your ablation says flat skill weighting beats category weighting —
doesn't that undercut your whole skills taxonomy?"** It's genuinely mixed
at full 100K scale, not a clean loss — category weighting actually wins
NDCG@10 (0.828 vs 0.822), flat wins NDCG@50 and MAP. We say why in
`metrics_report.md` directly rather than hiding the split: the pseudo-label
target is now dominated by tier 0 (179/300 — see the honeypot question
below), and skill weighting doesn't touch honeypot filtering at all, so the
NDCG/MAP gap is decided by a smaller, noisier remainder (the 121 non-tier-0
survivors). The actual case for category weighting is qualitative — our
first baseline (plain keyword overlap) put a Mechanical Engineer with 6
stuffed skill tags in the top 10 on a manual spot-check; category weighting
is what fixes that, not this NDCG table.

**"How do you know your honeypot detector isn't just overfit to this exact
dataset?"** Here's the honest answer, including where we were wrong: we
originally counted the three signals against an 18,745-row copy of
`candidates.jsonl` that turned out to be a truncated dev-time download, not
the real release (38+33+7=78, which we thought matched spec's stated ~80).
Re-run against the actual 100,000-row file, the same three rules find 200 +
175 + 35 = 410 candidates (0.41% of the population — proportionally close
to the original 78/18,745 rate, but 5x the spec's ~80 in absolute terms).
More concerning: at full scale, zero candidates trip more than one signal,
where the spec's own worked example describes a honeypot with *multiple*
simultaneous red flags. Our founding-year-violation rule in particular has
a median severity of just 1 year early, milder than the spec's 3-year
example. Our honest read: this rule is partly firing on generator noise in
the wider release, not exclusively on designed honeypots, and we didn't
retune the threshold after learning the "expected" answer — that would be
fitting to a number we were told, not to evidence. What doesn't change:
our actual top 100 has 0 flagged honeypots regardless, so the thing the
grader actually checks (honeypot rate in the top 100) still passes.

**"What didn't work?"** Three things, kept visible in `exp_log.md` rather
than removed: keyword-overlap-only scoring (the original anti-keyword trap
victim), honeypot probability as an additive feature instead of a
multiplicative gate (a strong honeypot still ranked 4th), and fitting the
SVD basis on stage-1 survivors only (made scores non-reproducible across
`--stage1-k` settings).

**"Why sklearn instead of LightGBM if you designed for LightGBM?"**
`train_ranker.py` tries lightgbm first and falls back to
`sklearn.GradientBoostingRegressor` with the same train/score interface,
logged with a warning every time it fires. `requirements.txt` deliberately
does NOT pin lightgbm (we discovered during the Docker clean-room test that
the file had been truncated mid-comment and the pin was never actually
there — and on reflection, leaving it out is the right call): the committed
`submission.csv` came from the sklearn backend, and installing lightgbm in
the grading sandbox could produce a *different* top 100 than the file we
submitted — a Stage-3 reproduction mismatch. Every number in
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
- Tighten the founding-year-violation honeypot signal (see §4's honeypot
  question) — likely require severity >=2 years and/or co-occurrence with
  another signal before counting a candidate, instead of any single
  1-year-early date. Not done yet because we don't have ground truth to
  validate a new threshold against; the current signal's effect on the
  actual top 100 is zero either way, so this is a diagnostic-accuracy fix,
  not a submission-correctness one.
- Deploy the sandbox link — outside what this build can do on its own.
