# Challenge Analysis: Redrob Intelligent Candidate Discovery & Ranking

This is a reverse-engineering pass over the released materials, written
before we wrote the ranking code. Everything below is checked against the
actual `candidates.jsonl` (100,000 records, all parse cleanly -- see "data
quality" note at the end), not just the prose in the JD/spec docs.

**2026-07-02 update:** the numbers below were re-measured against the real,
full `candidates.jsonl` (100,000 rows). Our original build (see
`experiments/exp_log.md`, entries through 2026-06-22) ran against a copy
that turned out to be a truncated 18,745-row subset, not the actual release
-- we didn't have the real file at the time. The proportions held up
(bait-tier skills are still ~5% of the population, signal-tier still ~1.3%
each), which is reassuring, but one finding did NOT hold up at full scale --
see the honeypot section below, flagged there rather than glossed over.

## 1. Explicit objective

Per `submission_spec.md`: rank the top 100 of 100,000 candidates against one
job description (Senior AI Engineer, Founding Team, Redrob AI), scored as

```
composite = 0.50 x NDCG@10 + 0.30 x NDCG@50 + 0.15 x MAP + 0.05 x P@10
```

against a hidden ground truth we never see. Format and compute constraints
(CPU-only, <=5 min, <=16GB, no network) are enforced separately at Stage 3
and disqualify regardless of composite score.

## 2. Actual objective (what we think the hidden labels reward)

The JD's own closing section spells this out unusually directly for a
hackathon brief:

> "The 'right answer' to this JD is not 'find candidates whose skills
> section contains the most AI keywords.' That's a trap we've explicitly
> built into the dataset."

We took that at face value and checked it against the data instead of just
trusting the prose. Two measurements changed how we built the ranker:

**Finding A -- skill frequency is inversely correlated with informativeness.**
We counted how often every skill tag appears across all 100,000 candidates.
The distribution is sharply bimodal:

| Skill | Candidates with it | Tier |
|---|---|---|
| Pinecone | 5,062 | bait |
| RAG | 4,995 | bait |
| Embeddings | 5,080 | bait |
| Sentence Transformers | 5,081 | bait |
| Vector Search | 5,065 | bait |
| LangChain | 5,162 | bait |
| Computer Vision | 4,755 | bait |
| BM25 | 1,382 | signal |
| Elasticsearch / OpenSearch / Milvus / Qdrant / Weaviate / pgvector | 1,286-1,394 each | signal |
| Learning to Rank | 1,383 | signal |
| Python / PyTorch / TensorFlow / scikit-learn | 1,288-1,381 each | signal |
| LoRA / QLoRA / PEFT | 1,371-1,401 each | signal |

The "bait" tier sits at ~5% of the population, scattered almost uniformly
across every title in the dataset -- the top titles holding Pinecone are
Civil Engineer (349), Graphic Designer (332), Accountant (327), Sales
Executive (325), Business Analyst (319), Operations Manager (315), and
Mechanical Engineer (304). The "signal" tier sits at ~1.3-1.4% per skill.
Framing "concentration" precisely (a single per-skill count doesn't tell you
this on its own): 750 candidates hold a genuinely ML/AI-track title
(tier_5_exact + tier_4_close in `weighted_job_representation.json`), and
97.7% of them (733/750) hold at least one signal-tier skill, versus only
13.8% of everyone else. That's the real gap -- not "signal skills only exist
on ML titles" (they don't; a low background rate of ~13.8% shows up
everywhere, consistent with the bait-tier's uniform-noise pattern), but "if
you're ML-titled, you almost certainly carry a signal skill; if you're not,
it's a coin-flip-adjacent minority." We didn't choose this framing in the
abstract -- we ran the frequency count first, then noticed it matches the
JD's *own* required-skill list almost exactly: Pinecone and FAISS (both
named explicitly in the JD as example vector DBs) are themselves in the
bait tier, while the less "famous" vector DBs the JD also names (Weaviate,
Qdrant, Milvus, OpenSearch, Elasticsearch) are in the signal tier. The more
recognizable the buzzword, the more it's been used as bait. A system that
rewards keyword overlap with the JD's required-skills list is, on this
dataset, rewarding exactly the candidates the JD says it doesn't want.

**Finding B -- the dataset's #1 candidate by pure semantic similarity is a
honeypot.** We built a TF-IDF+SVD similarity score against the JD text
purely as a diagnostic before writing the honeypot detector. The single
highest-scoring candidate by raw semantic similarity (0.921 cosine, among
the 300 stage-2 survivors) is `CAND_0077337`. Its profile independently
trips our honeypot detector: it claims "expert" proficiency in 11 skills
(QLoRA, pgvector, BM25, Information Retrieval, and others) -- a level that,
per the proficiency distribution below, only 200 candidates out of 100,000
use at all. This is the clearest demonstration we found that embedding
similarity and honeypot detection have to be separate, orthogonal layers: a
model that only scores semantic match would rank a known-suspicious profile
first.

## 3. Inferred recruiter logic

Reading the JD's "how to read between the lines" section as a literal
scoring spec rather than color commentary, the ideal candidate is:

- 6-8 years total experience, 4-5 of which are applied ML/AI *at a product
  company* (not pure services)
- Has shipped a ranking/search/recommendation system to real users at scale
  (a claim that should show up in `career_history[].description`, not just
  a skill tag)
- Has opinions on retrieval/evaluation/LLM-integration tradeoffs they can
  defend with reference to systems they built
- In or willing to relocate to Pune/Noida (the JD is explicit that
  non-India candidates are case-by-case, no visa sponsorship)
- Active on the platform -- behavioral signals matter because "for hiring
  purposes" an unreachable perfect-on-paper candidate isn't actually
  available

The JD also names explicit disqualifiers that aren't captured by any
single field: pure-research-only history, "LangChain + OpenAI in the last
12 months" with no pre-LLM production evidence, title-chasing (sub-18-month
average tenure across 3+ employers), CV/speech-only without NLP/IR, and
service-only career history (TCS/Wipro/Infosys/Accenture/Cognizant/
Capgemini/HCL/Mphasis) for the *entire* career, not just the current role.

## 4. Anti-keyword traps we found (and built against)

1. **Skill-tag stuffing on irrelevant titles** -- see Finding A above. We
   counter this with category-and-depth-weighted skill scoring
   (`src/features.py:skill_score`), not boolean keyword presence: a skill
   only counts at full weight if its claimed duration, endorsements, and
   (where available) Redrob assessment score back it up. We measured the
   gap directly: for the same skill (Pinecone), ML-titled candidates
   average 38.9 months of use / 21.3 endorsements / 68.6 assessment score
   vs. 15.7 months / 5.9 endorsements / 54.7 for non-ML-titled holders of
   the identical tag. Real, but a few points -- not a wall, which is why we
   weight title and career-text evidence more heavily than skill tags
   alone in the final composite.

2. **Honeypots** -- ~80 candidates per the spec, with "impossible dates,"
   "too many expert skills," and "duration inconsistency" named explicitly.
   We counted these directly rather than guessing thresholds, and this is
   the one place the full 100K re-run genuinely disagreed with our original
   (18,745-row) numbers, so we're reporting the disagreement rather than
   picking whichever count looks better:
   - 200 candidates use `proficiency: "expert"` on at least one skill (1,311
     "expert" tags total). No other candidate in the dataset uses this
     level at all (max is "advanced" for everyone else, 109,585 instances).
   - 175 candidates claim a `career_history` start date before a verifiable
     real-world founding year for their employer (checked against CRED
     2018, Razorpay 2014, Swiggy 2014, Zomato 2008 -- the only employers in
     the dataset with a public founding date).
   - 35 candidates have a `duration_months` value that disagrees with
     `end_date - start_date` by more than 3 months.
   - These three signals sum to **410** candidates (0.41% of the
     population) -- and, unlike our original dev-subset run, are
     **completely disjoint at full scale: zero candidates trip more than
     one signal.** That's a red flag against our own detector, not for it:
     the spec's own worked example ("8 years of experience at a company
     founded 3 years ago; expert proficiency in 10 skills with 0 years
     used") describes a *combined* profile with multiple simultaneous
     red flags, and the founding-year violations we find have a median
     severity of just 1 year early (vs. the spec's 3-year example) -- mild
     enough, and isolated enough, that we think this signal is partly
     firing on generator noise in the wider 100K release rather than only
     on the ~80 deliberately designed honeypots. We did not retune the
     threshold post-hoc to force the count back down to ~80 (that would be
     fitting to a number we were told, not to evidence); instead we're
     flagging it as an open gap. What we can say with confidence: our
     actual submission's top 100 has 0 flagged honeypots either way, so
     this doesn't change Stage-3 pass/fail, only the accuracy of the
     diagnostic count in our own docs. See `src/honeypot.py` for the
     scoring function and `experiments/exp_log.md` (2026-07-02 entry) for
     the full re-investigation.

3. **Title-chasing and framework-only profiles** -- harder to verify against
   ground truth since the spec doesn't quantify these, but we built explicit
   penalties (`src/features.py:negative_signal_multiplier`) for: 3+ employers
   averaging under 18 months tenure each; LangChain present with no
   pre-LLM-era production-ML skill backing it; CV/speech title with no
   NLP/IR skill at all.

## 5. Weak / strong / winning teams

**Weak teams** will compute embedding similarity between candidate skill
lists and the JD's skill list, sort descending, and submit. This scores
well on the bait-tier skills by construction and ranks honeypots highly
whenever the honeypot's text happens to be keyword-dense (exactly the
`CAND_0077337` case above). It also has no defense at Stage 4: "we computed
cosine similarity" is not a methodology a judge can probe.

**Strong teams** will build a honeypot detector and a skills taxonomy that
weights production-ML/IR signals over LLM-framework buzzwords, and will use
behavioral signals as a multiplier. This catches most of the trap but still
risks conflating "skills only" matching with team's the JD calls out
explicitly: a candidate who never touched a vector DB but has a
recommendation-systems shipping record buried in a career_history
description, under a "Backend Engineer" title.

**Winning teams** will additionally: (a) verify their honeypot/skill-tier
thresholds against actual frequency counts in the released data rather than
intuition, (b) treat title-tier and skill-tag evidence as independently
gateable rather than additively interchangeable, so a perfect skill list
under a "Marketing Manager" title cannot out-rank a real ML practitioner
under a "Backend Engineer" title with weaker tags but real career evidence,
(c) be honest in their offline evaluation about not having access to the
real ground truth (see `metrics_report.md`), and (d) be able to defend every
number in their pipeline in a live walkthrough, because Stage 5 is a
30-minute interview, not another scoring pass.

## 6. Data quality note

The real, full `candidates.jsonl` release (100,000 lines) parses cleanly --
0 malformed lines. Our original dev build (through 2026-06-22, see
`experiments/exp_log.md`) worked against a truncated 18,746-line copy where
the last record failed to parse; `src/ingest.py`'s skip-and-warn-rather-than-
crash behavior was built against that copy and is kept because malformed
input is still a real possibility in an unfamiliar grading sandbox, even
though the actual release doesn't trigger it.
