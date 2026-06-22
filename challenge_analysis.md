# Challenge Analysis: Redrob Intelligent Candidate Discovery & Ranking

This is a reverse-engineering pass over the released materials, written
before we wrote the ranking code. Everything below is checked against the
actual `candidates.jsonl` (18,745 usable records -- see "data quality" note
at the end), not just the prose in the JD/spec docs.

## 1. Explicit objective

Per `submission_spec.md`: rank the top 100 of 18,745 candidates against one
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
We counted how often every skill tag appears across all 18,745 candidates.
The distribution is sharply bimodal:

| Skill | Candidates with it | Tier |
|---|---|---|
| Pinecone | 965 | bait |
| RAG | 938 | bait |
| Embeddings | 958 | bait |
| Sentence Transformers | 979 | bait |
| Vector Search | 950 | bait |
| LangChain | 965 | bait |
| Computer Vision | 918 | bait |
| BM25 | 233 | signal |
| Elasticsearch / OpenSearch / Milvus / Qdrant / Weaviate / pgvector | 247-270 each | signal |
| Learning to Rank | 268 | signal |
| Python / PyTorch / TensorFlow / scikit-learn | 254-261 each | signal |
| LoRA / QLoRA / PEFT | 251-258 each | signal |

The "bait" tier sits at ~5% of the population, scattered almost uniformly
across every title in the dataset -- 68 Sales Executives, 67 Graphic
Designers, 49 Mechanical Engineers, and 47 HR Managers all list Pinecone as
a skill. The "signal" tier sits at ~1.3-1.4%, concentrated almost entirely
inside the 226 candidates whose current title is genuinely ML/AI-track. We
didn't choose this framing in the abstract -- we ran the frequency count
first, then noticed it matches the JD's *own* required-skill list almost
exactly: Pinecone and FAISS (both named explicitly in the JD as example
vector DBs) are themselves in the bait tier, while the less "famous" vector
DBs the JD also names (Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch)
are in the signal tier. The more recognizable the buzzword, the more it's
been used as bait. A system that rewards keyword overlap with the JD's
required-skills list is, on this dataset, rewarding exactly the candidates
the JD says it doesn't want.

**Finding B -- the dataset's #1 candidate by pure semantic similarity is a
honeypot.** We built a TF-IDF+SVD similarity score against the JD text
(details in `challenge_analysis.md` section 4) purely as a diagnostic before
writing the honeypot detector. The single highest-scoring candidate by raw
semantic similarity (0.897 cosine) is `CAND_0005538`, titled "Senior AI
Engineer." Its profile independently trips our honeypot detector (it has
"expert"-level proficiency tags -- a level that does not appear anywhere
else among the 18,707 non-honeypot candidates in the released data). This
is the clearest demonstration we found that embedding similarity and
honeypot detection have to be separate, orthogonal layers: a model that only
scores semantic match would rank a known-impossible profile first.

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
   average 39 months of use / 21 endorsements / 63 assessment score vs.
   16 months / 6 endorsements / 57 for non-ML-titled holders of the
   identical tag. Real, but a few points -- not a wall, which is why we
   weight title and career-text evidence more heavily than skill tags
   alone in the final composite.

2. **Honeypots** -- ~80 candidates per the spec, with "impossible dates,"
   "too many expert skills," and "duration inconsistency" named explicitly.
   We counted these directly rather than guessing thresholds:
   - 38 candidates use `proficiency: "expert"` on at least one skill. No
     other candidate in the dataset uses this level at all (max is
     "advanced" for everyone else, 20,597 instances). This alone is close
     to a ground-truth tell.
   - 33 candidates claim a `career_history` start date before a verifiable
     real-world founding year for their employer (checked against CRED
     2018, Razorpay 2014, Swiggy 2014, Zomato 2008 -- the only employers in
     the dataset with a public founding date).
   - 7 candidates have a `duration_months` value that disagrees with
     `end_date - start_date` by more than 3 months (some by over 10 years).
   These three signals are almost entirely disjoint and sum to **78**
   candidates -- against a spec-stated "~80." See `src/honeypot.py` for the
   scoring function and `experiments/exp_log.md` for the count-first
   methodology.

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
`CAND_0005538` case above). It also has no defense at Stage 4: "we computed
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

The released `candidates.jsonl` has 18,746 lines; the last one
(`CAND_0018746`) is truncated mid-string and fails to parse as JSON. We
treat this as a data artifact (most likely a copy/upload truncation, not an
intentional honeypot, since the failure mode is a syntactically broken
record rather than a semantically odd-but-valid one) and skip it with a
logged warning rather than crashing the pipeline. See `src/ingest.py`.
