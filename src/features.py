"""
Candidate feature engineering.

The skill-category weights and the "rare skill = signal, common skill =
noise" framing below come directly from counting skill frequencies across
all 18,745 candidates (see experiments/exp_log.md, "iteration 2"). Two
findings drove the design:

1. Skills like Pinecone, RAG, FAISS, Embeddings, Vector Search, LangChain,
   Computer Vision, etc. show up in 850-980 candidates each -- roughly 5% of
   the whole population, spread almost uniformly across every title
   including Mechanical Engineer and Sales Executive. These are the
   "anti-keyword trap" the JD warns about: the more recognizable a buzzword
   is, the more it's been scattered as bait.
2. Skills like BM25, Elasticsearch, OpenSearch, Milvus, Qdrant, Weaviate,
   pgvector, Learning to Rank, Python, PyTorch, Deep Learning, NLP, LoRA,
   QLoRA, PEFT sit at 230-280 candidates -- concentrated almost entirely in
   the ~226 candidates whose titles are genuinely ML/AI roles. These are the
   real tells.

We did not hand-pick this split after the fact -- we ran the frequency
count first, then matched it against which skills the JD names explicitly,
and found the JD's own required list (Pinecone, FAISS, RAG, embeddings)
overlaps heavily with the bait tier. That's the headline finding in
challenge_analysis.md: a system that rewards candidates for having the
JD's exact keyword list is rewarding exactly the candidates the JD is
trying to filter out.

Given that, we weight each skill match by:
  category_weight (production ML > IR > retrieval infra > LLM ops > generic infra > irrelevant)
  x claim_strength (duration_months, endorsements, and Redrob's own skill
    assessment score where available -- these are the parts of a skill claim
    a candidate can't easily pad for free)

claim_strength matters because of a second thing we measured: for the same
skill (e.g. Pinecone), candidates with ML-track titles average 39 months of
use, 21 endorsements, and a 63 assessment score; candidates with unrelated
titles holding the identical skill tag average 16 months, 6 endorsements,
and a 57 assessment score. The gap is real but it's a few points, not a
wall -- which is exactly why we don't rely on skill claims alone and weight
title/career evidence more heavily in the final composite.
"""

import math
from datetime import datetime

TODAY = datetime(2026, 6, 22)

# --- skill categories, ordered Production ML > IR > Retrieval > LLM Ops > Infra ---
SKILL_CATEGORY_WEIGHT = {}
for s in ["Recommendation Systems", "Ranking Systems", "Learning to Rank", "Machine Learning",
          "Deep Learning", "Statistical Modeling", "Feature Engineering", "Reinforcement Learning"]:
    SKILL_CATEGORY_WEIGHT[s] = 1.00  # production ML
for s in ["Information Retrieval", "Information Retrieval Systems", "BM25", "Semantic Search",
          "Search & Discovery", "Search Backend", "NLP", "Natural Language Processing"]:
    SKILL_CATEGORY_WEIGHT[s] = 0.95  # IR
for s in ["Vector Search", "Vector Representations", "Embeddings", "Sentence Transformers", "FAISS",
          "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch", "Elasticsearch", "pgvector",
          "RAG", "Haystack", "LlamaIndex", "Hugging Face Transformers", "Content Matching"]:
    SKILL_CATEGORY_WEIGHT[s] = 0.85  # retrieval infra
for s in ["Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT", "Model Adaptation", "Prompt Engineering",
          "LangChain", "LLMs", "MLflow", "Weights & Biases", "BentoML", "Kubeflow", "MLOps",
          "Text Encoders"]:
    SKILL_CATEGORY_WEIGHT[s] = 0.65  # LLM ops
for s in ["Python", "PyTorch", "TensorFlow", "scikit-learn"]:
    SKILL_CATEGORY_WEIGHT[s] = 0.90  # core ML tooling -- JD calls Python out by name
for s in ["AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Kafka", "Spark", "Airflow",
          "Databricks", "Snowflake", "Hadoop", "Apache Beam", "Apache Flink", "BigQuery",
          "Data Pipelines", "ETL", "Workflow Orchestration", "CI/CD", "Microservices", "gRPC",
          "REST APIs", "PostgreSQL", "MongoDB", "Redis", "dbt", "Go", "Rust", "Java", "JavaScript",
          "TypeScript"]:
    SKILL_CATEGORY_WEIGHT[s] = 0.35  # generic infra/eng -- relevant but not differentiating
for s in ["Computer Vision", "CNN", "Object Detection", "Image Classification", "OpenCV", "GANs",
          "Diffusion Models", "YOLO", "ASR", "TTS", "Speech Recognition", "Data Science",
          "Time Series", "Forecasting"]:
    SKILL_CATEGORY_WEIGHT[s] = 0.30  # CV/speech -- JD explicitly deprioritizes without NLP/IR

EVALUATION_SKILLS = {"NDCG", "MAP", "MRR", "Offline Evaluation", "A/B Testing", "Evaluation Frameworks"}


def claim_strength(skill, assessment_scores):
    """0..1 multiplier on how credible a single skill claim is, from signals
    a candidate can't pad for free: how long they say they've used it, how
    many people endorsed it, and (if available) their Redrob assessment
    score for that exact skill."""
    duration_component = min(skill.get("duration_months", 0) / 36.0, 1.0)
    endorsement_component = min(skill.get("endorsements", 0) / 20.0, 1.0)
    assessment = assessment_scores.get(skill["name"])
    assessment_component = (assessment / 100.0) if assessment is not None else 0.55  # neutral prior
    return 0.4 * duration_component + 0.3 * endorsement_component + 0.3 * assessment_component


def skill_score(candidate, job_repr):
    """retrieval_skill_match: weighted, depth-checked match against the
    required/preferred skill lists in weighted_job_representation.json."""
    required = set()
    for group in job_repr["required_skills"].values():
        if isinstance(group, list):
            required.update(group)
    preferred = set()
    for group in job_repr["preferred_skills"].values():
        if isinstance(group, list):
            preferred.update(group)

    assessment_scores = candidate["redrob_signals"]["skill_assessment_scores"]
    total = 0.0
    matched_required, matched_preferred = [], []
    for s in candidate["skills"]:
        cat_w = SKILL_CATEGORY_WEIGHT.get(s["name"], 0.05)  # unlisted/irrelevant skills barely count
        depth = claim_strength(s, assessment_scores)
        if s["name"] in required:
            total += 1.0 * cat_w * depth
            matched_required.append(s["name"])
        elif s["name"] in preferred:
            total += 0.5 * cat_w * depth
            matched_preferred.append(s["name"])
        else:
            total += 0.05 * cat_w * depth  # small credit for ML-adjacent skills outside the lists

    # normalize against a realistic max so scores land roughly in [0,1]
    # (a candidate matching ~6 strong required skills at full depth saturates this)
    normalized = min(total / 6.0, 1.0)
    return normalized, matched_required, matched_preferred


def evaluation_fit_score(candidate):
    names = {s["name"] for s in candidate["skills"]}
    hits = names & EVALUATION_SKILLS
    # NDCG/MAP/MRR rarely appear as literal skill tags in this dataset (checked: they don't),
    # so we also credit roles whose description text mentions evaluation work, and treat
    # "Learning to Rank" + "BM25" co-occurrence as indirect evidence of ranking evaluation exposure.
    text = " ".join(r.get("description", "") for r in candidate["career_history"]).lower()
    keyword_hits = sum(1 for kw in ["ndcg", "map@", "mean average precision", "a/b test",
                                     "offline evaluation", "evaluation framework", "precision@",
                                     "recall@"] if kw in text)
    score = min(0.5 * len(hits) + 0.25 * keyword_hits, 1.0)
    return score


def title_tier_score(candidate, job_repr):
    title = candidate["profile"]["current_title"]
    tiers = job_repr["title_tiers"]
    if title in tiers["tier_5_exact"]:
        base = 1.0
    elif title in tiers["tier_4_close"]:
        base = 0.75
    elif title in tiers["tier_3_adjacent"]:
        base = 0.45
    elif title in tiers["tier_2_weak_adjacent"]:
        base = 0.2
    else:
        base = 0.05
    penalty_note = tiers["penalized_titles"].get(title)
    if penalty_note:
        base *= 0.6
    return base, penalty_note


def company_quality_score(candidate, job_repr):
    industry = candidate["profile"]["current_industry"]
    ai_industries = {"AI/ML", "Conversational AI", "AI Services", "HealthTech AI", "Voice AI"}
    product_industries = {"Fintech", "E-commerce", "Food Delivery", "EdTech", "SaaS", "Gaming",
                           "Transportation", "Internet", "AdTech", "Insurance Tech"}
    if industry in ai_industries:
        score = 1.0
    elif industry in product_industries:
        score = 0.6
    elif industry in {"IT Services", "Consulting"}:
        score = 0.25
    else:  # Manufacturing, Paper Products, Conglomerate, etc -- generic non-tech employer
        score = 0.35
    return score


def career_progression_score(candidate):
    """Rewards stable, escalating tenure at a small number of employers;
    penalizes the JD's named 'title-chaser' pattern (3+ employers, <18mo avg
    tenure each)."""
    history = candidate["career_history"]
    n = len(history)
    if n == 0:
        return 0.5
    avg_tenure = sum(h["duration_months"] for h in history) / n
    if n >= 3 and avg_tenure < 18:
        return 0.2  # title-chaser pattern
    if avg_tenure >= 30:
        return 1.0
    return max(0.3, avg_tenure / 30.0)


def experience_band_fit(candidate, job_repr):
    years = candidate["profile"]["years_of_experience"]
    lo, hi = job_repr["experience_band"]["min_years"], job_repr["experience_band"]["max_years"]
    if lo <= years <= hi:
        return 1.0
    distance = (lo - years) if years < lo else (years - hi)
    return max(0.0, 1.0 - distance / 5.0)  # soft falloff, ~5 years outside band -> 0


def negative_signal_multiplier(candidate, job_repr, is_service_only_fn):
    """Returns a multiplier in (0, 1] applied to the *whole* composite score.
    These represent 'wrong type of candidate', not 'slightly weaker
    candidate', so they scale the total rather than one sub-score -- a
    candidate who is otherwise perfect but consulting-only for their entire
    career should not rank above a solid generalist."""
    mult = 1.0
    reasons = []
    neg = job_repr["negative_signals"]

    if is_service_only_fn(candidate):
        mult *= (1 - neg["service_only_career"]["penalty_if_entire_history"])
        reasons.append("entire career history at IT-services/consulting firms (JD soft negative)")

    skill_names = {s["name"] for s in candidate["skills"]}
    has_langchain = "LangChain" in skill_names
    pre_llm_evidence = any(
        s["name"] in {"Information Retrieval", "BM25", "Learning to Rank", "Machine Learning",
                       "Deep Learning", "Recommendation Systems"} and s.get("duration_months", 0) > 24
        for s in candidate["skills"]
    )
    if has_langchain and not pre_llm_evidence:
        mult *= (1 - neg["framework_only"]["penalty"])
        reasons.append("LangChain listed with no pre-LLM-era production ML evidence (framework-only pattern)")

    cv_speech = {"Computer Vision", "Speech Recognition", "ASR", "TTS", "Object Detection",
                 "Image Classification", "CNN", "OpenCV", "YOLO"}
    nlp_ir = {"NLP", "Information Retrieval", "BM25", "Semantic Search", "Embeddings",
              "Sentence Transformers", "Vector Search"}
    # Iteration 2 (see experiments/exp_log.md): originally this checked skill_names &
    # nlp_ir, i.e. presence only. The ablation study found that decoy candidates
    # (e.g. a Computer Vision Engineer who also lists Embeddings/Semantic Search as
    # bait-tier tags, with no real depth) slip through, because tag *presence*
    # doesn't distinguish "has genuinely worked on retrieval" from "has the same
    # keyword-stuffed tag everyone in the bait tier has." We now require at least
    # one nlp_ir skill to clear a claim-strength bar before it counts as real
    # cross-disciplinary evidence.
    assessment_scores = candidate["redrob_signals"]["skill_assessment_scores"]
    real_nlp_ir_evidence = any(
        s["name"] in nlp_ir and claim_strength(s, assessment_scores) >= 0.4
        for s in candidate["skills"]
    )
    if (skill_names & cv_speech) and not real_nlp_ir_evidence and \
       candidate["profile"]["current_title"] in {"Computer Vision Engineer"}:
        mult *= (1 - neg["cv_speech_without_nlp_ir"]["penalty"])
        reasons.append("CV/speech background without NLP/IR exposure (JD explicitly deprioritizes this profile)")

    history = candidate["career_history"]
    if len(history) >= 3:
        avg_tenure = sum(h["duration_months"] for h in history) / len(history)
        if avg_tenure < 18:
            mult *= (1 - neg["title_chaser"]["penalty"])
            reasons.append(f"{len(history)} employers averaging {avg_tenure:.0f} months tenure (title-chaser pattern)")

    if candidate["profile"]["current_title"] == "AI Research Engineer":
        # JD: "pure research environments ... without any production deployment -- we will not move forward"
        # we can't see "academic lab" directly, but a pure-research title + no production-scale
        # signals (no vector-db/infra skill, no current_company_size >= 201-500) is the closest proxy.
        has_infra_signal = bool(skill_names & {"FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
                                                "OpenSearch", "Elasticsearch", "Kubernetes", "Docker"})
        if not has_infra_signal:
            mult *= 0.5
            reasons.append("AI Research Engineer title with no production/deployment infra skills on record")

    return mult, reasons


def behavior_score(candidate):
    sig = candidate["redrob_signals"]
    last_active = datetime.strptime(sig["last_active_date"], "%Y-%m-%d")
    days_inactive = (TODAY - last_active).days
    recency = max(0.0, 1.0 - days_inactive / 180.0)  # linear decay to 0 at 6 months inactive

    response = sig["recruiter_response_rate"]
    interview_completion = sig["interview_completion_rate"]
    offer_accept = sig["offer_acceptance_rate"]
    offer_component = 0.5 if offer_accept < 0 else offer_accept  # -1 sentinel = no offer history yet

    demand = min((sig["search_appearance_30d"] / 200.0 + sig["saved_by_recruiters_30d"] / 10.0) / 2, 1.0)

    return float(0.30 * recency + 0.25 * response + 0.20 * interview_completion +
                 0.10 * offer_component + 0.15 * demand)


def availability_score(candidate, job_repr):
    sig = candidate["redrob_signals"]
    if not sig["open_to_work_flag"]:
        return 0.15  # not zero -- a passive candidate can still be worth surfacing, just unlikely to respond

    notice = sig["notice_period_days"]
    notice_curve = job_repr["notice_period_curve"]
    if notice <= 30:
        notice_score = notice_curve["<=30"]
    elif notice <= 60:
        notice_score = notice_curve["31-60"]
    elif notice <= 90:
        notice_score = notice_curve["61-90"]
    else:
        notice_score = notice_curve["91-180"]

    loc = candidate["profile"]["location"]
    country = candidate["profile"]["country"]
    loc_pref = job_repr["location_preference"]
    if any(p.lower() in loc.lower() for p in loc_pref["preferred"]):
        loc_score = 1.0
    elif country == loc_pref["country_required_soft"]:
        loc_score = 0.7
    elif sig["willing_to_relocate"]:
        loc_score = 0.5
    else:
        loc_score = 0.2

    return float(0.5 * notice_score + 0.5 * loc_score)


def trust_score(candidate, honeypot_probability):
    sig = candidate["redrob_signals"]
    completeness = sig["profile_completeness_score"] / 100.0
    verification = (int(sig["verified_email"]) + int(sig["verified_phone"]) +
                     int(sig["linkedin_connected"])) / 3.0
    base = 0.5 * completeness + 0.5 * verification
    return float(base * (1 - honeypot_probability))


def compute_feature_dict(candidate, job_repr, semantic_sim, bm25_score_normalized,
                          honeypot_probability, is_service_only_fn):
    """One row of the feature table used for both the rule score and the
    learned reranker. Keeping this as a flat dict (not nested) is on purpose
    -- it goes straight into a pandas DataFrame for LightGBM/GBR training,
    and a flat schema is easier to diff between experiment runs."""
    skill_match, matched_required, matched_preferred = skill_score(candidate, job_repr)
    title_score, title_penalty_note = title_tier_score(candidate, job_repr)
    company_score = company_quality_score(candidate, job_repr)
    progression = career_progression_score(candidate)
    exp_band = experience_band_fit(candidate, job_repr)
    neg_mult, neg_reasons = negative_signal_multiplier(candidate, job_repr, is_service_only_fn)
    behavior = behavior_score(candidate)
    availability = availability_score(candidate, job_repr)
    trust = trust_score(candidate, honeypot_probability)
    eval_fit = evaluation_fit_score(candidate)

    experience_fit = (0.35 * title_score + 0.20 * progression +
                       0.15 * company_score + 0.30 * exp_band)

    return {
        "candidate_id": candidate["candidate_id"],
        "semantic": semantic_sim,
        "retrieval_skill_match": skill_match,
        "bm25_score": bm25_score_normalized,
        "experience_fit": experience_fit,
        "title_score": title_score,
        "company_score": company_score,
        "progression_score": progression,
        "experience_band_fit": exp_band,
        "behavior": behavior,
        "evaluation_fit": eval_fit,
        "trust": trust,
        "availability": availability,
        "negative_multiplier": neg_mult,
        "honeypot_probability": honeypot_probability,
        "years_of_experience": candidate["profile"]["years_of_experience"],
        "matched_required_skills": matched_required,
        "matched_preferred_skills": matched_preferred,
        "negative_reasons": neg_reasons,
        "title_penalty_note": title_penalty_note,
    }


def rule_based_composite(feat, job_repr):
    """The transparent score: a fixed weighted sum (weights come from
    weighted_job_representation.json, matching the brief's formula) scaled
    by the negative-signal multiplier and the honeypot penalty. This is what
    we'd defend in a live walkthrough -- every term traces to a named field
    in the candidate record, no black box.

    Honeypot handling: rather than zeroing honeypots outright (which would
    make their rank arbitrary and let ties decide it, including possibly
    near the top by accident), we apply a steep penalty that g