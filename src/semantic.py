"""
Dense semantic similarity stage.

We tried sentence-transformers (all-MiniLM-L6-v2) first since that's the
production-standard choice and what the JD itself names. We dropped it for
this submission for three concrete reasons, not just "it's heavier":

1. Torch + the model weights add ~500MB to the dependency footprint. Stage 3
   reproduces our ranking step in an organizer-controlled sandbox we don't
   control and can't pre-warm; a dependency that fails to install (or pulls
   a model from the internet, which the rules forbid at ranking time) is a
   disqualification risk, not a quality risk. A TF-IDF+SVD pipeline has zero
   external downloads and is pure scikit-learn, which is already a
   requirement-stable dependency.
2. We measured this, not just assumed it (see experiments/exp_log.md,
   "iteration 4"): on our heuristic validation labels, swapping in
   MiniLM embeddings changed NDCG@10 by +0.012, inside the noise band we
   got from changing the pseudo-label thresholds by one notch. Given the
   ranking signal here is dominated by structured fields (title tier, skill
   depth, honeypot status) rather than paraphrase-level semantics, the
   extra dependency wasn't paying for itself.
3. Latency: TF-IDF+SVD over 18,745 candidates fits comfortably inside the
   5-minute budget with room to spare for the rest of the pipeline (BM25,
   feature engineering, LightGBM/GBR inference). MiniLM inference for the
   same set took ~40x longer on the 2-core box we built this on, and we'd
   rather spend that latency budget on candidates than on the embedding step.

We're explicit in challenge_analysis.md and the README that this is a
documented tradeoff, not an oversight -- a team with a GPU-backed Stage 3
sandbox guarantee would reasonably make the other call.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

_STOPWORDS_EXTRA = {"experience", "years", "year", "working", "worked", "work", "role", "team"}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    return text


def candidate_text_blob(candidate):
    parts = [
        candidate["profile"]["headline"],
        candidate["profile"]["summary"],
        candidate["profile"]["current_title"],
    ]
    for role in candidate["career_history"]:
        parts.append(role["title"])
        parts.append(role["description"])
    parts.extend(s["name"] for s in candidate["skills"])
    return clean_text(" ".join(parts))


def job_text_blob(job_description_text, job_repr):
    parts = [job_description_text]
    for group in job_repr["required_skills"].values():
        if isinstance(group, list):
            parts.extend(group)
    for group in job_repr["preferred_skills"].values():
        if isinstance(group, list):
            parts.extend(group)
    return clean_text(" ".join(parts))


class SemanticIndex:
    """TF-IDF -> SVD (LSA). Fit once over the candidate corpus + JD text,
    cache to disk with joblib so re-running rank.py doesn't refit every time
    (refit only happens if candidates.jsonl changes -- see rank.py cache key)."""

    def __init__(self, n_components=128, random_state=42):
        self.vectorizer = TfidfVectorizer(
            max_features=20000, min_df=2, max_df=0.6, ngram_range=(1, 2),
            sublinear_tf=True, stop_words=list(_STOPWORDS_EXTRA),
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, texts):
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def transform(self, texts):
        tfidf = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf)

    def similarity_to_query(self, candidate_vectors, query_vector):
        return cosine_similarity(candidate_vectors, query_vector.reshape(1, -1)).ravel()
