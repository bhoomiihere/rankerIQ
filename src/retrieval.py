"""
Multi-stage retrieval funnel: BM25 (sparse) -> TF-IDF/SVD (dense) -> full
feature scoring -> final rerank.

At 18,745 candidates this funnel is not strictly necessary for correctness
-- we could feature-score everyone in the time budget. We built it as a
real funnel anyway (not just for show) because:
  (a) the JD explicitly describes a 100K+ candidate pool, where scoring
      every candidate with the full feature pipeline (career-history
      parsing, honeypot checks, skill depth scoring) would not fit 5
      minutes on CPU, so the architecture has to assume a much larger pool
      than this dataset gives us, and
  (b) it gave us a free sanity check during development: if a candidate we
      know is a strong fit gets cut at stage 1, that's a bug in the BM25
      query construction, not the reranker -- worth catching early rather
      than burying it inside one big scoring function.
"""

from rank_bm25 import BM25Okapi

from .semantic import candidate_text_blob, clean_text


def stage1_bm25_filter(candidates, query_text, keep_top=1000):
    """Sparse lexical filter. Cheap, no model loading, scales to the full
    pool. Fast indicator of 'does this profile use any vocabulary remotely
    related to the JD' before we spend cycles on anything heavier."""
    corpus_tokens = [candidate_text_blob(c).split() for c in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = clean_text(query_text).split()
    scores = bm25.get_scores(query_tokens)
    ranked_idx = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    keep_idx = ranked_idx[:keep_top]
    return keep_idx, scores


def stage2_dense_filter(candidates, idx_subset, semantic_index, query_vector, keep_top=300):
    """Dense semantic filter on the stage-1 survivors only -- this is where
    TF-IDF/SVD transform cost actually matters, so we only pay it for 1000
    candidates, not 18,745 or (in the real pool) 100K+."""
    texts = [candidate_text_blob(candidates[i]) for i in idx_subset]
    vectors = semantic_index.transform(texts)
    sims = semantic_index.similarity_to_query(vectors, query_vector)
    order = sorted(range(len(idx_subset)), key=lambda j: sims[j], reverse=True)
    keep_local = order[:keep_top]
    keep_idx = [idx_subset[j] for j in keep_local]
    sim_by_idx = {idx_subset[j]: float(sims[j]) for j in range(len(idx_subset))}
    return keep_idx, sim_by_idx
