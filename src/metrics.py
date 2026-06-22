"""
NDCG@k, MAP, P@k against an ordered list of relevance labels (already
sorted by predicted score -- these functions just measure quality of that
ordering, they don't sort anything themselves).

Implemented directly instead of pulling in a metrics library because the
challenge's exact NDCG definition (graded relevance 0-4, standard
log2-discount gain) is simple enough that a dependency would cost more in
"does their library define NDCG the same way we think it does" risk than
it saves in code.
"""

import math


def dcg_at_k(relevances, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_relevances, k):
    actual_dcg = dcg_at_k(ranked_relevances, k)
    ideal = sorted(ranked_relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(ranked_relevances, k, relevance_threshold=3):
    top_k = list(ranked_relevances[:k])
    if len(top_k) == 0:
        return 0.0
    relevant = sum(1 for r in top_k if r >= relevance_threshold)
    return relevant / len(top_k)


def average_precision(ranked_relevances, relevance_threshold=3):
    """MAP for a single ranked list: precision averaged at each rank where
    a relevant item appears."""
    relevant_seen = 0
    precisions = []
    for i, rel in enumerate(ranked_relevances, start=1):
        if rel >= relevance_threshold:
            relevant_seen += 1
            precisions.append(relevant_seen / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)
