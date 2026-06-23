"""
Learned reranker: gradient-boosted trees over the engineered feature table,
trained to predict the heuristic relevance tier (0-4) from pseudo_labels.py.

Model choice: we wanted LightGBM's LambdaMART objective (lgb.LGBMRanker,
objective="lambdarank"), since it directly optimizes NDCG and is what the
brief asks for. In practice, our dev sandbox's outbound network to PyPI was
unreliable (proxy timeouts, not a deliberate choice), so we wrote this with
a fallback: try LightGBM first, and if it isn't importable, fall back to
sklearn's GradientBoostingRegressor trained on the tier as a continuous
target. We kept the fallback in the shipped code (not just as a dev hack)
because Stage 3 reproduction happens in a sandbox we don't control, and a
pip install that silently fails on an unfamiliar machine is exactly the
kind of failure mode the compute-constraints section warns about. If
LightGBM is available, it's used; the code logs which path actually ran so
metrics_report.md's numbers are traceable to a specific model, not a
guess.

Update: the fallback originally only triggered on ImportError (lightgbm not
installed at all). Reproducing on a clean Windows/Python 3.14 machine
surfaced a second failure mode in the same risk category -- lightgbm==4.3.0
imports fine but crashes inside model.fit() under numpy>=2.0, because its
internal `_list_to_1d_numpy` calls `np.array(data, dtype=dtype,
copy=False)`, and numpy 2.0 changed `copy=False` from "copy only if
needed" to "never copy, raise if a copy is required." That's exactly the
"pip install succeeds but the library doesn't actually work here" case this
fallback exists for, so we widened the try/except to cover a failed fit,
not just a failed import, instead of pinning numpy<2.0 (which has no
Python 3.14 wheel and isn't a real fix for the underlying lightgbm/numpy
incompatibility anyway).

This is a single-query ranking problem (one JD, ~18.7K candidates) -- there
is no cross-query structure to learn from, unlike typical LTR benchmarks
(MSLR-WEB30K, Yahoo LTR) with thousands of queries. LambdaMART's group
mechanism still works with one group, but "generalization" here means
"generalizes across held-out candidates for this JD," not "generalizes
across JDs." We say this again in metrics_report.md because it's easy to
read a cross-validated NDCG number as more than that.
"""

import logging

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "semantic", "retrieval_skill_match", "bm25_score", "experience_fit",
    "title_score", "company_score", "progression_score", "experience_band_fit",
    "behavior", "evaluation_fit", "trust", "availability",
    "negative_multiplier", "honeypot_probability", "years_of_experience",
]


def _try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        return None


def _fit_sklearn_gbr(X, y, random_state):
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=random_state,
    )
    model.fit(X, y)
    return model, "sklearn-gbr"


def train_model(X, y, random_state=42):
    lgb = _try_import_lightgbm()
    if lgb is None:
        logger.warning("lightgbm not importable -- falling back to sklearn GradientBoostingRegressor. "
                        "See src/train_ranker.py module docstring for why this fallback exists.")
        return _fit_sklearn_gbr(X, y, random_state)

    try:
        model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=200, num_leaves=15,
            learning_rate=0.05, min_child_samples=10, random_state=random_state,
        )
        model.fit(X, y, group=[len(y)])
        return model, "lightgbm-lambdamart"
    except Exception as exc:
        logger.warning("lightgbm imported but failed at fit time (%r) -- falling back to sklearn "
                        "GradientBoostingRegressor. Known cause: lightgbm<4.4 + numpy>=2.0 group-array "
                        "construction. See src/train_ranker.py module docstring.", exc)
        return _fit_sklearn_gbr(X, y, random_state)


def cross_validate(feature_df, labels, n_splits=5, random_state=42):
    """K-fold over candidates (see module docstring for why this is the
    only valid notion of CV with a single query). Returns per-fold and
    aggregate NDCG@10 / NDCG@50 / MAP / P@10 against the pseudo labels."""
    from .metrics import ndcg_at_k, average_precision, precision_at_k

    X = feature_df[FEATURE_COLUMNS].values
    y = np.array(labels)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        model, backend = train_model(X[train_idx], y[train_idx], random_state=random_state)
        preds = model.predict(X[test_idx])
        test_labels = y[test_idx]

        order = np.argsort(-preds)
        ranked_labels = test_labels[order]

        fold_results.append({
            "fold": fold_i,
            "backend": backend,
            "ndcg@10": ndcg_at_k(ranked_labels, 10),
            "ndcg@50": ndcg_at_k(ranked_labels, 50),
            "map": average_precision(ranked_labels, relevance_threshold=3),
            "p@10": precision_at_k(ranked_labels, 10, relevance_threshold=3),
        })
    return fold_results
