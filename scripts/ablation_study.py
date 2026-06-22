#!/usr/bin/env python3
"""
Ablation study: re-run the stage-3 feature/scoring step under modified
conditions and measure ranking quality against a FIXED set of pseudo-labels
(generated once, from the normal/full pipeline) so every variant is judged
against the same target ordering. Writes ablation_results.json, consumed by
metrics_report.md.

Variants:
  full_ensemble        - what rank.py actually ships (rule score blended
                          50/50 with the learned reranker)
  rule_only             - rule-based composite score alone, no learned model
  model_only            - learned reranker score alone (5-fold out-of-fold
                          predictions, so this isn't just memorizing the
                          training set)
  no_honeypot_detection - honeypot_probability forced to 0 for everyone
  flat_skill_weighting  - every skill counts equally (category weight = 1.0
                          for everything), to reproduce the keyword-stuffing
                          failure mode we found during development
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import load_candidates
from src.honeypot import score_honeypot, is_service_only
from src.semantic import SemanticIndex, candidate_text_blob, job_text_blob
from src.retrieval import stage1_bm25_filter, stage2_dense_filter
from src import features as feat_mod
from src.pseudo_labels import label_dataset
from src.train_ranker import train_model, FEATURE_COLUMNS
from src.metrics import ndcg_at_k, average_precision, precision_at_k
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).parent.parent


def normalize(values):
    values = np.array(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def rank_quality(scores, tiers, k10=10, k50=50):
    order = np.argsort(-np.array(scores))
    ranked = np.array(tiers)[order]
    return {
        "ndcg@10": ndcg_at_k(ranked, k10),
        "ndcg@50": ndcg_at_k(ranked, k50),
        "map": average_precision(ranked, relevance_threshold=3),
        "p@10": precision_at_k(ranked, 10, relevance_threshold=3),
    }


def main():
    job_repr = json.loads((REPO_ROOT / "weighted_job_representation.json").read_text())
    jd_text = (REPO_ROOT / "docs" / "job_description.txt").read_text()
    query_text = job_text_blob(jd_text, job_repr)

    candidates, _ = load_candidates(str(REPO_ROOT / "candidates.jsonl"))

    keep_idx_1000, bm25_scores_all = stage1_bm25_filter(candidates, query_text, keep_top=1000)
    all_texts = [candidate_text_blob(c) for c in candidates]
    semantic_index = SemanticIndex(n_components=128, random_state=42)
    semantic_index.fit(all_texts + [query_text])
    query_vector = semantic_index.transform([query_text])[0]
    keep_idx_300, sim_by_idx = stage2_dense_filter(
        candidates, keep_idx_1000, semantic_index, query_vector, keep_top=300)
    bm25_norm = normalize(bm25_scores_all)

    survivors = [candidates[i] for i in keep_idx_300]
    sims = [sim_by_idx[i] for i in keep_idx_300]
    bm25s = [float(bm25_norm[i]) for i in keep_idx_300]

    # ---- baseline (full/normal) feature computation -> fixed pseudo-labels ----
    def compute_rows(skill_weight_override=None, force_honeypot_zero=False):
        rows = []
        orig_weights = dict(feat_mod.SKILL_CATEGORY_WEIGHT)
        if skill_weight_override is not None:
            for k in feat_mod.SKILL_CATEGORY_WEIGHT:
                feat_mod.SKILL_CATEGORY_WEIGHT[k] = skill_weight_override
        try:
            for c, sim, bm25 in zip(survivors, sims, bm25s):
                hp_prob, _ = score_honeypot(c)
                if force_honeypot_zero:
                    hp_prob = 0.0
                row = feat_mod.compute_feature_dict(
                    c, job_repr, semantic_sim=sim, bm25_score_normalized=bm25,
                    honeypot_probability=hp_prob, is_service_only_fn=is_service_only)
                row["rule_score"] = feat_mod.rule_based_composite(row, job_repr)
                rows.append(row)
        finally:
            feat_mod.SKILL_CATEGORY_WEIGHT.clear()
            feat_mod.SKILL_CATEGORY_WEIGHT.update(orig_weights)
        return rows

    baseline_rows = compute_rows()
    baseline_tiers = label_dataset(baseline_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    results = {}

    # full_ensemble: rule score + out-of-fold learned model, 50/50
    X = baseline_df[FEATURE_COLUMNS].values
    y = np.array(baseline_tiers)
    oof_pred = np.zeros(len(y))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    backend_used = None
    for train_idx, test_idx in kf.split(X):
        model, backend_used = train_model(X[train_idx], y[train_idx], random_state=42)
        oof_pred[test_idx] = model.predict(X[test_idx])
    model_score_norm = normalize(oof_pred)
    ensemble_score = 0.5 * baseline_df["rule_score"].values + 0.5 * model_score_norm

    results["full_ensemble"] = rank_quality(ensemble_score, baseline_tiers)
    results["rule_only"] = rank_quality(baseline_df["rule_score"].values, baseline_tiers)
    results["model_only"] = rank_quality(model_score_norm, baseline_tiers)
    results["full_ensemble"]["backend"] = backend_used
    results["model_only"]["backend"] = backend_used

    # no_honeypot_detection: recompute scores with honeypot forced to 0,
    # evaluate against the SAME baseline_tiers (which do reflect real honeypot
    # status), so this measures "how many honeypots would we have ranked highly
    # if we hadn't built the detector"
    no_hp_rows = compute_rows(force_honeypot_zero=True)
    no_hp_scores = [r["rule_score"] for r in no_hp_rows]
    no_hp_order = np.argsort(-np.array(no_hp_scores))
    no_hp_top10_honeypot_count = sum(
        1 for i in no_hp_order[:10] if baseline_rows[i]["honeypot_probability"] >= 0.5)
    no_hp_top100_honeypot_count = sum(
        1 for i in no_hp_order[:100] if baseline_rows[i]["honeypot_probability"] >= 0.5)
    results["no_honeypot_detection"] = rank_quality(no_hp_scores, baseline_tiers)
    results["no_honeypot_detection"]["honeypots_in_top10"] = no_hp_top10_honeypot_count
    results["no_honeypot_detection"]["honeypots_in_top100"] = no_hp_top100_honeypot_count

    # compare: how many honeypots are in top10/top100 with detection ON (the real pipeline)
    real_order = np.argsort(-baseline_df["rule_score"].values)
    real_top10_hp = sum(1 for i in real_order[:10] if baseline_rows[i]["honeypot_probability"] >= 0.5)
    real_top100_hp = sum(1 for i in real_order[:100] if baseline_rows[i]["honeypot_probability"] >= 0.5)
    results["with_honeypot_detection"] = {
        "honeypots_in_top10": real_top10_hp,
        "honeypots_in_top100": real_top100_hp,
    }

    # flat_skill_weighting: every skill category weight = 1.0
    flat_rows = compute_rows(skill_weight_override=1.0)
    flat_scores = [r["rule_score"] for r in flat_rows]
    results["flat_skill_weighting"] = rank_quality(flat_scores, baseline_tiers)

    # decoy check: Mechanical Engineer / unrelated-title candidate with stuffed
    # skills -- did flat weighting let them rank higher than under category weighting?
    decoy_ids = [c["candidate_id"] for c in survivors
                 if c["profile"]["current_title"] not in
                 job_repr["title_tiers"]["tier_5_exact"] + job_repr["title_tiers"]["tier_4_close"]
                 and c["profile"]["current_title"] not in job_repr["title_tiers"]["tier_3_adjacent"]]
    decoy_rank_baseline = []
    decoy_rank_flat = []
    base_order_ids = [survivors[i]["candidate_id"] for i in np.argsort(-baseline_df["rule_score"].values)]
    flat_order_ids = [survivors[i]["candidate_id"] for i in np.argsort(-np.array(flat_scores))]
    for cid in decoy_ids:
        if cid in base_order_ids:
            decoy_rank_baseline.append(base_order_ids.index(cid) + 1)
        if cid in flat_order_ids:
            decoy_rank_flat.append(flat_order_ids.index(cid) + 1)
    results["decoy_title_check"] = {
        "n_unrelated_title_candidates_in_stage2_pool": len(decoy_ids),
        "best_rank_under_category_weighting": min(decoy_rank_baseline) if decoy_rank_baseline else None,
        "best_rank_under_flat_weighting": min(decoy_rank_flat) if decoy_rank_flat else None,
    }

    out_path = REPO_ROOT / "ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
