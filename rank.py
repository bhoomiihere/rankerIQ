#!/usr/bin/env python3
"""
Entry point: python rank.py --candidates candidates.jsonl --out submission.csv

End-to-end funnel: BM25 (stage 1) -> TF-IDF/SVD dense similarity (stage 2)
-> full feature scoring + honeypot detection (stage 3) -> learned reranker
+ rule score ensemble, final top-100 (stage 4). See README.md for the
runtime/memory budget this was built against and docs/architecture.md for
why each stage exists.

Everything before the BM25 call is just I/O. Everything after the final
sort is just CSV formatting. The interesting work is in src/.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.ingest import load_candidates
from src.honeypot import score_honeypot, is_service_only
from src.semantic import SemanticIndex, candidate_text_blob, job_text_blob
from src.retrieval import stage1_bm25_filter, stage2_dense_filter
from src.features import compute_feature_dict, rule_based_composite
from src.pseudo_labels import label_dataset
from src.train_ranker import train_model, cross_validate, FEATURE_COLUMNS
from src.reasoning import generate_reasoning

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rank")

REPO_ROOT = Path(__file__).parent


def normalize(values):
    values = np.array(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output submission CSV path")
    parser.add_argument("--job-repr", default=str(REPO_ROOT / "weighted_job_representation.json"))
    parser.add_argument("--job-text", default=str(REPO_ROOT / "docs" / "job_description.txt"))
    parser.add_argument("--stage1-k", type=int, default=1000)
    parser.add_argument("--stage2-k", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--metrics-out", default=str(REPO_ROOT / "metrics_report_data.json"),
                         help="Intermediate metrics JSON consumed by scripts/build_metrics_report.py")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()

    job_repr = json.loads(Path(args.job_repr).read_text())
    jd_text = Path(args.job_text).read_text()
    query_text = job_text_blob(jd_text, job_repr)

    logger.info("loading candidates from %s", args.candidates)
    candidates, n_skipped = load_candidates(args.candidates)
    logger.info("loaded %d candidates (%d skipped as malformed)", len(candidates), len(n_skipped))

    # ---- Stage 1: BM25 sparse filter ----
    t0 = time.time()
    keep_idx_1000, bm25_scores_all = stage1_bm25_filter(candidates, query_text, keep_top=args.stage1_k)
    logger.info("stage 1 (BM25): kept %d / %d candidates in %.1fs",
                len(keep_idx_1000), len(candidates), time.time() - t0)

    # ---- Stage 2: dense semantic filter ----
    # Fit TF-IDF/SVD on the full corpus once (representative vocabulary + IDF + SVD basis),
    # then only pay the transform cost on the stage-1 survivors -- see src/retrieval.py docstring.
    t0 = time.time()
    all_texts = [candidate_text_blob(c) for c in candidates]
    semantic_index = SemanticIndex(n_components=128, random_state=args.seed)
    semantic_index.fit(all_texts + [query_text])
    query_vector = semantic_index.transform([query_text])[0]
    logger.info("semantic index fit in %.1fs", time.time() - t0)

    t0 = time.time()
    keep_idx_300, sim_by_idx = stage2_dense_filter(
        candidates, keep_idx_1000, semantic_index, query_vector, keep_top=args.stage2_k)
    logger.info("stage 2 (dense): kept %d / %d candidates in %.1fs",
                len(keep_idx_300), len(keep_idx_1000), time.time() - t0)

    # ---- Stage 3: full feature scoring (honeypot + skill/title/behavior/trust/...) ----
    t0 = time.time()
    bm25_norm = normalize(bm25_scores_all)
    feature_rows, candidate_by_id = [], {}
    for idx in keep_idx_300:
        c = candidates[idx]
        candidate_by_id[c["candidate_id"]] = c
        hp_prob, hp_reasons = score_honeypot(c)
        feat = compute_feature_dict(
            c, job_repr,
            semantic_sim=sim_by_idx[idx],
            bm25_score_normalized=float(bm25_norm[idx]),
            honeypot_probability=hp_prob,
            is_service_only_fn=is_service_only,
        )
        feat["honeypot_reasons"] = hp_reasons
        feat["rule_score"] = rule_based_composite(feat, job_repr)
        feature_rows.append(feat)
    logger.info("stage 3 (feature scoring): scored %d candidates in %.1fs", len(feature_rows), time.time() - t0)

    feature_df = pd.DataFrame(feature_rows)
    tiers = label_dataset(feature_rows)
    feature_df["pseudo_tier"] = tiers

    # ---- Stage 4: learned reranker + ensemble, final top-k ----
    t0 = time.time()
    cv_results = cross_validate(feature_df, tiers, n_splits=5, random_state=args.seed)
    model, backend = train_model(feature_df[FEATURE_COLUMNS].values, np.array(tiers), random_state=args.seed)
    model_scores = model.predict(feature_df[FEATURE_COLUMNS].values)
    model_scores_norm = normalize(model_scores)

    feature_df["model_score"] = model_scores_norm
    feature_df["final_score"] = 0.5 * feature_df["rule_score"] + 0.5 * feature_df["model_score"]
    logger.info("stage 4 (rerank, backend=%s): trained + scored in %.1fs", backend, time.time() - t0)

    # BUG FIX: sorting on the full-precision final_score but writing a 6-decimal
    # rounded score to the CSV meant two rows that were merely *close* (not exactly
    # tied) pre-rounding could land adjacent with the wrong candidate_id order once
    # rounded -- validate_submission.py caught this ("Equal scores at ranks 61/62,
    # tie-break requires candidate_id ascending"). Round first, then sort on the
    # rounded value, so the tie-break is applied at the same precision we output.
    feature_df["final_score_rounded"] = feature_df["final_score"].round(6)
    ranked = feature_df.sort_values(
        ["final_score_rounded", "candidate_id"], ascending=[False, True]
    ).head(args.top_k).reset_index(drop=True)

    # Tie-break and monotonicity: validator requires score non-increasing by rank and,
    # on exact ties, candidate_id ascending. We sort that way already; to guarantee strict
    # monotonicity isn't *required* (ties are explicitly allowed by the spec) we leave equal
    # scores as-is rather than injecting artificial jitter that would misrepresent confidence.
    rows_out = []
    for rank, row in enumerate(ranked.itertuples(), start=1):
        candidate = candidate_by_id[row.candidate_id]
        feat_row = feature_df[feature_df.candidate_id == row.candidate_id].iloc[0].to_dict()
        reasoning = generate_reasoning(candidate, feat_row)
        rows_out.append({
            "candidate_id": row.candidate_id,
            "rank": rank,
            "score": round(float(row.final_score), 6),
            "reasoning": reasoning,
        })

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(out_df))

    metrics_payload = {
        "backend": backend,
        "n_candidates_total": len(candidates),
        "n_skipped_malformed": n_skipped,
        "stage1_kept": len(keep_idx_1000),
        "stage2_kept": len(keep_idx_300),
        "top_k": args.top_k,
        "cv_results": cv_results,
        "honeypot_count_in_stage2": int((feature_df["honeypot_probability"] >= 0.5).sum()),
        "honeypot_count_in_top_k": int((ranked["honeypot_probability"] >= 0.5).sum()),
        "tier_distribution_stage2": pd.Series(tiers).value_counts().sort_index().to_dict(),
        "wall_clock_seconds": round(time.time() - t_start, 1),
    }
    Path(args.metrics_out).write_text(json.dumps(metrics_payload, indent=2, default=str))
    logger.info("wrote %s", args.metrics_out)
    logger.info("total wall clock: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
