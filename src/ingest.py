"""
Load candidates.jsonl into plain Python dicts.

Why not pandas.read_json(lines=True) straight away: we hit a corrupt last
line in the released dataset (CAND_0018746's record is truncated mid-string -
looks like a copy/upload artifact, not an intentional honeypot, since it
fails on a dangling string rather than a semantically odd but valid record).
pandas' line reader throws on the first bad line and stops, which would
silently drop the whole file in a pipeline run. We parse line-by-line
instead so one bad record costs us one candidate, not all 18,745.
"""

import json
import logging

logger = logging.getLogger(__name__)


def load_candidates(path):
    """Returns (candidates: list[dict], skipped: list[int]) where skipped
    holds the 1-indexed line numbers that failed to parse."""
    candidates = []
    skipped = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                skipped.append(line_no)
    if skipped:
        logger.warning(
            "Skipped %d malformed line(s) in %s: %s",
            len(skipped), path, skipped[:10],
        )
    return candidates, skipped


def candidate_index(candidates):
    """candidate_id -> record, for O(1) lookup during reasoning generation."""
    return {c["candidate_id"]: c for c in candidates}
