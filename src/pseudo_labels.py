"""
Heuristic relevance tiers (0-4) used as weak supervision for the learned
reranker, and as the labels behind every NDCG/MAP number in
metrics_report.md.

Read this before trusting any metric in this repo: there is no hidden
ground truth available to us before submission. These tiers are *our*
judgment about what the JD rewards, encoded as rules, not a leaked or
inferred version of Redrob's actual grading. Cross-validation against these
labels checks whether our feature pipeline and learned model are internally
consistent (do they agree with our own stated logic, can the model recover
signal from the features) -- it does not estimate hidden-leaderboard
accuracy, and we say so everywhere this number appears.

The reason to do this at all, instead of skipping straight to the rule
score: it gives us a concrete way to ablate features and sanity-check
composite weights before the one submission we get to make counts. A model
that can't beat a trivial baseline on labels we wrote ourselves would be a
red flag about the feature pipeline, not just the labels.
"""


def assign_tier(feat):
    """0 = honeypot or clearly irrelevant, 4 = ideal-candidate match per the
    JD's own 'how to read between the lines' section."""
    if feat["honeypot_probability"] >= 0.5:
        return 0

    title = feat["title_score"]
    skill = feat["retrieval_skill_match"]
    neg = feat["negative_multiplier"]
    exp = feat["experience_band_fit"]

    if title == 0.05 and skill < 0.1:
        return 0  # unrelated title, no real skill evidence -- not a honeypot, just not a fit

    if title >= 0.75 and skill >= 0.45 and neg >= 0.8 and exp >= 0.4:
        return 4
    if title >= 0.45 and skill >= 0.28 and neg >= 0.6:
        return 3
    if title >= 0.2 or skill >= 0.15:
        return 2
    return 1


def label_dataset(feature_rows):
    return [assign_tier(f) for f in feature_rows]
