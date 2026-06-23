"""
Sandbox demo for the submission portal's mandatory sandbox/demo link
(submission_spec.docx, Section 10.2). Wraps the real rank.py CLI -- this
calls the exact same code path as the actual submission, not a simplified
or mocked version, so the demo and the real pipeline can't drift apart.

Deploy: push this repo to GitHub, then on share.streamlit.io -> New app ->
point at this file. No GPU/network needed at runtime, matches the
compute-constraints section of the spec.
"""
import os
import subprocess
import tempfile

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Redrob Candidate Ranker -- Team Bk", layout="wide")
st.title("Redrob Candidate Ranker -- Team Bk")
st.write(
    "Demo of the actual `rank.py` pipeline: BM25 -> TF-IDF/SVD dense filter -> "
    "rule-based feature scoring + honeypot detection -> learned reranker. "
    "Upload a `candidates.jsonl`, or use the bundled 200-row sample, and run "
    "the real ranking code below."
)

uploaded = st.file_uploader("candidates.jsonl (optional -- leave empty to use the bundled sample)",
                             type=["jsonl"])

if st.button("Run ranking", type="primary"):
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "submission.csv")
        if uploaded is not None:
            in_path = os.path.join(tmp, "candidates.jsonl")
            with open(in_path, "wb") as f:
                f.write(uploaded.read())
        else:
            in_path = "sample_candidates.jsonl"

        with st.spinner("Running rank.py end to end (BM25 -> dense filter -> feature scoring -> rerank)..."):
            result = subprocess.run(
                ["python", "rank.py", "--candidates", in_path, "--out", out_path],
                capture_output=True, text=True, timeout=280,
            )

        if result.returncode != 0:
            st.error("rank.py failed:")
            st.code(result.stderr[-3000:])
        else:
            df = pd.read_csv(out_path)
            st.success(f"Ranked {len(df)} candidates in this run.")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download submission.csv", df.to_csv(index=False),
                                file_name="submission.csv", mime="text/csv")

            with st.expander("Pipeline log"):
                st.code(result.stderr)

st.caption(
    "Source, methodology writeup, ablation results, and the honest "
    "experiment log (including rejected approaches) are in the GitHub repo "
    "linked in this submission's metadata."
)
