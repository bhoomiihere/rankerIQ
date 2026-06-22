# CPU-only by design -- the compute constraints section of the brief rules out
# GPU inference, so there's no reason to ship a CUDA base image and the
# multi-hundred-MB pull time that comes with it.
FROM python:3.11-slim

WORKDIR /app

# System deps for python-docx (only used by scripts/extract_job_description.py,
# kept minimal since rank.py itself doesn't need it at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default invocation matches the brief's required one-command execution.
# Override --candidates / --out at `docker run` time if needed:
#   docker run -v $(pwd)/data:/app/data redrob-ranker \
#       python rank.py --candidates /app/data/candidates.jsonl --out /app/data/submission.csv
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "candidates.jsonl", "--out", "submission.csv"]
