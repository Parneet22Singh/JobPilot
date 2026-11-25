"""Load sample jobs:  python -m scripts.seed"""

import json
from pathlib import Path

from app import config
from app.embeddings import get_embedder
from app.search import ensure_embeddings
from app.store import Store

store = Store(config.DB_PATH)
jobs = json.loads(
    (Path(__file__).resolve().parent.parent / "data/sample_jobs.json").read_text()
)
changed = sum(store.upsert_job(j) for j in jobs)
print(f"upserted {changed} jobs, embedded {ensure_embeddings(store, get_embedder())}")
