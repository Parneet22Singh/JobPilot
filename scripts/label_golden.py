"""Interactively build a golden set from REAL jobs in your DB.
Pooling: candidates = union of top-10 from BM25, dense and hybrid, shuffled and shown without saying which
method found them, so the labels don't favour one method.   python -m scripts.label_golden
Keys: y = relevant, n = not, s = skip query, q = save & quit
"""

import json
import random
from pathlib import Path

from app import config
from app.embeddings import get_embedder
from app.search import Index
from app.store import Store

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals/golden_real.json"

store = Store(config.DB_PATH)
index = Index(store, get_embedder())
golden = json.loads(OUT.read_text()) if OUT.exists() else []
done = {g["query"] for g in golden}
queries = [
    q.strip()
    for q in (ROOT / "evals/queries.txt").read_text().splitlines()
    if q.strip()
]
by_id = {j["id"]: j for j in store.all_jobs()}

for q in queries:
    if q in done:
        continue
    pool = {}
    for mode in ("bm25", "dense", "hybrid"):
        for r in index.search(q, k=10, mode=mode):
            pool[r["id"]] = r
    cands = list(pool.values())
    random.Random(q).shuffle(cands)
    print(f"\n=== QUERY: {q}  ({len(cands)} candidates) ===")
    relevant, skip = [], False
    for r in cands:
        j = by_id[r["id"]]
        print(
            f"\n{j['title']} @ {j['company']} ({j['location']})\n  {j['description'][:220]}..."
        )
        ans = input("relevant? [y/n/s/q] ").strip().lower()
        if ans == "q":
            OUT.write_text(json.dumps(golden, indent=1))
            raise SystemExit
        if ans == "s":
            skip = True
            break
        if ans == "y":
            relevant.append(r["id"])
    if not skip and relevant:
        golden.append({"query": q, "relevant": relevant})
        OUT.write_text(json.dumps(golden, indent=1))
print(f"saved {len(golden)} labeled queries to {OUT}")
