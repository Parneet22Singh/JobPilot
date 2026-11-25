"""Compare BM25 vs dense vs hybrid on the golden set.  Usage: python -m evals.run_retrieval"""

import argparse
import json
import statistics
from pathlib import Path

from app.embeddings import get_embedder
from app.search import Index, ensure_embeddings
from app.store import Store
from evals.metrics import ndcg_at_k, recall_at_k

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        help="use jobs already in this SQLite DB (real data) instead of the 20 samples",
    )
    ap.add_argument("--golden", default=str(ROOT / "evals/golden.json"))
    a = ap.parse_args()
    embedder = get_embedder()
    if a.db:
        store = Store(a.db)
    else:
        store = Store(":memory:")
        for j in json.loads((ROOT / "data/sample_jobs.json").read_text()):
            store.upsert_job(j)
        ensure_embeddings(store, embedder)
    index, golden = Index(store, embedder), json.loads(Path(a.golden).read_text())

    live = {j["id"] for j in store.all_jobs()}
    gone = sum(1 for g in golden for i in g["relevant"] if i not in live)
    if gone:
        print(
            f"WARNING: {gone} labeled jobs are no longer in the index (listings expire). "
            "Evaluate against a frozen copy of the DB: see README 'Evaluating on live data'."
        )
    print(
        f"embedder={type(embedder).__name__}  jobs={len(store.all_jobs())}  queries={len(golden)}"
    )
    print(f"{'mode':8} {'nDCG@10':>8} {'Recall@5':>9}")
    for mode in ("bm25", "dense", "hybrid"):
        nd, rc = [], []
        for g in golden:
            ids = [r["id"] for r in index.search(g["query"], k=10, mode=mode)]
            nd.append(ndcg_at_k(ids, set(g["relevant"]), 10))
            rc.append(recall_at_k(ids, set(g["relevant"]), 5))
        print(f"{mode:8} {statistics.mean(nd):8.3f} {statistics.mean(rc):9.3f}")


if __name__ == "__main__":
    main()
