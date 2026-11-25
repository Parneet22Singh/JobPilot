"""Hybrid retrieval with BM25 and dense embeddings."""

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import tokenize
from .store import Store, job_text

RRF_K = 60


def ensure_embeddings(
    store: Store, embedder, chunk: int = 100, retries: int = 5
) -> int:
    """Embed changed jobs in resumable chunks."""
    import time

    todo = store.jobs_missing_embeddings()
    for i in range(0, len(todo), chunk):
        batch = todo[i : i + chunk]
        for attempt in range(retries):
            try:
                vecs = embedder.embed_docs([job_text(j) for j in batch])
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt * 5)
        store.set_embeddings([j["id"] for j in batch], vecs)
        print(f"embedded {min(i + chunk, len(todo))}/{len(todo)}", flush=True)
    return len(todo)


class Index:
    def __init__(self, store: Store, embedder):
        self.store, self.embedder = store, embedder
        self._seen_version = -1
        self.jobs: list[dict] = []

    def _reload(self):
        if self._seen_version == self.store.version:
            return
        self.jobs = [j for j in self.store.all_jobs() if j["embedding"] is not None]
        self.mat = (
            np.stack([j["embedding"] for j in self.jobs])
            if self.jobs
            else np.zeros((0, 1))
        )
        self.bm25 = (
            BM25Okapi([tokenize(job_text(j)) for j in self.jobs]) if self.jobs else None
        )
        self._seen_version = self.store.version

    @staticmethod
    def _ranks(scores: np.ndarray, mask: np.ndarray) -> dict[int, int]:
        order = [i for i in np.argsort(-scores) if mask[i]]
        return {i: r for r, i in enumerate(order)}

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "hybrid",
        location: str | None = None,
        remote_only: bool = False,
    ) -> list[dict]:
        self._reload()
        if not self.jobs:
            return []
        mask = np.ones(len(self.jobs), dtype=bool)
        for i, j in enumerate(self.jobs):
            if remote_only and not j["remote"]:
                mask[i] = False
            if (
                location
                and location.lower() not in (j["location"] or "").lower()
                and not j["remote"]
            ):
                mask[i] = False

        ranks: list[dict[int, int]] = []
        if mode in ("bm25", "hybrid"):
            ranks.append(
                self._ranks(np.array(self.bm25.get_scores(tokenize(query))), mask)
            )
        if mode in ("dense", "hybrid"):
            ranks.append(self._ranks(self.mat @ self.embedder.embed_query(query), mask))

        fused: dict[int, float] = {}
        for r in ranks:
            for i, rank in r.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            {
                "score": round(fused[i], 5),
                **{
                    x: self.jobs[i][x]
                    for x in (
                        "id",
                        "title",
                        "company",
                        "location",
                        "remote",
                        "url",
                        "tags",
                        "posted_at",
                    )
                },
            }
            for i in top
        ]
