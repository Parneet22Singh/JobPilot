"""Embedding implementations for live search and offline tests."""

import hashlib
import re

import numpy as np

from . import config

TOKEN_RE = re.compile(r"[a-z0-9+#]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-9)


class HashEmbedder:
    """Bag-of-words hashing embedder. No semantics beyond lexical overlap; for tests only."""

    dim = 512

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]


class GeminiEmbedder:
    def __init__(self):
        from google import genai

        self._client = genai.Client(api_key=config.GEMINI_API_KEY)

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import types

        out = []
        for i in range(0, len(texts), 100):
            res = self._client.models.embed_content(
                model=config.EMBED_MODEL,
                contents=texts[i : i + 100],
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=config.EMBED_DIM
                ),
            )
            out.extend(e.values for e in res.embeddings)
        return _normalize(np.array(out, dtype=np.float32))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def get_embedder():
    return GeminiEmbedder() if config.GEMINI_API_KEY else HashEmbedder()
