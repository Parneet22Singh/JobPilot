from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, llm
from .agent import chat as agent_chat
from .embeddings import get_embedder
from .search import Index, ensure_embeddings
from .store import Store

app = FastAPI(title="JobPilot", version="0.1.0")
store = Store(config.DB_PATH)
embedder = get_embedder()
index = Index(store, embedder)


class JobIn(BaseModel):
    id: str
    title: str
    company: str = ""
    location: str = ""
    remote: bool = False
    url: str = ""
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    source: str = "api"
    posted_at: float | None = None


class IngestReq(BaseModel):
    jobs: list[JobIn]


class SearchReq(BaseModel):
    query: str
    k: int = 10
    mode: str = "hybrid"
    location: str | None = None
    remote_only: bool = False


class ResumeReq(BaseModel):
    session_id: str
    text: str


class ChatReq(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health():
    return {
        "ok": True,
        "embedder": type(embedder).__name__,
        "jobs": len(store.all_jobs()),
    }


@app.post("/jobs/ingest")
def ingest(req: IngestReq):
    changed = sum(store.upsert_job(j.model_dump()) for j in req.jobs)
    embedded = ensure_embeddings(store, embedder)
    return {"received": len(req.jobs), "new_or_changed": changed, "embedded": embedded}


@app.post("/search")
def search(req: SearchReq):
    if req.mode not in ("bm25", "dense", "hybrid"):
        raise HTTPException(400, "mode must be bm25, dense or hybrid")
    return {
        "results": index.search(
            req.query, req.k, req.mode, req.location, req.remote_only
        )
    }


@app.post("/resume/parse")
def parse_resume(req: ResumeReq):
    profile = llm.parse_resume(req.text).model_dump()
    store.set_profile(req.session_id, profile)
    return profile


@app.get("/recommend/{session_id}")
def recommend(session_id: str, k: int = 5):
    profile = store.get_profile(session_id)
    if not profile:
        raise HTTPException(404, "Upload a resume first via /resume/parse")
    query = " ".join(profile["target_roles"] + profile["skills"])
    candidates = index.search(query, k=20)
    ranked = llm.rerank(profile, candidates).ranked
    by_id = {c["id"]: c for c in candidates}
    out = [
        {**by_id[r.job_id], "fit": r.score, "reason": r.reason}
        for r in sorted(ranked, key=lambda r: -r.score)
        if r.job_id in by_id
    ]
    return {"recommendations": out[:k]}


@app.post("/chat")
def chat(req: ChatReq):
    return agent_chat(req.session_id, req.message, store, index)
