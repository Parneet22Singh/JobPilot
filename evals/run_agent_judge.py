"""LLM-as-judge on agent answers (needs GEMINI_API_KEY).  Usage: python -m evals.run_agent_judge"""

import json
import statistics
from pathlib import Path

from app import llm
from app.agent import chat
from app.embeddings import get_embedder
from app.search import Index, ensure_embeddings
from app.store import Store

ROOT = Path(__file__).resolve().parent.parent

QUESTIONS = [
    "I know Python and FastAPI. What backend roles fit me?",
    "Any remote roles working on LLM agents or evals?",
    "I'm into computer vision and OpenCV, what's out there?",
    "Find me something in Gurugram involving cloud or ML deployment.",
    "Is there a job at a company called Zephyr Robotics?",
]


def main():
    store, embedder = Store(":memory:"), get_embedder()
    for j in json.loads((ROOT / "data/sample_jobs.json").read_text()):
        store.upsert_job(j)
    ensure_embeddings(store, embedder)
    index = Index(store, embedder)
    context = "\n".join(
        f"{j['id']}: {j['title']} @ {j['company']} ({j['location']})"
        for j in store.all_jobs()
    )
    g, h = [], []
    for i, q in enumerate(QUESTIONS):
        out = chat(f"eval{i}", q, store, index)
        r = llm.judge(q, out["reply"], context)
        g.append(r.groundedness)
        h.append(r.helpfulness)
        print(
            f"Q: {q}\n tools={out['tools_used']} grounded={r.groundedness} helpful={r.helpfulness}\n {r.rationale}\n"
        )
    print(
        f"mean groundedness={statistics.mean(g):.2f}  mean helpfulness={statistics.mean(h):.2f}"
    )


if __name__ == "__main__":
    main()
