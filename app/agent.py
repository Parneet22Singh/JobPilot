"""Tool-using career agent with persistent conversational memory (Gemini automatic function calling)."""

from google.genai import types

from . import config
from .llm import client
from .search import Index
from .store import Store

SYSTEM = (
    "You are JobPilot, a career assistant for students and early-career candidates. "
    "ALWAYS use search_jobs before recommending roles; never invent jobs. "
    "Use get_user_profile to personalise. Cite job ids and explain fit in 1-2 sentences each. "
    "If the tools return nothing relevant, say so."
)


def build_tools(session_id: str, store: Store, index: Index):
    def search_jobs(
        query: str, location: str = "", remote_only: bool = False
    ) -> list[dict]:
        """Search the live jobs index. query: natural-language role/skills. location: optional city. remote_only: only remote roles."""
        return index.search(
            query, k=5, location=location or None, remote_only=remote_only
        )

    def get_job_details(job_id: str) -> dict:
        """Get full description of one job by its id."""
        return store.get_job(job_id) or {"error": "job not found"}

    def get_user_profile() -> dict:
        """Get the parsed resume profile of the current user (skills, target roles, experience)."""
        return store.get_profile(session_id) or {"error": "no resume uploaded yet"}

    return [search_jobs, get_job_details, get_user_profile]


def chat(session_id: str, message: str, store: Store, index: Index) -> dict:
    history = store.get_messages(session_id, limit=20)
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    res = client().models.generate_content(
        model=config.CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=build_tools(session_id, store, index),
            temperature=0.3,
        ),
    )
    reply = res.text or ""
    used = []
    for c in getattr(res, "automatic_function_calling_history", None) or []:
        for p in c.parts or []:
            if getattr(p, "function_call", None):
                used.append(p.function_call.name)

    store.add_message(session_id, "user", message)
    store.add_message(session_id, "model", reply)
    return {"reply": reply, "tools_used": used}
