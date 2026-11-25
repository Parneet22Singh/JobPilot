"""Gemini calls with structured (schema-constrained) outputs: resume parsing, reranking, LLM-as-judge."""

import json

from pydantic import BaseModel, Field

from . import config

_client = None


def client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


class ResumeProfile(BaseModel):
    name: str = ""
    skills: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    years_experience: float = 0
    education: str = ""
    summary: str = Field("", description="Two-sentence summary of the candidate")


class RerankItem(BaseModel):
    job_id: str
    score: int = Field(description="Fit 0-100")
    reason: str = Field(
        description="One sentence citing concrete skill overlap or gaps"
    )


class RerankResult(BaseModel):
    ranked: list[RerankItem]


class JudgeResult(BaseModel):
    groundedness: int = Field(
        description="1-5: claims supported by the provided jobs only"
    )
    helpfulness: int = Field(description="1-5: answers the user's question")
    rationale: str


def structured(prompt: str, schema: type[BaseModel], system: str = "") -> BaseModel:
    from google.genai import types

    res = client().models.generate_content(
        model=config.CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system or None,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        ),
    )
    return schema.model_validate(json.loads(res.text))


def parse_resume(text: str) -> ResumeProfile:
    return structured(
        f"Extract a profile from this resume. Do not invent facts.\n\n{text[:12000]}",
        ResumeProfile,
    )


def rerank(profile: dict, candidates: list[dict]) -> RerankResult:
    listing = "\n".join(
        f"- id={c['id']} | {c['title']} @ {c['company']} | {', '.join(c['tags'])}"
        for c in candidates
    )
    prompt = (
        f"Candidate profile:\n{json.dumps(profile)}\n\nJobs:\n{listing}\n\n"
        "Rank ALL jobs by fit for this candidate. Use only the information given."
    )
    return structured(prompt, RerankResult)


def judge(question: str, answer: str, context: str) -> JudgeResult:
    prompt = (
        f"User question: {question}\n\nJobs available (ground truth):\n{context}\n\n"
        f"Assistant answer:\n{answer}\n\nScore the answer strictly."
    )
    return structured(
        prompt,
        JudgeResult,
        system="You are a strict evaluator of a job-search assistant.",
    )
