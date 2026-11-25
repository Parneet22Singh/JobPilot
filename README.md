# JobPilot: AI Career Agent with Hybrid Semantic Job Search

Tool-using LLM agent (Gemini) with persistent memory, resume parsing to structured JSON, hybrid retrieval
(BM25 + embeddings fused with RRF) over a LIVE jobs index, LLM reranking with explanations, and an eval harness.

## Quick start (live jobs only)
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
set GEMINI_API_KEY=...
python -m pip install -r requirements.txt
python -m scripts.refresh --filter "intern|engineer|developer|data|ml|ai"
python -m uvicorn app.main:app --reload
```
Optional for India coverage: free key at developer.adzuna.com, then
`set ADZUNA_APP_ID=...`, `set ADZUNA_APP_KEY=...`, and `set ADZUNA_COUNTRY=in`.

## Where jobs come from (all current openings)
| Source | What it is |
|---|---|
| greenhouse / lever | Public feeds of company career boards. List companies in `data/company_boards.json` |
| remotive, remoteok, arbeitnow | Free public job APIs (mostly remote/Europe; credit remotive.com / remoteok.com if you display jobs) |
| adzuna | Aggregated listings incl. India (needs free key). Gives short descriptions only |
| jsonld | `python -m scripts.scrape_jobs jsonld --sitemap <url> --match /careers/` for any careers page with schema.org JobPosting data (obeys robots.txt) |

LinkedIn, Indeed, Glassdoor and Naukri are NOT scraped (their terms forbid it); the script refuses those domains.

## How listings stay current
- Company boards: a job missing from the board's latest listing is deleted (it was taken down).
- Every job stores `last_seen`; jobs not seen for `--prune-days` (default 14) are deleted.
- Jobs with a posting date older than `--max-age-days` (default 30) are never loaded.
- Only new/changed jobs are re-embedded. Schedule `python -m scripts.refresh` daily (cron / GitHub Actions).
- `data/sample_jobs.json` is synthetic and used ONLY by unit tests and `scripts.seed` (offline demo). Don't use it for results.

## Evaluating on live data
Live jobs expire, so labels can go stale. Freeze a copy first:
```bash
cp data/jobpilot.db data/eval_snapshot.db
DB_PATH=data/eval_snapshot.db python -m scripts.label_golden        # edit evals/queries.txt first
python -m evals.run_retrieval --db data/eval_snapshot.db --golden evals/golden_real.json
```
`label_golden` pools results from BM25, dense and hybrid and hides which method found what.
`evals/golden.json` + sample data is only a pipeline smoke test.

## API
`/jobs/ingest`, `/search`, `/resume/parse`, `/recommend/{session}`, `/chat`, `/health`

## Other
`python -m pytest` (7 tests) | `docker compose up --build` | `python -m evals.run_agent_judge` (needs Gemini key)
