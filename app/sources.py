"""Fetchers for live job feeds. Each returns a list of job dicts (see store.upsert_job)."""

import os

import requests

from .ingest import clean_html, to_epoch

UA = "jobpilot-student-project/0.1 (personal research)"


def _get(url, **kw):
    return requests.get(url, headers={"User-Agent": UA}, timeout=30, **kw)


def map_remotive(j: dict) -> dict:
    return {
        "id": f"remotive-{j['id']}",
        "source": "remotive",
        "title": j["title"],
        "company": j["company_name"],
        "location": j.get("candidate_required_location") or "Remote",
        "remote": True,
        "url": j["url"],
        "tags": j.get("tags") or [],
        "posted_at": to_epoch(j.get("publication_date")),
        "description": clean_html(j.get("description"))[:3000],
    }


def fetch_remotive() -> list[dict]:
    """Remotive asks API users to fetch sparingly (a few times a day) and to credit remotive.com."""
    return [
        map_remotive(j)
        for j in _get("https://remotive.com/api/remote-jobs").json().get("jobs", [])
    ]


def map_arbeitnow(j: dict) -> dict:
    return {
        "id": f"arbeitnow-{j['slug']}",
        "source": "arbeitnow",
        "title": j["title"],
        "company": j["company_name"],
        "location": j.get("location", ""),
        "remote": bool(j.get("remote")),
        "url": j.get("url", ""),
        "tags": j.get("tags") or [],
        "posted_at": to_epoch(j.get("created_at")),
        "description": clean_html(j.get("description"))[:3000],
    }


def fetch_arbeitnow(pages: int = 3) -> list[dict]:
    out = []
    for p in range(1, pages + 1):
        r = _get("https://www.arbeitnow.com/api/job-board-api", params={"page": p})
        if r.status_code != 200:
            break
        out += [map_arbeitnow(j) for j in r.json().get("data", [])]
    return out


def map_remoteok(j: dict) -> dict:
    return {
        "id": f"remoteok-{j['id']}",
        "source": "remoteok",
        "title": j["position"],
        "company": j.get("company", ""),
        "location": j.get("location") or "Remote",
        "remote": True,
        "url": j.get("url", ""),
        "tags": j.get("tags") or [],
        "posted_at": to_epoch(j.get("epoch") or j.get("date")),
        "description": clean_html(j.get("description"))[:3000],
    }


def fetch_remoteok() -> list[dict]:
    """RemoteOK's terms ask you to link back to remoteok.com if you show their jobs."""
    data = _get("https://remoteok.com/api").json()
    return [map_remoteok(j) for j in data if isinstance(j, dict) and "position" in j]


def map_adzuna(j: dict) -> dict:
    return {
        "id": f"adzuna-{j['id']}",
        "source": "adzuna",
        "title": j["title"],
        "company": (j.get("company") or {}).get("display_name", ""),
        "location": (j.get("location") or {}).get("display_name", ""),
        "remote": "remote" in (j.get("title", "") + j.get("description", "")).lower(),
        "url": j.get("redirect_url", ""),
        "tags": [],
        "posted_at": to_epoch(j.get("created")),
        "description": clean_html(j.get("description"))[:3000],
    }


def fetch_adzuna(country: str = None, what: str = None, pages: int = 3) -> list[dict]:
    """Free API key from developer.adzuna.com -> set ADZUNA_APP_ID and ADZUNA_APP_KEY.
    Best option here for India (ADZUNA_COUNTRY=in) since the other feeds skew remote/global.
    """
    app_id, app_key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        print("  adzuna: set ADZUNA_APP_ID and ADZUNA_APP_KEY to enable (skipped)")
        return []
    country = country or os.getenv("ADZUNA_COUNTRY", "in")
    what = what or os.getenv("ADZUNA_QUERY", "software engineer")
    out = []
    for p in range(1, pages + 1):
        r = _get(
            f"https://api.adzuna.com/v1/api/jobs/{country}/search/{p}",
            params={
                "app_id": app_id,
                "app_key": app_key,
                "what": what,
                "results_per_page": 50,
            },
        )
        if r.status_code != 200:
            print(f"  adzuna: HTTP {r.status_code}")
            break
        out += [map_adzuna(j) for j in r.json().get("results", [])]
    return out
