"""Collect jobs from sources that allow it, then upsert + embed them.

  python -m scripts.scrape_jobs greenhouse                 # company boards in data/company_boards.json
  python -m scripts.scrape_jobs lever --filter "intern|engineer"
  python -m scripts.scrape_jobs jsonld --urls data/job_urls.txt
  python -m scripts.scrape_jobs jsonld --sitemap https://example.com/sitemap.xml --match /jobs/ --max 200

Sources:
  greenhouse / lever : the public JSON endpoints companies publish for their own career pages.
  jsonld             : any careers page that embeds schema.org JobPosting data (Google Jobs uses it).
                       Checks robots.txt, rate-limits itself, identifies itself with a User-Agent.
Not supported on purpose: LinkedIn, Indeed, Glassdoor, Naukri (their terms forbid scraping).
Use the Kaggle LinkedIn dataset (scripts/ingest_csv.py) for that data instead.
"""

import argparse
import hashlib
import html
import json
import re
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

from app import config
from app.embeddings import get_embedder
from app.ingest import save, to_epoch
from app.search import ensure_embeddings
from app.store import Store

UA = "jobpilot-student-project/0.1 (personal research; contact: set-your-email)"
DELAY = 1.0
BLOCKED = ("linkedin.com", "indeed.com", "glassdoor.com", "naukri.com")
ROOT = Path(__file__).resolve().parent.parent


def to_text(s: str) -> str:
    s = html.unescape(s or "")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def get(url: str, **kw):
    time.sleep(DELAY)
    return requests.get(url, headers={"User-Agent": UA}, timeout=30, **kw)


# ---------- Greenhouse / Lever ----------
def greenhouse_jobs(token: str) -> list[dict]:
    r = get(
        f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs",
        params={"content": "true"},
    )
    if r.status_code != 200:
        print(f"  skip greenhouse/{token}: HTTP {r.status_code}")
        return []
    return [map_greenhouse(j, token) for j in r.json().get("jobs", [])]


def map_greenhouse(j: dict, token: str) -> dict:
    loc = (j.get("location") or {}).get("name", "")
    return {
        "id": f"gh-{token}-{j['id']}",
        "title": j["title"],
        "company": token.title(),
        "location": loc,
        "source": f"gh-{token}",
        "posted_at": to_epoch(j.get("updated_at")),
        "remote": "remote" in loc.lower(),
        "url": j.get("absolute_url", ""),
        "tags": [],
        "description": to_text(j.get("content", ""))[:3000],
    }


def lever_jobs(company: str) -> list[dict]:
    r = get(f"https://api.lever.co/v0/postings/{company}", params={"mode": "json"})
    if r.status_code != 200:
        print(f"  skip lever/{company}: HTTP {r.status_code}")
        return []
    return [map_lever(j, company) for j in r.json()]


def map_lever(j: dict, company: str) -> dict:
    cat = j.get("categories") or {}
    loc = cat.get("location", "")
    return {
        "id": f"lv-{company}-{j['id']}",
        "title": j["text"],
        "company": company.title(),
        "location": loc,
        "source": f"lv-{company}",
        "posted_at": to_epoch(j.get("createdAt")),
        "remote": (j.get("workplaceType") == "remote") or "remote" in loc.lower(),
        "url": j.get("hostedUrl", ""),
        "tags": [t for t in (cat.get("team"), cat.get("commitment")) if t],
        "description": to_text(j.get("descriptionPlain") or j.get("description", ""))[
            :3000
        ],
    }


# ---------- Generic schema.org JobPosting pages ----------
LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I
)


def _find_postings(node):
    if isinstance(node, list):
        for n in node:
            yield from _find_postings(n)
    elif isinstance(node, dict):
        t = node.get("@type")
        if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
            yield node
        for v in node.get("@graph", []) if isinstance(node.get("@graph"), list) else []:
            yield from _find_postings(v)


def parse_jsonld(page_html: str, page_url: str) -> list[dict]:
    out = []
    for block in LD_RE.findall(page_html):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        for p in _find_postings(data):
            loc = p.get("jobLocation") or {}
            loc = loc[0] if isinstance(loc, list) and loc else loc
            addr = (loc.get("address") or {}) if isinstance(loc, dict) else {}
            place = ", ".join(
                x
                for x in (
                    addr.get("addressLocality"),
                    addr.get("addressRegion"),
                    addr.get("addressCountry"),
                )
                if isinstance(x, str) and x
            )
            org = p.get("hiringOrganization") or {}
            url = p.get("url") or page_url
            out.append(
                {
                    "id": "ld-" + hashlib.md5(url.encode()).hexdigest()[:12],
                    "source": "ld-" + urlparse(url).netloc,
                    "posted_at": to_epoch(p.get("datePosted")),
                    "title": p.get("title", ""),
                    "company": (
                        org.get("name", "") if isinstance(org, dict) else str(org)
                    ),
                    "location": place,
                    "remote": p.get("jobLocationType") == "TELECOMMUTE"
                    or "remote" in place.lower(),
                    "url": url,
                    "tags": [
                        t for t in [p.get("employmentType")] if isinstance(t, str)
                    ],
                    "description": to_text(p.get("description", ""))[:3000],
                }
            )
    return [j for j in out if j["title"]]


_robots: dict[str, urllib.robotparser.RobotFileParser] = {}


def allowed(url: str) -> bool:
    host = urlparse(url).netloc
    if any(host.endswith(b) for b in BLOCKED):
        print(
            f"  refusing {host}: scraping not allowed by their terms. Use scripts/ingest_csv.py instead."
        )
        return False
    base = f"{urlparse(url).scheme}://{host}"
    if base not in _robots:
        rp = urllib.robotparser.RobotFileParser(base + "/robots.txt")
        try:
            rp.read()
        except Exception:
            pass
        _robots[base] = rp
    try:
        return _robots[base].can_fetch(UA, url)
    except Exception:
        return True


def jsonld_jobs(urls: list[str]) -> list[dict]:
    jobs = []
    for u in urls:
        if not allowed(u):
            print(f"  skip (robots.txt/blocked): {u}")
            continue
        try:
            r = get(u)
            if r.status_code == 200:
                jobs += parse_jsonld(r.text, u)
            else:
                print(f"  skip {u}: HTTP {r.status_code}")
        except requests.RequestException as e:
            print(f"  skip {u}: {e}")
    return jobs


def sitemap_urls(sitemap: str, match: str, limit: int) -> list[str]:
    if not allowed(sitemap):
        return []
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", get(sitemap).text)
    return [u for u in locs if match in u][:limit]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", choices=["greenhouse", "lever", "jsonld"])
    ap.add_argument("--boards", default=str(ROOT / "data/company_boards.json"))
    ap.add_argument("--urls", help="text file, one job-page URL per line (jsonld)")
    ap.add_argument("--sitemap", help="sitemap.xml URL to discover job pages (jsonld)")
    ap.add_argument(
        "--match", default="job", help="only sitemap URLs containing this text"
    )
    ap.add_argument("--max", type=int, default=100, help="max pages from a sitemap")
    ap.add_argument(
        "--filter",
        help="regex; keep only jobs whose title matches (e.g. 'intern|engineer')",
    )
    a = ap.parse_args()

    jobs: list[dict] = []
    if a.source in ("greenhouse", "lever"):
        names = json.loads(Path(a.boards).read_text()).get(a.source, [])
        for n in names:
            print(f"{a.source}/{n}")
            jobs += (greenhouse_jobs if a.source == "greenhouse" else lever_jobs)(n)
    else:
        urls = []
        if a.urls:
            urls += [
                u.strip() for u in Path(a.urls).read_text().splitlines() if u.strip()
            ]
        if a.sitemap:
            urls += sitemap_urls(a.sitemap, a.match, a.max)
        if not urls:
            ap.error("jsonld needs --urls and/or --sitemap")
        jobs = jsonld_jobs(urls)

    if a.filter:
        jobs = [j for j in jobs if re.search(a.filter, j["title"], re.I)]
    store = Store(config.DB_PATH)
    print(save(store, jobs, snapshot=a.source in ("greenhouse", "lever")))
    ensure_embeddings(store, get_embedder())


if __name__ == "__main__":
    main()
