"""Pull CURRENT jobs from every live source, drop closed/old ones, embed what's new.

  python -m scripts.refresh                         # all sources
  python -m scripts.refresh --sources greenhouse,lever --filter "intern|engineer|developer"
Run it on a schedule (cron / GitHub Actions) so the index tracks what is open today.
How listings stay current:
  * company boards (greenhouse/lever): a job missing from the board's latest listing is deleted (taken down)
  * every job records last_seen; anything not seen for --prune-days is deleted
  * jobs with a known posting date older than --max-age-days are not loaded
"""

import argparse
import json
import re
import time
from pathlib import Path

from app import config, sources
from app.embeddings import get_embedder
from app.ingest import save
from app.search import ensure_embeddings
from app.store import Store
from scripts.scrape_jobs import greenhouse_jobs, lever_jobs

ROOT = Path(__file__).resolve().parent.parent
FEEDS = {
    "remotive": sources.fetch_remotive,
    "arbeitnow": sources.fetch_arbeitnow,
    "remoteok": sources.fetch_remoteok,
    "adzuna": sources.fetch_adzuna,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sources", default="remotive,arbeitnow,remoteok,adzuna,greenhouse,lever"
    )
    ap.add_argument("--boards", default=str(ROOT / "data/company_boards.json"))
    ap.add_argument("--filter", help="regex on title, e.g. 'intern|engineer|developer'")
    ap.add_argument("--max-age-days", type=float, default=30)
    ap.add_argument("--prune-days", type=float, default=14)
    a = ap.parse_args()

    store = Store(config.DB_PATH)
    boards = json.loads(Path(a.boards).read_text()) if Path(a.boards).exists() else {}
    cutoff = time.time() - a.max_age_days * 86400
    for name in a.sources.split(","):
        name = name.strip()
        print(f"[{name}]")
        try:
            if name in FEEDS:
                jobs, snapshot = FEEDS[name](), False
            elif name in ("greenhouse", "lever"):
                fn = greenhouse_jobs if name == "greenhouse" else lever_jobs
                jobs, snapshot = [j for b in boards.get(name, []) for j in fn(b)], True
            else:
                print("  unknown source")
                continue
        except Exception as e:
            print(f"  failed: {e}")
            continue
        jobs = [j for j in jobs if not j.get("posted_at") or j["posted_at"] >= cutoff]
        if a.filter:
            jobs = [j for j in jobs if re.search(a.filter, j["title"], re.I)]
        print(" ", save(store, jobs, snapshot))

    embedded = ensure_embeddings(store, get_embedder())
    pruned = store.prune_stale(a.prune_days)
    print(
        f"embedded={embedded} pruned_stale={pruned} live_jobs_now={len(store.all_jobs())}"
    )


if __name__ == "__main__":
    main()
