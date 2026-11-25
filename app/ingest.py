"""Shared helpers for loading live jobs."""

import re
from datetime import datetime, timezone

from .store import Store


def to_epoch(v) -> float | None:
    """Turn API date values (ISO string, epoch seconds or ms) into epoch seconds."""
    if v in (None, ""):
        return None
    try:
        x = float(v)
        return x / 1000 if x > 1e11 else x
    except (TypeError, ValueError):
        pass
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    except ValueError:
        return None


def clean_html(s: str) -> str:
    import html

    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(s or ""))).strip()


def save(store: Store, jobs: list[dict], snapshot: bool) -> dict:
    """Upsert jobs and remove missing listings for complete source snapshots."""
    changed = sum(store.upsert_job(j) for j in jobs)
    expired = 0
    if snapshot:
        groups: dict[str, set[str]] = {}
        for j in jobs:
            groups.setdefault(j["source"], set()).add(j["id"])
        for src, ids in groups.items():
            expired += store.expire_missing(src, ids)
    return {"fetched": len(jobs), "new_or_changed": changed, "expired": expired}
