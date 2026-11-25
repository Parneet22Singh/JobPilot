import json
from pathlib import Path

from app.embeddings import HashEmbedder
from app.search import Index, ensure_embeddings
from app.store import Store
from evals.metrics import ndcg_at_k, recall_at_k

JOBS = json.loads(
    (Path(__file__).resolve().parent.parent / "data/sample_jobs.json").read_text()
)


def make_index():
    s, e = Store(":memory:"), HashEmbedder()
    for j in JOBS:
        s.upsert_job(j)
    ensure_embeddings(s, e)
    return s, e, Index(s, e)


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(["a", "b"], {"a", "b"}) == 1.0
    assert ndcg_at_k(["x", "y"], {"a"}) == 0.0
    assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5


def test_hybrid_finds_kubernetes_job():
    _, _, idx = make_index()
    assert idx.search("kubernetes", k=3)[0]["id"] == "j05"


def test_remote_filter():
    _, _, idx = make_index()
    assert all(r["remote"] for r in idx.search("engineer", k=10, remote_only=True))


def test_upsert_only_reembeds_changed_jobs():
    s, e, _ = make_index()
    assert s.upsert_job(JOBS[0]) is False
    assert ensure_embeddings(s, e) == 0
    s.upsert_job({**JOBS[0], "description": "Brand new description about rust"})
    assert ensure_embeddings(s, e) == 1


def test_scraper_parsers():
    from scripts.scrape_jobs import map_greenhouse, map_lever, parse_jsonld, allowed

    gh = map_greenhouse(
        {
            "id": 1,
            "title": "SWE Intern",
            "location": {"name": "Remote - India"},
            "content": "&lt;p&gt;Build &lt;b&gt;APIs&lt;/b&gt;&lt;/p&gt;",
            "absolute_url": "http://x",
        },
        "acme",
    )
    assert gh["remote"] and gh["description"] == "Build APIs"
    lv = map_lever(
        {
            "id": "a",
            "text": "ML Engineer",
            "categories": {"location": "Pune", "team": "AI"},
            "descriptionPlain": "Train models",
            "hostedUrl": "http://y",
        },
        "acme",
    )
    assert lv["location"] == "Pune" and "AI" in lv["tags"]
    page = (
        '<script type="application/ld+json">{"@type":"JobPosting","title":"Dev","hiringOrganization":{"name":"Co"},'
        '"jobLocation":{"address":{"addressLocality":"Delhi","addressCountry":"IN"}},"description":"<p>Hi</p>"}</script>'
    )
    j = parse_jsonld(page, "http://z/job/1")[0]
    assert j["title"] == "Dev" and j["location"] == "Delhi, IN" and j["company"] == "Co"
    assert allowed("https://www.linkedin.com/jobs/view/1") is False


def test_expiry_and_freshness():
    from app.ingest import save, to_epoch

    s = Store(":memory:")
    a = {"id": "gh-x-1", "source": "gh-x", "title": "Dev", "description": "build"}
    b = {"id": "gh-x-2", "source": "gh-x", "title": "QA", "description": "test"}
    assert save(s, [a, b], snapshot=True)["new_or_changed"] == 2
    r = save(s, [a], snapshot=True)
    assert r["expired"] == 1 and s.get_job("gh-x-2") is None
    assert s.prune_stale(days=0.0) >= 0
    assert (
        to_epoch("2024-01-01T00:00:00Z") == 1704067200.0
        and to_epoch(1704067200000) == 1704067200.0
    )


def test_feed_mappers():
    from app.sources import map_remotive, map_arbeitnow, map_remoteok, map_adzuna

    assert (
        map_remotive(
            {
                "id": 1,
                "title": "T",
                "company_name": "C",
                "url": "u",
                "description": "<b>x</b>",
                "publication_date": "2026-11-01T00:00:00",
            }
        )["description"]
        == "x"
    )
    assert (
        map_arbeitnow(
            {
                "slug": "s",
                "title": "T",
                "company_name": "C",
                "description": "d",
                "created_at": 1700000000,
            }
        )["id"]
        == "arbeitnow-s"
    )
    assert map_remoteok({"id": 5, "position": "P", "description": "d"})["remote"]
    assert (
        map_adzuna(
            {
                "id": 9,
                "title": "T",
                "company": {"display_name": "C"},
                "location": {"display_name": "Pune"},
            }
        )["location"]
        == "Pune"
    )
