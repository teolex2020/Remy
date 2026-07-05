"""Unit tests for source_filter (Phase 2)."""

from __future__ import annotations

from remy.core.retrieval.source_filter import (
    CLASS_GITHUB,
    CLASS_MIRROR,
    CLASS_OFFICIAL_DOCS,
    CLASS_RESEARCH,
    CLASS_SEO,
    CLASS_UNKNOWN,
    annotate,
    classify,
    enforce_site_constraint,
    extract_site_constraint,
    rerank,
)


def _c(title: str, uri: str, snippet: str = "") -> dict:
    return {"title": title, "uri": uri, "snippet": snippet}


def test_classify_arxiv_abs_is_research():
    s = classify(_c("Some Paper", "https://arxiv.org/abs/2312.10997"))
    assert s.source_class == CLASS_RESEARCH
    assert s.source_score >= 2
    assert "research_host" in s.source_signals


def test_classify_python_docs_is_official():
    s = classify(_c("asyncio TaskGroup", "https://docs.python.org/3/library/asyncio-task.html"))
    assert s.source_class == CLASS_OFFICIAL_DOCS
    assert s.source_score >= 2


def test_classify_github_repo_is_github():
    s = classify(_c("repo", "https://github.com/python/cpython"))
    assert s.source_class == CLASS_GITHUB


def test_classify_seo_leaderboard_is_seo():
    s = classify(_c("LLM Leaderboard", "https://onyx.app/llm-leaderboard"))
    assert s.source_class == CLASS_SEO
    assert s.source_score <= -2
    assert any(sig.startswith("seo_") for sig in s.source_signals)


def test_classify_best_of_year_title_is_seo():
    s = classify(_c("The 10 Best LLMs of 2026", "https://randomblog.example/post/123"))
    assert s.source_class == CLASS_SEO


def test_classify_researchgate_is_mirror():
    s = classify(_c("Some paper", "https://www.researchgate.net/publication/12345"))
    assert s.source_class == CLASS_MIRROR
    assert s.source_score < 0


def test_classify_unknown_when_nothing_matches():
    s = classify(_c("Random", "https://example.com/"))
    assert s.source_class == CLASS_UNKNOWN


def test_rerank_boosts_research_over_unknown():
    cands = annotate([
        _c("unknown", "https://example.com/a"),
        _c("arxiv paper", "https://arxiv.org/abs/2312.10997"),
        _c("another unknown", "https://example.net/b"),
    ])
    ranked = rerank(cands)
    assert ranked[0]["uri"] == "https://arxiv.org/abs/2312.10997"


def test_rerank_drops_seo_when_requested():
    cands = annotate([
        _c("LLM Leaderboard", "https://onyx.app/llm-leaderboard"),
        _c("arxiv paper", "https://arxiv.org/abs/2312.10997"),
    ])
    ranked = rerank(cands, drop_classes={"seo"})
    uris = [c["uri"] for c in ranked]
    assert "https://onyx.app/llm-leaderboard" not in uris
    assert "https://arxiv.org/abs/2312.10997" in uris


def test_annotate_preserves_original_fields():
    cands = annotate([_c("t", "https://arxiv.org/abs/2312.10997", "snippet text")])
    assert cands[0]["title"] == "t"
    assert cands[0]["snippet"] == "snippet text"
    assert "source_class" in cands[0]
    assert "source_score" in cands[0]
    assert "source_signals" in cands[0]


def test_rerank_preserves_order_on_ties():
    # Two unknown hosts; should keep input order.
    cands = annotate([
        _c("first", "https://example.com/first"),
        _c("second", "https://example.org/second"),
    ])
    ranked = rerank(cands)
    assert [c["uri"] for c in ranked] == [
        "https://example.com/first",
        "https://example.org/second",
    ]


# ── Phase 2: site: constraint extraction ─────────────────────────────────────


def test_extract_site_constraint_basic():
    assert extract_site_constraint("site:arxiv.org moe survey") == "arxiv.org"


def test_extract_site_constraint_trailing():
    assert extract_site_constraint("asyncio task group site:docs.python.org") == "docs.python.org"


def test_extract_site_constraint_case_insensitive():
    assert extract_site_constraint("Site:ArXiv.ORG paper") == "arxiv.org"


def test_extract_site_constraint_strips_www():
    assert extract_site_constraint("site:www.python.org") == "python.org"


def test_extract_site_constraint_none_when_absent():
    assert extract_site_constraint("just a normal query") is None


def test_extract_site_constraint_empty_query():
    assert extract_site_constraint("") is None
    assert extract_site_constraint(None) is None


# ── Phase 2: site: constraint enforcement ────────────────────────────────────


def test_enforce_site_constraint_drops_off_domain():
    cands = [
        _c("paper", "https://arxiv.org/abs/2312.10997"),
        _c("blog", "https://spotintelligence.com/2026/04/09/moe/"),
        _c("mindstudio", "https://www.mindstudio.ai/blog/moe"),
        _c("nips", "https://papers.nips.cc/paper_files/paper/2022/..."),
    ]
    kept = enforce_site_constraint(cands, "arxiv.org")
    assert len(kept) == 1
    assert kept[0]["uri"] == "https://arxiv.org/abs/2312.10997"


def test_enforce_site_constraint_keeps_subdomains():
    cands = [
        _c("docs", "https://docs.python.org/3/library/asyncio.html"),
        _c("www subdomain", "https://www.docs.python.org/3/"),
        _c("off", "https://realpython.com/python-asyncio/"),
    ]
    kept = enforce_site_constraint(cands, "docs.python.org")
    hosts = sorted({c["uri"] for c in kept})
    assert "https://realpython.com/python-asyncio/" not in hosts
    assert len(kept) == 2


def test_enforce_site_constraint_none_passes_through():
    cands = [_c("a", "https://a.com"), _c("b", "https://b.com")]
    assert enforce_site_constraint(cands, None) == cands


def test_enforce_site_constraint_empty_candidates():
    assert enforce_site_constraint([], "arxiv.org") == []
    assert enforce_site_constraint(None, "arxiv.org") == []


def test_enforce_site_constraint_strips_www_prefix():
    cands = [_c("a", "https://www.arxiv.org/abs/1234.5678")]
    assert len(enforce_site_constraint(cands, "arxiv.org")) == 1


# ── Phase 2: source-class primary/secondary/reject expectations ──────────────


def test_class_primary_sources_outrank_mirrors():
    """When a primary research host and a mirror both appear, primary wins."""
    cands = annotate([
        _c("mirror copy", "https://paperswithcode.com/paper/some-paper"),
        _c("arxiv abs", "https://arxiv.org/abs/2312.10997"),
    ])
    ranked = rerank(cands)
    assert ranked[0]["uri"] == "https://arxiv.org/abs/2312.10997"
    assert ranked[0]["source_class"] == CLASS_RESEARCH


def test_class_seo_dropped_official_docs_kept():
    cands = annotate([
        _c("top 10 python tips", "https://somesite.example/blog/top-10-python-tips"),
        _c("docs", "https://docs.python.org/3/tutorial/index.html"),
    ])
    ranked = rerank(cands, drop_classes={CLASS_SEO})
    uris = [c["uri"] for c in ranked]
    assert "https://docs.python.org/3/tutorial/index.html" in uris
    assert all("top-10-python-tips" not in u for u in uris)


def test_class_mirror_kept_but_demoted():
    cands = annotate([
        _c("mirror", "https://paperswithcode.com/paper/x"),
        _c("unknown", "https://randomblog.example/x"),
    ])
    ranked = rerank(cands)
    # Mirror is demoted (-2) below unknown (0), so unknown appears first.
    assert ranked[0]["uri"] == "https://randomblog.example/x"
    mirror_candidate = next(c for c in ranked if c["source_class"] == CLASS_MIRROR)
    assert mirror_candidate["source_score"] < 0
