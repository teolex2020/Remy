"""Unit tests for retrieval.evidence (Phase 3)."""

from __future__ import annotations

from remy.core.retrieval.evidence import (
    Artifact,
    Candidate,
    EvidencePacket,
    build_packet,
    extract_claim_identifiers,
    verify_claim_against_packet,
)


def _artifact(url: str, title: str = "", content: str = "", site: str = "") -> dict:
    return {"url": url, "title": title, "content": content, "site": site}


# ── Host match ────────────────────────────────────────────────────────────


def test_host_match_exact():
    pkt = build_packet(
        _artifact("https://arxiv.org/abs/2412.15803", "WebLLM: In-Browser LLM Inference"),
        requested_url="https://arxiv.org/abs/2412.15803",
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["host_match"] == "ok"
    assert pkt.ok


def test_host_match_mismatch_mirror():
    pkt = build_packet(
        _artifact("https://researchgate.net/publication/999", "Some paper"),
        requested_url="https://arxiv.org/abs/2412.15803",
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["host_match"] == "mismatch"
    assert pkt.has_mismatch


# ── Title match ───────────────────────────────────────────────────────────


def test_title_match_ok_when_overlap():
    pkt = build_packet(
        _artifact("https://arxiv.org/abs/2412.15803", "WebLLM In-Browser LLM Inference"),
        requested_url="https://arxiv.org/abs/2412.15803",
        expected_title="WebLLM In-Browser LLM Inference Paper",
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["title_match"] == "ok"


def test_title_match_mismatch_when_unrelated():
    # This is the c05 adversarial case from the benchmark: real URL, wrong claim.
    pkt = build_packet(
        _artifact("https://arxiv.org/abs/2412.15803", "WebLLM: In-Browser LLM Inference"),
        requested_url="https://arxiv.org/abs/2412.15803",
        expected_title="Thermodynamic Epistemic Governance of Quantum Societies",
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["title_match"] == "mismatch"
    assert pkt.has_mismatch


def test_title_match_unknown_when_no_claim():
    pkt = build_packet(_artifact("https://arxiv.org/abs/2312.10997", "RAG Survey"))
    names = {c.name for c in pkt.identity_checks}
    assert "title_match" not in names


# ── Identifier present ────────────────────────────────────────────────────


def test_identifier_present_arxiv():
    pkt = build_packet(
        _artifact(
            "https://arxiv.org/abs/2312.10997",
            "Retrieval Augmented Generation Survey",
            content="This paper, 2312.10997, surveys RAG methods ...",
        ),
        expected_identifier="2312.10997",
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["identifier_present"] == "ok"


def test_identifier_absent_is_mismatch():
    pkt = build_packet(
        _artifact(
            "https://arxiv.org/abs/2412.15803",
            "WebLLM: In-Browser LLM Inference",
            content="Abstract. We present WebLLM, a browser-native inference stack ...",
        ),
        expected_identifier="2401.99999",
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["identifier_present"] == "mismatch"
    assert pkt.has_mismatch


# ── Source class preservation ─────────────────────────────────────────────


def test_source_class_preserved_research_to_research():
    candidate = {
        "title": "t",
        "uri": "https://arxiv.org/abs/2312.10997",
        "source_class": "research",
        "source_score": 3,
    }
    pkt = build_packet(
        _artifact("https://arxiv.org/abs/2312.10997", "RAG Survey"),
        requested_url="https://arxiv.org/abs/2312.10997",
        candidate=candidate,
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["source_class_preserved"] == "ok"
    assert pkt.source_class == "research"


def test_source_class_drift_research_to_mirror_is_mismatch():
    candidate = {
        "title": "t",
        "uri": "https://arxiv.org/abs/2312.10997",
        "source_class": "research",
        "source_score": 3,
    }
    pkt = build_packet(
        _artifact("https://www.researchgate.net/publication/12345", "Mirror copy"),
        requested_url="https://arxiv.org/abs/2312.10997",
        candidate=candidate,
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["source_class_preserved"] == "mismatch"
    assert pkt.has_mismatch


# ── Packet shape + dict serialization ─────────────────────────────────────


def test_packet_to_dict_roundtrips_core_fields():
    pkt = build_packet(
        _artifact("https://docs.python.org/3/library/asyncio.html", "asyncio docs"),
        requested_url="https://docs.python.org/3/library/asyncio.html",
    )
    d = pkt.to_dict()
    assert d["host"] == "docs.python.org"
    assert d["source_class"] == "official_docs"
    assert d["title"] == "asyncio docs"
    assert d["ok"] is True
    assert isinstance(d["identity_checks"], list)


def test_packet_ok_when_no_checks_fail():
    pkt = build_packet(_artifact("https://example.com/", ""))
    # No explicit checks requested; host unknown class but no mismatch.
    assert pkt.ok
    assert not pkt.has_mismatch


def test_candidate_from_dict_preserves_fields():
    c = Candidate.from_dict({
        "title": "t",
        "uri": "https://arxiv.org/abs/2312.10997",
        "snippet": "s",
        "source_class": "research",
        "source_score": 3,
        "source_signals": ["research_host"],
    })
    assert c.source_class == "research"
    assert c.source_score == 3
    assert "research_host" in c.source_signals


def test_artifact_from_dict_missing_fields_ok():
    a = Artifact.from_dict({"url": "https://x.test/"})
    assert a.url == "https://x.test/"
    assert a.title == ""
    assert a.content == ""


# ── Phase 3: author mismatch ─────────────────────────────────────────────────


def _artifact_with_author(url, title, author, content=""):
    return {"url": url, "title": title, "author": author, "content": content}


def test_author_match_ok_on_surname_overlap():
    pkt = build_packet(
        _artifact_with_author(
            "https://arxiv.org/abs/2312.10997",
            "RAG Survey",
            "Gao, Yunfan and Liang, Xiaoxi",
        ),
        expected_authors="Yunfan Gao et al",
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["author_match"] == "ok"


def test_author_match_mismatch_on_wrong_authors():
    pkt = build_packet(
        _artifact_with_author(
            "https://arxiv.org/abs/2412.15803",
            "WebLLM",
            "Ruan and Wang and Zhao",
        ),
        expected_authors="Charlotte Bronte and Emily Dickinson",
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["author_match"] == "mismatch"
    assert pkt.has_mismatch


def test_author_match_unknown_when_fetched_author_blank():
    pkt = build_packet(
        _artifact_with_author(
            "https://arxiv.org/abs/2312.10997",
            "RAG Survey",
            "",
        ),
        expected_authors="Yunfan Gao",
    )
    names = {c.name: c.status for c in pkt.identity_checks}
    assert names["author_match"] == "unknown"
    # Unknown is not a mismatch — packet should still be ok.
    assert pkt.ok


# ── Phase 3: claim_span passthrough ──────────────────────────────────────────


def test_claim_span_preserved_on_packet():
    pkt = build_packet(
        _artifact("https://arxiv.org/abs/2312.10997", "RAG Survey"),
        claim_span="This paper surveys retrieval-augmented generation methods.",
    )
    assert pkt.claim_span.startswith("This paper surveys")
    assert pkt.to_dict()["claim_span"].startswith("This paper surveys")


def test_claim_span_defaults_to_empty():
    pkt = build_packet(_artifact("https://example.com/", ""))
    assert pkt.claim_span == ""


# ── Phase 3: claim identifier extraction ─────────────────────────────────────


def test_extract_claim_identifiers_arxiv():
    ids = extract_claim_identifiers("As shown in 2312.10997, the survey covers...")
    assert "2312.10997" in ids["arxiv_ids"]
    assert ids["dois"] == []


def test_extract_claim_identifiers_doi():
    ids = extract_claim_identifiers("See 10.1145/1234567.8901234 for details.")
    assert ids["dois"] == ["10.1145/1234567.8901234"]


def test_extract_claim_identifiers_mixed_and_dedup():
    ids = extract_claim_identifiers(
        "Two refs: 2312.10997 and 2312.10997 plus 10.1000/xyz123"
    )
    assert ids["arxiv_ids"] == ["2312.10997"]
    assert "10.1000/xyz123" in ids["dois"]


def test_extract_claim_identifiers_empty():
    assert extract_claim_identifiers("no identifiers here") == {"arxiv_ids": [], "dois": []}
    assert extract_claim_identifiers("") == {"arxiv_ids": [], "dois": []}


# ── Phase 3: verify_claim_against_packet ─────────────────────────────────────


def test_verify_no_packet_is_no_evidence():
    v = verify_claim_against_packet("any claim", None)
    assert v.status == "no_evidence"
    assert v.packet is None


def test_verify_ok_packet_no_identifiers_is_grounded():
    pkt = build_packet(
        _artifact("https://docs.python.org/3/library/asyncio.html", "asyncio"),
        requested_url="https://docs.python.org/3/library/asyncio.html",
    )
    v = verify_claim_against_packet("asyncio has TaskGroup support", pkt)
    assert v.status == "grounded"


def test_verify_mismatch_surfaces_reference_identity_mismatch():
    # c05 adversarial: real URL, wrong claimed title.
    pkt = build_packet(
        _artifact("https://arxiv.org/abs/2412.15803", "WebLLM: In-Browser LLM Inference"),
        requested_url="https://arxiv.org/abs/2412.15803",
        expected_title="Thermodynamic Epistemic Governance of Quantum Societies",
    )
    v = verify_claim_against_packet("...claim...", pkt)
    assert v.status == "reference_identity_mismatch"
    assert any("title_match" in r for r in v.reasons)


def test_verify_claim_mentions_arxiv_id_not_in_content():
    # Claim says "2401.99999" but we fetched content about 2412.15803.
    pkt = build_packet(
        _artifact(
            "https://arxiv.org/abs/2412.15803",
            "WebLLM: In-Browser LLM Inference",
            content="Abstract. We present WebLLM, a browser-native inference stack.",
        ),
        requested_url="https://arxiv.org/abs/2412.15803",
    )
    v = verify_claim_against_packet(
        "Per 2401.99999, this result follows directly.", pkt
    )
    assert v.status == "reference_identity_mismatch"
    assert any("2401.99999" in r for r in v.reasons)


def test_verify_claim_arxiv_id_matches_content():
    pkt = build_packet(
        _artifact(
            "https://arxiv.org/abs/2312.10997",
            "RAG Survey",
            content="This paper, 2312.10997, surveys retrieval-augmented generation.",
        ),
        requested_url="https://arxiv.org/abs/2312.10997",
    )
    v = verify_claim_against_packet("As noted in 2312.10997, RAG helps.", pkt)
    assert v.status == "grounded"


def test_verify_verdict_to_dict_serialises():
    pkt = build_packet(_artifact("https://example.com/", "x"))
    v = verify_claim_against_packet("claim", pkt)
    d = v.to_dict()
    assert d["status"] in {"grounded", "unverifiable"}
    assert "reasons" in d
    assert d["packet"] is not None
