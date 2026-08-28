from remy.core.source_provenance_graph import (
    apply_provenance_to_marginal_evidence,
    build_source_provenance_graph,
)


def _source(url, unique, **extra):
    return {
        "url": url,
        "title": "Database consistency evidence",
        "content": (
            "Database consistency replication evidence covers quorum and transactions. "
            f"{unique} " * 18
        ),
        **extra,
    }


def test_three_independent_sources_create_three_roots():
    graph = build_source_provenance_graph(
        [
            _source("https://one.example/a", "majority failure tolerance"),
            _source("https://two.example/b", "serializable anomaly prevention"),
            _source("https://three.example/c", "leader election log replication"),
        ]
    )

    assert graph["node_count"] == 3
    assert graph["independent_root_count"] == 3


def test_explicit_syndication_collapses_different_domains_to_one_root():
    original = "https://origin.example/report"
    graph = build_source_provenance_graph(
        [
            _source("https://one.example/a", "edition one", original_url=original),
            _source("https://two.example/b", "edition two", original_source_url=original),
            _source("https://three.example/c", "edition three", syndicated_from=original),
        ]
    )

    assert graph["independent_root_count"] == 1
    assert {node["evidence_root"] for node in graph["nodes"]} == {original}
    assert sum(edge["relation"] == "derived_from" for edge in graph["edges"]) == 3


def test_according_to_same_publication_is_not_independent_consensus():
    origin = "https://origin.example/study"
    graph = build_source_provenance_graph(
        [
            _source("https://news-one.example/a", f"according to {origin} analysis one"),
            _source("https://news-two.example/b", f"according to {origin} analysis two"),
        ]
    )

    assert graph["independent_root_count"] == 1
    assert graph["derived_source_count"] == 2


def test_near_identical_content_creates_duplicate_edge_and_one_root():
    content = _source("https://one.example/a", "identical report material")["content"]
    graph = build_source_provenance_graph(
        [
            {"url": "https://one.example/a", "content": content},
            {"url": "https://two.example/b", "content": content},
        ]
    )

    assert graph["independent_root_count"] == 1
    assert any(edge["relation"] == "near_duplicate_of" for edge in graph["edges"])


def test_official_documentation_is_classified_as_primary():
    graph = build_source_provenance_graph(
        [_source("https://docs.python.org/3/library/pathlib.html", "pathlib api reference")]
    )

    assert graph["nodes"][0]["authority_role"] == "primary"
    assert graph["primary_root_count"] == 1


def test_mirror_packet_is_classified_as_aggregator():
    graph = build_source_provenance_graph(
        [
            _source(
                "https://mirror.example/copy",
                "copied report",
                evidence_packet={"source_class": "mirror", "ok": True},
            )
        ]
    )

    assert graph["nodes"][0]["authority_role"] == "aggregator"


def test_identity_mismatch_reduces_authority_score():
    good = build_source_provenance_graph(
        [_source("https://example.org/a", "good", evidence_packet={"ok": True})]
    )
    bad = build_source_provenance_graph(
        [_source("https://example.org/b", "bad", evidence_packet={"has_mismatch": True})]
    )

    assert bad["nodes"][0]["authority_score"] < good["nodes"][0]["authority_score"]


def test_generic_reference_creates_citation_edge_without_forcing_derivation():
    cited = "https://study.example/original"
    graph = build_source_provenance_graph(
        [_source("https://analysis.example/a", f"discussion reference {cited}")]
    )

    assert any(edge["relation"] == "cites" and edge["to"] == cited for edge in graph["edges"])
    assert graph["independent_root_count"] == 1


def test_provenance_gate_reopens_false_three_domain_consensus():
    origin = "https://origin.example/report"
    sources = [
        _source(f"https://site{i}.example/a", f"unique commentary {i}", original_url=origin)
        for i in range(3)
    ]
    graph = build_source_provenance_graph(sources)
    marginal = {
        "sufficient": True,
        "stop_recommended": True,
        "decision": "stop_sufficient",
        "saturated": False,
        "accepted_urls": [source["url"] for source in sources],
        "reasons": [],
        "repair_queries": [],
    }
    updated = apply_provenance_to_marginal_evidence(marginal, graph)

    assert updated["sufficient"] is False
    assert updated["decision"] == "continue_provenance_diversify"
    assert updated["provenance"]["accepted_root_count"] == 1
    assert "not_enough_independent_evidence_roots" in updated["reasons"]


def test_provenance_gate_preserves_real_independent_consensus():
    sources = [
        _source("https://site1.example/a", "quorum majority failure tolerance"),
        _source("https://site2.example/a", "serializable isolation anomaly prevention"),
        _source("https://site3.example/a", "leader election replicated consensus log"),
    ]
    graph = build_source_provenance_graph(sources)
    marginal = {
        "sufficient": True,
        "accepted_urls": [source["url"] for source in sources],
    }
    updated = apply_provenance_to_marginal_evidence(marginal, graph)

    assert updated["sufficient"] is True
    assert updated["provenance"]["accepted_root_count"] == 3
