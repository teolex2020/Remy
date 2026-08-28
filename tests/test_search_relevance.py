from remy.core.search_relevance import assess_query_relevance, query_terms


def test_query_terms_remove_operators_and_generic_words():
    terms = query_terms("site:docs.python.org best Python async context manager")

    assert "site" not in terms
    assert "best" not in terms
    assert {"python", "async", "context", "manager"}.issubset(set(terms))


def test_title_and_content_overlap_produce_explainable_score():
    result = assess_query_relevance(
        "distributed database consistency",
        title="Distributed database consistency models",
        content="This document compares consistency guarantees across replicated databases.",
    )

    assert result["relevant"] is True
    assert result["score"] > 0.7
    assert set(result["matched_terms"]) == {"distributed", "database", "consistency"}


def test_unrelated_content_is_marked_insufficient():
    result = assess_query_relevance(
        "distributed database consistency",
        title="Cooking pasta",
        content="Boil salted water and prepare a tomato sauce.",
    )

    assert result["relevant"] is False
    assert result["score"] == 0.0
    assert result["reason"] == "insufficient_query_overlap"
