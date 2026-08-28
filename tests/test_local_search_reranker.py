from remy.core.local_search_reranker import (
    LocalMultilingualReranker,
    multilingual_tokens,
    token_similarity,
)


def test_unicode_tokenizer_keeps_technical_cyrillic_terms_and_drops_function_words():
    tokens = multilingual_tokens("Як налаштувати резервне копіювання PostgreSQL у Windows?")

    assert "як" not in tokens
    assert "у" not in tokens
    assert {"налаштувати", "резервне", "копіювання", "postgresql", "windows"}.issubset(
        set(tokens)
    )


def test_token_similarity_handles_transposition_and_cyrillic_inflection():
    assert token_similarity("databse", "database") > 0.9
    assert token_similarity("налаштуваня", "налаштування") > 0.9
    assert token_similarity("бази", "база") > 0.9
    assert token_similarity("postgresql", "cooking") == 0.0


def test_reranker_rewards_typo_recovery_in_title():
    reranker = LocalMultilingualReranker()
    scores = reranker.score_candidates(
        "daatbase transaction isolation levels",
        [
            {
                "title": "Database transaction isolation levels",
                "snippet": "Official database transaction reference.",
                "uri": "https://postgresql.org/docs/isolation",
            },
            {
                "title": "Transaction isolation levels",
                "snippet": "An unrelated overview that does not identify a database.",
                "uri": "https://blog.test/isolation",
            },
        ],
    )

    assert scores[0]["score"] > scores[1]["score"]
    assert scores[0]["matched_fuzzy"]["daatbase"] == "database"


def test_reranker_matches_ukrainian_word_forms_without_translation_service():
    reranker = LocalMultilingualReranker()
    score = reranker.score_candidates(
        "відновлення резервної копії бази даних",
        [
            {
                "title": "Як відновити резервну копію бази даних",
                "snippet": "Інструкція з відновлення PostgreSQL.",
                "uri": "https://docs.test/backup-restore",
            }
        ],
    )[0]

    assert score["score"] > 0.75
    assert score["matched_fuzzy"]["резервної"] == "резервну"
    assert not score["missing_terms"]


def test_reranker_is_explainable_for_unrelated_result():
    score = LocalMultilingualReranker().score_candidates(
        "kubernetes rolling deployment",
        [
            {
                "title": "Pasta cooking guide",
                "snippet": "Boil water and prepare sauce.",
                "uri": "https://example.test/food",
            }
        ],
    )[0]

    assert score["score"] == 0.0
    assert score["missing_terms"] == ["kubernetes", "rolling", "deployment"]
