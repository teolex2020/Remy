from remy.config.settings import settings
from remy.core.learning_review import (
    decide_learning_review,
    list_learning_reviews,
    stage_learning_review,
)


def test_only_explicit_correction_or_preference_is_staged(tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    try:
        assert stage_learning_review(
            session_id="s1", user_text="Розкажи мені про цей сайт", assistant_text="Draft"
        ) is None
        item = stage_learning_review(
            session_id="s1",
            user_text="Не вигадуй опис сайту, якщо інструмент не повернув його вміст.",
            assistant_text="Виправлю.",
        )
        assert item["category"] == "correction"
        assert item["status"] == "pending"
        assert len(list_learning_reviews()) == 1
    finally:
        settings.DATA_DIR = original


def test_review_requires_separate_decision_and_keeps_audit_record(tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    try:
        item = stage_learning_review(
            session_id="s2",
            user_text="Запам'ятай: завжди показуй джерело для опису сайту.",
        )
        decided = decide_learning_review(
            item["review_id"], status="approved", memory_record_id="memory-1"
        )

        assert decided["status"] == "approved"
        assert decided["memory_record_id"] == "memory-1"
        assert list_learning_reviews(status="pending") == []
        assert list_learning_reviews(status="approved")[0]["review_id"] == item["review_id"]
    finally:
        settings.DATA_DIR = original
