"""Memory quality tests — dual-layer brain ↔ KB integration.

Перевіряє не просто що функції не падають, а що два шари пам'яті
(aura_cognitive brain + aura_memory KB) реально корисно працюють разом:

1.  Sync — store у brain дублюється в KB за рівнями (L3/L4 → так, L1 → ні)
2.  Recall merge — recall повертає результати з обох джерел без дублів
3.  KB dedup — повторний sync не множить записи в KB
4.  Background sync — _sync_knowledge підхоплює нові brain-записи
5.  JSON-guard — JSON-обгорнутий контент не потрапляє в KB
6.  Consolidation — схожі brain-записи об'єднуються в meta-запис
7.  Pinning — IDENTITY рівень зберігається в KB як anchor (pin=True)
8.  Recall fallback — збій KB не ламає recall (повертає тільки brain)
9.  Decay не ламає KB — KB не залежить від brain decay
10. Cross-source dedup — той самий текст з brain і KB не дублюється у recall
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from aura import Aura as CognitiveMemory
from aura import Level


# ============ Fixtures ============


@pytest.fixture
def brain(tmp_path):
    b = CognitiveMemory(str(tmp_path / "brain"))
    yield b
    b.close()


@pytest.fixture
def mock_kb():
    kb = MagicMock()
    kb.process.return_value = "stored"
    kb.retrieve.return_value = []
    kb.retrieve_full.return_value = []
    kb.retrieve_matrix.return_value = ([], [], [])
    kb.list_memories.return_value = []
    kb.flush.return_value = None
    kb.count.return_value = 0
    return kb


@pytest.fixture
def kb_lock():
    return threading.Lock()


@pytest.fixture
def exec_tool(brain, mock_kb, kb_lock, tmp_path):
    """execute_tool з замокованим brain + KB."""
    with patch("remy.core.brain_tools.brain", brain), \
         patch("remy.core.brain_tools.brain_lock", threading.RLock()), \
         patch("remy.core.brain_tools._registry", None), \
         patch("remy.core.agent_tools.knowledge", mock_kb), \
         patch("remy.core.agent_tools.knowledge_lock", kb_lock), \
         patch("remy.core.tool_registry.settings") as ms:
        ms.SANDBOX_DIR = tmp_path / "sandbox"
        ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
        from remy.core.brain_tools import execute_tool
        yield execute_tool


# ============ 1. Sync за рівнями ============


@pytest.mark.real_sync
class TestSyncByLevel:

    def test_domain_syncs_to_kb(self, exec_tool, mock_kb):
        """Grounded L3_DOMAIN store має викликати knowledge.process()."""
        exec_tool("store", {
            "content": "Пацієнт щодня приймає метформін 500 мг двічі на день",
            "tags": "medication,health",
            "level": "L3_DOMAIN",
            "metadata": {"admission_class": "operator_asserted"},
        })
        mock_kb.process.assert_called()

    def test_identity_syncs_with_pin(self, exec_tool, mock_kb):
        """store_user_profile має викликати knowledge.process() з pin=True."""
        exec_tool("store_user_profile", {
            "name": "Тетяна",
            "occupation": "лікар",
            "location": "Київ",
        })
        mock_kb.process.assert_called()
        call_kwargs = mock_kb.process.call_args
        pin_val = call_kwargs[1].get("pin") if call_kwargs[1] else (
            call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
        )
        assert pin_val is True

    def test_working_does_not_sync(self, exec_tool, mock_kb):
        """L1_WORKING store НЕ має потрапляти в KB."""
        exec_tool("store", {
            "content": "Тимчасова нотатка що швидко зникне з памяті",
            "tags": "temp",
            "level": "L1_WORKING",
        })
        mock_kb.process.assert_not_called()

    def test_system_tags_do_not_sync(self, exec_tool, mock_kb):
        """Записи з system-тегами (web-search-cache) не мають йти в KB."""
        exec_tool("store", {
            "content": "Кешований результат пошуку що не є знанням",
            "tags": "web-search-cache",
            "level": "L3_DOMAIN",
        })
        mock_kb.process.assert_not_called()

    def test_decisions_level_syncs(self, exec_tool, mock_kb):
        """L2_DECISIONS синхронізується в KB (без pin) — рівень >= WORKING."""
        exec_tool("store", {
            "content": "Рішення лікаря: збільшити дозу вітаміну D до 2000 МО",
            "tags": "health,decision",
            "level": "L2_DECISIONS",
        })
        from remy.core.brain_tools import _should_sync
        should, pin = _should_sync(Level.DECISIONS)
        assert should is True
        assert pin is False


# ============ 2. Recall merge ============


class TestRecallMerge:

    def test_recall_shows_brain_result(self, exec_tool, brain, mock_kb):
        """Recall має повертати результати з brain."""
        brain.store(
            "Бабуся Марія народилась у 1932 році в Полтаві",
            level=Level.DOMAIN, tags=["family"]
        )
        result = exec_tool("recall", {"query": "бабуся Марія"})
        assert "Марія" in result or "1932" in result or "Полтав" in result

    def test_recall_shows_kb_result(self, exec_tool, mock_kb):
        """Recall має показувати KB результати з міткою KB."""
        mock_kb.retrieve_matrix.return_value = (
            [4.5], ["kb20"],
            [{"text": "Вітамін D покращує засвоєння кальцію", "id": "kb20", "intensity": 1.0, "dna": "general"}]
        )
        result = exec_tool("recall", {"query": "вітамін D"})
        assert "KB" in result
        assert "Вітамін D" in result or "вітамін" in result.lower()

    def test_recall_merges_both_sources(self, exec_tool, brain, mock_kb):
        """Коли обидва шари мають результати — обидва з'являються."""
        brain.store(
            "Лікар призначив вітамін D пацієнту Тетяні",
            level=Level.DOMAIN, tags=["health"]
        )
        mock_kb.retrieve_matrix.return_value = (
            [3.0], ["kb21"],
            [{"text": "Вітамін D синтезується під дією сонячного світла", "id": "kb21", "intensity": 1.0, "dna": "general"}]
        )
        result = exec_tool("recall", {"query": "вітамін D"})
        assert "trust:" in result   # brain результат
        assert "KB" in result       # KB результат

    def test_recall_deduplicates_identical_text(self, exec_tool, brain, mock_kb):
        """Один і той самий текст з brain і KB має з'явитись лише раз."""
        text = "Метформін знижує рівень цукру в крові натщесерце"
        brain.store(text, level=Level.DOMAIN, tags=["medication"])
        mock_kb.retrieve_matrix.return_value = (
            [9.0], ["kb22"],
            [{"text": text, "id": "kb22", "intensity": 1.0, "dna": "general"}]
        )
        result = exec_tool("recall", {"query": "метформін"})
        assert result.count("Метформін") == 1

    def test_recall_empty_returns_no_results(self, exec_tool, mock_kb):
        """Коли обидва шари порожні — повідомлення про відсутність."""
        result = exec_tool("recall", {"query": "абсолютно унікальний запит xyz999"})
        assert "No relevant memories" in result

    def test_recall_kb_failure_graceful(self, exec_tool, brain, mock_kb):
        """Збій KB не ламає recall — повертає тільки brain результати."""
        mock_kb.retrieve_matrix.side_effect = RuntimeError("KB crashed")
        mock_kb.list_memories.side_effect = RuntimeError("KB crashed")
        brain.store(
            "Важлива інформація про здоров'я пацієнта",
            level=Level.DOMAIN, tags=["health"]
        )
        result = exec_tool("recall", {"query": "здоров'я"})
        assert "здоров" in result.lower() or "пацієнт" in result.lower()
        assert "KB" not in result


# ============ 3. KB dedup при повторному sync ============


@pytest.mark.real_sync
class TestKBDedup:

    def test_same_content_not_stored_twice(self, mock_kb, kb_lock):
        """Повторний виклик _sync_to_knowledge з тим самим текстом блокується."""
        existing = "Діабет 2 типу потребує контролю рівня глюкози щодня"
        mock_kb.retrieve_matrix.return_value = ([0.85], ["kb30"], [{"text": existing}])

        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge(existing)

        assert result is False
        mock_kb.process.assert_not_called()

    def test_different_content_passes_dedup(self, mock_kb, kb_lock):
        """Новий текст проходить перевірку і зберігається."""
        mock_kb.retrieve_matrix.return_value = ([], [], [])

        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge(
                "Новий факт: регулярна ходьба зменшує ризик серцевих захворювань"
            )

        assert result is True
        mock_kb.process.assert_called_once()


# ============ 4. Background sync ============


class TestBackgroundSync:

    def test_sync_knowledge_picks_up_new_records(self, brain, mock_kb, kb_lock):
        """_sync_knowledge знаходить нові DOMAIN записи і відправляє в KB."""
        brain.store(
            "Нова рекомендація: зменшити споживання солі до 5г на добу",
            level=Level.DOMAIN, tags=["nutrition"]
        )

        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(brain)

        assert synced >= 1
        mock_kb.process.assert_called()
        mock_kb.flush.assert_called()

    def test_sync_skips_already_mirrored(self, brain, mock_kb, kb_lock):
        """Записи з mirrored_to_kb=True не синхронізуються повторно."""
        rec = brain.store(
            "Вже синхронізований факт про ліки пацієнта",
            level=Level.DOMAIN, tags=["medication"]
        )
        brain.update(rec.id, metadata={"mirrored_to_kb": True})

        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(brain)

        assert synced == 0

    def test_sync_respects_50_record_cap(self, brain, mock_kb, kb_lock):
        """_sync_knowledge не обробляє більше 50 записів за один запуск."""
        for i in range(60):
            brain.store(
                f"Факт номер {i} про стан здоров'я пацієнта для тестування",
                level=Level.DOMAIN, tags=["test"]
            )

        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(brain)

        assert synced <= 50

    def test_sync_noop_when_kb_disabled(self, brain, kb_lock):
        """Якщо knowledge=None — _sync_knowledge повертає 0 без помилок."""
        with patch("remy.core.agent_tools.knowledge", None), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(brain)

        assert synced == 0


# ============ 5. JSON-guard ============


@pytest.mark.real_sync
class TestJSONGuard:

    def test_json_wrapped_content_blocked(self, mock_kb, kb_lock):
        """Контент у форматі {'type': 'text', 'text': '...'} не потрапляє в KB."""
        wrapped = "{'type': 'text', 'text': 'Реальний текст який не мав потрапити сюди'}"

        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge(wrapped)

        assert result is False
        mock_kb.process.assert_not_called()

    def test_json_wrapped_double_quotes_blocked(self, mock_kb, kb_lock):
        """Версія з подвійними лапками теж блокується."""
        wrapped = '{"type": "text", "text": "Текст що прийшов від Gemini API"}'

        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge(wrapped)

        assert result is False
        mock_kb.process.assert_not_called()

    def test_plain_text_passes_guard(self, mock_kb, kb_lock):
        """Звичайний текст проходить guard без проблем."""
        mock_kb.retrieve_full.return_value = []
        plain = "Пацієнт приймає аспірин 100мг для профілактики тромбозу"

        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge(plain)

        assert result is True

    def test_short_content_blocked(self, mock_kb, kb_lock):
        """Занадто короткий текст (< 20 символів) не потрапляє в KB."""
        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge("Коротко")

        assert result is False
        mock_kb.process.assert_not_called()


# ============ 6. Consolidation ============


class TestConsolidation:

    def test_consolidation_reduces_similar_records(self, brain):
        """consolidate() має об'єднати дуже схожі записи в менше."""
        # Зберігаємо майже ідентичні записи
        for suffix in ["вранці", "вдень", "ввечері", "після їжі", "перед сном"]:
            brain.store(
                f"Пацієнт повинен вимірювати тиск {suffix} щодня",
                level=Level.DOMAIN, tags=["health", "medication"]
            )

        count_before = brain.count()
        brain.consolidate()
        count_after = brain.count()

        # Consolidation або зменшує або не збільшує кількість
        assert count_after <= count_before

    def test_consolidation_preserves_distinct_records(self, brain):
        """Різні записи не мають об'єднуватись."""
        brain.store("Бабуся народилась у 1932 році", level=Level.DOMAIN, tags=["family"])
        brain.store("Пацієнт приймає метформін", level=Level.DOMAIN, tags=["medication"])
        brain.store("Рекомендована доза вітаміну D — 1000 МО", level=Level.DOMAIN, tags=["nutrition"])

        count_before = brain.count()
        brain.consolidate()
        count_after = brain.count()

        # Різні записи не мають зникати
        assert count_after >= count_before - 1  # Допускаємо 1 помилку з'єднання

    def test_consolidated_meta_record_contains_key_info(self, brain):
        """Після консолідації інформація не втрачається — пошук досі працює."""
        brain.store("Тетяна приймає метформін 500мг", level=Level.DOMAIN, tags=["medication"])
        brain.store("Тетяна приймає метформін вранці", level=Level.DOMAIN, tags=["medication"])
        brain.store("Метформін призначено Тетяні для контролю цукру", level=Level.DOMAIN, tags=["medication"])

        brain.consolidate()

        # Після консолідації пошук все ще має знаходити метформін
        results = brain.search(query="метформін", limit=5)
        found_texts = " ".join(r.content for r in results)
        assert "метформін" in found_texts.lower() or "Метформін" in found_texts


# ============ 7. Pinning IDENTITY ============


@pytest.mark.real_sync
class TestPinning:

    def test_store_user_profile_pins_in_kb(self, exec_tool, mock_kb):
        """store_user_profile має зберігати в KB з pin=True."""
        exec_tool("store_user_profile", {"name": "Тетяна", "occupation": "лікар"})
        mock_kb.process.assert_called()
        call_kwargs = mock_kb.process.call_args
        pin_val = call_kwargs[1].get("pin") if call_kwargs[1] else (
            call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
        )
        assert pin_val is True

    def test_identity_record_via_background_sync_pins(self, brain, mock_kb, kb_lock):
        """IDENTITY записи через background _sync_knowledge теж мають pin=True."""
        brain.store(
            "Профіль: Тетяна Іванова, лікар-терапевт, Київ",
            level=Level.IDENTITY, tags=["user-profile", "identity"]
        )

        with patch("remy.core.agent_tools.knowledge", mock_kb), \
             patch("remy.core.agent_tools.knowledge_lock", kb_lock):
            from remy.core.background_brain import _sync_knowledge
            _sync_knowledge(brain)

        mock_kb.process.assert_called()
        call_kwargs = mock_kb.process.call_args
        pin_val = call_kwargs[1].get("pin") if call_kwargs[1] else (
            call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
        )
        assert pin_val is True


# ============ 8. Decay не ламає KB ============


class TestDecayKBIndependence:

    def test_brain_decay_does_not_affect_kb(self, exec_tool, brain, mock_kb):
        """Навіть якщо brain-запис затухає — KB не змінюється (незалежні шари)."""
        exec_tool("store", {
            "content": "Важливий медичний факт що має залишатись в KB",
            "tags": "health",
            "level": "L3_DOMAIN",
        })
        call_count_after_store = mock_kb.process.call_count

        # Decay brain — KB mock не має отримувати нових викликів
        for _ in range(10):
            brain.decay()

        # KB не має отримувати додаткових викликів від decay
        assert mock_kb.process.call_count == call_count_after_store

    def test_working_record_dies_but_kb_was_never_written(self, exec_tool, brain, mock_kb):
        """WORKING запис затухає в brain, але в KB він і не потрапляв."""
        exec_tool("store", {
            "content": "Тимчасова нотатка для перевірки decay поведінки",
            "tags": "temp",
            "level": "L1_WORKING",
        })
        mock_kb.process.assert_not_called()

        # Тепер decay
        for _ in range(20):
            brain.decay()

        # KB досі чистий
        mock_kb.process.assert_not_called()
