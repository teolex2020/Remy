# -*- coding: utf-8 -*-
"""Memory Quality Audit -- interaktyvna perevirka prodakshn bazy.

Запуск:
    cd <repo>/app
    .venv/Scripts/python.exe scripts/memory_audit.py

Що перевіряє:
    1. Brain stats         — скільки записів, рівні, теги
    2. KB stats            — скільки записів, чи є сміття
    3. Recall precision    — наскільки добре brain знаходить релевантні результати
    4. KB hit rate         — скільки запитів отримують KB результати
    5. Sync coverage       — скільки brain-записів вже синхронізовано в KB
    6. Duplicate rate      — скільки потенційних дублів в brain
    7. Decay health        — розподіл strength (живі vs слабкі vs мертві)
    8. Connection density  — скільки записів мають зв'язки (graph density)
    9. JSON garbage check  — чи є JSON-обгорнуті записи в KB
   10. Top tags            — найпопулярніші теги (що агент запам'ятовує найбільше)
"""

import os
import sys
import time
from pathlib import Path
from collections import Counter

# Add src/ to path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load env
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from aura import Aura as CognitiveMemory
from aura import Level


# ============ Config ============

BRAIN_PATH = ROOT / "data" / "brain"
KB_PATH = ROOT / "data" / "knowledge"

# Запити для перевірки recall precision
RECALL_TEST_QUERIES = [
    ("health",         ["ліки", "лікар", "здоров", "vitamin", "medication", "symptom", "health"]),
    ("family",         ["сім'я", "мама", "батько", "бабуся", "дідусь", "family", "mom", "dad"]),
    ("user profile",   ["ім'я", "name", "профіль", "profile", "user"]),
    ("medication",     ["метформін", "metformin", "таблетк", "pill", "dose", "дозу"]),
    ("nutrition",      ["їжа", "дієта", "diet", "вітамін", "vitamin", "калорі"]),
]

SEP = "-" * 60


def print_section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def fmt_level(level) -> str:
    names = {1: "WORKING", 2: "DECISIONS", 3: "DOMAIN", 4: "IDENTITY"}
    try:
        return names.get(int(level), str(level))
    except Exception:
        return str(level)


def pct(n, total) -> str:
    if total == 0:
        return "0%"
    return f"{n/total*100:.1f}%"


# ============ 1. Brain stats ============


def audit_brain(brain) -> dict:
    print_section("1. Brain Stats")

    total = brain.count()
    print(f"  Всього записів: {total}")

    if total == 0:
        print("  (brain порожній)")
        return {"total": 0}

    all_records = brain.list_records()
    print(f"  list_records повернув: {len(all_records)}")

    # Рівні
    level_counts = Counter()
    tag_counts = Counter()
    strength_buckets = {"strong (>0.7)": 0, "medium (0.3-0.7)": 0, "weak (<0.3)": 0}
    connected = 0
    mirrored = 0

    for r in all_records:
        level_counts[fmt_level(r.level)] += 1
        for t in (r.tags or []):
            tag_counts[t] += 1
        s = getattr(r, "strength", 1.0)
        if s > 0.7:
            strength_buckets["strong (>0.7)"] += 1
        elif s > 0.3:
            strength_buckets["medium (0.3-0.7)"] += 1
        else:
            strength_buckets["weak (<0.3)"] += 1
        if r.connections:
            connected += 1
        meta = getattr(r, "metadata", {}) or {}
        if meta.get("mirrored_to_kb"):
            mirrored += 1

    print("\n  Розподіл по рівнях:")
    for lvl, cnt in sorted(level_counts.items()):
        print(f"    {lvl:12s}: {cnt:4d}  ({pct(cnt, total)})")

    print("\n  Стан записів (strength):")
    for bucket, cnt in strength_buckets.items():
        bar = "█" * min(40, int(cnt / max(total, 1) * 40))
        print(f"    {bucket:22s}: {cnt:4d}  {bar}")

    print(f"\n  Записів зі зв'язками: {connected} ({pct(connected, total)})")
    print(f"  Синхронізовано в KB:  {mirrored} ({pct(mirrored, total)})")

    print("\n  Топ-15 тегів:")
    for tag, cnt in tag_counts.most_common(15):
        print(f"    [{tag}] × {cnt}")

    return {
        "total": total,
        "all_records": all_records,
        "level_counts": level_counts,
        "tag_counts": tag_counts,
        "connected": connected,
        "mirrored": mirrored,
    }


# ============ 2. KB stats ============


def audit_kb() -> dict:
    print_section("2. Knowledge Base (KB) Stats")

    try:
        from aura_memory import AuraMemory
    except ImportError:
        print("  ⚠ aura_memory не встановлено — KB вимкнений")
        return {"enabled": False}

    if not KB_PATH.exists():
        print(f"  ⚠ KB директорія не існує: {KB_PATH}")
        return {"enabled": False}

    kb = AuraMemory(str(KB_PATH))
    sdr_count = kb.count()
    records, total = kb.list_memories(limit=500)
    print(f"  SDR nodes (internal): {sdr_count}")
    print(f"  Logical records:      {total}")

    # Перевіримо на сміття
    json_wrapped = 0
    empty = 0
    short = 0
    sample_bad = []

    for r in records:
        text = r.get("text", "").strip()
        if not text:
            empty += 1
        elif text.startswith("{'type':") or text.startswith('{"type":'):
            json_wrapped += 1
            if len(sample_bad) < 3:
                sample_bad.append(text[:60])
        elif len(text) < 20:
            short += 1

    clean = total - json_wrapped - empty - short
    print(f"\n  Якість записів (перших 500):")
    print(f"    Чистих:          {clean}")
    print(f"    JSON-обгорнутих: {json_wrapped}  {'⚠ потребує cleanup' if json_wrapped > 0 else '✓'}")
    print(f"    Порожніх:        {empty}  {'⚠' if empty > 0 else '✓'}")
    print(f"    Занадто коротких:{short}  {'⚠' if short > 0 else '✓'}")

    if sample_bad:
        print("\n  Приклади поганих записів:")
        for s in sample_bad:
            print(f"    ↳ {s}...")

    # Розмір файлів
    total_size = sum(
        f.stat().st_size for f in KB_PATH.iterdir() if f.is_file()
    )
    print(f"\n  Розмір на диску: {total_size // 1024} KB")

    return {
        "enabled": True,
        "sdr_count": sdr_count,
        "logical_total": total,
        "json_wrapped": json_wrapped,
        "empty": empty,
        "clean": clean,
    }


# ============ 3. Recall precision ============


def audit_recall_precision(brain) -> dict:
    print_section("3. Recall Precision (brain)")

    if brain.count() == 0:
        print("  (brain порожній — пропускаємо)")
        return {}

    hits = 0
    total = len(RECALL_TEST_QUERIES)

    for query, keywords in RECALL_TEST_QUERIES:
        result = brain.recall(query, session_id="audit")
        found = any(kw.lower() in result.lower() for kw in keywords)
        status = "✓" if found else "✗"
        print(f"  {status} '{query}' → {'знайшов' if found else 'не знайшов'} ({len(result)} chars)")
        if not found and result != "No relevant memories found.":
            print(f"      → {result[:100]}...")
        if found:
            hits += 1

    precision = hits / total * 100
    print(f"\n  Precision: {hits}/{total} ({precision:.0f}%)")
    if precision < 50:
        print("  ⚠ Низька точність recall — можливо brain ще порожній або теги не відповідають")
    elif precision < 80:
        print("  ○ Помірна точність")
    else:
        print("  ✓ Гарна точність")

    return {"hits": hits, "total": total, "precision": precision}


# ============ 4. KB hit rate ============


def audit_kb_hit_rate() -> dict:
    print_section("4. KB Hit Rate")

    try:
        from aura_memory import AuraMemory
        kb = AuraMemory(str(KB_PATH))
    except Exception as e:
        print(f"  ⚠ KB недоступний: {e}")
        return {}

    hits = 0
    queries = [q for q, _ in RECALL_TEST_QUERIES]

    for query in queries:
        results = kb.retrieve_full(query[:100], top_k=3)
        found = bool(results)
        status = "✓" if found else "✗"
        top_score = results[0].get("score", 0) if found else 0
        print(f"  {status} '{query}' → {len(results)} results (top score: {top_score:.2f})")
        if found:
            hits += 1

    hit_rate = hits / len(queries) * 100
    print(f"\n  Hit rate: {hits}/{len(queries)} ({hit_rate:.0f}%)")
    if hit_rate < 40:
        print("  ⚠ Низький hit rate — KB може бути майже порожнім або sync не працював")
    elif hit_rate < 70:
        print("  ○ Помірний hit rate")
    else:
        print("  ✓ Гарний hit rate")

    return {"hits": hits, "total": len(queries), "hit_rate": hit_rate}


# ============ 5. Sync coverage ============


def audit_sync_coverage(brain_data: dict) -> dict:
    print_section("5. Sync Coverage (brain → KB)")

    all_records = brain_data.get("all_records", [])
    if not all_records:
        print("  (немає записів)")
        return {}

    syncable = [r for r in all_records if int(getattr(r, "level", 1)) >= 3]
    mirrored = [r for r in syncable
                if (getattr(r, "metadata", {}) or {}).get("mirrored_to_kb")]

    print(f"  Syncable записів (DOMAIN+IDENTITY): {len(syncable)}")
    print(f"  Вже синхронізовано (mirrored_to_kb): {len(mirrored)}")

    if syncable:
        coverage = len(mirrored) / len(syncable) * 100
        print(f"  Coverage: {coverage:.1f}%")
        if coverage < 30:
            print("  ⚠ Низьке покриття — можливо background sync не запускався")
        elif coverage < 70:
            print("  ○ Помірне покриття — background sync частково працює")
        else:
            print("  ✓ Гарне покриття")
    else:
        print("  (немає syncable записів)")
        coverage = 0

    # Які записи ще не синхронізовані
    not_mirrored = [r for r in syncable if r not in mirrored]
    if not_mirrored:
        print(f"\n  Топ-5 несинхронізованих:")
        for r in not_mirrored[:5]:
            print(f"    [{fmt_level(r.level)}] {r.content[:60]}...")

    return {"syncable": len(syncable), "mirrored": len(mirrored), "coverage": coverage}


# ============ 6. Duplicate rate ============


def audit_duplicates(brain_data: dict) -> dict:
    print_section("6. Duplicate Rate")

    all_records = brain_data.get("all_records", [])
    if not all_records:
        print("  (немає записів)")
        return {}

    # Простий текстовий dedup: перші 50 символів
    seen = {}
    exact_dupes = 0
    near_dupes = []

    for r in all_records:
        key = r.content[:50].lower().strip()
        if key in seen:
            exact_dupes += 1
            if len(near_dupes) < 5:
                near_dupes.append((seen[key].content[:50], r.content[:50]))
        else:
            seen[key] = r

    total = len(all_records)
    dupe_rate = exact_dupes / total * 100 if total else 0

    print(f"  Всього записів: {total}")
    print(f"  Потенційних дублів (перші 50 символів): {exact_dupes} ({dupe_rate:.1f}%)")

    if near_dupes:
        print("\n  Приклади:")
        for orig, dup in near_dupes:
            print(f"    orig: {orig}")
            print(f"    dup:  {dup}")
            print()

    if dupe_rate > 10:
        print("  ⚠ Багато дублів — _check_duplicates може не спрацьовувати")
    elif dupe_rate > 3:
        print("  ○ Помірна кількість дублів")
    else:
        print("  ✓ Дублів мало")

    return {"total": total, "dupes": exact_dupes, "dupe_rate": dupe_rate}


# ============ 7. Connection density ============


def audit_connections(brain_data: dict) -> dict:
    print_section("7. Connection Density (Knowledge Graph)")

    all_records = brain_data.get("all_records", [])
    if not all_records:
        print("  (немає записів)")
        return {}

    total = len(all_records)
    connected = brain_data.get("connected", 0)
    total_connections = sum(len(r.connections or {}) for r in all_records)

    density = connected / total * 100 if total else 0
    avg_connections = total_connections / total if total else 0

    print(f"  Записів зі зв'язками: {connected}/{total} ({density:.1f}%)")
    print(f"  Загальна кількість зв'язків: {total_connections}")
    print(f"  Середня кількість зв'язків на запис: {avg_connections:.2f}")

    # Топ-5 найзв'язаніших вузлів
    top_connected = sorted(all_records, key=lambda r: len(r.connections or {}), reverse=True)[:5]
    if top_connected and top_connected[0].connections:
        print("\n  Топ-5 вузлів за кількістю зв'язків:")
        for r in top_connected:
            n = len(r.connections or {})
            if n == 0:
                break
            print(f"    ({n} зв'язків) {r.content[:55]}...")

    if density < 5:
        print("\n  ⚠ Граф майже без зв'язків — агент не використовує connect_records або reflect() не запускався")
    elif density < 20:
        print("\n  ○ Помірна щільність графу")
    else:
        print("\n  ✓ Гарна щільність графу")

    return {"connected": connected, "total": total, "density": density}


# ============ Main ============


def main():
    print("\n" + "=" * 60)
    print("  REMY MEMORY QUALITY AUDIT")
    print(f"  Brain: {BRAIN_PATH}")
    print(f"  KB:    {KB_PATH}")
    print("=" * 60)

    if not BRAIN_PATH.exists():
        print(f"\n⚠ Brain директорія не існує: {BRAIN_PATH}")
        print("  Запусти додаток хоча б раз щоб ініціалізувати brain.")
        sys.exit(1)

    t0 = time.time()

    brain = CognitiveMemory(str(BRAIN_PATH))

    try:
        brain_data = audit_brain(brain)
        kb_data = audit_kb()
        recall_data = audit_recall_precision(brain)
        kb_hit_data = audit_kb_hit_rate()
        sync_data = audit_sync_coverage(brain_data)
        dupe_data = audit_duplicates(brain_data)
        conn_data = audit_connections(brain_data)
    finally:
        brain.close()

    elapsed = time.time() - t0

    # ============ Summary ============
    print_section("SUMMARY")

    checks = []

    # Brain health
    total_records = brain_data.get("total", 0)
    checks.append(("Brain записів", total_records > 0, str(total_records)))

    # KB health
    if kb_data.get("enabled"):
        kb_bad = kb_data.get("json_wrapped", 0) + kb_data.get("empty", 0)
        checks.append(("KB чистота", kb_bad == 0, f"{kb_bad} поганих записів"))
    else:
        checks.append(("KB", False, "не встановлено"))

    # Recall precision
    if recall_data:
        prec = recall_data.get("precision", 0)
        checks.append(("Recall precision", prec >= 60, f"{prec:.0f}%"))

    # KB hit rate
    if kb_hit_data:
        hr = kb_hit_data.get("hit_rate", 0)
        checks.append(("KB hit rate", hr >= 40, f"{hr:.0f}%"))

    # Sync coverage
    if sync_data:
        cov = sync_data.get("coverage", 0)
        checks.append(("Sync coverage", cov >= 30, f"{cov:.0f}%"))

    # Duplicates
    if dupe_data:
        dr = dupe_data.get("dupe_rate", 0)
        checks.append(("Duplicate rate", dr <= 5, f"{dr:.1f}%"))

    # Connections
    if conn_data:
        dens = conn_data.get("density", 0)
        checks.append(("Graph density", dens >= 5, f"{dens:.1f}%"))

    passed = sum(1 for _, ok, _ in checks if ok)
    total_checks = len(checks)

    for name, ok, val in checks:
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name:25s} {val}")

    print(f"\n  Результат: {passed}/{total_checks} перевірок пройшли")
    print(f"  Час аудиту: {elapsed:.1f}s")

    if passed == total_checks:
        print("\n  ✓ Все виглядає добре!")
    elif passed >= total_checks * 0.7:
        print("\n  ○ Є деякі проблеми — перевір деталі вище")
    else:
        print("\n  ⚠ Кілька серйозних проблем — рекомендується виправлення")

    print()


if __name__ == "__main__":
    main()
