# Brain Health — що і коли запускати

## Щоденно (або після активного використання)

```bash
python scripts/inspect_brain.py
```

Показує: кількість записів по рівнях, розподіл сили, топ-10 записів, concepts/causal/policy.

Якщо бачиш багато Working записів з strength 1.0 — агент давно не запускався. Запусти його хоча б на 5 хвилин щоб maintenance відпрацював.

---

## Що означають числа

**Records by Level:**
- `Identity` — хто ти, контакти, критичні факти. Decay ~0.01%/цикл. Живе місяцями.
- `Domain` — знання, факти, преференції. Decay ~0.5%/цикл. Живе тижнями.
- `Decisions` — задачі, плани, рішення. Decay ~1%/цикл. Живе днями.
- `Working` — веб-пошук, поточний контекст. Decay ~2%/цикл. Має зникати за 1-2 дні.

**Strength distribution:**
- Норма: багато записів 0.8-1.0 на Identity/Domain, мало на Working
- Проблема: Working записи тижнями тримають 1.0 → агент не запускався

**Surfaced Concepts/Causal/Policy — порожньо?**
Норма на початку. Формуються після 20+ записів на одну тему + кількох тижнів роботи агента.

---

## Якщо щось не так

**Всі записи `fact`, немає `preference`/`decision`:**
Норма — semantic_type більше не впливає на поведінку системи (з v1.4.1).
Level визначає важливість, не тип.

**Working записів більше 20:**
Агент давно не запускався. Запусти → один цикл maintenance прибере застарілі веб-пошуки.

**Identity/Domain записів 0:**
Агент нічого не зберіг на постійній основі. Перевір що агент правильно класифікує рівні при store.

---

## Rollback пам'яті (якщо щось зламалось)

```bash
# переглянути snapshots
python -c "from aura import Aura; b = Aura('data/brain'); print(b.list_snapshots()); b.close()"

# відкотитись
python -c "from aura import Aura; b = Aura('data/brain'); b.rollback('snapshot_name'); b.close()"
```

---

## Версія SDK

```bash
python -c "import aura_memory; print(aura_memory.__version__)"
```

Поточна: `1.4.1` — повний 5-шаровий когнітивний стек активний.
