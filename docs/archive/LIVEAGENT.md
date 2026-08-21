# LIVEAGENT — Повний аналіз архітектури агента Remy

> Живий документ. Оновлюється по мірі аналізу та змін в кодовій базі.
> Останнє оновлення: 2026-03-09
> Джерела: code review (Claude Opus), зовнішній аналіз (2 незалежні AI-системи)

## Зміст

1. [Огляд архітектури](#1-огляд-архітектури)
2. [Точки входу (канали)](#2-точки-входу-канали)
3. [Підготовка контексту перед LLM](#3-підготовка-контексту-перед-llm)
4. [LLM + ReAct Loop (LangGraph)](#4-llm--react-loop-langgraph)
5. [Виконання інструментів](#5-виконання-інструментів)
6. [Пам'ять (Aura SDK)](#6-память-aura-sdk)
7. [Пріоритизація та ранжування даних](#7-пріоритизація-та-ранжування-даних)
8. [Кешування (3 рівні)](#8-кешування-3-рівні)
9. [Фонові процеси](#9-фонові-процеси)
10. [Від результату до відповіді](#10-від-результату-до-відповіді)
11. [Повна схема потоку](#11-повна-схема-потоку)
12. [Висновки по архітектурі](#12-висновки-по-архітектурі)
13. [Вузькі місця та архітектурні ризики](#13-вузькі-місця-та-архітектурні-ризики)
14. [Стратегічні рекомендації (v2.x roadmap)](#14-стратегічні-рекомендації-v2x-roadmap)
15. [Відкриті питання / TODO](#15-відкриті-питання--todo)
16. [Changelog](#16-changelog)

---

## 1. Огляд архітектури

**Кодова база:** `remy/app/src/remy/`

Агент Remy — це ReAct-агент на базі LangGraph з 4 каналами вводу, єдиним мозком (Aura SDK, Rust core) та ~50 інструментами.

```
┌────────────────────────────────────────────────────────────────┐
│                        4 КАНАЛИ                                │
│  Desktop (PyWebView)  Telegram  Voice (Gemini Live)  Autonomous│
│       ↓                  ↓            ↓                  ↓     │
│       └──────── invoke_agent() ───────┘                  │     │
│                    (agent.py)          Окремий шлях       │     │
│                        │              (gemini_live.py)    │     │
│                        ▼                                  │     │
│              LangGraph StateGraph                         │     │
│              model ⇄ tools ⇄ model                       │     │
│                        │                                  │     │
│                        ▼                                  │     │
│              execute_tool()  ←─────────────────────────────┘    │
│              (tool_dispatch.py)                                 │
│                        │                                       │
│                        ▼                                       │
│               Aura SDK (Rust core)                             │
│               data/brain/ — єдине сховище                      │
└────────────────────────────────────────────────────────────────┘
```

**Ключовий принцип:** Один мозок, один агент, різні канали — чиста архітектура. Voice — єдиний виняток (прямий Gemini Live Audio API, несумісний з LangGraph).

---

## 2. Точки входу (канали)

### 2.1. Як повідомлення потрапляє в агента

| Канал | Файл | Метод | Виклик агента |
|-------|------|-------|---------------|
| **Desktop** | `web/session.py:152` | `WebSessionManager.gemini_respond()` | `invoke_agent(msg, sid, "desktop", log, history)` |
| **Desktop Stream** | `web/session.py:255` | `gemini_respond_stream()` | `invoke_agent_stream(...)` |
| **Desktop Multimodal** | `web/session.py:167` | `gemini_respond_multimodal()` | `invoke_agent(HumanMessage, ...)` |
| **Telegram** | `core/telegram_bot.py` | `handle_message()` → `_gemini_respond()` | `invoke_agent(msg, sid, "telegram", log, history)` |
| **Autonomous** | `core/autonomy.py:844` | `_decide_and_act()` | `invoke_agent(prompt, sid, "autonomous", log, history)` |
| **Proactive** | `core/autonomy.py:490` | `_start_proactive_session()` | `invoke_agent(msg, sid, "proactive", log, history=[])` |
| **Voice** | `core/gemini_live.py` | Прямий Gemini Live Audio | **НЕ через invoke_agent** |

### 2.2. Параметри invoke_agent

```python
async def invoke_agent(
    user_message: str | HumanMessage,  # Текст або мультимодальне повідомлення
    session_id: str,                    # UUID сесії (для co-activation tracking)
    channel: str,                       # "desktop" | "telegram" | "autonomous" | "proactive"
    session_log: list,                  # Лог активності (для session summary)
    history: list | None = None,        # Попередня історія LangChain messages
) -> tuple[str, list, list]:
    # Повертає: (response_text, updated_messages, updated_session_log)
```

### 2.3. Session Management

- **Desktop:** `WebSession` dataclass в RAM. Одна сесія на юзера. Закривається при виході → JSON dump + session summary.
- **Telegram:** Per-chat сесії, 30 хв timeout.
- **Autonomous:** Кожен cycle = окрема "сесія" в рамках AutonomousLoop.

---

## 3. Підготовка контексту перед LLM

Коли `invoke_agent()` отримує повідомлення, **7 етапів підготовки** перед першим LLM-викликом:

> **Примітка v2.4:** `invoke_agent()` (line 797) делегує до `_invoke_agent_inner()` (line 841) через interactive priority gate — механізм пріоритизації інтерактивних запитів над автономними.

### 3.1. Invalidate caches (`agent.py:853`)

```python
invalidate_system_instruction_cache(session_id)
```
- Зкидає кеш системної інструкції
- Зкидає кеш proactive context
- Системна інструкція перебудовується ОДИН раз на початку запиту, потім кешується для всіх tool iterations

### 3.2. Compact History (adaptive, `agent.py:862`, `agent.py:654`)

```python
keep_recent = _estimate_keep_recent(channel, user_message)
messages = compact_history(messages, keep_recent=keep_recent)
```

**Adaptive keep_recent** (v2.4, `agent.py:87`):

```python
_KEEP_RECENT = {
    "autonomous": 20,   # v2.4: reduced from 40 (scratchpad compensates)
    "research":   24,   # v2.4: reduced from 32 (scratchpad compensates)
    "default":    16,   # ~5 tool iterations — звичайна розмова
}
```

`_estimate_keep_recent(channel, user_message)` (`agent.py:94`):
- `channel == "autonomous"` → делегує до `context_window.dynamic_keep_recent()` (AUTON-12), fallback → 20
- keywords ∈ `_RESEARCH_KEYWORDS` (EN+UK) → 24
- інакше → 16 (default)

> **v2.4:** Значення зменшені з 40/32 до 20/24 — scratchpad (робочий блокнот) компенсує втрату контексту. Для autonomous каналу використовується динамічне визначення розміру через `context_window.dynamic_keep_recent()`, яке враховує складність поточної задачі.

Алгоритм compact_history (`agent.py:654`):
1. **Truncate** `ToolMessage` довше 300 символів → `...[truncated]`
2. Якщо messages <= keep_recent → повернути як є
3. Якщо > keep_recent → розбити на `old_part` + `recent_part`
4. Не розривати tool sequences (AIMessage+ToolMessage разом)
5. Стиснути `old_part`:
   - Витягнути User/AI текст (до 300 символів кожен)
   - Зберегти перші 3 рядки (початок сесії) + останні
   - Ліміт ~8000 символів (~2000 токенів)
6. Обгорнути в `SystemMessage("Earlier in this conversation: ...")`
7. **`_sanitize_tool_sequences()`** (`agent.py:592`) — фіксує orphaned ToolMessages після компресії (Gemini вимагає AI→Tool ordering)
8. **Synthetic HumanMessage injection** — якщо останнє повідомлення не HumanMessage, додається placeholder для Gemini positional rules

### 3.3. In-Session Insights (`agent.py:526`)

```python
messages = check_session_insights(session_id, messages)
```

- Кожне 5-те повідомлення (`INSIGHT_CHECK_INTERVAL = 5`)
- Перевіряє `brain.insights()`:
  - `decay_risk` — спогади, що затухають
  - `conflict` — протиріччя між записами
  - `hot_topic` — часто згадувані теми
- Інжектує як SystemMessage: `"[INTERNAL BRAIN INSIGHT — mention naturally if relevant]"`
- **Zero LLM calls** — чистий Python

### 3.4. Build System Instruction (modular, `system_instruction.py:96`)

Будується **мега-промпт** з 10+ компонентів. **Modular rules** (v2.1) — правила розбиті на ядро + умовні блоки:

| # | Компонент | Джерело | ~Токенів | Умова |
|---|-----------|---------|----------|-------|
| 1 | **Persona** | brain запис з tag `agent-persona`, fallback `_DEFAULT_PERSONA` | ~200 | завжди |
| 2 | **Core Rules** | Хардкодед ядро у `system_instruction.py` | ~1200 | завжди |
| 2a | **Interactive Rules** | `_INTERACTIVE_RULES` (lines 23-33) | ~250 | channel ∈ voice/telegram/desktop |
| 2b | **Browser Rules** | `_BROWSER_RULES` (lines 85-93) | ~400 | BROWSER_ENABLED=True **і** channel ∉ voice/proactive |
| 2c | **Research Rules** | `_RESEARCH_RULES` (lines 38-47) | ~200 | channel ∉ voice/proactive |
| 2d | **Planning Rules** | `_PLANNING_RULES` (lines 49-59) | ~200 | channel ∈ autonomous/desktop/telegram |
| 2e | **Execution Guard Rules** | `_EXECUTION_GUARD_RULES` (lines 61-75) | ~300 | channel == autonomous **або** (BROWSER_ENABLED і channel ∉ voice/proactive) |
| 2f | **Delegation Rules** | `_DELEGATION_RULES` (lines 77-83) | ~150 | channel ∉ voice/proactive |
| 3 | **Channel hints** | `if channel == "autonomous"/"telegram"/"desktop"/"voice"/"proactive"` | ~100 | завжди |
| 4 | **User identity** | `_build_user_identity()` → `brain.search(tags=["user-profile"])` | ~200 | завжди |
| 5 | **Tech stack self-awareness** | Хардкодед (Aura, LangGraph, Gemini) | ~100 | завжди |
| 6 | **Brain context** (prev sessions) | `brain.recall("session start recent topics", token_budget=512)` | ~512 | завжди |
| 6a | **Tier stats** | `brain.tier_stats()` → cognitive/core breakdown (line 336) | ~50 | завжди |
| 7 | **Background insights** | `get_transient_insights()` з `background_brain.py` | ~100 | завжди |
| 8 | **Temporal context** | `datetime.now()` → день, час, підказки по часу доби | ~50 | завжди |
| 9 | **Proactive context** | scheduled tasks + session summaries + failures + last dialogue | ~500 | завжди |
| 10 | **Active TODOs** | `brain.search(tags=["todo-item"])` + cross-ref з failures | ~300 | завжди |
| 11 | **Feedback adaptation** | `get_recent_feedback_summary()` | ~100 | завжди |

**Модульні правила** (`system_instruction.py:23-93`):
```python
_INTERACTIVE_RULES = """..."""        # Правила для інтерактивних каналів (~250 tokens)
_RESEARCH_RULES = """..."""           # v2.3: правила для research tasks (~200 tokens)
_PLANNING_RULES = """..."""           # v2.3: правила для planning (~200 tokens)
_EXECUTION_GUARD_RULES = """..."""    # v2.3: guard rails для execution (~300 tokens)
_DELEGATION_RULES = """..."""         # v2.3: правила для delegation (~150 tokens)
_BROWSER_RULES = """..."""            # Правила для browser tools (~400 tokens)

# Умовне додавання (system_instruction.py:202-223):
if channel in ("voice", "telegram", "desktop"):
    base += _INTERACTIVE_RULES
if settings.BROWSER_ENABLED and channel not in ("voice", "proactive"):
    base += _BROWSER_RULES
if channel not in ("voice", "proactive"):
    base += _RESEARCH_RULES
    base += _DELEGATION_RULES
if channel in ("autonomous", "desktop", "telegram"):
    base += _PLANNING_RULES
if channel == "autonomous" or (settings.BROWSER_ENABLED and channel not in ("voice", "proactive")):
    base += _EXECUTION_GUARD_RULES
```

**Загальний розмір: ~2500-4500 токенів** (voice/proactive = ~2500; desktop/telegram = ~3500; autonomous з browser = ~4500)

#### Proactive Context деталі (`proactive_context.py:135`, `_get_proactive_context_locked`)

Кешується на 5 хв (`_PROACTIVE_CACHE_TTL_SEC = 300`). Включає:
1. **Scheduled tasks** — due today / due tomorrow
2. **Recent session summaries** — brain.search(tags=["session-summary"], limit=2)
3. **Last dialogue** — з JSON файлу `data/history/*.json` (останні 8 реплік)
4. **Recent failures** — brain.search(tags=["outcome-failure"], limit=5)
5. **Recent autonomous outcomes** — brain.search(tags=["autonomous-outcome"], limit=5)

### 3.5. Inject Brain Context — RAG (`agent.py:1332`)

```python
context_msg = _inject_context(state)
```

Окремо від системної інструкції, **для кожного повідомлення користувача**:
- Мінімум 5 символів, мінімум 2 слова
- `_expand_relative_dates(user_text)` (`agent.py:1288`) — розгортає "вчора", "минулого тижня" тощо в конкретні дати для кращого recall
- `brain.recall(recall_query, token_budget=1200)` — семантичний пошук (v2.4: збільшено з 600)
- **Context deduplication** (`_extract_record_ids()`, `agent.py:1278`) — видаляє записи вже присутні в system instruction (за `[id:xxx]` маркерами)
- **Temporal supplement** (v2.4, `agent.py:1387-1415`) — якщо запит містить temporal signals ("вчора", "сьогодні", "last week"), додатково дотягує до 15 останніх записів за 7 днів через `brain.search(limit=30)`
- Якщо знайдено → SystemMessage після sys_instruction
- Ліміт: 8000 символів (~2000 токенів) (v2.4: збільшено з 4800)
- Попередження: `"Recalled context may contain outdated info. Focus on user's CURRENT question."`

### 3.6. Session Context — Anti-Contradiction (`agent.py:1208`)

```python
session_ctx = _build_session_context(state["messages"])
```

Сканує **поточну розмову** (не мозок!):
- Мінімум 4 повідомлення в історії
- Витягує останні 8 тверджень користувача (перші 120 символів)
- Витягує останні 10 дій агента (tool_name + args + results)
- SystemMessage: `"Do NOT propose actions that were already completed or contradict..."`
- **Zero LLM calls** — чистий text extraction

---

## 4. LLM + ReAct Loop (LangGraph)

### 4.1. StateGraph Structure

```python
class AgentState(TypedDict):                    # agent.py:170
    messages: Annotated[list, add_messages]
    session_id: str
    channel: str              # "desktop" | "telegram" | "autonomous" | "proactive"
    session_log: list         # Activity log for session summary
    enabled_tools: set        # Extended tools enabled via enable_tools meta-tool
    _cached_session_ctx: str  # v2.4: session context text (computed once per invoke, "" if empty)
    _cached_scratchpad: str   # v2.4: scratchpad text (computed once per invoke, "" if empty)
```

```
┌───────┐     ┌──────────────┐     ┌───────┐
│ model │────>│should_continue│────>│ tools │
│       │<────┤              │     │       │
└───────┘     │    END ─────────> response │
              └──────────────┘     └───────┘
                   │
              "model" (wrap-up on limit)
```

Граф кешується per-channel: `_compiled_graphs: dict[str, object]` (`agent.py:484`)

### 4.2. call_model (`agent.py:184`)

Порядок messages:
1. `SystemMessage` — system instruction (кешовано per-request) (`agent.py:191-196`)
2. `SystemMessage` — brain RAG context (якщо є) (`agent.py:199-202`, position 1)
3. `SystemMessage` — session context anti-contradiction (якщо є) (`agent.py:206-210`)
4. `SystemMessage` — **scratchpad** working notes (якщо є) (`agent.py:214-217`) — v2.4
5. Решта messages (history + новий HumanMessage)

**Чистка проміжного тексту:**
```python
# Якщо AIMessage має tool_calls + content → обнулити content
# Запобігає "подвійній відповіді" (model repeats intermediate thinking)
if isinstance(msg, AIMessage) and msg.tool_calls and msg.content:
    cleaned.append(AIMessage(content="", tool_calls=msg.tool_calls, id=msg.id))
```

**Selective Tool Loading:**
- `autonomous` / `proactive` → `get_all_tools()` (всі ~50)
- `desktop` / `telegram` → `CORE_TOOL_NAMES` only + `enabled_tools` (розширюються через meta-tool `enable_tools`)

**Error handling:**
- `TimeoutError` / `OSError` → AIMessage з поясненням, не crash
- Non-AIMessage response → обгортка в AIMessage

### 4.3. should_continue (`agent.py:428`)

Routing logic:
- AIMessage з `tool_calls` → `"tools"`
- Інакше → `END`
- **Guard: tool iteration limit**
  - Normal: `MAX_TOOL_ITERATIONS = 30`
  - Research/browser: `50` (якщо поточні tools ∈ `_RESEARCH_TOOLS`)
- При досягненні ліміту: inject ToolMessage `"[SYSTEM: Tool limit reached. Summarize...]"` → route to `"model"`
- Якщо модель **ігнорує wrap-up** і знову викликає tools → **hard stop** (return END)

### 4.4. call_tools (`agent.py:333`)

Для кожного tool_call:
1. `event_bus.emit("tool_call", {tool, args_summary, channel})` — для Activity stream (`agent.py:366`)
2. `tool.invoke(args)` → делегує до `execute_tool()` (`agent.py:375`)
3. `event_bus.emit("tool_result", {tool, result_preview, channel})` — результат для Activity stream (`agent.py:385`)
4. Результат → `ToolMessage(content, tool_call_id)` (`agent.py:392`)
5. Лог → `session_log.append({type, tool, args, result})` (`agent.py:397`)
6. Спеціальна обробка `enable_tools` → оновлює `state["enabled_tools"]` (`agent.py:405`)

### 4.5. Recursion Limits (adaptive, `agent.py:42`)

```
quick:      35  (~11 tool iterations) — короткі запитання (<8 слів)
normal:     70  (~23 tool iterations) — звичайна розмова
research:  150  (~50 tool iterations) — browser, web_search, delegate
autonomous: 150 — як research
```

Визначення: `_estimate_recursion_limit(channel, user_message)`
- autonomous → завжди research
- keywords (досліди, browse, зареєструй, navigate...) → research
- <8 слів → quick
- інакше → normal

### 4.6. Streaming (`agent.py:983`)

`invoke_agent_stream()` (`agent.py:983`) → `_invoke_agent_stream_inner()` (`agent.py:1025`) — async generator, yields events:
- `{"type": "token", "content": str}` — LLM токени
- `{"type": "tool_start", "tool": str}` — початок tool
- `{"type": "tool_end", "tool": str}` — кінець tool
- `{"type": "final", "text": str, "messages": list}` — фінальний пакет

**Token buffering:** Буферизує токени з проміжних model calls. Flush тільки коли model call final (немає tool_calls). Запобігає "потрійній відповіді".

---

## 5. Виконання інструментів

### 5.1. Точка входу: execute_tool (`tool_dispatch.py:35`)

```
execute_tool(name, args, session_id, channel)
     │
     ├─ name == "delegate_task"
     │   → ПОЗА brain_lock (workers самі блокують)
     │   → _handle_delegate_task(args, session_id, channel)
     │
     ├─ name ∈ ("browse_page", "browser_act", "browser_close")
     │   → Trust validation ПІД brain_lock
     │   → Execution ПОЗА brain_lock (async I/O)
     │   → _handle_browser_tool(name, args, session_id, channel)
     │
     └─ Все інше
         → brain_lock → _execute_tool_locked(name, args, session_id, channel)
```

### 5.2. Захисні шари (_execute_tool_locked, `tool_dispatch.py:80`)

| Шар | Перевірка | Файл | Реакція |
|-----|-----------|------|---------|
| **Circuit Breaker** | `tool_health.is_available(name)` | `tool_health.py` | 3 відмови за 10 хв → блок на 10 хв |
| **Trust Enforcement** | `_validate_action_data(name, args)` | `provenance.py` | Блокує дії з unverified sensitive data |
| **Approval Queue** | `needs_approval("store", args)` | `approval_queue.py` | Фінансові операції → user confirmation |
| **SSRF Protection** | `_check_ssrf(url)` | `tool_utils.py` | Блокує приватні IP, localhost, file:// |
| **Audit Trail** | `is_critical(name)` | `audit_trail.py` | Логує finance/registration/identity tools |

### 5.3. Recall (найчастіший tool, `tool_dispatch.py:410`)

**Два шляхи** (v2.1, hasattr guard для backward compatibility):

#### Шлях A: Unified recall (Aura SDK ≥1.3.0)

```
recall("мої ліки")
  │
  ├── 1. IN-MEMORY CACHE CHECK
  │    _get_cached_recall(query) — TTL 5 хв, max 50
  │    Hit? → повернути миттєво
  │
  ├── 2. brain.recall_full(query, top_k=15, include_failures=True)
  │    ОДИН Rust виклик замість трьох:
  │    ├─ Stage 1: recall_core() — RRF pipeline (SDR + ngram + tags)
  │    ├─ Stage 2: substring match (merged, ONE read lock)
  │    └─ Stage 3: failure-aware recall (same read lock as stage 2)
  │    Повертає: [(score, Record), ...] — вже відсортовано
  │
  ├── 3. SORT by effective_trust (найвищий → перший)
  │    _compute_effective_trust(metadata, now)
  │
  ├── 4. FILTER: trust < 0.35 → відкинути
  │
  ├── 5. DEDUP: перші 80 символів → set()
  │
  ├── 6. FORMAT з метаданими:
  │    [id:xxx] [trust: 0.7 | 3d | interactive] Текст... [tags]
  │    Якщо >300 символів → truncated + "use get_full_record"
  │
  ├── 7. CACHE result → _cache_recall_result(query, text)
  │
  └── 8. METRICS: record_recall_latency(total_ms, total_ms, 0.0, 0.0)
```

#### Шлях B: Legacy fallback (Aura SDK <1.3.0)

```
recall("мої ліки")
  │
  ├── 1. IN-MEMORY CACHE CHECK (як вище)
  │
  ├── 2. SEMANTIC SEARCH (RRF)
  │    brain.recall_structured(query, top_k=15, session_id)
  │    Rust Aura → Reciprocal Rank Fusion scoring
  │
  ├── 3. SUBSTRING FALLBACK
  │    brain.search(query, limit=10)
  │    Ловить exact keyword matches, які RRF пропускає
  │    Додає з score=0.6 (якщо не дублікат)
  │
  ├── 4. FAILURE-AWARE RECALL
  │    brain.search(query, tags=["outcome-failure"], limit=5)
  │    Примусово піднімає провали (score=0.8!)
  │
  ├── 5-9. SORT → FILTER → DEDUP → FORMAT → CACHE (як вище)
  │
  └── 10. METRICS: record_recall_latency(total_ms, rrf_ms, substr_ms, failure_ms)
```

**Guard:** `brain._has_recall_full` — автоматично обирає шлях A або B. Since Aura SDK 1.3.2, `recall_full()` is always available (Path A).

### 5.4. Store (`tool_dispatch.py:588`)

Пайплайн:

```
store(content="...", tags="health,medication", level="L3_DOMAIN")
  │
  ├── 1. _clean_tag() → транслітерація кирилиці, lowercase, strip
  │
  ├── 2. _auto_protect_tags(content, tags) → email/wallet → додає "financial"/"credential"
  │
  ├── 3. needs_approval() → фінансові теги? → approval_queue.request_approval_sync()
  │
  ├── 4. _check_duplicates(content[:100], tags) → brain.search() → similarity check
  │
  ├── 5. _stamp_provenance(metadata, channel, tags):
  │    source     = "agent-interactive" (desktop/telegram)
  │    trust_score = 0.7
  │    verified   = false
  │    volatility = "stable" / "moderate" / "volatile" (inferred from tags)
  │    timestamp  = now ISO
  │
  ├── 6. _apply_store_guard(content, tags, channel):
  │    Чутливі дані (email, wallet, password) → actionable = false
  │
  ├── 7. brain.store(content, level, tags, metadata, channel)
  │    Rust Aura → creates Record → returns ID
  │
  └── 8. clear_recall_cache() → інвалідує весь recall cache
```

### 5.5. Web Search

- `_get_cached_search(query)` — brain records з tag `web-search-cache`, TTL 24h
- Retry з backoff: 2 спроби, delays [2s, 5s]
- Результати → cache в brain як records
- SSRF check перед HTTP запитами

### 5.6. Browser Tools

- `browse_page(url)` → Playwright → screenshot → vision model → structured JSON
- `browser_act(action, selector, text)` → click/type/scroll/fill_form
- `browser_close()` → close browser
- Daily action limit: 200
- Idle timeout: 300s
- SSRF protection для URLs
- Trust validation для sensitive actions (browser_act with financial URLs)

### 5.7. Orchestrator & Capability Packs (Multi-Agent)

- `orchestrator.py:dispatch_worker(goal)` → routes goal to specialized worker
- `capability_packs.py:resolve_pack(goal)` → 3-tier resolution: explicit → keyword inference → general fallback
- **5 packs**: `signup_operator` (browser, 15 steps, 90s), `publisher` (browser, approval-first, 90s), `market_research` (research, 20 steps, 180s), `monitoring` (read-only, 120s), `general` (fallback)
- Each pack defines: worker type, guardrails, approval_mode, metrics_family, step_budget, timeout_sec
- Worker types: `browser_worker`, `research_worker`, `monitoring_worker`, `generic`
- Worker provenance: channel="worker-{role}", trust=0.35 (нижчий за interactive)
- Mission-first routing: `focus_execution_goals()` prioritizes runnable mission tasks
- Pack-level metrics via `execution_log.py` and `task_metrics`

---

## 6. Пам'ять (Aura SDK)

### 6.1. Архітектура

```
agent_tools.py:
  brain = _AuraCompat(path)   # Global instance, created at import time
        → Aura(path)           # Rust core (_core.pyd)
          → data/brain/        # Файлове сховище
```

- **_AuraCompat**: Шар сумісності — адаптує API Aura SDK до очікувань Remy (metadata serialization, Level compat, tier helpers)
- **brain_lock**: `threading.RLock()` — захист від concurrent access з Web (FastAPI), Telegram та Background потоків

### 6.2. Рівні пам'яті

```
РІВЕНЬ          DECAY RATE    ТРИВАЛІСТЬ    ПРИЗНАЧЕННЯ
─────────────────────────────────────────────────────────
L4_IDENTITY     ~0.99         Місяці/роки   Профіль, контакти, персона агента
L3_DOMAIN       ~0.95         Тижні/місяці  Факти, знання, дослідження, здоров'я
L2_DECISIONS    ~0.90         Дні           Рішення, вибори, преференції
L1_WORKING      ~0.80         Години        Тимчасові замітки, проміжні дані
```

### 6.3. Структура Record

```
Record:
  id:               UUID str
  content:          str           # Основний текст
  level:            Level         # L1-L4
  tags:             [str]         # Теги для фільтрації
  strength:         float         # 0.0-1.0, зменшується з часом
  activation_count: int           # Скільки разів recall "зачепив"
  created_at:       timestamp     # Коли створено
  metadata: {                     # Розширена інформація
    source:         str           # "agent-interactive" / "agent-autonomous" / "user-confirmed" / "agent-worker"
    verified:       bool          # Верифіковано користувачем?
    trust_score:    float         # Початковий рівень довіри (store time)
    volatility:     str           # "stable" / "moderate" / "volatile"
    timestamp:      str           # ISO datetime
    actionable:     bool | None   # Чи можна використовувати в зовнішніх діях?
    ... (tool-specific fields)
  }
```

### 6.4. Metadata Serialization

Aura Rust core вимагає **всі metadata values = string**. `_AuraCompat` автоматично:
- **На запис:** `_stringify_metadata()` — bool→"true"/"false", int→"123", list→JSON
- **На читання:** `_deserialize_metadata()` — зворотна конвертація in-place
- `_STRING_METADATA_KEYS` — set ключів, які НЕ конвертуються в числа (IDs, timestamps)

### 6.5. Методи пошуку

| Метод | Опис | Швидкість |
|-------|------|-----------|
| `recall(query, token_budget)` | Семантичний + keyword, повертає **текст** | ~200ms |
| `recall_structured(query, top_k)` | RRF scoring, повертає **список tuple(score, Record)** | ~200ms |
| **`recall_full(query, top_k, include_failures, ...)`** | **Unified: RRF + substring + failures в ОДНОМУ Rust виклику** (v1.0.5) | **~200ms** |
| `search(query, tags, limit)` | Substring + tag filter, повертає **Record list** | ~50ms |
| `get(id)` | За UUID | ~1ms |

**recall_full** (Aura SDK ≥1.3.0):
- Параметри: `query, top_k=20, include_failures=True, min_strength=0.1, expand_connections=True, session_id=None`
- Stage 1: `recall_core()` — RRF pipeline (own lock cycle, write for activation)
- Stage 2+3: ONE read lock — substring match + failure records merged
- Повертає `[(score, Record), ...]` з повним metadata (включаючи trust_score, source)
- Python wrapper: `_AuraCompat.recall_full()` з `_deserialize_metadata()` для кожного результату

---

## 7. Пріоритизація та ранжування даних

### 7.1. Effective Trust Score (формула)

```
effective_trust = base_trust × source_authority × age_factor × volatility_factor
```

| Компонент | Значення |
|-----------|----------|
| `base_trust` | `metadata.trust_score` (0.35 - 1.0), встановлюється при store |
| `source_authority` | множник по джерелу (табл. нижче) |
| `age_factor` | зниження з часом |
| `volatility_factor` | stable=1.0, moderate=0.9, volatile=0.7 |

### 7.2. Source Authority

```
ДЖЕРЕЛО              TRUST AT STORE   AUTHORITY AT RECALL
──────────────────────────────────────────────────────────
user-confirmed       1.0              1.2   ← Найвищий
user-telegram        —                1.2
user-desktop         —                1.2
agent-interactive    0.7              1.0
system               0.6              0.9
agent                0.5              0.85
agent-autonomous     0.4              0.75
agent-worker         0.35             0.70
agent-inference      —                0.65  ← Найнижчий
```

### 7.3. Volatility Classification

```
STABLE tags:    identity, contact, credential, financial, person, agent-persona
VOLATILE tags:  market, price, scheduled-task, todo-item, outcome-*, session-*, action-plan, feedback-signal
MODERATE:       все інше
```

### 7.4. Recall Filtering

- **Threshold:** trust < 0.35 → запис не показується
- **Sorting:** effective_trust descending
- **Dedup:** перші 80 символів content → не повторювати
- **Truncation:** >300 chars → truncate + hint "use get_full_record"
- **Failure boost:** outcome-failure records отримують score=0.8 (примусово піднімаються)

### 7.5. Memory Guards (3 шари)

1. **STORE GUARD** — при store: чутливі дані → `actionable=false` автоматично
2. **ACTION GUARD** — при browser_act/http_get: перевіряє `actionable=true` + `trust >= 0.8`
3. **HALLUCINATION GUARD** — дані НЕ знайдені в пам'яті → блок (запобігає вигадуванню)

---

## 8. Кешування (3 рівні)

| Кеш | Де | TTL | Max Size | Інвалідація |
|-----|-----|-----|----------|-------------|
| **System Instruction** | `agent.py:154` `invalidate_system_instruction_cache()` | Per-request | 50 entries | `invalidate_system_instruction_cache()` — на початку кожного invoke_agent |
| **Recall** | `tool_utils.py` `_recall_cache` | 5 хв | 50 entries | `clear_recall_cache(new_content)` — selective (keyword overlap) або full clear |
| **Proactive Context** | `proactive_context.py:107` `_proactive_context_cache` | 5 хв | 1 entry | Разом з sys instruction cache |
| **Web Search** | Brain records, tag `web-search-cache` | 24 год | Unlimited (brain) | TTL check при read |
| **LangChain Tools** | `langgraph_tools.py:281` `_cached_tools` | In-memory (process lifetime) | 1 list | `invalidate_tool_cache()` при sandbox approve |
| **Graph** | `agent.py:484` `_compiled_graphs` | In-memory (process lifetime) | Per-channel | `invalidate_graph_cache()` |
| **Session Context** | `AgentState._cached_session_ctx` | Per-invoke | 1 per state | v2.4: computed once, reused across tool iterations |
| **Scratchpad** | `AgentState._cached_scratchpad` | Per-invoke | 1 per state | v2.4: computed once, reused across tool iterations |

### 8.1. Recall Latency Monitoring (v2.1, `metrics.py`)

Prometheus gauge `remy_recall_latency_ms` з розбивкою по стадіях:

| Stage | Що вимірює |
|-------|-----------|
| `total` | Повний recall від початку до кінця |
| `rrf` | Тільки recall_structured / recall_core |
| `substring` | Тільки brain.search() substring fallback |
| `failure` | Тільки brain.search(tags=["outcome-failure"]) |

**MetricsCollector** (`metrics.py:70`):
```python
metrics_collector.record_recall_latency(total_ms, rrf_ms, substr_ms, failure_ms)
```

**Unified path (recall_full):** `rrf_ms = total_ms`, `substr_ms = 0`, `failure_ms = 0` (стадії об'єднані в Rust).

**Legacy path:** Кожна стадія вимірюється окремо.

Доступно: `GET /api/metrics` → `remy_recall_latency_ms{stage="total|rrf|substring|failure"}`

---

## 9. Фонові процеси

### 9.1. Background Brain (`background_brain.py`)

Запускається: `remy --background`

```
brain.run_maintenance() → MaintenanceReport (8 native phases in Rust):
  Phase 1: DECAY        — зменшує strength за рівнем (Working 0.80, Decisions 0.90, Domain 0.95, Identity 0.99)
  Phase 2: REFLECT      — аналіз зв'язків, promotion candidates
  Phase 3: INSIGHTS     — 9 zero-LLM detectors (decay_risk, clusters, conflicts, hot_topics, stale, co-activation, hubs, causal chains, promotion)
  Phase 4: CONSOLIDATION — MinHash-based duplicate merging
  Phase 5: SYNTHESIS    — cross-connections via causal walks
  Phase 6: ARCHIVAL     — remove stale records per configurable rules
  Phase 7: LEVEL FIX    — correct misclassified records (every Nth cycle)
  Phase 8: TASK REMINDERS — find overdue scheduled tasks

Python-side phases (background_brain.py):
  Phase P1: LLM CLUSTER MERGE — кластеризує схожі записи → LLM підсумовує в meta-record
  Phase P2: SCHEDULED TASKS CHECK — шукає tasks з due_date = сьогодні/завтра
  Phase P3: PROACTIVE NOTIFICATIONS → Telegram (якщо PROACTIVE_CHAT_ID налаштований)
```

### 9.2. Memory Level Fixing (`background_brain.py:111`)

`fix_memory_levels()` — автоматичний даунгрейд:
- Записи з тегами `autonomous-outcome`, `research-project`, `session-summary` тощо **не мають бути на L4_IDENTITY**
- Правило: якщо tag ∈ `_NON_IDENTITY_TAGS` і level == IDENTITY → даунгрейд до DOMAIN
- Виняток: activation_count >= 20 і strength >= 0.9 → keep (earned IDENTITY)
- Ідемпотентна операція

### 9.3. Transient Insights

- Зберігаються як brain record з tag `background-insights-latest`
- Завантажуються lazy (при першому зверненні)
- Інжектуються в system instruction через `get_transient_insights()`

---

## 10. Від результату до відповіді

### 10.1. Після завершення ReAct loop

```python
# agent.py (inside _invoke_agent_inner)
response_text = ""
for msg in reversed(result_messages):
    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
        response_text = msg.content.strip()
        break
```

**Fallback якщо немає тексту:**
- Збирає tool_names з session_log → `"I reached the step limit. Tools used: ..."`
- Або generic: `"I couldn't generate a response."`

### 10.2. Post-processing

| Крок | Файл | Опис |
|------|------|------|
| Clean messages | `agent.py` (inside _invoke_agent_inner) | Видаляє SystemMessage з history |
| Eval metrics | `eval_metrics.py` | compute_response_metrics → store (non-critical) |
| Feedback signals | `brain_tools.py` | detect_feedback_signals → store (desktop/telegram only) |

### 10.3. Session close

При закритті сесії (`session.py:84`):
1. **Session history** → JSON файл: `data/history/{timestamp}_{session_id}.json`
2. **Session summary** → LLM → `brain.store(tags=["session-summary"])` (level=DOMAIN)
3. **End session** → `brain.end_session(session_id)` → Rust оновлює co-activation

---

## 11. Повна схема потоку

```
USER MESSAGE
     │
     ▼
[1] CHANNEL (desktop/telegram/autonomous/proactive)
     │
     ▼
[2] invoke_agent(message, session_id, channel, log, history)
     │   → _invoke_agent_inner() [priority gate]
     │
     ├─ invalidate caches (sys instruction + proactive)
     ├─ compact_history(keep_recent=adaptive: 16/24/20 or dynamic)
     ├─ check_session_insights() [every 5 msgs]
     ├─ compute & cache session context + scratchpad in state
     │
     ▼
[3] build_agent_graph(channel) → StateGraph [cached per channel]
     │
     ▼
[4] NODE: call_model
     ├─ SystemMessage #1: build_system_instruction()
     │    ├─ persona (brain)
     │    ├─ rules (modular, ~1200-2500 tokens depending on channel)
     │    ├─ conditional: interactive/browser/research/planning/execution/delegation rules
     │    ├─ channel hints
     │    ├─ user profile (brain)
     │    ├─ brain.recall("context", 512 tokens) + tier stats
     │    ├─ background insights
     │    ├─ temporal (time/day)
     │    ├─ proactive context (tasks, summaries, failures, dialogue)
     │    ├─ active TODOs (brain)
     │    └─ feedback adaptation
     ├─ SystemMessage #2: _inject_context()
     │    └─ brain.recall(user_text, 1200 tokens) + temporal supplement — episodic RAG
     │    └─ context deduplication via [id:xxx] markers
     ├─ SystemMessage #3: _build_session_context()
     │    └─ user facts + actions from THIS conversation (cached in state)
     ├─ SystemMessage #4: scratchpad notes (v2.4, cached in state)
     │    └─ working memory notes from ring buffer
     ├─ Clean intermediate AIMessage text (anti-duplication)
     ├─ Selective tools (CORE for interactive, ALL for autonomous)
     └─ call_llm(messages, tools) → Gemini API
     │
     ▼
[5] NODE: should_continue
     ├─ tool_calls? → YES → "tools" (max 30, research=50)
     ├─ limit reached? → wrap-up message → "model"
     ├─ wrap-up ignored? → hard stop → END
     └─ no tool_calls → END
     │
     ▼
[6] NODE: call_tools
     │
     ▼
[7] execute_tool(name, args, session_id, channel)
     ├─ Circuit breaker check
     ├─ Trust enforcement
     ├─ Approval queue (financial)
     │
     ▼
[8] _execute_tool_inner() — BIG SWITCH:
     │
     ├─ recall:  cache → recall_full() [unified Rust] or 3-call legacy
     │           → sort by trust → filter ≥0.35 → dedup → format → cache → metrics
     │
     ├─ store:   clean tags → auto_protect → approval? → check_dups
     │           → stamp_provenance → store_guard → brain.store → clear cache
     │
     ├─ search:  tag-only or semantic via recall_structured → filter → format
     ├─ web_search: ssrf check → retry+backoff → cache result
     ├─ browse_page: Playwright → screenshot → vision model → structured JSON
     ├─ browser_act: trust validation → action → screenshot verify
     ├─ dispatch_worker: orchestrator → resolve_pack → scoped worker execution
     └─ ... (~50 total tools)
     │
     ▼
[9] Back to call_model → should_continue → END
     │
     ▼
[10] EXTRACT RESPONSE
     ├─ Last AIMessage without tool_calls
     ├─ Fallback: tool summary or "couldn't generate"
     ├─ Eval metrics (non-critical)
     ├─ Feedback signals (desktop/telegram)
     └─ Return (text, clean_messages, log)
     │
     ▼
[11] SESSION CLOSE (eventually)
     ├─ History → JSON file (data/history/)
     ├─ Session summary → LLM → brain.store(tags=["session-summary"])
     └─ brain.end_session()
     │
     ▼
[12] BACKGROUND BRAIN (offline)
     ├─ decay → strength зменшується
     ├─ reflect → аналіз зв'язків
     ├─ insights → decay_risk, conflicts, hot_topics
     ├─ consolidate → MinHash merge + LLM cluster merge
     ├─ knowledge synthesis → 2-hop graph walk
     └─ notifications → Telegram
```

---

## 12. Висновки по архітектурі

### Що працює правильно

1. **Уніфікований потік** — 4 канали → 1 invoke_agent() → 1 brain. Zero duplication.
2. **Багаторівневий контекст** — system instruction + RAG + session context + scratchpad = модель має повну картину на кожному кроці (4 SystemMessages).
3. **Provenance tracking** — кожен запис має trust/source/verified — модель знає, чому довіряти.
4. **Фонові процеси** — decay/consolidation працюють як "забування" і "узагальнення" — природна модель пам'яті.
5. **Багаторівневе кешування** — 8 кешів (sys instruction, recall, proactive, web search, tools, graph, session ctx, scratchpad) запобігають повторним дорогим зверненням.
6. **Selective tool loading** — interactive каналам не потрібні всі 50 tools, менше confusion для LLM.
7. **Memory guards** — 3-layer protection (store guard, action guard, hallucination guard).
8. **Adaptive recursion limits** — research/browser задачі отримують більше кроків.
9. **Modular system prompt** — 6 conditional rule blocks (interactive, browser, research, planning, execution guard, delegation) — channel-appropriate token budget.

---

## 13. Вузькі місця та архітектурні ризики

> Зібрано з трьох незалежних аналізів. Пріоритизовано за ступенем впливу.

### 13.1. CRITICAL: "Lost in the Middle" — токенний бюджет уваги

**Де:** Етап 2 (підготовка контексту), Етап 4 (call_model)

**Симптом:** Системний промпт ~2.5-4.5K токенів + RAG ~2000 токенів (v2.4: token_budget=1200, limit=8000 chars) + session context + scratchpad. Разом ~7-10K токенів контексту **до** того як юзер щось сказав.

**Проблема:** Хоча Gemini має гігантське контекстне вікно, LLM-и страждають від ефекту "Lost in the Middle" — інформація на початку і в кінці промпту обробляється краще, ніж в середині. Коли агент одночасно має думати про:
- Persona (хто я)
- 2000 токенів Rules (як поводитись)
- Active TODOs (що робити)
- Proactive context (tasks, failures, summaries)
- Brain RAG context (що я пам'ятаю)
- Session context (що вже обговорили)

...увага розсіюється.

**Наслідок:** В автономному режимі агент може ігнорувати конкретні правила (Rules), бо вони "загубилися" між Proactive context та Brain context. Наприклад, правило `STOP ON REPEATED FAILURES` може бути проігнороване коли навколо нього 6K інших токенів.

**Severity:** HIGH (впливає на якість кожної відповіді)

### 13.2. ~~CRITICAL~~ MITIGATED: Агресивна компресія історії

> **Вирішено у v2.1** — adaptive compact_history (Розділ 3.2, Розділ 14.7)

**Було:** `compact_history(messages, keep_recent=16)` — фіксований threshold для всіх задач.

**Стало:** `_estimate_keep_recent(channel, user_message)` → adaptive threshold:
- `autonomous` → dynamic via `context_window.dynamic_keep_recent()`, fallback 20 (v2.4: зменшено з 40)
- `research` keywords → 24 (v2.4: зменшено з 32)
- `default` → 16 (~5 tool iterations)

**Залишковий ризик:** Scratchpad (Розділ 14.3) та state summarization (AUTON-12) компенсують зменшення keep_recent. Для autonomous channel — dynamic sizing враховує складність задачі (12-28 повідомлень).

**Severity:** LOW (було HIGH)

### 13.3. ~~HIGH~~ MITIGATED: Потрійний I/O удар у Recall

> **Вирішено у v2.1** — `recall_full()` в Aura SDK 1.3.2 (Розділ 5.3, Розділ 14.1)

**Було:** 3 послідовні Python→Rust виклики: `recall_structured()` + `search()` + `search(tags=["outcome-failure"])`.

**Стало:** `brain.recall_full(query, top_k=15, include_failures=True)` — ОДИН Rust виклик:
- Stage 1: `recall_core()` (RRF) — власний lock cycle
- Stage 2+3: substring + failure records — ONE read lock, один прохід по records

**Note:** Since Aura SDK 1.3.2, `recall_full()` is always available. Legacy 3-call fallback path remains for backward compatibility but is effectively dead code.

**Latency monitoring:** `remy_recall_latency_ms` Prometheus gauge відстежує деградацію (Розділ 8.1).

**Залишковий ризик:** `_inject_context()` — окремий `brain.recall()` на кожне повідомлення. Context deduplication (Розділ 14.5) вирішить.

**Severity:** LOW (було HIGH)

### 13.4. ~~MEDIUM~~ MITIGATED: Recall cache — повна інвалідація

> **Частково вирішено у v2.3** — Selective recall cache invalidation (Розділ 14.4)

**Де:** `tool_utils.py` — `clear_recall_cache(new_content)`

**Було:** Кожен `store()`, `update_record()`, `delete_record()`, `connect_records()` зкидав **ВЕСЬ** recall cache.

**Стало:** `clear_recall_cache(new_content)` — з аргументом витягує перші 10 keywords, видаляє тільки cache entries з keyword overlap. `store` та `update_record` передають content для selective clear. `delete_record`, `connect_records`, `update_persona` — full clear (немає content для selective).

**Залишковий ризик:** `delete_record` та `connect_records` все ще зкидають весь кеш.

**Severity:** LOW (було MEDIUM)

### 13.5. ~~MEDIUM~~ MITIGATED: Дублювання brain.recall в system instruction та RAG

> **Вирішено у v2.3** — Context deduplication (Розділ 14.5)

**Де:** `system_instruction.py` та `agent.py:1332`

**Факт:**
- `build_system_instruction()` робить `brain.recall("session start recent topics", token_budget=512)` — для загального контексту
- `_inject_context()` робить `brain.recall(user_text, token_budget=1200)` — для конкретного запиту (v2.4: збільшено з 600)

**Стало:** `_inject_context()` збирає record IDs `[id:xxx]` з system instruction і фільтрує дубльовані лінії з recall output. Якщо всі лінії дубльовані → нічого не інжектується.

**Severity:** LOW (було MEDIUM)

### 13.6. ~~MEDIUM~~ RESOLVED: brain_lock = NoopLock потенціал

> **Вирішено:** `brain_lock` тепер `threading.RLock()` (не NoopLock). Захищає concurrent access.

**Severity:** RESOLVED

### 13.7. LOW: Proactive context = 4 brain searches + 1 file read при "wake up"

**Де:** `proactive_context.py:135`

**Факт:** `_get_proactive_context_locked()` робить 4 brain.search() виклики + 1 filesystem read:
1. `brain.search(tags=["scheduled-task"], limit=20)` (line 146)
2. `brain.search(tags=["session-summary"], limit=2)` (line 184)
3. `data/history/*.json` — filesystem read останнього JSON файлу для попереднього діалогу (lines 194-211, **NOT** brain search)
4. `brain.search(tags=["outcome-failure"], limit=5)` (line 215)
5. `brain.search(tags=["autonomous-outcome"], limit=5)` (line 235)

**Проблема:** При великій базі це може бути 500ms+. Але кешується на 5 хв → тільки перший запит повільний.

**Severity:** LOW (amortized by cache)

### 13.8. LOW: Memory level drift

**Де:** `background_brain.py:111`

**Факт:** Без регулярного запуску `fix_memory_levels()` записи можуть "застрягти" на L4_IDENTITY хоча мають бути на L3_DOMAIN.

**Наслідок:** Записи не decay'яться (IDENTITY майже вічна) → база росте необмежено → recall стає повільнішим.

**Severity:** LOW (повільна деградація, не immediate failure)

---

## 14. Стратегічні рекомендації (v2.x roadmap)

### 14.1. ~~Перенести композицію пошуку в Rust~~ IMPLEMENTED (Aura SDK 1.3.2)

> **Реалізовано:** `recall_full()` в `D:\aura\src\aura.rs`, PyO3 wrapper `py_recall_full()`, Python shim `_AuraCompat.recall_full()`.

**Що зроблено:**
- `recall_full(query, top_k, include_failures, min_strength, expand_connections, session_id)` — Rust метод
- Stage 1: `recall_core()` (RRF + activation) — власний lock
- Stage 2+3: substring + failure records — merged в ONE read lock pass
- PyO3 dict з повним `metadata` (для trust scoring в Python)
- `hasattr` guard в `tool_dispatch.py` — BC з SDK <1.3.0
- Aura SDK version bump: 1.0.4 → 1.0.5

**Результат:** Recall 1 FFI crossing замість 3. Stages 2+3 = один pass по records замість двох.

### 14.2. ~~Динамічне завантаження правил~~ IMPLEMENTED (Lazy System Prompt)

> **Реалізовано:** `system_instruction.py` — модульні правила з умовним додаванням.

**Що зроблено:**
- `_INTERACTIVE_RULES` (~250 tokens) — витягнуто з inline правил, додається тільки для voice/telegram/desktop
- `_BROWSER_RULES` (~400 tokens) — витягнуто, додається тільки якщо `BROWSER_ENABLED=True`
- Core rules залишаються завжди

**Результат:** Autonomous без browser = ~650 tokens менше. Простий desktop запит без browser = ~400 tokens менше.

~~**Залишкове TODO:** Можна далі розбити на RESEARCH_RULES, FINANCIAL_RULES, AUTONOMOUS_RULES для ще більшої економії.~~ **DONE v2.3**: Extracted `_RESEARCH_RULES`, `_PLANNING_RULES`, `_EXECUTION_GUARD_RULES`, `_DELEGATION_RULES`. Voice/proactive ~500 tokens saved. 24 tests.

### 14.8. ~~Tool Trust Classification (AUTON-9)~~ IMPLEMENTED

> **Реалізовано:** `tool_trust.py` — AST-based аналіз безпеки sandbox-інструментів.

**Що зроблено:**
- `_SafetyVisitor` (ast.NodeVisitor) — детектує небезпечні паттерни: network calls, subprocess, file-write, dynamic-exec
- `classify_tool_source(code)` → `TrustClassification` (trust_level: safe/moderate/dangerous, reasons, patterns)
- `should_auto_approve(tool_path)` → `(bool, reason)` — блокує dangerous від auto-approve
- `check_progressive_trust(manifest)` → список інструментів для підвищення довіри (>10 successful runs)
- `find_retired_tools(manifest, days=90)` → інструменти без використання >90 днів
- `format_trust_report(classification)` → людино-читабельний звіт
- Інтеграція: `_sandbox_create_tool` автоматично класифікує, `_sandbox_test_tool` перевіряє trust перед auto-approve
- 26 тестів

### 14.9. ~~Proactive Error Escalation (AUTON-10)~~ IMPLEMENTED

> **Реалізовано:** `error_escalation.py` — автоматична ескалація та відновлення при деградації системи.

**Що зроблено:**
- `DegradationLevel` IntEnum: GREEN(0), YELLOW(1), RED(2)
- `SystemHealth` dataclass з health indicators (llm_ok, memory_ok, budget_pct, tool_failures, level)
- `assess_system_health()` — оцінює стан системи за budget, tool failures, memory
- `build_alert_message(health)` → структуроване повідомлення для Telegram
- `send_critical_alert(health)` → Telegram сповіщення (cooldown 10 хв)
- `attempt_auto_recovery(health)` → пробує дешевий LLM call перед ескалацією
- `get_detailed_health()` → JSON для `/api/health/detailed` ендпоінту
- `get_recovery_suggestions(health)` → список рекомендацій для кожного рівня
- Інтеграція: `_run_maintenance` в autonomy.py перевіряє health → auto-recovery → critical alert
- 32 тести

### 14.10. ~~Tool Health Visibility & Adaptive Routing (AUTON-11)~~ IMPLEMENTED

> **Реалізовано:** `tool_routing.py` — видимість здоров'я інструментів та adaptive fallback.

**Що зроблено:**
- `_ALTERNATIVE_ROUTES` — маршрути fallback (web_search→http_get, browse_page→http_get, тощо)
- `get_alternatives(tool_name)` → список альтернатив з оцінкою надійності
- `get_best_alternative(tool_name)` → найкраща доступна альтернатива
- `get_tool_status_report()` → JSON з healthy/degraded/unavailable інструментами
- `format_tool_health_for_prompt(report)` → текст для decision prompt
- `test_tools_on_startup()` → async перевірка при старті
- `tool_status` FunctionDeclaration — агент може запитати стан інструментів
- Інтеграція: `_build_decision_prompt` в autonomy.py показує деградовані інструменти + альтернативи
- 15 тестів

### 14.11. ~~Dynamic Context Window & State Summarization (AUTON-12)~~ IMPLEMENTED

> **Реалізовано:** `context_window.py` — динамічне управління контекстним вікном для автономного режиму.

**Що зроблено:**
- `estimate_complexity(goal, attempts)` → 0.0-1.0 складність за ключовими словами, довжиною, кроками, спробами
- `context_size_for_complexity(complexity)` → 12-28 повідомлень (v2.4: зменшено з 12-48; scratchpad компенсує)
- `dynamic_keep_recent(channel, goal)` → замінює статичні threshold для autonomous
- `score_message_importance(msg)` → 0.0-1.0 за типом (System=1.0, Human=0.7-0.9, Tool error=0.8, Tool ok=0.3)
- `select_important_messages(msgs, budget, always_keep_recent)` → intelligent pruning
- `SessionState` dataclass + `update_session_state()` — трекінг прогресу, findings, blockers
- `get_state_summary(session_id)` → summary після 10+ дій
- `should_inject_state(session_id, action_count)` → інжект кожні 10 дій
- Інтеграція: `_estimate_keep_recent` в agent.py використовує dynamic_keep_recent для autonomous channel
- 37 тестів

### 14.12. ~~Plan Invalidation & Re-Planning (AUTON-14)~~ IMPLEMENTED

> **Реалізовано:** `plan_invalidation.py` — динамічна інвалідація планів та адаптивне перепланування.

**Що зроблено:**
- `PlanHealthCheck` dataclass — valid, needs_update, abandon, confidence, suggested_action
- `check_plan_validity(plan, result, success, failures)` → health check після кожного кроку
- `_detect_prerequisite_needed(result)` → keyword detection для "need to install", "permission denied", тощо
- `update_plan_confidence(plan_id, success)` / `get_plan_confidence()` / `should_replan()` — confidence tracking
- Confidence decay: -0.2 per failure, +0.05 per success, replan trigger при <0.3
- `insert_prerequisite(plan, step)` → вставка prerequisite кроку в позицію current_step
- `abandon_plan(plan)` → позначає план як abandoned + event_bus emission
- `build_replan_context(plan, failures)` → контекст для перегенерації плану
- `process_step_result(plan, result, success)` → health check + confidence update (integration helper)
- Інтеграція: autonomy.py замінює пряме advance_plan на process_step_result → abandon/replan/continue
- 27 тестів

### 14.13. ~~Confidence-Based Autonomy Levels (AUTON-15)~~ IMPLEMENTED

> **Реалізовано:** `confidence_autonomy.py` — градієнтна автономність на основі впевненості.

**Що зроблено:**
- `AutonomyAction`: EXECUTE_SILENT (>0.8), EXECUTE_NOTIFY (0.5-0.8), REQUEST_GUIDANCE (0.3-0.5), SKIP (<0.3)
- `ConfidenceFactors` dataclass — domain_familiarity, tool_reliability, goal_clarity, budget_health, recent_success_rate
- `compute_confidence(factors)` → зважене середнє (domain 35%, tool 20%, recent 20%, clarity 15%, budget 10%)
- `infer_domain(description)` → 6 доменів (research, file_ops, memory, web, planning, communication)
- `get_domain_confidence()` / `record_domain_outcome()` — per-domain success tracking з recency weighting
- `record_user_decision()` / `get_calibrated_thresholds()` — user trust calibration (approvals → нижчі пороги)
- `assess_action_confidence(description, budget, tool_issues, successes, failures)` → (score, action)
- `_assess_goal_clarity(description)` → оцінка специфічності опису (довжина, ключові слова, дієслова)
- `format_confidence_info(confidence, action, domain)` → людино-читабельний рядок
- Інтеграція: autonomy.py записує domain outcomes після кожного evaluation
- 31 тести

### 14.3. ~~Working Memory Ring (Scratchpad)~~ IMPLEMENTED

> **Реалізовано:** `scratchpad.py` — робочий блокнот агента з ring buffer евікцією.

**Що зроблено:**
- `scratchpad` brain tool (write/read/clear) — в CORE_TOOL_NAMES (завжди доступний)
- Notes зберігаються як L1_WORKING записи з тегом `scratchpad`
- Ring buffer: макс. 20 нотаток, найстаріші автоматично видаляються при overflow
- Monotonic `seq` counter для стабільного сортування навіть при однаковому `time.time()`
- `get_scratchpad_context()` → `[SCRATCHPAD — N working notes]` — інжектується в `call_model()` як SystemMessage
- Інтеграція: agent.py інжектує scratchpad після session context + brain context
- 21 тест

### 14.4. ~~Selective recall cache invalidation~~ IMPLEMENTED

> **Реалізовано:** `tool_utils.py` — `clear_recall_cache(new_content)` з keyword matching.

**Що зроблено:**
- `clear_recall_cache(new_content="")` — без аргументів = full clear (backward compatible)
- З `new_content` — витягує перші 10 keywords, видаляє тільки cache entries з keyword overlap
- Call sites в tool_dispatch.py: `store` та `update_record` передають content
- `delete_record`, `connect_records`, `update_persona` — full clear (немає content для selective)
- 10 тестів

### 14.5. ~~Context deduplication~~ IMPLEMENTED

> **Реалізовано:** `agent.py` — `_extract_record_ids()` + dedup фільтр в `_inject_context()`.

**Що зроблено:**
- `_extract_record_ids(text)` — regex `\[id:([^\]]+)\]` витягує record IDs з recall output
- `_inject_context()` збирає record IDs з system instruction (messages[0])
- Лінії recall output, чиї record IDs вже є в system instruction, видаляються
- Лінії без record IDs — зберігаються (не фільтруються)
- Якщо всі лінії дубльовані → повертає None (нічого не інжектується)
- 10 тестів

### 14.6. ~~Recall latency monitoring~~ IMPLEMENTED

> **Реалізовано:** `metrics.py` + `tool_dispatch.py` — Prometheus gauge з per-stage timing.

**Що зроблено:**
- `MetricsCollector.record_recall_latency(total_ms, rrf_ms, substr_ms, failure_ms)` — збирає дані
- `remy_recall_latency_ms{stage="total|rrf|substring|failure"}` — Prometheus gauge
- `remy_recall_calls_total` — counter
- Timing в обох шляхах recall (unified + legacy)
- Logger: `"recall_full: %.0fms query=%r"` / `"recall pipeline: total=%.0fms rrf=%.0fms substr=%.0fms fail=%.0fms"`

Деталі: Розділ 8.1.

### 14.7. ~~Adaptive compact_history~~ IMPLEMENTED

> **Реалізовано:** `agent.py` — `_estimate_keep_recent()` + `_KEEP_RECENT` dict.

**Що зроблено:**
- `_KEEP_RECENT = {"autonomous": 20, "research": 24, "default": 16}` (v2.4: зменшено з 40/32 — scratchpad компенсує)
- `_estimate_keep_recent(channel, user_message)` (`agent.py:94`) — перевіряє channel + `_RESEARCH_KEYWORDS` (EN+UK)
- Для autonomous: делегує до `context_window.dynamic_keep_recent()` (AUTON-12)
- Підтримує `str` та `HumanMessage` (multimodal → fallback на default)
- Обидва виклики (invoke_agent + invoke_agent_stream) оновлені
- 11 тестів в `test_agent.py::TestEstimateKeepRecent`

Деталі: Розділ 3.2.

---

## 15. Відкриті питання / TODO

### Аналіз підсистем
- [ ] Проаналізувати consolidation детальніше (MinHash + LLM cluster merge pipeline)
- [ ] Проаналізувати worker.py (як worker отримує tools, як merge результатів)
- [ ] Проаналізувати approval_queue.py (як саме працює sync wait + Telegram reply)
- [ ] Проаналізувати browser pipeline (screenshot → vision → structured JSON)
- [ ] Проаналізувати eval_metrics і feedback signals flow
- [ ] Проаналізувати gemini_live.py voice path (чому окремий від LangGraph?)

### Тестування edge cases
- [ ] Що буде якщо brain = пустий при першому запуску?
- [x] Дублювання контексту між system_instruction recall та _inject_context recall — наскільки overlap? — **FIXED v2.3**: Context deduplication (14.5) removes duplicate `[id:xxx]` lines
- [ ] Виміряти реальний recall latency на 1K / 5K / 10K записів
- [x] Протестувати compact_history при 20+ tool calls — що втрачається? — **FIXED v2.3**: tool-heavy conversations now preserve recent tool sequences (12 tests in test_compact_history_deep.py)

### Рекомендації до імплементації (з Розділу 14)
- [x] **v2.1:** Adaptive compact_history (Розділ 14.7) — **DONE**
- [x] **v2.1:** Recall latency monitoring (Розділ 14.6) — **DONE**
- [x] **v2.1:** Lazy System Prompt (Розділ 14.2) — **DONE** (partial: interactive + browser rules)
- [x] **v2.3:** Selective recall cache invalidation (Розділ 14.4) — **DONE**
- [x] **v2.1:** Aura unified recall (Розділ 14.1) — **DONE** (recall_full in Aura SDK 1.3.2)
- [x] **v2.3:** Working Memory Ring / Scratchpad (Розділ 14.3) — **DONE**
- [x] **v2.3:** Context deduplication (Розділ 14.5) — **DONE**
- [x] **v2.2:** Tool Trust Classification, AUTON-9 (Розділ 14.8) — **DONE**
- [x] **v2.2:** Proactive Error Escalation, AUTON-10 (Розділ 14.9) — **DONE**
- [x] **v2.2:** Tool Health Visibility & Adaptive Routing, AUTON-11 (Розділ 14.10) — **DONE**
- [x] **v2.2:** Dynamic Context Window, AUTON-12 (Розділ 14.11) — **DONE**
- [x] **v2.2:** Plan Invalidation & Re-Planning, AUTON-14 (Розділ 14.12) — **DONE**
- [x] **v2.2:** Confidence-Based Autonomy Levels, AUTON-15 (Розділ 14.13) — **DONE**

---

## 16. Changelog

### v2.4 (2026-03-02) — Context Tuning & Documentation Sync

**Архітектурне тюнінг** — оптимізація token budgets та context sizing на основі production досвіду:

| # | Зміна | Файл | Деталі |
|---|-------|------|--------|
| 1 | **Reduced `_KEEP_RECENT`** | `agent.py:87` | autonomous: 40→20, research: 32→24 (scratchpad компенсує) |
| 2 | **Increased RAG token_budget** | `agent.py:1366` | `_inject_context()`: 600→1200 tokens (health metrics, time-sensitive data) |
| 3 | **Increased RAG char limit** | `agent.py:1428` | 4800→8000 chars (~2000 tokens max) |
| 4 | **Temporal supplement** | `agent.py:1387` | Fetches recent 7-day records for temporal queries (yesterday, today, last week) |
| 5 | **`_expand_relative_dates()`** | `agent.py:1288` | Expands "вчора", "минулого тижня" to concrete dates for better recall |
| 6 | **AgentState caching** | `agent.py:170` | `_cached_session_ctx` + `_cached_scratchpad` — computed once per invoke |
| 7 | **Priority gate** | `agent.py:797` | `invoke_agent()` → `_invoke_agent_inner()` з interactive priority mechanism |
| 8 | **Dynamic context window max** | `context_window.py` | MAX reduced 48→28 (scratchpad compensates) |

**Ключові зміни:**
- **Context budget redistribution** — менший keep_recent (менше старих повідомлень) + більший RAG budget (більше релевантних пам'яті) + scratchpad (робочі нотатки) = краща якість контексту
- **Temporal awareness** — запити з temporal signals автоматично доповнюються останніми записами за 7 днів
- **State caching** — session context та scratchpad обчислюються один раз per-invoke замість per-tool-iteration
- **LIVEAGENT.md sync** — документ оновлено з актуальними лінійними номерами, значеннями, та структурними описами

---

### v2.3 (2026-02-27) — Memory & Context Optimization

**7 архітектурних змін**, реалізовані за рекомендаціями з Розділу 14:

| # | Зміна | Файл | Тести |
|---|-------|------|-------|
| 1 | **Working Memory Ring / Scratchpad** | `scratchpad.py`, `agent.py`, `tool_declarations.py`, `tool_dispatch.py` | 21 |
| 2 | **Selective recall cache invalidation** | `tool_utils.py`, `tool_dispatch.py` | 10 |
| 3 | **Context deduplication** | `agent.py` | 10 |
| 4 | **compact_history fix** | `agent.py` | 12 |
| 5 | **Autonomous scratchpad** | `autonomy.py` | — |
| 6 | **Modular system prompt rules** | `system_instruction.py` | 24 |

**Ключові зміни:**
- **Scratchpad** — `scratchpad` tool (write/read/clear) в CORE_TOOL_NAMES. L1_WORKING записи з ring buffer (max 20). Контекст автоматично інжектується в call_model()
- **Selective cache** — `clear_recall_cache(new_content)` зкидає тільки cache entries з keyword overlap замість повного clear. Batch operations (extract_facts, research) більше не вбивають весь кеш
- **Context dedup** — `_inject_context()` витягує record IDs з system instruction через regex `[id:xxx]` і фільтрує дубльовані лінії з recall output
- **compact_history fix** — tool-heavy conversations (20+ tool calls) тепер зберігають recent tool sequences замість видалення всього. Якщо немає HumanMessage в recent_part — tool sequences лишаються; тільки orphan ToolMessages видаляються
- **Autonomous scratchpad** — `_build_decision_prompt()` в autonomy.py інжектує scratchpad контекст між циклами
- **Modular rules** — 4 нових conditional blocks витягнуто з монолітного `base`: `_RESEARCH_RULES`, `_PLANNING_RULES`, `_EXECUTION_GUARD_RULES`, `_DELEGATION_RULES`. Voice/proactive = ~500 tokens менше. Desktop/autonomous = повний набір

**Тести:** 1650 Python tests pass, 0 failures (77 нових тестів)

---

### v2.2 (2026-02-27) — Autonomous Intelligence Sprint (AUTON-7 → AUTON-15)

**9 нових модулів** для розумнішої, безпечнішої та адаптивнішої автономності:

| # | Модуль | Файл | AUTON | Тести |
|---|--------|------|-------|-------|
| 1 | **Tool Trust Classification** | `tool_trust.py` | AUTON-9 | 26 |
| 2 | **Proactive Error Escalation** | `error_escalation.py` | AUTON-10 | 32 |
| 3 | **Tool Health Visibility & Routing** | `tool_routing.py` | AUTON-11 | 15 |
| 4 | **Dynamic Context Window** | `context_window.py` | AUTON-12 | 37 |
| 5 | **Plan Invalidation & Re-Planning** | `plan_invalidation.py` | AUTON-14 | 27 |
| 6 | **Confidence-Based Autonomy** | `confidence_autonomy.py` | AUTON-15 | 31 |

**Також реалізовані (AUTON-1 → AUTON-8, AUTON-13)** в попередніх сесіях:
- AUTON-1: Smart Goal Generation
- AUTON-2: Sub-Goal Decomposition
- AUTON-3: Human-in-the-Loop Approval
- AUTON-4: Session Reflection
- AUTON-5: Adaptive Strategy Analysis
- AUTON-6: External Tools (read/write file, http_get)
- AUTON-7: Action Plans (multi-step sequential plans)
- AUTON-8: Decision Tree Plans (branching plans)
- AUTON-13: Event Bus (inter-component communication)

**Ключові архітектурні зміни v2.2:**
- **Gradient autonomy** — 4 рівні автономності замість binary execute/skip
- **Domain tracking** — система вчиться з кожного домену (research, web, file_ops, memory, planning, communication)
- **Plan health** — confidence decay при failures, auto-replan при <0.3, prerequisite discovery
- **AST safety** — sandbox-інструменти класифікуються як safe/moderate/dangerous перед auto-approve
- **Error escalation** — GREEN/YELLOW/RED рівні з auto-recovery та Telegram alerts
- **Tool routing** — fallback альтернативи при circuit breaker open
- **Dynamic context** — 12-48 повідомлень залежно від складності задачі

**Нові ендпоінти:**
- `GET /api/health/detailed` — детальний стан системи
- `tool_status` brain tool — стан інструментів для агента

**Тести:** 1573 Python tests pass, 0 failures (168 нових тестів)

---

### v2.1 (2026-02-26) — Performance & Quality Sprint

**4 архітектурні зміни**, реалізовані за рекомендаціями з Розділу 14:

| # | Зміна | Файли | Тести |
|---|-------|-------|-------|
| 1 | **Adaptive compact_history** | `agent.py`, `test_agent.py` (+11 tests) | `_estimate_keep_recent()`, `_KEEP_RECENT` dict |
| 2 | **Lazy System Prompt** | `system_instruction.py` | `_INTERACTIVE_RULES`, `_BROWSER_RULES` extracted |
| 3 | **Recall latency monitoring** | `tool_dispatch.py`, `metrics.py` | `record_recall_latency()`, Prometheus gauge |
| 4 | **Unified recall (Rust)** | `aura.rs`, `Cargo.toml`, `pyproject.toml`, `agent_tools.py`, `tool_dispatch.py`, `test_proactivity.py` | `recall_full()`, `py_recall_full()`, hasattr guard |

**Aura SDK update:**
- Метод `recall_full()` — unified RRF + substring + failure recall
- PyO3 wrapper `py_recall_full()` з повним metadata dict
- As of v1.3.2: `store_with_channel()`, `feedback()`, `connect()`, namespace isolation, native `tier_stats()`, `promotion_candidates()`, 8-phase `run_maintenance()` with `MaintenanceReport`

**Ризики mitigated:**
- 13.2 (aggressive compact_history): HIGH → LOW
- 13.3 (triple I/O recall): HIGH → LOW

**Тести:** 1103 Python tests pass, 33 E2E skipped
