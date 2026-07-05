# Remy - Local AI Workflow Automation

Remy is a local-first AI automation app for documents, memory, research, reminders, and scheduled workflows on your PC.

The product promise is not "another chat assistant." Remy turns your own model key or local Ollama setup into ready workflows: summarize documents, extract deadlines, create daily briefs, research a topic, monitor changes, and save reviewable memory.

The reliability layer remains the technical moat: every action passes through consequence memory (has this failed before?), epistemic governance (is this claim verified?), and a factual boundary (is this a grounded fact or the agent's own generated text?). These systems should appear to users as dry runs, run history, local secrets, human approval, failure auto-pause, and unverified-memory labels.

Runs fully locally. No cloud subscriptions beyond LLM API keys; supports Ollama for fully offline operation.

---

## The Reliability Layer

Three deterministic (non-LLM) subsystems gate everything the agent does. This is what separates Remy from a tool-calling loop.

### 1. Consequence Memory & Scar Protection

Every action outcome is stored as a lived consequence — **SUPPORTS** (it worked) or **REFUTES** (it failed). Before the agent acts or revises a plan, the consequence gate queries this history:

- Actions previously refuted in a similar situation are surfaced as hard constraints to the planner — not suggestions.
- **Scar protection:** a refutation that was later "washed out" by occasional successes is *still* surfaced as a warning. Frequency cannot overwrite a lived failure. This prevents the classic agent pathology of retrying a known-bad approach because it worked once.

Consequence memory influences plan revision, autonomous step execution, and research source selection (hosts that produced low-quality findings are demoted).

### 2. Epistemic Governance

Every claim the agent makes is classified along three axes — claim class × epistemic status × knowledge origin — producing an entitlement decision: *allowed*, *requires evidence*, *requires downgrade*, or *forbidden*.

- **Phantom detection:** external claims without verification are counted; when the unverified ratio crosses thresholds, responses are rewritten with uncertainty markers or blocked entirely.
- Claim types are tracked explicitly: observed, memory-based, inference, research, temporal ("latest", "today"). Unverified claims get caveats; phantom citations are flagged.
- Sensitive data (emails, wallets, credentials) captured by autonomous agents is stored as non-actionable until a human verifies it.

### 3. Factual Boundary (Admission Classes)

Every memory record carries an admission class. When the agent assembles an evidence packet to answer a factual question, records classed as `working_state`, `plan`, `reflection`, or `generated_analysis` are **structurally excluded** — only `grounded_external_fact`, `operator_asserted`, and `research_artifact` pass through.

The agent architecturally cannot recall its own speculation as a fact.

---

## Architecture

Remy v3 is a **mission-driven runtime**: missions contain tasks, tasks are scheduled and delegated to specialist agents, and every execution passes through governance gates.

```
┌─ Channels: Web GUI · Desktop · Telegram · Voice · Headless ──────┐
│                                                                   │
│  Mission Runtime                                                  │
│    Scheduler ─► ChiefAgent.run_cycle()                            │
│       ├─ GoalTracker          (mission → tasks)                   │
│       ├─ ExecutionGate        (approval · risk · budget checks)   │
│       ├─ ConsequenceGate      (refuted-action blocking)           │
│       ├─ DelegationEngine ──► Specialist agents (tool access)     │
│       └─ EvaluationEngine     (outcome verdict + factuality)      │
│                                                                   │
│  Governance: PolicyEngine · BudgetEngine · ApprovalEngine ·       │
│              AuditEngine · Epistemic Governance                   │
│                                                                   │
│  Memory: Aura (Rust/PyO3) — recall · promotion · trust decay ·    │
│          consequence verdicts · admission classes                 │
│                                                                   │
│  Tools: web search · browser (Playwright) · files · HTTP ·        │
│         documents (PDF/PPTX) · sandbox tools · scheduling         │
└───────────────────────────────────────────────────────────────────┘
```

- Outcomes are persisted per cycle; an outcome learner extracts reusable playbooks from repeated successes.
- The legacy v2 goal-driven loop remains available (`AUTONOMY_V3=false`) during the v3 transition.
- All channels share one brain — memory stored in a web session is available in Telegram and vice versa, protected by a single lock across concurrent channels.

---

## Channels

| Channel | Description |
|---------|-------------|
| **Web GUI** | Browser-based chat at `localhost:8080` with real-time streaming |
| **Desktop** | Native window via PyWebView — no browser needed |
| **Telegram** | Full chat and operator supervision over Telegram bot |
| **Voice** | Real-time conversation via Gemini Live (microphone input, spoken responses) |
| **Autonomous** | Runs without any user — pursues missions on its own schedule |

---

## Memory System

Persistent cognitive memory powered by [aura-memory](https://pypi.org/project/aura-memory/) (Rust native extension, sub-millisecond recall).

| Level | Purpose | Persistence |
|-------|---------|-------------|
| `L1_WORKING` | Conversation context, search results, ephemeral data | Decays quickly |
| `L2_DECISIONS` | Action history, choices made, outcomes, consequences | Medium-term |
| `L3_DOMAIN` | Stable facts, curated knowledge, plans, research | Long-lived |
| `L4_IDENTITY` | Core beliefs about user and agent | Permanent |

**Memory operations:** semantic recall ranked by relevance/recency/trust, exact-match lookup, graph edges between records with weighted traversal, auto-consolidation of near-duplicates (85%+ similarity), user verification (trust boost), stale-marking without deletion, and protected records (credentials require user verification to read).

**Memory safety:** auto-quarantine of a corrupted store with replay-based recovery, promotion gates (edges only created if content passes factuality filters), trust/confidence decay over time, thread safety across all channels.

---

## Autonomous Missions

When autonomy is enabled, Remy runs an independent mission loop with layered controls.

### Governance & Budgets

Budget is checked **before** each LLM call; sessions auto-stop when exhausted.

| Limit | Default | Setting |
|-------|---------|---------|
| Daily tokens | 100,000 | `AUTONOMY_DAILY_TOKEN_LIMIT` |
| Hourly tokens | 20,000 | `AUTONOMY_HOURLY_TOKEN_LIMIT` |
| Daily cost cap | $5.00 | `AUTONOMY_DAILY_COST_LIMIT_USD` |
| Max actions/hour | 20 | `AUTONOMY_MAX_ACTIONS_PER_HOUR` |
| Session duration | 30 min | `AUTONOMY_MAX_SESSION_MINUTES` |
| Quiet hours | 23:00–07:00 | `AUTONOMY_QUIET_HOURS_*` |

### Human-in-the-Loop Approval

Critical actions pause and wait for explicit confirmation:

- Auto-triggered for wallet/payment/financial tools, browser actions on financial or registration URLs, storage of sensitive data (seed phrases, keys, IBAN, cards), and any tool call with money-movement arguments.
- Approval card appears in the Web GUI and as a Telegram message; timeout or rejection unblocks the mission safely.
- Optional multi-model safety review runs before presenting to the human.
- REST API for manual resolution: `POST /api/approvals/{id}/approve` / `/reject`.

### Planning & Replanning

- Linear plans and decision-tree plans (branching success/failure paths, per-node retry limits).
- Plans are revised automatically on repeated step failure — and the revision prompt receives consequence-memory constraints, so the new plan cannot reuse a refuted approach.

### Specialist Delegation

Complex missions are delegated to specialist agents (researcher, planner, executor, analyst) running in parallel, all sharing the same governed tool layer. When stuck, the agent asks the operator for guidance over Telegram instead of guessing.

### Capability Packs

Structured operating profiles that scope tools and guardrails per task type:

| Pack | Purpose | Key guardrails |
|------|---------|----------------|
| **market_research** | Competitive analysis, OSINT | 2+ sources required, contradictions tagged, citations mandatory |
| **signup_operator** | Account registration flows | No payment info, stop on captcha/SMS, 3 login failures → stop |
| **publisher** | Content publishing | Never publish live without approval, stop at draft |
| **monitoring** | Track changes on websites | Read-only, report diffs vs snapshot |
| **general** | Default mode | No additional restrictions |

---

## Web Research

Multi-cycle research pipeline that runs until the question is actually answered:

1. Query planning (2–7 targeted queries per mode) → multi-backend search (DuckDuckGo, Brave, Google, Mojeek, Startpage, Yahoo) with 24h cache
2. Content extraction to clean Markdown (`trafilatura`)
3. Deterministic source ranking: official docs > research papers > news > forums > SEO spam
4. Exact + semantic deduplication (cosine threshold 0.82)
5. Outcome evaluation — an LLM judges whether the question is answered; unanswered sub-questions become next-cycle queries
6. Contradiction resolution — conflicting sources trigger tie-breaker queries for a third authoritative source
7. Saturation detection — stops when 3 consecutive cycles add nothing new

Research sessions persist across restarts: queries, sources fetched/accepted/rejected, findings with confidence scores, citation coverage.

---

## Tools & Capabilities

| Area | What's included |
|------|-----------------|
| **Browser automation** | Playwright (Chromium): navigate, analyze interactive elements, click/type/fill forms. Persistent cookies/localStorage, daily action limits, SSRF protection, financial URLs auto-route to approval queue |
| **Document generation** | PDF reports (sections, tables, findings, audit trail), PowerPoint decks, AI images. Full Cyrillic/Unicode support |
| **Files & HTTP** | Read from allowed paths, write restricted to `data/`, raw HTTP GET for APIs |
| **Task management** | Todos with priority/category/deadline/recurrence, cron-style scheduled actions, active tasks surfaced in context automatically |
| **Sandbox tool system** | The agent (or user) writes new Python tools at runtime, tests them in isolated subprocesses, and deploys them after approval. Skills export/import as portable packages |
| **People & profile** | Contact and relationship tracking, structured user profile, editable agent persona (tone, traits, language) |
| **Runtime directives** | Persistent injected instructions ("always respond in Ukrainian"), session-scoped or permanent, with optional TTL |

---

## Cost Governance

All LLM calls are tracked in real time: per-model USD breakdown, hourly/daily/session/lifetime totals, budget enforcement *before* each call (GCRA-style). Visible in the Web GUI settings panel and via `/api/autonomy/status`.

An optional **LLM optimization layer** in the Web GUI reduces prompt size (context reduction), routes simple/demanding turns to cheaper/stronger models, answers exact repeated questions from a verified answer cache, and records measured savings (tokens saved, calls avoided, latency deltas). A/B mode lets you compare normal vs optimized answers before enabling.

---

## Security Model

Remy is a **local desktop-style agent, not a hosted SaaS app**. The Web UI has no accounts or login — the local server binds to `127.0.0.1` only, so the interface is reachable only from the machine that launched it. On Windows, the user account and filesystem permissions are the security boundary.

Telegram is the one external channel: set `TELEGRAM_ALLOWED_CHAT_IDS` to restrict who can operate the agent.

Additional layers: a PII vault tokenizes sensitive data before LLM API calls and restores it only in local responses; browser automation blocks localhost and private IP ranges (SSRF protection).

---

## Installation

```bash
git clone <repo>
cd remy/app
python -m venv .venv
.venv\Scripts\activate

pip install -e .
pip install vendor/aura_memory-1.5.4-cp312-cp312-win_amd64.whl

cp .env.example .env
# optional for development: edit .env and set GEMINI_API_KEY

remy-app             # installed desktop-style launcher target
remy --desktop       # developer CLI: native desktop window
remy --serve         # developer CLI: local web interface at 127.0.0.1:8080
remy --autonomous-v3 # developer CLI: v3 mission runtime, headless
remy                 # developer CLI: voice mode, microphone required
```

If the project folder was moved or renamed, refresh the editable install from
the `app` directory:

```powershell
python -m pip install -e . --no-deps
```

For end users the target distribution is a Windows installer/executable with a Start Menu shortcut — no Docker, Python, or terminal required. First run opens the app even without an API key; chat stays read-only until keys are added from Settings. The local server must remain bound to `127.0.0.1`.

---

## Configuration

All settings via `.env` or environment variables; runtime settings changed in the Web UI survive restarts and override `.env`.

```env
# LLM
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=...           # optional
OPENROUTER_API_KEY=...       # optional
OLLAMA_BASE_URL=http://127.0.0.1:11434   # optional local LLMs

# Channels
TELEGRAM_BOT_TOKEN=...
PROACTIVE_CHAT_ID=...
WEB_PORT=8080

# Autonomy
AUTONOMY_ENABLED=false
AUTONOMY_V3=true             # v3 mission runtime (v2 goal loop if false)
AUTONOMY_DAILY_TOKEN_LIMIT=100000
AUTONOMY_DAILY_COST_LIMIT_USD=5.00
AUTONOMY_QUIET_HOURS_START=23
AUTONOMY_QUIET_HOURS_END=7

# Safety
APPROVAL_QUEUE_ENABLED=true
APPROVAL_TIMEOUT_SEC=120
REVIEW_ENABLED=true
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + uvicorn (Python 3.10+) |
| Agent framework | LangGraph + Google Gemini (`google-genai`) |
| Memory | aura-memory (Rust/PyO3 native extension) |
| Web search | ddgs — DuckDuckGo, Brave, Google, Mojeek, Startpage, Yahoo |
| Content extraction | trafilatura |
| Browser automation | Playwright (Chromium) |
| Desktop UI | PyWebView |
| Voice | Gemini Live API |
| Telegram | python-telegram-bot |
| Documents | ReportLab, python-pptx, PyMuPDF |
| Optional LLMs | OpenAI, Anthropic Claude, Ollama (local) |

---

## Engineering Notes

- **Deterministic gates, not LLM vibes** — consequence memory, epistemic governance, and admission classes are plain code with tests, not prompt engineering
- **Graceful degradation** — each subsystem fails independently; a broken search cache or missing optional package doesn't crash the agent
- **Lazy imports** — LangGraph/langchain load on first chat message, not at startup
- **Multi-channel safe** — a single brain lock prevents memory corruption across concurrent Web + Telegram + autonomous access
- **No mandatory cloud** — fully offline except LLM API calls; Ollama supported for local-only operation
- **Tested** — 3,200+ test functions across 200 files, including dedicated suites for admission classes, retrieval boundaries, and consequence gating
