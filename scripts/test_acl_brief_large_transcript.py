"""Large-transcript A/B test: ACL brief vs. realistic autonomous-loop history.

Simulates what actually accumulates in the autonomous channel after N
iterations of tool calls: each iteration adds AI thought + tool call +
tool result, with occasional large `list_records` dumps. This is the
context shape that was blowing past Gemini's 1M token limit.

Measures:
  - Context size at 100, 300, 600, 1000 iterations.
  - Hallucination rate at each size when asked grounded factual questions.
  - Compression ratio vs. the same ACL brief.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from aura import Aura
from remy.core.cognitive_brief import build_cognitive_brief, estimate_tokens


BRAIN_PATH = str(Path(__file__).parent.parent / "data" / "brain")


def realistic_transcript(brain, iterations: int = 300) -> str:
    """Simulate real autonomous-loop accumulation.

    Each iteration ≈ 4-6 messages (AI thought, tool call, tool result,
    sometimes large record dump). Every ~20 iterations we include one
    real cognitive digest to make the signal *available* — the test is
    whether the LLM can find it in the noise.
    """
    aura = brain._aura
    lines = []

    try:
        digest = aura.get_memory_health_digest(8)
        real_digest_lines = [
            f"[tool_result:get_memory_health_digest] total_records={digest.total_records}, "
            f"dominant_phase={digest.latest_dominant_phase}, "
            f"contradiction_cluster_count={digest.contradiction_cluster_count}, "
            f"high_volatility_belief_count={digest.high_volatility_belief_count}",
        ]
        for iss in digest.top_issues[:5]:
            real_digest_lines.append(
                f"  issue: ns={iss.namespace} title={iss.title} score={iss.score:.2f}"
            )
    except Exception:
        real_digest_lines = ["[tool_result:get_memory_health_digest] (unavailable)"]

    try:
        cases = aura.get_conflict_cases(5)
        real_conflict_lines = [
            f"[tool_result:get_conflict_cases] returned {len(cases)} cases"
        ]
        for c in cases[:5]:
            real_conflict_lines.append(
                f"  conflict: belief_id={c.belief_id[:16]} severity={c.severity} strength={c.evidence_strength:.2f}"
            )
    except Exception:
        real_conflict_lines = ["[tool_result:get_conflict_cases] (unavailable)"]

    noise_block = [
        "[AI thought] Let me check the survival status before proceeding.",
        "[tool_call:check_survival]",
        "[tool_result:check_survival] balance=0.00 runway_days=0 status=CRITICAL model=free",
        "[AI thought] No funds available, sticking to free model and earning-focused tasks.",
        "[tool_call:get_active_goals]",
        "[tool_result:get_active_goals] 3 goals: survival (immortal), aurasdk_promotion, content_drafting",
        "[AI thought] Current task is drafting an outreach post about Aura SDK.",
        "[tool_call:search_records] q='outreach post draft' limit=5",
        "[tool_result:search_records] 4 hits:",
        "  [id:a1b2c3] Draft v1: 'Aura SDK — the cognitive layer your AI agent needs...'",
        "  [id:d4e5f6] Draft v2: 'How to give your agent persistent memory without embeddings...'",
        "  [id:g7h8i9] Research note: LinkedIn AI/ML groups best posting times",
        "  [id:j0k1l2] Feedback from previous post: 3 upvotes, 1 comment asking for demo",
        "[AI thought] Let me refine draft v2 — it has clearer value proposition.",
        "[tool_call:web_search] q='cognitive memory AI agent LinkedIn traction'",
        "[tool_result:web_search] 8 results, top 3 shown:",
        "  Result 1: 'Why agent memory is the next frontier' (medium.com, 2026-04-08)",
        "  Result 2: 'Vector DB vs cognitive layers — a practical comparison' (substack)",
        "  Result 3: 'Aura SDK benchmarks' (hn thread, 142 points)",
        "[AI thought] Interesting — there's an HN thread. Let me extract and read it.",
        "[tool_call:extract_content] url='https://news.ycombinator.com/item?id=...'",
        "[tool_result:extract_content] (2.3 KB of thread content)",
        "[AI thought] Mostly positive, a few criticisms about lack of benchmarks.",
        "[tool_call:store] content='HN feedback: users want published benchmarks' ns=marketing",
        "[tool_result:store] stored as id=m2n3o4",
    ]

    i = 0
    while i < iterations:
        # Every 20 iterations, inject the real signal (so LLM COULD find it).
        if i % 20 == 0:
            lines.extend(real_digest_lines)
            lines.extend(real_conflict_lines)
        lines.extend(noise_block)
        i += len(noise_block) // 4  # rough iteration proxy
    return "\n".join(lines)


QUESTIONS = [
    ("Скільки всього записів зберігає мозок агента зараз?", "total_records"),
    ("Який домінуючий епістемічний шар (dominant_phase) у поточному стані?", "dominant_phase"),
    ("Скільки активних конфліктних кластерів у пам'яті агента?", "contradiction_cluster_count"),
    ("Скільки переконань з високою волатильністю зафіксовано?", "high_volatility_belief_count"),
]


def ground_truth(aura):
    d = aura.get_memory_health_digest(8)
    return {
        "total_records": d.total_records,
        "dominant_phase": d.latest_dominant_phase,
        "contradiction_cluster_count": d.contradiction_cluster_count,
        "high_volatility_belief_count": d.high_volatility_belief_count,
    }


def ask_gemini(context: str, question: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    system = (
        "Ти — агент Remy. Відповідай коротко і ТОЧНО на основі наданого контексту. "
        "Якщо в контексті немає інформації для відповіді — скажи 'НЕ ВІДОМО'. "
        "Не вигадуй числа."
    )
    prompt = f"{system}\n\n=== CONTEXT ===\n{context}\n\n=== QUESTION ===\n{question}\n\n=== ANSWER ==="
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def check_answer(answer: str, truth_value) -> tuple[bool, str]:
    a = answer.lower()
    if "не відомо" in a or "не знаю" in a or "не надано" in a or "відсутн" in a:
        return False, "refusal"
    if isinstance(truth_value, int):
        if str(truth_value) in answer:
            return True, "contains number"
        import re
        nums = re.findall(r"\d+", answer)
        if nums:
            return False, f"wrong number(s): {nums} vs. truth {truth_value}"
        return False, "no number found"
    else:
        return (str(truth_value).lower() in a), f"expected '{truth_value}'"


def run_condition(label, context, truth, question_limit=None):
    questions = QUESTIONS[:question_limit] if question_limit else QUESTIONS
    correct = refusal = hallucination = 0
    details = []
    for q, key in questions:
        try:
            ans = ask_gemini(context, q)
        except Exception as exc:
            ans = f"ERROR: {exc}"
        ok, reason = check_answer(ans, truth[key])
        if ok:
            correct += 1; tag = "OK "
        elif reason == "refusal":
            refusal += 1; tag = "REF"
        else:
            hallucination += 1; tag = "HAL"
        details.append((q, truth[key], ans.replace("\n", " ")[:100], tag))
        time.sleep(1.0)
    return correct, refusal, hallucination, details


def main():
    a = Aura(BRAIN_PATH)
    class B: pass
    brain = B(); brain._aura = a

    truth = ground_truth(a)
    print("=== GROUND TRUTH (from Aura SDK) ===")
    for k, v in truth.items():
        print(f"  {k} = {v}")
    print()

    brief = build_cognitive_brief(brain, locale="ua")
    print(f"=== ACL BRIEF ===  {len(brief)} chars, ~{estimate_tokens(brief)} tokens")
    print()

    # Test brief once as baseline.
    print("[Baseline] Asking brief…")
    bc, br, bh, _ = run_condition("brief", brief, truth)
    print(f"  brief: correct={bc}/4, refusals={br}, hallucinations={bh}")
    print()

    # Growing transcript.
    for iters in [100, 300, 600, 1000]:
        tr = realistic_transcript(brain, iterations=iters)
        tr_chars = len(tr)
        tr_tokens = estimate_tokens(tr)
        compression = 1 - (len(brief) / max(1, tr_chars))

        print(f"=== TRANSCRIPT @ {iters} iters ===")
        print(f"  size       : {tr_chars:>7} chars, ~{tr_tokens:>6} tokens")
        print(f"  compression: brief is {compression*100:.1f}% smaller")

        # Guard: if transcript > 900K tokens, skip (Gemini limit is 1M).
        if tr_tokens > 900_000:
            print("  [skip] transcript exceeds 900K tokens, Gemini would reject")
            continue

        tc, tr_ref, th, details = run_condition(f"transcript@{iters}", tr, truth)
        print(f"  result     : correct={tc}/4, refusals={tr_ref}, hallucinations={th}")
        for q, tv, ans, tag in details:
            print(f"    [{tag}] {q[:45]:45s} truth={tv}  answer={ans}")
        print()


if __name__ == "__main__":
    main()
