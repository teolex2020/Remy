"""A/B test: ACL cognitive brief vs. raw message transcript.

Measures:
  1. Context size (chars + approximate tokens) for brief vs. full transcript.
  2. Hallucination rate: we ask Gemini factual questions whose answers are
     grounded in the real brain state. We compare:
       - Answer when LLM receives the ACL brief.
       - Answer when LLM receives a simulated long transcript (noise).
       - Answer when LLM receives NEITHER (baseline hallucination).
     Ground truth is read directly from the Aura SDK for comparison.

Run: ACL_BRIEF_ENABLED=1 python scripts/test_acl_brief_vs_transcript.py
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


def fake_transcript(brain, n_messages: int = 40) -> str:
    """Simulate a long tool-call transcript of the kind that accumulates
    in the autonomous loop — a lot of operational chatter with the real
    cognitive signal buried inside.
    """
    aura = brain._aura
    lines = []
    # Real cognitive signal (what ACL brief would capture).
    try:
        digest = aura.get_memory_health_digest(8)
        lines.append(f"[tool:get_memory_health_digest] total_records={digest.total_records}, dominant_phase={digest.latest_dominant_phase}")
        for iss in digest.top_issues[:5]:
            lines.append(f"  issue: ns={iss.namespace} title={iss.title} score={iss.score:.2f}")
    except Exception:
        pass
    try:
        cases = aura.get_conflict_cases(5)
        lines.append(f"[tool:get_conflict_cases] returned {len(cases)} cases")
        for c in cases:
            lines.append(f"  conflict: belief_id={c.belief_id[:16]} severity={c.severity} strength={c.evidence_strength:.2f}")
    except Exception:
        pass

    # Now pad with typical noise — the kind of iteration chatter that blows up context.
    noise_patterns = [
        "[tool:search_records] query='daily plan' hits=0",
        "[tool:get_active_goals] returned 3 goals",
        "[AI thought] Let me check the survival status before proceeding.",
        "[tool:check_survival] balance=0.00 runway_days=0 status=CRITICAL",
        "[AI thought] No funds — need to use free model and focus on earning.",
        "[tool:list_records] ns='marketing' returned 12 records, showing first 3:",
        "[AI thought] I see several outreach drafts. Let me review them.",
        "[tool:read_record] id='abc123' content='Draft post about Aura SDK...'",
        "[tool:web_search] q='aurasdk' returned 8 results",
        "[AI thought] Checking if there are any new mentions online.",
    ]
    idx = 0
    while len(lines) < n_messages:
        lines.append(noise_patterns[idx % len(noise_patterns)])
        idx += 1
    return "\n".join(lines)


QUESTIONS = [
    (
        "Скільки всього записів зберігає мозок агента зараз?",
        "total_records",
    ),
    (
        "Який домінуючий епістемічний шар (dominant_phase) у поточному стані?",
        "dominant_phase",
    ),
    (
        "Скільки активних конфліктних кластерів у пам'яті агента?",
        "contradiction_cluster_count",
    ),
    (
        "Скільки переконань з високою волатильністю зафіксовано?",
        "high_volatility_belief_count",
    ),
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
    """One-shot Gemini call. Uses free lite model to avoid spending wallet."""
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
    """Returns (matches, reason). Lenient matching — the model may paraphrase."""
    a = answer.lower()
    if "не відомо" in a or "не знаю" in a or "не надано" in a or "відсутн" in a:
        return False, "refusal"
    if isinstance(truth_value, int):
        # Number must appear in answer.
        if str(truth_value) in answer:
            return True, "contains number"
        # Check for obvious hallucinated numbers.
        import re
        nums = re.findall(r"\d+", answer)
        if nums:
            return False, f"wrong number(s): {nums} vs. truth {truth_value}"
        return False, "no number found"
    else:
        # String (dominant_phase).
        return (str(truth_value).lower() in a), f"expected '{truth_value}'"


def main():
    a = Aura(BRAIN_PATH)
    class B: pass
    brain = B(); brain._aura = a

    truth = ground_truth(a)
    print("=== GROUND TRUTH (from Aura SDK) ===")
    for k, v in truth.items():
        print(f"  {k} = {v}")
    print()

    # Build contexts.
    brief = build_cognitive_brief(brain, locale="ua")
    transcript = fake_transcript(brain, n_messages=40)

    print("=== CONTEXT SIZES ===")
    print(f"  ACL brief     : {len(brief):>6} chars, ~{estimate_tokens(brief):>5} tokens")
    print(f"  Raw transcript: {len(transcript):>6} chars, ~{estimate_tokens(transcript):>5} tokens")
    compression = 1 - (len(brief) / max(1, len(transcript)))
    print(f"  Compression   : {compression*100:.1f}% smaller")
    print()

    # Ask each question under 3 conditions.
    results = {"brief": {"correct": 0, "refusal": 0, "wrong": 0, "details": []},
               "transcript": {"correct": 0, "refusal": 0, "wrong": 0, "details": []},
               "none": {"correct": 0, "refusal": 0, "wrong": 0, "details": []}}

    for question, key in QUESTIONS:
        truth_val = truth[key]
        print(f"\n--- Q: {question}")
        print(f"    truth: {truth_val}")

        for cond, ctx in [("brief", brief), ("transcript", transcript), ("none", "(no context)")]:
            try:
                ans = ask_gemini(ctx, question)
            except Exception as exc:
                ans = f"ERROR: {exc}"
            correct, reason = check_answer(ans, truth_val)
            bucket = results[cond]
            if correct:
                bucket["correct"] += 1
                tag = "OK "
            elif reason == "refusal":
                bucket["refusal"] += 1
                tag = "REF"
            else:
                bucket["wrong"] += 1
                tag = "HAL"
            ans_short = ans.replace("\n", " ")[:120]
            print(f"    [{cond:10s}] {tag}  {ans_short}")
            bucket["details"].append((question, truth_val, ans_short, tag))
            time.sleep(1.0)  # stay polite to API

    print("\n=== SUMMARY ===")
    n = len(QUESTIONS)
    for cond, r in results.items():
        print(f"  {cond:10s}: correct={r['correct']}/{n}, refusals={r['refusal']}, hallucinations={r['wrong']}")

    print("\n=== INTERPRETATION ===")
    print(f"  Hallucination rate with ACL brief:    {results['brief']['wrong']}/{n}")
    print(f"  Hallucination rate with transcript:   {results['transcript']['wrong']}/{n}")
    print(f"  Hallucination rate with NO context:   {results['none']['wrong']}/{n}")
    print(f"  Context saved vs. transcript:         {compression*100:.1f}%")


if __name__ == "__main__":
    main()
