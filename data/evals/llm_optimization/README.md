# LLM Optimization Eval Corpus

This directory contains repeatable eval cases for Remy's context reduction and
session-state experiments.

The goal is not to prove that shorter prompts are always better. The goal is to
measure three things together:

- accuracy: the answer must preserve required facts exactly;
- prompt cost: raw transcript vs projected state vs incremental state;
- break-even point: whether the state update cost is worth paying.

## Case Schema

Each line in `cases.jsonl` is one JSON object:

```json
{
  "id": "exact_authorization_code",
  "category": "exact_facts",
  "risk": "medium",
  "messages": [
    {"role": "user", "content": "Critical fact: authorization code is RX-4471."},
    {"role": "assistant", "content": "Stored."}
  ],
  "question": "What is the authorization code? Answer only the code.",
  "expected_fragments": ["RX-4471"],
  "expected_any": [["terse", "brevity", "brief", "concise"]],
  "must_not_contain": ["RX-0000"],
  "notes": "Exact value retention after noisy context."
}
```

`expected_fragments` are strict required substrings. Use them for exact values,
IDs, dates, paths, and other facts that must survive verbatim.

`expected_any` is optional and contains synonym groups. At least one fragment in
each group must appear. Use it sparingly for semantic wording checks where exact
word choice is not the point.

Use synthetic data only. Do not place real personal, medical, customer, or
production secrets in this corpus.

## Running

Dry-run validates parsing and local prompt sizing without API calls:

```powershell
python tools\validation\llm_optimization_corpus_eval.py --dry-run
```

Real projected-state run:

```powershell
python tools\validation\llm_optimization_corpus_eval.py --mode projected --model gemini-flash-lite-latest
```

Full comparison:

```powershell
python tools\validation\llm_optimization_corpus_eval.py --mode all --model gemini-flash-lite-latest --delay-sec 1
```

Reports are written to `data/llm_optimization/`.
