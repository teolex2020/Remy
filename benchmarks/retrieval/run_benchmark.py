"""Brain-native retrieval benchmark runner.

Reads benchmark_v1.yaml, invokes the live web_search tool per case, collects
candidates, and emits a per-case report plus aggregate metrics.

Scope of Week 0 runner:
  - static assertions only: domain matches, candidate counts, source class heuristics
  - agent-level assertions (e.g. "agent_asserts_authors_without_fetch") are
    logged as TODO — they require Phase 1 tool-split enforcement before they
    can be meaningfully evaluated.

Usage:
    python benchmarks/retrieval/run_benchmark.py
    python benchmarks/retrieval/run_benchmark.py --case c03_site_constraint_arxiv
    python benchmarks/retrieval/run_benchmark.py --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from remy.core.tool_dispatch import execute_tool  # noqa: E402
from remy.core.brain_tools import (  # noqa: E402
    _reset_search_intent_counter,
    _same_intent_counter,
    _normalize_search_intent,
)
from remy.core.agent_tools import brain  # noqa: E402


# Metadata fields that count as "freshness signal" for brain_writes_without_
# freshness_metadata. Phase 4 will formalise this; for Week 0 we accept any
# of these as sufficient to prove the write was time-aware.
_FRESHNESS_FIELDS = (
    "freshness",
    "volatility",
    "fresh_until",
    "retrieved_at",
    "ingested_at",
    "fetched_at",
)


def _brain_record_count_safe() -> int:
    try:
        return int(brain.count())
    except Exception:
        return -1


def _same_intent_count(session_id: str, query: str) -> int:
    """Peek at the same-intent counter for this (session, query) pair."""
    key = (session_id, _normalize_search_intent(query))
    return _same_intent_counter.get(key, 0)


def _snapshot_cache_ids() -> set:
    """Snapshot ids of web-search-cache records (the infra write path)."""
    try:
        recs = brain.search(query="", tags=["web-search-cache"], limit=500)
        return {getattr(r, "id", None) for r in recs if getattr(r, "id", None)}
    except Exception:
        return set()


def _collect_new_cache_records(prior_ids: set) -> list:
    try:
        recs = brain.search(query="", tags=["web-search-cache"], limit=500)
    except Exception:
        return []
    return [r for r in recs if getattr(r, "id", None) and r.id not in prior_ids]


def _records_have_freshness(records) -> bool:
    for rec in records:
        meta = getattr(rec, "metadata", None) or {}
        if any(f in meta for f in _FRESHNESS_FIELDS):
            return True
    return False


# Infra-only tags that do NOT represent analysis-path claim writes. A record
# carrying one of these is bookkeeping (cache, deduplication index, telemetry)
# — it never promotes into durable belief. Counting them against the learning
# boundary would mis-classify the retrieval subsystem's own cache as a belief
# write. brain_writes_analysis excludes them; brain_writes_total still records
# the raw delta for observability.
_INFRA_TAGS = frozenset({"web-search-cache"})


def _is_infra_record(rec) -> bool:
    tags = getattr(rec, "tags", None) or []
    meta = getattr(rec, "metadata", None) or {}
    if any(t in _INFRA_TAGS for t in tags):
        return True
    if meta.get("type") == "web_search_cache":
        return True
    return False


def _bust_web_search_cache() -> int:
    """Delete any cached web_search records so the run is truly live.

    Without this, a prior run's cache serves every query in <0.1s and
    brain_writes measurements become meaningless (nothing actually fetched,
    nothing actually written). Returns count busted.
    """
    try:
        cached = list(brain.search(query="", tags=["web-search-cache"], limit=500))
    except Exception:
        return 0
    busted = 0
    for rec in cached:
        rid = getattr(rec, "id", None)
        if not rid:
            continue
        try:
            brain.delete(rid)
            busted += 1
        except Exception:
            pass
    return busted


_SEO_MIRROR_PATTERNS = (
    "best", "top-", "top10", "top-10",
    "medium.com", "towardsdatascience.com",
    "paperswithcode.com/paper/",
    "deeplearn.org", "alphaxiv.org", "arxiv-sanity",
)


def _domain(url: str) -> str:
    return urllib.parse.urlsplit(url or "").netloc.lower().removeprefix("www.")


def _is_seo_mirror(url: str, title: str) -> bool:
    u = (url or "").lower()
    t = (title or "").lower()
    for p in _SEO_MIRROR_PATTERNS:
        if p in u or p in t:
            return True
    return False


def _run_case(case: dict) -> dict:
    case_id = case["id"]
    query = case["query"]
    expected = case.get("expected", {}) or {}
    forbidden = case.get("forbidden", {}) or {}
    success_class = case.get("success")

    # Each case gets a fresh session id so the same-intent counter is scoped
    # per-case, not shared across the benchmark run.
    session_id = f"bench:{case_id}"
    _reset_search_intent_counter()

    # Brain baseline: total count + snapshot of cache-tagged ids.
    #
    # The Aura SDK's search/list_records has an in-process caching layer that
    # can return stale views of newly-inserted records during the same
    # process lifetime, so we cannot rely on an "all records" ID diff. Instead
    # we use two complementary signals:
    #   • brain.count() delta — reliable "was anything written" counter
    #   • search(tags=[web-search-cache]) id diff — reliable infra-write
    #     classifier because the tag query path does refresh per-call
    # Any count-delta beyond cache writes is attributed to the analysis path.
    brain_count_before = _brain_record_count_safe()
    prior_cache_ids = _snapshot_cache_ids()

    t0 = time.time()
    try:
        raw = execute_tool("web_search", {"query": query}, session_id=session_id)
        payload = json.loads(raw)
        error = payload.get("error")
    except Exception as exc:
        payload = {}
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0

    sources = payload.get("sources") or []
    candidate_count = len(sources)
    mode = payload.get("mode")

    issues: list[str] = []
    checks: dict[str, bool | int | str] = {}

    if "min_candidates" in expected:
        ok = candidate_count >= expected["min_candidates"]
        checks[f"min_candidates>={expected['min_candidates']}"] = ok
        if not ok:
            issues.append(
                f"min_candidates failed: got {candidate_count} < {expected['min_candidates']}"
            )

    # For no_result / honest_refusal cases, max_candidates is waived if the
    # tool returned a structured honest_refusal OR if a same-intent retry cap
    # is demonstrated on the 4th call. This matches Phase 1 semantics: the
    # correct behavior isn't "return <=N junk candidates" but "refuse honestly
    # when the intent is unanswerable."
    if "max_candidates" in expected and success_class != "honest_refusal":
        ok = candidate_count <= expected["max_candidates"]
        checks[f"max_candidates<={expected['max_candidates']}"] = ok
        if not ok:
            issues.append(
                f"max_candidates failed: got {candidate_count} > {expected['max_candidates']}"
            )

    if success_class == "honest_refusal":
        # Run 3 more calls with the same query; the 4th must be mode=honest_refusal.
        cap_fired = False
        for _ in range(3):
            again = json.loads(execute_tool("web_search", {"query": query}, session_id=session_id))
            if again.get("mode") == "honest_refusal":
                cap_fired = True
                break
        checks["same_intent_cap_fires"] = cap_fired
        checks["first_call_mode"] = mode
        if not cap_fired:
            issues.append(
                "honest_refusal expected: same-intent retry cap did not fire within 4 calls"
            )

    if "expected_domains_any_of" in expected:
        want = {d.lower() for d in expected["expected_domains_any_of"]}
        got_domains = {_domain(s.get("uri", "")) for s in sources}
        hit = any(
            d == w or d.endswith("." + w) for d in got_domains for w in want
        )
        checks[f"any_domain_in_{sorted(want)}"] = hit
        if not hit:
            issues.append(
                f"no candidate from expected domains {sorted(want)}; got {sorted(got_domains)}"
            )

    if "all_candidates_must_match_domain" in expected:
        want = expected["all_candidates_must_match_domain"].lower()
        all_match = all(
            _domain(s.get("uri", "")) == want
            or _domain(s.get("uri", "")).endswith("." + want)
            for s in sources
        ) if sources else False
        checks[f"all_match_{want}"] = all_match
        if sources and not all_match:
            off = [
                s.get("uri") for s in sources
                if not (
                    _domain(s.get("uri", "")) == want
                    or _domain(s.get("uri", "")).endswith("." + want)
                )
            ]
            issues.append(
                f"site-constraint violated: off-domain candidates present: {off}"
            )

    if forbidden.get("seo_mirror_in_top_3"):
        top3 = sources[:3]
        seo_hits = [s.get("uri") for s in top3 if _is_seo_mirror(s.get("uri", ""), s.get("title", ""))]
        ok = len(seo_hits) == 0
        checks["no_seo_mirror_in_top_3"] = ok
        if not ok:
            issues.append(f"SEO/mirror in top-3: {seo_hits}")

    if forbidden.get("off_domain_in_top_10"):
        want = expected.get("all_candidates_must_match_domain", "").lower()
        if want:
            top10 = sources[:10]
            off = [
                s.get("uri") for s in top10
                if not (
                    _domain(s.get("uri", "")) == want
                    or _domain(s.get("uri", "")).endswith("." + want)
                )
            ]
            ok = len(off) == 0
            checks["no_off_domain_in_top_10"] = ok
            if not ok:
                issues.append(f"off-domain in top-10: {off}")

    # ── Brain-level assertions (activated Week 0 + Phase 1/2 learning boundary) ──
    # Measure the full case window: primary call + honest_refusal re-calls. Then
    # split writes into infra (web-search-cache: bookkeeping, not belief) vs
    # analysis (anything else: that's the learning-boundary violation). The
    # cache write is retrieval subsystem bookkeeping — it never promotes into
    # durable belief — so it's excluded from analysis accounting.
    brain_count_after = _brain_record_count_safe()
    brain_writes_total = (
        brain_count_after - brain_count_before
        if brain_count_before >= 0 and brain_count_after >= 0
        else -1
    )
    new_cache_records = _collect_new_cache_records(prior_cache_ids)
    brain_writes_infra = len(new_cache_records)
    brain_writes_analysis = max(0, brain_writes_total - brain_writes_infra)
    # For freshness inspection we have the infra (cache) records directly; any
    # analysis-path records are unobservable here, but that case is a hard
    # fail anyway so the distinction doesn't matter.
    new_records = list(new_cache_records)

    checks["brain_writes_total"] = brain_writes_total
    checks["brain_writes_analysis"] = brain_writes_analysis
    checks["brain_writes_infra"] = brain_writes_infra

    if forbidden.get("brain_writes"):
        ok = brain_writes_analysis == 0
        checks["no_analysis_brain_writes"] = ok
        if not ok:
            issues.append(
                f"analysis path wrote to brain: {brain_writes_analysis} non-infra record(s)"
            )

    if forbidden.get("silent_overwrite_of_prior_belief"):
        # Week 0 proxy: without a conflict ledger (Phase 4), the only way
        # analysis could silently mutate belief is via a non-infra brain
        # write. Infra cache writes do not count.
        ok = brain_writes_analysis == 0
        checks["no_silent_overwrite"] = ok
        if not ok:
            issues.append(
                f"conflict probe caused {brain_writes_analysis} analysis write(s); "
                "silent overwrite risk until Phase 4 conflict ledger lands"
            )

    if forbidden.get("brain_writes_without_freshness_metadata"):
        # Every write in the case window (infra + analysis) must carry some
        # freshness signal — especially important on volatile queries like c10.
        if not new_records:
            checks["freshness_on_writes"] = True
        else:
            ok = _records_have_freshness(new_records)
            checks["freshness_on_writes"] = ok
            if not ok:
                issues.append(
                    f"{len(new_records)} brain write(s) on volatile query "
                    "without any freshness/volatility metadata"
                )

    if forbidden.get("retry_count_gt_3"):
        # Cap = 3 primary calls before honest_refusal takes over.
        all_calls = _same_intent_count(session_id, query)
        ok = all_calls <= 3 or (
            success_class == "honest_refusal"
            and checks.get("same_intent_cap_fires") is True
        )
        checks["retry_count_le_3"] = ok
        checks["retry_count_observed"] = all_calls
        if not ok:
            issues.append(
                f"same-intent retry cap exceeded: {all_calls} calls without "
                "honest_refusal"
            )

    # Remaining agent-level TODOs (require full chat-turn runner, not tool shell)
    todo_assertions = [
        k for k in forbidden
        if k in {
            "fabricated_authors",
            "agent_asserts_title_not_in_fetched_content",
            "agent_asserts_authors_without_fetch",
            "agent_fabricates_sources",
        }
    ]

    passed = len(issues) == 0 and error is None

    return {
        "id": case_id,
        "query": query,
        "success_class": case.get("success"),
        "elapsed_s": round(elapsed, 2),
        "candidate_count": candidate_count,
        "brain_writes_total": brain_writes_total,
        "brain_writes_analysis": brain_writes_analysis,
        "brain_writes_infra": brain_writes_infra,
        "intent_calls_primary": _same_intent_count(session_id, query),
        "error": error,
        "mode": payload.get("mode"),
        "top_candidates": [
            {"title": s.get("title"), "uri": s.get("uri")}
            for s in sources[:5]
        ],
        "checks": checks,
        "issues": issues,
        "todo_assertions_phase1": todo_assertions,
        "passed": passed,
    }


def _run_phase4_probe() -> dict:
    """End-to-end probe for Phase 4 truth-pressure surfacing.

    The main benchmark only calls web_search, so it can't validate that
    store_research actually stamps prior records with unresolved_conflict.
    This probe:
      1. Seeds a prior 'langchain 0.1.0' research record
      2. Calls store_research with conflicting 'langchain 0.4.0' findings
      3. Asserts conflict=True + flagged_prior_ids non-empty in the JSON
      4. Re-reads the prior record and asserts unresolved_conflict=True
      5. Re-calls store_research with conflict_resolution='replace' and
         asserts prior gets superseded_by=<new_id>
    """
    from remy.core.retrieval.freshness import truth_status
    session_id = "bench:phase4_probe"

    issues: list[str] = []
    checks: dict[str, object] = {}

    # Clean any prior probe residue from earlier runs.
    try:
        residue = brain.search(query="", tags=["research", "langchain-version-probe"], limit=50) or []
        for r in residue:
            try:
                brain.delete(r.id)
            except Exception:
                pass
    except Exception:
        pass

    # Step 1: seed prior belief.
    seed_raw = execute_tool(
        "store_research",
        {
            "topic": "langchain-version-probe",
            "findings": "langchain 0.1.0 is the current stable release as of the seed.",
            "volatility": "high",
        },
        session_id=session_id,
    )
    seed = json.loads(seed_raw)
    seed_id = seed.get("id")
    checks["seed_stored"] = bool(seed_id)
    if not seed_id:
        issues.append(f"seed step did not return an id: {seed}")
        return {"id": "phase4_probe", "passed": False, "checks": checks, "issues": issues}

    # Step 2: call with conflicting findings, default (flag) mode.
    conflict_raw = execute_tool(
        "store_research",
        {
            "topic": "langchain-version-probe",
            "findings": "langchain 0.4.0 is now the current stable release.",
            "volatility": "high",
        },
        session_id=session_id,
    )
    conflict_resp = json.loads(conflict_raw)
    checks["flag_response_stored_false"] = conflict_resp.get("stored") is False
    checks["flag_response_conflict_true"] = conflict_resp.get("conflict") is True
    flagged_ids = conflict_resp.get("flagged_prior_ids") or []
    checks["flagged_ids_non_empty"] = len(flagged_ids) > 0
    checks["seed_in_flagged_ids"] = seed_id in flagged_ids

    if conflict_resp.get("stored") is not False:
        issues.append(f"expected stored=False on conflict flag, got {conflict_resp.get('stored')}")
    if not flagged_ids:
        issues.append("flagged_prior_ids empty — conflict_flag_metadata not persisted")

    # Step 3: re-read prior and verify unresolved_conflict=True.
    prior = brain.get(seed_id)
    prior_meta = dict(getattr(prior, "metadata", {}) or {}) if prior else {}
    checks["prior_unresolved_conflict"] = prior_meta.get("unresolved_conflict") is True
    checks["prior_truth_status"] = truth_status(prior_meta)
    if prior_meta.get("unresolved_conflict") is not True:
        issues.append(
            f"prior record {seed_id} not stamped unresolved_conflict; meta={prior_meta}"
        )
    if truth_status(prior_meta) != "conflict_unresolved":
        issues.append(
            f"truth_status on flagged prior != conflict_unresolved (got {truth_status(prior_meta)})"
        )

    # Step 4: retry with conflict_resolution='replace' — expect supersession stamp.
    replace_raw = execute_tool(
        "store_research",
        {
            "topic": "langchain-version-probe",
            "findings": "langchain 0.4.0 is now the current stable release.",
            "volatility": "high",
            "conflict_resolution": "replace",
        },
        session_id=session_id,
    )
    replace_resp = json.loads(replace_raw)
    new_id = replace_resp.get("id")
    checks["replace_response_stored_true"] = replace_resp.get("stored") is True
    stamped = replace_resp.get("stamped_superseded_ids") or []
    checks["stamped_superseded_ids_non_empty"] = len(stamped) > 0
    if not new_id:
        issues.append(f"replace step did not return new id: {replace_resp}")
    if not stamped:
        issues.append("stamped_superseded_ids empty — supersede_metadata not persisted")

    # Step 5: re-read prior after replace — expect superseded status.
    prior2 = brain.get(seed_id)
    prior2_meta = dict(getattr(prior2, "metadata", {}) or {}) if prior2 else {}
    checks["prior_superseded_by"] = prior2_meta.get("superseded_by")
    checks["prior_truth_status_after_replace"] = truth_status(prior2_meta)
    if str(prior2_meta.get("superseded_by")) != str(new_id):
        issues.append(
            f"prior.superseded_by != new_id ({prior2_meta.get('superseded_by')!r} vs {new_id!r})"
        )
    if truth_status(prior2_meta) != "superseded":
        issues.append(
            f"truth_status after replace != superseded (got {truth_status(prior2_meta)})"
        )

    # Cleanup probe records so repeated runs don't accumulate.
    try:
        for rid in {seed_id, new_id} - {None, ""}:
            brain.delete(rid)
    except Exception:
        pass

    return {
        "id": "phase4_probe",
        "passed": len(issues) == 0,
        "checks": checks,
        "issues": issues,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench",
        default=str(Path(__file__).parent / "benchmark_v1.yaml"),
    )
    ap.add_argument("--case", help="Run only this case id")
    ap.add_argument("--out", help="Write JSON report to this path")
    ap.add_argument(
        "--no-cache-bust",
        action="store_true",
        help="Skip deleting web-search cache before run (default: bust cache so "
             "timings and brain-write measurements reflect a live run, not cache hits).",
    )
    args = ap.parse_args()

    if not args.no_cache_bust:
        busted = _bust_web_search_cache()
        print(f"[setup] cache-bust: cleared {busted} web-search-cache record(s)")

    with open(args.bench, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    cases = doc.get("cases", [])
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case with id={args.case}", file=sys.stderr)
            return 2

    results = []
    for case in cases:
        print(f"[run] {case['id']}: {case['query']!r}")
        r = _run_case(case)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"       -> {status}  candidates={r['candidate_count']}  "
            f"elapsed={r['elapsed_s']}s"
        )
        for issue in r["issues"]:
            print(f"         ! {issue}")
        if r["todo_assertions_phase1"]:
            print(
                f"         ~ phase1 todo: {r['todo_assertions_phase1']}"
            )

    # Phase 4 probe: end-to-end conflict surfacing through store_research.
    # Only runs when we're executing the full benchmark (no --case filter).
    phase4 = None
    if not args.case:
        print()
        print("[run] phase4_probe: store_research conflict surfacing")
        phase4 = _run_phase4_probe()
        p4_status = "PASS" if phase4["passed"] else "FAIL"
        print(f"       -> {p4_status}  checks={len(phase4['checks'])}")
        for issue in phase4["issues"]:
            print(f"         ! {issue}")

    summary = {
        "backend": doc.get("backend"),
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "phase4_probe_passed": bool(phase4 and phase4["passed"]),
    }
    print()
    print(f"summary: {summary['passed']}/{summary['total']} passed")
    if phase4 is not None:
        print(f"phase4_probe: {'PASS' if phase4['passed'] else 'FAIL'}")

    report = {"summary": summary, "results": results, "phase4_probe": phase4}
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"report written to {args.out}")

    overall_pass = summary["failed"] == 0 and (phase4 is None or phase4["passed"])
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
