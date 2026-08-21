"""Prototype a binary/state readout layer for LLM optimization research.

This is intentionally not a compression runner. It models the AuraSDK idea:
stable answers are indexed offline into a scoped binary snapshot and runtime
requests either hit a governed packet, block, or fall back to raw context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from remy.core.session_state_wrapper import estimate_tokens  # noqa: E402
from tools.validation.llm_optimization_corpus_eval import (  # noqa: E402
    DEFAULT_CASES,
    GeminiMeter,
    SYSTEM_PROMPT,
    build_prompt_for_mode,
    check_answer,
    load_cases,
)


SNAPSHOT_MODES = ("raw", "binary_snapshot_readout", "binary_snapshot_or_fallback")
TRAFFIC_PROFILES = ("exact", "paraphrase", "mixed")
DEFAULT_SCOPE = "remy-local"
DEFAULT_DOMAIN = "llm-optimization"
DEFAULT_SOURCE = "synthetic-corpus"
LOOKUP_MAGIC = b"RBSL"
STRINGS_MAGIC = b"RBSS"
SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class SnapshotRecord:
    query: str
    context_key: str
    answer_text: str
    status: str = "answered"
    source_id: str = ""
    record_kind: str = "answer_packet"


@dataclass(frozen=True)
class RuntimeRequest:
    case: Any
    query: str
    request_kind: str


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def context_component(value: str, fallback: str) -> str:
    normalized = normalize_text(value)
    return normalized.replace("|", "-").replace(":", "-") if normalized else fallback


def source_scope_context_key(*, scope: str, domain: str, source: str) -> str:
    return (
        f"scope:{context_component(scope, DEFAULT_SCOPE)}|"
        f"domain:{context_component(domain, DEFAULT_DOMAIN)}|"
        f"source:{context_component(source, DEFAULT_SOURCE)}"
    )


def hash64(value: str) -> int:
    return struct.unpack("<Q", sha256(str(value or "").casefold().encode("utf-8")).digest()[:8])[0]


def composite_key_hash(query: str, context_key: str) -> int:
    return hash64(f"{query.casefold()}\x1f{context_key.casefold()}")


def answer_text_for_case(case: Any) -> str:
    parts = list(case.expected_fragments)
    for group in case.expected_any:
        if group:
            parts.append(group[0])
    return " ".join(parts)


def runtime_requests_for_cases(cases: list[Any], *, traffic_profile: str) -> list[RuntimeRequest]:
    requests: list[RuntimeRequest] = []
    for index, case in enumerate(cases):
        if traffic_profile == "exact":
            query = case.question
            request_kind = "exact"
        elif traffic_profile == "paraphrase":
            query = f"Using the saved context, answer this in the same meaning: {case.question}"
            request_kind = "paraphrase"
        elif traffic_profile == "mixed":
            if index % 2 == 0:
                query = case.question
                request_kind = "exact"
            else:
                query = f"Using the saved context, answer this in the same meaning: {case.question}"
                request_kind = "paraphrase"
        else:
            raise ValueError(f"Unsupported traffic profile: {traffic_profile}")
        requests.append(RuntimeRequest(case=case, query=query, request_kind=request_kind))
    return requests


def case_with_runtime_question(case: Any, query: str) -> Any:
    return replace(case, question=query)


def snapshot_build_cases(cases: list[Any], *, coverage_pct: float) -> list[Any]:
    if coverage_pct <= 0:
        return []
    if coverage_pct >= 100:
        return cases
    count = int((len(cases) * coverage_pct) / 100)
    if count <= 0 and cases:
        count = 1
    return cases[:count]


def snapshot_records_from_cases(
    cases: list[Any],
    *,
    scope: str = DEFAULT_SCOPE,
    domain: str = DEFAULT_DOMAIN,
    source: str = DEFAULT_SOURCE,
) -> list[SnapshotRecord]:
    context_key = source_scope_context_key(scope=scope, domain=domain, source=source)
    return [
        SnapshotRecord(
            query=case.question,
            context_key=context_key,
            answer_text=answer_text_for_case(case),
            source_id=case.case_id,
            record_kind=case.category,
        )
        for case in cases
    ]


def _write_segment_strings(path: Path, strings: list[str]) -> list[tuple[int, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    refs: list[tuple[int, int]] = []
    for value in strings:
        data = str(value or "").encode("utf-8")
        refs.append((len(blob), len(data)))
        blob.extend(data)
    with path.open("wb") as handle:
        handle.write(STRINGS_MAGIC)
        handle.write(struct.pack("<IQ", SNAPSHOT_VERSION, len(blob)))
        handle.write(blob)
    return refs


def write_binary_snapshot(records: list[SnapshotRecord], snapshot_dir: Path) -> dict[str, Any]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    lookup_path = snapshot_dir / "lookup.bin"
    strings_path = snapshot_dir / "strings.seg"

    strings: list[str] = []
    for record in records:
        strings.extend(
            [
                record.query,
                record.context_key,
                record.answer_text,
                record.status,
                record.source_id,
                record.record_kind,
            ]
        )
    refs = _write_segment_strings(strings_path, strings)

    with lookup_path.open("wb") as handle:
        handle.write(LOOKUP_MAGIC)
        handle.write(struct.pack("<IQ", SNAPSHOT_VERSION, len(records)))
        ref_index = 0
        for record in records:
            handle.write(
                struct.pack(
                    "<QQQII",
                    composite_key_hash(record.query, record.context_key),
                    hash64(record.query),
                    hash64(record.context_key),
                    1 if record.status == "answered" else 2,
                    0,
                )
            )
            for _field in range(6):
                offset, length = refs[ref_index]
                handle.write(struct.pack("<QI", offset, length))
                ref_index += 1

    return {
        "snapshot_dir": str(snapshot_dir),
        "lookup_path": str(lookup_path),
        "strings_path": str(strings_path),
        "record_count": len(records),
        "snapshot_bytes": lookup_path.stat().st_size + strings_path.stat().st_size,
    }


def _read_strings(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 16 or data[:4] != STRINGS_MAGIC:
        raise ValueError("invalid strings segment")
    version, blob_len = struct.unpack_from("<IQ", data, 4)
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported strings version: {version}")
    blob = data[16:]
    if len(blob) != blob_len:
        raise ValueError("strings segment length mismatch")
    return blob


def _blob_string(blob: bytes, offset: int, length: int) -> str:
    end = offset + length
    if end > len(blob):
        raise ValueError("string reference outside segment")
    return blob[offset:end].decode("utf-8")


def read_binary_snapshot(snapshot_dir: Path) -> list[SnapshotRecord]:
    lookup_path = snapshot_dir / "lookup.bin"
    strings_path = snapshot_dir / "strings.seg"
    if not lookup_path.exists() or not strings_path.exists():
        return []
    blob = _read_strings(strings_path)
    data = lookup_path.read_bytes()
    if len(data) < 16 or data[:4] != LOOKUP_MAGIC:
        raise ValueError("invalid lookup segment")
    version, count = struct.unpack_from("<IQ", data, 4)
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported lookup version: {version}")
    offset = 16
    records: list[SnapshotRecord] = []
    for _index in range(count):
        # key_hash/query_hash/context_hash/status_code/reserved
        struct.unpack_from("<QQQII", data, offset)
        offset += 32
        refs = []
        for _field in range(6):
            ref_offset, ref_length = struct.unpack_from("<QI", data, offset)
            offset += 12
            refs.append((ref_offset, ref_length))
        values = [_blob_string(blob, ref_offset, ref_length) for ref_offset, ref_length in refs]
        records.append(
            SnapshotRecord(
                query=values[0],
                context_key=values[1],
                answer_text=values[2],
                status=values[3],
                source_id=values[4],
                record_kind=values[5],
            )
        )
    if offset != len(data):
        raise ValueError("lookup segment has trailing bytes")
    return records


def preflight_snapshot(records: list[SnapshotRecord]) -> dict[str, Any]:
    groups: dict[tuple[int, str, str], int] = {}
    for record in records:
        identity = (
            composite_key_hash(record.query, record.context_key),
            normalize_text(record.query),
            record.context_key,
        )
        groups[identity] = groups.get(identity, 0) + 1
    duplicate_count = sum(1 for count in groups.values() if count > 1)
    unsafe = []
    if duplicate_count:
        unsafe.append("DuplicateCompositeSnapshotKeys")
    return {
        "status": "Fail" if unsafe else "Pass",
        "record_count": len(records),
        "duplicate_key_count": duplicate_count,
        "unsafe_reasons": unsafe,
        "answer_permission_granted_by_preflight": False,
        "truth_asserted": False,
        "admits_evidence": False,
    }


def resolve_snapshot(
    records: list[SnapshotRecord],
    query: str,
    *,
    scope: str = DEFAULT_SCOPE,
    domain: str = DEFAULT_DOMAIN,
    source: str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    context_key = source_scope_context_key(scope=scope, domain=domain, source=source)
    key = composite_key_hash(query, context_key)
    matches = [
        record
        for record in records
        if composite_key_hash(record.query, record.context_key) == key
        and normalize_text(record.query) == normalize_text(query)
        and record.context_key == context_key
    ]
    if len(matches) > 1:
        return {
            "status": "blocked",
            "block_reason": "AmbiguousBinarySnapshotMatch",
            "candidate_found": True,
            "context_key": context_key,
            "answer_text": "",
            "answer_permission_granted": False,
            "product_answer_use_allowed": False,
        }
    if not matches:
        return {
            "status": "blocked",
            "block_reason": "NoBinarySnapshotMatch",
            "candidate_found": False,
            "context_key": context_key,
            "answer_text": "",
            "answer_permission_granted": False,
            "product_answer_use_allowed": False,
        }
    record = matches[0]
    allowed = record.status == "answered"
    return {
        "status": "answered" if allowed else "blocked",
        "block_reason": "" if allowed else "SnapshotRecordNotAnswerable",
        "candidate_found": True,
        "context_key": context_key,
        "source_id": record.source_id,
        "record_kind": record.record_kind,
        "answer_text": record.answer_text if allowed else "",
        "answer_permission_granted": allowed,
        "product_answer_use_allowed": allowed,
        "truth_asserted": False,
        "admits_evidence": False,
        "route_memory_write_allowed": False,
        "crystallization_memory_write_allowed": False,
        "changes_claim_state": False,
    }


def review_snapshot_answer_authority(packet: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed authority gate for binary snapshot answer use.

    This mirrors the Aura-clean rule: readout/candidate material is not a
    product answer unless a separate boundary grants answer authority.
    """
    block_reasons: list[str] = []
    if packet.get("status") != "answered":
        block_reasons.append(str(packet.get("block_reason") or "SnapshotPacketNotAnswered"))
    if not bool(packet.get("candidate_found")):
        block_reasons.append("SnapshotCandidateMissing")
    if not bool(packet.get("answer_permission_granted")):
        block_reasons.append("SnapshotAnswerPermissionMissing")
    if bool(packet.get("truth_asserted")):
        block_reasons.append("SnapshotAssertedTruth")
    if bool(packet.get("admits_evidence")):
        block_reasons.append("SnapshotAdmittedEvidence")
    if bool(packet.get("route_memory_write_allowed")) or bool(packet.get("changes_claim_state")):
        block_reasons.append("SnapshotTriedToMutateState")

    decision = "Stop" if block_reasons else "Go"
    return {
        "schema": "remy_binary_snapshot_answer_authority_v1",
        "decision": decision,
        "status": "PASS" if decision == "Go" else "FAIL",
        "block_reasons": list(dict.fromkeys(block_reasons)),
        "candidate_found": bool(packet.get("candidate_found")),
        "answer_permission_granted": bool(packet.get("answer_permission_granted")),
        "product_answer_use_allowed": decision == "Go",
        "truth_asserted": False,
        "admits_evidence": False,
        "llm_called_by_gate": False,
        "web_called_by_gate": False,
    }


async def run_binary_snapshot_eval(
    cases: list[Any],
    *,
    modes: list[str],
    model: str,
    dry_run: bool,
    delay_sec: float,
    snapshot_dir: Path,
    traffic_profile: str = "exact",
    snapshot_coverage_pct: float = 100.0,
) -> dict[str, Any]:
    build_cases = snapshot_build_cases(cases, coverage_pct=snapshot_coverage_pct)
    requests = runtime_requests_for_cases(cases, traffic_profile=traffic_profile)
    snapshot_meta = write_binary_snapshot(snapshot_records_from_cases(build_cases), snapshot_dir)
    records = read_binary_snapshot(snapshot_dir)
    preflight = preflight_snapshot(records)
    answer_func = None if dry_run else GeminiMeter(model, delay_sec=delay_sec)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for request in requests:
        case = request.case
        runtime_case = case_with_runtime_question(case, request.query)
        raw_prompt, _extra = await build_prompt_for_mode(runtime_case, "raw", answer_func)
        raw_prompt_tokens = estimate_tokens(raw_prompt)
        raw_answer = " ".join(case.expected_fragments + [group[0] for group in case.expected_any])
        if not dry_run and answer_func is not None and "raw" in modes:
            before_prompt = answer_func.prompt_tokens
            before_output = answer_func.output_tokens
            raw_answer, _meta = answer_func(raw_prompt)
            raw_provider_total = (
                answer_func.prompt_tokens - before_prompt + answer_func.output_tokens - before_output
            )
        else:
            raw_provider_total = raw_prompt_tokens

        for mode in modes:
            candidate_found = False
            authority_decision = ""
            if mode == "raw":
                answer = raw_answer
                provider_total = raw_provider_total
                prompt_tokens = raw_prompt_tokens
                status = "answered"
                block_reason = ""
                hit = False
            else:
                packet = resolve_snapshot(records, request.query)
                authority = review_snapshot_answer_authority(packet)
                hit = authority["decision"] == "Go"
                candidate_found = bool(authority["candidate_found"])
                authority_decision = str(authority["decision"])
                if hit:
                    answer = packet["answer_text"]
                    provider_total = 0
                    prompt_tokens = estimate_tokens(json.dumps(packet, ensure_ascii=False))
                    status = "answered"
                    block_reason = ""
                elif mode == "binary_snapshot_or_fallback":
                    answer = raw_answer
                    provider_total = raw_provider_total
                    prompt_tokens = raw_prompt_tokens
                    status = "fallback_raw"
                    block_reason = ";".join(authority["block_reasons"])
                else:
                    answer = ""
                    provider_total = 0
                    prompt_tokens = estimate_tokens(json.dumps(authority, ensure_ascii=False))
                    status = "blocked"
                    block_reason = ";".join(authority["block_reasons"])
            correctness = check_answer(
                answer,
                case.expected_fragments,
                case.must_not_contain,
                case.expected_any,
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "request_kind": request.request_kind,
                    "query": request.query,
                    "mode": mode,
                    "status": status,
                    "block_reason": block_reason,
                    "snapshot_hit": hit,
                    "candidate_found": candidate_found,
                    "authority_decision": authority_decision,
                    "correct": correctness["correct"],
                    "missing": correctness["missing"],
                    "missing_any": correctness["missing_any"],
                    "forbidden": correctness["forbidden"],
                    "prompt_tokens_estimate": prompt_tokens,
                    "raw_prompt_tokens_estimate": raw_prompt_tokens,
                    "provider_total_tokens": provider_total,
                    "provider_total_delta_vs_raw": raw_provider_total - provider_total,
                    "provider_total_saved_pct_vs_raw": round(
                        ((raw_provider_total - provider_total) / max(1, raw_provider_total)) * 100,
                        2,
                    ),
                    "answer_preview": answer[:500],
                }
            )

    by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        correct = sum(1 for row in mode_rows if row["correct"])
        hits = sum(1 for row in mode_rows if row["snapshot_hit"])
        fallbacks = sum(1 for row in mode_rows if row["status"] == "fallback_raw")
        blocked = sum(1 for row in mode_rows if row["status"] == "blocked")
        provider_total = sum(int(row["provider_total_tokens"]) for row in mode_rows)
        raw_provider_total = sum(
            int(row["provider_total_tokens"]) for row in rows if row["mode"] == "raw"
        )
        by_mode[mode] = {
            "cases": len(mode_rows),
            "accuracy": correct / len(mode_rows) if mode_rows else 0.0,
            "correct": correct,
            "snapshot_hits": hits,
            "snapshot_hit_rate": hits / len(mode_rows) if mode_rows else 0.0,
            "fallback_count": fallbacks,
            "fallback_rate": fallbacks / len(mode_rows) if mode_rows else 0.0,
            "blocked_count": blocked,
            "blocked_rate": blocked / len(mode_rows) if mode_rows else 0.0,
            "prompt_tokens_estimate": sum(int(row["prompt_tokens_estimate"]) for row in mode_rows),
            "provider_total_tokens": provider_total,
            "provider_total_saved_pct_vs_raw": round(
                ((raw_provider_total - provider_total) / max(1, raw_provider_total)) * 100,
                2,
            )
            if mode != "raw"
            else 0.0,
        }

    negative_probes = run_negative_probes(records)
    return {
        "schema": "remy_binary_snapshot_eval_v1",
        "summary": {
            "model": model,
            "dry_run": dry_run,
            "cases": len(cases),
            "traffic_profile": traffic_profile,
            "snapshot_coverage_pct": snapshot_coverage_pct,
            "snapshot_build_cases": len(build_cases),
            "modes": modes,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "snapshot": snapshot_meta,
            "preflight": preflight,
            "negative_probes": negative_probes,
            "by_mode": by_mode,
        },
        "rows": rows,
    }


def run_negative_probes(records: list[SnapshotRecord]) -> dict[str, Any]:
    if not records:
        return {"status": "blocked", "reason": "empty_snapshot"}
    first = records[0]
    wrong_scope = resolve_snapshot(records, first.query, scope="wrong-scope")
    wrong_domain = resolve_snapshot(records, first.query, domain="wrong-domain")
    wrong_source = resolve_snapshot(records, first.query, source="wrong-source")
    duplicate_preflight = preflight_snapshot(records + [first])
    passed = (
        wrong_scope["status"] == "blocked"
        and wrong_domain["status"] == "blocked"
        and wrong_source["status"] == "blocked"
        and duplicate_preflight["status"] == "Fail"
    )
    return {
        "status": "Pass" if passed else "Fail",
        "wrong_scope_blocked": wrong_scope["status"] == "blocked",
        "wrong_domain_blocked": wrong_domain["status"] == "blocked",
        "wrong_source_blocked": wrong_source["status"] == "blocked",
        "duplicate_preflight_failed": duplicate_preflight["status"] == "Fail",
    }


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in SNAPSHOT_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown snapshot mode(s): {', '.join(unknown)}")
    return modes


def _parse_traffic_profile(value: str) -> str:
    if value not in TRAFFIC_PROFILES:
        raise argparse.ArgumentTypeError(
            f"Unknown traffic profile {value!r}; expected one of {', '.join(TRAFFIC_PROFILES)}"
        )
    return value


def _write_report(report: dict[str, Any]) -> Path:
    from remy.config.settings import settings

    out_dir = Path(settings.DATA_DIR) / "llm_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"binary_snapshot_eval_{stamp}_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _main_async(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    if args.case_limit:
        cases = cases[: args.case_limit]
    snapshot_dir = args.snapshot_dir or (ROOT / "data" / "llm_optimization" / "binary_snapshot")
    report = await run_binary_snapshot_eval(
        cases,
        modes=args.mode,
        model=args.model,
        dry_run=args.dry_run,
        delay_sec=args.delay_sec,
        snapshot_dir=snapshot_dir,
        traffic_profile=args.traffic_profile,
        snapshot_coverage_pct=args.snapshot_coverage_pct,
    )
    path = _write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {path}")
    target = report["summary"]["by_mode"].get(args.target_mode, {})
    probes_ok = report["summary"]["negative_probes"]["status"] == "Pass"
    passed = probes_ok and float(target.get("accuracy") or 0.0) >= args.min_accuracy
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--mode", type=_parse_modes, default=["raw", "binary_snapshot_readout"])
    parser.add_argument("--target-mode", default="binary_snapshot_readout")
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--case-limit", type=int, default=30)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--traffic-profile", type=_parse_traffic_profile, default="exact")
    parser.add_argument("--snapshot-coverage-pct", type=float, default=100.0)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
