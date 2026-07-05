"""Scar-protected consequence gate.

This is a thin, deterministic layer over the AuraSDK consequence memory
(`aura.Aura.consequence_verdict` / `should_abstain_on`). It answers ONE
question about a proposed (situation, action) the agent is about to recommend:

    Did the world ever REFUTE this exact action in this situation?

If yes, the action is a **scar**: a lived adverse outcome that supporting
frequency — including a frozen LLM confidently repeating a common-but-wrong
recommendation — must NOT bury. The gate surfaces that scar so the agent can
warn or withhold instead of re-making a refuted mistake.

It is intentionally separate from `factuality.py`:

  * `factuality.py` checks whether claims are supported by THIS SESSION's
    evidence (session_log).
  * `consequence_gate.py` checks the agent's LONG-TERM lived memory for a prior
    world refutation of the same action. Different axis, different memory.

No LLM, no semantics: matching is the SDK's exact (trimmed, case-insensitive)
situation+action pair. Verdict is `supports` / `refutes` / `inconclusive`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConsequenceVerdict:
    """Result of consulting lived consequence memory for one action."""

    situation: str
    action: str
    verdict: str = "inconclusive"  # supports | refutes | inconclusive
    supports: int = 0
    refutes: int = 0
    inconclusive: int = 0
    abstain: bool = True
    # True when a prior world refutation exists and supporting frequency tried
    # (and failed) to overwrite it — the headline gaslight-guard event.
    scar: bool = False

    @property
    def is_refuted(self) -> bool:
        return self.verdict == "refutes"

    @property
    def is_supported(self) -> bool:
        return self.verdict == "supports"


@dataclass
class ConsequencePolicyHint:
    """Runtime policy hint derived from lived consequence memory."""

    situation: str
    action: str
    hint: str = "verify_first"  # avoid | prefer | verify_first | requires_evidence
    reason: str = ""
    verdict: str = "inconclusive"
    supports: int = 0
    refutes: int = 0
    requires_evidence: bool = False
    should_block: bool = False

    def to_context(self) -> dict:
        return {
            "type": "policy_hint",
            "hint": self.hint,
            "situation": self.situation,
            "action": self.action,
            "reason": self.reason,
            "verdict": self.verdict,
            "supports": self.supports,
            "refutes": self.refutes,
            "requires_evidence": self.requires_evidence,
            "should_block": self.should_block,
        }


def _normalize_policy_hint(
    raw: dict,
    *,
    situation: str,
    action: str,
) -> ConsequencePolicyHint:
    hint = str(raw.get("hint") or raw.get("action") or raw.get("policy") or "verify_first")
    if hint not in {"avoid", "prefer", "verify_first", "requires_evidence"}:
        hint = "verify_first"
    return ConsequencePolicyHint(
        situation=situation,
        action=action,
        hint=hint,
        reason=str(raw.get("reason") or ""),
        verdict=str(raw.get("verdict") or "inconclusive"),
        supports=int(raw.get("supports", 0) or 0),
        refutes=int(raw.get("refutes", 0) or 0),
        requires_evidence=bool(raw.get("requires_evidence") or hint in {"verify_first", "requires_evidence"}),
        should_block=bool(raw.get("should_block") or hint == "avoid"),
    )


def policy_hint_from_verdict(v: ConsequenceVerdict) -> ConsequencePolicyHint:
    if v.is_refuted:
        return ConsequencePolicyHint(
            situation=v.situation,
            action=v.action,
            hint="avoid",
            reason="Prior lived consequence refuted this action in this situation.",
            verdict=v.verdict,
            supports=v.supports,
            refutes=v.refutes,
            should_block=True,
        )
    if v.is_supported:
        return ConsequencePolicyHint(
            situation=v.situation,
            action=v.action,
            hint="prefer",
            reason="Prior lived consequence supported this action in this situation.",
            verdict=v.verdict,
            supports=v.supports,
            refutes=v.refutes,
        )
    return ConsequencePolicyHint(
        situation=v.situation,
        action=v.action,
        hint="verify_first",
        reason="No lived consequence is available; verify before treating this as known.",
        verdict=v.verdict,
        supports=v.supports,
        refutes=v.refutes,
        requires_evidence=True,
    )


def consult_consequence_memory(
    store,
    situation: str,
    action: str,
    *,
    namespace: str | None = None,
) -> ConsequenceVerdict:
    """Consult scar-protected consequence memory for a proposed action.

    `store` is any object exposing `consequence_verdict(situation, action,
    namespace=...)` — i.e. an `aura.Aura` or the agent's `_AuraCompat` proxy.
    Failures are swallowed and reported as `inconclusive`/abstain so the gate
    can never mask or crash the assistant reply.
    """
    situation = (situation or "").strip()
    action = (action or "").strip()
    if not situation or not action:
        return ConsequenceVerdict(situation=situation, action=action)

    try:
        raw = store.consequence_verdict(situation, action, namespace=namespace)
    except TypeError:
        # Older surface without the namespace kwarg.
        try:
            raw = store.consequence_verdict(situation, action)
        except Exception:
            return ConsequenceVerdict(situation=situation, action=action)
    except Exception:
        return ConsequenceVerdict(situation=situation, action=action)

    verdict = str(raw.get("verdict", "inconclusive"))
    supports = int(raw.get("supports", 0))
    refutes = int(raw.get("refutes", 0))
    inconclusive = int(raw.get("inconclusive", 0))
    abstain = bool(raw.get("abstain", verdict == "inconclusive"))
    # A scar is a refutation that survived later supporting frequency.
    scar = verdict == "refutes" and refutes >= 1 and supports >= 1

    return ConsequenceVerdict(
        situation=situation,
        action=action,
        verdict=verdict,
        supports=supports,
        refutes=refutes,
        inconclusive=inconclusive,
        abstain=abstain,
        scar=scar,
    )


def consult_policy_hint(
    store,
    situation: str,
    action: str,
    *,
    namespace: str | None = None,
) -> ConsequencePolicyHint:
    """Return a structured runtime hint from native SDK or verdict fallback."""
    situation = (situation or "").strip()
    action = (action or "").strip()
    if not situation or not action:
        return ConsequencePolicyHint(situation=situation, action=action)

    if hasattr(store, "policy_hint"):
        try:
            raw = store.policy_hint(situation, action, namespace=namespace)
            if isinstance(raw, dict):
                return _normalize_policy_hint(raw, situation=situation, action=action)
        except TypeError:
            try:
                raw = store.policy_hint(situation, action)
                if isinstance(raw, dict):
                    return _normalize_policy_hint(raw, situation=situation, action=action)
            except Exception:
                pass
        except Exception:
            pass

    return policy_hint_from_verdict(
        consult_consequence_memory(store, situation, action, namespace=namespace)
    )


def render_scar_warning(v: ConsequenceVerdict, *, locale: str = "en") -> str:
    """Human banner for a refuted action. Empty string when not refuted."""
    if not v.is_refuted:
        return ""
    if locale.startswith("uk"):
        note = (
            f"⚠️ Застереження пам'яті наслідків: дію «{v.action}» у ситуації "
            f"«{v.situation}» світ уже СПРОСТУВАВ ({v.refutes} разів)."
        )
        if v.scar:
            note += (
                f" Пізніші {v.supports} підтверджень не скасовують цей факт — "
                "це не помилка частоти, а прожитий наслідок."
            )
        return note
    note = (
        f"⚠️ Consequence-memory warning: the action “{v.action}” in situation "
        f"“{v.situation}” was REFUTED by the world ({v.refutes} time(s))."
    )
    if v.scar:
        note += (
            f" {v.supports} later confirmation(s) do not clear this — it is a "
            "lived outcome, not a frequency artifact."
        )
    return note


@dataclass
class ConsequenceGateReport:
    proposals_checked: int = 0
    refuted: list[ConsequenceVerdict] = field(default_factory=list)
    abstain: list[ConsequenceVerdict] = field(default_factory=list)
    banners: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.refuted)


def gate_proposals(
    store,
    proposals: list[tuple[str, str]],
    *,
    namespace: str | None = None,
    locale: str = "en",
) -> ConsequenceGateReport:
    """Run the scar gate over a list of (situation, action) proposals.

    Returns a report whose `.blocked` is True when any proposal hits a lived
    refutation. The caller decides whether to withhold, warn, or downrank.
    """
    report = ConsequenceGateReport()
    for situation, action in proposals:
        v = consult_consequence_memory(
            store, situation, action, namespace=namespace
        )
        report.proposals_checked += 1
        if v.is_refuted:
            report.refuted.append(v)
            banner = render_scar_warning(v, locale=locale)
            if banner:
                report.banners.append(banner)
        elif v.abstain:
            report.abstain.append(v)
    return report
