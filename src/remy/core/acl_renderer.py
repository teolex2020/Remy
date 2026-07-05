"""
ACL Renderer — deterministic text rendering for cognitive outputs.

Mirrors the Rust ACL renderer (src/acl_render.rs) in Python for use in
the Remi agent runtime. Produces operator-readable text from structured
cognitive data without any LLM involvement.

Design rule: ACL owns meaning; this module owns phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ── Locale ──────────────────────────────────────────────────────────────────


class Locale(Enum):
    EN = "en"
    UA = "ua"

    @classmethod
    def from_str(cls, s: str) -> "Locale":
        if s.lower() in ("ua", "uk", "ukr", "ukrainian"):
            return cls.UA
        return cls.EN


# ── Thermal Phase ───────────────────────────────────────────────────────────


class ThermalPhase(Enum):
    COLD = "cold"
    WARM = "warm"
    HOT = "hot"
    BOILING = "boiling"

    @classmethod
    def from_temperature(cls, t: float) -> "ThermalPhase":
        if t >= 0.80:
            return cls.BOILING
        if t >= 0.40:
            return cls.HOT
        if t >= 0.20:
            return cls.WARM
        return cls.COLD


# ── Locale Fragments ───────────────────────────────────────────────────────

_PHASE_LABELS = {
    (ThermalPhase.COLD, Locale.EN): "Cold",
    (ThermalPhase.COLD, Locale.UA): "Холодний",
    (ThermalPhase.WARM, Locale.EN): "Warm",
    (ThermalPhase.WARM, Locale.UA): "Теплий",
    (ThermalPhase.HOT, Locale.EN): "Hot",
    (ThermalPhase.HOT, Locale.UA): "Гарячий",
    (ThermalPhase.BOILING, Locale.EN): "Boiling",
    (ThermalPhase.BOILING, Locale.UA): "Кипить",
}

_GAP_TYPE_LABELS = {
    ("mechanism", Locale.EN): "mechanism",
    ("mechanism", Locale.UA): "механізм",
    ("procedure", Locale.EN): "procedure",
    ("procedure", Locale.UA): "процедура",
    ("constraint", Locale.EN): "constraint",
    ("constraint", Locale.UA): "обмеження",
    ("cause", Locale.EN): "cause",
    ("cause", Locale.UA): "причина",
    ("policy", Locale.EN): "policy",
    ("policy", Locale.UA): "політика",
}

_POLICY_ACTION_LABELS = {
    ("prefer", Locale.EN): "PREFER",
    ("prefer", Locale.UA): "ПЕРЕВАГА",
    ("recommend", Locale.EN): "RECOMMEND",
    ("recommend", Locale.UA): "РЕКОМЕНДАЦІЯ",
    ("verify_first", Locale.EN): "VERIFY FIRST",
    ("verify_first", Locale.UA): "ПЕРЕВІРИТИ",
    ("avoid", Locale.EN): "AVOID",
    ("avoid", Locale.UA): "УНИКАТИ",
    ("warn", Locale.EN): "WARNING",
    ("warn", Locale.UA): "ПОПЕРЕДЖЕННЯ",
}


def _phase_label(phase: ThermalPhase, locale: Locale) -> str:
    return _PHASE_LABELS.get((phase, locale), phase.value)


def _confidence_hedge(confidence: float, locale: Locale) -> str:
    if locale == Locale.UA:
        if confidence >= 0.85:
            return ""
        if confidence >= 0.65:
            return "ймовірно "
        if confidence >= 0.40:
            return "можливо "
        return "слабкий сигнал: "
    # EN
    if confidence >= 0.85:
        return ""
    if confidence >= 0.65:
        return "likely "
    if confidence >= 0.40:
        return "possibly "
    return "weak signal: "


def _urgency_suffix(
    temperature: Optional[float],
    phase: Optional[ThermalPhase],
    locale: Locale,
) -> str:
    if temperature is None or phase is None:
        return ""
    if phase == ThermalPhase.BOILING:
        tag = "URGENT" if locale == Locale.EN else "ТЕРМІНОВО"
        return f" [{tag}, {temperature:.2f}]"
    if phase == ThermalPhase.HOT:
        tag = _phase_label(phase, locale)
        return f" [{tag}, {temperature:.2f}]"
    if phase == ThermalPhase.WARM:
        tag = _phase_label(phase, locale)
        return f" [{tag}, {temperature:.2f}]"
    return ""


# ── Thermal Summary ────────────────────────────────────────────────────────


@dataclass
class HotZone:
    label: str
    temperature: float
    phase: ThermalPhase
    node_count: int = 0
    has_conflict: bool = False
    has_unresolved: bool = False


@dataclass
class ThermalSummaryExpr:
    total_beliefs: int = 0
    total_edges: int = 0
    total_energy: float = 0.0
    mean_temperature: float = 0.0
    hot_zones: list[HotZone] = field(default_factory=list)
    cold_count: int = 0
    isolated_hot_count: int = 0


def thermal_summary_from_report(report) -> ThermalSummaryExpr:
    """Project a ThermalReport (from thermal_advisor.py) into ACL expression."""
    hot_zones = []
    for c in report.clusters:
        tags_str = ", ".join(t for t, _ in c.dominant_tags[:3])
        hot_zones.append(HotZone(
            label=tags_str or f"cluster_{len(hot_zones)}",
            temperature=c.avg_temperature,
            phase=ThermalPhase.from_temperature(c.avg_temperature),
            node_count=len(c.nodes),
            has_conflict=c.has_conflict,
            has_unresolved=c.has_unresolved,
        ))
    hot_zones.sort(key=lambda z: z.temperature, reverse=True)

    # Count isolated hot nodes not in clusters
    clustered_ids = set()
    for c in report.clusters:
        clustered_ids.update(c.nodes)
    isolated = sum(1 for n in report.top_hot if n.belief_id not in clustered_ids and n.temperature > 0.40)

    return ThermalSummaryExpr(
        total_beliefs=report.node_count,
        total_edges=report.edge_count,
        total_energy=report.total_energy,
        mean_temperature=report.mean_temperature,
        hot_zones=hot_zones,
        cold_count=report.cold_mass_count,
        isolated_hot_count=isolated,
    )


def render_thermal_summary(expr: ThermalSummaryExpr, locale: Locale = Locale.EN) -> str:
    """Render thermal summary as operator-readable text."""
    lines = []

    if locale == Locale.EN:
        overall_phase = ThermalPhase.from_temperature(expr.mean_temperature)
        lines.append(
            f"[THERMAL] Cognitive state: {_phase_label(overall_phase, locale)} "
            f"(mean {expr.mean_temperature:.3f}). "
            f"{expr.total_beliefs} beliefs, {expr.total_edges} edges, "
            f"energy {expr.total_energy:.1f}."
        )

        if not expr.hot_zones:
            lines.append("No hot zones detected.")
        else:
            for z in expr.hot_zones:
                flags = []
                if z.has_conflict:
                    flags.append("CONFLICT")
                if z.has_unresolved:
                    flags.append("UNRESOLVED")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                lines.append(
                    f"  {_phase_label(z.phase, locale)} zone{flag_str}: "
                    f"{z.label} ({z.node_count} beliefs, {z.temperature:.3f})"
                )

        if expr.isolated_hot_count > 0:
            lines.append(f"  {expr.isolated_hot_count} isolated hot belief(s) — may be emerging concerns.")

        if expr.total_beliefs > 0:
            cold_ratio = expr.cold_count / expr.total_beliefs
            if cold_ratio > 0.7:
                lines.append(
                    f"  Cold mass: {expr.cold_count}/{expr.total_beliefs} "
                    f"({cold_ratio:.0%}) stable — safe to skip in maintenance."
                )
            elif cold_ratio > 0.4:
                lines.append(
                    f"  Warm graph: {expr.cold_count}/{expr.total_beliefs} "
                    f"({cold_ratio:.0%}) cold — moderate activity."
                )

    else:  # UA
        overall_phase = ThermalPhase.from_temperature(expr.mean_temperature)
        lines.append(
            f"[THERMAL] Когнітивний стан: {_phase_label(overall_phase, locale)} "
            f"(середня {expr.mean_temperature:.3f}). "
            f"{expr.total_beliefs} переконань, {expr.total_edges} зв'язків, "
            f"енергія {expr.total_energy:.1f}."
        )

        if not expr.hot_zones:
            lines.append("Гарячих зон не виявлено.")
        else:
            for z in expr.hot_zones:
                flags = []
                if z.has_conflict:
                    flags.append("КОНФЛIКТ")
                if z.has_unresolved:
                    flags.append("НЕВИРIШЕНО")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                lines.append(
                    f"  {_phase_label(z.phase, locale)} зона{flag_str}: "
                    f"{z.label} ({z.node_count} переконань, {z.temperature:.3f})"
                )

        if expr.isolated_hot_count > 0:
            lines.append(f"  {expr.isolated_hot_count} ізольованих гарячих переконань — можуть бути новими проблемами.")

        if expr.total_beliefs > 0:
            cold_ratio = expr.cold_count / expr.total_beliefs
            if cold_ratio > 0.7:
                lines.append(
                    f"  Холодна маса: {expr.cold_count}/{expr.total_beliefs} "
                    f"({cold_ratio:.0%}) стабільні — можна пропустити в maintenance."
                )
            elif cold_ratio > 0.4:
                lines.append(
                    f"  Теплий граф: {expr.cold_count}/{expr.total_beliefs} "
                    f"({cold_ratio:.0%}) холодних — помірна активність."
                )

    return "\n".join(lines)


# ── Conflict Report ─────────────────────────────────────────────────────────


@dataclass
class ConflictReportExpr:
    left_claim: str
    left_confidence: float
    right_claim: str
    right_confidence: float
    evidence_strength: float
    severity: str  # "low" | "medium" | "high"
    recommended_action: Optional[str] = None
    temperature: Optional[float] = None
    phase: Optional[ThermalPhase] = None


def render_conflict_report(expr: ConflictReportExpr, locale: Locale = Locale.EN) -> str:
    urgency = _urgency_suffix(expr.temperature, expr.phase, locale)
    hedge_l = _confidence_hedge(expr.left_confidence, locale)
    hedge_r = _confidence_hedge(expr.right_confidence, locale)

    if locale == Locale.EN:
        text = (
            f"[CONFLICT] CONFLICT{urgency}: "
            f"{hedge_l}\"{expr.left_claim}\" ({expr.left_confidence:.0%}) vs "
            f"{hedge_r}\"{expr.right_claim}\" ({expr.right_confidence:.0%}). "
            f"Evidence strength: {expr.evidence_strength:.2f}. Severity: {expr.severity}."
        )
        if expr.recommended_action:
            text += f" Recommended: {expr.recommended_action}."
    else:
        text = (
            f"[CONFLICT] КОНФЛIКТ{urgency}: "
            f"{hedge_l}\"{expr.left_claim}\" ({expr.left_confidence:.0%}) проти "
            f"{hedge_r}\"{expr.right_claim}\" ({expr.right_confidence:.0%}). "
            f"Сила доказів: {expr.evidence_strength:.2f}. Серйозність: {expr.severity}."
        )
        if expr.recommended_action:
            ua_action = {
                "escalate": "ескалювати",
                "review": "переглянути",
                "monitor": "моніторити",
            }.get(expr.recommended_action, expr.recommended_action)
            text += f" Рекомендація: {ua_action}."

    return text


# ── Gap Report ──────────────────────────────────────────────────────────────


@dataclass
class GapReportExpr:
    anchor_label: str
    gap_type: str  # "mechanism" | "procedure" | "constraint" | "cause" | "policy"
    severity: float
    uncertainty: float
    evidence_count: int = 0
    description: Optional[str] = None
    temperature: Optional[float] = None
    phase: Optional[ThermalPhase] = None


def render_gap_report(expr: GapReportExpr, locale: Locale = Locale.EN) -> str:
    urgency = _urgency_suffix(expr.temperature, expr.phase, locale)
    gap_label = _GAP_TYPE_LABELS.get((expr.gap_type, locale), expr.gap_type)

    if locale == Locale.EN:
        text = (
            f"[GAP] GAP{urgency}: Missing {gap_label} for \"{expr.anchor_label}\". "
            f"Severity: {expr.severity:.2f}, uncertainty: {expr.uncertainty:.2f}."
        )
        if expr.evidence_count > 0:
            text += f" {expr.evidence_count} evidence record(s)."
        if expr.description:
            text += f" Detail: {expr.description}."
    else:
        text = (
            f"[GAP] ПРОГАЛИНА{urgency}: Відсутній {gap_label} для \"{expr.anchor_label}\". "
            f"Серйозність: {expr.severity:.2f}, невизначеність: {expr.uncertainty:.2f}."
        )
        if expr.evidence_count > 0:
            text += f" {expr.evidence_count} запис(ів) доказів."
        if expr.description:
            text += f" Деталі: {expr.description}."

    return text


# ── Policy Report ───────────────────────────────────────────────────────────


@dataclass
class PolicyReportExpr:
    domain: str
    action: str  # "prefer" | "recommend" | "verify_first" | "avoid" | "warn"
    recommendation: str
    confidence: float
    policy_strength: float
    risk_score: float
    state: str  # "stable" | "candidate" | "suppressed" | "rejected"
    temperature: Optional[float] = None
    phase: Optional[ThermalPhase] = None


def render_policy_report(expr: PolicyReportExpr, locale: Locale = Locale.EN) -> str:
    urgency = _urgency_suffix(expr.temperature, expr.phase, locale)
    action_label = _POLICY_ACTION_LABELS.get((expr.action, locale), expr.action.upper())
    hedge = _confidence_hedge(expr.confidence, locale)

    if locale == Locale.EN:
        text = (
            f"[POLICY] POLICY{urgency}: [{action_label}] {hedge}{expr.recommendation} "
            f"(domain: {expr.domain}). Strength: {expr.policy_strength:.2f}, "
            f"risk: {expr.risk_score:.2f}. State: {expr.state}."
        )
        if expr.action in ("avoid", "warn") and expr.risk_score >= 0.60:
            text += " High risk — review recommended."
    else:
        state_ua = {
            "stable": "стабільний",
            "candidate": "кандидат",
            "suppressed": "пригнічений",
            "rejected": "відхилений",
        }.get(expr.state, expr.state)
        text = (
            f"[POLICY] ПОЛIТИКА{urgency}: [{action_label}] {hedge}{expr.recommendation} "
            f"(домен: {expr.domain}). Сила: {expr.policy_strength:.2f}, "
            f"ризик: {expr.risk_score:.2f}. Стан: {state_ua}."
        )
        if expr.action in ("avoid", "warn") and expr.risk_score >= 0.60:
            text += " Високий ризик — рекомендується перегляд."

    return text


# ── Pruning Report ──────────────────────────────────────────────────────────


@dataclass
class PruningReportExpr:
    examined: int
    removed: int
    namespace_filter: Optional[str] = None
    cognitive_load_freed_pct: float = 0.0


def render_pruning_report(expr: PruningReportExpr, locale: Locale = Locale.EN) -> str:
    if locale == Locale.EN:
        text = (
            f"[BRAIN] PRUNED: {expr.removed} record(s) removed out of {expr.examined} examined. "
            f"Cognitive load freed: {expr.cognitive_load_freed_pct:.1f}%."
        )
        if expr.namespace_filter:
            text += f" Namespace: {expr.namespace_filter}."
    else:
        text = (
            f"[BRAIN] ВИДАЛЕНО: {expr.removed} запис(ів) видалено з {expr.examined} перевірених. "
            f"Звільнено когнітивного навантаження: {expr.cognitive_load_freed_pct:.1f}%."
        )
        if expr.namespace_filter:
            text += f" Простір імен: {expr.namespace_filter}."

    return text


# ── Insight Rendering ───────────────────────────────────────────────────────


def render_insight(insight_type: str, payload: dict, locale: Locale = Locale.EN) -> Optional[str]:
    """Render a background brain insight as ACL-style text.

    Replaces the ad-hoc f-string formatting in background_brain._format_insight().
    """
    if locale == Locale.EN:
        if insight_type == "decay":
            names = payload.get("names", [])
            if names:
                return f"[BRAIN INSIGHT] Memories fading: {'; '.join(names[:5])}. Consider revisiting."
        elif insight_type == "conflict":
            a = payload.get("content_a", "?")[:50]
            b = payload.get("content_b", "?")[:50]
            return f"[BRAIN INSIGHT] CONTRADICTION: '{a}' vs '{b}'. May need clarification."
        elif insight_type == "cluster":
            tags = payload.get("tags", [])[:5]
            size = payload.get("size", 0)
            return f"[BRAIN INSIGHT] Knowledge cluster: {', '.join(tags)} ({size} connected records)."
        elif insight_type == "promotion":
            names = payload.get("names", [])
            if names:
                return f"[BRAIN INSIGHT] Memories promoted: {'; '.join(names[:5])}."
    else:  # UA
        if insight_type == "decay":
            names = payload.get("names", [])
            if names:
                return f"[BRAIN INSIGHT] Спогади згасають: {'; '.join(names[:5])}. Варто повернутись."
        elif insight_type == "conflict":
            a = payload.get("content_a", "?")[:50]
            b = payload.get("content_b", "?")[:50]
            return f"[BRAIN INSIGHT] КОНФЛIКТ: '{a}' проти '{b}'. Потребує уточнення."
        elif insight_type == "cluster":
            tags = payload.get("tags", [])[:5]
            size = payload.get("size", 0)
            return f"[BRAIN INSIGHT] Кластер знань: {', '.join(tags)} ({size} пов'язаних записів)."
        elif insight_type == "promotion":
            names = payload.get("names", [])
            if names:
                return f"[BRAIN INSIGHT] Спогади дозріли: {'; '.join(names[:5])}."

    return None


# ── Brain Voice (proactive messages) ───────────────────────────────────────


def render_brain_voice(event: dict, locale: Locale = Locale.EN) -> Optional[str]:
    """Render a BrainVoiceEvent dict as a short user-facing message.

    Event schema:
      {
        "kind": "hot_zone_spike" | "pruning_burst" | "routing_shift",
        "severity": "info" | "notable" | "urgent",
        "payload": {...},
      }

    Returns None if the event kind is unknown. Adding a new locale means
    adding a branch here — the brain side stays locale-agnostic.
    """
    kind = event.get("kind")
    payload = event.get("payload", {}) or {}
    severity = event.get("severity", "notable")

    if kind == "hot_zone_spike":
        delta = int(payload.get("delta", 0))
        now = int(payload.get("hot_zones_now", 0))
        conflicts = int(payload.get("conflict_clusters", 0))
        temp = float(payload.get("mean_temperature", 0.0))
        if locale == Locale.UA:
            prefix = "🔥 Увага:" if severity == "urgent" else "🔥"
            return (
                f"{prefix} помічаю напругу — з'явилось +{delta} гарячих зон "
                f"(усього {now}, у {conflicts} з них конфлікт). "
                f"Середня температура: {temp:.1%}."
            )
        prefix = "🔥 Heads up:" if severity == "urgent" else "🔥"
        return (
            f"{prefix} I'm noticing tension — {delta} new hot zone(s) "
            f"(total {now}, {conflicts} with conflict). "
            f"Mean temperature: {temp:.1%}."
        )

    if kind == "epistemic_claim":
        # Deterministic phrasing for a single epistemic claim, keyed by
        # EpistemicStatus. The brain emits only (status, origin, content);
        # natural language is chosen here.
        status = str(payload.get("status", "Unknown"))
        content = str(payload.get("content", "") or "").strip()
        origin = str(payload.get("origin", "") or "")
        if not content:
            return None
        _EN = {
            "Observed":     ("I see: ", ""),
            "Supported":    ("There is evidence that ", ""),
            "Believed":     ("The system holds that ", ""),
            "Hypothesis":   ("I'm guessing — ", " — this isn't verified."),
            "Contradicted": ("Signals conflict: ", " — I won't commit either way yet."),
            "Unknown":      ("I don't have grounded data on this: ", ""),
        }
        _UA = {
            "Observed":     ("Я бачу: ", ""),
            "Supported":    ("Є підтвердження, що ", ""),
            "Believed":     ("Система вважає, що ", ""),
            "Hypothesis":   ("Я припускаю — ", " — це не підтверджено."),
            "Contradicted": ("Є суперечливі сигнали: ", " — я поки не займаю бік."),
            "Unknown":      ("Я не маю підтверджених даних про це: ", ""),
        }
        mapping = _UA if locale == Locale.UA else _EN
        prefix, suffix = mapping.get(status, mapping["Unknown"])
        tail = f" [origin: {origin}]" if origin else ""
        return f"{prefix}{content}{suffix}{tail}".strip()

    if kind == "epistemic_downgrade":
        # Pre-mouth honest-phrasing banner emitted when a claim is rewritten
        # or blocked by epistemic_governance. Payload carries structured
        # counts, NEVER natural-language text from the brain.
        status = payload.get("status", "Hypothesis")
        mode = payload.get("mode", "soft")  # soft | aggressive | block
        phantom = int(payload.get("phantom_count", 0))
        total = int(payload.get("external_total", 0))
        if locale == Locale.UA:
            if mode == "block":
                return (
                    "🛑 Я не можу це підтвердити. "
                    f"{phantom} з {total} зовнішніх посилань не пройшли перевірку "
                    "в інструментах цього ходу, тож відповідь перероблено у чесну невизначеність."
                )
            if mode == "aggressive":
                return (
                    "⚠ Формулювання пом'якшено: частина згаданих посилань не підтверджена "
                    f"({phantom}/{total}). Розглядай як гіпотезу, не як факт."
                )
            return (
                "ℹ Невелика частина посилань не підтверджена "
                f"({phantom}/{total}). Трактуй відповідні твердження обережно."
            )
        if mode == "block":
            return (
                "🛑 I can't honestly confirm that. "
                f"{phantom} of {total} external references failed structural "
                "verification in this turn's tools, so the answer was rewritten into honest uncertainty."
            )
        if mode == "aggressive":
            return (
                "⚠ Phrasing softened: some of the references I mentioned are not verified "
                f"({phantom}/{total}). Treat this as a hypothesis, not a fact."
            )
        return (
            "ℹ A small number of references are unverified "
            f"({phantom}/{total}). Treat the affected statements cautiously."
        )

    if kind == "pruning_burst":
        delta = int(payload.get("delta", 0))
        pruned_now = int(payload.get("pruned_now", 0))
        healthy = int(payload.get("healthy", 0))
        if locale == Locale.UA:
            prefix = "🧹 Великий прибирання:" if severity == "urgent" else "🧹"
            return (
                f"{prefix} відсіяв +{delta} слабких зв'язків "
                f"(усього видалено: {pruned_now}, здорових: {healthy}). "
                f"Голова стала чистішою."
            )
        prefix = "🧹 Major cleanup:" if severity == "urgent" else "🧹"
        return (
            f"{prefix} pruned {delta} weak link(s) this cycle "
            f"(total pruned: {pruned_now}, healthy: {healthy}). "
            f"My head is clearer now."
        )

    if kind == "routing_shift":
        from_mode = str(payload.get("from_mode", "?"))
        to_mode = str(payload.get("to_mode", "?"))
        cycle = int(payload.get("cycle_number", 0))
        if locale == Locale.UA:
            return (
                f"🔄 Режим циклу змінився: {from_mode} → {to_mode} "
                f"(цикл #{cycle})."
            )
        return (
            f"🔄 Cycle mode shifted: {from_mode} → {to_mode} "
            f"(cycle #{cycle})."
        )

    return None
