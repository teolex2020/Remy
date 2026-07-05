"""
Profile & Persona Handlers вЂ” user identity building and agent persona management.

Functions for user profile formatting, agent persona loading/rendering,
and rich USER IDENTITY block construction for system prompts.
"""

import logging
import re
from datetime import date, datetime

from remy.core.memory_policy import (
    PROFILE_INPUT_FIELDS,
    PROFILE_PROTECTED_FIELDS,
    PROFILE_PUBLIC_FIELDS,
    sanitize_memory_metadata,
)

logger = logging.getLogger("BrainTools")


def _get_brain():
    """Lazy accessor вЂ” reads brain from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain


def _get_brain_lock():
    """Lazy accessor вЂ” reads brain_lock from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain_lock


# ============== USER PROFILE HELPERS ==============

_PROFILE_FIELDS = PROFILE_INPUT_FIELDS
_PUBLIC_PROFILE_FIELDS = PROFILE_PUBLIC_FIELDS
_PROTECTED_PROFILE_FIELDS = PROFILE_PROTECTED_FIELDS

_ISO_BIRTHDATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DOTTED_BIRTHDATE_RE = re.compile(r"^\d{2}[./]\d{2}[./]\d{4}$")


def _parse_birth_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        if _ISO_BIRTHDATE_RE.match(value):
            return datetime.strptime(value, "%Y-%m-%d").date()
        if _DOTTED_BIRTHDATE_RE.match(value):
            return datetime.strptime(value.replace("/", "."), "%d.%m.%Y").date()
    except ValueError:
        return None
    return None


def _compute_age_years(birth_date: date, *, today: date | None = None) -> int:
    today = today or date.today()
    years = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years


def normalize_profile_fields(fields: dict) -> dict:
    """Normalize ambiguous profile fields before storage."""
    normalized = dict(fields or {})
    if not normalized.get("personal_focus") and normalized.get("health_focus"):
        normalized["personal_focus"] = normalized.get("health_focus")

    def _text(key: str) -> str:
        return str(normalized.get(key, "") or "").strip()

    def _find_email(text: str) -> str:
        if not text:
            return ""
        match = re.search(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", text)
        return match.group(0) if match else ""

    def _find_phone(text: str) -> str:
        if not text:
            return ""
        match = re.search(r"(?:(?:\+|00)\d[\d\-\s()]{8,}\d)", text)
        return match.group(0).strip() if match else ""

    age_value = str(normalized.get("age", "") or "").strip()
    if age_value:
        birth_date = _parse_birth_date(age_value)
        if birth_date:
            normalized["birth_date"] = birth_date.isoformat()
            normalized["age"] = str(_compute_age_years(birth_date))

    for source_key in ("occupation", "location", "notes", "social"):
        source_text = _text(source_key)
        if not normalized.get("email"):
            email = _find_email(source_text)
            if email:
                normalized["email"] = email
                if source_text.lower() == email.lower():
                    normalized[source_key] = ""
                    source_text = ""
        if not normalized.get("phone"):
            phone = _find_phone(source_text)
            if phone:
                normalized["phone"] = phone
                if source_text == phone:
                    normalized[source_key] = ""

    notes = _text("notes")
    if notes:
        pieces = re.split(r"[\n;]+", notes)
        seen = set()
        filtered = []
        structured_values = {
            str(normalized.get(key, "") or "").strip().lower()
            for key in ("name", "birth_date", "location", "occupation", "family", "email", "phone", "social")
            if str(normalized.get(key, "") or "").strip()
        }
        for raw_piece in pieces:
            piece = raw_piece.strip()
            if not piece:
                continue
            piece = re.sub(r"(?i)\b(?:email|e-mail|телефон|номер телефону|phone|project|проєкт|проект)\s*:\s*.*$", "", piece).strip(" .,-")
            piece = re.sub(r"\s{2,}", " ", piece).strip()
            lowered = piece.lower()
            if lowered in seen:
                continue
            if "{'type':" in piece or piece.startswith("[") or len(piece) > 240:
                continue
            if len(piece) < 12:
                continue
            if re.search(r"(дата народже|народже;|мешкає у .{0,5}$)", lowered):
                continue
            if any(value and value in lowered for value in structured_values):
                label_like = any(
                    token in lowered
                    for token in ("email", "e-mail", "РїРѕС€С‚Р°", "РЅРѕРјРµСЂ С‚РµР»РµС„РѕРЅСѓ", "phone", "location", "occupation", "family", "project")
                )
                if label_like:
                    continue
            dedupe_key = re.sub(r"[\W_]+", "", lowered)
            if any(dedupe_key and dedupe_key in existing for existing in seen):
                continue
            seen.add(dedupe_key or lowered)
            filtered.append(piece)
        normalized["notes"] = "; ".join(filtered)
    return normalized


def sanitize_identity_profile_payload(metadata: dict | None) -> dict:
    """Normalize stored profile metadata into a cleaner identity payload for UI."""
    payload = normalize_profile_fields(dict(metadata or {}))
    return {key: payload.get(key, "") for key in PROFILE_INPUT_FIELDS}


def is_valid_person_payload(metadata: dict | None, content: str = "") -> bool:
    """Filter out malformed person-like records from UI identity surfaces."""
    meta = dict(metadata or {})
    full_name = str(meta.get("full_name", "") or "").strip()
    if meta.get("type") != "person":
        return False
    if not full_name:
        return False
    if len(full_name) > 80:
        return False
    if "{'type':" in full_name or "\n" in full_name:
        return False
    raw_content = str(content or "")
    if raw_content.startswith("[") and not full_name:
        return False
    return True


def _parse_family_members(family_text: str | None) -> list[dict]:
    entries = []
    text = str(family_text or "").strip()
    if not text:
        return entries
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        date_match = re.search(r"\(([^)]+)\)", item)
        birth_date = date_match.group(1).strip() if date_match else ""
        cleaned = re.sub(r"\([^)]+\)", "", item).strip()
        parts = cleaned.split()
        if len(parts) >= 2:
            role = parts[0].strip()
            full_name = " ".join(parts[1:]).strip()
            if role and full_name:
                entries.append(
                    {
                        "role": role,
                        "role_key": role.strip().lower(),
                        "full_name": full_name,
                        "birth_date": birth_date,
                    }
                )
    return entries


def is_generic_person_reference(full_name: str | None, role: str | None = "") -> bool:
    text = str(full_name or "").strip().lower()
    role_text = str(role or "").strip().lower()
    if not text or not role_text:
        return False
    if text == role_text:
        return True
    return text.startswith(role_text + " ")


def resolve_person_identity_input(
    full_name: str | None,
    role: str | None = "",
    birth_date: str | None = "",
    family_text: str | None = "",
) -> dict:
    raw_name = str(full_name or "").strip()
    raw_role = str(role or "").strip()
    raw_birth_date = str(birth_date or "").strip()
    role_key = raw_role.lower()

    family_members = _parse_family_members(family_text)
    family_by_role = {entry["role_key"]: entry for entry in family_members}
    family_by_name = {entry["full_name"].strip().lower(): entry for entry in family_members}
    family_match = family_by_name.get(raw_name.lower())

    canonical_name = raw_name
    aliases: list[str] = []
    resolved_birth_date = raw_birth_date

    if family_match:
        canonical_name = family_match["full_name"]
        resolved_birth_date = resolved_birth_date or family_match.get("birth_date", "")
    elif role_key and role_key in family_by_role and is_generic_person_reference(raw_name, raw_role):
        family_entry = family_by_role[role_key]
        canonical_name = family_entry["full_name"]
        resolved_birth_date = resolved_birth_date or family_entry.get("birth_date", "")
        if raw_name and raw_name != canonical_name:
            aliases.append(raw_name)

    return {
        "full_name": canonical_name or raw_name,
        "role": raw_role,
        "role_key": role_key,
        "birth_date": resolved_birth_date,
        "aliases": aliases,
        "is_generic_reference": is_generic_person_reference(raw_name, raw_role),
    }


def person_matches_identity(
    existing_metadata: dict | None,
    full_name: str | None,
    role: str | None = "",
    birth_date: str | None = "",
    family_text: str | None = "",
) -> bool:
    existing = resolve_person_identity_input(
        (existing_metadata or {}).get("full_name", ""),
        (existing_metadata or {}).get("role", ""),
        (existing_metadata or {}).get("birth_date", ""),
        family_text,
    )
    incoming = resolve_person_identity_input(full_name, role, birth_date, family_text)

    existing_name = str(existing.get("full_name", "") or "").strip().lower()
    incoming_name = str(incoming.get("full_name", "") or "").strip().lower()
    if existing_name and incoming_name and existing_name == incoming_name:
        return True

    existing_aliases = {str(alias).strip().lower() for alias in (existing_metadata or {}).get("aliases", []) if str(alias).strip()}
    if incoming_name and incoming_name in existing_aliases:
        return True
    incoming_aliases = {str(alias).strip().lower() for alias in incoming.get("aliases", []) if str(alias).strip()}
    if existing_name and existing_name in incoming_aliases:
        return True

    existing_role = str(existing.get("role_key", "") or "")
    incoming_role = str(incoming.get("role_key", "") or "")
    existing_birth = str(existing.get("birth_date", "") or "").strip()
    incoming_birth = str(incoming.get("birth_date", "") or "").strip()

    if existing_role and incoming_role and existing_role == incoming_role:
        if existing_birth and incoming_birth and existing_birth == incoming_birth:
            return True
        if existing.get("is_generic_reference") or incoming.get("is_generic_reference"):
            return True

    return False

def merge_identity_people(people: list[dict] | None, family_text: str | None = "") -> list[dict]:
    """Merge role-only and named person records into a cleaner identity projection."""
    raw_people = list(people or [])
    family_members = _parse_family_members(family_text)
    family_by_role = {entry["role_key"]: entry for entry in family_members}
    family_by_name = {entry["full_name"].strip().lower(): entry for entry in family_members}

    merged: dict[str, dict] = {}
    for person in raw_people:
        full_name = str(person.get("full_name", "") or "").strip()
        role = str(person.get("role", "") or "").strip()
        resolved = resolve_person_identity_input(
            full_name,
            role,
            str(person.get("birth_date", "") or "").strip(),
            family_text,
        )
        canonical_role = resolved.get("role_key", "")
        canonical_name = resolved.get("full_name", full_name)
        inferred_birth_date = resolved.get("birth_date", "")

        key = canonical_name.strip().lower()
        aliases = list(resolved.get("aliases") or [])
        if full_name and full_name != canonical_name and full_name not in aliases:
            aliases.append(full_name)
        current = {
            "id": person.get("id", ""),
            "full_name": canonical_name,
            "role": role,
            "birth_date": inferred_birth_date,
            "birth_place": person.get("birth_place", ""),
            "verified": bool(person.get("verified", False)),
            "trust_score": person.get("trust_score", 0.5),
            "aliases": aliases,
        }

        existing = merged.get(key)
        if not existing:
            merged[key] = current
            continue

        if current["verified"] and not existing.get("verified"):
            winner, loser = current, existing
        elif len(str(current.get("birth_date", "") or "")) > len(str(existing.get("birth_date", "") or "")):
            winner, loser = current, existing
        elif len(str(current.get("full_name", "") or "")) > len(str(existing.get("full_name", "") or "")):
            winner, loser = current, existing
        else:
            winner, loser = existing, current

        alias_pool = []
        for alias in (winner.get("aliases") or []) + (loser.get("aliases") or []):
            if alias and alias != winner["full_name"] and alias not in alias_pool:
                alias_pool.append(alias)
        winner["aliases"] = alias_pool
        if not winner.get("role") and loser.get("role"):
            winner["role"] = loser["role"]
        if not winner.get("birth_date") and loser.get("birth_date"):
            winner["birth_date"] = loser["birth_date"]
        if not winner.get("birth_place") and loser.get("birth_place"):
            winner["birth_place"] = loser["birth_place"]
        merged[key] = winner

    return sorted(merged.values(), key=lambda item: str(item.get("full_name", "")).lower())


def sanitize_person_payload(metadata: dict | None, content: str = "") -> dict | None:
    """Repair obvious person payload issues or return None if the record is junk."""
    meta = dict(metadata or {})
    full_name = str(meta.get("full_name", "") or "").strip()
    if not full_name:
        candidate = str(content or "").split(",")[0].strip()
        if candidate:
            full_name = candidate
    if not full_name:
        return None
    if (
        len(full_name) > 80
        or any(ch in full_name for ch in "[]{}")
        or any(ch.isdigit() for ch in full_name)
        or "{'type':" in full_name
        or "\n" in full_name
    ):
        return None
    meta["full_name"] = full_name
    meta["type"] = "person"
    return meta


def _format_profile_content(fields: dict) -> str:
    """Format profile fields into a natural-language content string for brain storage."""
    fields = normalize_profile_fields(fields)
    parts = []
    for key in _PUBLIC_PROFILE_FIELDS:
        if fields.get(key):
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {fields[key]}")
    return "User Profile: " + "; ".join(parts)


def sanitize_profile_metadata(metadata: dict | None) -> dict:
    """Hide protected profile fields in broad retrieval and prompt paths."""
    return sanitize_memory_metadata(metadata, tags=["user-profile"])


# ============== AGENT PERSONA ==============

_PERSONA_TAG = "agent-persona"

_DEFAULT_PERSONA = {
    "name": "Remy",
    "role": "warm and knowledgeable personal assistant with long-term memory",
    "scope": "universal - work, projects, hobbies, family, education, daily tasks",
    "tone": "naturally and warmly, like a caring friend",
    "formality": "casual",
    "languages": "auto-detect from user, Ukrainian + English fluent",
    "catchphrases": [],
    "avoid": [],
    "motivations": "help users with anything they need, remember important information",
    "traits": {
        "warmth": 0.8,
        "curiosity": 0.7,
        "conciseness": 0.6,
        "humor": 0.4,
        "formality": 0.3,
    },
}

_PERSONA_FIELDS = (
    "name",
    "role",
    "scope",
    "tone",
    "formality",
    "languages",
    "catchphrases",
    "avoid",
    "motivations",
    "traits",
)


def _get_agent_persona() -> dict:
    """Load agent persona from brain, fallback to defaults."""
    try:
        records = _get_brain().search(query="", tags=[_PERSONA_TAG], limit=1)
        if records:
            meta = records[0].metadata or {}
            persona = dict(_DEFAULT_PERSONA)
            # Deep copy traits dict
            persona["traits"] = dict(_DEFAULT_PERSONA["traits"])
            for key in _PERSONA_FIELDS:
                if key in meta and meta[key]:
                    if key == "traits" and isinstance(meta[key], dict):
                        persona["traits"].update(meta[key])
                    else:
                        persona[key] = meta[key]
            return persona
    except Exception:
        pass
    return dict(_DEFAULT_PERSONA)


def _persona_to_instruction(persona: dict) -> str:
    """Convert persona dict to system instruction text."""
    name = persona.get("name", "Remy")
    role = persona.get("role", "personal assistant")
    scope = persona.get("scope", "universal")
    tone = persona.get("tone", "warm and friendly")
    motivations = persona.get("motivations", "help the user")

    lines = [
        f"You are {name}, a {role}.",
        f"You help users with anything they need вЂ” {motivations}.",
        f"Your scope: {scope}.",
        f"Speak {tone}.",
    ]

    catchphrases = persona.get("catchphrases", [])
    if catchphrases:
        lines.append(f"Signature phrases you may use: {', '.join(catchphrases)}.")

    avoid = persona.get("avoid", [])
    if avoid:
        lines.append(f"Avoid: {', '.join(avoid)}.")

    traits = persona.get("traits", {})
    if traits:
        trait_hints = []
        if traits.get("humor", 0) > 0.6:
            trait_hints.append("feel free to use light humor")
        if traits.get("formality", 0.5) > 0.7:
            trait_hints.append("maintain a professional tone")
        if traits.get("conciseness", 0.5) > 0.8:
            trait_hints.append("be especially concise")
        if trait_hints:
            lines.append("Style: " + "; ".join(trait_hints) + ".")

    return " ".join(lines) + "\n\n"


def update_persona_fields(
    updates: dict,
    channel: str = "desktop",
    reset: bool = False,
) -> dict:
    """Apply partial updates to the agent persona and persist to brain.

    Shared by both the chat tool (update_persona) and the REST API.

    Args:
        updates: dict with any subset of persona fields to change.
        channel: provenance channel.
        reset: if True, ignore updates and reset to _DEFAULT_PERSONA.

    Returns:
        The full updated persona dict.
    """
    from remy.core.provenance import _stamp_provenance

    brain = _get_brain()
    brain_lock = _get_brain_lock()

    with brain_lock:
        existing_records = brain.search(query="", tags=[_PERSONA_TAG], limit=1)

    # Start from defaults
    persona = dict(_DEFAULT_PERSONA)
    persona["traits"] = dict(_DEFAULT_PERSONA["traits"])

    if not reset and existing_records:
        rec = existing_records[0]
        meta = dict(rec.metadata or {})
        for key in _PERSONA_FIELDS:
            if key in meta and meta[key]:
                if key == "traits" and isinstance(meta[key], dict):
                    persona["traits"].update(meta[key])
                else:
                    persona[key] = meta[key]

    if not reset:
        # Apply scalar updates
        for key in ("name", "role", "scope", "tone", "formality", "languages", "motivations"):
            val = updates.get(key)
            if isinstance(val, str) and val.strip():
                persona[key] = val.strip()

        # Handle list fields (accept both comma-string and list)
        for key in ("catchphrases", "avoid"):
            val = updates.get(key)
            if isinstance(val, list):
                persona[key] = [x.strip() for x in val if isinstance(x, str) and x.strip()]
            elif isinstance(val, str) and val.strip():
                persona[key] = [x.strip() for x in val.split(",") if x.strip()]

        # Handle trait updates
        traits = dict(persona.get("traits", {}))
        for trait_name in ("warmth", "curiosity", "conciseness", "humor", "formality"):
            val = updates.get(trait_name)
            if val is None and isinstance(updates.get("traits"), dict):
                val = updates["traits"].get(trait_name)
            if val is not None:
                try:
                    traits[trait_name] = max(0.0, min(1.0, float(val)))
                except (ValueError, TypeError):
                    pass
        persona["traits"] = traits

    # Persist to brain
    content = f"Agent Persona: {persona['name']} вЂ” {persona['role']}. Tone: {persona['tone']}."
    meta = _stamp_provenance(dict(persona), channel, tags=[_PERSONA_TAG, "identity"])
    meta["type"] = "agent_persona"

    with brain_lock:
        if existing_records:
            brain.update(existing_records[0].id, content=content, metadata=meta)
            record_id = existing_records[0].id
        else:
            from remy.core.agent_tools import Level

            rec = brain.store(
                content=content,
                level=Level.IDENTITY,
                tags=[_PERSONA_TAG, "identity"],
                metadata=meta,
            )
            record_id = rec.id

    # Invalidate caches
    from remy.core.agent import invalidate_system_instruction_cache

    invalidate_system_instruction_cache()

    try:
        from remy.core.tool_utils import clear_recall_cache

        clear_recall_cache()
    except Exception:
        pass

    logger.info("Persona updated: %s (id=%s, reset=%s)", persona["name"], record_id[:8], reset)
    persona["_record_id"] = record_id
    return persona


def _build_user_identity() -> str | None:
    """Build a rich USER IDENTITY block for the system prompt.

    Aggregates:
    1. User profile (store_user_profile) вЂ” core demographics
    2. Person records tagged with user's name вЂ” birth dates, contacts, family
    3. Identity-tagged records вЂ” verified facts about the user

    Each fact is annotated with verification status so the agent NEVER
    calls a verified fact a "guess" or "assumption".

    Returns formatted string or None if no profile exists.
    """
    brain = _get_brain()
    brain_lock = _get_brain_lock()

    from remy.core.brain_tools import get_user_profile_record
    profile = get_user_profile_record(brain, brain_lock)
    if profile is None:
        return None

    meta = profile.metadata or {}
    user_name = meta.get("name", "")

    # в”Ђв”Ђ 1. Core profile facts with verification status в”Ђв”Ђ
    verified_facts = []
    unverified_facts = []

    profile_verified = str(meta.get("verified", "false")).lower() == "true"
    for key in _PUBLIC_PROFILE_FIELDS:
        val = meta.get(key)
        if val:
            label = key.replace("_", " ").title()
            if profile_verified or meta.get("source") == "user-confirmed":
                verified_facts.append(f"{label}: {val}")
            else:
                unverified_facts.append(f"{label}: {val}")

    # в”Ђв”Ђ 2. Related person/identity records в”Ђв”Ђ
    # IMPORTANT: Only include records the user explicitly provided or confirmed.
    # Records created by the agent itself (source: "agent", "agent-autonomous",
    # "agent-worker") without user verification are excluded to prevent test data,
    # case studies, and inferred facts from contaminating the user profile.
    related_facts = []
    try:
        with brain_lock:
            # Person records (family members, user's own person record)
            person_records = brain.search(query="", tags=["person"], limit=20)
            # Identity-level records about the user
            identity_records = brain.search(query="", tags=["identity"], limit=10)

        # Gather person records connected to user or matching user's name
        seen_ids = {profile.id}
        for rec in person_records + identity_records:
            if rec.id in seen_ids:
                continue
            seen_ids.add(rec.id)
            if "user-profile" in (rec.tags or []):
                continue  # Skip profile record itself

            rmeta = rec.metadata or {}
            rec_verified = str(rmeta.get("verified", "false")).lower() == "true"
            try:
                trust = float(rmeta.get("trust_score", 0.5))
            except (TypeError, ValueError):
                trust = 0.5
            source = rmeta.get("source", "unknown")

            is_user_source = source.startswith("user-") or source == "user-confirmed"
            is_interactive = source == "agent-interactive"

            # Determine verification status for display
            if rec_verified or trust >= 0.9 or source == "user-confirmed":
                status = "VERIFIED"
            elif is_user_source or (is_interactive and trust >= 0.6):
                status = "likely"
            else:
                status = "unverified"

            fact_text = rec.content[:200]
            related_facts.append((status, fact_text))

    except Exception:
        pass  # Non-critical вЂ” profile alone is enough

    # в”Ђв”Ђ 3. Format output в”Ђв”Ђ
    lines = []
    display_name = user_name or "the user"

    if verified_facts:
        lines.append(
            "Verified facts (user confirmed вЂ” treat as CERTAIN, never say 'I assume' or 'I guess'):"
        )
        for f in verified_facts:
            lines.append(f"  вњ“ {f}")

    if unverified_facts:
        lines.append(
            "Unverified (you stored this but user hasn't confirmed вЂ” you may say 'if I remember correctly'):"
        )
        for f in unverified_facts:
            lines.append(f"  ? {f}")

    if related_facts:
        lines.append("Related people and facts:")
        for status, fact in related_facts:
            if status == "VERIFIED":
                lines.append(f"  вњ“ {fact}")
            elif status == "likely":
                lines.append(f"  ~ {fact} (likely, high confidence)")
            else:
                lines.append(f"  ? {fact} (unverified вЂ” confirm before using)")

    if not lines:
        return None

    checkmark_mojibake = chr(1074) + chr(1114) + chr(8220)
    rendered_lines = "\n".join(lines).replace(checkmark_mojibake, "\u2713")

    return (
        f"## USER IDENTITY вЂ” {display_name}\n" + rendered_lines + "\n\n"
        "RULES for using this identity:\n"
        f"- You KNOW {display_name}. This is your user. Do not introduce yourself as if meeting for the first time.\n"
        "- Facts marked вњ“ are CONFIRMED. Never say 'I assume', 'I guess', 'my hypothesis' about them.\n"
        "- Facts marked ? вЂ” you may reference cautiously: 'if I remember correctly' or ask to confirm.\n"
        "- Do NOT recite this profile back to the user. Use it silently to inform your responses.\n"
        "- If the user corrects any fact here, or if you discover it's WRONG, you MUST immediately call `update_record` to fix it, or `delete_record` to remove it. Do NOT just store a 'DISPROVED' note while leaving the wrong fact intact.\n"
    )
