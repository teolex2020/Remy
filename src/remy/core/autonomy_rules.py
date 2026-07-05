"""
Behavioral Rules Engine (AUTON-1) — meta-learning from repeated failures.

Rules are brain records that encode lessons learned from failures. Unlike
strategy_hints (text for LLM to interpret), rules are structured conditions
that the autonomy loop can apply programmatically:

1. Auto-generate rules after 3+ similar failures (same goal type + same issues)
2. Apply rules before strategy selection in _decide_and_act()
3. Inject matching rules into decision_prompt so LLM follows them
4. Track rule effectiveness and decay unused rules

Design:
- Rules stored as brain records with tags ["behavioral-rule"]
- Metadata: {condition_type, condition_value, action, confidence, applied_count,
             success_after_apply, source_failures, created_at}
- Rule types: "goal_keyword" (match goal description), "tool_failure" (tool failed),
              "critique_pattern" (repeated critique issues)
"""

import json
import logging
from datetime import datetime, timedelta

from remy.core.agent_tools import brain
from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy.Rules")

RULE_TAGS = ["behavioral-rule"]

# Minimum failures before generating a rule
MIN_FAILURES_FOR_RULE = 3

# Days before an unused rule is archived
RULE_EXPIRY_DAYS = 7


def load_active_rules() -> list[dict]:
    """Load all active behavioral rules from brain.

    Returns list of rule dicts with fields:
        record_id, condition_type, condition_value, action, confidence,
        applied_count, success_after_apply
    """
    try:
        from remy.core.agent_tools import brain_lock

        with brain_lock:
            records = brain.search(query="", tags=RULE_TAGS, limit=50)

        rules = []
        for r in records:
            meta = getattr(r, "metadata", None) or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            if meta.get("archived"):
                continue

            rules.append(
                {
                    "record_id": r.id if hasattr(r, "id") else str(r),
                    "content": r.content,
                    "condition_type": meta.get("condition_type", ""),
                    "condition_value": meta.get("condition_value", ""),
                    "action": meta.get("action", ""),
                    "confidence": float(meta.get("confidence", 0.5)),
                    "applied_count": int(meta.get("applied_count", 0)),
                    "success_after_apply": int(meta.get("success_after_apply", 0)),
                    "created_at": meta.get("created_at", ""),
                }
            )

        return rules

    except Exception as e:
        logger.warning("Failed to load behavioral rules: %s", e)
        return []


def match_rules(goal_description: str, rules: list[dict] | None = None) -> list[dict]:
    """Find rules that match the current goal.

    Matching logic:
    - "goal_keyword": condition_value substring found in goal description
    - "tool_failure": always matches (general tool guidance)
    - "critique_pattern": always matches (general quality guidance)
    """
    if rules is None:
        rules = load_active_rules()

    if not rules:
        return []

    goal_lower = goal_description.lower()
    matched = []

    for rule in rules:
        ctype = rule["condition_type"]
        cvalue = rule.get("condition_value", "").lower()

        if ctype == "goal_keyword" and cvalue and cvalue in goal_lower:
            matched.append(rule)
        elif ctype in ("tool_failure", "critique_pattern"):
            matched.append(rule)

    # Sort by confidence (highest first)
    matched.sort(key=lambda r: r["confidence"], reverse=True)
    return matched[:10]  # Cap at 10 rules


def format_rules_for_prompt(rules: list[dict]) -> str:
    """Format matched rules as text for injection into decision prompt."""
    if not rules:
        return ""

    lines = []
    for rule in rules:
        lines.append(f"- [{rule['condition_type']}] {rule['action']}")

    return (
        "\nBEHAVIORAL RULES (learned from past failures — follow these):\n"
        + "\n".join(lines)
        + "\n"
    )


def record_rule_applied(rule: dict, success: bool) -> None:
    """Update rule stats after it was applied in a cycle."""
    try:
        record_id = rule["record_id"]
        from remy.core.agent_tools import brain_lock

        with brain_lock:
            record = brain.get(record_id)
        if not record:
            return

        meta = getattr(record, "metadata", None) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        meta["applied_count"] = int(meta.get("applied_count", 0)) + 1
        if success:
            meta["success_after_apply"] = int(meta.get("success_after_apply", 0)) + 1
        meta["last_applied"] = datetime.now().isoformat()

        with brain_lock:
            brain.update(record_id, metadata=meta)

        logger.debug(
            "Rule %s applied (success=%s, total=%d)",
            record_id[:8],
            success,
            meta["applied_count"],
        )

    except Exception as e:
        logger.warning("Failed to update rule stats: %s", e)


def check_and_generate_rules(
    recent_critiques: list[dict] | None = None,
    recent_outcomes: list[dict] | None = None,
) -> list[str]:
    """Analyze recent failures/critiques and generate new rules if patterns found.

    Called periodically (e.g., after each cycle or in background).
    Returns list of new rule record IDs.
    """
    new_rule_ids = []

    # Strategy 1: Generate rules from repeated critique issues
    if recent_critiques is None:
        recent_critiques = _load_recent_critiques()

    if len(recent_critiques) >= MIN_FAILURES_FOR_RULE:
        # Categorize issues into standard buckets instead of comparing raw strings.
        # LLM critique text varies ("Cannot find file" vs "File missing" vs "Failed to read"),
        # so we normalize to canonical categories for reliable pattern detection.
        category_counts: dict[str, int] = {}
        for c in recent_critiques:
            for issue in c.get("issues", []):
                category = _categorize_issue(issue)
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1

        for category, count in category_counts.items():
            if count >= MIN_FAILURES_FOR_RULE:
                existing = _rule_exists("critique_pattern", category)
                if not existing:
                    rule_id = _create_rule(
                        condition_type="critique_pattern",
                        condition_value=category,
                        action=_CATEGORY_ACTIONS.get(
                            category,
                            f"Avoid: '{category}' pattern detected {count} times.",
                        ),
                        confidence=min(0.9, 0.5 + count * 0.1),
                        source_info=f"{count} critique occurrences (category: {category})",
                    )
                    if rule_id:
                        new_rule_ids.append(rule_id)

    # Strategy 2: Generate rules from goal type failures
    if recent_outcomes is None:
        recent_outcomes = _load_recent_outcomes()

    goal_failures: dict[str, int] = {}
    for o in recent_outcomes:
        if not o.get("success") and o.get("goal_type"):
            gtype = o["goal_type"].lower()
            goal_failures[gtype] = goal_failures.get(gtype, 0) + 1

    for goal_type, fail_count in goal_failures.items():
        if fail_count >= MIN_FAILURES_FOR_RULE:
            existing = _rule_exists("goal_keyword", goal_type)
            if not existing:
                rule_id = _create_rule(
                    condition_type="goal_keyword",
                    condition_value=goal_type,
                    action=(
                        f"Goals containing '{goal_type}' have failed {fail_count} times. "
                        f"Decompose immediately or try a different approach."
                    ),
                    confidence=min(0.9, 0.5 + fail_count * 0.1),
                    source_info=f"{fail_count} goal failures",
                )
                if rule_id:
                    new_rule_ids.append(rule_id)

    if new_rule_ids:
        logger.info("Generated %d new behavioral rules", len(new_rule_ids))
        event_bus.emit("rules_generated", {"count": len(new_rule_ids)})

    return new_rule_ids


def decay_stale_rules() -> int:
    """Archive rules that haven't been applied or confirmed in RULE_EXPIRY_DAYS.

    Returns count of archived rules.
    """
    rules = load_active_rules()
    archived = 0
    cutoff = datetime.now() - timedelta(days=RULE_EXPIRY_DAYS)

    for rule in rules:
        created_str = rule.get("created_at", "")
        if not created_str:
            continue

        try:
            created = datetime.fromisoformat(created_str)
        except ValueError:
            continue

        # Skip if created recently
        if created > cutoff:
            continue

        # Archive if never applied or low success rate
        applied = rule["applied_count"]
        successes = rule["success_after_apply"]

        if applied == 0 or (applied >= 3 and successes / applied < 0.3):
            try:
                from remy.core.agent_tools import brain_lock

                record_id = rule["record_id"]

                with brain_lock:
                    record = brain.get(record_id)
                if record:
                    meta = getattr(record, "metadata", None) or {}
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except (json.JSONDecodeError, TypeError):
                            meta = {}
                    meta["archived"] = True
                    meta["archived_reason"] = "stale_or_ineffective"
                    meta["archived_at"] = datetime.now().isoformat()
                    with brain_lock:
                        brain.update(record_id, metadata=meta)
                    archived += 1
            except Exception as e:
                logger.warning("Failed to archive rule %s: %s", rule["record_id"][:8], e)

    if archived:
        logger.info("Archived %d stale behavioral rules", archived)

    return archived


# ============== Issue categorization ==============

# Canonical categories for normalizing free-text LLM critique issues.
# Maps keyword patterns → category name.  This avoids the problem where
# "Cannot find file", "File missing", and "Failed to read file" are treated
# as three different issues — they all map to "file_not_found".
_ISSUE_CATEGORIES: list[tuple[str, list[str]]] = [
    (
        "file_not_found",
        [
            "file not found",
            "file missing",
            "cannot find file",
            "failed to read",
            "no such file",
            "filenotfounderror",
        ],
    ),
    ("timeout", ["timeout", "timed out", "took too long", "deadline exceeded"]),
    (
        "network_error",
        [
            "network error",
            "connection refused",
            "connection error",
            "dns resolution",
            "unreachable",
            "socket error",
            "http error",
            "status 5",
            "502",
            "503",
            "504",
        ],
    ),
    (
        "auth_error",
        [
            "unauthorized",
            "forbidden",
            "403",
            "401",
            "authentication",
            "access denied",
            "permission denied",
        ],
    ),
    ("rate_limit", ["rate limit", "too many requests", "429", "quota exceeded"]),
    (
        "empty_result",
        ["empty result", "no results", "nothing found", "no records", "no data", "empty response"],
    ),
    (
        "tool_error",
        ["tool failed", "tool error", "execution error", "tool not available", "tool not found"],
    ),
    ("memory_duplicate", ["duplicate", "already exists", "similar record", "already stored"]),
    ("parse_error", ["parse error", "json error", "invalid format", "syntax error", "malformed"]),
    (
        "loop_detected",
        ["repeated", "same action", "stuck", "infinite loop", "no progress", "going in circles"],
    ),
    (
        "no_action_taken",
        [
            "no tools",
            "no tool calls",
            "planned",
            "will do",
            "i'll",
            "let me",
            "i can",
            "i would",
            "i could",
            "no concrete",
            "just text",
            "no execution",
        ],
    ),
    (
        "fabricated_data",
        [
            "fabricated",
            "made up",
            "unverified",
            "not verified",
            "assumed",
            "guessed",
            "estimated without",
        ],
    ),
]

_CATEGORY_ACTIONS: dict[str, str] = {
    "file_not_found": "Verify file paths before reading. Use list_directory first.",
    "timeout": "Use shorter timeouts. Break large operations into smaller steps.",
    "network_error": "Check connectivity before web operations. Have a fallback plan.",
    "auth_error": "Verify credentials/permissions before accessing protected resources.",
    "rate_limit": "Add delays between API calls. Use cached results when available.",
    "empty_result": "Try broader search queries. Check if the data source is correct.",
    "tool_error": "Verify tool availability. Use alternative tools if primary fails.",
    "memory_duplicate": "Always recall before storing. Update existing records instead.",
    "parse_error": "Validate data format before processing. Handle malformed input.",
    "loop_detected": "Decompose the goal. Try a fundamentally different approach.",
    "no_action_taken": "STOP TALKING. Call a tool RIGHT NOW. No plans, no descriptions — execute.",
    "fabricated_data": "NEVER state facts without verifying. Use http_get, web_search, or recall to check data first.",
}


def _categorize_issue(issue_text: str) -> str:
    """Normalize a free-text issue description into a canonical category.

    Returns the category name, or empty string if no match.
    """
    text_lower = issue_text.strip().lower()
    if not text_lower:
        return ""

    for category, keywords in _ISSUE_CATEGORIES:
        for kw in keywords:
            if kw in text_lower:
                return category

    # No match — return a truncated version as fallback category
    # This still groups exact duplicates but won't help with paraphrases
    return ""


# ============== Internal helpers ==============


def _load_recent_critiques(limit: int = 20) -> list[dict]:
    """Load recent critique records from brain.

    Critique content format: "Self-critique for action X: quality=0.40. Issues: msg1; msg2. Suggestions: ..."
    We parse issues from the content text since they aren't stored in metadata.
    """
    try:
        from remy.core.agent_tools import brain_lock
        from remy.core.autonomy_critique import CRITIQUE_TAGS

        with brain_lock:
            records = brain.search(query="", tags=CRITIQUE_TAGS, limit=limit)

        critiques = []
        for r in records:
            meta = getattr(r, "metadata", None) or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            # Parse issues from content: "Issues: msg1; msg2. Suggestions: ..."
            issues = []
            content = getattr(r, "content", "") or ""
            if "Issues:" in content:
                after_issues = content.split("Issues:", 1)[1]
                # Cut at "Suggestions:" if present
                if "Suggestions:" in after_issues:
                    after_issues = after_issues.split("Suggestions:", 1)[0]
                # Split by semicolons, strip, filter empty/none
                for part in after_issues.strip().rstrip(".").split(";"):
                    part = part.strip()
                    if part and part.lower() != "none":
                        issues.append(part)

            critiques.append(
                {
                    "quality": float(meta.get("quality", 0.5)),
                    "issues": issues,
                    "goal": meta.get("goal", ""),
                }
            )
        return critiques

    except Exception:
        return []


def _load_recent_outcomes(limit: int = 20) -> list[dict]:
    """Load recent outcome records from brain."""
    try:
        from remy.core.agent_tools import brain_lock

        with brain_lock:
            records = brain.search(
                query="",
                tags=["autonomous-outcome"],
                limit=limit,
            )

        outcomes = []
        for r in records:
            meta = getattr(r, "metadata", None) or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            # Extract goal type from content heuristic
            content_lower = r.content.lower()
            goal_type = ""
            for kw in ("research", "write", "organize", "analyze", "learn", "health", "plan"):
                if kw in content_lower:
                    goal_type = kw
                    break

            outcomes.append(
                {
                    "success": "outcome-success" in (getattr(r, "tags", None) or []),
                    "goal_type": goal_type,
                    "content": r.content[:200],
                }
            )
        return outcomes

    except Exception:
        return []


def _rule_exists(condition_type: str, condition_value: str) -> bool:
    """Check if a similar rule already exists."""
    rules = load_active_rules()
    cv_lower = condition_value.lower()
    for rule in rules:
        if (
            rule["condition_type"] == condition_type
            and rule.get("condition_value", "").lower() == cv_lower
        ):
            return True
    return False


def _create_rule(
    condition_type: str,
    condition_value: str,
    action: str,
    confidence: float = 0.5,
    source_info: str = "",
) -> str | None:
    """Create a new behavioral rule in brain."""
    try:
        from remy.core.agent_tools import Level

        content = f"Behavioral rule [{condition_type}]: {action}"

        record = brain.store(
            content=content,
            level=Level.DOMAIN,  # Rules are persistent knowledge
            tags=RULE_TAGS,
            metadata={
                "condition_type": condition_type,
                "condition_value": condition_value,
                "action": action,
                "confidence": confidence,
                "applied_count": 0,
                "success_after_apply": 0,
                "source_info": source_info,
                "created_at": datetime.now().isoformat(),
            },
        )

        record_id = record.id if hasattr(record, "id") else str(record)
        logger.info(
            "Created behavioral rule: %s (type=%s, confidence=%.2f)",
            record_id[:8],
            condition_type,
            confidence,
        )
        return record_id

    except Exception as e:
        logger.warning("Failed to create behavioral rule: %s", e)
        return None
