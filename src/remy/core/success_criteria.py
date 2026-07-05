"""
Success Criteria for Goals (AUTON-5) — formal, machine-checkable goal completion.

Instead of relying solely on a 200-token LLM self-eval to guess whether a goal
is done, each goal can carry structured success criteria that are verified
programmatically before (or instead of) the LLM check.

Criteria Types:
- record_stored(tags): brain record with given tags exists
- research_complete(topic): a research project for topic is marked complete
- brain_count_increased(min_delta): brain has N+ more records than at goal start
- tool_result(...): session_log contains a matching verified tool outcome
- artifact_created(...): session_log contains a concrete artifact like record_id/url/file
- numeric_result(...): session_log contains a numeric result meeting a threshold
- signup_completed(...): browser evidence indicates account creation/login flow finished
- post_published(...): browser evidence indicates a post is live/published
- custom(description): free-text, requires LLM to verify (fallback)

Design:
- Criteria stored in goal metadata as `success_criteria: [...]`
- `verify_criteria()` returns (met_count, total, details)
- Integrated into _evaluate_outcome: if all criteria met → auto-complete
"""

import logging
import re
from pathlib import Path

from remy.core.agent_tools import brain

logger = logging.getLogger("Autonomy.Criteria")


# ============== Criterion verification ==============


def verify_criterion(
    criterion: dict,
    *,
    session_log: list[dict] | None = None,
) -> tuple[bool, str]:
    """Verify a single success criterion. Returns (met, reason)."""
    ctype = criterion.get("type", "custom")

    try:
        if ctype == "record_stored":
            return _verify_record_stored(criterion)
        elif ctype == "research_complete":
            return _verify_research_complete(criterion)
        elif ctype == "brain_count_increased":
            return _verify_brain_count(criterion)
        elif ctype == "file_exists":
            return _verify_file_exists(criterion)
        elif ctype == "tool_result":
            return _verify_tool_result(criterion, session_log=session_log)
        elif ctype == "artifact_created":
            return _verify_artifact_created(criterion, session_log=session_log)
        elif ctype == "numeric_result":
            return _verify_numeric_result(criterion, session_log=session_log)
        elif ctype == "signup_completed":
            return _verify_signup_completed(criterion, session_log=session_log)
        elif ctype == "post_published":
            return _verify_post_published(criterion, session_log=session_log)
        elif ctype == "draft_created":
            return _verify_draft_created(criterion, session_log=session_log)
        elif ctype == "custom":
            # Custom criteria can't be verified programmatically
            return False, f"Requires LLM: {criterion.get('description', '?')}"
        else:
            return False, f"Unknown criterion type: {ctype}"
    except Exception as e:
        logger.debug("Criterion check failed: %s", e)
        return False, f"Check error: {e}"


def verify_criteria(
    criteria: list[dict],
    *,
    session_log: list[dict] | None = None,
) -> tuple[int, int, list[dict]]:
    """Verify all criteria for a goal.

    Returns:
        (met_count, total_count, details)
        where details = [{"type": ..., "met": bool, "reason": str}, ...]
    """
    if not criteria:
        return 0, 0, []

    details = []
    met = 0

    for c in criteria:
        is_met, reason = verify_criterion(c, session_log=session_log)
        details.append(
            {
                "type": c.get("type", "custom"),
                "description": c.get("description", ""),
                "met": is_met,
                "reason": reason,
            }
        )
        if is_met:
            met += 1

    return met, len(criteria), details


def format_criteria_for_prompt(
    criteria: list[dict],
    *,
    session_log: list[dict] | None = None,
) -> str:
    """Format criteria as text for the decision/evaluation prompt."""
    if not criteria:
        return ""

    met_count, total, details = verify_criteria(criteria, session_log=session_log)

    lines = [f"\nSUCCESS CRITERIA ({met_count}/{total} met):"]
    for d in details:
        mark = "[x]" if d["met"] else "[ ]"
        desc = d.get("description") or d.get("type", "?")
        lines.append(f"  {mark} {desc}")
        if not d["met"] and d.get("reason"):
            lines.append(f"      → {d['reason']}")

    return "\n".join(lines) + "\n"


def all_criteria_met(
    criteria: list[dict],
    *,
    session_log: list[dict] | None = None,
) -> bool:
    """Quick check: are all criteria satisfied?"""
    if not criteria:
        return False  # No criteria = can't auto-verify

    met, total, _ = verify_criteria(criteria, session_log=session_log)
    return met == total and total > 0


# ============== Criteria generation ==============


def infer_goal_template(goal_description: str) -> dict | None:
    """Infer a lightweight goal template for common autonomous job families."""
    desc_lower = goal_description.lower()
    target_url = _extract_url_like_target(goal_description)
    target_number = _extract_target_number(goal_description)

    if any(
        kw in desc_lower
        for kw in ("register", "sign up", "signup", "create account", "log in", "login")
    ):
        return {
            "name": "signup_operator",
            "goal_type": "signup",
            "target_url": target_url,
        }

    if any(kw in desc_lower for kw in ("publish", "post", "tweet", "submit post", "share post")):
        return {
            "name": "publisher",
            "goal_type": "publish",
            "target_url": target_url,
        }

    if any(kw in desc_lower for kw in ("market", "competitive", "competitor", "osint")) and any(
        kw in desc_lower for kw in ("research", "analysis", "analyze", "report", "map")
    ):
        return {
            "name": "market_research",
            "goal_type": "market",
            "target_count": target_number or 3,
        }

    return None


def build_template_criteria(goal_description: str, template: dict) -> list[dict]:
    """Build success criteria for an inferred goal template."""
    name = (template or {}).get("name", "")
    criteria: list[dict] = []

    if name == "signup_operator":
        target_url = template.get("target_url")
        if target_url:
            criteria.append(
                {
                    "type": "tool_result",
                    "tool": ["browse_page", "browser_act"],
                    "verified": True,
                    "url_contains": target_url,
                    "description": f"Verified browser step reached '{target_url}'",
                }
            )
        criteria.append(
            {
                "type": "signup_completed",
                "description": "Signup/login flow reached a logged-in destination",
                **({"url_contains": target_url} if target_url else {}),
            }
        )
        return criteria

    if name == "publisher":
        target_url = template.get("target_url")
        if target_url:
            criteria.append(
                {
                    "type": "tool_result",
                    "tool": ["browse_page", "browser_act"],
                    "verified": True,
                    "url_contains": target_url,
                    "description": f"Verified browser step reached '{target_url}'",
                }
            )
        # Publisher pack stops at draft — draft creation IS success.
        # post_published is an optional bonus, not required.
        criteria.append(
            {
                "type": "draft_created",
                "description": "A draft was created and content was filled in (ready for review)",
                **({"url_contains": target_url} if target_url else {}),
            }
        )
        return criteria

    if name == "market_research":
        target_count = max(int(template.get("target_count", 3) or 3), 1)
        criteria.extend(
            [
                {
                    "type": "research_complete",
                    "topic": goal_description[:80],
                    "description": f"Research on '{goal_description[:80]}' is complete",
                },
                {
                    "type": "numeric_result",
                    "tool": "add_research_finding",
                    "fields": ["findings_count"],
                    "min_value": target_count,
                    "description": f"Collected at least {target_count} research findings",
                },
                {
                    "type": "artifact_created",
                    "tool": ["complete_research", "generate_report"],
                    "artifact_fields": ["report_id", "record_id", "url", "filename"],
                    "description": "A concrete research artifact was produced",
                },
            ]
        )
        return criteria

    return criteria


def generate_criteria_for_goal(goal_description: str) -> list[dict]:
    """Generate success criteria from a goal description. Zero LLM — heuristic.

    Returns a list of criterion dicts, or empty list if we can't infer.
    """
    from datetime import datetime

    desc_lower = goal_description.lower()
    template = infer_goal_template(goal_description)
    criteria = build_template_criteria(goal_description, template) if template else []
    now_iso = datetime.now().isoformat()

    # Research goals → research project completion
    research_kw = ("research", "investigate", "study", "find out", "learn about")
    for kw in research_kw:
        if kw in desc_lower and not any(c.get("type") == "research_complete" for c in criteria):
            # Extract topic (rough: take everything after the keyword)
            idx = desc_lower.index(kw) + len(kw)
            topic = goal_description[idx:].strip().rstrip(".")[:80]
            if topic:
                criteria.append(
                    {
                        "type": "research_complete",
                        "topic": topic,
                        "description": f"Research on '{topic}' is complete",
                    }
                )
            break

    # Storage goals → check record exists (time-scoped, not global count)
    store_kw = ("store", "save", "record", "remember", "track")
    for kw in store_kw:
        if kw in desc_lower:
            criteria.append(
                {
                    "type": "brain_count_increased",
                    "min_delta": 1,
                    "start_time": now_iso,
                    "description": "At least 1 new record stored",
                }
            )
            break

    target_url = _extract_url_like_target(goal_description)
    browser_kw = (
        "browse",
        "open",
        "navigate",
        "go to",
        "visit",
        "register",
        "sign up",
        "signup",
        "log in",
        "login",
        "post",
        "publish",
        "submit",
    )
    if (
        target_url
        and any(kw in desc_lower for kw in browser_kw)
        and not any(
            c.get("type") == "tool_result" and c.get("url_contains") == target_url for c in criteria
        )
    ):
        criteria.append(
            {
                "type": "tool_result",
                "tool": ["browse_page", "browser_act"],
                "verified": True,
                "url_contains": target_url,
                "description": f"Verified browser step reached '{target_url}'",
            }
        )

    signup_kw = ("register", "sign up", "signup", "create account", "log in", "login")
    if any(kw in desc_lower for kw in signup_kw) and not any(
        c.get("type") == "signup_completed" for c in criteria
    ):
        signup_criterion = {
            "type": "signup_completed",
            "description": "Signup/login flow reached a logged-in destination",
        }
        if target_url:
            signup_criterion["url_contains"] = target_url
        criteria.append(signup_criterion)

    publish_kw = ("publish", "post", "tweet", "submit post", "share post")
    if any(kw in desc_lower for kw in publish_kw) and not any(
        c.get("type") in ("post_published", "draft_created") for c in criteria
    ):
        # Draft creation is the success threshold for publisher pack.
        # The agent stops at draft — publishing requires user approval.
        draft_criterion = {
            "type": "draft_created",
            "description": "A draft was created and content was filled in (ready for review)",
        }
        if target_url:
            draft_criterion["url_contains"] = target_url
        criteria.append(draft_criterion)

    report_kw = ("report", "summary", "analysis", "market analysis", "pdf")
    if any(kw in desc_lower for kw in report_kw) and not any(
        c.get("type") == "artifact_created" for c in criteria
    ):
        criteria.append(
            {
                "type": "artifact_created",
                "tool": ["complete_research", "generate_report"],
                "artifact_fields": ["report_id", "record_id", "url", "filename"],
                "description": "A concrete report artifact was produced",
            }
        )

    lead_kw = ("lead", "leads", "prospect", "prospects", "client list", "contacts")
    target_number = _extract_target_number(goal_description)
    if (
        target_number
        and any(kw in desc_lower for kw in lead_kw)
        and not any(c.get("type") == "numeric_result" for c in criteria)
    ):
        criteria.append(
            {
                "type": "numeric_result",
                "fields": ["count", "stored_count", "findings_count"],
                "min_value": target_number,
                "description": f"Collected at least {target_number} lead-like items",
            }
        )

    # Metric tracking goals -> generic metric record
    if any(kw in desc_lower for kw in ("metric", "track", "measure", "score")):
        criteria.append(
            {
                "type": "record_stored",
                "tags": ["metric"],
                "description": "Metric record stored",
            }
        )

    # If we couldn't generate specific criteria, add a generic custom one
    if not criteria:
        criteria.append(
            {
                "type": "custom",
                "description": f"Goal achieved: {goal_description[:100]}",
            }
        )

    return criteria


# ============== Internal verifiers ==============


def _verify_record_stored(criterion: dict) -> tuple[bool, str]:
    """Check if a brain record with specified tags exists."""
    tags = criterion.get("tags", [])
    if not tags:
        return False, "No tags specified"

    from remy.core.agent_tools import brain_lock

    with brain_lock:
        records = brain.search(query="", tags=tags, limit=1)

    if records:
        return True, f"Found record with tags {tags}"
    return False, f"No record with tags {tags}"


def _verify_research_complete(criterion: dict) -> tuple[bool, str]:
    """Check if a research project for the topic is marked complete."""
    topic = criterion.get("topic", "")
    if not topic:
        return False, "No topic specified"

    from remy.core.agent_tools import brain_lock

    with brain_lock:
        records = brain.search(
            query=topic,
            tags=["research-project"],
            limit=5,
        )

    for r in records:
        meta = getattr(r, "metadata", None) or {}
        if isinstance(meta, str):
            import json

            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                continue
        if meta.get("status") == "completed":
            return True, f"Research project completed: {r.content[:60]}"

    return False, f"No completed research project for '{topic[:40]}'"


def _verify_brain_count(criterion: dict) -> tuple[bool, str]:
    """Check if new records were created since the goal started.

    Instead of comparing global brain.count() (which can shrink due to
    background decay/archival), we search for records created after start_time.
    Falls back to start_count comparison if start_time is not available.
    """
    min_delta = criterion.get("min_delta", 1)
    start_time = criterion.get("start_time")
    start_count = criterion.get("start_count")

    from remy.core.agent_tools import brain_lock

    # Preferred: count records created after goal start time
    if start_time:
        from datetime import datetime

        with brain_lock:
            all_records = brain.list_records(limit=200)
        # Count records with created_at > start_time
        new_count = 0
        for r in all_records:
            created = getattr(r, "created_at", None)
            if created is None:
                meta = getattr(r, "metadata", None) or {}
                created = meta.get("created_at") if isinstance(meta, dict) else None
            if created:
                try:
                    if isinstance(created, str):
                        created_dt = datetime.fromisoformat(created)
                    else:
                        created_dt = created
                    start_dt = (
                        datetime.fromisoformat(start_time)
                        if isinstance(start_time, str)
                        else start_time
                    )
                    if created_dt > start_dt:
                        new_count += 1
                except (ValueError, TypeError):
                    continue
        if new_count >= min_delta:
            return True, f"{new_count} new records since goal started (needed {min_delta})"
        return False, f"{new_count} new records since goal started (need {min_delta})"

    # Fallback: global count comparison (imprecise — background tasks can alter count)
    if start_count is None:
        return False, "No start_time or start_count baseline"

    with brain_lock:
        current = brain.count()

    delta = current - start_count
    if delta >= min_delta:
        return True, f"Brain grew by {delta} records (needed {min_delta})"
    return False, f"Brain grew by {delta} records (need {min_delta})"


def _verify_file_exists(criterion: dict) -> tuple[bool, str]:
    """Check if a file exists at the specified path."""
    path_str = criterion.get("path", "")
    if not path_str:
        return False, "No path specified"

    path = Path(path_str)
    if path.exists():
        return True, f"File exists: {path.name}"
    return False, f"File not found: {path_str}"


def _verify_tool_result(
    criterion: dict,
    *,
    session_log: list[dict] | None = None,
) -> tuple[bool, str]:
    """Check whether a structured tool result exists in the current runtime log."""
    if not session_log:
        return False, "No session log available for runtime verification"

    matched = _matching_tool_entries(session_log, criterion)
    min_calls = max(int(criterion.get("min_calls", 1) or 1), 1)

    if len(matched) < min_calls:
        tool_desc = _tool_desc(criterion)
        return (
            False,
            f"Only found {len(matched)}/{min_calls} matching runtime results for {tool_desc}",
        )

    latest = matched[-1]
    latest_evidence = latest.get("evidence") if isinstance(latest.get("evidence"), dict) else {}
    latest_url = (
        latest_evidence.get("page_url")
        or latest_evidence.get("requested_url")
        or latest.get("requested_url")
        or "n/a"
    )
    return True, f"Matched {len(matched)} runtime result(s); latest evidence URL: {latest_url}"


def _verify_artifact_created(
    criterion: dict,
    *,
    session_log: list[dict] | None = None,
) -> tuple[bool, str]:
    """Check for concrete artifacts like record IDs, URLs, files, or reports."""
    if not session_log:
        return False, "No session log available for artifact verification"

    artifact_fields = criterion.get("artifact_fields") or [
        "record_id",
        "report_id",
        "finding_id",
        "id",
        "url",
        "path",
        "filename",
    ]
    if isinstance(artifact_fields, str):
        artifact_fields = [artifact_fields]

    for entry in reversed(_matching_tool_entries(session_log, criterion)):
        for field in artifact_fields:
            value = entry.get(field)
            if value not in (None, "", False):
                return True, f"Artifact field '{field}' created: {value}"

    return False, f"No matching artifact found in fields {artifact_fields}"


def _verify_numeric_result(
    criterion: dict,
    *,
    session_log: list[dict] | None = None,
) -> tuple[bool, str]:
    """Check whether any matching tool result reports a numeric threshold."""
    if not session_log:
        return False, "No session log available for numeric verification"

    fields = criterion.get("fields") or criterion.get("field") or ["count"]
    if isinstance(fields, str):
        fields = [fields]
    min_value = float(criterion.get("min_value", 1))

    best_value = None
    best_field = None
    for entry in _matching_tool_entries(session_log, criterion):
        for field in fields:
            value = entry.get(field)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric = float(value)
            else:
                try:
                    numeric = float(str(value))
                except (TypeError, ValueError):
                    continue
            if best_value is None or numeric > best_value:
                best_value = numeric
                best_field = field

    if best_value is None:
        return False, f"No numeric result found in fields {fields}"
    if best_value >= min_value:
        return True, f"Field '{best_field}' reached {best_value:g} (needed {min_value:g})"
    return False, f"Best '{best_field}' value was {best_value:g} (need {min_value:g})"


def _verify_signup_completed(
    criterion: dict,
    *,
    session_log: list[dict] | None = None,
) -> tuple[bool, str]:
    """Check if browser evidence suggests signup/login reached an account destination."""
    if not session_log:
        return False, "No session log available for signup verification"

    matched = _matching_tool_entries(
        session_log,
        {
            **criterion,
            "tool": criterion.get("tool") or ["browse_page", "browser_act"],
            "verified": criterion.get("verified", True),
        },
    )
    if not matched:
        return False, "No verified browser results matched the signup flow"

    positive_url_markers = criterion.get("positive_url_markers") or [
        "dashboard",
        "account",
        "profile",
        "welcome",
        "onboarding",
        "getting-started",
    ]
    positive_text_markers = criterion.get("success_indicators") or [
        "welcome",
        "dashboard",
        "my account",
        "profile",
        "logout",
        "sign out",
        "getting started",
        "account created",
    ]
    negative_markers = criterion.get("negative_indicators") or [
        "captcha",
        "invalid",
        "error",
        "sign up",
        "register",
        "create account",
        "verification code",
        "confirm your email",
        "check your email",
        "try again",
    ]

    for entry in reversed(matched):
        url = _entry_url(entry)
        text = _entry_text_blob(entry)

        if any(marker in url for marker in positive_url_markers):
            if not any(marker in text for marker in negative_markers):
                return True, f"Signup flow reached account-like URL: {url}"

        if any(marker in text for marker in positive_text_markers) and not any(
            marker in text for marker in negative_markers
        ):
            return True, f"Signup flow shows logged-in indicators at {url or 'observed page'}"

    return False, "No verified signup evidence showed a logged-in/account destination"


def _verify_post_published(
    criterion: dict,
    *,
    session_log: list[dict] | None = None,
) -> tuple[bool, str]:
    """Check if browser evidence suggests a post/publication is live."""
    if not session_log:
        return False, "No session log available for post verification"

    matched = _matching_tool_entries(
        session_log,
        {
            **criterion,
            "tool": criterion.get("tool") or ["browse_page", "browser_act"],
            "verified": criterion.get("verified", True),
        },
    )
    if not matched:
        return False, "No verified browser results matched the publishing flow"

    positive_url_markers = criterion.get("positive_url_markers") or [
        "/status/",
        "/posts/",
        "/post/",
        "/p/",
        "/article/",
        "/articles/",
    ]
    positive_text_markers = criterion.get("success_indicators") or [
        "published",
        "posted",
        "live",
        "view post",
        "view article",
        "your post",
        "publication successful",
        "post is live",
    ]
    negative_markers = criterion.get("negative_indicators") or [
        "draft",
        "validation error",
        "failed",
        "captcha",
        "try again",
        "not independently verified",
        "compose",
        "new post",
    ]

    for entry in reversed(matched):
        url = _entry_url(entry)
        text = _entry_text_blob(entry)

        if any(marker in url for marker in positive_url_markers):
            if not any(marker in text for marker in negative_markers):
                return True, f"Observed live post URL: {url}"

        if any(marker in text for marker in positive_text_markers) and not any(
            marker in text for marker in negative_markers
        ):
            return True, f"Publishing flow reported live/public state at {url or 'observed page'}"

    return False, "No verified evidence showed a live published post"


def _verify_draft_created(
    criterion: dict,
    *,
    session_log: list[dict] | None = None,
) -> tuple[bool, str]:
    """Check if browser evidence suggests a draft was created/filled in.

    Publisher pack stops at draft state — so draft creation IS success.
    We look for: compose/editor pages visited, text entered, draft saved.
    """
    if not session_log:
        return False, "No session log available for draft verification"

    matched = _matching_tool_entries(
        session_log,
        {
            **criterion,
            "tool": criterion.get("tool") or ["browse_page", "browser_act"],
            "verified": criterion.get("verified", True),
        },
    )
    if not matched:
        return False, "No verified browser results matched a drafting flow"

    # Evidence that a draft/compose flow was engaged
    draft_url_markers = [
        "/compose",
        "/new",
        "/edit",
        "/draft",
        "/create",
        "/intent/tweet",
        "/submit",
        "/write",
    ]
    draft_text_markers = [
        "draft",
        "compose",
        "new post",
        "create post",
        "write",
        "saved",
        "editor",
        "text area",
        "content",
        "title",
        "schedule",
        "queue",
        "review",
    ]

    for entry in reversed(matched):
        url = _entry_url(entry)
        text = _entry_text_blob(entry)

        if any(marker in url for marker in draft_url_markers):
            return True, f"Draft flow detected at URL: {url}"

        if any(marker in text for marker in draft_text_markers):
            return True, f"Draft activity detected at {url or 'observed page'}"

    return False, "No verified evidence showed a draft was created"


def _matching_tool_entries(session_log: list[dict], criterion: dict) -> list[dict]:
    """Filter tool-call log entries according to shared runtime matching rules."""
    tools = criterion.get("tool") or criterion.get("tools")
    if isinstance(tools, str):
        tools = [tools]
    tool_set = set(tools or [])
    expected_verified = criterion.get("verified")
    status_in = criterion.get("status_in")
    if isinstance(status_in, str):
        status_in = [status_in]
    status_set = set(status_in or [])
    action = criterion.get("action")
    url_contains = str(criterion.get("url_contains", "") or "").lower()
    result_contains = str(criterion.get("result_contains", "") or "").lower()
    evidence_keys = set(criterion.get("evidence_keys", []) or [])

    matched = []
    for entry in session_log:
        if entry.get("type") != "tool_call":
            continue
        if tool_set and entry.get("tool") not in tool_set:
            continue

        verification = entry.get("verification")
        verified = entry.get("verified")
        if verified is None and isinstance(verification, dict):
            verified = verification.get("verified")
        if expected_verified is not None and verified is not expected_verified:
            continue

        status = entry.get("status")
        if status is None and isinstance(verification, dict):
            status = verification.get("status")
        if status_set and status not in status_set:
            continue

        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        if action and evidence.get("action") != action:
            continue
        if evidence_keys and not evidence_keys.issubset(set(evidence.keys())):
            continue

        if url_contains:
            urls = [
                str(entry.get("url", "") or "").lower(),
                str(entry.get("requested_url", "") or "").lower(),
                str(evidence.get("requested_url", "") or "").lower(),
                str(evidence.get("page_url", "") or "").lower(),
            ]
            if not any(url_contains in url for url in urls):
                continue

        result_text = str(entry.get("result", "") or "").lower()
        description_text = str(entry.get("description", "") or "").lower()
        if (
            result_contains
            and result_contains not in result_text
            and result_contains not in description_text
        ):
            continue

        matched.append(entry)

    return matched


def _entry_url(entry: dict) -> str:
    evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
    return str(
        evidence.get("page_url")
        or entry.get("url")
        or evidence.get("requested_url")
        or entry.get("requested_url")
        or ""
    ).lower()


def _entry_text_blob(entry: dict) -> str:
    evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
    parts = [
        entry.get("result", ""),
        entry.get("description", ""),
        entry.get("answer", ""),
        entry.get("page_state", ""),
        entry.get("auth_state", ""),
        evidence.get("page_text_snippet", ""),
    ]
    return " ".join(str(part) for part in parts if part).lower()


def _tool_desc(criterion: dict) -> str:
    tools = criterion.get("tool") or criterion.get("tools")
    if isinstance(tools, str):
        return tools
    if tools:
        return ", ".join(sorted(str(t) for t in tools))
    return "any tool"


def _extract_url_like_target(text: str) -> str | None:
    """Extract a URL-like target from a goal description for browser verification."""
    url_match = re.search(r"(https?://[^\s)]+)", text, flags=re.IGNORECASE)
    if url_match:
        return url_match.group(1).rstrip(".,)")

    bare_domain = re.search(
        r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s)]*)?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if bare_domain:
        return bare_domain.group(1).rstrip(".,)")

    return None


def _extract_target_number(text: str) -> int | None:
    """Extract the first plausible target number from a goal description."""
    match = re.search(r"\b(\d{1,4})\b", text)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None
