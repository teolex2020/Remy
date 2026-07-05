"""
Browser Tool Dispatch — Playwright browser automation with smart circuit breaker.

Handles browse_page, browser_act, browser_close tools. Includes error pattern
analysis, consecutive failure tracking, and persistent browser event loop.
"""

import asyncio
import inspect
import json
import logging
import re
import threading
from urllib.parse import urlparse

logger = logging.getLogger("BrainTools")


# ============== BROWSER STATE ==============
# Mutable state lives in brain_tools.py (tests patch it there).
# Constants are local — they're immutable so re-import works fine.

_BROWSER_ANALYZE_THRESHOLD: int = 3  # after 3 failures → analyze & suggest pivot
_BROWSER_HARD_STOP_LIMIT: int = 6  # after 6 failures → hard stop (agent had its chance)
_LOGIN_ESCALATION_THRESHOLD: int = 3  # identical login error 3x → escalate to blocked_external


def _get_bt():
    """Lazy accessor for brain_tools module (mutable browser state lives there)."""
    import remy.core.brain_tools as _bt

    return _bt


def _is_login_page(page_url: str, page_text: str) -> bool:
    """Check if the current page is a login/signup form."""
    haystack = f"{page_url} {page_text}".lower()
    return any(
        m in haystack
        for m in (
            "sign in",
            "log in",
            "login",
            "signin",
            "signup",
            "sign up",
            "register",
            "create account",
            "accounts.google",
            "github.com/login",
            "auth",
            "identifier",
        )
    )


def _check_login_failure_escalation(parsed: dict) -> dict:
    """Track identical login error texts; escalate after threshold repeats.

    Mutates parsed dict: adds login_failure_escalation=True + blocker fields
    if the same visible_error_text appears _LOGIN_ESCALATION_THRESHOLD times.
    """
    visible_error = (parsed.get("visible_error_text") or "").strip()
    if not visible_error:
        return parsed

    page_url = parsed.get("url") or parsed.get("requested_url") or ""
    page_text = parsed.get("page_text_snippet") or ""
    if not _is_login_page(page_url, page_text):
        return parsed

    bt = _get_bt()
    bt._login_error_history.append(visible_error)
    if len(bt._login_error_history) > 15:
        bt._login_error_history = bt._login_error_history[-15:]

    # Count identical errors
    count = sum(1 for e in bt._login_error_history if e == visible_error)
    if count >= _LOGIN_ESCALATION_THRESHOLD:
        logger.warning(
            "Login failure escalation: '%s' seen %d times — marking blocked_external",
            visible_error[:80],
            count,
        )
        parsed["external_blocker_likely"] = True
        parsed["blocker_reason"] = f"repeated login failure: {visible_error[:120]}"
        parsed["blocker_count"] = count
        parsed["login_failure_escalation"] = True
        bt._login_error_history.clear()

    return parsed


def _analyze_browser_errors(errors: list[str]) -> str:
    """Analyze repeated browser errors to identify the pattern and suggest alternatives.

    Zero-LLM: pure string analysis of error messages.
    Returns a structured instruction for the agent.
    """
    # Detect common patterns
    timeout_errors = [e for e in errors if "Timeout" in e or "timeout" in e]
    not_found = [e for e in errors if "not found" in e.lower() or "no element" in e.lower()]
    detached = [e for e in errors if "detach" in e.lower() or "navigat" in e.lower()]

    # Find repeated selectors
    selectors = []
    for e in errors:
        m = re.search(r"selector='([^']+)'", e)
        if not m:
            m = re.search(r'locator\("([^"]+)"\)', e)
        if m:
            selectors.append(m.group(1))

    unique_selectors = list(dict.fromkeys(selectors))  # dedupe, keep order

    # Build diagnosis
    if timeout_errors and len(timeout_errors) >= len(errors) // 2:
        diagnosis = (
            "Elements exist in DOM but are NOT interactive (click timeout). "
            "Common causes: SPA hydration delay, overlay blocking clicks, "
            "elements rendered by JS framework but not yet attached to event handlers."
        )
        alternatives = (
            "1) Try browser_act(action='wait', text='3000') then retry ONE selector\n"
            "2) Try JavaScript click: browser_act(action='evaluate', text='document.querySelector(\"SELECTOR\").click()')\n"
            "3) If this is a SPA/Next.js site, the page may need more load time — try scrolling first\n"
            "4) If nothing works, this page may require human interaction — store as failed-hypothesis and move on"
        )
    elif not_found:
        diagnosis = (
            "Selectors don't match any elements on the page. "
            "The page structure may differ from what vision analysis reported."
        )
        alternatives = (
            "1) Call browse_page again to get fresh selectors\n"
            "2) Try broader selectors (tag name instead of specific class)\n"
            "3) The content may be inside an iframe — try browser_act(action='evaluate', text='document.querySelectorAll(\"iframe\").length')\n"
            "4) If the page is dynamically loaded, wait and retry"
        )
    elif detached:
        diagnosis = (
            "Elements are being detached/replaced by the page (likely SPA navigation). "
            "The page is changing between screenshot analysis and action execution."
        )
        alternatives = (
            "1) Call browse_page again immediately before acting\n"
            "2) Use browser_act(action='wait', text='2000') to let the page stabilize\n"
            "3) Try a different navigation path to reach the same content"
        )
    else:
        diagnosis = f"Repeated browser interaction failures. Errors: {'; '.join(e[:100] for e in errors[-3:])}"
        alternatives = (
            "1) Re-analyze the page with browse_page to get updated selectors\n"
            "2) Try a completely different approach to achieve the goal\n"
            "3) If this site is not cooperating, find an alternative site/method"
        )

    tried_selectors = ""
    if unique_selectors:
        tried_selectors = f"\nFailed selectors (do NOT retry these): {', '.join(s[:80] for s in unique_selectors[:5])}"

    return (
        f"⚠️ BROWSER ERROR PATTERN DETECTED ({len(errors)} consecutive failures)\n\n"
        f"Diagnosis: {diagnosis}\n"
        f"{tried_selectors}\n\n"
        f"REQUIRED ACTIONS:\n"
        f"1. Store this as negative knowledge:\n"
        f"   store(content='DISPROVED: [describe what you tried on this page]. "
        f"Reason: {diagnosis[:100]}. Alternative: [what to try instead]', "
        f"tags='failed-hypothesis,browser', level='L2_DECISIONS')\n"
        f"2. Try ONE alternative approach:\n{alternatives}\n"
        f"3. If the alternative also fails, call browser_close and report to the user.\n\n"
        f"Do NOT repeat the same selectors. Think about WHY it fails and try something fundamentally different."
    )


def _same_origin(url_a: str, url_b: str) -> bool:
    """Return True when two URLs share scheme + hostname."""
    if not url_a or not url_b:
        return False
    try:
        a = urlparse(url_a)
        b = urlparse(url_b)
        return a.scheme == b.scheme and a.hostname == b.hostname
    except Exception:
        return False


def _browser_backend_mode() -> str:
    from remy.config.settings import settings

    return (settings.BROWSER_BACKEND or "playwright").strip().lower()


def _pinchtab_enabled() -> bool:
    from remy.core.pinchtab import PinchTabManager

    mode = _browser_backend_mode()
    return mode in {"hybrid", "pinchtab"} and PinchTabManager.enabled()


def _selector_looks_like_pinchtab_ref(selector: str | None) -> bool:
    if not selector:
        return False
    text = selector.strip()
    if text.startswith("ref:"):
        text = text[4:]
    if text.startswith("pinch:"):
        text = text[6:]
    return bool(re.fullmatch(r"e\d+", text))


def _should_try_pinchtab_browse() -> bool:
    return _pinchtab_enabled()


def _should_try_pinchtab_act(action: str, selector: str | None) -> bool:
    if not _pinchtab_enabled():
        return False
    if action in {"goto", "open", "back", "forward", "wait"}:
        return True
    return _selector_looks_like_pinchtab_ref(selector)


def _pinchtab_browse_result(result: dict) -> dict:
    page_url = result.get("url", "")
    requested_url = result.get("requested_url", "")
    page_text = result.get("page_text", "")
    dom_elements = result.get("dom_elements", [])
    verification = _verify_browser_step(
        tool="browse_page",
        action=None,
        requested_url=requested_url,
        page_url=page_url,
        page_text=page_text,
    )
    data = {
        "backend": "pinchtab",
        "tab_id": result.get("tab_id", ""),
        "url": page_url,
        "requested_url": requested_url,
        "screenshot": "",
        "description": result.get("description") or "PinchTab text-first extraction completed.",
        "answer": result.get("answer") or "",
        "page_state": result.get("page_state") or ("normal" if page_text else "unknown"),
        "page_text_snippet": page_text[:500],
        "page_text": page_text[:5000],
        "dom_elements": dom_elements,
        "dom_form_fields": result.get("dom_form_fields", []),
        "snapshot_count": result.get("snapshot_count", len(dom_elements)),
        "verification": verification,
        "evidence": _build_browser_evidence(
            tool="browse_page",
            requested_url=requested_url,
            page_url=page_url,
            screenshot_path="",
            page_text=page_text,
        ),
    }
    data["verified"] = verification["verified"]
    data["status"] = verification["status"]
    return data


def _pinchtab_act_result(action: str, selector: str | None, result: dict) -> dict:
    page_url = result.get("url", "")
    requested_url = result.get("requested_url", "")
    page_text = result.get("page_text", "")
    verification = _verify_browser_step(
        tool="browser_act",
        action=action,
        requested_url=requested_url,
        page_url=page_url,
        page_text=page_text,
    )
    data = {
        "backend": "pinchtab",
        "tab_id": result.get("tab_id", ""),
        "action": action,
        "url": page_url,
        "requested_url": requested_url,
        "screenshot": "",
        "description": result.get("description") or verification["reason"],
        "page_state": result.get("page_state") or ("normal" if page_text else "unknown"),
        "page_text_snippet": page_text[:500],
        "page_text": page_text[:5000],
        "verification": verification,
        "evidence": _build_browser_evidence(
            tool="browser_act",
            action=action,
            selector=selector,
            requested_url=requested_url,
            page_url=page_url,
            screenshot_path="",
            page_text=page_text,
        ),
    }
    data["verified"] = verification["verified"]
    data["status"] = verification["status"]
    return data


def _build_browser_evidence(
    *,
    tool: str,
    requested_url: str,
    page_url: str,
    screenshot_path: str,
    action: str | None = None,
    selector: str | None = None,
    page_text: str = "",
) -> dict:
    """Build a compact evidence artifact for browser steps."""
    return {
        "tool": tool,
        "action": action or "",
        "requested_url": requested_url,
        "page_url": page_url,
        "screenshot": screenshot_path,
        "selector": selector or "",
        "page_text_snippet": page_text[:500],
    }


def _verify_browser_step(
    *,
    tool: str,
    action: str | None,
    requested_url: str,
    page_url: str,
    page_text: str,
) -> dict:
    """Produce a conservative verification result for browser steps."""
    if tool == "browse_page":
        if page_url and (not requested_url or _same_origin(requested_url, page_url)):
            return {
                "verified": True,
                "status": "verified",
                "reason": "Page loaded and current URL matches the requested origin.",
            }
        return {
            "verified": False,
            "status": "attempted",
            "reason": "Navigation executed, but the final page URL could not be verified.",
        }

    if action in {"goto", "back", "forward"}:
        if page_url:
            return {
                "verified": True,
                "status": "verified",
                "reason": "Navigation action completed and a current page URL is available.",
            }
        return {
            "verified": False,
            "status": "attempted",
            "reason": "Navigation was attempted, but no resulting page URL was observed.",
        }

    if action == "click":
        if page_url and page_text.strip():
            return {
                "verified": True,
                "status": "verified",
                "reason": "Click executed and the page remained observable afterward.",
            }
        return {
            "verified": False,
            "status": "attempted",
            "reason": "Click executed, but the outcome was not independently verified.",
        }

    return {
        "verified": False,
        "status": "executed_unverified",
        "reason": (
            f"Action '{action or tool}' executed, but semantic success was not independently verified. "
            "Treat as attempted until a follow-up check confirms the outcome."
        ),
    }


def _adjust_verification_with_analysis(
    verification: dict,
    analysis: dict,
    *,
    tool: str,
    action: str | None,
) -> dict:
    """Downgrade optimistic browser verification when post-action analysis shows blockers/errors."""
    if not isinstance(analysis, dict):
        return verification

    page_state = str(analysis.get("page_state", "") or "").lower()
    auth_state = str(analysis.get("auth_state", "") or "").lower()
    blocking_overlay = str(analysis.get("blocking_overlay", "") or "").strip()
    answer = str(analysis.get("answer", "") or "").lower()
    description = str(analysis.get("description", "") or "").lower()
    text = " ".join(part for part in (answer, description) if part)

    if page_state in {"captcha", "error", "loading", "cookie_banner", "popup"}:
        reason_map = {
            "captcha": "CAPTCHA detected after the action. Treat this step as blocked, not completed.",
            "error": "The page shows an error/validation state after the action.",
            "loading": "The page still appears to be loading after the action.",
            "cookie_banner": "A cookie/banner overlay is blocking the page after the action.",
            "popup": "A blocking popup/modal is present after the action.",
        }
        return {
            "verified": False,
            "status": "attempted",
            "reason": reason_map.get(page_state, verification.get("reason", "")),
        }

    if blocking_overlay:
        return {
            "verified": False,
            "status": "attempted",
            "reason": f"Blocking overlay detected after the action: {blocking_overlay}",
        }

    if tool == "browser_act" and action in {"click", "goto"}:
        if any(
            marker in text
            for marker in (
                "required field",
                "required fields",
                "validation error",
                "invalid email",
                "invalid password",
                "please correct",
                "try again",
                "submission failed",
            )
        ):
            return {
                "verified": False,
                "status": "attempted",
                "reason": "The page reports validation or submission errors after the action.",
            }
        if auth_state == "logged_out" and any(
            marker in text for marker in ("sign up", "register", "log in", "login")
        ):
            return {
                "verified": False,
                "status": "attempted",
                "reason": "The page still appears logged out after the action.",
            }

    return verification


def _extract_visible_error_text(analysis: dict) -> str:
    """Extract short page-visible blocker/error text for reporting."""
    if not isinstance(analysis, dict):
        return ""

    page_state = str(analysis.get("page_state", "") or "").lower()
    auth_state = str(analysis.get("auth_state", "") or "").lower()
    answer = str(analysis.get("answer", "") or "").strip()
    description = str(analysis.get("description", "") or "").strip()
    blocking_overlay = str(analysis.get("blocking_overlay", "") or "").strip()

    interesting_markers = (
        "invalid",
        "error",
        "required",
        "captcha",
        "verify",
        "verification",
        "incorrect",
        "wrong",
        "try again",
        "blocked",
        "phone",
        "email",
        "password",
        "login",
        "log in",
        "sign in",
        "sign up",
        "register",
    )

    candidates = []
    for text in (answer, description, blocking_overlay):
        if text and text not in candidates:
            candidates.append(text)

    if page_state in {"error", "captcha", "cookie_banner", "popup"} or auth_state == "logged_out":
        for text in candidates:
            lowered = text.lower()
            if any(marker in lowered for marker in interesting_markers):
                return text[:220]
        if candidates:
            return candidates[0][:220]
    return ""


def _structured_browser_value_error(
    *,
    action: str,
    selector: str | None,
    requested_url: str,
    error_text: str,
) -> dict:
    """Convert browser-side ValueError into structured attempted/error evidence."""
    error_text = str(error_text or "").strip()
    lowered = error_text.lower()
    page_state = "error"
    blocker_reason = ""
    agent_hint = ""

    if lowered.startswith("pre-submit blocked:"):
        blocker_reason = error_text
        agent_hint = "Fix the required fields or validation errors before retrying submit."
    elif lowered.startswith("disabled submit:"):
        blocker_reason = error_text
        agent_hint = "Do not retry the disabled button. Complete missing inputs or resolve validation before saving."
    else:
        page_state = "unknown"

    return {
        "action": action,
        "requested_url": requested_url,
        "url": requested_url,
        "selector": selector or "",
        "error": error_text,
        "description": error_text,
        "page_state": page_state,
        "visible_error_text": error_text[:220],
        "verification": {
            "verified": False,
            "status": "attempted",
            "reason": error_text,
        },
        "verified": False,
        "status": "attempted",
        "evidence": {
            "tool": "browser_act",
            "action": action,
            "requested_url": requested_url,
            "page_url": requested_url,
            "selector": selector or "",
            "screenshot": "",
            "page_text_snippet": "",
        },
        "blocker_reason": blocker_reason,
        "_agent_hint": agent_hint,
    }


def _check_accidental_publish(result_data: dict, page_url: str, page_text: str) -> dict:
    """Post-action check: detect if content was accidentally published live.

    If the page URL or text suggests content went live (not draft), add a
    warning to the result so the agent knows and the orchestrator can flag it.
    """
    from remy.core.approval_queue import approval_queue

    url_is_publish_platform = getattr(approval_queue, "url_is_publish_platform", None)
    if not callable(url_is_publish_platform) or not url_is_publish_platform(page_url):
        return result_data

    url_lower = (page_url or "").lower()
    text_lower = (page_text or "")[:2000].lower()

    # URL patterns that indicate published (not draft) content
    live_url_markers = (
        "/status/",
        "/posts/",
        "/p/",
        "/article/",
        "published",
        "is-live",
        "post-created",
    )
    # Text patterns that indicate content went live
    live_text_markers = (
        "your post is live",
        "published successfully",
        "post created",
        "shared successfully",
        "tweet sent",
        "article published",
        "your article is now live",
        "posted successfully",
    )

    is_live = any(m in url_lower for m in live_url_markers) or any(
        m in text_lower for m in live_text_markers
    )
    if is_live:
        logger.warning("DRAFT ENFORCEMENT: content appears published live at %s", page_url)
        result_data["_agent_hint"] = (
            "WARNING: Content appears to have been PUBLISHED LIVE. "
            "Autonomous publish should stay in draft/queue. "
            "Report this to the user immediately. Do NOT publish more content."
        )
        result_data["accidental_publish"] = True
        result_data["external_blocker_likely"] = True
        result_data["blocker_reason"] = f"accidental publish detected at {page_url}"

    return result_data


def _get_effective_url(name: str, args: dict, run_async) -> str:
    """Resolve the best available URL for approval and reporting."""
    url = (args.get("url") or "").strip()
    if url:
        return url
    if name != "browser_act":
        return ""
    try:
        from remy.core.browser import BrowserManager

        mgr = BrowserManager.get()
        current_url = run_async(mgr.get_page_url())
        return str(current_url or "").strip()
    except Exception:
        return ""


def _extract_failed_selectors(errors: list[str]) -> list[str]:
    selectors = []
    for error in errors:
        match = re.search(r"selector='([^']+)'", error)
        if not match:
            match = re.search(r'locator\("([^"]+)"\)', error)
        if match:
            selectors.append(match.group(1))
    return selectors


def _shape_repeated_retry(name: str, args: dict, current_url: str, bt) -> dict | None:
    """Stop obvious repeat attempts on the same selector/action after recent failures."""
    if name != "browser_act":
        return None

    action = str(args.get("action") or "").strip().lower()
    selector = str(args.get("selector") or "").strip()
    if action not in {"click", "type", "select"} or not selector:
        return None
    if bt._consecutive_browser_failures < 2:
        return None

    selectors = _extract_failed_selectors(bt._browser_error_history[-5:])
    repeated_count = sum(1 for item in selectors if item == selector)
    if repeated_count < 2:
        return None

    return {
        "action": action,
        "requested_url": current_url,
        "url": current_url,
        "verified": False,
        "status": "attempted",
        "retry_shaped": True,
        "retry_reason": "repeated_selector_failure",
        "repeated_selector": selector,
        "repeated_selector_count": repeated_count,
        "description": (
            f"Blocked repeated retry: selector '{selector}' already failed {repeated_count} times in this browser session."
        ),
        "verification": {
            "verified": False,
            "status": "attempted",
            "reason": (
                f"Selector '{selector}' was already tried {repeated_count} times. "
                "Do not repeat the same interaction without a new page analysis."
            ),
        },
        "evidence": {
            "tool": name,
            "action": action,
            "requested_url": current_url,
            "page_url": current_url,
            "selector": selector,
            "page_text_snippet": "",
        },
        "_agent_hint": (
            "Do not retry the same selector again. Call browse_page for fresh selectors, "
            "or choose a different interaction path."
        ),
    }


def _suggest_selector_alternatives(
    current_selector: str, preferred_selectors: list[dict], limit: int = 3
) -> list[str]:
    """Return ranked alternative selectors excluding the current one."""
    alternatives = []
    seen = set()
    for item in preferred_selectors or []:
        selector = str(item.get("selector") or "").strip()
        if not selector or selector == current_selector or selector in seen:
            continue
        seen.add(selector)
        alternatives.append(selector)
        if len(alternatives) >= limit:
            break
    return alternatives


def _record_browser_failure_from_result(name: str, args: dict, parsed: dict) -> None:
    """Persist a browser failure/degraded outcome for later diagnostics and routing."""
    if not isinstance(parsed, dict):
        return
    try:
        from remy.core.browser_failure_memory import record_browser_failure

        evidence = parsed.get("evidence") if isinstance(parsed.get("evidence"), dict) else {}
        verification = (
            parsed.get("verification") if isinstance(parsed.get("verification"), dict) else {}
        )
        url = (
            parsed.get("url")
            or parsed.get("requested_url")
            or evidence.get("page_url")
            or evidence.get("requested_url")
            or args.get("url")
            or ""
        )
        text = (
            parsed.get("error")
            or verification.get("reason")
            or parsed.get("answer")
            or parsed.get("description")
            or evidence.get("page_text_snippet")
            or ""
        )
        record_browser_failure(
            tool=name,
            action=str(parsed.get("action") or args.get("action") or ""),
            url=str(url),
            text=str(text),
            selector=str(
                parsed.get("selector") or evidence.get("selector") or args.get("selector") or ""
            ),
            status=str(parsed.get("status") or ("failed" if parsed.get("error") else "attempted")),
        )
    except Exception:
        logger.debug("Browser failure memory update failed", exc_info=True)


def _record_browser_success_from_result(name: str, args: dict, parsed: dict) -> None:
    """Persist a verified browser outcome as a reusable playbook signal."""
    if not isinstance(parsed, dict):
        return
    try:
        from remy.core.browser_failure_memory import record_browser_success

        evidence = parsed.get("evidence") if isinstance(parsed.get("evidence"), dict) else {}
        url = (
            parsed.get("url")
            or parsed.get("requested_url")
            or evidence.get("page_url")
            or evidence.get("requested_url")
            or args.get("url")
            or ""
        )
        text = (
            parsed.get("answer")
            or parsed.get("description")
            or evidence.get("page_text_snippet")
            or ""
        )
        record_browser_success(
            tool=name,
            action=str(parsed.get("action") or args.get("action") or ""),
            url=str(url),
            text=str(text),
            selector=str(
                parsed.get("selector") or evidence.get("selector") or args.get("selector") or ""
            ),
            status=str(parsed.get("status") or "verified"),
        )
    except Exception:
        logger.debug("Browser success playbook update failed", exc_info=True)


def _apply_execution_memory_policy(name: str, args: dict, parsed: dict) -> dict:
    """Annotate browser results with known domain-level blocker/playbook signals."""
    if not isinstance(parsed, dict):
        return parsed

    try:
        from remy.core.browser_failure_memory import get_browser_execution_hints
    except Exception:
        return parsed

    evidence = parsed.get("evidence") if isinstance(parsed.get("evidence"), dict) else {}
    url = str(
        parsed.get("url")
        or parsed.get("requested_url")
        or evidence.get("page_url")
        or evidence.get("requested_url")
        or args.get("url")
        or ""
    )
    action = str(parsed.get("action") or args.get("action") or "")
    text = " ".join(
        str(part or "")
        for part in (
            parsed.get("page_state"),
            parsed.get("answer"),
            parsed.get("description"),
            parsed.get("error"),
            parsed.get("verification", {}).get("reason")
            if isinstance(parsed.get("verification"), dict)
            else "",
            evidence.get("page_text_snippet"),
        )
    ).strip()
    if not url:
        return parsed

    hints = get_browser_execution_hints(url=url, text=text, action=action, limit=2)
    failure_hints = hints.get("failure_hints") or []
    success_hints = hints.get("success_hints") or []
    if not failure_hints and not success_hints:
        return parsed

    parsed["execution_memory"] = {
        "domain": hints.get("domain"),
        "flow": hints.get("flow"),
        "known_failures": [
            {
                "signature": rec.get("signature"),
                "count": rec.get("count", 0),
                "tool": rec.get("tool", ""),
                "action": rec.get("action", ""),
            }
            for rec in failure_hints
        ],
        "known_successes": [
            {
                "flow": rec.get("flow"),
                "count": rec.get("count", 0),
                "tool": rec.get("tool", ""),
                "action": rec.get("action", ""),
            }
            for rec in success_hints
        ],
        "avoided_selectors": hints.get("avoided_selectors") or [],
        "preferred_selectors": hints.get("preferred_selectors") or [],
    }

    selector = str(parsed.get("selector") or evidence.get("selector") or args.get("selector") or "")
    selector_alternatives = _suggest_selector_alternatives(
        selector,
        parsed["execution_memory"]["preferred_selectors"],
    )
    if selector_alternatives:
        parsed["selector_alternatives"] = selector_alternatives
        parsed["suggested_selector"] = selector_alternatives[0]

    if selector:
        for item in parsed["execution_memory"]["avoided_selectors"]:
            if item.get("selector") == selector and int(item.get("count", 0)) >= 2:
                parsed["retry_shaped"] = True
                parsed["retry_reason"] = "historical_selector_failure"
                parsed["repeated_selector"] = selector
                parsed["repeated_selector_count"] = int(item.get("count", 0))
                parsed["_agent_hint"] = (
                    f"Selector '{selector}' has already failed {item.get('count', 0)} times on {hints.get('domain')}. "
                    f"Avoid it unless a fresh page analysis proves the DOM changed. "
                    f"{'Try ' + selector_alternatives[0] + ' instead.' if selector_alternatives else 'Call browse_page for a fresh selector map.'}"
                )
                break

    hard_blockers = {
        "captcha": "captcha challenge",
        "email_verification": "email verification required",
        "phone_verification": "phone verification required",
        "kyc_verification": "kyc/manual verification required",
        "payment_block": "payment step required",
    }
    if parsed.get("verified") is False:
        for rec in failure_hints:
            signature = str(rec.get("signature") or "")
            count = int(rec.get("count", 0) or 0)
            if signature in hard_blockers and count >= 2:
                parsed["external_blocker_likely"] = True
                parsed["blocker_reason"] = hard_blockers[signature]
                parsed["blocker_count"] = count
                parsed["_agent_hint"] = (
                    f"Known blocker on {hints.get('domain')}: {signature} seen {count} times. "
                    "Do not keep retrying the same flow. Mark blocked_external unless the current page shows a new path forward."
                )
                break

    if parsed.get("verified") is not True and success_hints and "_agent_hint" not in parsed:
        best = success_hints[0]
        preferred = parsed["execution_memory"]["preferred_selectors"]
        selector_hint = ""
        if preferred:
            selector_hint = f" Prefer selectors like {preferred[0]['selector']}."
        parsed["_agent_hint"] = (
            f"Known reusable playbook on {hints.get('domain')}: "
            f"{best.get('flow', 'navigation')} x{best.get('count', 0)} via "
            f"{best.get('tool', 'browser')}{'/' + best.get('action') if best.get('action') else ''}.{selector_hint} "
            "Prefer that pattern before trying novel interactions."
        )

    return parsed


def _handle_browser_tool(name: str, args: dict, session_id: str | None, channel: str | None) -> str:
    """Dispatch browser tools. Runs OUTSIDE brain_lock (async I/O + vision API).

    Smart circuit breaker with two thresholds:
    - After ANALYZE_THRESHOLD (3) consecutive failures: returns error analysis with
      diagnosis, failed selectors, and instructions to store as failed-hypothesis
      and try a different approach. Agent gets ONE more chance to pivot.
    - After HARD_STOP_LIMIT (6) consecutive failures: hard stop. Agent had its
      chance to adapt and failed — force browser_close.
    browser_close is always allowed.

    Mutable state (_consecutive_browser_failures, _browser_error_history) lives
    in brain_tools.py so tests can patch it there.
    """
    # Access mutable state + patchable references via brain_tools module
    bt = _get_bt()
    run_async = bt._run_async  # tests patch brain_tools._run_async
    bt_settings = bt.settings  # tests patch brain_tools.settings

    if not bt_settings.BROWSER_ENABLED:
        return json.dumps(
            {"error": "Browser tools are disabled. Set BROWSER_ENABLED=True to enable."}
        )

    # Hard stop — agent had its chance to pivot after the analysis warning
    if name != "browser_close" and bt._consecutive_browser_failures >= _BROWSER_HARD_STOP_LIMIT:
        logger.warning(
            "Browser HARD circuit breaker after %d failures. Forcing stop.",
            bt._consecutive_browser_failures,
        )
        bt._consecutive_browser_failures = 0
        bt._browser_error_history.clear()
        return json.dumps(
            {
                "error": (
                    f"HARD STOP: {_BROWSER_HARD_STOP_LIMIT} consecutive browser failures. "
                    "You were already warned to change approach but kept failing. "
                    "Call browser_close NOW and tell the user what happened. "
                    "Do NOT make any more browser_act calls."
                )
            }
        )

    # Soft stop — analyze error pattern, instruct agent to pivot
    if name == "browser_act" and bt._consecutive_browser_failures >= _BROWSER_ANALYZE_THRESHOLD:
        analysis = _analyze_browser_errors(bt._browser_error_history)
        logger.info(
            "Browser smart circuit breaker: %d failures, sending error analysis to agent.",
            bt._consecutive_browser_failures,
        )
        return json.dumps({"error": analysis})

    # Human-in-the-loop: financial / registration URLs require user approval
    from remy.core.approval_queue import approval_queue, build_approval_description, needs_approval

    url = _get_effective_url(name, args, run_async)

    shaped_retry = _shape_repeated_retry(name, args, url, bt)
    if shaped_retry is not None:
        shaped_retry = _apply_execution_memory_policy(name, args, shaped_retry)
        _record_browser_failure_from_result(name, args, shaped_retry)
        return json.dumps(shaped_retry, ensure_ascii=False)

    if name != "browser_close" and needs_approval(name, args, url, channel=channel):
        description = build_approval_description(name, args, url)
        if name == "browse_page":
            action_fn = lambda: run_async(_handle_browse_page(args, session_id, channel))
        else:  # browser_act
            action_fn = lambda: run_async(_handle_browser_act(args, session_id, channel))
        return approval_queue.request_approval_sync(
            description,
            action_fn,
            tool_name=name,
            tool_args=args,
            url=url,
        )

    if name == "browse_page":
        result = run_async(_handle_browse_page(args, session_id, channel))
    elif name == "browser_act":
        result = run_async(_handle_browser_act(args, session_id, channel))
    elif name == "browser_close":
        # browser_close resets the failure state
        bt._consecutive_browser_failures = 0
        bt._browser_error_history.clear()
        bt._login_error_history.clear()
        return run_async(_handle_browser_close(args, session_id, channel))
    else:
        return json.dumps({"error": f"Unknown browser tool: {name}"})

    # Track failures for both navigation and interaction steps.
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            parsed = _apply_execution_memory_policy(name, args, parsed)
            parsed = _check_login_failure_escalation(parsed)
            result = json.dumps(parsed, ensure_ascii=False)
        if isinstance(parsed, dict) and "error" in parsed:
            error_msg = str(parsed["error"])
            _record_browser_failure_from_result(name, args, parsed)
            bt._consecutive_browser_failures += 1
            bt._browser_error_history.append(error_msg)
            if len(bt._browser_error_history) > 10:
                bt._browser_error_history = bt._browser_error_history[-10:]
            logger.debug("Browser consecutive failures: %d", bt._consecutive_browser_failures)
        elif isinstance(parsed, dict) and parsed.get("verified") is False:
            _record_browser_failure_from_result(name, args, parsed)
            bt._consecutive_browser_failures = 0
            bt._browser_error_history.clear()
        elif isinstance(parsed, dict) and parsed.get("verified") is True:
            _record_browser_success_from_result(name, args, parsed)
            bt._consecutive_browser_failures = 0
            bt._browser_error_history.clear()
        else:
            bt._consecutive_browser_failures = 0
            bt._browser_error_history.clear()
    except (json.JSONDecodeError, TypeError):
        bt._consecutive_browser_failures += 1
        bt._browser_error_history.append(f"Parse error on result: {str(result)[:200]}")

    return result


class _BrowserLoop:
    """Persistent event loop in a dedicated daemon thread for browser operations."""

    _instance: "_BrowserLoop | None" = None
    _lock = threading.Lock()

    def __init__(self):
        import warnings

        warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="browser-loop")
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @classmethod
    def get(cls) -> "_BrowserLoop":
        if cls._instance is None or cls._instance._loop.is_closed():
            with cls._lock:
                if cls._instance is None or cls._instance._loop.is_closed():
                    cls._instance = cls()
        return cls._instance

    def run(self, coro):
        """Submit a coroutine to the persistent loop and wait for the result."""
        import concurrent.futures

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        while True:
            try:
                return future.result(timeout=0.5)
            except concurrent.futures.TimeoutError:
                continue

    def shutdown(self):
        """Stop the persistent loop (for cleanup/tests)."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)


def _run_async(coro) -> str:
    """Run an async coroutine from sync context using a persistent browser loop."""
    return _BrowserLoop.get().run(coro)


async def _handle_browse_page(args: dict, session_id: str | None, channel: str | None) -> str:
    """Navigate to URL, screenshot, analyze with vision model + DOM extraction."""
    from remy.core.browser import BrowserManager, DailyLimitError, SSRFError
    from remy.core.browser_vision import analyze_screenshot
    from remy.core.pinchtab import PinchTabError, PinchTabManager

    url = args.get("url", "")
    question = args.get("question", "")

    if not url:
        return json.dumps({"error": "url is required"})

    try:
        if _should_try_pinchtab_browse():
            try:
                pinch = PinchTabManager.get()
                pinch_result = await pinch.browse_page(url=url, question=question)
                return json.dumps(_pinchtab_browse_result(pinch_result), ensure_ascii=False)
            except Exception as exc:
                if _browser_backend_mode() == "pinchtab":
                    logger.error("PinchTab browse_page failed: %s", exc)
                    return json.dumps({"error": f"Browse failed via PinchTab: {exc}"})
                logger.info("PinchTab browse_page failed, falling back to Playwright: %s", exc)

        mgr = BrowserManager.get()
        normalized_url = BrowserManager.normalize_url(url)
        screenshot_png = await mgr.navigate(normalized_url)
        page_url = await mgr.get_page_url()
        page_text = await mgr.get_page_text()

        dom_elements = []
        extract_elements = getattr(mgr, "extract_interactive_elements", None)
        if callable(extract_elements):
            extracted = extract_elements()
            if inspect.isawaitable(extracted):
                extracted = await extracted
            if isinstance(extracted, list):
                dom_elements = extracted
        screenshot_path = ""
        if screenshot_png:
            filename = mgr.save_screenshot(screenshot_png)
            screenshot_path = f"/api/browser_screenshots/{filename}"
            analysis = await analyze_screenshot(
                screenshot_png=screenshot_png,
                question=question,
                page_url=page_url,
                page_text=page_text,
            )
        else:
            analysis = {
                "answer": "",
                "description": "Screenshot unavailable; relying on DOM extraction and page text only.",
                "page_state": "unknown",
            }

        if dom_elements:
            analysis["dom_elements"] = dom_elements
            forms_from_dom = []
            inputs = [el for el in dom_elements if el["tag"] in ("input", "select", "textarea")]
            if inputs:
                forms_from_dom = [
                    {
                        "label": el.get("visible_label")
                        or el.get("placeholder")
                        or el.get("name")
                        or el.get("aria_label")
                        or el["tag"],
                        "selector": el["selector"],
                        "type": el.get("type") or el["tag"],
                    }
                    for el in inputs
                ]
            if forms_from_dom:
                analysis["dom_form_fields"] = forms_from_dom
        visible_error_text = _extract_visible_error_text(analysis)

        verification = _verify_browser_step(
            tool="browse_page",
            action=None,
            requested_url=normalized_url,
            page_url=page_url,
            page_text=page_text,
        )
        result_data = {
            "url": page_url,
            "requested_url": normalized_url,
            "screenshot": screenshot_path,
            "verification": _adjust_verification_with_analysis(
                verification,
                analysis,
                tool="browse_page",
                action=None,
            ),
            "evidence": _build_browser_evidence(
                tool="browse_page",
                requested_url=normalized_url,
                page_url=page_url,
                screenshot_path=screenshot_path,
                page_text=page_text,
            ),
            **analysis,
        }
        if visible_error_text:
            result_data["visible_error_text"] = visible_error_text
        result_data["verified"] = result_data["verification"]["verified"]
        result_data["status"] = result_data["verification"]["status"]

        page_state = analysis.get("page_state", "normal")
        if page_state == "captcha":
            result_data["_agent_hint"] = (
                "CAPTCHA detected. Attempt to solve it:\n"
                "1. CHECKBOX captcha ('I'm not a robot'): use browser_act click on the checkbox element.\n"
                "2. IMAGE captcha (select images, puzzles): take a screenshot, analyze with vision, "
                "then click the correct images/answers using browser_act.\n"
                "3. TEXT captcha: read the distorted text from the screenshot and type it.\n"
                "Try up to 3 attempts. If you cannot solve it after 3 tries, "
                "use request_guidance to ask the user for help."
            )
        blocking = analysis.get("blocking_overlay")
        if blocking and page_state not in ("captcha",):
            result_data["_agent_hint"] = (
                f"Blocking overlay detected: {blocking}. "
                "Try to dismiss it by clicking the accept/close/dismiss button. "
                "Look in the elements list for a dismiss button."
            )

        return json.dumps(result_data, ensure_ascii=False)

    except SSRFError as e:
        return json.dumps({"error": f"Blocked: {e}"})
    except DailyLimitError as e:
        return json.dumps({"error": str(e)})
    except PinchTabError as e:
        return json.dumps({"error": f"Browse failed via PinchTab: {e}"})
    except Exception as e:
        logger.error("browse_page failed: %s", e)
        return json.dumps({"error": f"Browse failed: {e}"})


async def _handle_browser_act(args: dict, session_id: str | None, channel: str | None) -> str:
    """Perform action on current page, screenshot, analyze."""
    from remy.core.browser import BrowserManager, DailyLimitError, SSRFError
    from remy.core.browser_vision import analyze_screenshot
    from remy.core.pinchtab import PinchTabError, PinchTabManager

    action = args.get("action", "")
    selector = args.get("selector")
    text = args.get("text")
    url = args.get("url")

    if not action:
        return json.dumps({"error": "action is required"})

    try:
        if _should_try_pinchtab_act(action, selector):
            try:
                pinch = PinchTabManager.get()
                pinch_result = await pinch.act(
                    action=action,
                    selector=selector,
                    text=text,
                    url=url,
                )
                return json.dumps(_pinchtab_act_result(action, selector, pinch_result), ensure_ascii=False)
            except Exception as exc:
                if _browser_backend_mode() == "pinchtab":
                    logger.error("PinchTab browser_act failed: %s", exc)
                    return json.dumps({"error": f"Action failed via PinchTab: {exc}"})
                logger.info("PinchTab browser_act failed, falling back to Playwright: %s", exc)

        mgr = BrowserManager.get()
        requested_url = BrowserManager.normalize_url(url) if url else ""
        screenshot_png = await mgr.act(
            action=action,
            selector=selector,
            text=text,
            url=requested_url or url,
        )
        page_url = await mgr.get_page_url()
        page_text = await mgr.get_page_text()

        screenshot_path = ""
        if screenshot_png:
            filename = mgr.save_screenshot(screenshot_png)
            screenshot_path = f"/api/browser_screenshots/{filename}"
        verification = _verify_browser_step(
            tool="browser_act",
            action=action,
            requested_url=requested_url,
            page_url=page_url,
            page_text=page_text,
        )

        # v2.4: Skip expensive vision analysis for non-navigating actions.
        # type/scroll/wait/select don't change page layout — saves ~2,000 tokens each.
        _SKIP_VISION_ACTIONS = {"type", "scroll_down", "scroll_up", "scroll", "wait", "select"}
        if action in _SKIP_VISION_ACTIONS:
            result_data = {
                "action": action,
                "url": page_url,
                "requested_url": requested_url,
                "screenshot": screenshot_path,
                "description": verification["reason"],
                "verification": verification,
                "verified": verification["verified"],
                "status": verification["status"],
                "evidence": _build_browser_evidence(
                    tool="browser_act",
                    action=action,
                    selector=selector,
                    requested_url=requested_url,
                    page_url=page_url,
                    screenshot_path=screenshot_path,
                    page_text=page_text,
                ),
                "page_text_snippet": (page_text or "")[:500],
            }
            return json.dumps(result_data, ensure_ascii=False)

        if action == "click":
            question = (
                "Result after clicking. Check: did the page show any validation errors, "
                "error messages, or required field warnings? If so, list them in the answer field."
            )
        else:
            question = f"Result after action: {action}"

        if screenshot_png:
            analysis = await analyze_screenshot(
                screenshot_png=screenshot_png,
                question=question,
                page_url=page_url,
                page_text=page_text,
            )
        else:
            analysis = {
                "answer": "",
                "description": "Screenshot unavailable after action; relying on page text only.",
                "page_state": "unknown",
            }
        visible_error_text = _extract_visible_error_text(analysis)

        result_data = {
            "action": action,
            "url": page_url,
            "requested_url": requested_url,
            "screenshot": screenshot_path,
            "verification": _adjust_verification_with_analysis(
                verification,
                analysis,
                tool="browser_act",
                action=action,
            ),
            "evidence": _build_browser_evidence(
                tool="browser_act",
                action=action,
                selector=selector,
                requested_url=requested_url,
                page_url=page_url,
                screenshot_path=screenshot_path,
                page_text=page_text,
            ),
            **analysis,
        }
        if visible_error_text:
            result_data["visible_error_text"] = visible_error_text
        result_data["verified"] = result_data["verification"]["verified"]
        result_data["status"] = result_data["verification"]["status"]

        page_state = analysis.get("page_state", "normal")
        if page_state == "error":
            result_data["_agent_hint"] = (
                "The page shows validation errors. Read the error messages in the 'answer' field, "
                "fix the field values, and try submitting again."
            )

        # Draft enforcement: detect if content was accidentally published live
        if action == "click":
            result_data = _check_accidental_publish(result_data, page_url, page_text)

        return json.dumps(result_data, ensure_ascii=False)

    except SSRFError as e:
        return json.dumps({"error": f"Blocked: {e}"})
    except DailyLimitError as e:
        return json.dumps({"error": str(e)})
    except PinchTabError as e:
        return json.dumps({"error": f"Action failed via PinchTab: {e}"})
    except ValueError as e:
        return json.dumps(
            _structured_browser_value_error(
                action=action,
                selector=selector,
                requested_url=requested_url or url or "",
                error_text=str(e),
            ),
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("browser_act failed: %s", e)
        return json.dumps({"error": f"Action failed: {e}"})


async def _handle_browser_close(args: dict, session_id: str | None, channel: str | None) -> str:
    """Close browser and free resources."""
    from remy.core.browser import BrowserManager
    from remy.core.pinchtab import PinchTabManager

    try:
        mgr = BrowserManager.get()
        await mgr.close()
        if _pinchtab_enabled():
            await PinchTabManager.get().close()
        return json.dumps({"closed": True, "message": "Browser closed successfully."})
    except Exception as e:
        logger.error("browser_close failed: %s", e)
        return json.dumps({"error": f"Close failed: {e}"})
