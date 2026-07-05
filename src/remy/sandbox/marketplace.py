"""
Skill Marketplace — GitHub-based discovery and installation.

Skills are hosted in a GitHub repo with structure:
    skills/<name>/tool.py + skill.json
    index.json  — auto-generated catalog

Usage:
    browse_marketplace() → list of available skills
    install_from_marketplace("crypto_tracker") → import + register
"""

import json
import logging
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from remy.config.settings import settings

logger = logging.getLogger(__name__)

# Default marketplace repo (can be overridden in settings)
DEFAULT_MARKETPLACE_URL = getattr(
    settings,
    "SKILL_MARKETPLACE_URL",
    "https://raw.githubusercontent.com/teolex2020/remy-skills/main",
)
INDEX_FILE = "index.json"
FETCH_TIMEOUT_SEC = 15
MAX_DOWNLOAD_BYTES = 512 * 1024  # 512KB


def _fetch_url(url: str, timeout: int = FETCH_TIMEOUT_SEC) -> bytes:
    """Fetch URL content with timeout and size limit."""
    req = Request(url, headers={"User-Agent": "Remy-Agent/2.3"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"Download too large (>{MAX_DOWNLOAD_BYTES} bytes)")
        return data


def browse_marketplace(marketplace_url: str | None = None) -> list[dict]:
    """Fetch and return the marketplace index.

    Returns list of dicts with: name, version, description, author, tags, downloads.
    Returns empty list on network errors (graceful degradation).
    """
    base_url = marketplace_url or DEFAULT_MARKETPLACE_URL
    index_url = f"{base_url}/{INDEX_FILE}"

    try:
        data = _fetch_url(index_url)
        index = json.loads(data.decode("utf-8"))
        if not isinstance(index, dict) or "skills" not in index:
            logger.warning("Invalid marketplace index format")
            return []
        return index["skills"]
    except (URLError, json.JSONDecodeError, ValueError, OSError) as e:
        logger.warning("Failed to fetch marketplace index: %s", e)
        return []


def install_from_marketplace(
    skill_name: str,
    marketplace_url: str | None = None,
) -> dict:
    """Download and install a skill from the marketplace.

    Flow:
    1. Fetch skill.json from marketplace
    2. Fetch tool.py from marketplace
    3. Validate structure
    4. Register in manifest (status=draft)
    5. User must test + approve

    Returns dict with: installed, name, status, message
    """
    from remy.sandbox.runner import install_dependencies, validate_tool_file

    base_url = marketplace_url or DEFAULT_MARKETPLACE_URL
    skill_base = f"{base_url}/skills/{skill_name}"

    # 1. Fetch skill.json
    try:
        meta_data = _fetch_url(f"{skill_base}/skill.json")
        skill_meta = json.loads(meta_data.decode("utf-8"))
    except (URLError, json.JSONDecodeError, ValueError, OSError) as e:
        return {"installed": False, "error": f"Failed to fetch skill metadata: {e}"}

    # Validate required fields
    for key in ("name", "version", "description"):
        if key not in skill_meta:
            return {"installed": False, "error": f"skill.json missing required field: {key}"}

    name = skill_meta["name"]
    description = skill_meta["description"]
    dependencies = skill_meta.get("dependencies", [])

    # 2. Fetch tool.py
    try:
        tool_data = _fetch_url(f"{skill_base}/tool.py")
        tool_code = tool_data.decode("utf-8")
    except (URLError, ValueError, OSError) as e:
        return {"installed": False, "error": f"Failed to fetch tool code: {e}"}

    # 3. Write to sandbox tools dir
    tools_dir = Path(settings.SANDBOX_TOOLS_DIR)
    tools_dir.mkdir(parents=True, exist_ok=True)
    tool_file = tools_dir / f"{name}.py"

    if tool_file.exists():
        return {
            "installed": False,
            "error": f"Tool '{name}' already exists locally. Remove it first.",
        }

    tool_file.write_text(tool_code, encoding="utf-8")

    # 4. Validate via AST
    valid, msg = validate_tool_file(tool_file)
    if not valid:
        tool_file.unlink()
        return {"installed": False, "error": f"Tool validation failed: {msg}"}

    # 5. Install dependencies
    if dependencies:
        ok, dep_msg = install_dependencies(dependencies)
        if not ok:
            tool_file.unlink()
            return {"installed": False, "error": f"Dependency install failed: {dep_msg}"}

    # 6. Register in manifest
    from remy.core.tool_registry_mgmt import get_registry

    manifest = get_registry().manifest
    tool_entry = manifest.add_tool(
        name=name,
        file=f"{name}.py",
        description=description,
        parameters=skill_meta.get("parameters", {}),
        required=skill_meta.get("required", []),
        dependencies=dependencies,
    )

    # Store marketplace metadata
    tool_entry["version"] = skill_meta.get("version", "1.0.0")
    tool_entry["author"] = skill_meta.get("author", "unknown")
    tool_entry["tags"] = skill_meta.get("tags", [])
    tool_entry["marketplace_source"] = base_url
    manifest.save()

    # Trust classification
    try:
        from remy.core.tool_trust import classify_tool_source, format_trust_report

        classification = classify_tool_source(tool_code)
        tool_entry["trust_level"] = classification.trust_level
        tool_entry["trust_report"] = format_trust_report(classification)
        manifest.save()
    except Exception as e:
        logger.debug("Trust classification skipped: %s", e)

    logger.info(
        "Installed skill '%s' v%s from marketplace (%s)",
        name,
        skill_meta.get("version", "?"),
        base_url,
    )

    return {
        "installed": True,
        "name": name,
        "version": skill_meta.get("version", "1.0.0"),
        "author": skill_meta.get("author", "unknown"),
        "description": description,
        "status": "draft",
        "trust_level": tool_entry.get("trust_level", "unknown"),
        "message": (
            f"Skill '{name}' installed from marketplace. "
            f"Run sandbox_test_tool('{name}') to test, then approve."
        ),
    }
