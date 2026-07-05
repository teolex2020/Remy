"""
Skill Package — portable format for sharing sandbox tools.

Export: skill_package.tar.gz containing tool.py + skill.json
Import: validate → register in manifest → user approves

Usage:
    export_skill("my_tool") → Path to .tar.gz
    import_skill("/path/to/my_tool.skill.tar.gz") → dict with status
"""

import json
import logging
import tarfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

from remy.config.settings import settings

logger = logging.getLogger(__name__)

SKILL_JSON_SCHEMA_KEYS = {"name", "version", "description"}
SKILL_JSON_OPTIONAL_KEYS = {"author", "dependencies", "tags", "min_remy_version", "license"}
SKILL_ARCHIVE_SUFFIX = ".skill.tar.gz"
MAX_SKILL_SIZE_BYTES = 512 * 1024  # 512KB max archive size


def _get_skill_meta_from_manifest(tool_entry: dict) -> dict:
    """Build skill.json content from manifest entry."""
    return {
        "name": tool_entry["name"],
        "version": tool_entry.get("version", "1.0.0"),
        "description": tool_entry["description"],
        "author": tool_entry.get("author", "unknown"),
        "dependencies": tool_entry.get("dependencies", []),
        "tags": tool_entry.get("tags", []),
        "min_remy_version": "2.3.0",
        "exported_at": datetime.now().isoformat(),
    }


def export_skill(name: str, output_dir: Path | None = None) -> Path:
    """Export an approved sandbox tool as a .skill.tar.gz package.

    Args:
        name: Tool name in the manifest.
        output_dir: Directory to write the archive. Defaults to data/sandbox/exports/.

    Returns:
        Path to the created archive.

    Raises:
        ValueError: If tool not found or not approved.
        FileNotFoundError: If tool file missing.
    """
    from remy.core.tool_registry_mgmt import get_registry

    manifest = get_registry().manifest
    tool = manifest.get_tool(name)
    if not tool:
        raise ValueError(f"Tool '{name}' not found in manifest")
    if tool["status"] not in ("approved", "tested"):
        raise ValueError(
            f"Tool '{name}' must be approved or tested to export (current: {tool['status']})"
        )

    tool_path = Path(settings.SANDBOX_TOOLS_DIR) / tool["file"]
    if not tool_path.exists():
        raise FileNotFoundError(f"Tool file missing: {tool_path}")

    # Build skill.json
    skill_meta = _get_skill_meta_from_manifest(tool)

    # Output directory
    if output_dir is None:
        output_dir = settings.SANDBOX_DIR / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"{name}{SKILL_ARCHIVE_SUFFIX}"
    archive_path = output_dir / archive_name

    with tarfile.open(archive_path, "w:gz") as tar:
        # Add tool.py
        tar.add(tool_path, arcname="tool.py")

        # Add skill.json
        skill_json_bytes = json.dumps(skill_meta, indent=2, ensure_ascii=False).encode("utf-8")
        info = tarfile.TarInfo(name="skill.json")
        info.size = len(skill_json_bytes)
        tar.addfile(info, BytesIO(skill_json_bytes))

    logger.info(
        "Exported skill '%s' to %s (%d bytes)", name, archive_path, archive_path.stat().st_size
    )
    return archive_path


def _validate_skill_archive(archive_path: Path) -> tuple[dict, str]:
    """Validate a skill archive. Returns (skill_meta, tool_code) or raises ValueError."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if archive_path.stat().st_size > MAX_SKILL_SIZE_BYTES:
        raise ValueError(
            f"Archive too large: {archive_path.stat().st_size} bytes (max {MAX_SKILL_SIZE_BYTES})"
        )

    with tarfile.open(archive_path, "r:gz") as tar:
        # Security: check for path traversal
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                raise ValueError(f"Unsafe path in archive: {member.name}")
            if member.name not in ("tool.py", "skill.json"):
                raise ValueError(f"Unexpected file in archive: {member.name}")

        # Extract skill.json
        try:
            skill_json_file = tar.extractfile("skill.json")
            if skill_json_file is None:
                raise ValueError("skill.json not found in archive")
            skill_meta = json.loads(skill_json_file.read().decode("utf-8"))
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid skill.json: {e}") from e

        # Validate required fields
        missing = SKILL_JSON_SCHEMA_KEYS - set(skill_meta.keys())
        if missing:
            raise ValueError(f"skill.json missing required fields: {missing}")

        # Extract tool.py
        try:
            tool_py_file = tar.extractfile("tool.py")
            if tool_py_file is None:
                raise ValueError("tool.py not found in archive")
            tool_code = tool_py_file.read().decode("utf-8")
        except KeyError as e:
            raise ValueError(f"tool.py not found in archive: {e}") from e

    return skill_meta, tool_code


def import_skill(archive_path: str | Path) -> dict:
    """Import a skill from a .skill.tar.gz archive.

    The tool is registered in the manifest as 'tested' status.
    User must approve it before it becomes active.

    Args:
        archive_path: Path to the .skill.tar.gz file.

    Returns:
        dict with keys: imported, name, status, description, message
    """
    from remy.sandbox.runner import validate_tool_file

    archive_path = Path(archive_path)

    # Validate archive structure
    try:
        skill_meta, tool_code = _validate_skill_archive(archive_path)
    except (ValueError, FileNotFoundError) as e:
        return {"imported": False, "error": str(e)}

    name = skill_meta["name"]
    description = skill_meta["description"]
    dependencies = skill_meta.get("dependencies", [])

    # Write tool file
    tools_dir = Path(settings.SANDBOX_TOOLS_DIR)
    tools_dir.mkdir(parents=True, exist_ok=True)
    tool_file = tools_dir / f"{name}.py"

    # Don't overwrite existing tools without explicit intent
    if tool_file.exists():
        return {"imported": False, "error": f"Tool '{name}' already exists. Remove it first."}

    tool_file.write_text(tool_code, encoding="utf-8")

    # Validate tool structure via AST
    valid, msg = validate_tool_file(tool_file)
    if not valid:
        tool_file.unlink()
        return {"imported": False, "error": f"Tool validation failed: {msg}"}

    # Install dependencies if any
    if dependencies:
        from remy.sandbox.runner import install_dependencies

        ok, dep_msg = install_dependencies(dependencies)
        if not ok:
            tool_file.unlink()
            return {"imported": False, "error": f"Dependency install failed: {dep_msg}"}

    # Register in manifest
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

    # Store extra metadata from skill.json
    tool_entry["version"] = skill_meta.get("version", "1.0.0")
    tool_entry["author"] = skill_meta.get("author", "unknown")
    tool_entry["tags"] = skill_meta.get("tags", [])
    tool_entry["imported_from"] = str(archive_path.name)
    tool_entry["imported_at"] = datetime.now().isoformat()
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
        "Imported skill '%s' v%s from %s", name, skill_meta.get("version", "?"), archive_path.name
    )

    return {
        "imported": True,
        "name": name,
        "version": skill_meta.get("version", "1.0.0"),
        "author": skill_meta.get("author", "unknown"),
        "description": description,
        "status": "draft",
        "trust_level": tool_entry.get("trust_level", "unknown"),
        "message": f"Skill '{name}' imported. Run sandbox_test_tool('{name}') to test, then approve.",
    }


def list_exportable_skills() -> list[dict]:
    """List all skills that can be exported (approved or tested)."""
    from remy.core.tool_registry_mgmt import get_registry

    manifest = get_registry().manifest
    return [
        {
            "name": t["name"],
            "status": t["status"],
            "description": t["description"][:80],
            "version": t.get("version", "1.0.0"),
            "author": t.get("author", "unknown"),
        }
        for t in manifest.tools
        if t["status"] in ("approved", "tested")
    ]
