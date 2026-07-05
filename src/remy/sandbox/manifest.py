"""
Sandbox Manifest — JSON-backed registry for agent-created tools.

Status flow: draft → tested → pending → approved | rejected

Manifest file: data/sandbox/manifest.json
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Valid status transitions
VALID_TRANSITIONS = {
    "draft": {"tested", "rejected"},
    "tested": {"pending", "approved", "rejected"},
    "pending": {"approved", "rejected"},
    "approved": {"rejected"},
    "rejected": {"draft"},
}


class SandboxManifest:
    """JSON registry for sandbox tools."""

    def __init__(self, manifest_path: Path):
        self.path = Path(manifest_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load manifest: %s", e)
        return {"version": 1, "tools": []}

    def save(self):
        from remy.core.file_utils import atomic_write
        atomic_write(self.path, json.dumps(self._data, indent=2, ensure_ascii=False))

    @property
    def tools(self) -> list[dict]:
        return self._data["tools"]

    def get_tool(self, name: str) -> Optional[dict]:
        for t in self.tools:
            if t["name"] == name:
                return t
        return None

    def add_tool(self, name: str, file: str, description: str,
                 parameters: dict, required: list[str],
                 dependencies: list[str] | None = None) -> dict:
        existing = self.get_tool(name)
        if existing:
            existing.update({
                "file": file,
                "description": description,
                "parameters": parameters,
                "required": required,
                "dependencies": dependencies or [],
                "status": "draft",
                "test_result": None,
            })
            self.save()
            return existing

        entry = {
            "name": name,
            "file": file,
            "description": description,
            "parameters": parameters,
            "required": required,
            "dependencies": dependencies or [],
            "status": "draft",
            "test_result": None,
        }
        self.tools.append(entry)
        self.save()
        return entry

    def update_status(self, name: str, new_status: str) -> bool:
        tool = self.get_tool(name)
        if not tool:
            return False
        current = tool["status"]
        if new_status not in VALID_TRANSITIONS.get(current, set()):
            logger.warning("Invalid transition: %s → %s for %s", current, new_status, name)
            return False
        tool["status"] = new_status
        self.save()
        return True

    def set_test_result(self, name: str, passed: int, failed: int, output: str = ""):
        tool = self.get_tool(name)
        if tool:
            tool["test_result"] = {"passed": passed, "failed": failed, "output": output[:500]}
            if failed == 0 and passed > 0:
                tool["status"] = "tested"
            self.save()

    def record_telemetry(self, name: str, success: bool, duration_ms: int) -> bool:
        """Record execution telemetry for a sandbox tool."""
        tool = self.get_tool(name)
        if not tool:
            return False

        telemetry = tool.setdefault("telemetry", {})
        telemetry["call_count"] = int(telemetry.get("call_count", 0)) + 1
        telemetry["error_count"] = int(telemetry.get("error_count", 0))
        telemetry["total_ms"] = int(telemetry.get("total_ms", 0)) + max(0, int(duration_ms))

        if not success:
            telemetry["error_count"] += 1

        self.save()
        return True

    def get_approved_tools(self) -> list[dict]:
        return [t for t in self.tools if t["status"] == "approved"]

    def get_pending_tools(self) -> list[dict]:
        return [t for t in self.tools if t["status"] == "pending"]

    def submit_for_approval(self, name: str) -> bool:
        tool = self.get_tool(name)
        if not tool:
            return False
        if tool["status"] != "tested":
            return False
        tool["status"] = "pending"
        self.save()
        return True

    def auto_approve_tested(self) -> list[str]:
        """Auto-approve all tested tools with passing tests. Returns approved names.

        Only used in autonomous mode with AUTONOMY_AUTO_APPROVE_SANDBOX=True.
        """
        approved = []
        for tool in self.tools:
            if tool["status"] == "tested":
                test = tool.get("test_result", {})
                if test and test.get("passed", 0) > 0 and test.get("failed", 0) == 0:
                    tool["status"] = "approved"
                    approved.append(tool["name"])
        if approved:
            self.save()
        return approved

    def summary(self) -> list[dict]:
        return [
            {"name": t["name"], "status": t["status"],
             "description": t["description"][:80],
             "test_result": t.get("test_result"),
             **({"telemetry": t["telemetry"]} if "telemetry" in t else {})}
            for t in self.tools
        ]
