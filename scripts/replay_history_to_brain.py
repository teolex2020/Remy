from __future__ import annotations

import argparse
import json
from pathlib import Path

from remy.config.settings import settings
from remy.core.agent_tools import brain
from remy.core.history_replay import replay_history
from remy.core.tool_dispatch import execute_tool


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay safe memory-writing tool calls from session history into the current Aura brain.")
    parser.add_argument("--history-dir", default=str(settings.DATA_DIR / "history"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stats = replay_history(
        lambda tool, tool_args: execute_tool(tool, tool_args, session_id="history-replay", channel="desktop"),
        history_dir=Path(args.history_dir),
        dry_run=args.dry_run,
        force=args.force,
        count_records_fn=lambda: len(brain.search(query="", limit=20)),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
