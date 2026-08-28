"""Structured model-facing contracts for tools with ambiguous neighbours.

Schemas constrain arguments. These contracts constrain selection: they tell a
model when to choose a tool, where to route instead, and which boundaries are
hard. Execution policy remains authoritative and independent of this text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


REQUIRED_CONTRACT_SECTIONS = (
    "WHEN TO USE:",
    "WHEN NOT TO USE:",
    "DO NOT USE FOR:",
    "USAGE:",
    "EXAMPLES:",
)


@dataclass(frozen=True, slots=True)
class ToolContract:
    name: str
    summary: str
    returns: str
    when_to_use: tuple[str, ...]
    when_not_to_use: tuple[str, ...]
    do_not_use_for: tuple[str, ...]
    usage: tuple[str, ...]
    examples: tuple[str, ...]

    def render(self) -> str:
        def section(title: str, values: tuple[str, ...]) -> str:
            return title + "\n- " + "\n- ".join(values)

        return "\n\n".join(
            (
                f"{self.summary} Returns {self.returns}",
                section("WHEN TO USE:", self.when_to_use),
                section("WHEN NOT TO USE:", self.when_not_to_use),
                section("DO NOT USE FOR:", self.do_not_use_for),
                section("USAGE:", self.usage),
                section("EXAMPLES:", self.examples),
            )
        )

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        for field in (
            "name",
            "summary",
            "returns",
            "when_to_use",
            "when_not_to_use",
            "do_not_use_for",
            "usage",
            "examples",
        ):
            if not getattr(self, field):
                errors.append(f"{self.name or '<unnamed>'}: {field} is empty")
        if len(self.when_to_use) < 2:
            errors.append(f"{self.name}: WHEN TO USE needs at least two scenarios")
        if len(self.examples) < 2:
            errors.append(f"{self.name}: EXAMPLES needs at least two examples")
        rendered = self.render()
        for heading in REQUIRED_CONTRACT_SECTIONS:
            if heading not in rendered:
                errors.append(f"{self.name}: missing {heading}")
        return tuple(errors)


def _contract(
    name: str,
    summary: str,
    returns: str,
    when: tuple[str, ...],
    when_not: tuple[str, ...],
    never: tuple[str, ...],
    usage: tuple[str, ...],
    examples: tuple[str, ...],
) -> ToolContract:
    return ToolContract(name, summary, returns, when, when_not, never, usage, examples)


TOOL_CONTRACTS: dict[str, ToolContract] = {
    "recall": _contract(
        "recall",
        "Retrieve semantically relevant information from Remy's project memory and knowledge base.",
        "ranked remembered records and knowledge context, not live web evidence.",
        ("The user refers to prior work, decisions, preferences, people, or saved facts.", "Project context may already answer the question before external research."),
        ("Use search for explicit record/tag lookup.", "Use web_search only for current or missing external information after memory is checked."),
        ("Do not use for live news, prices, schedules, or arbitrary internet discovery.", "Do not store or modify memory."),
        ("Pass a focused semantic query; use a bounded token_budget when only a short preamble is needed.",),
        ("Recall what we decided about the release pipeline.", "Recall the user's preferred report format."),
    ),
    "search": _contract(
        "search",
        "Search Remy's stored records using a query and/or tags.",
        "matching local memory records, not internet results.",
        ("The user asks for saved records with known words or tags.", "You need a narrower local lookup than semantic recall."),
        ("Use recall for broad semantic project context.", "Use web_search for external sources."),
        ("Do not present results as current web evidence.", "Do not use for searching workspace file contents; use fs_search."),
        ("Provide query, comma-separated tags, or both.",),
        ("Find saved records tagged client,meeting.", "Search memory for invoice 293."),
    ),
    "web_search": _contract(
        "web_search",
        "Discover candidate web sources for a topic.",
        "titles, URLs, snippets, and ranking metadata; discovered candidates are not verified evidence.",
        ("The user needs current external information or unknown source URLs.", "You must discover several independent candidate sources before evidence collection."),
        ("Use extract_content when a specific human-readable URL is already known.", "Use http_get for a known API/JSON URL; use browse_page for interaction or JavaScript-only content."),
        ("Do not cite snippets as verified factual evidence.", "Do not repeatedly rephrase the same search instead of fetching a selected source."),
        ("Use a focused query in the most relevant language; then fetch selected URLs. Prefer at least three independent relevant sources for substantive research.",),
        ("Find current PostgreSQL 18 release documentation.", "Discover independent sources comparing local AI agent sandboxes."),
    ),
    "extract_content": _contract(
        "extract_content",
        "Fetch and clean the readable content of a known web page.",
        "article text, metadata, links/tables when requested, and an evidence identity packet.",
        ("A specific article, documentation, paper, or product-page URL is known.", "Static human-readable content must be grounded before making claims."),
        ("Use web_search when no source URL is known.", "Use http_get for APIs/JSON; use browse_page for login, forms, interaction, or JavaScript-only pages."),
        ("Do not use search-result snippets as a substitute for fetched content.", "Do not interact with page controls or submit forms."),
        ("Pass the exact URL. Add expected_title or expected_identifier for identity-sensitive evidence; force_refresh only for time-sensitive content.",),
        ("Extract the Vercel course page at this URL.", "Fetch an arXiv page and verify the expected paper identifier."),
    ),
    "http_get": _contract(
        "http_get",
        "Fetch a known HTTP endpoint directly.",
        "the endpoint response suitable for APIs, JSON, feeds, or machine-readable resources.",
        ("A concrete API, JSON, RSS, or machine-readable URL is already known.", "A lightweight direct request is preferable to launching a browser."),
        ("Use web_search to discover URLs.", "Use extract_content for readable articles/docs; use browse_page for interaction or browser-rendered content."),
        ("Do not use for authentication flows, form submission, or visual interaction.", "Do not treat an error page as source evidence."),
        ("Pass one exact URL and inspect the returned status/content before relying on it.",),
        ("Fetch a known GitHub API endpoint returning JSON.", "Read a public RSS feed from its direct URL."),
    ),
    "browse_page": _contract(
        "browse_page",
        "Open a known page in the real browser and inspect rendered content and controls.",
        "verified page text, page state, interactive elements, and forms.",
        ("The site requires JavaScript rendering, navigation, login, or form interaction.", "Visual or interactive page state matters to the task."),
        ("Use extract_content for static articles and documentation.", "Use http_get for APIs/JSON; use web_search when the URL is unknown."),
        ("Do not infer factual page content from the URL or screenshot description alone.", "Do not use a browser merely to read a simple static page."),
        ("Pass a full URL and an optional focused question. Ground claims only in returned page_text.",),
        ("Open the signed-in dashboard and identify its visible controls.", "Inspect a JavaScript-rendered pricing table."),
    ),
    "fs_read": _contract(
        "fs_read",
        "Read a known file inside a user-approved local workspace.",
        "bounded file content with paging metadata; binary data may be encoded.",
        ("The exact workspace file path is known.", "You need a bounded line range from a text file."),
        ("Use fs_search when the file or matching location is unknown.", "Use list_directory to inspect directory entries; use fs_write to change content."),
        ("Do not execute files or commands.", "Do not access paths outside approved workspace permissions."),
        ("Use workspace:// paths when possible; page large files with offset and limit.",),
        ("Read workspace://project/pyproject.toml.", "Read lines 500-700 from a known log file."),
    ),
    "fs_search": _contract(
        "fs_search",
        "Search file names or file contents inside a Read-approved workspace.",
        "bounded matching paths and optional line snippets.",
        ("You need to locate files by glob.", "You need to find a regex or symbol across multiple files."),
        ("Use fs_read when the exact file is already known.", "Use list_directory for a simple directory listing; use web_search for internet sources."),
        ("Do not modify files or execute commands.", "Do not search outside approved workspace boundaries."),
        ("Choose glob for names or grep for content; set max_results and a narrow path when possible.",),
        ("Find every Python file under src.", "Search the workspace for calls to prepareContainerRuntime."),
    ),
    "fs_write": _contract(
        "fs_write",
        "Write or append content inside a Write-approved local workspace.",
        "an audited write receipt.",
        ("The user requested a concrete file creation or modification.", "Generated content must be persisted to an approved workspace."),
        ("Use fs_read to inspect existing content first when overwriting could lose work.", "Use shell_exec only when an actual command/build/test is required."),
        ("Do not write outside an approved workspace.", "Do not use to simulate command execution or bypass approval policy."),
        ("Pass path, content, and write/append mode explicitly. Prefer minimal scoped edits.",),
        ("Create Documents/report.md with the generated report.", "Append one audited line to an approved log file."),
    ),
    "shell_exec": _contract(
        "shell_exec",
        "Execute a shell command in an Execute-approved local workspace after required approval.",
        "bounded stdout, stderr, exit status, and an audit receipt.",
        ("A build, test, formatter, package command, or other real process must run.", "No narrower read, search, or write tool can perform the operation."),
        ("Use fs_read for reading files, fs_search for locating content, and fs_write for direct edits.", "Use Agent Lab container execution when untrusted generated code requires a hard boundary."),
        ("Do not bypass workspace permissions, approval gates, or sandbox policy.", "Do not use shell commands merely to read, search, or write when a dedicated tool exists."),
        ("Provide a bounded timeout and an Execute-approved working_dir. Treat non-zero exit status as failure evidence.",),
        ("Run the approved project's test suite.", "Execute the formatter in the selected workspace."),
    ),
    "sandbox_create_tool": _contract(
        "sandbox_create_tool",
        "Create a reusable sandbox tool proposal for a recurring capability gap.",
        "a validated draft registration that still requires testing and approval.",
        ("The same missing operation will be needed repeatedly.", "A narrow reusable tool is safer and clearer than repeated general shell commands."),
        ("Use existing tools for one-off work.", "Use Agent Lab for building and testing a larger application or artifact."),
        ("Do not create a tool to bypass permissions, approvals, or network policy.", "Do not claim the new tool is verified or approved before independent tests pass."),
        ("Provide minimal dependency-free code, a precise schema, explicit limits, and no embedded secrets.",),
        ("Create a reusable parser for this recurring local file format.", "Create a bounded calculator tool used by several workflows."),
    ),
}


def validate_tool_contracts(contracts: Mapping[str, ToolContract] = TOOL_CONTRACTS) -> tuple[str, ...]:
    errors: list[str] = []
    for name, contract in contracts.items():
        if name != contract.name:
            errors.append(f"{name}: registry key differs from contract name {contract.name}")
        errors.extend(contract.validate())
    return tuple(errors)


def apply_tool_contracts(declarations: Iterable[Any]) -> list[Any]:
    """Return declarations with registered descriptions replaced by v2 contracts."""
    compiled: list[Any] = []
    for declaration in declarations:
        contract = TOOL_CONTRACTS.get(str(getattr(declaration, "name", "")))
        if contract is None:
            compiled.append(declaration)
            continue
        description = contract.render()
        if hasattr(declaration, "model_copy"):
            compiled.append(declaration.model_copy(update={"description": description}))
        elif hasattr(declaration, "copy"):
            compiled.append(declaration.copy(update={"description": description}))
        else:
            declaration.description = description
            compiled.append(declaration)
    return compiled


@dataclass(frozen=True, slots=True)
class RoutingCase:
    case_id: str
    prompt: str
    expected_tool: str
    forbidden_tools: tuple[str, ...] = ()


ROUTING_BENCHMARK_CASES: tuple[RoutingCase, ...] = (
    RoutingCase("memory-context", "What did we decide about the release pipeline?", "recall", ("web_search",)),
    RoutingCase("memory-tag", "Find my saved records tagged client,meeting.", "search", ("web_search", "fs_search")),
    RoutingCase("web-discovery", "Find three current independent sources about local AI sandboxes.", "web_search", ("browse_page", "shell_exec")),
    RoutingCase("known-article", "Read and extract the article at https://example.com/post.", "extract_content", ("web_search", "browse_page")),
    RoutingCase("known-api", "Fetch JSON from https://api.example.com/status.", "http_get", ("browse_page", "web_search")),
    RoutingCase("interactive-web", "Open the dashboard and inspect the sign-in form.", "browse_page", ("extract_content", "http_get")),
    RoutingCase("known-file", "Read workspace://project/pyproject.toml.", "fs_read", ("shell_exec", "fs_search")),
    RoutingCase("file-search", "Find all references to prepareContainerRuntime in the workspace.", "fs_search", ("shell_exec", "web_search")),
    RoutingCase("file-write", "Write this report to Documents/report.md.", "fs_write", ("shell_exec",)),
    RoutingCase("real-command", "Run the approved project test suite.", "shell_exec", ("fs_read", "fs_search")),
    RoutingCase("reusable-tool", "Create a reusable parser tool for this recurring file format.", "sandbox_create_tool", ("shell_exec",)),
)


def run_tool_routing_benchmark(
    selector: Callable[[RoutingCase], str],
    cases: Iterable[RoutingCase] = ROUTING_BENCHMARK_CASES,
) -> dict[str, Any]:
    """Evaluate any model/router adapter against stable tool-selection cases."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            selected = str(selector(case) or "")
            error = ""
        except Exception as exc:
            selected = ""
            error = str(exc)[:240]
        correct = selected == case.expected_tool
        unsafe = selected in case.forbidden_tools
        rows.append(
            {
                "case_id": case.case_id,
                "expected_tool": case.expected_tool,
                "selected_tool": selected,
                "correct": correct,
                "forbidden_selection": unsafe,
                "error": error,
            }
        )
    total = len(rows)
    correct_count = sum(1 for row in rows if row["correct"])
    forbidden_count = sum(1 for row in rows if row["forbidden_selection"])
    return {
        "total": total,
        "correct": correct_count,
        "accuracy": round(correct_count / total, 4) if total else 0.0,
        "forbidden_selections": forbidden_count,
        "safe_routing_rate": round((total - forbidden_count) / total, 4) if total else 0.0,
        "passed": bool(total and correct_count == total and forbidden_count == 0),
        "cases": rows,
    }
