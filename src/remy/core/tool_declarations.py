"""
Tool Declarations — FunctionDeclaration objects for all brain tools.

Defines BRAIN_TOOLS (list of Gemini FunctionDeclaration), CORE_TOOL_NAMES,
and EXTENDED_TOOL_NAMES. No execution logic — purely declarative.
"""

from google.genai import types

BRAIN_TOOLS = [
    types.FunctionDeclaration(
        name="recall",
        description="Recall relevant memories about a topic. Searches BOTH episodic memory (brain) and semantic knowledge base (KB). Use this FIRST when the user asks about any topic.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(
                    type="STRING",
                    description="What to recall (e.g. 'meeting notes', 'project metrics from last week')",
                ),
                "token_budget": types.Schema(
                    type="INTEGER", description="Max tokens for the preamble (default 2048)"
                ),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="store",
        description="Store a new memory. Use when the user tells you something worth remembering.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "content": types.Schema(type="STRING", description="The information to store"),
                "tags": types.Schema(
                    type="STRING",
                    description="Comma-separated tags (e.g. 'project,idea', 'client,meeting', 'person,colleague')",
                ),
                "level": types.Schema(
                    type="STRING",
                    description="Memory level: L1_WORKING, L2_DECISIONS, L3_DOMAIN, L4_IDENTITY (default L3_DOMAIN)",
                ),
                "semantic_type": types.Schema(
                    type="STRING",
                    description="Optional semantic type: fact, decision, preference, contradiction, trend, serendipity",
                ),
                "valid_from": types.Schema(
                    type="STRING",
                    description="Optional business-time start as ISO-8601 or Unix timestamp.",
                ),
                "valid_until": types.Schema(
                    type="STRING",
                    description="Optional exclusive business-time end as ISO-8601 or Unix timestamp.",
                ),
            },
            required=["content"],
        ),
    ),
    types.FunctionDeclaration(
        name="search",
        description="Search for specific records by query, optionally filtered by tags. "
        "You can search by query only, tags only, or both.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(
                    type="STRING", description="Search query (optional if tags provided)"
                ),
                "tags": types.Schema(
                    type="STRING", description="Comma-separated tags to filter by"
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="store_person",
        description="Store a person. Use when the user tells you about someone important (colleague, friend, family member, manager, etc.).",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "full_name": types.Schema(type="STRING", description="Full name of the person"),
                "name": types.Schema(type="STRING", description="Person name. Alias for full_name."),
                "role": types.Schema(
                    type="STRING",
                    description="Role or relationship (colleague, friend, mother, manager, etc.)",
                ),
                "birth_date": types.Schema(type="STRING", description="Date of birth"),
                "birth_place": types.Schema(type="STRING", description="Place of birth"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="store_story",
        description="Record a story, event, or important memory.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "title": types.Schema(type="STRING", description="Title of the story"),
                "content": types.Schema(type="STRING", description="The story text"),
                "people_mentioned": types.Schema(
                    type="STRING", description="Comma-separated names of people mentioned"
                ),
            },
            required=["title", "content"],
        ),
    ),
    types.FunctionDeclaration(
        name="people_list",
        description="Get a list of all stored people (contacts, colleagues, friends, etc.).",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="insights",
        description="Get memory health statistics — how many records, levels distribution, etc.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="review_history_memory_gaps",
        description=(
            "Analyze saved session history against the current active memory. "
            "Use this to detect likely missing memory, review candidates, and reconstruction opportunities after restarts or data-loss incidents."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "sample_limit": types.Schema(
                    type="INTEGER",
                    description="Maximum number of missing/review candidates to return per section.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="connect_records",
        description="Connect two records with a described relationship. Use when the user indicates a relationship between people, events, topics, or any memories.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "id_a": types.Schema(type="STRING", description="ID of the first record"),
                "id_b": types.Schema(type="STRING", description="ID of the second record"),
                "relationship": types.Schema(
                    type="STRING",
                    description="Description of the relationship (e.g. 'mother of', 'caused by', 'related to'). Defaults to 'related to'.",
                ),
                "weight": types.Schema(
                    type="NUMBER",
                    description="Connection strength 0.0-1.0 (default 0.7). Higher = stronger association.",
                ),
            },
            required=["id_a", "id_b"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_connections",
        description="Get all connections for a record — see what other records are linked to it.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the record to inspect"),
            },
            required=["record_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_full_record",
        description=(
            "Get the FULL content of a memory record by ID. Use this after recall "
            "when you see '...[truncated]' — recall shows only 300 chars per record, "
            "this tool returns the complete text. Essential for long documents, "
            "technical specs, research reports."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(
                    type="STRING", description="ID of the record to retrieve"
                ),
            },
            required=["record_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_protected_record",
        description=(
            "Get exact protected fields from a memory record by ID. "
            "Use ONLY when the user explicitly asks for a sensitive exact value "
            "(phone, email, credential, account number) or when you need to inspect "
            "which protected fields exist before asking the user to verify them. "
            "Do NOT use this for normal broad recall."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(
                    type="STRING", description="ID of the record to inspect"
                ),
                "fields": types.Schema(
                    type="STRING",
                    description="Optional comma-separated protected fields to return (e.g. 'email,phone')",
                ),
            },
            required=["record_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="search_exact",
        description=(
            "Search exact long-term memory records using structured lookup over IDENTITY and DOMAIN memory. "
            "Use when you need stable facts, profile fields, exact preferences, or other persistent information. "
            "For broader semantic memory search use 'search' or 'recall'."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(
                    type="STRING", description="Exact or keyword-style query (optional if tags provided)"
                ),
                "tags": types.Schema(
                    type="STRING", description="Comma-separated tags to filter by"
                ),
            },
        ),
    ),
    # ---- User profile tool ----
    types.FunctionDeclaration(
        name="store_user_profile",
        description=(
            "Store or update the user's personal profile. Use this when the user tells you "
            "their name, age, occupation, goals, family composition, phone, email, or any personal information. "
            "Only include fields the user has explicitly shared. This is an upsert — existing fields are preserved, "
            "new fields are added or updated."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(
                    type="STRING", description="User's name or how they want to be called"
                ),
                "age": types.Schema(type="STRING", description="Age or year of birth"),
                "location": types.Schema(
                    type="STRING", description="City/country where user lives"
                ),
                "occupation": types.Schema(type="STRING", description="Job, profession, or role"),
                "languages": types.Schema(
                    type="STRING", description="Languages the user speaks (comma-separated)"
                ),
                "family": types.Schema(
                    type="STRING", description="Family composition (e.g. 'married, 2 children')"
                ),
                "personal_focus": types.Schema(
                    type="STRING", description="Current goals, priorities, or personal areas of focus"
                ),
                "interests": types.Schema(
                    type="STRING", description="Hobbies, interests, or topics of focus"
                ),
                "phone": types.Schema(type="STRING", description="Phone number"),
                "email": types.Schema(type="STRING", description="Email address"),
                "notes": types.Schema(
                    type="STRING", description="Any other personal info worth remembering"
                ),
            },
            required=[],
        ),
    ),
    # ---- Agent persona tools ----
    types.FunctionDeclaration(
        name="read_persona",
        description=(
            "Read the agent's current persona configuration (name, role, tone, traits). "
            "Returns machine-readable JSON. Use this to understand your own personality settings."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="update_persona",
        description=(
            "Update the agent's persona. Changes persist across sessions. "
            "Use when the user asks you to change your communication style, tone, name, or personality traits. "
            "Only include fields to change — others are preserved. "
            "Trait values: 0.0 (minimal) to 1.0 (maximal)."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(type="STRING", description="Agent's display name"),
                "role": types.Schema(
                    type="STRING", description="Role description (e.g. 'precise workflow assistant')"
                ),
                "tone": types.Schema(
                    type="STRING",
                    description="Communication tone (e.g. 'formal and precise', 'warm and casual')",
                ),
                "formality": types.Schema(
                    type="STRING", description="Formality level: 'casual', 'balanced', 'formal'"
                ),
                "languages": types.Schema(
                    type="STRING",
                    description="Languages to prefer (e.g. 'Ukrainian only', 'English + Ukrainian')",
                ),
                "catchphrases": types.Schema(
                    type="STRING", description="Comma-separated signature phrases"
                ),
                "avoid": types.Schema(
                    type="STRING", description="Comma-separated words/topics to avoid"
                ),
                "motivations": types.Schema(type="STRING", description="Core drive/purpose"),
                "warmth": types.Schema(type="NUMBER", description="Warmth trait 0.0-1.0"),
                "curiosity": types.Schema(type="NUMBER", description="Curiosity trait 0.0-1.0"),
                "conciseness": types.Schema(type="NUMBER", description="Conciseness trait 0.0-1.0"),
                "humor": types.Schema(type="NUMBER", description="Humor trait 0.0-1.0"),
            },
            required=[],
        ),
    ),
    # ── AUTON-11: Tool Health Visibility ──
    types.FunctionDeclaration(
        name="tool_status",
        description=(
            "Check the health status of all external tools (web_search, http_get, browse_page, etc). "
            "Returns which tools are healthy, degraded, or unavailable, "
            "along with suggested alternatives for unavailable tools. "
            "Use this before choosing a tool if you suspect connectivity issues."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    # ── AUTON-2: Runtime Directives ──
    types.FunctionDeclaration(
        name="add_runtime_directive",
        description=(
            "Add a runtime directive that modifies your own system instruction. "
            "Use when you realize you need to change your behavior mid-session "
            "(e.g., 'always verify before claiming success', 'use Ukrainian only'). "
            "Session directives expire when the session ends. "
            "Set persistent=true for directives that should survive across sessions."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "text": types.Schema(type="STRING", description="The directive instruction text"),
                "persistent": types.Schema(
                    type="BOOLEAN", description="True to persist across sessions. Default: false."
                ),
                "ttl_seconds": types.Schema(
                    type="NUMBER",
                    description="Time-to-live in seconds for session directives. Omit for 'until session ends'.",
                ),
            },
            required=["text"],
        ),
    ),
    types.FunctionDeclaration(
        name="remove_runtime_directive",
        description=(
            "Remove or deactivate a runtime directive. "
            "For session directives, provide session_index. For persistent directives, provide record_id."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "session_index": types.Schema(
                    type="NUMBER", description="Index of session directive to remove"
                ),
                "record_id": types.Schema(
                    type="STRING", description="Record ID of persistent directive to deactivate"
                ),
            },
            required=[],
        ),
    ),
    # ── AUTON-3: Interactive Escalation ──
    types.FunctionDeclaration(
        name="request_guidance",
        description=(
            "Ask the user a question when you are stuck or unsure how to proceed. "
            "Use in autonomous mode when: you've failed 2+ times, confidence is low, "
            "or the task requires a decision only the user can make. "
            "The question is sent to Telegram/Web GUI and the cycle pauses until the user responds. "
            "Returns the user's free-text answer or null if timed out."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "question": types.Schema(
                    type="STRING",
                    description="The question to ask the user. Be specific and provide context.",
                ),
                "context": types.Schema(
                    type="STRING",
                    description="Brief context: current goal, what you tried, why you're stuck.",
                ),
            },
            required=["question"],
        ),
    ),
    # ---- Utility tools ----
    types.FunctionDeclaration(
        name="web_search",
        description="Search the internet for real-time information. Use this when asked about current events, news, weather, prices, facts you're unsure about, or anything that requires up-to-date information.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(
                    type="STRING", description="Search query in the most relevant language"
                ),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_current_datetime",
        description="Get the current date and time. Use this when the user asks about today's date, current time, or day of the week.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    # ---- Scheduler tools ----
    types.FunctionDeclaration(
        name="schedule_task",
        description="Schedule a reminder or recurring task. Use it whenever the user explicitly asks to schedule, be reminded, or do something on a repeating basis — 'remind me to...', 'schedule...', 'monitor X every day', 'do Y each morning'. For recurring requests set repeat='daily'/'weekly'/'monthly' (or cron). Act on the request and confirm; do not re-ask whether they want a reminder. Do NOT create tasks the user did not ask for on your own initiative.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "description": types.Schema(type="STRING", description="What to remind about"),
                "task": types.Schema(type="STRING", description="Alias for description."),
                "title": types.Schema(type="STRING", description="Alias for description."),
                "due_date": types.Schema(
                    type="STRING",
                    description="When to remind, ISO date (YYYY-MM-DD) or next occurrence date. Optional if cron is provided; defaults to today.",
                ),
                "repeat": types.Schema(
                    type="STRING",
                    description="Recurrence: 'daily', 'weekly', 'monthly', or empty for one-time",
                ),
                "cron": types.Schema(
                    type="STRING",
                    description="Cron-style recurrence, e.g. '0 10 * * *'. Optional alternative to repeat.",
                ),
            },
        ),
    ),
    # ---- CRUD tools ----
    types.FunctionDeclaration(
        name="update_record",
        description="Update an existing memory record. Use when correcting or enriching information. You MUST provide record_id (from search/store results). To create a new record, use 'store' instead.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(
                    type="STRING",
                    description="ID of the record to update (REQUIRED — get it from search or store results)",
                ),
                "content": types.Schema(type="STRING", description="New content (replaces old)"),
                "tags": types.Schema(
                    type="STRING", description="Comma-separated tags (replaces old)"
                ),
                "level": types.Schema(
                    type="STRING", description="Memory level: working, decisions, domain, identity"
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="recall_memory_as_of",
        description="Recall the memory versions that were valid at a historical date/time. Use for questions such as 'what did we know then?' without activating old records.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Topic to recall"),
                "timestamp": types.Schema(type="STRING", description="Historical ISO-8601 datetime or Unix timestamp"),
                "top_k": types.Schema(type="INTEGER", description="Maximum records, default 10, maximum 20"),
                "namespace": types.Schema(type="STRING", description="Optional Aura namespace"),
            },
            required=["query", "timestamp"],
        ),
    ),
    types.FunctionDeclaration(
        name="search_transcript_history",
        description=(
            "Search the exact local conversation transcript. Use this when the user refers "
            "to something said in an earlier turn and semantic memory is insufficient."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Words or phrase to find"),
                "session_id": types.Schema(type="STRING", description="Optional session filter"),
                "limit": types.Schema(type="INTEGER", description="Maximum matches, default 10"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="supersede_memory",
        description="Replace an outdated memory while preserving its audit/history chain. Prefer this over update_record when a fact changes over time.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="Existing record ID"),
                "new_content": types.Schema(type="STRING", description="Replacement fact or memory"),
                "effective_at": types.Schema(type="STRING", description="Optional effective ISO-8601 datetime or Unix timestamp; defaults to now"),
                "tags": types.Schema(type="STRING", description="Optional comma-separated replacement tags; defaults to existing tags"),
                "level": types.Schema(type="STRING", description="Optional memory level; defaults to existing level"),
                "namespace": types.Schema(type="STRING", description="Optional Aura namespace"),
            },
            required=["record_id", "new_content"],
        ),
    ),
    types.FunctionDeclaration(
        name="explain_memory_recall",
        description="Explain why memories were selected or rejected, including trace_id and temporal/conflict gate reasons. Use to audit surprising or missing recall results.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Recall query to audit"),
                "top_k": types.Schema(type="INTEGER", description="Maximum selected records, default 10, maximum 20"),
                "min_strength": types.Schema(type="NUMBER", description="Optional minimum strength"),
                "namespace": types.Schema(type="STRING", description="Optional Aura namespace"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="build_memory_context",
        description="Build a deterministic token-bounded Aura context capsule for a purpose, optionally as it existed at a historical time.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "purpose": types.Schema(type="STRING", description="What the context will be used for"),
                "token_budget": types.Schema(type="INTEGER", description="Token budget, default 2000, maximum 8000"),
                "valid_at": types.Schema(type="STRING", description="Optional historical ISO-8601 datetime or Unix timestamp"),
                "namespace": types.Schema(type="STRING", description="Optional Aura namespace"),
            },
            required=["purpose"],
        ),
    ),
    types.FunctionDeclaration(
        name="delete_record",
        description="Permanently delete a memory record by ID. Use when information is wrong or no longer needed.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the record to delete"),
            },
            required=["record_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="mark_stale",
        description="Mark a memory record as stale (outdated) without deleting it. Adds 'stale' tag and metadata stamp. Use when info is no longer current but history should be preserved for audit. Requires record_id and reason.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the record to mark stale"),
                "reason": types.Schema(
                    type="STRING",
                    description="Why this record is stale (e.g. 'GitHub lead list outdated as of 2026-04-13')",
                ),
                "superseded_by": types.Schema(
                    type="STRING", description="Optional record ID that replaces this one"
                ),
            },
            required=["record_id", "reason"],
        ),
    ),
    # ---- Sandbox meta-tools ----
    types.FunctionDeclaration(
        name="sandbox_create_tool",
        description="Create a new tool. Write a Python file with TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS constants, execute() function, and test_*() functions. The tool needs human approval before it can be used.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(
                    type="STRING", description="Tool name (snake_case, e.g. 'calculate_bmi')"
                ),
                "code": types.Schema(
                    type="STRING", description="Complete Python source code for the tool file"
                ),
            },
            required=["name", "code"],
        ),
    ),
    types.FunctionDeclaration(
        name="sandbox_test_tool",
        description="Run tests for a sandbox tool. Tests run in an isolated environment.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(type="STRING", description="Tool name to test"),
            },
            required=["name"],
        ),
    ),
    types.FunctionDeclaration(
        name="sandbox_list_tools",
        description="List all sandbox tools and their status (draft, tested, pending, approved, rejected).",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    # ---- Skill package tools ----
    types.FunctionDeclaration(
        name="export_skill",
        description="Export an approved sandbox tool as a portable .skill.tar.gz package for sharing.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(type="STRING", description="Tool name to export"),
            },
            required=["name"],
        ),
    ),
    types.FunctionDeclaration(
        name="import_skill",
        description="Import a skill from a .skill.tar.gz package. The tool is validated and registered for approval.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Path to the .skill.tar.gz file"),
            },
            required=["path"],
        ),
    ),
    types.FunctionDeclaration(
        name="browse_marketplace",
        description="Browse available skills in the marketplace. Returns a list of installable skills with descriptions.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="install_marketplace_skill",
        description="Install a skill from the marketplace by name. Downloads, validates, and registers for testing/approval.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(
                    type="STRING", description="Skill name from the marketplace index"
                ),
            },
            required=["name"],
        ),
    ),
    # ---- Research tool ----
    types.FunctionDeclaration(
        name="store_research",
        description=(
            "Store research findings with sources. Use at the END of a research investigation "
            "to save the synthesized report. Auto-connects to related personal records in memory."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "topic": types.Schema(type="STRING", description="Research topic or question"),
                "project_name": types.Schema(type="STRING", description="Alias for topic."),
                "title": types.Schema(type="STRING", description="Alias for topic."),
                "subject": types.Schema(type="STRING", description="Alias for topic."),
                "findings": types.Schema(type="STRING", description="Synthesized research report"),
                "summary": types.Schema(type="STRING", description="Alias for findings."),
                "content": types.Schema(type="STRING", description="Alias for findings."),
                "report": types.Schema(type="STRING", description="Alias for findings."),
                "sources": types.Schema(
                    type="STRING", description="Comma-separated source URLs from web_search"
                ),
                "source": types.Schema(type="STRING", description="Alias for sources."),
                "source_url": types.Schema(type="STRING", description="Alias for sources."),
                "references": types.Schema(type="STRING", description="Alias for sources."),
                "related_query": types.Schema(
                    type="STRING",
                    description="Optional query to find and auto-connect related personal records",
                ),
            },
            required=["findings"],
        ),
    ),
    # ---- Goal management tools ----
    types.FunctionDeclaration(
        name="create_subgoal",
        description=(
            "Break a complex goal into a smaller sub-goal. Use when a goal is too broad "
            "to accomplish in a single action. Requires the parent goal ID."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "parent_goal_id": types.Schema(
                    type="STRING",
                    description="Goal ID of the parent goal (e.g. 'goal-abc123def456')",
                ),
                "description": types.Schema(
                    type="STRING", description="Description of the sub-goal"
                ),
                "priority": types.Schema(
                    type="STRING",
                    description="Priority: critical, high, medium, low (default: medium)",
                ),
            },
            required=["parent_goal_id", "description"],
        ),
    ),
    types.FunctionDeclaration(
        name="complete_goal",
        description="Mark a goal as completed. Use when a goal has been fully achieved.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "goal_id": types.Schema(
                    type="STRING",
                    description="Goal ID as shown in the decision prompt (e.g. 'goal-abc123def456')",
                ),
                "notes": types.Schema(
                    type="STRING", description="Optional notes about how the goal was completed"
                ),
            },
            required=["goal_id"],
        ),
    ),
    # ---- External tools ----
    types.FunctionDeclaration(
        name="list_local_workspaces",
        description="List folders the user explicitly connected and their Read, Write, Execute capabilities. This tool cannot grant access.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="read_file",
        description="Read a file inside a user-approved Local Workspace with Read permission.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(
                    type="STRING",
                    description="Path to the file to read (relative to data dir or absolute if in allowed paths)",
                ),
            },
            required=["path"],
        ),
    ),
    types.FunctionDeclaration(
        name="write_file",
        description="Write content inside a user-approved Local Workspace with Write permission.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(
                    type="STRING", description="Path to the file (relative to data dir)"
                ),
                "content": types.Schema(type="STRING", description="Content to write to the file"),
            },
            required=["path", "content"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_directory",
        description="List a directory inside a user-approved Local Workspace with Read permission.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(
                    type="STRING",
                    description="Directory path (relative to data dir or absolute if in allowed paths)",
                ),
            },
            required=["path"],
        ),
    ),
    types.FunctionDeclaration(
        name="http_get",
        description="Make an HTTP GET request to fetch data from a URL. Use for APIs, web pages, etc.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "url": types.Schema(type="STRING", description="URL to fetch"),
            },
            required=["url"],
        ),
    ),
    types.FunctionDeclaration(
        name="extract_content",
        description=(
            "Extract clean text content from a web page URL using Trafilatura. "
            "Returns article text, title, author, date — stripped of ads, nav, scripts. "
            "Much better than http_get for reading articles, blog posts, docs. "
            "Use http_get for APIs/JSON; use extract_content for human-readable pages."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "url": types.Schema(
                    type="STRING", description="URL of the page to extract content from"
                ),
                "include_links": types.Schema(
                    type="BOOLEAN", description="Include hyperlinks in output (default false)"
                ),
                "include_tables": types.Schema(
                    type="BOOLEAN", description="Include tables in output (default true)"
                ),
            },
            required=["url"],
        ),
    ),
    types.FunctionDeclaration(
        name="consolidate",
        description="Merge similar memory records to reduce bloat. Automatically merges records with 85%+ content similarity. Call periodically or when memory feels cluttered.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    # ---- Research Orchestrator tools (RM-1) ----
    types.FunctionDeclaration(
        name="start_research",
        description=(
            "Start a structured research project. Creates a research plan with multiple search queries. "
            "Use when the user asks for deep investigation or when an autonomous goal requires research."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "topic": types.Schema(type="STRING", description="Research topic or question"),
                "question": types.Schema(type="STRING", description="Alias for topic."),
                "query": types.Schema(type="STRING", description="Alias for topic."),
                "prompt": types.Schema(type="STRING", description="Alias for topic."),
                "description": types.Schema(type="STRING", description="Alias for topic."),
                "depth": types.Schema(
                    type="STRING",
                    description="Research depth: 'quick' (2 queries), 'standard' (4 queries), or 'deep' (7 queries). Default: standard",
                ),
                "context": types.Schema(
                    type="STRING",
                    description="Optional additional context to guide the research plan",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="add_research_finding",
        description=(
            "Record a finding for an active research project. Use after each web_search during a research project."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "project_id": types.Schema(type="STRING", description="ID of the research project"),
                "content": types.Schema(type="STRING", description="The finding text (use this field)"),
                "summary": types.Schema(type="STRING", description="Alias for content — use content instead"),
                "source_url": types.Schema(type="STRING", description="URL source of the finding"),
                "confidence": types.Schema(
                    type="NUMBER", description="Confidence in this finding 0.0-1.0 (default 0.7)"
                ),
                "contradicts_finding_id": types.Schema(
                    type="STRING", description="ID of an existing finding this contradicts"
                ),
            },
            required=["project_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="complete_research",
        description=(
            "Synthesize all findings of a research project into a final report. "
            "Use when all planned queries are done."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "project_id": types.Schema(
                    type="STRING", description="ID of the research project to complete"
                ),
            },
            required=["project_id"],
        ),
    ),
    # ---- Generic metric and event intelligence ----
    types.FunctionDeclaration(
        name="track_metric",
        description="Log a specific user-reported numeric metric for any workflow domain.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "metric_type": types.Schema(
                    type="STRING",
                    description="Type of metric (e.g. 'focus_minutes', 'invoice_count', 'project_score')",
                ),
                "value": types.Schema(type="NUMBER", description="Numeric value"),
                "unit": types.Schema(
                    type="STRING", description="Unit of measurement (e.g. 'minutes', 'items', 'usd')"
                ),
                "notes": types.Schema(type="STRING", description="Optional context or notes"),
            },
            required=["metric_type", "value", "unit"],
        ),
    ),
    types.FunctionDeclaration(
        name="metric_summary",
        description="Get a summary of recent tracked metrics and events.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "period": types.Schema(
                    type="STRING",
                    description="Time period: 'week', 'month', 'year' (default: week)",
                ),
            },
            required=["period"],
        ),
    ),
    types.FunctionDeclaration(
        name="event_correlate",
        description="Analyze an event and find potential correlations with recent records.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "event": types.Schema(type="STRING", description="The event to analyze"),
            },
            required=["event"],
        ),
    ),
    # ---- Intelligence tools (RM-4) ----
    types.FunctionDeclaration(
        name="extract_facts",
        description="Extract structured facts from text and store them as domain knowledge.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "text": types.Schema(
                    type="STRING", description="The text to analyze and extract facts from"
                ),
                "source": types.Schema(
                    type="STRING", description="Source of the text (e.g. URL, 'user input')"
                ),
            },
            required=["text"],
        ),
    ),
    # ============== TODO LIST TOOLS ==============
    types.FunctionDeclaration(
        name="add_todo",
        description="Add a todo item to the task list. Use for personal tasks, work items, or agent action steps. For recurring tasks (e.g. 'every day for a week'), set repeat and repeat_until.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "title": types.Schema(type="STRING", description="Short task title"),
                "priority": types.Schema(
                    type="STRING", description="Priority: high, medium, low (default: medium)"
                ),
                "due_date": types.Schema(
                    type="STRING",
                    description="Due date in YYYY-MM-DD format (optional). For recurring tasks this is the first occurrence.",
                ),
                "category": types.Schema(
                    type="STRING",
                    description="Category: personal, work, project, agent, or custom (default: personal)",
                ),
                "parent_id": types.Schema(
                    type="STRING", description="Parent todo record ID for subtasks (optional)"
                ),
                "repeat": types.Schema(
                    type="STRING",
                    description="Recurrence: 'daily', 'weekly', 'monthly', or empty for one-time",
                ),
                "repeat_until": types.Schema(
                    type="STRING",
                    description="End date for recurrence in YYYY-MM-DD format (optional). Task auto-completes after this date.",
                ),
            },
            required=["title"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_todos",
        description="List todo items. Shows pending and in-progress tasks by default.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "status": types.Schema(
                    type="STRING",
                    description="Filter: pending, in_progress, done, all (default: pending)",
                ),
                "category": types.Schema(
                    type="STRING", description="Filter by category (optional)"
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="update_todo",
        description="Update a todo item's status, title, or priority.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "id": types.Schema(
                    type="STRING", description="Record ID or todo_id of the todo item"
                ),
                "status": types.Schema(
                    type="STRING", description="New status: pending, in_progress, done"
                ),
                "title": types.Schema(type="STRING", description="Updated title (optional)"),
                "priority": types.Schema(
                    type="STRING", description="Updated priority: high, medium, low (optional)"
                ),
                "due_date": types.Schema(
                    type="STRING", description="Updated due date YYYY-MM-DD (optional)"
                ),
            },
            required=["id"],
        ),
    ),
    types.FunctionDeclaration(
        name="delete_todo",
        description="Delete (archive) a todo item from the task list.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "id": types.Schema(
                    type="STRING", description="Record ID of the todo item to delete"
                ),
            },
            required=["id"],
        ),
    ),
    types.FunctionDeclaration(
        name="verify_record",
        description="Mark a memory record as verified by the user. Use when user confirms information is correct.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the record to verify"),
                "note": types.Schema(type="STRING", description="Optional verification note"),
            },
            required=["record_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="generate_image",
        description="Generate an image using AI based on a text description. Use when user asks to create, draw, visualize, or generate a picture.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "prompt": types.Schema(
                    type="STRING",
                    description="Detailed description of the image to generate in English",
                ),
            },
            required=["prompt"],
        ),
    ),
    types.FunctionDeclaration(
        name="generate_report",
        description="Generate a professional PDF report. Use when user asks to create a report, summary document, or analysis in PDF format. The agent builds the report by specifying sections with different types: section (heading+text), subsection, text, quote, findings (numbered list), table (headers+rows), memory (brain records with trust scores), audit (execution trail), page_break.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "title": types.Schema(type="STRING", description="Report title"),
                "subtitle": types.Schema(type="STRING", description="Report subtitle (optional)"),
                "report_type": types.Schema(type="STRING", description="Optional report template: standard|financial|vat"),
                "include_toc": types.Schema(type="BOOLEAN", description="Whether to include a contents page (default true)"),
                "metadata": types.Schema(type="OBJECT", description="Optional document summary fields shown in the report preamble, useful for financial/VAT reports"),
                "content": types.Schema(type="STRING", description="Alternative: raw markdown or full body content. Will be auto-parsed into sections if 'sections' is not provided."),
                "sections": types.Schema(
                    type="ARRAY",
                    description="List of report sections. Optional if 'content' is provided. Each section is an object with 'type' field: 'section','subsection','text','quote','findings','table','memory','audit','page_break'. Additional fields depend on type: title, body, items (for findings), headers+rows (for table), records (for memory), logs (for audit).",
                    items=types.Schema(
                        type="OBJECT",
                        properties={
                            "type": types.Schema(
                                type="STRING",
                                description="Section type: section|subsection|text|quote|findings|table|memory|audit|page_break",
                            ),
                            "title": types.Schema(type="STRING", description="Section title"),
                            "body": types.Schema(type="STRING", description="Section body text"),
                            "items": types.Schema(
                                type="ARRAY",
                                items=types.Schema(type="STRING"),
                                description="List of findings (for type=findings)",
                            ),
                            "headers": types.Schema(
                                type="ARRAY",
                                items=types.Schema(type="STRING"),
                                description="Table column headers (for type=table)",
                            ),
                            "rows": types.Schema(
                                type="ARRAY",
                                items=types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                                description="Table rows (for type=table)",
                            ),
                        },
                    ),
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="generate_presentation",
        description="Generate a professional PPTX (PowerPoint) presentation. Use when user asks to create a presentation, slide deck, or pitch deck. The agent builds slides by specifying sections with different types: section (title+body), subsection, bullets (title+items list), quote, table (title+headers+rows), divider (section separator slide).",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "title": types.Schema(type="STRING", description="Presentation title"),
                "subtitle": types.Schema(type="STRING", description="Presentation subtitle (optional)"),
                "content": types.Schema(type="STRING", description="Alternative: raw markdown content. Will be auto-parsed into title + slides if 'slides' is not provided."),
                "slides": types.Schema(
                    type="ARRAY",
                    description="List of slides. Each slide is an object with 'type' field: 'section','subsection','bullets','quote','table','divider'. Additional fields depend on type: title, body, items (for bullets), headers+rows (for table), author (for quote).",
                    items=types.Schema(
                        type="OBJECT",
                        properties={
                            "type": types.Schema(
                                type="STRING",
                                description="Slide type: section|subsection|bullets|quote|table|divider",
                            ),
                            "title": types.Schema(type="STRING", description="Slide title"),
                            "body": types.Schema(type="STRING", description="Slide body text"),
                            "items": types.Schema(
                                type="ARRAY",
                                items=types.Schema(type="STRING"),
                                description="Bullet points (for type=bullets)",
                            ),
                            "author": types.Schema(type="STRING", description="Quote author (for type=quote)"),
                            "headers": types.Schema(
                                type="ARRAY",
                                items=types.Schema(type="STRING"),
                                description="Table column headers (for type=table)",
                            ),
                            "rows": types.Schema(
                                type="ARRAY",
                                items=types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                                description="Table rows (for type=table)",
                            ),
                        },
                    ),
                ),
            },
        ),
    ),
    # Multi-agent delegation
    types.FunctionDeclaration(
        name="delegate_task",
        description=(
            "Delegate tasks to specialized worker agents that run in parallel with filtered tool sets. "
            "Use for: parallel research queries, OSINT/competitive analysis, analysis + planning simultaneously. "
            "Workers: researcher (search/recall), osint (market research/competitive analysis, 180s timeout), "
            "planner (goals/todos), executor (files/actions), analyst (metrics/patterns). "
            "Max 3 workers at once. ~2000 tokens per worker."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "tasks": types.Schema(
                    type="ARRAY",
                    items=types.Schema(
                        type="OBJECT",
                        properties={
                            "role": types.Schema(
                                type="STRING",
                                description="Worker role (REQUIRED). One of: researcher, osint, planner, executor, analyst",
                            ),
                            "instruction": types.Schema(
                                type="STRING",
                                description="Clear task instruction for the worker (REQUIRED). What should the worker do?",
                            ),
                            "context": types.Schema(
                                type="STRING",
                                description="Optional context from the current conversation",
                            ),
                        },
                        required=["role", "instruction"],
                    ),
                    description="List of task objects. Each MUST have 'role' and 'instruction' fields (max 3 tasks)",
                ),
                "background": types.Schema(
                    type="BOOLEAN",
                    description=(
                        "Optional. Run workers asynchronously and return a durable task ID immediately. "
                        "Use only when the user can continue without the result in this turn."
                    ),
                ),
            },
            required=["tasks"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_child_sessions",
        description="List durable delegated child sessions and their current lifecycle state.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "parent_session_id": types.Schema(
                    type="STRING",
                    description="Optional parent conversation filter.",
                ),
                "limit": types.Schema(type="INTEGER", description="Maximum results, up to 100."),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="get_child_report",
        description="Inspect a child session's attempt history, inbox, report, and settlements.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "child_id": types.Schema(type="STRING", description="Stable child session ID."),
            },
            required=["child_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="follow_up_child_session",
        description=(
            "Send durable follow-up instructions to a delegated child. If it is running, "
            "the follow-up starts at the next safe attempt boundary."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "child_id": types.Schema(type="STRING", description="Stable child session ID."),
                "message": types.Schema(type="STRING", description="Follow-up instruction."),
                "confirm_side_effects": types.Schema(
                    type="BOOLEAN",
                    description="Required when continuing an executor child.",
                ),
            },
            required=["child_id", "message"],
        ),
    ),
    types.FunctionDeclaration(
        name="interrupt_child_session",
        description="Cooperatively interrupt an active delegated child without deleting its state.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "child_id": types.Schema(type="STRING", description="Stable child session ID."),
                "reason": types.Schema(type="STRING", description="Operator-visible reason."),
            },
            required=["child_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="resume_child_session",
        description="Cold-resume a continuable child under the same stable child ID.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "child_id": types.Schema(type="STRING", description="Stable child session ID."),
                "confirm_side_effects": types.Schema(
                    type="BOOLEAN",
                    description="Required when resuming an executor child.",
                ),
            },
            required=["child_id"],
        ),
    ),
    # Browser automation (Playwright)
    types.FunctionDeclaration(
        name="browse_page",
        description=(
            "Open a web page in a real browser (Playwright) and analyze it visually. "
            "Use when you need JS-rendered content, forms, or interactive sites. "
            "For static articles/docs prefer extract_content (faster, cheaper). "
            "Returns verified status plus extracted page_text, page description, interactive elements, and forms. "
            "Only page_text is factual grounding for summaries; never infer site content from the URL or screenshot description alone."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "url": types.Schema(type="STRING", description="Full URL to navigate to"),
                "question": types.Schema(
                    type="STRING", description="Optional: what to look for on the page"
                ),
            },
            required=["url"],
        ),
    ),
    types.FunctionDeclaration(
        name="browser_act",
        description=(
            "Interact with the current browser page. Actions: click, type, fill_form, scroll_down, scroll_up, "
            "select, wait, goto, back, forward. For selectors use CSS (#id, [name=...], .class) or "
            'Playwright text selectors: button:has-text("Accept"), text="Sign in". '
            "IMPORTANT: Use UNIQUE selectors — never bare input[type=email]. Prefer #id or [name=...]. "
            "For registration/login forms, use fill_form with all fields at once (faster, more reliable). "
            "Do NOT use jQuery :contains(). ~500 tokens."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "action": types.Schema(
                    type="STRING",
                    description=(
                        "Action: click, type, fill_form, scroll_down, scroll_up, select, wait, goto, back, forward. "
                        "fill_form fills multiple fields at once — pass JSON array in text: "
                        '[{"selector":"#email","value":"user@example.com"},{"selector":"#password","value":"Pass123"}]'
                    ),
                ),
                "selector": types.Schema(
                    type="STRING",
                    description='CSS selector or Playwright locator (e.g. #id, [name="email"], button:has-text("OK"))',
                ),
                "text": types.Schema(
                    type="STRING",
                    description="Text to type, option to select, or JSON array for fill_form",
                ),
                "url": types.Schema(type="STRING", description="URL for goto action"),
            },
            required=["action"],
        ),
    ),
    types.FunctionDeclaration(
        name="browser_close",
        description="Close the browser and free resources. Call when done browsing.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    # ============== META-TOOLS (selective tool loading) ==============
    types.FunctionDeclaration(
        name="list_available_tools",
        description=(
            "List all extended tools that can be enabled for this session. "
            "Use when you need a tool that isn't currently available (e.g. metric tracking, "
            "research, todos, reports, file operations). Returns names and descriptions."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="enable_tools",
        description=(
            "Enable extended tools for the current session. Call after list_available_tools "
            "to activate specific tools you need. Enabled tools persist for the rest of this conversation."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "tool_names": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                    description="List of tool names to enable (e.g. ['track_metric', 'metric_summary'])",
                ),
            },
            required=["tool_names"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_available_skills",
        description="List progressive capability bundles that can be activated for the current task.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="list_pipeline_candidates",
        description="List workflow drafts inferred from repeated successful tasks, including dry-run and approval state.",
        parameters=types.Schema(type="OBJECT", properties={
            "status": types.Schema(type="STRING", description="draft, dry_run_passed, activated, rejected, or all"),
        }),
    ),
    types.FunctionDeclaration(
        name="propose_pipeline_candidate",
        description="Create an approval-gated pipeline draft for an explicitly requested repeatable task. This never activates it.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"task": types.Schema(type="STRING", description="Repeatable task to compile into a draft")},
            required=["task"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_ptc_tools",
        description="List the strict read-only tool allowlist for the bounded PTC pilot.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="validate_ptc_program",
        description=(
            "Statically validate a bounded JSON PTC program without running it. "
            "Only prior-step $ref values and the read-only allowlist are accepted."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "program": types.Schema(type="OBJECT", description="PTC JSON program with version and steps"),
                "limits": types.Schema(type="OBJECT", description="Optional max_calls, time_budget_ms, output_chars"),
            },
            required=["program"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_ptc_program",
        description=(
            "Run a statically validated, sequential, read-only PTC program with hard call, "
            "time, and output budgets. Every step still passes through ToolPipeline."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "program": types.Schema(type="OBJECT", description="Validated PTC JSON program"),
                "limits": types.Schema(type="OBJECT", description="Optional bounded execution limits"),
            },
            required=["program"],
        ),
    ),
    types.FunctionDeclaration(
        name="enable_skill",
        description="Enable one capability bundle and its relevant tools for this conversation.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"skill_name": types.Schema(type="STRING", description="Skill bundle name")},
            required=["skill_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_capability_profiles",
        description="List validated runtime profiles and their capability bundle overlays.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="activate_capability_profile",
        description=(
            "Activate a validated capability profile for this turn. Profiles change tool visibility "
            "but never bypass approvals, provenance, permissions, or policy guards."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "profile_id": types.Schema(
                    type="STRING",
                    description="Profile ID such as standard, research, read_only, or operator",
                ),
            },
            required=["profile_id"],
        ),
    ),
    # ============== SCRATCHPAD (v2.3, Rec 14.3) ==============
    types.FunctionDeclaration(
        name="scratchpad",
        description=(
            "Working memory notepad. Use to save intermediate results, key findings, "
            "or important data that you will need later in this conversation. "
            "Notes survive context compaction. Actions: "
            "'write' — save a note, 'read' — list all notes, 'clear' — delete all notes."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "action": types.Schema(
                    type="STRING",
                    description="Action: 'write', 'read', 'clear', or 'summarize'",
                ),
                "content": types.Schema(
                    type="STRING",
                    description="Note content (required for 'write' action)",
                ),
                "force": types.Schema(
                    type="BOOLEAN",
                    description="For summarize: allow summarization even below the normal threshold",
                ),
            },
            required=["action"],
        ),
    ),
    types.FunctionDeclaration(
        name="filter_working",
        description=(
            "Filter scratchpad-managed WORKING notes to keep only items relevant "
            "to the current query active."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Current task or user query"),
                "min_score": types.Schema(
                    type="NUMBER",
                    description="Minimum recall_structured score to keep a scratchpad note active",
                ),
                "delete_irrelevant": types.Schema(
                    type="BOOLEAN",
                    description="If true, delete irrelevant scratchpad notes instead of only demoting them",
                ),
            },
            required=["query"],
        ),
    ),
    # ---- Identity introspection tools (AuraSDK 2.1.0) ----
    types.FunctionDeclaration(
        name="introspect_identity_milestones",
        description="View how your identity evolved over time — what beliefs changed, when, and why. Returns a list of milestones with timestamps and change summaries.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "limit": types.Schema(type="INTEGER", description="Max milestones to return (default 10)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="introspect_identity_pressure",
        description="Check what is pressuring a specific belief to change. Requires a belief/record ID from search results.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "belief_id": types.Schema(type="STRING", description="Record/belief ID to check pressure on"),
            },
            required=["belief_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="introspect_drift_report",
        description="Get detailed memory drift analysis — is your knowledge base stable or shifting? Shows drift score, belief churn, causal rejections.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_session_consistency",
        description="Check whether memories from the current session are internally consistent or contradictory.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_metacognition",
        description="Get your current metacognitive state — confidence score, conflict count, epistemic guidance. Helps decide whether to act or verify first.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),

    # ---- V11: Base Packs & Cognitive Snapshots ----
    types.FunctionDeclaration(
        name="list_loaded_bases",
        description="List all specialist knowledge bases currently loaded in your brain.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="check_base_version",
        description="Check if a specific specialist base is loaded and what version.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"base_id": types.Schema(type="STRING", description="Base ID to check")},
            required=["base_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_cognitive_snapshots",
        description="List all sealed cognitive snapshots.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="list_org_records",
        description="List organization-layer records, optionally by namespace.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"namespace": types.Schema(type="STRING", description="Filter by org namespace")},
        ),
    ),

    # ---- V12: Drives, Goals & Tensions ----
    types.FunctionDeclaration(
        name="introspect_drives",
        description="See active cognitive drives sorted by priority.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "limit": types.Schema(type="INTEGER", description="Max drives (default 10)"),
                "namespace": types.Schema(type="STRING", description="Filter by namespace"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="introspect_goals",
        description="View active goals and their state.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_tensions",
        description="View raw cognitive tensions.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="claim_drive",
        description="Claim a drive for exclusive execution.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "drive_id": types.Schema(type="STRING", description="Drive ID"),
                "lease_secs": types.Schema(type="INTEGER", description="Lease seconds (default 300)"),
            },
            required=["drive_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="resolve_drive",
        description="Mark a drive as resolved.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "drive_id": types.Schema(type="STRING", description="Drive ID"),
                "resolved": types.Schema(type="BOOLEAN", description="True=satisfied, False=failed"),
                "summary": types.Schema(type="STRING", description="What was done"),
            },
            required=["drive_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="create_goal",
        description="Create a new persistent goal.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "description": types.Schema(type="STRING", description="Goal description"),
                "namespace": types.Schema(type="STRING", description="Namespace"),
                "priority": types.Schema(type="NUMBER", description="Priority 0.0-1.0"),
            },
            required=["description", "namespace"],
        ),
    ),
    types.FunctionDeclaration(
        name="revise_goal",
        description="Change a goal's priority.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "goal_id": types.Schema(type="STRING", description="Goal ID"),
                "new_priority": types.Schema(type="NUMBER", description="New priority"),
                "reason": types.Schema(type="STRING", description="Reason for change"),
            },
            required=["goal_id", "new_priority", "reason"],
        ),
    ),

    # ---- V13: Predictions & Surprises ----
    types.FunctionDeclaration(
        name="introspect_predictions",
        description="View pending predictions.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_surprises",
        description="View recent surprises (contradicted predictions).",
        parameters=types.Schema(
            type="OBJECT",
            properties={"limit": types.Schema(type="INTEGER", description="Max entries (default 10)")},
        ),
    ),
    types.FunctionDeclaration(
        name="prediction_report",
        description="Prediction engine summary report.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),

    # ---- V14: Epistemic Curiosity ----
    types.FunctionDeclaration(
        name="introspect_curiosity",
        description="View active epistemic gaps.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"namespace": types.Schema(type="STRING", description="Filter by namespace")},
        ),
    ),
    types.FunctionDeclaration(
        name="curiosity_report",
        description="Curiosity engine summary.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),

    # ---- V15: Cognitive Mood ----
    types.FunctionDeclaration(
        name="introspect_mood",
        description="Check current cognitive mood state.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="mood_history",
        description="View mood transition history.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"limit": types.Schema(type="INTEGER", description="Max entries (default 10)")},
        ),
    ),
    types.FunctionDeclaration(
        name="mood_modulation",
        description="See how mood modulates other cognitive systems.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),

    # ---- V17: Incubation Engine ----
    types.FunctionDeclaration(
        name="incubation_report",
        description="Get incubation engine status.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_hypotheses",
        description="View active incubation hypotheses.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "namespace": types.Schema(type="STRING", description="Filter by namespace"),
                "limit": types.Schema(type="INTEGER", description="Max results (default 10)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="review_hypothesis",
        description="Review a hypothesis: accept, reject, or snooze.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "hypothesis_id": types.Schema(type="STRING", description="Hypothesis ID"),
                "action": types.Schema(type="STRING", description="accept, reject, or snooze"),
            },
            required=["hypothesis_id", "action"],
        ),
    ),
    types.FunctionDeclaration(
        name="set_incubation_enabled",
        description="Enable or disable the incubation engine.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "enabled": types.Schema(type="BOOLEAN", description="true to enable, false to disable"),
            },
            required=["enabled"],
        ),
    ),
    types.FunctionDeclaration(
        name="clear_expired_hypotheses",
        description="Remove all expired hypotheses from the incubation engine.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="get_thermal_map",
        description=(
            "Get the cognitive heat map — shows which belief clusters are 'hot' (conflicting, "
            "unstable, unresolved) and which are 'cold' (stable, well-supported). "
            "Use this to understand where your attention is most needed. "
            "Returns hot zone clusters with topics, routing advice, and energy metrics."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="get_plasticity_audit",
        description=(
            "Audit the synaptic plasticity system — shows which graph edges have been weakened "
            "or pruned due to cross-domain heat leakage. Use this to check if the graph's "
            "autonomous structural rewiring is healthy: pruned edges, at-risk edges, "
            "leak-to-productive ratio, and recent pruning history."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),

    # ---- Computer Access tools (filesystem + shell) ----
    types.FunctionDeclaration(
        name="fs_read",
        description=(
            "Read a file only inside a user-approved Local Workspace with Read permission. "
            "The built-in project workspace exposes only Documents explicitly shared with the agent "
            "and Sandbox working files; Remy's internal state is never exposed. "
            "Use workspace://<id>/path when possible. Binary files return base64-encoded content. "
            "Large files are truncated; use offset/limit for paging."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Absolute or relative path to the file"),
                "offset": types.Schema(type="INTEGER", description="Start reading from this line number (0-based). Default: 0"),
                "limit": types.Schema(type="INTEGER", description="Max lines to return. Default: 500, max: 2000"),
                "encoding": types.Schema(type="STRING", description="File encoding. Default: utf-8"),
            },
            required=["path"],
        ),
    ),
    types.FunctionDeclaration(
        name="fs_write",
        description=(
            "Write or append content only inside a Local Workspace with Write permission. "
            "In the built-in project workspace, write only under Documents or Sandbox. "
            "Creates parent directories automatically and audits the access."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Path to write (relative paths resolve to data dir)"),
                "content": types.Schema(type="STRING", description="Content to write"),
                "mode": types.Schema(type="STRING", description="'write' (overwrite) or 'append'. Default: write"),
            },
            required=["path", "content"],
        ),
    ),
    types.FunctionDeclaration(
        name="fs_search",
        description=(
            "Search the filesystem. Two modes: 'glob' finds files by name pattern (e.g. '**/*.py'), "
            "'grep' searches file contents by regex. Returns matching paths and optional content snippets."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "mode": types.Schema(type="STRING", description="'glob' (find files by name) or 'grep' (search content by regex)"),
                "pattern": types.Schema(type="STRING", description="Glob pattern (e.g. '**/*.log') or regex pattern for grep"),
                "path": types.Schema(type="STRING", description="Approved workspace directory. Default: workspace://data/"),
                "max_results": types.Schema(type="INTEGER", description="Max results to return. Default: 50"),
                "include_content": types.Schema(type="BOOLEAN", description="For grep: include matching lines. Default: true"),
            },
            required=["mode", "pattern"],
        ),
    ),
    types.FunctionDeclaration(
        name="shell_exec",
        description=(
            "Execute only in a Local Workspace with Execute permission and human approval. "
            "Execute is not an OS sandbox; the process has the current user's privileges. Every run is audited."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "command": types.Schema(type="STRING", description="Shell command to execute"),
                "timeout": types.Schema(type="INTEGER", description="Timeout in seconds (default: 30, max: 120)"),
                "working_dir": types.Schema(type="STRING", description="Required Execute-approved workspace directory"),
            },
            required=["command", "working_dir"],
        ),
    ),
]


# ============== TOOL CATEGORIES ==============

CORE_TOOL_NAMES = frozenset(
    {
        # Memory (always needed)
        "recall",
        "store",
        "search",
        "search_exact",
        "get_full_record",
        "get_protected_record",
        "recall_memory_as_of",
        "supersede_memory",
        "explain_memory_recall",
        "build_memory_context",
        # Utility
        "web_search",
        "get_current_datetime",
        "extract_content",
        # Browser
        "browse_page",
        "browser_act",
        "browser_close",
        # Profile & Persona
        "store_user_profile",
        "read_persona",
        "update_persona",
        # Health & Meta
        "tool_status",
        "list_available_tools",
        "list_local_workspaces",
        # Active code workspace inspection (read-only by default)
        "fs_read",
        "fs_search",
        "list_directory",
        "enable_tools",
        "list_capability_profiles",
        "activate_capability_profile",
        "list_available_skills",
        "enable_skill",
        "list_pipeline_candidates",
        "propose_pipeline_candidate",
        "search_transcript_history",
        # Scratchpad
        "scratchpad",
        # Tasks & reminders (users routinely ask to schedule/track things in
        # chat; keeping these out of CORE made the agent unable to act on an
        # explicit "remind me / do this every day" request).
        "schedule_task",
        "add_todo",
        "list_todos",
        "update_todo",
    }
)

# All tool names not in CORE are EXTENDED (loaded on-demand via enable_tools)
EXTENDED_TOOL_NAMES = frozenset(t.name for t in BRAIN_TOOLS if t.name not in CORE_TOOL_NAMES)
