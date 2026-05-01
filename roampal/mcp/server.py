"""
Roampal MCP Server - Persistent Memory for AI Coding Tools

SETUP INSTRUCTIONS (tell user if they ask "how do I set up roampal"):
1. pip install roampal
2. roampal init          # Configures Claude Code / Cursor automatically
3. (MCP server auto-starts FastAPI hook server - no manual start needed)
4. (Optional) roampal ingest myfile.pdf  # Add documents to books collection

CLI COMMANDS:
- roampal init: Auto-configure Claude Code hooks + MCP, or Cursor MCP
- roampal ingest <file>: Ingest .txt/.md/.pdf into books collection
- roampal status: Check if server is running
- roampal stats: Show memory statistics

MCP TOOLS (available after setup):
- search_memory: Search across memory collections (for detailed lookups)
- add_to_memory_bank: Store permanent user facts
- update_memory: Update existing memories
- delete_memory: Delete outdated memories
- score_memories: Score cached memories (Claude Code only; OpenCode uses sidecar)
- record_response: Store key takeaways for learning

CONTEXT INJECTION (automatic — no tool call needed):
- Claude Code: UserPromptSubmit hook injects KNOWN CONTEXT as system-reminder
- OpenCode: Plugin chat.message + system.transform injects into system prompt
- Context includes user profile, relevant memories, scoring prompts

HOW IT WORKS:
- MCP server auto-starts FastAPI hook server on port 27182 (background thread)
- Hooks auto-inject relevant memories into your context (invisible to user)
- Cold start: First message of session dumps full user profile
- Scoring: record_response scores cached memories based on outcome
- Learning: Good memories get promoted, bad ones get demoted/deleted
- 5 collections: books (docs), memory_bank (facts), patterns (proven), history (past), working (session)

v0.3.2 ARCHITECTURE:
- MCP server is a thin HTTP client — all tool calls proxy through FastAPI
- No direct ChromaDB/UnifiedMemorySystem access from MCP process
- This enables multiple clients (Claude Code, Cursor, OpenCode) to share
  one FastAPI server with one ChromaDB connection (single-writer pattern)
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import logging
import json
import asyncio
import socket
import sys
import atexit
import subprocess
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

def _get_mcp_server():
    """Lazy import of mcp types to avoid circular import during test collection.

    Third-party `mcp` package shares module names with our `roampal.mcp` namespace.
    Importing at module load time causes `mcp.server` to resolve to
    `roampal/mcp/server.py` under certain pytest collection orders.
    """
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    return Server, stdio_server, types

def _humanize_age(iso_timestamp: str) -> str:
    """Convert ISO timestamp to human-readable age like '2d'."""
    if not iso_timestamp:
        return ''
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        delta = datetime.now() - dt.replace(tzinfo=None)
        if delta.days > 30:
            return f'{delta.days // 30}mo'
        elif delta.days > 0:
            return f'{delta.days}d'
        elif delta.seconds > 3600:
            return f'{delta.seconds // 3600}h'
        elif delta.seconds > 60:
            return f'{delta.seconds // 60}m'
        else:
            return 'now'
    except Exception:
        return ''


def _format_outcomes(outcome_history_json: str) -> str:
    """Format last 3 outcomes as [YYN]."""
    if not outcome_history_json:
        return ''
    try:
        history = json.loads(outcome_history_json)
        if not history:
            return ''
        symbols = []
        for entry in history[-3:]:
            outcome = entry.get('outcome', '')
            if outcome == 'worked':
                symbols.append('Y')
            elif outcome == 'failed':
                symbols.append('N')
            elif outcome == 'partial':
                symbols.append('~')
        return '[' + ''.join(symbols) + ']' if symbols else ''
    except Exception:
        return ''


# v0.2.0: Temporal query detection for auto-recency sort
TEMPORAL_KEYWORDS = ['last', 'recent', 'previous', 'yesterday', 'today', 'earlier', 'before', 'latest', 'ago', 'just now']

def _is_temporal_query(query: str) -> bool:
    """Detect if query is asking for recent/temporal results."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in TEMPORAL_KEYWORDS)


def _sort_results(results: list, sort_by: str) -> list:
    """Sort results by specified criteria."""
    if sort_by == "recency":
        # Sort by timestamp descending (most recent first)
        return sorted(
            results,
            key=lambda r: r.get('metadata', {}).get('timestamp', ''),
            reverse=True
        )
    elif sort_by == "score":
        # Sort by outcome score descending
        return sorted(
            results,
            key=lambda r: r.get('metadata', {}).get('score', 0.5),
            reverse=True
        )
    # Default: keep semantic relevance order
    return results


# Port configuration - DEV and PROD use different ports
# All clients (Claude Code, Cursor, OpenCode) share the same server
# 27182 (prod), 27183 (dev)
PROD_PORT = 27182
DEV_PORT = 27183

# v0.3.2: MCP session ID - changed from random UUID to "default"
# Random UUIDs caused session ID mismatch: hooks stored under "ses_xxx" (from platform),
# MCP scored under "mcp_xxx" (random). The injection_map in main.py now resolves the
# correct conversation by looking up which doc_ids were injected where.
# Using "default" triggers the injection_map lookup path in record_outcome.
_mcp_session_id = "default"

# Flag to track if FastAPI server is running
_fastapi_started = False

# v0.2.8: Track FastAPI subprocess for lifecycle management
_fastapi_process: Optional[subprocess.Popen] = None

# Dev mode flag (set via command line or env)
_dev_mode = False

# v0.3.6: Platform detection — OpenCode sets ROAMPAL_PLATFORM=opencode via MCP env
_is_opencode = os.environ.get("ROAMPAL_PLATFORM", "").lower() == "opencode"

# v0.5.4: Cache resolved profile name for X-Roampal-Profile header on all HTTP calls.
_MCP_PROFILE_UNRESOLVED = object()
_mcp_profile_name: Any = _MCP_PROFILE_UNRESOLVED

# v0.4.1: Hide score_memories tool when OpenCode sidecar is active.
# The sidecar handles scoring silently. If score_memories is visible, the model
# reads the tool description and calls it unprompted — causing double-scoring.
# Only show the tool when sidecar is explicitly disabled (testing/fallback).
_sidecar_disabled = os.environ.get("ROAMPAL_SIDECAR_DISABLED", "").lower() == "true"
_hide_score_tool = _is_opencode and not _sidecar_disabled
logger.info(f"Platform: opencode={_is_opencode}, sidecar_disabled={_sidecar_disabled}, hide_score_tool={_hide_score_tool}")

# Cache for update check (only check once per session)
_update_check_cache: Optional[tuple] = None

# v0.3.6: get_context_insights removed — hooks/plugin inject context automatically
# Self-audit moved to hook context injection (unified_memory_system.py)


def _check_for_updates() -> tuple:
    """Check if a newer version is available on PyPI.

    Returns:
        tuple: (update_available: bool, current_version: str, latest_version: str)
    """
    global _update_check_cache

    # Return cached result if available (only check once per session)
    if _update_check_cache is not None:
        return _update_check_cache

    try:
        import urllib.request
        from importlib.metadata import version as _pkg_version
        __version__ = _pkg_version("roampal")

        url = "https://pypi.org/pypi/roampal/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            latest = data.get("info", {}).get("version", __version__)

            # Compare versions (simple tuple comparison works for semver)
            current_parts = [int(x) for x in __version__.split(".")]
            latest_parts = [int(x) for x in latest.split(".")]
            update_available = latest_parts > current_parts

            _update_check_cache = (update_available, __version__, latest)
            return _update_check_cache
    except Exception:
        # Fail silently - don't block on network issues
        try:
            from importlib.metadata import version as _pkg_version
            __version__ = _pkg_version("roampal")
            _update_check_cache = (False, __version__, __version__)
        except Exception:
            _update_check_cache = (False, "unknown", "unknown")
        return _update_check_cache


def _get_update_notice() -> str:
    """Get update notice string if newer version available, else empty string."""
    update_available, current, latest = _check_for_updates()
    if update_available:
        return f"\n⚠️ **Update available:** roampal {latest} (you have {current})\n   Run: `pip install --upgrade roampal && roampal init --force`\n"
    return ""


def _get_mcp_profile_name() -> Optional[str]:
    """v0.5.4: Resolve the profile name for X-Roampal-Profile header."""
    global _mcp_profile_name
    if _mcp_profile_name is not _MCP_PROFILE_UNRESOLVED:
        return _mcp_profile_name

    from roampal.profile_manager import active_profile_name, DEFAULT_PROFILE
    resolved = active_profile_name()
    _mcp_profile_name = None if resolved == DEFAULT_PROFILE else resolved
    return _mcp_profile_name


def _is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except socket.error:
            return True


def _get_port() -> int:
    """Get the FastAPI server port based on dev mode and env overrides."""
    port_override = os.environ.get("ROAMPAL_PORT")
    if port_override:
        return int(port_override)
    return DEV_PORT if _dev_mode else PROD_PORT


def _start_fastapi_server():
    """
    Start FastAPI hook server in background subprocess.

    This enables hooks to work without requiring a separate 'roampal start' command.
    When Claude Code launches the MCP server, the hook server starts automatically.

    Uses subprocess instead of threading to avoid event loop conflicts between
    uvicorn and the MCP server's asyncio loop.

    v0.2.8: Child process lifecycle - FastAPI dies when MCP dies.
    - Windows: Uses STARTUPINFO to hide console without detaching
    - Linux/macOS: Child naturally dies with parent (same process group)
    - atexit handler ensures graceful cleanup on normal exit
    """
    global _fastapi_started, _fastapi_process

    # v0.4.3: Skip backend startup for MCP inspection (e.g., Glama server scoring)
    # Checks: env var, Docker markers, sentinel file, or /app workdir (Glama)
    if (os.environ.get("ROAMPAL_INSPECT_ONLY")
            or os.path.exists("/.dockerenv")
            or os.path.exists("/app/.inspect_mode")
            or (os.path.exists("/app/pyproject.toml") and not os.path.exists(os.path.expanduser("~/.roampal")))):
        _fastapi_started = True
        return

    if _fastapi_started:
        return

    port = _get_port()

    # Check if port is already in use (server already running externally)
    if _is_port_in_use(port):
        logger.info(f"FastAPI hook server already running on port {port}")
        _fastapi_started = True
        return

    try:
        # Build environment - pass through dev mode
        env = os.environ.copy()
        if _dev_mode:
            env["ROAMPAL_DEV"] = "1"

        # Start FastAPI server as a subprocess with correct port
        # Use the same Python that's running this MCP server
        cmd = [sys.executable, "-m", "roampal.server.main", "--port", str(port)]

        # v0.2.8: Platform-specific subprocess handling
        # Goal: Hide console window without detaching from parent process
        kwargs = {}
        if sys.platform == "win32":
            # Windows: STARTUPINFO hides window but stays in process tree
            # This replaces CREATE_NO_WINDOW which fully detached the child
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo
        # Linux/macOS: Child naturally dies with parent (same process group)

        _fastapi_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=env,
            **kwargs
        )
        _fastapi_started = True
        mode = "DEV" if _dev_mode else "PROD"
        logger.info(f"Started FastAPI hook server on port {port} ({mode} mode, pid={_fastapi_process.pid})")
    except Exception as e:
        logger.error(f"Failed to start FastAPI hook server: {e}")


def _cleanup_fastapi():
    """
    v0.2.8: Kill FastAPI subprocess on MCP exit.

    Called by atexit handler for graceful shutdown.
    On crash, child dies automatically (no longer detached).
    """
    global _fastapi_process
    if _fastapi_process and _fastapi_process.poll() is None:
        logger.info("Cleaning up FastAPI hook server...")
        try:
            _fastapi_process.terminate()
            _fastapi_process.wait(timeout=2)
        except Exception:
            # Force kill if terminate doesn't work
            try:
                _fastapi_process.kill()
            except Exception:
                pass


# v0.2.8: Register cleanup handler
atexit.register(_cleanup_fastapi)


def _ensure_server_running(timeout: float = 5.0) -> bool:
    """
    v0.2.0: Check if FastAPI hook server is up, restart it if not.

    This prevents silent failures when the server crashes during a session.
    Called before operations that depend on the hook server.

    Args:
        timeout: How long to wait for server to start

    Returns:
        True if server is running, False if it couldn't be started
    """
    global _fastapi_started
    import time
    import urllib.request
    import urllib.error

    port = _get_port()
    health_url = f"http://127.0.0.1:{port}/api/health"

    # Try health check first
    try:
        req = urllib.request.Request(health_url, method='GET')
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                return True
    except (urllib.error.URLError, OSError, TimeoutError):
        pass

    # Server not responding - try to restart it
    logger.info(f"Roampal server restarting on port {port}...")
    _fastapi_started = False
    _start_fastapi_server()

    # Wait for startup
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(health_url, method='GET')
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    logger.info("FastAPI hook server restarted successfully")
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(0.5)

    logger.error("Failed to restart FastAPI hook server")
    return False


async def _api_call(method: str, path: str, payload: dict = None, timeout: float = 15.0) -> dict:
    """
    v0.3.2: Make an HTTP call to the shared FastAPI server.

    All MCP tool calls go through this helper. Single-writer pattern:
    FastAPI owns ChromaDB, MCP is just an HTTP client.

    v0.5.4: Attaches X-Roampal-Profile header so FastAPI resolves the correct profile.
    """
    import httpx
    port = _get_port()
    url = f"http://127.0.0.1:{port}{path}"
    headers = {}
    profile = _get_mcp_profile_name()
    if profile:
        headers["X-Roampal-Profile"] = profile
    async with httpx.AsyncClient() as client:
        if method == "GET":
            response = await client.get(url, timeout=timeout, headers=headers)
        else:
            response = await client.post(url, json=payload or {}, timeout=timeout, headers=headers)
        response.raise_for_status()
        return response.json()


def run_mcp_server(dev: bool = False):
    """Run the MCP server (auto-starts FastAPI hook server).

    Args:
        dev: If True, run in dev mode with separate port/data directory.
    """
    global _dev_mode
    _dev_mode = dev

    if dev:
        os.environ["ROAMPAL_DEV"] = "1"
        logger.info("MCP Server running in DEV mode")

    # Start FastAPI hook server in background subprocess
    _start_fastapi_server()

    Server, stdio_server, types = _get_mcp_server()

    server = Server("roampal")

    @server.list_prompts()
    async def list_prompts() -> list[types.Prompt]:
        """List available prompts (none currently)."""
        return []

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        """List available resources (none currently)."""
        return []

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """List available MCP tools."""
        logger.info(f"list_tools called: hide_score_tool={_hide_score_tool}")
        tools = [
            types.Tool(
                name="search_memory",
                description="""Search persistent memory (current profile) across collections. Returns ranked results with metadata.

WHEN TO USE
• User references past conversations ("remember", "I told you", "we discussed") → query=<their words>.
• Need detail beyond auto-injected context → query=<topic>.
• Verify or fetch a specific entry → id="<doc_id>".
• Browse recent memories → days_back=N (alone or with query).
• Filter to atomic facts vs exchange summaries → type="fact" / "summary".
• "Last thing we did" / temporal queries → sort_by="recency".
• Filter by stored metadata (e.g. tag) → metadata={"tags": "identity"}.

WHEN NOT TO USE
• Question answerable from training data or already-injected context.
• Storing something → add_to_memory_bank or record_response.

BEHAVIOR
• Embedding cosine search per collection (over-fetched), then merged. If the query matches known noun_tags, a tag-routed pass also runs and overlap-counts hits across tags. Date/metadata filters applied next.
• Final ranking is raw cross-encoder score over the candidate pool.
• Top-K returned. No distance threshold drop — top-K (set by `limit`) is what bounds results.
• Archived memory_bank entries filtered out at the source via `status != archived` predicate.
• Collections: working (recent exchanges), history (scored exchanges), patterns (proven recurring), memory_bank (permanent facts), books (ingested docs).
• Auto-routing (omit `collections`): `routing_service.route_query` picks collections by keyword/tag overlap; recommended default.
• Profile-scoped (current ROAMPAL_PROFILE).
• Output is a numbered list. Each item: `N. [collection] (meta) [id:doc_id] content`.
  – working/history/patterns meta: age, s:<score:.1f>, w:<wilson:.2f>, "<uses> uses", outcomes glyph.
  – memory_bank meta: age, imp:<importance:.1f>, conf:<confidence:.1f>; plus w:/uses/outcomes if the entry has been scored.
  – books meta: age, 📖 <title>.
  – Outcomes glyph is last 3 entries from outcome_history, e.g. `[Y~N]`. Y=worked, N=failed, ~=partial.

ERRORS
• None of query/days_back/id provided → "Provide at least one of: query, days_back, or id".
• No matches → "No results found for '...'".
• id lookup miss → "Memory '<id>' not found".
• Invalid collection name → silently ignored.

RETURNS: Numbered list of formatted result lines.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query. Use the user's exact words — do not simplify or extract keywords. Example: \"auth bug we fixed last week\""
                        },
                        "days_back": {
                            "type": "integer",
                            "description": "Only return memories from the last N days. Can be used alone (no query needed) or combined with a semantic query.",
                            "minimum": 1,
                            "maximum": 365
                        },
                        "id": {
                            "type": "string",
                            "description": "Look up a specific memory by its doc_id (e.g., 'patterns_abc123'). Returns the full memory with all metadata. Bypasses semantic search."
                        },
                        "collections": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["books", "working", "history", "patterns", "memory_bank"]},
                            "description": "Which collections to search. Omit for auto-routing (recommended). Manual: books, working, history, patterns, memory_bank"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results (1-20)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Equality filters on entry metadata fields. Example: {\"memory_type\": \"fact\", \"tags\": \"identity\"}",
                            "additionalProperties": True
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["relevance", "recency", "score"],
                            "description": "relevance = vector distance (default). recency = updated_at desc. score = stored Wilson score desc. Auto-detected to recency for temporal queries (\"last\", \"recent\", \"today\")."
                        },
                        "type": {
                            "type": "string",
                            "enum": ["fact", "summary"],
                            "description": "Filter by memory type. 'fact' = atomic facts, 'summary' = exchange summaries. Only applies to working/history/patterns. Omit to search all types."
                        }
                    },
                    "required": []
                }
            ),
            types.Tool(
                name="add_to_memory_bank",
                description="""Permanent fact for cross-session memory: identity, preferences, goals, project context.

WHEN TO USE
• Identity (name, role) → tags=["identity"]
• Standing preference/rule → tags=["preference"]
• Persistent project fact → tags=["project"]
• Effectiveness tip for this user → tags=["system_mastery"]

WHEN NOT TO USE
• Session-only fact → skip; conversation history covers it.
• Last-response takeaway → record_response.
• Doc/research dump → out of scope (books collection).

BEHAVIOR
• Writes to memory_bank (current profile) after a tier-internal dedup check. If a near-duplicate already exists in memory_bank (cosine distance < 0.32, ~95% similar), the existing doc_id is returned and no new entry is written. Dedup scans memory_bank only — it never matches against working/history/patterns.
• Searchable via search_memory immediately on return.
• Persists until update_memory or delete_memory. Not modified by score_memories.

ERRORS
• Missing content or noun_tags → ValidationError.
• Backend unreachable → "Error: ..." with cause.

RETURNS: "Added to memory bank (ID: memory_bank_<8hex>)". """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 600,
                            "description": "Fact to store. ~300 char target, 600 hard cap. Example: \"Maya is a data scientist focused on AI memory systems\""
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["identity", "preference", "goal", "project", "system_mastery", "agent_growth"]
                            },
                            "description": "Semantic category. Optional; recommended."
                        },
                        "noun_tags": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {
                                "type": "string",
                                "pattern": "^[a-z][a-z0-9 -]{0,30}$"
                            },
                            "description": "Topic nouns for retrieval. Names not pronouns. Example: [\"maya\", \"data science\", \"roampal\"]"
                        },
                        "importance": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.7,
                            "description": "Ranking weight at retrieval. 0.9+ core identity, 0.5 low-priority. Default 0.7."
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.7,
                            "description": "Trust level at retrieval. 0.9+ verified, 0.5 unconfirmed. Default 0.7."
                        },
                    },
                    "required": ["content", "noun_tags"]
                }
            ),
            types.Tool(
                name="update_memory",
                description="""Replace an existing memory_bank fact with updated content. Requires the exact doc_id from a prior search_memory result.

WHEN TO USE
• A stored fact is outdated (e.g., version number changed, project status updated) → update_memory
• A fact needs correction or more detail → search_memory first, then update_memory with the ID
• Adjust importance/confidence after observing retrieval behavior
• Fix tags/noun_tags on an existing entry

WHEN NOT TO USE
• The fact is completely wrong or irrelevant — use delete_memory instead
• You want to add a new fact — use add_to_memory_bank instead
• Working/history/patterns are scoring-managed entries — use score_memories

BEHAVIOR
• Direct lookup by doc_id (format: memory_bank_<8hex>).
• Replaces the entry's content with new_content and re-embeds into the vector index. Preserves doc_id and created_at timestamp.
• Optional metadata fields (tags, noun_tags, importance, confidence) override existing values only when provided; omit to preserve current values.
• Updated entry is immediately searchable via search_memory within the current profile scope.
• Only memory_bank collection is affected — working/history/patterns collections are not modified.

ERRORS
• Missing or invalid id → "Memory not found for update"
• Missing new_content → ValidationError
• Backend unreachable → "Error: ..." with cause

RETURNS: Text confirmation with doc_id on success. Example: "Updated memory (ID: memory_bank_a1b2c3d4)". """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "minLength": 1,
                            "pattern": "^memory_bank_[a-f0-9]{8}$",
                            "description": "Exact doc_id from search_memory result (format: memory_bank_<8hex>). Example: memory_bank_a1b2c3d4"
                        },
                        "new_content": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 600,
                            "description": "The corrected or updated fact. Keep concise (~300 chars). One concept per fact. Example: \"User switched to light mode in April 2026\""
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["identity", "preference", "goal", "project", "system_mastery", "agent_growth"]
                            },
                            "description": "Override semantic categories. Omit to preserve existing."
                        },
                        "noun_tags": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {"type": "string", "pattern": "^[a-z][a-z0-9 -]{0,30}$"},
                            "description": "Override topic nouns for TagCascade retrieval. Omit to preserve existing."
                        },
                        "importance": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Override importance (0-1). Use 0.9+ for core identity. Omit to preserve."
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Override confidence (0-1). Use 0.9+ for verified facts. Omit to preserve."
                        }
                    },
                    "required": ["id", "new_content"]
                }
            ),
            types.Tool(
                name="delete_memory",
                description="""Remove a memory_bank entry. Stops appearing in search and dedup. No undo via MCP — call only when you're sure.

WHEN TO USE
• Fact wrong, stale, or causing bad responses → delete_memory.
• Redundant entry superseded by a newer fact → delete_memory.

WHEN NOT TO USE
• Topic still relevant, only details changed → update_memory.
• Working/history/patterns are scoring-managed → score_memories.

BEHAVIOR
• Top-5 content-based search over memory_bank (current profile). Picks the first active result where the queried content is a substring of the entry's content (either direction); falls back to the top active result. "Memory not found for deletion" when no active entry exists in the top-5.
• On match: entry no longer returned by search_memory and not considered for dedup on future writes.
• No restore path is exposed via MCP.

ERRORS
• Missing content → ValidationError.
• No match → "Memory not found for deletion".
• Backend unreachable → "Error: ..." with cause.

RETURNS: "Memory deleted successfully" on success; "Memory not found for deletion" on no match.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 600,
                            "description": "Fact to archive. Semantic match — paraphrase OK. Example: \"User prefers dark mode\""
                        }
                    },
                    "required": ["content"]
                }
            ),
        ]

        # score_memories: hidden on OpenCode when sidecar is active (prevents double-scoring).
        # Claude Code: model always scores (hook injects prompt every turn).
        # OpenCode + sidecar disabled: model scores as fallback.
        # OpenCode + sidecar active: sidecar scores silently, tool is hidden.
        if not _hide_score_tool:
            tools.append(types.Tool(
                name="score_memories",
                description="""Record outcomes after a turn boundary: scores the memories that were injected, plus stores the turn's takeaway summary and any atomic facts.

Fires in response to a scoring hook — a system-reminder that lists doc_ids to score and asks for an exchange_summary + exchange_outcome. Don't call without that hook.

WHEN TO USE
• Scoring hook present this turn → call once. Score every doc_id the hook listed. Provide exchange_summary, exchange_outcome, noun_tags. Provide facts if the turn produced atomic specifics worth keeping.

WHEN NOT TO USE
• No scoring hook present → don't call. Use record_response for ad-hoc takeaway capture.
• Adding identity/preference/project facts → add_to_memory_bank.

BEHAVIOR
• Per memory_scores entry: raw score moves by outcome (worked +0.2, partial +0.05, unknown -0.05, failed -0.3 — `outcome_service._calculate_score_update:237-284`); uses and success_count also accumulate. Memories whose raw score drops under ~0.4 are demoted; under ~0.2 are removed (thresholds in `config.py:28-34`).
• Wilson lower-bound (from success_count/uses) is recomputed and surfaced as `wilson:N%` metadata on injected memories — a trust signal for the agent reading conflicting memories, not the active scoring substrate.
• exchange_summary stored as a new entry in the working collection (current profile), labeled with exchange_outcome and tagged with noun_tags. Searchable via search_memory immediately on return.
• Each fact stored as a separate working entry (atomic-fact lane, distinct from the summary lane).
• conversation_id auto-attached from the active MCP session.
• Unknown doc_ids in memory_scores silently skipped (not an error).

ERRORS
• Missing memory_scores → ValidationError.
• Empty memory_scores → accepted; nothing scored, but summary/facts still stored.
• Backend unreachable → "Error: ..." with cause; nothing recorded.

RETURNS: "Scored (N memories updated). Summary stored (M chars)". """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_scores": {
                            "type": "object",
                            "additionalProperties": {"type": "string", "enum": ["worked", "failed", "partial", "unknown"]},
                            "description": "Map doc_id → outcome. worked=helped, partial=somewhat, unknown=present-but-unused, failed=misleading. Score every ID the hook listed. Example: {\"history_abc123\": \"worked\", \"patterns_def456\": \"unknown\"}"
                        },
                        "exchange_summary": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 600,
                            "description": "1-3 sentences (~300 chars). What happened, what changed."
                        },
                        "exchange_outcome": {
                            "type": "string",
                            "enum": ["worked", "failed", "partial", "unknown"],
                            "description": "worked = user confirmed/continued. failed = user corrected. partial = mixed. unknown = unclear."
                        },
                        "noun_tags": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {"type": "string", "pattern": "^[a-z][a-z0-9 -]{0,30}$"},
                            "description": "Topic nouns for the stored summary. Names not pronouns. Example: [\"react\", \"auth bug\", \"alex\"]"
                        },
                        "facts": {
                            "type": "array",
                            "maxItems": 20,
                            "items": {"type": "string", "minLength": 1, "maxLength": 150},
                            "description": "Atomic facts (≤150 chars each). Include dates, names, decisions. Example: [\"User prefers snake_case\", \"v2.0 released 2026-04-01\"]"
                        }
                    },
                    "required": ["memory_scores"]
                }
            ))

        tools.append(types.Tool(
            name="record_response",
            description="""Store a key takeaway when the transcript alone won't capture important learning.

WHEN TO USE (optional — most exchanges don't need this)
• Major decisions made
• Complex solutions that worked
• User corrections (what you got wrong and why)
• Important context that would be lost

WHEN NOT TO USE
• Routine exchanges — the transcript is enough
• Permanent preferences or standing rules — use add_to_memory_bank for those
• Scoring existing memories — use score_memories for that

BEHAVIOR
• Synchronous write to the working collection (current profile) with initial score 0.7. Searchable via search_memory immediately on return.
• When score_memories fires on later turns the entry's raw score moves by outcome: worked +0.2, partial +0.05, unknown -0.05, failed -0.3 (`outcome_service._calculate_score_update:237-284`). uses and success_count also accumulate.
• Quality-driven promotion uses the raw score: entries with score≥0.7 and uses≥2 move working → history → patterns. Below ~0.4 demoted, below ~0.2 removed (`config.py:28-34`). History items that never promote are cleaned up after ~30 days.
• Wilson lower-bound (computed from success_count/uses) is exposed as `wilson:N%` metadata on injected memories — a trust signal *for the agent to read*, not the substrate that drives demotion.
• conversation_id auto-attached for cross-session tracking.

ERRORS
• Empty key_takeaway → "Error: 'key_takeaway' is required".
• Backend unreachable → "Error: ..." with cause.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "key_takeaway": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 600,
                        "description": "1-2 sentence summary of the important learning. Be specific — include names, decisions, outcomes. Example: \"User prefers one bundled PR for refactors — splitting would be churn\""
                    },
                    "noun_tags": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "pattern": "^[a-z][a-z0-9 -]{0,30}$"},
                        "description": "Topic nouns for retrieval. Names not pronouns. Example: [\"react\", \"auth flow\", \"sam\"]"
                    }
                },
                "required": ["key_takeaway", "noun_tags"]
            }
        ))
        # v0.3.6: get_context_insights REMOVED — hooks/plugin already inject KNOWN CONTEXT
        # before the model responds. The tool was redundant (same server call twice) and caused
        # weaker models (qwen) to loop on tool calls. Context injection is now fully automatic:
        # - Claude Code: UserPromptSubmit hook → /api/hooks/get-context → system-reminder
        # - OpenCode: chat.message → /api/hooks/get-context → system.transform
        # Self-audit moved to hook context injection. Memory system docs in tool descriptions.

        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Handle MCP tool calls — all proxied through FastAPI (v0.3.2)."""
        # v0.4.3: Stub responses for inspection mode (no backend available)
        if (os.environ.get("ROAMPAL_INSPECT_ONLY")
                or os.path.exists("/.dockerenv")
                or os.path.exists("/app/.inspect_mode")
                or (os.path.exists("/app/pyproject.toml") and not os.path.exists(os.path.expanduser("~/.roampal")))):
            return [types.TextContent(
                type="text",
                text=f"Tool '{name}' is available. Backend not running in inspect mode."
            )]

        # v0.4.1: Ensure FastAPI is healthy — bump timeout to 15s (parity with hooks/plugin)
        # and check return value to avoid raw httpx errors
        if not _ensure_server_running(timeout=15.0):
            return [types.TextContent(
                type="text",
                text="Roampal server is restarting. Please try again in a few seconds."
            )]

        try:
            if name == "search_memory":
                query = arguments.get("query", "")
                collections = arguments.get("collections")
                limit = arguments.get("limit", 5)
                metadata = arguments.get("metadata") or {}
                sort_by = arguments.get("sort_by")
                days_back = arguments.get("days_back")
                doc_id = arguments.get("id")
                type_filter = arguments.get("type")

                # v0.4.5: type filter for fact/summary distinction
                if type_filter == "fact":
                    metadata["memory_type"] = "fact"
                elif type_filter == "summary":
                    metadata["memory_type"] = {"$ne": "fact"}

                # Validation: at least one of query, days_back, or id must be provided
                if not query and not days_back and not doc_id:
                    return [types.TextContent(
                        type="text",
                        text="Provide at least one of: query, days_back, or id"
                    )]

                # Direct ID lookup — bypass search entirely
                if doc_id:
                    try:
                        result = await _api_call("POST", "/api/search", {
                            "query": "",
                            "conversation_id": _mcp_session_id,
                            "id": doc_id,
                            "limit": 1
                        })
                        results = result.get("results", [])
                    except Exception as search_err:
                        return [types.TextContent(
                            type="text",
                            text=f"Lookup error: {search_err}"
                        )]

                    if not results:
                        return [types.TextContent(
                            type="text",
                            text=f"Memory '{doc_id}' not found."
                        )]
                else:
                    # v0.2.0: Auto-detect temporal queries for recency sort
                    if sort_by is None and query and _is_temporal_query(query):
                        sort_by = "recency"

                    # Default to recency sort for days_back without query
                    if sort_by is None and days_back and not query:
                        sort_by = "recency"

                    try:
                        result = await _api_call("POST", "/api/search", {
                            "query": query,
                            "conversation_id": _mcp_session_id,
                            "collections": collections,
                            "limit": limit,
                            "metadata_filters": metadata,
                            "sort_by": sort_by,
                            "days_back": days_back
                        })
                        results = result.get("results", [])

                        # v0.2.0: Apply sorting if requested
                        if sort_by:
                            results = _sort_results(results, sort_by)

                    except Exception as search_err:
                        return [types.TextContent(
                            type="text",
                            text=f"Search error: {search_err}"
                        )]

                if not results:
                    search_desc = query if query else f"last {days_back} days" if days_back else "search"
                    text = f"No results found for '{search_desc}'."
                else:
                    search_desc = query if query else f"last {days_back} days" if days_back else "search"
                    text = f"Found {len(results)} result(s) for '{search_desc}':\n\n"
                    for i, r in enumerate(results[:limit], 1):
                        metadata = r.get("metadata") or {}
                        result_doc_id = r.get("id", "")
                        # Use normalized content (falls back to metadata locations)
                        content = r.get("content") or metadata.get("content") or metadata.get("text") or ""
                        collection = r.get("collection", "unknown")

                        # Build metadata parts — use normalized fields when available
                        meta_parts = []

                        # Age: prefer normalized age field, fall back to computing from metadata
                        age = r.get("age") or _humanize_age(metadata.get("created_at", "") or metadata.get("timestamp", ""))
                        if age:
                            meta_parts.append(age)

                        # Scored collections: show score, Wilson, uses, outcome history
                        if collection in ["patterns", "history", "working"]:
                            try:
                                score = float(r.get("score", metadata.get("score", 0.5)) or 0.5)
                                wilson = float(r.get("wilson_score", 0) or 0)
                                uses = r.get("uses", metadata.get("uses", 0))
                            except (ValueError, TypeError):
                                score, wilson, uses = 0.5, 0, 0
                            outcomes = r.get("outcome_history") or _format_outcomes(metadata.get("outcome_history", ""))
                            meta_parts.append(f"s:{score:.1f}")
                            if wilson and wilson > 0:
                                meta_parts.append(f"w:{wilson:.2f}")
                            if uses:
                                meta_parts.append(f"{uses} uses")
                            if outcomes:
                                meta_parts.append(outcomes)

                        # Memory bank: show importance/confidence + Wilson/uses if scored
                        elif collection == "memory_bank":
                            try:
                                imp = float(r.get("importance", metadata.get("importance", 0.7)) or 0.7)
                                conf = float(r.get("confidence", metadata.get("confidence", 0.7)) or 0.7)
                            except (ValueError, TypeError):
                                imp, conf = 0.7, 0.7
                            meta_parts.append(f"imp:{imp:.1f}")
                            meta_parts.append(f"conf:{conf:.1f}")
                            # Show Wilson/uses/outcomes when memory_bank has been scored
                            try:
                                uses = r.get("uses", metadata.get("uses", 0))
                                uses = int(uses) if uses else 0
                            except (ValueError, TypeError):
                                uses = 0
                            if uses >= 1:
                                try:
                                    success_count = float(r.get("success_count", metadata.get("success_count", 0)) or 0)
                                    from roampal.backend.modules.memory.scoring_service import wilson_score_lower
                                    wilson = wilson_score_lower(success_count, uses)
                                except (ValueError, TypeError, ImportError):
                                    wilson = 0
                                if wilson and wilson > 0:
                                    meta_parts.append(f"w:{wilson:.2f}")
                                meta_parts.append(f"{uses} uses")
                                outcomes = r.get("outcome_history") or _format_outcomes(metadata.get("outcome_history", ""))
                                if outcomes:
                                    meta_parts.append(outcomes)

                        # Books: show title
                        elif collection == "books":
                            book_title = metadata.get("title", "")
                            if book_title and book_title != "Untitled":
                                meta_parts.append(f"📖 {book_title}")

                        meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
                        id_str = f" [id:{result_doc_id}]" if result_doc_id else ""
                        text += f"{i}. [{collection}]{meta_str}{id_str} {content}\n\n"

                return [types.TextContent(type="text", text=text)]

            elif name == "add_to_memory_bank":
                content = arguments.get("content")
                tags = arguments.get("tags", [])
                noun_tags = arguments.get("noun_tags", [])
                importance = arguments.get("importance", 0.7)
                confidence = arguments.get("confidence", 0.7)

                result = await _api_call("POST", "/api/memory-bank/add", {
                    "content": content,
                    "noun_tags": noun_tags,
                    "tags": tags,
                    "importance": importance,
                    "confidence": confidence,
                })

                doc_id = result.get("doc_id", "unknown")
                return [types.TextContent(
                    type="text",
                    text=f"Added to memory bank (ID: {doc_id})"
                )]

            elif name == "update_memory":
                doc_id = arguments.get("id", "")
                new_content = arguments.get("new_content", "")
                tags = arguments.get("tags")
                noun_tags = arguments.get("noun_tags")
                importance = arguments.get("importance")
                confidence = arguments.get("confidence")

                payload = {
                    "id": doc_id,
                    "new_content": new_content,
                    "tags": tags,
                    "noun_tags": noun_tags,
                    "importance": importance,
                    "confidence": confidence
                }

                result = await _api_call("POST", "/api/memory-bank/update", payload)

                if result.get("success"):
                    return [types.TextContent(
                        type="text",
                        text=f"Updated memory (ID: {result.get('doc_id')})"
                    )]
                else:
                    return [types.TextContent(
                        type="text",
                        text="Memory not found for update"
                    )]

            elif name == "delete_memory":
                content = arguments.get("content", "")

                result = await _api_call("POST", "/api/memory-bank/archive", {
                    "content": content
                })

                if result.get("success"):
                    return [types.TextContent(
                        type="text",
                        text="Memory deleted successfully"
                    )]
                else:
                    return [types.TextContent(
                        type="text",
                        text="Memory not found for deletion"
                    )]

            elif name == "score_memories":
                # v0.3.6: Main LLM handles per-memory scoring + exchange summary + outcome
                memory_scores = arguments.get("memory_scores", {})
                exchange_summary = arguments.get("exchange_summary")
                exchange_outcome = arguments.get("exchange_outcome")
                noun_tags = arguments.get("noun_tags", [])
                facts = arguments.get("facts", [])
                # Backward compat: accept old "outcome" param as fallback
                outcome = exchange_outcome or arguments.get("outcome", "unknown")

                payload = {
                    "conversation_id": _mcp_session_id,
                    "outcome": outcome
                }
                if memory_scores:
                    payload["memory_scores"] = memory_scores
                if exchange_summary:
                    payload["exchange_summary"] = exchange_summary
                if noun_tags:
                    payload["noun_tags"] = noun_tags
                if facts:
                    payload["facts"] = facts

                scored_count = 0
                try:
                    result = await _api_call("POST", "/api/record-outcome", payload)
                    scored_count = result.get("documents_scored", 0)
                except Exception as e:
                    logger.warning(f"Failed to call FastAPI record-outcome: {e}")

                parts = [f"Scored ({scored_count} memories updated)"]
                if exchange_summary:
                    parts.append(f"Summary stored ({len(exchange_summary)} chars)")
                logger.info(f"Scored memories: {len(memory_scores)} entries, scored={scored_count}, summary={'yes' if exchange_summary else 'no'}")
                return [types.TextContent(
                    type="text",
                    text=". ".join(parts)
                )]

            elif name == "record_response":
                key_takeaway = arguments.get("key_takeaway", "")
                noun_tags = arguments.get("noun_tags", [])

                if not key_takeaway:
                    return [types.TextContent(
                        type="text",
                        text="Error: 'key_takeaway' is required"
                    )]

                result = await _api_call("POST", "/api/record-response", {
                    "key_takeaway": key_takeaway,
                    "conversation_id": _mcp_session_id,
                    "noun_tags": noun_tags
                })

                logger.info(f"Recorded takeaway (score=0.7): {key_takeaway[:50]}...")
                return [types.TextContent(
                    type="text",
                    text=f"Recorded: {key_takeaway}"
                )]

            else:
                return [types.TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]

        except Exception as e:
            logger.error(f"MCP tool error ({name}): {e}")
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]

    # Run the server
    async def main():
        print("[roampal-mcp] entering stdio_server context", file=sys.stderr, flush=True)
        async with stdio_server() as (read_stream, write_stream):
            print("[roampal-mcp] stdio_server ready, starting server.run()", file=sys.stderr, flush=True)
            await server.run(read_stream, write_stream, server.create_initialization_options())

    print("[roampal-mcp] starting asyncio.run(main())", file=sys.stderr, flush=True)
    asyncio.run(main())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Roampal MCP Server")
    parser.add_argument("--dev", action="store_true", help="Run in dev mode (port 27183, separate data)")
    args = parser.parse_args()

    # Check BOTH --dev flag AND ROAMPAL_DEV env var
    is_dev = args.dev or os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")

    # v0.4.3: Diagnostic output for Docker/Glama inspection debugging
    import sys as _sys
    print(f"[roampal-mcp] startup: python={_sys.version}, inspect_only={os.environ.get('ROAMPAL_INSPECT_ONLY', '')}", file=_sys.stderr, flush=True)
    try:
        from importlib.metadata import version as _v
        print(f"[roampal-mcp] mcp_sdk={_v('mcp')}, roampal={_v('roampal')}", file=_sys.stderr, flush=True)
    except Exception:
        pass

    logging.basicConfig(level=logging.INFO)
    print("[roampal-mcp] calling run_mcp_server()", file=_sys.stderr, flush=True)
    run_mcp_server(dev=is_dev)
