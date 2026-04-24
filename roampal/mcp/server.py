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

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

logger = logging.getLogger(__name__)

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
                description="""Search persistent memory across all collections. Use when you need details beyond what was automatically provided in context.

WHEN TO USE:
• User references past conversations ("remember", "I told you", "we discussed")
• You need more detail than the injected context provided
• You want to verify a memory by ID before acting on it (use id= parameter)
• You want to browse recent memories by time (use days_back= parameter)

WHEN NOT TO USE:
• General knowledge questions — use your training data
• The injected context already answers the question
• You want to store something — use add_to_memory_bank or record_response

BEHAVIOR:
• Searches across 5 collections: working (recent), history (scored), patterns (proven), memory_bank (permanent facts), books (documents)
• Omit 'collections' for automatic routing (recommended)
• Returns ranked results with scores, age, and usage metadata
• Results include formatted indicators: [YYN] (outcome history), s:0.7 (score), w:0.68 (confidence), use count, age, [id:doc_id]

ERROR HANDLING:
• No query, days_back, or id provided → returns "Provide at least one of: query, days_back, or id"
• No matches found → returns "No results found for '...'"
• Invalid collection name → collection is ignored
• ID lookup with nonexistent ID → returns "Memory 'id' not found."

RETURNS:
Formatted text with numbered results. Each result shows: [collection] (age, score, confidence, uses, outcome_history) [id:doc_id] content""",
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
                            "description": "Optional metadata filters. Example: {\"memory_type\": \"fact\"}",
                            "additionalProperties": True
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["relevance", "recency", "score"],
                            "description": "Sort order. 'recency' for temporal queries like 'last thing we did'. Auto-detected if omitted."
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
                description="""Store a permanent fact for cross-session continuity. Use for identity, preferences, goals, and project context.

WHEN TO USE:
• User shares identity information (name, role, background) — use tags=["identity"]
• User states a preference or standing rule
• Important project context that should persist across all sessions
• Knowledge that helps you be more effective for this user

WHEN NOT TO USE:
• Session-specific details or temporary context — let working memory handle it
• Raw conversation content — auto-captured by the scoring system
• Exchange takeaways — use record_response instead (those get outcome-scored)
• Research dumps or large documents — use books collection via CLI

BEHAVIOR:
• Creates a new memory in the memory_bank collection with a generated doc_id
• Memory_bank facts are permanent and NOT outcome-scored — they persist until you update or delete them
• If always_inject=true, the fact appears in every context injection (use only for core identity)
• Duplicate content is allowed — check with search_memory first to avoid redundancy
• Keep facts concise (~300 chars). One concept per fact. Put key info first.

ERROR HANDLING:
• Missing noun_tags → returns validation error (required field)
• Empty content → accepted but not useful (avoid)

RETURNS:
Text confirmation with assigned doc_id. Example: "Added to memory bank (ID: memory_bank_a1b2c3d4)" """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact to store. Keep concise (~300 chars). One concept per fact. Example: \"Logan is a data scientist focused on AI memory systems\""},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Semantic categories: 'identity' (name, role), 'preference' (workflow, style), 'goal' (objectives), 'project' (codebases), 'system_mastery' (effectiveness tips), 'agent_growth' (meta-learning). Use 'identity' for user profile facts."},
                        "noun_tags": {"type": "array", "items": {"type": "string"}, "description": "Topic nouns for tag-based retrieval. Lowercase, 1-3 words each, max 8. Use names not pronouns. Example: [\"logan\", \"data science\", \"roampal\"]"},
                        "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "How critical this fact is (0.0-1.0, default: 0.7). Use 0.9+ for core identity facts."},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "How certain you are (0.0-1.0, default: 0.7). Use 0.9+ only for verified facts. Use 0.5 for unconfirmed claims."},
                        "always_inject": {"type": "boolean", "default": False, "description": "If true, this fact appears in every context injection. Use only for core identity. Default: false."}
                    },
                    "required": ["content", "noun_tags"]
                }
            ),
            types.Tool(
                name="update_memory",
                description="""Replace an existing memory_bank fact with updated content. Use when information changes but the topic is still relevant.

WHEN TO USE:
• A stored fact is outdated (e.g., version number changed, project status updated)
• A fact needs correction or more detail
• You gave wrong info because of a stale memory — fix it immediately

WHEN NOT TO USE:
• The fact is completely wrong or irrelevant — use delete_memory instead
• You want to update working/history/patterns memories — those are managed automatically by scoring
• You want to add a new fact — use add_to_memory_bank instead

BEHAVIOR:
• Finds the closest semantic match to old_content in memory_bank
• Replaces the matched memory's content with new_content, preserving its doc_id and metadata
• Only searches memory_bank collection — cannot update working/history/patterns
• old_content does not need to be exact — a close paraphrase works (semantic matching)

ERROR HANDLING:
• No semantic match found → returns "Memory not found for update", no changes made
• Both parameters are required — omitting either returns an error

RETURNS:
Text confirmation with doc_id on success. Example: "Updated memory (ID: memory_bank_a1b2c3d4)" """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "old_content": {"type": "string", "description": "The existing fact to find and replace. Matched by semantic similarity — exact text not required. Example: \"User prefers dark mode\""},
                        "new_content": {"type": "string", "description": "The corrected or updated fact. Keep concise (~300 chars). One concept per fact. Example: \"User switched to light mode in April 2026\""}
                    },
                    "required": ["old_content", "new_content"]
                }
            ),
            types.Tool(
                name="delete_memory",
                description="""Permanently remove a memory_bank fact. This action is irreversible.

WHEN TO USE:
• A fact is completely wrong, stale, or misleading
• A fact is redundant (superseded by a newer memory)
• A memory_bank fact keeps causing incorrect responses — remove it directly

WHEN NOT TO USE:
• The topic is still relevant but details changed — use update_memory instead
• You want to remove working/history/patterns memories — those are managed by scoring automatically (score them "failed" to demote)

BEHAVIOR:
• Searches memory_bank by semantic similarity to find the closest match
• Deletes the best-matching memory permanently
• Only operates on memory_bank — cannot delete from other collections
• Content is matched by meaning, not exact text — a paraphrase works
• If multiple memories are similar, only the closest match is deleted

ERROR HANDLING:
• No semantic match found → returns "Memory not found for deletion", nothing deleted
• Empty content → still attempts search (avoid — provide a meaningful description)

RETURNS:
Text confirmation on success: "Memory deleted successfully". On no match: "Memory not found for deletion".""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Describe the memory to delete in natural language. Matched by semantic similarity — a paraphrase works. Example: \"User's old email address\""}
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
                description="""Score memories that were retrieved in your previous context. Each memory gets an outcome rating that adjusts its future retrieval priority.

WHEN TO USE:
• When prompted by the scoring hook (appears as a scoring prompt in context)
• Score every memory ID listed in the prompt

WHEN NOT TO USE:
• To store new information — use record_response or add_to_memory_bank instead
• Without a scoring prompt — this tool evaluates existing memories, not new ones

BEHAVIOR:
• Updates confidence scores on each scored memory
• Memories repeatedly scored "failed" are automatically demoted or removed
• exchange_summary is stored as a new working memory for future retrieval
• facts (if provided) are stored as separate working memories
• Unknown memory IDs in memory_scores are silently skipped

ERROR HANDLING:
• Missing memory_scores → returns an error
• Empty memory_scores map → accepted but scores nothing
• Server unreachable → returns "Error: ..." with details

RETURNS:
Text confirmation. Example: "Scored (8 memories updated). Summary stored (300 chars)" """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_scores": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "string",
                                "enum": ["worked", "failed", "partial", "unknown"]
                            },
                            "description": "Object mapping memory IDs to outcomes. Keys are doc_ids, values are: 'worked' (helped your response), 'partial' (somewhat relevant), 'unknown' (present but unused), 'failed' (misleading — caused incorrect response). Example: {\"history_abc123\": \"worked\", \"patterns_def456\": \"unknown\"}"
                        },
                        "exchange_summary": {
                            "type": "string",
                            "description": "1-3 sentence summary of the previous exchange (~300 chars). Captures what happened and the outcome. Stored as a working memory for future retrieval."
                        },
                        "exchange_outcome": {
                            "type": "string",
                            "enum": ["worked", "failed", "partial", "unknown"],
                            "description": "Overall result of the previous exchange. 'worked' = user confirmed or continued. 'failed' = user corrected you. 'partial' = mixed. 'unknown' = unclear."
                        },
                        "noun_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Topic nouns from the exchange for tag-based retrieval. Lowercase, 1-3 words each, max 8. Use names not pronouns. Example: [\"react\", \"auth bug\", \"logan\"]"
                        },
                        "facts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Atomic facts from this exchange, stored as separate working memories. One fact per string, max 150 chars. Include specifics (dates, names, decisions). Example: [\"User prefers snake_case\", \"v2.0 released 2026-04-01\"]"
                        }
                    },
                    "required": ["memory_scores"]
                }
            ))

        tools.append(types.Tool(
            name="record_response",
            description="""Store a key takeaway when the transcript alone won't capture important learning.

WHEN TO USE (optional — most exchanges don't need this):
• Major decisions made
• Complex solutions that worked
• User corrections (what you got wrong and why)
• Important context that would be lost

WHEN NOT TO USE:
• Routine exchanges — the transcript is enough
• Permanent preferences or standing rules — use add_to_memory_bank for those
• Scoring existing memories — use score_memories for that

BEHAVIOR:
• Stores the takeaway as a new working memory with initial score 0.7
• Working memories are automatically scored on subsequent turns (+0.2 worked, +0.05 partial, -0.3 failed)
• Over time, useful takeaways promote to history (30d) then patterns (permanent)
• Returns text confirmation with the stored takeaway

ERROR HANDLING:
• Empty key_takeaway → returns "Error: 'key_takeaway' is required"
• Missing noun_tags → accepted but reduces retrieval quality""",
            inputSchema={
                "type": "object",
                "properties": {
                    "key_takeaway": {
                        "type": "string",
                        "description": "1-2 sentence summary of the important learning. Be specific — include names, decisions, outcomes. Example: \"User prefers one bundled PR for refactors — splitting would be churn\""
                    },
                    "noun_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topic nouns for tag-based retrieval. Lowercase, 1-3 words each, max 8. Use names not pronouns. Example: [\"react\", \"auth flow\", \"logan\"]"
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
                always_inject = arguments.get("always_inject", False)

                result = await _api_call("POST", "/api/memory-bank/add", {
                    "content": content,
                    "noun_tags": noun_tags,
                    "tags": tags,
                    "importance": importance,
                    "confidence": confidence,
                    "always_inject": always_inject
                })

                doc_id = result.get("doc_id", "unknown")
                return [types.TextContent(
                    type="text",
                    text=f"Added to memory bank (ID: {doc_id})"
                )]

            elif name == "update_memory":
                old_content = arguments.get("old_content", "")
                new_content = arguments.get("new_content", "")

                result = await _api_call("POST", "/api/memory-bank/update", {
                    "old_content": old_content,
                    "new_content": new_content
                })

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
