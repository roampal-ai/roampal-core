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
- score_memories: Score cached memories from previous context
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

import logging
import json
import asyncio
import socket
import os
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
# Used to return a leaner score_memories tool description (no exchange summary/outcome)
_is_opencode = os.environ.get("ROAMPAL_PLATFORM", "").lower() == "opencode"

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
        from roampal import __version__

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
            from roampal import __version__
            _update_check_cache = (False, __version__, __version__)
        except Exception:
            _update_check_cache = (False, "unknown", "unknown")
        return _update_check_cache


def _get_update_notice() -> str:
    """Get update notice string if newer version available, else empty string."""
    update_available, current, latest = _check_for_updates()
    if update_available:
        return f"\n⚠️ **Update available:** roampal {latest} (you have {current})\n   Run: `pip install --upgrade roampal`\n"
    return ""


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
    logger.warning(f"FastAPI hook server not responding on port {port}, attempting restart...")
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
    """
    import httpx
    port = _get_port()
    url = f"http://127.0.0.1:{port}{path}"
    async with httpx.AsyncClient() as client:
        if method == "GET":
            response = await client.get(url, timeout=timeout)
        else:
            response = await client.post(url, json=payload or {}, timeout=timeout)
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
        return [
            types.Tool(
                name="search_memory",
                description="""Search your persistent memory. Use when you need details beyond what KNOWN CONTEXT provided.

WHEN TO SEARCH:
• User says "remember", "I told you", "we discussed" → search immediately
• You need more detail than the context provided
• You see [id:...] in KNOWN CONTEXT and want full details → use id= parameter

WHEN NOT TO SEARCH:
• General knowledge questions (use your training)
• KNOWN CONTEXT already gave you the answer

Collections: working (24h then auto-promotes), history (30d scored), patterns (permanent scored), memory_bank (permanent), books (permanent docs)
Omit 'collections' parameter for auto-routing (recommended).

TEMPORAL SEARCH:
Use days_back=N to search by time without a semantic query.
Examples: days_back=7 → last 7 days. days_back=1, collections=["working"] → today's working memory.
Combine with query for time-filtered semantic search: query="auth", days_back=14.

ID LOOKUP:
Use id="patterns_abc123" to look up a specific memory directly (bypasses search).
Useful when you see [id:...] tags in KNOWN CONTEXT and want full metadata.

READING RESULTS:
• [YYN] = outcome history (last 3: Y=worked, ~=partial, N=failed)
• s:0.7 = outcome score (0-1, higher = more successful outcomes)
• w:0.68 = Wilson confidence score (statistical lower bound, 0-1)
• 8 uses = times this memory has been surfaced and scored
• 5d = age of memory
• [id:patterns_abc123] = memory ID for lookup or scoring

VERIFICATION USE:
• Before stating a fact from memory as definitive, search for it
• If you find conflicting memories on the same topic, prefer the
  newer one and score the older one "failed" on the next scoring prompt
• Use id= lookup to verify specific memories referenced in KNOWN CONTEXT
  before building conclusions on them""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - use the users EXACT words/phrases, do NOT simplify or extract keywords"
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
                            "description": "Which collections to search. Omit for auto-routing (recommended). Manual: books, working, history, patterns, memory_bank",
                            "default": None
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
                            "description": "Optional filters. Use sparingly. Examples: created_at='2025-11-12', last_outcome='worked', has_code=true",
                            "additionalProperties": True
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["relevance", "recency", "score"],
                            "description": "Sort order. 'recency' for temporal queries like 'last thing we did'. Auto-detected if omitted.",
                            "default": None
                        }
                    },
                    "required": []
                }
            ),
            types.Tool(
                name="add_to_memory_bank",
                description="""Store PERMANENT facts that help maintain continuity across sessions.

WHAT BELONGS HERE:
• User identity (name, role, background) - MUST use tags=["identity"]
• Preferences (communication style, tools, workflows)
• Goals and projects (what they're working on, priorities)
• Progress tracking (what worked, what failed, strategy iterations)
• Useful context that would be lost between sessions

WHAT DOES NOT BELONG:
• Raw conversation exchanges (auto-captured in working/history)
• Temporary session facts (current task details)
• Every fact you hear - be SELECTIVE, this is for PERMANENT knowledge

TAG MEANINGS:
• identity - name, role, background (REQUIRED for cold start detection)
• preference - communication style, tools they like, how they work
• goal - what they're trying to achieve long-term
• project - current projects, codebases, repos
• system_mastery - things you learned about being effective for THIS user
• agent_growth - meta-learning about how to improve as an assistant

ALWAYS_INJECT:
Use sparingly. Only for facts needed on EVERY message (core identity).
Most facts should NOT be always_inject - they surface via semantic search when relevant.

SIZE GUIDANCE:
• Keep facts AS SMALL AS POSSIBLE - aim for ~300 chars or less
• The first ~300 chars of each fact appear in cold start profile summaries
• Longer facts work but only the beginning shows on cold start - put the key info first
• Research dumps belong in books collection, not memory_bank
• If you notice massive facts (1000+ chars), offer to condense them
• One concept per fact - split multi-topic content into separate memories

Rule of thumb: If it helps maintain continuity across sessions OR enables learning/improvement, store it. If it's session-specific, don't.

Note: memory_bank is NOT outcome-scored. Facts persist until deleted.

STORAGE DISCIPLINE:
• confidence=0.9+ is reserved for facts you VERIFIED against source
  (code, docs, tool output). Not for things you "know"
• confidence=0.7 is the default — use it unless you have evidence
• confidence=0.5 for unverified claims, approximate numbers, or
  things the user told you that you couldn't confirm
• Before storing, ask: "Will this still be true next week?" If not,
  it doesn't belong in memory_bank — let working/history handle it
• One fact per memory. Don't combine unrelated information""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact to remember"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Categories: identity, preference, goal, project, system_mastery, agent_growth"},
                        "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "How critical (0.0-1.0)"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "How certain (0.0-1.0)"},
                        "always_inject": {"type": "boolean", "default": False, "description": "If true, this memory appears in EVERY context (use for core identity only)"}
                    },
                    "required": ["content"]
                }
            ),
            types.Tool(
                name="update_memory",
                description="Update existing memory when information changes or needs correction.\n\nWHEN TO USE: When you discover a memory_bank fact is outdated, wrong, or incomplete — fix it immediately. Don't wait to be told. If you gave wrong info because of a stale memory, update it in the same turn you discover the error. This is YOUR memory — maintain it.\n\nUPDATE vs DELETE: Update when the fact is still relevant but details changed (version, path, status). Delete when the fact is no longer relevant at all or has been replaced by a different memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "old_content": {"type": "string", "description": "Old/incorrect fact to find"},
                        "new_content": {"type": "string", "description": "Corrected/updated fact"}
                    },
                    "required": ["old_content", "new_content"]
                }
            ),
            types.Tool(
                name="delete_memory",
                description="Delete outdated/irrelevant memories from memory_bank.\n\nWHEN TO USE: When a memory_bank fact is wrong, stale, redundant, or harmful. Deleting bad memories is as important as storing good ones. If a memory has been misleading you across sessions, don't just score it 'failed' — delete it. memory_bank is not outcome-scored, so scoring alone won't remove it. You must act directly.\n\nUPDATE vs DELETE: Delete when the fact is completely wrong or no longer relevant. Update (use update_memory instead) when the fact is still relevant but details changed.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory to delete (semantic match)"}
                    },
                    "required": ["content"]
                }
            ),
            types.Tool(
                name="score_memories",
                description=(
                    # v0.3.6: OpenCode gets a lean description — per-memory scoring ONLY
                    # Exchange summary + outcome are handled by the sidecar, not the main LLM
                    """Score cached memories from your previous context. ONLY call when <roampal-score-required> appears.

Score each memory ID listed in the scoring prompt:
• worked = this memory was helpful
• partial = somewhat helpful
• unknown = didn't use this memory
• failed = this memory was MISLEADING

⚠️ "failed" means MISLEADING, not just unused. If you didn't use it, mark "unknown".

Memory IDs correspond to [id:...] tags in KNOWN CONTEXT from the previous turn."""
                    if _is_opencode else
                    """Score individual cached memories from your previous context. ONLY use when the <roampal-score-required> hook prompt appears.

Your job here is to:
1. Score each cached memory individually — was it helpful, misleading, or unused?
2. Summarize the previous exchange in ~300 chars
3. Rate the exchange outcome (worked/failed/partial/unknown)

⚠️ IMPORTANT: This tool is ONLY for scoring when prompted by the hook. Do NOT call it at other times.
For storing important learnings at any time, use record_response(key_takeaway="...") instead.

FINDING MEMORY IDs:
The scoring prompt lists memory IDs to score. These IDs correspond to [id:...] tags
in the KNOWN CONTEXT block from the previous turn. Look at KNOWN CONTEXT to see what
each memory contained, then score based on whether it helped your response.

PER-MEMORY SCORING:
You MUST score each memory ID listed in the scoring prompt:
• worked = this memory was helpful
• partial = somewhat helpful
• unknown = didn't use this memory
• failed = this memory was MISLEADING (gave bad advice that led you astray)

You MAY also score any other memory visible in KNOWN CONTEXT or earlier conversation.

⚠️ CRITICAL - "failed" means MISLEADING, not just unused.
If you didn't use a memory, mark it "unknown" not "failed".

ACTIVE MEMORY MANAGEMENT:
• "failed" is not punishment — it's pruning. A memory scored "failed"
  3 times will be demoted or deleted. This is GOOD. Use it.
• If a memory was MISLEADING in your response (you gave wrong info
  because of it), score it "failed" even if the exchange went okay
• If you notice a pattern memory surfacing repeatedly with "unknown"
  scores, consider: is this memory actually useful? If not, scoring
  "failed" once helps the system stop wasting context on it
• You are the gardener. Pull the weeds."""
                ),
                inputSchema=(
                    # OpenCode: memory_scores only, no exchange fields
                    {
                        "type": "object",
                        "properties": {
                            "memory_scores": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "string",
                                    "enum": ["worked", "failed", "partial", "unknown"]
                                },
                                "description": "Score for each memory: doc_id -> outcome. MUST include all cached memories."
                            }
                        },
                        "required": ["memory_scores"]
                    }
                    if _is_opencode else
                    {
                        "type": "object",
                        "properties": {
                            "memory_scores": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "string",
                                    "enum": ["worked", "failed", "partial", "unknown"]
                                },
                                "description": "Score for each memory: doc_id -> outcome. MUST include all cached memories. MAY include extras from context."
                            },
                            "exchange_summary": {
                                "type": "string",
                                "description": "~300 char summary of the previous exchange"
                            },
                            "exchange_outcome": {
                                "type": "string",
                                "enum": ["worked", "failed", "partial", "unknown"],
                                "description": "Was the previous response effective?"
                            }
                        },
                        "required": ["memory_scores"]
                    }
                )
            ),
            types.Tool(
                name="record_response",
                description="""Store a key takeaway when the transcript alone won't capture important learning.

OPTIONAL - Only use for significant exchanges:
• Major decisions made
• Complex solutions that worked
• User corrections (what you got wrong and why)
• Important context that would be lost

Most routine exchanges don't need this - the transcript is enough.

Key takeaways start at 0.7 (user explicitly asked to remember = higher confidence).
Scoring happens via score_memories on the next turn: +0.2 worked, +0.05 partial, -0.3 failed.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key_takeaway": {
                            "type": "string",
                            "description": "1-2 sentence summary of the important learning"
                        }
                    },
                    "required": ["key_takeaway"]
                }
            ),
            # v0.3.6: get_context_insights REMOVED — hooks/plugin already inject KNOWN CONTEXT
            # before the model responds. The tool was redundant (same server call twice) and caused
            # weaker models (qwen) to loop on tool calls. Context injection is now fully automatic:
            # - Claude Code: UserPromptSubmit hook → /api/hooks/get-context → system-reminder
            # - OpenCode: chat.message → /api/hooks/get-context → system.transform
            # Self-audit moved to hook context injection. Memory system docs in tool descriptions.
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Handle MCP tool calls — all proxied through FastAPI (v0.3.2)."""
        # Ensure FastAPI hook server is healthy before any operation
        _ensure_server_running(timeout=3.0)

        try:
            if name == "search_memory":
                query = arguments.get("query", "")
                collections = arguments.get("collections")
                limit = arguments.get("limit", 5)
                metadata = arguments.get("metadata")
                sort_by = arguments.get("sort_by")
                days_back = arguments.get("days_back")
                doc_id = arguments.get("id")

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

                        # Memory bank: show importance/confidence
                        elif collection == "memory_bank":
                            try:
                                imp = float(r.get("importance", metadata.get("importance", 0.7)) or 0.7)
                                conf = float(r.get("confidence", metadata.get("confidence", 0.7)) or 0.7)
                            except (ValueError, TypeError):
                                imp, conf = 0.7, 0.7
                            meta_parts.append(f"imp:{imp:.1f}")
                            meta_parts.append(f"conf:{conf:.1f}")

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
                importance = arguments.get("importance", 0.7)
                confidence = arguments.get("confidence", 0.7)
                always_inject = arguments.get("always_inject", False)

                result = await _api_call("POST", "/api/memory-bank/add", {
                    "content": content,
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

                if not key_takeaway:
                    return [types.TextContent(
                        type="text",
                        text="Error: 'key_takeaway' is required"
                    )]

                result = await _api_call("POST", "/api/record-response", {
                    "key_takeaway": key_takeaway,
                    "conversation_id": _mcp_session_id
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
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Roampal MCP Server")
    parser.add_argument("--dev", action="store_true", help="Run in dev mode (port 27183, separate data)")
    args = parser.parse_args()

    # Check BOTH --dev flag AND ROAMPAL_DEV env var
    is_dev = args.dev or os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")

    logging.basicConfig(level=logging.INFO)
    run_mcp_server(dev=is_dev)
