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
- get_context_insights: Get context before responding (user profile, relevant memories)
- search_memory: Search across memory collections (for detailed lookups)
- add_to_memory_bank: Store permanent user facts
- update_memory: Update existing memories
- delete_memory: Delete outdated memories
- record_response: Complete the interaction (key_takeaway + outcome scoring)

WORKFLOW:
1. get_context_insights(query) - Get what you know about this topic
2. search_memory() - If you need more details
3. Respond to user
4. record_response(key_takeaway, outcome) - Close the loop for learning

HOW IT WORKS:
- MCP server auto-starts FastAPI hook server on port 27182 (background thread)
- Hooks auto-inject relevant memories into your context (invisible to user)
- Cold start: First message of session dumps full user profile
- Scoring: record_response scores cached memories based on outcome
- Learning: Good memories get promoted, bad ones get demoted/deleted
- 5 collections: books (docs), memory_bank (facts), patterns (proven), history (past), working (session)
"""

import logging
import json
import asyncio
import threading
import socket
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from roampal.backend.modules.memory import UnifiedMemorySystem

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



# Port configuration - DEV and PROD use different ports
PROD_PORT = 27182
DEV_PORT = 27183

# Global memory system
_memory: Optional[UnifiedMemorySystem] = None

# Session cache for outcome tracking
_mcp_search_cache: Dict[str, Dict[str, Any]] = {}

# Flag to track if FastAPI server is running
_fastapi_started = False

# Dev mode flag (set via command line or env)
_dev_mode = False

# Cache for update check (only check once per session)
_update_check_cache: Optional[tuple] = None


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


def _start_fastapi_server():
    """
    Start FastAPI hook server in background subprocess.

    This enables hooks to work without requiring a separate 'roampal start' command.
    When Claude Code launches the MCP server, the hook server starts automatically.

    Uses subprocess instead of threading to avoid event loop conflicts between
    uvicorn and the MCP server's asyncio loop.
    """
    global _fastapi_started

    if _fastapi_started:
        return

    # Use correct port based on dev mode
    port = DEV_PORT if _dev_mode else PROD_PORT

    # Check if port is already in use (server already running externally)
    if _is_port_in_use(port):
        logger.info(f"FastAPI hook server already running on port {port}")
        _fastapi_started = True
        return

    import subprocess

    try:
        # Build environment - pass through dev mode
        env = os.environ.copy()
        if _dev_mode:
            env["ROAMPAL_DEV"] = "1"

        # Start FastAPI server as a subprocess with correct port
        # Use the same Python that's running this MCP server
        cmd = [sys.executable, "-m", "roampal.server.main", "--port", str(port)]

        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Don't inherit stdin (MCP uses it)
            stdin=subprocess.DEVNULL,
            # Pass environment with dev flag
            env=env,
            # Detach from parent process group on Windows
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        _fastapi_started = True
        mode = "DEV" if _dev_mode else "PROD"
        logger.info(f"Started FastAPI hook server on port {port} ({mode} mode)")
    except Exception as e:
        logger.error(f"Failed to start FastAPI hook server: {e}")


def _detect_mcp_client() -> str:
    """Detect MCP client for session tracking."""
    # Use "default" to match the hook's fallback conversation_id
    # This ensures MCP tool calls and hook injections share the same cache
    return "default"


async def _initialize_memory():
    """Initialize memory system if needed."""
    global _memory
    if _memory is None:
        _memory = UnifiedMemorySystem()
        await _memory.initialize()
        logger.info("MCP: Memory system initialized")


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

    # Start FastAPI hook server in background thread
    _start_fastapi_server()

    server = Server("roampal")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """List available MCP tools."""
        return [
            types.Tool(
                name="search_memory",
                description="""Search your persistent memory. Use when you need details beyond what get_context_insights returned.

WHEN TO SEARCH:
• User says "remember", "I told you", "we discussed" → search immediately
• get_context_insights recommended a collection → search that collection
• You need more detail than the context provided

WHEN NOT TO SEARCH:
• General knowledge questions (use your training)
• get_context_insights already gave you the answer

Collections: working (24h then auto-promotes), history (30d scored), patterns (permanent scored), memory_bank (permanent), books (permanent docs)
Omit 'collections' parameter for auto-routing (recommended).""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - use the users EXACT words/phrases, do NOT simplify or extract keywords"
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
                            "description": "Optional filters. Use sparingly. Examples: timestamp='2025-11-12', last_outcome='worked', has_code=true",
                            "additionalProperties": True
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="add_to_memory_bank",
                description="""Store PERMANENT facts that help maintain continuity across sessions.

WHAT BELONGS HERE:
• User identity (name, role, background)
• Preferences (communication style, tools, workflows)
• Goals and projects (what they're working on, priorities)
• Progress tracking (what worked, what failed, strategy iterations)
• Useful context that would be lost between sessions

WHAT DOES NOT BELONG:
• Raw conversation exchanges (auto-captured in working/history)
• Temporary session facts (current task details)
• Every fact you hear - be SELECTIVE, this is for PERMANENT knowledge

Rule of thumb: If it helps maintain continuity across sessions OR enables learning/improvement, store it. If it's session-specific, don't.

Note: memory_bank is NOT outcome-scored. Facts persist until archived.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact to remember"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Categories: identity, preference, goal, project, system_mastery, agent_growth"},
                        "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "How critical (0.0-1.0)"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "How certain (0.0-1.0)"}
                    },
                    "required": ["content"]
                }
            ),
            types.Tool(
                name="update_memory",
                description="Update existing memory when information changes or needs correction.",
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
                description="Delete outdated/irrelevant memories from memory_bank.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory to delete (semantic match)"}
                    },
                    "required": ["content"]
                }
            ),
            types.Tool(
                name="score_response",
                description="""Score the previous exchange. ONLY use when the <roampal-score-required> hook prompt appears.

⚠️ IMPORTANT: This tool is ONLY for scoring when prompted by the hook. Do NOT call it at other times.
For storing important learnings at any time, use record_response(key_takeaway="...") instead.

OUTCOME DETECTION (read user's reaction):
✓ worked = user satisfied, says thanks, moves on
✗ failed = user corrects you, says "no", "that's wrong", provides the right answer
~ partial = user says "kind of" or takes some but not all of your answer
? unknown = no clear signal from user

⚠️ CRITICAL - "failed" OUTCOMES ARE ESSENTIAL:
• If user says you were wrong → outcome="failed"
• If memory you retrieved was outdated → outcome="failed"
• If user had to correct you → outcome="failed"
• If you gave advice that didn't help → outcome="failed"

Failed outcomes are how bad memories get deleted. Without them, wrong info persists forever.
Don't default to "worked" just to be optimistic. Wrong memories MUST be demoted.

SELECTIVE SCORING (optional):
If the scoring prompt shows "Memories surfaced:", you can specify which were actually relevant:
• related=["doc_id_1", "doc_id_2"] → only those get scored
• Omit related → all surfaced memories get scored (backwards compatible)
Unrelated memories get 0 (neutral) - they're not penalized, just skipped.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "enum": ["worked", "failed", "partial", "unknown"],
                            "description": "How helpful was your response based on user's reaction"
                        },
                        "related": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: doc_ids of memories that were actually relevant. Omit to score all surfaced memories."
                        }
                    },
                    "required": ["outcome"]
                }
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

INITIAL SCORING (optional):
You can score the takeaway at creation time based on the current exchange:
• initial_score="worked" → starts at 0.7 (boosted)
• initial_score="failed" → starts at 0.2 (demoted, but still stored as "what not to do")
• Omit → starts at 0.5 (neutral default)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key_takeaway": {
                            "type": "string",
                            "description": "1-2 sentence summary of the important learning"
                        },
                        "initial_score": {
                            "type": "string",
                            "enum": ["worked", "failed", "partial"],
                            "description": "Optional: Score based on current exchange outcome. Omit for neutral 0.5 start."
                        }
                    },
                    "required": ["key_takeaway"]
                }
            ),
            types.Tool(
                name="get_context_insights",
                description="""Search your memory before responding. Returns what you know about this user/topic.

WORKFLOW (follow these steps):
1. get_context_insights(query) ← YOU ARE HERE
2. Read the context returned
3. search_memory() if you need more details
4. Respond to user
5. record_response() to complete

Returns: Known facts, past solutions, recommended collections, tool stats.
Uses semantic search across your memory collections.

PROACTIVE MEMORY: If you learn something NEW about the user during the conversation
(name, preference, goal, project context), use add_to_memory_bank() to store it.
Don't wait to be asked - good assistants remember what matters.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query/topic you're considering (use user's exact words)"
                        }
                    },
                    "required": ["query"]
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Handle MCP tool calls."""
        await _initialize_memory()
        session_id = _detect_mcp_client()

        try:
            if name == "search_memory":
                query = arguments.get("query", "")
                collections = arguments.get("collections")
                limit = arguments.get("limit", 5)
                metadata = arguments.get("metadata")

                try:
                    results = await _memory.search(
                        query=query,
                        collections=collections,
                        limit=limit,
                        metadata_filters=metadata
                    )
                except Exception as search_err:
                    return [types.TextContent(
                        type="text",
                        text=f"Search error: {search_err}\n\nData path: {_memory.data_path}\nCollections: {list(_memory.collections.keys())}"
                    )]

                # Cache doc_ids for outcome scoring (last call only)
                cached_doc_ids = [r.get("id") for r in results if r.get("id")]
                _mcp_search_cache[session_id] = {
                    "doc_ids": cached_doc_ids,
                    "query": query,
                    "timestamp": datetime.now()
                }

                if not results:
                    # Debug info
                    coll_counts = {name: coll.collection.count() if coll.collection else 0 for name, coll in _memory.collections.items()}
                    text = f"No results found for '{query}'.\n\nDebug: data_path={_memory.data_path}, collections={coll_counts}"
                else:
                    text = f"Found {len(results)} result(s) for '{query}':\n\n"
                    for i, r in enumerate(results[:limit], 1):
                        metadata = r.get("metadata", {})
                        doc_id = r.get("id", "")
                        # Content can be in multiple places
                        content = r.get("content") or metadata.get("content") or metadata.get("text") or r.get("text", "")
                        collection = r.get("collection", "unknown")

                        # Build metadata parts
                        meta_parts = []

                        # Age from created_at
                        age = _humanize_age(metadata.get("created_at", ""))
                        if age:
                            meta_parts.append(age)

                        # Outcome history for scored collections
                        if collection in ["patterns", "history", "working"]:
                            score = metadata.get("score", 0.5)
                            outcomes = _format_outcomes(metadata.get("outcome_history", ""))
                            meta_parts.append(f"s:{score:.1f}")
                            if outcomes:
                                meta_parts.append(outcomes)

                        meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
                        id_str = f" [id:{doc_id[:8]}]" if doc_id else ""
                        text += f"{i}. [{collection}]{meta_str}{id_str} {content}\n\n"

                return [types.TextContent(type="text", text=text)]

            elif name == "add_to_memory_bank":
                content = arguments.get("content")
                tags = arguments.get("tags", [])
                importance = arguments.get("importance", 0.7)
                confidence = arguments.get("confidence", 0.7)

                doc_id = await _memory.store_memory_bank(
                    text=content,
                    tags=tags,
                    importance=importance,
                    confidence=confidence
                )

                return [types.TextContent(
                    type="text",
                    text=f"Added to memory bank (ID: {doc_id})"
                )]

            elif name == "update_memory":
                old_content = arguments.get("old_content", "")
                new_content = arguments.get("new_content", "")

                doc_id = await _memory.update_memory_bank(
                    old_content=old_content,
                    new_content=new_content
                )

                if doc_id:
                    return [types.TextContent(
                        type="text",
                        text=f"Updated memory (ID: {doc_id})"
                    )]
                else:
                    return [types.TextContent(
                        type="text",
                        text="Memory not found for update"
                    )]

            elif name == "delete_memory":
                content = arguments.get("content", "")

                success = await _memory.delete_memory_bank(content)

                if success:
                    return [types.TextContent(
                        type="text",
                        text="Memory deleted successfully"
                    )]
                else:
                    return [types.TextContent(
                        type="text",
                        text="Memory not found for deletion"
                    )]

            elif name == "score_response":
                outcome = arguments.get("outcome", "unknown")
                related = arguments.get("related")  # Optional list of doc_ids to score

                # Score via FastAPI endpoint (has access to hook-injected doc_ids cache)
                # Use correct port based on dev mode
                port = DEV_PORT if _dev_mode else PROD_PORT
                scored_count = 0
                try:
                    import httpx
                    payload = {
                        "conversation_id": session_id,
                        "outcome": outcome
                    }
                    # Only include related if explicitly provided (backwards compatible)
                    if related is not None:
                        payload["related"] = related

                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"http://127.0.0.1:{port}/api/record-outcome",
                            json=payload,
                            timeout=5.0
                        )
                        if response.status_code == 200:
                            result = response.json()
                            scored_count = result.get("documents_scored", 0)
                except Exception as e:
                    logger.warning(f"Failed to call FastAPI record-outcome: {e}")

                # Fall back to MCP cache if FastAPI returned 0 or failed
                if scored_count == 0 and session_id in _mcp_search_cache:
                        cached = _mcp_search_cache[session_id]
                        doc_ids = cached.get("doc_ids", [])
                        # Apply related filter if provided
                        if related is not None:
                            doc_ids = [d for d in doc_ids if d in related]
                        if doc_ids:
                            result = await _memory.record_outcome(doc_ids, outcome)
                            scored_count = result.get("documents_updated", 0)
                        del _mcp_search_cache[session_id]

                logger.info(f"Scored response: outcome={outcome}, related={related}, scored={scored_count}")
                return [types.TextContent(
                    type="text",
                    text=f"Scored (outcome={outcome}, {scored_count} memories updated)"
                )]

            elif name == "record_response":
                key_takeaway = arguments.get("key_takeaway", "")
                initial_score = arguments.get("initial_score")  # Optional: worked, failed, partial

                if not key_takeaway:
                    return [types.TextContent(
                        type="text",
                        text="Error: 'key_takeaway' is required"
                    )]

                # Calculate starting score based on initial_score
                # Matches score deltas from ARCHITECTURE.md: worked +0.20, failed -0.30, partial +0.05
                starting_score = 0.5  # neutral default
                if initial_score == "worked":
                    starting_score = 0.7  # 0.5 + 0.2
                elif initial_score == "failed":
                    starting_score = 0.2  # 0.5 - 0.3
                elif initial_score == "partial":
                    starting_score = 0.55  # 0.5 + 0.05

                # Store the takeaway in working memory
                doc_id = await _memory.store_working(
                    content=f"Key takeaway: {key_takeaway}",
                    conversation_id=session_id,
                    metadata={
                        "type": "key_takeaway",
                        "timestamp": datetime.now().isoformat(),
                        "initial_outcome": initial_score
                    },
                    initial_score=starting_score
                )
                logger.info(f"Recorded takeaway (score={starting_score}): {key_takeaway[:50]}...")
                return [types.TextContent(
                    type="text",
                    text=f"Recorded: {key_takeaway}"
                )]

            elif name == "get_context_insights":
                query = arguments.get("query", "")

                if not query:
                    return [types.TextContent(
                        type="text",
                        text="Error: 'query' is required"
                    )]

                # Get context from memory system
                context = await _memory.get_context_for_injection(query)

                # Cache doc_ids for scoring (last call only)
                cached_doc_ids = context.get("doc_ids", [])
                if cached_doc_ids:
                    _mcp_search_cache[session_id] = {
                        "doc_ids": cached_doc_ids,
                        "query": query,
                        "source": "get_context_insights",
                        "timestamp": datetime.now()
                    }

                # Format response
                user_facts = context.get("user_facts", [])
                memories = context.get("relevant_memories", [])

                text = f"Known Context for '{query}':\n\n"

                if user_facts:
                    text += "**Memory Bank:**\n"
                    for fact in user_facts:
                        text += f"• {fact.get('content', '')}\n"
                    text += "\n"

                if memories:
                    text += "**Relevant Memories:**\n"
                    for mem in memories:
                        coll = mem.get("collection", "unknown")
                        content = mem.get("content") or mem.get("metadata", {}).get("content", "")
                        score = mem.get("metadata", {}).get("score", 0.5)
                        text += f"• [{coll}] (score:{score:.2f}) {content}\n"
                    text += "\n"

                if not user_facts and not memories:
                    text += "No relevant context found. This may be a new topic or first interaction.\n"

                text += f"\n_Cached {len(cached_doc_ids)} doc_ids for outcome scoring._"

                # Add update notice if available (checked once per session)
                text += _get_update_notice()

                return [types.TextContent(type="text", text=text)]

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

    logging.basicConfig(level=logging.INFO)
    run_mcp_server(dev=args.dev)
