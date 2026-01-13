"""
Roampal FastAPI Server

Provides:
- /api/hooks/get-context: Called by UserPromptSubmit hook (injects scoring prompt)
- /api/hooks/stop: Called by Stop hook (stores exchange, enforces record_response)
- /api/health: Health check endpoint
- MCP server (optional, for power users)

Hook Enforcement Flow:
1. UserPromptSubmit hook calls /api/hooks/get-context
   - If previous exchange unscored, injects scoring prompt
   - Adds relevant memories to context
2. LLM MUST call record_response(outcome) to score previous exchange
3. Stop hook calls /api/hooks/stop
   - Stores current exchange with doc_id
   - Can block (exit 2) if record_response wasn't called
"""

import logging
import json
import asyncio
import os
import sys
from pathlib import Path

# Fix Windows encoding issues with unicode characters (emojis, box drawing, etc.)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass  # Ignore if already reconfigured or in test environment
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import memory system and session manager
from roampal.backend.modules.memory import UnifiedMemorySystem
from roampal.backend.modules.memory.unified_memory_system import ActionOutcome
from roampal.backend.modules.memory.content_graph import ContentGraph
from roampal.hooks import SessionManager

logger = logging.getLogger(__name__)

# Global instances
_memory: Optional[UnifiedMemorySystem] = None
_session_manager: Optional[SessionManager] = None

# Search result cache for outcome scoring (session_id -> doc_ids)
_search_cache: Dict[str, Dict[str, Any]] = {}

# Port constants for dev/prod isolation
PROD_PORT = 27182
DEV_PORT = 27183

# Cache for update check (only check once per server session)
_update_check_cache: Optional[tuple] = None

# Cold start tag priorities - one fact per category (v0.2.7)
TAG_PRIORITIES = ["identity", "preference", "goal", "project", "system_mastery", "agent_growth"]

# v0.2.8: Parent process monitoring for lifecycle management
_parent_monitor_task: Optional[asyncio.Task] = None


def _is_parent_alive(parent_pid: int) -> bool:
    """
    v0.2.8: Check if parent process is still alive. Cross-platform, no dependencies.

    Uses OS-native APIs:
    - Windows: ctypes + kernel32 OpenProcess/GetExitCodeProcess
    - Unix: os.kill with signal 0
    """
    if sys.platform == "win32":
        # Windows: Use ctypes to check process
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, parent_pid)
            if not handle:
                return False  # Can't open = doesn't exist

            exit_code = ctypes.c_ulong()
            result = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)

            if not result:
                return False
            return exit_code.value == STILL_ACTIVE
        except Exception:
            # If ctypes fails, assume parent is alive (fail open)
            return True
    else:
        # Unix: signal 0 checks if process exists without actually sending a signal
        try:
            os.kill(parent_pid, 0)
            return True
        except OSError:
            return False


async def _monitor_parent_process():
    """
    v0.2.8: Exit if parent MCP process dies.

    Runs every 2 seconds. If parent PID no longer exists, FastAPI exits.
    This catches SIGKILL and crashes that bypass atexit handlers.

    No external dependencies - uses ctypes (Windows) or os.kill (Unix).
    """
    parent_pid = os.getppid()
    logger.info(f"v0.2.8: Monitoring parent process {parent_pid} for lifecycle management")

    while True:
        await asyncio.sleep(2)  # Check every 2 seconds

        if not _is_parent_alive(parent_pid):
            logger.info(f"Parent process {parent_pid} died. Shutting down FastAPI server.")
            # Use os._exit for immediate termination (no cleanup needed for zombie prevention)
            os._exit(0)


def _first_sentence(text: str, max_chars: int = 300) -> str:
    """
    Extract first sentence from text, capped at max_chars.

    v0.2.7: Used for cold start truncation - prevents massive facts
    from overwhelming context. Full facts still available via search.
    Bumped from 150 to 300 chars for better summary context.
    """
    if not text:
        return ""
    # Find first sentence ending
    for end_char in ['. ', '.\n', '!', '?']:
        idx = text.find(end_char)
        if idx > 0:
            first = text[:idx + 1].strip()
            if len(first) <= max_chars:
                return first
            break
    # No sentence ending found or sentence too long - truncate
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rsplit(' ', 1)[0] + "..."


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


def _get_update_injection() -> Optional[str]:
    """Get update notice for hook injection if newer version available.

    Returns directive for Claude to tell the user about the update.
    """
    update_available, current, latest = _check_for_updates()
    if update_available:
        return f"""<roampal-update>
A newer version of roampal is available: {latest} (user has {current}).
IMPORTANT: Briefly mention this to the user. Say something like:
"Quick note: roampal {latest} is available. Run `pip install --upgrade roampal` to update."
Only mention once per conversation.
</roampal-update>
"""
    return None


async def _build_cold_start_profile() -> Optional[str]:
    """
    Build the cold start user profile injection.

    v0.2.7: Lean but rich - 10 facts max with balanced coverage:
    1. Always-inject memories (identity core)
    2. One fact per tag category (identity, preference, goal, project, system_mastery, agent_growth)
    3. (Future) Content KG entities - read path exists but entity extraction not yet wired up

    Returns:
        Formatted user profile string, or None if no facts exist
    """
    if not _memory:
        return None

    try:
        # Get all facts from memory_bank, sorted by quality (importance, confidence)
        all_memory_bank = _memory._memory_bank_service.list_all(include_archived=False)

        # Sort by quality score (importance first, then confidence)
        sorted_facts = sorted(
            all_memory_bank,
            key=lambda f: (
                f.get("metadata", {}).get("importance", 0.5),
                f.get("metadata", {}).get("confidence", 0.5)
            ),
            reverse=True
        )

        # Pick HIGHEST QUALITY fact for EACH tag category (one per tag)
        all_facts = []
        seen_tags = set()
        for fact in sorted_facts:
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except:
                    tags = []
            else:
                tags = tags_raw or []

            # Find which priority tag this fact matches (if any)
            for tag in TAG_PRIORITIES:
                if tag in tags and tag not in seen_tags:
                    all_facts.append(fact)
                    seen_tags.add(tag)
                    break  # One fact per tag

            # Stop once we have one fact per tag category
            if len(seen_tags) == len(TAG_PRIORITIES):
                break

        # Check if user has NO identity at all
        identity_content = []
        for fact in all_facts:
            content = fact.get("text") or fact.get("content") or fact.get("metadata", {}).get("content", "")
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except:
                    tags = []
            else:
                tags = tags_raw or []
            if "identity" in tags:
                identity_content.append(content)

        if not identity_content:
            # Check if they have history (existing user) or not (truly new)
            has_history = await _memory.search(
                query="",
                collections=["history", "patterns"],
                limit=1
            )

            if has_history:
                return """<roampal-identity-missing>
You've been working with this user but don't have their identity stored yet.

When natural, consider asking their name to personalize future sessions. Store with:
  add_to_memory_bank(content="User's name is [NAME]", tags=["identity"])

No rush - just when it fits the conversation.
</roampal-identity-missing>
"""
            else:
                return """<roampal-new-user>
NEW USER: You don't have any stored information about this user yet.

Consider naturally asking for their name and what they're working on. Store with:
  add_to_memory_bank(content="User's name is [NAME]", tags=["identity"])
  add_to_memory_bank(content="Working on [PROJECT]", tags=["project"])

Keep it conversational - don't interrogate. A simple "I don't think we've met - what's your name?" works.
</roampal-new-user>
"""

        # Build narrative profile - one line per category (compact ~200 chars)
        tag_labels = {
            "identity": "Identity",
            "preference": "Preference",
            "goal": "Goal",
            "project": "Project",
            "system_mastery": "System Mastery",
            "agent_growth": "Agent Growth"
        }

        # Group facts by their primary tag, keeping only FIRST fact per category
        category_facts = {}
        for fact in all_facts:
            content = fact.get("text") or fact.get("content") or fact.get("metadata", {}).get("content", "")
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except:
                    tags = []
            else:
                tags = tags_raw or []

            for tag in TAG_PRIORITIES:
                if tag in tags and tag not in category_facts:
                    category_facts[tag] = content
                    break

        # Build compact narrative
        profile_parts = ["<roampal-user-profile>"]
        for tag in TAG_PRIORITIES:
            if tag in category_facts:
                profile_parts.append(f"{tag_labels[tag]}: {_first_sentence(category_facts[tag])}")
        profile_parts.append("</roampal-user-profile>")

        logger.info(f"Cold start: {len(all_facts)} facts, {len(category_facts)} categories")
        return "\n".join(profile_parts)

    except Exception as e:
        logger.error(f"Error building cold start profile: {e}")
        return None


# ==================== Request/Response Models ====================

class GetContextRequest(BaseModel):
    """Request for hook context injection."""
    query: str
    conversation_id: Optional[str] = None
    recent_messages: Optional[List[Dict[str, Any]]] = None


class GetContextResponse(BaseModel):
    """Response with context to inject."""
    formatted_injection: str
    user_facts: List[Dict[str, Any]]
    relevant_memories: List[Dict[str, Any]]
    context_summary: str
    scoring_required: bool = False  # True if previous exchange needs scoring


class StopHookRequest(BaseModel):
    """Request from Stop hook after LLM responds."""
    conversation_id: str
    user_message: str
    assistant_response: str
    transcript: Optional[str] = None  # Full transcript to check for record_response call


class StopHookResponse(BaseModel):
    """Response to Stop hook."""
    stored: bool
    doc_id: str
    scoring_complete: bool  # Did the LLM call record_response?
    should_block: bool  # Should hook block with exit code 2?
    block_message: Optional[str] = None


class SearchRequest(BaseModel):
    """Request for searching memory."""
    query: str
    conversation_id: Optional[str] = None
    collections: Optional[List[str]] = None
    limit: int = 10


class MemoryBankAddRequest(BaseModel):
    """Request to add to memory bank."""
    content: str
    tags: Optional[List[str]] = None
    importance: float = 0.7
    confidence: float = 0.7


class MemoryBankUpdateRequest(BaseModel):
    """Request to update memory bank."""
    old_content: str
    new_content: str


class RecordOutcomeRequest(BaseModel):
    """Request to record outcome for scoring."""
    conversation_id: str
    outcome: str  # worked, failed, partial, unknown
    related: Optional[List[str]] = None  # DEPRECATED: use memory_scores instead
    memory_scores: Optional[Dict[str, str]] = None  # v0.2.8: Per-memory scoring (doc_id -> outcome)


# ==================== Lifecycle ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage memory system lifecycle."""
    global _memory, _session_manager, _parent_monitor_task
    logger.info("Starting Roampal server...")

    # Check for dev mode or custom data path
    dev_mode = os.environ.get('ROAMPAL_DEV', '').lower() in ('1', 'true', 'yes')
    data_path = os.environ.get("ROAMPAL_DATA_PATH")

    # If dev mode and no custom path, use DEV data directory
    if dev_mode and not data_path:
        if os.name == 'nt':  # Windows
            appdata = os.environ.get('APPDATA', str(Path.home()))
            data_path = str(Path(appdata) / "Roampal_DEV" / "data")
        elif sys.platform == 'darwin':  # macOS
            data_path = str(Path.home() / "Library" / "Application Support" / "Roampal_DEV" / "data")
        else:  # Linux
            data_path = str(Path.home() / ".local" / "share" / "roampal_dev" / "data")
        logger.info(f"DEV MODE enabled - using: {data_path}")
    elif data_path:
        logger.info(f"Using custom data path: {data_path}")

    # Initialize memory system
    _memory = UnifiedMemorySystem(data_path=data_path)
    await _memory.initialize()
    logger.info("Memory system initialized")

    # v0.2.9: Cleanup legacy archived memories (one-time migration)
    if _memory._memory_bank_service:
        cleaned = _memory._memory_bank_service.cleanup_archived()
        if cleaned > 0:
            logger.info(f"v0.2.9 migration: cleaned up {cleaned} archived memories")

    # Initialize session manager (uses same data path)
    _session_manager = SessionManager(_memory.data_path)
    logger.info("Session manager initialized")

    # v0.2.8: Start parent process monitor (kills FastAPI when MCP dies)
    _parent_monitor_task = asyncio.create_task(_monitor_parent_process())

    yield

    # Cleanup
    logger.info("Shutting down Roampal server...")
    if _parent_monitor_task:
        _parent_monitor_task.cancel()
        try:
            await _parent_monitor_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Roampal",
        description="Persistent memory for AI coding tools",
        version="0.1.0",
        lifespan=lifespan
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==================== Hook Endpoints ====================

    @app.post("/api/hooks/get-context", response_model=GetContextResponse)
    async def get_context(request: GetContextRequest):
        """
        Called by UserPromptSubmit hook BEFORE the LLM sees the message.

        Returns:
        1. Cold start user profile (on first message of session)
        2. Scoring prompt if previous exchange needs scoring AND assistant completed
        3. Relevant memories from search
        4. User facts from memory_bank
        """
        if not _memory or not _session_manager:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            formatted_parts = []
            scoring_required = False
            conversation_id = request.conversation_id or "default"

            # 0. Check for cold start (first message of session)
            is_cold_start = _session_manager.is_first_message(conversation_id)

            if is_cold_start:
                # Check for updates on cold start only (once per conversation)
                update_injection = _get_update_injection()
                if update_injection:
                    formatted_parts.append(update_injection)

                # Dump full user profile on first message
                cold_start_profile = await _build_cold_start_profile()
                if cold_start_profile:
                    formatted_parts.append(cold_start_profile)
                    logger.info(f"Cold start: injected user profile for {conversation_id}")

                # Mark first message as seen
                _session_manager.mark_first_message_seen(conversation_id)

            # v0.2.7: Get context for BOTH cold start and regular messages
            # Cold start uses it for KNOWN CONTEXT (recent work), non-cold start also uses for scoring
            context = await _memory.get_context_for_injection(
                query=request.query,
                conversation_id=conversation_id,
                recent_conversation=request.recent_messages
            )

            # v0.2.7: On cold start, append KNOWN CONTEXT after profile (recent work context)
            if is_cold_start and context.get("formatted_injection"):
                formatted_parts.append(context["formatted_injection"])
                logger.info(f"Cold start: added KNOWN CONTEXT for {conversation_id}")

            # 2. Check if assistant completed a response (vs user interrupting mid-work)
            assistant_completed = _session_manager.check_and_clear_completed(conversation_id)

            # Only inject scoring prompt if:
            # - Assistant completed their previous response (not mid-work interruption)
            # - There's an unscored exchange to score
            # - NOT a cold start (no previous exchange to score on first message)
            if assistant_completed and not is_cold_start:
                previous = await _session_manager.get_previous_exchange(conversation_id)

                if previous and not previous.get("scored", False):
                    # Build list of surfaced memories for selective scoring
                    surfaced_memories = context.get("relevant_memories", [])

                    # Inject scoring prompt with surfaced memories
                    scoring_prompt = _session_manager.build_scoring_prompt(
                        previous_exchange=previous,
                        current_user_message=request.query,
                        surfaced_memories=surfaced_memories if surfaced_memories else None
                    )
                    formatted_parts.append(scoring_prompt)
                    scoring_required = True
                    # Track that we injected scoring prompt (for Stop hook to check)
                    _session_manager.set_scoring_required(conversation_id, True)
                    logger.info(f"Injecting scoring prompt for conversation {conversation_id} with {len(surfaced_memories)} memories")
                else:
                    # No unscored exchange, but assistant did complete
                    _session_manager.set_scoring_required(conversation_id, False)
            else:
                # User interrupted mid-work OR cold start - no scoring needed
                _session_manager.set_scoring_required(conversation_id, False)
                if not is_cold_start:
                    logger.info(f"Skipping scoring - user interrupted mid-work for {conversation_id}")

            # 3. Add memory context after scoring prompt (only if not cold start)
            # v0.2.7: Cold start already added KNOWN CONTEXT above, don't duplicate
            if not is_cold_start and context.get("formatted_injection"):
                formatted_parts.append(context["formatted_injection"])

            # 4. Cache doc_ids for outcome scoring via record_response
            # This ensures hook-injected memories can be scored later
            injected_doc_ids = context.get("doc_ids", [])
            if injected_doc_ids:
                _search_cache[conversation_id] = {
                    "doc_ids": injected_doc_ids,
                    "query": request.query,
                    "source": "hook_injection",
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"Cached {len(injected_doc_ids)} doc_ids from hook injection for {conversation_id}")

            return GetContextResponse(
                formatted_injection="\n".join(formatted_parts),
                user_facts=context.get("user_facts", []),
                relevant_memories=context.get("relevant_memories", []),
                context_summary=context.get("context_summary", ""),
                scoring_required=scoring_required
            )

        except Exception as e:
            import traceback
            logger.error(f"Error getting context: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/hooks/stop", response_model=StopHookResponse)
    async def stop_hook(request: StopHookRequest):
        """
        Called by Stop hook AFTER the LLM responds.

        1. Stores the exchange with doc_id for later scoring
        2. Checks if record_response was called (with retry for race condition)
        3. Returns should_block=True if scoring was required but not done
        """
        if not _memory or not _session_manager:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            conversation_id = request.conversation_id or "default"

            # Skip storing if either message is empty/blank
            user_msg = (request.user_message or "").strip()
            assistant_msg = (request.assistant_response or "").strip()

            if not user_msg or not assistant_msg:
                logger.warning(f"Skipping empty exchange storage: user={bool(user_msg)}, assistant={bool(assistant_msg)}")
                return StopHookResponse(
                    status="skipped",
                    message="Empty exchange not stored"
                )

            # Store the exchange in working memory
            content = f"User: {user_msg}\n\nAssistant: {assistant_msg}"
            doc_id = await _memory.store_working(
                content=content,
                conversation_id=conversation_id,
                metadata={
                    "turn_type": "exchange",
                    "timestamp": datetime.now().isoformat()
                }
            )

            # Store exchange in session file
            await _session_manager.store_exchange(
                conversation_id=conversation_id,
                user_message=request.user_message,
                assistant_response=request.assistant_response,
                doc_id=doc_id
            )

            # IMPORTANT: Check scoring flags BEFORE set_completed() resets them
            scoring_was_required = _session_manager.was_scoring_required(conversation_id)

            # Race condition fix: If scoring was required, wait briefly for record_response
            # MCP tool call to complete. The tool might be in-flight when Stop hook fires.
            scored_this_turn = _session_manager.was_scored_this_turn(conversation_id)

            if scoring_was_required and not scored_this_turn:
                # Wait up to 500ms with 50ms intervals for the MCP tool to complete
                for _ in range(10):
                    await asyncio.sleep(0.05)  # 50ms
                    scored_this_turn = _session_manager.was_scored_this_turn(conversation_id)
                    if scored_this_turn:
                        logger.info(f"Race condition resolved: record_response completed after {(_ + 1) * 50}ms")
                        break

            # Mark assistant as completed - this signals UserPromptSubmit that
            # scoring is needed on the NEXT user message
            # Note: This resets scoring_required, so we check it above first
            _session_manager.set_completed(conversation_id)
            logger.info(f"Marked assistant as completed for {conversation_id}")

            # Determine blocking based on scoring state
            scoring_complete = False
            should_block = False
            block_message = None

            if scored_this_turn:
                scoring_complete = True
                logger.info(f"record_response was called this turn for {conversation_id}")
            elif scoring_was_required:
                # Scoring was required this turn but LLM didn't call record_response
                # SOFT ENFORCE: Log warning but don't block (prompt injection does 95% of the work)
                # We trade guaranteed enforcement for smooth UX - no block_message either
                should_block = False
                block_message = None  # Don't send any message to avoid UI noise
                logger.warning(f"Soft enforce: record_response not called for {conversation_id}")
            else:
                # Scoring wasn't required (user interrupted mid-work) - don't block
                logger.info(f"No scoring required this turn for {conversation_id} - not blocking")

            return StopHookResponse(
                stored=True,
                doc_id=doc_id,
                scoring_complete=scoring_complete,
                should_block=should_block,
                block_message=block_message
            )

        except Exception as e:
            logger.error(f"Error in stop hook: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Memory API Endpoints ====================

    @app.post("/api/search")
    async def search_memory(request: SearchRequest):
        """Search across memory collections."""
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            results = await _memory.search(
                query=request.query,
                collections=request.collections,
                limit=request.limit
            )

            # Cache doc_ids for outcome scoring
            if request.conversation_id:
                doc_ids = [r.get("id") for r in results if r.get("id")]
                _search_cache[request.conversation_id] = {
                    "doc_ids": doc_ids,
                    "query": request.query,
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "query": request.query,
                "count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memory-bank/add")
    async def add_to_memory_bank(request: MemoryBankAddRequest):
        """Add a fact to memory bank."""
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            doc_id = await _memory.store_memory_bank(
                text=request.content,
                tags=request.tags,
                importance=request.importance,
                confidence=request.confidence
            )

            return {"success": True, "doc_id": doc_id}

        except Exception as e:
            logger.error(f"Error adding to memory bank: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memory-bank/update")
    async def update_memory_bank(request: MemoryBankUpdateRequest):
        """Update a memory bank entry."""
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            doc_id = await _memory.update_memory_bank(
                old_content=request.old_content,
                new_content=request.new_content
            )

            return {
                "success": doc_id is not None,
                "doc_id": doc_id
            }

        except Exception as e:
            logger.error(f"Error updating memory bank: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memory-bank/archive")
    async def delete_memory_bank(request: Dict[str, str]):
        """Archive a memory bank entry."""
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        content = request.get("content", "")
        if not content:
            raise HTTPException(status_code=400, detail="Content required")

        try:
            success = await _memory.delete_memory_bank(content)
            return {"success": success}

        except Exception as e:
            logger.error(f"Error archiving: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/ingest")
    async def ingest_document(request: Dict[str, Any]):
        """
        Ingest a document into the books collection.

        Called by CLI when server is running, so data is immediately searchable.

        Request body:
            content: Document text
            title: Document title
            source: Source file path
            chunk_size: Characters per chunk (default 1000)
            chunk_overlap: Overlap between chunks (default 200)
        """
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        content = request.get("content", "")
        title = request.get("title", "Untitled")
        source = request.get("source", "unknown")
        chunk_size = request.get("chunk_size", 1000)
        chunk_overlap = request.get("chunk_overlap", 200)

        if not content:
            raise HTTPException(status_code=400, detail="Content required")

        try:
            doc_ids = await _memory.store_book(
                content=content,
                title=title,
                source=source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            logger.info(f"Ingested '{title}' in {len(doc_ids)} chunks")

            return {
                "success": True,
                "title": title,
                "chunks": len(doc_ids),
                "doc_ids": doc_ids
            }

        except Exception as e:
            logger.error(f"Error archiving: {e}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get("/api/books")
    async def list_books():
        """List all ingested books."""
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            books = await _memory.list_books()
            return {"success": True, "books": books}
        except Exception as e:
            logger.error(f"Error listing books: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/remove-book")
    async def remove_book(request: Dict[str, Any]):
        """Remove a book by title."""
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        title = request.get("title", "")
        if not title:
            raise HTTPException(status_code=400, detail="Title required")

        try:
            result = await _memory.remove_book(title)
            return {
                "success": result.get("removed", 0) > 0,
                "removed": result.get("removed", 0),
                "title": title,
                "message": result.get("message", "")
            }
        except Exception as e:
            logger.error(f"Error removing book: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # v0.2.3: Deferred background task for Action KG updates
    async def _deferred_action_kg_updates(
        doc_ids: List[str],
        outcome: str,
        cached_query: str
    ):
        """
        Background task for heavy Action KG and routing updates.

        v0.2.3: Extracted from record_outcome endpoint for performance.
        """
        try:
            # Detect context type
            context_type = await _memory.detect_context_type() or "general"
            collections_updated = set()

            for doc_id in doc_ids:
                # Extract collection from doc_id prefix
                collection = None
                for coll_name in ["memory_bank", "books", "working", "history", "patterns"]:
                    if doc_id.startswith(coll_name):
                        collection = coll_name
                        break

                # Track in Action KG
                action = ActionOutcome(
                    action_type="score_response",
                    context_type=context_type,
                    outcome=outcome,
                    doc_id=doc_id,
                    collection=collection
                )
                await _memory.record_action_outcome(action)

                if collection:
                    collections_updated.add(collection)

            # Update Routing KG
            if cached_query:
                for collection in collections_updated:
                    await _memory._update_kg_routing(cached_query, collection, outcome)

            logger.info(f"[Background] Action KG updates completed for {len(doc_ids)} docs")

        except Exception as e:
            logger.error(f"[Background] Action KG update error: {e}")

    @app.post("/api/record-outcome")
    async def record_outcome(request: RecordOutcomeRequest):
        """
        Record outcome for learning.

        Called by the score_response MCP tool.
        Scores:
        1. Most recent unscored exchange (across ALL sessions - handles MCP/hook ID mismatch)
        2. Cached search results (from _search_cache)
        """
        if not _memory or not _session_manager:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            conversation_id = request.conversation_id or "default"
            doc_ids_scored = []

            # Track that score_response was called this turn (for Stop hook blocking)
            _session_manager.set_scored_this_turn(conversation_id, True)

            # v0.2.3: Skip session file scan - doc_ids are in _search_cache or request.related
            # Old approach scanned ALL session files (O(n×m) I/O) - now O(1) cache lookup

            # Score cached search results
            # First try exact conversation_id match
            cached = _search_cache.get(conversation_id, {})
            cached_doc_ids = cached.get("doc_ids", [])
            cache_key_used = conversation_id

            # If no cache for this ID, find the most recent cache entry
            # This handles MCP using "default" while hook caches under real session_id
            if not cached_doc_ids and _search_cache:
                most_recent_key = max(_search_cache.keys(),
                    key=lambda k: _search_cache[k].get("timestamp", ""))
                cached = _search_cache.get(most_recent_key, {})
                cached_doc_ids = cached.get("doc_ids", [])
                cache_key_used = most_recent_key
                if cached_doc_ids:
                    logger.info(f"Using cache from session {most_recent_key} (MCP used {conversation_id})")

            # v0.2.8: Per-memory scoring - process each memory with individual outcome
            if request.memory_scores:
                for doc_id, mem_outcome in request.memory_scores.items():
                    if mem_outcome in ["worked", "failed", "partial"]:
                        await _memory.record_outcome(doc_ids=[doc_id], outcome=mem_outcome)
                        doc_ids_scored.append(doc_id)
                logger.info(f"Per-memory scoring: {len(doc_ids_scored)} memories with individual outcomes")

            # DEPRECATED: related param (backward compat)
            elif request.related is not None:
                doc_ids_scored.extend(request.related)
                logger.info(f"Direct scoring (deprecated): {len(request.related)} doc_ids from related param")
                if doc_ids_scored and request.outcome in ["worked", "failed", "partial"]:
                    await _memory.record_outcome(doc_ids=doc_ids_scored, outcome=request.outcome)

            # Fallback: score all cached with exchange outcome
            elif cached_doc_ids:
                doc_ids_scored.extend(cached_doc_ids)
                logger.info(f"Cache scoring: {len(cached_doc_ids)} doc_ids")
                if doc_ids_scored and request.outcome in ["worked", "failed", "partial"]:
                    await _memory.record_outcome(doc_ids=doc_ids_scored, outcome=request.outcome)

            # Log final result and trigger background updates
            if doc_ids_scored:
                logger.info(f"Scored {len(doc_ids_scored)} documents")

                # ========== FAST PATH COMPLETE (v0.2.3) ==========
                # Score recorded. Defer heavy Action KG and routing updates to background.
                cached_query = cached.get("query", "")
                asyncio.create_task(
                    _deferred_action_kg_updates(
                        doc_ids=doc_ids_scored,
                        outcome=request.outcome,
                        cached_query=cached_query
                    )
                )

            # Clear search cache for the key we used
            if cache_key_used in _search_cache:
                del _search_cache[cache_key_used]

            return {
                "success": True,
                "outcome": request.outcome,
                "documents_scored": len(doc_ids_scored)
            }

        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Health/Status Endpoints ====================

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "memory_initialized": _memory is not None and _memory.initialized,
            "session_manager_ready": _session_manager is not None,
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/api/stats")
    async def get_stats():
        """Get memory system statistics."""
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        return _memory.get_stats()

    return app


def start_server(host: str = "127.0.0.1", port: int = None, dev: bool = False):
    """
    Start the Roampal server with dev/prod isolation.
    
    Args:
        host: Server host
        port: Server port (auto-determined from dev mode if not specified)
        dev: Run in dev mode (port 27183, Roampal_DEV data)
    """
    # Determine mode and port
    dev_mode = dev or os.environ.get('ROAMPAL_DEV', '').lower() in ('1', 'true', 'yes')

    # Set env var so lifespan() can read it
    if dev_mode:
        os.environ['ROAMPAL_DEV'] = '1'

    if port is None:
        port = DEV_PORT if dev_mode else PROD_PORT
    
    # Validate port matches mode
    if dev_mode and port != DEV_PORT:
        raise ValueError(f"DEV mode requires port {DEV_PORT}, got {port}")
    if not dev_mode and port != PROD_PORT:
        raise ValueError(f"PROD mode requires port {PROD_PORT}, got {port}")
    
    # Startup banner (ASCII-safe for Windows cp1252)
    mode_str = "DEV" if dev_mode else "PROD"
    data_hint = "Roampal_DEV" if dev_mode else "Roampal"
    print(f"""
===================================================
  ROAMPAL SERVER - {mode_str} MODE
  Port: {port}
  Data: %APPDATA%/{data_hint}/data
===================================================
""")
    
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Roampal FastAPI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 27182 prod, 27183 dev)")
    parser.add_argument("--dev", action="store_true", help="Run in dev mode (port 27183, Roampal_DEV data)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    start_server(host=args.host, port=args.port, dev=args.dev)
