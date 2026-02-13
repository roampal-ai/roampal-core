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
from pydantic import BaseModel, Field
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

# v0.3.2: Injection map - tracks which doc_ids were injected to which conversation
# This enables robust multi-session scoring by matching doc_ids to their source conversation
# instead of relying on "most recent unscored" heuristics
# Format: {doc_id: {"conversation_id": str, "injected_at": str, "exchange_doc_id": str}}
_injection_map: Dict[str, Dict[str, Any]] = {}

# Port constants for dev/prod isolation
PROD_PORT = 27182
DEV_PORT = 27183

# Cache for update check (only check once per server session)
_update_check_cache: Optional[tuple] = None

# Cold start tag priorities - one fact per category (v0.2.7)
TAG_PRIORITIES = ["identity", "preference", "goal", "project", "system_mastery", "agent_growth"]


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

When natural, ask their name to personalize future sessions. Store with:
  add_to_memory_bank(content="User's name is [NAME]", tags=["identity"])
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
    # v0.3.2: Split fields for OpenCode plugin — scoring in user message, context in system prompt
    scoring_prompt: str = ""  # Just the scoring block (for prepending to user message)
    scoring_prompt_simple: str = ""  # Simplified scoring prompt for non-Claude models (no XML tags, plain language)
    context_only: str = ""  # Just the memory context without scoring (for system prompt)
    # v0.3.2: Raw scoring data for independent LLM scoring call (OpenCode plugin)
    scoring_exchange: Optional[Dict[str, str]] = None  # {"user": "...", "assistant": "..."} previous exchange
    scoring_memories: Optional[List[Dict[str, str]]] = None  # [{"id": "doc_id", "content": "full memory content"}, ...]


class StopHookRequest(BaseModel):
    """Request from Stop hook after LLM responds."""
    conversation_id: str
    user_message: str = ""  # v0.3.6: Optional — Claude Code sends empty (state-only)
    assistant_response: str = ""  # v0.3.6: Optional — Claude Code sends empty (state-only)
    transcript: Optional[str] = None  # Full transcript to check for record_response call
    metadata: Optional[Dict[str, Any]] = None  # v0.3.6: Extra metadata (e.g., memory_type: "exchange_summary")
    lifecycle_only: bool = False  # v0.3.6: Track in session JSONL but skip ChromaDB storage


class StopHookResponse(BaseModel):
    """Response to Stop hook."""
    stored: bool
    doc_id: str
    scoring_complete: bool  # Did the LLM call record_response?
    should_block: bool  # Should hook block with exit code 2?
    block_message: Optional[str] = None


class SearchRequest(BaseModel):
    """Request for searching memory."""
    query: Optional[str] = Field("", max_length=2000)
    days_back: Optional[int] = Field(None, ge=1, le=365)
    id: Optional[str] = Field(None, max_length=200)
    conversation_id: Optional[str] = Field(None, max_length=200)
    collections: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=500)
    metadata_filters: Optional[Dict[str, Any]] = None
    sort_by: Optional[str] = Field(None, pattern="^(relevance|recency|score)$")


class MemoryBankAddRequest(BaseModel):
    """Request to add to memory bank."""
    content: str
    tags: Optional[List[str]] = None
    importance: float = 0.7
    confidence: float = 0.7
    always_inject: bool = False


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
    exchange_summary: Optional[str] = None  # v0.3.6: ~300 char summary from main LLM
    exchange_outcome: Optional[str] = None  # v0.3.6: backward compat alias for outcome


class RecordResponseRequest(BaseModel):
    """Request to record a key takeaway (MCP tool proxy)."""
    key_takeaway: str
    conversation_id: str


class UpdateContentRequest(BaseModel):
    """Request to update a memory's content (v0.3.6 summarization)."""
    doc_id: str
    collection: str
    new_content: str


# ==================== Lifecycle ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage memory system lifecycle."""
    global _memory, _session_manager
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

    # v0.3.2: Parent process monitoring removed. FastAPI now outlives any single
    # MCP client so multiple clients (Claude Code, Cursor, OpenCode) can share it.
    # First MCP to start auto-starts FastAPI; others detect port in use and skip.

    yield

    # Cleanup
    logger.info("Shutting down Roampal server...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Roampal",
        description="Persistent memory for AI coding tools",
        version="0.3.6",
        lifespan=lifespan
    )

    # CORS — localhost only (v0.3.5: tightened from allow_origins=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
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
            scoring_prompt_text = ""  # v0.3.2: track separately for OpenCode split delivery
            scoring_exchange_data = None  # v0.3.2: raw exchange data for independent LLM scoring
            scoring_memories_data = None  # v0.3.2: raw surfaced memory data for independent LLM scoring
            scoring_prompt_simple_text = ""  # v0.3.2: simplified scoring for non-Claude models
            context_parts = []  # v0.3.2: non-scoring context parts for split delivery
            conversation_id = request.conversation_id or "default"

            # 0. Check for cold start (first message of session)
            is_cold_start = _session_manager.is_first_message(conversation_id)

            if is_cold_start:
                # Check for updates on cold start only (once per conversation)
                update_injection = _get_update_injection()
                if update_injection:
                    formatted_parts.append(update_injection)
                    context_parts.append(update_injection)

                # Dump full user profile on first message
                cold_start_profile = await _build_cold_start_profile()
                if cold_start_profile:
                    formatted_parts.append(cold_start_profile)
                    context_parts.append(cold_start_profile)
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
                context_parts.append(context["formatted_injection"])
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
                    scoring_prompt_simple = _session_manager.build_scoring_prompt_simple(
                        previous_exchange=previous,
                        current_user_message=request.query,
                        surfaced_memories=surfaced_memories if surfaced_memories else None
                    )
                    formatted_parts.append(scoring_prompt)
                    scoring_prompt_text = scoring_prompt  # v0.3.2: track for split delivery
                    scoring_prompt_simple_text = scoring_prompt_simple  # v0.3.2: simplified for non-Claude
                    scoring_required = True
                    # v0.3.2: Raw data for independent LLM scoring (OpenCode plugin)
                    scoring_exchange_data = {
                        "user": previous.get("user", ""),
                        "assistant": previous.get("assistant", "")
                    }
                    scoring_memories_data = []
                    if surfaced_memories:
                        for mem in surfaced_memories:
                            mem_id = mem.get("id", mem.get("doc_id", "unknown"))
                            content = mem.get("content", mem.get("text", ""))
                            scoring_memories_data.append({
                                "id": mem_id,
                                "content": content,
                                "content_hint": content[:60] if content else ""  # v0.3.5: brief hint for SCORING REFERENCE
                            })
                    # Track that we injected scoring prompt (for Stop hook to check)
                    _session_manager.set_scoring_required(conversation_id, True)
                    logger.info(f"Injecting scoring prompt for conversation {conversation_id} with {len(surfaced_memories)} memories")
                else:
                    # No unscored exchange, but assistant did complete
                    _session_manager.set_scoring_required(conversation_id, False)

                # v0.3.6: Exchange summarization handled by main LLM via score_memories
                # (no background sidecar needed — see Change 9 platform-split architecture)
            else:
                # User interrupted mid-work OR cold start - no scoring needed
                _session_manager.set_scoring_required(conversation_id, False)
                if not is_cold_start:
                    logger.info(f"Skipping scoring - user interrupted mid-work for {conversation_id}")

            # 3. Add memory context after scoring prompt (only if not cold start)
            # v0.2.7: Cold start already added KNOWN CONTEXT above, don't duplicate
            if not is_cold_start and context.get("formatted_injection"):
                formatted_parts.append(context["formatted_injection"])
                context_parts.append(context["formatted_injection"])

            # 4. Cache doc_ids for outcome scoring via record_response
            # This ensures hook-injected memories can be scored later
            injected_doc_ids = context.get("doc_ids", [])
            timestamp = datetime.now().isoformat()
            if injected_doc_ids:
                _search_cache[conversation_id] = {
                    "doc_ids": injected_doc_ids,
                    "query": request.query,
                    "source": "hook_injection",
                    "timestamp": timestamp
                }
                logger.info(f"Cached {len(injected_doc_ids)} doc_ids from hook injection for {conversation_id}")

                # v0.3.2: Populate injection map for robust multi-session scoring
                # Each doc_id maps back to the conversation that received it
                # This enables matching by doc_id instead of "most recent unscored" heuristics
                for doc_id in injected_doc_ids:
                    _injection_map[doc_id] = {
                        "conversation_id": conversation_id,
                        "injected_at": timestamp,
                        "query": request.query
                    }
                logger.info(f"Added {len(injected_doc_ids)} doc_ids to injection map for {conversation_id}")

            return GetContextResponse(
                formatted_injection="\n".join(formatted_parts),
                user_facts=context.get("user_facts", []),
                relevant_memories=context.get("relevant_memories", []),
                context_summary=context.get("context_summary", ""),
                scoring_required=scoring_required,
                # v0.3.2: Split fields for OpenCode — scoring in user message, context in system prompt
                scoring_prompt=scoring_prompt_text,
                scoring_prompt_simple=scoring_prompt_simple_text,
                context_only="\n".join(context_parts) if context_parts else "",
                scoring_exchange=scoring_exchange_data,
                scoring_memories=scoring_memories_data
            )

        except Exception as e:
            import traceback
            logger.error(f"Error getting context: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/hooks/check-scored")
    async def check_scored(conversation_id: str = ""):
        """Check if score_memories was already called for this conversation this turn.
        Used by OpenCode plugin to skip sidecar if main LLM already scored."""
        if not _session_manager:
            return {"scored": False}
        scored = _session_manager.was_scored_this_turn(conversation_id)
        return {"scored": scored}

    @app.post("/api/hooks/stop", response_model=StopHookResponse)
    async def stop_hook(request: StopHookRequest):
        """
        Called by Stop hook AFTER the LLM responds.

        v0.3.6: Two modes via lifecycle_only flag:
        - Claude Code (lifecycle_only=True): JSONL exchange tracking + state management.
          Exchanges stored in session JSONL for scoring prompt generation, but NOT in ChromaDB.
          Main LLM stores summaries directly via score_memories tool.
        - OpenCode (lifecycle_only=False): Full exchange storage in both JSONL and ChromaDB.
          Sidecar summarizes later via session.idle event.

        Always handles turn lifecycle: checks scoring compliance, marks turn complete.
        """
        if not _memory or not _session_manager:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            conversation_id = request.conversation_id or "default"

            # Store exchange based on mode:
            # - lifecycle_only=True (Claude Code): Store in session JSONL only (for scoring prompts)
            #   Main LLM stores summaries in ChromaDB via score_memories tool.
            # - lifecycle_only=False (OpenCode): Store in both JSONL and ChromaDB working memory
            #   Sidecar summarizes later via session.idle event.
            user_msg = (request.user_message or "").strip()
            assistant_msg = (request.assistant_response or "").strip()
            doc_id = ""

            if user_msg and assistant_msg:
                if request.lifecycle_only:
                    # Claude Code path: JSONL only (for get_previous_exchange scoring lifecycle)
                    # Skip ChromaDB — main LLM stores summaries via score_memories
                    await _session_manager.store_exchange(
                        conversation_id=conversation_id,
                        user_message=request.user_message,
                        assistant_response=request.assistant_response,
                        doc_id=""  # No ChromaDB doc — main LLM handles storage
                    )
                    logger.info(f"Lifecycle-only exchange stored in JSONL for {conversation_id}")
                else:
                    # OpenCode path: Full storage in both ChromaDB and JSONL
                    content = f"User: {user_msg}\n\nAssistant: {assistant_msg}"
                    store_metadata = {
                        "turn_type": "exchange",
                        "timestamp": datetime.now().isoformat()
                    }
                    if request.metadata:
                        store_metadata.update(request.metadata)

                    doc_id = await _memory.store_working(
                        content=content,
                        conversation_id=conversation_id,
                        metadata=store_metadata
                    )

                    await _session_manager.store_exchange(
                        conversation_id=conversation_id,
                        user_message=request.user_message,
                        assistant_response=request.assistant_response,
                        doc_id=doc_id
                    )
                    logger.info(f"Full exchange {doc_id} stored for {conversation_id}")
            else:
                logger.info(f"State-only stop hook for {conversation_id} (no exchange data)")

            # === State management (always runs) ===

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

            # v0.3.2: Session ID mismatch fallback - OpenCode plugin hooks use ses_xxx
            # but MCP tools use mcp_xxx. Check if any MCP session was scored this turn.
            if scoring_was_required and not scored_this_turn:
                state = _session_manager._load_completion_state()
                for sid, sdata in state.items():
                    if sid.startswith("mcp_") and sdata.get("scored_this_turn"):
                        scored_this_turn = True
                        _session_manager.set_scored_this_turn(sid, False)
                        logger.info(f"Session ID mismatch resolved: MCP {sid} scored for plugin {conversation_id}")
                        break

            # Mark assistant as completed - this signals UserPromptSubmit that
            # scoring is needed on the NEXT user message
            _session_manager.set_completed(conversation_id)
            logger.info(f"Marked assistant as completed for {conversation_id}")

            # Determine blocking based on scoring state
            scoring_complete = False
            should_block = False
            block_message = None

            if scored_this_turn:
                scoring_complete = True
                logger.info(f"score_memories was called this turn for {conversation_id}")
            elif scoring_was_required:
                # SOFT ENFORCE: Log warning but don't block
                should_block = False
                block_message = None
                logger.warning(f"Soft enforce: score_memories not called for {conversation_id}")
            else:
                logger.info(f"No scoring required this turn for {conversation_id} - not blocking")

            # v0.3.6: Spawn background auto-summarize task (non-blocking)
            # One old memory cleaned up per stop hook cycle.
            # Follows existing pattern from _deferred_action_kg_updates().
            asyncio.create_task(_auto_summarize_one_memory())

            return StopHookResponse(
                stored=bool(doc_id),
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
            # Direct ID lookup — bypass search entirely
            if request.id:
                doc = _memory.get_by_id(request.id)
                results = [doc] if doc else []
                return {
                    "query": request.id,
                    "count": len(results),
                    "results": results
                }

            # Convert days_back to date filter
            metadata_filters = request.metadata_filters or {}
            if request.days_back:
                from datetime import timedelta
                cutoff = (datetime.now() - timedelta(days=request.days_back)).isoformat()
                metadata_filters["created_at"] = {"$gte": cutoff}

            results = await _memory.search(
                query=request.query or "",
                collections=request.collections,
                limit=request.limit,
                metadata_filters=metadata_filters if metadata_filters else None,
                sort_by=request.sort_by
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
                confidence=request.confidence,
                always_inject=request.always_inject
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

        # v0.3.5: Enforce size limit at API layer (store_book also checks, but fail fast)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail=f"Content too large ({len(content)} bytes, max {max_size})")

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

    # ==================== MCP Tool Proxy Endpoints ====================

    @app.post("/api/record-response")
    async def record_response_endpoint(request: RecordResponseRequest):
        """Record a key takeaway in working memory (MCP tool proxy)."""
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            doc_id = await _memory.store_working(
                content=f"Key takeaway: {request.key_takeaway}",
                conversation_id=request.conversation_id,
                metadata={
                    "type": "key_takeaway",
                    "timestamp": datetime.now().isoformat()
                },
                initial_score=0.7
            )
            logger.info(f"Recorded takeaway (score=0.7): {request.key_takeaway[:50]}...")
            return {"success": True, "doc_id": doc_id}

        except Exception as e:
            logger.error(f"Error recording response: {e}")
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
                    action_type="score_memories",
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

        Called by the score_memories MCP tool.
        Scores:
        1. Most recent unscored exchange (across ALL sessions - handles MCP/hook ID mismatch)
        2. Cached search results (from _search_cache)
        """
        if not _memory or not _session_manager:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            conversation_id = request.conversation_id or "default"
            doc_ids_scored = []

            # Track that score_memories was called this turn (for Stop hook blocking)
            _session_manager.set_scored_this_turn(conversation_id, True)

            # v0.3.6: Resolve the plugin session ID via injection_map BEFORE outcome check.
            # The MCP session ID (conversation_id) differs from the plugin session (ses_xxx).
            # We need to mark the plugin session as scored so check-scored returns true.
            exchange_conv_id = conversation_id
            resolved_via = None
            if request.memory_scores and _injection_map:
                for doc_id in request.memory_scores.keys():
                    injection = _injection_map.get(doc_id)
                    if injection:
                        exchange_conv_id = injection["conversation_id"]
                        _session_manager.set_scored_this_turn(exchange_conv_id, True)
                        resolved_via = f"injection_map (doc_id={doc_id})"
                        logger.info(f"Resolved conversation {exchange_conv_id} via {resolved_via}")
                        break

            # Fallback: if injection_map didn't resolve, check _last_exchange_cache
            # for most recent unscored exchange (handles empty memory_scores case)
            if not resolved_via and _session_manager:
                best_ts = ""
                for cid, exc in _session_manager._last_exchange_cache.items():
                    if cid != conversation_id and not exc.get("scored", False):
                        ts = exc.get("timestamp", "")
                        if ts > best_ts:
                            best_ts = ts
                            exchange_conv_id = cid
                            resolved_via = "last_exchange_cache (most recent unscored)"
                if resolved_via:
                    _session_manager.set_scored_this_turn(exchange_conv_id, True)
                    logger.info(f"Resolved conversation {exchange_conv_id} via {resolved_via}")

            # Look up the previous exchange doc_id (needed for summary replacement on OpenCode)
            # Claude Code: exchange_doc_id will be None (stop hook doesn't store exchanges)
            exchange_doc_id = None
            previous = _session_manager._last_exchange_cache.get(exchange_conv_id)
            if not previous:
                previous = _session_manager._last_exchange_cache.get(conversation_id)
                if previous:
                    exchange_conv_id = conversation_id

            if not previous:
                # Last resort: Find the MOST RECENT unscored exchange across all sessions
                best_ts = ""
                for cid, exc in _session_manager._last_exchange_cache.items():
                    if not exc.get("scored", False):
                        exc_ts = exc.get("timestamp", "")
                        if exc_ts > best_ts:
                            best_ts = exc_ts
                            previous = exc
                            exchange_conv_id = cid
                if previous:
                    logger.warning(f"Used 'most recent unscored' fallback - resolved to {exchange_conv_id}")

            if previous and previous.get("doc_id") and not previous.get("scored", False):
                exchange_doc_id = previous["doc_id"]
                # Score the exchange with the outcome (skip for "unknown" — no signal)
                if request.outcome in ["worked", "failed", "partial"]:
                    await _memory.record_outcome(doc_ids=[exchange_doc_id], outcome=request.outcome)
                await _session_manager.mark_scored(exchange_conv_id, exchange_doc_id, request.outcome)
                logger.info(f"Scored exchange {exchange_doc_id} with outcome={request.outcome}")

            # v0.2.3: Skip session file scan - doc_ids are in _search_cache or request.related
            # Old approach scanned ALL session files (O(n×m) I/O) - now O(1) cache lookup

            # Score cached search results
            # v0.3.2: Try resolved exchange_conv_id first (from injection_map), then original conversation_id
            cached = _search_cache.get(exchange_conv_id, {})
            cached_doc_ids = cached.get("doc_ids", [])
            cache_key_used = exchange_conv_id

            # Fallback to original conversation_id if exchange_conv_id didn't have cache
            if not cached_doc_ids and exchange_conv_id != conversation_id:
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
                    if mem_outcome in ["worked", "failed", "partial", "unknown"]:
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

            # v0.3.6: Store exchange summary
            # Claude Code: main LLM stores summary directly (no prior exchange to replace)
            # OpenCode: replace the full exchange stored by stop hook with summary
            summary_stored = False
            if request.exchange_summary and _memory:
                try:
                    if exchange_doc_id:
                        # OpenCode path: replace existing full exchange with summary
                        adapter = _memory.collections.get("working")
                        if adapter:
                            doc = adapter.get_fragment(exchange_doc_id)
                            if doc:
                                metadata = doc.get("metadata", {})
                                metadata["text"] = request.exchange_summary
                                metadata["content"] = request.exchange_summary
                                metadata["memory_type"] = "exchange_summary"
                                metadata["exchange_outcome"] = request.outcome
                                metadata["summarized_at"] = datetime.now().isoformat()
                                metadata["original_length"] = len(doc.get("content", ""))

                                embedding = await _memory._embedding_service.embed_text(request.exchange_summary)
                                await adapter.upsert_vectors(
                                    ids=[exchange_doc_id],
                                    vectors=[embedding],
                                    metadatas=[metadata],
                                )
                                summary_stored = True
                                logger.info(f"Replaced exchange {exchange_doc_id} with summary ({len(request.exchange_summary)} chars, was {metadata.get('original_length', '?')})")
                    else:
                        # Claude Code path: store summary directly as new working memory
                        summary_doc_id = await _memory.store_working(
                            content=request.exchange_summary,
                            conversation_id=conversation_id,
                            metadata={
                                "memory_type": "exchange_summary",
                                "exchange_outcome": request.outcome or "unknown",
                                "summarized_at": datetime.now().isoformat(),
                                "turn_type": "exchange",
                            }
                        )
                        summary_stored = True
                        logger.info(f"Stored exchange summary {summary_doc_id} directly ({len(request.exchange_summary)} chars)")
                except Exception as e:
                    logger.error(f"Failed to store exchange summary: {e}")

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

            # v0.3.2: Clean up injection_map for scored doc_ids
            # This prevents the map from growing indefinitely
            for doc_id in doc_ids_scored:
                if doc_id in _injection_map:
                    del _injection_map[doc_id]

            return {
                "success": True,
                "outcome": request.outcome,
                "documents_scored": len(doc_ids_scored),
                "summary_stored": summary_stored
            }

        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Content Update (v0.3.6 Summarization) ====================

    @app.post("/api/memory/update-content")
    async def update_memory_content(request: UpdateContentRequest):
        """
        Update a memory's content and re-embed it.

        v0.3.6: Used by `roampal summarize` to replace long memories with summaries.
        """
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        try:
            collection = request.collection
            if collection not in _memory.collections:
                raise HTTPException(status_code=400, detail=f"Unknown collection: {collection}")

            adapter = _memory.collections[collection]
            doc = adapter.get_fragment(request.doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document not found: {request.doc_id}")

            # Update metadata with new content
            metadata = doc.get("metadata", {})
            metadata["text"] = request.new_content
            metadata["content"] = request.new_content
            metadata["summarized_at"] = datetime.now().isoformat()
            metadata["original_length"] = len(doc.get("content", ""))

            # Re-embed with new content
            embedding = await _memory._embedding_service.embed_text(request.new_content)
            await adapter.upsert_vectors(
                ids=[request.doc_id],
                vectors=[embedding],
                metadatas=[metadata]
            )

            logger.info(f"Updated content for {request.doc_id}: {metadata['original_length']} -> {len(request.new_content)} chars")
            return {"success": True, "doc_id": request.doc_id, "new_length": len(request.new_content)}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating content: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Auto-Summarize (v0.3.6 Change 10) ====================

    async def _do_auto_summarize_one() -> Dict[str, Any]:
        """
        Find one long memory and summarize it. Shared by endpoint and background task.

        v0.3.6: Sidecar-first routing — uses user's configured backend (Custom > Haiku > Zen > Ollama).
        Returns dict with {summarized, reason, doc_id, old_len, new_len} or
        {summarized: false, reason, doc_id, collection, content, old_len} for plugin Zen fallback.
        """
        if not _memory:
            return {"summarized": False, "reason": "memory_not_ready"}

        # Search for candidates across working, history, patterns
        candidates = []
        for coll_name in ["working", "history", "patterns"]:
            try:
                results = await _memory.search(
                    query="",
                    collections=[coll_name],
                    limit=10,
                    sort_by="recency"
                )
                for r in results:
                    content = r.get("content", "")
                    metadata = r.get("metadata", {})
                    if len(content) > 400 and not metadata.get("summarized_at"):
                        candidates.append((coll_name, r))
                        if len(candidates) >= 1:
                            break
            except Exception as e:
                logger.warning(f"Auto-summarize search error for {coll_name}: {e}")

            if candidates:
                break

        if not candidates:
            return {"summarized": False, "reason": "no_candidates"}

        coll_name, memory = candidates[0]
        content = memory.get("content", "")
        doc_id = memory.get("id", memory.get("doc_id", ""))

        # Summarize via sidecar (Custom > Haiku > Zen > Ollama > claude -p)
        try:
            from roampal.sidecar_service import summarize_only
            summary = await asyncio.to_thread(summarize_only, content)
        except Exception as e:
            logger.warning(f"Auto-summarize backend error: {e}")
            return {
                "summarized": False, "reason": "backend_failed",
                "doc_id": doc_id, "collection": coll_name,
                "content": content, "old_len": len(content)
            }

        if not summary:
            return {
                "summarized": False, "reason": "backend_failed",
                "doc_id": doc_id, "collection": coll_name,
                "content": content, "old_len": len(content)
            }

        # Enforce summary length to prevent re-summarization loops
        if len(summary) > 400:
            summary = summary[:380] + "... [truncated]"

        # Update memory with summary
        try:
            adapter = _memory.collections.get(coll_name)
            if not adapter:
                return {"summarized": False, "reason": "collection_not_found"}

            doc = adapter.get_fragment(doc_id)
            if not doc:
                return {"summarized": False, "reason": "doc_not_found"}

            metadata = doc.get("metadata", {})
            metadata["text"] = summary
            metadata["content"] = summary
            metadata["summarized_at"] = datetime.now().isoformat()
            metadata["original_length"] = len(content)

            embedding = await _memory._embedding_service.embed_text(summary)
            await adapter.upsert_vectors(
                ids=[doc_id],
                vectors=[embedding],
                metadatas=[metadata]
            )

            logger.info(f"Auto-summarized {doc_id}: {len(content)} -> {len(summary)} chars")
            return {
                "summarized": True, "doc_id": doc_id,
                "old_len": len(content), "new_len": len(summary)
            }
        except Exception as e:
            logger.error(f"Auto-summarize update error: {e}")
            return {"summarized": False, "reason": "update_failed"}

    async def _auto_summarize_one_memory():
        """
        Background task: summarize one long memory. Non-blocking.

        v0.3.6: Spawned at end of stop_hook handler via asyncio.create_task().
        Follows existing pattern from _deferred_action_kg_updates().
        """
        try:
            result = await _do_auto_summarize_one()
            if result.get("summarized"):
                logger.info(f"[Background] Auto-summarized {result['doc_id']}: {result['old_len']} -> {result['new_len']} chars")
            elif result.get("reason") == "no_candidates":
                logger.debug("[Background] No auto-summarize candidates found")
            else:
                logger.info(f"[Background] Auto-summarize skipped: {result.get('reason', 'unknown')}")
        except Exception as e:
            logger.error(f"[Background] Auto-summarize error: {e}")

    @app.post("/api/memory/auto-summarize-one")
    async def auto_summarize_one():
        """
        Find and summarize one long memory.

        v0.3.6: Called by OpenCode plugin (session.idle) and Claude Code background task.
        Sidecar-first: uses user's configured backend before plugin falls back to Zen.

        Returns:
            {summarized: true, doc_id, old_len, new_len} on success
            {summarized: false, reason: "no_candidates"} if nothing to summarize
            {summarized: false, reason: "backend_failed", doc_id, collection, content, old_len}
                if sidecar failed (plugin can fall back to Zen)
        """
        if not _memory:
            raise HTTPException(status_code=503, detail="Memory system not ready")

        result = await _do_auto_summarize_one()
        return result

    # ==================== Health/Status Endpoints ====================

    @app.get("/api/health")
    async def health_check():
        """
        Health check endpoint.

        v0.3.0: Actually tests embedding functionality to catch PyTorch state corruption.
        Returns 503 if embeddings are broken, allowing auto-restart.
        """
        embedding_ok = False
        embedding_error = None

        if _memory and _memory.initialized and _memory._embedding_service:
            try:
                # Actually test embedding - catches [Errno 22] corruption
                test_vector = await _memory._embedding_service.embed_text("health check")
                embedding_ok = len(test_vector) > 0
            except Exception as e:
                embedding_error = str(e)

        if not embedding_ok:
            # Return 503 so _ensure_server_running knows to restart
            raise HTTPException(
                status_code=503,
                detail=f"Embedding service unhealthy: {embedding_error or 'not initialized'}"
            )

        return {
            "status": "healthy",
            "memory_initialized": _memory is not None and _memory.initialized,
            "session_manager_ready": _session_manager is not None,
            "embedding_ok": embedding_ok,
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
