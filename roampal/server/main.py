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

    Dumps ALL memory_bank facts to give the LLM a complete picture
    of who the user is at the start of each session.

    Returns:
        Formatted user profile string, or None if no facts exist
    """
    if not _memory:
        return None

    try:
        # Get memory_bank facts - use a broad query to find user-related facts
        # Empty query returns random results, so we search for user-specific content
        # Search for user identity and preference facts
        all_facts = await _memory.search(
            query="user name identity who is preference goal project communication style background",
            collections=["memory_bank"],
            limit=50  # Get up to 50 facts for cold start
        )

        if not all_facts:
            return None

        # Group facts by tags for organization
        identity_facts = []
        preference_facts = []
        goal_facts = []
        project_facts = []
        other_facts = []

        for fact in all_facts:
            # Content can be in 'text', 'content', or metadata
            content = fact.get("text") or fact.get("content") or fact.get("metadata", {}).get("content", "")
            # Tags can be a list or JSON string
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    import json as json_module
                    tags = json_module.loads(tags_raw)
                except:
                    tags = []
            else:
                tags = tags_raw

            # Debug logging
            logger.debug(f"Cold start fact: content={content[:50]}..., tags={tags}")

            if "identity" in tags:
                identity_facts.append(content)
            elif "preference" in tags:
                preference_facts.append(content)
            elif "goal" in tags:
                goal_facts.append(content)
            elif "project" in tags:
                project_facts.append(content)
            else:
                other_facts.append(content)

        # Build formatted profile
        profile_parts = ["<roampal-user-profile>"]
        profile_parts.append("COLD START: Here's everything you know about this user:\n")

        if identity_facts:
            profile_parts.append("IDENTITY:")
            for fact in identity_facts:
                profile_parts.append(f"  - {fact}")

        if preference_facts:
            profile_parts.append("\nPREFERENCES:")
            for fact in preference_facts:
                profile_parts.append(f"  - {fact}")

        if goal_facts:
            profile_parts.append("\nGOALS:")
            for fact in goal_facts:
                profile_parts.append(f"  - {fact}")

        if project_facts:
            profile_parts.append("\nPROJECTS:")
            for fact in project_facts:
                profile_parts.append(f"  - {fact}")

        if other_facts:
            profile_parts.append("\nOTHER FACTS:")
            for fact in other_facts:
                profile_parts.append(f"  - {fact}")

        profile_parts.append("\nUse this context to personalize your responses.")
        profile_parts.append("</roampal-user-profile>\n")

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
    related: Optional[List[str]] = None  # Optional: filter to only score these doc_ids


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

    # Initialize session manager (uses same data path)
    _session_manager = SessionManager(_memory.data_path)
    logger.info("Session manager initialized")

    yield

    # Cleanup
    logger.info("Shutting down Roampal server...")


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

            # 1. Get memory context FIRST (needed for scoring prompt to include surfaced memories)
            context = await _memory.get_context_for_injection(
                query=request.query,
                conversation_id=conversation_id,
                recent_conversation=request.recent_messages
            )

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

            # 3. Add memory context after scoring prompt
            if context.get("formatted_injection"):
                formatted_parts.append(context["formatted_injection"])

            # 3. Cache doc_ids for outcome scoring via record_response
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
            logger.error(f"Error getting context: {e}")
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

            # Store the exchange in working memory
            content = f"User: {request.user_message}\n\nAssistant: {request.assistant_response}"
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

            # v0.2.3: Skip session file scan - trust the search cache
            # The slow get_most_recent_unscored_exchange() scanned ALL session files (O(n×m) I/O)
            # but doc_ids are already available in _search_cache or request.related
            # See: https://github.com/roampal/roampal-core/issues/XXX (performance fix)

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

            # 2a. If related doc_ids provided, use directly (bypass stale cache)
            # This fixes the timing issue where cache is overwritten before scoring
            if request.related is not None and len(request.related) > 0:
                doc_ids_scored.extend(request.related)
                logger.info(f"Direct scoring: {len(request.related)} doc_ids from related param")
            elif cached_doc_ids:
                # No related filter - score all cached (backwards compatible)
                doc_ids_scored.extend(cached_doc_ids)
                logger.info(f"Cache scoring: {len(cached_doc_ids)} doc_ids")

            # 3. Apply outcome to filtered documents
            if doc_ids_scored and request.outcome in ["worked", "failed", "partial"]:
                result = await _memory.record_outcome(
                    doc_ids=doc_ids_scored,
                    outcome=request.outcome
                )
                logger.info(f"Scored {len(doc_ids_scored)} documents with outcome '{request.outcome}'")

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
            else:
                result = {"outcome": request.outcome, "documents_updated": 0}

            # Clear search cache for the key we used
            if cache_key_used in _search_cache:
                del _search_cache[cache_key_used]

            return {
                "success": True,
                "outcome": request.outcome,
                "documents_scored": len(doc_ids_scored),
                **result
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
