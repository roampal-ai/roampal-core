"""
Tests for FastAPI endpoints (server/main.py).

Tests the HTTP API layer that all clients (Claude Code, Cursor, OpenCode)
proxy through. Uses FastAPI TestClient with mocked memory/session backends.

v0.3.2: Covers all endpoints including /api/record-response and split delivery fields.
v0.3.6: /api/context-insights removed (hooks/plugin inject context automatically).
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_memory():
    """Create a mock UnifiedMemorySystem."""
    memory = MagicMock()
    memory.initialize = AsyncMock()
    memory.search = AsyncMock(return_value=[])
    memory.store_memory_bank = AsyncMock(return_value="mb_test123")
    memory.update_memory_bank = AsyncMock(return_value="mb_test123")
    memory.delete_memory_bank = AsyncMock(return_value=True)
    memory.store_working = AsyncMock(return_value="working_test123")
    memory.record_outcome = AsyncMock()
    memory.get_context_for_injection = AsyncMock(return_value={
        "formatted_injection": "<test-context>memories here</test-context>",
        "user_facts": [{"content": "User is a developer"}],
        "relevant_memories": [{"content": "Previous work on testing", "collection": "working"}],
        "context_summary": "Test context",
        "doc_ids": ["working_abc", "patterns_def"]
    })
    memory.list_books = AsyncMock(return_value=[])
    memory.remove_book = AsyncMock(return_value={"removed": 1, "message": "Removed"})
    memory.store_book = AsyncMock(return_value=["book_1", "book_2"])
    memory.detect_context_type = AsyncMock(return_value="general")
    memory.record_action_outcome = AsyncMock()
    memory._update_kg_routing = AsyncMock()
    memory.data_path = "/tmp/roampal_test"
    memory._memory_bank_service = MagicMock()
    memory._memory_bank_service.cleanup_archived = MagicMock(return_value=0)
    memory._memory_bank_service.list_all = MagicMock(return_value=[])
    memory.get_always_inject = MagicMock(return_value=[])
    return memory


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager."""
    sm = MagicMock()
    sm.is_first_message = MagicMock(return_value=False)
    sm.mark_first_message_seen = MagicMock()
    sm.check_and_clear_completed = MagicMock(return_value=False)
    sm.get_previous_exchange = AsyncMock(return_value=None)
    sm.build_scoring_prompt = MagicMock(return_value="<score>Score this</score>")
    sm.build_scoring_prompt_simple = MagicMock(return_value="REQUIRED: Call score_memories to score if these memories were helpful.")
    sm.set_scoring_required = MagicMock()
    sm.store_exchange = AsyncMock()
    sm.was_scoring_required = MagicMock(return_value=False)
    sm.was_scored_this_turn = MagicMock(return_value=False)
    sm.set_completed = MagicMock()
    sm.set_scored_this_turn = MagicMock()
    sm._last_exchange_cache = {}
    return sm


@pytest.fixture
def client(mock_memory, mock_session_manager):
    """Create a FastAPI TestClient with mocked backends."""
    from httpx import AsyncClient, ASGITransport
    from roampal.server import main

    # Patch globals before creating app
    original_memory = main._memory
    original_sm = main._session_manager
    main._memory = mock_memory
    main._session_manager = mock_session_manager

    # Create app without lifespan (we already set globals)
    app = main.create_app()

    # Override lifespan to avoid real initialization
    @app.on_event("startup")
    async def skip_startup():
        pass

    yield app, mock_memory, mock_session_manager

    # Restore
    main._memory = original_memory
    main._session_manager = original_sm


@pytest.fixture
async def async_client(client):
    """Create async HTTP client for testing."""
    from httpx import AsyncClient, ASGITransport
    app, mock_memory, mock_session_manager = client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, mock_memory, mock_session_manager


# ============================================================================
# Health Endpoint
# ============================================================================

class TestHealthEndpoint:
    """Test /api/health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200_when_healthy(self, async_client):
        """Health check passes when memory is initialized with working embeddings."""
        ac, mock_mem, _ = async_client
        # Health check requires: _memory.initialized, _memory._embedding_service.embed_text()
        mock_mem.initialized = True
        mock_embedding = MagicMock()
        mock_embedding.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_mem._embedding_service = mock_embedding

        response = await ac.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_returns_503_when_not_initialized(self, async_client):
        """Health check returns 503 when memory not initialized."""
        from roampal.server import main
        ac, _, _ = async_client

        original = main._memory
        main._memory = None
        try:
            response = await ac.get("/api/health")
            assert response.status_code == 503
        finally:
            main._memory = original


# ============================================================================
# Get Context Endpoint
# ============================================================================

class TestGetContextEndpoint:
    """Test /api/hooks/get-context endpoint."""

    @pytest.mark.asyncio
    async def test_basic_context_request(self, async_client):
        """Basic context request returns formatted injection."""
        ac, mock_mem, _ = async_client
        response = await ac.post("/api/hooks/get-context", json={
            "query": "test query",
            "conversation_id": "test_session"
        })
        assert response.status_code == 200
        data = response.json()
        assert "formatted_injection" in data
        assert "user_facts" in data
        assert "relevant_memories" in data
        assert "scoring_required" in data

    @pytest.mark.asyncio
    async def test_split_delivery_fields_present(self, async_client):
        """v0.3.2: Response includes scoring_prompt and context_only fields."""
        ac, _, _ = async_client
        response = await ac.post("/api/hooks/get-context", json={
            "query": "test query",
            "conversation_id": "test_session"
        })
        assert response.status_code == 200
        data = response.json()
        assert "scoring_prompt" in data
        assert "context_only" in data

    @pytest.mark.asyncio
    async def test_scoring_prompt_when_exchange_unscored(self, async_client):
        """Scoring prompt injected when previous exchange is unscored."""
        ac, mock_mem, mock_sm = async_client

        # Simulate: assistant completed, previous exchange exists and is unscored
        mock_sm.check_and_clear_completed.return_value = True
        mock_sm.get_previous_exchange = AsyncMock(return_value={
            "user_message": "hello",
            "assistant_response": "hi there",
            "scored": False,
            "doc_id": "working_123"
        })

        response = await ac.post("/api/hooks/get-context", json={
            "query": "new question",
            "conversation_id": "scoring_test"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["scoring_required"] is True
        assert data["scoring_prompt"] != ""

    @pytest.mark.asyncio
    async def test_no_scoring_on_cold_start(self, async_client):
        """No scoring prompt on cold start (first message of session)."""
        ac, _, mock_sm = async_client
        mock_sm.is_first_message.return_value = True

        response = await ac.post("/api/hooks/get-context", json={
            "query": "first message",
            "conversation_id": "cold_start_test"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["scoring_required"] is False

    @pytest.mark.asyncio
    async def test_context_caches_doc_ids(self, async_client):
        """Context endpoint caches doc_ids for later scoring."""
        from roampal.server import main

        ac, _, _ = async_client
        response = await ac.post("/api/hooks/get-context", json={
            "query": "test caching",
            "conversation_id": "cache_test"
        })
        assert response.status_code == 200

        # Verify doc_ids were cached
        cached = main._search_cache.get("cache_test")
        assert cached is not None
        assert "doc_ids" in cached
        # Clean up
        main._search_cache.pop("cache_test", None)

    @pytest.mark.asyncio
    async def test_503_when_memory_not_ready(self, async_client):
        """Returns 503 when memory system not initialized."""
        from roampal.server import main
        ac, _, _ = async_client

        original = main._memory
        main._memory = None
        try:
            response = await ac.post("/api/hooks/get-context", json={
                "query": "test",
                "conversation_id": "test"
            })
            assert response.status_code == 503
        finally:
            main._memory = original


# ============================================================================
# Stop Hook Endpoint
# ============================================================================

class TestStopHookEndpoint:
    """Test /api/hooks/stop endpoint."""

    @pytest.mark.asyncio
    async def test_lifecycle_only_skips_chromadb(self, async_client):
        """lifecycle_only=True stores in JSONL but skips ChromaDB."""
        ac, mock_mem, mock_sm = async_client
        response = await ac.post("/api/hooks/stop", json={
            "conversation_id": "lifecycle_test",
            "user_message": "What is Python?",
            "assistant_response": "Python is a programming language.",
            "lifecycle_only": True
        })
        assert response.status_code == 200
        data = response.json()
        assert data["stored"] is False  # No ChromaDB doc_id
        # ChromaDB store_working must NOT be called
        mock_mem.store_working.assert_not_called()
        # But session JSONL store_exchange MUST be called
        mock_sm.store_exchange.assert_called_once_with(
            conversation_id="lifecycle_test",
            user_message="What is Python?",
            assistant_response="Python is a programming language.",
            doc_id=""
        )

    @pytest.mark.asyncio
    async def test_stores_exchange(self, async_client):
        """Stop hook stores exchange in working memory."""
        ac, mock_mem, mock_sm = async_client
        response = await ac.post("/api/hooks/stop", json={
            "conversation_id": "stop_test",
            "user_message": "What is Python?",
            "assistant_response": "Python is a programming language."
        })
        assert response.status_code == 200
        data = response.json()
        assert data["stored"] is True
        assert data["doc_id"] is not None

        # Verify store_working was called
        mock_mem.store_working.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_empty_exchange(self, async_client):
        """Stop hook skips storage for empty messages and returns stored=False."""
        ac, mock_mem, _ = async_client
        response = await ac.post("/api/hooks/stop", json={
            "conversation_id": "empty_test",
            "user_message": "",
            "assistant_response": ""
        })
        assert response.status_code == 200
        data = response.json()
        assert data["stored"] is False
        assert data["should_block"] is False
        mock_mem.store_working.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_block_when_scoring_not_required(self, async_client):
        """Stop hook doesn't block when scoring wasn't required."""
        ac, _, mock_sm = async_client
        mock_sm.was_scoring_required.return_value = False

        response = await ac.post("/api/hooks/stop", json={
            "conversation_id": "no_block",
            "user_message": "hello",
            "assistant_response": "hi"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["should_block"] is False

    @pytest.mark.asyncio
    async def test_marks_assistant_completed(self, async_client):
        """Stop hook marks assistant as completed for next scoring cycle."""
        ac, _, mock_sm = async_client
        response = await ac.post("/api/hooks/stop", json={
            "conversation_id": "complete_test",
            "user_message": "hello",
            "assistant_response": "hi"
        })
        assert response.status_code == 200
        mock_sm.set_completed.assert_called_once_with("complete_test")


# ============================================================================
# Search Endpoint
# ============================================================================

class TestSearchEndpoint:
    """Test /api/search endpoint."""

    @pytest.mark.asyncio
    async def test_basic_search(self, async_client):
        """Basic search returns results."""
        ac, mock_mem, _ = async_client
        mock_mem.search = AsyncMock(return_value=[
            {"id": "test_1", "content": "Test result", "collection": "working",
             "metadata": {"score": 0.8}}
        ])

        response = await ac.post("/api/search", json={
            "query": "test query",
            "conversation_id": "search_test",
            "limit": 5
        })
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["results"]) == 1

    @pytest.mark.asyncio
    async def test_search_caches_doc_ids(self, async_client):
        """Search caches doc_ids for scoring."""
        from roampal.server import main

        ac, mock_mem, _ = async_client
        mock_mem.search = AsyncMock(return_value=[
            {"id": "doc_abc", "content": "result", "collection": "patterns"}
        ])

        response = await ac.post("/api/search", json={
            "query": "test",
            "conversation_id": "cache_search_test",
            "limit": 5
        })
        assert response.status_code == 200

        cached = main._search_cache.get("cache_search_test")
        assert cached is not None
        assert "doc_abc" in cached["doc_ids"]
        main._search_cache.pop("cache_search_test", None)

    @pytest.mark.asyncio
    async def test_search_with_collections_filter(self, async_client):
        """Search passes collection filters to memory system."""
        ac, mock_mem, _ = async_client
        mock_mem.search = AsyncMock(return_value=[])

        response = await ac.post("/api/search", json={
            "query": "test",
            "collections": ["patterns", "memory_bank"],
            "limit": 3
        })
        assert response.status_code == 200
        mock_mem.search.assert_called_once()
        call_kwargs = mock_mem.search.call_args
        assert call_kwargs.kwargs.get("collections") == ["patterns", "memory_bank"]


# ============================================================================
# Memory Bank Endpoints
# ============================================================================

class TestMemoryBankEndpoints:
    """Test /api/memory-bank/* endpoints."""

    @pytest.mark.asyncio
    async def test_add_to_memory_bank(self, async_client):
        """Add fact to memory bank."""
        ac, mock_mem, _ = async_client
        response = await ac.post("/api/memory-bank/add", json={
            "content": "User prefers dark mode",
            "tags": ["preference"],
            "importance": 0.8,
            "confidence": 0.9
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["doc_id"] == "mb_test123"

    @pytest.mark.asyncio
    async def test_add_with_always_inject(self, async_client):
        """v0.3.2: Add with always_inject flag."""
        ac, mock_mem, _ = async_client
        response = await ac.post("/api/memory-bank/add", json={
            "content": "User's name is Test",
            "tags": ["identity"],
            "always_inject": True
        })
        assert response.status_code == 200
        mock_mem.store_memory_bank.assert_called_once()
        call_kwargs = mock_mem.store_memory_bank.call_args
        assert call_kwargs.kwargs.get("always_inject") is True

    @pytest.mark.asyncio
    async def test_update_memory_bank(self, async_client):
        """Update existing memory bank entry."""
        ac, mock_mem, _ = async_client
        response = await ac.post("/api/memory-bank/update", json={
            "old_content": "User prefers light mode",
            "new_content": "User prefers dark mode"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_update_not_found(self, async_client):
        """Update returns success=false when memory not found."""
        ac, mock_mem, _ = async_client
        mock_mem.update_memory_bank = AsyncMock(return_value=None)

        response = await ac.post("/api/memory-bank/update", json={
            "old_content": "nonexistent",
            "new_content": "new content"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_archive_memory_bank(self, async_client):
        """Archive (delete) memory bank entry."""
        ac, mock_mem, _ = async_client
        response = await ac.post("/api/memory-bank/archive", json={
            "content": "outdated fact"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_archive_empty_content_400(self, async_client):
        """Archive with empty content returns 400."""
        ac, _, _ = async_client
        response = await ac.post("/api/memory-bank/archive", json={
            "content": ""
        })
        assert response.status_code == 400


# ============================================================================
# Record Outcome Endpoint
# ============================================================================

class TestRecordOutcomeEndpoint:
    """Test /api/record-outcome endpoint."""

    @pytest.mark.asyncio
    async def test_record_outcome_with_memory_scores(self, async_client):
        """v0.2.8: Per-memory scoring with individual outcomes."""
        ac, mock_mem, mock_sm = async_client
        response = await ac.post("/api/record-outcome", json={
            "conversation_id": "score_test",
            "outcome": "worked",
            "memory_scores": {
                "patterns_abc": "worked",
                "working_def": "partial"
            }
        })
        assert response.status_code == 200
        data = response.json()
        assert data["documents_scored"] >= 0

        # Verify record_outcome was called for each scored memory
        assert mock_mem.record_outcome.call_count >= 1

    @pytest.mark.asyncio
    async def test_record_outcome_marks_scored(self, async_client):
        """Record outcome marks the turn as scored."""
        ac, _, mock_sm = async_client
        response = await ac.post("/api/record-outcome", json={
            "conversation_id": "mark_scored",
            "outcome": "worked",
            "memory_scores": {}
        })
        assert response.status_code == 200
        mock_sm.set_scored_this_turn.assert_called_once_with("mark_scored", True)


# ============================================================================
# Injection Map Tests (v0.3.2 - Bug 9 fix)
# ============================================================================

class TestInjectionMap:
    """Test injection_map for multi-session scoring correlation.

    v0.3.2: The injection_map tracks which doc_ids were injected to which
    conversation, enabling correct scoring even when MCP uses a different
    session ID than hooks.
    """

    @pytest.mark.asyncio
    async def test_get_context_populates_injection_map(self, async_client):
        """get-context should populate _injection_map with injected doc_ids."""
        from roampal.server.main import _injection_map

        ac, mock_mem, mock_sm = async_client

        # Clear injection map before test
        _injection_map.clear()

        # Configure mock to return doc_ids
        mock_mem.get_context_for_injection = AsyncMock(return_value={
            "formatted_injection": "<test>context</test>",
            "user_facts": [],
            "relevant_memories": [],
            "context_summary": "",
            "doc_ids": ["working_abc123", "patterns_xyz789"]
        })

        response = await ac.post("/api/hooks/get-context", json={
            "query": "test query",
            "conversation_id": "ses_unique123"
        })
        assert response.status_code == 200

        # Injection map should now have both doc_ids mapped to this conversation
        assert "working_abc123" in _injection_map
        assert "patterns_xyz789" in _injection_map
        assert _injection_map["working_abc123"]["conversation_id"] == "ses_unique123"
        assert _injection_map["patterns_xyz789"]["conversation_id"] == "ses_unique123"

    @pytest.mark.asyncio
    async def test_record_outcome_uses_injection_map(self, async_client):
        """record_outcome should resolve conversation via injection_map."""
        from roampal.server.main import _injection_map, _search_cache

        ac, mock_mem, mock_sm = async_client

        # Set up injection_map with doc_id -> conversation mapping
        _injection_map.clear()
        _injection_map["patterns_test456"] = {
            "conversation_id": "ses_actual789",
            "injected_at": "2024-01-01T00:00:00",
            "query": "test"
        }

        # Set up search cache under the actual session ID
        _search_cache["ses_actual789"] = {
            "doc_ids": ["patterns_test456"],
            "query": "test",
            "timestamp": "2024-01-01T00:00:00"
        }

        # MCP calls with "default" session ID but scores a doc_id from injection_map
        response = await ac.post("/api/record-outcome", json={
            "conversation_id": "default",  # MCP uses "default"
            "outcome": "worked",
            "memory_scores": {
                "patterns_test456": "worked"  # This doc_id is in injection_map
            }
        })
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_injection_map_cleaned_after_scoring(self, async_client):
        """Scored doc_ids should be removed from injection_map."""
        from roampal.server.main import _injection_map

        ac, mock_mem, mock_sm = async_client

        # Set up injection_map
        _injection_map.clear()
        _injection_map["working_cleanup_test"] = {
            "conversation_id": "ses_cleanup",
            "injected_at": "2024-01-01T00:00:00",
            "query": "test"
        }

        # Score the document
        response = await ac.post("/api/record-outcome", json={
            "conversation_id": "default",
            "outcome": "worked",
            "memory_scores": {
                "working_cleanup_test": "worked"
            }
        })
        assert response.status_code == 200

        # Doc_id should be removed from injection_map after scoring
        assert "working_cleanup_test" not in _injection_map


# ============================================================================
# Record Response Endpoint (v0.3.2)
# ============================================================================

class TestRecordResponseEndpoint:
    """Test /api/record-response endpoint."""

    @pytest.mark.asyncio
    async def test_record_key_takeaway(self, async_client):
        """Record a key takeaway in working memory."""
        ac, mock_mem, _ = async_client
        response = await ac.post("/api/record-response", json={
            "key_takeaway": "User prefers concise responses",
            "conversation_id": "takeaway_test"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["doc_id"] is not None

        # Verify stored with initial score 0.7
        mock_mem.store_working.assert_called_once()
        call_kwargs = mock_mem.store_working.call_args
        assert call_kwargs.kwargs.get("initial_score") == 0.7


# ============================================================================
# Context Insights Endpoint — REMOVED in v0.3.6
# /api/context-insights removed: hooks/plugin inject context automatically
# ============================================================================


# ============================================================================
# Ingest / Books Endpoints
# ============================================================================

class TestBooksEndpoints:
    """Test /api/ingest, /api/books, /api/remove-book endpoints."""

    @pytest.mark.asyncio
    async def test_ingest_document(self, async_client):
        """Ingest a document into books collection."""
        ac, mock_mem, _ = async_client
        response = await ac.post("/api/ingest", json={
            "content": "This is a test document with some content.",
            "title": "Test Doc",
            "source": "test.md"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["chunks"] == 2

    @pytest.mark.asyncio
    async def test_ingest_empty_content_400(self, async_client):
        """Ingest with empty content returns 400."""
        ac, _, _ = async_client
        response = await ac.post("/api/ingest", json={
            "content": "",
            "title": "Empty"
        })
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_list_books(self, async_client):
        """List books returns book list."""
        ac, _, _ = async_client
        response = await ac.get("/api/books")
        assert response.status_code == 200
        data = response.json()
        assert "books" in data

    @pytest.mark.asyncio
    async def test_remove_book(self, async_client):
        """Remove book by title."""
        ac, _, _ = async_client
        response = await ac.post("/api/remove-book", json={
            "title": "Test Doc"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# ============================================================================
# Check-Scored Endpoint (v0.3.6)
# ============================================================================

class TestCheckScoredEndpoint:
    """Test /api/hooks/check-scored endpoint.

    v0.3.6: OpenCode plugin calls this to detect if the main LLM already
    called score_memories, so the sidecar can skip double-scoring.
    """

    @pytest.mark.asyncio
    async def test_returns_false_when_not_scored(self, async_client):
        """Default state: no scoring has happened this turn."""
        ac, _, mock_sm = async_client
        mock_sm.was_scored_this_turn.return_value = False

        response = await ac.get("/api/hooks/check-scored", params={
            "conversation_id": "test_session"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["scored"] is False
        mock_sm.was_scored_this_turn.assert_called_once_with("test_session")

    @pytest.mark.asyncio
    async def test_returns_true_after_scoring(self, async_client):
        """Returns true after record-outcome was called for this conversation."""
        ac, _, mock_sm = async_client
        mock_sm.was_scored_this_turn.return_value = True

        response = await ac.get("/api/hooks/check-scored", params={
            "conversation_id": "scored_session"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["scored"] is True

    @pytest.mark.asyncio
    async def test_returns_false_when_session_manager_none(self, async_client):
        """Graceful fallback when session manager not initialized."""
        from roampal.server import main
        ac, _, _ = async_client

        original = main._session_manager
        main._session_manager = None
        try:
            response = await ac.get("/api/hooks/check-scored", params={
                "conversation_id": "no_sm"
            })
            assert response.status_code == 200
            data = response.json()
            assert data["scored"] is False
        finally:
            main._session_manager = original

    @pytest.mark.asyncio
    async def test_empty_conversation_id(self, async_client):
        """Empty conversation_id still returns valid response."""
        ac, _, mock_sm = async_client
        mock_sm.was_scored_this_turn.return_value = False

        response = await ac.get("/api/hooks/check-scored")
        assert response.status_code == 200
        data = response.json()
        assert data["scored"] is False

    @pytest.mark.asyncio
    async def test_scoring_lifecycle_integration(self, async_client):
        """Full lifecycle: check-scored false -> record-outcome -> check-scored true."""
        ac, mock_mem, mock_sm = async_client

        # Step 1: Not scored yet
        mock_sm.was_scored_this_turn.return_value = False
        response = await ac.get("/api/hooks/check-scored", params={
            "conversation_id": "lifecycle_test"
        })
        assert response.json()["scored"] is False

        # Step 2: Score via record-outcome
        response = await ac.post("/api/record-outcome", json={
            "conversation_id": "lifecycle_test",
            "outcome": "worked",
            "memory_scores": {"working_abc": "worked"}
        })
        assert response.status_code == 200
        mock_sm.set_scored_this_turn.assert_called_with("lifecycle_test", True)

        # Step 3: Now check-scored should reflect the scoring
        mock_sm.was_scored_this_turn.return_value = True
        response = await ac.get("/api/hooks/check-scored", params={
            "conversation_id": "lifecycle_test"
        })
        assert response.json()["scored"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
