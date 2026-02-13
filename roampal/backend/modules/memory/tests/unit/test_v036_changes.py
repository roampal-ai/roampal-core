"""
v0.3.6 Change Verification Tests

Covers test gaps identified in the v0.3.6 audit:
1. SearchService._apply_collection_boost() distance multiplier (Change 2)
2. Promotion carry-forward: success_count/uses NOT reset (Change 3)
3. Batch promotion promoted_to_history_at (Change 4)
4. POST /api/memory/auto-summarize-one endpoint (Change 10)
5. _auto_summarize_one_memory() background task (Change 10)
6. ROAMPAL_SUMMARIZE_MODEL override in sidecar_service (Change 10)
7. roampal context --recent-exchanges CLI output (Change 11)
"""

import sys
import os
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

from roampal.backend.modules.memory.scoring_service import wilson_score_lower


# ============================================================================
# Test 1: SearchService._apply_collection_boost() (Changes 1-2)
# ============================================================================

class TestCollectionBoostDistanceMultiplier:
    """Test the distance multiplier math in _apply_collection_boost."""

    @pytest.fixture
    def search_service(self):
        """Create a SearchService with mocked dependencies."""
        from roampal.backend.modules.memory.search_service import SearchService
        from roampal.backend.modules.memory.scoring_service import ScoringService
        from roampal.backend.modules.memory.config import MemoryConfig

        config = MemoryConfig()
        scoring = ScoringService(config=config)
        routing = MagicMock()
        kg = MagicMock()
        # _calculate_entity_boost needs _doc_entities on kg_service
        kg._doc_entities = {}
        embed_fn = AsyncMock(return_value=[0.1, 0.2, 0.3])

        svc = SearchService(
            collections={},
            scoring_service=scoring,
            routing_service=routing,
            kg_service=kg,
            embed_fn=embed_fn,
            config=config
        )
        return svc

    def test_memory_bank_50_50_blend_high_quality_bad_wilson(self, search_service):
        """v0.3.6: 50/50 blend — high quality + bad Wilson penalized."""
        result = {
            "distance": 1.0,
            "id": "mb_test1",
            "metadata": {
                "importance": 0.9,
                "confidence": 0.9,
                "uses": 10,
                "success_count": 3.0  # bad wilson
            }
        }
        search_service._apply_collection_boost(result, "memory_bank", "test query")

        # quality = 0.9 * 0.9 = 0.81
        # wilson = wilson_score_lower(3.0, 10) ≈ 0.147
        # blended = 0.5 * 0.81 + 0.5 * 0.147 ≈ 0.479
        # metadata_boost = 1.0 - 0.479 * 0.4 ≈ 0.808
        assert 0.7 < result["distance"] < 0.9

    def test_memory_bank_50_50_blend_high_quality_good_wilson(self, search_service):
        """v0.3.6: 50/50 blend — high quality + good Wilson = strong boost."""
        result = {
            "distance": 1.0,
            "id": "mb_test2",
            "metadata": {
                "importance": 0.9,
                "confidence": 0.9,
                "uses": 10,
                "success_count": 9.0  # great wilson
            }
        }
        search_service._apply_collection_boost(result, "memory_bank", "test query")

        # quality = 0.81, wilson ≈ 0.74
        # blended ≈ 0.775
        # metadata_boost = 1.0 - 0.775 * 0.4 ≈ 0.69
        assert 0.55 < result["distance"] < 0.75

    def test_memory_bank_cold_start_quality_only(self, search_service):
        """Cold start (uses < 3) uses quality only — no Wilson."""
        result = {
            "distance": 1.0,
            "id": "mb_test3",
            "metadata": {
                "importance": 0.8,
                "confidence": 0.8,
                "uses": 1,
                "success_count": 0.0
            }
        }
        search_service._apply_collection_boost(result, "memory_bank", "test query")

        # quality = 0.64, no wilson blend
        # metadata_boost = 1.0 - 0.64 * 0.4 = 0.744
        assert abs(result["distance"] - 0.744) < 0.01

    def test_0_4_multiplier_not_0_8(self, search_service):
        """v0.3.6: Perfect score gives 0.6 distance (40% boost), NOT 0.2 (80%)."""
        result = {
            "distance": 1.0,
            "id": "mb_test4",
            "metadata": {
                "importance": 1.0,
                "confidence": 1.0,
                "uses": 0,  # cold start — quality only
                "success_count": 0.0
            }
        }
        search_service._apply_collection_boost(result, "memory_bank", "test query")

        # quality = 1.0 * 1.0 = 1.0 (cold start, quality only)
        # metadata_boost = 1.0 - 1.0 * 0.4 = 0.6
        assert abs(result["distance"] - 0.6) < 0.01

    def test_patterns_get_0_9_boost(self, search_service):
        """Patterns collection gets 10% distance reduction."""
        result = {"distance": 1.0, "id": "patterns_test"}
        search_service._apply_collection_boost(result, "patterns", "test query")
        assert abs(result["distance"] - 0.9) < 0.01

    def test_working_gets_no_collection_boost(self, search_service):
        """Working collection gets no special distance boost."""
        result = {"distance": 1.0, "id": "working_test", "metadata": {}}
        search_service._apply_collection_boost(result, "working", "test query")
        assert abs(result["distance"] - 1.0) < 0.01

    def test_entity_boost_applies_to_all_collections(self, search_service):
        """v0.3.6: Entity boost applies after collection-specific boosts."""
        # Mock _calculate_entity_boost to return > 1.0
        search_service._calculate_entity_boost = MagicMock(return_value=1.5)

        for coll in ["working", "history", "patterns"]:
            result = {"distance": 1.0, "id": f"{coll}_test", "metadata": {}}
            search_service._apply_collection_boost(result, coll, "test query")
            # distance = 1.0 / 1.5 ≈ 0.667 (for working/history which have no other boost)
            assert result["distance"] < 1.0, f"Entity boost not applied to {coll}"

    def test_wilson_uses_proper_function_not_naive_ratio(self, search_service):
        """v0.3.6: Uses wilson_score_lower(), not naive success_count/uses."""
        result = {
            "distance": 1.0,
            "id": "mb_test5",
            "metadata": {
                "importance": 0.7,
                "confidence": 0.7,
                "uses": 1,
                "success_count": 1.0  # naive = 1.0, wilson ≈ 0.206
            }
        }
        # uses < 3 = cold start, but let's test uses >= 3
        result["metadata"]["uses"] = 3
        result["metadata"]["success_count"] = 3.0

        search_service._apply_collection_boost(result, "memory_bank", "test query")

        # If naive: blended = 0.5 * 0.49 + 0.5 * 1.0 = 0.745
        # If wilson: wilson(3,3) ≈ 0.438, blended = 0.5 * 0.49 + 0.5 * 0.438 = 0.464
        # metadata_boost_naive = 1.0 - 0.745 * 0.4 = 0.702
        # metadata_boost_wilson = 1.0 - 0.464 * 0.4 = 0.814
        # If using wilson, distance should be > 0.75 (not < 0.75 like naive)
        assert result["distance"] > 0.75, "Should use wilson_score_lower, not naive ratio"


# ============================================================================
# Test 2: Promotion Carry-Forward (Change 3)
# ============================================================================

class TestPromotionCarryForward:
    """Test that success_count and uses are NOT reset on promotion."""

    @pytest.fixture
    def promotion_service(self):
        """Create a PromotionService with mocked collections."""
        from roampal.backend.modules.memory.promotion_service import PromotionService
        from roampal.backend.modules.memory.config import MemoryConfig

        working = MagicMock()
        history = MagicMock()
        history.upsert_vectors = AsyncMock()

        collections = {"working": working, "history": history}
        embed_fn = AsyncMock(return_value=[0.1, 0.2, 0.3])

        svc = PromotionService(
            collections=collections,
            embed_fn=embed_fn,
            config=MemoryConfig()
        )
        return svc, working, history

    @pytest.mark.asyncio
    async def test_working_to_history_carries_success_count(self, promotion_service):
        """v0.3.6: success_count NOT reset on working → history."""
        svc, working, history = promotion_service

        metadata = {
            "text": "test memory content",
            "score": 0.8,
            "uses": 5,
            "success_count": 3.0,
            "timestamp": datetime.now().isoformat()
        }

        doc = {"content": "test memory content", "metadata": metadata}
        working.get_fragment = MagicMock(return_value=doc)
        working.delete_vectors = MagicMock()

        result = await svc._promote_working_to_history(
            "working_test123", doc, metadata.copy(), 0.8, 5
        )

        # Verify upsert was called
        assert history.upsert_vectors.called

        # Get the metadata that was upserted
        call_args = history.upsert_vectors.call_args
        upserted_metadata = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas", [{}])
        meta = upserted_metadata[0]

        # v0.3.6: success_count and uses should be carried forward, NOT reset
        assert meta["success_count"] == 3.0, "success_count should NOT be reset to 0"
        assert meta["uses"] == 5, "uses should NOT be reset to 0"
        assert "promoted_to_history_at" in meta

    @pytest.mark.asyncio
    async def test_working_to_history_sets_promoted_to_history_at(self, promotion_service):
        """promoted_to_history_at timestamp is set on promotion."""
        svc, working, history = promotion_service

        metadata = {
            "text": "test content",
            "score": 0.8,
            "uses": 3,
            "success_count": 2.0,
            "timestamp": datetime.now().isoformat()
        }

        doc = {"content": "test content", "metadata": metadata}
        working.get_fragment = MagicMock(return_value=doc)
        working.delete_vectors = MagicMock()

        await svc._promote_working_to_history(
            "working_test456", doc, metadata.copy(), 0.8, 3
        )

        call_args = history.upsert_vectors.call_args
        upserted_metadata = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas", [{}])
        meta = upserted_metadata[0]

        assert "promoted_to_history_at" in meta
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(meta["promoted_to_history_at"])


# ============================================================================
# Test 3: Batch Promotion Parity (Change 4)
# ============================================================================

class TestBatchPromotionParity:
    """Test batch promotion sets promoted_to_history_at and doesn't reset counters."""

    @pytest.fixture
    def promotion_service(self):
        from roampal.backend.modules.memory.promotion_service import PromotionService
        from roampal.backend.modules.memory.config import MemoryConfig

        working = MagicMock()
        history = MagicMock()
        history.upsert_vectors = AsyncMock()

        collections = {"working": working, "history": history}
        embed_fn = AsyncMock(return_value=[0.1, 0.2, 0.3])

        svc = PromotionService(
            collections=collections,
            embed_fn=embed_fn,
            config=MemoryConfig()
        )
        return svc, working, history

    @pytest.mark.asyncio
    async def test_batch_promotion_sets_promoted_to_history_at(self, promotion_service):
        """v0.3.6: Batch promotion includes promoted_to_history_at for parity."""
        svc, working, history = promotion_service

        # Mock the working adapter's collection and methods
        working.collection = MagicMock()  # Truthy so _do_batch_promotion doesn't bail
        working.list_all_ids = MagicMock(return_value=["working_batch1"])
        working.get_fragment = MagicMock(return_value={
            "content": "batch test content",
            "metadata": {
                "text": "batch test content",
                "score": 0.9,
                "uses": 3,
                "success_count": 2.0,
                "timestamp": datetime.now().isoformat()
            }
        })
        working.delete_vectors = MagicMock()

        await svc._do_batch_promotion()

        # Verify history.upsert_vectors was called
        assert history.upsert_vectors.called

        call_args = history.upsert_vectors.call_args
        upserted_metadata = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas", [{}])
        meta = upserted_metadata[0]

        assert "promoted_to_history_at" in meta
        assert meta["promotion_reason"] == "batch_promotion"
        # Counters should be carried forward (via **metadata spread)
        assert meta["success_count"] == 2.0
        assert meta["uses"] == 3


# ============================================================================
# Test 4: Auto-Summarize Endpoint (Change 10)
# ============================================================================

class TestAutoSummarizeEndpoint:
    """Test POST /api/memory/auto-summarize-one endpoint."""

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[])
        memory.collections = {}
        memory._embedding_service = MagicMock()
        memory._embedding_service.embed_text = AsyncMock(return_value=[0.1, 0.2])
        return memory

    @pytest.fixture
    def mock_session_manager(self):
        sm = MagicMock()
        sm.is_first_message = MagicMock(return_value=False)
        sm.mark_first_message_seen = MagicMock()
        sm.check_and_clear_completed = MagicMock(return_value=False)
        sm.get_previous_exchange = AsyncMock(return_value=None)
        sm.build_scoring_prompt = MagicMock(return_value="<score>Score</score>")
        sm.build_scoring_prompt_simple = MagicMock(return_value="Score prompt")
        sm.set_scoring_required = MagicMock()
        sm.store_exchange = AsyncMock()
        sm.was_scoring_required = MagicMock(return_value=False)
        sm.was_scored_this_turn = MagicMock(return_value=False)
        sm.set_completed = MagicMock()
        sm.set_scored_this_turn = MagicMock()
        sm._last_exchange_cache = {}
        return sm

    @pytest.fixture
    async def async_client(self, mock_memory, mock_session_manager):
        from httpx import AsyncClient, ASGITransport
        from roampal.server import main

        original_memory = main._memory
        original_sm = main._session_manager
        main._memory = mock_memory
        main._session_manager = mock_session_manager

        app = main.create_app()

        @app.on_event("startup")
        async def skip_startup():
            pass

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, mock_memory

        main._memory = original_memory
        main._session_manager = original_sm

    @pytest.mark.asyncio
    async def test_auto_summarize_no_candidates(self, async_client):
        """Returns no_candidates when all memories are short or already summarized."""
        ac, mock_mem = async_client
        mock_mem.search = AsyncMock(return_value=[
            {"content": "short", "metadata": {}, "id": "working_1"}
        ])

        response = await ac.post("/api/memory/auto-summarize-one")
        assert response.status_code == 200
        data = response.json()
        assert data["summarized"] is False
        assert data["reason"] == "no_candidates"

    @pytest.mark.asyncio
    async def test_auto_summarize_skips_already_summarized(self, async_client):
        """Skips memories that already have summarized_at metadata."""
        ac, mock_mem = async_client
        mock_mem.search = AsyncMock(return_value=[
            {
                "content": "x" * 500,  # Long enough
                "metadata": {"summarized_at": "2026-01-01T00:00:00"},
                "id": "working_already"
            }
        ])

        response = await ac.post("/api/memory/auto-summarize-one")
        data = response.json()
        assert data["summarized"] is False
        assert data["reason"] == "no_candidates"

    @pytest.mark.asyncio
    async def test_auto_summarize_returns_503_when_memory_not_ready(self, async_client):
        """Returns 503 when memory system not initialized."""
        from roampal.server import main
        ac, _ = async_client

        original = main._memory
        main._memory = None
        try:
            response = await ac.post("/api/memory/auto-summarize-one")
            assert response.status_code == 503
        finally:
            main._memory = original

    @pytest.mark.asyncio
    async def test_auto_summarize_success(self, async_client):
        """Successfully summarizes a long memory."""
        ac, mock_mem = async_client

        # Mock search to return a long memory
        mock_mem.search = AsyncMock(return_value=[
            {
                "content": "x" * 500,
                "metadata": {},
                "id": "working_long1",
                "doc_id": "working_long1"
            }
        ])

        # Mock the collection adapter
        adapter = MagicMock()
        adapter.get_fragment = MagicMock(return_value={
            "content": "x" * 500,
            "metadata": {"text": "x" * 500}
        })
        adapter.upsert_vectors = AsyncMock()
        mock_mem.collections = {"working": adapter}

        # Mock summarize_only to return a short summary
        with patch("roampal.sidecar_service.summarize_only", return_value="Short summary of the exchange"):
            response = await ac.post("/api/memory/auto-summarize-one")

        data = response.json()
        assert data["summarized"] is True
        assert data["doc_id"] == "working_long1"
        assert data["old_len"] == 500
        assert data["new_len"] > 0

    @pytest.mark.asyncio
    async def test_auto_summarize_backend_failed_returns_content(self, async_client):
        """When sidecar fails, returns content for plugin Zen fallback."""
        ac, mock_mem = async_client

        long_content = "y" * 600
        mock_mem.search = AsyncMock(return_value=[
            {
                "content": long_content,
                "metadata": {},
                "id": "working_fail1",
                "doc_id": "working_fail1"
            }
        ])

        with patch("roampal.sidecar_service.summarize_only", return_value=None):
            response = await ac.post("/api/memory/auto-summarize-one")

        data = response.json()
        assert data["summarized"] is False
        assert data["reason"] == "backend_failed"
        assert data["doc_id"] == "working_fail1"
        assert data["content"] == long_content
        assert data["old_len"] == 600

    @pytest.mark.asyncio
    async def test_auto_summarize_truncates_long_summary(self, async_client):
        """Summaries over 400 chars are truncated to prevent re-summarization loops."""
        ac, mock_mem = async_client

        mock_mem.search = AsyncMock(return_value=[
            {
                "content": "z" * 500,
                "metadata": {},
                "id": "working_trunc1",
                "doc_id": "working_trunc1"
            }
        ])

        adapter = MagicMock()
        adapter.get_fragment = MagicMock(return_value={
            "content": "z" * 500,
            "metadata": {"text": "z" * 500}
        })
        adapter.upsert_vectors = AsyncMock()
        mock_mem.collections = {"working": adapter}

        # Return a summary that's too long (> 400 chars)
        long_summary = "a" * 450
        with patch("roampal.sidecar_service.summarize_only", return_value=long_summary):
            response = await ac.post("/api/memory/auto-summarize-one")

        data = response.json()
        assert data["summarized"] is True
        # new_len should be <= 400 (truncated to 380 + "... [truncated]")
        assert data["new_len"] <= 400


# ============================================================================
# Test 5: ROAMPAL_SUMMARIZE_MODEL Override (Change 10)
# ============================================================================

class TestSummarizeModelOverride:
    """Test ROAMPAL_SUMMARIZE_MODEL env var in sidecar_service."""

    def test_summarize_model_env_var_read(self):
        """ROAMPAL_SUMMARIZE_MODEL env var is read at import time."""
        from roampal.sidecar_service import SUMMARIZE_MODEL
        # Just verify the constant exists (it reads from env)
        assert isinstance(SUMMARIZE_MODEL, str)

    def test_get_backend_info_reports_summarize_model(self):
        """get_backend_info() reports SUMMARIZE_MODEL when set."""
        with patch.dict(os.environ, {
            "ROAMPAL_SUMMARIZE_MODEL": "claude-sonnet-4-5-20250929",
            "ANTHROPIC_API_KEY": "sk-ant-test123"
        }):
            # Need to reload to pick up env var
            import importlib
            import roampal.sidecar_service as svc
            original = svc.SUMMARIZE_MODEL
            svc.SUMMARIZE_MODEL = "claude-sonnet-4-5-20250929"
            try:
                result = svc.get_backend_info()
                assert "claude-sonnet-4-5-20250929" in result
                assert "ROAMPAL_SUMMARIZE_MODEL" in result
            finally:
                svc.SUMMARIZE_MODEL = original

    def test_summarize_only_uses_model_override(self):
        """summarize_only() calls _call_anthropic_model when SUMMARIZE_MODEL is set."""
        import roampal.sidecar_service as svc
        original = svc.SUMMARIZE_MODEL
        svc.SUMMARIZE_MODEL = "claude-haiku-3-5-20241022"

        try:
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}), \
                 patch.object(svc, "_call_anthropic_model", return_value={"summary": "test summary"}) as mock_call:
                result = svc.summarize_only("Some long content that needs summarizing")
                assert result == "test summary"
                mock_call.assert_called_once()
                # Verify correct model was passed
                assert mock_call.call_args[0][1] == "claude-haiku-3-5-20241022"
        finally:
            svc.SUMMARIZE_MODEL = original

    def test_summarize_only_falls_back_on_model_failure(self):
        """Falls back to sidecar chain when SUMMARIZE_MODEL fails."""
        import roampal.sidecar_service as svc
        original = svc.SUMMARIZE_MODEL
        svc.SUMMARIZE_MODEL = "bad-model"

        try:
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}), \
                 patch.object(svc, "_call_anthropic_model", return_value=None), \
                 patch.object(svc, "_call_llm", return_value={"summary": "fallback summary"}) as mock_llm:
                result = svc.summarize_only("Some long content")
                assert result == "fallback summary"
                mock_llm.assert_called_once()
        finally:
            svc.SUMMARIZE_MODEL = original

    def test_summarize_only_skips_model_without_api_key(self):
        """Skips SUMMARIZE_MODEL when ANTHROPIC_API_KEY not set."""
        import roampal.sidecar_service as svc
        original = svc.SUMMARIZE_MODEL
        svc.SUMMARIZE_MODEL = "claude-sonnet-4-5-20250929"

        try:
            with patch.dict(os.environ, {}, clear=False), \
                 patch.object(svc, "_call_anthropic_model") as mock_anthropic, \
                 patch.object(svc, "_call_llm", return_value={"summary": "chain summary"}) as mock_llm:
                # Remove ANTHROPIC_API_KEY if present
                env = os.environ.copy()
                env.pop("ANTHROPIC_API_KEY", None)
                with patch.dict(os.environ, env, clear=True):
                    result = svc.summarize_only("Some content")
                    mock_anthropic.assert_not_called()
                    assert result == "chain summary"
        finally:
            svc.SUMMARIZE_MODEL = original


# ============================================================================
# Test 6: CLI context --recent-exchanges (Change 11)
# ============================================================================

class TestContextRecentExchanges:
    """Test roampal context --recent-exchanges CLI output."""

    def test_recent_exchanges_formats_output(self):
        """Verify output format matches expected RECENT EXCHANGES block."""
        from unittest.mock import patch
        import io
        from contextlib import redirect_stdout

        # Mock httpx.post to return exchange summaries
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"content": "User asked about auth tokens. Fixed TTL clamping.", "metadata": {"recency": "2min ago"}},
                {"content": "Discussed sidecar scoring. Decided on Haiku.", "metadata": {"recency": "8min ago"}},
                {"content": "Dropped dual storage. Just store summary.", "metadata": {"recency": "15min ago"}},
                {"content": "Verified token savings: 78% reduction.", "metadata": {"recency": "22min ago"}},
            ]
        }

        with patch("httpx.post", return_value=mock_response):
            f = io.StringIO()
            with redirect_stdout(f):
                from roampal.cli import cmd_context
                args = MagicMock()
                args.recent_exchanges = True
                args.port = None
                args.dev = False
                try:
                    cmd_context(args)
                except SystemExit:
                    pass

            output = f.getvalue()
            # Should contain the header
            assert "RECENT EXCHANGES" in output
            # Should contain exchange content
            assert "auth tokens" in output or "token savings" in output


# ============================================================================
# Test 7: History-to-Patterns Gate with Lifetime Successes (Change 3)
# ============================================================================

class TestPatternsGateLifetimeSuccesses:
    """Test that the patterns gate counts lifetime successes (no reset)."""

    @pytest.fixture
    def promotion_service(self):
        from roampal.backend.modules.memory.promotion_service import PromotionService
        from roampal.backend.modules.memory.config import MemoryConfig

        working = MagicMock()
        history = MagicMock()
        patterns = MagicMock()
        patterns.upsert_vectors = AsyncMock()

        collections = {"working": working, "history": history, "patterns": patterns}
        embed_fn = AsyncMock(return_value=[0.1, 0.2, 0.3])

        config = MemoryConfig()
        svc = PromotionService(
            collections=collections,
            embed_fn=embed_fn,
            config=config
        )
        return svc, history, patterns, config

    @pytest.mark.asyncio
    async def test_history_to_patterns_requires_5_lifetime_successes(self, promotion_service):
        """Memory with 5 lifetime successes (3 from working + 2 from history) qualifies."""
        svc, history, patterns, config = promotion_service

        metadata = {
            "text": "proven memory content",
            "score": 0.95,
            "uses": 8,
            "success_count": 5.0,  # 3 from working + 2 from history (no reset!)
            "timestamp": datetime.now().isoformat()
        }
        doc = {"content": "proven memory content", "metadata": metadata}
        history.get_fragment = MagicMock(return_value=doc)
        history.delete_vectors = MagicMock()

        result = await svc.handle_promotion(
            doc_id="history_proven1",
            collection="history",
            score=0.95,
            uses=8,
            metadata=metadata.copy()
        )

        # Should have been promoted (patterns.upsert_vectors called)
        assert patterns.upsert_vectors.called

    @pytest.mark.asyncio
    async def test_history_to_patterns_blocked_with_4_successes(self, promotion_service):
        """Memory with only 4 lifetime successes does NOT qualify."""
        svc, history, patterns, config = promotion_service

        metadata = {
            "text": "almost proven content",
            "score": 0.95,
            "uses": 7,
            "success_count": 4.0,  # Not enough
            "timestamp": datetime.now().isoformat()
        }
        doc = {"content": "almost proven content", "metadata": metadata}
        history.get_fragment = MagicMock(return_value=doc)

        result = await svc.handle_promotion(
            doc_id="history_almost1",
            collection="history",
            score=0.95,
            uses=7,
            metadata=metadata.copy()
        )

        # Should NOT have been promoted
        assert not patterns.upsert_vectors.called
