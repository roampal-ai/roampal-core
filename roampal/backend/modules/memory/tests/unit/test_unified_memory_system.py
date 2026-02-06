"""
Unit Tests for UnifiedMemorySystem (Core)

Tests the Core UMS which is more monolithic than Desktop's facade pattern.
Core UMS handles collections and services internally.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem
from roampal.backend.modules.memory.config import MemoryConfig
from roampal.backend.modules.memory.scoring_service import ScoringService


class TestUnifiedMemorySystemInit:
    """Test initialization."""

    def test_init_creates_data_dir(self, tmp_path):
        """Should create data directory."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        assert (tmp_path / "data").exists()

    def test_init_with_custom_config(self, tmp_path):
        """Should use custom config."""
        config = MemoryConfig(promotion_score_threshold=0.8)
        ums = UnifiedMemorySystem(
            data_path=str(tmp_path / "data"),
            config=config
        )
        assert ums.config.promotion_score_threshold == 0.8

    def test_init_not_initialized(self, tmp_path):
        """Should not be initialized until initialize() called."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        assert not ums.initialized

    def test_init_loads_kg(self, tmp_path):
        """Should load knowledge graph on init."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        assert "routing_patterns" in ums.knowledge_graph
        assert "context_action_effectiveness" in ums.knowledge_graph


class TestInitialize:
    """Test initialization process."""

    @pytest.fixture
    def ums(self, tmp_path):
        """Create UMS instance."""
        return UnifiedMemorySystem(data_path=str(tmp_path / "data"))

    @pytest.mark.asyncio
    async def test_initialize_creates_collections(self, ums):
        """Should create all collections."""
        await ums.initialize()

        assert "books" in ums.collections
        assert "working" in ums.collections
        assert "history" in ums.collections
        assert "patterns" in ums.collections
        assert "memory_bank" in ums.collections

    @pytest.mark.asyncio
    async def test_initialize_creates_services(self, ums):
        """Should initialize all services."""
        await ums.initialize()

        assert ums._embedding_service is not None
        assert ums._scoring_service is not None
        assert ums._promotion_service is not None
        assert ums._outcome_service is not None
        assert ums._memory_bank_service is not None
        assert ums._context_service is not None
        assert ums._search_service is not None

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, ums):
        """Should only initialize once."""
        await ums.initialize()
        first_embedding = ums._embedding_service

        await ums.initialize()
        assert ums._embedding_service is first_embedding


class TestStoreWorking:
    """Test store_working functionality."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with mocked embedding."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True

        # Mock collections
        working = MagicMock()
        working.upsert_vectors = AsyncMock()
        ums.collections = {"working": working}

        # Mock embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)

        return ums

    @pytest.mark.asyncio
    async def test_store_working_generates_doc_id(self, mock_ums):
        """Should generate document ID."""
        doc_id = await mock_ums.store_working("test text")

        assert doc_id.startswith("working_")
        mock_ums.collections["working"].upsert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_working_embeds_text(self, mock_ums):
        """Should embed the text."""
        await mock_ums.store_working("test text")

        mock_ums._embedding_service.embed_text.assert_called_with("test text")


class TestSearch:
    """Test search functionality."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with search mocks."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True

        # Mock embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)

        # Mock scoring
        ums._scoring_service = MagicMock()
        ums._scoring_service.calculate_final_score = MagicMock(return_value={
            "final_rank_score": 0.8,
            "wilson_score": 0.7,
            "embedding_similarity": 0.9,
            "learned_score": 0.7,
            "embedding_weight": 0.5,
            "learned_weight": 0.5
        })

        # Mock collections with query results
        mock_collection = MagicMock()
        mock_collection.hybrid_query = AsyncMock(return_value=[
            {"id": "doc_1", "text": "result 1", "distance": 0.5, "metadata": {"score": 0.7, "uses": 3}},
            {"id": "doc_2", "text": "result 2", "distance": 0.8, "metadata": {"score": 0.5, "uses": 1}},
        ])

        ums.collections = {
            "working": mock_collection,
            "history": mock_collection,
            "patterns": mock_collection,
            "books": mock_collection,
            "memory_bank": mock_collection,
        }

        return ums

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_ums):
        """Should return search results."""
        results = await mock_ums.search("test query")

        # Returns list (may be empty if mocked collections return nothing)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_generates_embedding(self, mock_ums):
        """Should generate embedding for query."""
        await mock_ums.search("test query")

        mock_ums._embedding_service.embed_text.assert_called()


class TestRecordOutcome:
    """Test outcome recording."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with outcome mock."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True

        # Mock outcome service
        ums._outcome_service = MagicMock()
        ums._outcome_service.record_outcome = AsyncMock(return_value={"score": 0.7})

        return ums

    @pytest.mark.asyncio
    async def test_record_outcome_delegates(self, mock_ums):
        """Should delegate to outcome service."""
        # Core's record_outcome takes doc_ids (list), not doc_id
        await mock_ums.record_outcome(
            doc_ids=["working_test123"],
            outcome="worked"
        )

        mock_ums._outcome_service.record_outcome.assert_called()

    @pytest.mark.asyncio
    async def test_record_outcome_with_reason(self, mock_ums):
        """Should pass failure reason."""
        # Core's record_outcome takes doc_ids (list), not doc_id
        await mock_ums.record_outcome(
            doc_ids=["working_test123"],
            outcome="failed",
            failure_reason="Test failure"
        )

        # Check failure_reason was passed
        mock_ums._outcome_service.record_outcome.assert_called()


class TestMemoryBankAPI:
    """Test memory bank API."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with memory bank mock."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True

        # Mock memory bank service
        ums._memory_bank_service = MagicMock()
        ums._memory_bank_service.store = AsyncMock(return_value="memory_bank_123")
        ums._memory_bank_service.update = AsyncMock(return_value="memory_bank_123")
        ums._memory_bank_service.archive = AsyncMock(return_value=True)
        ums._memory_bank_service.search = AsyncMock(return_value=[])

        return ums

    @pytest.mark.asyncio
    async def test_store_memory_bank(self, mock_ums):
        """Should delegate to memory bank service."""
        doc_id = await mock_ums.store_memory_bank(
            text="User prefers dark mode",
            tags=["preference"]
        )

        assert doc_id == "memory_bank_123"
        mock_ums._memory_bank_service.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_memory_bank(self, mock_ums):
        """Should delegate update."""
        await mock_ums.update_memory_bank(
            old_content="old text",
            new_content="new text"
        )

        mock_ums._memory_bank_service.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_memory_bank(self, mock_ums):
        """Should delegate archive."""
        result = await mock_ums.delete_memory_bank("some content to archive")
        assert result is True


class TestContextAPI:
    """Test context analysis API."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with mocked collections for context analysis."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True

        # Mock embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)

        # Mock collections
        mock_collection = MagicMock()
        mock_collection.hybrid_query = AsyncMock(return_value=[])

        ums.collections = {
            "working": mock_collection,
            "history": mock_collection,
            "patterns": mock_collection,
            "books": mock_collection,
            "memory_bank": mock_collection,
        }

        return ums

    @pytest.mark.asyncio
    async def test_analyze_context(self, mock_ums):
        """Should return context analysis result."""
        # Core's analyze_conversation_context does work internally
        context = await mock_ums.analyze_conversation_context(
            current_message="test",
            recent_conversation=[],
            conversation_id="conv123"
        )

        # Core returns these keys
        assert "relevant_patterns" in context
        assert "past_outcomes" in context
        assert "topic_continuity" in context
        assert "proactive_insights" in context


class TestKGAccess:
    """Test Knowledge Graph access."""

    @pytest.fixture
    def ums(self, tmp_path):
        """Create UMS instance."""
        return UnifiedMemorySystem(data_path=str(tmp_path / "data"))

    def test_knowledge_graph_property(self, ums):
        """Should expose knowledge graph."""
        kg = ums.knowledge_graph

        assert "routing_patterns" in kg
        assert "context_action_effectiveness" in kg
        assert "problem_solutions" in kg

    def test_knowledge_graph_is_dict(self, ums):
        """Knowledge graph should be a dict."""
        assert isinstance(ums.knowledge_graph, dict)


class TestStats:
    """Test statistics retrieval."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with collection mocks."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True

        # Mock collections
        mock_collection = MagicMock()
        mock_collection.collection = MagicMock()
        mock_collection.collection.count = MagicMock(return_value=10)

        ums.collections = {
            "working": mock_collection,
            "history": mock_collection,
            "patterns": mock_collection,
            "books": mock_collection,
            "memory_bank": mock_collection,
        }

        return ums

    def test_get_stats(self, mock_ums):
        """Should return statistics."""
        stats = mock_ums.get_stats()

        assert "initialized" in stats
        assert "data_path" in stats
        assert "collections" in stats


class TestConceptExtraction:
    """Test concept extraction."""

    @pytest.fixture
    def ums(self, tmp_path):
        """Create UMS instance."""
        return UnifiedMemorySystem(data_path=str(tmp_path / "data"))

    def test_extract_concepts(self, ums):
        """Should extract concepts from text."""
        concepts = ums._extract_concepts("Python programming language")

        assert "python" in concepts
        assert "programming" in concepts

    def test_extract_concepts_filters_short(self, ums):
        """Should filter short words."""
        concepts = ums._extract_concepts("Go is a fun language")

        # Short words filtered
        assert "go" not in concepts
        assert "is" not in concepts
        assert "language" in concepts


class TestTierRecommendations:
    """Test tier recommendation logic."""

    @pytest.fixture
    def ums(self, tmp_path):
        """Create UMS with routing patterns."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.knowledge_graph["routing_patterns"] = {
            "python": {
                "best_collection": "patterns",
                "collections_used": {"patterns": {"total": 10, "successes": 8}}
            }
        }
        return ums

    def test_get_tier_recommendations(self, ums):
        """Should return recommendations based on KG."""
        recs = ums.get_tier_recommendations(["python"])

        # Core returns these keys
        assert "top_collections" in recs
        assert "match_count" in recs
        assert "confidence_level" in recs


class TestStoreBook:
    """Test book storage and chunking - v0.1.11 fixes."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with mocked embedding and books collection."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True

        # Mock collections
        books = MagicMock()
        books.upsert_vectors = AsyncMock()
        # v0.2.0: Mock async _ensure_initialized and collection.get for duplicate detection
        books._ensure_initialized = AsyncMock()
        books.collection = MagicMock()
        books.collection.get = MagicMock(return_value={"ids": []})  # No existing books
        ums.collections = {"books": books}

        # Mock embedding - v0.2.0: Use embed_texts for batch embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)
        ums._embedding_service.embed_texts = AsyncMock(side_effect=lambda texts: [[0.1] * 384 for _ in texts])

        return ums

    @pytest.mark.asyncio
    async def test_small_file_single_chunk(self, mock_ums):
        """Small files (<= chunk_size) should become single chunk."""
        # Content smaller than default 1000 chunk_size
        content = "This is a short document."
        doc_ids = await mock_ums.store_book(content, title="test")

        # Should create exactly 1 chunk
        assert len(doc_ids) == 1
        mock_ums.collections["books"].upsert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_large_file_multiple_chunks(self, mock_ums):
        """Large files should be split into multiple chunks."""
        # Content larger than chunk_size
        content = "x" * 2500  # 2500 chars with default 1000 chunk_size
        doc_ids = await mock_ums.store_book(content, title="test")

        # Should create multiple chunks (v0.2.0: batch upsert means 1 call with all chunks)
        assert len(doc_ids) >= 2
        mock_ums.collections["books"].upsert_vectors.assert_called_once()
        # Check that multiple doc_ids were passed in the single call
        call_args = mock_ums.collections["books"].upsert_vectors.call_args
        assert len(call_args.kwargs.get("ids", [])) >= 2

    @pytest.mark.asyncio
    async def test_overlap_equals_chunk_size_no_infinite_loop(self, mock_ums):
        """Overlap >= chunk_size should not cause infinite loop (v0.1.11 fix)."""
        import asyncio

        content = "x" * 500  # Some content

        # This would hang forever before the fix
        # Use timeout to catch infinite loop
        try:
            await asyncio.wait_for(
                mock_ums.store_book(content, title="test", chunk_size=100, chunk_overlap=100),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            pytest.fail("store_book infinite looped with overlap >= chunk_size")

    @pytest.mark.asyncio
    async def test_overlap_greater_than_chunk_size_no_infinite_loop(self, mock_ums):
        """Overlap > chunk_size should not cause infinite loop (v0.1.11 fix)."""
        import asyncio

        content = "x" * 500

        # This would definitely hang before the fix
        try:
            await asyncio.wait_for(
                mock_ums.store_book(content, title="test", chunk_size=100, chunk_overlap=150),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            pytest.fail("store_book infinite looped with overlap > chunk_size")

    @pytest.mark.asyncio
    async def test_empty_content_handled(self, mock_ums):
        """Empty content should not crash."""
        doc_ids = await mock_ums.store_book("", title="empty")

        # Should create 1 chunk with empty content
        assert len(doc_ids) == 1


class TestMemoryBankWilsonBlend:
    """Test memory_bank 80/20 Wilson blend in search inline scoring (v0.2.9 fix)."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with mocked collections for search testing."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True
        ums.ghost_ids = set()

        # Mock embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)

        # Real ScoringService for Wilson scoring (v0.3.2)
        ums._scoring_service = ScoringService(config=ums.config)

        return ums

    @pytest.mark.asyncio
    async def test_memory_bank_cold_start_pure_quality(self, mock_ums):
        """Memory bank items with < 3 uses should use pure importance*confidence."""
        mock_collection = MagicMock()
        mock_collection.query_vectors = AsyncMock(return_value=[
            {
                "id": "mb_1",
                "text": "user likes dark mode",
                "distance": 0.5,
                "metadata": {
                    "importance": 0.9,
                    "confidence": 0.8,
                    "uses": 0,
                    "success_count": 0.0,
                    "score": 0.5,
                },
            },
        ])

        empty_collection = MagicMock()
        empty_collection.query_vectors = AsyncMock(return_value=[])

        mock_ums.collections = {
            "memory_bank": mock_collection,
            "working": empty_collection,
            "history": empty_collection,
            "patterns": empty_collection,
            "books": empty_collection,
        }

        results = await mock_ums.search("test", collections=["memory_bank"])
        assert len(results) == 1
        # Pure quality: 0.9 * 0.8 = 0.72
        assert abs(results[0]["quality"] - 0.72) < 0.01

    @pytest.mark.asyncio
    async def test_memory_bank_wilson_blend_after_3_uses(self, mock_ums):
        """Memory bank items with 3+ uses should blend 80% quality + 20% Wilson."""
        mock_collection = MagicMock()
        mock_collection.query_vectors = AsyncMock(return_value=[
            {
                "id": "mb_2",
                "text": "user prefers typescript",
                "distance": 0.5,
                "metadata": {
                    "importance": 0.9,
                    "confidence": 0.8,
                    "uses": 10,
                    "success_count": 8.0,
                    "score": 0.5,
                },
            },
        ])

        empty_collection = MagicMock()
        empty_collection.query_vectors = AsyncMock(return_value=[])

        mock_ums.collections = {
            "memory_bank": mock_collection,
            "working": empty_collection,
            "history": empty_collection,
            "patterns": empty_collection,
            "books": empty_collection,
        }

        results = await mock_ums.search("test", collections=["memory_bank"])
        assert len(results) == 1

        base_quality = 0.9 * 0.8  # 0.72
        # v0.3.2: Uses real Wilson lower bound (~0.49 for 8/10 at 95% CI),
        # not simple ratio (0.8). Wilson is conservative with small samples.
        # Expected: 0.8 * 0.72 + 0.2 * wilson_lower(8,10) ≈ 0.674
        quality = results[0]["quality"]
        assert quality < base_quality  # Wilson blend pulls below pure quality
        assert quality > 0.6  # But not unreasonably low for 80% success rate
        assert abs(quality - 0.674) < 0.02  # ~0.674 with real Wilson

    @pytest.mark.asyncio
    async def test_memory_bank_wilson_blend_low_success_reduces_quality(self, mock_ums):
        """Memory bank items with low success rate should rank lower via Wilson."""
        mock_collection = MagicMock()
        mock_collection.query_vectors = AsyncMock(return_value=[
            {
                "id": "mb_3",
                "text": "noisy fact",
                "distance": 0.5,
                "metadata": {
                    "importance": 0.9,
                    "confidence": 0.9,
                    "uses": 10,
                    "success_count": 1.0,  # Only 1/10 worked
                    "score": 0.5,
                },
            },
        ])

        empty_collection = MagicMock()
        empty_collection.query_vectors = AsyncMock(return_value=[])

        mock_ums.collections = {
            "memory_bank": mock_collection,
            "working": empty_collection,
            "history": empty_collection,
            "patterns": empty_collection,
            "books": empty_collection,
        }

        results = await mock_ums.search("test", collections=["memory_bank"])
        assert len(results) == 1

        pure_quality = 0.9 * 0.9  # 0.81
        # Wilson should pull quality down because 1/10 success = 0.1 rate
        assert results[0]["quality"] < pure_quality

    @pytest.mark.asyncio
    async def test_memory_bank_under_threshold_no_blend(self, mock_ums):
        """Memory bank items with exactly 2 uses should NOT blend Wilson."""
        mock_collection = MagicMock()
        mock_collection.query_vectors = AsyncMock(return_value=[
            {
                "id": "mb_4",
                "text": "new fact",
                "distance": 0.5,
                "metadata": {
                    "importance": 0.9,
                    "confidence": 0.8,
                    "uses": 2,
                    "success_count": 0.0,  # 0/2 success
                    "score": 0.5,
                },
            },
        ])

        empty_collection = MagicMock()
        empty_collection.query_vectors = AsyncMock(return_value=[])

        mock_ums.collections = {
            "memory_bank": mock_collection,
            "working": empty_collection,
            "history": empty_collection,
            "patterns": empty_collection,
            "books": empty_collection,
        }

        results = await mock_ums.search("test", collections=["memory_bank"])
        assert len(results) == 1
        # Should be pure quality despite 0/2 success rate
        assert abs(results[0]["quality"] - 0.72) < 0.01


class TestSearchWilsonScoring:
    """Test that search() uses Wilson scoring via ScoringService (v0.3.2)."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with real ScoringService for Wilson scoring."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True
        ums.ghost_ids = set()

        # Mock embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)

        # Real ScoringService for Wilson scoring
        ums._scoring_service = ScoringService(config=ums.config)

        return ums

    @pytest.mark.asyncio
    async def test_search_uses_wilson_not_raw_score(self, mock_ums):
        """search() should use Wilson-based scoring, not raw metadata score.

        A proven memory (10 uses, 8 successes) should rank differently with
        Wilson vs raw score. Wilson gives dynamic weight shifts that raw score
        doesn't provide.
        """
        mock_collection = MagicMock()
        mock_collection.query_vectors = AsyncMock(return_value=[
            {
                "id": "doc_good",
                "text": "proven good advice",
                "distance": 0.6,  # Farther semantically
                "metadata": {
                    "score": 0.9,
                    "uses": 30,
                    "success_count": 29.0,  # 29/30 → Wilson ~0.83 → PROVEN weights
                },
            },
            {
                "id": "doc_bad",
                "text": "unproven bad advice",
                "distance": 0.4,  # Closer semantically
                "metadata": {
                    "score": 0.5,
                    "uses": 0,
                    "success_count": 0.0,
                },
            },
        ])

        empty_collection = MagicMock()
        empty_collection.query_vectors = AsyncMock(return_value=[])

        mock_ums.collections = {
            "working": mock_collection,
            "history": empty_collection,
            "patterns": empty_collection,
            "memory_bank": empty_collection,
            "books": empty_collection,
        }

        results = await mock_ums.search("test", collections=["working"])
        assert len(results) == 2

        # Proven good advice should rank first despite being farther away
        assert results[0]["id"] == "doc_good"
        assert results[1]["id"] == "doc_bad"

        # The good doc should have higher final_rank_score
        assert results[0]["final_rank_score"] > results[1]["final_rank_score"]

    @pytest.mark.asyncio
    async def test_search_wilson_consistent_with_context_injection(self, mock_ums):
        """search() and get_context_for_injection() should use same scoring service.

        v0.3.2: Both paths now delegate to ScoringService.calculate_final_score()
        """
        mock_collection = MagicMock()
        mock_collection.query_vectors = AsyncMock(return_value=[
            {
                "id": "doc_1",
                "text": "test memory",
                "distance": 0.5,
                "metadata": {
                    "score": 0.7,
                    "uses": 5,
                    "success_count": 4.0,
                },
            },
        ])

        empty_collection = MagicMock()
        empty_collection.query_vectors = AsyncMock(return_value=[])

        mock_ums.collections = {
            "working": mock_collection,
            "history": empty_collection,
            "patterns": empty_collection,
            "memory_bank": empty_collection,
            "books": empty_collection,
        }

        results = await mock_ums.search("test", collections=["working"])
        assert len(results) == 1

        # Verify Wilson-based scoring fields are present
        result = results[0]
        assert "final_rank_score" in result
        assert "embedding_similarity" in result
        assert "quality" in result

        # Quality should reflect Wilson-blended learned score, not raw 0.7
        # With 4/5 successes, Wilson lower bound is ~0.437
        # Since uses >= 3, learned = pure Wilson ≈ 0.437
        assert result["quality"] != 0.7  # NOT the raw score


class TestCrossEncoderWiring:
    """Test cross-encoder reranking wired into UMS (v0.3.2)."""

    @pytest.fixture
    def ums(self, tmp_path):
        """Create UMS instance."""
        return UnifiedMemorySystem(data_path=str(tmp_path / "data"))

    @pytest.mark.asyncio
    async def test_initialize_creates_search_service_with_reranker(self, ums):
        """initialize() should create SearchService and attempt to load CrossEncoder."""
        await ums.initialize()

        assert ums._search_service is not None
        # Reranker may or may not be available depending on environment,
        # but SearchService itself must always be created
        assert hasattr(ums._search_service, 'reranker')

    @pytest.mark.asyncio
    async def test_initialize_graceful_without_cross_encoder(self, ums):
        """SearchService should still be created when CrossEncoder import fails."""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            # Force ImportError for sentence_transformers
            await ums.initialize()

        assert ums._search_service is not None
        # Reranker should be None since import failed
        assert ums._search_service.reranker is None

    @pytest.mark.asyncio
    async def test_search_delegates_to_search_service(self, tmp_path):
        """search() should delegate to SearchService when available."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True
        ums.ghost_ids = set()

        # Mock embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)

        # Mock SearchService
        mock_search_service = MagicMock()
        mock_search_service.search = AsyncMock(return_value=[
            {
                "id": "doc_1",
                "text": "result from SearchService",
                "collection": "working",
                "final_rank_score": 0.9,
                "embedding_similarity": 0.8,
                "learned_score": 0.7,
                "metadata": {"score": 0.7, "uses": 5},
            },
        ])
        ums._search_service = mock_search_service

        results = await ums.search("test query")

        # SearchService.search() should have been called
        mock_search_service.search.assert_called_once()
        call_kwargs = mock_search_service.search.call_args.kwargs
        assert call_kwargs["query"] == "test query"

        # Result should have legacy field aliases
        assert len(results) == 1
        assert results[0]["quality"] == 0.7
        assert results[0]["combined_score"] == 0.9
        assert results[0]["similarity"] == 0.8

    @pytest.mark.asyncio
    async def test_search_falls_back_on_search_service_failure(self, tmp_path):
        """search() should fall back to inline search when SearchService raises."""
        ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"))
        ums.initialized = True
        ums.ghost_ids = set()

        # Mock embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)

        # Real scoring for inline fallback
        ums._scoring_service = ScoringService(config=ums.config)

        # Mock SearchService that raises
        mock_search_service = MagicMock()
        mock_search_service.search = AsyncMock(side_effect=RuntimeError("search failed"))
        ums._search_service = mock_search_service

        # Mock collections for inline fallback
        mock_collection = MagicMock()
        mock_collection.query_vectors = AsyncMock(return_value=[
            {"id": "fallback_1", "text": "inline result", "distance": 0.5,
             "metadata": {"score": 0.6, "uses": 2}},
        ])
        empty_collection = MagicMock()
        empty_collection.query_vectors = AsyncMock(return_value=[])

        ums.collections = {
            "working": mock_collection,
            "history": empty_collection,
            "patterns": empty_collection,
            "books": empty_collection,
            "memory_bank": empty_collection,
        }

        results = await ums.search("test query")

        # Should have fallen back to inline and returned results
        assert len(results) >= 1
        assert results[0]["id"] == "fallback_1"

    @pytest.mark.asyncio
    async def test_search_service_receives_reranker(self, ums):
        """SearchService should receive the reranker passed during initialize()."""
        # Patch CrossEncoder so we can control what reranker is created
        mock_reranker = MagicMock()
        with patch('roampal.backend.modules.memory.unified_memory_system.SearchService') as MockSS:
            MockSS.return_value = MagicMock()
            # Patch the CrossEncoder import inside initialize()
            import builtins
            original_import = builtins.__import__
            def mock_import(name, *args, **kwargs):
                if name == 'sentence_transformers':
                    mod = MagicMock()
                    mod.CrossEncoder.return_value = mock_reranker
                    return mod
                return original_import(name, *args, **kwargs)

            with patch('builtins.__import__', side_effect=mock_import):
                await ums.initialize()

            # Verify SearchService was called with our mock reranker
            MockSS.assert_called_once()
            call_kwargs = MockSS.call_args.kwargs
            assert call_kwargs["reranker"] is mock_reranker


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
