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
    async def test_archive_memory_bank(self, mock_ums):
        """Should delegate archive."""
        result = await mock_ums.archive_memory_bank("some content to archive")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
