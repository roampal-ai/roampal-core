"""
Unit Tests for SearchService

Tests the extracted search logic.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from roampal.backend.modules.memory.search_service import SearchService
from roampal.backend.modules.memory.scoring_service import ScoringService
from roampal.backend.modules.memory.routing_service import RoutingService
from roampal.backend.modules.memory.knowledge_graph_service import KnowledgeGraphService
from roampal.backend.modules.memory.config import MemoryConfig


class TestSearchServiceInit:
    """Test SearchService initialization."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        collections = {
            "working": MagicMock(),
            "history": MagicMock(),
            "patterns": MagicMock(),
            "books": MagicMock(),
            "memory_bank": MagicMock(),
        }
        scoring = MagicMock(spec=ScoringService)
        routing = MagicMock(spec=RoutingService)
        kg = MagicMock(spec=KnowledgeGraphService)
        kg.knowledge_graph = {"routing_patterns": {}, "context_action_effectiveness": {}}
        embed_fn = AsyncMock(return_value=[0.1] * 384)

        return {
            "collections": collections,
            "scoring_service": scoring,
            "routing_service": routing,
            "kg_service": kg,
            "embed_fn": embed_fn,
        }

    def test_init_with_all_dependencies(self, mock_dependencies):
        """Should initialize with all dependencies."""
        service = SearchService(**mock_dependencies)
        assert service.collections == mock_dependencies["collections"]
        assert service.scoring_service == mock_dependencies["scoring_service"]
        assert service.routing_service == mock_dependencies["routing_service"]
        assert service.kg_service == mock_dependencies["kg_service"]

    def test_init_with_optional_reranker(self, mock_dependencies):
        """Should accept optional reranker."""
        mock_reranker = MagicMock()
        service = SearchService(**mock_dependencies, reranker=mock_reranker)
        assert service.reranker == mock_reranker


class TestMainSearch:
    """Test main search functionality."""

    @pytest.fixture
    def mock_service(self):
        """Create SearchService with mocks."""
        collections = {
            "working": MagicMock(),
            "history": MagicMock(),
        }

        # Mock hybrid_query to return sample results
        async def mock_hybrid_query(**kwargs):
            return [
                {"id": "doc_1", "text": "test result 1", "distance": 0.5, "metadata": {"score": 0.7, "uses": 3}},
                {"id": "doc_2", "text": "test result 2", "distance": 0.8, "metadata": {"score": 0.5, "uses": 1}},
            ]

        for coll in collections.values():
            coll.hybrid_query = AsyncMock(side_effect=mock_hybrid_query)

        scoring = MagicMock(spec=ScoringService)
        scoring.apply_scoring_to_results = MagicMock(side_effect=lambda x, **kwargs: x)

        routing = MagicMock(spec=RoutingService)
        routing.route_query = MagicMock(return_value=["working", "history"])
        routing.preprocess_query = MagicMock(side_effect=lambda x: x)

        kg = MagicMock(spec=KnowledgeGraphService)
        kg.knowledge_graph = {"routing_patterns": {}, "context_action_effectiveness": {}}
        kg.find_known_solutions = AsyncMock(return_value=[])
        kg.extract_concepts = MagicMock(return_value=["test"])
        kg.content_graph = MagicMock()
        kg.content_graph._doc_entities = {}

        embed_fn = AsyncMock(return_value=[0.1] * 384)

        return SearchService(
            collections=collections,
            scoring_service=scoring,
            routing_service=routing,
            kg_service=kg,
            embed_fn=embed_fn,
        )

    @pytest.mark.asyncio
    async def test_search_routes_query(self, mock_service):
        """Should route query when collections not specified."""
        await mock_service.search("test query", limit=5)
        mock_service.routing_service.route_query.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_uses_explicit_collections(self, mock_service):
        """Should use explicit collections when provided."""
        await mock_service.search("test query", collections=["history"], limit=5)
        mock_service.routing_service.route_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_preprocesses_query(self, mock_service):
        """Should preprocess query before embedding."""
        await mock_service.search("test query", limit=5)
        mock_service.routing_service.preprocess_query.assert_called()

    @pytest.mark.asyncio
    async def test_search_generates_embedding(self, mock_service):
        """Should generate embedding for query."""
        await mock_service.search("test query", limit=5)
        mock_service.embed_fn.assert_called()

    @pytest.mark.asyncio
    async def test_search_applies_scoring(self, mock_service):
        """Should apply scoring to results."""
        await mock_service.search("test query", limit=5)
        mock_service.scoring_service.apply_scoring_to_results.assert_called()

    @pytest.mark.asyncio
    async def test_search_returns_list_by_default(self, mock_service):
        """Should return list when return_metadata=False."""
        result = await mock_service.search("test query", limit=5)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_returns_dict_with_metadata(self, mock_service):
        """Should return dict when return_metadata=True."""
        result = await mock_service.search("test query", limit=5, return_metadata=True)
        assert isinstance(result, dict)
        assert "results" in result
        assert "total" in result
        assert "has_more" in result

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, mock_service):
        """Should respect limit parameter."""
        result = await mock_service.search("test query", limit=1)
        assert len(result) <= 1

    @pytest.mark.asyncio
    async def test_search_handles_empty_query(self, mock_service):
        """Empty query should return all items."""
        # Mock get for empty query path
        for coll in mock_service.collections.values():
            coll.collection = MagicMock()
            coll.collection.get = MagicMock(return_value={
                'ids': ['id1', 'id2'],
                'documents': ['doc1', 'doc2'],
                'metadatas': [{'score': 0.5}, {'score': 0.6}]
            })

        result = await mock_service.search("", limit=10)
        assert len(result) > 0


class TestEntityBoost:
    """Test entity boost calculation."""

    @pytest.fixture
    def mock_service(self):
        """Create SearchService with entity mocks."""
        kg = MagicMock(spec=KnowledgeGraphService)
        kg.extract_concepts = MagicMock(return_value=["python", "django"])
        kg.content_graph = MagicMock()
        kg.content_graph._doc_entities = {
            "doc_1": {"python", "django", "web"},
        }
        kg.content_graph.entities = {
            "python": {"avg_quality": 0.9},
            "django": {"avg_quality": 0.8},
        }
        kg.knowledge_graph = {"routing_patterns": {}, "context_action_effectiveness": {}}

        return SearchService(
            collections={},
            scoring_service=MagicMock(),
            routing_service=MagicMock(),
            kg_service=kg,
            embed_fn=AsyncMock(),
        )

    def test_entity_boost_with_matches(self, mock_service):
        """Should boost documents with matching high-quality entities."""
        boost = mock_service._calculate_entity_boost("python django", "doc_1")
        # python (0.9) + django (0.8) = 1.7 quality
        # boost = 1.0 + min(1.7 * 0.2, 0.5) = 1.0 + 0.34 = 1.34
        assert boost > 1.0
        assert boost <= 1.5

    def test_entity_boost_no_matches(self, mock_service):
        """Should return 1.0 when no entity matches."""
        boost = mock_service._calculate_entity_boost("python django", "doc_unknown")
        assert boost == 1.0

    def test_entity_boost_empty_query(self, mock_service):
        """Should return 1.0 for empty query."""
        mock_service.kg_service.extract_concepts = MagicMock(return_value=[])
        boost = mock_service._calculate_entity_boost("", "doc_1")
        assert boost == 1.0


class TestDocEffectiveness:
    """Test document effectiveness calculation."""

    @pytest.fixture
    def mock_service(self):
        """Create SearchService with effectiveness data."""
        kg = MagicMock(spec=KnowledgeGraphService)
        kg.knowledge_graph = {
            "routing_patterns": {},
            "context_action_effectiveness": {
                "context|action|coll": {
                    "examples": [
                        {"doc_id": "doc_1", "outcome": "worked"},
                        {"doc_id": "doc_1", "outcome": "worked"},
                        {"doc_id": "doc_1", "outcome": "failed"},
                        {"doc_id": "doc_2", "outcome": "partial"},
                    ]
                }
            }
        }

        return SearchService(
            collections={},
            scoring_service=MagicMock(),
            routing_service=MagicMock(),
            kg_service=kg,
            embed_fn=AsyncMock(),
        )

    def test_doc_effectiveness_calculates_rate(self, mock_service):
        """Should calculate success rate correctly."""
        eff = mock_service.get_doc_effectiveness("doc_1")
        assert eff is not None
        assert eff["successes"] == 2
        assert eff["failures"] == 1
        assert eff["total_uses"] == 3
        # success_rate = (2 + 0) / 3 = 0.667
        assert abs(eff["success_rate"] - 0.667) < 0.01

    def test_doc_effectiveness_unknown_doc(self, mock_service):
        """Should return None for unknown document."""
        eff = mock_service.get_doc_effectiveness("unknown_doc")
        assert eff is None

    def test_doc_effectiveness_partial_counts(self, mock_service):
        """Should count partial as 0.5 success."""
        eff = mock_service.get_doc_effectiveness("doc_2")
        assert eff is not None
        assert eff["partials"] == 1
        # success_rate = (0 + 1*0.5) / 1 = 0.5
        assert eff["success_rate"] == 0.5


class TestCollectionBoosts:
    """Test collection-specific distance boosts."""

    @pytest.fixture
    def service(self):
        """Create SearchService."""
        kg = MagicMock(spec=KnowledgeGraphService)
        kg.extract_concepts = MagicMock(return_value=[])
        kg.content_graph = MagicMock()
        kg.content_graph._doc_entities = {}
        kg.content_graph.entities = {}
        kg.knowledge_graph = {"routing_patterns": {}, "context_action_effectiveness": {}}

        return SearchService(
            collections={},
            scoring_service=MagicMock(),
            routing_service=MagicMock(),
            kg_service=kg,
            embed_fn=AsyncMock(),
        )

    def test_patterns_boost(self, service):
        """Patterns should get 10% distance reduction."""
        result = {"distance": 1.0, "metadata": {}}
        service._apply_collection_boost(result, "patterns", "query")
        assert result["distance"] == 0.9

    def test_memory_bank_quality_boost(self, service):
        """Memory bank should boost by quality score."""
        result = {
            "distance": 1.0,
            "id": "doc_1",
            "metadata": {"importance": 0.9, "confidence": 0.9}
        }
        service._apply_collection_boost(result, "memory_bank", "query")
        # quality = 0.81, metadata_boost = 1.0 - 0.81*0.8 = 0.352
        assert result["distance"] < 1.0

    def test_books_recent_upload_boost(self, service):
        """Recent books should get boost."""
        result = {
            "distance": 1.0,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "metadata": {}
        }
        service._apply_collection_boost(result, "books", "query")
        assert result["distance"] == 0.7


class TestCaching:
    """Test doc_id caching for outcome scoring."""

    @pytest.fixture
    def service(self):
        """Create SearchService."""
        kg = MagicMock(spec=KnowledgeGraphService)
        kg.extract_concepts = MagicMock(return_value=["test"])
        kg.knowledge_graph = {"routing_patterns": {}}

        return SearchService(
            collections={},
            scoring_service=MagicMock(),
            routing_service=MagicMock(),
            kg_service=kg,
            embed_fn=AsyncMock(),
        )

    def test_track_search_caches_doc_ids(self, service):
        """Should cache doc_ids from scoreable collections."""
        results = [
            {"id": "working_1", "collection": "working"},
            {"id": "history_1", "collection": "history"},
            {"id": "books_1", "collection": "books"},  # Not cached
        ]

        service._track_search_results("query", results, None)

        cached = service.get_cached_doc_ids('default')
        assert "working_1" in cached
        assert "history_1" in cached
        assert "books_1" not in cached  # Books not cached

    def test_caching_per_session(self, service):
        """Should cache separately per session."""
        results1 = [{"id": "doc_1", "collection": "working"}]
        results2 = [{"id": "doc_2", "collection": "working"}]

        ctx1 = MagicMock()
        ctx1.session_id = "session_1"
        ctx2 = MagicMock()
        ctx2.session_id = "session_2"

        service._track_search_results("q1", results1, ctx1)
        service._track_search_results("q2", results2, ctx2)

        assert service.get_cached_doc_ids("session_1") == ["doc_1"]
        assert service.get_cached_doc_ids("session_2") == ["doc_2"]


class TestParseNumeric:
    """Test numeric parsing helper."""

    @pytest.fixture
    def service(self):
        return SearchService(
            collections={},
            scoring_service=MagicMock(),
            routing_service=MagicMock(),
            kg_service=MagicMock(),
            embed_fn=AsyncMock(),
        )

    def test_parse_float(self, service):
        assert service._parse_numeric(0.9) == 0.9

    def test_parse_int(self, service):
        assert service._parse_numeric(1) == 1.0

    def test_parse_list(self, service):
        assert service._parse_numeric([0.8, 0.9]) == 0.8

    def test_parse_string_high(self, service):
        assert service._parse_numeric("high") == 0.9

    def test_parse_string_medium(self, service):
        assert service._parse_numeric("medium") == 0.7

    def test_parse_string_low(self, service):
        assert service._parse_numeric("low") == 0.5

    def test_parse_none(self, service):
        assert service._parse_numeric(None) == 0.7

    def test_parse_invalid(self, service):
        assert service._parse_numeric("invalid") == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
