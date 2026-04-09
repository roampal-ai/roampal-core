"""
Tests for SearchService — TagCascade retrieval with CE reranking.

v0.4.5: Tests the benchmark-validated tag_cascade_cosine algorithm:
- Tags-first cascade pool construction (overlap tiers, cosine within tier)
- CE reranking with raw CE score (no Wilson blend)
- Cosine fallback when no tags match
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from roampal.backend.modules.memory.search_service import SearchService
from roampal.backend.modules.memory.scoring_service import ScoringService
from roampal.backend.modules.memory.routing_service import RoutingService
from roampal.backend.modules.memory.tag_service import TagService
from roampal.backend.modules.memory.config import MemoryConfig


def _make_result(doc_id, distance=0.5, collection="working", noun_tags=None, metadata=None):
    """Helper to create a mock search result."""
    meta = metadata or {}
    meta.setdefault("text", f"Content for {doc_id}")
    meta.setdefault("uses", 0)
    meta.setdefault("score", 0.5)
    meta.setdefault("success_count", 0.0)
    if noun_tags:
        meta["noun_tags"] = noun_tags
    return {
        "id": doc_id,
        "distance": distance,
        "text": meta["text"],
        "content": meta["text"],
        "metadata": meta,
        "collection": collection,
    }


@pytest.fixture
def config():
    return MemoryConfig()


@pytest.fixture
def scoring_service(config):
    return ScoringService(config=config)


@pytest.fixture
def routing_service():
    return RoutingService()


@pytest.fixture
def tag_service():
    ts = TagService()
    ts.add_known_tags(["calvin", "boston", "muscle car", "joanna"])
    return ts


@pytest.fixture
def embed_fn():
    return AsyncMock(return_value=[0.1] * 384)


@pytest.fixture
def mock_collections():
    """Create mock collection adapters for all 5 collections."""
    collections = {}
    for name in ["working", "history", "patterns", "books", "memory_bank"]:
        adapter = MagicMock()
        adapter.collection = MagicMock()
        adapter.collection.count = MagicMock(return_value=10)
        adapter.collection.get = MagicMock(return_value={"ids": [], "metadatas": [], "documents": []})
        adapter.hybrid_query = AsyncMock(return_value=[])
        adapter.get_fragment = MagicMock(return_value=None)
        collections[name] = adapter
    return collections


@pytest.fixture
def search_service(mock_collections, scoring_service, routing_service, tag_service, embed_fn, config):
    return SearchService(
        collections=mock_collections,
        scoring_service=scoring_service,
        routing_service=routing_service,
        tag_service=tag_service,
        embed_fn=embed_fn,
        config=config,
    )


class TestSearchServiceInit:
    """Test SearchService initialization."""

    def test_accepts_tag_service(self, mock_collections, scoring_service, routing_service, tag_service, embed_fn):
        """v0.4.5: SearchService requires tag_service."""
        svc = SearchService(
            collections=mock_collections,
            scoring_service=scoring_service,
            routing_service=routing_service,
            tag_service=tag_service,
            embed_fn=embed_fn,
        )
        assert svc.tag_service is tag_service

    def test_accepts_kg_service_kwarg(self, mock_collections, scoring_service, routing_service, tag_service, embed_fn):
        """v0.4.5: Should accept and ignore kg_service via **kwargs."""
        svc = SearchService(
            collections=mock_collections,
            scoring_service=scoring_service,
            routing_service=routing_service,
            tag_service=tag_service,
            embed_fn=embed_fn,
            kg_service=MagicMock(),  # Should not crash
        )
        assert svc.tag_service is tag_service


class TestTagRoutedSearch:
    """Test TagCascade pool construction."""

    @pytest.mark.asyncio
    async def test_tier_filling_order(self, search_service, mock_collections):
        """Pool fills from highest overlap tier down."""
        # Memory matching 2 tags (calvin + boston)
        mem_2tags = _make_result("mem_2", distance=0.8, noun_tags='["calvin", "boston"]')
        # Memory matching 1 tag (calvin only) but closer cosine
        mem_1tag = _make_result("mem_1", distance=0.2, noun_tags='["calvin"]')

        # Setup: "calvin" query returns both, "boston" returns only mem_2
        def hybrid_side_effect(query_vector, query_text, top_k, filters=None):
            if filters and "noun_tags" in filters:
                tag_filter = filters["noun_tags"]["$contains"]
                if '"calvin"' in tag_filter:
                    return [mem_2tags.copy(), mem_1tag.copy()]
                elif '"boston"' in tag_filter:
                    return [mem_2tags.copy()]
            return []

        mock_collections["working"].hybrid_query = AsyncMock(side_effect=hybrid_side_effect)

        results = await search_service._tag_routed_search(
            query_embedding=[0.1] * 384,
            processed_query="What did Calvin do in Boston?",
            collections=["working"],
            limit=10,
            matched_tags=["calvin", "boston"],
            metadata_filters=None,
        )

        # mem_2 (overlap=2) should come before mem_1 (overlap=1)
        ids = [r["id"] for r in results]
        assert ids.index("mem_2") < ids.index("mem_1")

    @pytest.mark.asyncio
    async def test_cosine_tiebreaker_within_tier(self, search_service, mock_collections):
        """Within same overlap tier, closer cosine distance ranks first."""
        mem_close = _make_result("close", distance=0.1, noun_tags='["calvin"]')
        mem_far = _make_result("far", distance=0.9, noun_tags='["calvin"]')

        mock_collections["working"].hybrid_query = AsyncMock(
            return_value=[mem_far.copy(), mem_close.copy()]
        )

        results = await search_service._tag_routed_search(
            query_embedding=[0.1] * 384,
            processed_query="Tell me about Calvin",
            collections=["working"],
            limit=10,
            matched_tags=["calvin"],
            metadata_filters=None,
        )

        ids = [r["id"] for r in results]
        assert ids.index("close") < ids.index("far")

    @pytest.mark.asyncio
    async def test_cosine_fill_when_tags_insufficient(self, search_service, mock_collections):
        """Untagged cosine results fill remaining pool slots."""
        tagged = _make_result("tagged", distance=0.3, noun_tags='["calvin"]')
        untagged = _make_result("untagged", distance=0.2)

        # Tag query returns tagged mem
        mock_collections["working"].hybrid_query = AsyncMock(return_value=[tagged.copy()])
        # Cosine search returns untagged (patched via _search_collections)

        with patch.object(search_service, '_search_collections', new_callable=AsyncMock) as mock_cosine:
            mock_cosine.return_value = [untagged.copy()]

            results = await search_service._tag_routed_search(
                query_embedding=[0.1] * 384,
                processed_query="Calvin",
                collections=["working"],
                limit=10,
                matched_tags=["calvin"],
                metadata_filters=None,
            )

        ids = [r["id"] for r in results]
        assert "tagged" in ids
        assert "untagged" in ids
        # Tagged should come first (overlap=1 > overlap=0)
        assert ids.index("tagged") < ids.index("untagged")

    @pytest.mark.asyncio
    async def test_pool_size_capped(self, search_service, mock_collections):
        """Pool does not exceed CE_CANDIDATE_POOL size."""
        many_results = [
            _make_result(f"mem_{i}", distance=i * 0.01, noun_tags='["calvin"]')
            for i in range(50)
        ]
        mock_collections["working"].hybrid_query = AsyncMock(return_value=many_results)

        results = await search_service._tag_routed_search(
            query_embedding=[0.1] * 384,
            processed_query="Calvin",
            collections=["working"],
            limit=10,
            matched_tags=["calvin"],
            metadata_filters=None,
        )

        assert len(results) <= search_service.CE_CANDIDATE_POOL

    @pytest.mark.asyncio
    async def test_tag_overlap_count_set(self, search_service, mock_collections):
        """Results have tag_overlap_count metadata."""
        mem = _make_result("mem", distance=0.3, noun_tags='["calvin", "boston"]')

        def hybrid_side_effect(query_vector, query_text, top_k, filters=None):
            if filters and "noun_tags" in filters:
                return [mem.copy()]
            return []

        mock_collections["working"].hybrid_query = AsyncMock(side_effect=hybrid_side_effect)

        results = await search_service._tag_routed_search(
            query_embedding=[0.1] * 384,
            processed_query="Calvin in Boston",
            collections=["working"],
            limit=10,
            matched_tags=["calvin", "boston"],
            metadata_filters=None,
        )

        assert results[0]["tag_overlap_count"] == 2


class TestCEReranking:
    """Test cross-encoder reranking uses raw CE score."""

    def test_raw_ce_score_no_wilson_blend(self, search_service):
        """final_rank_score should be raw CE score, not Wilson blend."""
        results = [
            _make_result("high_wilson", distance=0.3, metadata={"text": "content A", "uses": 10, "score": 0.9, "success_count": 9.0}),
            _make_result("low_wilson", distance=0.3, metadata={"text": "content B", "uses": 10, "score": 0.2, "success_count": 1.0}),
        ]

        # Mock CE to score low_wilson higher than high_wilson
        with patch.object(search_service, '_load_ce', return_value=True):
            with patch.object(search_service, '_ce_predict', return_value=[1.0, 5.0]):
                reranked = search_service._rerank_with_ce("query", results)

        # low_wilson has higher CE score (5.0) so should rank first
        assert reranked[0]["id"] == "low_wilson"
        # final_rank_score should be raw CE, not blended
        assert reranked[0]["final_rank_score"] == 5.0
        assert reranked[1]["final_rank_score"] == 1.0

    def test_ce_score_preserved(self, search_service):
        """ce_score field preserved on results."""
        results = [_make_result("a", metadata={"text": "content"})]

        with patch.object(search_service, '_load_ce', return_value=True):
            with patch.object(search_service, '_ce_predict', return_value=[3.14]):
                reranked = search_service._rerank_with_ce("query", results)

        assert reranked[0]["ce_score"] == 3.14

    def test_ce_unavailable_returns_original(self, search_service):
        """When CE model not loaded, returns results unchanged."""
        results = [_make_result("a"), _make_result("b")]
        with patch.object(search_service, '_load_ce', return_value=False):
            reranked = search_service._rerank_with_ce("query", results)
        assert [r["id"] for r in reranked] == ["a", "b"]


class TestCosineOnlyFallback:
    """Test behavior when no tags match."""

    @pytest.mark.asyncio
    async def test_no_tags_uses_cosine(self, search_service, mock_collections, embed_fn):
        """When no tags match, search falls back to cosine + CE."""
        cosine_result = _make_result("cosine_hit", distance=0.2)
        mock_collections["working"].hybrid_query = AsyncMock(return_value=[cosine_result])

        # Empty tag service (no known tags to match)
        search_service.tag_service = TagService()

        results = await search_service.search(
            query="something with no matching tags",
            limit=5,
            collections=["working"],
        )

        assert len(results) > 0


class TestCollectionBoost:
    """Test collection-specific distance boosts."""

    def test_patterns_boost(self, search_service):
        """Patterns collection gets 0.9x distance boost."""
        result = {"distance": 1.0, "metadata": {}}
        search_service._apply_collection_boost(result, "patterns", "query")
        assert result["distance"] == 0.9

    def test_memory_bank_cold_start_boost(self, search_service):
        """Memory bank with <3 uses gets quality-based boost."""
        result = {
            "distance": 1.0,
            "metadata": {"importance": 0.9, "confidence": 0.9, "uses": 0}
        }
        search_service._apply_collection_boost(result, "memory_bank", "query")
        assert result["distance"] < 1.0

    def test_memory_bank_no_boost_after_3_uses(self, search_service):
        """Memory bank with 3+ uses gets no cold start boost."""
        result = {
            "distance": 1.0,
            "metadata": {"importance": 0.9, "confidence": 0.9, "uses": 5}
        }
        search_service._apply_collection_boost(result, "memory_bank", "query")
        assert result["distance"] == 1.0
