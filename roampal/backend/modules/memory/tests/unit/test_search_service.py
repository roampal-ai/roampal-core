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
    return AsyncMock(return_value=[0.1] * 768)


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


class TestTagCascadePythonFilter:
    """v0.5.3: Python-side tag membership filter after over-fetch."""

    @pytest.mark.asyncio
    async def test_filters_by_tag_membership(self, search_service, mock_collections):
        """Results without matching tag should be filtered out."""
        mem_match = _make_result("match", distance=0.2, noun_tags='["calvin"]')
        mem_no_match = _make_result("no_match", distance=0.1, noun_tags='["boston"]')

        mock_collections["working"].hybrid_query = AsyncMock(
            return_value=[mem_match.copy(), mem_no_match.copy()]
        )

        # Mock cosine fill to prevent untagged results from being added back.
        with patch.object(search_service, '_search_collections', new_callable=AsyncMock) as mock_cosine:
            mock_cosine.return_value = []

            results = await search_service._tag_routed_search(
                query_embedding=[0.1] * 768,
                processed_query="Calvin",
                collections=["working"],
                limit=10,
                matched_tags=["calvin"],
                metadata_filters=None,
            )

        ids = [r["id"] for r in results]
        assert "match" in ids
        assert "no_match" not in ids

    @pytest.mark.asyncio
    async def test_json_encoded_noun_tags(self, search_service, mock_collections):
        """Should parse JSON-encoded noun_tags strings."""
        mem = _make_result("json_mem", distance=0.2, noun_tags='["calvin"]')
        mem_no_tag = _make_result("no_tag", distance=0.1)

        mock_collections["working"].hybrid_query = AsyncMock(
            return_value=[mem.copy(), mem_no_tag.copy()]
        )

        with patch.object(search_service, '_search_collections', new_callable=AsyncMock) as mock_cosine:
            mock_cosine.return_value = []

            results = await search_service._tag_routed_search(
                query_embedding=[0.1] * 768,
                processed_query="Calvin",
                collections=["working"],
                limit=10,
                matched_tags=["calvin"],
                metadata_filters=None,
            )

        ids = [r["id"] for r in results]
        assert "json_mem" in ids
        assert "no_tag" not in ids

    @pytest.mark.asyncio
    async def test_list_noun_tags(self, search_service, mock_collections):
        """Should handle already-parsed list noun_tags."""
        mem = _make_result("list_mem", distance=0.2, noun_tags=["calvin"])

        mock_collections["working"].hybrid_query = AsyncMock(
            return_value=[mem.copy()]
        )

        results = await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
            processed_query="Calvin",
            collections=["working"],
            limit=10,
            matched_tags=["calvin"],
            metadata_filters=None,
        )

        ids = [r["id"] for r in results]
        assert "list_mem" in ids

    @pytest.mark.asyncio
    async def test_multiple_tag_queries(self, search_service, mock_collections):
        """Multiple tag queries should each filter independently."""
        mem_calvin = _make_result("calvin", distance=0.2, noun_tags='["calvin"]')
        mem_boston = _make_result("boston", distance=0.3, noun_tags='["boston"]')

        def hybrid_side_effect(query_vector, query_text, top_k, filters=None):
            if "Calvin" in query_text:
                return [mem_calvin.copy(), mem_boston.copy()]
            elif "Boston" in query_text:
                return [mem_calvin.copy(), mem_boston.copy()]
            return []

        mock_collections["working"].hybrid_query = AsyncMock(side_effect=hybrid_side_effect)

        results = await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
            processed_query="Calvin",
            collections=["working"],
            limit=10,
            matched_tags=["calvin", "boston"],
            metadata_filters=None,
        )

        # Both tags match, so both results should pass through
        ids = [r["id"] for r in results]
        assert "calvin" in ids
        assert "boston" in ids

    @pytest.mark.asyncio
    async def test_over_fetch_factor(self, search_service, mock_collections):
        """Should over-fetch by ×8 instead of ×2."""
        mem = _make_result("mem", distance=0.2, noun_tags='["calvin"]')
        mock_collections["working"].hybrid_query = AsyncMock(return_value=[mem.copy()])

        await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
            processed_query="Calvin",
            collections=["working"],
            limit=5,
            matched_tags=["calvin"],
            metadata_filters=None,
        )

        # v0.5.3: Tag search uses top_k=limit*8; cosine fill may also call with limit*3.
        # Check that at least one call used the ×8 over-fetch factor.
        tag_search_top_k = [
            c[1]["top_k"] for c in mock_collections["working"].hybrid_query.call_args_list
            if "top_k" in c[1]
        ]
        assert 40 in tag_search_top_k, \
            f"Expected top_k=40 (5×8) in {tag_search_top_k}"

    @pytest.mark.asyncio
    async def test_no_tag_filter_passed_to_chroma(self, search_service, mock_collections):
        """Should NOT pass tag filter to ChromaDB."""
        mem = _make_result("mem", distance=0.2, noun_tags='["calvin"]')
        mock_collections["working"].hybrid_query = AsyncMock(return_value=[mem.copy()])

        await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
            processed_query="Calvin",
            collections=["working"],
            limit=10,
            matched_tags=["calvin"],
            metadata_filters=None,
        )

        call_args = mock_collections["working"].hybrid_query.call_args
        filters = call_args[1].get("filters") or (call_args[0][3] if len(call_args[0]) > 3 else None)
        assert not filters or "noun_tags" not in filters, \
            f"Tag filter should not be passed to ChromaDB: {filters}"

    @pytest.mark.asyncio
    async def test_empty_tag_list_filters_all(self, search_service, mock_collections):
        """If no results match the tag after Python filtering, returns empty."""
        mem = _make_result("mem", distance=0.2, noun_tags='["boston"]')

        mock_collections["working"].hybrid_query = AsyncMock(
            return_value=[mem.copy()]
        )

        with patch.object(search_service, '_search_collections', new_callable=AsyncMock) as mock_cosine:
            mock_cosine.return_value = []

            results = await search_service._tag_routed_search(
                query_embedding=[0.1] * 768,
                processed_query="Calvin",
                collections=["working"],
                limit=10,
                matched_tags=["calvin"],
                metadata_filters=None,
            )

        assert len(results) == 0


class TestMemoryBankFilterWrapping:
    """v0.5.3: memory_bank filter wrapping with $and for multi-key filters."""

    @pytest.mark.asyncio
    async def test_archived_exclusion_applied(self, search_service, mock_collections):
        """memory_bank should exclude archived results via status=$ne filter."""
        mem = _make_result("mem", distance=0.2, noun_tags='["calvin"]', metadata={"status": "active"})

        captured_filters = {}

        async def capture_hybrid(query_vector, query_text, top_k, filters=None):
            captured_filters["filters"] = filters
            return [mem.copy()]

        mock_collections["memory_bank"].hybrid_query = AsyncMock(side_effect=capture_hybrid)

        await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
            processed_query="Calvin",
            collections=["memory_bank"],
            limit=10,
            matched_tags=["calvin"],
            metadata_filters=None,
        )

        filters = captured_filters["filters"]
        assert filters is not None
        # Should use $and for the status filter
        if "$and" in filters:
            status_cond = [c.get("status") for c in filters["$and"] if "status" in c]
            assert any(s == {"$ne": "archived"} for s in status_cond)

    @pytest.mark.asyncio
    async def test_user_filters_merged_with_status(self, search_service, mock_collections):
        """User metadata filters should be merged with archived exclusion."""
        mem = _make_result("mem", distance=0.2, noun_tags='["calvin"]', metadata={})

        captured_filters = {}

        async def capture_hybrid(query_vector, query_text, top_k, filters=None):
            captured_filters["filters"] = filters
            return [mem.copy()]

        mock_collections["memory_bank"].hybrid_query = AsyncMock(side_effect=capture_hybrid)

        user_filters = {"importance": {"$gte": 0.7}}

        await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
            processed_query="Calvin",
            collections=["memory_bank"],
            limit=10,
            matched_tags=["calvin"],
            metadata_filters=user_filters,
        )

        filters = captured_filters["filters"]
        assert filters is not None
        # Should have both user filter and status exclusion in $and
        if "$and" in filters:
            keys_in_and = []
            for cond in filters["$and"]:
                keys_in_and.extend(cond.keys())
            assert "status" in keys_in_and
            assert "importance" in keys_in_and

    @pytest.mark.asyncio
    async def test_non_memory_bank_no_status_filter(self, search_service, mock_collections):
        """Non-memory_bank collections should NOT get status filter."""
        mem = _make_result("mem", distance=0.2, noun_tags='["calvin"]', metadata={})

        captured_filters = {}

        async def capture_hybrid(query_vector, query_text, top_k, filters=None):
            captured_filters["filters"] = filters
            return [mem.copy()]

        mock_collections["working"].hybrid_query = AsyncMock(side_effect=capture_hybrid)

        await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
            processed_query="Calvin",
            collections=["working"],
            limit=10,
            matched_tags=["calvin"],
            metadata_filters=None,
        )

        filters = captured_filters["filters"]
        assert not filters or "status" not in str(filters), \
            f"Non-memory_bank should not get status filter: {filters}"

    @pytest.mark.asyncio
    async def test_memory_bank_no_user_filters(self, search_service, mock_collections):
        """memory_bank with no user filters should only have archived exclusion."""
        mem = _make_result("mem", distance=0.2, noun_tags='["calvin"]', metadata={})

        captured_filters = {}

        async def capture_hybrid(query_vector, query_text, top_k, filters=None):
            captured_filters["filters"] = filters
            return [mem.copy()]

        mock_collections["memory_bank"].hybrid_query = AsyncMock(side_effect=capture_hybrid)

        await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
            processed_query="Calvin",
            collections=["memory_bank"],
            limit=10,
            matched_tags=["calvin"],
            metadata_filters=None,
        )

        filters = captured_filters["filters"]
        assert filters is not None


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

        # v0.5.3: No tag filter passed to ChromaDB — returns all results for Python-side filtering
        mock_collections["working"].hybrid_query = AsyncMock(
            return_value=[mem_2tags.copy(), mem_1tag.copy()]
        )

        results = await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
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
            query_embedding=[0.1] * 768,
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
                query_embedding=[0.1] * 768,
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
            query_embedding=[0.1] * 768,
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

        # v0.5.3: No tag filter — returns all results for Python-side filtering
        mock_collections["working"].hybrid_query = AsyncMock(return_value=[mem.copy()])

        results = await search_service._tag_routed_search(
            query_embedding=[0.1] * 768,
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


class TestV052CERerankOffload:
    """v0.5.2: _rerank_with_ce must be invoked via asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_ce_rerank_offloaded_to_executor(self, search_service):
        """search() must hand _rerank_with_ce off to a thread so ONNX inference
        does not block the event loop."""
        import asyncio

        captured = []
        original_to_thread = asyncio.to_thread

        async def spy_to_thread(fn, /, *args, **kwargs):
            captured.append((getattr(fn, "__name__", repr(fn)), args))
            return await original_to_thread(fn, *args, **kwargs)

        # Bypass the heavy pool path — feed search() a result set directly by
        # mocking the pool builder to return one item, then run CE rerank.
        search_service._build_search_pool = AsyncMock(return_value=[{
            "id": "d1",
            "text": "hello",
            "content": "hello",
            "metadata": {"score": 0.5, "uses": 0, "success_count": 0.0},
            "distance": 0.2,
            "collection": "working",
        }])

        # Force the CE path to be invoked even without a real model loaded.
        rerank_stub = MagicMock(side_effect=lambda q, r, top_k: r)
        rerank_stub.__name__ = "_rerank_with_ce"
        search_service._rerank_with_ce = rerank_stub

        with patch("asyncio.to_thread", side_effect=spy_to_thread):
            await search_service.search(query="hello", limit=4)

        rerank_calls = [c for c in captured if c[0] == "_rerank_with_ce"]
        assert rerank_calls, (
            f"_rerank_with_ce never went through asyncio.to_thread; "
            f"captured calls: {captured}"
        )
