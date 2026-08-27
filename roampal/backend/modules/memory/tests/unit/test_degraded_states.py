"""
v0.5.9 Item 7: degraded states must surface instead of silently returning an
empty result. Embedder-down propagates an explicit EmbedderUnavailable (never
a bare []), reranker-down annotates via the skip flag but still returns
results, and get_status() exposes both to /api/status.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from roampal.backend.modules.memory.search_service import (
    SearchService,
    EmbedderUnavailable,
)
from roampal.backend.modules.memory.scoring_service import ScoringService
from roampal.backend.modules.memory.routing_service import RoutingService
from roampal.backend.modules.memory.tag_service import TagService
from roampal.backend.modules.memory.config import MemoryConfig


def _make_result(doc_id, distance=0.5, collection="working"):
    meta = {"text": f"Content for {doc_id}", "uses": 0, "score": 0.5,
            "success_count": 0.0}
    return {
        "id": doc_id,
        "distance": distance,
        "text": meta["text"],
        "content": meta["text"],
        "metadata": meta,
        "collection": collection,
    }


@pytest.fixture
def search_service():
    collections = {}
    for name in ["working", "history", "patterns", "books", "memory_bank"]:
        adapter = MagicMock()
        adapter.collection = MagicMock()
        adapter.collection.count = MagicMock(return_value=10)
        adapter.collection.get = MagicMock(
            return_value={"ids": [], "metadatas": [], "documents": []})
        adapter.hybrid_query = AsyncMock(return_value=[])
        adapter.get_fragment = MagicMock(return_value=None)
        collections[name] = adapter
    return SearchService(
        collections=collections,
        scoring_service=MagicMock(spec=ScoringService),
        routing_service=MagicMock(spec=RoutingService),
        tag_service=MagicMock(spec=TagService),
        embed_fn=AsyncMock(return_value=[0.1] * 768),
        config=MemoryConfig(),
    )


class TestEmbedderUnavailable:
    async def test_embed_failure_raises_not_bare_empty(self, search_service):
        """Embedder-down must surface as EmbedderUnavailable, never []."""
        search_service.embed_fn = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(EmbedderUnavailable, match="Embedding service unavailable"):
            await search_service.search("hello")

    async def test_embedder_unavailable_is_exception(self):
        assert issubclass(EmbedderUnavailable, Exception)


class TestRerankerDegraded:
    async def test_ce_unavailable_returns_results_and_flags(self, search_service):
        """CE-down falls back to cosine-only: same results, flagged."""
        results = [_make_result("a", 0.2), _make_result("b", 0.3)]
        with patch.object(search_service, "_load_ce", return_value=False):
            out = search_service._rerank_with_ce("query", results)
        assert out == results
        assert search_service._rerank_skipped is True

    async def test_ce_throw_returns_results_and_flags(self, search_service):
        results = [_make_result("a", 0.2)]
        with patch.object(search_service, "_load_ce", return_value=True), \
             patch.object(search_service, "_ce_predict",
                          side_effect=RuntimeError("ce exploded")):
            out = search_service._rerank_with_ce("query", results)
        assert out == results
        assert search_service._rerank_skipped is True

    async def test_flag_clears_on_successful_rerank(self, search_service):
        results = [_make_result("a", 0.2), _make_result("b", 0.3)]
        search_service._rerank_skipped = True  # poisoned by a previous search
        with patch.object(search_service, "_load_ce", return_value=True), \
             patch.object(search_service, "_ce_predict", return_value=[1.0, 2.0]):
            out = search_service._rerank_with_ce("query", results)
        assert search_service._rerank_skipped is False
        assert out[0]["id"] == "b"  # highest CE score first
        assert out[0]["final_rank_score"] == 2.0


class TestStatusShape:
    def test_get_status_keys(self, search_service):
        status = search_service.get_status()
        assert set(status.keys()) == {"available", "skipped_last_search", "model"}
        assert status["available"] is False  # never loaded in unit env
        assert status["skipped_last_search"] is False

    def test_get_status_reflects_loaded_ce(self, search_service):
        search_service._ce_loaded = True
        search_service._ce_session = MagicMock()
        status = search_service.get_status()
        assert status["available"] is True
        assert "mmarco" in status["model"]
