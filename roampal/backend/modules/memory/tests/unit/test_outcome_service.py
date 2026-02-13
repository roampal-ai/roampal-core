"""
Unit Tests for OutcomeService

Tests the extracted outcome recording logic.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from roampal.backend.modules.memory.outcome_service import OutcomeService
from roampal.backend.modules.memory.config import MemoryConfig


class TestOutcomeServiceInit:
    """Test OutcomeService initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default config."""
        service = OutcomeService(collections={})
        assert service.config.promotion_score_threshold == 0.7

    def test_init_with_custom_config(self):
        """Should use custom config."""
        config = MemoryConfig(promotion_score_threshold=0.8)
        service = OutcomeService(collections={}, config=config)
        assert service.config.promotion_score_threshold == 0.8

    def test_init_with_services(self):
        """Should accept KG and promotion services."""
        kg_mock = MagicMock()
        promo_mock = MagicMock()
        service = OutcomeService(
            collections={},
            kg_service=kg_mock,
            promotion_service=promo_mock
        )
        assert service.kg_service == kg_mock
        assert service.promotion_service == promo_mock


class TestRecordOutcome:
    """Test outcome recording."""

    @pytest.fixture
    def mock_collections(self):
        """Create mock collections."""
        working = MagicMock()
        working.get_fragment = MagicMock(return_value={
            "content": "test content",
            "metadata": {
                "text": "test content",
                "score": 0.5,
                "uses": 0,
                "outcome_history": "[]"
            }
        })
        working.update_fragment_metadata = MagicMock()
        working.collection = MagicMock()
        working.collection.count = MagicMock(return_value=10)

        return {"working": working, "history": MagicMock()}

    @pytest.fixture
    def service(self, mock_collections):
        """Create OutcomeService instance."""
        return OutcomeService(collections=mock_collections)

    @pytest.mark.asyncio
    async def test_record_worked_outcome(self, service, mock_collections):
        """Should increase score for worked outcome."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="worked"
        )

        assert result is not None
        assert result["score"] > 0.5  # Score increased
        assert result["uses"] == 1
        assert result["success_count"] == 1.0  # v0.2.8: Worked adds 1 success
        assert result["last_outcome"] == "worked"
        mock_collections["working"].update_fragment_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_failed_outcome(self, service, mock_collections):
        """Should decrease score for failed outcome."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="failed",
            failure_reason="Did not work"
        )

        assert result is not None
        assert result["score"] < 0.5  # Score decreased
        assert result["uses"] == 1  # v0.2.8: Failed now increments uses
        assert result["success_count"] == 0.0  # v0.2.8: Failed adds 0 success
        assert result["last_outcome"] == "failed"

    @pytest.mark.asyncio
    async def test_record_partial_outcome(self, service, mock_collections):
        """Should slightly increase score for partial outcome."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="partial"
        )

        assert result is not None
        assert result["score"] > 0.5  # Score slightly increased
        assert result["uses"] == 1  # Uses incremented for partial
        assert result["success_count"] == 0.5  # v0.2.8: Partial adds 0.5 success
        assert result["last_outcome"] == "partial"

    @pytest.mark.asyncio
    async def test_safeguard_books(self, service):
        """Should not score book chunks."""
        result = await service.record_outcome(
            doc_id="books_test123",
            outcome="worked"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_outcome_all_collections(self, service, mock_collections):
        """v0.3.6: unknown = +1 use, +0.25 success, -0.05 raw score for ALL collections."""
        # Test unknown outcome on working collection
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="unknown"
        )
        assert result is not None
        assert result["uses"] == 1  # Incremented
        assert result["success_count"] == 0.25  # v0.2.9: unknown = 0.25 success for all collections
        assert result["score"] == 0.45  # v0.3.6: -0.05 raw score penalty (0.5 - 0.05)

    @pytest.mark.asyncio
    async def test_not_found(self, service, mock_collections):
        """Should return None for non-existent document."""
        mock_collections["working"].get_fragment = MagicMock(return_value=None)

        result = await service.record_outcome(
            doc_id="working_nonexistent",
            outcome="worked"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_outcome_history_tracking(self, service, mock_collections):
        """Should track outcome history."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="worked"
        )

        history = json.loads(result["outcome_history"])
        assert len(history) == 1
        assert history[0]["outcome"] == "worked"

    @pytest.mark.asyncio
    async def test_failure_reason_tracking(self, service, mock_collections):
        """Should track failure reasons."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="failed",
            failure_reason="Test failure"
        )

        reasons = json.loads(result["failure_reasons"])
        assert len(reasons) == 1
        assert reasons[0]["reason"] == "Test failure"

    @pytest.mark.asyncio
    async def test_success_context_tracking(self, service, mock_collections):
        """Should track success contexts."""
        result = await service.record_outcome(
            doc_id="working_test123",
            outcome="worked",
            context={"topic": "test"}
        )

        contexts = json.loads(result["success_contexts"])
        assert len(contexts) == 1
        assert contexts[0]["topic"] == "test"


class TestScoreCalculation:
    """Test score calculation logic."""

    @pytest.fixture
    def service(self):
        return OutcomeService(collections={})

    def test_worked_increases_score(self, service):
        """Worked should increase score."""
        delta, new_score, uses, success_delta = service._calculate_score_update(
            "worked", 0.5, 0
        )
        assert delta > 0
        assert new_score > 0.5
        assert uses == 1
        assert success_delta == 1.0  # v0.2.8

    def test_failed_decreases_score(self, service):
        """Failed should decrease score."""
        delta, new_score, uses, success_delta = service._calculate_score_update(
            "failed", 0.5, 0
        )
        assert delta < 0
        assert new_score < 0.5
        assert uses == 1  # v0.2.8: Failed now increments uses
        assert success_delta == 0.0  # v0.2.8

    def test_partial_slightly_increases(self, service):
        """Partial should slightly increase score."""
        delta, new_score, uses, success_delta = service._calculate_score_update(
            "partial", 0.5, 0
        )
        assert delta > 0
        assert delta < 0.1  # Small increase
        assert new_score > 0.5
        assert uses == 1
        assert success_delta == 0.5  # v0.2.8

    def test_score_capped_at_1(self, service):
        """Score should not exceed 1.0."""
        delta, new_score, uses, _ = service._calculate_score_update(
            "worked", 0.95, 0
        )
        assert new_score <= 1.0

    def test_score_capped_at_0(self, service):
        """Score should not go below 0.0."""
        delta, new_score, uses, _ = service._calculate_score_update(
            "failed", 0.1, 0
        )
        assert new_score >= 0.0


class TestCountSuccesses:
    """Test success counting from history."""

    @pytest.fixture
    def service(self):
        return OutcomeService(collections={})

    def test_empty_returns_zero(self, service):
        """Empty history returns 0."""
        assert service.count_successes_from_history("") == 0
        assert service.count_successes_from_history("[]") == 0

    def test_worked_counts_as_one(self, service):
        """Worked outcomes count as 1."""
        history = json.dumps([{"outcome": "worked"}])
        assert service.count_successes_from_history(history) == 1.0

    def test_partial_counts_as_half(self, service):
        """Partial outcomes count as 0.5."""
        history = json.dumps([{"outcome": "partial"}])
        assert service.count_successes_from_history(history) == 0.5

    def test_failed_counts_as_zero(self, service):
        """Failed outcomes count as 0."""
        history = json.dumps([{"outcome": "failed"}])
        assert service.count_successes_from_history(history) == 0

    def test_mixed_outcomes(self, service):
        """Mixed outcomes sum correctly."""
        history = json.dumps([
            {"outcome": "worked"},
            {"outcome": "partial"},
            {"outcome": "failed"},
            {"outcome": "worked"}
        ])
        assert service.count_successes_from_history(history) == 2.5

    def test_invalid_json_returns_zero(self, service):
        """Invalid JSON returns 0."""
        assert service.count_successes_from_history("invalid") == 0


class TestOutcomeStats:
    """Test outcome statistics retrieval."""

    @pytest.fixture
    def mock_collections(self):
        working = MagicMock()
        working.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {
                "score": 0.8,
                "uses": 5,
                "last_outcome": "worked",
                "outcome_history": json.dumps([
                    {"outcome": "worked"},
                    {"outcome": "worked"},
                    {"outcome": "partial"},
                    {"outcome": "failed"},
                    {"outcome": "worked"}
                ])
            }
        })
        return {"working": working}

    @pytest.fixture
    def service(self, mock_collections):
        return OutcomeService(collections=mock_collections)

    def test_get_outcome_stats(self, service):
        """Should return correct outcome stats."""
        stats = service.get_outcome_stats("working_test123")

        assert stats["doc_id"] == "working_test123"
        assert stats["collection"] == "working"
        assert stats["score"] == 0.8
        assert stats["uses"] == 5
        assert stats["last_outcome"] == "worked"
        assert stats["outcomes"]["worked"] == 3
        assert stats["outcomes"]["partial"] == 1
        assert stats["outcomes"]["failed"] == 1
        assert stats["total_outcomes"] == 5

    def test_get_stats_not_found(self, service, mock_collections):
        """Should return error for non-existent document."""
        mock_collections["working"].get_fragment = MagicMock(return_value=None)

        stats = service.get_outcome_stats("working_nonexistent")
        assert stats["error"] == "not_found"


class TestKGIntegration:
    """Test KG service integration."""

    @pytest.fixture
    def mock_kg_service(self):
        kg = MagicMock()
        kg.extract_concepts = MagicMock(return_value=["test", "concept"])
        kg.update_kg_routing = AsyncMock()
        kg.build_concept_relationships = MagicMock()
        kg.add_problem_category = MagicMock()
        kg.add_solution_pattern = MagicMock()
        kg.update_success_rate = MagicMock()
        kg.add_failure_pattern = MagicMock()
        kg.add_problem_solution = MagicMock()
        kg.add_solution_pattern_entry = MagicMock()
        kg.debounced_save_kg = AsyncMock()
        return kg

    @pytest.fixture
    def mock_collections(self):
        working = MagicMock()
        working.get_fragment = MagicMock(return_value={
            "content": "test solution",
            "metadata": {
                "text": "test solution",
                "query": "test problem",
                "score": 0.5,
                "uses": 0,
                "outcome_history": "[]"
            }
        })
        working.update_fragment_metadata = MagicMock()
        working.collection = MagicMock()
        working.collection.count = MagicMock(return_value=10)
        return {"working": working}

    @pytest.fixture
    def service(self, mock_collections, mock_kg_service):
        return OutcomeService(
            collections=mock_collections,
            kg_service=mock_kg_service
        )

    @pytest.mark.asyncio
    async def test_updates_kg_routing(self, service, mock_kg_service):
        """Should update KG routing on outcome."""
        await service.record_outcome("working_test123", "worked")

        assert mock_kg_service.update_kg_routing.call_count >= 1  # Desktop calls twice

    @pytest.mark.asyncio
    async def test_builds_relationships_on_success(self, service, mock_kg_service):
        """Should build concept relationships on worked outcome."""
        await service.record_outcome("working_test123", "worked")
        await asyncio.sleep(0.1)  # v0.2.3: Let deferred learning task run

        mock_kg_service.build_concept_relationships.assert_called()
        mock_kg_service.add_problem_category.assert_called()

    @pytest.mark.asyncio
    async def test_tracks_failure_patterns(self, service, mock_kg_service):
        """Should track failure patterns on failed outcome."""
        await service.record_outcome(
            "working_test123",
            "failed",
            failure_reason="Test failure"
        )
        await asyncio.sleep(0.1)  # v0.2.3: Let deferred learning task run

        mock_kg_service.add_failure_pattern.assert_called()

    @pytest.mark.asyncio
    async def test_saves_kg_after_update(self, service, mock_kg_service):
        """Should save KG after updates."""
        await service.record_outcome("working_test123", "worked")
        await asyncio.sleep(0.1)  # v0.2.3: Let deferred learning task run

        mock_kg_service.debounced_save_kg.assert_called()


class TestPromotionIntegration:
    """Test promotion service integration."""

    @pytest.fixture
    def mock_promotion_service(self):
        promo = MagicMock()
        promo.handle_promotion = AsyncMock()
        return promo

    @pytest.fixture
    def mock_collections(self):
        working = MagicMock()
        working.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"score": 0.5, "uses": 0, "outcome_history": "[]"}
        })
        working.update_fragment_metadata = MagicMock()
        working.collection = MagicMock()
        working.collection.count = MagicMock(return_value=10)
        return {"working": working}

    @pytest.fixture
    def service(self, mock_collections, mock_promotion_service):
        return OutcomeService(
            collections=mock_collections,
            promotion_service=mock_promotion_service
        )

    @pytest.mark.asyncio
    async def test_calls_promotion_handler(self, service, mock_promotion_service):
        """Should call promotion handler after outcome."""
        await service.record_outcome("working_test123", "worked")
        await asyncio.sleep(0.1)  # v0.2.3: Let deferred learning task run

        mock_promotion_service.handle_promotion.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_correct_params_to_promotion(self, service, mock_promotion_service):
        """Should pass correct parameters to promotion handler."""
        await service.record_outcome("working_test123", "worked")
        await asyncio.sleep(0.1)  # v0.2.3: Let deferred learning task run

        call_args = mock_promotion_service.handle_promotion.call_args
        assert call_args[1]["doc_id"] == "working_test123"
        assert call_args[1]["collection"] == "working"
        assert "score" in call_args[1]
        assert "uses" in call_args[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
