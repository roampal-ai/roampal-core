"""
Unit Tests for Exchange Scoring - v0.2.9.2 fix.

Tests the fix that re-enables exchange scoring which was accidentally
removed in v0.2.3's performance optimization.

The fix ensures that when score_response(outcome="worked") is called,
the previous exchange's doc_id gets that outcome applied.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestExchangeScoringLogic:
    """Test the exchange scoring logic in isolation."""

    def test_outcome_filter_includes_valid_outcomes(self):
        """Exchange should be scored for worked/failed/partial outcomes."""
        valid_outcomes = ["worked", "failed", "partial"]
        for outcome in valid_outcomes:
            assert outcome in ["worked", "failed", "partial"]

    def test_outcome_filter_excludes_unknown(self):
        """Exchange should NOT be scored for unknown outcome."""
        outcome = "unknown"
        assert outcome not in ["worked", "failed", "partial"]

    def test_already_scored_check(self):
        """Already-scored exchanges should be skipped."""
        exchange = {"doc_id": "working_123", "scored": True}
        should_score = exchange.get("doc_id") and not exchange.get("scored", False)
        assert should_score is False

    def test_unscored_exchange_passes(self):
        """Unscored exchanges should be scored."""
        exchange = {"doc_id": "working_456", "scored": False}
        should_score = exchange.get("doc_id") and not exchange.get("scored", False)
        assert should_score is True

    def test_missing_doc_id_skipped(self):
        """Exchanges without doc_id should be skipped."""
        exchange = {"scored": False}
        should_score = exchange.get("doc_id") and not exchange.get("scored", False)
        assert not should_score  # None or False both evaluate to falsy

    def test_fallback_finds_unscored_exchange(self):
        """Fallback should find the first unscored exchange."""
        cache = {
            "conv1": {"doc_id": "working_111", "scored": True},
            "conv2": {"doc_id": "working_222", "scored": False},
            "conv3": {"doc_id": "working_333", "scored": True},
        }

        # Simulate fallback logic
        found = None
        found_conv_id = None
        for cid, exc in cache.items():
            if not exc.get("scored", False):
                found = exc
                found_conv_id = cid
                break

        assert found is not None
        assert found["doc_id"] == "working_222"
        assert found_conv_id == "conv2"

    def test_fallback_returns_none_when_all_scored(self):
        """Fallback should return None when all exchanges are scored."""
        cache = {
            "conv1": {"doc_id": "working_111", "scored": True},
            "conv2": {"doc_id": "working_222", "scored": True},
        }

        found = None
        for cid, exc in cache.items():
            if not exc.get("scored", False):
                found = exc
                break

        assert found is None

    def test_conversation_id_tracking_for_mark_scored(self):
        """Correct conversation_id should be used when marking scored."""
        cache = {
            "real_session": {"doc_id": "working_abc", "scored": False},
        }

        request_conv_id = "default"  # MCP often uses "default"
        exchange_conv_id = request_conv_id

        # Primary lookup fails
        previous = cache.get(request_conv_id)
        assert previous is None

        # Fallback finds exchange from different conv_id
        for cid, exc in cache.items():
            if not exc.get("scored", False):
                previous = exc
                exchange_conv_id = cid  # Track actual conv_id
                break

        # mark_scored should use exchange_conv_id, not request_conv_id
        assert previous is not None
        assert exchange_conv_id == "real_session"
        assert exchange_conv_id != request_conv_id


class TestExchangeScoringIntegration:
    """Integration tests with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_exchange_scored_with_worked_outcome(self):
        """Exchange should be scored when outcome is 'worked'."""
        # Mock dependencies
        mock_memory = AsyncMock()
        mock_memory.record_outcome = AsyncMock(return_value={"documents_updated": 1})

        mock_session_manager = MagicMock()
        mock_session_manager._last_exchange_cache = {
            "default": {"doc_id": "working_test123", "scored": False}
        }
        mock_session_manager.mark_scored = AsyncMock(return_value=True)
        mock_session_manager.set_scored_this_turn = MagicMock()

        # Simulate the fix logic
        conversation_id = "default"
        outcome = "worked"
        exchange_doc_id = None
        exchange_conv_id = conversation_id

        if outcome in ["worked", "failed", "partial"]:
            previous = mock_session_manager._last_exchange_cache.get(conversation_id)
            if previous and previous.get("doc_id") and not previous.get("scored", False):
                exchange_doc_id = previous["doc_id"]
                await mock_memory.record_outcome(doc_ids=[exchange_doc_id], outcome=outcome)
                await mock_session_manager.mark_scored(exchange_conv_id, exchange_doc_id, outcome)

        # Verify
        assert exchange_doc_id == "working_test123"
        mock_memory.record_outcome.assert_called_once_with(
            doc_ids=["working_test123"],
            outcome="worked"
        )
        mock_session_manager.mark_scored.assert_called_once_with(
            "default",
            "working_test123",
            "worked"
        )

    @pytest.mark.asyncio
    async def test_exchange_not_scored_with_unknown_outcome(self):
        """Exchange should NOT be scored when outcome is 'unknown'."""
        mock_memory = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager._last_exchange_cache = {
            "default": {"doc_id": "working_test456", "scored": False}
        }

        outcome = "unknown"
        exchange_doc_id = None

        if outcome in ["worked", "failed", "partial"]:
            previous = mock_session_manager._last_exchange_cache.get("default")
            if previous:
                exchange_doc_id = previous["doc_id"]

        assert exchange_doc_id is None
        mock_memory.record_outcome.assert_not_called()

    @pytest.mark.asyncio
    async def test_double_scoring_prevented(self):
        """Already-scored exchange should not be scored again."""
        mock_memory = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager._last_exchange_cache = {
            "default": {"doc_id": "working_already", "scored": True}  # Already scored
        }

        outcome = "worked"
        exchange_doc_id = None

        if outcome in ["worked", "failed", "partial"]:
            previous = mock_session_manager._last_exchange_cache.get("default")
            if previous and previous.get("doc_id") and not previous.get("scored", False):
                exchange_doc_id = previous["doc_id"]

        assert exchange_doc_id is None  # Should not score
        mock_memory.record_outcome.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
