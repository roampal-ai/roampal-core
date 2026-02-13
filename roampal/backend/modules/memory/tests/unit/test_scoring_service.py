"""
Unit Tests for ScoringService

Tests the extracted scoring logic to ensure it matches the original behavior.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import json
import pytest

from roampal.backend.modules.memory.scoring_service import ScoringService, wilson_score_lower
from roampal.backend.modules.memory.config import MemoryConfig


class TestWilsonScoreLower:
    """Test the Wilson score function."""

    def test_zero_uses_returns_neutral(self):
        """Zero uses should return 0.5 (neutral)."""
        score = wilson_score_lower(0, 0)
        assert score == 0.5

    def test_perfect_record_low_samples(self):
        """1/1 should have lower score than proven record."""
        new_score = wilson_score_lower(1, 1)
        proven_score = wilson_score_lower(90, 100)

        # Proven should beat newcomer
        assert proven_score > new_score
        # New score should be around 0.2
        assert 0.1 < new_score < 0.4
        # Proven score should be around 0.84
        assert 0.8 < proven_score < 0.9

    def test_score_range(self):
        """Score should always be between 0 and 1."""
        test_cases = [
            (0, 1), (1, 1), (5, 10), (50, 100),
            (99, 100), (100, 100), (0, 100),
        ]

        for successes, total in test_cases:
            score = wilson_score_lower(successes, total)
            assert 0 <= score <= 1, f"Score {score} out of range for {successes}/{total}"

    def test_monotonic_increase(self):
        """Higher success rate should give higher score (same sample size)."""
        scores = [wilson_score_lower(i, 10) for i in range(11)]

        for i in range(1, len(scores)):
            assert scores[i] >= scores[i-1], \
                f"Score should increase: {scores[i-1]} -> {scores[i]}"

    def test_sample_size_effect(self):
        """Same success rate with more samples should give higher score."""
        score_10 = wilson_score_lower(5, 10)
        score_100 = wilson_score_lower(50, 100)
        score_1000 = wilson_score_lower(500, 1000)

        assert score_100 > score_10
        assert score_1000 > score_100

    def test_matches_expected_values(self):
        """Should match expected mathematical values from Wilson formula."""
        # Values verified against original unified_memory_system.py implementation
        expected = {
            (0, 0): 0.5,
            (1, 1): 0.2065,
            (90, 100): 0.8256,  # Actual output from Wilson formula
            (50, 100): 0.4038,
        }

        for (successes, total), expected_approx in expected.items():
            new_score = wilson_score_lower(successes, total)
            assert abs(new_score - expected_approx) < 0.01, \
                f"Mismatch for {successes}/{total}: got {new_score}, expected ~{expected_approx}"


class TestScoringService:
    """Test the ScoringService class."""

    @pytest.fixture
    def service(self):
        return ScoringService()

    @pytest.fixture
    def custom_config_service(self):
        config = MemoryConfig(
            embedding_weight_proven=0.3,
            learned_weight_proven=0.7,
        )
        return ScoringService(config)

    def test_default_config(self, service):
        """Should use default config if none provided."""
        assert service.config.high_value_threshold == 0.9
        assert service.config.promotion_score_threshold == 0.7

    def test_custom_config(self, custom_config_service):
        """Should use custom config if provided."""
        assert custom_config_service.config.embedding_weight_proven == 0.3
        assert custom_config_service.config.learned_weight_proven == 0.7

    def test_count_successes_empty(self, service):
        """Empty history should return 0."""
        assert service.count_successes_from_history("") == 0
        assert service.count_successes_from_history("[]") == 0

    def test_count_successes_worked(self, service):
        """Worked outcomes count as 1."""
        history = json.dumps([{"outcome": "worked"}])
        assert service.count_successes_from_history(history) == 1.0

    def test_count_successes_partial(self, service):
        """Partial outcomes count as 0.5."""
        history = json.dumps([{"outcome": "partial"}])
        assert service.count_successes_from_history(history) == 0.5

    def test_count_successes_mixed(self, service):
        """Mixed outcomes should sum correctly."""
        history = json.dumps([
            {"outcome": "worked"},
            {"outcome": "partial"},
            {"outcome": "failed"},
            {"outcome": "worked"},
        ])
        assert service.count_successes_from_history(history) == 2.5

    def test_count_successes_invalid_json(self, service):
        """Invalid JSON should return 0."""
        assert service.count_successes_from_history("invalid") == 0

    def test_calculate_learned_score_no_uses(self, service):
        """No uses should return raw score."""
        learned, wilson = service.calculate_learned_score(0.7, 0)
        assert learned == 0.7
        assert wilson == 0.5

    def test_calculate_learned_score_with_history(self, service):
        """Should calculate from outcome history."""
        history = json.dumps([
            {"outcome": "worked"},
            {"outcome": "worked"},
            {"outcome": "partial"},
        ])
        learned, wilson = service.calculate_learned_score(0.5, 3, history)
        expected_wilson = wilson_score_lower(2.5, 3)
        assert abs(wilson - expected_wilson) < 0.01

    def test_calculate_learned_score_with_success_count(self, service):
        """v0.2.8: Should use success_count when provided (bypasses 10-entry cap)."""
        # Simulate 100 uses with 90 worked (90% success rate)
        # Old behavior: outcome_history capped at 10 entries would break Wilson
        # New behavior: success_count=90.0 is used directly
        learned, wilson = service.calculate_learned_score(
            raw_score=0.5,
            uses=100,
            outcome_history="[]",  # Empty history - would fail without success_count
            success_count=90.0
        )
        expected_wilson = wilson_score_lower(90, 100)
        assert abs(wilson - expected_wilson) < 0.01
        assert wilson > 0.8  # 90% success with 100 uses should be high confidence

    def test_success_count_overrides_history(self, service):
        """v0.2.8: success_count should take precedence over outcome_history."""
        # History says 1 success (10 worked out of 10)
        # But success_count says 50 successes
        history = json.dumps([{"outcome": "worked"} for _ in range(10)])
        learned, wilson = service.calculate_learned_score(
            raw_score=0.5,
            uses=100,
            outcome_history=history,  # Would give 10 successes
            success_count=50.0  # Should use this instead
        )
        expected_wilson = wilson_score_lower(50, 100)
        assert abs(wilson - expected_wilson) < 0.01

    def test_dynamic_weights_proven(self, service):
        """Proven memories should have high learned weight."""
        emb_w, learn_w = service.get_dynamic_weights(5, 0.85, "history")
        assert emb_w == 0.2
        assert learn_w == 0.8

    def test_dynamic_weights_established(self, service):
        """Established memories should have moderately high learned weight."""
        emb_w, learn_w = service.get_dynamic_weights(3, 0.75, "history")
        assert emb_w == 0.25
        assert learn_w == 0.75

    def test_dynamic_weights_new(self, service):
        """New memories should favor embedding."""
        emb_w, learn_w = service.get_dynamic_weights(0, 0.5, "history")
        assert emb_w > learn_w

    def test_dynamic_weights_failing(self, service):
        """Failing memories should heavily favor embedding."""
        emb_w, learn_w = service.get_dynamic_weights(3, 0.3, "history")
        assert emb_w == 0.7
        assert learn_w == 0.3

    def test_dynamic_weights_memory_bank_high_quality(self, service):
        """High quality memory_bank should use quality-based weights."""
        emb_w, learn_w = service.get_dynamic_weights(
            0, 0.5, "memory_bank", importance=0.95, confidence=0.9
        )
        assert emb_w == 0.45
        assert learn_w == 0.55

    def test_dynamic_weights_memory_bank_normal(self, service):
        """Normal memory_bank should be balanced."""
        emb_w, learn_w = service.get_dynamic_weights(
            0, 0.5, "memory_bank", importance=0.7, confidence=0.7
        )
        assert emb_w == 0.5
        assert learn_w == 0.5

    def test_calculate_final_score_basic(self, service):
        """Should calculate final score correctly."""
        metadata = {"score": 0.7, "uses": 0}
        result = service.calculate_final_score(metadata, distance=1.0, collection="history")

        assert "final_rank_score" in result
        assert "wilson_score" in result
        assert "embedding_similarity" in result
        assert "learned_score" in result
        assert 0 <= result["final_rank_score"] <= 1

    def test_calculate_final_score_distance_conversion(self, service):
        """Distance should be converted to similarity correctly."""
        metadata = {"score": 0.5, "uses": 0}

        result0 = service.calculate_final_score(metadata, distance=0.0, collection="history")
        assert result0["embedding_similarity"] == 1.0

        result1 = service.calculate_final_score(metadata, distance=1.0, collection="history")
        assert result1["embedding_similarity"] == 0.5

        result10 = service.calculate_final_score(metadata, distance=10.0, collection="history")
        assert result10["embedding_similarity"] < result1["embedding_similarity"]

    def test_apply_scoring_to_results(self, service):
        """Should apply scoring and sort results."""
        results = [
            {"metadata": {"score": 0.3, "uses": 0}, "distance": 2.0, "collection": "history"},
            {"metadata": {"score": 0.9, "uses": 5}, "distance": 0.5, "collection": "history"},
            {"metadata": {"score": 0.5, "uses": 2}, "distance": 1.0, "collection": "history"},
        ]

        scored = service.apply_scoring_to_results(results)

        scores = [r["final_rank_score"] for r in scored]
        assert scores == sorted(scores, reverse=True)

        for r in scored:
            assert "final_rank_score" in r
            assert "wilson_score" in r
            assert "embedding_weight" in r


class TestScoringConsistency:
    """Test that new scoring matches original behavior."""

    @pytest.fixture
    def service(self):
        return ScoringService()

    def test_memory_bank_quality_scoring_cold_start(self, service):
        """Memory bank with < 3 uses should use pure importance*confidence."""
        metadata = {
            "score": 0.5,
            "uses": 0,
            "importance": 0.9,
            "confidence": 0.8,
        }
        result = service.calculate_final_score(metadata, distance=0.5, collection="memory_bank")
        # Pure quality: 0.9 * 0.8 = 0.72
        assert abs(result["learned_score"] - 0.72) < 0.01

    def test_memory_bank_quality_scoring_under_threshold(self, service):
        """Memory bank with 1-2 uses should still use pure quality (cold start protection)."""
        metadata = {
            "score": 0.5,
            "uses": 2,
            "importance": 0.9,
            "confidence": 0.8,
            "success_count": 2.0,
            "outcome_history": "[]",
        }
        result = service.calculate_final_score(metadata, distance=0.5, collection="memory_bank")
        # Still pure quality below 3 uses: 0.9 * 0.8 = 0.72
        assert abs(result["learned_score"] - 0.72) < 0.01

    def test_memory_bank_wilson_blend_after_3_uses(self, service):
        """v0.3.6: Memory bank should blend 50% quality + 50% Wilson after 3+ uses."""
        metadata = {
            "score": 0.5,
            "uses": 10,
            "importance": 0.9,
            "confidence": 0.8,
            "success_count": 8.0,
            "outcome_history": "[]",
        }
        result = service.calculate_final_score(metadata, distance=0.5, collection="memory_bank")
        quality = 0.9 * 0.8  # 0.72
        wilson = result["wilson_score"]
        expected = 0.5 * quality + 0.5 * wilson
        # Verify the 50/50 blend formula (v0.3.6)
        assert abs(result["learned_score"] - expected) < 0.01
        # Verify Wilson actually influenced the score (not pure quality)
        assert abs(result["learned_score"] - quality) > 0.001

    def test_memory_bank_wilson_blend_at_threshold(self, service):
        """v0.3.6: Memory bank should start blending 50/50 at exactly 3 uses."""
        metadata = {
            "score": 0.5,
            "uses": 3,
            "importance": 0.7,
            "confidence": 0.7,
            "success_count": 3.0,
            "outcome_history": "[]",
        }
        result = service.calculate_final_score(metadata, distance=0.5, collection="memory_bank")
        quality = 0.7 * 0.7  # 0.49
        wilson = result["wilson_score"]
        expected = 0.5 * quality + 0.5 * wilson
        assert abs(result["learned_score"] - expected) < 0.01

    def test_memory_bank_wilson_blend_low_success(self, service):
        """Memory bank with low success rate should rank lower via Wilson blend."""
        # High quality memory that keeps failing
        metadata = {
            "score": 0.5,
            "uses": 10,
            "importance": 0.9,
            "confidence": 0.9,
            "success_count": 1.0,  # Only 1/10 worked
            "outcome_history": "[]",
        }
        result = service.calculate_final_score(metadata, distance=0.5, collection="memory_bank")
        quality = 0.9 * 0.9  # 0.81
        # Wilson should pull the score down from 0.81
        assert result["learned_score"] < quality


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
