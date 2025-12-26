"""
Unit Tests for RoutingService.

Tests intelligent collection routing and query preprocessing.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
from unittest.mock import MagicMock


class TestRoutingServiceInit:
    """Test RoutingService initialization."""

    def test_init_with_kg_service(self):
        """Should initialize with KG service."""
        from roampal.backend.modules.memory.routing_service import RoutingService

        kg_service = MagicMock()
        kg_service.knowledge_graph = {}

        service = RoutingService(kg_service=kg_service)

        assert service.kg_service == kg_service


class TestQueryPreprocessing:
    """Test query preprocessing with acronym expansion."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        from roampal.backend.modules.memory.routing_service import RoutingService

        kg_service = MagicMock()
        kg_service.knowledge_graph = {}

        return RoutingService(kg_service=kg_service)

    def test_expand_api_acronym(self, service):
        """Should expand API to full form."""
        result = service.preprocess_query("How to use API")

        assert "application programming interface" in result.lower()
        assert "api" in result.lower()  # Original preserved

    def test_expand_multiple_acronyms(self, service):
        """Should expand multiple acronyms."""
        result = service.preprocess_query("API and SDK docs")

        assert "application programming interface" in result.lower()
        assert "software development kit" in result.lower()

    def test_no_expansion_needed(self, service):
        """Should return unchanged if no acronyms."""
        result = service.preprocess_query("simple query")

        assert result == "simple query"

    def test_empty_query(self, service):
        """Empty query should return empty."""
        result = service.preprocess_query("")

        assert result == ""

    def test_normalize_whitespace(self, service):
        """Should normalize whitespace."""
        result = service.preprocess_query("multiple   spaces   here")

        assert "  " not in result


class TestTierScoreCalculation:
    """Test tier score calculation."""

    @pytest.fixture
    def service(self):
        """Create service with routing patterns."""
        from roampal.backend.modules.memory.routing_service import RoutingService

        kg_service = MagicMock()
        kg_service.knowledge_graph = {
            "routing_patterns": {
                "python": {
                    "collections_used": {
                        "patterns": {"successes": 8, "failures": 2, "partials": 1, "total": 11},
                        "history": {"successes": 3, "failures": 2, "partials": 0, "total": 5}
                    }
                }
            }
        }

        return RoutingService(kg_service=kg_service)

    def test_calculate_tier_scores(self, service):
        """Should calculate scores for collections."""
        scores = service.calculate_tier_scores(["python"])

        assert "patterns" in scores
        assert "history" in scores
        assert "working" in scores
        assert scores["patterns"] > 0
        assert scores["patterns"] > scores["history"]  # More successful

    def test_no_patterns_returns_zeros(self, service):
        """Unknown concepts should return zero scores."""
        scores = service.calculate_tier_scores(["unknown_concept"])

        assert all(s == 0.0 for s in scores.values())

    def test_empty_concepts_returns_zeros(self, service):
        """Empty concepts should return zero scores."""
        scores = service.calculate_tier_scores([])

        assert all(s == 0.0 for s in scores.values())


class TestQueryRouting:
    """Test query routing logic."""

    @pytest.fixture
    def service_exploration(self):
        """Create service in exploration phase (no patterns)."""
        from roampal.backend.modules.memory.routing_service import RoutingService

        kg_service = MagicMock()
        kg_service.knowledge_graph = {"routing_patterns": {}}
        kg_service.extract_concepts.return_value = ["unknown"]

        return RoutingService(kg_service=kg_service)

    @pytest.fixture
    def service_confident(self):
        """Create service with high confidence patterns."""
        from roampal.backend.modules.memory.routing_service import RoutingService

        kg_service = MagicMock()
        kg_service.knowledge_graph = {
            "routing_patterns": {
                "python": {
                    "collections_used": {
                        "patterns": {"successes": 30, "failures": 2, "partials": 1, "total": 33},
                    },
                    "last_used": ""
                }
            }
        }
        kg_service.extract_concepts.return_value = ["python"]

        return RoutingService(kg_service=kg_service)

    def test_exploration_searches_all(self, service_exploration):
        """Exploration phase should search all collections."""
        from roampal.backend.modules.memory.routing_service import ALL_COLLECTIONS

        result = service_exploration.route_query("unknown query")

        assert set(result) == set(ALL_COLLECTIONS)

    def test_confident_narrows_search(self, service_confident):
        """High confidence should narrow to top collections."""
        result = service_confident.route_query("python question")

        # Should include patterns (high confidence there)
        assert "patterns" in result
        # Should not include all 5
        assert len(result) < 5


class TestTierRecommendations:
    """Test tier recommendations for insights."""

    @pytest.fixture
    def service(self):
        """Create service."""
        from roampal.backend.modules.memory.routing_service import RoutingService

        kg_service = MagicMock()
        kg_service.knowledge_graph = {
            "routing_patterns": {
                "python": {
                    "collections_used": {
                        "patterns": {"successes": 10, "failures": 0, "partials": 0, "total": 10}
                    }
                }
            }
        }

        return RoutingService(kg_service=kg_service)

    def test_get_tier_recommendations_structure(self, service):
        """Should return expected structure."""
        result = service.get_tier_recommendations(["python"])

        assert "top_collections" in result
        assert "match_count" in result
        assert "confidence_level" in result

    def test_no_concepts_exploration(self, service):
        """No concepts should return exploration level."""
        result = service.get_tier_recommendations([])

        assert result["confidence_level"] == "exploration"
        assert len(result["top_collections"]) == 5  # All collections


class TestAcronymDictionary:
    """Test acronym dictionary completeness."""

    def test_common_tech_acronyms(self):
        """Should have common tech acronyms."""
        from roampal.backend.modules.memory.routing_service import RoutingService

        assert "api" in RoutingService.ACRONYM_DICT
        assert "sdk" in RoutingService.ACRONYM_DICT
        assert "ui" in RoutingService.ACRONYM_DICT
        assert "ml" in RoutingService.ACRONYM_DICT
        assert "ai" in RoutingService.ACRONYM_DICT

    def test_reverse_mapping(self):
        """Should have reverse mapping for bidirectional matching."""
        from roampal.backend.modules.memory.routing_service import RoutingService

        assert len(RoutingService.EXPANSION_TO_ACRONYM) > 0
        assert "application programming interface" in RoutingService.EXPANSION_TO_ACRONYM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
