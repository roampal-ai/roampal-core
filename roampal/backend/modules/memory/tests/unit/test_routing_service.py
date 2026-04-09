"""
Unit Tests for RoutingService (v0.4.5).

v0.4.5: KG-based routing removed. RoutingService now always returns all 5 collections.
Tests focus on query preprocessing (acronym expansion) and route_query always returning all.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest


class TestRoutingServiceInit:
    """Test RoutingService initialization."""

    def test_init_no_args(self):
        """Should initialize without any args."""
        from roampal.backend.modules.memory.routing_service import RoutingService
        service = RoutingService()
        assert service.config is not None

    def test_init_with_config(self):
        """Should accept optional config."""
        from roampal.backend.modules.memory.routing_service import RoutingService
        from roampal.backend.modules.memory.config import MemoryConfig
        config = MemoryConfig()
        service = RoutingService(config=config)
        assert service.config is config


class TestQueryPreprocessing:
    """Test query preprocessing with acronym expansion."""

    @pytest.fixture
    def service(self):
        from roampal.backend.modules.memory.routing_service import RoutingService
        return RoutingService()

    def test_expand_api_acronym(self, service):
        result = service.preprocess_query("How to use API")
        assert "application programming interface" in result.lower()
        assert "api" in result.lower()

    def test_expand_multiple_acronyms(self, service):
        result = service.preprocess_query("API and SDK docs")
        assert "application programming interface" in result.lower()
        assert "software development kit" in result.lower()

    def test_no_expansion_needed(self, service):
        result = service.preprocess_query("simple query")
        assert result == "simple query"

    def test_empty_query(self, service):
        result = service.preprocess_query("")
        assert result == ""

    def test_normalize_whitespace(self, service):
        result = service.preprocess_query("multiple   spaces   here")
        assert "  " not in result


class TestQueryRouting:
    """Test query routing — v0.4.5: always returns all collections."""

    @pytest.fixture
    def service(self):
        from roampal.backend.modules.memory.routing_service import RoutingService
        return RoutingService()

    def test_route_query_returns_all_collections(self, service):
        """v0.4.5: Should always return all 5 collections."""
        from roampal.backend.modules.memory.routing_service import ALL_COLLECTIONS
        result = service.route_query("any query")
        assert set(result) == set(ALL_COLLECTIONS)

    def test_route_query_returns_copy(self, service):
        """Should return a copy, not the original list."""
        result1 = service.route_query("query")
        result2 = service.route_query("query")
        assert result1 is not result2

    def test_route_empty_query(self, service):
        """Empty query should still return all collections."""
        from roampal.backend.modules.memory.routing_service import ALL_COLLECTIONS
        result = service.route_query("")
        assert set(result) == set(ALL_COLLECTIONS)


class TestAcronymDictionary:
    """Test acronym dictionary completeness."""

    def test_common_tech_acronyms(self):
        from roampal.backend.modules.memory.routing_service import RoutingService
        assert "api" in RoutingService.ACRONYM_DICT
        assert "sdk" in RoutingService.ACRONYM_DICT
        assert "ui" in RoutingService.ACRONYM_DICT
        assert "ml" in RoutingService.ACRONYM_DICT
        assert "ai" in RoutingService.ACRONYM_DICT

    def test_reverse_mapping(self):
        from roampal.backend.modules.memory.routing_service import RoutingService
        assert len(RoutingService.EXPANSION_TO_ACRONYM) > 0
        assert "application programming interface" in RoutingService.EXPANSION_TO_ACRONYM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
