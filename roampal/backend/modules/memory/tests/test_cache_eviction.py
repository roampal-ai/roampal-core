"""
Tests for cache TTL eviction — _evict_stale_entries() in server/main.py.

Covers:
- Fresh entries are NOT evicted
- Stale entries (>30 min old) ARE evicted
- Entries with invalid timestamps are evicted
- Mixed fresh/stale entries only evict stale ones
"""

from datetime import datetime, timedelta

import pytest


# Import the function and globals under test
from roampal.server.main import (
    _evict_stale_entries,
    _search_cache,
    _injection_map,
    _CACHE_TTL_SECONDS,
)


@pytest.fixture(autouse=True)
def clean_caches():
    """Clear both caches before and after each test."""
    _search_cache.clear()
    _injection_map.clear()
    yield
    _search_cache.clear()
    _injection_map.clear()


class TestFreshEntriesNotEvicted:
    """Fresh entries (< 30 min old) should survive eviction."""

    def test_fresh_search_cache_entry_survives(self):
        now = datetime.now().isoformat()
        _search_cache["session_1"] = {
            "doc_ids": ["doc_a"],
            "query": "test",
            "timestamp": now,
        }

        _evict_stale_entries()

        assert "session_1" in _search_cache

    def test_fresh_injection_map_entry_survives(self):
        now = datetime.now().isoformat()
        _injection_map["doc_a"] = {
            "conversation_id": "session_1",
            "injected_at": now,
            "query": "test",
        }

        _evict_stale_entries()

        assert "doc_a" in _injection_map

    def test_entry_at_exactly_ttl_boundary_survives(self):
        """Entry at exactly 30 minutes should NOT be evicted (> check, not >=)."""
        boundary = (datetime.now() - timedelta(seconds=_CACHE_TTL_SECONDS)).isoformat()
        _search_cache["boundary"] = {
            "doc_ids": ["doc_b"],
            "timestamp": boundary,
        }

        _evict_stale_entries()

        # At exactly TTL, (now - entry_time) == TTL, which is NOT > TTL
        assert "boundary" in _search_cache


class TestStaleEntriesEvicted:
    """Entries older than 30 minutes should be evicted."""

    def test_stale_search_cache_entry_evicted(self):
        old_time = (datetime.now() - timedelta(seconds=_CACHE_TTL_SECONDS + 1)).isoformat()
        _search_cache["old_session"] = {
            "doc_ids": ["doc_old"],
            "query": "old query",
            "timestamp": old_time,
        }

        _evict_stale_entries()

        assert "old_session" not in _search_cache

    def test_stale_injection_map_entry_evicted(self):
        old_time = (datetime.now() - timedelta(seconds=_CACHE_TTL_SECONDS + 1)).isoformat()
        _injection_map["doc_old"] = {
            "conversation_id": "old_session",
            "injected_at": old_time,
            "query": "old query",
        }

        _evict_stale_entries()

        assert "doc_old" not in _injection_map

    def test_very_old_entry_evicted(self):
        """Entry from 24 hours ago should definitely be evicted."""
        ancient = (datetime.now() - timedelta(hours=24)).isoformat()
        _search_cache["ancient"] = {
            "doc_ids": ["doc_ancient"],
            "timestamp": ancient,
        }

        _evict_stale_entries()

        assert "ancient" not in _search_cache


class TestInvalidTimestampsEvicted:
    """Entries with unparseable timestamps should be evicted."""

    def test_garbage_timestamp_evicted_from_search_cache(self):
        _search_cache["bad_ts"] = {
            "doc_ids": ["doc_x"],
            "timestamp": "not-a-date",
        }

        _evict_stale_entries()

        assert "bad_ts" not in _search_cache

    def test_garbage_timestamp_evicted_from_injection_map(self):
        _injection_map["doc_bad"] = {
            "conversation_id": "ses_1",
            "injected_at": "garbage",
        }

        _evict_stale_entries()

        assert "doc_bad" not in _injection_map

    def test_empty_timestamp_not_evicted(self):
        """Empty string timestamp: the code checks `if ts:` — empty string is falsy, so it skips.
        These entries are NOT evicted (they just don't have a parseable timestamp to check)."""
        _search_cache["no_ts"] = {
            "doc_ids": ["doc_y"],
            "timestamp": "",
        }
        _injection_map["doc_no_ts"] = {
            "conversation_id": "ses_2",
            "injected_at": "",
        }

        _evict_stale_entries()

        # Empty string is falsy, so `if ts:` is False, entry is NOT added to stale list
        assert "no_ts" in _search_cache
        assert "doc_no_ts" in _injection_map

    def test_missing_timestamp_key_not_evicted(self):
        """Missing timestamp key: entry.get('timestamp', '') returns '', which is falsy."""
        _search_cache["missing_key"] = {
            "doc_ids": ["doc_z"],
        }

        _evict_stale_entries()

        assert "missing_key" in _search_cache

    def test_none_timestamp_type_error_evicted(self):
        """None as timestamp: `if ts:` is False for None, so it is NOT evicted."""
        _injection_map["doc_none"] = {
            "conversation_id": "ses_3",
            "injected_at": None,
        }

        _evict_stale_entries()

        # None is falsy, so `if ts:` skips it — NOT evicted
        assert "doc_none" in _injection_map


class TestMixedFreshAndStale:
    """Only stale entries should be evicted; fresh ones should remain."""

    def test_mixed_search_cache(self):
        now = datetime.now()
        fresh_ts = now.isoformat()
        stale_ts = (now - timedelta(seconds=_CACHE_TTL_SECONDS + 60)).isoformat()

        _search_cache["fresh_session"] = {
            "doc_ids": ["doc_fresh"],
            "timestamp": fresh_ts,
        }
        _search_cache["stale_session"] = {
            "doc_ids": ["doc_stale"],
            "timestamp": stale_ts,
        }

        _evict_stale_entries()

        assert "fresh_session" in _search_cache
        assert "stale_session" not in _search_cache

    def test_mixed_injection_map(self):
        now = datetime.now()
        fresh_ts = now.isoformat()
        stale_ts = (now - timedelta(seconds=_CACHE_TTL_SECONDS + 60)).isoformat()

        _injection_map["doc_fresh"] = {
            "conversation_id": "ses_a",
            "injected_at": fresh_ts,
        }
        _injection_map["doc_stale"] = {
            "conversation_id": "ses_b",
            "injected_at": stale_ts,
        }

        _evict_stale_entries()

        assert "doc_fresh" in _injection_map
        assert "doc_stale" not in _injection_map

    def test_mixed_with_invalid_timestamp(self):
        """Invalid timestamp entries should be evicted alongside stale ones."""
        now = datetime.now()
        fresh_ts = now.isoformat()

        _search_cache["fresh"] = {"doc_ids": ["a"], "timestamp": fresh_ts}
        _search_cache["invalid"] = {"doc_ids": ["b"], "timestamp": "bad"}
        _search_cache["stale"] = {
            "doc_ids": ["c"],
            "timestamp": (now - timedelta(hours=1)).isoformat(),
        }

        _evict_stale_entries()

        assert "fresh" in _search_cache
        assert "invalid" not in _search_cache
        assert "stale" not in _search_cache

    def test_many_entries_partial_eviction(self):
        """Bulk test: 5 fresh + 5 stale = 5 remain."""
        now = datetime.now()
        for i in range(5):
            _search_cache[f"fresh_{i}"] = {
                "doc_ids": [f"doc_{i}"],
                "timestamp": now.isoformat(),
            }
        for i in range(5):
            _search_cache[f"stale_{i}"] = {
                "doc_ids": [f"doc_s_{i}"],
                "timestamp": (now - timedelta(hours=1)).isoformat(),
            }

        _evict_stale_entries()

        assert len(_search_cache) == 5
        for i in range(5):
            assert f"fresh_{i}" in _search_cache
            assert f"stale_{i}" not in _search_cache
