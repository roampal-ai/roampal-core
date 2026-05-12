"""Tests for SessionManager._cleanup_completion_state.

v0.5.7: Core-side GC for `_completion_state.json`. Without this prune,
the state file accumulates one entry per conversation_id ever seen, which
both inflates per-write I/O and poisons the cross-session scoring fallback
in `server/main.py` with stale `scored_this_turn=True` flags.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_entry(ts: datetime, **extras) -> dict:
    """Build a state entry with first_message_timestamp set to `ts`."""
    entry = {
        "first_message_seen": True,
        "first_message_timestamp": ts.isoformat(),
        "timestamp": ts.isoformat(),
    }
    entry.update(extras)
    return entry


def _safe_id(conversation_id: str) -> str:
    """Mirror SessionManager._get_session_file's sanitization."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in conversation_id)


def _write_state_and_jsonls(
    data_path: Path,
    entries: dict,
    *,
    create_jsonl_for: set | None = None,
) -> Path:
    """Seed `_completion_state.json` and matching JSONL transcripts.

    By default a JSONL is created for every entry. Pass `create_jsonl_for`
    to restrict which conversation_ids get a transcript on disk.
    """
    sessions_dir = data_path / "mcp_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    state_file = sessions_dir / "_completion_state.json"
    state_file.write_text(json.dumps(entries))

    create_for = entries.keys() if create_jsonl_for is None else create_jsonl_for
    for conv_id in create_for:
        (sessions_dir / f"{_safe_id(conv_id)}.jsonl").write_text("")

    return state_file


class TestCleanupCompletionState:
    """v0.5.7 Item 1: prune stale entries from `_completion_state.json`."""

    def test_prunes_stale_first_message_timestamp(self, tmp_path):
        """Entries with first_message_timestamp older than max_age_days are dropped.

        v0.5.7 default TTL is 30 days (matched to the JSONL transcript TTL).
        """
        from roampal.hooks.session_manager import SessionManager

        now = datetime.now()
        entries = {
            "fresh": _make_entry(now - timedelta(days=1)),
            "stale_45d": _make_entry(now - timedelta(days=45)),
            "stale_90d": _make_entry(now - timedelta(days=90)),
        }
        state_file = _write_state_and_jsonls(tmp_path, entries)

        sm = SessionManager(tmp_path)

        final = json.loads(state_file.read_text())
        assert "fresh" in final
        assert "stale_45d" not in final
        assert "stale_90d" not in final
        assert len(final) == 1

        # Sanity: the SessionManager object exists (avoid unused-var noise).
        assert sm.sessions_dir == tmp_path / "mcp_sessions"

    def test_prunes_entries_with_missing_jsonl(self, tmp_path):
        """Entries whose <conversation_id>.jsonl is missing are dropped, even if fresh."""
        from roampal.hooks.session_manager import SessionManager

        now = datetime.now()
        entries = {
            "has_jsonl": _make_entry(now - timedelta(hours=1)),
            "no_jsonl": _make_entry(now - timedelta(hours=1)),
        }
        # Only create the JSONL for one of them
        state_file = _write_state_and_jsonls(
            tmp_path, entries, create_jsonl_for={"has_jsonl"}
        )

        SessionManager(tmp_path)

        final = json.loads(state_file.read_text())
        assert "has_jsonl" in final
        assert "no_jsonl" not in final

    def test_max_entries_ceiling_after_age_pass(self, tmp_path):
        """When too many fresh entries survive the age pass, evict oldest by timestamp."""
        from roampal.hooks.session_manager import SessionManager

        now = datetime.now()
        # 600 fresh entries, each with progressively newer timestamps.
        # max_entries default is 500 → 100 oldest should be evicted.
        entries = {}
        for i in range(600):
            conv_id = f"conv_{i:04d}"
            # i=0 is oldest (5 days ago), i=599 is newest (a few seconds ago)
            ts = now - timedelta(days=5) + timedelta(minutes=i)
            entries[conv_id] = _make_entry(ts)

        state_file = _write_state_and_jsonls(tmp_path, entries)

        SessionManager(tmp_path)

        final = json.loads(state_file.read_text())
        assert len(final) == 500
        # Newest entry must survive
        assert "conv_0599" in final
        # Oldest entry must be evicted
        assert "conv_0000" not in final
        # Boundary: keep indices 100..599 (last 500)
        assert "conv_0100" in final
        assert "conv_0099" not in final

    def test_atomic_write_preserves_valid_json_on_failure(self, tmp_path):
        """If the atomic write fails mid-prune, original file is left valid JSON."""
        from roampal.hooks.session_manager import SessionManager

        now = datetime.now()
        entries = {
            "fresh": _make_entry(now - timedelta(days=1)),
            "stale": _make_entry(now - timedelta(days=60)),
        }
        state_file = _write_state_and_jsonls(tmp_path, entries)
        original_bytes = state_file.read_bytes()

        # Force write_json_atomic to fail. write_json_atomic itself unlinks
        # the temp file on exception, so the original file must remain
        # byte-for-byte unchanged.
        with patch(
            "roampal.utils.atomic_json.os.replace",
            side_effect=OSError("simulated disk error"),
        ):
            SessionManager(tmp_path)

        # File still exists and parses as valid JSON
        assert state_file.exists()
        loaded = json.loads(state_file.read_text())
        assert isinstance(loaded, dict)

        # Original content untouched (atomic write contract)
        assert state_file.read_bytes() == original_bytes

        # No stray .tmp files left behind in sessions_dir
        tmp_leftovers = list((tmp_path / "mcp_sessions").glob("*.tmp"))
        assert tmp_leftovers == [], f"Leftover tmp files: {tmp_leftovers}"
