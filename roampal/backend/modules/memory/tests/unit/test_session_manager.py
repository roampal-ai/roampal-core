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


class TestMarkScoredAtomicRewrite:
    """v0.5.8 Item 2: atomic transcript rewrite in SessionManager.mark_scored()."""

    def _seed_session(self, data_path: Path, conversation_id: str, lines: list[str]) -> None:
        """Create a session JSONL file with the given lines."""
        sessions_dir = data_path / "mcp_sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        safe_id = _safe_id(conversation_id)
        (sessions_dir / f"{safe_id}.jsonl").write_text("".join(lines))

    async def test_happy_path_marks_last_matching_record(self, tmp_path):
        """mark_scored updates the last matching assistant record and returns True."""
        from roampal.hooks.session_manager import SessionManager

        lines = [
            json.dumps({"role": "user", "content": "hello"}) + "\n",
            json.dumps({"role": "assistant", "doc_id": "doc_1", "content": "hi back"}) + "\n",
        ]
        self._seed_session(tmp_path, "conv1", lines)

        sm = SessionManager(tmp_path)
        result = await sm.mark_scored("conv1", "doc_1", "worked")

        assert result is True
        # Verify the assistant record was updated
        records = [json.loads(line.strip()) for line in open(tmp_path / "mcp_sessions" / "conv1.jsonl")]
        assistant_record = [r for r in records if r.get("role") == "assistant"][0]
        assert assistant_record["scored"] is True
        assert assistant_record["outcome"] == "worked"
        # User record untouched
        user_record = [r for r in records if r.get("role") == "user"][0]
        assert "scored" not in user_record

    async def test_no_tmp_residue_on_success(self, tmp_path):
        """After a successful mark_scored, no .tmp files are left behind."""
        from roampal.hooks.session_manager import SessionManager

        lines = [
            json.dumps({"role": "user", "content": "hello"}) + "\n",
            json.dumps({"role": "assistant", "doc_id": "doc_1", "content": "hi back"}) + "\n",
        ]
        self._seed_session(tmp_path, "conv2", lines)

        sm = SessionManager(tmp_path)
        await sm.mark_scored("conv2", "doc_1", "partial")

        tmp_leftovers = list((tmp_path / "mcp_sessions").glob("*.tmp"))
        assert tmp_leftovers == [], f"Leftover tmp files: {tmp_leftovers}"

    async def test_replace_failure_preserves_original(self, tmp_path):
        """If os.replace fails mid-atomic-write, original file is byte-for-byte intact."""
        from roampal.hooks.session_manager import SessionManager

        lines = [
            json.dumps({"role": "user", "content": "hello"}) + "\n",
            json.dumps({"role": "assistant", "doc_id": "doc_1", "content": "hi back"}) + "\n",
        ]
        self._seed_session(tmp_path, "conv3", lines)

        sm = SessionManager(tmp_path)
        original_bytes = (tmp_path / "mcp_sessions" / "conv3.jsonl").read_bytes()

        with patch(
            "roampal.hooks.session_manager.os.replace",
            side_effect=OSError("simulated disk error"),
        ):
            result = await sm.mark_scored("conv3", "doc_1", "worked")

        assert result is False
        # Original file unchanged (atomic write contract)
        assert (tmp_path / "mcp_sessions" / "conv3.jsonl").read_bytes() == original_bytes
        # No stray .tmp files left behind
        tmp_leftovers = list((tmp_path / "mcp_sessions").glob("*.tmp"))
        assert tmp_leftovers == [], f"Leftover tmp files: {tmp_leftovers}"

    async def test_write_failure_cleans_tmp(self, tmp_path):
        """If the write phase fails (not replace), original file is intact and tmp is cleaned."""
        from roampal.hooks.session_manager import SessionManager

        lines = [
            json.dumps({"role": "user", "content": "hello"}) + "\n",
            json.dumps({"role": "assistant", "doc_id": "doc_1", "content": "hi back"}) + "\n",
        ]
        self._seed_session(tmp_path, "conv4", lines)

        sm = SessionManager(tmp_path)
        original_bytes = (tmp_path / "mcp_sessions" / "conv4.jsonl").read_bytes()

        with patch(
            "roampal.hooks.session_manager.os.fdopen",
            side_effect=OSError("simulated write error"),
        ):
            result = await sm.mark_scored("conv4", "doc_1", "worked")

        assert result is False
        # Original file unchanged
        assert (tmp_path / "mcp_sessions" / "conv4.jsonl").read_bytes() == original_bytes
        # No stray .tmp files left behind (inner cleanup ran)
        tmp_leftovers = list((tmp_path / "mcp_sessions").glob("*.tmp"))
        assert tmp_leftovers == [], f"Leftover tmp files: {tmp_leftovers}"

    async def test_missing_session_file_returns_false(self, tmp_path):
        """If the session JSONL doesn't exist, mark_scored returns False without creating anything."""
        from roampal.hooks.session_manager import SessionManager

        sm = SessionManager(tmp_path)
        result = await sm.mark_scored("conv5", "doc_1", "worked")

        assert result is False
        sessions_dir = tmp_path / "mcp_sessions"
        # No JSONL file created
        assert not (sessions_dir / "conv5.jsonl").exists()
        # No .tmp files anywhere
        if sessions_dir.exists():
            tmp_leftovers = list(sessions_dir.glob("*.tmp"))
            assert tmp_leftovers == []

    async def test_doc_id_not_found_no_write(self, tmp_path):
        """If no matching doc_id is found, mark_scored returns False and performs zero writes."""
        from roampal.hooks.session_manager import SessionManager

        lines = [
            json.dumps({"role": "user", "content": "hello"}) + "\n",
            json.dumps({"role": "assistant", "doc_id": "doc_1", "content": "hi back"}) + "\n",
        ]
        self._seed_session(tmp_path, "conv6", lines)

        sm = SessionManager(tmp_path)
        original_bytes = (tmp_path / "mcp_sessions" / "conv6.jsonl").read_bytes()

        # Request a doc_id that doesn't exist in the file
        result = await sm.mark_scored("conv6", "doc_999", "worked")

        assert result is False
        # File must be byte-for-byte unchanged (no write path taken)
        assert (tmp_path / "mcp_sessions" / "conv6.jsonl").read_bytes() == original_bytes

    async def test_corrupt_lines_skipped_and_last_match_wins(self, tmp_path):
        """mark_scored skips corrupt JSONL lines and updates only the last matching assistant record."""
        from roampal.hooks.session_manager import SessionManager

        # Two assistant records with same doc_id; one garbage line between them.
        # mark_scored scans in reverse, so it should find the NEWER (last) match.
        lines = [
            json.dumps({"role": "user", "content": "first"}) + "\n",
            json.dumps({"role": "assistant", "doc_id": "doc_1", "content": "old response"}) + "\n",
            "THIS IS GARBAGE NOT JSON\n",  # corrupt line — should be skipped
            json.dumps({"role": "user", "content": "second"}) + "\n",
            json.dumps({"role": "assistant", "doc_id": "doc_1", "content": "new response"}) + "\n",
        ]
        self._seed_session(tmp_path, "conv7", lines)

        sm = SessionManager(tmp_path)
        result = await sm.mark_scored("conv7", "doc_1", "worked")

        assert result is True
        # Parse only valid JSON lines to find assistant records
        all_lines = open(tmp_path / "mcp_sessions" / "conv7.jsonl").readlines()
        assistants = []
        for line in all_lines:
            try:
                rec = json.loads(line.strip())
                if rec.get("role") == "assistant":
                    assistants.append(rec)
            except json.JSONDecodeError:
                pass  # garbage lines are skipped by parsing, preserved in file
        # Only the last (newest) assistant record should be marked scored
        assert len(assistants) == 2
        assert not assistants[0].get("scored"), "Older match must NOT be updated"
        assert assistants[1]["scored"] is True, "Newer match MUST be updated"
        assert assistants[1]["outcome"] == "worked"
        # Garbage line preserved verbatim in the file (atomic rewrite writes all lines back)
        raw_content = open(tmp_path / "mcp_sessions" / "conv7.jsonl").read()
        assert "THIS IS GARBAGE NOT JSON" in raw_content

    async def test_cache_flags_updated(self, tmp_path):
        """mark_scored updates _last_exchange_cache when doc_id matches (regression guard)."""
        from roampal.hooks.session_manager import SessionManager

        lines = [
            json.dumps({"role": "user", "content": "hello"}) + "\n",
            json.dumps({"role": "assistant", "doc_id": "doc_1", "content": "hi back"}) + "\n",
        ]
        self._seed_session(tmp_path, "conv8", lines)

        sm = SessionManager(tmp_path)
        # Pre-seed the cache with a matching entry (simulates an in-flight exchange)
        sm._last_exchange_cache["conv8"] = {"doc_id": "doc_1"}

        await sm.mark_scored("conv8", "doc_1", "worked")

        assert sm._last_exchange_cache["conv8"]["scored"] is True
        assert sm._last_exchange_cache["conv8"]["outcome"] == "worked"
