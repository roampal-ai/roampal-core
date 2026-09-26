"""
Task 37 guard: no silent truncation of memory text.

v0.6.0 user standing rule (memory_bank_93aecd4f): memory text is never
silently cut — over-limit writes get REJECTED with an actionable
message. One source of truth for every limit lives in
`roampal/memory_limits.py`; the only allowed slicing is the two display
cuts (RECENT EXCHANCES and the cold-start profile block, 300 each),
which must be explicit and never touch a write path.

The scan test fails the build if any source file under roampal/ slices
memory text outside the allowlist — a new silent truncation cannot ship
unnoticed (it either uses memory_limits or adds a reviewed, justified
allowlist entry).
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
)

import re
from pathlib import Path

import pytest

from roampal.memory_limits import (
TARGET_CHARS,
FACT_MAX,
MEMORY_BANK_ENTRY_MAX,
RECENT_EXCHANGES_DISPLAY_CUT,
SERVER_BACKSTOP_MAX,
SUMMARY_TAKEAWAY_MAX,
TARGET_PHRASE,
check_length,
fact_too_long,
memory_bank_entry_too_long,
sidecar_reprompt_too_long,
summary_takeaway_too_long,
over_backstop,
)


ROOT = Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
)
SOURCE_DIRS = [ROOT / "roampal"]
SCAN_SUFFIXES = {".py", ".ts"}

# Matches both Python slices and JS slices of a fixed size:
#   content[:200]   content[:2000]   .slice(0, 200)
_SLICE_PATTERN = re.compile(r"\[\s*:\s*(\d+)\s*\]|\.slice\(\s*0\s*,\s*(\d+)\s*\)")

# Log lines, fingerprints and previews cut display-only text to keep logs
# readable — they never touch stored memory text. Whitelist by (file
# fragment, line fragment); each keeper carries a justification.
_LOG_LINE = re.compile(r"(_diag\(|logger\.|debugLog\(|f\"Recorded takeaway|Preview:|description=f)")


def _is_log_or_id_line(stripped: str) -> bool:
    """Log/debug lines, hashes and doc-id generation never store memory
    text — cutting them is deliberately fine (not memory truncation)."""
    if _LOG_LINE.search(stripped):
        return True
    if "uuid.uuid4().hex[:8]" in stripped or "hexdigest()[:12]" in stripped:
        return True  # id generation
    return False


def _iter_sources():
    for d in SOURCE_DIRS:
        for path in d.rglob("*"):
            if path.suffix in SCAN_SUFFIXES:
                if "tests" in path.parts or "dev" in path.parts:
                    continue  # tests and dev docs may slice freely
                yield path


def _matches_in(paths):
    violations = []
    for path in paths:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if not _SLICE_PATTERN.search(line):
                continue
            if stripped.startswith("#") or "def test_" in line:
                continue
            if _is_log_or_id_line(stripped):
                continue
            if any(fn in rel and frag in stripped for fn, frag, _ in ALLOWED):
                continue
            violations.append(f"{rel}:{lineno}: {stripped}")
    return violations


# Explicit keepers — every entry carries its justification. NEW slicing
# shows up as a failing test; the author either uses memory_limits or
# documents the keeper here for review.
ALLOWED = [
    # --- intentional INPUT caps (task spec: not memory truncation) ---
    ("sidecar_service.py", "text[:2000]",
     "sidecar summarizer prompt input cap"),
    ("sidecar_service.py", "text[:8000]",
     "sidecar 8000-char-per-side input cap (explicit Task 35 allowance)"),
    # --- intentional LIST (not text) caps ---
    ("sidecar_service.py", "cleaned[:8]", "max 8 tags"),
    ("sidecar_service.py", "cleaned[:10]", "max 10 tags"),
    ("tag_service.py", "[:8]", "tag-count caps"),
    ("context_service.py", "[:1]", "insight item-count cap"),
    ("context_service.py", "[:3]", "concept item-count cap"),
    ("context_service.py", "[:10]", "concept item-count cap"),
    ("unified_memory_system.py", "concepts[:10]", "concept item-count cap"),
    ("search_service.py", "results[:1]", "result item-count cap"),
    ("memory_cmds.py", "samples[:3]", "sample item-count cap"),
    ("memory_cmds.py", "created_at'][:10]", "date display only"),
    ("memory_cmds.py", "resp.text[:200]", "HTTP response display only"),
    ("sidecar_service.py", '"value": summary_val[:100]',
     "sidecar self-check diagnostic preview (not stored memory)"),
    ("mcp/server.py", "response.text[:300]",
     "error-detail extraction serving the rejection text to the model (display, never storage)"),
    ("main.py", "content[:60]",
     "content_hint - 60-char retrieval hint (explicit Task 35 allowance)"),
    ("main.py", "_tags[:5]", "tag-list display cap"),
    # --- fingerprints / id generation (never memory content) ---
    ("roampal.ts", "fpInput", "exchange fingerprint input"),
    ("roampal.ts", ".slice(0, 8000)",
     "system-prompt 8000-char-per-side input cap (explicit Task 35 allowance)"),
    ("roampal.ts", ".toString(16).padStart(8, '0').slice(0, 12)", "fingerprint hash id"),
]


class TestNoSilentTruncation:
    """Task 37 guard: scanning write/inject paths for fixed-size slicing."""

    def test_no_unallowlisted_slices_in_source(self):
        violations = _matches_in(_iter_sources())
        assert not violations, (
            "Potential silent truncation (write/inject paths must not slice "
            "memory text outside memory_limits' display cuts). Either route "
            "through roampal/memory_limits.py or add a justified allowlist "
            f"entry in test_memory_limits.py:\n" + "\n".join(violations[:30])
        )

    def test_ts_display_cut_mirrors_python(self):
        """Task 37: the plugin keeps TS constants mirrored to the Python
        module — the RECENT EXCHANGES display cut must agree everywhere,
        and the summary hard cap drives the plugin's re-ask sanity bound."""
        ts = (ROOT / "roampal" / "plugins" / "opencode" / "roampal.ts").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r"const RECENT_EXCHANGES_DISPLAY_CUT = (\d+)", ts
        )
        assert match, "TS RECENT_EXCHANGES_DISPLAY_CUT constant missing"
        assert int(match.group(1)) == RECENT_EXCHANGES_DISPLAY_CUT

        cap = re.search(r"const SUMMARY_TAKEAWAY_MAX = (\d+)", ts)
        assert cap, "TS SUMMARY_TAKEAWAY_MAX constant missing"
        assert int(cap.group(1)) == SUMMARY_TAKEAWAY_MAX

        # The TS display cut must be word-boundary + "…" (clipDisplay), not a
        # bare slice. Both RECENT-EXCHANGES formatter sites use the helper.
        assert re.search(r"function clipDisplay\(body: string\)", ts)
        assert 'lastIndexOf(" ")' in ts
        assert ts.count("clipDisplay(body)") >= 2, (
            "both recent-exchanges formatters must route through clipDisplay"
        )
        assert '+ "…"' in ts, "clipped TS entries must end in the … marker"


# =====================================================================


# ============================================================================
# Behavior: rejection messages + nothing-stored (async endpoint level)
# ============================================================================

from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
async def limits_client(tmp_path, monkeypatch):
    """Same shape as test_fastapi_endpoints' self-contained harness: mocked
    profile registry so handlers see a mock memory instead of real init."""
    import roampal.server.main as main

    mock_memory = MagicMock()
    mock_memory.store_memory_bank = AsyncMock(return_value="memory_bank_deadbeef")
    mock_memory.update_memory_bank = AsyncMock(return_value="memory_bank_deadbeef")
    mock_memory.store_working = AsyncMock(return_value="working_deadbeef")
    mock_memory.record_outcome = AsyncMock(return_value={})
    mock_memory.collections = {}
    mock_memory.initialized = True

    mock_session = MagicMock()
    mock_session.set_scored_this_turn = MagicMock()
    mock_session.store_exchange = AsyncMock(return_value=None)
    mock_session.was_scoring_required = MagicMock(return_value=False)
    original = dict(main._memory_by_profile)
    original_sm = dict(main._session_manager_by_profile)
    main._memory_by_profile["default"] = mock_memory
    main._session_manager_by_profile["default"] = mock_session

    with patch("roampal.server.main._resolve_profile_name", return_value="default"):
        app = main.create_app()

        async def fake_lifespan(_app):
            yield

        app.router.lifespan_context = fake_lifespan

        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac, mock_memory

    main._memory_by_profile.clear()
    main._memory_by_profile.update(original)
    main._session_manager_by_profile.clear()
    main._session_manager_by_profile.update(original_sm)


class TestServerRejectionsInsteadOfCuts:
    """Task 37: over-limit writes get OUR message (not a silent cut, not
    a generic validation error); nothing is stored on rejection."""

    @pytest.mark.asyncio
    async def test_add_entry_over_600_rejected_with_split_message(self, limits_client):
        ac, mock_memory = limits_client
        resp = await ac.post(
            "/api/memory-bank/add",
            json={"content": "x" * (MEMORY_BANK_ENTRY_MAX + 1), "tags": ["project"]},
        )
        assert resp.status_code == 400
        assert str(MEMORY_BANK_ENTRY_MAX) in resp.text
        assert "add_to_memory_bank" in resp.text
        mock_memory.store_memory_bank.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_add_entry_at_limit_round_trips(self, limits_client):
        """At-limit content stores byte-identical (never cut)."""
        ac, mock_memory = limits_client
        exact = "w" * MEMORY_BANK_ENTRY_MAX
        resp = await ac.post(
            "/api/memory-bank/add",
            json={"content": exact, "tags": ["project"]},
        )
        assert resp.status_code == 200
        mock_memory.store_memory_bank.assert_awaited_once()
        stored_text = mock_memory.store_memory_bank.call_args.kwargs.get("text")
        assert stored_text == exact

    @pytest.mark.asyncio
    async def test_huge_write_rejected(self, limits_client):
        """A 2000+-char takeaway gets rejected (nothing stored). The 600
        rule fires first, so the model sees the split/rewrite message —
        either way nothing is stored and nothing is cut."""
        ac, mock_memory = limits_client
        resp = await ac.post(
            "/api/record-response",
            json={
                "key_takeaway": "y" * (SERVER_BACKSTOP_MAX + 10),
                "conversation_id": "s1",
            },
        )
        assert resp.status_code == 400
        assert "Nothing was stored" in resp.text or "Rewrite in ~300 chars" in resp.text
        mock_memory.store_working.assert_not_called()

    @pytest.mark.asyncio
    async def test_takeaway_over_600_rejected_with_split_message(self, limits_client):
        ac, mock_memory = limits_client
        resp = await ac.post(
            "/api/record-response",
            json={"key_takeaway": "y" * (SUMMARY_TAKEAWAY_MAX + 1),
                  "conversation_id": "s1"},
        )
        assert resp.status_code == 400
        assert "record_response" in resp.text  # split instruction
        mock_memory.store_working.assert_not_called()

    @pytest.mark.asyncio
    async def test_takeaway_at_limit_stores(self, limits_client):
        ac, mock_memory = limits_client
        exact = "x" * SUMMARY_TAKEAWAY_MAX
        resp = await ac.post(
            "/api/record-response",
            json={"key_takeaway": exact, "conversation_id": "s1"},
        )
        assert resp.status_code == 200
        stored = mock_memory.store_working.call_args.kwargs.get("content")
        assert stored == f"Key takeaway: {exact}"

    @pytest.mark.asyncio
    async def test_update_over_600_rejected(self, limits_client):
        ac, mock_memory = limits_client
        resp = await ac.post(
            "/api/memory-bank/update",
            json={"id": "memory_bank_deadbeef",
                  "new_content": "z" * (MEMORY_BANK_ENTRY_MAX + 1)},
        )
        assert resp.status_code == 400
        assert "add_to_memory_bank" in resp.text
        mock_memory.update_memory_bank.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_outcome_summary_over_600_rejected(self, limits_client):
        """The score_memories/sidecar write path had NO limits at all —
        a sidecar summary could be ~16k chars and stored as-is. Now a
        >600 summary is rejected with the rewrite message and NOTHING is
        stored (summary block, facts loop — both skipped)."""
        from unittest.mock import MagicMock
        ac, mock_memory = limits_client

        def no_fact_store(*a, **kw):  # any store_working call beyond mocked counts is unexpected
            raise AssertionError("nothing should be stored on rejection")

        resp = await ac.post(
            "/api/record-outcome",
            json={
                "conversation_id": "s1",
                "outcome": "worked",
                "exchange_summary": "s" * (SUMMARY_TAKEAWAY_MAX + 1),
                "facts": ["f" * (FACT_MAX + 1)],  # ALSO over-limit — both must reject
            },
        )
        assert resp.status_code == 400
        assert "record_response" in resp.text  # summary split message wins
        # The store path must never have been reached:
        sc_helper = mock_memory.store_working
        sc_helper.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_outcome_fact_over_150_rejected(self, limits_client):
        """Summary within limits, but a single fact over 150 -> rejected
        with the per-fact message."""
        ac, mock_memory = limits_client
        resp = await ac.post(
            "/api/record-outcome",
            json={
                "conversation_id": "s1",
                "outcome": "worked",
                "exchange_summary": "short summary",
                "facts": ["ok fact", "f" * (FACT_MAX + 1)],
            },
        )
        assert resp.status_code == 400
        assert "150" in resp.text
        mock_memory.store_working.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_outcome_within_limits_stores(self, limits_client):
        ac, mock_memory = limits_client
        resp = await ac.post(
            "/api/record-outcome",
            json={
                "conversation_id": "s1",
                "outcome": "worked",
                "exchange_summary": "x" * SUMMARY_TAKEAWAY_MAX,
                "facts": ["a" * FACT_MAX],
            },
        )
        assert resp.status_code == 200
        # summary stored + fact stored (mock counts both)
        assert mock_memory.store_working.await_count >= 1


class TestCheckLengthBehavior:
    """The single check_length + the exact messages the model must see."""

    def test_within_limits_returns_empty(self):
        assert check_length("summary", "a" * SUMMARY_TAKEAWAY_MAX) == ""
        assert check_length("takeaway", "b" * (TARGET_CHARS + 10)) == ""
        assert check_length("memory_bank", "c" * MEMORY_BANK_ENTRY_MAX) == ""
        assert check_length("fact", "d" * FACT_MAX) == ""
        assert check_length("summary", None) == ""

    def test_over_limit_returns_our_messages(self):
        err = check_length("summary", "a" * (SUMMARY_TAKEAWAY_MAX + 1))
        assert err == summary_takeaway_too_long(SUMMARY_TAKEAWAY_MAX + 1)
        assert "Rewrite in" in err and "record_response" in err

        err = check_length("memory_bank", "b" * (MEMORY_BANK_ENTRY_MAX + 1))
        assert "add_to_memory_bank" in err and "~300 chars" in err

        err = check_length("fact", "c" * (FACT_MAX + 1))
        assert err == fact_too_long(FACT_MAX + 1)
        assert "One fact per item" in err

        err = sidecar_reprompt_too_long(700)
        assert err.startswith("Too long: 700/600") and "~300 chars" in err

    def test_backstop_catches_every_kind(self):
        big = "x" * (SERVER_BACKSTOP_MAX + 1)
        for kind in ("anything-else", None):
            err = check_length(kind, big)
            assert "Nothing was stored" in err
            assert over_backstop(len(big)) == err
        # Known kinds fire their (kinder, more specific) rule first — the
        # backstop tier only covers callers without a kind rule.
        assert "Nothing was stored" in over_backstop(SERVER_BACKSTOP_MAX + 10)


# ============================================================================
# Task 37 chunk B: stop-endpoint summary-only (sidecar path)
# ============================================================================

class TestStopEndpointSummaryOnly:
    """The /api/hooks/stop sidecar path (lifecycle_only=False) stores ONLY the
    summary: over-limit is rejected HTTP 400 with our rewrite message and
    NOTHING is stored; at-limit content round-trips byte-identical."""

    @pytest.mark.asyncio
    async def test_sidecar_summary_over_600_rejected_nothing_stored(self, limits_client):
        ac, mock_memory = limits_client
        resp = await ac.post(
            "/api/hooks/stop",
            json={
                "conversation_id": "s1",
                "user_message": "u",
                "assistant_response": "s" * (SUMMARY_TAKEAWAY_MAX + 1),
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert f"{SUMMARY_TAKEAWAY_MAX + 1}/{SUMMARY_TAKEAWAY_MAX}" in detail
        assert "1-2 sentences" in detail
        # Sidecar-reprompt split instruction is NOT the record_response advice —
        # sidecar consumers can only rewrite, not spawn record_response calls.
        assert "record_response" not in detail
        mock_memory.store_working.assert_not_called()

    @pytest.mark.asyncio
    async def test_sidecar_summary_at_limit_stores_summary_only(self, limits_client):
        """No 'User: ... / Assistant: ...' wrapper — the summary is the content."""
        ac, mock_memory = limits_client
        exact = "m" * SUMMARY_TAKEAWAY_MAX
        resp = await ac.post(
            "/api/hooks/stop",
            json={
                "conversation_id": "s1",
                "user_message": "u",
                "assistant_response": exact,
            },
        )
        assert resp.status_code == 200
        stored = mock_memory.store_working.call_args.kwargs.get("content")
        assert stored == exact  # byte-identical, no wrapper


# ============================================================================
# Task 37 chunk B: `/api/memory/update-content` (roampal summarize path)
# ============================================================================

class TestUpdateContentLimit:
    """update-content had NO limits at all — now rejects over-600 updates."""

    @pytest.mark.asyncio
    async def test_update_content_over_600_rejected(self, limits_client):
        ac, mock_memory = limits_client
        adapter = MagicMock()
        adapter.get_fragment = MagicMock(return_value={"id": "w1", "content": "old", "metadata": {}})
        adapter.upsert_vectors = AsyncMock()
        mock_memory.collections = {"working": adapter}
        embedder = MagicMock()
        embedder.embed_text = AsyncMock(return_value=[0.1])
        mock_memory._embedding_service = embedder

        resp = await ac.post(
            "/api/memory/update-content",
            json={
                "doc_id": "w1",
                "collection": "working",
                "new_content": "z" * (SUMMARY_TAKEAWAY_MAX + 1),
            },
        )
        assert resp.status_code == 400
        assert "600" in resp.text
        adapter.upsert_vectors.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_content_at_limit_round_trips(self, limits_client):
        ac, mock_memory = limits_client
        adapter = MagicMock()
        adapter.get_fragment = MagicMock(return_value={"id": "w1", "content": "old", "metadata": {}})
        adapter.upsert_vectors = AsyncMock()
        mock_memory.collections = {"working": adapter}
        embedder = MagicMock()
        embedder.embed_text = AsyncMock(return_value=[0.1])
        mock_memory._embedding_service = embedder

        resp = await ac.post(
            "/api/memory/update-content",
            json={
                "doc_id": "w1",
                "collection": "working",
                "new_content": "z" * SUMMARY_TAKEAWAY_MAX,
            },
        )
        assert resp.status_code == 200
        metadata = adapter.upsert_vectors.call_args.kwargs["metadatas"][0]
        assert metadata["content"] == "z" * SUMMARY_TAKEAWAY_MAX


# ============================================================================
# Task 37 chunk B: cold-start profile display cut + clipped block footer
# ============================================================================

class TestColdStartDisplayCut:
    """_first_sentence cuts at PROFILE_BLOCK_DISPLAY_CUT (word boundary, "…")
    and the profile block ends with the clipped-lines note — only when
    something was actually clipped."""

    def _mock_mem(self, facts):
        mock = MagicMock()
        mock._memory_bank_service.list_all = MagicMock(return_value=facts)
        mock._data_path = MagicMock()
        mock._data_path.__truediv__ = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        mock.search = AsyncMock(return_value=[])
        return mock

    async def _fact(self, tag, text, n=1):
        return {"id": f"{tag}{n}", "metadata": {"tags": f'["{tag}"]'}, "text": text}

    @pytest.mark.asyncio
    async def test_clipped_lines_get_footer(self):
        import roampal.server.main as main
        from roampal.memory_limits import PROFILE_BLOCK_DISPLAY_CUT

        facts = [
            await self._fact("identity", "I" * 400),          # over the cut
            await self._fact("preference", "Prefers pytest"),  # short — no cut
        ]
        result = await main._build_cold_start_profile(self._mock_mem(facts))
        assert result.endswith("</roampal-user-profile>")
        assert "Lines ending in … are clipped — use search_memory" in result
        identity_line = [ln for ln in result.splitlines() if ln.startswith("Identity: ")][0]
        assert len(identity_line) <= len("Identity: ") + PROFILE_BLOCK_DISPLAY_CUT
        assert identity_line.endswith("…")

    @pytest.mark.asyncio
    async def test_no_clipping_no_footer(self):
        import roampal.server.main as main
        facts = [await self._fact("identity", "User is a data scientist")]
        result = await main._build_cold_start_profile(self._mock_mem(facts))
        assert "</roampal-user-profile>" in result
        assert "Lines ending in" not in result


# ============================================================================
# Task 37 spec: Claude Code hook output (platform limit 10,000 then 2k preview)
# — the worst-case LEGAL block must stay under the platform cap.
# ============================================================================

class TestHookOutputWorstCaseUnderPlatformCap:
    """
    Every write path is now capped (600 summaries/entries, 150 facts, 300
    display cuts). This test composes the worst-case legal hook output —
    cold-start profile block (one line per tag) + KNOWN CONTEXT (8 retrieval
    memories at full text) + the lean scoring prompt — and pins that the
    write-side limits keep the total under Claude Code's 10,000-char hook
    platform limit. If a cap is raised without re-budgeting, this fails.
    """

    def test_worst_case_block_under_10000(self, tmp_path):
        from roampal.hooks.session_manager import SessionManager
        import roampal.server.main as main
        from roampal.memory_limits import (
            PROFILE_BLOCK_DISPLAY_CUT,
            SUMMARY_TAKEAWAY_MAX,
        )

        # KNOWN CONTEXT worst case: retrieval cap (8 memories/turn, full text,
        # never cut) with EVERY memory at the summary hard cap + one formatting row.
        text_at_cap = "m" * SUMMARY_TAKEAWAY_MAX
        mem_line = f"• {text_at_cap} [id:working_01234567] (17h, working, wilson:70%, used:3x, last:worked)"
        known_context = "KNOWN CONTEXT (8 memories):\n" + "\n".join([mem_line] * 8)

        # Cold-start profile worst case: one clipped line per tag category.
        profile = "<roampal-user-profile>\n" + "\n".join(
            f"{label}: {main._first_sentence(text_at_cap)}"
            for label in ("Identity", "Preference", "Goal", "Project",
                          "System Mastery", "Agent Growth")
        ) + "\n</roampal-user-profile>"

        # Lean scoring prompt (Claude Code path) with 8 surfaced memories.
        sm = SessionManager(tmp_path)
        surfaced = [{"id": f"working_{i:012d}", "content": text_at_cap}
                    for i in range(8)]
        scoring_prompt = sm.build_scoring_prompt({}, "user msg", surfaced)

        total = len(known_context) + len(profile) + len(scoring_prompt)
        assert total < 10000, f"worst-case hook block {total} >= 10000 platform cap"
        assert total > 5000, "vacuous composition — worst case must be substantial"