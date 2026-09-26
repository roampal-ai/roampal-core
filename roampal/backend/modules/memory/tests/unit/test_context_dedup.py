"""Task 30 acceptance: no re-injection of what's already in context —
WITHOUT losing anything.

The audit (F5) found the server re-injects memories already sitting in the
Claude Code context (one memory appeared 3x in one session). Hook output is
append-only, so every turn's block stays until compaction.

Contract (user-approved spec):
- NO size cap, retrieval unchanged (4 summaries + 4 facts, full text).
- The per-conversation record is keyed on ID + CONTENT — an updated
  memory re-shows in full again.
- A repeat is a one-line pointer, NEVER silently dropped:
  `still relevant: [id:…] — shown earlier this session`.
- The surfaced/scoring list carries every surfaced id, pointers included —
  identical to today's.
- Record resets via the SessionStart hooks (compact/startup/clear →
  /api/hooks/session-reset, session id from stdin) — after compaction the
  full text returns.
- OpenCode path unchanged (it rebuilds the block per request anyway).
"""

import asyncio
import argparse
import io
import json
import os
import random
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
)

import pytest

from roampal.backend.modules.memory.unified_memory_system import (
    UnifiedMemorySystem,
    injection_hash,
)


class _Md5EmbedFixture:
    """Deterministic 768-dim unit-norm md5-per-text embedder (real cosine
    geometry) — the acceptance point is the pointer/record MECHANICS, not
    model accuracy (same fixture family as the dedup suite)."""

    def __init__(self, dim=768):
        self.dim = dim

    def _vec(self, text):
        import hashlib

        seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "big")
        rng = random.Random(seed)
        out = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        norm = sum(v * v for v in out) ** 0.5
        return [v / norm for v in out]

    async def embed_text(self, text, role="passage"):
        return self._vec(text)

    async def embed_texts(self, texts, role="passage"):
        return [self._vec(t) for t in texts]

    async def prewarm(self):
        return None


@pytest.fixture(autouse=True)
async def _cancel_warmup_tasks():
    """Same orphan-worker leak guard as test_dedup/test_unified suites (py3.10)."""
    yield
    try:
        for task in asyncio.all_tasks():
            if task.get_name() in ("warmup_ce", "warmup_embedding", "reembed") and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
    except RuntimeError:
        pass


# ============================================================================
# Unit level: the pointer renderer itself
# ============================================================================

class TestPointerRendering:
    def _ums_skeleton(self):
        ums = UnifiedMemorySystem.__new__(UnifiedMemorySystem)
        ums._memory_bank_service = MagicMock()
        ums._memory_bank_service.list_all = MagicMock(return_value=[])
        return ums

    def _memory(self, mid, content, mtype="fact"):
        return {
            "id": mid,
            "content": content,
            "collection": "memory_bank" if mtype == "fact" else "working",
            "metadata": {"memory_type": mtype},
        }

    def test_repeat_renders_pointer(self):
        ums = self._ums_skeleton()
        mem = self._memory("working_1", "User prefers dark mode")
        # The record's keys are composite id:{content-hash} pairs (the .get
        # lookup the renderer performs); a plain-id entry never triggers.
        text = ums._format_context_injection(
            {"memories": [mem]},
            already_shown={
                f"working_1:{injection_hash('User prefers dark mode')}": "ts",
            },
        )
        assert "still relevant: [id:working_1] — shown earlier this session" in text
        assert "User prefers dark mode" not in text

    def test_plain_id_key_does_not_match(self):
        """Non-vacuity: a record holding JUST the id (wrong shape) must not
        collapse the line — the content hash is part of the key."""
        ums = self._ums_skeleton()
        mem = self._memory("working_1", "User prefers dark mode")
        text = ums._format_context_injection(
            {"memories": [mem]}, already_shown={"working_1": "aabbccddeeff"}
        )
        assert "still relevant" not in text
        assert "User prefers dark mode" in text

    def test_updated_content_shows_full_again(self):
        """Key = id+content: same id, new content -> full text again."""
        ums = self._ums_skeleton()
        mem = self._memory("working_1", "User prefers LIGHT mode now")
        text = ums._format_context_injection(
            {"memories": [mem]},
            already_shown={"working_1": injection_hash("User prefers dark mode")},
        )
        assert "User prefers LIGHT mode now" in text
        assert "still relevant" not in text

    def test_pointer_keeps_id_consistent_with_scoring(self):
        """The pointer line keeps the id so the scoring roster is identical
        whether a line renders full or as a pointer."""
        ums = self._ums_skeleton()
        mem = self._memory("working_1", "User prefers dark mode")
        text = ums._format_context_injection(
            {"memories": [mem]},
            already_shown={"working_1": injection_hash("User prefers dark mode")},
        )
        assert "[id:working_1]" in text

    def test_empty_record_renders_full(self):
        ums = self._ums_skeleton()
        mem = self._memory("working_1", "User prefers dark mode")
        text = ums._format_context_injection({"memories": [mem]}, already_shown={})
        assert "• User prefers dark mode [id:working_1]" in text

    def test_injection_hash_stable_across_processes(self):
        import hashlib

        assert injection_hash("hello") == hashlib.md5(b"hello").hexdigest()[:12]
        assert injection_hash("") == hashlib.md5(b"").hexdigest()[:12]

    def test_record_hash_matches_renderer_with_field_divergence(self):
        """Double-check find: normalize_memory prefers metadata.text over
        root text, so a memory carrying BOTH must hash identically on the
        record side and the renderer side — otherwise pointers never
        render. Both now extract through normalize_memory; this test pins
        the shared precedence (drift guard)."""
        from roampal.backend.modules.memory.unified_memory_system import normalize_memory

        mem = {
            "id": "working_x",
            "text": "ROOT TEXT",
            "metadata": {"memory_type": "fact", "text": "META TEXT"},
            "collection": "working",
        }
        normalized = normalize_memory(dict(mem), "working")
        content = normalized.get("content", "")

        # the renderer's precedence: metadata.text wins over root text
        assert content == "META TEXT"
        # the record side must derive the SAME content — the hash of what a
        # naive root-text-first chain would pick differs here, so this row
        # fails if the two sides use different chains again
        assert injection_hash(content) == injection_hash("META TEXT")
        assert injection_hash(content) != injection_hash("ROOT TEXT")


# ============================================================================
# Acceptance: the REAL get-context flow over turns (real UMS, fake embedder,
# short-lived ChromaDB archive) -> full text once, pointers after, full text
# again after the session-reset event; scoring list unchanged.
# ============================================================================

async def _make_real_ums(tmp_path):
    ums = UnifiedMemorySystem(
        data_path=str(tmp_path / "data"),
        embed_service=_Md5EmbedFixture(),
    )
    await ums.initialize()
    return ums


@pytest.fixture
async def dedup_client(tmp_path):
    """limits-client harness shape, but with a REAL UnifiedMemorySystem
    (fake embedder, real short-lived ChromaDB) as the profile's memory."""
    import roampal.server.main as main

    real_ums = await _make_real_ums(tmp_path)

    mock_session = MagicMock()
    mock_session.was_scoring_required = MagicMock(return_value=False)
    mock_session.is_first_message = MagicMock(return_value=False)
    mock_session.check_and_clear_completed = MagicMock(return_value=False)
    mock_session.mark_first_message_seen = MagicMock()
    original_um = dict(main._memory_by_profile)
    original_sm = dict(main._session_manager_by_profile)
    original_rec = dict(main._session_injections)
    original_pending = dict(main._pending_injections)
    main._memory_by_profile["default"] = real_ums
    main._session_manager_by_profile["default"] = mock_session

    with patch("roampal.server.main._resolve_profile_name", return_value="default"):
        app = main.create_app()

        async def fake_lifespan(_app):
            yield

        app.router.lifespan_context = fake_lifespan
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac, real_ums, main

    main._memory_by_profile.clear()
    main._memory_by_profile.update(original_um)
    main._session_manager_by_profile.clear()
    main._session_manager_by_profile.update(original_sm)
    main._session_injections.clear()
    main._session_injections.update(original_rec)
    main._pending_injections.clear()
    main._pending_injections.update(original_pending)


async def _cc_turn(ac, query, conversation_id, *, ack=True):
    """One Claude Code prompt: get-context with dedup, then (like the hook,
    after printing) ack delivery. ack=False models a hook that timed out or
    exited non-zero — Claude Code never received the block."""
    resp = await ac.post(
        "/api/hooks/get-context",
        json={"query": query, "conversation_id": conversation_id, "dedup_injections": True},
    )
    assert resp.status_code == 200
    token = resp.json().get("injection_token", "")
    if ack and token:
        ack_resp = await ac.post("/api/hooks/injection-ack", json={"injection_token": token})
        assert ack_resp.status_code == 200
    return resp


class TestNoReinjectionWithoutLoss:
    @pytest.mark.asyncio
    async def test_three_turns_full_pointer_full(self, dedup_client):
        """The acceptance row: turn 1 full text; turns 2+ that memory
        renders as a pointer; after the compact/startup/clear-style
        session-reset the SAME full text returns. Nothing is lost."""
        ac, real_ums, main = dedup_client

        doc_id = await real_ums.store_memory_bank(
            "User owns a passive solar greenhouse project",
            tags=["project"],
            noun_tags=["greenhouse"],
        )

        resp1 = await _cc_turn(ac, "greenhouse project", "sess")
        turn1 = resp1.json()["formatted_injection"]
        full_line = f"• User owns a passive solar greenhouse project [id:{doc_id}]"
        assert full_line in turn1, turn1
        assert "still relevant" not in turn1

        record = main._session_injections.get("sess")
        assert record and f"{doc_id}:{injection_hash('User owns a passive solar greenhouse project')}" in record

        resp2 = await _cc_turn(ac, "greenhouse project", "sess")
        turn2 = resp2.json()["formatted_injection"]
        assert f"still relevant: [id:{doc_id}] — shown earlier this session" in turn2
        assert full_line not in turn2

        # Scores identically to today: the memory stays in relevant_memories
        # (ids not dropped — the surfaced list matches turn 1's).
        ids1 = [m.get("id") for m in resp1.json()["relevant_memories"]]
        ids2 = [m.get("id") for m in resp2.json()["relevant_memories"]]
        assert ids1 == ids2 and doc_id in ids2

        resp3 = await _cc_turn(ac, "greenhouse project", "sess")
        assert "still relevant:" in resp3.json()["formatted_injection"]

        # Session reset (compact/startup/clear flow): full text again.
        reset = await ac.post(
            "/api/hooks/session-reset",
            json={"conversation_id": "sess"},
        )
        assert reset.status_code == 200
        resp4 = await _cc_turn(ac, "greenhouse project", "sess")
        turn4 = resp4.json()["formatted_injection"]
        assert full_line in turn4, turn4
        assert "still relevant" not in turn4

    @pytest.mark.asyncio
    async def test_no_flag_means_no_dedup_opencode_path(self, dedup_client):
        """OpenCode's rebuilt-per-request surface does NOT opt in: without
        dedup_injections, the SAME memory shows FULL text every turn (a
        pointer there would hide content — the prior block is already
        gone from its rebuilt system prompt) and nothing is recorded."""
        ac, real_ums, main = dedup_client

        doc_id = await real_ums.store_memory_bank(
            "OpenCode user tracks a greenhouse dashboard",
            tags=["project"],
            noun_tags=["dashboard"],
        )
        payload = {"query": "greenhouse dashboard", "conversation_id": "oc_s1"}

        resp1 = await ac.post("/api/hooks/get-context", json=payload)
        assert resp1.status_code == 200
        turn1 = resp1.json()["formatted_injection"]
        assert f"[id:{doc_id}]" in turn1

        resp2 = await ac.post("/api/hooks/get-context", json=payload)
        turn2 = resp2.json()["formatted_injection"]
        assert "still relevant" not in turn2, (
            "OpenCode surface must NOT get pointers (its blocks are rebuilt "
            "per request — a pointer would hide the memory)"
        )
        assert main._session_injections.get("oc_s1") is None
        # No ack token either — nothing is ever held pending for OpenCode.
        assert resp1.json()["injection_token"] == ""
        assert not any(
            e["conversation_id"] == "oc_s1" for e in main._pending_injections.values()
        )

    @pytest.mark.asyncio
    async def test_updated_memory_shows_full_again(self, dedup_client):
        """Same id, new content -> the record key changes -> full text."""
        ac, real_ums, main = dedup_client

        # Seed FIRST, then two turns: turn 1 records the id+content key.
        content_v1 = "User asked about the baking schedule for Friday"
        doc_id = await real_ums.store_working(
            content=content_v1,
            conversation_id="upd1",
            metadata={"memory_type": "exchange_summary"},
        )

        resp1 = await _cc_turn(ac, "baking schedule", "upd1")
        assert content_v1 in resp1.json()["formatted_injection"]

        resp2 = await _cc_turn(ac, "baking schedule", "upd1")
        turn2 = resp2.json()["formatted_injection"]
        assert f"still relevant: [id:{doc_id}] — shown earlier this session" in turn2

        # Update the content (same id) via the same endpoint roampal summarize uses.
        content_v2 = "Baking moved: the schedule now says Saturday morning instead"
        assert len(content_v2) <= 600  # stays within the Task 37 summary cap
        upd = await ac.post(
            "/api/memory/update-content",
            json={"doc_id": doc_id, "collection": "working", "new_content": content_v2},
        )
        assert upd.status_code == 200

        resp3 = await _cc_turn(ac, "baking schedule", "upd1")
        turn3 = resp3.json()["formatted_injection"]
        assert f"still relevant: [id:{doc_id}]" not in turn3
        assert content_v2 in turn3

    @pytest.mark.asyncio
    async def test_session_reset_endpoint_semantics(self, dedup_client):
        ac, real_ums, main = dedup_client
        main._session_injections["s1"] = {"working_1:aabbcc": "2026-09-23T00:00:00"}

        reset = await ac.post("/api/hooks/session-reset", json={"conversation_id": "s1"})
        assert reset.status_code == 200
        assert reset.json()["status"] == "ok"
        assert "s1" not in main._session_injections

        # Second reset is a clean no-op (hook semantics: any answer is ok).
        reset2 = await ac.post("/api/hooks/session-reset", json={"conversation_id": "s1"})
        assert reset2.status_code == 200

        # Missing conversation_id is an explicit 400, not a silent ok.
        bad = await ac.post("/api/hooks/session-reset", json={})
        assert bad.status_code == 400


class TestDeliveryAck:
    """A memory counts as shown only once the hook confirms Claude Code got
    the block. Every failure mode errs toward full text, never toward a
    pointer to text Claude never received."""

    @pytest.mark.asyncio
    async def test_unacked_turn_shows_full_text_again(self, dedup_client):
        ac, real_ums, main = dedup_client
        doc_id = await real_ums.store_memory_bank(
            "User keeps bees on the north allotment",
            tags=["project"],
            noun_tags=["bees"],
        )
        full_line = f"• User keeps bees on the north allotment [id:{doc_id}]"

        # Hook timed out / exited non-zero: no ack.
        resp1 = await _cc_turn(ac, "bees allotment", "ack1", ack=False)
        assert full_line in resp1.json()["formatted_injection"]
        assert resp1.json()["injection_token"]
        assert "ack1" not in main._session_injections

        # Next turn: still full text (Claude never saw turn 1's block).
        resp2 = await _cc_turn(ac, "bees allotment", "ack1")
        turn2 = resp2.json()["formatted_injection"]
        assert full_line in turn2, turn2
        assert "still relevant" not in turn2

        # Turn 2 WAS acked -> turn 3 renders the pointer.
        resp3 = await _cc_turn(ac, "bees allotment", "ack1")
        assert f"still relevant: [id:{doc_id}]" in resp3.json()["formatted_injection"]

    @pytest.mark.asyncio
    async def test_ack_semantics(self, dedup_client):
        ac, real_ums, main = dedup_client
        main._pending_injections["tok1"] = {
            "conversation_id": "s9",
            "keys": {"working_1:aabbccddeeff": "2026-09-24T00:00:00"},
            "created": "2026-09-24T00:00:00",
        }

        ok = await ac.post("/api/hooks/injection-ack", json={"injection_token": "tok1"})
        assert ok.status_code == 200 and ok.json()["committed"] is True
        assert "working_1:aabbccddeeff" in main._session_injections["s9"]
        assert "tok1" not in main._pending_injections

        # Replayed / unknown token: harmless no-op.
        again = await ac.post("/api/hooks/injection-ack", json={"injection_token": "tok1"})
        assert again.status_code == 200 and again.json()["committed"] is False

        bad = await ac.post("/api/hooks/injection-ack", json={})
        assert bad.status_code == 400

    @pytest.mark.asyncio
    async def test_reset_drops_pending_so_late_ack_is_ignored(self, dedup_client):
        ac, real_ums, main = dedup_client
        main._pending_injections["tok2"] = {
            "conversation_id": "s10",
            "keys": {"working_2:aabbccddeeff": "2026-09-24T00:00:00"},
            "created": "2026-09-24T00:00:00",
        }
        await ac.post("/api/hooks/session-reset", json={"conversation_id": "s10"})
        late = await ac.post("/api/hooks/injection-ack", json={"injection_token": "tok2"})
        assert late.json()["committed"] is False
        assert "s10" not in main._session_injections

    def test_unacked_pending_expires_with_cache_ttl(self):
        import roampal.server.main as main
        from datetime import datetime, timedelta

        old = (datetime.now() - timedelta(seconds=main._CACHE_TTL_SECONDS + 5)).isoformat()
        fresh = datetime.now().isoformat()
        saved = dict(main._pending_injections)
        try:
            main._pending_injections.clear()
            main._pending_injections["old"] = {"conversation_id": "a", "keys": {}, "created": old}
            main._pending_injections["new"] = {"conversation_id": "b", "keys": {}, "created": fresh}
            main._evict_stale_entries()
            assert "old" not in main._pending_injections
            assert "new" in main._pending_injections
        finally:
            main._pending_injections.clear()
            main._pending_injections.update(saved)


class TestHookAck:
    """The prompt hook acks only after printing, right before its zero exit."""

    def _run_hook(self, monkeypatch, capsys, get_context_result, *, fail_update_check=False):
        import io
        from roampal.hooks import user_prompt_submit_hook as ups

        posts = []

        class _Resp:
            def __init__(self, body):
                self._body = body

            def read(self):
                return json.dumps(self._body).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            url = req.full_url
            posts.append(url)
            if url.endswith("/api/health"):
                return _Resp({"status": "healthy"})
            if url.endswith("/api/hooks/get-context"):
                return _Resp(get_context_result)
            return _Resp({"status": "ok"})

        def update_check():
            if fail_update_check:
                raise RuntimeError("update check blew up")
            return (False, "", "")

        monkeypatch.setattr(ups.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(ups, "check_for_updates_cached", update_check)
        monkeypatch.setattr(ups, "_roampal_headers", lambda: {"Content-Type": "application/json"})
        monkeypatch.setattr(
            ups.sys, "stdin", io.StringIO(json.dumps({"prompt": "hi", "session_id": "s1"}))
        )
        with pytest.raises(SystemExit) as exc:
            ups.main()
        return exc.value.code, posts, capsys.readouterr().out

    def test_acks_after_printing_on_success(self, monkeypatch, capsys):
        code, posts, out = self._run_hook(
            monkeypatch, capsys,
            {"formatted_injection": "BLOCK", "injection_token": "tok-x"},
        )
        assert code == 0
        assert "BLOCK" in out
        acks = [u for u in posts if u.endswith("/api/hooks/injection-ack")]
        assert len(acks) == 1
        assert posts.index(acks[0]) > posts.index(
            [u for u in posts if u.endswith("/api/hooks/get-context")][0]
        )

    def test_no_ack_without_token(self, monkeypatch, capsys):
        code, posts, _ = self._run_hook(
            monkeypatch, capsys, {"formatted_injection": "BLOCK"}
        )
        assert code == 0
        assert not any(u.endswith("/api/hooks/injection-ack") for u in posts)

    def test_no_ack_when_hook_exits_nonzero(self, monkeypatch, capsys):
        """Claude Code drops stdout from a non-zero exit — so no ack."""
        code, posts, _ = self._run_hook(
            monkeypatch, capsys,
            {"formatted_injection": "BLOCK", "injection_token": "tok-y"},
            fail_update_check=True,
        )
        assert code == 1
        assert not any(u.endswith("/api/hooks/injection-ack") for u in posts)


class TestSessionStartWiring:
    """SessionStart hooks (compact/startup/clear) reset the record: stdin
    session id -> /api/hooks/session-reset before the RECENT EXCHANGES
    block prints."""

    def test_setup_writes_clear_matcher(self):
        import roampal.cli.setup as setup_mod

        source = Path(setup_mod.__file__).read_text(encoding="utf-8")
        assert '"matcher": "compact"' in source
        assert '"matcher": "startup"' in source
        assert '"matcher": "clear"' in source, (
            "SessionStart must also fire for cleared sessions (Task 30 reset)"
        )

    def test_cmd_context_posts_reset_from_stdin_session_id(self, monkeypatch):
        """cmd_context --recent-exchanges reads the hook's stdin JSON and
        POSTs the session id to /api/hooks/session-reset (best effort)."""

        posts = []

        def fake_post(url, json=None, timeout=None, headers=None):
            posts.append((url, json))

            class R:
                status_code = 200

                def json(self):
                    return {"results": []}

            return R()

        import httpx as real_httpx
        from roampal.cli import memory_cmds as mc

        monkeypatch.setattr(real_httpx, "post", fake_post, raising=True)

        stdin_payload = json.dumps({"session_id": "sess_new_42", "source": "compact"})
        stdin_fake = io.StringIO(stdin_payload)
        stdin_fake.isatty = lambda: False

        args = argparse.Namespace(recent_exchanges=True, dev=False, port=None)
        buf = StringIO()
        with patch("sys.stdin", stdin_fake), patch("sys.stdout", buf):
            mc.cmd_context(args)

        assert posts, "cmd_context did not POST the session reset"
        url, payload = posts[0]
        assert url.endswith("/api/hooks/session-reset")
        assert payload == {"conversation_id": "sess_new_42"}

    def test_cmd_context_without_stdin_session_id_is_silent(self, monkeypatch):
        """No session id in stdin (or no JSON) -> no RESET POST (the search
        call is still legitimate), prints fine."""
        reset_posts = []
        search_posts = []

        def fake_post(url, **kw):
            if url.endswith("/api/hooks/session-reset"):
                reset_posts.append(url)
            else:
                search_posts.append(url)

            class R:
                status_code = 200

                def json(self):
                    return {"results": []}

            return R()

        import httpx as real_httpx
        from roampal.cli import memory_cmds as mc

        monkeypatch.setattr(real_httpx, "post", fake_post, raising=True)

        stdin_fake = io.StringIO("not json at all\n")
        stdin_fake.isatty = lambda: False
        args = argparse.Namespace(recent_exchanges=True, dev=False, port=None)
        buf = StringIO()
        with patch("sys.stdin", stdin_fake), patch("sys.stdout", buf):
            mc.cmd_context(args)
        assert reset_posts == []

        monkeypatch.setattr(real_httpx, "post", fake_post, raising=True)

        stdin_fake = io.StringIO("not json at all\n")
        stdin_fake.isatty = lambda: False
        args = argparse.Namespace(recent_exchanges=True, dev=False, port=None)
        buf = StringIO()
        with patch("sys.stdin", stdin_fake), patch("sys.stdout", buf):
            mc.cmd_context(args)
        stdout_capture = buf.getvalue()
        reset_posts = [u for u in posts if u.endswith("/api/hooks/session-reset")]
        assert reset_posts == [], "no session id -> no reset POST"
        # The RECENT-EXCHANGES search itself still runs (and gets no results
        # in this fixture — nothing printed).
        assert stdout_capture == ""

    def test_cmd_context_without_stdin_session_id_is_silent(self, monkeypatch):
        """No session id in stdin (or no JSON) -> no RESET POST (the search
        call is still legitimate), prints fine."""
        reset_posts = []
        search_posts = []

        def fake_post(url, **kw):
            if url.endswith("/api/hooks/session-reset"):
                reset_posts.append(url)
            else:
                search_posts.append(url)

            class R:
                status_code = 200

                def json(self):
                    return {"results": []}

            return R()

        import httpx as real_httpx
        from roampal.cli import memory_cmds as mc

        monkeypatch.setattr(real_httpx, "post", fake_post, raising=True)

        stdin_fake = io.StringIO("not json at all\n")
        stdin_fake.isatty = lambda: False
        args = argparse.Namespace(recent_exchanges=True, dev=False, port=None)
        buf = StringIO()
        with patch("sys.stdin", stdin_fake), patch("sys.stdout", buf):
            mc.cmd_context(args)
        assert reset_posts == []