"""Task 45: one stored summary per OpenCode exchange.

Found in the Task 34 live check (2026-09-26): every OpenCode exchange summary
was stored twice — the plugin sent it with its scoring call
(/api/record-outcome stores it as a new working memory when no exchange doc
exists) and again via tryStoreSummary -> /api/hooks/stop. The stop hook now
merges into an identical summary stored moments ago in the same
conversation, and the plugin no longer sends the summary with its scores.
"""

import asyncio
import hashlib
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem
from roampal.server.main import SUMMARY_MERGE_WINDOW_SECONDS, _merge_into_recent_summary

PLUGIN = Path(__file__).resolve().parents[5] / "plugins" / "opencode" / "roampal.ts"


class _FakeEmbed:
    def _vec(self, text):
        rng = random.Random(int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big"))
        v = [rng.gauss(0.0, 1.0) for _ in range(768)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    async def embed_text(self, text, role="passage"):
        return self._vec(text)

    async def embed_texts(self, texts, role="passage"):
        return [self._vec(t) for t in texts]

    async def prewarm(self):
        return None


@pytest.fixture(autouse=True)
async def _cancel_warmup_tasks():
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


async def _ums(tmp_path):
    ums = UnifiedMemorySystem(data_path=str(tmp_path / "data"), embed_service=_FakeEmbed())
    await ums.initialize()
    return ums


SUMMARY = "User asked for a quick test; I replied without running anything."


async def _record_outcome_copy(ums, conv="ses_1", text=SUMMARY):
    """What /api/record-outcome's no-exchange-doc branch stores (copy 1)."""
    return await ums.store_working(
        content=text,
        conversation_id=conv,
        metadata={"memory_type": "exchange_summary", "exchange_outcome": "worked", "turn_type": "exchange"},
        noun_tags=["test"],
    )


def _stop_metadata():
    return {"memory_type": "exchange_summary", "timestamp": datetime.now().isoformat(),
            "exchange_fingerprint": "79795dcb", "sidecar_outcome": "worked"}


@pytest.mark.asyncio
async def test_identical_summary_merges_into_the_first_copy(tmp_path):
    ums = await _ums(tmp_path)
    first = await _record_outcome_copy(ums)
    merged = await _merge_into_recent_summary(ums, "ses_1", SUMMARY, _stop_metadata())
    assert merged == first
    meta = ums.collections["working"].get_fragment(first)["metadata"]
    assert meta["exchange_fingerprint"] == "79795dcb" and meta["sidecar_outcome"] == "worked"
    assert await ums.collections["working"].get_collection_count() == 1


@pytest.mark.asyncio
async def test_different_text_or_conversation_is_stored_normally(tmp_path):
    ums = await _ums(tmp_path)
    await _record_outcome_copy(ums)
    assert await _merge_into_recent_summary(ums, "ses_1", "A different summary.", _stop_metadata()) is None
    assert await _merge_into_recent_summary(ums, "ses_2", SUMMARY, _stop_metadata()) is None


@pytest.mark.asyncio
async def test_an_old_identical_summary_is_not_merged(tmp_path):
    ums = await _ums(tmp_path)
    first = await _record_outcome_copy(ums)
    old = (datetime.now() - timedelta(seconds=SUMMARY_MERGE_WINDOW_SECONDS + 60)).isoformat()
    ums.collections["working"].update_fragment_metadata(first, {"created_at": old})
    assert await _merge_into_recent_summary(ums, "ses_1", SUMMARY, _stop_metadata()) is None


@pytest.mark.asyncio
async def test_non_summary_memory_with_same_text_is_ignored(tmp_path):
    ums = await _ums(tmp_path)
    await ums.store_working(content=SUMMARY, conversation_id="ses_1",
                            metadata={"memory_type": "fact"}, noun_tags=["test"])
    assert await _merge_into_recent_summary(ums, "ses_1", SUMMARY, _stop_metadata()) is None


def _record_outcome_bodies(src):
    bodies, i = [], src.find("/record-outcome`")
    while i != -1:
        bodies.append(src[i: src.index("signal", i) if "signal" in src[i:i + 900] else i + 900])
        i = src.find("/record-outcome`", i + 1)
    return bodies


def _sends_summary(src):
    return [b for b in _record_outcome_bodies(src) if re.search(r"^\s*exchange_summary\s*:", b, re.M)]


def test_plugin_scoring_calls_no_longer_send_the_summary():
    src = PLUGIN.read_text(encoding="utf-8")
    assert len(_record_outcome_bodies(src)) >= 3
    assert not _sends_summary(src)


def test_stop_hook_uses_the_merge_before_storing():
    src = (Path(__file__).resolve().parents[5] / "server" / "main.py").read_text(encoding="utf-8")
    i = src.index("OpenCode path (sidecar): summary-only storage")
    block = src[i: src.index("Sidecar summary {doc_id} stored", i)]
    assert block.index("_merge_into_recent_summary(") < block.index("_memory.store_working(")
