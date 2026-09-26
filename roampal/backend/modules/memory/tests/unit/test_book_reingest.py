"""A removed book can be ingested again (real ChromaDB, fake embedder).

Found by the Task 34 live smoke (2026-09-26): `remove` ghosts a book's chunks
(v0.2.2, non-destructive) but store_book's v0.2.0 duplicate check matched ANY
chunk with the title — ghosted ones included — and returned those ids as if
stored. Once removed, a title could never be ingested again: the CLI printed
"Stored ... 1 chunks", `books` stayed empty. `remove` of an already-removed
title also reported success.
"""

import asyncio
import hashlib
import random

import pytest

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem


class _FakeEmbed:
    """md5-seeded unit vectors (768-dim) standing in for the ONNX model."""

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


def _titles(books):
    return [b["title"] for b in books]


@pytest.mark.asyncio
async def test_removed_book_can_be_ingested_again(tmp_path):
    ums = await _ums(tmp_path)
    first = await ums.store_book("Version one of the doc.", title="doc")
    assert _titles(await ums.list_books()) == ["doc"]

    assert (await ums.remove_book("doc"))["removed"] == len(first)
    assert await ums.list_books() == []

    second = await ums.store_book("Version two of the doc.", title="doc")
    assert second and not set(second) & set(first), "re-ingest returned the removed chunks"
    assert not set(second) & ums.ghost_ids
    assert _titles(await ums.list_books()) == ["doc"]


@pytest.mark.asyncio
async def test_live_duplicate_title_still_skipped(tmp_path):
    """The v0.2.0 duplicate guard still holds for a book that is present."""
    ums = await _ums(tmp_path)
    first = await ums.store_book("Some content.", title="dup")
    again = await ums.store_book("Other content.", title="dup")
    assert again == first


@pytest.mark.asyncio
async def test_removing_an_already_removed_book_is_not_found(tmp_path):
    ums = await _ums(tmp_path)
    await ums.store_book("Short doc.", title="gone")
    assert (await ums.remove_book("gone"))["removed"] == 1
    second = await ums.remove_book("gone")
    assert second["removed"] == 0
    assert "No book found" in second.get("message", "")


@pytest.mark.asyncio
async def test_reingest_survives_a_restart(tmp_path):
    """Ghost registry reloads from disk; the re-ingested copy stays visible."""
    ums = await _ums(tmp_path)
    await ums.store_book("Old.", title="doc")
    await ums.remove_book("doc")
    await ums.store_book("New.", title="doc")

    reopened = await _ums(tmp_path)
    assert _titles(await reopened.list_books()) == ["doc"]


@pytest.mark.asyncio
async def test_stats_book_count_excludes_removed_books(tmp_path):
    """Task 46: `stats` counted removed (ghosted) chunks, so it showed more
    books than `roampal books` listed. The count is now the live chunks, with
    the hidden ones reported separately."""
    ums = await _ums(tmp_path)
    await ums.store_book("Old version.", title="doc")
    assert ums.get_stats()["collections"]["books"] == {"count": 1}

    await ums.remove_book("doc")
    assert ums.get_stats()["collections"]["books"] == {"count": 0, "removed": 1}

    await ums.store_book("New version.", title="doc")
    assert ums.get_stats()["collections"]["books"] == {"count": 1, "removed": 1}
    assert len(await ums.list_books()) == 1
