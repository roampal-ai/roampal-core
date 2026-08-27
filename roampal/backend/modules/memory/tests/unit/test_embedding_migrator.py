"""
Tests for embedding_migrator (v0.5.9 Item 2a): per-collection migration state,
composite model::onnx artifact keying, legacy meta upgrade, compare-and-swap,
the single-runner lock — and the dry-run guarantee that embeddings_meta.json
is never touched (post-review fix 2026-08-27).
"""

import json
import os
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from roampal.backend.modules.memory import embedding_migrator as em


class FakeChromaCollection:
    """Mimics the chromadb collection surface _reembed_collection uses."""

    def __init__(self, metas):
        self._metas = dict(metas)
        self.update_calls = []
        self.get_calls = 0
        self.collection = MagicMock()
        self.collection.count = MagicMock(return_value=len(self._metas))
        self.collection.get = MagicMock(side_effect=self._get)
        self.collection.update = MagicMock(side_effect=self._update)

    def list_all_ids(self):
        return list(self._metas.keys())

    def _get(self, ids=None, include=None):
        self.get_calls += 1
        return {"ids": list(ids), "metadatas": [dict(self._metas[i]) for i in ids]}

    def _update(self, ids=None, embeddings=None, **kw):
        self.update_calls.append((list(ids), embeddings))


def _make_ums(metas, dim=4):
    ums = MagicMock()
    adapter = FakeChromaCollection(metas)
    ums.collections = {"working": adapter}
    ums._embedding_service.embed_text = AsyncMock(
        return_value=[0.1] * dim)
    ums._embedding_service.get_embedding_dimension = MagicMock(return_value=dim)
    return ums, adapter


@pytest.fixture
def data_path(tmp_path):
    return tmp_path


class TestArtifactKey:
    def test_composite_format(self):
        assert em.artifact_key("repoA", "onnx/m.onnx") == "repoA::onnx/m.onnx"


class TestNeedsMigration:
    META = {"collections": {"working": "repoA::onnx/model_qint8_avx512_vnni.onnx"}}

    def test_same_model_and_file_is_current(self):
        assert em.collection_needs_migration(
            self.META, "working", "repoA", "onnx/model_qint8_avx512_vnni.onnx") is False

    def test_same_filename_different_model_detected(self):
        """The 2026-08-26 blind spot: mpnet and e5-base INT8 share one relative
        ONNX path. Filename-only comparison must never come back."""
        assert em.collection_needs_migration(
            self.META, "working", "repoB", "onnx/model_qint8_avx512_vnni.onnx") is True

    def test_missing_collection_needs_migration(self):
        assert em.collection_needs_migration(
            self.META, "history", "repoA", "onnx/model_qint8_avx512_vnni.onnx") is True

    def test_empty_meta_needs_migration(self):
        assert em.collection_needs_migration({}, "working", "repoA", "onnx/x.onnx") is True


class TestMetaUpgrade:
    def test_legacy_bare_filenames_upgraded_in_memory(self, data_path):
        (data_path / em.META_FILENAME).write_text(json.dumps({
            "model": "repoA", "onnx_file": "onnx/model_qint8_avx512_vnni.onnx",
            "collections": {"working": "onnx/model_qint8_avx512_vnni.onnx"},
        }), encoding="utf-8")
        meta = em.read_meta(data_path)
        assert meta["collections"]["working"] == \
            "repoA::onnx/model_qint8_avx512_vnni.onnx"

    def test_composite_values_pass_through(self, data_path):
        key = "repoA::onnx/x.onnx"
        (data_path / em.META_FILENAME).write_text(json.dumps({
            "model": "repoA", "collections": {"working": key},
        }), encoding="utf-8")
        assert em.read_meta(data_path)["collections"]["working"] == key

    def test_corrupt_meta_treated_as_empty(self, data_path):
        (data_path / em.META_FILENAME).write_text("{not json", encoding="utf-8")
        assert em.read_meta(data_path) == {}


class TestDryRun:
    async def test_dry_run_writes_no_meta_and_no_vectors(self, data_path):
        """--dry-run must not mark collections migrated (2026-08-27 fix: the
        per-collection write_meta used to fire unconditionally)."""
        metas = {"a": {"text": "hello"}, "b": {"content": "world"}}
        ums, adapter = _make_ums(metas)
        em.write_meta(data_path, "old-repo", "onnx/old.onnx", {})

        n = await em.migrate_profile(
            ums, data_path, model="new-repo", onnx_file="onnx/new.onnx",
            dry_run=True)

        assert n == 2
        assert adapter.update_calls == []
        meta = em.read_meta(data_path)
        assert em.collection_needs_migration(
            meta, "working", "new-repo", "onnx/new.onnx") is True
        assert meta["model"] == "old-repo"

    async def test_real_run_writes_meta_and_vectors(self, data_path):
        metas = {"a": {"text": "hello"}, "b": {"content": "world"}}
        ums, adapter = _make_ums(metas)

        n = await em.migrate_profile(
            ums, data_path, model="new-repo", onnx_file="onnx/new.onnx")

        assert n == 2
        assert len(adapter.update_calls) == 1
        updated_ids, updated_vecs = adapter.update_calls[0]
        assert sorted(updated_ids) == ["a", "b"]
        assert len(updated_vecs) == 2
        meta = em.read_meta(data_path)
        assert meta["collections"]["working"] == "new-repo::onnx/new.onnx"

    async def test_force_reruns_current_collection(self, data_path):
        metas = {"a": {"text": "hello"}}
        ums, adapter = _make_ums(metas)
        em.write_meta(data_path, "new-repo", "onnx/new.onnx",
                      {"working": em.artifact_key("new-repo", "onnx/new.onnx")})

        n = await em.migrate_profile(
            ums, data_path, model="new-repo", onnx_file="onnx/new.onnx", force=True)
        assert n == 1
        assert len(adapter.update_calls) == 1

        n2 = await em.migrate_profile(
            ums, data_path, model="new-repo", onnx_file="onnx/new.onnx", force=False)
        assert n2 == 0


class TestCompareAndSwap:
    async def test_record_rewritten_mid_batch_is_skipped(self, data_path):
        """A record whose text changes between the first read and the CAS
        re-read keeps its fresh vector — it must not be in the update."""
        ums, adapter = _make_ums({"a": {"text": "v1"}, "b": {"text": "stable"}})

        original_get = adapter.collection.get.side_effect

        def get_with_race(ids=None, include=None):
            result = original_get(ids=ids, include=include)
            if "a" in (ids or []):
                # Concurrent writer rewrites record a between read and recheck.
                adapter._metas["a"] = {"text": "v2-rewritten"}
            return result

        adapter.collection.get = MagicMock(side_effect=get_with_race)

        n = await em.migrate_profile(
            ums, data_path, model="new-repo", onnx_file="onnx/new.onnx")

        assert n == 1
        updated_ids = adapter.update_calls[0][0]
        assert updated_ids == ["b"]


class TestSingleRunnerLock:
    def test_second_acquire_blocked_in_process(self, data_path):
        """In-process reentrancy guard (2026-08-27): two coroutines share one
        PID, so the PID-liveness lock file alone cannot serialize them."""
        assert em._acquire_lock(data_path) is True
        assert em._acquire_lock(data_path) is False
        em._release_lock(data_path)
        assert em._acquire_lock(data_path) is True
        em._release_lock(data_path)

    def test_two_profiles_lock_independently(self, tmp_path):
        """Round-3 fix (2026-08-27): the in-process guard is keyed by lock
        path, not global — profile A migrating must never lock profile B
        (different data_path) out of its own migration."""
        pa = tmp_path / "profile_a"
        pb = tmp_path / "profile_b"
        pa.mkdir()
        pb.mkdir()

        assert em._acquire_lock(pa) is True
        assert em._acquire_lock(pb) is True   # different profile -> allowed
        assert em._acquire_lock(pa) is False  # same profile -> still guarded
        em._release_lock(pa)
        assert em._acquire_lock(pa) is True   # freed while pb still held
        em._release_lock(pb)
        em._release_lock(pa)

    def test_live_pid_blocks(self, data_path):
        (data_path / em.LOCK_FILENAME).write_text(json.dumps(
            {"pid": os.getpid(), "started_at": time.time()}), encoding="utf-8")
        assert em._acquire_lock(data_path) is False
        em._release_lock(data_path)

    def test_stale_lock_taken_over(self, data_path):
        (data_path / em.LOCK_FILENAME).write_text(json.dumps(
            {"pid": 999999, "started_at": time.time() - em.LOCK_STALE_SECONDS - 10}),
            encoding="utf-8")
        assert em._acquire_lock(data_path) is True
        em._release_lock(data_path)
        assert not (data_path / em.LOCK_FILENAME).exists()

    async def test_migrate_profile_returns_zero_when_locked(self, data_path):
        """Locked-out runner returns 0 and must NOT release the holder's lock
        (the early return happens before the try/finally)."""
        ums, adapter = _make_ums({"a": {"text": "hello"}})
        assert em._acquire_lock(data_path) is True
        n = await em.migrate_profile(
            ums, data_path, model="new-repo", onnx_file="onnx/new.onnx")
        assert n == 0
        assert adapter.update_calls == []
        assert (data_path / em.LOCK_FILENAME).exists()
        em._release_lock(data_path)
