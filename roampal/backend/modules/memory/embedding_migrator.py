"""Automatic background re-embedding after an embedder model-family change.

v0.5.9 Item 2a. When the embedder ONNX artifact changes (e.g. mpnet ->
e5-base), every stored vector must be recomputed so query and stored vectors
live in the same embedding space. The migration is:

- completed per collection: a collection is only marked migrated once every
  record has been re-embedded. Mid-flight a collection holds a mix of old and
  new vectors, but the per-collection marker and compare-and-swap make an
  interrupted run safely resumable.
- resumable (per-collection markers in embeddings_meta.json),
- safe against concurrent edits via compare-and-swap (a record rewritten
  mid-migration keeps its fresh vector),
- lock-guarded so the server and a concurrent `roampal reembed` never migrate
  the same profile at once.
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

META_FILENAME = "embeddings_meta.json"
LOCK_FILENAME = ".reembed.lock"
LOCK_STALE_SECONDS = 3600  # 1 hour
BATCH_SIZE = 32
DEFAULT_BATCH_SLEEP = 0.05

# Smaller collections first so the highest-value memory is searchable sooner.
COLLECTION_ORDER = ["memory_bank", "patterns", "working", "history", "books"]

# In-process reentrancy guard: the PID-liveness lock file cannot distinguish
# "another process" from "another coroutine in this process" (the UMS
# background task and a direct migrate_profile call share one PID). Without
# this, `roampal reembed` could race its own background scheduler for the lock
# and one of them would silently skip. KEYED BY RESOLVED LOCK PATH: the server
# is multi-profile, and profile A's migration on data_path_a must never lock
# profile B out of data_path_b (post-review round 3 fix, 2026-08-27 — the
# first cut used one global bool and did exactly that). Single event loop per
# process; the check-then-add below is await-free.
_in_process_migration_active: set = set()


def _lock_key(data_path) -> str:
    return str(_lock_path(data_path).resolve())


def _acquire_lock(data_path) -> bool:
    key = _lock_key(data_path)
    if key in _in_process_migration_active:
        return False
    p = _lock_path(data_path)
    if p.exists():
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            age = time.time() - info.get("started_at", 0)
            if age < LOCK_STALE_SECONDS and _pid_alive(info.get("pid")):
                return False
        except Exception:
            pass
        # stale or unreadable -> take over
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"pid": os.getpid(), "started_at": time.time()}),
            encoding="utf-8",
        )
        _in_process_migration_active.add(key)
        return True
    except Exception as e:
        logger.warning(f"[reembed] failed to acquire lock {p}: {e}")
        return False


def _release_lock(data_path):
    _in_process_migration_active.discard(_lock_key(data_path))
    try:
        _lock_path(data_path).unlink(missing_ok=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Metadata tracking
# --------------------------------------------------------------------------- #
def _meta_path(data_path: Path) -> Path:
    return Path(data_path) / META_FILENAME


def artifact_key(model: str, onnx_file: str) -> str:
    """Identity of the artifact that produced a collection's vectors.

    Must include the model repo, not just the ONNX filename: different repos
    can (and do) ship their INT8 export at the same relative path (e.g. both
    mpnet and e5-base use "onnx/model_qint8_avx512_vnni.onnx"), so filename
    alone can't distinguish a model-family change — found live 2026-08-26,
    where switching HF_REPO with the onnx_file unchanged left the migration
    detector blind to the swap and a collection silently mixed embedding
    spaces across a restart.
    """
    return f"{model}::{onnx_file}"


def read_meta(data_path) -> dict:
    p = _meta_path(Path(data_path))
    if not p.exists():
        return {}
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # corrupt meta -> treat as no meta
        logger.warning(f"[reembed] failed to read {p}: {e}")
        return {}
    # Upgrade pre-2026-08-26 meta files (per-collection value was a bare
    # onnx_file, no model) to the composite key in memory. Self-heals on the
    # next write_meta call; only correct if every collection really was
    # produced by the top-level "model" field, which was always true under
    # the old single-artifact-per-profile scheme.
    top_model = meta.get("model")
    if top_model:
        meta["collections"] = {
            c: (v if "::" in v else artifact_key(top_model, v))
            for c, v in meta.get("collections", {}).items()
        }
    return meta


def write_meta(data_path, model: str, onnx_file: str, collections: Dict[str, str]) -> dict:
    p = _meta_path(Path(data_path))
    meta = {
        "model": model,
        "onnx_file": onnx_file,
        "collections": collections,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(p)  # atomic on POSIX + Windows (same filesystem)
    return meta


def collection_needs_migration(meta: dict, collection: str, model: str, onnx_file: str) -> bool:
    return meta.get("collections", {}).get(collection) != artifact_key(model, onnx_file)


def meta_has_mismatch(meta: dict, model: str, onnx_file: str, collection_names: List[str]) -> bool:
    if not meta:
        return False
    return any(collection_needs_migration(meta, c, model, onnx_file) for c in collection_names)


# --------------------------------------------------------------------------- #
# Single-runner lock
# --------------------------------------------------------------------------- #
def _lock_path(data_path) -> Path:
    return Path(data_path) / LOCK_FILENAME


def _pid_alive(pid) -> bool:
    if pid is None:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


# --------------------------------------------------------------------------- #
# Core migration
# --------------------------------------------------------------------------- #
def _extract_text(metadata) -> str:
    # Mirror the write path: stored vectors are computed from text or content.
    m = metadata or {}
    return (m.get("text") or m.get("content") or "").strip()


async def _reembed_collection(ums, adapter, coll_name: str, onnx_file: str,
                              dry_run: bool = False) -> int:
    ids = adapter.list_all_ids()
    if not ids:
        return 0

    embed_fn = ums._embedding_service.embed_text
    dim = ums._embedding_service.get_embedding_dimension()
    batch_sleep = float(os.environ.get("ROAMPAL_REEMBED_BATCH_SLEEP", str(DEFAULT_BATCH_SLEEP)))

    updated = 0
    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start:start + BATCH_SIZE]
        records = adapter.collection.get(ids=batch_ids, include=["metadatas"])
        metas = records.get("metadatas", []) or []

        texts = [_extract_text(m) for m in metas]
        # Index-preserving embeds (per-record, so blank records stay aligned).
        vectors = []
        for t in texts:
            if t:
                vectors.append(await embed_fn(t, role="passage"))
            else:
                vectors.append([0.0] * dim)

        # Compare-and-swap: re-read; skip any record whose text changed
        # (e.g. a concurrent summarization/promotion rewrote it).
        recheck = adapter.collection.get(ids=batch_ids, include=["metadatas"])
        re_metas = recheck.get("metadatas", []) or []

        keep_ids, keep_vecs = [], []
        for j, bid in enumerate(batch_ids):
            rt = _extract_text(re_metas[j]) if j < len(re_metas) else None
            if rt == texts[j]:
                keep_ids.append(bid)
                keep_vecs.append(vectors[j])

        skipped = len(batch_ids) - len(keep_ids)
        if skipped:
            logger.info(f"[reembed] {coll_name}: {skipped} record(s) changed/deleted "
                        f"mid-batch, left unchanged")

        if keep_ids and not dry_run:
            # ChromaDB update() preserves documents + metadatas; a missing id
            # silently no-ops, so a deleted record costs nothing.
            adapter.collection.update(ids=keep_ids, embeddings=keep_vecs)
        updated += len(keep_ids)

        if batch_sleep:
            await asyncio.sleep(batch_sleep)

    return updated


async def migrate_profile(
    ums,
    data_path,
    *,
    model: str,
    onnx_file: str,
    force: bool = False,
    dry_run: bool = False,
    only_collection: Optional[str] = None,
) -> int:
    """Re-embed any collection whose stored vectors were produced by a different
    ONNX artifact. Returns the number of records updated."""
    data_path = Path(data_path)
    if not _acquire_lock(data_path):
        logger.info("[reembed] another process holds the lock; skipping")
        return 0
    total = 0
    try:
        collections = ums.collections
        order = [c for c in COLLECTION_ORDER if c in collections]
        if only_collection:
            order = [only_collection] if only_collection in collections else []
        # smallest-first within the requested set
        try:
            order.sort(key=lambda c: (collections[c].collection.count()
                                       if collections[c].collection else 0))
        except Exception:
            pass

        meta = read_meta(data_path)
        completed = dict(meta.get("collections", {}))

        for coll_name in order:
            adapter = collections[coll_name]
            if not force and not collection_needs_migration(meta, coll_name, model, onnx_file):
                logger.info(f"[reembed] {coll_name}: up to date, skipping")
                continue
            n = await _reembed_collection(ums, adapter, coll_name, onnx_file, dry_run=dry_run)
            total += n
            if not dry_run:
                # Only mark the collection migrated when vectors were actually
                # rewritten — a dry-run must leave embeddings_meta.json untouched
                # so a later real run still fires. (Post-review fix 2026-08-27:
                # this write used to be unconditional, so --dry-run silently
                # marked every needing-migration collection as done.)
                completed[coll_name] = artifact_key(model, onnx_file)
                write_meta(data_path, model, onnx_file, completed)
            logger.info(f"[reembed] {coll_name}: {n} record(s) "
                        f"{'would be ' if dry_run else ''}migrated")

        if not dry_run:
            write_meta(data_path, model, onnx_file, completed)
        logger.info(f"[reembed] done. {total} record(s) updated across {len(order)} collection(s)")
        return total
    finally:
        _release_lock(data_path)
