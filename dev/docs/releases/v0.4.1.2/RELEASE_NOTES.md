# v0.4.1.2 Hotfix Release Notes

**Platforms:** All (Core Backend)
**Theme:** Fix broken memory injection caused by BM25 index misalignment

---

## Overview

v0.4.1 introduced a bug in the BM25 hybrid search index that caused memory injection to return fewer than 4 memories. The root cause was an index-to-ID misalignment in the BM25 arrays, compounded by phantom entries in ChromaDB's HNSW vector index from TTL-deleted working memories. This hotfix removes BM25 entirely (it was an optional enhancement over pure vector search) and adds defensive filters for stale vector index entries.

---

## Critical

### 1. BM25 index-to-ID misalignment breaks memory injection
**File:** `chromadb_adapter.py`

**Problem:** v0.4.1 added `if doc` filtering to BM25 tokenization to handle None documents, but did not apply the same filter to `bm25_ids`, `bm25_docs`, and `bm25_metadatas` arrays. This caused BM25 index positions to shift — position N in the BM25 index no longer mapped to `bm25_ids[N]`. The RRF merge in `hybrid_query` then returned wrong IDs with empty content. The content filter in `get_context_for_injection` correctly dropped these, but the 4-slot allocation had no backfill, resulting in 2-3 memories injected instead of 4.

**Fix:** Removed BM25 hybrid search entirely. BM25 was an optional lexical supplement to vector search that caused more problems than it solved (OOM on large collections, index alignment bugs, optional dependency issues). Pure vector search is sufficient for the memory sizes involved. Removed: `rank_bm25` import, `_build_bm25_index()` method, BM25 instance variables, `hybrid` optional dependency group.

### 2. Phantom HNSW vector entries return None documents
**File:** `chromadb_adapter.py`

**Problem:** When working memories are deleted (24h TTL expiry), ChromaDB's HNSW vector index may retain stale entries. `query_vectors` returns these phantom results with `document=None` and `metadata=None`. They appear as valid search results but contain no content.

**Fix:** Added filter in `query_vectors` to skip results where both document and metadata are None/empty.

---

## Medium

### 3. Memory slot backfill when reserved slots are empty
**File:** `unified_memory_system.py`

**Problem:** The 4-slot allocation (1 working + 1 history + 2 best matches) did not backfill when a reserved slot returned empty. If the working or history search found no valid result, only 3 memories were injected.

**Fix:** Backfill from best matches to always reach 4 total, regardless of how many reserved slots are filled.

---

## Files Modified

| File | Changes |
|------|---------|
| `chromadb_adapter.py` | Removed BM25 import, `_build_bm25_index()`, BM25 instance vars, BM25 code in `hybrid_query`; added phantom entry filter in `query_vectors` |
| `unified_memory_system.py` | Slot allocation backfills to always inject 4 memories |
| `pyproject.toml` | Removed `[hybrid]` optional dependency group; version bump to 0.4.1.2 |
| `__init__.py` | Version bump to 0.4.1.2 |

---

## Verification

**484 tests passing. 0 failures.**

Tested 10 queries against live ChromaDB with stale HNSW entries — all returned exactly 4 memories (previously 2-3 on affected queries).

- [x] BM25 code fully removed (import, index builder, instance variables)
- [x] `hybrid_query` uses pure vector search
- [x] Phantom HNSW entries filtered in `query_vectors`
- [x] Slot allocation backfills to 4 regardless of reserved slot availability
- [x] `rank-bm25` removed from optional dependencies
