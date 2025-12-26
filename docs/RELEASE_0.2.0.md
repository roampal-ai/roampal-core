# roampal-core v0.2.0 Release Plan

## Pre-Release Checklist

- [ ] Bump version in `pyproject.toml` (0.1.10 → 0.2.0)
- [ ] Bump version in `roampal/__init__.py` (0.1.10 → 0.2.0)
- [ ] Run full test suite
- [ ] Test on fresh install

---

## CRITICAL: Action KG Not Syncing (P0)

**Problem:** `record_action_outcome()` updates never visible to SearchService.

**Symptoms:**
- Call `score_response(outcome="worked")` → action recorded
- SearchService queries `context_action_effectiveness` → empty
- Causal learning never takes effect

**Root Cause (Book-Informed Analysis):**

This is a classic **"Reading Your Own Writes"** bug from *Designing Data-Intensive Applications* (p.162):

> *"If the user views the data shortly after making a write, the new data may not yet have reached the replica."*

The architecture has **two copies of the same data**:

```
UMS.knowledge_graph (line 165)          ← record_action_outcome() writes HERE
         ↓ saves to disk
    knowledge_graph.json
         ↓ loaded at init (once)
_kg_service.knowledge_graph (line 110)  ← SearchService reads HERE (stale!)
```

When `record_action_outcome()` runs:
1. Updates `self.knowledge_graph["context_action_effectiveness"]` ✓
2. Calls `self._save_kg()` → writes to disk ✓
3. `_kg_service.knowledge_graph` never refreshed ✗

**Fix: Single Source of Truth**

Following the book's principle (p.366): *"All nodes decide to deliver the same messages in the same order... messages are not duplicated."*

**Option A (RECOMMENDED): Delegate to KG Service**
```python
# BEFORE: UMS maintains its own copy
self.knowledge_graph["context_action_effectiveness"][key] = stats
self._save_kg()

# AFTER: Single source of truth
await self._kg_service.record_action_outcome(key, stats)
# _kg_service handles save + in-memory update
```

**Option B: Read-Your-Own-Writes Pattern**
```python
# After saving, force reload
self._save_kg()
self._kg_service.reload_kg()  # Sync the two copies
```

**Files to Modify:**
- `unified_memory_system.py` - Remove `self.knowledge_graph`, delegate to `_kg_service`
- `knowledge_graph_service.py` - Add `record_action_outcome()` method
- `tests/integration/test_action_kg_sync.py` (new) - End-to-end test

**Why Tests Didn't Catch This:**

| Gap | Reason |
|-----|--------|
| Unit tests passed | Each component works in isolation |
| No integration test | No test covers write→read handoff |
| Design bug | Code works as written; design has two copies |

**New Integration Test to Add:**
```python
async def test_action_kg_visible_after_record():
    """Record action outcome → SearchService must see it (regression test)."""
    # 1. Record action via UMS
    await ums.record_action_outcome(ActionOutcome(...))

    # 2. Query via SearchService (uses _kg_service)
    stats = search_service.get_doc_action_stats("doc_id")

    # 3. Must see the recorded action
    assert stats is not None
    assert stats["total_uses"] >= 1
```

**Effort:** 3h (including integration test)

---

## Issue: Metadata Time Filters Don't Work

**Problem:** `search_memory(metadata={"timestamp": "2025-12-18"})` returns nothing.

**Root Cause (VERIFIED):**
1. Timestamps stored as ISO strings: `2025-11-30T12:24:53.312762`
2. Simple filter does exact string match → no match
3. **ChromaDB's `$gte`/`$lte` operators only work on NUMBERS, not strings**

```
>>> patterns.query(where={'timestamp': {'$gte': '2025-12-17'}})
Error: Expected operand value to be an int or a float for operator $gte, got 2025-12-17
```

**Current metadata structure in ChromaDB:**
```python
{
    'timestamp': '2025-11-30T12:24:53.312762',  # ISO string (NOT numeric)
    'last_used': '2025-12-15T07:54:22.132506',
    'score': 0.727,
    'uses': 9,
    'last_outcome': 'failed',
    'outcome_history': '[...]',
    'promoted_from': 'history',
    'collection': 'working',
    'source': 'mcp_claude_desktop_main'
}
```

---

## Fix Options Evaluated

### Option A: Add `timestamp_epoch` numeric field
- Store `int(datetime.timestamp())` alongside ISO string
- Enables `{"timestamp_epoch": {"$gte": 1702857600}}`
- **Requires migration of existing data**
- More complex

### Option B: Post-query filtering in Python ✓ RECOMMENDED
- Fetch more results than needed from ChromaDB
- Filter in Python after: `[r for r in results if r['timestamp'] >= '2025-12-18']`
- **ISO strings sort correctly alphabetically** (`'2025-12-17' < '2025-12-18'`)
- No migration needed
- Simple to implement

---

## Fix 1: Post-Query Date Filtering

**Location:** `roampal/backend/modules/memory/search_service.py`

**Approach:** Intercept date-related metadata filters, remove from ChromaDB query, apply as post-filter.

```python
# In search_service.py

DATE_FIELDS = ('timestamp', 'last_used', 'created_at')

def _extract_date_filters(self, filters: Optional[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Separate date filters (for post-query) from other filters (for ChromaDB).

    Returns:
        (chromadb_filters, date_filters)
    """
    if not filters:
        return None, None

    chromadb_filters = {}
    date_filters = {}

    for key, value in filters.items():
        if key in DATE_FIELDS:
            date_filters[key] = value
        else:
            chromadb_filters[key] = value

    return chromadb_filters or None, date_filters or None


def _apply_date_filters(self, results: List[Dict], date_filters: Dict) -> List[Dict]:
    """Apply date filtering in Python (ISO strings sort alphabetically)."""
    filtered = results

    for field, condition in date_filters.items():
        if isinstance(condition, str):
            # Exact date: "2025-12-18" matches "2025-12-18T..."
            if len(condition) == 10:  # YYYY-MM-DD
                filtered = [r for r in filtered
                           if r.get('metadata', {}).get(field, '').startswith(condition)]
            else:
                # Full timestamp match
                filtered = [r for r in filtered
                           if r.get('metadata', {}).get(field) == condition]

        elif isinstance(condition, dict):
            # Range operators
            for op, val in condition.items():
                if op == '$gte':
                    filtered = [r for r in filtered
                               if r.get('metadata', {}).get(field, '') >= val]
                elif op == '$gt':
                    filtered = [r for r in filtered
                               if r.get('metadata', {}).get(field, '') > val]
                elif op == '$lte':
                    filtered = [r for r in filtered
                               if r.get('metadata', {}).get(field, '') <= val]
                elif op == '$lt':
                    filtered = [r for r in filtered
                               if r.get('metadata', {}).get(field, '') < val]

    return filtered
```

**Usage in search():**
```python
async def search(self, query, metadata_filters=None, ...):
    # Separate date filters from ChromaDB filters
    chromadb_filters, date_filters = self._extract_date_filters(metadata_filters)

    # Fetch more results if we'll be post-filtering
    fetch_limit = limit * 3 if date_filters else limit

    # Search with non-date filters only
    results = await self._search_collections(
        query_embedding, processed_query, collections, fetch_limit, chromadb_filters
    )

    # Apply date filters in Python
    if date_filters:
        results = self._apply_date_filters(results, date_filters)

    # Continue with scoring, reranking, pagination...
```

---

## Fix 2: Add `sort_by` Parameter to search_memory

**Problem:** Semantic search returns by relevance, not recency. When user asks "what was the last thing we did", recency should dominate.

**Location:** `roampal/mcp/server.py` - search_memory tool

**Change:** Add optional `sort_by` parameter.

```python
# Tool schema addition:
"sort_by": {
    "type": "string",
    "enum": ["relevance", "recency", "score"],
    "description": "Sort order. 'recency' for temporal queries like 'last thing we did'",
    "default": "relevance"
}
```

```python
# In search handler:
if sort_by == "recency":
    results.sort(key=lambda x: x.get('metadata', {}).get('timestamp', ''), reverse=True)
elif sort_by == "score":
    results.sort(key=lambda x: x.get('metadata', {}).get('score', 0.5), reverse=True)
# else: keep semantic relevance order (default)
```

---

## Fix 3: Auto-Detect Temporal Queries

**Problem:** LLM shouldn't have to manually specify `sort_by=recency`.

**Location:** `roampal/mcp/server.py`

**Change:** Detect temporal keywords and auto-apply recency sort.

```python
TEMPORAL_KEYWORDS = ['last', 'recent', 'previous', 'yesterday', 'today', 'earlier', 'before', 'latest']

def _is_temporal_query(query: str) -> bool:
    query_lower = query.lower()
    return any(kw in query_lower for kw in TEMPORAL_KEYWORDS)

# In search_memory handler:
if sort_by is None and _is_temporal_query(query):
    sort_by = "recency"
```

---

## Testing

```python
# Test 1: Exact date filter
results = search_memory("session", metadata={"timestamp": "2025-12-17"})
assert len(results) > 0
assert all(r['metadata']['timestamp'].startswith('2025-12-17') for r in results)

# Test 2: Date range filter
results = search_memory("work", metadata={"timestamp": {"$gte": "2025-12-15"}})
assert all(r['metadata']['timestamp'] >= '2025-12-15' for r in results)

# Test 3: Recency sort auto-detection
results = search_memory("last thing we worked on")
timestamps = [r['metadata']['timestamp'] for r in results]
assert timestamps == sorted(timestamps, reverse=True)  # Most recent first

# Test 4: Explicit sort_by
results = search_memory("work", sort_by="recency")
timestamps = [r['metadata']['timestamp'] for r in results]
assert timestamps == sorted(timestamps, reverse=True)
```

---

## Migration Notes

- No breaking changes
- No data migration needed (post-query filtering works with existing ISO strings)
- Existing queries continue to work
- New filter syntax is additive

---

## Files to Modify

1. `roampal/backend/modules/memory/search_service.py`
   - Add `_extract_date_filters()` method
   - Add `_apply_date_filters()` method
   - Modify `search()` to use post-query filtering

2. `roampal/mcp/server.py`
   - Add `sort_by` parameter to search_memory tool schema
   - Add `_is_temporal_query()` helper
   - Apply sorting based on `sort_by` or auto-detection

3. `tests/unit/test_search_service.py`
   - Add date filter unit tests
   - Add sort_by tests

---

## Summary

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Date filter returns nothing | ChromaDB `$gte`/`$lte` don't work on strings | Post-query filtering in Python |
| "Last thing" returns wrong order | Semantic search prioritizes relevance | `sort_by=recency` + auto-detection |

---

# Book Ingestion Audit

**Status:** Retrieval working (verified via test)

## Current Architecture

### Entry Points
| Method | Location | Status |
|--------|----------|--------|
| CLI | `roampal ingest <file>` | Working |
| API | `POST /api/ingest` | Working |
| MCP | No direct tool | Search only |

### Supported Formats
- `.txt`, `.md` - Direct UTF-8 read
- `.pdf` - Via `pypdf` (optional dependency)
- **Not supported:** `.docx`, `.epub`, `.html`

### Storage
- Collection: `roampal_books`
- ID format: `book_<uuid>_chunk_<index>`
- Metadata: `title`, `source`, `chunk_index`, `total_chunks`, `created_at`
- **Special:** Books exempt from outcome scoring (static reference material)

---

## Pending Issues

### Issue 1: No Batch Embedding (10-50x slower)
**Location:** `unified_memory_system.py:727`

**Problem:** Embeds chunks one at a time in a loop instead of batching.

```python
# CURRENT (slow)
for i, chunk in enumerate(chunks):
    embedding = await self.embedding_service.embed_text(chunk)  # One at a time
    ...

# FIX (fast)
embeddings = await self.embedding_service.embed_texts(chunks)  # Batch all
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    ...
```

**Impact:** 100-page PDF takes 30+ seconds instead of 3 seconds.

---

### Issue 2: Character-Based Chunking Breaks Semantics
**Location:** `unified_memory_system.py:711-720`

**Problem:** Chunks split at character count, not sentence boundaries.

```python
# CURRENT
while start < len(content):
    end = start + chunk_size
    chunk = content[start:end]  # May cut mid-sentence
    ...
```

**Example:** Chunk ends with "The solution is to use..." (no conclusion)

**Fix:** Use sentence tokenization with NLTK (already a dependency).

```python
import nltk
sentences = nltk.sent_tokenize(content)

chunks = []
current_chunk = []
current_size = 0

for sentence in sentences:
    if current_size + len(sentence) > chunk_size and current_chunk:
        chunks.append(' '.join(current_chunk))
        # Keep overlap sentences
        overlap_sentences = []
        overlap_size = 0
        for s in reversed(current_chunk):
            if overlap_size + len(s) <= chunk_overlap:
                overlap_sentences.insert(0, s)
                overlap_size += len(s)
            else:
                break
        current_chunk = overlap_sentences
        current_size = overlap_size
    current_chunk.append(sentence)
    current_size += len(sentence)

if current_chunk:
    chunks.append(' '.join(current_chunk))
```

---

### Issue 3: No Duplicate Detection
**Location:** `unified_memory_system.py:store_book()`

**Problem:** Ingesting same document twice creates duplicate chunks.

**Fix:** Add content hash check before storage.

```python
import hashlib

def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]

async def store_book(self, content, title, ...):
    content_hash = self._content_hash(content)

    # Check for existing book with same hash
    existing = await self.collections["books"].get(
        where={"content_hash": content_hash},
        limit=1
    )
    if existing and existing.get("ids"):
        return {"status": "duplicate", "existing_title": existing["metadatas"][0]["title"]}

    # Store with hash in metadata
    metadata["content_hash"] = content_hash
    ...
```

---

### Issue 4: No File Size Limits
**Location:** `cli.py:cmd_ingest()`

**Problem:** Can ingest arbitrarily large files → OOM crashes.

**Fix:** Add size check at CLI entry.

```python
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def cmd_ingest(file, ...):
    file_path = Path(file)
    if file_path.stat().st_size > MAX_FILE_SIZE:
        console.print(f"[red]File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)[/red]")
        return
```

---

### Issue 5: No Transaction Rollback
**Location:** `unified_memory_system.py:store_book()` upsert loop

**Problem:** If embedding fails mid-batch, partial book stored.

**Fix:** Batch all embeddings first, then upsert atomically.

```python
async def store_book(self, content, title, ...):
    chunks = self._chunk_content(content, chunk_size, chunk_overlap)

    # Embed ALL chunks first (fail fast)
    try:
        embeddings = await self.embedding_service.embed_texts(chunks)
    except Exception as e:
        return {"status": "error", "message": f"Embedding failed: {e}"}

    # Then upsert all at once
    doc_ids = [f"book_{uuid4().hex[:8]}_chunk_{i}" for i in range(len(chunks))]
    await self.collections["books"].upsert_vectors(
        ids=doc_ids,
        vectors=embeddings,
        metadatas=[...for each chunk...]
    )
```

---

## Medium Priority Issues

### Issue 6: PDF Extraction Loses Formatting
**Location:** `cli.py:458-461`

**Problem:** `pypdf.extract_text()` loses tables, columns, headers.

**Fix:** Consider `pymupdf` for layout-aware extraction in v0.3.

---

### Issue 8: `--dev` Flag Ignored in Remove Command
**Location:** `cli.py:562`

**Problem:** `cmd_remove()` always uses port 27182, ignores dev mode.

**Fix:** Pass dev flag to port selection.

---

### Issue 9: Book Title Not Shown in Search Results
**Problem:** MCP search returns book content but no indication of source.

**Fix:** Include title in formatted output.

```python
# In server.py search result formatting
if collection == "books":
    title = metadata.get("title", "Unknown")
    meta_parts.append(f"from: {title}")
```

---

## Files to Modify

0. `roampal/backend/modules/memory/chromadb_adapter.py` **(PRIORITY)**
   - Line 225-228: Add `embedding_function=None` to collection refresh

1. `roampal/backend/modules/memory/unified_memory_system.py`
   - `store_book()`: Batch embedding, sentence chunking, duplicate detection
   - Add `_content_hash()` helper
   - Add `_chunk_by_sentences()` helper

2. `roampal/cli.py`
   - `cmd_ingest()`: Add file size validation
   - `cmd_remove()`: Fix dev mode port

3. `roampal/mcp/server.py`
   - Search result formatting: Show book title

4. `tests/unit/test_book_ingestion.py` (new)
   - Batch embedding test
   - Sentence chunking test
   - Duplicate detection test
   - Large file handling test

---

## Book Ingestion Summary

| Issue | Impact | Fix Effort |
|-------|--------|------------|
| No batch embedding | 10-50x slower | 2h |
| Character chunking | Poor retrieval | 4h |
| No duplicate detection | Wasted storage | 2h |
| No file size limits | OOM crashes | 1h |
| No transaction rollback | Corrupted data | 2h |
| PDF loses formatting | Bad extraction | 6h (v0.3) |
| Dev flag ignored | Can't remove dev books | 30m |
| No title in search | No source context | 1h |

---

## Server Robustness: Auto-Restart on Crash

**Problem:** Server dies silently → user gets no response → has to manually restart.

**Impact:** Users hit silent failures, think roampal is broken.

**Root Cause:** uvicorn process dies with no watchdog. MCP doesn't know server is dead until timeout.

**Current flow:**
```
MCP tool called → hit HTTP endpoint → if server dead → timeout/error
```

**Fixed flow:**
```
MCP tool called → check health → if dead → start server → retry → hit endpoint
```

**Implementation:**

**Location:** `roampal/mcp/server.py`

```python
import subprocess
import time

def ensure_server_running(port: int, timeout: float = 10.0) -> bool:
    """Check if server is up, start it if not."""
    health_url = f"http://127.0.0.1:{port}/api/health"

    # Check if already running
    try:
        resp = httpx.get(health_url, timeout=2.0)
        if resp.status_code == 200:
            return True
    except:
        pass

    # Not running - start it
    logger.info(f"Server not responding, starting on port {port}...")
    env = os.environ.copy()
    subprocess.Popen(
        [sys.executable, "-m", "roampal.cli", "start", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True  # Detach from parent
    )

    # Wait for startup
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(health_url, timeout=1.0)
            if resp.status_code == 200:
                logger.info("Server started successfully")
                return True
        except:
            pass
        time.sleep(0.5)

    logger.error("Failed to start server")
    return False
```

**Usage (wrap each MCP tool):**
```python
# Before any HTTP call to the server
if not ensure_server_running(get_port()):
    return {"error": "Could not start server"}
```

**Files to Modify:**
- `roampal/mcp/server.py` - Add `ensure_server_running()` helper
- `roampal/mcp/server.py` - Wrap tool handlers with health check before HTTP calls

**Effort:** 30 min

**Priority:** P1 - users hitting silent failures is bad UX

---

# Merged from v0.1.11

The following fixes from v0.1.11 are included in v0.2.0:

## Critical Fixes

| Fix | Impact | Files |
|-----|--------|-------|
| Remove hardcoded name from cold-start query | All users get correct identity query | `server/main.py` |
| Preserve env section in `roampal init` | DEV users won't lose config | `cli.py` |
| Add `--dev` flag for init/server | Fresh DEV setup works | `cli.py`, `mcp/server.py` |
| Fix `embedding_function=None` in collection refresh | Books search returns results | `chromadb_adapter.py` |
| Fix infinite loop on small file chunks | Prevents hang on book ingestion | `unified_memory_system.py` |

## Test Coverage Added

| Test File | Count | Coverage |
|-----------|-------|----------|
| `test_embedding_service.py` | 9 | Text embedding, batch ops, prewarm |
| `test_context_service.py` | 11 | Context analysis, continuity detection |
| `test_routing_service.py` | 12 | Query routing, acronym expansion |
| `test_promotion_service.py` | 11 | Promotion/demotion, batch ops |
| `test_server_main.py` | 15 | Cold-start query, PII guard |
| `test_unified_memory_system.py` | 5 | store_book chunking |
| `test_chromadb_integration.py` | 18 | Real ChromaDB operations |

**Total tests:** 252 (172 existing + 80 new)

---

# v0.2.0 Summary

| Priority | Issue | Fix | Effort |
|----------|-------|-----|--------|
| **P0** | Action KG not syncing | Single source of truth | 3h |
| P1 | Date filter returns nothing | Post-query filtering | 2h |
| P1 | "Last thing" wrong order | sort_by + auto-detect | 1h |
| P1 | Server dies silently | Auto-restart | 30m |
| P2 | Book embedding slow | Batch embedding | 2h |
| P2 | Character chunking | Sentence-based | 4h |
| P2 | Duplicate books | Content hash | 2h |
| P2 | Large file OOM | Size limits + rollback | 3h |
| P3 | --dev flag in remove | Pass dev flag | 30m |
| P3 | No book title in search | Add to meta | 1h |

**Total estimated:** ~19h
