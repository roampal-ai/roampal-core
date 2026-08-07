# Roampal Core v0.5.8

**Release Date:** 2026-07-30
**Type:** Critical data integrity fix (SQLite WAL journaling)
**Triggered by:** Repeated hard server crashes (May 18–26, 2026) causing ChromaDB SQLite catalog corruption with total metadata loss while binary segment files survived on disk.

## Summary

ChromaDB's embedded SQLite catalog (`chroma.sqlite3`) was running in default `DELETE` journal mode. Any hard process termination mid-write (port conflict, external kill, system crash) corrupted the SQLite header/catalog tables, disconnecting all collections from their binary segment files. The fix switches to WAL journal mode with FULL durability at client initialization time.

---

## Item 1 — ChromaDB SQLite catalog corruption on hard crash

### Root cause

`PersistentClient(path=self.db_path)` in `chromadb_adapter.py:140` creates an embedded SQLite database using ChromaDB's default settings. The SQLite journal mode defaults to `DELETE`, which means:
- During a write transaction, changes go into a temporary rollback journal file
- If the process is killed mid-write (e.g., external port conflict on 27182), the journal is deleted but the main DB may have partial writes
- The catalog tables (`segments`, `collections`) become inconsistent with the binary HNSW segment files

### Corruption mechanism observed

On a prod install after ~30 days of usage:
- SQLite `segments` table reported 14 ghost entries across 5 collections (roampal_books, working, history, patterns, memory_bank)
- Binary `.bin` segment directories (~77MB total across ~9 UUID dirs) existed on disk with valid HNSW indexes
- The metadata layer had lost all document text content — `embedding_metadata_array` table was empty
- Collection-to-segment mapping survived but individual document IDs, metadatas, and documents were gone

### External crash source (not Roampal code)

Windows Event Viewer showed `eos.exe` and `LM Studio` crashes around the same timestamps as Roampal terminations. These external processes held conflicting ports/resources on port 27182. The Roampal server subprocess output was sent to `DEVNULL`, hiding crash errors from logs. No Roampal code or packages changed since December (v0.5.6).

### Fix — WAL journal mode and durability configuration at startup

In `chromadb_adapter.py`, new `_configure_sqlite_wal(db_path)` function runs immediately after `PersistentClient` creation:
- Opens a brief connection to the SQLite file
- Executes `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=FULL`
- Commits and closes. WAL is stored in the DB header and persists for future connections. `synchronous=FULL` is requested on the configuration connection, but SQLite treats it as connection-local and ChromaDB's Rust backend does not expose a public connection-level PRAGMA hook.

WAL (Write-Ahead Logging) keeps modifications in a separate `.wal` file until commit, so reads never block on writes and interrupted writes can be recovered without leaving the main database partially written. The helper requests FULL durability on its brief configuration connection, but ChromaDB's Rust backend does not expose a supported way to verify or set that option on its own connection.

### Acceptance criteria

- Journal mode changes from `delete` to `wal` on first PersistentClient init
- WAL persists across all connections (stored in DB header)
- No regression in query/upsert/delete paths
- Silent no-op if SQLite file doesn't exist yet (e.g., fresh install before first write)

### Files affected

| File | Change |
|---|---|
| `roampal/backend/modules/memory/chromadb_adapter.py` | New `_configure_sqlite_wal(db_path)` helper at line 21–43; called after `PersistentClient(path=...)` at line 146. Also imports `sqlite3` and adds logger warning on failure. |
| `pyproject.toml` | version 0.5.7 → 0.5.8 |
| `roampal/__init__.py` | `__version__` 0.5.7 → 0.5.8 |

### Coordination

- **No data migration required.** Existing installs will get WAL configured on next server start after installing v0.5.8. Corrupted databases need manual cleanup (delete chromadb directory, restart).
- **Desktop:** ships the fix by bumping bundled core version to v0.5.8.

### Why this matters

Without WAL mode, every hard crash risks total metadata loss — collections become empty even though binary data survives on disk. This is not a rare edge case: Windows port conflicts with other Python processes (LM Studio, eos.exe) caused 30+ days of accumulated memory to be lost in May 2026. WAL mode gives the SQLite catalog a crash-recovery path at the cost of ~5–10ms extra latency per write batch (negligible compared to embedding generation which takes seconds).

---

## Implementation status

Implemented and verified 2026-05-27.

- `_configure_sqlite_wal(db_path)` added at `chromadb_adapter.py:21-43`, called from `initialize()` after PersistentClient creation at line 146.
- Verified journal mode change on test database: `delete` → `wal`.
- Verified WAL persists across connections — second connection reads `wal` as current mode without re-applying. `synchronous=FULL` is verified only on the connection where it is requested because SQLite treats it as connection-local.
- Defensive: returns silently if SQLite file doesn't exist yet (fresh install). Logs warning on failure but doesn't block startup.

## Verification

1. **SQLite journal mode test.** Created fresh PersistentClient, ran `_configure_sqlite_wal`, verified `PRAGMA journal_mode` returns `wal`. Second connection without re-applying still reads `wal` — header persistence confirmed.
2. **Fresh install path.** Verified function returns silently when SQLite file doesn't exist yet (no crash or error).

## Automated test coverage

v0.5.8 ships with 16 new automated tests covering both fixes:

| Test file | Class | Tests | Coverage |
|---|---|---|---|
| `tests/unit/test_sqlite_wal.py` | `TestConfigureSqliteWal` | 6 | WAL mode applied, header persistence across connections, FULL durability requested on the configuration connection, missing-file no-op, failure logs warning (doesn't raise), uncommitted write rollback integrity |
| `tests/unit/test_session_manager.py` | `TestMarkScoredAtomicRewrite` | 8 | Happy path marks last matching record, no .tmp residue on success, os.replace failure preserves original, fdopen write failure cleans tmp and preserves original, missing session file returns False, doc_id not found performs zero writes, corrupt JSONL lines skipped with last-match-wins semantics, cache flags updated |
| `tests/integration/test_chromadb_integration.py` | `TestSqliteWalIntegration` | 2 | Real ChromaDBAdapter.initialize() enables WAL (primary acceptance test), normal upsert/query/delete ops regression under WAL |

Full suite: **733 passed** (all green). No regressions.

## Bugfix discovered during testing

During implementation of the atomic transcript rewrite tests, a bug was found in `session_manager.py`'s exception handler for `mark_scored()`: if `os.fdopen()` raised an exception, the file descriptor from `tempfile.mkstemp()` was left open, preventing Windows from unlinking the temp file (Windows cannot delete an open handle). The fix adds `os.close(fd)` before cleanup in the except block:

```python
except Exception:
    try:
        os.close(fd)  # v0.5.8-hotfix: close fd so Windows can unlink tmp
    except OSError:
        pass
    try:
        tmp_path.unlink()
    except OSError:
        pass
    raise
```

This is included in the v0.5.8 commit alongside the atomic rewrite itself.

## Files changed

| File | Change |
|---|---|
| `roampal/backend/modules/memory/chromadb_adapter.py` | New `_configure_sqlite_wal` helper + call from `initialize()` (+31 lines) |
| `roampal/hooks/session_manager.py` | Atomic transcript rewrite in `mark_scored()` + fdclose bugfix (+26 lines net) |
| `pyproject.toml` | version 0.5.7 → 0.5.8, dev deps (pytest-timeout, pytest-forked, build, twine) |
| `roampal/__init__.py` | `__version__` 0.5.7 → 0.5.8 |
| `tests/unit/test_sqlite_wal.py` | NEW — 6 unit tests for `_configure_sqlite_wal` |
| `tests/unit/test_session_manager.py` | +8 tests in new `TestMarkScoredAtomicRewrite` class |
| `tests/integration/test_chromadb_integration.py` | +2 integration tests in new `TestSqliteWalIntegration` class |

---

## Item 2 — Session transcript rewrite is not atomic (hotfix)

### Root cause

`SessionManager.mark_scored()` in `roampal/hooks/session_manager.py` rewrites the entire session `.jsonl` file in place with `open(session_file, "w")` when marking an exchange as scored. A crash or power loss during this rewrite truncates the file and loses the conversation transcript.

### Fix

Use the same temp-file + `os.replace()` atomic dance already used for JSON state files:
- Write updated lines to a sibling `.tmp` file
- Atomically replace the original session file with `os.replace()`
- Clean up the temp file if anything fails (including closing fd on Windows)

### Files affected

| File | Change |
|---|---|
| `roampal/hooks/session_manager.py` | Import `tempfile`; make `mark_scored()` rewrite session JSONL atomically; close fd before cleanup to fix Windows unlink bug |
| `dev/docs/releases/v0.5.8/RELEASE_NOTES.md` | Document hotfix and discovered bugfix |

---

## Pre-existing test debt fixed (not a v0.5.8 regression)

The following 20 tests were failing due to outdated assertions from v0.5.x behavioral changes, not production bugs:

| File | Tests fixed | Root cause |
|---|---|---|
| `test_fastapi_endpoints.py` (19 failures) | All endpoint tests | Memory IDs changed from predictable (`mb_test123`, `working_abc`) to UUID-based (`memory_bank_<hex8>`, `working_<hex8>`). Fixture mocks updated to return UUID-format IDs. Search test was mocking wrong method (`get_context_for_injection` instead of `search`). |
| `test_server_main.py` (1 failure) | Banner profile assertion | System's active profile file contained `"main"` instead of `"default"`, causing banner to show a Profile line the test didn't expect. Fixed by patching `active_profile_name()` in the test helper. |

These were fixed as part of the v0.5.8 test coverage pass to ensure the full suite is green before release.
