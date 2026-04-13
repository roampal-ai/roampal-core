# v0.4.6 Release Notes

**Platforms:** Claude Code, OpenCode
**Theme:** Context quality
**Status:** Planning

---

## Changes

### 1. Improved Facts section framing in context injection

Auto-extracted facts in KNOWN CONTEXT are now clearly labeled as directional, not authoritative. This helps the LLM treat them as leads to verify rather than ground truth.

**Fix:** Changed Facts header in `unified_memory_system.py` from `Facts:` to:
```
Facts (auto-extracted from conversation — use for direction, not authority. Verify before citing as true):
```

### 2. Remove regex tag extraction from book ingestion

Removed `extract_tags_regex()` from `store_book()`. Further testing is required before deciding whether to implement tag extraction for book memory ingestion, and which approach (LLM-based, statistical, or regex) would be appropriate.

Users who want tags on existing memories can run `roampal summarize`, which extracts tags via the sidecar LLM.

### 3. Remove automatic tag migration on startup

Removed the background `TagMigration` that ran on first server startup to backfill `noun_tags` via regex. New users get tags through the normal flow (LLM extraction via MCP tools or sidecar). Existing users who want to backfill tags on old memories can use `roampal summarize`.

### 4. `roampal summarize` — remove fact extraction, detect existing tags

- **Removed fact extraction.** Previously extracted atomic facts from original content and stored them as new working memories. These fragments were low-signal and cluttered context injection.
- **Skip tag extraction for memories that already have tags.** If a memory already has `noun_tags` in metadata, don't re-extract — prints "tags=existing" instead.

---

## Carried Forward from v0.4.5

- Test sidecar model during setup — still useful, deferred
- Tag cleanup via Wilson averages (`roampal cleanup`) — deferred pending real-world data

---

## Files Changed

| File | Change |
|------|--------|
| `unified_memory_system.py` | Facts header reworded; tag extraction removed from book ingestion; tag migration removed from startup |
| `tag_migration.py` | No longer imported or triggered (file kept for reference) |
| `cli.py` | `roampal summarize`: removed fact extraction, added existing tag detection |

---

## Verification

- [ ] KNOWN CONTEXT shows updated Facts header
- [ ] Book ingestion stores chunks without `noun_tags` metadata
- [ ] No tag migration runs on startup
- [ ] `roampal summarize` skips tag extraction on memories that already have tags
- [ ] `roampal summarize` no longer creates fact working memories
- [ ] Existing tagged memories unaffected
