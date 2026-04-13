# v0.4.6.1 Hotfix Release Notes

**Platforms:** Claude Code (MCP tools), OpenCode (sidecar plugin)
**Theme:** Improve MCP tool documentation for Glama quality score

---

## Overview

Glama's quality score evaluates MCP tool descriptions across 6 dimensions: purpose clarity, usage guidelines, behavioral transparency, parameter semantics, conciseness, and contextual completeness. Several tools had sparse descriptions that scored poorly, dragging the overall quality grade to B (75%). This hotfix expands tool documentation to target an A grade.

---

## Changes

### 1. `update_memory` tool description (1.3/5.0 -> targeting A)
**File:** `mcp/server.py`

- Added full memory lifecycle explanation (5 collections, promotion flow)
- Added matching behavior docs (semantic similarity, not exact string)
- Added return value documentation
- Expanded parameter descriptions from one-liners to full sentences

### 2. `score_memories` tool description (2.1/5.0 -> targeting A)
**File:** `mcp/server.py`

- Added "HOW IT WORKS" section explaining Wilson confidence scoring feedback loop
- Added sibling tool differentiation (vs record_response)
- Added SIDE EFFECTS section (what scoring does to the system)
- Added parameter examples (e.g., `{"patterns_abc123": "worked"}`)
- Added return value documentation

### 3. `delete_memory` tool description (2.9/5.0 -> targeting A)
**File:** `mcp/server.py`

- Added irreversibility warning in first line
- Added scope clarification (only memory_bank, not scored collections)
- Added matching behavior docs (semantic match, closest-match deletion)
- Added return value documentation
- Expanded parameter description

### 4. `search_memory` tool description (3.1/5.0 -> targeting A)
**File:** `mcp/server.py`

- Added RETURNS section documenting result structure (content, metadata fields, formatted display)
- Added HOW THIS DIFFERS section listing all 4 sibling tools

### 5. `add_to_memory_bank` tool description (3.2/5.0 -> targeting A)
**File:** `mcp/server.py`

- Added HOW THIS DIFFERS FROM record_response section (permanent/unscored vs scored pipeline)
- Added return value documentation

### 6. Dockerfile updated
**File:** `Dockerfile.glama-test`

- Updated commit hash to `d1901f9` (includes tool description improvements)

---

## Files Modified

| File | Changes |
|------|---------|
| `mcp/server.py` | Expanded descriptions on 5 MCP tools with lifecycle, returns, matching, sibling differentiation |
| `Dockerfile.glama-test` | Updated commit hash to d1901f9 |
| `pyproject.toml` | Version bump 0.4.6 -> 0.4.6.1 |
| `roampal/__init__.py` | Version bump 0.4.6 -> 0.4.6.1 |

---

## Verification

- [x] 505 tests passing
- [ ] Glama quality score re-evaluated after release
- [ ] awesome-mcp-servers PR #1915 updated if AAA achieved
