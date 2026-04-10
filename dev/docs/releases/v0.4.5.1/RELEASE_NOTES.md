# v0.4.5.1 Hotfix Release Notes

**Platforms:** Claude Code (MCP tools), OpenCode (sidecar plugin)
**Theme:** Fix noun_tag extraction prompts to match benchmark

---

## Overview

The noun_tag extraction instructions in both OpenCode's combined sidecar prompt and Claude Code's MCP tool descriptions were condensed one-liners that dropped most of the filtering rules from the benchmark-validated prompt. The benchmark prompt (`roampal-labs/strategies/entity_routed.py:79-91`) includes explicit pronoun skip lists, 12 meta-word filters, verb filtering, and a WHO/WHAT directive. Production prompts had 5 meta-words and no verb filtering.

---

## Fixes

### 1. OpenCode combined sidecar NOUN_TAGS section
**File:** `plugins/opencode/roampal.ts`

**Before (1 condensed line):**
```
NOUN_TAGS: Extract key topic nouns from your summary - people's names, places, objects,
specific things the exchange was about. Lowercase, 1-3 words each, max 8. Skip pronouns
and meta-words (user, assistant, memory, response, question).
```

**After (matches benchmark):**
```
NOUN_TAGS: Extract the key TOPIC nouns from this exchange - people's names, places, objects,
and specific things the exchange is actually about.
- Use actual names, not pronouns (skip 'he', 'she', 'they', 'user', 'assistant')
- Keep each tag as a short noun phrase (1-3 words), lowercase, max 8
- Include both proper nouns and important common nouns
- Skip meta-words about the conversation itself: source, answer, details, accuracy, response,
  question, topic, context, information, correction, update, memory
- Skip generic verbs/actions: said, told, mentioned, discussed, talked, asked
- Focus on WHO and WHAT the exchange is about, not how it was communicated
```

**What was missing:** Explicit pronoun list, 7 of 12 meta-words (source, answer, details, accuracy, topic, context, information, correction, update), all verb filtering (said, told, mentioned, discussed, talked, asked), WHO/WHAT directive.

### 2. Claude Code MCP tool noun_tags descriptions
**File:** `mcp/server.py`

Three MCP tools had the same vague one-liner for their `noun_tags` parameter:
- `add_to_memory_bank` — "Key nouns from the content - people names, places, objects, specific things. Lowercase."
- `score_memories` — "Key nouns from the exchange summary - people names, places, topics. Lowercase."
- `record_response` — "Key nouns from the takeaway - people names, places, objects, specific things. Lowercase."

All three updated to include the full benchmark filtering rules: pronoun skip list, 12 meta-words, verb filtering, WHO/WHAT directive. Claude Code has no sidecar — the main LLM extracts tags directly, so these tool descriptions are the only instructions it gets.

---

## Benchmark reference

The canonical tag extraction prompt lives in `roampal-labs/strategies/entity_routed.py:79-91`. The standalone `extract_tags()` in `sidecar_service.py:576-589` already matched the benchmark. Only the combined/tool-description versions were divergent.

---

## Files Modified

| File | Changes |
|------|---------|
| `plugins/opencode/roampal.ts` | Expanded NOUN_TAGS section from 1 line to 7 lines matching benchmark prompt |
| `mcp/server.py` | Updated `noun_tags` description on 3 MCP tools (`add_to_memory_bank`, `score_memories`, `record_response`) with full benchmark filtering rules |

---

## Verification

- [ ] OpenCode sidecar NOUN_TAGS section matches entity_routed.py rules
- [ ] All 3 Claude Code MCP tool noun_tags descriptions include: pronoun skip, 12 meta-words, verb filtering, WHO/WHAT
- [ ] Standalone `extract_tags()` in sidecar_service.py unchanged (already matched)
- [ ] Tests passing
