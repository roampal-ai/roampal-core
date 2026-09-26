"""
One source of truth for every memory length rule (v0.6.0 Task 37).

Governance ask (2026-09-18): stop juggling different limits in different
places. This module holds every number, the three rejection messages, and
a single `check_length(kind, text)` gate that runs once at the SERVER on
every write path (MCP tools, sidecar summaries/facts, `roampal
summarize`, `record_response`). MCP tool descriptions are built from
these same values so the writing model sees the rules it will be held to.

Rules (Task 35 user decisions, standing rule memory_bank_93aecd4f):
- Target everywhere: ~300 chars, 1-2 sentences. Never silently truncate.
- Summary/takeaway hard cap: 600 (also the MCP schema maxLength).
- Memory-bank entry hard cap: 600; single fact: 150.
- Server backstop (no-memory-limits callers): 2000 — REJECT, never cut.
- Display cuts (text shown to the model, never a write path): 300 each
  for RECENT EXCHANGES and the cold-start profile block; clipped entries
  are word-boundary cut and end in "…" so the model knows to
  search_memory for the full text.

Overflow: if content genuinely needs more room, the main LLM makes
another memory (a separate record_response for summaries/takeaways,
another add_to_memory_bank call for entries); the sidecar only rewrites
shorter (it summarizes one exchange).
"""

TARGET_CHARS = 300
TARGET_PHRASE = "~300 chars, 1-2 sentences"

SUMMARY_TAKEAWAY_MAX = 600
MEMORY_BANK_ENTRY_MAX = 600
FACT_MAX = 150
SERVER_BACKSTOP_MAX = 2000

RECENT_EXCHANGES_DISPLAY_CUT = 300
PROFILE_BLOCK_DISPLAY_CUT = 300


def summary_takeaway_too_long(n: int) -> str:
    """record_response / score_memories summary rejection."""
    return (
        f"Too long: {n}/{SUMMARY_TAKEAWAY_MAX} chars. "
        f"Rewrite in {TARGET_PHRASE}. "
        "Put anything extra in a separate record_response."
    )


def memory_bank_entry_too_long(n: int) -> str:
    """add/update memory-bank entry rejection."""
    return (
        f"Too long: {n}/{MEMORY_BANK_ENTRY_MAX} chars. "
        f"Rewrite in ~300 chars, 1-2 sentences. "
        "Put anything extra in another add_to_memory_bank call."
    )


def fact_too_long(n: int) -> str:
    """Per-fact rejection (memory-bank facts list)."""
    return f"Too long: {n}/{FACT_MAX} chars. One fact per item, {FACT_MAX} chars max."


def sidecar_reprompt_too_long(n: int) -> str:
    """Text the plugin re-sends to the sidecar when its summary is long."""
    return f"Too long: {n}/{SUMMARY_TAKEAWAY_MAX} chars. Rewrite in ~300 chars, 1-2 sentences."


def over_backstop(n: int) -> str:
    """Server backstop rejection — nothing stored; the model must split."""
    return (
        f"Content too long: {n} chars (hard limit {SERVER_BACKSTOP_MAX}). "
        f"Split it into separate memories, each {TARGET_PHRASE}, written as "
        "separate calls. Nothing was stored."
    )


def check_length(kind: str, text):
    """Validate one write.

    Returns '' when the text is within its rule; otherwise the exact
    rejection message to surface to the client/model. A rejected write
    stores NOTHING (callers raise before storing).

    Kinds: 'summary'/'takeaway' (record_response, score_memories
    summaries), 'memory_bank' (add/update entries), 'fact' (per-item in
    the memory-bank facts list).
    """
    if text is None:
        return ""
    text = str(text)
    n = len(text)
    if kind in ("summary", "takeaway"):
        if n > SUMMARY_TAKEAWAY_MAX:
            return summary_takeaway_too_long(n)
    elif kind == "memory_bank":
        if n > MEMORY_BANK_ENTRY_MAX:
            return memory_bank_entry_too_long(n)
    elif kind == "fact":
        if n > FACT_MAX:
            return fact_too_long(n)
    if n > SERVER_BACKSTOP_MAX:
        return over_backstop(n)
    return ""
