"""
Tag Service — Noun tag extraction and matching for tag-routed retrieval.

v0.4.5: Replaces Knowledge Graph with simpler, faster tag-based routing.
v0.4.9: LLM-only tag extraction (matches benchmark). No regex fallback.

Tag extraction:
- LLM-only: Sidecar/main LLM extraction. Gets semantic nouns (kickboxing, pottery class).
- No regex fallback: If LLM fails, returns empty list (matches benchmark).

Query matching uses simple word matching (zero latency in search path).
"""

import json
import logging
import re
import threading
from typing import Callable, List, Optional, Set

logger = logging.getLogger(__name__)


# Common English words that start sentences but aren't proper nouns
_COMMON_SENTENCE_STARTERS = {
    # Pronouns/articles/determiners
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "we",
    "he",
    "she",
    "they",
    "you",
    "my",
    "our",
    "his",
    "her",
    # Conjunctions/prepositions
    "there",
    "here",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "how",
    "why",
    "if",
    "but",
    "and",
    "or",
    "so",
    "yet",
    "for",
    "nor",
    # Adverbs
    "not",
    "no",
    "yes",
    "just",
    "also",
    "still",
    "even",
    "only",
    "now",
    "after",
    "before",
    "since",
    "while",
    "during",
    "until",
    "once",
    "both",
    "each",
    "every",
    "all",
    "any",
    "some",
    "most",
    "many",
    "much",
    "few",
    "several",
    "other",
    "another",
    "such",
    "same",
    "however",
    "therefore",
    "meanwhile",
    "otherwise",
    "furthermore",
    "although",
    "because",
    "unless",
    "whether",
    "instead",
    "despite",
    "pretty",
    "really",
    "very",
    "quite",
    "rather",
    "always",
    "never",
    "often",
    "sometimes",
    "recently",
    "currently",
    "finally",
    "already",
    "overall",
    "basically",
    "essentially",
    "apparently",
    "generally",
    # Common verbs that start sentences (past tense / imperative)
    "agreed",
    "asked",
    "confirmed",
    "explained",
    "found",
    "noted",
    "fixed",
    "added",
    "removed",
    "updated",
    "replaced",
    "checked",
    "went",
    "got",
    "made",
    "took",
    "came",
    "said",
    "told",
    "gave",
    "ran",
    "set",
    "put",
    "let",
    "cut",
    "run",
    "get",
    "see",
    "saw",
    "stop",
    "start",
    "keep",
    "done",
    "doing",
    "being",
    "tried",
    "used",
    "moved",
    "changed",
    "created",
    "built",
    "wrote",
    "read",
    "sent",
    "left",
    "lost",
    "won",
    "paid",
    "met",
    "hit",
    "showed",
    "proved",
    "tested",
    "verified",
    "merged",
    "resolved",
    "noticed",
    "realized",
    "decided",
    "considered",
    "suggested",
    "reviewed",
    "analyzed",
    "compared",
    "looked",
    "searched",
    "wanted",
    "needed",
    "liked",
    "loved",
    "thought",
    "felt",
    "called",
    "helped",
    "worked",
    "played",
    "lived",
    "turned",
    "opened",
    "closed",
    "finished",
    "started",
    "ended",
    "continued",
    "learned",
    "discovered",
    "remembered",
    "forgot",
    "mentioned",
    "recommended",
    "proposed",
    "requested",
    "required",
    "expected",
    "completed",
    "implemented",
    "deployed",
    "configured",
    "installed",
    "scanned",
    "rewrote",
    "refactored",
    "debugged",
    "committed",
    "clarified",
    "identified",
    "designed",
    "happened",
    "recurring",
    "overwrites",
    "removes",
    "standalone",
    "clickable",
    "increases",
    "decreases",
    "returns",
    "produces",
    "provides",
    "requires",
    "supports",
    "includes",
    "excludes",
    "contains",
    "enables",
    "prevents",
    "allows",
    "ensures",
    "maintains",
    "represents",
    "improved",
    "reduced",
    "increased",
    "decreased",
    "eliminated",
    "applied",
    "stored",
    "loaded",
    "fetched",
    "queried",
    "returned",
    "matched",
    "filtered",
    "sorted",
    "ranked",
    "scored",
    "computed",
    "calculated",
    "extracted",
    "injected",
    "inserted",
    "deleted",
    "archived",
    "promoted",
    "demoted",
    "migrated",
    "converted",
    "validated",
    "confirmed",
    "rejected",
    "accepted",
    "approved",
    "assigned",
    "attached",
    "detected",
    "triggered",
    "handled",
    "processed",
    "generated",
    "formatted",
    "parsed",
    "serialized",
    "posted",
    "submitted",
    "published",
    "released",
    "launched",
    # Common adjectives that start sentences
    "hard",
    "easy",
    "fast",
    "slow",
    "full",
    "empty",
    "ready",
    "able",
    "sure",
    "clear",
    "true",
    "false",
    "right",
    "wrong",
    "next",
    "previous",
    "main",
    "current",
    "recent",
    "similar",
    "different",
    "specific",
    "important",
    "interesting",
    "useful",
    "possible",
    "necessary",
    "bigger",
    "smaller",
    "better",
    "worse",
    "later",
    "earlier",
    # Common words that aren't useful as tags
    "would",
    "could",
    "should",
    "must",
    "might",
    "need",
    "over",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "hundred",
    "thousand",
    "million",
}

# Tags that are noise — meta-words, months, generic adjectives
_NOISE_TAGS = {
    # Meta/abstract terms (about the conversation, not the content)
    "source",
    "answer",
    "details",
    "accuracy",
    "response",
    "question",
    "information",
    "content",
    "data",
    "system",
    "memory",
    "result",
    "user",
    "assistant",
    "context",
    "type",
    "value",
    "name",
    "list",
    "example",
    "note",
    "item",
    "thing",
    "way",
    "time",
    "part",
    "topic",
    "conversation",
    "discussion",
    "exchange",
    "summary",
    "key",
    "takeaway",
    "outcome",
    "score",
    "status",
    "update",
    "focus",
    "approach",
    "method",
    "process",
    "based",
    "level",
    "architecture",
    "benchmark",
    "release",
    "version",
    "issue",
    "problem",
    "solution",
    "fix",
    "change",
    "idea",
    "step",
    "task",
    "plan",
    "goal",
    "progress",
    "work",
    "fact",
    "conclusion",
    "section",
    "point",
    "reason",
    "case",
    # Note: months and days of week NOT filtered — temporal tags are useful for routing
    # Generic adjectives
    "old",
    "new",
    "big",
    "small",
    "long",
    "short",
    "high",
    "low",
    "good",
    "great",
    "nice",
    "bad",
    "best",
    "worst",
    "first",
    "last",
    "major",
    "minor",
    "total",
    "single",
    "double",
    "whole",
}

# Short acronyms that ARE valid tags
_VALID_SHORT_TAGS = {"ai", "ml", "uk", "us", "eu", "nz"}


def _is_adjective_not_place(word: str) -> bool:
    """Check if word is a nationality/language adjective, not a place name."""
    suffixes = ("ian", "ish", "ese", "ean", "ic")
    return any(word.endswith(s) for s in suffixes) and len(word) > 4


def _strip_file_extensions(word: str) -> str:
    """Remove common file extensions stuck to words."""
    return re.sub(
        r"\.(md|py|js|ts|json|yaml|yml|toml|txt|csv|html|css)$",
        "",
        word,
        flags=re.IGNORECASE,
    )


def _dedup_substrings(tags: List[str]) -> List[str]:
    """If 'ford mustang' is a tag, remove 'ford' and 'mustang' individually."""
    multi_word = [t for t in tags if " " in t]
    if not multi_word:
        return tags
    result = []
    for tag in tags:
        if " " not in tag:
            if any(tag in mw.split() for mw in multi_word):
                continue
        result.append(tag)
    return result


# v0.4.9: REMOVED - extract_tags_regex() function
# Benchmark uses LLM-only tag extraction. No regex fallback.


class TagService:
    """
    Tag extraction and matching for tag-routed retrieval.

    v0.4.9: LLM-only tag extraction (matches benchmark). No regex fallback.

    Manages:
    - Noun tag extraction (LLM-only)
    - Known tag index (in-memory, rebuilt from ChromaDB on init)
    - Query-to-tag matching (simple word matching)
    """

    def __init__(
        self, llm_extract_fn: Optional[Callable[[str], Optional[List[str]]]] = None
    ):
        """
        Args:
            llm_extract_fn: Optional callable(text) -> List[str] for LLM extraction.
                            If None or returns None, falls back to regex.
        """
        self._llm_extract_fn = llm_extract_fn
        self._known_tags: Set[str] = set()
        self._tags_lock = threading.Lock()

    # --- Extraction ---

    def extract_tags(self, text: str) -> List[str]:
        """
        Extract noun tags from text using LLM only (matches benchmark).
        No regex fallback — if LLM fails, returns empty list.

        Returns: List of lowercase tag strings, max 8.
        """
        import inspect

        if not self._llm_extract_fn:
            logger.debug("No LLM extract function available, returning []")
            return []

        if inspect.iscoroutinefunction(self._llm_extract_fn):
            logger.warning(
                "TagService.extract_tags() called with an async llm_extract_fn — "
                "sync path can't await it. Use extract_tags_async() instead. "
                "Returning []."
            )
            return []

        try:
            tags = self._llm_extract_fn(text)
            if tags:
                tags = self._normalize_llm_tags(tags)
                with self._tags_lock:
                    self._known_tags.update(tags)
                return tags
        except Exception as e:
            logger.warning(f"LLM tag extraction failed: {e}")

        return []

    @staticmethod
    def _normalize_llm_tags(tags: List[str]) -> List[str]:
        """Normalize LLM-extracted tags: lowercase, dedup, filter noise, cap at 8."""
        seen = set()
        result = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            tag = tag.lower().strip()
            if len(tag) < 2 or tag in _NOISE_TAGS or tag in seen:
                continue
            seen.add(tag)
            result.append(tag)
        result.sort(key=len, reverse=True)
        return result[:8]

    # --- Query Matching ---

    def match_query_tags(self, query: str) -> List[str]:
        """
        Match query text against known tag index.

        Uses simple word matching (fast, no LLM latency in search path).
        Uses word-boundary matching to prevent "log" matching "logan".

        Returns: List of matched known tags, sorted by length (most specific first).
        """
        if not query or not self._known_tags:
            return []

        query_lower = query.lower()
        matches = []

        # Sort known tags by length descending (longer/more specific first)
        sorted_tags = sorted(self._known_tags, key=len, reverse=True)

        for tag in sorted_tags:
            if len(tag) < 2:
                continue
            # Word boundary match
            pattern = r"\b" + re.escape(tag) + r"\b"
            if re.search(pattern, query_lower):
                matches.append(tag)

        return matches[:8]

    # --- Known Tag Index ---

    def rebuild_known_tags(self, collections: dict):
        """
        Scan all ChromaDB collections and rebuild in-memory known_tags set.

        Called once during initialization. Fast — reads metadata only.
        """
        with self._tags_lock:
            self._known_tags.clear()
            total = 0

            for coll_name, adapter in collections.items():
                try:
                    collection_obj = adapter.collection
                    if not collection_obj:
                        continue
                    items = collection_obj.get(limit=10000, include=["metadatas"])
                    for meta in items.get("metadatas", []):
                        if not meta:
                            continue
                        noun_tags_raw = meta.get("noun_tags", "")
                        if not noun_tags_raw:
                            continue
                        try:
                            if isinstance(noun_tags_raw, str):
                                tags = json.loads(noun_tags_raw)
                            else:
                                tags = noun_tags_raw
                            if isinstance(tags, list):
                                self._known_tags.update(
                                    t for t in tags if isinstance(t, str)
                                )
                                total += len(tags)
                        except (json.JSONDecodeError, TypeError):
                            pass
                except Exception as e:
                    logger.warning(f"Failed to scan {coll_name} for tags: {e}")

            logger.info(
            f"Rebuilt known tag index: {len(self._known_tags)} unique tags from {total} total"
        )

    def add_known_tags(self, tags: List[str]):
        """Add tags to the known index (called after storing a new memory)."""
        with self._tags_lock:
            self._known_tags.update(tags)

    @property
    def known_tags(self) -> Set[str]:
        """Get a copy of the known tag set."""
        return self._known_tags.copy()

    @property
    def known_tag_count(self) -> int:
        return len(self._known_tags)
