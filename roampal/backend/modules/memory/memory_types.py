"""
Memory System Type Definitions

Centralizes all type definitions, dataclasses, and type aliases used throughout
the memory system. Extracted from UnifiedMemorySystem module-level definitions.

Replaces loose JSON string serialization with proper typed dataclasses.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
import json


# Type aliases (from lines 45, 94)
CollectionName = Literal["books", "working", "history", "patterns", "memory_bank"]
ContextType = str  # LLM discovers topics organically (coding, fitness, finance, etc.)


@dataclass
class OutcomeEntry:
    """
    A single outcome event for a memory.

    Replaces JSON strings like:
    {"outcome": "worked", "timestamp": "2024-12-10T...", "context": "coding"}
    """
    outcome: Literal["worked", "failed", "partial", "unknown"]
    timestamp: str
    context: Optional[str] = None
    confidence: float = 1.0
    implicit: bool = False
    reason: Optional[str] = None


@dataclass
class OutcomeHistory:
    """
    History of outcomes for a memory item.

    Provides proper serialization/deserialization instead of raw JSON strings
    stored in ChromaDB metadata.
    """
    entries: List[OutcomeEntry] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string for ChromaDB storage."""
        return json.dumps([asdict(e) for e in self.entries])

    @classmethod
    def from_json(cls, data: str) -> "OutcomeHistory":
        """Deserialize from JSON string."""
        if not data:
            return cls()
        try:
            entries = [OutcomeEntry(**e) for e in json.loads(data)]
            return cls(entries=entries)
        except (json.JSONDecodeError, TypeError):
            return cls()

    def add_outcome(
        self,
        outcome: Literal["worked", "failed", "partial", "unknown"],
        context: Optional[str] = None,
        confidence: float = 1.0,
        implicit: bool = False,
        reason: Optional[str] = None
    ):
        """Add a new outcome entry."""
        self.entries.append(OutcomeEntry(
            outcome=outcome,
            timestamp=datetime.now().isoformat(),
            context=context,
            confidence=confidence,
            implicit=implicit,
            reason=reason
        ))

    @property
    def success_count(self) -> int:
        """Count of worked + partial outcomes."""
        return sum(1 for e in self.entries if e.outcome in ("worked", "partial"))

    @property
    def failure_count(self) -> int:
        """Count of failed outcomes."""
        return sum(1 for e in self.entries if e.outcome == "failed")

    @property
    def total_count(self) -> int:
        """Total outcome count."""
        return len(self.entries)


@dataclass
class PromotionRecord:
    """Record of a single promotion/demotion event."""
    from_collection: str
    to_collection: str
    timestamp: str
    score: float
    uses: int
    reason: str = "score_threshold"


@dataclass
class PromotionHistory:
    """
    History of promotions/demotions for a memory item.
    """
    promotions: List[PromotionRecord] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string for ChromaDB storage."""
        return json.dumps([asdict(p) for p in self.promotions])

    @classmethod
    def from_json(cls, data: str) -> "PromotionHistory":
        """Deserialize from JSON string."""
        if not data:
            return cls()
        try:
            promotions = [PromotionRecord(**p) for p in json.loads(data)]
            return cls(promotions=promotions)
        except (json.JSONDecodeError, TypeError):
            return cls()

    def add_promotion(
        self,
        from_collection: str,
        to_collection: str,
        score: float,
        uses: int,
        reason: str = "score_threshold"
    ):
        """Record a promotion/demotion event."""
        self.promotions.append(PromotionRecord(
            from_collection=from_collection,
            to_collection=to_collection,
            timestamp=datetime.now().isoformat(),
            score=score,
            uses=uses,
            reason=reason
        ))


@dataclass
class ActionOutcome:
    """
    Tracks individual action outcomes with topic-based context awareness (v0.2.1 Causal Learning).

    Copied from original UnifiedMemorySystem lines 97-153.

    Enables learning: "In topic X, action Y leads to outcome Z"

    Examples:
    - For CODING: search_memory → 92% success (searching code patterns works well)
    - For FITNESS: create_memory → 88% success (storing workout logs works well)
    - For FINANCE: archive_memory → 75% success (archiving expenses works well)
    """
    action_type: str  # Tool name: "search_memory", "create_memory", "update_memory", etc.
    context_type: ContextType  # LLM-classified topic: "coding", "fitness", "finance", etc.
    outcome: Literal["worked", "failed", "partial"]
    timestamp: datetime = field(default_factory=datetime.now)

    # Action details
    action_params: Dict[str, Any] = field(default_factory=dict)  # Tool parameters
    doc_id: Optional[str] = None  # If action involved a document
    collection: Optional[str] = None  # Which collection was accessed

    # Outcome details
    failure_reason: Optional[str] = None
    success_context: Optional[Dict[str, Any]] = None

    # Causal attribution
    chain_position: int = 0  # Position in action chain (0 = first action)
    chain_length: int = 1  # Total actions in chain
    caused_final_outcome: bool = True  # Did this action cause the final outcome?

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for KG storage."""
        return {
            "action_type": self.action_type,
            "context_type": self.context_type,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "action_params": self.action_params,
            "doc_id": self.doc_id,
            "collection": self.collection,
            "failure_reason": self.failure_reason,
            "success_context": self.success_context,
            "chain_position": self.chain_position,
            "chain_length": self.chain_length,
            "caused_final_outcome": self.caused_final_outcome,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionOutcome":
        """Deserialize from dict."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class MemoryMetadata:
    """
    Structured metadata for a memory item.

    Provides type-safe access to commonly used metadata fields.
    """
    id: str
    timestamp: str
    collection: str
    score: float = 0.5
    uses: int = 0
    importance: float = 0.7
    confidence: float = 0.7
    tags: List[str] = field(default_factory=list)
    outcome_history: Optional[str] = None  # JSON string
    promotion_history: Optional[str] = None  # JSON string
    conversation_id: Optional[str] = None
    context_type: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for ChromaDB storage."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMetadata":
        """Create from ChromaDB metadata dict."""
        # Only include fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class SearchResult:
    """
    Structured search result.

    Provides type-safe access to search result fields.
    """
    text: str
    collection: str
    distance: float
    metadata: Dict[str, Any]
    final_rank_score: float = 0.0
    wilson_score: float = 0.5
    embedding_similarity: float = 0.0
    learned_score: float = 0.5
    ce_score: Optional[float] = None  # Cross-encoder score

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create from raw search result dict."""
        return cls(
            text=data.get("text", ""),
            collection=data.get("collection", ""),
            distance=data.get("distance", 0.0),
            metadata=data.get("metadata", {}),
            final_rank_score=data.get("final_rank_score", 0.0),
            wilson_score=data.get("wilson_score", 0.5),
            embedding_similarity=data.get("embedding_similarity", 0.0),
            learned_score=data.get("learned_score", 0.5),
            ce_score=data.get("ce_score"),
        )


# Type aliases for search results
MemoryResult = Dict[str, Any]  # Generic memory result dict


@dataclass
class SearchMetadata:
    """
    Metadata about a search operation.

    Returned alongside results when return_metadata=True.
    """
    query: str
    collections_searched: List[str]
    total_results: int
    routing_phase: str = "unknown"  # exploration, medium, high
    tier_scores: Dict[str, float] = field(default_factory=dict)
    cached_doc_ids: List[str] = field(default_factory=list)
    entity_boost_applied: bool = False
    cross_encoder_used: bool = False
    search_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)
