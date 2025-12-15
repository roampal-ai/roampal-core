"""
Memory System - Core memory storage and retrieval

Provides:
- UnifiedMemorySystem: Main orchestrator
- ChromaDBAdapter: Vector storage
- Services: Search, Scoring, Outcome, Context, MemoryBank
"""

from .config import MemoryConfig
from .memory_types import (
    CollectionName,
    OutcomeEntry,
    OutcomeHistory,
    SearchResult,
    MemoryMetadata,
)
from .unified_memory_system import UnifiedMemorySystem
from .chromadb_adapter import ChromaDBAdapter
from .embedding_service import EmbeddingService
from .scoring_service import ScoringService
from .outcome_service import OutcomeService
from .memory_bank_service import MemoryBankService
from .context_service import ContextService

__all__ = [
    # Main system
    "UnifiedMemorySystem",
    # Types and config
    "MemoryConfig",
    "CollectionName",
    "OutcomeEntry",
    "OutcomeHistory",
    "SearchResult",
    "MemoryMetadata",
    # Services
    "ChromaDBAdapter",
    "EmbeddingService",
    "ScoringService",
    "OutcomeService",
    "MemoryBankService",
    "ContextService",
]
