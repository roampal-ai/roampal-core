"""
Memory System Configuration

Centralizes all magic numbers and configuration constants for the memory system.
Extracted from UnifiedMemorySystem lines 202-206, 709, 993, 398, 4221, and scattered
hardcoded values throughout the codebase.

This replaces scattered class constants with a single, configurable dataclass.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryConfig:
    """
    Configuration for the unified memory system.

    All values have production defaults matching the original codebase.
    Can be overridden for testing or different deployment scenarios.
    """

    # Threshold constants (from lines 202-206)
    high_value_threshold: float = 0.9
    """Memories above this score are preserved beyond retention period"""

    promotion_score_threshold: float = 0.7
    """Minimum score for working->history or history->patterns promotion"""

    demotion_score_threshold: float = 0.4
    """Below this, patterns demote to history"""

    deletion_score_threshold: float = 0.2
    """Below this, history items are deleted"""

    new_item_deletion_threshold: float = 0.1
    """More lenient deletion threshold for items < 7 days old"""

    # Search and scoring (from lines 709, 1580-1581)
    cross_encoder_blend_ratio: float = 0.6
    """Weight given to cross-encoder in final ranking (0.4 original, 0.6 cross-encoder)"""

    embedding_weight_proven: float = 0.2
    """Embedding weight for proven memories (high use count)"""

    learned_weight_proven: float = 0.8
    """Learned/outcome weight for proven memories"""

    embedding_weight_new: float = 0.8
    """Embedding weight for new memories (low use count)"""

    learned_weight_new: float = 0.2
    """Learned/outcome weight for new memories"""

    # Multipliers (from lines 1359, 1398, 1413, 1421)
    search_multiplier: int = 3
    """
    Fetch limit * search_multiplier results for better ranking.
    Currently hardcoded as `limit * 3` in 4 locations.
    """

    # Storage limits (from line 4221)
    max_memory_bank_items: int = 1000
    """Maximum items in memory_bank collection"""

    # Timing (from line 398)
    kg_debounce_seconds: int = 5
    """Debounce window for knowledge graph saves"""

    # Default values (from line 993)
    default_importance: float = 0.7
    """Default importance for new memories"""

    default_confidence: float = 0.7
    """Default confidence for new memories"""

    # Promotion timing
    promotion_use_threshold: int = 3
    """Minimum uses before considering for promotion"""

    promotion_age_days: int = 7
    """Minimum age in days before considering for promotion"""

    # Retention periods
    working_memory_retention_hours: int = 24
    """Working memory cleanup threshold"""

    history_retention_days: int = 30
    """History cleanup threshold for low-value items"""

    # Cross-encoder settings
    cross_encoder_top_k: int = 30
    """Number of candidates to rerank with cross-encoder"""

    # Wilson score
    wilson_confidence: float = 0.95
    """Confidence level for Wilson score calculation"""


# Default configuration instance
DEFAULT_CONFIG = MemoryConfig()
