"""
Scoring Service

Handles all score calculations for the memory system including:
- Wilson score lower bound calculation
- Final rank score calculation with dynamic weighting
- Memory maturity-based weight adjustments

Extracted from UnifiedMemorySystem lines 47-90 (wilson_score_lower) and
lines 1514-1656 (scoring logic in search()).
"""

import math
import json
import logging
from typing import Dict, Any, Tuple, Optional
from scipy import stats

from .config import MemoryConfig

logger = logging.getLogger(__name__)


def wilson_score_lower(successes: float, total: int, confidence: float = 0.95) -> float:
    """
    Calculate the lower bound of Wilson score confidence interval (v0.2.5).

    This solves the "cold start" ranking problem where a memory with 1 success / 1 use (100%)
    would outrank a proven memory with 90/100 (90%). Wilson score uses statistical confidence
    intervals to favor proven records over lucky new ones.

    Args:
        successes: Number of successful outcomes (works + partial)
        total: Total number of uses
        confidence: Confidence level (0.95 = 95% confidence interval)

    Returns:
        Lower bound of confidence interval (0.0 to 1.0)
        - 1/1 success → ~0.20 (low confidence due to small sample)
        - 90/100 success → ~0.84 (high confidence due to large sample)
        - 0/0 → 0.5 (neutral for untested memories)

    Formula: Wilson score interval lower bound
    p̃ = (p + z²/2n - z√(p(1-p)/n + z²/4n²)) / (1 + z²/n)

    Reference: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
    """
    if total == 0:
        return 0.5  # Neutral score for untested memories

    # z-score for confidence level (1.96 for 95% confidence)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    p = successes / total  # Observed proportion
    n = total

    # Wilson score formula
    denominator = 1 + z * z / n
    center = p + z * z / (2 * n)

    # Variance term under the square root
    variance = p * (1 - p) / n + z * z / (4 * n * n)

    # Lower bound of confidence interval
    lower_bound = (center - z * math.sqrt(variance)) / denominator

    return max(0.0, lower_bound)  # Ensure non-negative


class ScoringService:
    """
    Service for calculating memory scores.

    Responsibilities:
    - Wilson score calculation for statistical confidence
    - Final rank score calculation with dynamic weighting
    - Memory maturity-based weight adjustments
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the scoring service.

        Args:
            config: Memory configuration. Uses defaults if not provided.
        """
        self.config = config or MemoryConfig()

    def calculate_wilson_score(
        self,
        successes: float,
        total: int,
        confidence: Optional[float] = None
    ) -> float:
        """
        Calculate Wilson score lower bound.

        Args:
            successes: Number of successful outcomes
            total: Total number of uses
            confidence: Confidence level (uses config default if not provided)

        Returns:
            Wilson score lower bound (0.0 to 1.0)
        """
        conf = confidence or self.config.wilson_confidence
        return wilson_score_lower(successes, total, conf)

    def count_successes_from_history(self, outcome_history: str) -> float:
        """
        Count successes from outcome history JSON string.

        Args:
            outcome_history: JSON string of outcome entries

        Returns:
            Number of successes (worked=1, partial=0.5)
        """
        successes = 0.0
        try:
            history = json.loads(outcome_history) if outcome_history else []
            for entry in history:
                if isinstance(entry, dict):
                    outcome = entry.get("outcome", "")
                    if outcome == "worked":
                        successes += 1.0
                    elif outcome == "partial":
                        successes += 0.5
        except (json.JSONDecodeError, TypeError):
            pass
        return successes

    def calculate_learned_score(
        self,
        raw_score: float,
        uses: int,
        outcome_history: str = ""
    ) -> Tuple[float, float]:
        """
        Calculate learned score with Wilson score blending.

        Args:
            raw_score: Raw score from metadata
            uses: Number of times memory was used
            outcome_history: JSON string of outcome history

        Returns:
            Tuple of (learned_score, wilson_score)
        """
        # Count successes from outcome history
        successes = self.count_successes_from_history(outcome_history)

        # Fallback: estimate from raw score if no history
        if successes == 0 and uses > 0:
            successes = raw_score * uses

        # Calculate Wilson score
        wilson = self.calculate_wilson_score(successes, uses)

        # Blend Wilson score with raw score based on sample size
        if uses == 0:
            learned = raw_score
        elif uses < 3:
            blend = uses / 3  # 0.33 for 1 use, 0.67 for 2 uses
            learned = (1 - blend) * raw_score + blend * wilson
        else:
            learned = wilson

        return learned, wilson

    def get_dynamic_weights(
        self,
        uses: int,
        learned_score: float,
        collection: str,
        importance: float = 0.7,
        confidence: float = 0.7
    ) -> Tuple[float, float]:
        """
        Get dynamic embedding/learned weights based on memory maturity.

        Args:
            uses: Number of times memory was used
            learned_score: Calculated learned score
            collection: Collection name
            importance: Memory importance (for memory_bank)
            confidence: Memory confidence (for memory_bank)

        Returns:
            Tuple of (embedding_weight, learned_weight)
        """
        if uses >= 5 and learned_score >= 0.8:
            # PROVEN HIGH-VALUE MEMORY
            return (self.config.embedding_weight_proven, self.config.learned_weight_proven)

        elif uses >= 3 and learned_score >= 0.7:
            # ESTABLISHED MEMORY
            return (0.25, 0.75)

        elif uses >= 2 and learned_score >= 0.5:
            # EMERGING PATTERN (positive)
            return (0.35, 0.65)

        elif uses >= 2:
            # FAILING PATTERN
            return (0.7, 0.3)

        elif collection == "memory_bank":
            # MEMORY BANK SPECIAL CASE - quality-based ranking
            quality = importance * confidence
            if quality >= 0.8:
                return (0.45, 0.55)
            else:
                return (0.5, 0.5)

        else:
            # NEW/UNKNOWN MEMORY
            return (self.config.embedding_weight_new, self.config.learned_weight_new)

    def calculate_final_score(
        self,
        metadata: Dict[str, Any],
        distance: float,
        collection: str
    ) -> Dict[str, float]:
        """
        Calculate final rank score for a search result.

        This is the main scoring function that combines:
        - Embedding similarity (from distance)
        - Learned score (from outcome history with Wilson scoring)
        - Dynamic weighting based on memory maturity

        Args:
            metadata: Memory metadata dict
            distance: L2 distance from embedding search
            collection: Collection name

        Returns:
            Dict with scoring details:
            {
                "final_rank_score": combined score,
                "wilson_score": statistical confidence,
                "embedding_similarity": 1/(1+distance),
                "learned_score": outcome-based score,
                "embedding_weight": weight used,
                "learned_weight": weight used
            }
        """
        raw_score = metadata.get("score", 0.5)
        uses = metadata.get("uses", 0)
        outcome_history = metadata.get("outcome_history", "")
        importance = metadata.get("importance", 0.7)
        confidence = metadata.get("confidence", 0.7)

        # Ensure numeric types
        try:
            importance = float(importance) if not isinstance(importance, (int, float)) else importance
            confidence = float(confidence) if not isinstance(confidence, (int, float)) else confidence
        except (ValueError, TypeError):
            importance = 0.7
            confidence = 0.7

        # Calculate learned score with Wilson blending
        learned_score, wilson_score = self.calculate_learned_score(
            raw_score, uses, outcome_history
        )

        # Special case: memory_bank uses quality as learned score
        if collection == "memory_bank":
            quality = importance * confidence
            learned_score = quality

        # Convert distance to similarity
        embedding_similarity = 1.0 / (1.0 + distance)

        # Get dynamic weights
        embedding_weight, learned_weight = self.get_dynamic_weights(
            uses, learned_score, collection, importance, confidence
        )

        # Calculate combined score
        final_score = (embedding_weight * embedding_similarity) + (learned_weight * learned_score)

        return {
            "final_rank_score": final_score,
            "wilson_score": wilson_score,
            "embedding_similarity": embedding_similarity,
            "learned_score": learned_score,
            "embedding_weight": embedding_weight,
            "learned_weight": learned_weight,
        }

    def apply_scoring_to_results(
        self,
        results: list,
        sort: bool = True
    ) -> list:
        """
        Apply scoring to a list of search results.

        Args:
            results: List of search result dicts
            sort: Whether to sort by final_rank_score (descending)

        Returns:
            List of results with scoring fields added
        """
        for r in results:
            metadata = r.get("metadata", {})
            distance = r.get("distance", 1.0)
            collection = r.get("collection", "")

            scores = self.calculate_final_score(metadata, distance, collection)

            # Add scores to result
            r.update(scores)
            r["original_distance"] = distance
            r["uses"] = metadata.get("uses", 0)

        if sort:
            results.sort(key=lambda x: x.get("final_rank_score", 0.0), reverse=True)

        return results
