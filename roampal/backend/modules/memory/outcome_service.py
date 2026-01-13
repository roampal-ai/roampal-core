"""
OutcomeService - Extracted from UnifiedMemorySystem

Handles outcome recording, score updates, and learning from feedback.

v0.2.3: Performance fix - defer heavy KG learning to background tasks.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal, Callable, Awaitable

from .config import MemoryConfig

logger = logging.getLogger(__name__)


class OutcomeService:
    """
    Service for recording outcomes and updating memory scores.

    Extracted from UnifiedMemorySystem.record_outcome and related methods.
    Handles:
    - Time-weighted score updates
    - Outcome history tracking
    - KG routing updates
    - Problem-solution pattern tracking
    - Batch cleanup of old working memory (every 50 scores)
    """

    def __init__(
        self,
        collections: Dict[str, Any],
        kg_service: Any = None,
        promotion_service: Any = None,
        config: Optional[MemoryConfig] = None
    ):
        """
        Initialize OutcomeService.

        Args:
            collections: Dict of collection name -> adapter
            kg_service: KnowledgeGraphService for routing updates
            promotion_service: PromotionService for promotion handling
            config: Memory configuration
        """
        self.collections = collections
        self.kg_service = kg_service
        self.promotion_service = promotion_service
        self.config = config or MemoryConfig()

        # Counter for batch cleanup trigger
        self._score_count = 0
        self._cleanup_interval = 50  # Trigger cleanup every N scores

    async def record_outcome(
        self,
        doc_id: str,
        outcome: Literal["worked", "failed", "partial"],
        failure_reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Record outcome and trigger learning.

        Args:
            doc_id: Document that was used
            outcome: Whether it worked
            failure_reason: Reason for failure (if applicable)
            context: Additional context for learning

        Returns:
            Updated metadata or None if document not found
        """
        # Find collection and document FIRST (needed for KG routing update)
        collection_name = None
        doc = None

        for coll_name, adapter in self.collections.items():
            if doc_id.startswith(coll_name):
                collection_name = coll_name
                doc = adapter.get_fragment(doc_id)
                break

        # UPDATE KG ROUTING FIRST - even for books/memory_bank
        # This allows KG to learn which collections answer which queries
        if doc and collection_name and self.kg_service:
            metadata = doc.get("metadata", {})
            # For quiz retrievals, use quiz_question from context; otherwise use stored query
            problem_text = ""
            if context and "quiz_question" in context:
                problem_text = context["quiz_question"]
            else:
                problem_text = metadata.get("query", "") or metadata.get("text", "")[:200]

            if problem_text:
                await self.kg_service.update_kg_routing(problem_text, collection_name, outcome)
                logger.info(f"[KG] Updated routing for '{problem_text[:50]}' -> {collection_name} (outcome={outcome})")

        # SAFEGUARD: Books are reference material, not scorable memories
        # But we still updated KG routing above so system learns to route to books
        if doc_id.startswith("books_"):
            logger.info(f"[KG] Learned routing pattern for books, but skipping score update (static reference material)")
            return None

        if not doc:
            logger.warning(f"Document {doc_id} not found")
            return None

        # Calculate score update
        metadata = doc.get("metadata", {})
        current_score = metadata.get("score", 0.5)
        uses = metadata.get("uses", 0)
        success_count = metadata.get("success_count", 0.0)  # v0.2.8: Track cumulative successes

        # Time-weighted score update
        time_weight = self._calculate_time_weight(metadata.get("last_used"))
        score_delta, new_score, uses, success_delta = self._calculate_score_update(
            outcome, current_score, uses, time_weight
        )
        success_count += success_delta  # v0.2.8: Accumulate successes (no cap)

        # Update context tracking
        if outcome == "worked" and context:
            contexts = json.loads(metadata.get("success_contexts", "[]"))
            contexts.append(context)
            metadata["success_contexts"] = json.dumps(contexts)
        elif outcome == "failed" and failure_reason:
            reasons = json.loads(metadata.get("failure_reasons", "[]"))
            reasons.append({
                "reason": failure_reason,
                "timestamp": datetime.now().isoformat()
            })
            metadata["failure_reasons"] = json.dumps(reasons)

        # Update outcome history
        outcome_history = json.loads(metadata.get("outcome_history", "[]"))
        outcome_history.append({
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "reason": failure_reason
        })
        outcome_history = outcome_history[-10:]  # Keep last 10

        # Update metadata
        metadata.update({
            "score": new_score,
            "uses": uses,
            "success_count": success_count,  # v0.2.8: Cumulative successes for Wilson score
            "last_outcome": outcome,
            "last_used": datetime.now().isoformat(),
            "outcome_history": json.dumps(outcome_history)
        })

        # Persist to collection
        self.collections[collection_name].update_fragment_metadata(doc_id, metadata)

        logger.info(
            f"Score update [{collection_name}]: {current_score:.2f} -> {new_score:.2f} "
            f"(outcome={outcome}, delta={score_delta:+.2f}, time_weight={time_weight:.2f}, uses={uses})"
        )

        # ========== FAST PATH COMPLETE (v0.2.3) ==========
        # Score updated, metadata persisted. Return immediately.
        # Heavy KG learning runs in background (fire-and-forget).

        if self.kg_service or self.promotion_service:
            asyncio.create_task(
                self._deferred_learning(
                    doc_id=doc_id,
                    outcome=outcome,
                    collection_name=collection_name,
                    doc=doc,
                    new_score=new_score,
                    uses=uses,
                    metadata=metadata,
                    failure_reason=failure_reason,
                    context=context
                )
            )

        logger.info(f"Outcome recorded: {doc_id} -> {outcome} (score: {new_score:.2f})")
        return metadata

    async def _deferred_learning(
        self,
        doc_id: str,
        outcome: str,
        collection_name: str,
        doc: Dict[str, Any],
        new_score: float,
        uses: int,
        metadata: Dict[str, Any],
        failure_reason: Optional[str],
        context: Optional[Dict[str, Any]]
    ):
        """
        Background task for heavy KG learning operations.

        Runs after score update returns. Fire-and-forget pattern.
        v0.2.3: Extracted from record_outcome for performance.
        """
        try:
            # Update KG with outcome (skip duplicate routing - already done above)
            if self.kg_service:
                problem_text = metadata.get("query", "")
                await self._update_kg_with_outcome(
                    doc_id, outcome, problem_text, doc.get("content", ""),
                    new_score, metadata, failure_reason, context
                )

            # Handle promotion/demotion
            if self.promotion_service:
                collection_size = self.collections[collection_name].collection.count()
                await self.promotion_service.handle_promotion(
                    doc_id=doc_id,
                    collection=collection_name,
                    score=new_score,
                    uses=uses,
                    metadata=metadata,
                    collection_size=collection_size
                )

            # Batch cleanup: trigger every N scores
            self._score_count += 1
            if self._score_count % self._cleanup_interval == 0 and self.promotion_service:
                cleaned = await self.promotion_service.cleanup_old_working_memory(max_age_hours=24.0)
                if cleaned > 0:
                    logger.info(f"Batch cleanup triggered: removed {cleaned} old working memories")

            logger.info(f"[Background] Deferred learning completed for {doc_id}")

        except Exception as e:
            logger.error(f"[Background] Deferred learning error for {doc_id}: {e}")

    def _calculate_time_weight(self, last_used: Optional[str]) -> float:
        """Calculate time weight for score updates."""
        if not last_used:
            return 1.0

        try:
            age_days = (datetime.now() - datetime.fromisoformat(last_used)).days
            return 1.0 / (1 + age_days / 30)  # Decay over month
        except:
            return 1.0

    def _calculate_score_update(
        self,
        outcome: str,
        current_score: float,
        uses: int,
        time_weight: float
    ) -> tuple:
        """
        Calculate score delta and new values.

        Returns:
            Tuple of (score_delta, new_score, new_uses, success_delta)

        v0.2.8: Added success_delta for Wilson score tracking.
        - worked: +1.0 success, +1 use
        - partial: +0.5 success, +1 use
        - failed: +0.0 success, +1 use

        v0.2.9: unknown handling (all collections).
        - unknown: +0.25 success, +1 use (weak negative signal for natural selection)
        """
        if outcome == "worked":
            score_delta = 0.2 * time_weight
            new_score = min(1.0, current_score + score_delta)
            uses += 1
            success_delta = 1.0
        elif outcome == "failed":
            score_delta = -0.3 * time_weight
            new_score = max(0.0, current_score + score_delta)
            uses += 1  # v0.2.8: Fixed - failed should increment uses
            success_delta = 0.0
        elif outcome == "partial":  # v0.2.8.1: explicit partial check
            score_delta = 0.05 * time_weight
            new_score = min(1.0, current_score + score_delta)
            uses += 1
            success_delta = 0.5
        elif outcome == "unknown":
            # v0.2.9: unknown = surfaced but not used (all collections)
            # Weak negative signal: 0.25 success creates gradual drift for noise
            score_delta = 0.0  # Don't change raw score
            new_score = current_score
            uses += 1
            success_delta = 0.25
        else:
            # Guard - invalid outcomes don't affect score
            logger.warning(f"Unexpected outcome '{outcome}' - no score change")
            return 0.0, current_score, uses, 0.0

        return score_delta, new_score, uses, success_delta

    async def _update_kg_with_outcome(
        self,
        doc_id: str,
        outcome: str,
        problem_text: str,
        solution_text: str,
        new_score: float,
        metadata: Dict[str, Any],
        failure_reason: Optional[str],
        context: Optional[Dict[str, Any]]
    ):
        """Update knowledge graph based on outcome."""
        if not self.kg_service:
            return

        # v0.2.3: Removed duplicate update_kg_routing call - already done in record_outcome

        if outcome == "worked" and problem_text and solution_text:
            # Extract concepts
            problem_concepts = self.kg_service.extract_concepts(problem_text)
            solution_concepts = self.kg_service.extract_concepts(solution_text)
            all_concepts = list(set(problem_concepts + solution_concepts))

            # Build relationships
            self.kg_service.build_concept_relationships(all_concepts)

            # Track problem category
            problem_key = "_".join(sorted(problem_concepts)[:3])
            self.kg_service.add_problem_category(problem_key, doc_id)

            # Track solution pattern
            self.kg_service.add_solution_pattern(
                doc_id, solution_text, new_score,
                [problem_key], solution_concepts[:5]
            )

            # Update success rate
            self.kg_service.update_success_rate(doc_id, outcome)

            # Track problem-solution mapping
            await self._track_problem_solution(doc_id, metadata, context)

        elif outcome == "failed":
            # Track failure
            self.kg_service.update_success_rate(doc_id, outcome)

            if failure_reason:
                self.kg_service.add_failure_pattern(
                    failure_reason[:50], doc_id, problem_text[:100]
                )

        elif outcome == "partial":
            self.kg_service.update_success_rate(doc_id, outcome)

        # Save KG (debounced)
        await self.kg_service.debounced_save_kg()

    async def _track_problem_solution(
        self,
        doc_id: str,
        metadata: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ):
        """Track successful problem->solution patterns for future reuse."""
        if not self.kg_service:
            return

        try:
            problem_text = metadata.get("original_context", "") or metadata.get("query", "")
            solution_text = metadata.get("text", "")

            if not problem_text or not solution_text:
                return

            # Create problem signature
            problem_concepts = self.kg_service.extract_concepts(problem_text)
            problem_signature = "_".join(sorted(problem_concepts[:5]))

            if not problem_signature:
                return

            # Track in KG
            self.kg_service.add_problem_solution(
                problem_signature=problem_signature,
                doc_id=doc_id,
                solution_text=solution_text,
                context=context
            )

            # Track solution pattern
            pattern_hash = f"{problem_signature}::{doc_id}"
            self.kg_service.add_solution_pattern_entry(
                pattern_hash=pattern_hash,
                problem_text=problem_text,
                solution_text=solution_text,
                outcome="worked"
            )

            logger.info(f"Tracked problem->solution: {problem_signature[:30]}... -> {doc_id}")

        except Exception as e:
            logger.error(f"Error tracking problem->solution: {e}")

    def count_successes_from_history(self, outcome_history_json: str) -> float:
        """
        Count successes from outcome history JSON.

        Args:
            outcome_history_json: JSON string of outcome history

        Returns:
            Weighted success count (worked=1, partial=0.5)
        """
        if not outcome_history_json or outcome_history_json == "[]":
            return 0

        try:
            history = json.loads(outcome_history_json)
            successes = 0.0
            for entry in history:
                outcome = entry.get("outcome", "")
                if outcome == "worked":
                    successes += 1.0
                elif outcome == "partial":
                    successes += 0.5
            return successes
        except json.JSONDecodeError:
            return 0

    def get_outcome_stats(self, doc_id: str) -> Dict[str, Any]:
        """
        Get outcome statistics for a document.

        Args:
            doc_id: Document ID

        Returns:
            Dict with outcome stats
        """
        for coll_name, adapter in self.collections.items():
            if doc_id.startswith(coll_name):
                doc = adapter.get_fragment(doc_id)
                if doc:
                    metadata = doc.get("metadata", {})
                    outcome_history = json.loads(metadata.get("outcome_history", "[]"))

                    worked = sum(1 for o in outcome_history if o.get("outcome") == "worked")
                    failed = sum(1 for o in outcome_history if o.get("outcome") == "failed")
                    partial = sum(1 for o in outcome_history if o.get("outcome") == "partial")

                    return {
                        "doc_id": doc_id,
                        "collection": coll_name,
                        "score": metadata.get("score", 0.5),
                        "uses": metadata.get("uses", 0),
                        "last_outcome": metadata.get("last_outcome"),
                        "outcomes": {
                            "worked": worked,
                            "failed": failed,
                            "partial": partial
                        },
                        "total_outcomes": len(outcome_history)
                    }

        return {"doc_id": doc_id, "error": "not_found"}
