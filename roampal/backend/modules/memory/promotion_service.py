"""
Promotion Service - Automatic memory promotion/demotion between collections.

Extracted from UnifiedMemorySystem as part of refactoring.

Responsibilities:
- Promote valuable memories (working -> history -> patterns)
- Demote failing patterns
- Delete persistently failing memories
- Batch promotion of valuable working memory
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .config import MemoryConfig

logger = logging.getLogger(__name__)


class PromotionService:
    """
    Handles automatic promotion and demotion of memories between collections.

    Promotion Flow:
    - working -> history: score >= 0.7, uses >= 2
    - history -> patterns: score >= 0.9 (HIGH_VALUE_THRESHOLD), uses >= 3

    Demotion Flow:
    - patterns -> history: score < 0.4 (DEMOTION_SCORE_THRESHOLD)

    Deletion:
    - Any collection: score < 0.2 (DELETION_SCORE_THRESHOLD)
    - New items (< 7 days): score < 0.1 (NEW_ITEM_DELETION_THRESHOLD)
    """

    def __init__(
        self,
        collections: Dict[str, Any],  # ChromaDBAdapter instances
        embed_fn: Callable[[str], Any],  # Async function to embed text
        add_relationship_fn: Optional[Callable] = None,  # For evolution tracking
        config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize PromotionService.

        Args:
            collections: Dict mapping collection name to ChromaDBAdapter
            embed_fn: Async function to generate embeddings
            add_relationship_fn: Optional callback to track evolution relationships
            config: Optional MemoryConfig for thresholds
        """
        self.collections = collections
        self.embed_fn = embed_fn
        self.add_relationship_fn = add_relationship_fn
        self.config = config or MemoryConfig()

        # Promotion lock to prevent concurrent promotion operations
        self._promotion_lock = asyncio.Lock()

    # =========================================================================
    # Main Promotion/Demotion Handler
    # =========================================================================

    async def handle_promotion(
        self,
        doc_id: str,
        collection: str,
        score: float,
        uses: int,
        metadata: Dict[str, Any],
        collection_size: int = 0
    ) -> Optional[str]:
        """
        Handle automatic promotion/demotion using outcome-based thresholds.

        Args:
            doc_id: Document ID to evaluate
            collection: Current collection name
            score: Current score (0-1)
            uses: Number of times used
            metadata: Document metadata
            collection_size: Current collection size (unused, kept for API compatibility)

        Returns:
            New doc_id if promoted/demoted, None otherwise
        """
        # Get full document for evaluation
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return None

        doc = self.collections[collection].get_fragment(doc_id)
        if not doc:
            logger.warning(f"Cannot evaluate {doc_id}: document not found")
            return None

        # Promotion: working -> history
        if collection == "working":
            if score >= self.config.promotion_score_threshold and uses >= 2:
                return await self._promote_working_to_history(doc_id, doc, metadata, score, uses)

        # Promotion: history -> patterns
        # v0.2.9: Require success_count >= 5 (must prove usefulness after entering history)
        elif collection == "history":
            success_count = float(metadata.get("success_count", 0.0))
            if score >= self.config.high_value_threshold and uses >= 3 and success_count >= 5:
                return await self._promote_history_to_patterns(doc_id, doc, metadata, score, uses)

        # Demotion: patterns -> history
        elif collection == "patterns":
            if score < self.config.demotion_score_threshold:
                return await self._demote_patterns_to_history(doc_id, metadata, score)

        # Deletion: score too low
        if score < self.config.deletion_score_threshold:
            await self._handle_deletion(doc_id, collection, metadata, score)

        return None

    async def _promote_working_to_history(
        self,
        doc_id: str,
        doc: Dict,
        metadata: Dict[str, Any],
        score: float,
        uses: int
    ) -> Optional[str]:
        """Promote from working to history collection."""
        new_id = doc_id.replace("working_", "history_")

        # Build promotion record
        promotion_record = {
            "from": "working",
            "to": "history",
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "uses": uses
        }
        promotion_history = json.loads(metadata.get("promotion_history", "[]"))
        promotion_history.append(promotion_record)
        metadata["promotion_history"] = json.dumps(promotion_history)
        metadata["promoted_from"] = "working"

        # v0.2.9: Reset counters on history entry - memory must prove itself fresh
        metadata["success_count"] = 0.0
        metadata["uses"] = 0
        metadata["promoted_to_history_at"] = datetime.now().isoformat()

        # Get text for embedding
        text_for_embedding = metadata.get("text") or metadata.get("content") or doc.get("content", "")
        if not text_for_embedding:
            logger.error(f"Cannot promote {doc_id}: no text found")
            return None

        try:
            await self.collections["history"].upsert_vectors(
                ids=[new_id],
                vectors=[await self.embed_fn(text_for_embedding)],
                metadatas=[metadata]
            )
            logger.info(f"Created history memory: {new_id}")
        except Exception as e:
            logger.error(f"Failed to create history memory {new_id}: {e}")
            return None

        # Only delete from working AFTER successful promotion
        self.collections["working"].delete_vectors([doc_id])

        # Track evolution relationship
        if self.add_relationship_fn:
            await self.add_relationship_fn(new_id, "evolution", {"parent": doc_id})

        logger.info(f"Promoted {doc_id} from working -> history (score: {score:.2f}, uses: {uses})")
        return new_id

    async def _promote_history_to_patterns(
        self,
        doc_id: str,
        doc: Dict,
        metadata: Dict[str, Any],
        score: float,
        uses: int
    ) -> Optional[str]:
        """Promote from history to patterns collection."""
        new_id = doc_id.replace("history_", "patterns_")

        # Build promotion record
        promotion_record = {
            "from": "history",
            "to": "patterns",
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "uses": uses
        }
        promotion_history = json.loads(metadata.get("promotion_history", "[]"))
        promotion_history.append(promotion_record)
        metadata["promotion_history"] = json.dumps(promotion_history)
        metadata["promoted_from"] = "history"

        # Get text for embedding
        text_for_embedding = metadata.get("text") or metadata.get("content") or doc.get("content", "")
        if not text_for_embedding:
            logger.error(f"Cannot promote {doc_id}: no text found")
            return None

        try:
            await self.collections["patterns"].upsert_vectors(
                ids=[new_id],
                vectors=[await self.embed_fn(text_for_embedding)],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Failed to create patterns memory {new_id}: {e}")
            return None

        self.collections["history"].delete_vectors([doc_id])

        if self.add_relationship_fn:
            await self.add_relationship_fn(new_id, "evolution", {"parent": doc_id})

        logger.info(f"Promoted {doc_id} from history -> patterns (score: {score:.2f}, uses: {uses})")
        return new_id

    async def _demote_patterns_to_history(
        self,
        doc_id: str,
        metadata: Dict[str, Any],
        score: float
    ) -> Optional[str]:
        """Demote from patterns back to history collection."""
        new_id = doc_id.replace("patterns_", "history_")

        text_for_embedding = metadata.get("text") or metadata.get("content", "")
        if not text_for_embedding:
            logger.error(f"Cannot demote {doc_id}: no text found")
            return None

        try:
            await self.collections["history"].upsert_vectors(
                ids=[new_id],
                vectors=[await self.embed_fn(text_for_embedding)],
                metadatas=[{**metadata, "demoted_from": "patterns"}]
            )
        except Exception as e:
            logger.error(f"Failed to demote {doc_id}: {e}")
            return None

        self.collections["patterns"].delete_vectors([doc_id])
        logger.info(f"Demoted {doc_id} to history (score: {score:.2f})")
        return new_id

    async def _handle_deletion(
        self,
        doc_id: str,
        collection: str,
        metadata: Dict[str, Any],
        score: float
    ):
        """Handle deletion of low-scoring memories."""
        # v0.2.9.1 FIX: memory_bank and books are PERMANENT - never delete based on score
        # Scores are only used for Wilson ranking, not lifecycle management
        if collection in ("memory_bank", "books"):
            logger.debug(f"[PROMOTION] Skipping deletion for {collection} (permanent collection)")
            return

        # Calculate age
        age_days = 0
        if metadata.get("timestamp"):
            try:
                age_days = (datetime.now() - datetime.fromisoformat(metadata["timestamp"])).days
            except Exception:
                pass

        # Newer items get more lenient threshold
        deletion_threshold = (
            self.config.deletion_score_threshold
            if age_days > 7
            else self.config.new_item_deletion_threshold
        )

        if score < deletion_threshold:
            self.collections[collection].delete_vectors([doc_id])
            logger.info(f"Deleted {doc_id} from {collection} (score {score:.2f} < threshold {deletion_threshold})")

    # =========================================================================
    # Batch Promotion
    # =========================================================================

    async def promote_valuable_working_memory(
        self,
        conversation_id: Optional[str] = None
    ) -> int:
        """
        Promote valuable working memory to history collection.

        This is typically run as a background task (hourly).

        Args:
            conversation_id: Optional filter (not enforced - working memory is global)

        Returns:
            Number of memories promoted
        """
        async with self._promotion_lock:
            return await self._do_batch_promotion()

    async def _do_batch_promotion(self) -> int:
        """Internal batch promotion logic."""
        try:
            working_adapter = self.collections.get("working")
            if not working_adapter or not working_adapter.collection:
                return 0

            promoted_count = 0
            checked_count = 0

            # Get all working memory items
            all_ids = working_adapter.list_all_ids()

            for doc_id in all_ids:
                doc = working_adapter.get_fragment(doc_id)
                if not doc:
                    continue

                metadata = doc.get("metadata", {})
                checked_count += 1

                # Get promotion criteria
                text = metadata.get("text", "")
                score = metadata.get("score", 0.5)
                uses = metadata.get("uses", 0)
                timestamp_str = metadata.get("timestamp") or metadata.get("created_at", "")

                # Calculate age
                age_hours = self._calculate_age_hours(timestamp_str)

                # Promote if: high score AND used multiple times
                if score >= self.config.promotion_score_threshold and uses >= 2:
                    new_id = doc_id.replace("working_", "history_")

                    await self.collections["history"].upsert_vectors(
                        ids=[new_id],
                        vectors=[await self.embed_fn(text)],
                        metadatas=[{
                            **metadata,
                            "promoted_from": "working",
                            "promotion_time": datetime.now().isoformat(),
                            "promotion_reason": "batch_promotion"
                        }]
                    )

                    working_adapter.delete_vectors([doc_id])
                    promoted_count += 1
                    logger.info(f"Promoted {doc_id} to history (score: {score:.2f}, uses: {uses}, age: {age_hours:.1f}h)")

                # Cleanup: Remove items older than 24 hours that weren't promoted
                elif age_hours > 24:
                    working_adapter.delete_vectors([doc_id])
                    logger.info(f"Cleaned up old working memory {doc_id} (age: {age_hours:.1f}h, score: {score:.2f})")

            if promoted_count > 0:
                logger.info(f"Batch promotion: checked {checked_count}, promoted {promoted_count} memories")

            return promoted_count

        except Exception as e:
            logger.error(f"Error in batch promotion: {e}")
            return 0

    def _calculate_age_hours(self, timestamp_str: str) -> float:
        """Calculate age in hours from timestamp string."""
        try:
            if timestamp_str:
                doc_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                return (datetime.now() - doc_time).total_seconds() / 3600
        except Exception:
            pass
        return 0.0

    # =========================================================================
    # Item Movement (generic)
    # =========================================================================

    async def promote_item(
        self,
        doc_id: str,
        from_collection: str,
        to_collection: str,
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Move an item from one collection to another (generic version).

        Args:
            doc_id: Source document ID
            from_collection: Source collection name
            to_collection: Target collection name
            metadata: Document metadata

        Returns:
            New document ID if successful, None otherwise
        """
        try:
            # Get the document from source collection
            doc = self.collections[from_collection].get_fragment(doc_id)
            if not doc:
                logger.warning(f"Cannot promote {doc_id}: not found in {from_collection}")
                return None

            # Add promotion metadata
            metadata["promoted_from"] = from_collection
            metadata["promoted_at"] = datetime.now().isoformat()
            metadata["original_id"] = doc_id

            # Generate new ID
            new_id = f"{to_collection}_{uuid.uuid4().hex[:8]}"

            # Get text for embedding
            text = doc.get("content", "") or metadata.get("text", "")
            if not text:
                logger.error(f"Cannot promote {doc_id}: no text found")
                return None

            # Store in target collection
            await self.collections[to_collection].upsert_vectors(
                ids=[new_id],
                vectors=[await self.embed_fn(text)],
                metadatas=[metadata]
            )

            logger.info(f"Promoted {doc_id} from {from_collection} to {to_collection} as {new_id}")
            return new_id

        except Exception as e:
            logger.error(f"Failed to promote item: {e}")
            return None

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup_old_working_memory(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up working memory items older than specified age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of items cleaned up
        """
        try:
            working_adapter = self.collections.get("working")
            if not working_adapter or not working_adapter.collection:
                return 0

            cleaned_count = 0
            all_ids = working_adapter.list_all_ids()

            for doc_id in all_ids:
                doc = working_adapter.get_fragment(doc_id)
                if not doc:
                    continue

                metadata = doc.get("metadata", {})
                timestamp_str = metadata.get("timestamp") or metadata.get("created_at", "")
                age_hours = self._calculate_age_hours(timestamp_str)

                if age_hours > max_age_hours:
                    working_adapter.delete_vectors([doc_id])
                    cleaned_count += 1
                    logger.debug(f"Cleaned up old working memory {doc_id} (age: {age_hours:.1f}h)")

            if cleaned_count > 0:
                logger.info(f"Working memory cleanup: removed {cleaned_count} old items")

            return cleaned_count

        except Exception as e:
            logger.error(f"Error in working memory cleanup: {e}")
            return 0

    async def cleanup_old_history(self, max_age_hours: float = 720.0) -> int:
        """
        Clean up history items older than specified age (default 30 days).

        History has a 30-day lifecycle: items older than 30 days are removed
        unless they've been promoted to patterns.

        Args:
            max_age_hours: Maximum age in hours (default 720 = 30 days)

        Returns:
            Number of items cleaned up
        """
        try:
            history_adapter = self.collections.get("history")
            if not history_adapter or not history_adapter.collection:
                return 0

            cleaned_count = 0
            all_ids = history_adapter.list_all_ids()

            for doc_id in all_ids:
                doc = history_adapter.get_fragment(doc_id)
                if not doc:
                    continue

                metadata = doc.get("metadata", {})
                timestamp_str = metadata.get("timestamp") or metadata.get("created_at", "")
                age_hours = self._calculate_age_hours(timestamp_str)

                if age_hours > max_age_hours:
                    history_adapter.delete_vectors([doc_id])
                    cleaned_count += 1
                    logger.debug(f"Cleaned up old history {doc_id} (age: {age_hours:.1f}h)")

            if cleaned_count > 0:
                logger.info(f"History cleanup: removed {cleaned_count} items older than {max_age_hours/24:.0f} days")

            return cleaned_count

        except Exception as e:
            logger.error(f"Error in history cleanup: {e}")
            return 0
