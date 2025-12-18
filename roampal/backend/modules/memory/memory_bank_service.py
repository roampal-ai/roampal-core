"""
MemoryBankService - Extracted from UnifiedMemorySystem

Handles memory_bank operations for user identity, preferences, and facts.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Awaitable

from .config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryBankService:
    """
    Service for managing memory_bank collection.

    Memory bank stores persistent user information like:
    - Identity (name, role, company)
    - Preferences (coding style, tools)
    - Goals and projects
    - Learned facts about the user

    Unlike working/history/patterns, memory_bank items:
    - Are not scored by outcomes
    - Use importance/confidence instead
    - Can be archived but not promoted/demoted
    """

    # Capacity limit for memory bank
    MAX_ITEMS = 500

    def __init__(
        self,
        collection: Any,
        embed_fn: Callable[[str], Awaitable[List[float]]],
        search_fn: Optional[Callable] = None,
        config: Optional[MemoryConfig] = None
    ):
        """
        Initialize MemoryBankService.

        Args:
            collection: Memory bank collection adapter
            embed_fn: Async function to embed text
            search_fn: Optional search function for queries
            config: Memory configuration
        """
        self.collection = collection
        self.embed_fn = embed_fn
        self.search_fn = search_fn
        self.config = config or MemoryConfig()

    async def store(
        self,
        text: str,
        tags: List[str],
        importance: float = 0.7,
        confidence: float = 0.7
    ) -> str:
        """
        Store user memory in memory_bank collection.

        Args:
            text: Memory content
            tags: List of tags (identity, preference, project, context, goal)
            importance: 0.0-1.0 (how critical is this memory)
            confidence: 0.0-1.0 (how sure are we about this)

        Returns:
            Document ID

        Raises:
            ValueError: If memory bank is at capacity
        """
        # Capacity check (skip in benchmark mode - uncapped in 0.2.8)
        benchmark_mode = os.environ.get("ROAMPAL_BENCHMARK_MODE", "").lower() == "true"
        if not benchmark_mode:
            current_count = self._get_count()
            if current_count >= self.MAX_ITEMS:
                error_msg = (
                    f"Memory bank at capacity ({current_count}/{self.MAX_ITEMS}). "
                    "Please archive or delete old memories."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        doc_id = f"memory_bank_{uuid.uuid4().hex[:8]}"

        # Generate embedding
        embedding = await self.embed_fn(text)

        # Build metadata
        metadata = {
            "text": text,
            "content": text,
            "tags": json.dumps(tags),
            "importance": importance,
            "confidence": confidence,
            "score": 1.0,  # Never decays
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "mentioned_count": 1,
            "last_mentioned": datetime.now().isoformat()
        }

        # Store in collection
        await self.collection.upsert_vectors(
            ids=[doc_id],
            vectors=[embedding],
            metadatas=[metadata]
        )

        logger.info(f"Stored memory_bank item: {text[:50]}... (tags: {tags})")
        return doc_id

    async def update(
        self,
        doc_id: str,
        new_text: str,
        reason: str = "llm_update"
    ) -> str:
        """
        Update memory with auto-archiving of old version.

        Args:
            doc_id: Memory to update
            new_text: New content
            reason: Why updating (for audit trail)

        Returns:
            Document ID
        """
        # Get current version
        old_doc = self.collection.get_fragment(doc_id)
        if not old_doc:
            logger.warning(f"Memory {doc_id} not found, creating new")
            return await self.store(new_text, tags=["updated"])

        # Auto-archive old version
        archive_id = f"{doc_id}_archived_{int(datetime.now().timestamp())}"
        old_text = old_doc.get("metadata", {}).get("content",
                   old_doc.get("metadata", {}).get("text", ""))
        old_embedding = await self.embed_fn(old_text)

        await self.collection.upsert_vectors(
            ids=[archive_id],
            vectors=[old_embedding],
            metadatas=[{
                **old_doc.get("metadata", {}),
                "status": "archived",
                "original_id": doc_id,
                "archive_reason": reason,
                "archived_at": datetime.now().isoformat()
            }]
        )

        # Update in-place
        new_embedding = await self.embed_fn(new_text)
        old_metadata = old_doc.get("metadata", {})

        await self.collection.upsert_vectors(
            ids=[doc_id],
            vectors=[new_embedding],
            metadatas=[{
                **old_metadata,
                "text": new_text,
                "content": new_text,
                "updated_at": datetime.now().isoformat(),
                "update_reason": reason
            }]
        )

        logger.info(f"Updated memory_bank item {doc_id}: {reason}")
        return doc_id

    async def archive(
        self,
        content: str,
        reason: str = "llm_decision"
    ) -> bool:
        """
        Archive memory by content (semantic match).

        Args:
            content: Content to find and archive (semantic match)
            reason: Why archiving

        Returns:
            Success status
        """
        # Search for the memory by content to find its doc_id
        if self.search_fn:
            results = await self.search_fn(
                query=content,
                collections=["memory_bank"],
                limit=5
            )

            # Find best match - look for exact or close content match
            doc_id = None
            for r in results:
                r_content = r.get("content") or r.get("metadata", {}).get("content", "")
                # Check if content matches (exact or substring)
                if content in r_content or r_content in content:
                    doc_id = r.get("id")
                    break

            if not doc_id and results:
                # Fall back to top result if no exact match
                doc_id = results[0].get("id")
        else:
            # No search function - try content as doc_id (backwards compat)
            doc_id = content

        if not doc_id:
            logger.warning(f"Could not find memory to archive: {content[:50]}...")
            return False

        doc = self.collection.get_fragment(doc_id)
        if not doc:
            logger.warning(f"Memory {doc_id} not found in collection")
            return False

        metadata = doc.get("metadata", {})
        metadata["status"] = "archived"
        metadata["archive_reason"] = reason
        metadata["archived_at"] = datetime.now().isoformat()

        self.collection.update_fragment_metadata(doc_id, metadata)
        logger.info(f"Archived memory_bank item {doc_id}: {reason}")
        return True

    async def search(
        self,
        query: str = None,
        tags: List[str] = None,
        include_archived: bool = False,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search memory_bank collection with filtering.

        Args:
            query: Semantic search query (None = get all)
            tags: Filter by tags
            include_archived: Include archived memories
            limit: Max results

        Returns:
            List of memories
        """
        # Use provided search function or basic vector search
        if query and self.search_fn:
            results = await self.search_fn(
                query=query,
                collections=["memory_bank"],
                limit=limit * 2  # Get extra for filtering
            )
        else:
            # Fallback: get all and filter
            results = await self._get_all_items(limit * 2)

        # Filter by status and tags
        filtered = []
        for r in results:
            metadata = r.get("metadata", {})
            status = metadata.get("status", "active")

            # Skip archived unless requested
            if status == "archived" and not include_archived:
                continue

            # Filter by tags if specified
            if tags:
                doc_tags = json.loads(metadata.get("tags", "[]"))
                if not any(tag in doc_tags for tag in tags):
                    continue

            filtered.append(r)

        return filtered[:limit]

    async def restore(self, doc_id: str) -> bool:
        """
        User manually restores archived memory.

        Args:
            doc_id: Memory to restore

        Returns:
            Success status
        """
        doc = self.collection.get_fragment(doc_id)
        if not doc:
            return False

        metadata = doc.get("metadata", {})
        metadata["status"] = "active"
        metadata["restored_at"] = datetime.now().isoformat()
        metadata["restored_by"] = "user"

        self.collection.update_fragment_metadata(doc_id, metadata)
        logger.info(f"User restored memory: {doc_id}")
        return True

    async def delete(self, doc_id: str) -> bool:
        """
        User permanently deletes memory (hard delete).

        Args:
            doc_id: Memory to delete

        Returns:
            Success status
        """
        try:
            self.collection.delete_vectors([doc_id])
            logger.info(f"User permanently deleted memory: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {doc_id}: {e}")
            return False

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.

        Args:
            doc_id: Memory ID

        Returns:
            Memory document or None
        """
        return self.collection.get_fragment(doc_id)

    def list_all(
        self,
        include_archived: bool = False,
        tags: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all memories with optional filtering.

        Args:
            include_archived: Include archived memories
            tags: Filter by tags

        Returns:
            List of memories
        """
        all_ids = self.collection.list_all_ids()
        results = []

        for doc_id in all_ids:
            doc = self.collection.get_fragment(doc_id)
            if doc:
                metadata = doc.get("metadata", {})
                status = metadata.get("status", "active")

                # Skip archived unless requested
                if status == "archived" and not include_archived:
                    continue

                # Filter by tags if specified
                if tags:
                    doc_tags = json.loads(metadata.get("tags", "[]"))
                    if not any(tag in doc_tags for tag in tags):
                        continue

                results.append({
                    "id": doc_id,
                    "content": doc.get("content", ""),
                    "metadata": metadata
                })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory bank statistics.

        Returns:
            Dict with stats
        """
        all_items = self.list_all(include_archived=True)

        active = [i for i in all_items if i["metadata"].get("status") == "active"]
        archived = [i for i in all_items if i["metadata"].get("status") == "archived"]

        # Count by tags
        tag_counts = {}
        for item in active:
            tags = json.loads(item["metadata"].get("tags", "[]"))
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Average importance/confidence
        importances = [i["metadata"].get("importance", 0.7) for i in active]
        confidences = [i["metadata"].get("confidence", 0.7) for i in active]

        return {
            "total": len(all_items),
            "active": len(active),
            "archived": len(archived),
            "capacity": self.MAX_ITEMS,
            "usage_percent": len(active) / self.MAX_ITEMS * 100,
            "tag_counts": tag_counts,
            "avg_importance": sum(importances) / len(importances) if importances else 0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0
        }

    def increment_mention(self, doc_id: str) -> bool:
        """
        Increment mention count for a memory.

        Args:
            doc_id: Memory ID

        Returns:
            Success status
        """
        doc = self.collection.get_fragment(doc_id)
        if not doc:
            return False

        metadata = doc.get("metadata", {})
        metadata["mentioned_count"] = metadata.get("mentioned_count", 0) + 1
        metadata["last_mentioned"] = datetime.now().isoformat()

        self.collection.update_fragment_metadata(doc_id, metadata)
        return True

    def _get_count(self) -> int:
        """Get current item count."""
        try:
            return self.collection.collection.count()
        except Exception as e:
            logger.warning(f"Could not get memory_bank count: {e}")
            return 0

    async def _get_all_items(self, limit: int) -> List[Dict[str, Any]]:
        """Get all items (fallback when no search query)."""
        all_ids = self.collection.list_all_ids()
        results = []

        for doc_id in all_ids[:limit]:
            doc = self.collection.get_fragment(doc_id)
            if doc:
                results.append({
                    "id": doc_id,
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "distance": 0  # No distance for direct fetch
                })

        return results
