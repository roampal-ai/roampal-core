"""
UnifiedMemorySystem - Facade coordinating memory services

Simplified from Roampal ui-implementation for roampal-core.
Stripped: Ollama, KG service, complex routing.
Kept: Core search, outcome tracking, memory bank operations.
"""

import asyncio
import logging
import json
import sys
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field

from .config import MemoryConfig
from .chromadb_adapter import ChromaDBAdapter
from .embedding_service import EmbeddingService
from .scoring_service import ScoringService
from .outcome_service import OutcomeService
from .memory_bank_service import MemoryBankService
from .context_service import ContextService
from .promotion_service import PromotionService
from .routing_service import RoutingService
from .tag_service import TagService
from .search_service import SearchService

logger = logging.getLogger(__name__)

CollectionName = Literal["books", "working", "history", "patterns", "memory_bank"]

# ContextType is any string - LLM discovers topics organically (coding, fitness, finance, etc.)
ContextType = str

# v0.2.0: Maximum content size for book ingestion (10MB)
MAX_BOOK_SIZE = 10 * 1024 * 1024  # 10MB in bytes


@dataclass
class ActionOutcome:
    """
    Tracks individual action outcomes with topic-based context awareness (v0.2.1 Causal Learning).

    Enables learning: "In topic X, action Y leads to outcome Z"

    Examples:
    - For CODING: search_memory → 92% success (searching code patterns works well)
    - For FITNESS: create_memory → 88% success (storing workout logs works well)

    Context is detected from conversation (coding, fitness, finance, creative_writing, etc.)
    """
    action_type: str  # Tool name: "search_memory", "create_memory", "score_memories", etc.
    context_type: ContextType  # Topic: "coding", "fitness", "finance", etc.
    outcome: Literal["worked", "failed", "partial"]
    timestamp: datetime = field(default_factory=datetime.now)

    # Action details
    action_params: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    collection: Optional[str] = None

    # Outcome details
    failure_reason: Optional[str] = None
    success_context: Optional[Dict[str, Any]] = None

    # Causal attribution
    chain_position: int = 0  # Position in action chain (0 = first action)
    chain_length: int = 1  # Total actions in chain
    caused_final_outcome: bool = True  # Did this action cause the final outcome?

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for KG storage"""
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
        """Deserialize from dict"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


def _humanize_age(iso_timestamp: str) -> str:
    """Convert ISO timestamp to human-readable relative age like '2d', '5h'."""
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        # Timestamps are stored as naive local time — compare against naive local now
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        now = datetime.now()
        delta = now - dt
        days = delta.days
        hours = delta.seconds // 3600
        minutes = delta.seconds // 60
        if days > 365:
            return f"{days // 365}y"
        elif days > 30:
            return f"{days // 30}mo"
        elif days > 0:
            return f"{days}d"
        elif hours > 0:
            return f"{hours}h"
        elif minutes > 0:
            return f"{minutes}m"
        else:
            return "now"
    except Exception:
        return ""


def normalize_memory(result: Dict, collection: str = None) -> Dict:
    """
    Standardize memory shape across all retrieval paths.

    After normalization, every memory has consistent fields:
    - content, id, collection at root level
    - created_at (standardized from created_at or timestamp)
    - age (relative time string)
    - score, uses, success_count, last_outcome, outcome_history
    - tags (parsed list, empty [] for non-memory_bank)
    - importance/confidence (memory_bank only)
    - metadata preserved as-is
    """
    metadata = result.get("metadata", {})

    # === Content: ONE canonical location ===
    content = (
        result.get("content")
        or metadata.get("content")
        or metadata.get("text")
        or result.get("text", "")
    )
    result["content"] = content
    result.pop("text", None)

    # === Collection ===
    if not collection:
        collection = result.get("collection") or ""
        if not collection and result.get("id"):
            for prefix in ["working", "history", "patterns", "memory_bank", "books"]:
                if result["id"].startswith(prefix + "_"):
                    collection = prefix
                    break
    result["collection"] = collection

    # === Timestamps: standardize to created_at ===
    created_at = (
        metadata.get("created_at")
        or metadata.get("timestamp")
        or ""
    )
    result["created_at"] = created_at
    metadata["created_at"] = created_at  # Write back so _apply_date_filters() sees it
    result["age"] = _humanize_age(created_at)

    # === Outcome scoring (always present, defaults for unscored) ===
    result["score"] = float(metadata.get("score", 0.5))
    result["uses"] = int(metadata.get("uses", 0))
    result["success_count"] = float(metadata.get("success_count", 0.0))
    result["last_outcome"] = metadata.get("last_outcome", "")

    # Format outcome history
    outcome_history_raw = metadata.get("outcome_history", "")
    if isinstance(outcome_history_raw, str) and outcome_history_raw:
        try:
            history = json.loads(outcome_history_raw)
            if history:
                symbols = []
                for entry in history[-3:]:
                    outcome = entry.get("outcome", "")
                    if outcome == "worked":
                        symbols.append("Y")
                    elif outcome == "failed":
                        symbols.append("N")
                    elif outcome == "partial":
                        symbols.append("~")
                result["outcome_history"] = "[" + "".join(symbols) + "]" if symbols else ""
            else:
                result["outcome_history"] = ""
        except (json.JSONDecodeError, TypeError):
            result["outcome_history"] = ""
    else:
        result["outcome_history"] = ""

    # Wilson score: compute if not already present
    if "wilson_score" not in result:
        uses = result["uses"]
        success_count = result["success_count"]
        if uses > 0 and success_count is not None:
            # Simple Wilson lower bound approximation
            p = success_count / uses if uses > 0 else 0.5
            z = 1.96  # 95% confidence
            n = uses
            denominator = 1 + z * z / n
            centre = p + z * z / (2 * n)
            spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
            result["wilson_score"] = round((centre - spread) / denominator, 4)
        else:
            result["wilson_score"] = 0.5  # Untested default

    if "learned_score" not in result:
        result["learned_score"] = result.get("score", 0.5)

    # === Tags: always present ===
    tags_raw = metadata.get("tags", "[]")
    if isinstance(tags_raw, str):
        try:
            result["tags"] = json.loads(tags_raw)
        except json.JSONDecodeError:
            result["tags"] = []
    elif isinstance(tags_raw, list):
        result["tags"] = tags_raw
    else:
        result["tags"] = []

    # === Memory bank specific: importance/confidence ===
    if collection == "memory_bank":
        result["importance"] = float(metadata.get("importance", 0.7))
        result["confidence"] = float(metadata.get("confidence", 0.7))

    # Preserve raw metadata
    result["metadata"] = metadata

    return result


class UnifiedMemorySystem:
    """
    The unified memory system for roampal-core.

    Coordinates services for:
    - Multi-collection vector search
    - Outcome-based learning
    - Memory bank (user facts)
    - Context analysis for hook injection

    5 Collections:
    - books: Uploaded reference material (never decays)
    - working: Current session context (session-scoped)
    - history: Past conversations (auto-promoted to patterns)
    - patterns: Proven solutions (what actually worked)
    - memory_bank: Persistent user facts (LLM-controlled, never decays)
    """

    def __init__(
        self,
        data_path: str = None,
        config: Optional[MemoryConfig] = None
    ):
        """
        Initialize UnifiedMemorySystem.

        Args:
            data_path: Path for ChromaDB storage. Defaults to %APPDATA%/Roampal/data
            config: Memory configuration
        """
        self.config = config or MemoryConfig()

        # Default data path - same as Roampal Desktop
        # Can override with ROAMPAL_DATA_PATH env var
        # Windows: %APPDATA%/Roampal/data (or dev-data for dev mode)
        # macOS: ~/Library/Application Support/Roampal/data
        # Linux: ~/.local/share/Roampal/data
        if data_path is None:
            # Check for env override first
            data_path = os.environ.get('ROAMPAL_DATA_PATH')

            if data_path is None:
                # Check for dev mode - matches Desktop's ROAMPAL_DATA_DIR convention
                # DEV: Roampal_DEV/data, PROD: Roampal/data
                dev_mode = os.environ.get('ROAMPAL_DEV', '').lower() in ('1', 'true', 'yes')
                app_folder = 'Roampal_DEV' if dev_mode else 'Roampal'

                if os.name == 'nt':  # Windows
                    appdata = os.environ.get('APPDATA', os.path.expanduser('~'))
                    data_path = os.path.join(appdata, app_folder, 'data')
                elif sys.platform == 'darwin':  # macOS
                    data_path = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', app_folder, 'data')
                else:  # Linux — v0.4.1: respect XDG_DATA_HOME
                    xdg_data = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser('~'), '.local', 'share'))
                    data_path = os.path.join(xdg_data, app_folder.lower(), 'data')
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Services (lazy initialized)
        self._embedding_service: Optional[EmbeddingService] = None
        self._scoring_service: Optional[ScoringService] = None
        self._promotion_service: Optional[PromotionService] = None
        self._outcome_service: Optional[OutcomeService] = None
        self._memory_bank_service: Optional[MemoryBankService] = None
        self._context_service: Optional[ContextService] = None
        self._search_service: Optional[SearchService] = None

        # Collections
        self.collections: Dict[str, ChromaDBAdapter] = {}
        self.initialized = False

        # v0.4.5: KG removed. Tag service handles routing now.

        # Ghost Registry - tracks deleted book IDs without modifying ChromaDB
        # v0.2.2: Non-destructive delete to avoid HNSW index corruption
        self.ghost_registry_path = self.data_path / "ghost_ids.json"
        self.ghost_ids: set = self._load_ghost_registry()

    # v0.4.5: _load_kg removed — KG replaced by tag-routed retrieval

    def _load_ghost_registry(self) -> set:
        """
        Load ghost registry from disk.

        v0.2.2: Ghost IDs are book chunk IDs that have been "deleted" but remain
        in ChromaDB to avoid HNSW index corruption. These get filtered from search.
        """
        if self.ghost_registry_path.exists():
            try:
                with open(self.ghost_registry_path, 'r') as f:
                    data = json.load(f)
                    return set(data.get("ghost_ids", []))
            except Exception as e:
                logger.warning(f"Failed to load ghost registry: {e}")
        return set()

    def _save_ghost_registry(self):
        """Save ghost registry to disk."""
        try:
            with open(self.ghost_registry_path, 'w') as f:
                json.dump({"ghost_ids": list(self.ghost_ids)}, f, indent=2)
            logger.debug(f"Saved ghost registry with {len(self.ghost_ids)} IDs")
        except Exception as e:
            logger.error(f"Failed to save ghost registry: {e}")

    def _migrate_chromadb_schema(self):
        """
        Migrate ChromaDB schema for compatibility across versions.

        v0.2.2: ChromaDB 1.x added 'topic' column to collections and segments tables.
        Users upgrading from ChromaDB 0.4.x/0.5.x will have old schema.
        This safely adds missing columns without affecting existing data.
        """
        import sqlite3

        chromadb_path = self.data_path / "chromadb"
        sqlite_path = chromadb_path / "chroma.sqlite3"

        if not sqlite_path.exists():
            logger.debug("No existing ChromaDB - skipping migration")
            return

        try:
            conn = sqlite3.connect(str(sqlite_path))
            cursor = conn.cursor()

            # Columns added in ChromaDB 1.x that may be missing
            migrations_needed = []

            # Check collections table
            cursor.execute("PRAGMA table_info(collections)")
            collections_columns = {col[1] for col in cursor.fetchall()}
            if 'topic' not in collections_columns:
                migrations_needed.append(('collections', 'topic', 'TEXT'))

            # Check segments table (also needs 'topic' in ChromaDB 1.x)
            cursor.execute("PRAGMA table_info(segments)")
            segments_columns = {col[1] for col in cursor.fetchall()}
            if 'topic' not in segments_columns:
                migrations_needed.append(('segments', 'topic', 'TEXT'))

            # Apply migrations
            for table, column, col_type in migrations_needed:
                try:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    logger.info(f"ChromaDB migration: Added {column} to {table}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        pass  # Column already exists, safe to ignore
                    else:
                        raise

            conn.commit()
            conn.close()

            if migrations_needed:
                logger.info(f"ChromaDB schema migration complete: {len(migrations_needed)} columns added")
            else:
                logger.debug("ChromaDB schema up to date")

        except Exception as e:
            logger.warning(f"ChromaDB schema migration failed (non-fatal): {e}")
            # Don't raise - let ChromaDB try to initialize anyway
            # Worst case: original error surfaces with better context

    async def initialize(self):
        """Initialize collections and services."""
        if self.initialized:
            return

        logger.info("Initializing UnifiedMemorySystem...")

        # v0.2.2: Migrate ChromaDB schema before initialization
        # Handles upgrades from ChromaDB 0.4.x to 1.x
        self._migrate_chromadb_schema()

        # Initialize embedding service
        self._embedding_service = EmbeddingService()
        await self._embedding_service.prewarm()

        # Initialize collections - use roampal_ prefix to match Desktop data
        # Keys are short names (for code), values use roampal_ prefix in chromadb
        collection_mapping = {
            "books": "roampal_books",
            "working": "roampal_working",
            "history": "roampal_history",
            "patterns": "roampal_patterns",
            "memory_bank": "roampal_memory_bank"
        }
        # v0.4.4: Initialize all collection adapters concurrently
        adapters = []
        for short_name, chroma_name in collection_mapping.items():
            adapter = ChromaDBAdapter(
                collection_name=chroma_name,
                persist_directory=str(self.data_path / "chromadb")
            )
            adapters.append((short_name, chroma_name, adapter))

        await asyncio.gather(*(adapter.initialize() for _, _, adapter in adapters))
        for short_name, chroma_name, adapter in adapters:
            self.collections[short_name] = adapter
            logger.info(f"Initialized collection: {short_name} -> {chroma_name}")

        # Initialize services with dependencies
        self._scoring_service = ScoringService(config=self.config)

        # v0.4.5: Tag service replaces KG
        # Regex-only extraction for now. Claude Code: main LLM will pass
        # noun_tags via MCP tool params (future). OpenCode: sidecar handles
        # tag extraction separately. No sidecar LLM calls from server side.
        self._tag_service = TagService()

        # Routing service (v0.4.5: simplified, no KG)
        self._routing_service = RoutingService(config=self.config)

        # Initialize PromotionService for auto-deletion/promotion when scores drop
        self._promotion_service = PromotionService(
            collections=self.collections,
            embed_fn=self._embedding_service.embed_text,
            config=self.config
        )

        # OutcomeService wired to PromotionService — garbage now gets deleted
        self._outcome_service = OutcomeService(
            collections=self.collections,
            promotion_service=self._promotion_service,
            config=self.config
        )

        self._memory_bank_service = MemoryBankService(
            collection=self.collections["memory_bank"],
            embed_fn=self._embedding_service.embed_text,
            config=self.config,
            search_fn=self.search
        )

        self._context_service = ContextService(
            collections=self.collections,
            embed_fn=self._embedding_service.embed_text,
            config=self.config
        )

        # v0.4.5: Tag-routed search replaces KG-based search
        self._search_service = SearchService(
            collections=self.collections,
            scoring_service=self._scoring_service,
            routing_service=self._routing_service,
            tag_service=self._tag_service,
            embed_fn=self._embedding_service.embed_text,
            config=self.config,
        )

        # v0.4.5: Rebuild known tag index from existing memories
        self._tag_service.rebuild_known_tags(self.collections)

        self.initialized = True
        logger.info("UnifiedMemorySystem initialized successfully")

        # Startup cleanup: delete garbage memories (score < 0.2)
        await self._startup_cleanup()


    async def _startup_cleanup(self):
        """
        Delete garbage memories on startup.

        Scans working, history, and patterns collections for items with score < 0.2
        and deletes them. This ensures garbage doesn't pile up between sessions.
        """
        deleted_count = 0
        collections_to_clean = ["working", "history", "patterns"]

        for coll_name in collections_to_clean:
            adapter = self.collections.get(coll_name)
            if not adapter:
                continue

            try:
                # Ensure collection is initialized before cleanup
                await adapter._ensure_initialized()
                if not adapter.collection:
                    continue
                # Get all documents in collection
                results = adapter.collection.get(include=["metadatas"])
                ids = results.get("ids", [])
                metadatas = results.get("metadatas", [])

                ids_to_delete = []
                for i, doc_id in enumerate(ids):
                    if i < len(metadatas):
                        score = metadatas[i].get("score", 0.5)
                        if score < self.config.deletion_score_threshold:
                            ids_to_delete.append(doc_id)

                if ids_to_delete:
                    adapter.delete_vectors(ids_to_delete)
                    deleted_count += len(ids_to_delete)
                    logger.info(f"Startup cleanup: deleted {len(ids_to_delete)} garbage items from {coll_name}")

            except Exception as e:
                logger.warning(f"Startup cleanup error for {coll_name}: {e}")

        if deleted_count > 0:
            logger.info(f"Startup cleanup: {deleted_count} garbage deleted")

        # v0.4.4: Run both cleanup tasks concurrently
        await asyncio.gather(
            self._promotion_service.cleanup_old_working_memory(max_age_hours=24.0),
            self._promotion_service.cleanup_old_history(max_age_hours=720.0),
        )

    # ==================== Core Search ====================

    async def search(
        self,
        query: str,
        limit: int = 10,
        collections: Optional[List[CollectionName]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memory with optional collection filtering.

        v0.4.5: Delegates to SearchService for TagCascade retrieval
        (tags-first cascade + cross-encoder reranking).

        Args:
            query: Search query
            limit: Max results per collection
            collections: Which collections to search (default: all)
            metadata_filters: Optional metadata filters
            sort_by: Sort order - "relevance" (default), "recency", or "score"

        Returns:
            Ranked results with scores
        """
        if not self.initialized:
            await self.initialize()

        # Delegate to SearchService (TagCascade + cross-encoder)
        if self._search_service:
            try:
                all_results = await self._search_service.search(
                    query=query,
                    limit=limit,
                    collections=collections,
                    metadata_filters=metadata_filters,
                    sort_by=sort_by,
                )

                # Filter ghost IDs (deleted books)
                all_results = [r for r in all_results if r.get("id", "") not in self.ghost_ids]

                # Add legacy field aliases for Desktop compatibility
                for r in all_results:
                    r["quality"] = r.get("learned_score", 0.5)
                    r["combined_score"] = r.get("final_rank_score", 0)
                    r["similarity"] = r.get("embedding_similarity", 0)

                # Apply sort_by override
                if sort_by == "recency":
                    def get_timestamp(x):
                        metadata = x.get("metadata", {})
                        ts = metadata.get("timestamp") or metadata.get("created_at") or ""
                        return ts if ts else "0"
                    all_results.sort(key=get_timestamp, reverse=True)
                    return all_results[:limit]
                elif sort_by == "score":
                    all_results.sort(key=lambda x: x.get("metadata", {}).get("score", 0.5), reverse=True)
                    return all_results[:limit]
                else:
                    # Default: already sorted by final_rank_score from SearchService
                    return all_results[:limit * 2]

            except Exception as e:
                logger.warning(f"SearchService search failed, falling back to inline: {e}")

        # Fallback: inline search (pre-initialization or SearchService failure)
        if collections is None:
            collections = list(self.collections.keys())

        query_vector = await self._embedding_service.embed_text(query)

        all_results = []
        for coll_name in collections:
            if coll_name not in self.collections:
                continue

            try:
                results = await self.collections[coll_name].query_vectors(
                    query_vector=query_vector,
                    top_k=limit,
                    filters=metadata_filters
                )

                for r in results:
                    doc_id = r.get("id", "")
                    if doc_id in self.ghost_ids:
                        continue

                    r["collection"] = coll_name
                    metadata = r.get("metadata", {})
                    distance = r.get("distance", 1.0)

                    if self._scoring_service:
                        scores = self._scoring_service.calculate_final_score(
                            metadata, distance, coll_name
                        )
                        embedding_similarity = scores["embedding_similarity"]
                        final_rank_score = scores["final_rank_score"]
                        quality = scores.get("learned_score", 0.5)
                    else:
                        embedding_similarity = 1.0 / (1.0 + distance)
                        quality = float(metadata.get("score", 0.5))
                        quality_boost = 1.0 - (quality * 0.8)
                        adjusted_distance = distance * quality_boost
                        adjusted_similarity = 1.0 / (1.0 + adjusted_distance)
                        final_rank_score = adjusted_similarity * (1.0 + quality)

                    r["embedding_similarity"] = embedding_similarity
                    r["final_rank_score"] = final_rank_score
                    r["quality"] = quality
                    r["combined_score"] = final_rank_score
                    r["similarity"] = embedding_similarity

                    all_results.append(r)

            except Exception as e:
                logger.warning(f"Error searching {coll_name}: {e}")

        if sort_by == "recency":
            def get_timestamp(x):
                metadata = x.get("metadata", {})
                ts = metadata.get("timestamp") or metadata.get("created_at") or ""
                return ts if ts else "0"
            all_results.sort(key=get_timestamp, reverse=True)
            return all_results[:limit]
        elif sort_by == "score":
            all_results.sort(key=lambda x: x.get("metadata", {}).get("score", 0.5), reverse=True)
            return all_results[:limit]
        else:
            all_results.sort(key=lambda x: x.get("final_rank_score", 0), reverse=True)
            return all_results[:limit * 2]

    # ==================== Generic Store (Desktop-compatible wrapper) ====================

    async def store(
        self,
        text: str,
        collection: str = "working",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store text in a collection (Desktop-compatible wrapper).

        Args:
            text: Text to store
            collection: Target collection (working, books, history, patterns, memory_bank)
            metadata: Optional metadata

        Returns:
            Document ID
        """
        if not self.initialized:
            await self.initialize()

        # Route to appropriate collection-specific method
        if collection == "working":
            return await self.store_working(content=text, metadata=metadata)
        elif collection == "books":
            # store_book returns list of chunk IDs; return first for Desktop compat
            chunk_ids = await self.store_book(text)
            return chunk_ids[0] if chunk_ids else None
        elif collection == "memory_bank":
            tags = metadata.get("tags", []) if metadata else []
            importance = metadata.get("importance", 0.7) if metadata else 0.7
            confidence = metadata.get("confidence", 0.7) if metadata else 0.7
            return await self.store_memory_bank(text, tags, importance, confidence)
        elif collection in ("patterns", "history"):
            # Direct store to patterns/history (Desktop-compatible)
            import uuid
            doc_id = f"{collection}_{uuid.uuid4().hex[:8]}"
            final_metadata = {
                "text": text,
                "content": text,
                "score": 0.5,
                "uses": 0,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            embedding = await self._embedding_service.embed_text(text)
            await self.collections[collection].upsert_vectors(
                ids=[doc_id],
                vectors=[embedding],
                metadatas=[final_metadata]
            )

            # v0.4.5: Extract noun tags for TagCascade retrieval
            if hasattr(self, '_tag_service') and self._tag_service:
                try:
                    tags = self._tag_service.extract_tags(text)
                    if tags:
                        self.collections[collection].update_fragment_metadata(
                            doc_id, {"noun_tags": json.dumps(tags)}
                        )
                except Exception as e:
                    logger.warning(f"Tag extraction failed for {doc_id}: {e}")

            return doc_id
        else:
            # Unknown collection - default to working
            return await self.store_working(content=text, metadata=metadata)

    # ==================== Memory Bank Operations ====================

    async def store_memory_bank(
        self,
        text: str,
        tags: List[str] = None,
        noun_tags: Optional[List[str]] = None,
        importance: float = 0.7,
        confidence: float = 0.7,
        always_inject: bool = False
    ) -> str:
        """
        Store a fact in memory_bank.

        Args:
            text: The fact to remember
            tags: Categories (identity, preference, goal, project)
            noun_tags: Content nouns for TagCascade retrieval (from LLM)
            importance: How critical (0.0-1.0)
            confidence: How certain (0.0-1.0)
            always_inject: If True, appears in every context

        Returns:
            Document ID
        """
        if not self.initialized:
            await self.initialize()

        doc_id = await self._memory_bank_service.store(
            text=text,
            tags=tags or [],
            importance=importance,
            confidence=confidence,
            always_inject=always_inject
        )

        # v0.4.5: noun_tags for TagCascade retrieval
        if noun_tags:
            self.collections["memory_bank"].update_fragment_metadata(
                doc_id, {"noun_tags": json.dumps(noun_tags)}
            )
            if hasattr(self, '_tag_service') and self._tag_service:
                self._tag_service.add_known_tags(noun_tags)
        elif hasattr(self, '_tag_service') and self._tag_service:
            try:
                extracted = self._tag_service.extract_tags(text)
                if extracted:
                    self.collections["memory_bank"].update_fragment_metadata(
                        doc_id, {"noun_tags": json.dumps(extracted)}
                    )
            except Exception as e:
                logger.warning(f"Tag extraction failed for {doc_id}: {e}")

        return doc_id

    async def update_memory_bank(
        self,
        old_content: str,
        new_content: str
    ) -> Optional[str]:
        """
        Update a memory_bank entry.

        Args:
            old_content: Content to find (semantic match)
            new_content: New content

        Returns:
            Document ID or None if not found
        """
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.update(old_content, new_content)

    async def delete_memory_bank(self, content: str) -> bool:
        """
        Archive a memory_bank entry.

        Args:
            content: Content to archive (semantic match)

        Returns:
            Success status
        """
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.archive(content)

    def get_always_inject(self) -> List[Dict[str, Any]]:
        """
        Get all memories marked with always_inject: true.

        These are core identity facts that should appear in EVERY context
        regardless of semantic relevance to the query.

        Returns:
            List of always_inject memories
        """
        if not self._memory_bank_service:
            return []
        return self._memory_bank_service.get_always_inject()

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID from any collection.

        Args:
            doc_id: Document ID (e.g., "memory_bank_abc123")

        Returns:
            Document dict with id, content, metadata, or None if not found
        """
        # Determine collection from doc_id prefix
        for coll_name, adapter in self.collections.items():
            if doc_id.startswith(coll_name + "_"):
                doc = adapter.get_fragment(doc_id)
                if doc:
                    result = {
                        "id": doc_id,
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "collection": coll_name
                    }
                    return normalize_memory(result, coll_name)
        return None

    # ==================== Outcome Recording ====================

    async def record_outcome(
        self,
        doc_ids: List[str] = None,
        outcome: Literal["worked", "failed", "partial", "unknown"] = "worked",
        failure_reason: Optional[str] = None,
        # Desktop-compatible parameters
        doc_id: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Record outcome for searched documents.

        This updates scores for memories that were used in a response.
        Called by record_response MCP tool.

        Args:
            doc_ids: Documents that were used (Core style)
            outcome: Whether the response worked
            failure_reason: Reason if failed
            doc_id: Single document ID (Desktop-compatible)
            context: Additional context (Desktop-compatible, unused)

        Returns:
            Summary of updates
        """
        if not self.initialized:
            await self.initialize()

        # Desktop compatibility: handle single doc_id
        if doc_id is not None and doc_ids is None:
            doc_ids = [doc_id]
        elif doc_ids is None:
            doc_ids = []

        updates = []
        for single_doc_id in doc_ids:
            result = await self._outcome_service.record_outcome(
                doc_id=single_doc_id,
                outcome=outcome,
                failure_reason=failure_reason
            )
            if result:
                updates.append({
                    "doc_id": single_doc_id,
                    "new_score": result.get("score"),
                    "outcome": outcome
                })

        logger.info(f"Recorded outcome '{outcome}' for {len(updates)} documents")
        return {
            "outcome": outcome,
            "documents_updated": len(updates),
            "updates": updates
        }

    # ==================== Context Analysis (for Hooks) ====================

    async def get_context_for_injection(
        self,
        query: str,
        conversation_id: str = None,
        recent_conversation: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get context to inject into LLM prompt via hooks.

        v0.4.5: TagCascade retrieval — tags-first cascade fills candidate pool,
        CE reranks, top results injected into LLM context.

        Always_inject memory_bank facts (identity, preferences) are fetched
        separately and included outside the 4 scored slots.

        Args:
            query: The user's message
            conversation_id: Current conversation ID
            recent_conversation: Recent messages for continuity

        Returns:
            Dict with memories, doc_ids for scoring, and formatted injection
        """
        if not self.initialized:
            await self.initialize()

        result = {
            "memories": [],
            "user_facts": [],
            "formatted_injection": "",
            "doc_ids": []
        }

        # 0. Fetch always_inject memories (core identity - always included)
        always_inject_memories = self._memory_bank_service.get_always_inject()
        if always_inject_memories:
            result["user_facts"] = always_inject_memories
            # Add their doc_ids for scoring
            for mem in always_inject_memories:
                if mem.get("id"):
                    result["doc_ids"].append(mem["id"])

        # v0.4.5: Two-lane retrieval matching benchmark (4 summaries + 4 facts = 8)
        all_collections = ["working", "patterns", "history", "memory_bank"]

        # Lane 1: summaries/context (4 slots)
        # memory_bank items included (no memory_type field, $ne "fact" includes them)
        summary_results = await self.search(
            query=query, limit=4, collections=all_collections,
            metadata_filters={"memory_type": {"$ne": "fact"}}
        )

        # Lane 2: facts (4 slots)
        # memory_bank excluded naturally (no items have memory_type: "fact")
        fact_results = await self.search(
            query=query, limit=4, collections=all_collections,
            metadata_filters={"memory_type": "fact"}
        )

        # Merge: 4 summaries + 4 facts
        all_results = summary_results + fact_results

        # Filter empty — keep all 8 (4 summaries + 4 facts)
        top_memories = [
            m for m in all_results
            if m.get("content") or m.get("text")
        ]

        result["memories"] = top_memories
        result["relevant_memories"] = top_memories  # Alias for selective scoring in hooks
        result["doc_ids"] = [m.get("id") for m in top_memories if m.get("id")]
        result["formatted_injection"] = self._format_context_injection(result)

        return result


    def _format_context_injection(self, context: Dict[str, Any]) -> str:
        """
        Format context for injection into LLM prompt.

        v0.4.5: Shows all memories (4 summaries + 4 facts from two-lane retrieval).
        Extracts user name from identity-tagged memories.
        """
        import re
        parts = []
        user_name = None

        # Find identity-tagged facts in memory_bank (no always_inject required)
        all_facts = self._memory_bank_service.list_all(include_archived=False)
        for fact in all_facts:
            # Check for identity tag
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except (json.JSONDecodeError, ValueError, TypeError):
                    tags = []
            else:
                tags = tags_raw or []

            if "identity" not in tags:
                continue

            content = fact.get("content") or fact.get("text") or fact.get("metadata", {}).get("text", "")
            content_lower = content.lower()

            # Look for name patterns
            if "name is" in content_lower or "i'm " in content_lower or "i am " in content_lower:
                # Try "name is X" pattern
                match = re.search(r"name is (\w+)", content, re.IGNORECASE)
                if match:
                    user_name = match.group(1)
                    break
                # Try "I'm X" or "I am X" pattern
                match = re.search(r"i[''`]?m (\w+)|i am (\w+)", content, re.IGNORECASE)
                if match:
                    user_name = match.group(1) or match.group(2)
                    break

        memories = context.get("memories", [])

        if user_name or memories:
            parts.append("You have persistent memory about this user via Roampal. Memory tags: wilson:N% = reliability from past scoring, used:Nx = times retrieved, last:worked/failed/partial/unknown = whether this memory was *helpful* last time (not whether a task succeeded). [id:...] tags can be looked up with search_memory(id=...). Memories may be outdated or wrong. Verify before treating as ground truth. The context below was retrieved from past conversations. If the user references past interactions or asks if you remember them, use this context — you DO remember.")
            parts.append("")
            parts.append("═══ KNOWN CONTEXT ═══")

            # Add user name simply if found
            if user_name:
                parts.append(f"User: {user_name}")

            # v0.4.5: Separate summaries and facts for clarity
            summaries = []
            facts = []
            for mem in memories:
                normalized = normalize_memory(dict(mem), mem.get("collection", "unknown"))
                mem_type = mem.get("metadata", {}).get("memory_type", "")
                if mem_type == "fact":
                    facts.append(normalized)
                else:
                    summaries.append(normalized)

            def _format_mem(normalized):
                content = normalized.get("content", "")
                collection = normalized.get("collection", "unknown")
                doc_id = normalized.get("id", "")
                age = normalized.get("age", "")
                uses = normalized.get("uses", 0)
                wilson = normalized.get("wilson_score", 0)
                last_outcome = normalized.get("last_outcome", "")

                tag_parts = []
                if age:
                    tag_parts.append(age)
                tag_parts.append(collection)
                if collection == "books":
                    tag_parts.append("reference")
                elif uses > 0:
                    tag_parts.append(f"wilson:{wilson:.0%}")
                    tag_parts.append(f"used:{uses}x")
                    if last_outcome:
                        tag_parts.append(f"last:{last_outcome}")

                id_str = f" [id:{doc_id}]" if doc_id else ""
                return f"• {content}{id_str} ({', '.join(tag_parts)})"

            if summaries:
                for s in summaries:
                    parts.append(_format_mem(s))

            if facts:
                parts.append("")
                parts.append("Facts (auto-extracted from conversation — use for direction, not authority. Verify before citing as true):")
                for f in facts:
                    parts.append(_format_mem(f))

            parts.append("═══ END CONTEXT ═══")
            parts.append("")
            return "\n".join(parts)

        return ""

    # ==================== Book/Document Operations ====================

    def _chunk_by_sentences(
        self,
        content: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        v0.2.0: Chunk content by sentence boundaries.

        Avoids cutting mid-sentence which degrades semantic quality.
        Falls back to character-based chunking for edge cases.

        Args:
            content: Full text to chunk
            chunk_size: Target size per chunk (chars)
            chunk_overlap: Overlap between chunks (chars)

        Returns:
            List of text chunks
        """
        import re

        # Guard: Small content gets single chunk
        if len(content) <= chunk_size:
            return [content]

        # Split by sentence boundaries (. ! ? followed by space/newline)
        # Keep the punctuation with the sentence
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, content)

        # Guard: If no sentences found (no punctuation), fall back to character chunking
        if len(sentences) <= 1:
            return self._chunk_by_chars(content, chunk_size, chunk_overlap)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_length + sentence_len > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))

                # Calculate overlap: keep sentences from end that fit in overlap
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s) + 1  # +1 for space
                    else:
                        break

                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks if chunks else [content]

    def _chunk_by_chars(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Fallback: Character-based chunking for content without sentence boundaries.
        """
        effective_overlap = min(chunk_overlap, chunk_size - 1)
        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunks.append(content[start:end])
            start = end - effective_overlap

        return chunks

    async def _check_book_exists(self, title: str) -> Optional[List[str]]:
        """
        v0.2.0: Check if a book with this title already exists.

        Args:
            title: Book title to check

        Returns:
            List of existing doc_ids if found, None otherwise
        """
        books_collection = self.collections.get("books")
        if not books_collection:
            return None

        await books_collection._ensure_initialized()
        if books_collection.collection is None:
            return None

        try:
            # Query for chunks with this title
            results = books_collection.collection.get(
                where={"title": title},
                include=["metadatas"]
            )

            if results and results.get("ids"):
                return results["ids"]
        except Exception as e:
            logger.warning(f"Error checking for existing book: {e}")

        return None

    async def store_book(
        self,
        content: str,
        title: str = None,
        source: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        Store a document in the books collection.

        Documents are chunked for better retrieval.

        Args:
            content: Full document text
            title: Document title
            source: Source file path or URL
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunk document IDs
        """
        if not self.initialized:
            await self.initialize()

        # v0.2.0: File size limit check
        content_size = len(content.encode('utf-8'))
        if content_size > MAX_BOOK_SIZE:
            size_mb = content_size / (1024 * 1024)
            raise ValueError(f"Content too large ({size_mb:.1f}MB > 10MB limit). Split into smaller files.")

        # v0.2.0: Duplicate detection - check if book with same title exists
        if title:
            existing = await self._check_book_exists(title)
            if existing:
                logger.info(f"Book '{title}' already exists ({len(existing)} chunks), skipping")
                return existing

        # v0.2.0: Sentence-based chunking (preserves semantic boundaries)
        chunks = self._chunk_by_sentences(content, chunk_size, chunk_overlap)

        base_id = f"books_{uuid.uuid4().hex[:8]}"

        # v0.2.0: Batch embed all chunks at once (much faster for large books)
        embeddings = await self._embedding_service.embed_texts(chunks)

        # Build all document IDs and metadata
        doc_ids = []
        metadatas = []
        created_at = datetime.now().isoformat()

        for i, chunk in enumerate(chunks):
            doc_id = f"{base_id}_chunk_{i}"
            doc_ids.append(doc_id)

            meta = {
                "content": chunk,
                "text": chunk,
                "title": title or "Untitled",
                "source": source or "unknown",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": created_at
            }

            metadatas.append(meta)

        # v0.2.0: Batch upsert with rollback on failure
        try:
            await self.collections["books"].upsert_vectors(
                ids=doc_ids,
                vectors=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            # Rollback: delete any chunks that may have been inserted
            logger.error(f"Failed to store book '{title}': {e}, rolling back...")
            try:
                books = self.collections.get("books")
                if books and books.collection:
                    books.collection.delete(ids=doc_ids)
            except Exception as rollback_err:
                logger.warning(f"Rollback failed: {rollback_err}")
            raise

        logger.info(f"Stored book '{title}' in {len(chunks)} chunks (batch embedded)")
        return doc_ids

    async def remove_book(self, title: str) -> Dict[str, Any]:
        """
        Remove a book by title using ghost registry.

        v0.2.2: Non-destructive approach - adds chunk IDs to ghost registry
        instead of deleting from ChromaDB. This avoids HNSW index corruption.
        Ghost IDs are filtered from search results, making them invisible.

        Args:
            title: The book title to remove

        Returns:
            Dict with removal stats
        """
        if not self.initialized:
            await self.initialize()

        books_collection = self.collections.get("books")
        if not books_collection:
            return {"removed": 0, "error": "Books collection not found"}

        # Ensure lazy initialization completes
        await books_collection._ensure_initialized()
        if books_collection.collection is None:
            return {"removed": 0, "error": "Books collection not initialized"}

        # Find all chunks with this title
        try:
            results = books_collection.collection.get(
                where={"title": title},
                include=["metadatas"]
            )
        except Exception as e:
            return {"removed": 0, "error": str(e)}

        doc_ids = results.get("ids", [])
        if not doc_ids:
            return {"removed": 0, "message": f"No book found with title '{title}'"}

        # v0.2.2: Add to ghost registry instead of deleting from ChromaDB
        # This avoids HNSW index corruption while making chunks invisible
        self.ghost_ids.update(doc_ids)
        self._save_ghost_registry()
        logger.info(f"Ghosted {len(doc_ids)} chunks for book '{title}' (non-destructive)")

        # v0.4.5: Action KG cleanup removed (KG deleted)

        return {
            "removed": len(doc_ids),
            "title": title,
            "method": "ghost_registry"
        }

    async def list_books(self) -> List[Dict[str, Any]]:
        """
        List all books with metadata.

        Returns:
            List of book info dicts grouped by title
        """
        if not self.initialized:
            await self.initialize()

        books_collection = self.collections.get("books")
        if not books_collection:
            return []

        # Ensure lazy initialization completes
        await books_collection._ensure_initialized()
        if books_collection.collection is None:
            return []

        try:
            results = books_collection.collection.get(include=["metadatas"])
        except Exception:
            return []

        # Group by title, filtering out ghosted chunks
        books_by_title = {}
        for i, doc_id in enumerate(results.get("ids", [])):
            # v0.2.2: Skip ghosted chunks
            if doc_id in self.ghost_ids:
                continue

            metadata = results.get("metadatas", [])[i] if i < len(results.get("metadatas", [])) else {}
            title = metadata.get("title", "Untitled")

            if title not in books_by_title:
                books_by_title[title] = {
                    "title": title,
                    "source": metadata.get("source", "unknown"),
                    "created_at": metadata.get("created_at", ""),
                    "chunk_count": 0
                }
            books_by_title[title]["chunk_count"] += 1

        return list(books_by_title.values())

    # v0.4.5: _save_kg removed — KG replaced by tag-routed retrieval

    # ==================== Working Memory Operations ====================

    async def store_working(
        self,
        content: str,
        conversation_id: str = None,
        metadata: Dict[str, Any] = None,
        initial_score: float = 0.5,
        noun_tags: Optional[List[str]] = None
    ) -> str:
        """
        Store content in working memory.

        Used for session context that may be promoted to patterns.

        Args:
            content: Content to store
            conversation_id: Session identifier
            metadata: Additional metadata
            initial_score: Starting score (default 0.5, can be boosted/demoted based on outcome)
            noun_tags: Content nouns for TagCascade retrieval (from LLM)

        Returns:
            Document ID
        """
        if not self.initialized:
            await self.initialize()

        doc_id = f"working_{uuid.uuid4().hex[:8]}"
        embedding = await self._embedding_service.embed_text(content)

        meta = {
            "content": content,
            "text": content,
            "score": initial_score,
            "uses": 0,
            "created_at": datetime.now().isoformat(),
            "conversation_id": conversation_id or "unknown"
        }
        if metadata:
            meta.update(metadata)

        # v0.4.5: noun_tags for TagCascade retrieval
        # Use provided tags (from LLM via MCP tool), fall back to regex for migration
        if noun_tags:
            meta["noun_tags"] = json.dumps(noun_tags)
            if hasattr(self, '_tag_service') and self._tag_service:
                self._tag_service.add_known_tags(noun_tags)
        elif hasattr(self, '_tag_service') and self._tag_service:
            try:
                tags = self._tag_service.extract_tags(content)
                if tags:
                    meta["noun_tags"] = json.dumps(tags)
            except Exception as e:
                logger.warning(f"Tag extraction failed for working memory: {e}")

        await self.collections["working"].upsert_vectors(
            ids=[doc_id],
            vectors=[embedding],
            metadatas=[meta]
        )

        return doc_id

    # ==================== Query Routing (Desktop-compatible) ====================

    def _route_query(self, query: str) -> List[str]:
        """Route query to appropriate collections (delegates to routing service)."""
        if self._routing_service:
            return self._routing_service.route_query(query)
        return ["working", "patterns", "history", "books", "memory_bank"]

    # ==================== Stats and Diagnostics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        stats = {
            "initialized": self.initialized,
            "data_path": str(self.data_path),
            "collections": {}
        }

        for name, adapter in self.collections.items():
            try:
                count = adapter.collection.count() if adapter.collection else 0
                stats["collections"][name] = {"count": count}
            except Exception as e:
                stats["collections"][name] = {"error": str(e)}

        return stats

    # ==================== Knowledge Graph Methods ====================

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract concepts from text using basic extraction.

        Returns: List of lowercase concept keywords
        """
        if not text:
            return []

        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                      'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'this', 'that',
                      'with', 'they', 'from', 'what', 'when', 'where', 'which', 'how', 'why',
                      'just', 'will', 'would', 'could', 'should', 'there', 'their', 'about'}

        concepts = [w for w in words if w not in stop_words]
        return concepts[:10]

    # v0.4.5: KG methods removed (get_tier_recommendations, get_action_effectiveness,
    # get_facts_for_entities, _get_doc_effectiveness, detect_context_type,
    # analyze_conversation_context, record_action_outcome, _update_kg_routing).
    # Tag-routed retrieval in SearchService handles all routing now.

    def get_tier_recommendations(self, concepts: List[str]) -> Dict[str, Any]:
        """v0.4.5: Stub for backward compat. Returns all collections."""
        ALL_COLLECTIONS = ["working", "patterns", "history", "books", "memory_bank"]
        return {"top_collections": ALL_COLLECTIONS.copy(), "match_count": 0, "confidence_level": "exploration"}

    # v0.4.5: All KG methods below this line removed. Only get_tier_recommendations
    # stub kept above for backward compatibility. Removed methods:
    # get_action_effectiveness, get_facts_for_entities, _get_doc_effectiveness,
    # detect_context_type, analyze_conversation_context, record_action_outcome, _update_kg_routing
    #
    # End of UnifiedMemorySystem class.
    # ==================== LEGACY KG CODE REMOVED v0.4.5 ====================
