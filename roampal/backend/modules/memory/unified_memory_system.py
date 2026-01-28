"""
UnifiedMemorySystem - Facade coordinating memory services

Simplified from Roampal ui-implementation for roampal-core.
Stripped: Ollama, KG service, complex routing.
Kept: Core search, outcome tracking, memory bank operations.
"""

import logging
import json
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
from .knowledge_graph_service import KnowledgeGraphService

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
    action_type: str  # Tool name: "search_memory", "create_memory", "score_response", etc.
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
                elif os.uname().sysname == 'Darwin':  # macOS
                    data_path = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', app_folder, 'data')
                else:  # Linux
                    data_path = os.path.join(os.path.expanduser('~'), '.local', 'share', app_folder.lower(), 'data')
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Services (lazy initialized)
        self._embedding_service: Optional[EmbeddingService] = None
        self._scoring_service: Optional[ScoringService] = None
        self._promotion_service: Optional[PromotionService] = None
        self._outcome_service: Optional[OutcomeService] = None
        self._memory_bank_service: Optional[MemoryBankService] = None
        self._context_service: Optional[ContextService] = None

        # Collections
        self.collections: Dict[str, ChromaDBAdapter] = {}
        self.initialized = False

        # Knowledge Graph - shared with Desktop
        self.kg_path = self.data_path / "knowledge_graph.json"
        self.knowledge_graph = self._load_kg()

        # Ghost Registry - tracks deleted book IDs without modifying ChromaDB
        # v0.2.2: Non-destructive delete to avoid HNSW index corruption
        self.ghost_registry_path = self.data_path / "ghost_ids.json"
        self.ghost_ids: set = self._load_ghost_registry()

    def _load_kg(self) -> Dict[str, Any]:
        """Load knowledge graph from disk, or return default structure."""
        if self.kg_path.exists():
            try:
                with open(self.kg_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load KG from {self.kg_path}: {e}")

        # Default empty KG structure
        return {
            "routing_patterns": {},
            "success_rates": {},
            "failure_patterns": {},
            "problem_categories": {},
            "problem_solutions": {},
            "solution_patterns": {},
            "context_action_effectiveness": {}
        }

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
        for short_name, chroma_name in collection_mapping.items():
            adapter = ChromaDBAdapter(
                collection_name=chroma_name,
                persist_directory=str(self.data_path / "chromadb")
            )
            await adapter.initialize()
            self.collections[short_name] = adapter
            logger.info(f"Initialized collection: {short_name} -> {chroma_name}")

        # Initialize services with dependencies
        self._scoring_service = ScoringService(config=self.config)

        # KG service (Desktop parity)
        self._kg_service = KnowledgeGraphService(
            kg_path=self.data_path / "knowledge_graph.json",
            content_graph_path=self.data_path / "content_graph.json",
            relationships_path=self.data_path / "memory_relationships.json",
            config=self.config
        )

        # Routing service (Desktop parity)
        self._routing_service = RoutingService(
            kg_service=self._kg_service,
            config=self.config
        )

        # Initialize PromotionService for auto-deletion/promotion when scores drop
        self._promotion_service = PromotionService(
            collections=self.collections,
            embed_fn=self._embedding_service.embed_text,
            config=self.config
        )

        # OutcomeService wired to PromotionService and KG - garbage now gets deleted
        self._outcome_service = OutcomeService(
            collections=self.collections,
            kg_service=self._kg_service,
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

        # Also clean old working memories (> 24 hours)
        await self._promotion_service.cleanup_old_working_memory(max_age_hours=24.0)

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

        if collections is None:
            collections = list(self.collections.keys())

        # Get query embedding
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
                    # v0.2.2: Filter out ghost IDs (deleted books)
                    doc_id = r.get("id", "")
                    if doc_id in self.ghost_ids:
                        continue

                    # Add collection info
                    r["collection"] = coll_name

                    # Get metadata and calculate base similarity
                    metadata = r.get("metadata", {})
                    distance = r.get("distance", 1.0)
                    embedding_similarity = 1.0 / (1.0 + distance)

                    # Calculate quality score based on collection type
                    if coll_name == "memory_bank":
                        # Memory bank: use importance × confidence
                        importance = float(metadata.get("importance", 0.7))
                        confidence = float(metadata.get("confidence", 0.7))
                        quality = importance * confidence
                    else:
                        # Other collections: use learned score from outcome history
                        quality = float(metadata.get("score", 0.5))

                    # Apply quality-weighted scoring (Desktop-compatible formula)
                    # Distance boost: adjust distance by quality
                    quality_boost = 1.0 - (quality * 0.8)  # High quality = lower effective distance
                    adjusted_distance = distance * quality_boost
                    adjusted_similarity = 1.0 / (1.0 + adjusted_distance)

                    # Final score: blend embedding similarity with quality
                    # This ensures high-quality results rank above low-quality semantic matches
                    final_rank_score = adjusted_similarity * (1.0 + quality)

                    # Store Desktop-compatible field names
                    r["embedding_similarity"] = embedding_similarity
                    r["final_rank_score"] = final_rank_score
                    r["quality"] = quality

                    # Keep legacy fields for backwards compatibility
                    r["combined_score"] = final_rank_score
                    r["similarity"] = embedding_similarity

                    all_results.append(r)

            except Exception as e:
                logger.warning(f"Error searching {coll_name}: {e}")

        # Sort based on sort_by parameter
        if sort_by == "recency":
            # Sort by timestamp (most recent first)
            def get_timestamp(x):
                metadata = x.get("metadata", {})
                ts = metadata.get("timestamp") or metadata.get("created_at") or ""
                return ts if ts else "0"
            all_results.sort(key=get_timestamp, reverse=True)
            # Recency sort: return exactly limit items
            return all_results[:limit]
        elif sort_by == "score":
            # Sort by outcome score
            all_results.sort(key=lambda x: x.get("metadata", {}).get("score", 0.5), reverse=True)
            return all_results[:limit]
        else:
            # Default: sort by final_rank_score (quality-weighted relevance)
            all_results.sort(key=lambda x: x.get("final_rank_score", 0), reverse=True)
            # Semantic search: return limit * 2 for better cross-collection coverage
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
            return doc_id
        else:
            # Unknown collection - default to working
            return await self.store_working(content=text, metadata=metadata)

    # ==================== Memory Bank Operations ====================

    async def store_memory_bank(
        self,
        text: str,
        tags: List[str] = None,
        importance: float = 0.7,
        confidence: float = 0.7,
        always_inject: bool = False
    ) -> str:
        """
        Store a fact in memory_bank.

        Args:
            text: The fact to remember
            tags: Categories (identity, preference, goal, project)
            importance: How critical (0.0-1.0)
            confidence: How certain (0.0-1.0)
            always_inject: If True, appears in every context

        Returns:
            Document ID
        """
        if not self.initialized:
            await self.initialize()

        return await self._memory_bank_service.store(
            text=text,
            tags=tags or [],
            importance=importance,
            confidence=confidence,
            always_inject=always_inject
        )

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
            if doc_id.startswith(coll_name):
                doc = adapter.get_fragment(doc_id)
                if doc:
                    return {
                        "id": doc_id,
                        "content": doc.get("content", ""),
                        "text": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "collection": coll_name
                    }
        return None

    # ==================== Outcome Recording ====================

    async def record_outcome(
        self,
        doc_ids: List[str] = None,
        outcome: Literal["worked", "failed", "partial"] = "worked",
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

        Uses KG-routed unified search: searches ALL collections, ranks by Wilson score,
        returns top 5 most relevant/proven memories regardless of collection.

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

        # 1. Extract concepts for KG routing insight
        concepts = self._extract_concepts(query)

        # 2. Get KG recommendations (informational - we still search all)
        kg_recs = self.get_tier_recommendations(concepts)

        # 3. Reserved slot: Always include 1 working memory for session context
        working_results = await self.search(
            query=query,
            limit=1,
            collections=["working"]
        )
        working_scored = self._scoring_service.apply_scoring_to_results(working_results)
        reserved_working = [m for m in working_scored if m.get("content") or m.get("text")][:1]
        reserved_ids = {m.get("id") for m in reserved_working if m.get("id")}

        # 4. Search remaining collections for other slots
        other_collections = ["patterns", "history", "memory_bank"]
        other_results = await self.search(
            query=query,
            limit=4,
            collections=other_collections
        )

        # 5. Apply Wilson scoring for proper ranking
        scored_results = self._scoring_service.apply_scoring_to_results(other_results)

        # 6. Filter empty memories, exclude any duplicates from reserved
        valid_results = [
            m for m in scored_results
            if (m.get("content") or m.get("text")) and m.get("id") not in reserved_ids
        ]

        # 7. Combine: 1 reserved working + top 2 from other collections
        top_memories = reserved_working + valid_results[:2]

        # 8. Enrich with Action KG effectiveness stats
        for mem in top_memories:
            coll = mem.get("collection", "unknown")
            eff = self.get_action_effectiveness("general", "search", coll)
            if eff:
                mem["effectiveness"] = eff.get("success_rate", 0)

        result["memories"] = top_memories
        result["relevant_memories"] = top_memories  # Alias for selective scoring in hooks
        result["doc_ids"] = [m.get("id") for m in top_memories if m.get("id")]
        result["formatted_injection"] = self._format_context_injection(result)

        return result


    def _format_context_injection(self, context: Dict[str, Any]) -> str:
        """
        Format context for injection into LLM prompt.

        Shows top 3 memories across all collections with effectiveness stats.
        Extracts user name from identity-tagged memories.
        """
        import re
        import json as json_module
        parts = []
        user_name = None

        # Find identity-tagged facts in memory_bank (no always_inject required)
        all_facts = self._memory_bank_service.list_all(include_archived=False)
        for fact in all_facts:
            # Check for identity tag
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags = json_module.loads(tags_raw) if tags_raw else []
                except:
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
            parts.append("═══ KNOWN CONTEXT ═══")

            # Add user name simply if found
            if user_name:
                parts.append(f"User: {user_name}")

            # Add memories
            for mem in memories[:3]:

                # Get content from various possible locations
                content = mem.get("content") or mem.get("text") or mem.get("metadata", {}).get("text", "")
                collection = mem.get("collection", "unknown")

                # Get Wilson score and effectiveness
                wilson = mem.get("wilson_score", 0)
                effectiveness = mem.get("effectiveness", 0)

                # Format with collection and score info
                if wilson >= 0.7:
                    parts.append(f"• {content} ({int(wilson*100)}% proven, {collection})")
                elif effectiveness > 0:
                    parts.append(f"• {content} ({int(effectiveness*100)}% effective, {collection})")
                else:
                    parts.append(f"• {content} ({collection})")

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

        base_id = f"book_{uuid.uuid4().hex[:8]}"

        # v0.2.0: Batch embed all chunks at once (much faster for large books)
        embeddings = await self._embedding_service.embed_texts(chunks)

        # Build all document IDs and metadata
        doc_ids = []
        metadatas = []
        created_at = datetime.now().isoformat()

        for i, chunk in enumerate(chunks):
            doc_id = f"{base_id}_chunk_{i}"
            doc_ids.append(doc_id)

            metadatas.append({
                "content": chunk,
                "text": chunk,
                "title": title or "Untitled",
                "source": source or "unknown",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": created_at
            })

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

        # Clean Action KG references
        cleaned_refs = await self.cleanup_action_kg_for_doc_ids(doc_ids)

        return {
            "removed": len(doc_ids),
            "title": title,
            "cleaned_kg_refs": cleaned_refs,
            "method": "ghost_registry"
        }

    async def cleanup_action_kg_for_doc_ids(self, doc_ids: List[str]) -> int:
        """
        Remove Action KG examples that reference specific doc_ids.

        Called when books are deleted to maintain KG integrity.

        Args:
            doc_ids: List of document IDs being deleted

        Returns:
            Number of examples removed
        """
        if not doc_ids:
            return 0

        try:
            doc_id_set = set(doc_ids)
            cleaned = 0

            for key, stats in self.knowledge_graph.get("context_action_effectiveness", {}).items():
                examples = stats.get("examples", [])
                original_count = len(examples)
                stats["examples"] = [
                    ex for ex in examples
                    if ex.get("doc_id") not in doc_id_set
                ]
                cleaned += original_count - len(stats["examples"])

            if cleaned > 0:
                logger.info(f"Action KG cleanup: removed {cleaned} examples for deleted doc_ids")
                self._save_kg()

            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning Action KG for doc_ids: {e}")
            return 0

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

    def _save_kg(self):
        """Save knowledge graph to disk and sync to _kg_service.

        v0.2.0 FIX: After saving, reload _kg_service so SearchService sees updates.
        This fixes the "Reading Your Own Writes" bug where UMS writes to self.knowledge_graph
        but SearchService reads from _kg_service.knowledge_graph (which was stale).
        """
        try:
            with open(self.kg_path, 'w') as f:
                json.dump(self.knowledge_graph, f, indent=2)

            # v0.2.0: Sync to _kg_service so SearchService sees the update
            if hasattr(self, '_kg_service') and self._kg_service is not None:
                self._kg_service.reload_kg()

        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")

    # ==================== Working Memory Operations ====================

    async def store_working(
        self,
        content: str,
        conversation_id: str = None,
        metadata: Dict[str, Any] = None,
        initial_score: float = 0.5
    ) -> str:
        """
        Store content in working memory.

        Used for session context that may be promoted to patterns.

        Args:
            content: Content to store
            conversation_id: Session identifier
            metadata: Additional metadata
            initial_score: Starting score (default 0.5, can be boosted/demoted based on outcome)

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
            "score": initial_score,  # Can be adjusted based on outcome at creation time
            "uses": 0,
            "created_at": datetime.now().isoformat(),
            "conversation_id": conversation_id or "unknown"
        }
        if metadata:
            meta.update(metadata)

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

    def get_tier_recommendations(self, concepts: List[str]) -> Dict[str, Any]:
        """Query Routing KG for best collections given concepts."""
        ALL_COLLECTIONS = ["working", "patterns", "history", "books", "memory_bank"]

        if not concepts:
            return {"top_collections": ALL_COLLECTIONS.copy(), "match_count": 0, "confidence_level": "exploration"}

        collection_scores = {c: 0.0 for c in ALL_COLLECTIONS}
        match_count = 0

        routing_patterns = self.knowledge_graph.get("routing_patterns", {})
        problem_categories = self.knowledge_graph.get("problem_categories", {})

        for concept in concepts:
            if concept in routing_patterns:
                pattern_data = routing_patterns[concept]
                # Handle both old format (string) and new format (dict with best_collection)
                if isinstance(pattern_data, dict):
                    best_coll = pattern_data.get("best_collection", "")
                else:
                    best_coll = pattern_data
                if best_coll in collection_scores:
                    collection_scores[best_coll] += 1.0
                    match_count += 1

            if concept in problem_categories:
                preferred = problem_categories[concept]
                if isinstance(preferred, list):
                    for coll in preferred:
                        if coll in collection_scores:
                            collection_scores[coll] += 0.5
                            match_count += 1

        sorted_collections = sorted(collection_scores.items(), key=lambda x: x[1], reverse=True)

        if match_count >= 3:
            confidence = "high"
        elif match_count >= 1:
            confidence = "medium"
        else:
            confidence = "exploration"

        return {"top_collections": [c[0] for c in sorted_collections], "match_count": match_count, "confidence_level": confidence}

    def get_action_effectiveness(self, context_type: str, action_type: str, collection: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get effectiveness stats for an action in a specific context."""
        key = f"{context_type}|{action_type}|{collection or '*'}"
        return self.knowledge_graph.get("context_action_effectiveness", {}).get(key)

    async def get_facts_for_entities(self, entities: List[str], limit: int = 2) -> List[Dict[str, Any]]:
        """Query Content KG to retrieve matching memory_bank facts."""
        if not self.initialized:
            await self.initialize()

        facts = []
        seen_ids = set()

        for entity in entities:
            if len(facts) >= limit:
                break

            try:
                results = await self.search(query=entity, collections=["memory_bank"], limit=2)

                for result in results:
                    if len(facts) >= limit:
                        break

                    doc_id = result.get("id")
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        effectiveness = self._get_doc_effectiveness(doc_id)
                        facts.append({
                            "id": doc_id,
                            "content": result.get("content", result.get("text", "")),
                            "entity": entity,
                            "effectiveness": effectiveness
                        })
            except Exception as e:
                logger.warning(f"Error getting facts for entity '{entity}': {e}")

        return facts

    def _get_doc_effectiveness(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get doc effectiveness from Action KG examples."""
        successes = failures = 0

        for key, stats in self.knowledge_graph.get("context_action_effectiveness", {}).items():
            for ex in stats.get("examples", []):
                if ex.get("doc_id") == doc_id:
                    if ex.get("outcome") == "worked":
                        successes += 1
                    elif ex.get("outcome") == "failed":
                        failures += 1

        total = successes + failures
        return {"success_rate": successes / total, "total_uses": total, "successes": successes, "failures": failures} if total else None

    async def detect_context_type(self, system_prompts: List[str] = None, recent_messages: List[Dict[str, Any]] = None) -> str:
        """Detect context type from conversation (coding, fitness, general, etc.)."""
        all_text = ""
        if system_prompts:
            all_text += " ".join(system_prompts)
        if recent_messages:
            for msg in recent_messages:
                all_text += " " + msg.get("content", "")

        all_text = all_text.lower()

        context_keywords = {
            "coding": ["code", "function", "class", "error", "debug", "api", "database", "python", "javascript", "typescript", "react", "git", "build", "test"],
            "fitness": ["workout", "exercise", "gym", "weight", "muscle", "cardio", "diet", "protein", "calories"],
            "finance": ["money", "budget", "invest", "stock", "savings", "expense", "income", "tax"],
            "learning": ["learn", "study", "course", "book", "tutorial", "understand", "concept"],
            "writing": ["write", "essay", "article", "blog", "content", "draft", "edit"]
        }

        scores = {ctx: sum(1 for kw in kws if kw in all_text) for ctx, kws in context_keywords.items()}
        best_ctx = max(scores.items(), key=lambda x: x[1])
        return best_ctx[0] if best_ctx[1] > 0 else "general"

    async def analyze_conversation_context(self, current_message: str, recent_conversation: List[Dict[str, Any]], conversation_id: str) -> Dict[str, Any]:
        """Analyze conversation context for organic memory injection."""
        context = {"relevant_patterns": [], "past_outcomes": [], "topic_continuity": [], "proactive_insights": [], "matched_concepts": []}

        try:
            current_concepts = self._extract_concepts(current_message)
            context["matched_concepts"] = current_concepts

            if current_concepts:
                pattern_signature = "_".join(sorted(current_concepts[:3]))

                if pattern_signature in self.knowledge_graph.get("problem_categories", {}):
                    past_solutions = self.knowledge_graph["problem_categories"][pattern_signature]

                    for doc_id in past_solutions[:2]:
                        for coll_name in ["patterns", "history"]:
                            if coll_name in self.collections:
                                doc = self.collections[coll_name].get_fragment(doc_id)
                                if doc:
                                    metadata = doc.get("metadata", {})
                                    score = metadata.get("score", 0.5)
                                    uses = metadata.get("uses", 0)
                                    last_outcome = metadata.get("last_outcome", "unknown")

                                    if score >= 0.7 and last_outcome == "worked":
                                        context["relevant_patterns"].append({
                                            "text": doc.get("content", ""),
                                            "score": score,
                                            "uses": uses,
                                            "collection": coll_name,
                                            "insight": f"Based on {uses} past use(s), this approach had a {int(score*100)}% success rate"
                                        })
                                    break

            failure_patterns = self.knowledge_graph.get("failure_patterns", {})
            for failure_key, failures in failure_patterns.items():
                if any(concept in failure_key.lower() for concept in current_concepts):
                    for failure in failures[-2:]:
                        context["past_outcomes"].append({
                            "outcome": "failed",
                            "reason": failure_key,
                            "when": failure.get("timestamp", ""),
                            "insight": f"Note: Similar approach failed before due to: {failure_key}"
                        })

        except Exception as e:
            logger.warning(f"Error analyzing conversation context: {e}")

        return context

    # ==================== Action KG Tracking ====================

    async def record_action_outcome(self, action: ActionOutcome):
        """
        Record action-level outcome with context awareness (v0.2.1 Causal Learning).

        This enables learning: "In context X, action Y on collection Z leads to outcome W"
        Example: In "coding" context, search_memory on books → 90% success

        Works for ALL collections including memory_bank and books (at collection level,
        not individual doc scoring).

        Args:
            action: ActionOutcome with context type, action type, outcome, and causal attribution
        """
        # Build key for context-action-collection effectiveness tracking
        # Format: "{context_type}|{action_type}|{collection}"
        key = f"{action.context_type}|{action.action_type}|{action.collection or '*'}"

        # Initialize tracking structure if needed
        if key not in self.knowledge_graph["context_action_effectiveness"]:
            self.knowledge_graph["context_action_effectiveness"][key] = {
                "successes": 0,
                "failures": 0,
                "partials": 0,
                "success_rate": 0.0,
                "total_uses": 0,
                "first_seen": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
                "examples": []
            }

        stats = self.knowledge_graph["context_action_effectiveness"][key]

        # Update counts based on outcome
        if action.outcome == "worked":
            stats["successes"] += 1
        elif action.outcome == "failed":
            stats["failures"] += 1
        else:  # partial
            stats["partials"] += 1

        stats["total_uses"] += 1
        stats["last_used"] = datetime.now().isoformat()

        # Calculate success rate (successes / total, treating partials as 0.5)
        total = stats["successes"] + stats["failures"] + stats["partials"]
        if total > 0:
            weighted_successes = stats["successes"] + (stats["partials"] * 0.5)
            stats["success_rate"] = weighted_successes / total

        # Store example for debugging (keep last 5)
        example = {
            "timestamp": action.timestamp.isoformat(),
            "outcome": action.outcome,
            "doc_id": action.doc_id,
            "params": action.action_params,
            "chain_position": action.chain_position,
            "chain_length": action.chain_length,
            "caused_final": action.caused_final_outcome
        }
        if action.failure_reason:
            example["failure_reason"] = action.failure_reason

        stats["examples"] = (stats.get("examples", []) + [example])[-5:]

        # Log learning for transparency
        logger.info(
            f"[Causal Learning] {key}: {action.outcome} "
            f"(rate={stats['success_rate']:.2%}, uses={stats['total_uses']}, "
            f"chain={action.chain_position+1}/{action.chain_length})"
        )

        # Save KG to disk
        self._save_kg()

    async def _update_kg_routing(self, query: str, collection: str, outcome: str):
        """
        Update KG routing patterns based on outcome.

        Learns which collections work best for which query patterns.
        Works for ALL collections including memory_bank and books.

        Example: "Python tutorial" queries that work on books → routes future
        similar queries to books first.
        """
        if not query:
            return

        concepts = self._extract_concepts(query)

        for concept in concepts:
            if concept not in self.knowledge_graph["routing_patterns"]:
                self.knowledge_graph["routing_patterns"][concept] = {
                    "collections_used": {},
                    "best_collection": collection,
                    "success_rate": 0.5
                }

            pattern = self.knowledge_graph["routing_patterns"][concept]

            # Track collection performance
            if collection not in pattern["collections_used"]:
                pattern["collections_used"][collection] = {
                    "successes": 0,
                    "failures": 0,
                    "total": 0
                }

            stats = pattern["collections_used"][collection]
            stats["total"] += 1

            if outcome == "worked":
                stats["successes"] += 1
            elif outcome == "failed":
                stats["failures"] += 1

            # Update best collection based on success rates
            best_collection = collection
            best_rate = 0.0

            for coll_name, coll_stats in pattern["collections_used"].items():
                total_with_feedback = coll_stats["successes"] + coll_stats["failures"]
                if total_with_feedback > 0:
                    rate = coll_stats["successes"] / total_with_feedback
                else:
                    rate = 0.5  # Neutral baseline

                if rate > best_rate:
                    best_rate = rate
                    best_collection = coll_name

            pattern["best_collection"] = best_collection
            pattern["success_rate"] = best_rate if best_rate > 0 else 0.5

        # Save KG
        self._save_kg()
        logger.info(f"[Routing KG] Updated patterns for '{query[:50]}' → {collection} ({outcome})")
