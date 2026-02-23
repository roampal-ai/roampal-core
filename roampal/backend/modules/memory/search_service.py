"""
Search Service - Unified search with hybrid ranking and cross-encoder reranking.

Extracted from UnifiedMemorySystem as part of refactoring.

Responsibilities:
- Main search with hybrid ranking (vector + BM25)
- Cross-encoder reranking (optional)
- Entity boost calculation
- Result scoring and ranking
- Document effectiveness tracking
"""

import json
import logging
import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from .config import MemoryConfig
from .scoring_service import ScoringService, wilson_score_lower
from .routing_service import RoutingService, ALL_COLLECTIONS
from .knowledge_graph_service import KnowledgeGraphService

logger = logging.getLogger(__name__)


# Type aliases
CollectionName = Literal["working", "patterns", "history", "books", "memory_bank"]


class SearchService:
    """
    Unified search with hybrid ranking.

    Features:
    - KG-based intelligent routing
    - Vector similarity search with BM25 fusion
    - Wilson score learning-aware ranking
    - Cross-encoder reranking (optional)
    - Entity boost from Content KG
    """

    def __init__(
        self,
        collections: Dict[str, Any],  # ChromaDBAdapter instances
        scoring_service: ScoringService,
        routing_service: RoutingService,
        kg_service: KnowledgeGraphService,
        embed_fn: Callable[[str], Any],  # Async function to embed text
        config: Optional[MemoryConfig] = None,
        reranker: Optional[Any] = None,  # CrossEncoder instance
    ):
        """
        Initialize SearchService.

        Args:
            collections: Dict mapping collection name to ChromaDBAdapter
            scoring_service: ScoringService for ranking
            routing_service: RoutingService for query routing
            kg_service: KnowledgeGraphService for KG operations
            embed_fn: Async function to generate embeddings
            config: Optional MemoryConfig
            reranker: Optional CrossEncoder for reranking
        """
        self.collections = collections
        self.scoring_service = scoring_service
        self.routing_service = routing_service
        self.kg_service = kg_service
        self.embed_fn = embed_fn
        self.config = config or MemoryConfig()
        self.reranker = reranker

        # Cache for doc_ids per session (for outcome scoring)
        self._cached_doc_ids: Dict[str, List[str]] = {}

    # =========================================================================
    # Date Filter Helpers (v0.2.0)
    # =========================================================================

    # Date fields that ChromaDB can't filter with $gte/$lte (stored as ISO strings)
    DATE_FIELDS = ('timestamp', 'last_used', 'created_at', 'first_seen', 'last_seen')

    def _extract_date_filters(
        self,
        filters: Optional[Dict[str, Any]]
    ) -> tuple[Optional[Dict], Optional[Dict]]:
        """
        Separate date filters (for post-query) from other filters (for ChromaDB).

        v0.2.0: ChromaDB $gte/$lte only work on numbers, not ISO strings.
        We extract date filters and apply them in Python after fetching results.

        Returns:
            (chromadb_filters, date_filters)
        """
        if not filters:
            return None, None

        chromadb_filters = {}
        date_filters = {}

        for key, value in filters.items():
            if key in self.DATE_FIELDS:
                date_filters[key] = value
            else:
                chromadb_filters[key] = value

        return chromadb_filters or None, date_filters or None

    def _apply_date_filters(
        self,
        results: List[Dict],
        date_filters: Dict[str, Any]
    ) -> List[Dict]:
        """
        Apply date filtering in Python (ISO strings sort alphabetically).

        Supports:
        - Exact date match: {"timestamp": "2025-12-20"} matches "2025-12-20T..."
        - Range operators: {"timestamp": {"$gte": "2025-12-15", "$lte": "2025-12-20"}}
        """
        filtered = results

        for field, condition in date_filters.items():
            if isinstance(condition, str):
                # Exact date match: "2025-12-18" matches "2025-12-18T..."
                if len(condition) == 10:  # YYYY-MM-DD format
                    filtered = [
                        r for r in filtered
                        if r.get('metadata', {}).get(field, '').startswith(condition)
                    ]
                else:
                    # Full timestamp match
                    filtered = [
                        r for r in filtered
                        if r.get('metadata', {}).get(field) == condition
                    ]

            elif isinstance(condition, dict):
                # Range operators
                for op, val in condition.items():
                    if op == '$gte':
                        filtered = [
                            r for r in filtered
                            if r.get('metadata', {}).get(field, '') >= val
                        ]
                    elif op == '$gt':
                        filtered = [
                            r for r in filtered
                            if r.get('metadata', {}).get(field, '') > val
                        ]
                    elif op == '$lte':
                        filtered = [
                            r for r in filtered
                            if r.get('metadata', {}).get(field, '') <= val
                        ]
                    elif op == '$lt':
                        filtered = [
                            r for r in filtered
                            if r.get('metadata', {}).get(field, '') < val
                        ]

        return filtered

    # =========================================================================
    # Main Search
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        collections: Optional[List[CollectionName]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
        transparency_context: Optional[Any] = None,
        sort_by: Optional[str] = None
    ) -> Union[List[Dict], Dict]:
        """
        Search memory with intelligent routing and optional metadata filtering.

        Args:
            query: Search query
            limit: Max results
            offset: Pagination offset
            collections: Override automatic routing
            metadata_filters: ChromaDB where filters
            return_metadata: Include pagination metadata
            transparency_context: Optional context for tracking
            sort_by: Sort order (recency, score, relevance). Default: relevance for queries, recency for empty.

        Returns:
            Ranked results (list or dict with pagination metadata)
        """
        # Use KG to route query if collections not specified
        if collections is None:
            collections = self.routing_service.route_query(query)

        # Check for known problem->solution patterns
        known_solutions = await self.kg_service.find_known_solutions(
            query,
            self._get_fragment
        )

        # Special handling for empty query - return all items
        if not query or query.strip() == "":
            return await self._search_all(
                collections, limit, offset, return_metadata,
                metadata_filters=metadata_filters,
                sort_by=sort_by
            )

        # Preprocess query for better retrieval
        processed_query = self.routing_service.preprocess_query(query)

        # Generate query embedding
        try:
            query_embedding = await self.embed_fn(processed_query)
        except Exception as e:
            logger.error(f"Embedding generation failed for query '{query}': {e}")
            if return_metadata:
                return {"results": [], "total": 0, "limit": limit, "offset": offset, "has_more": False}
            return []

        # Track search start if context provided
        if transparency_context and hasattr(transparency_context, 'track_action'):
            transparency_context.track_action(
                action_type="memory_search",
                description=f"Searching: {query[:50]}{'...' if len(query) > 50 else ''}",
                detail=f"Collections: {', '.join(collections)}",
                status="executing"
            )

        # v0.2.0: Separate date filters from ChromaDB filters
        # ChromaDB can't do $gte/$lte on ISO string timestamps
        chromadb_filters, date_filters = self._extract_date_filters(metadata_filters)

        # Fetch more results if we'll be post-filtering by date
        fetch_limit = limit * 3 if date_filters else limit

        # Search specified collections
        all_results = await self._search_collections(
            query_embedding, processed_query, collections, fetch_limit, chromadb_filters
        )

        # v0.2.0: Apply date filters in Python (ISO strings sort alphabetically)
        if date_filters:
            all_results = self._apply_date_filters(all_results, date_filters)

        # Add known solutions to the beginning (they're already boosted)
        if known_solutions:
            existing_ids = {r.get("id") for r in all_results}
            unique_known = [s for s in known_solutions if s.get("id") not in existing_ids]
            all_results = unique_known + all_results

        # Apply scoring and ranking
        all_results = self.scoring_service.apply_scoring_to_results(all_results)

        # Cross-encoder reranking
        if self.reranker and len(all_results) > limit * 2:
            all_results = await self._rerank_with_cross_encoder(query, all_results, limit)

        # Track usage for KG learning
        paginated_results = all_results[offset:offset + limit]
        self._track_search_results(query, paginated_results, transparency_context)

        # Return results
        if return_metadata:
            return {
                "results": paginated_results,
                "total": len(all_results),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < len(all_results)
            }
        return paginated_results

    async def _search_all(
        self,
        collections: List[str],
        limit: int,
        offset: int,
        return_metadata: bool,
        metadata_filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None
    ) -> Union[List[Dict], Dict]:
        """Handle empty query - return all items with full scoring pipeline."""
        from .unified_memory_system import normalize_memory

        all_results = []

        for coll_name in collections:
            if coll_name not in self.collections:
                continue

            try:
                adapter = self.collections[coll_name]
                collection_obj = adapter.collection
                items = collection_obj.get(limit=100000)

                for i in range(len(items['ids'])):
                    metadata = items['metadatas'][i] if i < len(items['metadatas']) else {}
                    result = {
                        'id': items['ids'][i],
                        'content': items['documents'][i] if i < len(items['documents']) else '',
                        'metadata': metadata,
                        'collection': coll_name
                    }
                    # Normalize for consistent fields (created_at, score, tags, etc.)
                    result = normalize_memory(result, coll_name)
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error getting all items from {coll_name}: {e}")

        # Apply metadata + date filters (previously bypassed for empty queries)
        if metadata_filters:
            chromadb_filters, date_filters = self._extract_date_filters(metadata_filters)
            # v0.3.5: Apply non-date metadata filters in Python (chromadb_filters
            # can't be passed to .get() retroactively since we already fetched)
            if chromadb_filters:
                for key, value in chromadb_filters.items():
                    if isinstance(value, dict):
                        # Operator filters like {"$gte": x} — skip, these are for ChromaDB
                        continue
                    all_results = [r for r in all_results if (r.get("metadata") or {}).get(key) == value]
            if date_filters:
                all_results = self._apply_date_filters(all_results, date_filters)

        # Apply scoring pipeline (Wilson, learned_score, final_rank_score)
        all_results = self.scoring_service.apply_scoring_to_results(all_results, sort=False)

        # Sort by requested order (default: recency for no-query searches)
        if sort_by == "score":
            all_results.sort(key=lambda x: x.get("score", 0.5), reverse=True)
        elif sort_by == "relevance":
            all_results.sort(key=lambda x: x.get("final_rank_score", 0.0), reverse=True)
        else:
            # Default: recency — use normalized created_at
            all_results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        paginated_results = all_results[offset:offset + limit]
        if return_metadata:
            return {
                "results": paginated_results,
                "total": len(all_results),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < len(all_results)
            }
        return paginated_results

    async def _search_collections(
        self,
        query_embedding: List[float],
        processed_query: str,
        collections: List[str],
        limit: int,
        metadata_filters: Optional[Dict[str, Any]]
    ) -> List[Dict]:
        """Search specified collections and apply boosts."""
        all_results = []

        for coll_name in collections:
            if coll_name not in self.collections:
                continue

            results = await self._search_single_collection(
                coll_name, query_embedding, processed_query, limit, metadata_filters
            )

            # Apply collection-specific boosts
            for r in results:
                r["collection"] = coll_name
                self._apply_collection_boost(r, coll_name, processed_query)

            all_results.extend(results)

        return all_results

    async def _search_single_collection(
        self,
        coll_name: str,
        query_embedding: List[float],
        processed_query: str,
        limit: int,
        metadata_filters: Optional[Dict[str, Any]]
    ) -> List[Dict]:
        """Search a single collection."""
        adapter = self.collections[coll_name]
        multiplier = self.config.search_multiplier

        # Build filters for memory_bank
        filters = metadata_filters
        if coll_name == "memory_bank":
            filters = (metadata_filters or {}).copy()
            if "status" not in filters:
                filters["status"] = {"$ne": "archived"}

        # Hybrid query (vector + BM25)
        results = await adapter.hybrid_query(
            query_vector=query_embedding,
            query_text=processed_query,
            top_k=limit * multiplier,
            filters=filters
        )

        # Add recency metadata for working memory
        if coll_name == "working":
            self._add_recency_metadata(results)

        return results

    def _apply_collection_boost(self, result: Dict, coll_name: str, query: str):
        """Apply collection-specific distance boosts."""
        # Patterns get slight boost
        if coll_name == "patterns":
            result["distance"] = result.get("distance", 1.0) * 0.9

        # Memory bank: boost by importance * confidence, with Wilson influence (v0.2.9)
        elif coll_name == "memory_bank":
            metadata = result.get("metadata", {})
            importance = self._parse_numeric(metadata.get("importance", 0.7))
            confidence = self._parse_numeric(metadata.get("confidence", 0.7))
            quality_score = importance * confidence

            # v0.2.9: Blend quality with Wilson score if enough uses
            uses = int(metadata.get("uses", 0))
            success_count = float(metadata.get("success_count", 0.0))

            if uses >= 3:
                # Calculate Wilson score from memory's own outcomes
                wilson = wilson_score_lower(success_count, uses)
                # v0.3.7: 100% Wilson for proven facts — pure track record dominates embedding
                metadata_boost = 1.0 - wilson * 0.8
            else:
                # Not enough data - use quality only (cold start protection)
                metadata_boost = 1.0 - quality_score * 0.4
            result["distance"] = result.get("distance", 1.0) * metadata_boost

            # Doc effectiveness boost (cross-doc pattern tracking)
            doc_id = result.get("id") or result.get("doc_id")
            if doc_id:
                eff = self.get_doc_effectiveness(doc_id)
                if eff and eff.get("total_uses", 0) >= 3:
                    eff_multiplier = 0.7 + eff["success_rate"] * 0.6
                    result["distance"] = result["distance"] / eff_multiplier

        # Books: boost recent uploads
        elif coll_name == "books":
            if result.get("upload_timestamp"):
                try:
                    upload_time = datetime.fromisoformat(result["upload_timestamp"])
                    age_days = (datetime.utcnow() - upload_time).days
                    if age_days <= 7:
                        result["distance"] = result.get("distance", 1.0) * 0.7
                except Exception:
                    pass

            # Doc effectiveness boost
            doc_id = result.get("id") or result.get("doc_id")
            if doc_id:
                eff = self.get_doc_effectiveness(doc_id)
                if eff and eff.get("total_uses", 0) >= 3:
                    eff_multiplier = 0.7 + eff["success_rate"] * 0.6
                    result["distance"] = result.get("distance", 1.0) / eff_multiplier

        # v0.3.6: Entity boost for ALL collections (was memory_bank only)
        entity_boost = self._calculate_entity_boost(query, result.get("id", ""))
        if entity_boost > 1.0:
            result["distance"] = result.get("distance", 1.0) / entity_boost

    def _add_recency_metadata(self, results: List[Dict]):
        """Add recency metadata to working memory results."""
        for r in results:
            metadata = r.get("metadata", {})
            if metadata.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(metadata["timestamp"])
                    minutes_ago = (datetime.now() - timestamp).total_seconds() / 60

                    if minutes_ago < 1:
                        metadata["recency"] = "just now"
                    elif minutes_ago < 60:
                        metadata["recency"] = f"{int(minutes_ago)} minutes ago"
                    else:
                        hours_ago = minutes_ago / 60
                        metadata["recency"] = f"{int(hours_ago)} hours ago"

                    r["minutes_ago"] = minutes_ago
                except Exception:
                    r["minutes_ago"] = 999

    def _parse_numeric(self, value: Any) -> float:
        """Parse value to float, handling various formats."""
        if isinstance(value, (list, tuple)):
            return float(value[0]) if value else 0.7
        elif isinstance(value, str):
            level_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
            return level_map.get(value.lower(), 0.7)
        try:
            return float(value) if value else 0.7
        except (ValueError, TypeError):
            return 0.7

    # =========================================================================
    # Entity Boost Calculation
    # =========================================================================

    def _calculate_entity_boost(self, query: str, doc_id: str) -> float:
        """
        Calculate quality boost based on Content KG entities.

        v0.3.6: Applies to ALL collections (was memory_bank only).
        Boosts documents containing high-quality entities that match query concepts.

        Returns:
            Boost multiplier (1.0 = no boost, up to 1.5 = 50% boost)
        """
        try:
            query_concepts = self.kg_service.extract_concepts(query)
            query_entities = [c for c in query_concepts if len(c) >= 3]

            if not query_entities:
                return 1.0

            doc_entities = self.kg_service.content_graph._doc_entities.get(doc_id, set())

            if not doc_entities:
                return 1.0

            total_boost = 0.0
            for entity in query_entities:
                if entity in doc_entities and entity in self.kg_service.content_graph.entities:
                    entity_quality = self.kg_service.content_graph.entities[entity].get("avg_quality", 0.0)
                    total_boost += entity_quality

            # Cap boost at 50%
            boost_multiplier = 1.0 + min(total_boost * 0.2, 0.5)

            if boost_multiplier > 1.0:
                logger.debug(f"Entity boost for {doc_id}: {boost_multiplier:.2f}x")

            return boost_multiplier
        except Exception as e:
            logger.error(f"Error calculating entity boost: {e}")
            return 1.0

    # =========================================================================
    # Document Effectiveness
    # =========================================================================

    def get_doc_effectiveness(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Aggregate success rate for a specific doc from Action KG examples.

        Returns:
            Dict with success_rate, total_uses, etc., or None if no data
        """
        successes = 0
        failures = 0
        partials = 0

        for key, stats in self.kg_service.knowledge_graph.get("context_action_effectiveness", {}).items():
            for example in stats.get("examples", []):
                if example.get("doc_id") == doc_id:
                    if example["outcome"] == "worked":
                        successes += 1
                    elif example["outcome"] == "failed":
                        failures += 1
                    else:
                        partials += 1

        total = successes + failures + partials
        if total == 0:
            return None

        return {
            "successes": successes,
            "failures": failures,
            "partials": partials,
            "total_uses": total,
            "success_rate": (successes + partials * 0.5) / total
        }

    # =========================================================================
    # Cross-Encoder Reranking
    # =========================================================================

    async def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Rerank top candidates with cross-encoder for precision.

        Returns:
            Reranked results with cross-encoder scores
        """
        if not self.reranker:
            return candidates

        try:
            # Take top-30 candidates for reranking
            top_candidates = sorted(
                candidates,
                key=lambda x: x.get("final_rank_score", 0.0),
                reverse=True
            )[:30]

            # Prepare query-document pairs
            pairs = []
            for candidate in top_candidates:
                doc_text = candidate.get("text", "")
                if not doc_text and candidate.get("metadata"):
                    doc_text = candidate.get("metadata", {}).get("content", "")
                pairs.append([query, doc_text[:512]])

            # Score with cross-encoder
            ce_scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)

            # Blend scores
            # v0.3.2: CE weight drops to 0 for scored memories (uses >= 3).
            # Wilson has real outcome data at that point — don't let semantic
            # similarity undercut learned rankings. CE only helps cold-start.
            for i, candidate in enumerate(top_candidates):
                ce_score = float(ce_scores[i])
                candidate["ce_score"] = ce_score
                original_score = candidate.get("final_rank_score", 0.5)

                metadata = candidate.get("metadata", {})
                uses = int(metadata.get("uses", 0))
                collection = candidate.get("collection", "")

                # Scored memories: skip CE blending, preserve Wilson ranking
                if uses >= 3:
                    candidate["final_rank_score"] = original_score
                    continue

                # Cold-start memories: CE is valuable, blend it in
                if collection == "memory_bank":
                    importance = self._parse_numeric(metadata.get("importance", 0.7))
                    confidence = self._parse_numeric(metadata.get("confidence", 0.7))
                    quality = importance * confidence

                    ce_norm = (ce_score + 1) / 2
                    ce_weight = 0.3
                    quality_boost = 1.0 + quality * 0.3
                    blended = ((1 - ce_weight) * original_score + ce_weight * ce_norm) * quality_boost
                else:
                    ce_norm = (ce_score + 1) / 2
                    ce_weight = 0.4
                    blended = (1 - ce_weight) * original_score + ce_weight * ce_norm

                candidate["final_rank_score"] = blended

            # Re-sort by updated score
            top_candidates.sort(key=lambda x: x.get("final_rank_score", 0.0), reverse=True)

            # Merge back: use reranked top + remaining
            top_ids = {c.get("id") for c in top_candidates}
            remaining = [c for c in candidates if c.get("id") not in top_ids]

            return top_candidates + remaining

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return candidates

    # =========================================================================
    # Tracking and Caching
    # =========================================================================

    def _track_search_results(
        self,
        query: str,
        results: List[Dict],
        transparency_context: Optional[Any]
    ):
        """Track search results for KG learning."""
        # Track usage for returned results
        for result in results:
            if "collection" in result and "id" in result:
                self._track_usage(query, result["collection"], result["id"])

        # Cache doc_ids for outcome scoring
        session_id = 'default'
        if transparency_context and hasattr(transparency_context, 'session_id'):
            session_id = transparency_context.session_id

        cached_doc_ids = []
        for result in results:
            collection = result.get("collection", "")
            if collection in ["working", "history", "patterns"]:
                doc_id = result.get("id")
                if doc_id:
                    cached_doc_ids.append(doc_id)

        self._cached_doc_ids[session_id] = cached_doc_ids
        if cached_doc_ids:
            logger.debug(f"Cached {len(cached_doc_ids)} doc_ids for outcome scoring")

        # Track with transparency context
        if transparency_context and hasattr(transparency_context, 'track_memory_search'):
            confidence_scores = [math.exp(-r.get("distance", 0.5) / 100.0) for r in results]
            transparency_context.track_memory_search(
                query=query,
                collections=[r.get("collection") for r in results[:1]],
                results_count=len(results),
                confidence_scores=confidence_scores
            )

    def _track_usage(self, query: str, collection: str, doc_id: str):
        """Track which collection was used for which query."""
        concepts = self.kg_service.extract_concepts(query)
        for concept in concepts:
            if concept not in self.kg_service.knowledge_graph["routing_patterns"]:
                self.kg_service.knowledge_graph["routing_patterns"][concept] = {
                    "collections_used": {},
                    "best_collection": collection,
                    "success_rate": 0.5
                }

    def _get_fragment(self, collection: str, doc_id: str) -> Optional[Dict]:
        """Get a document fragment by collection and ID."""
        if collection in self.collections:
            return self.collections[collection].get_fragment(doc_id)
        return None

    def get_cached_doc_ids(self, session_id: str = 'default') -> List[str]:
        """Get cached doc_ids for a session."""
        return self._cached_doc_ids.get(session_id, [])

    # =========================================================================
    # Book Search (Specialized)
    # =========================================================================

    async def search_books(
        self,
        query: str,
        chunk_type: Optional[str] = None,
        has_code: Optional[bool] = None,
        code_language: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Enhanced book search with metadata filtering.

        Args:
            query: Search query text
            chunk_type: Filter by chunk type ("code", "prose", "mixed")
            has_code: Filter by presence of code
            code_language: Filter by programming language
            n_results: Number of results to return

        Returns:
            List of search results with enhanced context
        """
        if "books" not in self.collections:
            logger.warning("Books collection not initialized")
            return []

        # Build where clause
        where = {}
        if chunk_type:
            where["chunk_type"] = chunk_type
        if has_code is not None:
            where["has_code"] = has_code
        if code_language:
            where["code_language"] = code_language

        try:
            results = await self.collections["books"].query(
                query_texts=[query],
                n_results=n_results * 2,
                where=where if where else None
            )

            if not results or not results.get("ids"):
                return []

            # Format results
            formatted_results = []
            for i in range(min(n_results, len(results["ids"][0]))):
                result = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0,
                    "collection": "books"
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Book search failed: {e}")
            return []
