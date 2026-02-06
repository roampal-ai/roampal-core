"""
Knowledge Graph Service - Manages dual KG system (Routing KG + Content KG).

Extracted from UnifiedMemorySystem as part of refactoring.
Includes race condition fix for debounced KG saves.

Responsibilities:
- Loading/saving both routing KG and content KG
- Concept extraction from text
- Building concept relationships
- Tracking problem-solution patterns
- KG cleanup operations
- Entity/relationship queries for visualization
"""

import asyncio
import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from filelock import FileLock

from .config import MemoryConfig
from .content_graph import ContentGraph

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """
    Manages dual Knowledge Graph system.

    Dual KG Architecture:
    - Routing KG: Query patterns -> collection routing decisions
    - Content KG: Memory content -> entity relationships

    Includes race condition fix: Uses asyncio.Lock to protect
    concurrent access to _kg_save_task in debounced saves.
    """

    def __init__(
        self,
        kg_path: Path,
        content_graph_path: Path,
        relationships_path: Path,
        config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize KnowledgeGraphService.

        Args:
            kg_path: Path to routing KG JSON file
            content_graph_path: Path to content KG JSON file
            relationships_path: Path to memory relationships JSON file
            config: Optional MemoryConfig for thresholds
        """
        self.config = config or MemoryConfig()
        self.kg_path = kg_path
        self.content_graph_path = content_graph_path
        self.relationships_path = relationships_path

        # Load graphs
        self.knowledge_graph = self._load_kg()
        self.content_graph = self._load_content_graph()
        self.relationships = self._load_relationships()

        # Debounced save state with RACE CONDITION FIX
        self._kg_save_task: Optional[asyncio.Task] = None
        self._kg_save_pending = False
        self._kg_save_lock = asyncio.Lock()  # FIX: Protect concurrent task access

        logger.info(f"KnowledgeGraphService initialized from {kg_path}")

    # =========================================================================
    # Loading Methods
    # =========================================================================

    def _load_content_graph(self) -> ContentGraph:
        """
        Load Content Knowledge Graph from disk.

        CRITICAL: This is a core feature for entity relationship mapping.
        Do not disable or remove - required for dual KG visualization.
        """
        if self.content_graph_path.exists():
            try:
                return ContentGraph.load_from_file(str(self.content_graph_path))
            except Exception as e:
                logger.warning(f"Failed to load content graph, creating new: {e}")
        return ContentGraph()

    def _load_kg(self) -> Dict[str, Any]:
        """Load knowledge graph routing patterns."""
        default_kg = {
            "routing_patterns": {},      # concept -> best_collection
            "success_rates": {},         # collection -> success_rate
            "failure_patterns": {},      # concept -> failure_reasons
            "problem_categories": {},    # problem_type -> preferred_collections
            "problem_solutions": {},     # problem_signature -> [solution_ids]
            "solution_patterns": {},     # pattern_hash -> {problem, solution, success_rate}
            # v0.2.1 Causal Learning: Action-level effectiveness tracking
            "context_action_effectiveness": {}  # (context, action, collection) -> {success, fail, success_rate}
        }

        if self.kg_path.exists():
            try:
                with open(self.kg_path, 'r') as f:
                    loaded_kg = json.load(f)
                    # Ensure all required keys exist
                    for key in default_kg:
                        if key not in loaded_kg:
                            loaded_kg[key] = default_kg[key]
                    return loaded_kg
            except Exception:
                pass
        return default_kg

    def _load_relationships(self) -> Dict[str, Any]:
        """Load memory relationships."""
        if self.relationships_path.exists():
            try:
                with open(self.relationships_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "related": {},       # doc_id -> [related_doc_ids]
            "evolution": {},     # doc_id -> {parent, children}
            "conflicts": {}      # doc_id -> [conflicting_doc_ids]
        }

    def reload_kg(self):
        """Reload KG from disk to pick up changes from other processes."""
        self.knowledge_graph = self._load_kg()

    # =========================================================================
    # Saving Methods (with race condition fix)
    # =========================================================================

    def _save_kg_sync(self):
        """
        Synchronous save both routing KG and content KG.

        CRITICAL: Saves both graphs atomically to maintain consistency.
        Do not remove content graph save - it's required for entity tracking.
        """
        # Save routing KG
        lock_path = str(self.kg_path) + ".lock"
        try:
            with FileLock(lock_path, timeout=10):
                self.kg_path.parent.mkdir(exist_ok=True, parents=True)
                # Write to temp file first then rename (atomic operation)
                temp_path = self.kg_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(self.knowledge_graph, f, indent=2)
                temp_path.replace(self.kg_path)
        except PermissionError as e:
            logger.error(f"Permission denied saving routing KG: {e}")
        except Exception as e:
            logger.error(f"Failed to save routing KG: {e}", exc_info=True)

        # CRITICAL: Save content KG (entity relationships)
        try:
            self.content_graph.save_to_file(str(self.content_graph_path))
        except Exception as e:
            logger.error(f"Failed to save content KG: {e}", exc_info=True)

    async def _save_kg(self):
        """Save knowledge graph asynchronously."""
        await asyncio.to_thread(self._save_kg_sync)

    async def _debounced_save_kg(self):
        """
        Debounce KG saves to batch within 5-second window to reduce file I/O.

        RACE CONDITION FIX: Uses asyncio.Lock to protect concurrent access
        to _kg_save_task. Without this, multiple coroutines could both
        cancel the task and create new ones simultaneously.
        """
        async with self._kg_save_lock:  # FIX: Serialize access
            # Cancel existing pending save task
            if self._kg_save_task and not self._kg_save_task.done():
                self._kg_save_task.cancel()
                try:
                    await self._kg_save_task
                except asyncio.CancelledError:
                    pass

            # Create new delayed save task
            async def delayed_save():
                try:
                    await asyncio.sleep(self.config.kg_debounce_seconds)
                    await self._save_kg()
                    self._kg_save_pending = False
                except asyncio.CancelledError:
                    pass

            self._kg_save_pending = True
            self._kg_save_task = asyncio.create_task(delayed_save())

    async def debounced_save_kg(self):
        """Public alias for debounced KG save (used by OutcomeService)."""
        await self._debounced_save_kg()

    def update_success_rate(self, doc_id: str, outcome: str):
        """
        Update success rate tracking for a document in the routing KG.

        This tracks which documents (by ID) lead to successful outcomes,
        enabling the routing system to prefer historically successful sources.

        Args:
            doc_id: Document ID that had an outcome
            outcome: "worked", "failed", or "partial"
        """
        # Extract collection from doc_id (e.g., "working_abc123" -> "working")
        parts = doc_id.split("_")
        if len(parts) < 2:
            return

        collection = parts[0]

        # Track in routing_patterns
        if collection not in self.knowledge_graph["routing_patterns"]:
            self.knowledge_graph["routing_patterns"][collection] = {}

        stats = self.knowledge_graph["routing_patterns"][collection]
        # Ensure required keys exist (handles old-schema entries missing these)
        for key in ("successes", "failures", "partials", "total"):
            if key not in stats:
                stats[key] = 0
        if "success_rate" not in stats:
            stats["success_rate"] = 0.5

        stats["total"] += 1

        if outcome == "worked":
            stats["successes"] += 1
        elif outcome == "failed":
            stats["failures"] += 1
        else:
            stats["partials"] += 1

        # Recalculate success rate (partials count as 0.5)
        if stats["total"] > 0:
            weighted = stats["successes"] + (stats["partials"] * 0.5)
            stats["success_rate"] = weighted / stats["total"]

    def add_relationship(self, doc_id: str, rel_type: str, data: Dict[str, Any]):
        """
        Add a relationship to the relationships tracking structure.

        Args:
            doc_id: Document ID
            rel_type: Relationship type (e.g., "evolution", "related", "conflicts")
            data: Relationship data
        """
        if rel_type not in self.relationships:
            self.relationships[rel_type] = {}

        if doc_id not in self.relationships[rel_type]:
            self.relationships[rel_type][doc_id] = []

        self.relationships[rel_type][doc_id].append(data)



    def _save_relationships_sync(self):
        """Synchronous save memory relationships - to be called in thread with file locking."""
        lock_path = str(self.relationships_path) + ".lock"
        try:
            with FileLock(lock_path, timeout=10):
                self.relationships_path.parent.mkdir(exist_ok=True, parents=True)
                # Atomic write
                temp_path = self.relationships_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(self.relationships, f, indent=2)
                temp_path.replace(self.relationships_path)
        except PermissionError as e:
            logger.error(f"Permission denied saving relationships: {e}")
        except Exception as e:
            logger.error(f"Failed to save relationships: {e}", exc_info=True)

    async def save_relationships(self):
        """Save relationships asynchronously."""
        await asyncio.to_thread(self._save_relationships_sync)

    # =========================================================================
    # Concept Extraction
    # =========================================================================

    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract N-grams (unigrams, bigrams, trigrams) from text for KG routing.
        Implements architecture.md specification for concept extraction.
        """
        concepts: Set[str] = set()

        # Normalize and tokenize
        text_lower = text.lower()
        # Remove punctuation except hyphens and underscores
        text_clean = re.sub(r'[^\w\s\-_]', ' ', text_lower)
        words = text_clean.split()

        # Stop words (expanded set)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "to", "for", "of",
            "with", "in", "on", "at", "by", "from", "as", "be", "this", "that",
            "it", "i", "you", "we", "they", "my", "your", "our", "their", "what",
            "when", "where", "how", "why", "can", "could", "would", "should"
        }

        # v0.2.1: Blocklist for MCP tool names and internal function-like patterns
        # These pollute the Content KG with non-semantic entities
        tool_blocklist = {
            # MCP tool names
            "search_memory", "add_to_memory_bank", "create_memory", "update_memory",
            "archive_memory", "get_context_insights", "record_response", "validated",
            # Common patterns from tool descriptions
            "memory_bank", "working", "history", "patterns", "books",
            # Internal function-like terms
            "function", "parameter", "response", "request", "query", "result",
            "collection", "collections", "metadata", "timestamp", "document"
        }

        # Filter stop words
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]

        # 1. Extract UNIGRAMS (single words)
        for word in filtered_words:
            if len(word) > 3:  # Only meaningful words
                concepts.add(word)

        # 2. Extract BIGRAMS (2-word phrases)
        for i in range(len(filtered_words) - 1):
            bigram = f"{filtered_words[i]}_{filtered_words[i+1]}"
            concepts.add(bigram)

        # 3. Extract TRIGRAMS (3-word phrases)
        for i in range(len(filtered_words) - 2):
            trigram = f"{filtered_words[i]}_{filtered_words[i+1]}_{filtered_words[i+2]}"
            concepts.add(trigram)

        # Filter out blocklisted terms
        filtered_concepts = [
            c for c in concepts
            if not any(blocked in c for blocked in tool_blocklist)
        ]

        return filtered_concepts

    # =========================================================================
    # Concept Relationships
    # =========================================================================

    def build_concept_relationships(self, concepts: List[str]):
        """Build relationships between co-occurring concepts."""
        if "relationships" not in self.knowledge_graph:
            self.knowledge_graph["relationships"] = {}

        # Build relationships between all concept pairs
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Create bidirectional relationship key (sorted for consistency)
                rel_key = "|".join(sorted([concept1, concept2]))

                if rel_key not in self.knowledge_graph["relationships"]:
                    self.knowledge_graph["relationships"][rel_key] = {
                        "co_occurrence": 0,
                        "success_together": 0,
                        "failure_together": 0
                    }

                # Increment co-occurrence
                self.knowledge_graph["relationships"][rel_key]["co_occurrence"] += 1

    async def update_kg_routing(self, query: str, collection: str, outcome: str):
        """Update KG routing patterns and relationships based on outcome."""
        if not query:
            return

        concepts = self.extract_concepts(query)

        # Build relationships between concepts
        self.build_concept_relationships(concepts)

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

            # Update best collection
            best_collection = collection
            best_rate = 0.0

            for coll_name, coll_stats in pattern["collections_used"].items():
                # Calculate success rate: successes / (successes + failures)
                # Excludes "partial" and "unknown" outcomes per v0.1.6 spec
                total_with_feedback = coll_stats["successes"] + coll_stats["failures"]

                if total_with_feedback > 0:
                    rate = coll_stats["successes"] / total_with_feedback
                else:
                    rate = 0.5  # Neutral baseline (50%) for untested patterns

                if rate > best_rate:
                    best_rate = rate
                    best_collection = coll_name

            pattern["best_collection"] = best_collection
            # Default to 0.5 if no collections have been tested with explicit feedback
            pattern["success_rate"] = best_rate if best_rate > 0 else 0.5

        # Update relationship outcomes
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                rel_key = "|".join(sorted([concept1, concept2]))
                if rel_key in self.knowledge_graph.get("relationships", {}):
                    rel_data = self.knowledge_graph["relationships"][rel_key]
                    if outcome == "worked":
                        rel_data["success_together"] += 1
                    elif outcome == "failed":
                        rel_data["failure_together"] += 1

        # Save KG with proper await (debounced)
        await self._debounced_save_kg()


    def add_problem_category(self, problem_key: str, doc_id: str):
        """Add a document to a problem category."""
        if "problem_categories" not in self.knowledge_graph:
            self.knowledge_graph["problem_categories"] = {}

        if problem_key not in self.knowledge_graph["problem_categories"]:
            self.knowledge_graph["problem_categories"][problem_key] = []

        if doc_id not in self.knowledge_graph["problem_categories"][problem_key]:
            self.knowledge_graph["problem_categories"][problem_key].append(doc_id)

    def get_problem_categories(self) -> Dict[str, List[str]]:
        """Get all problem categories."""
        return self.knowledge_graph.get("problem_categories", {})

    # =========================================================================
    # Problem-Solution Tracking
    # =========================================================================

    async def find_known_solutions(self, query: str, get_fragment_fn: Callable) -> List[Dict[str, Any]]:
        """
        Find known solutions for similar problems.

        Args:
            query: Search query text
            get_fragment_fn: Function to retrieve document by ID (collection_name, doc_id) -> doc
        """
        try:
            if not query:
                return []

            # Extract concepts from the query
            query_concepts = self.extract_concepts(query)
            query_signature = "_".join(sorted(query_concepts[:5]))

            known_solutions = []

            # Ensure problem_solutions exists in knowledge graph
            if "problem_solutions" not in self.knowledge_graph:
                self.knowledge_graph["problem_solutions"] = {}

            # Look for exact problem matches
            if query_signature in self.knowledge_graph["problem_solutions"]:
                solutions = self.knowledge_graph["problem_solutions"][query_signature]

                # Sort by success count and recency
                sorted_solutions = sorted(
                    solutions,
                    key=lambda x: (x.get("success_count", 0), x.get("last_used", "")),
                    reverse=True
                )

                # Add top solutions to results
                for solution in sorted_solutions[:3]:
                    doc_id = solution.get("doc_id")
                    if doc_id:
                        # Try to find the actual document
                        for coll_name in ["patterns", "history", "memory_bank", "books"]:
                            if doc_id.startswith(coll_name):
                                doc = get_fragment_fn(coll_name, doc_id)
                                if doc:
                                    # Boost the score for known solutions
                                    doc["distance"] = doc.get("distance", 1.0) * 0.5  # 50% boost
                                    doc["is_known_solution"] = True
                                    doc["solution_success_count"] = solution.get("success_count", 0)
                                    known_solutions.append(doc)
                                    logger.info(f"Found known solution: {doc_id} (used {solution['success_count']} times)")
                                    break

            # Also check for partial matches (3+ concept overlap)
            for problem_sig, solutions in self.knowledge_graph["problem_solutions"].items():
                if problem_sig != query_signature:
                    problem_concepts_stored = set(problem_sig.split("_"))
                    overlap = len(set(query_concepts) & problem_concepts_stored)

                    if overlap >= 3:  # Significant overlap
                        for solution in solutions[:1]:  # Take best from partial matches
                            doc_id = solution.get("doc_id")
                            if doc_id and doc_id not in [s.get("id") for s in known_solutions]:
                                for coll_name in ["patterns", "history", "memory_bank", "books"]:
                                    if doc_id.startswith(coll_name):
                                        doc = get_fragment_fn(coll_name, doc_id)
                                        if doc:
                                            doc["distance"] = doc.get("distance", 1.0) * 0.7  # 30% boost
                                            doc["is_partial_solution"] = True
                                            doc["concept_overlap"] = overlap
                                            known_solutions.append(doc)
                                            break

            return known_solutions

        except Exception as e:
            logger.error(f"Error finding known solutions: {e}")
            return []

    async def track_problem_solution(
        self,
        doc_id: str,
        metadata: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ):
        """Track successful problem->solution patterns for future reuse."""
        try:
            # Extract problem signature from the original query/context
            problem_text = metadata.get("original_context", "") or metadata.get("query", "")
            solution_text = metadata.get("text", "")

            if not problem_text or not solution_text:
                return

            # Create problem signature (simplified concepts)
            problem_concepts = self.extract_concepts(problem_text)
            problem_signature = "_".join(sorted(problem_concepts[:5]))  # Top 5 concepts

            if not problem_signature:
                return

            # Store problem->solution mapping
            if problem_signature not in self.knowledge_graph["problem_solutions"]:
                self.knowledge_graph["problem_solutions"][problem_signature] = []

            solution_entry = {
                "doc_id": doc_id,
                "solution": solution_text,  # Store abbreviated solution
                "success_count": 1,
                "last_used": datetime.now().isoformat(),
                "contexts": [context] if context else []
            }

            # Check if this solution already exists for this problem
            existing_solutions = self.knowledge_graph["problem_solutions"][problem_signature]
            solution_found = False

            for existing in existing_solutions:
                if existing["doc_id"] == doc_id:
                    # Update existing solution
                    existing["success_count"] += 1
                    existing["last_used"] = datetime.now().isoformat()
                    if context and context not in existing.get("contexts", []):
                        existing.setdefault("contexts", []).append(context)
                    solution_found = True
                    break

            if not solution_found:
                existing_solutions.append(solution_entry)

            # Track solution patterns (for pattern matching)
            pattern_hash = f"{problem_signature}::{doc_id}"
            if pattern_hash not in self.knowledge_graph["solution_patterns"]:
                self.knowledge_graph["solution_patterns"][pattern_hash] = {
                    "problem": problem_text,
                    "solution": solution_text,
                    "success_count": 0,
                    "failure_count": 0,
                    "contexts": []
                }

            pattern = self.knowledge_graph["solution_patterns"][pattern_hash]
            pattern["success_count"] += 1
            pattern["success_rate"] = pattern["success_count"] / (pattern["success_count"] + pattern["failure_count"])

            # Save updated KG with proper await (debounced)
            await self._debounced_save_kg()

            logger.info(f"Tracked problem->solution pattern: {problem_signature[:30]}... -> {doc_id}")

        except Exception as e:
            logger.error(f"Error tracking problem->solution: {e}")

    # =========================================================================
    # Cleanup Methods
    # =========================================================================

    async def cleanup_dead_references(self, doc_exists_fn: Callable[[str], bool]) -> int:
        """
        Remove doc_id references that no longer exist in collections.

        Args:
            doc_exists_fn: Function to check if doc exists (doc_id) -> bool
        """
        try:
            cleaned = 0

            # Clean problem_categories
            for problem_key, doc_ids in list(self.knowledge_graph.get("problem_categories", {}).items()):
                valid_ids = [doc_id for doc_id in doc_ids if doc_exists_fn(doc_id)]
                if len(valid_ids) < len(doc_ids):
                    cleaned += len(doc_ids) - len(valid_ids)
                    if valid_ids:
                        self.knowledge_graph["problem_categories"][problem_key] = valid_ids
                    else:
                        del self.knowledge_graph["problem_categories"][problem_key]

            # Clean problem_solutions
            for problem_sig, solutions in list(self.knowledge_graph.get("problem_solutions", {}).items()):
                valid_solutions = [s for s in solutions if doc_exists_fn(s.get("doc_id"))]
                if len(valid_solutions) < len(solutions):
                    cleaned += len(solutions) - len(valid_solutions)
                    if valid_solutions:
                        self.knowledge_graph["problem_solutions"][problem_sig] = valid_solutions
                    else:
                        del self.knowledge_graph["problem_solutions"][problem_sig]

            # Clean routing_patterns (remove patterns with 0 total uses)
            for concept, pattern in list(self.knowledge_graph.get("routing_patterns", {}).items()):
                collections_used = pattern.get("collections_used", {})
                total_uses = sum(stats.get("total", 0) for stats in collections_used.values())
                if total_uses == 0:
                    del self.knowledge_graph["routing_patterns"][concept]
                    cleaned += 1

            if cleaned > 0:
                logger.info(f"KG cleanup: removed {cleaned} dead references")
                await self._save_kg()  # Immediate save for cleanup operation

            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning KG dead references: {e}")
            return 0

    async def cleanup_action_kg_for_doc_ids(self, doc_ids: List[str]) -> int:
        """
        Remove Action KG examples referencing specific doc_ids (v0.2.6).

        Called when books are deleted to prevent stale doc_id references
        in context_action_effectiveness examples.

        Args:
            doc_ids: List of document IDs to remove from Action KG examples

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

                # Filter out examples with matching doc_ids
                stats["examples"] = [
                    ex for ex in examples
                    if ex.get("doc_id") not in doc_id_set
                ]

                cleaned += original_count - len(stats["examples"])

            if cleaned > 0:
                logger.info(f"Action KG cleanup: removed {cleaned} examples for deleted doc_ids")
                await self._save_kg()

            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning Action KG for doc_ids: {e}")
            return 0

    # =========================================================================
    # Entity/Relationship Queries (for visualization)
    # =========================================================================

    async def get_kg_entities(
        self,
        filter_text: Optional[str] = None,
        limit: int = 25  # v0.3.0: Reduced from 200 - frontend only displays 20 nodes (MAX_NODES_DISPLAYED)
    ) -> List[Dict[str, Any]]:
        """
        Get entities from DUAL knowledge graph (Routing KG + Content KG merged).

        CRITICAL: This merges both graphs to provide complete entity view.
        - Routing KG: Query patterns -> collection routing
        - Content KG: Entity relationships from memory_bank content
        - Entities in both graphs get source="both" (purple nodes in UI)

        NOTE: Reloads KG from disk to pick up changes from MCP process.
        """
        # Reload KG from disk to pick up changes from MCP process
        self.reload_kg()

        entities_map: Dict[str, Dict[str, Any]] = {}

        # STEP 1: Get routing KG entities (query-based patterns)
        for concept, pattern in self.knowledge_graph.get("routing_patterns", {}).items():
            if filter_text and filter_text.lower() not in concept.lower():
                continue

            # Count routing connections
            routing_connections = 0
            for rel_key in self.knowledge_graph.get("relationships", {}).keys():
                if concept in rel_key.split("|"):
                    routing_connections += 1

            # Get total usage across all collections
            collections_used = pattern.get("collections_used", {})
            total_usage = sum(c.get("total", 0) for c in collections_used.values())

            entities_map[concept] = {
                "entity": concept,
                "source": "routing",  # Will be updated if also in content KG
                "routing_connections": routing_connections,
                "content_connections": 0,
                "total_connections": routing_connections,
                "success_rate": pattern.get("success_rate", 0.5),
                "best_collection": pattern.get("best_collection"),
                "collections_used": collections_used,
                "usage_count": total_usage,
                "mentions": 0,  # From content KG
                "last_used": pattern.get("last_used"),
                "created_at": pattern.get("created_at")
            }

        # STEP 2: Get content KG entities (memory-based relationships)
        # CRITICAL: Do not skip this step - provides green/purple nodes in UI
        content_entities = self.content_graph.get_all_entities(min_mentions=1)

        # v0.2.1: Blocklist for filtering out tool-like entities
        tool_blocklist_terms = {
            "search_memory", "add_to_memory_bank", "create_memory", "update_memory",
            "archive_memory", "get_context_insights", "record_response", "validated",
            "memory_bank", "working", "history", "patterns", "books",
            "function", "parameter", "response", "request", "query", "result",
            "collection", "collections", "metadata", "timestamp", "document"
        }

        for entity_data in content_entities:
            entity_name = entity_data["entity"]

            # v0.2.1: Skip entities that look like tool names
            is_tool_like = any(term in entity_name for term in tool_blocklist_terms)
            if is_tool_like:
                continue

            if filter_text and filter_text.lower() not in entity_name.lower():
                continue

            # Count content connections
            content_rels = self.content_graph.get_entity_relationships(entity_name, min_strength=0.0)
            content_connections = len(content_rels)

            if entity_name in entities_map:
                # Entity exists in BOTH graphs -> source="both" (purple node)
                entities_map[entity_name]["source"] = "both"
                entities_map[entity_name]["content_connections"] = content_connections
                entities_map[entity_name]["total_connections"] += content_connections
                entities_map[entity_name]["mentions"] = entity_data["mentions"]
            else:
                # Entity only in content KG -> source="content" (green node)
                entities_map[entity_name] = {
                    "entity": entity_name,
                    "source": "content",  # Content KG only
                    "routing_connections": 0,
                    "content_connections": content_connections,
                    "total_connections": content_connections,
                    "success_rate": 0.5,  # Neutral (no routing data)
                    "best_collection": "memory_bank",  # Content entities are from memory_bank
                    "collections_used": {"memory_bank": {"total": entity_data["mentions"]}},
                    "usage_count": entity_data["mentions"],
                    "mentions": entity_data["mentions"],
                    "last_used": entity_data.get("last_seen"),
                    "created_at": entity_data.get("first_seen")
                }

        # STEP 3: Get Action Effectiveness KG entities (context|action|collection patterns)
        # v0.2.1: Orange nodes showing action success rates per context
        for key, stats in self.knowledge_graph.get("context_action_effectiveness", {}).items():
            parts = key.split("|")
            if len(parts) >= 2:
                context_type = parts[0]
                action_type = parts[1]
                collection = parts[2] if len(parts) > 2 else "*"

                # Create readable label
                label = f"{action_type}@{context_type}"
                if collection != "*":
                    label += f"->{collection}"

                if filter_text and filter_text.lower() not in label.lower():
                    continue

                total_uses = stats.get("successes", 0) + stats.get("failures", 0)
                if total_uses == 0:
                    continue  # Skip unused patterns

                success_rate = stats.get("success_rate", 0.5)

                # Don't overwrite routing/content entities with same name
                if label not in entities_map:
                    entities_map[label] = {
                        "entity": label,
                        "source": "action",  # Orange nodes for action effectiveness
                        "routing_connections": 0,
                        "content_connections": 0,
                        "total_connections": 0,
                        "success_rate": success_rate,
                        "best_collection": collection if collection != "*" else None,
                        "collections_used": {collection: {"total": total_uses, "successes": stats.get("successes", 0), "failures": stats.get("failures", 0)}},
                        "usage_count": total_uses,
                        "mentions": 0,
                        "last_used": stats.get("last_used"),
                        "created_at": stats.get("first_used"),
                        # Action-specific metadata
                        "context_type": context_type,
                        "action_type": action_type,
                        "partials": stats.get("partials", 0)
                    }

        # Convert to list and sort by usage
        entities = list(entities_map.values())
        entities.sort(key=lambda x: x["usage_count"], reverse=True)
        return entities[:limit]

    async def get_kg_relationships(self, entity: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific entity (DUAL KG merged).

        CRITICAL: Merges routing + content relationships for complete view.
        """
        relationships_map: Dict[str, Dict[str, Any]] = {}

        # STEP 1: Get routing KG relationships
        for rel_key, rel_data in self.knowledge_graph.get("relationships", {}).items():
            concepts = rel_key.split("|")
            if entity in concepts:
                related = concepts[1] if concepts[0] == entity else concepts[0]
                relationships_map[related] = {
                    "related_entity": related,
                    "source": "routing",  # Will update if also in content
                    "strength": rel_data.get("co_occurrence", 0),
                    "total_strength": rel_data.get("co_occurrence", 0),
                    "success_together": rel_data.get("success_together", 0),
                    "failure_together": rel_data.get("failure_together", 0),
                    "content_strength": 0  # From content KG
                }

        # STEP 2: Get content KG relationships
        # CRITICAL: Do not skip - provides entity relationship visualization
        content_rels = self.content_graph.get_entity_relationships(entity, min_strength=0.0)
        for rel_data in content_rels:
            related = rel_data["related_entity"]
            content_strength = rel_data["strength"]

            if related in relationships_map:
                # Relationship exists in BOTH graphs
                relationships_map[related]["source"] = "both"
                relationships_map[related]["content_strength"] = content_strength
                relationships_map[related]["total_strength"] += content_strength
            else:
                # Relationship only in content KG
                relationships_map[related] = {
                    "related_entity": related,
                    "source": "content",  # Content KG only
                    "strength": 0,  # No routing data
                    "total_strength": content_strength,
                    "success_together": 0,
                    "failure_together": 0,
                    "content_strength": content_strength
                }

        relationships = list(relationships_map.values())
        relationships.sort(key=lambda x: x["total_strength"], reverse=True)
        return relationships

    # =========================================================================
    # Content Graph Integration
    # =========================================================================

    def add_failure_pattern(self, failure_reason: str, doc_id: str, problem_text: str):
        """Track a failure pattern for learning."""
        if "failure_patterns" not in self.knowledge_graph:
            self.knowledge_graph["failure_patterns"] = {}

        if failure_reason not in self.knowledge_graph["failure_patterns"]:
            self.knowledge_graph["failure_patterns"][failure_reason] = []

        self.knowledge_graph["failure_patterns"][failure_reason].append({
            "doc_id": doc_id,
            "problem_text": problem_text
        })

    def add_problem_solution(
        self,
        problem_signature: str,
        doc_id: str,
        solution_text: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Track a successful problem->solution mapping."""
        if "problem_solutions" not in self.knowledge_graph:
            self.knowledge_graph["problem_solutions"] = {}

        if problem_signature not in self.knowledge_graph["problem_solutions"]:
            self.knowledge_graph["problem_solutions"][problem_signature] = []

        # Add if not already present
        existing = self.knowledge_graph["problem_solutions"][problem_signature]
        if doc_id not in existing:
            existing.append(doc_id)

    def add_solution_pattern(
        self,
        doc_id: str,
        solution_text: str,
        score: float,
        problem_keys: List[str],
        solution_concepts: List[str]
    ):
        """Track a solution pattern for reuse."""
        if "solution_patterns" not in self.knowledge_graph:
            self.knowledge_graph["solution_patterns"] = {}

        pattern_key = f"{doc_id}::{'_'.join(problem_keys[:3])}"

        self.knowledge_graph["solution_patterns"][pattern_key] = {
            "doc_id": doc_id,
            "solution_text": solution_text[:500],
            "score": score,
            "problem_keys": problem_keys,
            "solution_concepts": solution_concepts,
            "uses": 1
        }

    def add_solution_pattern_entry(
        self,
        pattern_hash: str,
        problem_text: str,
        solution_text: str,
        outcome: str
    ):
        """Add or update a solution pattern entry."""
        if "solution_patterns" not in self.knowledge_graph:
            self.knowledge_graph["solution_patterns"] = {}

        if pattern_hash not in self.knowledge_graph["solution_patterns"]:
            self.knowledge_graph["solution_patterns"][pattern_hash] = {
                "problem_text": problem_text[:200],
                "solution_text": solution_text[:500],
                "successes": 0,
                "failures": 0,
                "success_rate": 0.5
            }

        entry = self.knowledge_graph["solution_patterns"][pattern_hash]

        if outcome == "worked":
            entry["successes"] = entry.get("successes", 0) + 1
        elif outcome == "failed":
            entry["failures"] = entry.get("failures", 0) + 1

        total = entry.get("successes", 0) + entry.get("failures", 0)
        if total > 0:
            entry["success_rate"] = entry.get("successes", 0) / total


    def add_entities_from_text(
        self,
        text: str,
        doc_id: str,
        collection: str,
        quality_score: Optional[float] = None
    ) -> List[str]:
        """
        Add entities from text to content graph.
        Wrapper around ContentGraph.add_entities_from_text.
        """
        return self.content_graph.add_entities_from_text(
            text=text,
            doc_id=doc_id,
            collection=collection,
            extract_concepts_fn=self.extract_concepts,
            quality_score=quality_score
        )

    def remove_entity_mention(self, doc_id: str):
        """Remove a document's entity mentions from content graph."""
        self.content_graph.remove_entity_mention(doc_id)

    # =========================================================================
    # Shutdown
    # =========================================================================

    async def cleanup(self):
        """Clean shutdown - save pending changes."""
        async with self._kg_save_lock:
            if self._kg_save_task and not self._kg_save_task.done():
                self._kg_save_task.cancel()
                try:
                    await self._kg_save_task
                except asyncio.CancelledError:
                    pass

        # Final save if pending
        if self._kg_save_pending:
            await self._save_kg()

        logger.info("KnowledgeGraphService cleaned up")
