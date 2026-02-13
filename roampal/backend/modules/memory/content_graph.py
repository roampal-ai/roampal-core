"""
Content Knowledge Graph - Indexes entity relationships from memory_bank content.

Part of dual KG system:
- Routing KG: Query patterns → collection routing decisions
- Content KG: Memory content → entity relationships (THIS FILE)

Author: Roampal AI
Date: 2025-11-06
Version: v0.2.0
"""

import json
import logging
from typing import Dict, List, Set, Tuple, Any, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger(__name__)


class ContentGraph:
    """
    Content-based knowledge graph that indexes entity relationships from memory_bank content.

    Key differences from routing KG:
    - Data source: memory_bank text content (not queries)
    - Purpose: Entity relationship mapping (not search routing)
    - Storage: content_graph.json (separate from knowledge_graph.json)

    Graph structure:
    {
        "entities": {
            "alice": {
                "mentions": 12,
                "collections": {"memory_bank": 12},
                "documents": ["doc_id_1", "doc_id_2", ...],
                "first_seen": "2025-11-06T10:30:00",
                "last_seen": "2025-11-06T14:45:00"
            }
        },
        "relationships": {
            "alice__acme_corp": {
                "entities": ["alice", "acme_corp"],
                "strength": 8.0,
                "co_occurrences": 8,
                "documents": ["doc_id_1", "doc_id_2", ...],
                "first_seen": "2025-11-06T10:30:00",
                "last_seen": "2025-11-06T14:45:00"
            }
        },
        "metadata": {
            "version": "0.2.0",
            "created": "2025-11-06T10:00:00",
            "last_updated": "2025-11-06T14:45:00",
            "total_documents": 29,
            "total_entities": 47,
            "total_relationships": 123
        }
    }
    """

    def __init__(self):
        """Initialize empty content graph."""
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, Dict[str, Any]] = {}
        self.entity_metadata: Dict[str, Dict[str, Any]] = {}
        self.metadata = {
            "version": "0.2.0",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_documents": 0,
            "total_entities": 0,
            "total_relationships": 0
        }

        # Document tracking for relationship building
        self._doc_entities: Dict[str, Set[str]] = defaultdict(set)

        logger.info("ContentGraph initialized (empty)")

    def add_entities_from_text(
        self,
        text: str,
        doc_id: str,
        collection: str,
        extract_concepts_fn: Callable[[str], List[str]],
        quality_score: Optional[float] = None
    ) -> List[str]:
        """
        Extract entities from text and index them in the content graph.

        Args:
            text: Raw text content to analyze
            doc_id: Unique document identifier
            collection: Collection name (e.g., "memory_bank")
            extract_concepts_fn: Function to extract concepts (reuses _extract_concepts from UnifiedMemorySystem)
            quality_score: Optional quality score (importance × confidence) for this document's entities

        Returns:
            List of extracted entity names

        Example:
            >>> cg = ContentGraph()
            >>> entities = cg.add_entities_from_text(
            ...     text="User prefers Docker Compose for local development environments",
            ...     doc_id="mem_001",
            ...     collection="memory_bank",
            ...     extract_concepts_fn=memory_system._extract_concepts
            ... )
            >>> entities
            ['docker', 'compose', 'local', 'development', 'environments']
        """
        # Extract concepts using same logic as routing KG (consistency)
        concepts = extract_concepts_fn(text)

        # Filter to meaningful entities (exclude very short/generic terms)
        entities = [c for c in concepts if len(c) >= 3 and not c.isdigit()]

        if not entities:
            logger.debug(f"No entities extracted from doc {doc_id}")
            return []

        now = datetime.now().isoformat()

        # Index each entity
        for entity in entities:
            if entity not in self.entities:
                self.entities[entity] = {
                    "mentions": 0,
                    "collections": defaultdict(int),
                    "documents": [],
                    "first_seen": now,
                    "last_seen": now,
                    "total_quality": 0.0,
                    "avg_quality": 0.0
                }

            # Update entity metadata
            self.entities[entity]["mentions"] += 1
            self.entities[entity]["collections"][collection] += 1
            self.entities[entity]["last_seen"] = now

            if doc_id not in self.entities[entity]["documents"]:
                self.entities[entity]["documents"].append(doc_id)

            # Update quality scores if provided
            if quality_score is not None:
                self.entities[entity]["total_quality"] += quality_score
                self.entities[entity]["avg_quality"] = (
                    self.entities[entity]["total_quality"] / self.entities[entity]["mentions"]
                )

        # Track entities in this document for relationship building
        self._doc_entities[doc_id].update(entities)

        # Build relationships (co-occurrence within same document)
        self._build_relationships(entities, doc_id, now)

        # Update metadata
        self.metadata["last_updated"] = now
        self.metadata["total_entities"] = len(self.entities)
        self.metadata["total_relationships"] = len(self.relationships)

        logger.info(f"Indexed {len(entities)} entities from doc {doc_id}: {entities[:5]}{'...' if len(entities) > 5 else ''}")

        return entities

    def _build_relationships(self, entities: List[str], doc_id: str, timestamp: str):
        """
        Build entity relationships based on co-occurrence in the same document.

        Args:
            entities: List of entities found in document
            doc_id: Document identifier
            timestamp: ISO timestamp for tracking

        Example:
            If doc contains ["alice", "acme_corp", "operations_manager"],
            creates relationships:
            - alice <-> acme_corp
            - alice <-> operations_manager
            - acme_corp <-> operations_manager
        """
        # Create pairwise relationships (undirected)
        unique_entities = list(set(entities))

        for i in range(len(unique_entities)):
            for j in range(i + 1, len(unique_entities)):
                entity_a = unique_entities[i]
                entity_b = unique_entities[j]

                # Create canonical relationship ID (sorted to ensure consistency)
                rel_id = "__".join(sorted([entity_a, entity_b]))

                if rel_id not in self.relationships:
                    self.relationships[rel_id] = {
                        "entities": sorted([entity_a, entity_b]),
                        "strength": 0.0,
                        "co_occurrences": 0,
                        "documents": [],
                        "first_seen": timestamp,
                        "last_seen": timestamp
                    }

                # Update relationship metadata
                self.relationships[rel_id]["co_occurrences"] += 1
                self.relationships[rel_id]["last_seen"] = timestamp

                if doc_id not in self.relationships[rel_id]["documents"]:
                    self.relationships[rel_id]["documents"].append(doc_id)

                # Calculate relationship strength (co-occurrence count with decay)
                # Simple formula: log2(co_occurrences + 1) for diminishing returns
                co_occur = self.relationships[rel_id]["co_occurrences"]
                self.relationships[rel_id]["strength"] = round(
                    (co_occur ** 0.5) * 2.0,  # sqrt for diminishing returns, * 2 for scaling
                    2
                )

    def remove_entity_mention(self, doc_id: str):
        """
        Remove a document's entity mentions (when memory archived/deleted).

        Args:
            doc_id: Document identifier to remove

        Notes:
            - Decrements mention counts
            - Removes doc from entity.documents
            - Removes relationships if no more co-occurrences
            - Does NOT delete entities (preserves historical context)
        """
        if doc_id not in self._doc_entities:
            logger.debug(f"Doc {doc_id} not tracked in content graph")
            return

        entities_in_doc = self._doc_entities[doc_id]
        now = datetime.now().isoformat()

        # Decrement entity mentions
        for entity in entities_in_doc:
            if entity in self.entities:
                self.entities[entity]["mentions"] -= 1

                if doc_id in self.entities[entity]["documents"]:
                    self.entities[entity]["documents"].remove(doc_id)

                # Keep entity metadata even if mentions reach 0 (historical context)

        # Remove relationships involving these entities
        rels_to_remove = []
        for rel_id, rel_data in self.relationships.items():
            if doc_id in rel_data["documents"]:
                rel_data["documents"].remove(doc_id)
                rel_data["co_occurrences"] -= 1

                # Recalculate strength
                if rel_data["co_occurrences"] > 0:
                    co_occur = rel_data["co_occurrences"]
                    rel_data["strength"] = round((co_occur ** 0.5) * 2.0, 2)
                else:
                    # Mark for deletion if no more co-occurrences
                    rels_to_remove.append(rel_id)

        # Clean up relationships with 0 co-occurrences
        for rel_id in rels_to_remove:
            del self.relationships[rel_id]

        # Remove from tracking
        del self._doc_entities[doc_id]

        # Update metadata
        self.metadata["last_updated"] = now
        self.metadata["total_relationships"] = len(self.relationships)

        logger.info(f"Removed doc {doc_id} from content graph ({len(entities_in_doc)} entities, {len(rels_to_remove)} relationships)")

    def get_entity_relationships(self, entity: str, min_strength: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get all relationships for a specific entity.

        Args:
            entity: Entity name to query
            min_strength: Minimum relationship strength threshold

        Returns:
            List of relationships with connected entities and metadata

        Example:
            >>> cg.get_entity_relationships("alice", min_strength=2.0)
            [
                {
                    "related_entity": "acme_corp",
                    "strength": 5.66,
                    "co_occurrences": 8,
                    "documents": ["mem_001", "mem_003", ...],
                    "first_seen": "2025-11-06T10:30:00",
                    "last_seen": "2025-11-06T14:45:00"
                },
                ...
            ]
        """
        if entity not in self.entities:
            logger.debug(f"Entity '{entity}' not found in content graph")
            return []

        relationships = []

        for rel_id, rel_data in self.relationships.items():
            if entity in rel_data["entities"] and rel_data["strength"] >= min_strength:
                # Find the other entity in the relationship
                related_entity = [e for e in rel_data["entities"] if e != entity][0]

                relationships.append({
                    "related_entity": related_entity,
                    "strength": rel_data["strength"],
                    "co_occurrences": rel_data["co_occurrences"],
                    "documents": rel_data["documents"],
                    "first_seen": rel_data["first_seen"],
                    "last_seen": rel_data["last_seen"]
                })

        # Sort by strength (descending)
        relationships.sort(key=lambda x: x["strength"], reverse=True)

        logger.debug(f"Found {len(relationships)} relationships for entity '{entity}'")
        return relationships

    def find_path(
        self,
        from_entity: str,
        to_entity: str,
        max_depth: int = 3
    ) -> Optional[List[str]]:
        """
        Find shortest path between two entities using BFS.

        Args:
            from_entity: Starting entity
            to_entity: Target entity
            max_depth: Maximum path length to search

        Returns:
            List of entities forming path, or None if no path found

        Example:
            >>> cg.find_path("alice", "acme_corp", max_depth=3)
            ["alice", "acme_corp"]  # Direct connection

            >>> cg.find_path("alice", "solar_energy", max_depth=3)
            ["alice", "acme_corp", "solar_energy"]  # 2-hop connection
        """
        if from_entity not in self.entities or to_entity not in self.entities:
            logger.debug(f"Entity not found: from='{from_entity}' to='{to_entity}'")
            return None

        if from_entity == to_entity:
            return [from_entity]

        # BFS to find shortest path
        queue = deque([(from_entity, [from_entity])])
        visited = {from_entity}

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            # Get neighbors
            for rel_id, rel_data in self.relationships.items():
                if current in rel_data["entities"]:
                    # Find connected entity
                    neighbor = [e for e in rel_data["entities"] if e != current][0]

                    if neighbor == to_entity:
                        # Found target
                        result_path = path + [neighbor]
                        logger.info(f"Found path: {' -> '.join(result_path)}")
                        return result_path

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

        logger.debug(f"No path found between '{from_entity}' and '{to_entity}' (max_depth={max_depth})")
        return None

    def get_all_entities(self, min_mentions: int = 1) -> List[Dict[str, Any]]:
        """
        Get all entities with metadata.

        Args:
            min_mentions: Minimum mention count threshold

        Returns:
            List of entities with full metadata
        """
        entities = [
            {
                "entity": name,  # Changed from "name" to "entity" for consistency with routing KG
                **data
            }
            for name, data in self.entities.items()
            if data["mentions"] >= min_mentions
        ]

        # Sort by avg_quality (descending), fallback to mentions for backward compatibility
        entities.sort(key=lambda x: x.get("avg_quality", 0.0), reverse=True)

        return entities

    def get_stats(self) -> Dict[str, Any]:
        """
        Get content graph statistics.

        Returns:
            Dictionary with graph metrics
        """
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "total_documents": len(self._doc_entities),
            "avg_mentions_per_entity": round(
                sum(e["mentions"] for e in self.entities.values()) / len(self.entities), 2
            ) if self.entities else 0.0,
            "avg_relationships_per_entity": round(
                (len(self.relationships) * 2) / len(self.entities), 2
            ) if self.entities else 0.0,
            "strongest_relationship": max(
                (r for r in self.relationships.values()),
                key=lambda x: x["strength"],
                default=None
            ),
            "most_mentioned_entity": max(
                self.entities.items(),
                key=lambda x: x[1]["mentions"],
                default=(None, None)
            )[0],
            "metadata": self.metadata
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize content graph to dictionary.

        Returns:
            Complete graph data as dictionary
        """
        # Convert defaultdicts to regular dicts for JSON serialization
        serializable_entities = {}
        for name, data in self.entities.items():
            entity_copy = data.copy()
            if isinstance(entity_copy.get("collections"), defaultdict):
                entity_copy["collections"] = dict(entity_copy["collections"])
            serializable_entities[name] = entity_copy

        return {
            "entities": serializable_entities,
            "relationships": self.relationships,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentGraph":
        """
        Deserialize content graph from dictionary.

        Args:
            data: Graph data dictionary

        Returns:
            ContentGraph instance
        """
        graph = cls()

        # Load entities
        graph.entities = {}
        for name, entity_data in data.get("entities", {}).items():
            entity_copy = entity_data.copy()
            # Convert collections dict back to defaultdict
            if "collections" in entity_copy:
                entity_copy["collections"] = defaultdict(int, entity_copy["collections"])
            graph.entities[name] = entity_copy

        # Load relationships
        graph.relationships = data.get("relationships", {})

        # Load metadata
        graph.metadata = data.get("metadata", graph.metadata)

        # Rebuild doc_entities tracking
        for entity_name, entity_data in graph.entities.items():
            for doc_id in entity_data.get("documents", []):
                graph._doc_entities[doc_id].add(entity_name)

        logger.info(
            f"ContentGraph loaded: {len(graph.entities)} entities, "
            f"{len(graph.relationships)} relationships, "
            f"{len(graph._doc_entities)} documents"
        )

        return graph

    def save_to_file(self, filepath: str):
        """
        Save content graph to JSON file.

        Args:
            filepath: Path to save file
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"ContentGraph saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save ContentGraph to {filepath}: {e}")
            raise

    @classmethod
    def load_from_file(cls, filepath: str) -> "ContentGraph":
        """
        Load content graph from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            ContentGraph instance
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            graph = cls.from_dict(data)
            logger.info(f"ContentGraph loaded from {filepath}")
            return graph
        except FileNotFoundError:
            logger.warning(f"ContentGraph file not found: {filepath}, creating new graph")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load ContentGraph from {filepath}: {e}")
            raise
