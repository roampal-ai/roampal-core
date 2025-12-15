"""
Routing Service - Intelligent collection routing using learned KG patterns.

Extracted from UnifiedMemorySystem as part of refactoring.

Responsibilities:
- Query preprocessing (acronym expansion)
- Intelligent routing based on learned patterns
- Tier score calculation for collections
- Tier recommendations for insights
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import MemoryConfig
from .knowledge_graph_service import KnowledgeGraphService

logger = logging.getLogger(__name__)


# Collection names for routing
ALL_COLLECTIONS = ["working", "patterns", "history", "books", "memory_bank"]


class RoutingService:
    """
    Intelligent collection routing using learned KG patterns.

    Implements architecture.md specification with learning phases:
    - Phase 1 (Exploration): total_score < 0.5 -> all 5 collections
    - Phase 2 (Medium Confidence): 0.5 <= total_score < 2.0 -> top 2-3 collections
    - Phase 3 (High Confidence): total_score >= 2.0 -> top 1-2 collections
    """

    # Acronym dictionary for query expansion
    ACRONYM_DICT = {
        # Technology
        "api": "application programming interface",
        "apis": "application programming interfaces",
        "sdk": "software development kit",
        "sdks": "software development kits",
        "ui": "user interface",
        "ux": "user experience",
        "db": "database",
        "sql": "structured query language",
        "html": "hypertext markup language",
        "css": "cascading style sheets",
        "js": "javascript",
        "ts": "typescript",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "llm": "large language model",
        "nlp": "natural language processing",
        "gpu": "graphics processing unit",
        "cpu": "central processing unit",
        "ram": "random access memory",
        "ssd": "solid state drive",
        "hdd": "hard disk drive",
        "os": "operating system",
        "ide": "integrated development environment",
        "cli": "command line interface",
        "gui": "graphical user interface",
        "ci": "continuous integration",
        "cd": "continuous deployment",
        "devops": "development operations",
        "qa": "quality assurance",
        "uat": "user acceptance testing",
        "mvp": "minimum viable product",
        "poc": "proof of concept",
        "saas": "software as a service",
        "paas": "platform as a service",
        "iaas": "infrastructure as a service",
        "iot": "internet of things",
        "vpn": "virtual private network",
        "dns": "domain name system",
        "http": "hypertext transfer protocol",
        "https": "hypertext transfer protocol secure",
        "ftp": "file transfer protocol",
        "ssh": "secure shell",
        "ssl": "secure sockets layer",
        "tls": "transport layer security",
        "jwt": "json web token",
        "oauth": "open authorization",
        "rest": "representational state transfer",
        "crud": "create read update delete",
        "orm": "object relational mapping",
        "json": "javascript object notation",
        "xml": "extensible markup language",
        "yaml": "yaml ain't markup language",
        "csv": "comma separated values",
        "pdf": "portable document format",
        "svg": "scalable vector graphics",
        "png": "portable network graphics",
        "jpg": "joint photographic experts group",
        "gif": "graphics interchange format",
        "aws": "amazon web services",
        "gcp": "google cloud platform",
        "vm": "virtual machine",
        "k8s": "kubernetes",
        "npm": "node package manager",
        "pip": "pip installs packages",
        "git": "git",  # Keep as is, well known
        "pr": "pull request",
        "mr": "merge request",
        "env": "environment",
        "prod": "production",
        "dev": "development",
        "repo": "repository",
        "config": "configuration",
        "auth": "authentication",
        "async": "asynchronous",
        "sync": "synchronous",

        # Locations
        "nyc": "new york city",
        "la": "los angeles",
        "sf": "san francisco",
        "dc": "washington dc",
        "uk": "united kingdom",
        "usa": "united states of america",
        "us": "united states",

        # Organizations
        "nasa": "national aeronautics and space administration",
        "fbi": "federal bureau of investigation",
        "cia": "central intelligence agency",
        "fda": "food and drug administration",
        "cdc": "centers for disease control",
        "mit": "massachusetts institute of technology",
        "ucla": "university of california los angeles",
        "stanford": "stanford university",
        "harvard": "harvard university",

        # Business
        "ceo": "chief executive officer",
        "cto": "chief technology officer",
        "cfo": "chief financial officer",
        "coo": "chief operating officer",
        "vp": "vice president",
        "hr": "human resources",
        "roi": "return on investment",
        "kpi": "key performance indicator",
        "okr": "objectives and key results",
        "b2b": "business to business",
        "b2c": "business to consumer",
        "erp": "enterprise resource planning",
        "crm": "customer relationship management",
        "eod": "end of day",
        "asap": "as soon as possible",
        "eta": "estimated time of arrival",
        "fyi": "for your information",
        "tbd": "to be determined",
        "pov": "point of view",
        "wfh": "work from home",
    }

    # Build reverse mapping (expansion -> acronym) for bidirectional matching
    EXPANSION_TO_ACRONYM = {v.lower(): k.lower() for k, v in ACRONYM_DICT.items()}

    def __init__(
        self,
        kg_service: KnowledgeGraphService,
        config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize RoutingService.

        Args:
            kg_service: KnowledgeGraphService for concept extraction and KG access
            config: Optional MemoryConfig for thresholds
        """
        self.kg_service = kg_service
        self.config = config or MemoryConfig()

    @property
    def knowledge_graph(self) -> Dict[str, Any]:
        """Access the knowledge graph from KG service."""
        return self.kg_service.knowledge_graph

    # =========================================================================
    # Query Preprocessing
    # =========================================================================

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess search query for better retrieval:
        1. Expand acronyms (API -> "API application programming interface")
        2. Normalize whitespace

        This improves recall when user queries with acronyms but facts stored with full names.

        Args:
            query: Original search query

        Returns:
            Enhanced query with expanded acronyms
        """
        if not query:
            return query

        # Normalize whitespace
        query = " ".join(query.split())

        # Find and expand acronyms in query
        words = query.split()
        expansions_to_add = []

        for word in words:
            word_lower = word.lower().strip(".,!?;:'\"()")

            # Check if word is a known acronym
            if word_lower in self.ACRONYM_DICT:
                expansion = self.ACRONYM_DICT[word_lower]
                # Add expansion if not already in query
                if expansion.lower() not in query.lower():
                    expansions_to_add.append(expansion)
                    logger.debug(f"[QUERY_PREPROCESS] Expanded '{word}' -> '{expansion}'")

        # Append expansions to query (keeps original + adds expanded versions)
        if expansions_to_add:
            enhanced_query = query + " " + " ".join(expansions_to_add)
            logger.debug(f"[QUERY_PREPROCESS] Enhanced query: '{query}' -> '{enhanced_query}'")
            return enhanced_query

        return query

    # =========================================================================
    # Tier Score Calculation
    # =========================================================================

    def calculate_tier_scores(self, concepts: List[str]) -> Dict[str, float]:
        """
        Calculate tier scores for each collection based on learned patterns.
        Implements architecture.md tier scoring formula:

        tier_score = success_rate * confidence
        where:
          success_rate = successes / (successes + failures)
          confidence = min(total_uses / 10, 1.0)

        Returns dict mapping collection_name -> total_score
        """
        collection_scores = {
            "working": 0.0,
            "patterns": 0.0,
            "history": 0.0,
            "books": 0.0,
            "memory_bank": 0.0
        }

        # Aggregate scores across all concepts
        for concept in concepts:
            if concept in self.knowledge_graph.get("routing_patterns", {}):
                pattern_data = self.knowledge_graph["routing_patterns"][concept]
                collections_used = pattern_data.get("collections_used", {})

                for collection, stats in collections_used.items():
                    if collection not in collection_scores:
                        continue  # Skip unknown collections

                    successes = stats.get("successes", 0)
                    failures = stats.get("failures", 0)
                    partials = stats.get("partials", 0)
                    total_uses = successes + failures + partials

                    # Calculate success_rate (exclude partials from denominator)
                    if successes + failures > 0:
                        success_rate = successes / (successes + failures)
                    else:
                        success_rate = 0.5  # Neutral for no confirmed outcomes

                    # Calculate confidence (reaches 1.0 after 10 uses)
                    confidence = min(total_uses / 10.0, 1.0)

                    # Tier score
                    tier_score = success_rate * confidence

                    # Add to collection's total score
                    collection_scores[collection] += tier_score

        return collection_scores

    # =========================================================================
    # Query Routing
    # =========================================================================

    def route_query(self, query: str) -> List[str]:
        """
        Intelligent routing using learned KG patterns.
        Implements architecture.md specification with learning phases:

        Phase 1 (Exploration): total_score < 0.5 -> search all 5 collections
        Phase 2 (Medium Confidence): 0.5 <= total_score < 2.0 -> search top 2-3 collections
        Phase 3 (High Confidence): total_score >= 2.0 -> search top 1-2 collections

        Returns list of collection names to search.
        """
        # Extract concepts from query using KG service
        concepts = self.kg_service.extract_concepts(query)

        if not concepts:
            logger.debug("[Routing] No concepts extracted, searching all collections")
            return ALL_COLLECTIONS.copy()

        # Calculate tier scores for each collection
        collection_scores = self.calculate_tier_scores(concepts)

        # Calculate total score (sum of all collection scores)
        total_score = sum(collection_scores.values())

        # Sort collections by score (highest first)
        sorted_collections = sorted(
            collection_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Apply routing thresholds
        if total_score < 0.5:
            # EXPLORATION PHASE: No learned patterns yet, search everything
            selected = ALL_COLLECTIONS.copy()
            logger.info(f"[Routing] Exploration phase (score={total_score:.2f}): searching all collections")

        elif total_score < 2.0:
            # MEDIUM CONFIDENCE: Search top 2-3 collections
            # Take top collections with score > 0.1, up to 3
            selected = [
                coll for coll, score in sorted_collections[:3]
                if score > 0.1
            ]
            if not selected:
                selected = [sorted_collections[0][0]]  # At least take top 1
            logger.info(f"[Routing] Medium confidence (score={total_score:.2f}): searching {selected}")

        else:
            # HIGH CONFIDENCE: Search top 1-2 collections
            # Take top collections with score > 0.5, up to 2
            selected = [
                coll for coll, score in sorted_collections[:2]
                if score > 0.5
            ]
            if not selected:
                selected = [sorted_collections[0][0]]  # At least take top 1
            logger.info(f"[Routing] High confidence (score={total_score:.2f}): searching {selected}")

        # Log concept extraction and scores for debugging
        logger.debug(f"[Routing] Concepts: {concepts[:5]}...")
        logger.debug(f"[Routing] Scores: {dict(sorted_collections[:3])}")

        # Track usage for KG visualization (increment 'total' for used patterns)
        # This makes MCP-searched patterns visible in UI even without explicit outcome feedback
        self._track_routing_usage(concepts, selected)

        return selected

    def _track_routing_usage(self, concepts: List[str], selected_collections: List[str]):
        """Track routing usage in KG for visualization."""
        for concept in concepts:
            if concept in self.knowledge_graph.get("routing_patterns", {}):
                pattern = self.knowledge_graph["routing_patterns"][concept]
                collections_used = pattern.get("collections_used", {})

                # Increment total for each collection that was selected for search
                for collection in selected_collections:
                    if collection in collections_used:
                        collections_used[collection]["total"] = collections_used[collection].get("total", 0) + 1
                    else:
                        # Initialize if this collection not tracked yet
                        collections_used[collection] = {
                            "successes": 0,
                            "failures": 0,
                            "partials": 0,
                            "total": 1
                        }

                # Update last_used timestamp
                pattern["last_used"] = datetime.now().isoformat()

    # =========================================================================
    # Tier Recommendations (for get_context_insights)
    # =========================================================================

    def get_tier_recommendations(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Query Routing KG for best collections given concepts (v0.2.6 Directive Insights).

        Uses the same logic as route_query but returns recommendations
        for get_context_insights output.

        Args:
            concepts: List of extracted concepts from user query

        Returns:
            Dict with top_collections, match_count, confidence_level
        """
        if not concepts:
            return {
                "top_collections": ALL_COLLECTIONS.copy(),
                "match_count": 0,
                "confidence_level": "exploration"
            }

        # Calculate tier scores
        collection_scores = self.calculate_tier_scores(concepts)
        total_score = sum(collection_scores.values())

        # Sort by score
        sorted_collections = sorted(
            collection_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Count matched patterns
        match_count = 0
        for concept in concepts:
            if concept in self.knowledge_graph.get("routing_patterns", {}):
                match_count += 1

        # Determine confidence level and top collections
        if total_score < 0.5:
            confidence_level = "exploration"
            top_collections = ALL_COLLECTIONS.copy()
        elif total_score < 2.0:
            confidence_level = "medium"
            top_collections = [coll for coll, score in sorted_collections[:3] if score > 0.1]
            if not top_collections:
                top_collections = [sorted_collections[0][0]]
        else:
            confidence_level = "high"
            top_collections = [coll for coll, score in sorted_collections[:2] if score > 0.5]
            if not top_collections:
                top_collections = [sorted_collections[0][0]]

        return {
            "top_collections": top_collections,
            "match_count": match_count,
            "confidence_level": confidence_level,
            "total_score": total_score,
            "scores": dict(sorted_collections[:3])  # Top 3 for visibility
        }
