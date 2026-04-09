"""
Routing Service — Query preprocessing and collection routing.

v0.4.5: Simplified. KG-based learned routing removed.
Tag-routed retrieval (in SearchService) handles precision now.
This service handles:
- Query preprocessing (acronym expansion)
- Collection routing (always all 5 — tag routing handles scoping)
"""

import logging
import re
from typing import List, Optional

from .config import MemoryConfig

logger = logging.getLogger(__name__)


# Collection names for routing
ALL_COLLECTIONS = ["working", "patterns", "history", "books", "memory_bank"]


class RoutingService:
    """
    Query preprocessing and collection routing.

    v0.4.5: Always routes to all 5 collections. Tag-routed retrieval
    in SearchService handles precision via noun tag overlap scoring.
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

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize RoutingService.

        Args:
            config: Optional MemoryConfig for thresholds
        """
        self.config = config or MemoryConfig()

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
    # Query Routing
    # =========================================================================

    def route_query(self, query: str) -> List[str]:
        """
        Route query to collections.

        v0.4.5: Always returns all 5 collections. Tag-routed retrieval
        in SearchService handles precision via noun tag overlap scoring.
        """
        return ALL_COLLECTIONS.copy()
