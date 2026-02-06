"""
Mock utilities for deterministic testing without LLM dependencies.
"""

import hashlib
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta


class MockEmbeddingService:
    """Generate deterministic embeddings from text hash (no actual model)"""

    def __init__(self):
        self.dimension = 768

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate deterministic 768d vector from text hash.
        Same text always produces same embedding.
        """
        # Use SHA-256 hash of text to seed random generation
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Convert hash bytes to floats
        embedding = []
        for i in range(self.dimension):
            # Use different portions of hash for each dimension
            byte_idx = (i * 2) % len(hash_bytes)
            value = (hash_bytes[byte_idx] + hash_bytes[byte_idx + 1 if byte_idx + 1 < len(hash_bytes) else 0]) / 512.0
            # Normalize to [-1, 1] range
            embedding.append(value * 2.0 - 1.0)

        return embedding

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts."""
        return [await self.embed_text(text) for text in texts]

    async def prewarm(self):
        """No-op prewarm for mock."""
        pass

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a ** 2 for a in vec1) ** 0.5
        mag2 = sum(b ** 2 for b in vec2) ** 0.5
        return dot_product / (mag1 * mag2) if mag1 and mag2 else 0.0


class MockLLMService:
    """Mock LLM for contextual prefix generation"""

    async def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Generate deterministic contextual prefix from prompt.
        Rule-based, no actual LLM.
        """
        # Extract collection and content from prompt
        if "memory_bank" in prompt.lower():
            prefix = "User memory"
        elif "patterns" in prompt.lower():
            prefix = "Proven solution pattern"
        elif "working" in prompt.lower():
            prefix = "Recent conversation"
        elif "history" in prompt.lower():
            prefix = "Past conversation"
        elif "books" in prompt.lower():
            prefix = "Reference material"
        else:
            prefix = "Memory chunk"

        # Check for high importance
        if "high importance" in prompt.lower():
            prefix += ", High importance"

        return prefix


class MockTimeManager:
    """Mock time for deterministic decay testing"""

    def __init__(self):
        self._current_time = datetime.now()
        self._original_now = datetime.now

    def set_time(self, dt: datetime):
        """Set current mocked time"""
        self._current_time = dt

    def advance_days(self, days: int):
        """Advance time by N days"""
        self._current_time += timedelta(days=days)

    def advance_hours(self, hours: int):
        """Advance time by N hours"""
        self._current_time += timedelta(hours=hours)

    def now(self) -> datetime:
        """Get current mocked time"""
        return self._current_time

    def reset(self):
        """Reset to real time"""
        self._current_time = datetime.now()


def mock_extract_concepts(text: str) -> List[str]:
    """
    Extract concepts from text (rule-based, no LLM).
    Simple tokenization + filtering.
    """
    # Lowercase and split
    words = text.lower().split()

    # Filter stopwords
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had"}
    concepts = [w.strip(".,!?") for w in words if w not in stopwords and len(w) > 2]

    # Add bigrams for compound concepts
    bigrams = []
    for i in range(len(concepts) - 1):
        bigram = f"{concepts[i]}_{concepts[i+1]}"
        bigrams.append(bigram)

    return list(set(concepts + bigrams))


def calculate_similarity(text1: str, text2: str, embedding_service: MockEmbeddingService) -> float:
    """
    Calculate similarity between two texts using mock embeddings.
    Returns value between 0.0 (different) and 1.0 (identical).
    """
    import asyncio

    # Get embeddings
    loop = asyncio.get_event_loop()
    emb1 = loop.run_until_complete(embedding_service.embed_text(text1))
    emb2 = loop.run_until_complete(embedding_service.embed_text(text2))

    # Calculate cosine similarity
    similarity = embedding_service.cosine_similarity(emb1, emb2)

    # Convert from [-1, 1] to [0, 1]
    return (similarity + 1.0) / 2.0


def mock_context_classifier(messages: List[str]) -> str:
    """
    Classify conversation context from messages (rule-based, no LLM).
    Returns context like "coding", "fitness", "general", etc.
    """
    combined = " ".join(messages).lower()

    # Rule-based classification
    if any(word in combined for word in ["docker", "kubernetes", "api", "database", "code", "python", "react"]):
        return "coding"
    elif any(word in combined for word in ["workout", "exercise", "fitness", "gym", "nutrition"]):
        return "fitness"
    elif any(word in combined for word in ["finance", "investment", "budget", "money", "stock"]):
        return "finance"
    elif any(word in combined for word in ["story", "character", "plot", "writing", "novel"]):
        return "creative_writing"
    else:
        return "general"


def create_test_metadata(collection: str, **kwargs) -> Dict[str, Any]:
    """Create standard test metadata for a collection"""
    base_metadata = {
        "collection": collection,
        "timestamp": datetime.now().isoformat(),
        "test": True
    }

    # Add collection-specific defaults
    if collection in ["working", "history", "patterns"]:
        base_metadata["score"] = kwargs.get("score", 0.5)
        base_metadata["uses"] = kwargs.get("uses", 0)
        base_metadata["last_outcome"] = kwargs.get("last_outcome", "unknown")

    if collection == "memory_bank":
        base_metadata["importance"] = kwargs.get("importance", 0.7)
        base_metadata["confidence"] = kwargs.get("confidence", 0.7)
        base_metadata["tags"] = kwargs.get("tags", [])
        base_metadata["status"] = kwargs.get("status", "active")

    if collection == "books":
        base_metadata["title"] = kwargs.get("title", "Test Book")
        base_metadata["author"] = kwargs.get("author", "Test Author")

    # Override with any additional kwargs
    base_metadata.update(kwargs)

    return base_metadata


def verify_doc_id_format(doc_id: str, expected_collection: str) -> bool:
    """
    Verify doc_id follows expected format:
    - v0.2.7+: {collection}_{hex8} (e.g., books_ca38b775)
    - Legacy: {collection}_{uuid}_{timestamp}
    """
    # Check collection prefix
    if not doc_id.startswith(f"{expected_collection}_"):
        return False

    # Remove collection prefix to get remaining parts
    remaining = doc_id[len(expected_collection) + 1:]
    parts = remaining.split("_")

    if len(parts) < 1:
        return False

    # First part should be hex (8 chars for v0.2.7+, or UUID for legacy)
    try:
        int(parts[0], 16)
    except ValueError:
        return False

    # v0.2.7+ format: just hex8, timestamp no longer required
    return True


def verify_embedding_dimension(embedding: List[float], expected_dim: int = 768) -> bool:
    """Verify embedding has correct dimension"""
    return len(embedding) == expected_dim and all(isinstance(v, float) for v in embedding)


def verify_metadata_persistence(original: Dict[str, Any], retrieved: Dict[str, Any]) -> bool:
    """Verify metadata persisted correctly (ignoring auto-added fields)"""
    # Fields that are auto-added and can be ignored
    ignore_fields = {"id", "distance", "embedding", "text", "content"}

    for key, value in original.items():
        if key in ignore_fields:
            continue

        if key not in retrieved:
            print(f"Missing field: {key}")
            return False

        if retrieved[key] != value:
            # Handle JSON serialization differences
            if isinstance(value, (list, dict)):
                if json.dumps(value, sort_keys=True) != json.dumps(retrieved[key], sort_keys=True):
                    print(f"Mismatch in {key}: {value} != {retrieved[key]}")
                    return False
            else:
                print(f"Mismatch in {key}: {value} != {retrieved[key]}")
                return False

    return True


def verify_kg_structure(kg: Dict[str, Any], kg_type: str) -> bool:
    """Verify knowledge graph has expected structure"""
    if kg_type == "routing":
        required_keys = ["routing_patterns", "success_rates", "failure_patterns", "problem_categories"]
        return all(key in kg for key in required_keys)

    elif kg_type == "content":
        required_keys = ["entities", "relationships", "metadata"]
        if not all(key in kg for key in required_keys):
            return False

        # Verify entity structure
        for entity_name, entity_data in kg["entities"].items():
            required_entity_keys = ["mentions", "collections", "documents"]
            if not all(key in entity_data for key in required_entity_keys):
                return False

        # Verify relationship structure
        for rel_id, rel_data in kg["relationships"].items():
            required_rel_keys = ["entities", "strength", "co_occurrences"]
            if not all(key in rel_data for key in required_rel_keys):
                return False

        return True

    elif kg_type == "action_effectiveness":
        # Should be in routing KG
        if "context_action_effectiveness" not in kg:
            return False

        # Verify structure of each action pattern
        for key, stats in kg["context_action_effectiveness"].items():
            required_stats = ["success_count", "failure_count", "success_rate", "total_uses"]
            if not all(stat in stats for stat in required_stats):
                return False

        return True

    return False
