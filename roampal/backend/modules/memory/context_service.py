"""
ContextService - Extracted from UnifiedMemorySystem

Handles conversation context analysis for organic memory injection.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Awaitable

from .config import MemoryConfig

logger = logging.getLogger(__name__)


class ContextService:
    """
    Service for analyzing conversation context.

    Extracted from UnifiedMemorySystem.analyze_conversation_context and related.
    Provides:
    - Pattern recognition from past conversations
    - Failure awareness for similar approaches
    - Topic continuity detection
    - Proactive insights based on routing patterns
    - Repetition detection
    """

    def __init__(
        self,
        collections: Dict[str, Any],
        embed_fn: Optional[Callable[[str], Awaitable[List[float]]]] = None,
        config: Optional[MemoryConfig] = None,
        **kwargs,  # Accept and ignore kg_service for backward compat
    ):
        """
        Initialize ContextService.

        Args:
            collections: Dict of collection name -> adapter
            embed_fn: Async function to embed text for similarity search
            config: Memory configuration
        """
        self.collections = collections
        self.embed_fn = embed_fn
        self.config = config or MemoryConfig()

    async def analyze_conversation_context(
        self,
        current_message: str,
        recent_conversation: List[Dict[str, Any]],
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Analyze conversation context for organic memory injection.

        Args:
            current_message: The current user message
            recent_conversation: List of recent conversation messages
            conversation_id: Current conversation ID

        Returns:
            Dict with relevant patterns, past outcomes, continuity, insights
        """
        context = {
            "relevant_patterns": [],
            "past_outcomes": [],
            "topic_continuity": [],
            "proactive_insights": []
        }

        try:
            # Extract concepts from current message
            current_concepts = self._extract_concepts(current_message)

            # 1. Pattern Recognition
            patterns = await self._find_relevant_patterns(current_concepts)
            context["relevant_patterns"] = patterns

            # 2. Failure Awareness
            past_outcomes = self._check_failure_patterns(current_concepts)
            context["past_outcomes"] = past_outcomes

            # 3. Topic Continuity
            continuity = self._detect_topic_continuity(
                current_concepts, recent_conversation
            )
            context["topic_continuity"] = continuity

            # 4. Proactive Insights
            insights = self._get_proactive_insights(current_concepts)
            context["proactive_insights"] = insights

            # 5. Repetition Detection
            if self.embed_fn and "working" in self.collections:
                repetitions = await self._detect_repetition(
                    current_message, conversation_id
                )
                context["proactive_insights"].extend(repetitions[:1])

        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")

        return context

    async def _find_relevant_patterns(
        self,
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Find relevant patterns from past conversations."""
        patterns = []

        # v0.4.5: KG removed. Pattern recognition via tag-routed retrieval now.
        return patterns

    def _check_failure_patterns(
        self,
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Check if similar attempts failed before. v0.4.5: KG removed."""
        return []

    def _detect_topic_continuity(
        self,
        current_concepts: List[str],
        recent_conversation: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect if continuing or switching topics."""
        continuity = []

        if not recent_conversation or len(recent_conversation) < 2:
            return continuity

        # Get last user message
        last_messages = [
            msg for msg in recent_conversation[-3:]
            if msg.get("role") == "user"
        ]

        if not last_messages:
            return continuity

        last_message = last_messages[-1].get("content", "")
        last_concepts = self._extract_concepts(last_message)

        # Check concept overlap
        overlap = set(current_concepts) & set(last_concepts)
        if overlap:
            continuity.append({
                "continuing": True,
                "common_concepts": list(overlap),
                "insight": f"Continuing discussion about: {', '.join(list(overlap)[:3])}"
            })
        else:
            continuity.append({
                "continuing": False,
                "insight": "Topic shift detected - loading new context"
            })

        return continuity

    def _get_proactive_insights(
        self,
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Get proactive insights. v0.4.5: KG removed."""
        return []

    async def _detect_repetition(
        self,
        current_message: str,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Detect if user asked similar question recently."""
        repetitions = []

        if not self.embed_fn or "working" not in self.collections:
            return repetitions

        try:
            # Get embedding
            query_vector = await self.embed_fn(current_message)

            # Search working memory
            working_items = await self.collections["working"].query_vectors(
                query_vector=query_vector,
                top_k=3
            )

            for item in working_items:
                metadata = item.get("metadata", {})
                if metadata.get("conversation_id") == conversation_id:
                    # Calculate similarity
                    similarity = 1.0 / (1.0 + item.get("distance", 1.0))
                    if similarity > 0.85:  # Very similar
                        # v0.2.8: Full content, no truncation
                        repetitions.append({
                            "text": item.get("content", ""),
                            "similarity": similarity,
                            "insight": f"You mentioned something similar recently (similarity: {int(similarity*100)}%)"
                        })

        except Exception as e:
            logger.warning(f"Error detecting repetition: {e}")

        return repetitions

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text using basic extraction."""
        return self._basic_concept_extraction(text)

    def _basic_concept_extraction(self, text: str) -> List[str]:
        """Basic concept extraction without KG service."""
        if not text:
            return []

        # Simple word extraction (lowercase, alphabetic only)
        words = text.lower().split()
        concepts = []

        # Filter stopwords and short words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now',
            'it', 'its', 'and', 'but', 'or', 'if', 'as', 'this', 'that',
            'i', 'me', 'my', 'you', 'your', 'he', 'she', 'we', 'they'
        }

        for word in words:
            # Clean word
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) >= 3 and clean_word not in stopwords:
                concepts.append(clean_word)

        return concepts[:10]

    async def find_known_solutions(self, query: str) -> List[Dict[str, Any]]:
        """Find known solutions. v0.4.5: KG removed — tag-routed search handles this."""
        return []

    def get_context_summary(
        self,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable summary of context analysis.

        Args:
            context: Result from analyze_conversation_context

        Returns:
            Summary string
        """
        parts = []

        # Patterns
        patterns = context.get("relevant_patterns", [])
        if patterns:
            parts.append(f"Found {len(patterns)} relevant pattern(s) from past conversations")

        # Failures
        failures = context.get("past_outcomes", [])
        if failures:
            parts.append(f"Warning: {len(failures)} similar approach(es) failed before")

        # Continuity
        continuity = context.get("topic_continuity", [])
        if continuity:
            cont = continuity[0]
            if cont.get("continuing"):
                concepts = cont.get("common_concepts", [])
                parts.append(f"Continuing discussion: {', '.join(concepts[:3])}")
            else:
                parts.append("New topic detected")

        # Insights
        insights = context.get("proactive_insights", [])
        if insights:
            parts.append(f"{len(insights)} proactive insight(s) available")

        return " | ".join(parts) if parts else "No significant context detected"
