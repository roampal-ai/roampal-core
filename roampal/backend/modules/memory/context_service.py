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
        kg_service: Any = None,
        embed_fn: Optional[Callable[[str], Awaitable[List[float]]]] = None,
        config: Optional[MemoryConfig] = None
    ):
        """
        Initialize ContextService.

        Args:
            collections: Dict of collection name -> adapter
            kg_service: KnowledgeGraphService for pattern/routing access
            embed_fn: Async function to embed text for similarity search
            config: Memory configuration
        """
        self.collections = collections
        self.kg_service = kg_service
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

        if not self.kg_service or not concepts:
            return patterns

        # Create pattern signature from concepts
        pattern_signature = "_".join(sorted(concepts[:3]))

        # Check problem categories
        problem_categories = self.kg_service.get_problem_categories()
        if pattern_signature in problem_categories:
            past_solutions = problem_categories[pattern_signature]

            for doc_id in past_solutions[:2]:
                # Look in patterns and history collections
                for coll_name in ["patterns", "history"]:
                    if coll_name not in self.collections:
                        continue

                    doc = self.collections[coll_name].get_fragment(doc_id)
                    if doc:
                        metadata = doc.get("metadata", {})
                        score = metadata.get("score", 0.5)
                        uses = metadata.get("uses", 0)
                        last_outcome = metadata.get("last_outcome", "unknown")

                        # Only include proven patterns
                        if score >= self.config.promotion_score_threshold and last_outcome == "worked":
                            # v0.2.8: Full content, no truncation
                            patterns.append({
                                "text": doc.get("content", ""),
                                "score": score,
                                "uses": uses,
                                "collection": coll_name,
                                "insight": f"Based on {uses} past use(s), this approach had a {int(score*100)}% success rate"
                            })
                        break

        return patterns

    def _check_failure_patterns(
        self,
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Check if similar attempts failed before."""
        past_outcomes = []

        if not self.kg_service:
            return past_outcomes

        failure_patterns = self.kg_service.get_failure_patterns()

        for failure_key, failures in failure_patterns.items():
            # Check if current message relates to past failures
            if any(concept in failure_key.lower() for concept in concepts):
                recent_failures = [f for f in failures if f.get("timestamp", "")][-2:]

                for failure in recent_failures:
                    # v0.2.8: Full content, no truncation
                    past_outcomes.append({
                        "outcome": "failed",
                        "reason": failure_key,
                        "when": failure.get("timestamp", ""),
                        "insight": f"Note: Similar approach failed before due to: {failure_key}"
                    })

        return past_outcomes

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
        """Get proactive insights based on routing patterns."""
        insights = []

        if not self.kg_service:
            return insights

        routing_patterns = self.kg_service.get_routing_patterns()

        for concept in concepts[:3]:
            if concept in routing_patterns:
                pattern = routing_patterns[concept]
                success_rate = pattern.get("success_rate", 0)
                best_collection = pattern.get("best_collection", "unknown")

                if success_rate > 0.7:
                    insights.append({
                        "concept": concept,
                        "success_rate": success_rate,
                        "recommendation": f"For '{concept}', check {best_collection} collection (historically {int(success_rate*100)}% effective)"
                    })

        return insights

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
        """
        Extract concepts from text.

        Uses KG service if available, otherwise basic extraction.
        """
        if self.kg_service:
            return self.kg_service.extract_concepts(text)

        # Basic extraction fallback
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
        """
        Find known solutions for similar problems.

        Args:
            query: The problem query

        Returns:
            List of known solutions with boost information
        """
        if not query or not self.kg_service:
            return []

        try:
            # Extract concepts
            query_concepts = self._extract_concepts(query)
            query_signature = "_".join(sorted(query_concepts[:5]))

            known_solutions = []
            problem_solutions = self.kg_service.get_problem_solutions()

            # Look for exact matches
            if query_signature in problem_solutions:
                solutions = problem_solutions[query_signature]

                # Sort by success count and recency
                sorted_solutions = sorted(
                    solutions,
                    key=lambda x: (x.get("success_count", 0), x.get("last_used", "")),
                    reverse=True
                )

                # Get actual documents
                for solution in sorted_solutions[:3]:
                    doc_id = solution.get("doc_id")
                    if doc_id:
                        doc = self._get_document(doc_id)
                        if doc:
                            doc["distance"] = doc.get("distance", 1.0) * 0.5  # 50% boost
                            doc["is_known_solution"] = True
                            doc["solution_success_count"] = solution.get("success_count", 0)
                            known_solutions.append(doc)
                            logger.info(f"Found known solution: {doc_id}")

            # Check partial matches
            for problem_sig, solutions in problem_solutions.items():
                if problem_sig != query_signature:
                    problem_concepts_stored = set(problem_sig.split("_"))
                    overlap = len(set(query_concepts) & problem_concepts_stored)

                    if overlap >= 3:  # Significant overlap
                        for solution in solutions[:1]:
                            doc_id = solution.get("doc_id")
                            existing_ids = [s.get("id") for s in known_solutions]

                            if doc_id and doc_id not in existing_ids:
                                doc = self._get_document(doc_id)
                                if doc:
                                    doc["distance"] = doc.get("distance", 1.0) * 0.7  # 30% boost
                                    doc["is_partial_solution"] = True
                                    doc["concept_overlap"] = overlap
                                    known_solutions.append(doc)

            return known_solutions

        except Exception as e:
            logger.error(f"Error finding known solutions: {e}")
            return []

    def _get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document from appropriate collection."""
        for coll_name, adapter in self.collections.items():
            if doc_id.startswith(coll_name + "_"):
                doc = adapter.get_fragment(doc_id)
                if doc:
                    return {
                        "id": doc_id,
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "distance": 1.0,
                        "collection": coll_name
                    }
        return None

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
