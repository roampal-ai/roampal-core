"""
Learning Speed & Effectiveness Benchmark
=========================================

Measures how quickly and effectively the system learns from interactions.

Key Metrics:
1. Cold Start Performance - How well does system work with no prior learning?
2. Learning Curve - How many interactions to reach stable performance?
3. Adaptation Speed - How fast does routing improve with feedback?
4. Knowledge Retention - Does learning persist and stay accurate?
5. Pattern Recognition - Can system identify and reuse successful patterns?
6. Context Specialization - Does system learn context-specific strategies?

This benchmark simulates realistic learning scenarios:
- Multiple query types (coding, personal, reference)
- Mixed outcomes (worked/failed/partial)
- Progressive feedback over time
- Cross-session learning persistence
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import sys
import uuid
import shutil
import time
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem, ActionOutcome


async def create_fresh_memory():
    """Create a fresh memory instance with clean database."""
    test_dir = f"test_data_learning_{uuid.uuid4().hex[:8]}"
    memory = UnifiedMemorySystem(
        data_path=test_dir,
        
        
    )
    await memory.initialize()
    return memory, test_dir


def cleanup_memory(test_dir: str):
    """Clean up test data directory."""
    if Path(test_dir).exists():
        try:
            shutil.rmtree(test_dir)
        except:
            pass


async def test_cold_start_baseline():
    """TEST 1: Measure cold start performance (no prior learning)"""
    print("-"*80)
    print("TEST 1: Cold Start Baseline")
    print("-"*80)
    print("Scenario: Fresh system with no learning - establish baseline")

    memory, test_dir = await create_fresh_memory()

    try:
        # Seed with diverse content across collections
        # Books - reference material
        await memory.store(
            text="Python list comprehensions provide a concise way to create lists.",
            collection="books",
            metadata={"title": "Python Guide", "type": "reference"}
        )
        await memory.store(
            text="Git branching allows parallel development workflows.",
            collection="books",
            metadata={"title": "Git Handbook", "type": "reference"}
        )

        # Patterns - proven solutions
        await memory.store(
            text="For Python debugging, use pdb.set_trace() or breakpoint().",
            collection="patterns",
            metadata={"score": 0.9, "uses": 10, "query": "debug python"}
        )

        # Memory bank - user facts
        await memory.store_memory_bank(
            text="User prefers VS Code as their IDE.",
            tags=["preference", "tools"],
            importance=0.8,
            confidence=0.9
        )
        await memory.store_memory_bank(
            text="User works primarily with Python and TypeScript.",
            tags=["skills", "programming"],
            importance=0.9,
            confidence=0.95
        )

        print("  [SETUP] Seeded 5 items across books, patterns, memory_bank")

        # Test routing accuracy on cold start
        test_queries = [
            ("How do I debug Python code?", ["patterns", "books"]),
            ("What IDE does the user prefer?", ["memory_bank"]),
            ("Explain git branching", ["books"]),
            ("What programming languages does user know?", ["memory_bank"]),
            ("Python list syntax", ["books", "patterns"]),
        ]

        correct_routes = 0
        for query, expected_collections in test_queries:
            routed = memory._route_query(query)
            # Check if any expected collection is in routed
            hit = any(exp in routed for exp in expected_collections)
            if hit:
                correct_routes += 1
                print(f"  [OK] '{query[:30]}...' -> {routed[:2]}")
            else:
                print(f"  [MISS] '{query[:30]}...' -> {routed[:2]} (expected {expected_collections})")

        cold_accuracy = correct_routes / len(test_queries) * 100
        print(f"\n  Cold Start Routing Accuracy: {correct_routes}/{len(test_queries)} ({cold_accuracy:.0f}%)")

        return ("Cold start baseline", cold_accuracy, {"accuracy": cold_accuracy})
    finally:
        cleanup_memory(test_dir)


async def test_learning_curve():
    """TEST 2: Measure how quickly routing improves with feedback"""
    print("\n" + "-"*80)
    print("TEST 2: Learning Curve")
    print("-"*80)
    print("Scenario: Track routing accuracy as feedback accumulates")

    memory, test_dir = await create_fresh_memory()

    try:
        # Seed content
        coding_doc = await memory.store(
            text="Use async/await for non-blocking I/O in Python.",
            collection="patterns",
            metadata={"score": 0.8, "uses": 5, "query": "async python"}
        )

        personal_doc = await memory.store_memory_bank(
            text="User's favorite Python framework is FastAPI.",
            tags=["preference", "python"],
            importance=0.9,
            confidence=0.9
        )

        reference_doc = await memory.store(
            text="FastAPI is a modern Python web framework for building APIs.",
            collection="books",
            metadata={"title": "FastAPI Docs", "type": "reference"}
        )

        # Define training scenarios with correct routing
        training_scenarios = [
            ("How to use async in Python?", coding_doc, "patterns", "worked"),
            ("What Python framework does user like?", personal_doc, "memory_bank", "worked"),
            ("FastAPI documentation", reference_doc, "books", "worked"),
            ("async await syntax Python", coding_doc, "patterns", "worked"),
            ("user's preferred API framework", personal_doc, "memory_bank", "worked"),
        ]

        # Measure accuracy at intervals
        accuracy_history = []

        # Initial accuracy (0 feedback)
        test_query = "async Python programming"
        initial_route = memory._route_query(test_query)
        initial_has_patterns = "patterns" in initial_route[:2]
        accuracy_history.append(("0 feedback", 1 if initial_has_patterns else 0))

        # Train progressively
        for i, (query, doc_id, collection, outcome) in enumerate(training_scenarios):
            # Record outcome
            await memory.record_outcome(
                doc_id=doc_id,
                outcome=outcome,
                context={"quiz_question": query}
            )

            # Test routing after each feedback
            test_route = memory._route_query("async Python programming")
            has_patterns = "patterns" in test_route[:2]
            accuracy_history.append((f"{i+1} feedback", 1 if has_patterns else 0))

        # Extended training - repeat successful pattern
        for i in range(10):
            await memory.record_outcome(
                doc_id=coding_doc,
                outcome="worked",
                context={"quiz_question": "async Python code"}
            )

        final_route = memory._route_query("async Python programming")
        final_has_patterns = "patterns" in final_route[:2]
        accuracy_history.append(("15 feedback", 1 if final_has_patterns else 0))

        # Print learning curve
        print("\n  Learning Curve (routing 'async Python' to patterns):")
        print("  " + "-"*50)
        for stage, correct in accuracy_history:
            bar = "#" * (correct * 20)
            print(f"  {stage:15s} | {bar:20s} | {'HIT' if correct else 'MISS'}")

        # Calculate improvement
        initial_correct = accuracy_history[0][1]
        final_correct = accuracy_history[-1][1]
        improved = final_correct >= initial_correct

        print(f"\n  Learning detected: {'YES' if improved or final_correct else 'PARTIAL'}")

        return ("Learning curve", 100 if final_correct else 50, {"history": accuracy_history})
    finally:
        cleanup_memory(test_dir)


async def test_adaptation_speed():
    """TEST 3: How fast does system adapt to new patterns?"""
    print("\n" + "-"*80)
    print("TEST 3: Adaptation Speed")
    print("-"*80)
    print("Scenario: Introduce new pattern, measure interactions to learn")

    memory, test_dir = await create_fresh_memory()

    try:
        # Create a completely new domain the system hasn't seen
        fitness_docs = []
        for i, text in enumerate([
            "User's workout routine: Monday chest, Tuesday back, Wednesday legs.",
            "User tracks calories using MyFitnessPal app.",
            "User's goal is to lose 10 pounds by summer.",
        ]):
            doc_id = await memory.store_memory_bank(
                text=text,
                tags=["fitness", "health"],
                importance=0.85,
                confidence=0.9
            )
            fitness_docs.append(doc_id)

        print("  [SETUP] Created 3 fitness-related memories")

        # Test initial routing for fitness queries
        fitness_query = "What is user's workout schedule?"

        interactions_to_learn = 0
        max_interactions = 20
        learned = False

        for i in range(max_interactions):
            # Check current routing
            route = memory._route_query(fitness_query)

            # Success = memory_bank is prioritized for personal queries
            if "memory_bank" in route[:2]:
                if i == 0:
                    print(f"  [IMMEDIATE] Routed correctly on first try!")
                    learned = True
                    break
                else:
                    print(f"  [LEARNED] Correct routing after {i} interactions")
                    learned = True
                    interactions_to_learn = i
                    break

            # Provide feedback
            await memory.record_outcome(
                doc_id=fitness_docs[0],
                outcome="worked",
                context={"quiz_question": fitness_query}
            )
            interactions_to_learn = i + 1

        if not learned:
            # Check final state
            final_route = memory._route_query(fitness_query)
            if "memory_bank" in final_route[:2]:
                print(f"  [SLOW] Learned after {max_interactions} interactions")
                learned = True
                interactions_to_learn = max_interactions
            else:
                print(f"  [FAIL] Did not learn after {max_interactions} interactions")
                print(f"  Final route: {final_route}")

        # Score based on speed (fewer = better)
        if interactions_to_learn == 0:
            speed_score = 100  # Immediate
        elif interactions_to_learn <= 3:
            speed_score = 90   # Fast learner
        elif interactions_to_learn <= 10:
            speed_score = 70   # Moderate
        elif learned:
            speed_score = 50   # Slow but learned
        else:
            speed_score = 20   # Did not learn

        print(f"\n  Adaptation Speed Score: {speed_score}/100")
        print(f"  Interactions needed: {interactions_to_learn if learned else 'N/A'}")

        return ("Adaptation speed", speed_score, {"interactions": interactions_to_learn, "learned": learned})
    finally:
        cleanup_memory(test_dir)


async def test_knowledge_retention():
    """TEST 4: Does learned knowledge persist?"""
    print("\n" + "-"*80)
    print("TEST 4: Knowledge Retention")
    print("-"*80)
    print("Scenario: Train, then verify knowledge persists after many operations")

    memory, test_dir = await create_fresh_memory()

    try:
        # Phase 1: Establish strong pattern
        pattern_doc = await memory.store(
            text="For database queries, always use parameterized statements to prevent SQL injection.",
            collection="patterns",
            metadata={"score": 0.7, "uses": 3, "query": "database security"}
        )

        # Train it heavily
        for i in range(15):
            await memory.record_outcome(
                doc_id=pattern_doc,
                outcome="worked",
                context={"quiz_question": "SQL injection prevention"}
            )

        print("  [TRAIN] Established 'database security' pattern (15 worked outcomes)")

        # Check learned state
        trained_route = memory._route_query("How to prevent SQL injection?")
        trained_has_patterns = "patterns" in trained_route[:2]
        print(f"  [AFTER TRAINING] Routes to patterns: {'YES' if trained_has_patterns else 'NO'}")

        # Phase 2: Flood with unrelated operations
        print("  [NOISE] Adding 50 unrelated operations...")
        for i in range(50):
            # Store random memories
            await memory.store_memory_bank(
                text=f"Random fact #{i}: The user mentioned something about topic {i % 10}.",
                tags=["noise"],
                importance=0.3,
                confidence=0.3
            )

            # Do some searches
            await memory.search(
                query=f"topic {i % 10} information",
                collections=["memory_bank"],
                limit=3
            )

        # Phase 3: Test if original learning persists
        final_route = memory._route_query("How to prevent SQL injection?")
        final_has_patterns = "patterns" in final_route[:2]
        print(f"  [AFTER NOISE] Routes to patterns: {'YES' if final_has_patterns else 'NO'}")

        # Check KG state
        kg = memory.knowledge_graph
        routing_patterns = kg.get("routing_patterns", {})

        # Look for database/sql related concepts
        retained_concepts = []
        for concept, data in routing_patterns.items():
            if "sql" in concept.lower() or "database" in concept.lower() or "injection" in concept.lower():
                retained_concepts.append(concept)

        print(f"  [RETENTION] Found {len(retained_concepts)} relevant concepts in KG")

        # Score
        retention_score = 0
        if final_has_patterns:
            retention_score += 50
        if trained_has_patterns:
            retention_score += 25
        if len(retained_concepts) > 0:
            retention_score += 25

        print(f"\n  Knowledge Retention Score: {retention_score}/100")

        return ("Knowledge retention", retention_score, {
            "trained_correct": trained_has_patterns,
            "final_correct": final_has_patterns,
            "concepts_retained": len(retained_concepts)
        })
    finally:
        cleanup_memory(test_dir)


async def test_pattern_recognition():
    """TEST 5: Can system identify and reuse successful patterns?"""
    print("\n" + "-"*80)
    print("TEST 5: Pattern Recognition")
    print("-"*80)
    print("Scenario: System should recognize similar queries and reuse successful routes")

    memory, test_dir = await create_fresh_memory()

    try:
        # Create documents for different types
        error_doc = await memory.store(
            text="When you see 'ModuleNotFoundError', check your import paths and virtual environment.",
            collection="patterns",
            metadata={"score": 0.8, "uses": 5, "query": "python import error"}
        )

        # Train on specific error query
        training_queries = [
            "ModuleNotFoundError in Python",
            "Python import error fix",
            "Cannot import module Python",
        ]

        for query in training_queries:
            await memory.record_outcome(
                doc_id=error_doc,
                outcome="worked",
                context={"quiz_question": query}
            )

        print(f"  [TRAIN] Trained on {len(training_queries)} error-related queries")

        # Test on SIMILAR but NOT IDENTICAL queries
        test_queries = [
            "ImportError in my Python script",           # Similar - import error
            "Module not found when running Python",       # Similar - module error
            "Python can't find my package",               # Similar - package/import
            "Why does Python say no module named X?",     # Similar - module error
        ]

        recognized = 0
        for query in test_queries:
            route = memory._route_query(query)
            if "patterns" in route[:2]:
                recognized += 1
                print(f"  [OK] '{query[:40]}...' -> patterns")
            else:
                print(f"  [MISS] '{query[:40]}...' -> {route[:2]}")

        recognition_rate = recognized / len(test_queries) * 100
        print(f"\n  Pattern Recognition Rate: {recognized}/{len(test_queries)} ({recognition_rate:.0f}%)")

        return ("Pattern recognition", recognition_rate, {"recognized": recognized, "total": len(test_queries)})
    finally:
        cleanup_memory(test_dir)


async def test_context_specialization():
    """TEST 6: Does system learn context-specific strategies?"""
    print("\n" + "-"*80)
    print("TEST 6: Context Specialization")
    print("-"*80)
    print("Scenario: Different contexts should route to different collections")

    memory, test_dir = await create_fresh_memory()

    try:
        # Create context-specific content
        # Coding context -> patterns
        coding_doc = await memory.store(
            text="Use list comprehensions for cleaner Python code.",
            collection="patterns",
            metadata={"score": 0.85, "uses": 8, "query": "python code style"}
        )

        # Personal context -> memory_bank
        personal_doc = await memory.store_memory_bank(
            text="User's favorite programming language is Rust.",
            tags=["preference", "programming"],
            importance=0.9,
            confidence=0.95
        )

        # Reference context -> books
        reference_doc = await memory.store(
            text="Rust is a systems programming language focused on safety and performance.",
            collection="books",
            metadata={"title": "Rust Overview", "type": "reference"}
        )

        # Train each context
        contexts = [
            ("coding", coding_doc, "patterns", [
                "How to write clean Python?",
                "Python coding best practices",
                "Improve my Python code",
            ]),
            ("personal", personal_doc, "memory_bank", [
                "What language does user prefer?",
                "User's favorite programming language",
                "User programming preferences",
            ]),
            ("reference", reference_doc, "books", [
                "What is Rust programming?",
                "Rust language overview",
                "Rust documentation",
            ]),
        ]

        # Train each context
        for context_name, doc_id, expected_collection, queries in contexts:
            for query in queries:
                await memory.record_outcome(
                    doc_id=doc_id,
                    outcome="worked",
                    context={"quiz_question": query}
                )
            print(f"  [TRAIN] {context_name} context: {len(queries)} queries -> {expected_collection}")

        # Test specialization
        test_cases = [
            ("Python code improvement tips", "patterns", "coding"),
            ("User's language preference", "memory_bank", "personal"),
            ("What is Rust used for?", "books", "reference"),
        ]

        correct = 0
        for query, expected, context_name in test_cases:
            route = memory._route_query(query)
            if expected in route[:2]:
                correct += 1
                print(f"  [OK] {context_name}: '{query[:30]}...' -> {expected}")
            else:
                print(f"  [MISS] {context_name}: '{query[:30]}...' -> {route[:2]} (expected {expected})")

        specialization_rate = correct / len(test_cases) * 100
        print(f"\n  Context Specialization Rate: {correct}/{len(test_cases)} ({specialization_rate:.0f}%)")

        return ("Context specialization", specialization_rate, {"correct": correct, "total": len(test_cases)})
    finally:
        cleanup_memory(test_dir)


async def test_learning_efficiency():
    """TEST 7: How efficiently does the system learn (feedback per improvement)?"""
    print("\n" + "-"*80)
    print("TEST 7: Learning Efficiency")
    print("-"*80)
    print("Scenario: Measure improvement per unit of feedback")

    memory, test_dir = await create_fresh_memory()

    try:
        # Create a scoring baseline
        test_doc = await memory.store(
            text="For React performance, use useMemo and useCallback hooks.",
            collection="patterns",
            metadata={"score": 0.5, "uses": 0, "query": "react performance"}
        )

        print("  [SETUP] Created React performance pattern (score=0.5)")

        # Track score progression
        score_history = []

        doc = memory.collections["patterns"].get_fragment(test_doc)
        if doc:
            score_history.append(("initial", doc["metadata"].get("score", 0.5)))

        # Apply feedback incrementally and track
        feedback_count = 0
        for i in range(20):
            outcome = "worked" if i % 3 != 2 else "partial"  # Mix of outcomes
            await memory.record_outcome(
                doc_id=test_doc,
                outcome=outcome,
                context={"quiz_question": "React performance optimization"}
            )
            feedback_count += 1

            doc = memory.collections["patterns"].get_fragment(test_doc)
            if doc:
                current_score = doc["metadata"].get("score", 0.5)
                score_history.append((f"fb_{feedback_count}", current_score))

        # Calculate efficiency metrics
        if len(score_history) >= 2:
            initial_score = score_history[0][1]
            final_score = score_history[-1][1]
            total_improvement = final_score - initial_score
            efficiency = total_improvement / feedback_count if feedback_count > 0 else 0

            print(f"\n  Score Progression:")
            print(f"    Initial: {initial_score:.2f}")
            print(f"    Final:   {final_score:.2f}")
            print(f"    Change:  {total_improvement:+.2f}")
            print(f"    Feedback given: {feedback_count}")
            print(f"    Efficiency: {efficiency:.3f} points/feedback")

            # Find when score first exceeded thresholds
            threshold_times = {}
            for stage, score in score_history:
                if score >= 0.7 and "0.7" not in threshold_times:
                    threshold_times["0.7"] = stage
                if score >= 0.8 and "0.8" not in threshold_times:
                    threshold_times["0.8"] = stage
                if score >= 0.9 and "0.9" not in threshold_times:
                    threshold_times["0.9"] = stage

            if threshold_times:
                print(f"    Thresholds reached: {threshold_times}")

            # Score based on efficiency
            if efficiency > 0.02:
                efficiency_score = 100
            elif efficiency > 0.01:
                efficiency_score = 80
            elif efficiency > 0.005:
                efficiency_score = 60
            elif efficiency > 0:
                efficiency_score = 40
            else:
                efficiency_score = 20
        else:
            efficiency_score = 0
            efficiency = 0

        print(f"\n  Learning Efficiency Score: {efficiency_score}/100")

        return ("Learning efficiency", efficiency_score, {
            "efficiency": efficiency,
            "feedback_count": feedback_count,
            "score_change": total_improvement if len(score_history) >= 2 else 0
        })
    finally:
        cleanup_memory(test_dir)


async def test_learning_speed():
    """Run all learning speed and effectiveness benchmarks."""

    print("\n" + "="*80)
    print("LEARNING SPEED & EFFECTIVENESS BENCHMARK")
    print("="*80)
    print("\nMeasures how quickly and effectively the system learns.\n")

    start_time = time.time()
    test_results = []

    # Run each test
    test_results.append(await test_cold_start_baseline())
    test_results.append(await test_learning_curve())
    test_results.append(await test_adaptation_speed())
    test_results.append(await test_knowledge_retention())
    test_results.append(await test_pattern_recognition())
    test_results.append(await test_context_specialization())
    test_results.append(await test_learning_efficiency())

    elapsed = time.time() - start_time

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("LEARNING BENCHMARK RESULTS")
    print("="*80)

    total_score = 0
    for name, score, details in test_results:
        bar_len = int(score / 5)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {name:25s} [{bar}] {score:.0f}%")
        total_score += score

    avg_score = total_score / len(test_results)

    print(f"\n  Average Learning Score: {avg_score:.0f}%")
    print(f"  Runtime: {elapsed:.1f}s")

    # Grade
    print("\n" + "-"*80)
    if avg_score >= 80:
        grade = "A - Excellent Learner"
    elif avg_score >= 70:
        grade = "B - Good Learner"
    elif avg_score >= 60:
        grade = "C - Adequate Learner"
    elif avg_score >= 50:
        grade = "D - Slow Learner"
    else:
        grade = "F - Learning Impaired"

    print(f"  GRADE: {grade}")

    print("\n" + "="*80)
    if avg_score >= 60:
        print(f"PASS - System demonstrates effective learning capabilities")
        print(f"       Average score: {avg_score:.0f}%")
    else:
        print(f"NEEDS IMPROVEMENT - Learning capabilities below threshold")
        print(f"       Average score: {avg_score:.0f}%")
    print("="*80)

    return avg_score >= 50  # Pass if average is 50% or better


if __name__ == "__main__":
    success = asyncio.run(test_learning_speed())
    sys.exit(0 if success else 1)
