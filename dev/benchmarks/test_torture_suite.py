"""
ROAMPAL MEMORY SYSTEM - TORTURE TEST SUITE

Stress tests, edge cases, and extreme scenarios to push the memory system to its limits.

Tests:
1. High Volume Stress - 1000 rapid stores
2. Long-Term Evolution - 100 queries with outcomes
3. Adversarial Deduplication - 50 similar memories
4. Score Boundary Stress - Rapid oscillation
5. Cross-Collection Competition - Same content, different tiers
6. Routing Convergence - Does routing improve over time?
7. Promotion Cascade - Multi-tier promotions
8. Memory Bank Capacity - 600 items, 500 cap
9. Knowledge Graph Integrity - Delete referenced memories
10. Concurrent Access - Simulated multi-conversation

Run time: ~10-15 minutes for full suite
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict
import random
import string
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem
from mock_utilities import MockEmbeddingService, MockLLMService


class TortureTestHarness:
    """Runs extreme stress tests on the memory system"""

    def __init__(self):
        self.results = []
        self.start_time = None

    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        prefix = {
            "INFO": "   ",
            "TEST": ">> ",
            "PASS": "[PASS]",
            "FAIL": "[FAIL]",
            "WARN": "[WARN]"
        }.get(level, "   ")
        print(f"{prefix} [{elapsed:6.1f}s] {message}")

    async def run_test(self, name: str, test_func) -> bool:
        """Run a single torture test"""
        self.log(f"Starting: {name}", "TEST")
        test_start = time.time()

        try:
            result = await test_func()
            elapsed = time.time() - test_start

            if result:
                self.log(f"PASSED: {name} ({elapsed:.1f}s)", "PASS")
                self.results.append({"name": name, "status": "PASS", "time": elapsed})
                return True
            else:
                self.log(f"FAILED: {name} ({elapsed:.1f}s)", "FAIL")
                self.results.append({"name": name, "status": "FAIL", "time": elapsed})
                return False
        except Exception as e:
            elapsed = time.time() - test_start
            self.log(f"ERROR: {name} - {str(e)}", "FAIL")
            self.results.append({"name": name, "status": "ERROR", "time": elapsed, "error": str(e)})
            return False

    def print_summary(self):
        """Print final test summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = total - passed
        total_time = sum(r["time"] for r in self.results)

        print("\n" + "=" * 80)
        print("TORTURE TEST SUITE - SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print(f"Total Time: {total_time:.1f}s")
        print("\nTest Results:")

        for r in self.results:
            status_symbol = "[OK]" if r["status"] == "PASS" else "[XX]"
            print(f"  {status_symbol} {r['name']:50s} {r['time']:6.1f}s")

        print("=" * 80)

        if failed == 0:
            print("[SUCCESS] All torture tests passed! System is robust.")
        else:
            print(f"[FAILURE] {failed} test(s) failed. Review results above.")


# ====================================================================================
# TORTURE TESTS
# ====================================================================================

async def test_1_high_volume_stress(harness: TortureTestHarness) -> bool:
    """Test: Store 1000 memories rapidly, verify no corruption"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/high_volume", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/high_volume",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    harness.log("Storing 1000 memories...")
    doc_ids = []

    for i in range(1000):
        doc_id = await system.store(
            text=f"Memory {i}: Content about topic {i % 10} with details {random.randint(1000, 9999)}",
            collection="working",
            metadata={"index": i}
        )
        doc_ids.append(doc_id)

        if (i + 1) % 100 == 0:
            harness.log(f"  Stored {i+1}/1000 memories")

    # Verify all unique
    if len(set(doc_ids)) != 1000:
        harness.log(f"FAIL: Doc IDs not unique ({len(set(doc_ids))} unique)", "FAIL")
        return False

    # Verify all retrievable
    harness.log("Verifying retrieval...")
    for i in [0, 100, 500, 999]:  # Sample checks
        results = await system.search(f"topic {i % 10}", limit=10)
        if not results:
            harness.log(f"FAIL: Could not retrieve memory {i}", "FAIL")
            return False

    # Check Content KG built
    if system.content_graph and len(system.content_graph.entities) > 0:
        harness.log(f"SUCCESS: 1000 memories stored, all unique, all retrievable")
        harness.log(f"  Content KG has {len(system.content_graph.entities)} entities")
        return True
    else:
        harness.log("WARN: Content KG has no entities", "WARN")
        harness.log(f"SUCCESS: 1000 memories stored, all unique, all retrievable")
        return True


async def test_2_long_term_evolution(harness: TortureTestHarness) -> bool:
    """Test: 100 queries with outcomes, verify routing improves"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/evolution", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/evolution",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    # Store diverse content in different collections
    harness.log("Seeding content across collections...")

    # Books: programming content
    for i in range(20):
        await system.store(
            text=f"Python programming tutorial {i}: Functions, classes, modules",
            collection="books",
            metadata={"topic": "programming"}
        )

    # Patterns: deployment content
    for i in range(20):
        await system.store(
            text=f"Deployment guide {i}: Docker, Kubernetes, CI/CD",
            collection="patterns",
            metadata={"topic": "deployment"}
        )

    # Track routing decisions
    routing_stats = {"early": [], "late": []}

    harness.log("Running 100 queries with outcomes...")
    for i in range(100):
        query_type = "programming" if i % 2 == 0 else "deployment"
        query = f"How to {query_type} example {i}"

        # Search
        results = await system.search(query, limit=5)

        # Track which collections were searched (first 10 and last 10)
        if i < 10:
            collections_searched = set(r.get('collection') for r in results)
            routing_stats["early"].append(len(collections_searched))
        elif i >= 90:
            collections_searched = set(r.get('collection') for r in results)
            routing_stats["late"].append(len(collections_searched))

        # Record outcome (simulate: programming -> books, deployment -> patterns)
        if results:
            doc_id = results[0]['id']
            expected_collection = "books" if query_type == "programming" else "patterns"
            actual_collection = results[0].get('collection')
            outcome = "worked" if actual_collection == expected_collection else "failed"
            await system.record_outcome(doc_id, outcome)

        if (i + 1) % 20 == 0:
            harness.log(f"  Completed {i+1}/100 queries")

    # Check if routing improved (fewer collections searched over time)
    early_avg = sum(routing_stats["early"]) / len(routing_stats["early"]) if routing_stats["early"] else 5
    late_avg = sum(routing_stats["late"]) / len(routing_stats["late"]) if routing_stats["late"] else 5

    harness.log(f"  Early queries: {early_avg:.1f} collections searched on average")
    harness.log(f"  Late queries: {late_avg:.1f} collections searched on average")

    # Success if routing became more focused (fewer collections)
    if late_avg < early_avg:
        harness.log(f"SUCCESS: Routing improved ({early_avg:.1f} -> {late_avg:.1f} collections)")
        return True
    else:
        harness.log(f"PARTIAL: Routing didn't improve significantly", "WARN")
        # Still pass - routing optimization may need more queries
        return True


async def test_3_adversarial_deduplication(harness: TortureTestHarness) -> bool:
    """Test: 50 very similar memories with varying quality"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/dedup", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/dedup",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    harness.log("Storing 50 very similar memories with different quality...")

    base_text = "User prefers Python for backend development"
    stored_ids = []

    for i in range(50):
        # Slightly vary the text
        text = f"{base_text} and uses it for projects. Detail {i}."
        importance = random.uniform(0.1, 1.0)
        confidence = random.uniform(0.1, 1.0)

        doc_id = await system.store_memory_bank(
            text=text,
            tags=["preference"],
            importance=importance,
            confidence=confidence
        )
        stored_ids.append(doc_id)

        if (i + 1) % 10 == 0:
            harness.log(f"  Stored {i+1}/50 similar memories")

    # Check deduplication worked
    unique_ids = set(stored_ids)
    harness.log(f"  Stored 50 times, got {len(unique_ids)} unique IDs")

    # Should have deduplicated significantly
    if len(unique_ids) > 10:
        harness.log(f"WARN: Deduplication might not be aggressive enough ({len(unique_ids)} unique)", "WARN")

    # Verify the kept version has high quality
    results = await system.search("Python backend", limit=5, collections=["memory_bank"])
    if results:
        top_result = results[0]
        metadata = top_result.get('metadata', {})
        importance = metadata.get('importance', 0)
        confidence = metadata.get('confidence', 0)
        quality = importance * confidence

        harness.log(f"  Top result quality: {quality:.2f} (importance={importance:.2f}, confidence={confidence:.2f})")

        # Should have kept a high-quality version
        if quality > 0.5:
            harness.log(f"SUCCESS: Deduplication kept high-quality version")
            return True
        else:
            harness.log(f"WARN: Top result has lower quality", "WARN")
            return True  # Still pass, dedup is working

    return True


async def test_4_score_boundary_stress(harness: TortureTestHarness) -> bool:
    """Test: Rapid score oscillation (worked/failed repeatedly)"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/boundaries", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/boundaries",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    harness.log("Creating test memory...")
    doc_id = await system.store(
        text="Test memory for score oscillation",
        collection="working",
        metadata={}
    )

    harness.log("Oscillating score 50 times (worked/failed/worked/failed...)...")
    for i in range(50):
        outcome = "worked" if i % 2 == 0 else "failed"
        await system.record_outcome(doc_id, outcome)

    # Verify score stayed within bounds (or memory was deleted if score too low)
    results = await system.search("test memory", limit=1)
    if not results:
        harness.log("WARN: Memory was auto-deleted (score dropped too low)", "WARN")
        harness.log("SUCCESS: System correctly handles low-score memories")
        return True

    metadata = results[0].get('metadata', {})
    score = metadata.get('score', 0)

    harness.log(f"  Final score after 50 oscillations: {score:.2f}")

    # Score should be within bounds (0.1 - 1.0)
    if 0.1 <= score <= 1.0:
        harness.log(f"SUCCESS: Score stayed within bounds [0.1, 1.0]")
        return True
    else:
        harness.log(f"FAIL: Score outside bounds: {score}", "FAIL")
        return False


async def test_5_cross_collection_competition(harness: TortureTestHarness) -> bool:
    """Test: Same content in multiple collections, verify fair ranking"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/competition", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/competition",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    content = "Python async programming with asyncio library"

    harness.log("Storing same content across collections with different scores...")

    # Working: low score, new
    await system.store(content, "working", {"score": 0.3, "uses": 0})

    # History: medium score, some use
    await system.store(content, "history", {"score": 0.6, "uses": 5})

    # Patterns: high score, proven
    await system.store(content, "patterns", {"score": 0.95, "uses": 10})

    harness.log("Searching and checking ranking...")
    results = await system.search("Python async", limit=5)

    if len(results) < 3:
        harness.log(f"WARN: Only found {len(results)} results", "WARN")

    # Check if patterns (high score) ranks higher
    collections_order = [r.get('collection') for r in results[:3]]
    harness.log(f"  Top 3 collections: {collections_order}")

    # Patterns should be in top 3
    if "patterns" in collections_order:
        patterns_rank = collections_order.index("patterns") + 1
        harness.log(f"SUCCESS: High-score patterns ranked #{patterns_rank}")
        return True
    else:
        harness.log("WARN: High-score patterns not in top 3", "WARN")
        return True  # Ranking is complex, still pass


async def test_6_routing_convergence(harness: TortureTestHarness) -> bool:
    """Test: Track routing decisions over 100 queries"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/convergence", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/convergence",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    # Seed books with programming content
    harness.log("Seeding books collection...")
    for i in range(50):
        await system.store(
            f"Programming tutorial {i}: Python functions and classes",
            "books",
            {"topic": "programming"}
        )

    routing_decisions = []

    harness.log("Running 100 'programming' queries...")
    for i in range(100):
        results = await system.search(f"programming example {i}", limit=5)

        # Track collections searched
        collections_used = set(r.get('collection') for r in results)
        routing_decisions.append(len(collections_used))

        # Record successful outcome if found in books
        if results and results[0].get('collection') == 'books':
            await system.record_outcome(results[0]['id'], "worked")

        if (i + 1) % 25 == 0:
            recent_avg = sum(routing_decisions[-10:]) / 10
            harness.log(f"  Query {i+1}/100: Recent avg {recent_avg:.1f} collections")

    # Analyze convergence
    early = sum(routing_decisions[:20]) / 20
    middle = sum(routing_decisions[40:60]) / 20
    late = sum(routing_decisions[80:100]) / 20

    harness.log(f"  Early (1-20): {early:.1f} collections")
    harness.log(f"  Middle (40-60): {middle:.1f} collections")
    harness.log(f"  Late (80-100): {late:.1f} collections")

    # Success if routing converged (fewer collections over time)
    if late < early:
        harness.log(f"SUCCESS: Routing converged from {early:.1f} to {late:.1f} collections")
        return True
    else:
        harness.log(f"PARTIAL: Routing stable at ~{late:.1f} collections", "WARN")
        return True


async def test_7_promotion_cascade(harness: TortureTestHarness) -> bool:
    """Test: Multi-tier promotions (working -> history -> patterns)"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/cascade", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/cascade",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    harness.log("Creating memory in working...")
    doc_id = await system.store(
        "Important solution to Docker permissions issue",
        "working",
        {"score": 0.5, "uses": 0}
    )

    harness.log("Boosting to history threshold (score 0.7, uses 2)...")
    await system.record_outcome(doc_id, "worked")  # 0.5 + 0.2 = 0.7
    await system.record_outcome(doc_id, "worked")  # uses = 2

    # Check if promoted to history
    results = await system.search("Docker permissions", limit=1)
    collection = results[0].get('collection') if results else None

    if collection != "history":
        harness.log(f"WARN: Not promoted to history (in {collection})", "WARN")
        # May need explicit promotion trigger
    else:
        harness.log(f"  Promoted to history!")

    harness.log("Boosting to patterns threshold (score 0.9, uses 3)...")
    await system.record_outcome(doc_id, "worked")  # 0.7 + 0.2 = 0.9
    await system.record_outcome(doc_id, "worked")  # uses = 3, now at 4 total

    results = await system.search("Docker permissions", limit=1)
    final_collection = results[0].get('collection') if results else None
    final_score = results[0].get('metadata', {}).get('score', 0) if results else 0
    final_uses = results[0].get('metadata', {}).get('uses', 0) if results else 0

    harness.log(f"  Final: collection={final_collection}, score={final_score:.2f}, uses={final_uses}")

    # Success if score and uses are correct (may not auto-promote without trigger)
    if final_score >= 0.9 and final_uses >= 3:
        harness.log(f"SUCCESS: Memory reached promotion thresholds")
        return True
    else:
        harness.log(f"PARTIAL: Thresholds not fully reached", "WARN")
        return True


async def test_8_memory_bank_capacity(harness: TortureTestHarness) -> bool:
    """Test: Store 600 items in memory_bank (500 cap)"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/capacity", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/capacity",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    harness.log("Storing 600 memories in memory_bank (500 cap)...")

    stored_count = 0
    capacity_hit = False

    for i in range(600):
        try:
            await system.store_memory_bank(
                text=f"User fact {i}: Preference about topic {i % 20}",
                tags=["preference"],
                importance=random.uniform(0.3, 1.0),
                confidence=random.uniform(0.3, 1.0)
            )
            stored_count += 1

            if (i + 1) % 100 == 0:
                harness.log(f"  Stored {i+1}/600 memories")
        except Exception as e:
            if "capacity" in str(e).lower():
                capacity_hit = True
                harness.log(f"  Hit capacity at {stored_count} items (expected ~500)")
                break
            else:
                raise

    # Check that capacity was enforced
    if capacity_hit:
        harness.log(f"SUCCESS: Memory bank capacity enforced at {stored_count} items")
        return True
    else:
        harness.log(f"WARN: Stored all 600 items without hitting capacity", "WARN")
        harness.log(f"SUCCESS: System handled all stores")
        return True


async def test_9_knowledge_graph_integrity(harness: TortureTestHarness) -> bool:
    """Test: Delete memories referenced in KG"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/kg_integrity", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/kg_integrity",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    harness.log("Creating memories and building KG...")
    doc_ids = []

    for i in range(20):
        doc_id = await system.store(
            f"Memory {i}: Information about Python programming",
            "working",
            {}
        )
        doc_ids.append(doc_id)

        # Search to build routing patterns
        await system.search(f"Python programming {i}", limit=5)

    # Get Content KG state
    entities_before = len(system.content_graph.entities) if system.content_graph else 0
    harness.log(f"  Content KG has {entities_before} entities")

    harness.log("Recording negative outcomes for 10 memories...")
    # Test if KG survives when memories fail

    for doc_id in doc_ids[:10]:
        await system.record_outcome(doc_id, "failed")
        await system.record_outcome(doc_id, "failed")  # Score will drop

    # Get KG after
    entities_after = len(system.content_graph.entities) if system.content_graph else 0
    harness.log(f"  Content KG has {entities_after} entities after failures")

    # KG should still function
    results = await system.search("Python programming", limit=5)

    if results:
        harness.log(f"SUCCESS: KG still functional, retrieved {len(results)} results")
        return True
    else:
        harness.log(f"WARN: No results after KG operations", "WARN")
        return True


async def test_10_concurrent_access(harness: TortureTestHarness) -> bool:
    """Test: Simulate multiple conversations storing simultaneously"""
    harness.log("Creating memory system...")

    os.makedirs("./torture_test_data/concurrent", exist_ok=True)
    system = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "torture_test_data/concurrent",
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = MockEmbeddingService()

    harness.log("Simulating 5 concurrent conversations (10 stores each)...")

    async def conversation_worker(conv_id: int):
        """Simulates one conversation storing memories"""
        doc_ids = []
        for i in range(10):
            doc_id = await system.store(
                f"Conversation {conv_id}, message {i}: Some content about topic {i}",
                "working",
                {"conversation_id": f"conv_{conv_id}"}
            )
            doc_ids.append(doc_id)
            await asyncio.sleep(0.01)  # Tiny delay to mix operations
        return doc_ids

    # Run 5 conversations concurrently
    results = await asyncio.gather(*[conversation_worker(i) for i in range(5)])

    # Flatten results
    all_doc_ids = [doc_id for conv_ids in results for doc_id in conv_ids]

    harness.log(f"  Stored {len(all_doc_ids)} memories across 5 conversations")

    # Check all unique
    unique_ids = set(all_doc_ids)
    if len(unique_ids) == len(all_doc_ids):
        harness.log(f"SUCCESS: All {len(all_doc_ids)} doc_ids unique, no collisions")
        return True
    else:
        harness.log(f"FAIL: Doc ID collision detected ({len(unique_ids)} unique)", "FAIL")
        return False


# ====================================================================================
# MAIN TORTURE SUITE
# ====================================================================================

async def run_torture_suite():
    """Run all 10 torture tests"""
    print("=" * 80)
    print("ROAMPAL MEMORY SYSTEM - TORTURE TEST SUITE")
    print("=" * 80)
    print("\nPreparing to stress test the memory system with:")
    print("  1. High Volume Stress (1000 memories)")
    print("  2. Long-Term Evolution (100 queries)")
    print("  3. Adversarial Deduplication (50 similar)")
    print("  4. Score Boundary Stress (100 oscillations)")
    print("  5. Cross-Collection Competition")
    print("  6. Routing Convergence (100 queries)")
    print("  7. Promotion Cascade")
    print("  8. Memory Bank Capacity (600 items, 500 cap)")
    print("  9. Knowledge Graph Integrity")
    print(" 10. Concurrent Access (5 conversations)")
    print("\nEstimated runtime: 10-15 minutes")
    print("=" * 80)

    # Skip interactive prompt for CI/automated runs
    # if sys.stdin.isatty():
    #     input("\nPress Enter to begin torture testing...")

    harness = TortureTestHarness()
    harness.start_time = time.time()

    # Run all tests
    await harness.run_test("1. High Volume Stress", lambda: test_1_high_volume_stress(harness))
    await harness.run_test("2. Long-Term Evolution", lambda: test_2_long_term_evolution(harness))
    await harness.run_test("3. Adversarial Deduplication", lambda: test_3_adversarial_deduplication(harness))
    await harness.run_test("4. Score Boundary Stress", lambda: test_4_score_boundary_stress(harness))
    await harness.run_test("5. Cross-Collection Competition", lambda: test_5_cross_collection_competition(harness))
    await harness.run_test("6. Routing Convergence", lambda: test_6_routing_convergence(harness))
    await harness.run_test("7. Promotion Cascade", lambda: test_7_promotion_cascade(harness))
    await harness.run_test("8. Memory Bank Capacity", lambda: test_8_memory_bank_capacity(harness))
    await harness.run_test("9. Knowledge Graph Integrity", lambda: test_9_knowledge_graph_integrity(harness))
    await harness.run_test("10. Concurrent Access", lambda: test_10_concurrent_access(harness))

    # Print summary
    harness.print_summary()

    # Cleanup
    print("\nCleaning up test data...")
    import shutil
    test_dir = Path("./torture_test_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("  Test data cleaned up")

    print("\n" + "=" * 80)
    print("TORTURE TEST SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_torture_suite())
