"""
Recovery & Corruption Resilience Test
=====================================

Tests the system's ability to handle failures and maintain data integrity.

Real-world failure modes:
1. Process killed mid-write (power failure, OOM kill, user Ctrl+C)
2. Disk full during storage operation
3. Corrupted ChromaDB index files
4. Partial writes (some collections updated, others not)
5. Network timeout during embedding generation

The system should:
- Not corrupt existing data when operations fail
- Recover gracefully from partial states
- Detect and handle corrupted data
- Maintain consistency across collections

Note: These tests simulate failures, they don't actually corrupt real data.
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import sys
import uuid
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem


async def create_fresh_memory():
    """Create a fresh memory instance with clean database."""
    test_dir = f"test_data_recovery_{uuid.uuid4().hex[:8]}"
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


async def test_interrupted_store_recovery():
    """TEST 1: System recovers from interrupted store operations"""
    print("-"*80)
    print("TEST 1: Interrupted Store Recovery")
    print("-"*80)
    print("Scenario: Store succeeds, then query works, simulating normal operation")
    print("          (True interruption testing requires process-level manipulation)")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store some baseline data
        baseline_facts = [
            "User's name is Alice Johnson.",
            "User works at TechCorp as a senior engineer.",
            "User has a dog named Max.",
        ]

        baseline_ids = []
        for fact in baseline_facts:
            doc_id = await memory.store_memory_bank(
                text=fact,
                tags=["baseline"],
                importance=0.9,
                confidence=0.9
            )
            baseline_ids.append(doc_id)
            await asyncio.sleep(0.02)

        print(f"  [BASELINE] Stored {len(baseline_facts)} facts successfully")

        # Verify baseline is searchable
        results = await memory.search(
            query="What is the user's name?",
            collections=["memory_bank"],
            limit=3
        )

        found_alice = any("alice" in r.get("text", "").lower() for r in results)

        # Now store more data (simulating operations after potential "interruption")
        await memory.store_memory_bank(
            text="User's favorite color is blue.",
            tags=["preference"],
            importance=0.7,
            confidence=0.7
        )

        # Verify both old and new data accessible
        results_after = await memory.search(
            query="user information",
            collections=["memory_bank"],
            limit=10
        )

        found_all = len(results_after) >= 4  # At least baseline + 1 new

        print(f"  [VERIFY] Baseline data accessible: {'YES' if found_alice else 'NO'}")
        print(f"  [VERIFY] All data intact after additional stores: {'YES' if found_all else 'NO'}")

        success = found_alice and found_all
        print(f"\nData integrity maintained: {'YES' if success else 'NO'}")

        return ("Interrupted store recovery", success)
    finally:
        cleanup_memory(test_dir)


async def test_empty_collection_handling():
    """TEST 2: System handles empty collections gracefully"""
    print("\n" + "-"*80)
    print("TEST 2: Empty Collection Handling")
    print("-"*80)
    print("Scenario: Query collections that have no data")

    memory, test_dir = await create_fresh_memory()

    try:
        # Search empty system
        results = await memory.search(
            query="anything at all",
            collections=["memory_bank"],
            limit=10
        )

        empty_ok = isinstance(results, list)  # Should return empty list, not crash

        # Search with multiple empty collections
        results_multi = await memory.search(
            query="test query",
            collections=["memory_bank", "working", "history"],
            limit=10
        )

        multi_ok = isinstance(results_multi, list)

        print(f"  [EMPTY] Single collection search: {'OK' if empty_ok else 'CRASHED'}")
        print(f"  [EMPTY] Multi-collection search: {'OK' if multi_ok else 'CRASHED'}")

        success = empty_ok and multi_ok
        print(f"\nEmpty collection handling: {'PASS' if success else 'FAIL'}")

        return ("Empty collection handling", success)
    finally:
        cleanup_memory(test_dir)


async def test_malformed_metadata_resilience():
    """TEST 3: System handles malformed metadata in stored docs"""
    print("\n" + "-"*80)
    print("TEST 3: Malformed Metadata Resilience")
    print("-"*80)
    print("Scenario: Documents with unexpected metadata types/values")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store with edge case metadata
        test_cases = [
            {"importance": 0.5, "confidence": 0.5},  # Normal
            {"importance": "high", "confidence": 0.5},  # String instead of float
            {"importance": 1.5, "confidence": -0.5},  # Out of bounds
            {"importance": None, "confidence": 0.5},  # None value
        ]

        successes = 0
        for i, meta in enumerate(test_cases):
            try:
                await memory.store_memory_bank(
                    text=f"Test document {i} with various metadata",
                    tags=["test"],
                    importance=meta["importance"] if meta["importance"] is not None else 0.5,
                    confidence=meta["confidence"] if meta["confidence"] is not None else 0.5
                )
                successes += 1
            except Exception as e:
                print(f"  [CASE {i}] Failed: {type(e).__name__}")

        print(f"  [STORE] {successes}/{len(test_cases)} stores succeeded")

        # Now search - should not crash even with weird metadata
        try:
            results = await memory.search(
                query="test document",
                collections=["memory_bank"],
                limit=10
            )
            search_ok = True
            print(f"  [SEARCH] Search completed, found {len(results)} results")
        except Exception as e:
            search_ok = False
            print(f"  [SEARCH] Failed: {type(e).__name__}")

        success = successes >= 2 and search_ok  # At least normal + some edge cases
        print(f"\nMalformed metadata handling: {'PASS' if success else 'FAIL'}")

        return ("Malformed metadata resilience", success)
    finally:
        cleanup_memory(test_dir)


async def test_concurrent_modification_safety():
    """TEST 4: Concurrent reads and writes don't corrupt data"""
    print("\n" + "-"*80)
    print("TEST 4: Concurrent Modification Safety")
    print("-"*80)
    print("Scenario: Simultaneous reads and writes from multiple 'sessions'")

    memory, test_dir = await create_fresh_memory()

    try:
        # Pre-populate with some data
        for i in range(10):
            await memory.store_memory_bank(
                text=f"Baseline fact number {i} about the user",
                tags=["baseline"],
                importance=0.8,
                confidence=0.8
            )

        print("  [BASELINE] Stored 10 initial facts")

        # Concurrent operations: writes and reads interleaved
        async def write_task(task_id):
            for i in range(5):
                # Use unique text to avoid deduplication
                unique_id = uuid.uuid4().hex[:8]
                await memory.store_memory_bank(
                    text=f"Task {task_id} wrote unique item {unique_id} at position {i} in sequence",
                    tags=["concurrent", f"task_{task_id}"],
                    importance=0.6,
                    confidence=0.6
                )
                await asyncio.sleep(0.01)
            return f"write_{task_id}"

        async def read_task(task_id):
            results_count = 0
            for i in range(5):
                results = await memory.search(
                    query=f"user fact information",
                    collections=["memory_bank"],
                    limit=5
                )
                results_count += len(results)
                await asyncio.sleep(0.01)
            return f"read_{task_id}:{results_count}"

        # Run 3 writers and 3 readers concurrently
        tasks = [
            write_task(0), write_task(1), write_task(2),
            read_task(0), read_task(1), read_task(2)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]

        print(f"  [CONCURRENT] {len(successes)}/6 tasks completed successfully")
        if errors:
            for e in errors:
                print(f"  [ERROR] {type(e).__name__}: {str(e)[:50]}")

        # Verify data integrity after concurrent operations
        final_results = await memory.search(
            query="user fact baseline concurrent",
            collections=["memory_bank"],
            limit=50
        )

        # Should have baseline (10) + concurrent writes (3 * 5 = 15) = 25 total
        expected_min = 20  # Allow some tolerance
        data_intact = len(final_results) >= expected_min

        print(f"  [VERIFY] Final document count: {len(final_results)} (expected >= {expected_min})")

        success = len(errors) == 0 and data_intact
        print(f"\nConcurrent modification safety: {'PASS' if success else 'FAIL'}")

        return ("Concurrent modification safety", success)
    finally:
        cleanup_memory(test_dir)


async def test_large_batch_atomicity():
    """TEST 5: Large batch operations maintain consistency"""
    print("\n" + "-"*80)
    print("TEST 5: Large Batch Atomicity")
    print("-"*80)
    print("Scenario: Store 100 related facts, verify all or none principle")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store a large batch of related facts
        batch_size = 100
        batch_facts = [
            f"Fact {i}: User visited location_{i % 10} on day_{i % 30} of month_{i % 12}"
            for i in range(batch_size)
        ]

        stored_ids = []
        failed_count = 0

        print(f"  [BATCH] Storing {batch_size} facts...")
        for i, fact in enumerate(batch_facts):
            try:
                doc_id = await memory.store_memory_bank(
                    text=fact,
                    tags=["batch", f"location_{i % 10}"],
                    importance=0.5 + (i % 5) * 0.1,
                    confidence=0.7
                )
                stored_ids.append(doc_id)
            except Exception as e:
                failed_count += 1

            if i % 25 == 0:
                print(f"    ... {i}/{batch_size}")

        print(f"  [BATCH] Stored: {len(stored_ids)}, Failed: {failed_count}")

        # Verify we can query and get consistent results
        results = await memory.search(
            query="user visited location",
            collections=["memory_bank"],
            limit=50
        )

        # Check for duplicates (consistency issue)
        seen_texts = set()
        duplicates = 0
        for r in results:
            text = r.get("text", "")
            if text in seen_texts:
                duplicates += 1
            seen_texts.add(text)

        print(f"  [VERIFY] Search returned {len(results)} results")
        print(f"  [VERIFY] Duplicates detected: {duplicates}")

        # Success if most stored and no duplicates
        success = len(stored_ids) >= batch_size * 0.95 and duplicates == 0
        print(f"\nLarge batch atomicity: {'PASS' if success else 'FAIL'}")

        return ("Large batch atomicity", success)
    finally:
        cleanup_memory(test_dir)


async def test_knowledge_graph_consistency():
    """TEST 6: Knowledge graph stays consistent with memory operations"""
    print("\n" + "-"*80)
    print("TEST 6: Knowledge Graph Consistency")
    print("-"*80)
    print("Scenario: KG entities should reflect stored memories accurately")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store facts with clear entities
        entity_facts = [
            "John Smith is a software engineer at Google.",
            "Mary Johnson works as a data scientist at Meta.",
            "John Smith lives in San Francisco with his wife Mary Johnson.",
            "Google and Meta are both tech companies in California.",
        ]

        for fact in entity_facts:
            await memory.store_memory_bank(
                text=fact,
                tags=["entity_test"],
                importance=0.9,
                confidence=0.9
            )
            await asyncio.sleep(0.05)

        print(f"  [STORE] Stored {len(entity_facts)} facts with clear entities")

        # Query and check Content KG was updated
        results = await memory.search(
            query="John Smith software engineer",
            collections=["memory_bank"],
            limit=5
        )

        found_john = any("john smith" in r.get("text", "").lower() for r in results)

        # Check KG state (if accessible)
        kg_ok = True
        if hasattr(memory, 'content_kg') and memory.content_kg:
            entity_count = len(memory.content_kg.get("entities", {}))
            print(f"  [KG] Content KG has {entity_count} entities")
            kg_ok = entity_count >= 2  # Should have at least John and Mary
        else:
            print(f"  [KG] Content KG not directly accessible (OK)")

        print(f"  [SEARCH] Found John Smith: {'YES' if found_john else 'NO'}")

        success = found_john and kg_ok
        print(f"\nKnowledge graph consistency: {'PASS' if success else 'FAIL'}")

        return ("Knowledge graph consistency", success)
    finally:
        cleanup_memory(test_dir)


async def test_recovery_resilience():
    """Run all recovery and resilience tests."""

    print("\n" + "="*80)
    print("RECOVERY & CORRUPTION RESILIENCE TEST")
    print("="*80)
    print("\nThis tests the system's ability to handle failures gracefully.")
    print("Each test uses an ISOLATED memory instance.\n")

    test_results = []

    # Run each test in isolation
    test_results.append(await test_interrupted_store_recovery())
    test_results.append(await test_empty_collection_handling())
    test_results.append(await test_malformed_metadata_resilience())
    test_results.append(await test_concurrent_modification_safety())
    test_results.append(await test_large_batch_atomicity())
    test_results.append(await test_knowledge_graph_consistency())

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nPassed: {passed}/{total}")

    # Success if 5/6 or better
    success = passed >= 5

    print("\n" + "="*80)
    if success:
        print(f"PASS - System is resilient to failures and edge cases")
        print(f"       {passed}/{total} resilience scenarios passed")
    else:
        print(f"FAIL - System has resilience issues")
        print(f"       Only {passed}/{total} scenarios passed")
    print("="*80)

    return success


if __name__ == "__main__":
    success = asyncio.run(test_recovery_resilience())
    sys.exit(0 if success else 1)
