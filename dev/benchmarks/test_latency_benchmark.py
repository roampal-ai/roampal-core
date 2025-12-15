import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

#!/usr/bin/env python3
"""
Latency Benchmark for Roampal Memory System

Tests search performance under realistic load:
- p50, p95, p99 latency metrics
- Multiple collection sizes (10, 50, 100, 500 memories)
- Real search queries

ADAPTED FOR ROAMPAL CORE

Usage:
    cd benchmarks
    python test_latency_benchmark.py
"""

import asyncio
import sys
import time
from pathlib import Path
import statistics
import shutil

# Add roampal-core root to path
core_root = Path(__file__).parent.parent
sys.path.insert(0, str(core_root))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem


async def benchmark_search_latency():
    """Benchmark search latency under realistic load."""

    print("=" * 80)
    print("ROAMPAL CORE - LATENCY BENCHMARK")
    print("=" * 80)
    print()

    # Initialize system - Core uses data_path
    test_data_dir = Path(__file__).parent / "test_latency_data"
    print("[1/5] Initializing memory system...")
    system = UnifiedMemorySystem(data_path=str(test_data_dir))
    await system.initialize()
    print("OK System initialized")
    print()

    # Test configurations (benchmark mode bypasses 500 limit)
    test_sizes = [10, 50, 100, 500]
    queries_per_test = 100

    all_results = {}

    topics = [
        "python programming", "machine learning", "web development",
        "data science", "system design", "cloud computing",
        "cybersecurity", "mobile development", "devops", "databases"
    ]

    for size in test_sizes:
        print(f"[Testing with {size} memories]")
        print("-" * 80)

        # Create fresh system for each size (Core doesn't have delete_all)
        size_test_dir = test_data_dir / f"size_{size}"
        system = UnifiedMemorySystem(data_path=str(size_test_dir))
        await system.initialize()

        # Seed memories
        print(f"  Seeding {size} memories...", end=" ", flush=True)
        seed_start = time.perf_counter()

        for i in range(size):
            topic = topics[i % len(topics)]
            await system.store_memory_bank(
                text=f"Memory #{i}: {topic} tutorial explaining advanced concepts and best practices",
                tags=[topic, "tutorial"],
                importance=0.8,
                confidence=0.9
            )

        seed_time = (time.perf_counter() - seed_start) * 1000
        print(f"OK ({seed_time:.1f}ms)")

        # Run search queries
        print(f"  Running {queries_per_test} search queries...", end=" ", flush=True)
        latencies = []

        queries = [
            "python programming tutorial",
            "machine learning basics",
            "web development guide",
            "data science examples",
            "system design patterns",
            "cloud computing architecture",
            "cybersecurity best practices",
            "mobile app development",
            "devops automation",
            "database optimization"
        ]

        for i in range(queries_per_test):
            query = queries[i % len(queries)]

            start = time.perf_counter()
            results = await system.search(query, collections=["memory_bank"], limit=5)
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        print("OK")

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies_sorted)
        p95_idx = int(len(latencies_sorted) * 0.95)
        p95 = latencies_sorted[p95_idx]
        p99_idx = int(len(latencies_sorted) * 0.99)
        p99 = latencies_sorted[p99_idx]
        avg = statistics.mean(latencies)

        all_results[size] = {
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "avg": avg,
            "min": min(latencies),
            "max": max(latencies)
        }

        print(f"  Results:")
        print(f"    p50:  {p50:6.2f}ms")
        print(f"    p95:  {p95:6.2f}ms")
        print(f"    p99:  {p99:6.2f}ms")
        print(f"    avg:  {avg:6.2f}ms")
        print(f"    min:  {min(latencies):6.2f}ms")
        print(f"    max:  {max(latencies):6.2f}ms")
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Memories':<12} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12} {'avg (ms)':<12}")
    print("-" * 80)

    for size in test_sizes:
        r = all_results[size]
        print(f"{size:<12} {r['p50']:<12.2f} {r['p95']:<12.2f} {r['p99']:<12.2f} {r['avg']:<12.2f}")

    print()

    # Check targets
    print("=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()

    target_size = 100  # Standard benchmark size
    if target_size in all_results:
        r = all_results[target_size]

        print(f"At {target_size} memories:")
        print(f"  p50: {r['p50']:.2f}ms")
        print(f"  p95: {r['p95']:.2f}ms")
        print(f"  p99: {r['p99']:.2f}ms")
        print()
        print("Note: Performance is hardware-dependent.")
        print("      These results validate search latency is reasonable.")

    print()
    print("=" * 80)

    # Cleanup
    shutil.rmtree(test_data_dir, ignore_errors=True)

    return all_results


if __name__ == "__main__":
    asyncio.run(benchmark_search_latency())
