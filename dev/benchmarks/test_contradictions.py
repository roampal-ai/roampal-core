"""
Memory Contradiction Test
=========================

Tests how the system handles CONFLICTING information stored over time.

This is critical for a personal memory system because:
1. Facts change (user moved, got married, changed jobs)
2. LLM may store incorrect info that gets corrected later
3. Same fact may be stated differently at different times
4. User may deliberately update preferences

The system should:
- Prioritize higher-quality (importance x confidence) facts
- Recognize when facts conflict
- Not return both "lives in NYC" and "lives in SF" equally

Test scenarios:
1. Direct contradictions (X is A vs X is B)
2. Temporal updates (was A, now is B)
3. Partial contradictions (likes coffee vs hates caffeine)
4. Confidence conflicts (certain A vs uncertain B)
5. Many-to-one conflicts (multiple wrong answers vs one right)

ADAPTED FOR ROAMPAL CORE
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import sys
import uuid
import shutil
from pathlib import Path
from datetime import datetime

# Add roampal-core root to path
core_root = Path(__file__).parent.parent
sys.path.insert(0, str(core_root))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem


async def create_fresh_memory():
    """Create a fresh memory instance with clean database."""
    test_dir = Path(__file__).parent / f"test_data_contradictions_{uuid.uuid4().hex[:8]}"
    memory = UnifiedMemorySystem(data_path=str(test_dir))
    await memory.initialize()
    return memory, str(test_dir)


def cleanup_memory(test_dir: str):
    """Clean up test data directory."""
    if Path(test_dir).exists():
        try:
            shutil.rmtree(test_dir)
        except:
            pass


async def test_direct_contradiction():
    """TEST 1: Direct Contradiction - Same fact, different values"""
    print("-"*80)
    print("TEST 1: Direct Contradiction")
    print("-"*80)
    print("Scenario: Two memories say different things about same fact")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store contradicting facts with different quality
        await memory.store_memory_bank(
            text="User's favorite color is blue.",
            tags=["preference", "color"],
            importance=0.3,  # Low - casual mention
            confidence=0.4
        )
        await asyncio.sleep(0.05)

        await memory.store_memory_bank(
            text="User's favorite color is definitely green, they mentioned it multiple times.",
            tags=["preference", "color", "verified"],
            importance=0.9,  # High - verified
            confidence=0.9
        )

        results = await memory.search(
            query="What is the user's favorite color?",
            collections=["memory_bank"],
            limit=3
        )

        top_text = results[0].get("text", "") if results else ""
        green_first = "green" in top_text.lower()

        print(f"\nLow quality says: blue (q=0.12)")
        print(f"High quality says: green (q=0.81)")
        print(f"\nTop result: {top_text[:60]}...")
        print(f"Correct answer (green) ranked #1: {'YES' if green_first else 'NO'}")

        return ("Direct contradiction", green_first)
    finally:
        cleanup_memory(test_dir)


async def test_many_wrong_vs_one_right():
    """TEST 2: Many Wrong vs One Right (5:1 noise)"""
    print("\n" + "-"*80)
    print("TEST 2: Many Wrong vs One Right (5:1 noise)")
    print("-"*80)
    print("Scenario: 5 low-quality wrong answers, 1 high-quality right answer")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store 5 wrong answers (low quality)
        wrong_answers = [
            "User works as a teacher.",
            "User's job is being a nurse.",
            "User is employed as a lawyer.",
            "User works in construction.",
            "User is a chef by profession.",
        ]
        for wrong in wrong_answers:
            await memory.store_memory_bank(
                text=wrong,
                tags=["job", "uncertain"],
                importance=0.3,
                confidence=0.3
            )
            await asyncio.sleep(0.02)

        # Store 1 right answer (high quality)
        await memory.store_memory_bank(
            text="User is a software engineer at a tech startup, confirmed in multiple conversations.",
            tags=["job", "verified"],
            importance=0.95,
            confidence=0.95
        )

        results = await memory.search(
            query="What does the user do for work?",
            collections=["memory_bank"],
            limit=10
        )

        top_text = results[0].get("text", "") if results else ""
        correct_first = "software engineer" in top_text.lower()

        print(f"\n5 wrong answers (q=0.09 each): teacher, nurse, lawyer, construction, chef")
        print(f"1 right answer (q=0.90): software engineer")
        print(f"\n[DEBUG] Full ranking:")
        for i, r in enumerate(results[:7]):
            text = r.get("text", "")[:45]
            meta = r.get("metadata", {})
            imp = meta.get("importance", 0.5)
            conf = meta.get("confidence", 0.5)
            q = imp * conf
            score = r.get("final_rank_score", 0)
            emb_sim = r.get("embedding_similarity", 0)
            print(f"  {i+1}. score={score:.3f} q={q:.2f} emb_sim={emb_sim:.3f} | {text}...")
        print(f"\nCorrect answer ranked #1: {'YES' if correct_first else 'NO'}")

        return ("5:1 wrong vs right", correct_first)
    finally:
        cleanup_memory(test_dir)


async def test_temporal_update():
    """TEST 3: Temporal Update (Old fact vs New fact)"""
    print("\n" + "-"*80)
    print("TEST 3: Temporal Update")
    print("-"*80)
    print("Scenario: Old fact vs updated fact (user moved)")

    memory, test_dir = await create_fresh_memory()

    try:
        await memory.store_memory_bank(
            text="User lives in Boston.",
            tags=["location", "old"],
            importance=0.5,
            confidence=0.5
        )
        await asyncio.sleep(0.05)

        await memory.store_memory_bank(
            text="User recently relocated to Seattle for a new job opportunity.",
            tags=["location", "current"],
            importance=0.9,
            confidence=0.9
        )

        results = await memory.search(
            query="Where does the user live?",
            collections=["memory_bank"],
            limit=5
        )

        top_text = results[0].get("text", "") if results else ""
        seattle_first = "seattle" in top_text.lower()

        print(f"\nOld location (q=0.25): Boston")
        print(f"Current location (q=0.81): Seattle")
        print(f"\n[DEBUG] Full ranking:")
        for i, r in enumerate(results[:3]):
            text = r.get("text", "")[:50]
            meta = r.get("metadata", {})
            q = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
            score = r.get("final_rank_score", 0)
            print(f"  {i+1}. score={score:.3f} q={q:.2f} | {text}...")
        print(f"\nCurrent location ranked #1: {'YES' if seattle_first else 'NO'}")

        return ("Temporal update", seattle_first)
    finally:
        cleanup_memory(test_dir)


async def test_confidence_conflict():
    """TEST 4: Confidence Conflict"""
    print("\n" + "-"*80)
    print("TEST 4: Confidence Conflict")
    print("-"*80)
    print("Scenario: Uncertain guess vs confident statement")

    memory, test_dir = await create_fresh_memory()

    try:
        await memory.store_memory_bank(
            text="User might be vegetarian, not sure though.",
            tags=["diet", "uncertain"],
            importance=0.7,  # Seems important
            confidence=0.2   # But very uncertain
        )
        await asyncio.sleep(0.05)

        await memory.store_memory_bank(
            text="User confirmed they eat meat and especially love steak.",
            tags=["diet", "confirmed"],
            importance=0.8,
            confidence=0.95
        )

        results = await memory.search(
            query="Is the user vegetarian?",
            collections=["memory_bank"],
            limit=5
        )

        top_text = results[0].get("text", "") if results else ""
        meat_first = "meat" in top_text.lower() or "steak" in top_text.lower()

        print(f"\nUncertain (q=0.14): might be vegetarian")
        print(f"Confident (q=0.76): loves steak")
        print(f"\n[DEBUG] Full ranking:")
        for i, r in enumerate(results[:3]):
            text = r.get("text", "")[:50]
            meta = r.get("metadata", {})
            q = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
            score = r.get("final_rank_score", 0)
            print(f"  {i+1}. score={score:.3f} q={q:.2f} | {text}...")
        print(f"\nConfident answer ranked #1: {'YES' if meat_first else 'NO'}")

        return ("Confidence conflict", meat_first)
    finally:
        cleanup_memory(test_dir)


async def test_implicit_contradiction():
    """TEST 5: Partial/Implicit Contradiction"""
    print("\n" + "-"*80)
    print("TEST 5: Partial/Implicit Contradiction")
    print("-"*80)
    print("Scenario: Statements that indirectly conflict")

    memory, test_dir = await create_fresh_memory()

    try:
        await memory.store_memory_bank(
            text="User hates anything with caffeine.",
            tags=["preference", "drinks"],
            importance=0.4,
            confidence=0.4
        )
        await asyncio.sleep(0.05)

        await memory.store_memory_bank(
            text="User drinks 3 cups of coffee every morning and loves espresso.",
            tags=["preference", "drinks", "verified"],
            importance=0.9,
            confidence=0.9
        )

        results = await memory.search(
            query="Does the user like coffee?",
            collections=["memory_bank"],
            limit=5
        )

        top_text = results[0].get("text", "") if results else ""
        coffee_lover_first = "coffee" in top_text.lower() and "loves" in top_text.lower()

        print(f"\nContradicts (q=0.16): hates caffeine")
        print(f"Direct (q=0.81): loves coffee")
        print(f"\n[DEBUG] Full ranking:")
        for i, r in enumerate(results[:3]):
            text = r.get("text", "")[:50]
            meta = r.get("metadata", {})
            q = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
            score = r.get("final_rank_score", 0)
            print(f"  {i+1}. score={score:.3f} q={q:.2f} | {text}...")
        print(f"\nCoffee-lover answer ranked #1: {'YES' if coffee_lover_first else 'NO'}")

        return ("Implicit contradiction", coffee_lover_first)
    finally:
        cleanup_memory(test_dir)


async def test_contradictions():
    """Run all contradiction tests with isolated memory instances."""

    print("\n" + "="*80)
    print("MEMORY CONTRADICTION TEST")
    print("="*80)
    print("\nThis tests whether the system correctly handles conflicting facts")
    print("and prioritizes the most reliable information.")
    print("Each test uses an ISOLATED memory instance.\n")

    test_results = []

    # Run each test in isolation
    test_results.append(await test_direct_contradiction())
    test_results.append(await test_many_wrong_vs_one_right())
    test_results.append(await test_temporal_update())
    test_results.append(await test_confidence_conflict())
    test_results.append(await test_implicit_contradiction())

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

    # Success if 4/5 or better (we allow one failure for edge cases)
    success = passed >= 4

    print("\n" + "="*80)
    if success:
        print(f"PASS - System correctly prioritizes high-quality information")
        print(f"       {passed}/{total} contradiction scenarios handled correctly")
    else:
        print(f"FAIL - System struggling with contradictions")
        print(f"       Only {passed}/{total} scenarios correct")
    print("="*80)

    return success


if __name__ == "__main__":
    success = asyncio.run(test_contradictions())
    sys.exit(0 if success else 1)
