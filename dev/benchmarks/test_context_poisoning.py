"""
Context Poisoning Test
======================

Tests whether adversarial or noisy data can CORRUPT the system's ability
to retrieve good memories.

"Context poisoning" is when bad data actively degrades the quality of
retrieval, not just by crowding out good data, but by confusing the
system's understanding of what's relevant.

Real-world scenarios:
1. Malicious user deliberately injects confusing data
2. LLM hallucinates incorrect but plausible-sounding facts
3. OCR errors from scanned documents introduce noise
4. Copy-paste errors introduce duplicate/corrupted text
5. Outdated API responses cached with current data

Test scenarios:
1. Exact duplicate poisoning (same text, different quality)
2. Near-duplicate confusion (slight variations)
3. Entity confusion (similar names, different people)
4. Temporal confusion (same event, different dates)
5. Negation poisoning (X is true vs X is NOT true)
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import sys
import uuid
import shutil
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem


async def create_fresh_memory():
    """Create a fresh memory instance with clean database."""
    test_dir = f"test_data_poisoning_{uuid.uuid4().hex[:8]}"
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


async def test_exact_duplicate_poisoning():
    """TEST 1: System handles exact duplicates with different qualities"""
    print("-"*80)
    print("TEST 1: Exact Duplicate Poisoning")
    print("-"*80)
    print("Scenario: Same text stored multiple times with varying quality")
    print("          System should deduplicate and keep highest quality version")

    memory, test_dir = await create_fresh_memory()

    try:
        base_text = "User's birthday is March 15, 1990."

        # Store low-quality version first
        await memory.store_memory_bank(
            text=base_text,
            tags=["personal", "uncertain"],
            importance=0.3,
            confidence=0.3
        )
        await asyncio.sleep(0.05)

        # Store high-quality version
        await memory.store_memory_bank(
            text=base_text,
            tags=["personal", "verified"],
            importance=0.95,
            confidence=0.95
        )
        await asyncio.sleep(0.05)

        # Store medium-quality version
        await memory.store_memory_bank(
            text=base_text,
            tags=["personal"],
            importance=0.6,
            confidence=0.6
        )

        print("  [POISON] Stored same text 3 times: q=0.09, q=0.90, q=0.36")

        # Query and check results
        results = await memory.search(
            query="When is the user's birthday?",
            collections=["memory_bank"],
            limit=5
        )

        # Count how many versions of the birthday fact appear
        birthday_results = [r for r in results if "march 15" in r.get("text", "").lower()]

        # Should either dedupe to 1, or if multiple, highest quality should be #1
        if len(birthday_results) == 1:
            print(f"  [DEDUP] System correctly deduplicated to 1 result")
            meta = birthday_results[0].get("metadata", {})
            quality = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
            kept_high_quality = quality > 0.8
            print(f"  [QUALITY] Kept version has quality={quality:.2f}")
        else:
            print(f"  [MULTI] System returned {len(birthday_results)} versions")
            # Check if highest quality is first
            if birthday_results:
                meta = birthday_results[0].get("metadata", {})
                quality = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
                kept_high_quality = quality > 0.8
                print(f"  [QUALITY] Top version has quality={quality:.2f}")
            else:
                kept_high_quality = False

        success = kept_high_quality
        print(f"\nHigh-quality version prioritized: {'YES' if success else 'NO'}")

        return ("Exact duplicate handling", success)
    finally:
        cleanup_memory(test_dir)


async def test_near_duplicate_confusion():
    """TEST 2: System handles near-duplicates (slight variations)"""
    print("\n" + "-"*80)
    print("TEST 2: Near-Duplicate Confusion")
    print("-"*80)
    print("Scenario: Similar but not identical facts compete for retrieval")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store several near-duplicates with varying quality
        variants = [
            ("User's favorite color is blue.", 0.3, 0.3),
            ("The user's fav color is blue!", 0.35, 0.35),
            ("User prefers the color blue.", 0.4, 0.4),
            ("User said their favorite color is green (verified in 3 conversations).", 0.95, 0.95),
            ("Users favorite colour is blue", 0.25, 0.25),  # typo variant
        ]

        for text, imp, conf in variants:
            await memory.store_memory_bank(
                text=text,
                tags=["preference"],
                importance=imp,
                confidence=conf
            )
            await asyncio.sleep(0.02)

        print("  [POISON] Stored 4 low-quality 'blue' variants + 1 high-quality 'green'")

        results = await memory.search(
            query="What is the user's favorite color?",
            collections=["memory_bank"],
            limit=5
        )

        # The high-quality GREEN should win over all the low-quality BLUE variants
        top_text = results[0].get("text", "") if results else ""
        green_first = "green" in top_text.lower()

        print(f"\n[DEBUG] Top 3 results:")
        for i, r in enumerate(results[:3]):
            text = r.get("text", "")[:50]
            meta = r.get("metadata", {})
            q = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
            score = r.get("final_rank_score", 0)
            print(f"  {i+1}. score={score:.3f} q={q:.2f} | {text}...")

        print(f"\nHigh-quality (green) beats low-quality noise (blue): {'YES' if green_first else 'NO'}")

        return ("Near-duplicate handling", green_first)
    finally:
        cleanup_memory(test_dir)


async def test_entity_confusion():
    """TEST 3: System distinguishes between similar but different entities"""
    print("\n" + "-"*80)
    print("TEST 3: Entity Confusion Attack")
    print("-"*80)
    print("Scenario: Facts about similar names should not confuse retrieval")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store facts about the REAL person (high quality)
        real_person_facts = [
            "John Smith (the user's manager) has 15 years of experience.",
            "John Smith manages the engineering team at TechCorp.",
            "John Smith's office is on the 5th floor.",
        ]
        for fact in real_person_facts:
            await memory.store_memory_bank(
                text=fact,
                tags=["work", "manager", "verified"],
                importance=0.9,
                confidence=0.9
            )
            await asyncio.sleep(0.02)

        # Store CONFUSING facts about different John Smiths (low quality)
        confusing_facts = [
            "John Smith the actor was born in 1945.",
            "John Smith (some random guy) likes pizza.",
            "A John Smith was mentioned in a news article.",
            "John Smith from accounting quit last month.",
            "John Smith, the historical figure, was a colonist.",
        ]
        for fact in confusing_facts:
            await memory.store_memory_bank(
                text=fact,
                tags=["random", "unverified"],
                importance=0.3,
                confidence=0.3
            )
            await asyncio.sleep(0.02)

        print("  [REAL] Stored 3 high-quality facts about user's manager John Smith")
        print("  [NOISE] Stored 5 low-quality facts about OTHER John Smiths")

        # Query specifically about the manager
        results = await memory.search(
            query="Who is John Smith the user's manager?",
            collections=["memory_bank"],
            limit=5
        )

        # Check if real manager facts are top-ranked
        top_text = results[0].get("text", "") if results else ""
        manager_first = "manager" in top_text.lower() or "techcorp" in top_text.lower()

        print(f"\n[DEBUG] Top 3 results:")
        for i, r in enumerate(results[:3]):
            text = r.get("text", "")[:55]
            meta = r.get("metadata", {})
            q = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
            score = r.get("final_rank_score", 0)
            print(f"  {i+1}. score={score:.3f} q={q:.2f} | {text}...")

        print(f"\nCorrect John Smith (manager) ranked #1: {'YES' if manager_first else 'NO'}")

        return ("Entity confusion resistance", manager_first)
    finally:
        cleanup_memory(test_dir)


async def test_temporal_confusion():
    """TEST 4: System handles conflicting temporal information"""
    print("\n" + "-"*80)
    print("TEST 4: Temporal Confusion Attack")
    print("-"*80)
    print("Scenario: Same event with different dates should not confuse system")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store WRONG dates (low quality)
        wrong_dates = [
            "User got married on June 5, 2018.",
            "User's wedding was in 2017.",
            "User got married sometime in spring 2019.",
            "The wedding happened in December 2018.",
        ]
        for fact in wrong_dates:
            await memory.store_memory_bank(
                text=fact,
                tags=["personal", "unverified"],
                importance=0.3,
                confidence=0.25
            )
            await asyncio.sleep(0.02)

        # Store CORRECT date (high quality, verified)
        await memory.store_memory_bank(
            text="User got married on September 14, 2019 (confirmed from wedding photos).",
            tags=["personal", "verified", "milestone"],
            importance=0.95,
            confidence=0.98
        )

        print("  [WRONG] Stored 4 incorrect wedding dates (q=0.08 each)")
        print("  [CORRECT] Stored 1 verified wedding date (q=0.93)")

        results = await memory.search(
            query="When did the user get married?",
            collections=["memory_bank"],
            limit=5
        )

        top_text = results[0].get("text", "") if results else ""
        correct_date_first = "september 14, 2019" in top_text.lower()

        print(f"\n[DEBUG] Top 3 results:")
        for i, r in enumerate(results[:3]):
            text = r.get("text", "")[:55]
            meta = r.get("metadata", {})
            q = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
            score = r.get("final_rank_score", 0)
            print(f"  {i+1}. score={score:.3f} q={q:.2f} | {text}...")

        print(f"\nCorrect date (Sept 14, 2019) ranked #1: {'YES' if correct_date_first else 'NO'}")

        return ("Temporal confusion resistance", correct_date_first)
    finally:
        cleanup_memory(test_dir)


async def test_negation_poisoning():
    """TEST 5: System handles negated facts correctly"""
    print("\n" + "-"*80)
    print("TEST 5: Negation Poisoning Attack")
    print("-"*80)
    print("Scenario: 'X is true' vs 'X is NOT true' - system must prioritize quality")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store FALSE negated facts (low quality)
        false_facts = [
            "User is NOT allergic to shellfish.",
            "User has no shellfish allergy.",
            "User can eat shellfish without problems.",
        ]
        for fact in false_facts:
            await memory.store_memory_bank(
                text=fact,
                tags=["health", "unverified"],
                importance=0.35,
                confidence=0.3
            )
            await asyncio.sleep(0.02)

        # Store TRUE fact (high quality - critical health info!)
        await memory.store_memory_bank(
            text="User IS allergic to shellfish - carries EpiPen (doctor confirmed).",
            tags=["health", "allergy", "critical", "verified"],
            importance=1.0,  # Maximum importance - life-threatening
            confidence=0.99
        )

        print("  [DANGER] Stored 3 false 'NOT allergic' facts (q=0.10)")
        print("  [TRUTH] Stored 1 verified 'IS allergic' fact (q=0.99)")

        results = await memory.search(
            query="Does the user have a shellfish allergy?",
            collections=["memory_bank"],
            limit=5
        )

        top_text = results[0].get("text", "") if results else ""
        # True allergy fact should be #1 (contains "IS allergic" or "EpiPen")
        truth_first = ("is allergic" in top_text.lower() and "not" not in top_text.lower()) or "epipen" in top_text.lower()

        print(f"\n[DEBUG] Top 3 results:")
        for i, r in enumerate(results[:3]):
            text = r.get("text", "")[:55]
            meta = r.get("metadata", {})
            q = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
            score = r.get("final_rank_score", 0)
            print(f"  {i+1}. score={score:.3f} q={q:.2f} | {text}...")

        print(f"\nTrue allergy fact (critical!) ranked #1: {'YES' if truth_first else 'NO'}")

        return ("Negation poisoning resistance", truth_first)
    finally:
        cleanup_memory(test_dir)


async def test_context_poisoning():
    """Run all context poisoning tests."""

    print("\n" + "="*80)
    print("CONTEXT POISONING TEST")
    print("="*80)
    print("\nThis tests whether adversarial/noisy data can corrupt retrieval.")
    print("Each test uses an ISOLATED memory instance.\n")

    test_results = []

    # Run each test in isolation
    test_results.append(await test_exact_duplicate_poisoning())
    test_results.append(await test_near_duplicate_confusion())
    test_results.append(await test_entity_confusion())
    test_results.append(await test_temporal_confusion())
    test_results.append(await test_negation_poisoning())

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

    # Success if 4/5 or better
    success = passed >= 4

    print("\n" + "="*80)
    if success:
        print(f"PASS - System is resistant to context poisoning attacks")
        print(f"       {passed}/{total} poisoning scenarios handled correctly")
    else:
        print(f"FAIL - System vulnerable to context poisoning")
        print(f"       Only {passed}/{total} scenarios passed")
    print("="*80)

    return success


if __name__ == "__main__":
    success = asyncio.run(test_context_poisoning())
    sys.exit(0 if success else 1)
