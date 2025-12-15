"""
Catastrophic Forgetting Test
============================

Tests whether the system maintains access to OLD knowledge after learning NEW knowledge.

In neural networks, "catastrophic forgetting" occurs when learning new information
destroys previously learned information. This is a critical failure mode for any
memory system that learns continuously.

For a personal memory system, this means:
1. User tells system about their family -> system stores it
2. User talks about work for weeks -> new work memories stored
3. User asks about family -> system should STILL remember, not have "forgotten"

Test scenarios:
1. OLD knowledge retention after adding UNRELATED new knowledge
2. OLD knowledge retention after adding SIMILAR domain new knowledge
3. OLD knowledge access after system receives 100+ new memories
4. Quality scores of OLD knowledge preserved after bulk inserts
5. Cross-domain interference (work memories don't hurt family memories)

Each test validates that old knowledge remains accessible and correctly ranked.
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
    test_dir = f"test_data_forgetting_{uuid.uuid4().hex[:8]}"
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


async def test_unrelated_domain_retention():
    """TEST 1: Old knowledge retained after adding unrelated new knowledge"""
    print("-"*80)
    print("TEST 1: Unrelated Domain Retention")
    print("-"*80)
    print("Scenario: Store FAMILY facts, add COOKING facts, query FAMILY")

    memory, test_dir = await create_fresh_memory()

    try:
        # Phase 1: Store FAMILY knowledge (old)
        family_facts = [
            "User's mother's name is Margaret and she lives in Florida.",
            "User has a younger brother named David who works in finance.",
            "User's grandmother passed away last year at age 92.",
        ]
        family_ids = []
        for fact in family_facts:
            doc_id = await memory.store_memory_bank(
                text=fact,
                tags=["family"],
                importance=0.9,
                confidence=0.9
            )
            family_ids.append(doc_id)
            await asyncio.sleep(0.02)

        print(f"  [OLD] Stored {len(family_facts)} family facts")

        # Query BEFORE adding new domain
        results_before = await memory.search(
            query="Tell me about the user's mother",
            collections=["memory_bank"],
            limit=3
        )
        found_mother_before = any("margaret" in r.get("text", "").lower() for r in results_before)

        # Phase 2: Add UNRELATED COOKING knowledge (new) - 20 facts to create interference
        cooking_facts = [
            f"User's favorite {food} recipe takes {i*10+30} minutes to prepare."
            for i, food in enumerate([
                "pasta", "curry", "stir-fry", "soup", "salad",
                "tacos", "pizza", "risotto", "sushi", "steak",
                "burger", "lasagna", "paella", "ramen", "burrito",
                "kebab", "biryani", "pad thai", "gnocchi", "carbonara"
            ])
        ]
        for fact in cooking_facts:
            await memory.store_memory_bank(
                text=fact,
                tags=["cooking", "recipes"],
                importance=0.7,
                confidence=0.8
            )
            await asyncio.sleep(0.01)

        print(f"  [NEW] Added {len(cooking_facts)} cooking facts (interference)")

        # Query AFTER adding new domain
        results_after = await memory.search(
            query="Tell me about the user's mother",
            collections=["memory_bank"],
            limit=3
        )
        found_mother_after = any("margaret" in r.get("text", "").lower() for r in results_after)

        # Also verify brother is still accessible
        results_brother = await memory.search(
            query="Does the user have any siblings?",
            collections=["memory_bank"],
            limit=3
        )
        found_brother = any("david" in r.get("text", "").lower() or "brother" in r.get("text", "").lower()
                          for r in results_brother)

        print(f"\n  Mother query BEFORE new facts: {'FOUND' if found_mother_before else 'LOST'}")
        print(f"  Mother query AFTER new facts: {'FOUND' if found_mother_after else 'LOST'}")
        print(f"  Brother query AFTER new facts: {'FOUND' if found_brother else 'LOST'}")

        success = found_mother_before and found_mother_after and found_brother
        print(f"\nOld family knowledge retained: {'YES' if success else 'NO'}")

        return ("Unrelated domain retention", success)
    finally:
        cleanup_memory(test_dir)


async def test_same_domain_retention():
    """TEST 2: Old knowledge retained after adding similar domain new knowledge"""
    print("\n" + "-"*80)
    print("TEST 2: Same Domain Retention")
    print("-"*80)
    print("Scenario: Store WORK facts, add MORE WORK facts, query OLD WORK")

    memory, test_dir = await create_fresh_memory()

    try:
        # Phase 1: Store OLD work knowledge (important, high quality)
        old_work_facts = [
            "User's first manager was Jennifer who taught them about leadership.",
            "User started their career at Google as an intern in 2015.",
            "User's first major project was the recommendation engine redesign.",
        ]
        for fact in old_work_facts:
            await memory.store_memory_bank(
                text=fact,
                tags=["work", "career", "old"],
                importance=0.95,  # Very important - formative experiences
                confidence=0.95
            )
            await asyncio.sleep(0.02)

        print(f"  [OLD] Stored {len(old_work_facts)} old work facts (high quality)")

        # Verify OLD facts accessible before new facts
        results_before = await memory.search(
            query="Who was the user's first manager?",
            collections=["memory_bank"],
            limit=3
        )
        found_jennifer_before = any("jennifer" in r.get("text", "").lower() for r in results_before)

        # Phase 2: Add 30 NEW work facts (recent, but lower quality)
        new_work_facts = [
            "User had a meeting about the Q3 roadmap today.",
            "User's current project deadline is next Friday.",
            "User got positive feedback on their presentation.",
            "User's team is growing by two new members.",
            "User attended a workshop on cloud architecture.",
        ] + [
            f"User worked on task {i} for the sprint planning." for i in range(25)
        ]

        for fact in new_work_facts:
            await memory.store_memory_bank(
                text=fact,
                tags=["work", "current"],
                importance=0.6,  # Medium - day-to-day stuff
                confidence=0.7
            )
            await asyncio.sleep(0.01)

        print(f"  [NEW] Added {len(new_work_facts)} new work facts (lower quality)")

        # Query OLD facts after adding new domain data
        results_after = await memory.search(
            query="Who was the user's first manager?",
            collections=["memory_bank"],
            limit=5
        )
        found_jennifer_after = any("jennifer" in r.get("text", "").lower() for r in results_after)

        # Also check if the OLD fact is still ranked highly
        jennifer_rank = None
        for i, r in enumerate(results_after):
            if "jennifer" in r.get("text", "").lower():
                jennifer_rank = i + 1
                break

        results_google = await memory.search(
            query="Where did the user start their career?",
            collections=["memory_bank"],
            limit=5
        )
        found_google = any("google" in r.get("text", "").lower() for r in results_google)

        print(f"\n  Jennifer query BEFORE new facts: {'FOUND' if found_jennifer_before else 'LOST'}")
        print(f"  Jennifer query AFTER new facts: {'FOUND' if found_jennifer_after else 'LOST'}")
        if jennifer_rank:
            print(f"  Jennifer fact rank: #{jennifer_rank}")
        print(f"  Google career start still accessible: {'YES' if found_google else 'NO'}")

        success = found_jennifer_before and found_jennifer_after and found_google
        print(f"\nOld work knowledge retained: {'YES' if success else 'NO'}")

        return ("Same domain retention", success)
    finally:
        cleanup_memory(test_dir)


async def test_bulk_insert_survival():
    """TEST 3: Old knowledge survives bulk insert of 100+ memories"""
    print("\n" + "-"*80)
    print("TEST 3: Bulk Insert Survival")
    print("-"*80)
    print("Scenario: Store KEY facts, bulk insert 100 random facts, verify KEY accessible")

    memory, test_dir = await create_fresh_memory()

    try:
        # Phase 1: Store KEY knowledge that MUST survive
        key_facts = [
            "User's social security number ends in 4567 (store securely).",
            "User is severely allergic to peanuts - carries an EpiPen.",
            "User's emergency contact is their spouse at 555-1234.",
        ]
        key_ids = []
        for fact in key_facts:
            doc_id = await memory.store_memory_bank(
                text=fact,
                tags=["critical", "safety"],
                importance=1.0,  # Maximum importance
                confidence=1.0
            )
            key_ids.append(doc_id)
            await asyncio.sleep(0.02)

        print(f"  [CRITICAL] Stored {len(key_facts)} safety-critical facts")

        # Phase 2: Bulk insert 100 random mundane facts
        print(f"  [BULK] Inserting 100 random facts...")
        for i in range(100):
            await memory.store_memory_bank(
                text=f"User mentioned random fact #{i} about some topic on day {i % 30}.",
                tags=["random", "mundane"],
                importance=0.3,
                confidence=0.4
            )
            if i % 20 == 0:
                print(f"    ... {i}/100")

        print(f"  [BULK] Inserted 100 random facts")

        # Query critical facts
        results_allergy = await memory.search(
            query="Does the user have any allergies?",
            collections=["memory_bank"],
            limit=5
        )
        found_peanut = any("peanut" in r.get("text", "").lower() for r in results_allergy)

        results_emergency = await memory.search(
            query="Who is the user's emergency contact?",
            collections=["memory_bank"],
            limit=5
        )
        found_spouse = any("spouse" in r.get("text", "").lower() or "555-1234" in r.get("text", "")
                         for r in results_emergency)

        # Check ranking - critical facts should be #1
        peanut_rank = None
        for i, r in enumerate(results_allergy):
            if "peanut" in r.get("text", "").lower():
                peanut_rank = i + 1
                break

        print(f"\n  Allergy fact found: {'YES' if found_peanut else 'NO'}")
        if peanut_rank:
            print(f"  Allergy fact rank: #{peanut_rank}")
        print(f"  Emergency contact found: {'YES' if found_spouse else 'NO'}")

        # Success requires critical facts found AND ranked highly (#1 or #2)
        success = found_peanut and found_spouse and (peanut_rank is not None and peanut_rank <= 2)
        print(f"\nCritical knowledge survives bulk insert: {'YES' if success else 'NO'}")

        return ("Bulk insert survival", success)
    finally:
        cleanup_memory(test_dir)


async def test_quality_preservation():
    """TEST 4: Quality scores preserved after bulk operations"""
    print("\n" + "-"*80)
    print("TEST 4: Quality Preservation")
    print("-"*80)
    print("Scenario: Verify high-quality facts maintain ranking advantage over time")
    print("         (with semantically similar competing facts)")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store ONE high-quality fact about MIT
        await memory.store_memory_bank(
            text="User attended MIT and graduated summa cum laude with a Computer Science degree in 2015.",
            tags=["education", "verified"],
            importance=0.95,
            confidence=0.95
        )
        print("  [HIGH] Stored 1 high-quality MIT education fact (q=0.90)")

        # Store 20 low-quality facts that ALSO mention universities
        # (Making them semantically similar, but lower quality)
        universities = [
            "MIT", "Stanford", "Harvard", "Berkeley", "Caltech",
            "Princeton", "Yale", "Columbia", "Chicago", "Duke",
            "Northwestern", "Cornell", "UCLA", "CMU", "Georgia Tech",
            "Michigan", "Penn", "Johns Hopkins", "NYU", "Brown"
        ]
        for i, uni in enumerate(universities):
            await memory.store_memory_bank(
                text=f"User might have attended {uni}, but this is unconfirmed hearsay.",
                tags=["education", "unverified"],
                importance=0.3,
                confidence=0.2  # Very low confidence - just rumors
            )
            await asyncio.sleep(0.01)
        print("  [LOW] Stored 20 low-quality university rumors (q=0.06 each)")

        # Query about MIT specifically - high-quality MIT fact should win
        results = await memory.search(
            query="Did the user attend MIT?",
            collections=["memory_bank"],
            limit=10
        )

        top_text = results[0].get("text", "") if results else ""
        # High quality fact has "summa cum laude" - that distinguishes it
        verified_mit_first = "summa cum laude" in top_text.lower() or "2015" in top_text

        # Show ranking
        print(f"\n[DEBUG] Top 5 results:")
        for i, r in enumerate(results[:5]):
            text = r.get("text", "")[:55]
            meta = r.get("metadata", {})
            q = meta.get("importance", 0.5) * meta.get("confidence", 0.5)
            score = r.get("final_rank_score", 0)
            print(f"  {i+1}. score={score:.3f} q={q:.2f} | {text}...")

        print(f"\nHigh-quality verified MIT fact ranked #1: {'YES' if verified_mit_first else 'NO'}")

        return ("Quality preservation", verified_mit_first)
    finally:
        cleanup_memory(test_dir)


async def test_cross_domain_isolation():
    """TEST 5: Cross-domain memories don't interfere with each other"""
    print("\n" + "-"*80)
    print("TEST 5: Cross-Domain Isolation")
    print("-"*80)
    print("Scenario: Different topics don't degrade each other's retrieval")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store facts in 5 different domains
        domains = {
            "pets": [
                "User has a golden retriever named Max who is 5 years old.",
                "Max loves playing fetch at the park every morning.",
            ],
            "health": [
                "User runs 5 miles every day for cardiovascular health.",
                "User takes vitamin D supplements every morning.",
            ],
            "hobbies": [
                "User plays guitar in a jazz band on weekends.",
                "User is learning to paint watercolors.",
            ],
            "travel": [
                "User's dream destination is visiting the temples of Kyoto.",
                "User has visited 23 countries so far.",
            ],
            "food": [
                "User's favorite cuisine is Thai food, especially pad see ew.",
                "User is trying to eat more vegetables this year.",
            ],
        }

        for domain, facts in domains.items():
            for fact in facts:
                await memory.store_memory_bank(
                    text=fact,
                    tags=[domain],
                    importance=0.85,
                    confidence=0.85
                )
                await asyncio.sleep(0.01)

        total_facts = sum(len(facts) for facts in domains.values())
        print(f"  [ALL] Stored {total_facts} facts across 5 domains")

        # Query each domain and verify correct facts retrieved
        queries = {
            "pets": ("What is the user's dog's name?", "max"),
            "health": ("How much does the user run?", "5 miles"),
            "hobbies": ("What instrument does the user play?", "guitar"),
            "travel": ("Where does the user want to visit?", "kyoto"),
            "food": ("What's the user's favorite cuisine?", "thai"),
        }

        successes = 0
        for domain, (query, keyword) in queries.items():
            results = await memory.search(
                query=query,
                collections=["memory_bank"],
                limit=3
            )
            found = any(keyword.lower() in r.get("text", "").lower() for r in results)
            if found:
                successes += 1
            print(f"  [{domain}] {keyword}: {'FOUND' if found else 'LOST'}")

        success = successes >= 4  # Allow 1 failure
        print(f"\nCross-domain isolation: {successes}/5 domains correctly retrieved")
        print(f"Test {'PASSED' if success else 'FAILED'}")

        return ("Cross-domain isolation", success)
    finally:
        cleanup_memory(test_dir)


async def test_catastrophic_forgetting():
    """Run all catastrophic forgetting tests."""

    print("\n" + "="*80)
    print("CATASTROPHIC FORGETTING TEST")
    print("="*80)
    print("\nThis tests whether learning NEW knowledge destroys OLD knowledge.")
    print("Each test uses an ISOLATED memory instance.\n")

    test_results = []

    # Run each test in isolation
    test_results.append(await test_unrelated_domain_retention())
    test_results.append(await test_same_domain_retention())
    test_results.append(await test_bulk_insert_survival())
    test_results.append(await test_quality_preservation())
    test_results.append(await test_cross_domain_isolation())

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
        print(f"PASS - System does NOT exhibit catastrophic forgetting")
        print(f"       {passed}/{total} knowledge retention scenarios passed")
    else:
        print(f"FAIL - System shows signs of catastrophic forgetting")
        print(f"       Only {passed}/{total} scenarios passed")
    print("="*80)

    return success


if __name__ == "__main__":
    success = asyncio.run(test_catastrophic_forgetting())
    sys.exit(0 if success else 1)
