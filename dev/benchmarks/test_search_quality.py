"""
Search Quality Test Suite
=========================

Tests the quality and effectiveness of the search system beyond basic functionality.

These tests expose potential improvements in:
1. Query understanding (synonyms, acronyms, typos)
2. Ranking quality (relevance vs recency vs quality)
3. Diversity (avoiding redundant results)
4. Edge cases in retrieval

This isn't a pass/fail test suite - it's a diagnostic that shows WHERE the search
could be improved and measures current baselines.
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import sys
import uuid
import shutil
from pathlib import Path
from collections import Counter

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem


async def create_fresh_memory():
    """Create a fresh memory instance with clean database."""
    test_dir = f"test_data_search_quality_{uuid.uuid4().hex[:8]}"
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


async def test_synonym_understanding():
    """TEST 1: Can the system understand synonyms?"""
    print("-"*80)
    print("TEST 1: Synonym Understanding")
    print("-"*80)
    print("Scenario: Store facts with one word, query with synonyms")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store facts using specific words
        facts = [
            ("User drives a red automobile to work every day.", "car"),
            ("User's canine companion is a golden retriever named Max.", "dog"),
            ("User resides in a two-bedroom apartment in downtown.", "lives"),
            ("User is employed as a software developer.", "works"),
            ("User's spouse is named Sarah.", "wife/husband"),
        ]

        for fact, _ in facts:
            await memory.store_memory_bank(
                text=fact,
                tags=["synonym_test"],
                importance=0.8,
                confidence=0.8
            )
            await asyncio.sleep(0.02)

        print(f"  [STORE] Stored {len(facts)} facts")

        # Query with synonyms
        synonym_queries = [
            ("What car does the user drive?", "automobile", "car query for 'automobile'"),
            ("Does the user have a dog?", "canine", "dog query for 'canine'"),
            ("Where does the user live?", "resides", "lives query for 'resides'"),
            ("What is the user's job?", "employed", "job query for 'employed'"),
            ("Who is the user's wife?", "spouse", "wife query for 'spouse'"),
        ]

        successes = 0
        for query, stored_word, description in synonym_queries:
            results = await memory.search(
                query=query,
                collections=["memory_bank"],
                limit=3
            )

            found = any(stored_word in r.get("text", "").lower() for r in results)
            if found:
                successes += 1
                print(f"  [OK] {description}")
            else:
                print(f"  [MISS] {description}")

        accuracy = successes / len(synonym_queries) * 100
        print(f"\n  Synonym understanding: {successes}/{len(synonym_queries)} ({accuracy:.0f}%)")

        return ("Synonym understanding", accuracy)
    finally:
        cleanup_memory(test_dir)


async def test_typo_tolerance():
    """TEST 2: Can the system handle typos in queries?"""
    print("\n" + "-"*80)
    print("TEST 2: Typo Tolerance")
    print("-"*80)
    print("Scenario: Query with common typos/misspellings")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store facts with correct spelling
        await memory.store_memory_bank(
            text="User works at Microsoft as a software engineer.",
            tags=["work"],
            importance=0.9,
            confidence=0.9
        )
        await memory.store_memory_bank(
            text="User's favorite restaurant is The Cheesecake Factory.",
            tags=["preference"],
            importance=0.8,
            confidence=0.8
        )
        await memory.store_memory_bank(
            text="User graduated from Massachusetts Institute of Technology.",
            tags=["education"],
            importance=0.9,
            confidence=0.9
        )

        print(f"  [STORE] Stored 3 facts with correct spelling")

        # Query with typos
        typo_queries = [
            ("Microsft", "microsoft", "Microsft -> Microsoft"),
            ("Cheescake Factory", "cheesecake", "Cheescake -> Cheesecake"),
            ("Masachusetts", "massachusetts", "Masachusetts -> Massachusetts"),
            ("sofware engineer", "software", "sofware -> software"),
            ("restarant", "restaurant", "restarant -> restaurant"),
        ]

        successes = 0
        for typo_query, correct_word, description in typo_queries:
            results = await memory.search(
                query=typo_query,
                collections=["memory_bank"],
                limit=3
            )

            found = any(correct_word in r.get("text", "").lower() for r in results)
            if found:
                successes += 1
                print(f"  [OK] {description}")
            else:
                print(f"  [MISS] {description}")

        accuracy = successes / len(typo_queries) * 100
        print(f"\n  Typo tolerance: {successes}/{len(typo_queries)} ({accuracy:.0f}%)")

        return ("Typo tolerance", accuracy)
    finally:
        cleanup_memory(test_dir)


async def test_acronym_expansion():
    """TEST 3: Can the system match acronyms to full names?"""
    print("\n" + "-"*80)
    print("TEST 3: Acronym Expansion")
    print("-"*80)
    print("Scenario: Query with acronyms, facts have full names (and vice versa)")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store with full names
        await memory.store_memory_bank(
            text="User lives in New York City in a studio apartment.",
            tags=["location"],
            importance=0.9,
            confidence=0.9
        )
        await memory.store_memory_bank(
            text="User works at the National Aeronautics and Space Administration.",
            tags=["work"],
            importance=0.9,
            confidence=0.9
        )
        await memory.store_memory_bank(
            text="User uses the Application Programming Interface for their project.",
            tags=["work"],
            importance=0.8,
            confidence=0.8
        )

        # Also store with acronyms
        await memory.store_memory_bank(
            text="User attended UCLA for undergraduate studies.",
            tags=["education"],
            importance=0.9,
            confidence=0.9
        )

        print(f"  [STORE] Stored 4 facts (mix of full names and acronyms)")

        # Query with acronyms for full names
        acronym_tests = [
            ("Where does the user live in NYC?", "new york city", "NYC -> New York City"),
            ("Does user work at NASA?", "aeronautics", "NASA -> National Aeronautics..."),
            ("User uses API?", "programming interface", "API -> Application Programming Interface"),
            ("Did user attend University of California Los Angeles?", "ucla", "Full name -> UCLA"),
        ]

        successes = 0
        for query, expected_word, description in acronym_tests:
            results = await memory.search(
                query=query,
                collections=["memory_bank"],
                limit=3
            )

            found = any(expected_word.lower() in r.get("text", "").lower() for r in results)
            if found:
                successes += 1
                print(f"  [OK] {description}")
            else:
                print(f"  [MISS] {description}")
                if results:
                    print(f"        Got: {results[0].get('text', '')[:60]}...")

        accuracy = successes / len(acronym_tests) * 100
        print(f"\n  Acronym understanding: {successes}/{len(acronym_tests)} ({accuracy:.0f}%)")

        return ("Acronym expansion", accuracy)
    finally:
        cleanup_memory(test_dir)


async def test_result_diversity():
    """TEST 4: Does search return diverse results or redundant ones?"""
    print("\n" + "-"*80)
    print("TEST 4: Result Diversity")
    print("-"*80)
    print("Scenario: Multiple facts about same topic - search should show variety")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store multiple facts about different aspects of the same entity
        pet_facts = [
            "User has a dog named Max.",
            "Max is a golden retriever breed.",
            "Max is 5 years old.",
            "Max loves playing fetch at the park.",
            "Max's favorite treat is bacon.",
            "Max was adopted from a shelter in 2019.",
            "Max goes to the vet every 6 months.",
            "Max sleeps on a dog bed in the living room.",
        ]

        for fact in pet_facts:
            await memory.store_memory_bank(
                text=fact,
                tags=["pet", "max"],
                importance=0.7,
                confidence=0.7
            )
            await asyncio.sleep(0.02)

        print(f"  [STORE] Stored {len(pet_facts)} facts about Max the dog")

        # Query for pet info
        results = await memory.search(
            query="Tell me about the user's pet",
            collections=["memory_bank"],
            limit=5
        )

        # Check diversity - count unique aspects
        aspects = []
        for r in results:
            text = r.get("text", "").lower()
            if "name" in text or "named" in text:
                aspects.append("name")
            elif "breed" in text or "retriever" in text:
                aspects.append("breed")
            elif "years old" in text or "age" in text:
                aspects.append("age")
            elif "fetch" in text or "play" in text:
                aspects.append("activity")
            elif "treat" in text or "bacon" in text:
                aspects.append("food")
            elif "adopted" in text or "shelter" in text:
                aspects.append("origin")
            elif "vet" in text:
                aspects.append("health")
            elif "sleep" in text or "bed" in text:
                aspects.append("living")
            else:
                aspects.append("other")

        unique_aspects = len(set(aspects))
        diversity_score = unique_aspects / len(results) * 100 if results else 0

        print(f"\n  Results returned: {len(results)}")
        print(f"  Unique aspects covered: {unique_aspects}")
        print(f"  Aspects: {Counter(aspects)}")
        print(f"  Diversity score: {diversity_score:.0f}%")

        return ("Result diversity", diversity_score)
    finally:
        cleanup_memory(test_dir)


async def test_recency_vs_relevance():
    """TEST 5: Balance between recent and relevant results"""
    print("\n" + "-"*80)
    print("TEST 5: Recency vs Relevance Balance")
    print("-"*80)
    print("Scenario: Old highly-relevant fact vs new somewhat-relevant fact")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store old but highly relevant fact
        await memory.store_memory_bank(
            text="User's favorite programming language is Python, which they've used for 10 years.",
            tags=["programming", "preference"],
            importance=0.95,
            confidence=0.95
        )
        await asyncio.sleep(0.1)  # Small delay to ensure different timestamps

        # Store many newer but less relevant facts
        for i in range(10):
            await memory.store_memory_bank(
                text=f"User mentioned something about code snippet {i} today.",
                tags=["programming", "recent"],
                importance=0.5,
                confidence=0.5
            )
            await asyncio.sleep(0.01)

        print(f"  [OLD] Stored 1 old, highly-relevant fact (q=0.90)")
        print(f"  [NEW] Stored 10 new, less-relevant facts (q=0.25 each)")

        # Query for programming preference
        results = await memory.search(
            query="What programming language does the user prefer?",
            collections=["memory_bank"],
            limit=5
        )

        # Check if Python fact is #1
        top_text = results[0].get("text", "") if results else ""
        python_first = "python" in top_text.lower()

        print(f"\n  Top result: {top_text[:60]}...")
        print(f"  Highly-relevant old fact ranked #1: {'YES' if python_first else 'NO'}")

        # Calculate how well relevance beats recency
        python_rank = None
        for i, r in enumerate(results):
            if "python" in r.get("text", "").lower():
                python_rank = i + 1
                break

        if python_rank:
            score = (5 - python_rank + 1) / 5 * 100  # Rank 1 = 100%, Rank 5 = 20%
        else:
            score = 0

        print(f"  Python fact rank: {python_rank if python_rank else 'Not in top 5'}")
        print(f"  Relevance score: {score:.0f}%")

        return ("Recency vs relevance", score)
    finally:
        cleanup_memory(test_dir)


async def test_partial_match_quality():
    """TEST 6: How well does partial matching work?"""
    print("\n" + "-"*80)
    print("TEST 6: Partial Match Quality")
    print("-"*80)
    print("Scenario: Query contains only part of the stored information")

    memory, test_dir = await create_fresh_memory()

    try:
        # Store detailed facts
        await memory.store_memory_bank(
            text="User's full name is Alexander Benjamin Thompson III, born in Boston, Massachusetts.",
            tags=["identity"],
            importance=0.95,
            confidence=0.95
        )
        await memory.store_memory_bank(
            text="User's phone number is +1 (555) 123-4567 extension 890.",
            tags=["contact"],
            importance=0.9,
            confidence=0.9
        )
        await memory.store_memory_bank(
            text="User's email is alex.thompson@example-company.org.",
            tags=["contact"],
            importance=0.9,
            confidence=0.9
        )

        print(f"  [STORE] Stored 3 detailed facts")

        # Query with partial info
        partial_queries = [
            ("Alexander Thompson", "alexander benjamin thompson", "First + Last name"),
            ("555-123", "555", "Partial phone"),
            ("alex.thompson", "alex.thompson", "Email prefix"),
            ("born in Boston", "boston", "Birth city"),
            ("example-company", "example-company", "Email domain"),
        ]

        successes = 0
        for query, expected, description in partial_queries:
            results = await memory.search(
                query=query,
                collections=["memory_bank"],
                limit=3
            )

            found = any(expected.lower() in r.get("text", "").lower() for r in results)
            if found:
                successes += 1
                print(f"  [OK] {description}")
            else:
                print(f"  [MISS] {description}")

        accuracy = successes / len(partial_queries) * 100
        print(f"\n  Partial match accuracy: {successes}/{len(partial_queries)} ({accuracy:.0f}%)")

        return ("Partial match quality", accuracy)
    finally:
        cleanup_memory(test_dir)


async def test_search_quality():
    """Run all search quality diagnostic tests."""

    print("\n" + "="*80)
    print("SEARCH QUALITY DIAGNOSTIC SUITE")
    print("="*80)
    print("\nThis suite measures search quality across multiple dimensions.")
    print("These aren't pass/fail - they establish baselines for improvement.\n")

    test_results = []

    # Run each test
    test_results.append(await test_synonym_understanding())
    test_results.append(await test_typo_tolerance())
    test_results.append(await test_acronym_expansion())
    test_results.append(await test_result_diversity())
    test_results.append(await test_recency_vs_relevance())
    test_results.append(await test_partial_match_quality())

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SEARCH QUALITY SCORECARD")
    print("="*80)

    total_score = 0
    for name, score in test_results:
        bar_length = int(score / 5)  # Scale to 20 chars
        bar = "#" * bar_length + "-" * (20 - bar_length)
        print(f"  {name:25s} [{bar}] {score:.0f}%")
        total_score += score

    avg_score = total_score / len(test_results)

    print(f"\n  Average Search Quality Score: {avg_score:.0f}%")

    print("\n" + "-"*80)
    print("IMPROVEMENT OPPORTUNITIES:")
    print("-"*80)

    # Identify areas for improvement
    for name, score in test_results:
        if score < 60:
            print(f"  [LOW] {name} at {score:.0f}% - needs attention")
        elif score < 80:
            print(f"  [MED] {name} at {score:.0f}% - room for improvement")

    print("\n" + "="*80)
    if avg_score >= 70:
        print(f"BASELINE ESTABLISHED - Average: {avg_score:.0f}%")
        print("Search quality is acceptable, but improvements possible")
    else:
        print(f"NEEDS IMPROVEMENT - Average: {avg_score:.0f}%")
        print("Search quality could benefit from enhancements")
    print("="*80)

    return avg_score >= 50  # Very lenient - this is diagnostic


if __name__ == "__main__":
    success = asyncio.run(test_search_quality())
    sys.exit(0 if success else 1)
