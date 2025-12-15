"""
Stale Data Test
===============

Tests whether the system correctly prioritizes FRESH, HIGH-QUALITY data
over STALE, LOW-QUALITY data when facts change over time.

Real-world scenario: User's information changes, but old memories persist.
The system should surface the most recent, highest-quality information.

Scenario:
- Store OLD facts about user (low quality, created first)
- Store NEW facts about user (high quality, created later)
- OLD and NEW facts CONFLICT (e.g., "lives in NYC" vs "moved to SF")
- System should prioritize NEW facts despite OLD having similar semantics
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem


async def test_stale_data():
    """
    Test: Can the system prioritize fresh data over stale data?

    Timeline:
    1. OLD facts stored (low quality, outdated info)
    2. NEW facts stored (high quality, current info)
    3. Query should return NEW facts, not OLD
    """

    print("\n" + "="*80)
    print("STALE DATA TEST - Fresh vs Outdated Information")
    print("="*80)

    # Initialize system
    memory = UnifiedMemorySystem(
        data_path=str(Path(__file__).parent / "test_data_stale")
    )
    await memory.initialize()

    # =========================================================================
    # PHASE 1: Store OLD (stale) facts - lower quality
    # =========================================================================
    print("\n[PHASE 1] Storing OLD facts (outdated, lower quality)...")

    old_facts = [
        "User lives in New York City and works at a startup.",
        "User's phone number is 555-0100.",
        "User prefers Python 2.7 for all projects.",
        "User is single and not looking to date.",
        "User's favorite restaurant is Joe's Pizza in NYC.",
    ]

    old_ids = []
    for i, fact in enumerate(old_facts):
        doc_id = await memory.store_memory_bank(
            text=fact,
            tags=["user_info", "outdated"],
            importance=0.5,  # Medium - was once reliable
            confidence=0.4   # Lower - outdated info
        )
        old_ids.append(doc_id)
        print(f"  [OLD] {i+1}/5 (q=0.20): {fact[:50]}...")

    # Small delay to ensure different timestamps
    await asyncio.sleep(0.1)

    # =========================================================================
    # PHASE 2: Store NEW (current) facts - higher quality
    # =========================================================================
    print("\n[PHASE 2] Storing NEW facts (current, higher quality)...")

    new_facts = [
        "User recently moved to San Francisco for a new job at a tech company.",
        "User's new phone number is 555-9999.",
        "User now exclusively uses Python 3.11+ with type hints.",
        "User got married last year and has a newborn.",
        "User's new favorite restaurant is Tartine Bakery in SF.",
    ]

    new_ids = []
    for i, fact in enumerate(new_facts):
        doc_id = await memory.store_memory_bank(
            text=fact,
            tags=["user_info", "current"],
            importance=0.95,  # High - verified current info
            confidence=0.95   # High - recently confirmed
        )
        new_ids.append(doc_id)
        print(f"  [NEW] {i+1}/5 (q=0.90): {fact[:50]}...")

    print(f"\n[VERIFY] Stored {len(old_facts)} old + {len(new_facts)} new = {len(old_facts) + len(new_facts)} total")
    print(f"[VERIFY] Quality gap: NEW=0.90 vs OLD=0.20 (4.5x higher)")

    # =========================================================================
    # PHASE 3: Query for conflicting information
    # =========================================================================
    print("\n" + "-"*80)
    print("RUNNING CONFLICT QUERIES")
    print("-"*80)

    test_cases = [
        {
            "query": "Where does the user live?",
            "old_answer": "New York City",
            "new_answer": "San Francisco",
        },
        {
            "query": "What is the user's phone number?",
            "old_answer": "555-0100",
            "new_answer": "555-9999",
        },
        {
            "query": "What Python version does the user prefer?",
            "old_answer": "Python 2.7",
            "new_answer": "Python 3.11+",
        },
        {
            "query": "Is the user married?",
            "old_answer": "single",
            "new_answer": "married",
        },
        {
            "query": "What's the user's favorite restaurant?",
            "old_answer": "Joe's Pizza",
            "new_answer": "Tartine Bakery",
        },
    ]

    new_wins = 0

    for test in test_cases:
        print(f"\n[QUERY] {test['query']}")
        print(f"        OLD answer: {test['old_answer']}")
        print(f"        NEW answer: {test['new_answer']}")

        results = await memory.search(
            query=test["query"],
            collections=["memory_bank"],
            limit=3
        )

        if not results:
            print("        RESULT: No results found")
            continue

        top_result = results[0]
        doc_id = top_result.get('id', '')
        text = top_result.get('text', '')[:60]
        metadata = top_result.get('metadata', {})
        quality = metadata.get('importance', 0.5) * metadata.get('confidence', 0.5)

        is_new = doc_id in new_ids
        is_old = doc_id in old_ids

        if is_new:
            marker = "NEW (correct)"
            new_wins += 1
        elif is_old:
            marker = "OLD (stale)"
        else:
            marker = "UNKNOWN"

        print(f"        RESULT: [{marker}] q={quality:.2f} | {text}...")

        # Show top 3 for context
        for i, r in enumerate(results[:3]):
            rid = r.get('id', '')
            rtext = r.get('text', '')[:40]
            rmeta = r.get('metadata', {})
            rq = rmeta.get('importance', 0.5) * rmeta.get('confidence', 0.5)
            rmarker = "NEW" if rid in new_ids else ("OLD" if rid in old_ids else "???")
            print(f"          {i+1}. [{rmarker}] q={rq:.2f} | {rtext}...")

    # =========================================================================
    # FINAL SCORING
    # =========================================================================

    accuracy = (new_wins / len(test_cases)) * 100

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"NEW (fresh) data ranked #1: {new_wins}/{len(test_cases)} queries ({accuracy:.1f}%)")
    print(f"Quality gap: NEW=0.90 vs OLD=0.20 (4.5x higher)")
    print(f"\n[MECHANISM] Quality-based ranking should prioritize recent, high-confidence data")

    # Success criteria: NEW should win at least 3/5 queries (60%)
    # Note: When OLD is semantically much closer to query, it may still win
    # This is expected - quality boost helps but can't fully overcome semantic gaps
    success = new_wins >= 3

    print("\n" + "="*80)
    if success:
        print(f"PASS - Fresh data prioritized when semantically competitive")
        print(f"       {new_wins}/5 queries returned current information")
        if new_wins < 5:
            print(f"       Note: {5 - new_wins} queries had OLD closer semantically (expected)")
    else:
        print(f"FAIL - Stale data dominating even when quality is much higher")
        print(f"       Quality ranking needs strengthening")
    print("="*80)

    # Cleanup
    import shutil
    for path in ["test_data_stale"]:
        if Path(path).exists():
            try:
                shutil.rmtree(path)
            except:
                pass

    return success


if __name__ == "__main__":
    success = asyncio.run(test_stale_data())
    sys.exit(0 if success else 1)
