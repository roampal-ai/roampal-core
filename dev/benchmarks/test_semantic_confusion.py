"""
Semantic Confusion STRESS TEST
==============================

A brutal test that throws maximum confusion at the memory system.

The challenge:
- 3 correct facts about "Sarah" (HIGH quality)
- 47 misleading facts about OTHER people with similar names/attributes (LOW quality)
- All semantically similar - names like Sara, Sarah M., Sandra, etc.
- Some noise has PARTIALLY correct info mixed with wrong details
- 15:1 noise ratio

This simulates real-world confusion where the LLM stored many similar-looking
facts and now needs to retrieve the RIGHT Sarah, not the wrong ones.

The system must use quality-based ranking to cut through the noise.

ADAPTED FOR ROAMPAL CORE
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import sys
from pathlib import Path

# Add roampal-core root to path
core_root = Path(__file__).parent.parent
sys.path.insert(0, str(core_root))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem


async def test_semantic_confusion():
    """
    STRESS TEST: Can the system find the real Sarah among 47 impostors?

    Ground truth (what we're looking for):
    - Sarah Chen, 34 years old, software engineer at TechCorp

    Confusion tactics:
    - Similar names: Sara, Sarah M., Sandra, Sera, etc.
    - Similar ages: 33, 34, 35, etc.
    - Similar jobs: developer, programmer, engineer, etc.
    - Similar companies: TechCo, TechStart, TechCorp Inc, etc.
    - Partially correct combos: "Sarah, 34, works at Google" (wrong company)
    """

    print("\n" + "="*80)
    print("SEMANTIC CONFUSION STRESS TEST - 15:1 Noise Ratio")
    print("="*80)
    print("\nChallenge: Find the REAL Sarah among 47 similar-looking impostors")

    # Initialize system - Core uses data_path instead of data_dir
    test_data_dir = Path(__file__).parent / "test_data_semantic_confusion"
    memory = UnifiedMemorySystem(data_path=str(test_data_dir))
    await memory.initialize()

    # =========================================================================
    # GROUND TRUTH - The correct facts (HIGH QUALITY)
    # =========================================================================
    print("\n[SETUP] Storing 3 ground truth facts about Sarah Chen (HIGH QUALITY)...")

    ground_truth = [
        "Sarah Chen is 34 years old and works as a software engineer at TechCorp.",
        "Sarah Chen's email is sarah.chen@techcorp.com and she lives in San Francisco.",
        "Sarah Chen graduated from Stanford with a CS degree and has been at TechCorp for 5 years."
    ]

    truth_ids = []
    for i, fact in enumerate(ground_truth):
        doc_id = await memory.store_memory_bank(
            text=fact,
            tags=["user_info", "sarah", "verified"],
            importance=0.95,
            confidence=0.98
        )
        truth_ids.append(doc_id)
        print(f"  [OK] Truth {i+1}/3 (quality=0.93): {fact[:60]}...")

    # =========================================================================
    # CONFUSION NOISE - Similar but WRONG facts (LOW QUALITY)
    # =========================================================================
    print("\n[SETUP] Storing 47 confusing noise facts (LOW QUALITY)...")

    noise_facts = [
        # Similar names, wrong details
        "Sara Chen is 28 years old and works at Microsoft as a PM.",
        "Sarah M. Chen is 34 and runs her own consulting business.",
        "Sarah Chen-Williams is a 34-year-old lawyer in New York.",
        "Sandra Chen is 35 and works as a software engineer at Amazon.",
        "Sera Chen is 33 years old, software developer at a startup.",
        "Sarah C. is 34, works in tech somewhere in California.",
        "S. Chen is an engineer, about 34, works at some tech company.",

        # Right name, wrong age
        "Sarah Chen is 28 years old and just started her career.",
        "Sarah Chen is 42 and has been in tech for 20 years.",
        "Sarah Chen is 31, recently graduated and joined Google.",
        "Sarah Chen, age 36, senior architect at Facebook.",
        "Sarah Chen is in her late 20s, junior developer.",

        # Right name, wrong company
        "Sarah Chen works at Google as a software engineer.",
        "Sarah Chen is employed by Amazon Web Services.",
        "Sarah Chen works at Meta, previously Facebook.",
        "Sarah Chen is at Apple, working on iOS development.",
        "Sarah Chen joined Netflix last year as an engineer.",
        "Sarah Chen works at a small startup called TechStart.",
        "Sarah Chen is at TechCo (not TechCorp) as a developer.",

        # Right name, wrong job
        "Sarah Chen is a product manager at TechCorp.",
        "Sarah Chen works as a data scientist.",
        "Sarah Chen is the CEO of her own company.",
        "Sarah Chen is a UX designer specializing in mobile.",
        "Sarah Chen works in HR at a tech company.",
        "Sarah Chen is a marketing manager in Silicon Valley.",

        # Partially correct but misleading combos
        "Sarah Chen, 34, is a software engineer at Google.",  # Wrong company!
        "Sarah Chen at TechCorp is 28 years old.",  # Wrong age!
        "Sarah Chen, software engineer, lives in New York.",  # Wrong city!
        "Sarah Chen works at TechCorp as a product manager.",  # Wrong job!
        "Sarah Chen, 34, engineer at TechCorp Inc (the consulting firm).",  # Wrong TechCorp!

        # Other people entirely
        "Michael Chen is 34 and works at TechCorp as an engineer.",
        "Sarah Williams is a 34-year-old software engineer in SF.",
        "Sarah Johnson, 34, software developer at TechCorp competitor.",
        "Chen Wei is 34 and works as an engineer at TechCorp.",
        "Sarah Park, software engineer, 34, works in tech.",

        # Vague/ambiguous
        "Someone named Sarah works in tech, maybe engineering.",
        "There's an engineer named Chen at some company.",
        "A software engineer in their 30s works at TechCorp.",
        "Sarah is a common name in tech companies.",
        "Chen is an engineer, age unknown, location unknown.",

        # Red herrings with high semantic similarity
        "The user mentioned Sarah Chen multiple times in conversation.",
        "Looking up information about Sarah Chen.",
        "Sarah Chen's details need to be verified.",
        "Is Sarah Chen the one who works at TechCorp?",
        "Confirming Sarah Chen's employment information.",
        "Sarah Chen inquiry - need to find correct details.",
        "User asked about Sarah Chen's background.",
    ]

    noise_ids = []
    for i, fact in enumerate(noise_facts):
        doc_id = await memory.store_memory_bank(
            text=fact,
            tags=["user_info", "person"],
            importance=0.25,  # LOW - uncertain info
            confidence=0.30   # LOW - not verified
        )
        noise_ids.append(doc_id)
        if (i + 1) % 10 == 0:
            print(f"  [OK] {i+1}/47 noise facts stored (quality=0.08)...")

    total = len(ground_truth) + len(noise_facts)
    noise_ratio = len(noise_facts) / len(ground_truth)

    print(f"\n[VERIFY] Total stored: {len(ground_truth)} truth + {len(noise_facts)} noise = {total} docs")
    print(f"[VERIFY] Noise ratio: {noise_ratio:.1f}:1")
    print(f"[VERIFY] Quality gap: Truth=0.93 vs Noise=0.08 ({0.93/0.08:.0f}x higher)")
    print(f"[DEBUG] Truth IDs: {truth_ids}")
    print(f"[DEBUG] First 5 Noise IDs: {noise_ids[:5]}")

    # =========================================================================
    # TEST QUERIES - Progressively harder
    # =========================================================================

    test_cases = [
        {
            "query": "Sarah Chen TechCorp engineer",
            "description": "Direct query - should be easy",
            "difficulty": "EASY"
        },
        {
            "query": "What does Sarah Chen do for work?",
            "description": "Natural language - medium difficulty",
            "difficulty": "MEDIUM"
        },
        {
            "query": "Sarah age job company",
            "description": "Sparse keywords - harder",
            "difficulty": "HARD"
        },
        {
            "query": "the user Sarah",
            "description": "Vague reference - very hard",
            "difficulty": "BRUTAL"
        },
        {
            "query": "software engineer 34 years old",
            "description": "No name! Must infer from quality",
            "difficulty": "NIGHTMARE"
        }
    ]

    print("\n" + "-"*80)
    print("RUNNING TEST QUERIES")
    print("-"*80)

    total_score = 0
    max_score = len(test_cases) * 3  # 3 truth docs to find per query
    truth_at_rank1 = 0  # Count queries where truth is ranked #1

    for test in test_cases:
        print(f"\n[{test['difficulty']}] Query: \"{test['query']}\"")
        print(f"         {test['description']}")

        results = await memory.search(
            query=test["query"],
            collections=["memory_bank"],
            limit=5
        )

        truth_found = sum(1 for r in results[:5] if r['id'] in truth_ids)
        noise_in_top5 = 5 - truth_found

        total_score += truth_found

        # Check if truth is ranked #1 (most important metric)
        if results and results[0]['id'] in truth_ids:
            truth_at_rank1 += 1

        print(f"         Results: {truth_found}/3 truth in top 5 ({noise_in_top5} noise)")

        # Show what we got
        for i, r in enumerate(results[:5]):
            doc_id = r.get('id', '')
            is_truth = doc_id in truth_ids
            marker = "TRUTH" if is_truth else "noise"
            metadata = r.get('metadata', {})
            importance = metadata.get('importance', 0.5)
            confidence = metadata.get('confidence', 0.5)
            quality = importance * confidence if isinstance(importance, (int, float)) and isinstance(confidence, (int, float)) else 0.5
            final_score = r.get('final_rank_score', 0)
            text = r.get('text', '')
            # Show full text for high quality "noise" entries to debug
            if marker == "noise" and quality > 0.5:
                print(f"           {i+1}. [{marker}] q={quality:.2f} rank={final_score:.3f} | FULL: {text}")
                print(f"               ID: {doc_id}")
            else:
                print(f"           {i+1}. [{marker}] q={quality:.2f} rank={final_score:.3f} | {text[:70]}...")

    # =========================================================================
    # FINAL SCORING
    # =========================================================================

    accuracy = (total_score / max_score) * 100
    rank1_rate = (truth_at_rank1 / len(test_cases)) * 100

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total truth docs found: {total_score}/{max_score} ({accuracy:.1f}%)")
    print(f"Truth ranked #1: {truth_at_rank1}/{len(test_cases)} queries ({rank1_rate:.1f}%)")
    print(f"Noise ratio survived: {noise_ratio:.0f}:1")
    print(f"\n[MECHANISM] 3-stage quality enforcement:")
    print(f"  1. Distance boost: adjusted_distance = L2_distance * (1.0 - quality * 0.8)")
    print(f"  2. L2->Similarity: similarity = 1 / (1 + distance)")
    print(f"  3. CE multiplier: final_score = blended_score * (1 + quality)")

    # Success criteria:
    # - Primary: Truth should be #1 in at least 3/5 queries (60%)
    # - BRUTAL query is expected to fail (no semantic match)
    # - This tests quality ranking cuts through noise, not total recall
    success = truth_at_rank1 >= 3

    print("\n" + "="*80)
    if success:
        print(f"PASS - Quality ranking works: truth #1 in {truth_at_rank1}/5 queries")
        print(f"       Survived {noise_ratio:.0f}:1 noise ratio")
    else:
        print(f"FAIL - Truth ranked #1 in only {truth_at_rank1}/5 queries (need >= 3)")
        print(f"       Quality boost not strong enough for {noise_ratio:.0f}:1 noise")
    print("="*80)

    # Cleanup
    import shutil
    if test_data_dir.exists():
        try:
            shutil.rmtree(test_data_dir)
        except:
            pass

    return success


if __name__ == "__main__":
    success = asyncio.run(test_semantic_confusion())
    sys.exit(0 if success else 1)
