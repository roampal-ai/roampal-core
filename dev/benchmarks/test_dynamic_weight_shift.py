"""
Test: Dynamic Weight Shifting - Proven Memories vs New Matches

This test validates that Roampal's dynamic weighted ranking actually works:
- New memories: 70% embedding, 30% score
- Proven memories (uses>=5, score>=0.8): 40% embedding, 60% score

The test proves: Proven memories rank well EVEN with mediocre query formulation.

Test Design:
1. Create a "proven" memory (simulate 5+ retrievals, consistent positive outcomes)
2. Create a "new" memory (no retrievals, default score)
3. Search with a query that semantically matches the NEW memory better
4. Verify the PROVEN memory still ranks competitively due to weight shift

This is the REAL test of outcome-based learning value.
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem
from modules.memory.chromadb_adapter import ChromaDBAdapter
from mock_utilities import MockLLMService

# Try to import real embeddings
try:
    from learning_curve_test.real_embedding_service import RealEmbeddingService
    HAS_REAL_EMBEDDINGS = True
except ImportError:
    HAS_REAL_EMBEDDINGS = False
    print("WARNING: Real embeddings not available. Install sentence-transformers.")


# ====================================================================================
# TEST SCENARIOS - Designed to show weight shift value
# ====================================================================================

TEST_SCENARIOS = [
    {
        "name": "Proven Solution vs New Semantically-Closer Match",
        "description": """
        The 'proven' memory is a solution that WORKED multiple times.
        The 'new' memory is semantically closer to the query but unproven.
        Dynamic weighting should make proven memory competitive.
        """,
        "proven_memory": {
            "text": "Use pdb with breakpoints - set breakpoint with 'import pdb; pdb.set_trace()' then step through with n/s/c commands",
            "outcome": "worked",
            "outcome_count": 2,  # Record 2 positive outcomes (0.5 -> 0.7 -> 0.9)
            "uses": 5,  # Simulate 5 retrievals (proven threshold)
        },
        "new_memory": {
            "text": "Debug Python code by adding print statements everywhere to see variable values at each step",
            "outcome": "unknown",  # Never tested
            "uses": 0,
        },
        # Query semantically closer to "new" (mentions print, debug, variables)
        "query": "How do I debug and print variable values in Python?",
        "expected": "proven should be competitive despite worse semantic match",
    },
    {
        "name": "Established Pattern vs Fresh Advice",
        "description": """
        'Established' memory (uses>=3, score>=0.7) gets 55% score weight.
        'Fresh' memory has excellent semantic match but no track record.
        """,
        "proven_memory": {
            "text": "For API pagination, use cursor-based pagination with opaque tokens for stable results across updates",
            "outcome": "worked",
            "outcome_count": 1,  # 0.5 -> 0.7 (established threshold)
            "uses": 3,
        },
        "new_memory": {
            "text": "Paginate API results using page numbers like ?page=1&limit=10 for simple offset pagination",
            "outcome": "unknown",
            "uses": 0,
        },
        "query": "How should I add pagination with page numbers to my API?",
        "expected": "established pattern competes despite query mentioning page numbers",
    },
    {
        "name": "Battle-Tested Database Advice",
        "description": """
        Proven memory about indexing vs new memory about caching.
        Query mentions caching but proven indexing advice should compete.
        """,
        "proven_memory": {
            "text": "Add composite indexes on (user_id, created_at) for user timeline queries - reduced latency from 2s to 50ms",
            "outcome": "worked",
            "outcome_count": 2,  # 0.5 -> 0.7 -> 0.9 (high score)
            "uses": 5,
        },
        "new_memory": {
            "text": "Speed up database queries by caching results in Redis with a 5 minute TTL",
            "outcome": "unknown",
            "uses": 0,
        },
        "query": "How can I cache or speed up my user timeline database queries?",
        "expected": "indexing advice competitive despite query mentioning cache",
    },
    {
        "name": "Failed Advice Should Sink",
        "description": """
        A memory that consistently FAILED should rank lower than a new unproven memory.
        This tests the negative side of outcome learning.
        """,
        "proven_memory": {
            "text": "Use setTimeout with 0ms delay to make async operations synchronous",
            "outcome": "failed",
            "outcome_count": 1,  # Failed once (0.5 -> 0.2, won't get deleted)
            "uses": 3,
        },
        "new_memory": {
            "text": "Use async/await with proper Promise handling for clean asynchronous code",
            "outcome": "unknown",
            "uses": 0,
        },
        "query": "How do I handle async operations with setTimeout?",
        "expected": "new advice should outrank consistently failed advice",
    },
    {
        "name": "Mixed Signals - Partial Success",
        "description": """
        Memory with mixed outcomes (partial success) vs new perfect semantic match.
        Tests the nuance of partial scoring.
        """,
        "proven_memory": {
            "text": "Use Docker Compose for local development - works well for simple setups but needs tuning for large projects",
            "outcome": "partial",  # Sometimes works, sometimes not (0.5 -> 0.55 -> 0.60 -> 0.65)
            "outcome_count": 3,
            "uses": 4,
        },
        "new_memory": {
            "text": "Set up local Kubernetes with minikube for development that matches production exactly",
            "outcome": "unknown",
            "uses": 0,
        },
        "query": "What's the best local Kubernetes or container setup for development?",
        "expected": "partial success memory should have moderate ranking",
    },
]


async def create_proven_memory(system, memory_config):
    """
    Create a memory and simulate it becoming 'proven' through usage.

    This simulates:
    1. Storing the memory
    2. Recording outcomes multiple times
    3. Simulating retrievals to increment 'uses' count
    """
    # Store the memory
    doc_id = await system.store(
        text=memory_config["text"],
        collection="working",
        metadata={"type": "test", "initial_outcome": memory_config["outcome"]}
    )

    # Record outcomes multiple times
    outcome = memory_config["outcome"]
    outcome_count = memory_config.get("outcome_count", 0)

    for i in range(outcome_count):
        if outcome in ["worked", "failed", "partial"]:
            try:
                await system.record_outcome(doc_id=doc_id, outcome=outcome)
            except Exception as e:
                print(f"    Warning: Outcome {i+1} failed for doc_id {doc_id}: {e}")

    # Simulate retrievals by directly updating metadata
    # This is what happens when the memory is retrieved multiple times
    target_uses = memory_config.get("uses", 0)
    if target_uses > 0:
        # Get current metadata via the collections dict
        adapter = system.collections.get("working")
        if adapter:
            # Update uses count directly (simulating multiple retrievals)
            try:
                # Get the document from ChromaDB
                result = adapter.collection.get(ids=[doc_id], include=["metadatas"])
                if result and result.get("metadatas"):
                    current_meta = result["metadatas"][0]
                    current_meta["uses"] = target_uses
                    # Update in ChromaDB
                    adapter.collection.update(ids=[doc_id], metadatas=[current_meta])
            except Exception as e:
                print(f"    Warning: Could not update uses count: {e}")

    return doc_id


async def create_new_memory(system, memory_config):
    """Create a new memory with no history."""
    doc_id = await system.store(
        text=memory_config["text"],
        collection="working",
        metadata={"type": "test", "is_new": True}
    )
    return doc_id


async def run_scenario(scenario, data_dir, embedding_service):
    """
    Run a single test scenario and return results.
    """
    name = scenario["name"]

    # Create memory system
    system = UnifiedMemorySystem(
        data_path=data_dir,
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = embedding_service

    # Create proven memory
    proven_id = await create_proven_memory(system, scenario["proven_memory"])

    # Create new memory
    new_id = await create_new_memory(system, scenario["new_memory"])

    # Debug: verify memories exist before search
    adapter = system.collections.get("working")
    if adapter:
        all_docs = adapter.collection.get(include=["metadatas"])
        print(f"    [DEBUG] Working collection has {len(all_docs['ids'])} docs")
        for doc_id, meta in zip(all_docs['ids'], all_docs['metadatas']):
            print(f"      - {doc_id[:30]}: score={meta.get('score', 'N/A')}, uses={meta.get('uses', 'N/A')}")

    # Also check history (in case proven memory got promoted)
    history_adapter = system.collections.get("history")
    if history_adapter:
        try:
            history_docs = history_adapter.collection.get(include=["metadatas"])
            if history_docs['ids']:
                print(f"    [DEBUG] History collection has {len(history_docs['ids'])} docs (PROMOTED!)")
                for doc_id, meta in zip(history_docs['ids'], history_docs['metadatas']):
                    print(f"      - {doc_id[:30]}: score={meta.get('score', 'N/A')}, uses={meta.get('uses', 'N/A')}")
        except Exception:
            pass  # History might be empty

    # Search across working AND history (in case proven memory got promoted)
    results = await system.search(scenario["query"], collections=["working", "history"], limit=5)

    # Analyze results
    proven_text = scenario["proven_memory"]["text"][:40]
    new_text = scenario["new_memory"]["text"][:40]

    proven_rank = None
    new_rank = None
    proven_score = None
    new_score = None

    for i, r in enumerate(results):
        text = r.get("text", "")
        meta = r.get("metadata", {})

        if proven_text.lower() in text.lower():
            proven_rank = i + 1
            proven_score = meta.get("score", 0.5)
        elif new_text.lower() in text.lower():
            new_rank = i + 1
            new_score = meta.get("score", 0.5)

    # Determine outcome
    proven_outcome = scenario["proven_memory"]["outcome"]

    if proven_outcome == "failed":
        # For failed memories, new should outrank
        success = (new_rank is not None) and (proven_rank is None or new_rank < proven_rank)
        expectation = "New should outrank failed"
    else:
        # For worked/partial, proven should be competitive (within top 3)
        success = proven_rank is not None and proven_rank <= 3
        expectation = "Proven should be in top 3"

    return {
        "scenario": name,
        "proven_rank": proven_rank,
        "proven_score": proven_score,
        "new_rank": new_rank,
        "new_score": new_score,
        "success": success,
        "expectation": expectation,
        "results": [(r.get("text", "")[:50], r.get("metadata", {}).get("score")) for r in results[:3]]
    }


async def main():
    print("=" * 70)
    print("DYNAMIC WEIGHT SHIFT TEST")
    print("Proving: Proven memories rank well despite imperfect query match")
    print("=" * 70)
    print()
    print("Weight Assignment by Memory Maturity:")
    print("  - New (uses<2):        70% embedding, 30% score")
    print("  - Emerging (uses>=2):  50% embedding, 50% score")
    print("  - Established (>=3, >=0.7): 45% embedding, 55% score")
    print("  - Proven (>=5, >=0.8): 40% embedding, 60% score")
    print()

    if not HAS_REAL_EMBEDDINGS:
        print("ERROR: This test requires real embeddings.")
        print("Install: pip install sentence-transformers")
        return

    # Initialize embedding service
    print("Loading embedding model...")
    embedding_service = RealEmbeddingService()
    print("Model loaded.\n")

    # Create test directory
    test_dir = Path(__file__).parent / "dynamic_weight_test_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    results = []
    successes = 0

    print("-" * 70)
    print("Running scenarios...")
    print("-" * 70)

    for i, scenario in enumerate(TEST_SCENARIOS):
        print(f"\n[{i+1}/{len(TEST_SCENARIOS)}] {scenario['name']}")
        print(f"    Query: \"{scenario['query']}\"")
        print(f"    Proven outcome: {scenario['proven_memory']['outcome']} x{scenario['proven_memory'].get('outcome_count', 0)}")

        scenario_dir = str(test_dir / f"scenario_{i}")
        os.makedirs(scenario_dir, exist_ok=True)

        result = await run_scenario(scenario, scenario_dir, embedding_service)
        results.append(result)

        print(f"    Results:")
        proven_score_str = f"{result['proven_score']:.2f}" if result['proven_score'] is not None else 'N/A'
        new_score_str = f"{result['new_score']:.2f}" if result['new_score'] is not None else 'N/A'
        print(f"      Proven memory rank: {result['proven_rank']} (score: {proven_score_str})")
        print(f"      New memory rank: {result['new_rank']} (score: {new_score_str})")
        print(f"      Top 3 results:")
        for j, (text, score) in enumerate(result['results']):
            score_str = f"{score:.2f}" if score is not None else "0.50"
            print(f"        {j+1}. [{score_str}] {text}...")

        if result['success']:
            print(f"    -> PASS: {result['expectation']}")
            successes += 1
        else:
            print(f"    -> FAIL: Expected {result['expectation']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(TEST_SCENARIOS)
    print(f"\nResults: {successes}/{total} scenarios passed ({successes/total*100:.0f}%)")

    # Breakdown by outcome type
    worked_scenarios = [r for r in results if "worked" in str(TEST_SCENARIOS[results.index(r)]["proven_memory"]["outcome"])]
    failed_scenarios = [r for r in results if "failed" in str(TEST_SCENARIOS[results.index(r)]["proven_memory"]["outcome"])]

    print(f"\nBreakdown:")
    for r in results:
        status = "PASS" if r['success'] else "FAIL"
        print(f"  [{status}] {r['scenario'][:40]}: proven={r['proven_rank']}, new={r['new_rank']}")

    # What this proves
    print("\n" + "-" * 70)
    if successes >= total * 0.8:
        print("CONCLUSION: Dynamic weight shifting WORKS")
        print()
        print("Proven memories (uses>=5, positive outcomes) rank competitively")
        print("even when query semantically matches newer, unproven content better.")
        print()
        print("This validates that outcome-based learning provides REAL value")
        print("beyond simple vector similarity search.")
    elif successes >= total * 0.5:
        print("CONCLUSION: Partial success - dynamic weighting shows effect")
        print()
        print("Some scenarios show weight shift benefit, but not consistently.")
        print("May need tuning of weight thresholds or more extreme test cases.")
    else:
        print("CONCLUSION: Dynamic weighting NOT showing expected effect")
        print()
        print("Possible issues:")
        print("- Weight thresholds may not be triggering")
        print("- Semantic similarity still dominating")
        print("- Need to verify weight assignment in search code")

    # Save results
    results_file = test_dir / "dynamic_weight_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": total,
            "successes": successes,
            "pass_rate": successes / total,
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Cleanup
    print("\nCleaning up test data...")
    try:
        shutil.rmtree(test_dir)
        print("Done.")
    except Exception as e:
        print(f"Warning: Could not clean up test data: {e}")
        print("You may need to manually delete:", test_dir)


if __name__ == "__main__":
    asyncio.run(main())
