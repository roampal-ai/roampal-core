"""
================================================================================
LEARNING CURVE BENCHMARK
================================================================================

Tests how accuracy improves as outcome history accumulates over time.

HYPOTHESIS:
-----------
More learning cycles = better adversarial resistance. As memories accumulate
"worked" outcomes, they should outrank semantically-matching-but-wrong answers.

MATURITY LEVELS TESTED:
-----------------------
- Level 0 (Cold Start): 0 uses, no outcomes - pure semantic matching
- Level 1 (Early):      3 uses, 2 worked - some signal
- Level 2 (Established): 5 uses, 4 worked - trusted pattern
- Level 3 (Proven):     10 uses, 8 worked - highly reliable
- Level 4 (Mature):     20 uses, 18 worked - battle-tested

EXPECTED RESULTS:
-----------------
Accuracy should increase with maturity level as outcome scores
increasingly override semantic similarity.

OUTPUT:
-------
- Per-level accuracy metrics
- Learning curve data (JSON) for graphing
- Statistical analysis of improvement rate

================================================================================
"""

import asyncio
import json
import os
import sys
import statistics
import shutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem
from mock_utilities import MockLLMService

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("ERROR: sentence-transformers required. pip install sentence-transformers")

os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"


class EmbeddingService768:
    def __init__(self):
        print("Loading embedding model: all-mpnet-base-v2 (768 dim)...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.dimension = 768

    async def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


# Maturity levels to test
MATURITY_LEVELS = [
    {"name": "cold_start", "uses": 0, "worked": 0, "failed": 0},
    {"name": "early", "uses": 3, "worked": 2, "failed": 0},
    {"name": "established", "uses": 5, "worked": 4, "failed": 0},
    {"name": "proven", "uses": 10, "worked": 8, "failed": 1},
    {"name": "mature", "uses": 20, "worked": 18, "failed": 1},
]

# Adversarial scenarios (query semantically matches WRONG answer)
SCENARIOS = [
    {
        "id": 1,
        "domain": "finance",
        "query": "Should I invest all my cash to beat inflation?",
        "good_advice": "Build 6-month emergency fund in high-yield savings before investing",
        "bad_advice": "Invest all extra money immediately - cash loses value to inflation",
    },
    {
        "id": 2,
        "domain": "finance",
        "query": "How do I pick stocks to beat the market?",
        "good_advice": "Invest in low-cost total market index funds - 90% of active managers underperform",
        "bad_advice": "Pick individual stocks based on research to beat the market",
    },
    {
        "id": 3,
        "domain": "health",
        "query": "Can I train myself to need less sleep?",
        "good_advice": "Sleep 7-9 hours consistently - sleep deprivation causes cognitive decline",
        "bad_advice": "Train your body to need less sleep - successful people sleep 4-5 hours",
    },
    {
        "id": 4,
        "domain": "health",
        "query": "Should I do intense HIIT when I have time?",
        "good_advice": "Consistent moderate exercise (150 min/week) beats intense sporadic workouts",
        "bad_advice": "Do intense HIIT workouts when you can - intensity matters more than consistency",
    },
    {
        "id": 5,
        "domain": "tech",
        "query": "Should I fix bugs first and test later?",
        "good_advice": "Write tests before fixing bugs - prevents regressions and documents behavior",
        "bad_advice": "Fix bugs quickly first, then add tests later if you have time",
    },
    {
        "id": 6,
        "domain": "tech",
        "query": "Can I keep secrets in a private repo?",
        "good_advice": "Use environment variables for config - never commit secrets to version control",
        "bad_advice": "Keep config files with secrets in a private repo - it's secure enough",
    },
    {
        "id": 7,
        "domain": "nutrition",
        "query": "Should I eat 2g protein per pound for gains?",
        "good_advice": "Protein needs are 0.7-1g per pound for muscle building - more isn't better",
        "bad_advice": "Eat as much protein as possible for maximum gains - 2g+ per pound",
    },
    {
        "id": 8,
        "domain": "nutrition",
        "query": "Should I avoid carbs after 6pm?",
        "good_advice": "Meal timing doesn't matter much - total daily intake matters more",
        "bad_advice": "Never eat carbs after 6pm - they turn to fat if not burned immediately",
    },
    {
        "id": 9,
        "domain": "crypto",
        "query": "Should I put all my savings in crypto?",
        "good_advice": "Only invest what you can afford to lose - crypto is highly speculative",
        "bad_advice": "Go all-in on crypto - it's the future of finance",
    },
    {
        "id": 10,
        "domain": "crypto",
        "query": "Should I invest in new altcoins early?",
        "good_advice": "Be skeptical of new tokens promising huge returns - 99% fail or are scams",
        "bad_advice": "Get in early on new altcoins for massive gains - early investors make fortunes",
    },
]


async def test_at_maturity_level(
    scenario: Dict,
    maturity: Dict,
    data_dir: str,
    embedding_service
) -> Dict:
    """Test a single scenario at a specific maturity level."""

    # Clean data directory
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    system = UnifiedMemorySystem(
        data_path=data_dir,
    )
    await system.initialize()
    system._embedding_service = embedding_service
    if system._search_service:
        system._search_service.embed_fn = embedding_service.embed_text

    # Store good advice
    good_id = await system.store(
        text=scenario["good_advice"],
        collection="working",
        metadata={"type": "good", "scenario": scenario["id"]}
    )

    # Store bad advice
    bad_id = await system.store(
        text=scenario["bad_advice"],
        collection="working",
        metadata={"type": "bad", "scenario": scenario["id"]}
    )

    # Apply maturity level outcomes by directly setting metadata
    # System reads: outcome_history (JSON array), score, uses (see unified_memory_system.py:1494-1519)
    adapter = system.collections.get("working")

    if adapter and maturity["uses"] > 0:
        # Build outcome_history for good advice (high success rate)
        good_outcome_history = []
        for i in range(maturity["worked"]):
            good_outcome_history.append({"outcome": "worked", "timestamp": f"2025-01-{i+1:02d}"})
        for i in range(maturity["failed"]):
            good_outcome_history.append({"outcome": "failed", "timestamp": f"2025-01-{20+i:02d}"})

        # Calculate raw score (simple success rate)
        good_success_rate = maturity["worked"] / maturity["uses"]

        # Update good advice metadata with fields system reads
        try:
            result = adapter.collection.get(ids=[good_id], include=["metadatas"])
            if result and result.get("metadatas"):
                meta = result["metadatas"][0]
                meta["uses"] = maturity["uses"]
                meta["score"] = good_success_rate
                meta["success_count"] = float(maturity["worked"])
                meta["outcome_history"] = json.dumps(good_outcome_history)
                meta["last_outcome"] = "worked"
                adapter.collection.update(ids=[good_id], metadatas=[meta])
        except Exception as e:
            print(f"Error updating good advice: {e}")

        # Bad advice gets worse outcomes (1/3 success rate)
        bad_worked = max(0, maturity["worked"] // 3)
        bad_failed = maturity["uses"] - bad_worked
        bad_success_rate = bad_worked / maturity["uses"]

        # Build outcome_history for bad advice
        bad_outcome_history = []
        for i in range(bad_worked):
            bad_outcome_history.append({"outcome": "worked", "timestamp": f"2025-01-{i+1:02d}"})
        for i in range(bad_failed):
            bad_outcome_history.append({"outcome": "failed", "timestamp": f"2025-01-{20+i:02d}"})

        # Update bad advice metadata
        try:
            result = adapter.collection.get(ids=[bad_id], include=["metadatas"])
            if result and result.get("metadatas"):
                meta = result["metadatas"][0]
                meta["uses"] = maturity["uses"]
                meta["score"] = bad_success_rate
                meta["success_count"] = float(bad_worked)
                meta["outcome_history"] = json.dumps(bad_outcome_history)
                meta["last_outcome"] = "failed"
                adapter.collection.update(ids=[bad_id], metadatas=[meta])
        except Exception as e:
            print(f"Error updating bad advice: {e}")

    # Search all relevant collections (working, history, patterns)
    results = await system.search(
        scenario["query"],
        collections=["working", "history", "patterns"],
        limit=10
    )

    # Determine ranking using bidirectional matching (like three-way comparison)
    good_rank = 0
    bad_rank = 0
    good_advice_lower = scenario["good_advice"].lower()
    bad_advice_lower = scenario["bad_advice"].lower()

    for i, r in enumerate(results):
        text = r.get("text", "").lower()
        # Bidirectional match: advice in text OR text in advice
        if good_rank == 0:
            if good_advice_lower[:50] in text or text[:50] in good_advice_lower:
                good_rank = i + 1
        if bad_rank == 0:
            if bad_advice_lower[:50] in text or text[:50] in bad_advice_lower:
                bad_rank = i + 1

    return {
        "scenario_id": scenario["id"],
        "domain": scenario["domain"],
        "maturity": maturity["name"],
        "uses": maturity["uses"],
        "good_rank": good_rank,
        "bad_rank": bad_rank,
        "top1_correct": good_rank == 1,
    }


async def main():
    print("=" * 80)
    print("LEARNING CURVE BENCHMARK")
    print("=" * 80)
    print()
    print("HYPOTHESIS: More outcome history = better adversarial resistance")
    print()
    print("MATURITY LEVELS:")
    for m in MATURITY_LEVELS:
        success_rate = (m["worked"] / m["uses"] * 100) if m["uses"] > 0 else 0
        print(f"  {m['name']:12s}: {m['uses']:2d} uses, {m['worked']:2d} worked ({success_rate:.0f}% success)")
    print()

    if not HAS_EMBEDDINGS:
        print("ERROR: sentence-transformers required")
        return

    embedding_service = EmbeddingService768()
    print("Model loaded.\n")

    test_dir = Path(__file__).parent / "learning_curve_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    # Results storage
    all_results = []
    level_accuracies = {m["name"]: [] for m in MATURITY_LEVELS}

    total_tests = len(SCENARIOS) * len(MATURITY_LEVELS)
    test_num = 0

    print("-" * 80)
    print(f"Running {len(SCENARIOS)} scenarios × {len(MATURITY_LEVELS)} maturity levels = {total_tests} tests")
    print("-" * 80)

    for scenario in SCENARIOS:
        for maturity in MATURITY_LEVELS:
            test_num += 1
            print(f"\r[{test_num:3d}/{total_tests}] {scenario['domain']:8s} @ {maturity['name']:12s}", end="", flush=True)

            data_dir = str(test_dir / f"s{scenario['id']}_{maturity['name']}")

            result = await test_at_maturity_level(
                scenario, maturity, data_dir, embedding_service
            )
            all_results.append(result)
            level_accuracies[maturity["name"]].append(1.0 if result["top1_correct"] else 0.0)

    print("\n\n")

    # Analysis
    print("=" * 80)
    print("RESULTS: LEARNING CURVE")
    print("=" * 80)

    print(f"\n{'Maturity Level':<15} {'Uses':<6} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 50)

    learning_curve_data = []
    for maturity in MATURITY_LEVELS:
        scores = level_accuracies[maturity["name"]]
        accuracy = statistics.mean(scores) * 100
        correct = sum(scores)
        total = len(scores)

        print(f"{maturity['name']:<15} {maturity['uses']:<6} {accuracy:>5.1f}%       {int(correct)}/{total}")

        learning_curve_data.append({
            "level": maturity["name"],
            "uses": maturity["uses"],
            "accuracy": accuracy,
            "correct": int(correct),
            "total": total,
        })

    # Calculate improvement
    cold_start_acc = learning_curve_data[0]["accuracy"]
    mature_acc = learning_curve_data[-1]["accuracy"]
    improvement = mature_acc - cold_start_acc

    print(f"\n{'='*50}")
    print(f"IMPROVEMENT: {cold_start_acc:.1f}% -> {mature_acc:.1f}% (+{improvement:.1f} pts)")
    print(f"{'='*50}")

    # Per-domain breakdown at mature level
    print(f"\nPER-DOMAIN ACCURACY (Mature Level):")
    domains = {}
    for r in all_results:
        if r["maturity"] == "mature":
            d = r["domain"]
            if d not in domains:
                domains[d] = {"correct": 0, "total": 0}
            domains[d]["total"] += 1
            if r["top1_correct"]:
                domains[d]["correct"] += 1

    for domain, counts in sorted(domains.items()):
        acc = counts["correct"] / counts["total"] * 100
        print(f"  {domain:<12}: {acc:5.1f}%")

    # Save results
    results_file = test_dir / "learning_curve_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(SCENARIOS),
            "n_maturity_levels": len(MATURITY_LEVELS),
            "learning_curve": learning_curve_data,
            "improvement": {
                "cold_start": cold_start_acc,
                "mature": mature_acc,
                "delta": improvement,
            },
            "per_domain_mature": {d: c["correct"]/c["total"]*100 for d, c in domains.items()},
            "all_results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if improvement > 10:
        print(f"\n  [PASS] LEARNING CURVE VALIDATED")
        print(f"    Accuracy improved by +{improvement:.1f} percentage points")
        print(f"    from cold start ({cold_start_acc:.1f}%) to mature ({mature_acc:.1f}%)")
        print(f"\n  This proves: More outcome history = better adversarial resistance")
    elif improvement > 0:
        print(f"\n  ~ MODEST IMPROVEMENT")
        print(f"    Accuracy improved by +{improvement:.1f} percentage points")
        print(f"    Consider increasing maturity levels or outcome contrast")
    else:
        print(f"\n  [FAIL] NO IMPROVEMENT DETECTED")
        print(f"    Learning curve hypothesis not validated in this test")


if __name__ == "__main__":
    asyncio.run(main())
