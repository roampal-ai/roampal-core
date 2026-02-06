"""
A/B Test: Outcome-Based Learning vs Plain Vector Search

This test proves that outcome-based scoring actually improves retrieval quality,
not just that "having memory is better than no memory."

Test Design:
- Condition A (Treatment): Full Roampal with outcome scoring
- Condition B (Control): Same storage, but NO outcome scoring applied

Both conditions:
- Use identical real embeddings (all-mpnet-base-v2, 768d)
- Store the same good + bad advice memories
- Use the same ChromaDB backend

The ONLY difference: whether outcome scores are applied after storage.

ADVERSARIAL DESIGN:
- Queries are designed to semantically match the BAD advice better
- Without outcome scoring, pure vector similarity returns the BAD advice first
- With outcome scoring, the good advice (score=0.9) outranks the bad (score=0.2)

This isolates the value of outcome-based learning under adversarial conditions.
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import json
import sys
import statistics
import math
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem
from mock_utilities import MockLLMService

# Try to import real embeddings
try:
    from learning_curve_test.real_embedding_service import RealEmbeddingService
    HAS_REAL_EMBEDDINGS = True
except ImportError:
    HAS_REAL_EMBEDDINGS = False
    print("WARNING: Real embeddings not available. Install sentence-transformers.")


# ====================================================================================
# ADVERSARIAL TEST SCENARIOS
# ====================================================================================
#
# Each scenario has:
#   - good_advice: The correct approach (worked in practice)
#   - bad_advice: The wrong approach (failed in practice)
#   - query: Designed to semantically match bad_advice better than good_advice
#
# Without outcome scoring: bad_advice ranks #1 (closer semantic match to query)
# With outcome scoring: good_advice ranks #1 (score boost overrides semantic distance)

TEST_SCENARIOS = [
    {
        "name": "Print Debugging vs Debugger",
        "good_advice": "Use IDE debugger with breakpoints and watch expressions for systematic debugging",
        "bad_advice": "Add print statements throughout the code to trace variable values and execution flow",
        "query": "How should I use print statements to check variable values in my code?",
    },
    {
        "name": "Cache Everything vs Connection Pool",
        "good_advice": "Use connection pooling with proper pool size tuning based on workload analysis",
        "bad_advice": "Cache every database query result in Redis to avoid hitting the database",
        "query": "How do I cache database queries in Redis to speed things up?",
    },
    {
        "name": "Catch-All Errors vs Typed Exceptions",
        "good_advice": "Create typed exception hierarchy with specific handlers for each error category",
        "bad_advice": "Use a single try/catch block that catches all exceptions to prevent crashes",
        "query": "How do I catch all exceptions so my application doesn't crash?",
    },
    {
        "name": "Manual Timing vs Profiler",
        "good_advice": "Use cProfile or py-spy profiler to identify actual bottlenecks with statistical sampling",
        "bad_advice": "Add time.time() calls around each function to measure how long things take",
        "query": "How do I add timing calls around my functions to find what's slow?",
    },
    {
        "name": "Sequential Await vs Concurrent",
        "good_advice": "Use asyncio.gather() or TaskGroup to run independent async operations concurrently",
        "bad_advice": "Await each async call one by one in sequence to keep the code simple and readable",
        "query": "Should I await async calls one by one to keep my code simple?",
    },
    {
        "name": "Force Push vs Revert Commit",
        "good_advice": "Use git revert to create a new commit that undoes changes, preserving history for the team",
        "bad_advice": "Use git push --force to remove bad commits and clean up the branch history",
        "query": "How do I force push to remove bad commits and clean up my branch?",
    },
    {
        "name": "Global State vs Dependency Injection",
        "good_advice": "Use dependency injection to pass shared state explicitly through constructors",
        "bad_advice": "Use global variables to share state between async tasks for simplicity",
        "query": "How do I use global variables to share state between async tasks?",
    },
    {
        "name": "Nested REST vs Flat Endpoints",
        "good_advice": "Use flat REST endpoints with query parameters for filtering and pagination",
        "bad_advice": "Create deeply nested URL paths to represent all entity relationships",
        "query": "How should I create nested URL paths to represent entity relationships?",
    },
    {
        "name": "Stock Picking vs Index Funds",
        "good_advice": "Invest in low-cost index funds - 90% of active managers underperform over 15 years (S&P SPIVA)",
        "bad_advice": "Pick individual stocks based on research to beat the market and get higher returns",
        "query": "How do I pick the best individual stocks to beat the market?",
    },
    {
        "name": "Market Timing vs Stay Invested",
        "good_advice": "Stay invested through market downturns - missing the 10 best days cuts returns by 50% (Schwab)",
        "bad_advice": "Sell when the market drops and buy back when it starts rising to avoid losses",
        "query": "When should I sell my stocks to avoid losses during a market drop?",
    },
]


# ====================================================================================
# STATISTICAL FUNCTIONS
# ====================================================================================

def cohens_d(treatment: List[float], control: List[float]) -> float:
    """Calculate Cohen's d effect size"""
    if not treatment or not control:
        return 0.0
    mean_t = statistics.mean(treatment)
    mean_c = statistics.mean(control)

    try:
        var_t = statistics.variance(treatment) if len(treatment) > 1 else 0
        var_c = statistics.variance(control) if len(control) > 1 else 0
    except:
        var_t, var_c = 0, 0

    n_t, n_c = len(treatment), len(control)
    if n_t + n_c <= 2:
        return 0.0
    pooled_var = ((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2)
    if pooled_var == 0:
        return float('inf') if mean_t != mean_c else 0.0
    pooled_std = math.sqrt(pooled_var)

    return (mean_t - mean_c) / pooled_std


def paired_t_test(treatment: List[float], control: List[float]) -> Tuple[float, float]:
    """Paired t-test using scipy for proper p-value computation."""
    if len(treatment) != len(control) or len(treatment) < 2:
        return (0.0, 1.0)

    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(treatment, control)
        if math.isnan(t_stat):
            # All differences are zero or constant
            return (0.0, 1.0)
        return (t_stat, p_value)
    except ImportError:
        # Fallback: manual computation with t-distribution approximation
        differences = [t - c for t, c in zip(treatment, control)]
        mean_diff = statistics.mean(differences)
        try:
            std_diff = statistics.stdev(differences)
        except:
            std_diff = 0.0
        if std_diff == 0:
            return (float('inf') if mean_diff > 0 else 0.0, 0.0 if mean_diff != 0 else 1.0)
        n = len(differences)
        t_stat = mean_diff / (std_diff / math.sqrt(n))
        # Note: p-value is approximate without scipy
        return (t_stat, float('nan'))


# ====================================================================================
# TEST FUNCTIONS
# ====================================================================================

async def run_scenario_with_outcomes(
    scenario: Dict,
    data_dir: str,
    embedding_service
) -> Dict:
    """
    Treatment: Store good + bad advice, apply outcome scoring, then search.
    Good advice gets "worked" (score boosted), bad gets "failed" (score demoted).
    """
    system = UnifiedMemorySystem(data_path=data_dir)
    await system.initialize()
    system._embedding_service = embedding_service
    if system._search_service:
        system._search_service.embed_fn = embedding_service.embed_text

    # Store both pieces of advice
    good_id = await system.store(
        text=scenario["good_advice"],
        collection="working",
        metadata={"type": "good", "scenario": scenario["name"]}
    )
    bad_id = await system.store(
        text=scenario["bad_advice"],
        collection="working",
        metadata={"type": "bad", "scenario": scenario["name"]}
    )

    # Apply outcomes by directly setting metadata (consistent with 4-way benchmark).
    # Avoids uses/success_count inconsistency from manual uses override after record_outcome.
    adapter = system.collections.get("working")
    if adapter:
        # Good advice: proven high-value (5 uses, 90% success rate)
        try:
            result = adapter.collection.get(ids=[good_id], include=["metadatas"])
            if result and result.get("metadatas"):
                meta = result["metadatas"][0]
                meta["uses"] = 5
                meta["score"] = 0.9
                meta["success_count"] = 4.5  # 4.5/5 = 90%
                meta["last_outcome"] = "worked"
                meta["outcome_history"] = json.dumps([
                    {"outcome": "worked", "timestamp": f"2025-01-{i+1:02d}"}
                    for i in range(5)
                ][:5])
                adapter.collection.update(ids=[good_id], metadatas=[meta])
        except Exception:
            pass

        # Bad advice: failing (5 uses, 20% success rate)
        try:
            result = adapter.collection.get(ids=[bad_id], include=["metadatas"])
            if result and result.get("metadatas"):
                meta = result["metadatas"][0]
                meta["uses"] = 5
                meta["score"] = 0.2
                meta["success_count"] = 1.0  # 1/5 = 20%
                meta["last_outcome"] = "failed"
                meta["outcome_history"] = json.dumps([
                    {"outcome": "failed", "timestamp": f"2025-01-{i+1:02d}"}
                    for i in range(4)
                ] + [{"outcome": "worked", "timestamp": "2025-01-05"}])
                adapter.collection.update(ids=[bad_id], metadatas=[meta])
        except Exception:
            pass

    # Search - should return good advice first due to score boost
    # Use limit=5 to retrieve enough candidates for quality scoring to rerank
    results = await system.search(
        scenario["query"],
        collections=["working", "history"],
        limit=5
    )

    good_ranked_first = False
    if results:
        top_text = results[0].get("text", "").lower()
        if scenario["good_advice"].lower()[:50] in top_text or top_text in scenario["good_advice"].lower():
            good_ranked_first = True

    return {
        "scenario": scenario["name"],
        "condition": "with_outcomes",
        "good_ranked_first": good_ranked_first,
        "precision": 1.0 if good_ranked_first else 0.0,
    }


async def run_scenario_without_outcomes(
    scenario: Dict,
    data_dir: str,
    embedding_service
) -> Dict:
    """
    Control: Store good + bad advice, NO outcome scoring, then search.
    Pure vector similarity determines ranking.
    """
    system = UnifiedMemorySystem(data_path=data_dir)
    await system.initialize()
    system._embedding_service = embedding_service
    if system._search_service:
        system._search_service.embed_fn = embedding_service.embed_text

    # Store good advice (no outcomes)
    await system.store(
        text=scenario["good_advice"],
        collection="working",
        metadata={"type": "good", "scenario": scenario["name"]}
    )

    # Store bad advice (no outcomes)
    await system.store(
        text=scenario["bad_advice"],
        collection="working",
        metadata={"type": "bad", "scenario": scenario["name"]}
    )

    # Search - should return bad advice first (adversarial query matches it better)
    # Use limit=5 to match treatment condition retrieval depth
    results = await system.search(
        scenario["query"],
        collections=["working"],
        limit=5
    )

    good_ranked_first = False
    if results:
        top_text = results[0].get("text", "").lower()
        if scenario["good_advice"].lower()[:50] in top_text or top_text in scenario["good_advice"].lower():
            good_ranked_first = True

    return {
        "scenario": scenario["name"],
        "condition": "without_outcomes",
        "good_ranked_first": good_ranked_first,
        "precision": 1.0 if good_ranked_first else 0.0,
    }


# ====================================================================================
# MAIN TEST
# ====================================================================================

async def main():
    print("=" * 70)
    print("A/B TEST: Outcome-Based Learning vs Plain Vector Search")
    print("=" * 70)
    print()
    print("ADVERSARIAL DESIGN: Queries match BAD advice better semantically.")
    print("The ONLY difference: whether outcome scores are applied.")
    print()

    if not HAS_REAL_EMBEDDINGS:
        print("ERROR: This test requires real embeddings.")
        print("Install: pip install sentence-transformers")
        return

    print("Loading embedding model (all-mpnet-base-v2, 768d)...")
    embedding_service = RealEmbeddingService()
    print("Model loaded.\n")

    test_dir = Path(__file__).parent / "ab_test_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    treatment_results = []
    control_results = []

    print("-" * 70)
    print(f"Running {len(TEST_SCENARIOS)} adversarial scenarios...")
    print("-" * 70)

    for i, scenario in enumerate(TEST_SCENARIOS):
        print(f"\n[{i+1}/{len(TEST_SCENARIOS)}] {scenario['name']}")
        print(f"    Query: \"{scenario['query'][:70]}...\"")

        # Treatment: WITH outcome scoring
        treatment_dir = str(test_dir / f"treatment_{i}")
        os.makedirs(treatment_dir, exist_ok=True)
        treatment = await run_scenario_with_outcomes(scenario, treatment_dir, embedding_service)
        treatment_results.append(treatment)
        marker = "PASS" if treatment['good_ranked_first'] else "FAIL"
        print(f"    [WITH outcomes]    Good advice #1: {treatment['good_ranked_first']} [{marker}]")

        # Control: WITHOUT outcome scoring
        control_dir = str(test_dir / f"control_{i}")
        os.makedirs(control_dir, exist_ok=True)
        control = await run_scenario_without_outcomes(scenario, control_dir, embedding_service)
        control_results.append(control)
        marker = "FAIL (expected)" if not control['good_ranked_first'] else "PASS (query not adversarial enough)"
        print(f"    [WITHOUT outcomes] Good advice #1: {control['good_ranked_first']} [{marker}]")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    treatment_precisions = [r['precision'] for r in treatment_results]
    control_precisions = [r['precision'] for r in control_results]

    mean_treatment = statistics.mean(treatment_precisions)
    mean_control = statistics.mean(control_precisions)

    print(f"\nPrecision (WITH outcomes):    {mean_treatment:.1%}")
    print(f"Precision (WITHOUT outcomes): {mean_control:.1%}")
    print(f"Improvement:                  {mean_treatment - mean_control:+.1%}")

    # Statistical analysis
    d = cohens_d(treatment_precisions, control_precisions)
    t_stat, p_value = paired_t_test(treatment_precisions, control_precisions)

    print(f"\nStatistical Analysis:")
    print(f"  Cohen's d:    {d:.2f}", end="")
    if d >= 0.8:
        print(" (LARGE effect)")
    elif d >= 0.5:
        print(" (medium effect)")
    elif d >= 0.2:
        print(" (small effect)")
    else:
        print(" (negligible)")

    print(f"  t-statistic:  {t_stat:.2f}")
    print(f"  p-value:      {p_value}")

    if p_value < 0.05:
        print(f"\n[PASS] STATISTICALLY SIGNIFICANT: Outcome scoring improves retrieval quality.")
    else:
        print(f"\n[WARN] NOT STATISTICALLY SIGNIFICANT: p={p_value}")

    # Save results
    results_file = test_dir / "ab_test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(TEST_SCENARIOS),
            "treatment_results": treatment_results,
            "control_results": control_results,
            "statistics": {
                "mean_treatment": mean_treatment,
                "mean_control": mean_control,
                "improvement": mean_treatment - mean_control,
                "cohens_d": d,
                "t_statistic": t_stat,
                "p_value": p_value
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Cleanup
    print("\nCleaning up test data...")
    try:
        shutil.rmtree(test_dir)
        print("Done.")
    except PermissionError:
        print("Note: Some test files locked by ChromaDB - will be cleaned up on next run.")


if __name__ == "__main__":
    asyncio.run(main())
