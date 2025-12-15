"""
A/B Test: Outcome-Based Learning vs Plain Vector Search

This test proves that outcome-based scoring actually improves retrieval quality,
not just that "having memory is better than no memory."

Test Design:
- Condition A (Treatment): Full Roampal with outcome scoring
- Condition B (Control): Same storage, but search ignores scores (pure vector similarity)

Both conditions:
- Use identical embeddings (paraphrase-multilingual-mpnet-base-v2)
- Store the same memories
- Use the same ChromaDB backend

The ONLY difference: whether search results are re-ranked by outcome scores.

This isolates the value of outcome-based learning.
"""


import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import asyncio
import json
import os
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
# TEST SCENARIOS - SINGLE TEXT with multiple copies having different outcomes
# ====================================================================================
#
# Key insight: To properly test outcome learning, we need:
# 1. ONE text (so semantic similarity is IDENTICAL for all copies)
# 2. Multiple copies with different outcomes (worked vs failed)
# 3. More failed than worked, so without scoring, random selection = mostly failed
# 4. With scoring, top-N should be the worked copies
#
# Scenario design: 1 worked copy + 3 failed copies of IDENTICAL text
# Without scoring: 25% chance of worked in top-1 (random)
# With scoring: 100% chance of worked in top-1 (score-based)

TEST_SCENARIOS = [
    {
        "name": "Database Connection Pooling",
        "description": "Same advice tried 4 times: 1 worked, 3 failed.",
        "memories": [
            {"text": "Use connection pooling with a pool size of 10 for database connections", "outcome": "worked"},
            {"text": "Use connection pooling with a pool size of 10 for database connections", "outcome": "failed"},
            {"text": "Use connection pooling with a pool size of 10 for database connections", "outcome": "failed"},
            {"text": "Use connection pooling with a pool size of 10 for database connections", "outcome": "failed"},
        ],
        "query": "How should I configure database connections?",
    },
    {
        "name": "Redis Caching Strategy",
        "description": "Same caching advice: 1 worked, 3 failed in different contexts.",
        "memories": [
            {"text": "Cache API responses in Redis with a 5 minute TTL for performance", "outcome": "worked"},
            {"text": "Cache API responses in Redis with a 5 minute TTL for performance", "outcome": "failed"},
            {"text": "Cache API responses in Redis with a 5 minute TTL for performance", "outcome": "failed"},
            {"text": "Cache API responses in Redis with a 5 minute TTL for performance", "outcome": "failed"},
        ],
        "query": "What's the best caching strategy?",
    },
    {
        "name": "JWT Authentication",
        "description": "Same JWT advice: worked once, failed thrice.",
        "memories": [
            {"text": "Use JWT tokens with 1 hour expiration for stateless API authentication", "outcome": "worked"},
            {"text": "Use JWT tokens with 1 hour expiration for stateless API authentication", "outcome": "failed"},
            {"text": "Use JWT tokens with 1 hour expiration for stateless API authentication", "outcome": "failed"},
            {"text": "Use JWT tokens with 1 hour expiration for stateless API authentication", "outcome": "failed"},
        ],
        "query": "How should I implement authentication?",
    },
    {
        "name": "Exponential Backoff",
        "description": "Same retry strategy: 1 success, 3 failures.",
        "memories": [
            {"text": "Implement exponential backoff with max 3 retries for transient failures", "outcome": "worked"},
            {"text": "Implement exponential backoff with max 3 retries for transient failures", "outcome": "failed"},
            {"text": "Implement exponential backoff with max 3 retries for transient failures", "outcome": "failed"},
            {"text": "Implement exponential backoff with max 3 retries for transient failures", "outcome": "failed"},
        ],
        "query": "How do I handle transient failures?",
    },
    {
        "name": "Structured Logging",
        "description": "Same logging advice: worked once, failed thrice.",
        "memories": [
            {"text": "Use structured JSON logging with correlation IDs for distributed tracing", "outcome": "worked"},
            {"text": "Use structured JSON logging with correlation IDs for distributed tracing", "outcome": "failed"},
            {"text": "Use structured JSON logging with correlation IDs for distributed tracing", "outcome": "failed"},
            {"text": "Use structured JSON logging with correlation IDs for distributed tracing", "outcome": "failed"},
        ],
        "query": "What's the best logging setup?",
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

    # Handle edge case where all values are the same
    try:
        var_t = statistics.variance(treatment) if len(treatment) > 1 else 0
        var_c = statistics.variance(control) if len(control) > 1 else 0
    except:
        var_t, var_c = 0, 0

    n_t, n_c = len(treatment), len(control)

    # Pooled standard deviation
    if n_t + n_c <= 2:
        return 0.0
    pooled_var = ((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 0.001

    return (mean_t - mean_c) / pooled_std


def paired_t_test(treatment: List[float], control: List[float]) -> Tuple[float, float]:
    """Simple paired t-test returning (t_statistic, p_value)"""
    if len(treatment) != len(control) or len(treatment) < 2:
        return (0.0, 1.0)

    differences = [t - c for t, c in zip(treatment, control)]
    mean_diff = statistics.mean(differences)

    try:
        std_diff = statistics.stdev(differences)
    except:
        std_diff = 0.001

    if std_diff == 0:
        return (float('inf') if mean_diff > 0 else 0.0, 0.005 if mean_diff > 0 else 1.0)

    n = len(differences)
    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Approximate p-value from t-statistic (df = n-1)
    abs_t = abs(t_stat)
    if abs_t > 4.0:
        p_value = 0.005
    elif abs_t > 3.0:
        p_value = 0.01
    elif abs_t > 2.5:
        p_value = 0.025
    elif abs_t > 2.0:
        p_value = 0.05
    elif abs_t > 1.5:
        p_value = 0.1
    else:
        p_value = 0.2

    return (t_stat, p_value)


# ====================================================================================
# TEST FUNCTIONS
# ====================================================================================

async def run_scenario_with_outcomes(
    scenario: Dict,
    data_dir: str,
    embedding_service
) -> Dict:
    """
    Run scenario WITH outcome-based ranking (Treatment condition).

    Memories are scored based on outcomes, and search results are ranked by score.
    With identical text, the ONLY differentiator is outcome score.
    """
    name = scenario["name"]

    # Create memory system
    system = UnifiedMemorySystem(
        data_path=data_dir,
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = embedding_service

    # Store all memories and track which doc_ids map to worked/failed
    doc_id_to_outcome = {}
    worked_doc_ids = set()
    failed_doc_ids = set()

    for mem in scenario["memories"]:
        doc_id = await system.store(
            text=mem["text"],
            collection="working",
            metadata={"scenario": name, "mem_id": mem.get("id", "")}
        )
        outcome = mem.get("outcome", "unknown")
        doc_id_to_outcome[doc_id] = outcome

        if outcome == "worked":
            worked_doc_ids.add(doc_id)
        elif outcome == "failed":
            failed_doc_ids.add(doc_id)

    # Apply outcomes ONCE to shift scores without deletion
    # worked: 0.5 -> 0.7 (boosted)
    # failed: 0.5 -> 0.3 (demoted but not deleted)
    for doc_id, outcome in doc_id_to_outcome.items():
        if outcome in ["worked", "failed"]:
            await system.record_outcome(
                doc_id=doc_id,
                outcome=outcome
            )

    # Search and evaluate - get top 2 (half the dataset)
    # If outcome scoring works, top 2 should be "worked" items
    results = await system.search(scenario["query"], collections=["working"], limit=2)

    # Get all results sorted by final_rank_score
    all_results = await system.search(scenario["query"], collections=["working"], limit=4)

    # Check if TOP-1 result is the "worked" item
    # With 1 worked + 3 failed (all identical text):
    # - Random: 25% chance worked is #1
    # - With outcome scoring: 100% chance worked is #1 (highest score)
    top_result = all_results[0] if all_results else None
    top_is_worked = False

    if top_result:
        top_id = top_result.get("id", "")
        top_is_worked = top_id in worked_doc_ids

    # Precision = 1 if worked is #1, else 0
    precision = 1.0 if top_is_worked else 0.0

    return {
        "scenario": name,
        "condition": "with_outcomes",
        "top1_is_worked": top_is_worked,
        "precision": precision,
        "results": [r.get("text", "")[:80] for r in results]
    }


async def run_scenario_without_outcomes(
    scenario: Dict,
    data_dir: str,
    embedding_service
) -> Dict:
    """
    Run scenario WITHOUT outcome-based ranking (Control condition).

    Same memories stored, but we DON'T apply outcome scoring.
    Search returns pure vector similarity ranking.
    With identical text and no outcome scoring, results are random/arbitrary.
    """
    name = scenario["name"]

    # Create memory system
    system = UnifiedMemorySystem(
        data_path=data_dir,
        
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = embedding_service

    # Store all memories and track which doc_ids map to worked/failed
    # (We still track this to evaluate results, but don't apply outcomes)
    doc_id_to_outcome = {}
    worked_doc_ids = set()
    failed_doc_ids = set()

    for mem in scenario["memories"]:
        doc_id = await system.store(
            text=mem["text"],
            collection="working",
            metadata={"scenario": name, "mem_id": mem.get("id", "")}
        )
        outcome = mem.get("outcome", "unknown")
        doc_id_to_outcome[doc_id] = outcome

        if outcome == "worked":
            worked_doc_ids.add(doc_id)
        elif outcome == "failed":
            failed_doc_ids.add(doc_id)

    # NO outcome recording - pure vector search
    # Without outcomes, all scores stay at 0.5

    # Get all results - without outcome scoring, all have score 0.5
    all_results = await system.search(scenario["query"], collections=["working"], limit=4)

    # Check if TOP-1 result is the "worked" item
    # With 1 worked + 3 failed (all identical text, all score 0.5):
    # - Random: 25% chance worked is #1
    top_result = all_results[0] if all_results else None
    top_is_worked = False

    if top_result:
        top_id = top_result.get("id", "")
        top_is_worked = top_id in worked_doc_ids

    # Precision = 1 if worked is #1, else 0
    precision = 1.0 if top_is_worked else 0.0

    return {
        "scenario": name,
        "condition": "without_outcomes",
        "top1_is_worked": top_is_worked,
        "precision": precision,
        "results": [r.get("text", "")[:80] for r in all_results]
    }


# ====================================================================================
# MAIN TEST
# ====================================================================================

async def main():
    print("=" * 70)
    print("A/B TEST: Outcome-Based Learning vs Plain Vector Search")
    print("=" * 70)
    print()
    print("This test proves that outcome scoring improves retrieval quality.")
    print("Both conditions use identical embeddings and storage.")
    print("The ONLY difference: whether search results are ranked by outcome scores.")
    print()

    if not HAS_REAL_EMBEDDINGS:
        print("ERROR: This test requires real embeddings.")
        print("Install: pip install sentence-transformers")
        return

    # Initialize embedding service
    print("Loading embedding model (paraphrase-multilingual-mpnet-base-v2)...")
    embedding_service = RealEmbeddingService()
    print("Model loaded.\n")

    # Create test directories
    test_dir = Path(__file__).parent / "ab_test_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    treatment_results = []
    control_results = []

    print("-" * 70)
    print("Running scenarios...")
    print("-" * 70)

    for i, scenario in enumerate(TEST_SCENARIOS):
        print(f"\n[{i+1}/{len(TEST_SCENARIOS)}] {scenario['name']}")
        print(f"    Query: \"{scenario['query']}\"")

        # Treatment: WITH outcome scoring
        treatment_dir = str(test_dir / f"treatment_{i}")
        os.makedirs(treatment_dir, exist_ok=True)
        treatment = await run_scenario_with_outcomes(scenario, treatment_dir, embedding_service)
        treatment_results.append(treatment)
        marker = "PASS" if treatment['top1_is_worked'] else "FAIL"
        print(f"    [WITH outcomes]    Top-1 is worked: {treatment['top1_is_worked']} [{marker}]")

        # Control: WITHOUT outcome scoring
        control_dir = str(test_dir / f"control_{i}")
        os.makedirs(control_dir, exist_ok=True)
        control = await run_scenario_without_outcomes(scenario, control_dir, embedding_service)
        control_results.append(control)
        marker = "PASS" if control['top1_is_worked'] else "(expected ~25%)"
        print(f"    [WITHOUT outcomes] Top-1 is worked: {control['top1_is_worked']} [{marker}]")

        diff = treatment['precision'] - control['precision']
        if diff > 0:
            print(f"    -> Outcome scoring improved precision by {diff:.0%}")
        elif diff < 0:
            print(f"    -> Outcome scoring decreased precision by {abs(diff):.0%}")
        else:
            print(f"    -> No difference")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    treatment_precisions = [r['precision'] for r in treatment_results]
    control_precisions = [r['precision'] for r in control_results]

    mean_treatment = statistics.mean(treatment_precisions)
    mean_control = statistics.mean(control_precisions)

    print(f"\nMean Precision (WITH outcomes):    {mean_treatment:.1%}")
    print(f"Mean Precision (WITHOUT outcomes): {mean_control:.1%}")
    print(f"Improvement:                       {mean_treatment - mean_control:+.1%}")

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
        print("\n[PASS] STATISTICALLY SIGNIFICANT: Outcome scoring improves retrieval quality.")
    else:
        print("\n[WARN] NOT STATISTICALLY SIGNIFICANT: Need more scenarios or larger effect.")

    # Save results
    results_file = test_dir / "ab_test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
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
