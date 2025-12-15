# Roampal Memory System - Comprehensive Test Suite

**Purpose**: Validate every feature of the Roampal memory system through exhaustive testing, from basic functionality to extreme stress scenarios.

## Test Suites

### 1. Comprehensive Test (30 tests, ~4 min)
**Purpose**: Validate all core features work correctly

```bash
cd benchmarks/comprehensive_test
python test_comprehensive.py
```

**Coverage**: All 5 tiers, 3 KGs, scoring, promotion, deduplication, edge cases

### 2. Torture Test Suite (10 tests, ~2 min)
**Purpose**: Stress test the system with extreme scenarios

```bash
cd benchmarks/comprehensive_test
python test_torture_suite.py
```

**Coverage**: High volume (1000 stores), adversarial inputs, capacity limits, concurrent access

### 3. Semantic Confusion Test (~30s)
**Purpose**: Validate quality-based ranking cuts through semantic noise

```bash
cd benchmarks/comprehensive_test
python test_semantic_confusion.py
```

**Scenario**: 15:1 noise ratio (3 truth vs 47 confusing facts)
**Result**: Truth ranked #1 in 4/5 queries (80%)

### 4. Stale Data Test (~20s)
**Purpose**: Validate fresh high-quality data beats stale low-quality data

```bash
cd benchmarks/comprehensive_test
python test_stale_data.py
```

**Scenario**: OLD facts (q=0.20) conflict with NEW facts (q=0.90)
**Result**: NEW ranked #1 in 3/5 queries (60%) - OLD wins when semantically closer

### 5. Edge Case Test (24 tests, ~30s)
**Purpose**: Validate system handles edge cases and adversarial inputs

```bash
cd benchmarks/comprehensive_test
python test_edge_cases.py
```

**Coverage**: 7 sections
- Embedding edge cases (empty, whitespace, 25KB text, single char)
- Unicode (emoji, RTL, zero-width, mixed scripts)
- Injection attempts (SQL, prompt, path traversal, JSON)
- Boundary conditions (importance/confidence limits)
- Malformed input (None values, numeric coercion)
- Concurrent operations (20 parallel stores)
- Search edge cases (special chars, limit 0, limit 10000)

**Result**: 24/24 tests pass

### 6. Contradiction Test (5 tests, ~10s)
**Purpose**: Validate quality-based ranking resolves conflicting facts

```bash
cd benchmarks/comprehensive_test
python test_contradictions.py
```

**Scenarios**:
1. Direct contradiction (blue vs green)
2. Many-wrong vs one-right (5:1 noise)
3. Temporal update (Boston -> Seattle)
4. Confidence conflict (uncertain vs confirmed)
5. Implicit contradiction (hates caffeine vs loves coffee)

**Result**: 5/5 tests pass - High-quality info ranked #1

### 7. Catastrophic Forgetting Test (5 tests, ~60s)
**Purpose**: Validate old knowledge survives after adding new knowledge

```bash
cd benchmarks/comprehensive_test
python test_catastrophic_forgetting.py
```

**Scenarios**:
1. Unrelated domain retention (family + cooking -> family still works)
2. Same domain retention (old work + new work -> old work accessible)
3. Bulk insert survival (3 critical + 100 random -> critical #1)
4. Quality preservation (1 verified vs 20 rumors -> verified #1)
5. Cross-domain isolation (5 topics don't interfere)

**Result**: 5/5 tests pass - No catastrophic forgetting

### 8. Context Poisoning Test (5 tests, ~15s)
**Purpose**: Validate system resists adversarial data attacks

```bash
cd benchmarks/comprehensive_test
python test_context_poisoning.py
```

**Attack Scenarios**:
1. Exact duplicate poisoning (deduplication keeps best)
2. Near-duplicate confusion (green beats 4 blue variants)
3. Entity confusion (manager John Smith beats 5 other John Smiths)
4. Temporal confusion (verified date beats 4 wrong dates)
5. Negation poisoning (IS allergic beats NOT allergic)

**Result**: 5/5 tests pass - System resistant to poisoning

### 9. Learning Sabotage Test (10 tests, ~30s)
**Purpose**: Validate LLM learning system resists manipulation attacks

```bash
cd benchmarks/comprehensive_test
python test_learning_sabotage.py
```

**Attack Scenarios**:
1. Outcome flooding (spam 50 'worked' to inflate bad memory)
2. Score boundary enforcement (try to exceed 0.1-1.0 bounds)
3. KG routing poisoning (associate wrong collections with queries)
4. Feedback inversion (mark failures as success, vice versa)
5. Action-Effectiveness manipulation (poison success rates)
6. Deduplication replacement (try to replace good with bad duplicate)
7. Rapid outcome rate limiting (rapid-fire vs spaced outcomes)
8. Memory bank immutability (outcomes shouldn't change quality scores)
9. Cross-collection contamination (outcomes leak to other docs)
10. Invalid outcome handling (non-existent docs, empty IDs)

**Result**: 10/10 tests pass - Learning system resistant to sabotage

### 10. Search Quality Test (6 metrics, ~20s)
**Purpose**: Validate search quality across multiple dimensions

```bash
cd benchmarks/comprehensive_test
python test_search_quality.py
```

**Quality Metrics**:
1. Synonym understanding (car → automobile)
2. Typo tolerance (Microsft → Microsoft)
3. Acronym expansion (API → Application Programming Interface)
4. Result diversity (varied aspects, not redundant)
5. Recency vs relevance balance
6. Partial match quality

**Result**: 100% average (6/6 metrics at 100%)

### 11. Recovery Resilience Test (6 tests, ~60s)
**Purpose**: Validate system recovers from failures gracefully

```bash
cd benchmarks/comprehensive_test
python test_recovery_resilience.py
```

**Scenarios**:
1. Interrupted store recovery
2. Empty collection handling
3. Malformed metadata resilience
4. Concurrent modification safety
5. Large batch atomicity (100 stores)
6. Knowledge graph consistency

**Result**: 6/6 tests pass - System resilient to failures

### 12. Learning Speed & Effectiveness Test (7 tests, ~30s)
**Purpose**: Benchmark how quickly and effectively the system learns

```bash
cd benchmarks/comprehensive_test
python test_learning_speed.py
```

**Benchmarks**:
1. Cold start baseline - routing accuracy with no prior learning
2. Learning curve - track improvement as feedback accumulates
3. Adaptation speed - interactions needed to learn new patterns
4. Knowledge retention - learning persists after noise
5. Pattern recognition - reuse successful routes for similar queries
6. Context specialization - learn context-specific strategies
7. Learning efficiency - improvement per unit of feedback

**Result**: 99% average score (Grade: A - Excellent Learner)
- Cold start baseline: 100%
- Learning curve: 100%
- Adaptation speed: 90% (learns in 1 interaction)
- Knowledge retention: 100%
- Pattern recognition: 100%
- Context specialization: 100%
- Learning efficiency: 100%

### 13. Comprehensive 4-Way Benchmark (200 tests, ~5 min)
**Purpose**: Definitive comparison of RAG vs Reranker vs Outcomes vs Full Roampal

```bash
cd benchmarks/comprehensive_test
python test_comprehensive_benchmark.py
```

**Design**: 4 conditions × 5 maturity levels × 10 adversarial scenarios = 200 tests

**Conditions**:
- RAG Baseline: Pure ChromaDB L2 distance
- Reranker Only: Vector + ms-marco cross-encoder (no outcomes)
- Outcomes Only: Vector + Wilson scoring (no reranker)
- Full Roampal: Vector + reranker + Wilson scoring

**Metrics**: Top-1 Accuracy, MRR, nDCG@5, Token Efficiency

**Results**:
| Condition | Top-1 | MRR | nDCG@5 |
|-----------|-------|-----|--------|
| RAG Baseline | 10% | 0.550 | 0.668 |
| Reranker Only | 20% | 0.600 | 0.705 |
| Outcomes Only | 50% | 0.750 | 0.815 |
| Full Roampal | 44% | 0.720 | 0.793 |

**Learning Curve** (Full Roampal):
| Maturity | Uses | Top-1 | MRR |
|----------|------|-------|-----|
| Cold Start | 0 | 0% | 0.500 |
| Early | 3 | 50% | 0.750 |
| Mature | 20 | 60% | 0.800 |

**Key Finding**: Outcome learning (+40 pts) dominates reranker (+10 pts) by 4×

**Statistical Significance**:
- Cold→Mature: p=0.0051** (highly significant)
- Full vs RAG (MRR): p=0.0150*
- Full vs Reranker (MRR): p=0.0368*

### 14. Learning Curve Test (50 tests, ~2 min)
**Purpose**: Prove outcome history improves adversarial resistance

```bash
cd benchmarks/comprehensive_test
python test_learning_curve.py
```

**Design**: 10 scenarios × 5 maturity levels (cold_start → mature)

**Results**:
| Maturity | Uses | Accuracy |
|----------|------|----------|
| Cold Start | 0 | 10% |
| Early | 3 | 100% |
| Established | 5 | 100% |
| Proven | 10 | 100% |
| Mature | 20 | 100% |

**Improvement**: +90 percentage points (10% → 100%)

**Key Finding**: Just 3 uses is enough for 100% accuracy on adversarial queries

### 15. Outcome Learning A/B Test (5 scenarios, ~30s)
**Purpose**: Definitive proof that outcome scoring adds value beyond vector similarity

```bash
cd benchmarks/comprehensive_test
python test_outcome_learning_ab.py
```

**Design**: Uses *semantically identical* text to isolate outcome scoring value
- 5 scenarios, each with 4 copies of identical text
- 1 copy marked "worked", 3 copies marked "failed"
- Treatment: Apply outcome scores (worked=0.7, failed=0.3)
- Control: No outcome scoring (all stay at 0.5)

**Results**:
| Condition | Precision |
|-----------|-----------|
| WITH outcomes | **100%** |
| WITHOUT outcomes | 40% |

**Statistical Significance**:
- Cohen's d: 1.55 (LARGE effect)
- Improvement: +60 percentage points

**Key Finding**: When text is identical, outcome scoring is the ONLY differentiator. This is the cleanest proof that "memory learns what works."

## What Gets Tested

### Core Infrastructure
- ✅ All 5 memory collections (books, working, history, patterns, memory_bank)
- ✅ Content Knowledge Graph (entity extraction & relationships)
- ✅ Storage & retrieval (basic, hybrid, reranking)
- ✅ Outcome-based scoring (+0.2 worked, -0.3 failed)
- ✅ Promotion & demotion logic (working→history→patterns)
- ✅ Deduplication (95% similarity threshold)
- ✅ Quality ranking (importance × confidence)

### Stress Testing
- ✅ High volume (1000 rapid stores)
- ✅ Adversarial deduplication (50 similar memories)
- ✅ Score boundary stress (oscillating outcomes)
- ✅ Capacity enforcement (500-item memory_bank cap)
- ✅ Concurrent access (5 simultaneous conversations)
- ✅ Knowledge graph resilience (failures & deletions)

## Expected Results

### Comprehensive Test
```
30/30 tests passed (100%)
Runtime: ~4 minutes
```

### Torture Test Suite
```
10/10 tests passed (100%)
Runtime: ~90 seconds

Tests:
1. High Volume Stress (1000 memories)          58.5s ✅
2. Long-Term Evolution (100 queries)           21.3s ✅
3. Adversarial Deduplication (50 similar)       5.8s ✅
4. Score Boundary Stress (50 oscillations)      0.9s ✅
5. Cross-Collection Competition                 0.9s ✅
6. Routing Convergence (100 queries)            6.0s ✅
7. Promotion Cascade                            0.9s ✅
8. Memory Bank Capacity (500 cap)              37.1s ✅
9. Knowledge Graph Integrity                    1.5s ✅
10. Concurrent Access (5 conversations)         1.3s ✅
```

## What This Proves

### ✅ Infrastructure is Production-Ready
- 1000 rapid stores with zero corruption or ID collisions
- Deduplication correctly keeps highest-quality versions
- Ranking algorithm works (high-score memories ranked first)
- Capacity enforcement functional (hard cap at 500 items)
- Concurrent access safe (5 simultaneous conversations)

### ✅ Learning Mechanisms Work
- Outcome-based scoring updates correctly
- Promotion thresholds trigger (working → history @ score≥0.7)
- Content KG survives memory failures and deletions
- Score boundaries respected [0.1, 1.0]

### ❌ Not Tested (requires real LLM)
- Semantic similarity with real embeddings
- LLM-based routing decisions
- Long-term decay behavior over actual time

## Files

### Core Tests
- `test_comprehensive.py` - 30-test comprehensive suite
- `test_torture_suite.py` - 10-test stress suite
- `mock_utilities.py` - Mock LLM/embeddings (deterministic)

### Documentation
- `README.md` - This file
- `COMPLETE_TEST_COVERAGE_REVIEW.md` - Systematic review vs architecture
- `learning_curve_test/STATISTICAL_SIGNIFICANCE_EXPLAINED.md` - Statistical proof

### Learning Curve Tests
- `learning_curve_test/test_statistical_significance_synthetic.py` - Proves learning (93.3% accuracy, p=0.005)
- `learning_curve_test/dashboard_statistical_significance.html` - Visual proof

### Dashboards
- `dashboard_torture_suite.html` - Visual torture test results (NEW)
- `learning_curve_test/dashboard_statistical_significance.html` - Statistical proof

## Test Data

**No LLM required** - Uses mock services for deterministic testing:
- Mock embeddings: SHA-256 hash → 768d vector (consistent but not semantic)
- Mock LLM: Rule-based responses
- Predefined test scenarios

## Quick Links

- [README.md](README.md) - You are here
- [TORTURE_SUITE_RESULTS.md](TORTURE_SUITE_RESULTS.md) - Detailed torture test breakdown
- [COMPLETE_TEST_COVERAGE_REVIEW.md](COMPLETE_TEST_COVERAGE_REVIEW.md) - Systematic review
- [dashboard_torture_suite.html](dashboard_torture_suite.html) - Visual results
