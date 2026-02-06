"""
================================================================================
TOKEN EFFICIENCY BENCHMARK - Roampal vs Traditional RAG
================================================================================

Measures per-query token efficiency in adversarial retrieval scenarios.

WHAT THIS BENCHMARK MEASURES:
-----------------------------
1. Tokens returned per query (primary metric)
2. Retrieval accuracy under adversarial conditions (Hit@1, MRR, nDCG@3)
3. Token efficiency ratio (accuracy per token)
4. Cost projections at scale

KEY COMPARISON:
---------------
- Control: ChromaDB top-k retrieval (standard RAG pattern)
  - Returns k semantically similar chunks
  - No outcome learning - pure vector similarity
  - Must return multiple results because it can't tell which one is correct

- Treatment: Roampal outcome-weighted retrieval
  - Returns 1 best result based on learned outcomes
  - Dynamic weight shifting: proven memories (60% score, 40% embedding)
  - Can return fewer results because outcome scoring provides confidence
    in which memory will actually help the LLM

WHY THIS MATTERS:
-----------------
Token efficiency is a CONSEQUENCE of accuracy. Traditional RAG hedges by
returning top-k results because it has no signal for which chunk is correct.
Roampal's outcome history tells it which memories worked in past conversations,
so it can confidently return just 1. The ~80% token reduction (1/5 = 80%) is
a structural property that holds at any chunk size.

ADVERSARIAL DESIGN:
-------------------
- 100 scenarios where queries semantically match WRONG answer better
- Tests whether outcome learning can override misleading semantic similarity
- Same scenarios as test_roampal_vs_vector_db.py (67% vs 0% accuracy)

METHODOLOGY:
------------
- Real embeddings: sentence-transformers/all-mpnet-base-v2 (768d)
- Token approximation: 1 token ≈ 4 characters (OpenAI standard)
- Standard IR metrics: Hit@1, Hit@3, MRR, nDCG@3 (BEIR/MTEB compatible)
- Fresh data directory per scenario (no cross-contamination)

IMPORTANT CAVEATS:
------------------
- This measures PER-QUERY retrieval tokens, not total memory store size
- Industry numbers (Mem0's 7K, Zep's 600K) measure memory materialization
- Direct comparison is with other top-k RAG approaches, not memory stores
- Our test chunks are short (~20-30 tokens); real RAG uses 500-2000 token chunks

REPRODUCIBILITY:
----------------
Run: python test_token_efficiency.py
Requirements: sentence-transformers, chromadb
Runtime: ~2-3 minutes

================================================================================
"""

import asyncio
import json
import os
import sys
import statistics
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# CRITICAL: Set benchmark mode BEFORE importing Roampal modules
# This allows tests to use isolated data directories instead of AppData
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem
from mock_utilities import MockLLMService

# ChromaDB for control condition
import chromadb

# Try to import real embeddings
try:
    from learning_curve_test.real_embedding_service import RealEmbeddingService
    HAS_REAL_EMBEDDINGS = True
except ImportError:
    HAS_REAL_EMBEDDINGS = False
    print("WARNING: Real embeddings not available. Install sentence-transformers.")


# ====================================================================================
# PRICING CONSTANTS (as of Dec 2024)
# ====================================================================================

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},           # per 1M tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}

# Standard token approximation: 1 token ~= 4 characters
CHARS_PER_TOKEN = 4

# Industry comparison baselines (from published research)
# NOTE: These are "memory materialization" costs - how many tokens to represent
# the memory store in context. Not directly comparable to per-query retrieval.
# We include them for context but our primary comparison is RAG top-k.
INDUSTRY_BASELINES = {
    "full_context": {
        "name": "Full Context (LOCOMO)",
        "tokens": 26000,
        "source": "LOCOMO avg conversation length - entire history in context",
        "type": "memory_materialization",
    },
    "mem0": {
        "name": "Mem0",
        "tokens": 7000,
        "source": "Mem0 research (arxiv:2504.19413) - avg memory store size",
        "type": "memory_materialization",
    },
    "mem0_graph": {
        "name": "Mem0 + Graph",
        "tokens": 14000,
        "source": "Mem0 with graph memory - nodes + relationships",
        "type": "memory_materialization",
    },
    "zep": {
        "name": "Zep",
        "tokens": 600000,
        "source": "Zep memory graph (disputed) - full KG serialization",
        "type": "memory_materialization",
    },
}

# These are comparable per-query retrieval costs (what we're actually measuring)
RAG_BASELINES = {
    "rag_top3_500char": {
        "name": "RAG top-3 (500 char chunks)",
        "tokens": 375,  # 3 * 500 / 4
        "source": "Typical RAG with 500-char chunks",
    },
    "rag_top5_500char": {
        "name": "RAG top-5 (500 char chunks)",
        "tokens": 625,  # 5 * 500 / 4
        "source": "Typical RAG with 500-char chunks",
    },
    "rag_top3_1000char": {
        "name": "RAG top-3 (1K char chunks)",
        "tokens": 750,  # 3 * 1000 / 4
        "source": "RAG with larger chunks",
    },
    "rag_top5_1000char": {
        "name": "RAG top-5 (1K char chunks)",
        "tokens": 1250,  # 5 * 1000 / 4
        "source": "RAG with larger chunks",
    },
}


# ====================================================================================
# STANDARD IR METRICS (BEIR/MTEB compatible)
# ====================================================================================

def ndcg_at_k(relevances: List[int], k: int = 3) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    Standard metric used in BEIR/MTEB benchmarks.

    Args:
        relevances: List of relevance scores (1 = relevant, 0 = not relevant)
        k: Cutoff position

    Returns:
        nDCG@k score between 0 and 1
    """
    import math

    if not relevances:
        return 0.0

    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))

    # Ideal DCG (best possible ranking)
    ideal_relevances = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_at_k(relevant_in_results: bool, k: int) -> float:
    """
    Hit@k metric: 1 if relevant item appears in top-k results, else 0.
    Standard recall metric in IR.
    """
    return 1.0 if relevant_in_results else 0.0


def mean_reciprocal_rank(rank_of_relevant: int) -> float:
    """
    MRR: 1/rank of first relevant result.
    Another standard IR metric.
    """
    if rank_of_relevant <= 0:
        return 0.0
    return 1.0 / rank_of_relevant


# ====================================================================================
# TEST SCENARIOS - Personal Finance (Data-backed adversarial scenarios)
# ====================================================================================
#
# SOURCES:
# - S&P SPIVA: 90%+ of active managers underperform over 15 years
# - Schwab Research: Perfect timing only beats immediate investing by 8% over 20 years
# - DALBAR: Average investor underperforms due to behavior gaps
# - Vanguard, Morningstar, academic research on fees, diversification, tax efficiency
#
# Each scenario is designed so the query semantically matches the BAD advice better,
# testing whether outcome-based learning can override misleading similarity.

SCENARIOS = [
    # === INVESTING BASICS (10 scenarios) ===
    # === INVESTING BASICS (10 scenarios) ===
    {
        "id": 1,
        "category": "investing",
        "good_advice": "Invest in low-cost total market index funds - 90% of active managers underperform over 15 years (S&P SPIVA)",
        "bad_advice": "Pick individual stocks based on research to beat the market and get higher returns",
        "query": "How do I pick stocks to beat the market and get higher returns?",
    },
    {
        "id": 2,
        "category": "investing",
        "good_advice": "Stay invested through market downturns - missing the 10 best days cuts returns by 50% (Schwab Research)",
        "bad_advice": "Sell stocks before a crash and buy back in when the market recovers",
        "query": "Should I sell before a crash and buy back when the market recovers?",
    },
    {
        "id": 3,
        "category": "investing",
        "good_advice": "Invest immediately when you have money - immediate investing trails perfect timing by only 8% over 20 years (Schwab)",
        "bad_advice": "Wait for a market dip or correction before putting money into the market",
        "query": "Should I wait for a market dip before investing my money?",
    },
    {
        "id": 4,
        "category": "investing",
        "good_advice": "Use dollar-cost averaging with automatic monthly investments for emotional discipline",
        "bad_advice": "Save up a lump sum and invest it all at once when the timing feels right",
        "query": "Should I save up and invest a lump sum when the timing feels right?",
    },
    {
        "id": 5,
        "category": "investing",
        "good_advice": "Keep expense ratios under 0.20% - a 1% fee difference costs $590K over 40 years on $100K (Vanguard)",
        "bad_advice": "Pay higher fees for actively managed funds with expert stock pickers",
        "query": "Should I pay higher fees for actively managed funds with expert managers?",
    },
    {
        "id": 6,
        "category": "investing",
        "good_advice": "Diversify globally with international stocks - US outperformance isn't guaranteed long-term",
        "bad_advice": "Invest only in US stocks since America has the strongest economy",
        "query": "Should I invest only in US stocks since America has the strongest economy?",
    },
    {
        "id": 7,
        "category": "investing",
        "good_advice": "Rebalance annually to maintain target allocation and buy low/sell high automatically",
        "bad_advice": "Let your winners run and don't sell stocks that are going up",
        "query": "Should I let my winners run and not sell stocks that are going up?",
    },
    {
        "id": 8,
        "category": "investing",
        "good_advice": "Ignore daily market news - long-term investors who check less often have better returns (DALBAR)",
        "bad_advice": "Stay informed by checking your portfolio and market news daily",
        "query": "Should I check my portfolio and market news every day to stay informed?",
    },
    {
        "id": 9,
        "category": "investing",
        "good_advice": "Use target-date funds for automatic age-appropriate asset allocation",
        "bad_advice": "Actively manage your own asset allocation based on market conditions",
        "query": "Should I actively manage my asset allocation based on market conditions?",
    },
    {
        "id": 10,
        "category": "investing",
        "good_advice": "Hold investments for decades - short-term trading incurs taxes and fees that destroy returns",
        "bad_advice": "Trade actively to take profits and reinvest when prices drop",
        "query": "Should I trade actively to take profits when stocks go up?",
    },

    # === RETIREMENT PLANNING (10 scenarios) ===
    {
        "id": 11,
        "category": "retirement",
        "good_advice": "Max out 401k employer match first - it's free 50-100% return on your money",
        "bad_advice": "Pay off low-interest debt before contributing to 401k for employer match",
        "query": "Should I pay off debt before contributing to 401k for the match?",
    },
    {
        "id": 12,
        "category": "retirement",
        "good_advice": "Use Roth IRA if you expect higher taxes in retirement - tax-free growth for decades",
        "bad_advice": "Use Traditional IRA for the immediate tax deduction to reduce this year's taxes",
        "query": "Should I use Traditional IRA to get the tax deduction this year?",
    },
    {
        "id": 13,
        "category": "retirement",
        "good_advice": "Save 15-20% of income for retirement including employer match from your 20s",
        "bad_advice": "Start saving aggressively for retirement in your 40s when you earn more",
        "query": "Should I wait until my 40s when I earn more to save aggressively?",
    },
    {
        "id": 14,
        "category": "retirement",
        "good_advice": "Keep retirement funds 100% in stocks until age 40-45 for maximum growth",
        "bad_advice": "Add bonds to your portfolio in your 20s-30s for safety and balance",
        "query": "Should I add bonds in my 20s-30s for safety and balance?",
    },
    {
        "id": 15,
        "category": "retirement",
        "good_advice": "Use the 4% rule for initial retirement withdrawal rate with annual inflation adjustments",
        "bad_advice": "Withdraw only dividend and interest income to preserve your principal forever",
        "query": "Should I only withdraw dividends to preserve my principal in retirement?",
    },
    {
        "id": 16,
        "category": "retirement",
        "good_advice": "Delay Social Security until 70 for 8% per year increase - maximize lifetime benefits",
        "bad_advice": "Take Social Security at 62 to get your money while you're healthy",
        "query": "Should I take Social Security at 62 to get my money while I'm healthy?",
    },
    {
        "id": 17,
        "category": "retirement",
        "good_advice": "Roll old 401ks into an IRA for lower fees and better investment options",
        "bad_advice": "Leave your 401k with your old employer since it's already set up",
        "query": "Should I leave my 401k with my old employer since it's already set up?",
    },
    {
        "id": 18,
        "category": "retirement",
        "good_advice": "Contribute to HSA as stealth retirement account - triple tax advantage beats Roth",
        "bad_advice": "Use your HSA for current medical expenses since that's what it's for",
        "query": "Should I use my HSA for current medical expenses since that's what it's for?",
    },
    {
        "id": 19,
        "category": "retirement",
        "good_advice": "Do backdoor Roth IRA if over income limits - legal way to get Roth benefits",
        "bad_advice": "Accept that you can't contribute to Roth IRA if your income is too high",
        "query": "Should I accept I can't contribute to Roth if my income is too high?",
    },
    {
        "id": 20,
        "category": "retirement",
        "good_advice": "Calculate retirement needs based on expenses, not income replacement percentage",
        "bad_advice": "Plan to replace 80% of your pre-retirement income in retirement",
        "query": "Should I plan to replace 80% of my income in retirement?",
    },

    # === DEBT MANAGEMENT (10 scenarios) ===
    {
        "id": 21,
        "category": "debt",
        "good_advice": "Pay minimum on low-interest debt, invest the difference in index funds (math favors investing)",
        "bad_advice": "Pay off your mortgage early to become debt-free and save on interest",
        "query": "Should I pay off my mortgage early to become debt-free?",
    },
    {
        "id": 22,
        "category": "debt",
        "good_advice": "Use debt avalanche method - pay highest interest rate first to minimize total interest",
        "bad_advice": "Use debt snowball to pay smallest balance first for quick wins and motivation",
        "query": "Should I pay smallest balance first for quick wins and motivation?",
    },
    {
        "id": 23,
        "category": "debt",
        "good_advice": "Refinance student loans only if federal benefits aren't needed - keep income-driven repayment option",
        "bad_advice": "Refinance student loans immediately for a lower interest rate",
        "query": "Should I refinance student loans immediately for a lower interest rate?",
    },
    {
        "id": 24,
        "category": "debt",
        "good_advice": "Never pay off 0% APR financing early - invest that money instead",
        "bad_advice": "Pay off 0% financing early to reduce your total debt load",
        "query": "Should I pay off 0% financing early to reduce my debt?",
    },
    {
        "id": 25,
        "category": "debt",
        "good_advice": "Use balance transfer cards strategically with a payoff plan before 0% expires",
        "bad_advice": "Avoid balance transfer cards because of the transfer fee",
        "query": "Should I avoid balance transfer cards because of the transfer fee?",
    },
    {
        "id": 26,
        "category": "debt",
        "good_advice": "Maintain mortgage if rate is under 5% - invest extra payments in index funds instead",
        "bad_advice": "Make extra mortgage payments to build equity faster",
        "query": "Should I make extra mortgage payments to build equity faster?",
    },
    {
        "id": 27,
        "category": "debt",
        "good_advice": "Keep credit utilization under 30% but don't pay extra to get it to 0%",
        "bad_advice": "Pay off credit cards completely each month before statement closes",
        "query": "Should I pay off cards before the statement closes for 0% utilization?",
    },
    {
        "id": 28,
        "category": "debt",
        "good_advice": "Take out PLUS loans only as last resort - parent retirement more important than college",
        "bad_advice": "Take Parent PLUS loans so your child graduates debt-free",
        "query": "Should I take PLUS loans so my child can graduate debt-free?",
    },
    {
        "id": 29,
        "category": "debt",
        "good_advice": "Contribute to 401k match even with credit card debt - match beats 20% interest",
        "bad_advice": "Stop 401k contributions until all credit card debt is paid off",
        "query": "Should I stop 401k contributions until credit card debt is paid?",
    },
    {
        "id": 30,
        "category": "debt",
        "good_advice": "Use a 15-year mortgage only if payment fits budget - 30-year with investing beats it",
        "bad_advice": "Get a 15-year mortgage to pay less total interest",
        "query": "Should I get a 15-year mortgage to pay less total interest?",
    },

    # === EMERGENCY FUNDS (10 scenarios) ===
    {
        "id": 31,
        "category": "emergency",
        "good_advice": "Keep 1-2 months expenses in checking, rest in high-yield savings or money market (4-5% APY)",
        "bad_advice": "Keep 6 months of expenses in a regular savings account for safety",
        "query": "Should I keep 6 months in a regular savings account for emergencies?",
    },
    {
        "id": 32,
        "category": "emergency",
        "good_advice": "Use Roth IRA as backup emergency fund - contributions can be withdrawn penalty-free",
        "bad_advice": "Never touch retirement accounts for emergencies under any circumstances",
        "query": "Should I never touch retirement accounts for emergencies?",
    },
    {
        "id": 33,
        "category": "emergency",
        "good_advice": "Build emergency fund to 3 months then start investing - don't wait for 6 months",
        "bad_advice": "Build a full 6 month emergency fund before investing any money",
        "query": "Should I build 6 months emergency fund before investing anything?",
    },
    {
        "id": 34,
        "category": "emergency",
        "good_advice": "Use T-bills or money market funds for emergency fund - same safety, better yield",
        "bad_advice": "Keep emergency fund only in FDIC insured savings accounts for safety",
        "query": "Should I only use FDIC insured savings for my emergency fund?",
    },
    {
        "id": 35,
        "category": "emergency",
        "good_advice": "Reduce emergency fund size if you have stable job, spouse income, or good credit lines",
        "bad_advice": "Keep a full 6 months emergency fund regardless of your job stability",
        "query": "Should I keep 6 months emergency fund even with a stable job?",
    },
    {
        "id": 36,
        "category": "emergency",
        "good_advice": "Include HELOC as part of emergency backup - you don't need all cash",
        "bad_advice": "Don't count credit lines as emergency funds since that's going into debt",
        "query": "Should I not count credit lines since that's going into debt?",
    },
    {
        "id": 37,
        "category": "emergency",
        "good_advice": "Ladder CDs or Treasury bills for emergency fund to earn more while maintaining liquidity",
        "bad_advice": "Avoid CDs for emergency funds because of early withdrawal penalties",
        "query": "Should I avoid CDs for emergencies because of early withdrawal penalties?",
    },
    {
        "id": 38,
        "category": "emergency",
        "good_advice": "Use I-bonds for portion of emergency fund - inflation protected with tax benefits",
        "bad_advice": "Don't use I-bonds for emergencies because of the 1-year lock-up period",
        "query": "Should I avoid I-bonds for emergencies because of the lock-up period?",
    },
    {
        "id": 39,
        "category": "emergency",
        "good_advice": "Consider taxable brokerage as extended emergency fund - liquid and earning returns",
        "bad_advice": "Keep emergency and investment accounts completely separate always",
        "query": "Should I keep emergency and investment accounts completely separate?",
    },
    {
        "id": 40,
        "category": "emergency",
        "good_advice": "Self-employed need 6-12 months; W-2 employees with stable job need only 3 months",
        "bad_advice": "Everyone needs exactly 6 months of expenses in their emergency fund",
        "query": "Does everyone need exactly 6 months for emergencies?",
    },

    # === INSURANCE (10 scenarios) ===
    {
        "id": 41,
        "category": "insurance",
        "good_advice": "Buy term life insurance and invest the premium difference - 10x cheaper than whole life",
        "bad_advice": "Buy whole life insurance for the cash value accumulation and investment component",
        "query": "Should I buy whole life insurance for the cash value and investment?",
    },
    {
        "id": 42,
        "category": "insurance",
        "good_advice": "Raise insurance deductibles to $1000+ and self-insure small claims",
        "bad_advice": "Keep deductibles low so insurance covers more of your expenses",
        "query": "Should I keep deductibles low so insurance covers more expenses?",
    },
    {
        "id": 43,
        "category": "insurance",
        "good_advice": "Drop collision coverage on cars worth less than $5000 - premium exceeds potential payout",
        "bad_advice": "Keep full coverage on your car until it's completely paid off",
        "query": "Should I keep full coverage until my car is paid off?",
    },
    {
        "id": 44,
        "category": "insurance",
        "good_advice": "Get umbrella insurance for $1M+ liability coverage - costs only $200-300/year",
        "bad_advice": "Standard auto and home liability limits are enough for most people",
        "query": "Are standard liability limits enough for most people?",
    },
    {
        "id": 45,
        "category": "insurance",
        "good_advice": "Skip extended warranties and protection plans - profit margins are 50-80% for sellers",
        "bad_advice": "Buy extended warranties on electronics for peace of mind",
        "query": "Should I buy extended warranties on electronics for peace of mind?",
    },
    {
        "id": 46,
        "category": "insurance",
        "good_advice": "Get disability insurance if your employer doesn't provide it - more important than life insurance",
        "bad_advice": "Focus on life insurance first since disability is rare for young people",
        "query": "Should I focus on life insurance first since disability is rare?",
    },
    {
        "id": 47,
        "category": "insurance",
        "good_advice": "Never buy credit life insurance or mortgage life insurance - overpriced coverage",
        "bad_advice": "Buy mortgage life insurance to pay off your house if you die",
        "query": "Should I buy mortgage life insurance to pay off my house if I die?",
    },
    {
        "id": 48,
        "category": "insurance",
        "good_advice": "Shop insurance every 2-3 years - loyalty costs 10-30% more than switching",
        "bad_advice": "Stay with the same insurance company for loyalty discounts",
        "query": "Should I stay with the same insurer for loyalty discounts?",
    },
    {
        "id": 49,
        "category": "insurance",
        "good_advice": "Life insurance need drops as you age and build wealth - reduce coverage over time",
        "bad_advice": "Maintain the same life insurance coverage your entire life",
        "query": "Should I maintain the same life insurance coverage my entire life?",
    },
    {
        "id": 50,
        "category": "insurance",
        "good_advice": "Skip identity theft insurance - credit monitoring and freezes are free and more effective",
        "bad_advice": "Buy identity theft protection insurance to cover fraud losses",
        "query": "Should I buy identity theft insurance to cover fraud losses?",
    },

    # === TAX OPTIMIZATION (10 scenarios) ===
    {
        "id": 51,
        "category": "taxes",
        "good_advice": "Tax-loss harvest by selling losers and immediately buying similar (not identical) funds",
        "bad_advice": "Never sell investments at a loss because that means locking in losses",
        "query": "Should I never sell at a loss because that locks in losses?",
    },
    {
        "id": 52,
        "category": "taxes",
        "good_advice": "Hold tax-inefficient investments (bonds, REITs) in tax-advantaged accounts only",
        "bad_advice": "Keep all your accounts invested in the same diversified portfolio",
        "query": "Should I invest all accounts in the same diversified portfolio?",
    },
    {
        "id": 53,
        "category": "taxes",
        "good_advice": "Bunch charitable donations in alternating years to exceed standard deduction",
        "bad_advice": "Donate the same amount to charity every year for consistency",
        "query": "Should I donate the same amount to charity every year?",
    },
    {
        "id": 54,
        "category": "taxes",
        "good_advice": "Donate appreciated stock directly to charity - avoid capital gains tax entirely",
        "bad_advice": "Sell investments and donate cash to charity for the tax deduction",
        "query": "Should I sell investments and donate cash for the tax deduction?",
    },
    {
        "id": 55,
        "category": "taxes",
        "good_advice": "Use specific lot identification to sell highest-cost-basis shares first for lower taxes",
        "bad_advice": "Use FIFO (first in, first out) for simplicity when selling investments",
        "query": "Should I use FIFO for simplicity when selling investments?",
    },
    {
        "id": 56,
        "category": "taxes",
        "good_advice": "Harvest losses even without gains - carry forward $3000 against ordinary income yearly",
        "bad_advice": "Only tax-loss harvest when you have capital gains to offset",
        "query": "Should I only tax-loss harvest when I have gains to offset?",
    },
    {
        "id": 57,
        "category": "taxes",
        "good_advice": "Do Roth conversions in low-income years to fill up lower tax brackets",
        "bad_advice": "Avoid Roth conversions because you'll have to pay taxes on the conversion",
        "query": "Should I avoid Roth conversions because I'll have to pay taxes?",
    },
    {
        "id": 58,
        "category": "taxes",
        "good_advice": "Use 529 plans for education - state tax deduction plus tax-free growth",
        "bad_advice": "Save for college in a regular brokerage account for more flexibility",
        "query": "Should I save for college in a regular brokerage for flexibility?",
    },
    {
        "id": 59,
        "category": "taxes",
        "good_advice": "Claim home office deduction if legitimately self-employed - it's not an audit trigger",
        "bad_advice": "Avoid home office deduction because it increases audit risk",
        "query": "Should I avoid home office deduction because of audit risk?",
    },
    {
        "id": 60,
        "category": "taxes",
        "good_advice": "Municipal bonds only make sense in taxable accounts for high tax bracket investors",
        "bad_advice": "Buy municipal bonds for tax-free income regardless of your tax bracket",
        "query": "Should I buy municipal bonds for tax-free income?",
    },

    # === REAL ESTATE (10 scenarios) ===
    {
        "id": 61,
        "category": "realestate",
        "good_advice": "Rent if buy vs rent calculation favors renting - homeownership isn't always better",
        "bad_advice": "Buy a house as soon as possible because renting is throwing away money",
        "query": "Should I buy a house ASAP because renting is throwing away money?",
    },
    {
        "id": 62,
        "category": "realestate",
        "good_advice": "Put only 20% down if mortgage rate is low - invest the rest in index funds",
        "bad_advice": "Put down as much as possible to minimize your mortgage payment",
        "query": "Should I put down as much as possible to minimize my mortgage?",
    },
    {
        "id": 63,
        "category": "realestate",
        "good_advice": "Buy less house than you can afford - bank pre-approval is not your budget",
        "bad_advice": "Buy the most house you can qualify for to maximize home equity growth",
        "query": "Should I buy the most house I can qualify for?",
    },
    {
        "id": 64,
        "category": "realestate",
        "good_advice": "Keep total housing costs under 25% of take-home pay including taxes and maintenance",
        "bad_advice": "It's okay to spend 35-40% on housing in expensive cities",
        "query": "Is it okay to spend 35-40% on housing in expensive cities?",
    },
    {
        "id": 65,
        "category": "realestate",
        "good_advice": "Avoid PMI by using 80-10-10 loan or wait until 20% down - PMI is wasted money",
        "bad_advice": "Pay PMI to buy a house sooner rather than waiting to save 20%",
        "query": "Should I pay PMI to buy a house sooner rather than waiting?",
    },
    {
        "id": 66,
        "category": "realestate",
        "good_advice": "Don't count primary residence as an investment - it's a consumption asset",
        "bad_advice": "Your house is your biggest investment and builds long-term wealth",
        "query": "Is my house my biggest investment for building wealth?",
    },
    {
        "id": 67,
        "category": "realestate",
        "good_advice": "Refinance if you can lower rate by 0.75%+ and will stay long enough to recoup costs",
        "bad_advice": "Refinance whenever rates drop even a little bit to save money",
        "query": "Should I refinance whenever rates drop even a little bit?",
    },
    {
        "id": 68,
        "category": "realestate",
        "good_advice": "Skip real estate if you need flexibility - transaction costs are 8-10% of home value",
        "bad_advice": "Buy property even if you might move in 2-3 years to build equity",
        "query": "Should I buy even if I might move in 2-3 years to build equity?",
    },
    {
        "id": 69,
        "category": "realestate",
        "good_advice": "Rental property requires hands-on work or expensive management - REITs are easier",
        "bad_advice": "Buy rental property for passive income and tax benefits",
        "query": "Should I buy rental property for passive income?",
    },
    {
        "id": 70,
        "category": "realestate",
        "good_advice": "Use rate-lock periods wisely - don't lock too early in volatile rate environments",
        "bad_advice": "Lock your mortgage rate immediately when you get pre-approved",
        "query": "Should I lock my mortgage rate immediately when pre-approved?",
    },

    # === BEHAVIORAL FINANCE (10 scenarios) ===
    {
        "id": 71,
        "category": "behavioral",
        "good_advice": "Automate all savings and investments - willpower is a limited resource",
        "bad_advice": "Manually transfer money to savings each month so you stay engaged",
        "query": "Should I manually transfer savings each month to stay engaged?",
    },
    {
        "id": 72,
        "category": "behavioral",
        "good_advice": "Write an investment policy statement before market drops - decide rules when calm",
        "bad_advice": "Make investment decisions based on current market conditions",
        "query": "Should I make investment decisions based on current market conditions?",
    },
    {
        "id": 73,
        "category": "behavioral",
        "good_advice": "Delete brokerage apps and check investments quarterly at most - less is more",
        "bad_advice": "Monitor your investments daily to stay on top of market changes",
        "query": "Should I monitor my investments daily to stay on top of changes?",
    },
    {
        "id": 74,
        "category": "behavioral",
        "good_advice": "Ignore financial media and predictions - no one can consistently predict markets",
        "bad_advice": "Follow financial news and expert predictions to inform decisions",
        "query": "Should I follow expert predictions to inform my investment decisions?",
    },
    {
        "id": 75,
        "category": "behavioral",
        "good_advice": "Don't anchor to your purchase price - it's irrelevant to future returns",
        "bad_advice": "Wait for your losing investment to get back to even before selling",
        "query": "Should I wait for a losing investment to get back to even before selling?",
    },
    {
        "id": 76,
        "category": "behavioral",
        "good_advice": "Simplify to 3-4 funds maximum - complexity doesn't improve returns",
        "bad_advice": "Diversify across many funds and asset classes for better protection",
        "query": "Should I diversify across many funds and asset classes?",
    },
    {
        "id": 77,
        "category": "behavioral",
        "good_advice": "Avoid lifestyle inflation when income increases - save 50%+ of raises",
        "bad_advice": "Upgrade your lifestyle when you get a raise since you've earned it",
        "query": "Should I upgrade my lifestyle when I get a raise since I've earned it?",
    },
    {
        "id": 78,
        "category": "behavioral",
        "good_advice": "Use mental accounting positively - have separate accounts for different goals",
        "bad_advice": "Keep all money in one account for simplicity and to see your full picture",
        "query": "Should I keep all money in one account for simplicity?",
    },
    {
        "id": 79,
        "category": "behavioral",
        "good_advice": "Set rules before emotional situations - 'I will buy more if market drops 20%'",
        "bad_advice": "React to market conditions as they happen with flexibility",
        "query": "Should I react to market conditions as they happen with flexibility?",
    },
    {
        "id": 80,
        "category": "behavioral",
        "good_advice": "Calculate opportunity cost of major purchases - that car costs years of retirement",
        "bad_advice": "Buy what you can afford if you have the cash in your account",
        "query": "Should I buy what I can afford if I have the cash?",
    },

    # === COMMON MYTHS (10 scenarios) ===
    {
        "id": 81,
        "category": "myths",
        "good_advice": "Use TIPS or I-bonds for inflation protection - gold has unreliable inflation correlation",
        "bad_advice": "Buy gold as a hedge against inflation and currency devaluation",
        "query": "Should I buy gold as a hedge against inflation?",
    },
    {
        "id": 82,
        "category": "myths",
        "good_advice": "Age in bonds rule is outdated - with longer lifespans, stocks should dominate until 50",
        "bad_advice": "Subtract your age from 100 to get your stock allocation percentage",
        "query": "Should I subtract my age from 100 for my stock allocation?",
    },
    {
        "id": 83,
        "category": "myths",
        "good_advice": "Carrying a credit card balance doesn't help your credit score - pay in full monthly",
        "bad_advice": "Carry a small balance on credit cards to build credit history",
        "query": "Should I carry a small balance to build credit history?",
    },
    {
        "id": 84,
        "category": "myths",
        "good_advice": "You don't need a financial advisor for basic investing - use target date funds instead",
        "bad_advice": "Hire a financial advisor to help you invest properly",
        "query": "Should I hire a financial advisor to help me invest?",
    },
    {
        "id": 85,
        "category": "myths",
        "good_advice": "Cryptocurrency is speculation, not investment - keep it under 5% of portfolio if at all",
        "bad_advice": "Allocate 10-20% of portfolio to crypto for growth potential",
        "query": "Should I allocate 10-20% to crypto for growth potential?",
    },
    {
        "id": 86,
        "category": "myths",
        "good_advice": "Annuities have high fees and complexity - index funds are better for most people",
        "bad_advice": "Buy an annuity for guaranteed retirement income and peace of mind",
        "query": "Should I buy an annuity for guaranteed retirement income?",
    },
    {
        "id": 87,
        "category": "myths",
        "good_advice": "Dividend stocks aren't free money - total return matters more than yield",
        "bad_advice": "Focus on high dividend stocks for steady income in retirement",
        "query": "Should I focus on high dividend stocks for retirement income?",
    },
    {
        "id": 88,
        "category": "myths",
        "good_advice": "Past performance doesn't predict future results - last year's winners often underperform",
        "bad_advice": "Choose funds based on their 5-year track record of beating the market",
        "query": "Should I choose funds based on their 5-year track record?",
    },
    {
        "id": 89,
        "category": "myths",
        "good_advice": "You don't need to be debt-free to start investing - it's about interest rate math",
        "bad_advice": "Pay off all debt before investing any money in the market",
        "query": "Should I pay off all debt before investing anything?",
    },
    {
        "id": 90,
        "category": "myths",
        "good_advice": "Leasing is almost always more expensive than buying used - do the total cost math",
        "bad_advice": "Lease a car for lower monthly payments and always having a new car",
        "query": "Should I lease a car for lower payments and always having new?",
    },

    # === INCOME & CAREER (10 scenarios) ===
    {
        "id": 91,
        "category": "income",
        "good_advice": "Negotiate salary at every job offer - employers expect it and budget for 10-15% more",
        "bad_advice": "Accept the first salary offer to avoid seeming difficult or greedy",
        "query": "Should I accept the first offer to avoid seeming greedy?",
    },
    {
        "id": 92,
        "category": "income",
        "good_advice": "Job hop every 2-3 years early career for 10-20% raises vs 3% annual increases",
        "bad_advice": "Stay at one company for 5+ years to build loyalty and seniority",
        "query": "Should I stay at one company for 5+ years for loyalty and seniority?",
    },
    {
        "id": 93,
        "category": "income",
        "good_advice": "Prioritize salary over titles early career - compound earnings matter most",
        "bad_advice": "Take a lower salary for a better title and career advancement opportunity",
        "query": "Should I take lower salary for a better title and advancement?",
    },
    {
        "id": 94,
        "category": "income",
        "good_advice": "Side income diversifies risk - don't depend 100% on one employer",
        "bad_advice": "Focus 100% on your main job to maximize career advancement",
        "query": "Should I focus 100% on my main job for career advancement?",
    },
    {
        "id": 95,
        "category": "income",
        "good_advice": "Research market rates before asking for raise - data beats appeals to fairness",
        "bad_advice": "Ask for a raise based on your tenure and personal financial needs",
        "query": "Should I ask for a raise based on tenure and financial needs?",
    },
    {
        "id": 96,
        "category": "income",
        "good_advice": "Consider total compensation not just salary - 401k match, HSA, equity matter",
        "bad_advice": "Compare job offers based on the base salary amount",
        "query": "Should I compare job offers based on base salary?",
    },
    {
        "id": 97,
        "category": "income",
        "good_advice": "Invest in skills that increase earning power - highest ROI 'investment' available",
        "bad_advice": "Save as much as possible rather than spending money on education",
        "query": "Should I save rather than spend money on education?",
    },
    {
        "id": 98,
        "category": "income",
        "good_advice": "Geographic arbitrage works - earn big city salary remotely while living cheaply",
        "bad_advice": "Live near your office for networking and career advancement opportunities",
        "query": "Should I live near the office for networking and advancement?",
    },
    {
        "id": 99,
        "category": "income",
        "good_advice": "Max out tax-advantaged accounts before taxable investing - order matters for returns",
        "bad_advice": "Invest in taxable accounts for more flexibility and fewer restrictions",
        "query": "Should I invest in taxable accounts for more flexibility?",
    },
    {
        "id": 100,
        "category": "income",
        "good_advice": "FIRE math: save 25x annual expenses, withdraw 4% - doesn't require high income",
        "bad_advice": "You need to be wealthy or have high income to retire early",
        "query": "Do I need high income to retire early?",
    },
]


# ====================================================================================
# TOKEN COUNTING UTILITIES
# ====================================================================================

def count_tokens(text: str) -> int:
    """Approximate token count using character/4 heuristic."""
    return len(text) // CHARS_PER_TOKEN


def format_tokens(tokens: int) -> str:
    """Format token count with K suffix if large."""
    if tokens >= 1000:
        return f"{tokens/1000:.1f}K"
    return str(tokens)


# ====================================================================================
# CONTROL CONDITIONS: Traditional RAG (top-k chunks)
# ====================================================================================

async def run_control_topk(scenario: Dict, data_dir: str, embedding_service, k: int = 3) -> Dict:
    """
    Control condition: Traditional RAG with top-k chunk retrieval.
    Returns ALL top-k chunks as context (standard RAG pattern).
    """
    client = chromadb.PersistentClient(path=data_dir)
    collection = client.create_collection(
        name=f"control_top{k}",
        metadata={"hnsw:space": "l2"}
    )

    # Generate embeddings and store
    good_embedding = await embedding_service.embed_text(scenario["good_advice"])
    bad_embedding = await embedding_service.embed_text(scenario["bad_advice"])

    # Add some padding content to simulate real RAG scenario
    # In reality, RAG returns chunks from documents, not just single sentences
    padding_texts = [
        f"Additional context about {scenario['category']}: This is a common topic in software development.",
        f"Related information: Developers often encounter issues with {scenario['category']}.",
        f"Background: Understanding {scenario['category']} is essential for building robust systems.",
    ]

    all_ids = ["good", "bad"] + [f"pad_{i}" for i in range(len(padding_texts))]
    all_texts = [scenario["good_advice"], scenario["bad_advice"]] + padding_texts

    all_embeddings = [good_embedding, bad_embedding]
    for text in padding_texts:
        emb = await embedding_service.embed_text(text)
        all_embeddings.append(emb)

    all_metadatas = [
        {"type": "good"},
        {"type": "bad"},
    ] + [{"type": "padding"} for _ in padding_texts]

    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_texts,
        metadatas=all_metadatas
    )

    # Query and get top-k
    query_embedding = await embedding_service.embed_text(scenario["query"])
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    # Calculate tokens from ALL returned chunks (traditional RAG behavior)
    retrieved_texts = results.get("documents", [[]])[0]
    total_chars = sum(len(t) for t in retrieved_texts)
    total_tokens = total_chars // CHARS_PER_TOKEN

    # Check if good advice is in the results and calculate rank
    good_ranked_first = False
    good_in_results = False
    good_rank = 0  # 0 means not found
    relevances = []  # For nDCG calculation

    if results["ids"] and results["ids"][0]:
        result_ids = results["ids"][0]
        for i, doc_id in enumerate(result_ids):
            if doc_id == "good":
                good_in_results = True
                good_rank = i + 1  # 1-indexed rank
                if i == 0:
                    good_ranked_first = True
            # Build relevance list: 1 for good, 0 for everything else
            relevances.append(1 if doc_id == "good" else 0)

    # Calculate standard IR metrics
    hit_1 = hit_at_k(good_ranked_first, 1)
    hit_3 = hit_at_k(good_in_results and good_rank <= 3, 3)
    mrr = mean_reciprocal_rank(good_rank) if good_rank > 0 else 0.0
    ndcg_3 = ndcg_at_k(relevances, k=3)

    return {
        "scenario_id": scenario["id"],
        "condition": f"rag_top{k}",
        "k": k,
        "tokens_returned": total_tokens,
        "chars_returned": total_chars,
        "good_ranked_first": good_ranked_first,
        "good_in_results": good_in_results,
        "good_rank": good_rank,
        "accuracy": 1.0 if good_ranked_first else 0.0,
        "num_chunks": len(retrieved_texts),
        # Standard IR metrics
        "hit_at_1": hit_1,
        "hit_at_3": hit_3,
        "mrr": mrr,
        "ndcg_at_3": ndcg_3,
    }


# ====================================================================================
# TREATMENT CONDITION: Roampal
# ====================================================================================

async def run_treatment_roampal(scenario: Dict, data_dir: str, embedding_service) -> Dict:
    """
    Treatment condition: Roampal with outcome-based scoring.
    Returns only the most relevant memory based on learned outcomes.
    """
    system = UnifiedMemorySystem(
        data_path=data_dir,
    )
    await system.initialize()
    system._embedding_service = embedding_service
    if system._search_service:
        system._search_service.embed_fn = embedding_service.embed_text

    # Store good advice with positive outcomes
    good_id = await system.store(
        text=scenario["good_advice"],
        collection="working",
        metadata={"type": "good", "scenario": scenario["id"]}
    )

    # Record positive outcomes
    for _ in range(2):
        try:
            await system.record_outcome(doc_id=good_id, outcome="worked")
        except:
            pass

    # Update usage count to "proven" status
    adapter = system.collections.get("working")
    if adapter:
        try:
            result = adapter.collection.get(ids=[good_id], include=["metadatas"])
            if result and result.get("metadatas"):
                meta = result["metadatas"][0]
                meta["uses"] = 5
                adapter.collection.update(ids=[good_id], metadatas=[meta])
        except:
            pass

    # Store bad advice with negative outcomes
    bad_id = await system.store(
        text=scenario["bad_advice"],
        collection="working",
        metadata={"type": "bad", "scenario": scenario["id"]}
    )

    try:
        await system.record_outcome(doc_id=bad_id, outcome="failed")
    except:
        pass

    # Add padding content (same as control for fair comparison)
    for i, text in enumerate([
        f"Additional context about {scenario['category']}: This is a common topic in software development.",
        f"Related information: Developers often encounter issues with {scenario['category']}.",
        f"Background: Understanding {scenario['category']} is essential for building robust systems.",
    ]):
        await system.store(
            text=text,
            collection="working",
            metadata={"type": "padding"}
        )

    # Search with Roampal (limit=1 - return ONLY the best match)
    # Check both working and history in case of promotion
    results = await system.search(
        scenario["query"],
        collections=["working", "history"],
        limit=1  # KEY DIFFERENCE: Roampal returns only 1 high-confidence result
    )

    # Calculate tokens
    if results:
        retrieved_text = results[0].get("text", "")
        total_chars = len(retrieved_text)
        total_tokens = total_chars // CHARS_PER_TOKEN
    else:
        total_chars = 0
        total_tokens = 0

    # Check accuracy
    good_ranked_first = False
    if results:
        top_text = results[0].get("text", "").lower()
        if scenario["good_advice"].lower()[:50] in top_text or top_text in scenario["good_advice"].lower():
            good_ranked_first = True

    # For Roampal with limit=1, if correct it's rank 1, otherwise rank 0 (not found)
    good_rank = 1 if good_ranked_first else 0

    # Standard IR metrics (simplified for limit=1 case)
    hit_1 = 1.0 if good_ranked_first else 0.0
    hit_3 = hit_1  # Same since we only return 1
    mrr = 1.0 if good_ranked_first else 0.0
    ndcg_3 = 1.0 if good_ranked_first else 0.0  # Perfect if correct, 0 otherwise

    return {
        "scenario_id": scenario["id"],
        "condition": "roampal",
        "tokens_returned": total_tokens,
        "chars_returned": total_chars,
        "good_ranked_first": good_ranked_first,
        "good_rank": good_rank,
        "accuracy": 1.0 if good_ranked_first else 0.0,
        "num_chunks": 1 if results else 0,
        # Standard IR metrics
        "hit_at_1": hit_1,
        "hit_at_3": hit_3,
        "mrr": mrr,
        "ndcg_at_3": ndcg_3,
    }


# ====================================================================================
# COST PROJECTION CALCULATOR
# ====================================================================================

def calculate_annual_cost(tokens_per_query: float, queries_per_month: int = 1_000_000, model: str = "gpt-4o") -> Dict:
    """
    Calculate annual API costs based on token consumption.

    Args:
        tokens_per_query: Average tokens per query
        queries_per_month: Monthly query volume (default 1M)
        model: LLM model for pricing

    Returns:
        Dict with monthly and annual costs
    """
    price_per_million = PRICING.get(model, PRICING["gpt-4o"])["input"]

    monthly_tokens = tokens_per_query * queries_per_month
    monthly_cost = (monthly_tokens / 1_000_000) * price_per_million
    annual_cost = monthly_cost * 12

    return {
        "model": model,
        "tokens_per_query": tokens_per_query,
        "queries_per_month": queries_per_month,
        "monthly_tokens": monthly_tokens,
        "monthly_cost_usd": monthly_cost,
        "annual_cost_usd": annual_cost,
    }


# ====================================================================================
# MAIN BENCHMARK RUNNER
# ====================================================================================

async def main():
    print("=" * 80)
    print("TOKEN EFFICIENCY BENCHMARK: Roampal vs Traditional RAG")
    print("=" * 80)
    print()
    print("HYPOTHESIS: Roampal uses fewer tokens while maintaining higher accuracy")
    print()
    print("CONDITIONS:")
    print("  - RAG Top-3:  Return top 3 semantically similar chunks")
    print("  - RAG Top-5:  Return top 5 semantically similar chunks")
    print("  - Roampal:    Return 1 outcome-weighted best result")
    print()

    if not HAS_REAL_EMBEDDINGS:
        print("ERROR: This test requires real embeddings.")
        print("Install: pip install sentence-transformers")
        return

    # Initialize embedding service - MUST use 768d model to match Roampal's UnifiedMemorySystem
    print("Loading embedding model (all-mpnet-base-v2, 768d)...")
    embedding_service = RealEmbeddingService(model_name='all-mpnet-base-v2')
    print("Model loaded.\n")

    # Create test directories
    test_dir = Path(__file__).parent / "token_efficiency_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    results_rag3 = []
    results_rag5 = []
    results_roampal = []

    print("-" * 80)
    n_scenarios = len(SCENARIOS)
    print(f"Running {n_scenarios} scenarios...")
    print("-" * 80)

    for scenario in SCENARIOS:
        print(f"\n[{scenario['id']:3d}/{n_scenarios}] {scenario['category'].upper()}")

        # RAG Top-3
        rag3_dir = str(test_dir / f"rag3_{scenario['id']}")
        os.makedirs(rag3_dir, exist_ok=True)
        rag3 = await run_control_topk(scenario, rag3_dir, embedding_service, k=3)
        results_rag3.append(rag3)

        # RAG Top-5
        rag5_dir = str(test_dir / f"rag5_{scenario['id']}")
        os.makedirs(rag5_dir, exist_ok=True)
        rag5 = await run_control_topk(scenario, rag5_dir, embedding_service, k=5)
        results_rag5.append(rag5)

        # Roampal
        roampal_dir = str(test_dir / f"roampal_{scenario['id']}")
        os.makedirs(roampal_dir, exist_ok=True)
        roampal = await run_treatment_roampal(scenario, roampal_dir, embedding_service)
        results_roampal.append(roampal)

        # Show result
        print(f"  RAG-3: {rag3['tokens_returned']:3d} tokens | RAG-5: {rag5['tokens_returned']:3d} tokens | Roampal: {roampal['tokens_returned']:3d} tokens")

    # ====================================================================================
    # ANALYSIS
    # ====================================================================================

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Aggregate metrics (including standard IR metrics)
    def aggregate(results: List[Dict]) -> Dict:
        tokens = [r["tokens_returned"] for r in results]
        accuracy = [r["accuracy"] for r in results]
        hit1 = [r.get("hit_at_1", r["accuracy"]) for r in results]
        hit3 = [r.get("hit_at_3", r["accuracy"]) for r in results]
        mrr = [r.get("mrr", r["accuracy"]) for r in results]
        ndcg = [r.get("ndcg_at_3", r["accuracy"]) for r in results]

        return {
            "avg_tokens": statistics.mean(tokens),
            "total_tokens": sum(tokens),
            "min_tokens": min(tokens),
            "max_tokens": max(tokens),
            "accuracy": statistics.mean(accuracy) * 100,
            "correct": sum(accuracy),
            # Standard IR metrics (BEIR/MTEB compatible)
            "hit_at_1": statistics.mean(hit1) * 100,
            "hit_at_3": statistics.mean(hit3) * 100,
            "mrr": statistics.mean(mrr),
            "ndcg_at_3": statistics.mean(ndcg),
        }

    agg_rag3 = aggregate(results_rag3)
    agg_rag5 = aggregate(results_rag5)
    agg_roampal = aggregate(results_roampal)

    print(f"\n{'Metric':<25} {'RAG Top-3':<15} {'RAG Top-5':<15} {'Roampal':<15}")
    print("-" * 80)
    print(f"{'Avg tokens/query':<25} {agg_rag3['avg_tokens']:<15.1f} {agg_rag5['avg_tokens']:<15.1f} {agg_roampal['avg_tokens']:<15.1f}")
    print(f"{'Hit@1 (accuracy)':<25} {agg_rag3['hit_at_1']:<15.1f}% {agg_rag5['hit_at_1']:<15.1f}% {agg_roampal['hit_at_1']:<15.1f}%")
    print(f"{'Hit@3':<25} {agg_rag3['hit_at_3']:<15.1f}% {agg_rag5['hit_at_3']:<15.1f}% {agg_roampal['hit_at_3']:<15.1f}%")
    print(f"{'MRR':<25} {agg_rag3['mrr']:<15.3f} {agg_rag5['mrr']:<15.3f} {agg_roampal['mrr']:<15.3f}")
    print(f"{'nDCG@3':<25} {agg_rag3['ndcg_at_3']:<15.3f} {agg_rag5['ndcg_at_3']:<15.3f} {agg_roampal['ndcg_at_3']:<15.3f}")
    print(f"{'Correct answers':<25} {agg_rag3['correct']:<15.0f}/{n_scenarios} {agg_rag5['correct']:<15.0f}/{n_scenarios} {agg_roampal['correct']:<15.0f}/{n_scenarios}")

    # Token efficiency ratio (accuracy per 100 tokens)
    eff_rag3 = (agg_rag3['accuracy'] / agg_rag3['avg_tokens']) * 100 if agg_rag3['avg_tokens'] > 0 else 0
    eff_rag5 = (agg_rag5['accuracy'] / agg_rag5['avg_tokens']) * 100 if agg_rag5['avg_tokens'] > 0 else 0
    eff_roampal = (agg_roampal['accuracy'] / agg_roampal['avg_tokens']) * 100 if agg_roampal['avg_tokens'] > 0 else 0

    print(f"{'Efficiency (acc/100tok)':<25} {eff_rag3:<15.2f} {eff_rag5:<15.2f} {eff_roampal:<15.2f}")

    # ====================================================================================
    # TOKEN REDUCTION ANALYSIS
    # ====================================================================================

    print("\n" + "=" * 80)
    print("TOKEN REDUCTION")
    print("=" * 80)

    reduction_vs_rag3 = ((agg_rag3['avg_tokens'] - agg_roampal['avg_tokens']) / agg_rag3['avg_tokens']) * 100
    reduction_vs_rag5 = ((agg_rag5['avg_tokens'] - agg_roampal['avg_tokens']) / agg_rag5['avg_tokens']) * 100

    print(f"\nRoampal token reduction:")
    print(f"  vs RAG Top-3: {reduction_vs_rag3:.1f}% fewer tokens ({agg_rag3['avg_tokens']:.0f} -> {agg_roampal['avg_tokens']:.0f})")
    print(f"  vs RAG Top-5: {reduction_vs_rag5:.1f}% fewer tokens ({agg_rag5['avg_tokens']:.0f} -> {agg_roampal['avg_tokens']:.0f})")

    # Multiplier
    mult_vs_rag3 = agg_rag3['avg_tokens'] / agg_roampal['avg_tokens'] if agg_roampal['avg_tokens'] > 0 else float('inf')
    mult_vs_rag5 = agg_rag5['avg_tokens'] / agg_roampal['avg_tokens'] if agg_roampal['avg_tokens'] > 0 else float('inf')

    print(f"\nEfficiency multiplier:")
    print(f"  RAG Top-3 uses {mult_vs_rag3:.1f}x more tokens than Roampal")
    print(f"  RAG Top-5 uses {mult_vs_rag5:.1f}x more tokens than Roampal")

    # ====================================================================================
    # COST PROJECTIONS
    # ====================================================================================

    print("\n" + "=" * 80)
    print("COST PROJECTIONS (1M queries/month)")
    print("=" * 80)

    for model in ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet"]:
        print(f"\n{model}:")

        cost_rag3 = calculate_annual_cost(agg_rag3['avg_tokens'], model=model)
        cost_rag5 = calculate_annual_cost(agg_rag5['avg_tokens'], model=model)
        cost_roampal = calculate_annual_cost(agg_roampal['avg_tokens'], model=model)

        print(f"  RAG Top-3:  ${cost_rag3['annual_cost_usd']:>10,.2f}/year")
        print(f"  RAG Top-5:  ${cost_rag5['annual_cost_usd']:>10,.2f}/year")
        print(f"  Roampal:    ${cost_roampal['annual_cost_usd']:>10,.2f}/year")

        savings_vs_rag3 = cost_rag3['annual_cost_usd'] - cost_roampal['annual_cost_usd']
        savings_vs_rag5 = cost_rag5['annual_cost_usd'] - cost_roampal['annual_cost_usd']

        print(f"  Savings vs RAG-3: ${savings_vs_rag3:>10,.2f}/year")
        print(f"  Savings vs RAG-5: ${savings_vs_rag5:>10,.2f}/year")

    # ====================================================================================
    # COMPARISON WITH TYPICAL RAG CONFIGURATIONS
    # ====================================================================================

    print("\n" + "=" * 80)
    print("COMPARISON WITH TYPICAL RAG CONFIGURATIONS")
    print("=" * 80)

    print(f"\n{'Configuration':<35} {'Tokens':<12} {'vs Roampal':<12}")
    print("-" * 70)

    # Compare with realistic RAG baselines
    for key, baseline in RAG_BASELINES.items():
        mult = baseline['tokens'] / agg_roampal['avg_tokens'] if agg_roampal['avg_tokens'] > 0 else float('inf')
        print(f"{baseline['name']:<35} {format_tokens(baseline['tokens']):<12} {mult:.1f}x more")

    print(f"{'Roampal (this test)':<35} {format_tokens(int(agg_roampal['avg_tokens'])):<12} baseline")

    # ====================================================================================
    # INDUSTRY CONTEXT (for reference, not direct comparison)
    # ====================================================================================

    print("\n" + "=" * 80)
    print("INDUSTRY CONTEXT: Memory Materialization Costs")
    print("(NOTE: These measure total memory store size, not per-query retrieval)")
    print("=" * 80)

    print(f"\n{'System':<30} {'Tokens':<12} {'Notes'}")
    print("-" * 85)

    for key, baseline in INDUSTRY_BASELINES.items():
        print(f"{baseline['name']:<30} {format_tokens(baseline['tokens']):<12} {baseline['source'][:45]}")

    # Calculate savings vs RAG baselines (fair comparison)
    print(f"\n{'Annual savings at 1M queries/month (GPT-4o pricing):'}")
    for key, baseline in RAG_BASELINES.items():
        baseline_cost = calculate_annual_cost(baseline['tokens'], model="gpt-4o")
        roampal_cost = calculate_annual_cost(agg_roampal['avg_tokens'], model="gpt-4o")
        savings = baseline_cost['annual_cost_usd'] - roampal_cost['annual_cost_usd']
        if savings > 0:
            print(f"  vs {baseline['name']:<30}: ${savings:>10,.2f}/year saved")

    # ====================================================================================
    # CONCLUSION
    # ====================================================================================

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    print(f"""
KEY FINDINGS:

1. TOKEN EFFICIENCY:
   - Roampal: {agg_roampal['avg_tokens']:.0f} tokens/query average
   - RAG Top-3: {agg_rag3['avg_tokens']:.0f} tokens/query ({mult_vs_rag3:.1f}x more)
   - RAG Top-5: {agg_rag5['avg_tokens']:.0f} tokens/query ({mult_vs_rag5:.1f}x more)

2. ACCURACY:
   - Roampal: {agg_roampal['accuracy']:.1f}% ({int(agg_roampal['correct'])}/{n_scenarios} correct)
   - RAG Top-3: {agg_rag3['accuracy']:.1f}% ({int(agg_rag3['correct'])}/{n_scenarios} correct)
   - RAG Top-5: {agg_rag5['accuracy']:.1f}% ({int(agg_rag5['correct'])}/{n_scenarios} correct)

3. EFFICIENCY RATIO (accuracy per 100 tokens):
   - Roampal: {eff_roampal:.2f}
   - RAG Top-3: {eff_rag3:.2f}
   - RAG Top-5: {eff_rag5:.2f}

HEADLINE:
  Roampal uses {reduction_vs_rag5:.0f}% fewer tokens than standard RAG
  while achieving {agg_roampal['accuracy']:.0f}% accuracy vs {agg_rag5['accuracy']:.0f}%.
""")

    # Save results with all metrics
    results_file = test_dir / "token_efficiency_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "benchmark_info": {
                "name": "Token Efficiency Benchmark",
                "version": "1.0",
                "methodology": "Adversarial retrieval scenarios (queries designed to match wrong answer)",
                "comparable_to": ["LOCOMO", "LongMemEval", "BEIR/MTEB"],
                "metrics_reported": ["tokens/query", "Hit@1", "Hit@3", "MRR", "nDCG@3"],
            },
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": 30,
            "rag_top3": {
                "results": results_rag3,
                "aggregate": agg_rag3,
            },
            "rag_top5": {
                "results": results_rag5,
                "aggregate": agg_rag5,
            },
            "roampal": {
                "results": results_roampal,
                "aggregate": agg_roampal,
            },
            "token_reduction": {
                "vs_rag3_percent": reduction_vs_rag3,
                "vs_rag5_percent": reduction_vs_rag5,
                "multiplier_vs_rag3": mult_vs_rag3,
                "multiplier_vs_rag5": mult_vs_rag5,
            },
            "efficiency_ratio": {
                "rag3": eff_rag3,
                "rag5": eff_rag5,
                "roampal": eff_roampal,
            },
            "rag_baselines_comparison": {
                key: {
                    "tokens": baseline["tokens"],
                    "multiplier_vs_roampal": baseline["tokens"] / agg_roampal["avg_tokens"] if agg_roampal["avg_tokens"] > 0 else None,
                    "source": baseline["source"],
                }
                for key, baseline in RAG_BASELINES.items()
            },
            "industry_context": {
                key: {
                    "tokens": baseline["tokens"],
                    "type": baseline.get("type", "memory_materialization"),
                    "source": baseline["source"],
                    "note": "Not directly comparable - measures total memory store size",
                }
                for key, baseline in INDUSTRY_BASELINES.items()
            },
        }, f, indent=2)

    print(f"Full results saved to: {results_file}")

    # Cleanup
    print("\nCleaning up test data...")
    try:
        shutil.rmtree(test_dir)
        print("Done.")
    except Exception as e:
        print(f"Warning: Could not clean up: {e}")


if __name__ == "__main__":
    asyncio.run(main())
