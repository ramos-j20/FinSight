"""Unit tests for the cost calculator."""
import pytest
from decimal import Decimal
from backend.core.cost_calculator import calculate_cost, PRICING


def test_haiku_cost_calculation():
    """Test claude-haiku-4-5 cost matches expected value for known token counts."""
    # Haiku input: 1.00/M, output: 5.00/M
    # 1000 input tokens = $0.001
    # 500 output tokens = $0.0025
    # Total = $0.0035
    breakdown = calculate_cost("claude-haiku-4-5", input_tokens=1000, output_tokens=500)
    assert breakdown.total_cost == Decimal("0.0035")
    assert breakdown.total_cost_without_optimizations == Decimal("0.0035")


def test_sonnet_cost_calculation():
    """Test claude-sonnet-4-5 cost matches expected value for known token counts."""
    # Sonnet input: 3.00/M, output: 15.00/M
    # 1000 input tokens = $0.003
    # 500 output tokens = $0.0075
    # Total = $0.0105
    breakdown = calculate_cost("claude-sonnet-4-5", input_tokens=1000, output_tokens=500)
    assert breakdown.total_cost == Decimal("0.0105")


def test_batch_discount():
    """Test batch discount correctly halves input and output costs for both models."""
    # Haiku with batch discount (50%): 0.50/M input, 2.50/M output
    # 1000 input = $0.0005
    # 500 output = $0.00125
    # Total = $0.00175
    breakdown = calculate_cost("claude-haiku-4-5", input_tokens=1000, output_tokens=500, is_batch=True)
    assert breakdown.total_cost == Decimal("0.00175")
    assert breakdown.savings_pct == 50.0


def test_cache_read_pricing():
    """Test cache_read_tokens billed at 10% of standard input rate."""
    # Sonnet input: 3.00/M, cache_read: 0.30/M (which IS 10%)
    # 1000 tokens, all cache_read
    breakdown = calculate_cost("claude-sonnet-4-5", input_tokens=1000, output_tokens=0, cache_read_tokens=1000)
    assert breakdown.total_cost == Decimal("0.0003")
    assert breakdown.savings_from_cache == Decimal("0.0027") # 0.003 - 0.0003


def test_total_cost_without_optimizations():
    """Test total_cost_without_optimizations ignores all discounts."""
    # Sonnet: 1000 total input, 500 output. 
    # Even if 500 are cached and it's a batch job, raw cost ignores that.
    breakdown = calculate_cost(
        "claude-sonnet-4-5", 
        input_tokens=1000, 
        output_tokens=500, 
        cache_read_tokens=500, 
        is_batch=True
    )
    # Raw = 1000 * 3/M + 500 * 15/M = 0.003 + 0.0075 = 0.0105
    assert breakdown.total_cost_without_optimizations == Decimal("0.0105")
    # Batch cost = (500 input * 1.5/M) + (500 output * 7.5/M) + (500 cache_read * 0.3/M)
    # = 0.00075 + 0.00375 + 0.00015 = 0.00465
    assert breakdown.total_cost == Decimal("0.00465")


def test_savings_pct_calculation():
    """Test savings_pct is mathematically correct."""
    # Standard: $0.01, Actual: $0.002
    # Savings: $0.008 (80%)
    # To get $0.01 with Haiku: 10000 input tokens ($0.01), 0 output.
    # To get $0.002 with caching: 10000 input, 10000 cache_read ($0.001) + 1000 cache_write ($0.00125)? No.
    # Let's just check the math.
    breakdown = calculate_cost("claude-haiku-4-5", input_tokens=10000, output_tokens=0, cache_read_tokens=8000)
    # Raw = 10000 * 1/M = 0.01
    # Cost = (2000 * 1/M) + (8000 * 0.1/M) = 0.002 + 0.0008 = 0.0028
    # Savings = 0.0072 (72%)
    assert float(breakdown.savings_pct) == 72.0
