"""
VALIDATION - Strategy Backtesting and Validation

The Validation module tests strategies before deployment:

1. Backtesting - Simulate strategy on historical data
2. Metrics - Calculate comprehensive performance metrics
3. Statistical testing - Verify significance of results

Components:
- Backtester: Event-driven backtesting engine
- MetricsCalculator: Calculates Sharpe, drawdown, trade stats
- TradeRecord: Record of individual trades

Usage:
    from validation import get_backtester, BacktestConfig
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=10_000_000,
        commission_pct=0.00015,
        slippage_pct=0.001
    )
    
    backtester = get_backtester(config)
    
    # Run backtest
    result = backtester.run(hypothesis, data)
    
    # Check if passed
    if result.passed_criteria:
        # Promote to paper trading
        registry.promote(hypothesis.hypothesis_id)
"""

from validation.metrics import (
    MetricsCalculator,
    TradeRecord,
    get_metrics_calculator,
)

from validation.backtester import (
    Backtester,
    BacktestConfig,
    BacktestMode,
    BacktestResult,
    BacktestPosition,
    get_backtester,
)


__all__ = [
    # Metrics
    'MetricsCalculator',
    'TradeRecord',
    'get_metrics_calculator',
    
    # Backtester
    'Backtester',
    'BacktestConfig',
    'BacktestMode',
    'BacktestResult',
    'BacktestPosition',
    'get_backtester',
]
