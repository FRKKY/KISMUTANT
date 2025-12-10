"""
HYPOTHESIS ENGINE - Trading Strategy Management

The Hypothesis Engine manages the lifecycle of trading strategies:

1. Generation - Create hypotheses from patterns or templates
2. Incubation - Backtest and validate hypotheses
3. Paper Trading - Validate with live (paper) trading
4. Live Trading - Deploy with real capital
5. Retirement - Remove underperforming strategies

Components:
- Hypothesis: Data structure for a trading strategy
- StrategyRegistry: Manages strategy lifecycle
- SignalGenerator: Produces trading signals from strategies
- HypothesisFactory: Creates strategies from patterns

Usage:
    from hypothesis import (
        get_registry,
        get_signal_generator,
        HypothesisFactory
    )
    
    # Create hypothesis from pattern
    hypothesis = HypothesisFactory.from_pattern(detected_pattern)
    
    # Register in system
    registry = get_registry()
    registry.register(hypothesis)
    
    # Generate signals
    generator = get_signal_generator()
    signals = generator.generate_signals(market_data)
"""

from hypothesis.models import (
    # Core types
    Hypothesis,
    TradingSignal,
    PerformanceMetrics,
    
    # Enums
    StrategyState,
    StrategyType,
    SignalType,
    
    # Criteria
    PromotionCriteria,
    DemotionCriteria,
    
    # Factory
    HypothesisFactory,
)

from hypothesis.registry import (
    StrategyRegistry,
    RegistryConfig,
    get_registry,
)

from hypothesis.signals import (
    SignalGenerator,
    SignalGeneratorConfig,
    RuleEvaluator,
    get_signal_generator,
)


__all__ = [
    # Core types
    'Hypothesis',
    'TradingSignal',
    'PerformanceMetrics',
    
    # Enums
    'StrategyState',
    'StrategyType', 
    'SignalType',
    
    # Criteria
    'PromotionCriteria',
    'DemotionCriteria',
    
    # Factory
    'HypothesisFactory',
    
    # Registry
    'StrategyRegistry',
    'RegistryConfig',
    'get_registry',
    
    # Signals
    'SignalGenerator',
    'SignalGeneratorConfig',
    'RuleEvaluator',
    'get_signal_generator',
]
