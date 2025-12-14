"""
LEARNING MODULE - Self-improvement system for the Living Trading System

This module enables the system to learn from its performance and evolve:
- Analyze what strategies work in different market conditions
- Optimize parameters based on historical performance
- Evolve strategies by modifying rules based on outcomes
- Detect market regime changes and adapt accordingly
"""

from learning.analyzer import PerformanceAnalyzer, get_analyzer
from learning.optimizer import ParameterOptimizer, get_optimizer
from learning.evolution import StrategyEvolver, get_evolver
from learning.regime import RegimeDetector, get_regime_detector

__all__ = [
    "PerformanceAnalyzer",
    "get_analyzer",
    "ParameterOptimizer",
    "get_optimizer",
    "StrategyEvolver",
    "get_evolver",
    "RegimeDetector",
    "get_regime_detector",
]
