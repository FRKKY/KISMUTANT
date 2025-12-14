"""
PERFORMANCE ANALYZER - Learn what works and what doesn't

Analyzes strategy performance across different:
- Market conditions (trending, ranging, volatile)
- Time periods (day of week, time of day, month)
- Asset characteristics (sector, volatility, liquidity)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class MarketCondition(str, Enum):
    """Market condition classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class PerformanceInsight:
    """A learned insight about performance."""
    insight_id: str
    insight_type: str  # "strength", "weakness", "pattern", "correlation"
    description: str
    confidence: float  # 0-1
    sample_size: int
    conditions: Dict[str, Any]  # When this insight applies
    recommendation: str
    discovered_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type,
            "description": self.description,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "conditions": self.conditions,
            "recommendation": self.recommendation,
            "discovered_at": self.discovered_at.isoformat(),
        }


@dataclass
class StrategyPerformanceProfile:
    """Comprehensive performance profile for a strategy."""
    strategy_id: str
    strategy_name: str

    # Overall metrics
    total_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0

    # Performance by condition
    performance_by_condition: Dict[MarketCondition, Dict[str, float]] = field(default_factory=dict)

    # Performance by time
    performance_by_day: Dict[int, Dict[str, float]] = field(default_factory=dict)  # 0=Mon
    performance_by_hour: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Insights
    strengths: List[PerformanceInsight] = field(default_factory=list)
    weaknesses: List[PerformanceInsight] = field(default_factory=list)
    patterns: List[PerformanceInsight] = field(default_factory=list)

    # Recommendations
    optimal_conditions: List[MarketCondition] = field(default_factory=list)
    avoid_conditions: List[MarketCondition] = field(default_factory=list)


class PerformanceAnalyzer:
    """
    Analyzes trading performance to learn what works.

    Key capabilities:
    1. Segment performance by market conditions
    2. Identify strengths and weaknesses
    3. Find patterns in winning/losing trades
    4. Generate actionable insights
    """

    _instance: Optional['PerformanceAnalyzer'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._trade_history: List[Dict[str, Any]] = []
        self._strategy_profiles: Dict[str, StrategyPerformanceProfile] = {}
        self._insights: List[PerformanceInsight] = []
        self._market_conditions: Dict[str, MarketCondition] = {}

        self._initialized = True
        logger.info("PerformanceAnalyzer initialized")

    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Record a completed trade for analysis."""
        self._trade_history.append({
            **trade,
            "recorded_at": datetime.utcnow(),
        })

        # Update running analysis
        strategy_id = trade.get("strategy_id")
        if strategy_id:
            self._update_strategy_profile(strategy_id)

    def _update_strategy_profile(self, strategy_id: str) -> None:
        """Update performance profile for a strategy."""
        trades = [t for t in self._trade_history if t.get("strategy_id") == strategy_id]

        if len(trades) < 5:
            return  # Need minimum trades

        if strategy_id not in self._strategy_profiles:
            self._strategy_profiles[strategy_id] = StrategyPerformanceProfile(
                strategy_id=strategy_id,
                strategy_name=trades[0].get("strategy_name", "Unknown")
            )

        profile = self._strategy_profiles[strategy_id]

        # Calculate overall metrics
        returns = [t.get("return_pct", 0) for t in trades]
        wins = [r for r in returns if r > 0]

        profile.total_trades = len(trades)
        profile.win_rate = len(wins) / len(trades) if trades else 0
        profile.avg_return = np.mean(returns) if returns else 0

        # Sharpe ratio (annualized, assuming daily)
        if len(returns) > 1 and np.std(returns) > 0:
            profile.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        # Max drawdown
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        profile.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profile.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Analyze by conditions
        self._analyze_by_conditions(profile, trades)

        # Generate insights
        self._generate_insights(profile, trades)

    def _analyze_by_conditions(
        self,
        profile: StrategyPerformanceProfile,
        trades: List[Dict[str, Any]]
    ) -> None:
        """Analyze performance segmented by market conditions."""
        # Group by day of week
        for day in range(7):
            day_trades = [t for t in trades
                         if t.get("entry_time", datetime.now()).weekday() == day]
            if day_trades:
                returns = [t.get("return_pct", 0) for t in day_trades]
                profile.performance_by_day[day] = {
                    "count": len(day_trades),
                    "win_rate": len([r for r in returns if r > 0]) / len(returns),
                    "avg_return": np.mean(returns),
                }

        # Group by market condition (if available)
        for condition in MarketCondition:
            condition_trades = [t for t in trades
                              if t.get("market_condition") == condition.value]
            if condition_trades:
                returns = [t.get("return_pct", 0) for t in condition_trades]
                profile.performance_by_condition[condition] = {
                    "count": len(condition_trades),
                    "win_rate": len([r for r in returns if r > 0]) / len(returns),
                    "avg_return": np.mean(returns),
                    "sharpe": (np.mean(returns) / np.std(returns) * np.sqrt(252))
                             if np.std(returns) > 0 else 0,
                }

    def _generate_insights(
        self,
        profile: StrategyPerformanceProfile,
        trades: List[Dict[str, Any]]
    ) -> None:
        """Generate actionable insights from performance data."""
        import uuid

        profile.strengths.clear()
        profile.weaknesses.clear()
        profile.patterns.clear()
        profile.optimal_conditions.clear()
        profile.avoid_conditions.clear()

        # Identify strengths (conditions where strategy excels)
        for condition, perf in profile.performance_by_condition.items():
            if perf["count"] >= 10 and perf["win_rate"] > 0.6:
                insight = PerformanceInsight(
                    insight_id=str(uuid.uuid4()),
                    insight_type="strength",
                    description=f"Strategy performs well in {condition.value} conditions",
                    confidence=min(perf["count"] / 30, 1.0),
                    sample_size=perf["count"],
                    conditions={"market_condition": condition.value},
                    recommendation=f"Increase position size during {condition.value}"
                )
                profile.strengths.append(insight)
                profile.optimal_conditions.append(condition)

        # Identify weaknesses
        for condition, perf in profile.performance_by_condition.items():
            if perf["count"] >= 10 and perf["win_rate"] < 0.4:
                insight = PerformanceInsight(
                    insight_id=str(uuid.uuid4()),
                    insight_type="weakness",
                    description=f"Strategy struggles in {condition.value} conditions",
                    confidence=min(perf["count"] / 30, 1.0),
                    sample_size=perf["count"],
                    conditions={"market_condition": condition.value},
                    recommendation=f"Reduce or avoid trading during {condition.value}"
                )
                profile.weaknesses.append(insight)
                profile.avoid_conditions.append(condition)

        # Identify day-of-week patterns
        best_day = max(profile.performance_by_day.items(),
                      key=lambda x: x[1].get("avg_return", 0),
                      default=(None, {}))
        worst_day = min(profile.performance_by_day.items(),
                       key=lambda x: x[1].get("avg_return", 0),
                       default=(None, {}))

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        if best_day[0] is not None and best_day[1].get("count", 0) >= 5:
            if best_day[1].get("avg_return", 0) > profile.avg_return * 1.5:
                insight = PerformanceInsight(
                    insight_id=str(uuid.uuid4()),
                    insight_type="pattern",
                    description=f"Best performance on {day_names[best_day[0]]}s",
                    confidence=min(best_day[1]["count"] / 20, 1.0),
                    sample_size=best_day[1]["count"],
                    conditions={"day_of_week": best_day[0]},
                    recommendation=f"Consider increasing activity on {day_names[best_day[0]]}s"
                )
                profile.patterns.append(insight)

        # Store all insights globally
        all_insights = profile.strengths + profile.weaknesses + profile.patterns
        for insight in all_insights:
            if insight not in self._insights:
                self._insights.append(insight)

    def get_strategy_profile(self, strategy_id: str) -> Optional[StrategyPerformanceProfile]:
        """Get the performance profile for a strategy."""
        return self._strategy_profiles.get(strategy_id)

    def get_all_insights(self) -> List[PerformanceInsight]:
        """Get all discovered insights."""
        return self._insights.copy()

    def get_recommendations(self, strategy_id: str) -> List[str]:
        """Get actionable recommendations for a strategy."""
        profile = self._strategy_profiles.get(strategy_id)
        if not profile:
            return []

        recommendations = []

        # From strengths
        for insight in profile.strengths:
            recommendations.append(f"[STRENGTH] {insight.recommendation}")

        # From weaknesses
        for insight in profile.weaknesses:
            recommendations.append(f"[WEAKNESS] {insight.recommendation}")

        # From patterns
        for insight in profile.patterns:
            recommendations.append(f"[PATTERN] {insight.recommendation}")

        return recommendations

    def should_trade(
        self,
        strategy_id: str,
        current_condition: Optional[MarketCondition] = None
    ) -> Tuple[bool, str]:
        """
        Determine if a strategy should trade given current conditions.

        Returns:
            (should_trade, reason)
        """
        profile = self._strategy_profiles.get(strategy_id)

        if not profile or profile.total_trades < 20:
            return True, "Insufficient data for recommendation"

        if current_condition:
            if current_condition in profile.avoid_conditions:
                return False, f"Strategy historically underperforms in {current_condition.value}"

            if current_condition in profile.optimal_conditions:
                return True, f"Optimal conditions: {current_condition.value}"

        # Check overall performance
        if profile.sharpe_ratio < 0.5:
            return False, f"Low Sharpe ratio: {profile.sharpe_ratio:.2f}"

        if profile.max_drawdown > 0.15:
            return False, f"High drawdown risk: {profile.max_drawdown:.1%}"

        return True, "Acceptable performance profile"

    def get_position_size_multiplier(
        self,
        strategy_id: str,
        current_condition: Optional[MarketCondition] = None
    ) -> float:
        """
        Get position size multiplier based on learned performance.

        Returns:
            Multiplier (0.5 = half size, 1.0 = normal, 1.5 = increased)
        """
        profile = self._strategy_profiles.get(strategy_id)

        if not profile or profile.total_trades < 20:
            return 1.0  # Default size

        multiplier = 1.0

        # Adjust based on current condition
        if current_condition:
            if current_condition in profile.optimal_conditions:
                multiplier *= 1.25  # Increase in optimal conditions
            elif current_condition in profile.avoid_conditions:
                multiplier *= 0.5  # Decrease in poor conditions

        # Adjust based on overall performance
        if profile.sharpe_ratio > 1.5:
            multiplier *= 1.2
        elif profile.sharpe_ratio < 0.75:
            multiplier *= 0.8

        # Clamp to reasonable range
        return max(0.25, min(2.0, multiplier))

    def export_learnings(self) -> Dict[str, Any]:
        """Export all learned knowledge for persistence."""
        return {
            "trade_count": len(self._trade_history),
            "strategies_analyzed": len(self._strategy_profiles),
            "insights_discovered": len(self._insights),
            "profiles": {
                sid: {
                    "total_trades": p.total_trades,
                    "win_rate": p.win_rate,
                    "sharpe_ratio": p.sharpe_ratio,
                    "optimal_conditions": [c.value for c in p.optimal_conditions],
                    "avoid_conditions": [c.value for c in p.avoid_conditions],
                    "strengths": [i.to_dict() for i in p.strengths],
                    "weaknesses": [i.to_dict() for i in p.weaknesses],
                }
                for sid, p in self._strategy_profiles.items()
            },
            "global_insights": [i.to_dict() for i in self._insights],
            "exported_at": datetime.utcnow().isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics summary."""
        return {
            "total_trades_analyzed": len(self._trade_history),
            "strategy_profiles": len(self._strategy_profiles),
            "global_insights": len(self._insights),
        }


# Singleton accessor
_analyzer_instance: Optional[PerformanceAnalyzer] = None

def get_analyzer() -> PerformanceAnalyzer:
    """Get the singleton PerformanceAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = PerformanceAnalyzer()
    return _analyzer_instance
