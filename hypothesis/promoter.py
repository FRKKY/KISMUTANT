"""
STRATEGY PROMOTER - Automated strategy lifecycle management

Provides intelligent automation for:
- Scheduling backtests for new hypotheses
- Monitoring paper trading for time-based promotion
- Priority queueing for promotions (best performers first)
- Capacity-aware promotion (promote when slots open)
- Notification dispatch for lifecycle events
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple
from collections import deque
from enum import Enum

from loguru import logger

from hypothesis.models import (
    Hypothesis,
    StrategyState,
    PerformanceMetrics
)
from hypothesis.registry import get_registry, StrategyRegistry
from core.events import get_event_bus, Event, EventType


class PromotionDecision(str, Enum):
    """Result of a promotion evaluation."""
    PROMOTE = "promote"
    WAIT = "wait"
    DEMOTE = "demote"
    RETIRE = "retire"
    NO_ACTION = "no_action"


@dataclass
class PromotionCandidate:
    """A hypothesis being evaluated for promotion."""
    hypothesis_id: str
    current_state: StrategyState
    priority_score: float
    days_in_state: int
    metrics: Optional[PerformanceMetrics]
    evaluation_reason: str
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    def __lt__(self, other: 'PromotionCandidate') -> bool:
        """Sort by priority score (higher is better)."""
        return self.priority_score > other.priority_score


@dataclass
class PromoterConfig:
    """Configuration for the strategy promoter."""

    # Evaluation frequency
    evaluation_interval_hours: int = 4

    # Backtest queue
    max_backtest_queue_size: int = 20
    backtest_timeout_minutes: int = 30

    # Paper trading requirements
    min_paper_days: int = 30
    min_paper_trades: int = 20

    # Priority weights for scoring
    sharpe_weight: float = 0.4
    profit_factor_weight: float = 0.2
    win_rate_weight: float = 0.2
    drawdown_weight: float = 0.2  # Negative weight (lower is better)

    # Capacity management
    promote_best_when_full: bool = True
    min_improvement_to_replace: float = 0.2  # 20% better to replace

    # Notifications
    notify_on_promotion: bool = True
    notify_on_demotion: bool = True
    notify_on_retirement: bool = True


class StrategyPromoter:
    """
    Manages automated strategy promotion lifecycle.

    Responsibilities:
    1. Evaluate hypotheses for promotion readiness
    2. Queue and prioritize promotions
    3. Handle capacity constraints
    4. Dispatch lifecycle notifications
    """

    def __init__(
        self,
        registry: Optional[StrategyRegistry] = None,
        config: Optional[PromoterConfig] = None
    ):
        self.registry = registry or get_registry()
        self.config = config or PromoterConfig()

        # Backtest queue
        self._backtest_queue: deque = deque(maxlen=self.config.max_backtest_queue_size)
        self._pending_backtests: Dict[str, datetime] = {}

        # Promotion candidates (priority queue)
        self._promotion_candidates: List[PromotionCandidate] = []

        # Statistics
        self._stats = {
            "evaluations": 0,
            "promotions": 0,
            "demotions": 0,
            "retirements": 0,
            "queued_backtests": 0
        }

        # Callbacks
        self._on_decision: List[Callable] = []

        logger.info("StrategyPromoter initialized")

    def queue_for_backtest(self, hypothesis_id: str) -> bool:
        """
        Queue a hypothesis for backtesting.

        Args:
            hypothesis_id: ID of hypothesis to backtest

        Returns:
            True if queued successfully
        """
        if hypothesis_id in self._pending_backtests:
            logger.debug(f"Hypothesis {hypothesis_id} already in backtest queue")
            return False

        if len(self._backtest_queue) >= self.config.max_backtest_queue_size:
            logger.warning("Backtest queue full, cannot queue")
            return False

        self._backtest_queue.append(hypothesis_id)
        self._pending_backtests[hypothesis_id] = datetime.utcnow()
        self._stats["queued_backtests"] += 1

        logger.debug(f"Queued {hypothesis_id} for backtest")
        return True

    def get_next_backtest(self) -> Optional[str]:
        """Get next hypothesis to backtest."""
        if not self._backtest_queue:
            return None

        hypothesis_id = self._backtest_queue.popleft()
        return hypothesis_id

    def complete_backtest(
        self,
        hypothesis_id: str,
        metrics: PerformanceMetrics
    ) -> None:
        """
        Mark a backtest as complete and update hypothesis.

        Args:
            hypothesis_id: ID of hypothesis
            metrics: Backtest results
        """
        if hypothesis_id in self._pending_backtests:
            del self._pending_backtests[hypothesis_id]

        hypothesis = self.registry.get(hypothesis_id)
        if hypothesis:
            hypothesis.backtest_metrics = metrics
            self.registry.update(hypothesis)

            # Re-evaluate for promotion
            self._evaluate_hypothesis(hypothesis)

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate all active hypotheses for promotion/demotion.

        Returns:
            Summary of evaluations and decisions
        """
        self._stats["evaluations"] += 1
        self._promotion_candidates.clear()

        results = {
            "evaluated": 0,
            "decisions": [],
            "promotions": [],
            "demotions": [],
            "retirements": []
        }

        # Evaluate incubating strategies
        for hypothesis in self.registry.get_all(StrategyState.INCUBATING):
            decision = self._evaluate_hypothesis(hypothesis)
            results["evaluated"] += 1
            results["decisions"].append({
                "hypothesis_id": hypothesis.hypothesis_id,
                "name": hypothesis.name,
                "decision": decision.value
            })

            if decision == PromotionDecision.PROMOTE:
                results["promotions"].append(hypothesis.hypothesis_id)

        # Evaluate paper trading strategies
        for hypothesis in self.registry.get_all(StrategyState.PAPER_TRADING):
            decision = self._evaluate_hypothesis(hypothesis)
            results["evaluated"] += 1
            results["decisions"].append({
                "hypothesis_id": hypothesis.hypothesis_id,
                "name": hypothesis.name,
                "decision": decision.value
            })

            if decision == PromotionDecision.PROMOTE:
                results["promotions"].append(hypothesis.hypothesis_id)
            elif decision == PromotionDecision.DEMOTE:
                results["demotions"].append(hypothesis.hypothesis_id)
            elif decision == PromotionDecision.RETIRE:
                results["retirements"].append(hypothesis.hypothesis_id)

        # Evaluate live strategies for demotion
        for hypothesis in self.registry.get_all(StrategyState.LIVE):
            decision = self._evaluate_hypothesis(hypothesis)
            results["evaluated"] += 1
            results["decisions"].append({
                "hypothesis_id": hypothesis.hypothesis_id,
                "name": hypothesis.name,
                "decision": decision.value
            })

            if decision == PromotionDecision.DEMOTE:
                results["demotions"].append(hypothesis.hypothesis_id)

        # Process promotions with capacity awareness
        self._process_promotion_queue()

        logger.info(
            f"Evaluation complete: {results['evaluated']} evaluated, "
            f"{len(results['promotions'])} promote, "
            f"{len(results['demotions'])} demote, "
            f"{len(results['retirements'])} retire"
        )

        return results

    def _evaluate_hypothesis(self, hypothesis: Hypothesis) -> PromotionDecision:
        """
        Evaluate a single hypothesis for lifecycle decision.

        Args:
            hypothesis: Hypothesis to evaluate

        Returns:
            PromotionDecision indicating recommended action
        """
        state = hypothesis.state

        if state == StrategyState.INCUBATING:
            return self._evaluate_incubating(hypothesis)
        elif state == StrategyState.PAPER_TRADING:
            return self._evaluate_paper_trading(hypothesis)
        elif state == StrategyState.LIVE:
            return self._evaluate_live(hypothesis)

        return PromotionDecision.NO_ACTION

    def _evaluate_incubating(self, hypothesis: Hypothesis) -> PromotionDecision:
        """Evaluate an incubating strategy for promotion to paper."""
        metrics = hypothesis.backtest_metrics

        if not metrics:
            # Queue for backtest if not already queued
            if hypothesis.hypothesis_id not in self._pending_backtests:
                self.queue_for_backtest(hypothesis.hypothesis_id)
            return PromotionDecision.WAIT

        # Check promotion criteria
        can_promote, failures = hypothesis.can_promote()

        if can_promote:
            # Calculate priority score
            score = self._calculate_priority_score(metrics)

            candidate = PromotionCandidate(
                hypothesis_id=hypothesis.hypothesis_id,
                current_state=StrategyState.INCUBATING,
                priority_score=score,
                days_in_state=self._days_in_state(hypothesis),
                metrics=metrics,
                evaluation_reason="Backtest passed all criteria"
            )
            self._promotion_candidates.append(candidate)

            return PromotionDecision.PROMOTE

        # Check if should retire (too many failures or very poor metrics)
        if metrics.sharpe_ratio < 0 or metrics.max_drawdown > 0.30:
            return PromotionDecision.RETIRE

        return PromotionDecision.WAIT

    def _evaluate_paper_trading(self, hypothesis: Hypothesis) -> PromotionDecision:
        """Evaluate a paper trading strategy for promotion/demotion."""
        metrics = hypothesis.paper_metrics
        days_in_paper = self._days_in_state(hypothesis)

        # Must meet minimum time requirement
        if days_in_paper < self.config.min_paper_days:
            return PromotionDecision.WAIT

        if not metrics:
            return PromotionDecision.WAIT

        # Check for demotion first
        should_demote, demote_reasons = hypothesis.should_demote()
        if should_demote:
            self._execute_demotion(hypothesis, "; ".join(demote_reasons))
            return PromotionDecision.DEMOTE

        # Check if trades requirement met
        if metrics.total_trades < self.config.min_paper_trades:
            return PromotionDecision.WAIT

        # Check promotion criteria
        can_promote, failures = hypothesis.can_promote()

        if can_promote:
            score = self._calculate_priority_score(metrics)

            candidate = PromotionCandidate(
                hypothesis_id=hypothesis.hypothesis_id,
                current_state=StrategyState.PAPER_TRADING,
                priority_score=score,
                days_in_state=days_in_paper,
                metrics=metrics,
                evaluation_reason=f"Paper trading passed ({days_in_paper} days, {metrics.total_trades} trades)"
            )
            self._promotion_candidates.append(candidate)

            return PromotionDecision.PROMOTE

        return PromotionDecision.WAIT

    def _evaluate_live(self, hypothesis: Hypothesis) -> PromotionDecision:
        """Evaluate a live strategy for demotion."""
        metrics = hypothesis.live_metrics

        if not metrics:
            return PromotionDecision.NO_ACTION

        should_demote, reasons = hypothesis.should_demote()

        if should_demote:
            self._execute_demotion(hypothesis, "; ".join(reasons))
            return PromotionDecision.DEMOTE

        return PromotionDecision.NO_ACTION

    def _calculate_priority_score(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate a priority score for promotion ordering.

        Higher score = higher priority for promotion.
        """
        if not metrics:
            return 0.0

        config = self.config

        # Normalize metrics (0-1 scale approximately)
        sharpe_score = max(0, min(metrics.sharpe_ratio / 3.0, 1.0))  # Sharpe 0-3 -> 0-1
        pf_score = max(0, min((metrics.profit_factor - 1) / 2.0, 1.0))  # PF 1-3 -> 0-1
        wr_score = max(0, min(metrics.win_rate, 1.0))  # Already 0-1
        dd_score = max(0, min(1 - metrics.max_drawdown / 0.20, 1.0))  # DD 0-20% -> 1-0

        # Weighted sum
        score = (
            config.sharpe_weight * sharpe_score +
            config.profit_factor_weight * pf_score +
            config.win_rate_weight * wr_score +
            config.drawdown_weight * dd_score
        )

        return score

    def _days_in_state(self, hypothesis: Hypothesis) -> int:
        """Calculate days hypothesis has been in current state."""
        state = hypothesis.state

        if state == StrategyState.INCUBATING and hypothesis.incubation_start:
            return (datetime.now() - hypothesis.incubation_start).days
        elif state == StrategyState.PAPER_TRADING and hypothesis.paper_start:
            return (datetime.now() - hypothesis.paper_start).days
        elif state == StrategyState.LIVE and hypothesis.live_start:
            return (datetime.now() - hypothesis.live_start).days

        return 0

    def _process_promotion_queue(self) -> List[str]:
        """
        Process promotion candidates with capacity awareness.

        Promotes best candidates up to capacity limits.

        Returns:
            List of promoted hypothesis IDs
        """
        if not self._promotion_candidates:
            return []

        # Sort by priority score (highest first)
        self._promotion_candidates.sort()

        promoted = []

        # Process paper -> live promotions first (more important)
        live_capacity = self.registry.config.max_live
        current_live = len(self.registry.get_all(StrategyState.LIVE))
        available_live_slots = live_capacity - current_live

        # Candidates for live promotion
        live_candidates = [
            c for c in self._promotion_candidates
            if c.current_state == StrategyState.PAPER_TRADING
        ]

        for candidate in live_candidates[:available_live_slots]:
            success, msg = self.registry.promote(candidate.hypothesis_id)
            if success:
                promoted.append(candidate.hypothesis_id)
                self._stats["promotions"] += 1
                self._notify_promotion(candidate, StrategyState.LIVE)

        # If full but promote_best_when_full is enabled, consider replacing
        if (available_live_slots == 0 and
            self.config.promote_best_when_full and
            live_candidates):

            # Find worst performing live strategy
            live_strategies = self.registry.get_all(StrategyState.LIVE)
            worst_live = self._find_worst_performer(live_strategies)
            best_candidate = live_candidates[0] if live_candidates else None

            if worst_live and best_candidate:
                worst_score = self._calculate_priority_score(worst_live.live_metrics)

                # Replace if candidate is significantly better
                if (best_candidate.priority_score - worst_score) > self.config.min_improvement_to_replace:
                    # Demote worst
                    self.registry.demote(worst_live.hypothesis_id, "Replaced by better performer")
                    self._stats["demotions"] += 1

                    # Promote best candidate
                    success, msg = self.registry.promote(best_candidate.hypothesis_id)
                    if success:
                        promoted.append(best_candidate.hypothesis_id)
                        self._stats["promotions"] += 1
                        self._notify_promotion(best_candidate, StrategyState.LIVE)

        # Process incubating -> paper promotions
        paper_capacity = self.registry.config.max_paper_trading
        current_paper = len(self.registry.get_all(StrategyState.PAPER_TRADING))
        available_paper_slots = paper_capacity - current_paper

        paper_candidates = [
            c for c in self._promotion_candidates
            if c.current_state == StrategyState.INCUBATING
        ]

        for candidate in paper_candidates[:available_paper_slots]:
            success, msg = self.registry.promote(candidate.hypothesis_id)
            if success:
                promoted.append(candidate.hypothesis_id)
                self._stats["promotions"] += 1
                self._notify_promotion(candidate, StrategyState.PAPER_TRADING)

        return promoted

    def _find_worst_performer(self, strategies: List[Hypothesis]) -> Optional[Hypothesis]:
        """Find the worst performing strategy from a list."""
        if not strategies:
            return None

        worst = None
        worst_score = float('inf')

        for strategy in strategies:
            metrics = strategy.get_current_metrics()
            if metrics:
                score = self._calculate_priority_score(metrics)
                if score < worst_score:
                    worst_score = score
                    worst = strategy

        return worst

    def _execute_demotion(self, hypothesis: Hypothesis, reason: str) -> bool:
        """Execute a demotion decision."""
        success, msg = self.registry.demote(hypothesis.hypothesis_id, reason)

        if success:
            self._stats["demotions"] += 1
            self._notify_demotion(hypothesis, reason)

        return success

    def _notify_promotion(
        self,
        candidate: PromotionCandidate,
        new_state: StrategyState
    ) -> None:
        """Send promotion notification."""
        if not self.config.notify_on_promotion:
            return

        hypothesis = self.registry.get(candidate.hypothesis_id)

        get_event_bus().publish(Event(
            event_type=EventType.SYSTEM_ALERT,
            source="strategy_promoter",
            payload={
                "type": "promotion",
                "hypothesis_id": candidate.hypothesis_id,
                "name": hypothesis.name if hypothesis else "Unknown",
                "new_state": new_state.value,
                "priority_score": candidate.priority_score,
                "reason": candidate.evaluation_reason
            }
        ))

    def _notify_demotion(self, hypothesis: Hypothesis, reason: str) -> None:
        """Send demotion notification."""
        if not self.config.notify_on_demotion:
            return

        get_event_bus().publish(Event(
            event_type=EventType.SYSTEM_ALERT,
            source="strategy_promoter",
            payload={
                "type": "demotion",
                "hypothesis_id": hypothesis.hypothesis_id,
                "name": hypothesis.name,
                "new_state": hypothesis.state.value,
                "reason": reason
            }
        ))

    def get_promotion_candidates(self) -> List[Dict[str, Any]]:
        """Get current promotion candidates with their scores."""
        return [
            {
                "hypothesis_id": c.hypothesis_id,
                "current_state": c.current_state.value,
                "priority_score": c.priority_score,
                "days_in_state": c.days_in_state,
                "reason": c.evaluation_reason
            }
            for c in sorted(self._promotion_candidates)
        ]

    def get_backtest_queue(self) -> List[str]:
        """Get current backtest queue."""
        return list(self._backtest_queue)

    def get_stats(self) -> Dict[str, Any]:
        """Get promoter statistics."""
        return {
            **self._stats,
            "pending_backtests": len(self._pending_backtests),
            "backtest_queue_size": len(self._backtest_queue),
            "promotion_candidates": len(self._promotion_candidates)
        }


# Singleton accessor
_promoter_instance: Optional[StrategyPromoter] = None


def get_promoter(config: Optional[PromoterConfig] = None) -> StrategyPromoter:
    """Get the singleton StrategyPromoter instance."""
    global _promoter_instance
    if _promoter_instance is None:
        _promoter_instance = StrategyPromoter(config=config)
    return _promoter_instance
