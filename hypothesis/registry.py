"""
STRATEGY REGISTRY - Manages Hypothesis Lifecycle

The Registry is responsible for:
1. Storing and retrieving hypotheses
2. Managing state transitions (promotion/demotion)
3. Enforcing lifecycle rules
4. Tracking strategy performance over time
5. Coordinating with capital allocator

State Machine:
    INCUBATING ──[backtest passes]──► PAPER_TRADING
         │                                  │
         │                      [30 days + metrics pass]
         │                                  │
         └────────────────────────────────► LIVE
                                            │
                              [underperforms]│
                                            ▼
                                        RETIRED
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from collections import defaultdict
import json
from pathlib import Path

from loguru import logger

from hypothesis.models import (
    Hypothesis,
    StrategyState,
    StrategyType,
    PerformanceMetrics,
    PromotionCriteria,
    DemotionCriteria,
    TradingSignal
)
from core.events import get_event_bus, Event, EventType


@dataclass
class RegistryConfig:
    """Configuration for strategy registry."""
    
    # Capacity limits
    max_incubating: int = 50         # Max strategies in incubation
    max_paper_trading: int = 10      # Max strategies in paper trading
    max_live: int = 3                # Max live strategies (matches capital allocation)
    
    # Lifecycle
    promotion_criteria: PromotionCriteria = field(default_factory=PromotionCriteria)
    demotion_criteria: DemotionCriteria = field(default_factory=DemotionCriteria)
    
    # Auto-management
    auto_promote: bool = True        # Automatically promote when criteria met
    auto_demote: bool = True         # Automatically demote underperformers
    
    # Persistence
    persist_path: Optional[Path] = None


class StrategyRegistry:
    """
    Central registry for all trading hypotheses.
    
    Manages the full lifecycle of strategies from incubation
    through live trading to retirement.
    """
    
    def __init__(self, config: Optional[RegistryConfig] = None):
        self.config = config or RegistryConfig()
        
        # Storage by state
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._by_state: Dict[StrategyState, List[str]] = {
            state: [] for state in StrategyState
        }
        
        # Indexes for fast lookup
        self._by_symbol: Dict[str, List[str]] = defaultdict(list)
        self._by_type: Dict[StrategyType, List[str]] = defaultdict(list)
        
        # Event callbacks
        self._on_promotion: List[Callable] = []
        self._on_demotion: List[Callable] = []
        self._on_retirement: List[Callable] = []
        
        # Statistics
        self._stats = {
            "total_created": 0,
            "total_promoted": 0,
            "total_demoted": 0,
            "total_retired": 0
        }
        
        logger.info("StrategyRegistry initialized")
    
    # ===== CRUD Operations =====
    
    def register(self, hypothesis: Hypothesis) -> str:
        """
        Register a new hypothesis in the registry.
        
        New hypotheses start in INCUBATING state.
        """
        # Check capacity
        if len(self._by_state[StrategyState.INCUBATING]) >= self.config.max_incubating:
            logger.warning("Incubation capacity reached, cannot register new hypothesis")
            raise ValueError("Incubation capacity reached")
        
        # Ensure INCUBATING state
        hypothesis.state = StrategyState.INCUBATING
        hypothesis.incubation_start = datetime.now()
        
        # Store
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis
        self._by_state[StrategyState.INCUBATING].append(hypothesis.hypothesis_id)
        
        # Update indexes
        for symbol in hypothesis.symbols:
            self._by_symbol[symbol].append(hypothesis.hypothesis_id)
        self._by_type[hypothesis.strategy_type].append(hypothesis.hypothesis_id)
        
        self._stats["total_created"] += 1
        
        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.STRATEGY_REGISTERED,
            source="strategy_registry",
            payload={
                "hypothesis_id": hypothesis.hypothesis_id,
                "name": hypothesis.name,
                "type": hypothesis.strategy_type.value
            }
        ))
        
        logger.info(f"Registered hypothesis: {hypothesis.name} ({hypothesis.hypothesis_id})")
        
        return hypothesis.hypothesis_id
    
    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)
    
    def get_all(self, state: Optional[StrategyState] = None) -> List[Hypothesis]:
        """Get all hypotheses, optionally filtered by state."""
        if state:
            ids = self._by_state.get(state, [])
            return [self._hypotheses[hid] for hid in ids if hid in self._hypotheses]
        return list(self._hypotheses.values())
    
    def get_by_symbol(self, symbol: str) -> List[Hypothesis]:
        """Get all hypotheses that trade a symbol."""
        ids = self._by_symbol.get(symbol, [])
        return [self._hypotheses[hid] for hid in ids if hid in self._hypotheses]
    
    def get_by_type(self, strategy_type: StrategyType) -> List[Hypothesis]:
        """Get all hypotheses of a type."""
        ids = self._by_type.get(strategy_type, [])
        return [self._hypotheses[hid] for hid in ids if hid in self._hypotheses]
    
    def get_live_strategies(self) -> List[Hypothesis]:
        """Get all live trading strategies."""
        return self.get_all(StrategyState.LIVE)
    
    def get_paper_strategies(self) -> List[Hypothesis]:
        """Get all paper trading strategies."""
        return self.get_all(StrategyState.PAPER_TRADING)
    
    def update(self, hypothesis: Hypothesis) -> None:
        """Update an existing hypothesis."""
        if hypothesis.hypothesis_id not in self._hypotheses:
            raise ValueError(f"Hypothesis {hypothesis.hypothesis_id} not found")
        
        hypothesis.updated_at = datetime.now()
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis
    
    def remove(self, hypothesis_id: str) -> bool:
        """Remove a hypothesis from registry."""
        if hypothesis_id not in self._hypotheses:
            return False
        
        hypothesis = self._hypotheses[hypothesis_id]
        
        # Remove from state list
        if hypothesis_id in self._by_state[hypothesis.state]:
            self._by_state[hypothesis.state].remove(hypothesis_id)
        
        # Remove from indexes
        for symbol in hypothesis.symbols:
            if hypothesis_id in self._by_symbol[symbol]:
                self._by_symbol[symbol].remove(hypothesis_id)
        
        if hypothesis_id in self._by_type[hypothesis.strategy_type]:
            self._by_type[hypothesis.strategy_type].remove(hypothesis_id)
        
        del self._hypotheses[hypothesis_id]
        
        logger.info(f"Removed hypothesis: {hypothesis_id}")
        return True
    
    # ===== State Transitions =====
    
    def _change_state(
        self,
        hypothesis: Hypothesis,
        new_state: StrategyState
    ) -> None:
        """Internal state change helper."""
        old_state = hypothesis.state
        
        # Remove from old state list
        if hypothesis.hypothesis_id in self._by_state[old_state]:
            self._by_state[old_state].remove(hypothesis.hypothesis_id)
        
        # Update state
        hypothesis.state = new_state
        hypothesis.updated_at = datetime.now()
        
        # Add to new state list
        self._by_state[new_state].append(hypothesis.hypothesis_id)
        
        # Update timestamps
        if new_state == StrategyState.PAPER_TRADING:
            hypothesis.paper_start = datetime.now()
        elif new_state == StrategyState.LIVE:
            hypothesis.live_start = datetime.now()
        elif new_state == StrategyState.RETIRED:
            hypothesis.retired_at = datetime.now()
        
        logger.info(
            f"State change: {hypothesis.name} "
            f"{old_state.value} → {new_state.value}"
        )
    
    def promote(self, hypothesis_id: str, force: bool = False) -> tuple[bool, str]:
        """
        Promote a hypothesis to the next state.
        
        INCUBATING → PAPER_TRADING → LIVE
        
        Args:
            hypothesis_id: ID of hypothesis to promote
            force: Skip validation checks
        
        Returns:
            Tuple of (success, message)
        """
        hypothesis = self.get(hypothesis_id)
        if not hypothesis:
            return False, "Hypothesis not found"
        
        # Check current state
        if hypothesis.state == StrategyState.INCUBATING:
            target_state = StrategyState.PAPER_TRADING
            capacity_list = self._by_state[StrategyState.PAPER_TRADING]
            max_capacity = self.config.max_paper_trading
        
        elif hypothesis.state == StrategyState.PAPER_TRADING:
            target_state = StrategyState.LIVE
            capacity_list = self._by_state[StrategyState.LIVE]
            max_capacity = self.config.max_live
        
        else:
            return False, f"Cannot promote from {hypothesis.state.value}"
        
        # Check capacity
        if len(capacity_list) >= max_capacity:
            return False, f"Capacity reached for {target_state.value} ({max_capacity})"
        
        # Validate promotion criteria
        if not force:
            can_promote, failures = hypothesis.can_promote()
            if not can_promote:
                return False, f"Criteria not met: {', '.join(failures)}"
        
        # Perform promotion
        self._change_state(hypothesis, target_state)
        self._stats["total_promoted"] += 1
        
        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.STRATEGY_PROMOTED,
            source="strategy_registry",
            payload={
                "hypothesis_id": hypothesis_id,
                "name": hypothesis.name,
                "new_state": target_state.value
            }
        ))
        
        # Call callbacks
        for callback in self._on_promotion:
            try:
                callback(hypothesis, target_state)
            except Exception as e:
                logger.error(f"Promotion callback error: {e}")
        
        return True, f"Promoted to {target_state.value}"
    
    def demote(self, hypothesis_id: str, reason: str = "") -> tuple[bool, str]:
        """
        Demote a hypothesis to previous state or retire.
        
        LIVE → PAPER_TRADING or RETIRED
        PAPER_TRADING → INCUBATING or RETIRED
        
        Args:
            hypothesis_id: ID of hypothesis to demote
            reason: Reason for demotion
        
        Returns:
            Tuple of (success, message)
        """
        hypothesis = self.get(hypothesis_id)
        if not hypothesis:
            return False, "Hypothesis not found"
        
        # Determine target state
        if hypothesis.state == StrategyState.LIVE:
            # Live strategies that fail go to paper for re-evaluation
            target_state = StrategyState.PAPER_TRADING
        elif hypothesis.state == StrategyState.PAPER_TRADING:
            # Paper strategies that fail get retired
            target_state = StrategyState.RETIRED
        else:
            return False, f"Cannot demote from {hypothesis.state.value}"
        
        # Perform demotion
        self._change_state(hypothesis, target_state)
        self._stats["total_demoted"] += 1
        
        hypothesis.notes += f"\nDemoted on {datetime.now().isoformat()}: {reason}"
        
        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.STRATEGY_DEMOTED,
            source="strategy_registry",
            payload={
                "hypothesis_id": hypothesis_id,
                "name": hypothesis.name,
                "new_state": target_state.value,
                "reason": reason
            }
        ))
        
        # Call callbacks
        for callback in self._on_demotion:
            try:
                callback(hypothesis, target_state, reason)
            except Exception as e:
                logger.error(f"Demotion callback error: {e}")
        
        return True, f"Demoted to {target_state.value}"
    
    def retire(self, hypothesis_id: str, reason: str = "") -> tuple[bool, str]:
        """
        Retire a hypothesis (terminal state).
        
        Args:
            hypothesis_id: ID of hypothesis to retire
            reason: Reason for retirement
        
        Returns:
            Tuple of (success, message)
        """
        hypothesis = self.get(hypothesis_id)
        if not hypothesis:
            return False, "Hypothesis not found"
        
        if hypothesis.state == StrategyState.RETIRED:
            return False, "Already retired"
        
        # Perform retirement
        self._change_state(hypothesis, StrategyState.RETIRED)
        self._stats["total_retired"] += 1
        
        hypothesis.notes += f"\nRetired on {datetime.now().isoformat()}: {reason}"
        
        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.STRATEGY_RETIRED,
            source="strategy_registry",
            payload={
                "hypothesis_id": hypothesis_id,
                "name": hypothesis.name,
                "reason": reason
            }
        ))
        
        # Call callbacks
        for callback in self._on_retirement:
            try:
                callback(hypothesis, reason)
            except Exception as e:
                logger.error(f"Retirement callback error: {e}")
        
        return True, "Retired"
    
    def pause(self, hypothesis_id: str, reason: str = "") -> tuple[bool, str]:
        """Pause a live or paper trading strategy."""
        hypothesis = self.get(hypothesis_id)
        if not hypothesis:
            return False, "Hypothesis not found"
        
        if hypothesis.state not in [StrategyState.LIVE, StrategyState.PAPER_TRADING]:
            return False, f"Cannot pause from {hypothesis.state.value}"
        
        # Store original state for resuming
        hypothesis.parameters["_paused_from"] = hypothesis.state.value
        self._change_state(hypothesis, StrategyState.PAUSED)
        
        hypothesis.notes += f"\nPaused on {datetime.now().isoformat()}: {reason}"
        
        return True, "Paused"
    
    def resume(self, hypothesis_id: str) -> tuple[bool, str]:
        """Resume a paused strategy."""
        hypothesis = self.get(hypothesis_id)
        if not hypothesis:
            return False, "Hypothesis not found"
        
        if hypothesis.state != StrategyState.PAUSED:
            return False, "Not paused"
        
        # Restore original state
        original_state = hypothesis.parameters.get("_paused_from", "paper_trading")
        target_state = StrategyState(original_state)
        
        del hypothesis.parameters["_paused_from"]
        self._change_state(hypothesis, target_state)
        
        hypothesis.notes += f"\nResumed on {datetime.now().isoformat()}"
        
        return True, f"Resumed to {target_state.value}"
    
    # ===== Performance Updates =====
    
    def update_metrics(
        self,
        hypothesis_id: str,
        metrics: PerformanceMetrics
    ) -> None:
        """Update performance metrics for a hypothesis."""
        hypothesis = self.get(hypothesis_id)
        if not hypothesis:
            return
        
        if hypothesis.state == StrategyState.INCUBATING:
            hypothesis.backtest_metrics = metrics
        elif hypothesis.state == StrategyState.PAPER_TRADING:
            hypothesis.paper_metrics = metrics
        elif hypothesis.state == StrategyState.LIVE:
            hypothesis.live_metrics = metrics
        
        hypothesis.updated_at = datetime.now()
        
        # Check for auto-promotion
        if self.config.auto_promote:
            can_promote, _ = hypothesis.can_promote()
            if can_promote:
                self.promote(hypothesis_id)
        
        # Check for auto-demotion
        if self.config.auto_demote:
            should_demote, reasons = hypothesis.should_demote()
            if should_demote:
                self.demote(hypothesis_id, "; ".join(reasons))
    
    # ===== Lifecycle Evaluation =====
    
    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate all active strategies for promotion/demotion.
        
        Call this periodically (e.g., daily) to manage lifecycle.
        """
        results = {
            "evaluated": 0,
            "promoted": [],
            "demoted": [],
            "retired": []
        }
        
        # Check incubating strategies for promotion
        for hypothesis in self.get_all(StrategyState.INCUBATING):
            results["evaluated"] += 1
            
            can_promote, failures = hypothesis.can_promote()
            if can_promote:
                success, msg = self.promote(hypothesis.hypothesis_id)
                if success:
                    results["promoted"].append(hypothesis.hypothesis_id)
        
        # Check paper trading strategies
        for hypothesis in self.get_all(StrategyState.PAPER_TRADING):
            results["evaluated"] += 1
            
            # Check for demotion first
            should_demote, reasons = hypothesis.should_demote()
            if should_demote:
                success, msg = self.demote(
                    hypothesis.hypothesis_id, 
                    "; ".join(reasons)
                )
                if success:
                    results["demoted"].append(hypothesis.hypothesis_id)
                continue
            
            # Check for promotion
            can_promote, failures = hypothesis.can_promote()
            if can_promote:
                success, msg = self.promote(hypothesis.hypothesis_id)
                if success:
                    results["promoted"].append(hypothesis.hypothesis_id)
        
        # Check live strategies for demotion
        for hypothesis in self.get_all(StrategyState.LIVE):
            results["evaluated"] += 1
            
            should_demote, reasons = hypothesis.should_demote()
            if should_demote:
                success, msg = self.demote(
                    hypothesis.hypothesis_id,
                    "; ".join(reasons)
                )
                if success:
                    results["demoted"].append(hypothesis.hypothesis_id)
        
        logger.info(
            f"Lifecycle evaluation: {results['evaluated']} evaluated, "
            f"{len(results['promoted'])} promoted, "
            f"{len(results['demoted'])} demoted"
        )
        
        return results
    
    # ===== Callbacks =====
    
    def on_promotion(self, callback: Callable) -> None:
        """Register callback for promotion events."""
        self._on_promotion.append(callback)
    
    def on_demotion(self, callback: Callable) -> None:
        """Register callback for demotion events."""
        self._on_demotion.append(callback)
    
    def on_retirement(self, callback: Callable) -> None:
        """Register callback for retirement events."""
        self._on_retirement.append(callback)
    
    # ===== Persistence =====
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save registry to file."""
        path = path or self.config.persist_path
        if not path:
            logger.warning("No persistence path configured")
            return
        
        data = {
            "hypotheses": {
                hid: h.to_dict() for hid, h in self._hypotheses.items()
            },
            "stats": self._stats,
            "saved_at": datetime.now().isoformat()
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Registry saved to {path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load registry from file."""
        path = path or self.config.persist_path
        if not path or not path.exists():
            logger.warning(f"No registry file found at {path}")
            return
        
        with open(path) as f:
            data = json.load(f)
        
        # Clear current state
        self._hypotheses.clear()
        for state in self._by_state:
            self._by_state[state].clear()
        self._by_symbol.clear()
        self._by_type.clear()
        
        # Load hypotheses
        for hid, hdata in data.get("hypotheses", {}).items():
            hypothesis = Hypothesis.from_dict(hdata)
            self._hypotheses[hid] = hypothesis
            self._by_state[hypothesis.state].append(hid)
            
            for symbol in hypothesis.symbols:
                self._by_symbol[symbol].append(hid)
            self._by_type[hypothesis.strategy_type].append(hid)
        
        self._stats = data.get("stats", self._stats)
        
        logger.info(f"Registry loaded: {len(self._hypotheses)} hypotheses")
    
    # ===== Statistics =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_hypotheses": len(self._hypotheses),
            "by_state": {
                state.value: len(ids) 
                for state, ids in self._by_state.items()
            },
            "by_type": {
                stype.value: len(ids) 
                for stype, ids in self._by_type.items()
            },
            "lifecycle": self._stats,
            "capacity": {
                "incubating": f"{len(self._by_state[StrategyState.INCUBATING])}/{self.config.max_incubating}",
                "paper_trading": f"{len(self._by_state[StrategyState.PAPER_TRADING])}/{self.config.max_paper_trading}",
                "live": f"{len(self._by_state[StrategyState.LIVE])}/{self.config.max_live}"
            }
        }


# Singleton instance
_registry: Optional[StrategyRegistry] = None


def get_registry(config: Optional[RegistryConfig] = None) -> StrategyRegistry:
    """Get the singleton StrategyRegistry instance."""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry(config)
    return _registry
