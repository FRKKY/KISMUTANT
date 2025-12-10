"""
CAPITAL ALLOCATOR - Position Sizing with Kelly Criterion

Implements fractional Kelly criterion for optimal capital allocation:
- Half-Kelly for conservative sizing (reduces volatility)
- Per-strategy capital limits
- Portfolio-level risk constraints
- Dynamic rebalancing

Kelly Formula:
    f* = (p * b - q) / b
    
    where:
    f* = fraction of capital to bet
    p = probability of winning
    q = probability of losing (1 - p)
    b = win/loss ratio (avg_win / avg_loss)

Half-Kelly = f* / 2 (more conservative, reduces drawdowns)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal
from enum import Enum
import math

from loguru import logger

from hypothesis.models import (
    Hypothesis,
    StrategyState,
    PerformanceMetrics,
    TradingSignal
)
from hypothesis.registry import get_registry
from core.events import get_event_bus, Event, EventType
from core.invariants import get_invariants


class AllocationMethod(str, Enum):
    """Methods for allocating capital."""
    EQUAL_WEIGHT = "equal_weight"
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    QUARTER_KELLY = "quarter_kelly"
    RISK_PARITY = "risk_parity"
    PERFORMANCE_WEIGHTED = "performance_weighted"


@dataclass
class AllocationConfig:
    """Configuration for capital allocation."""
    
    # Total capital
    total_capital: float = 10_000_000  # ₩10M default
    
    # Allocation method
    method: AllocationMethod = AllocationMethod.HALF_KELLY
    
    # Strategy limits
    max_strategies: int = 3
    max_per_strategy_pct: float = 0.40      # Max 40% per strategy
    min_per_strategy_pct: float = 0.10      # Min 10% per strategy
    
    # Position limits
    max_per_position_pct: float = 0.15      # Max 15% per position
    max_sector_pct: float = 0.30            # Max 30% per sector
    
    # Risk limits
    max_portfolio_risk_pct: float = 0.02    # Max 2% portfolio risk per day
    max_correlation: float = 0.7            # Max correlation between strategies
    
    # Kelly parameters
    kelly_fraction: float = 0.5             # Half-Kelly
    min_kelly_fraction: float = 0.05        # Minimum allocation
    max_kelly_fraction: float = 0.25        # Maximum Kelly allocation
    
    # Rebalancing
    rebalance_threshold_pct: float = 0.05   # Rebalance if drift > 5%
    rebalance_frequency_days: int = 7       # Max days between rebalances
    
    # Cash reserve
    min_cash_pct: float = 0.10              # Keep 10% cash minimum


@dataclass
class StrategyAllocation:
    """Capital allocation for a single strategy."""
    
    hypothesis_id: str
    strategy_name: str
    
    # Allocation
    allocated_capital: float
    allocation_pct: float
    
    # Kelly metrics
    kelly_fraction: float = 0.0
    win_rate: float = 0.0
    win_loss_ratio: float = 0.0
    
    # Current state
    deployed_capital: float = 0.0
    cash_available: float = 0.0
    
    # Positions
    num_positions: int = 0
    max_positions: int = 5
    
    # Risk metrics
    current_risk: float = 0.0
    max_risk: float = 0.0
    
    # Timestamps
    allocated_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "strategy_name": self.strategy_name,
            "allocated_capital": self.allocated_capital,
            "allocation_pct": self.allocation_pct,
            "kelly_fraction": self.kelly_fraction,
            "deployed_capital": self.deployed_capital,
            "cash_available": self.cash_available,
            "num_positions": self.num_positions
        }


@dataclass
class PositionSizing:
    """Position sizing calculation result."""
    
    signal_id: str
    hypothesis_id: str
    symbol: str
    
    # Sizing
    position_size_krw: float
    position_size_shares: int
    position_pct: float
    
    # Risk
    risk_per_share: float
    total_risk: float
    risk_pct: float
    
    # Limits applied
    limited_by: Optional[str] = None  # What constraint limited the size
    original_size: float = 0.0        # Size before limits
    
    # Validation
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "position_size_krw": self.position_size_krw,
            "position_size_shares": self.position_size_shares,
            "position_pct": self.position_pct,
            "risk_pct": self.risk_pct,
            "limited_by": self.limited_by,
            "is_valid": self.is_valid
        }


class KellyCalculator:
    """
    Calculates Kelly criterion for position sizing.
    
    The Kelly criterion maximizes long-term growth rate,
    but full Kelly is too aggressive. We use fractional Kelly.
    """
    
    @staticmethod
    def calculate_kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Args:
            win_rate: Probability of winning (0 to 1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
        
        Returns:
            Kelly fraction (can be negative if edge is negative)
        """
        if avg_loss == 0:
            return 0.0
        
        # Win/loss ratio
        b = avg_win / avg_loss
        
        # Kelly formula: f* = (p * b - q) / b
        p = win_rate
        q = 1 - win_rate
        
        kelly = (p * b - q) / b
        
        return kelly
    
    @staticmethod
    def calculate_kelly_from_metrics(
        metrics: PerformanceMetrics
    ) -> float:
        """Calculate Kelly fraction from performance metrics."""
        if not metrics or metrics.total_trades < 10:
            return 0.0
        
        if metrics.avg_loss == 0:
            return 0.0
        
        return KellyCalculator.calculate_kelly(
            win_rate=metrics.win_rate,
            avg_win=abs(metrics.avg_win),
            avg_loss=abs(metrics.avg_loss)
        )
    
    @staticmethod
    def apply_kelly_fraction(
        kelly: float,
        fraction: float = 0.5,
        min_kelly: float = 0.0,
        max_kelly: float = 0.25
    ) -> float:
        """
        Apply fractional Kelly with bounds.
        
        Args:
            kelly: Raw Kelly fraction
            fraction: Kelly fraction to use (0.5 = half Kelly)
            min_kelly: Minimum allocation
            max_kelly: Maximum allocation
        
        Returns:
            Bounded fractional Kelly
        """
        if kelly <= 0:
            return 0.0
        
        fractional = kelly * fraction
        
        # Apply bounds
        return max(min_kelly, min(fractional, max_kelly))


class CapitalAllocator:
    """
    Manages capital allocation across strategies and positions.
    
    Responsibilities:
    1. Allocate capital to live strategies
    2. Calculate position sizes for signals
    3. Enforce risk limits
    4. Track deployed capital
    5. Trigger rebalancing
    """
    
    def __init__(self, config: Optional[AllocationConfig] = None):
        self.config = config or AllocationConfig()
        self._registry = get_registry()
        self._invariants = get_invariants()
        
        # Allocations
        self._allocations: Dict[str, StrategyAllocation] = {}
        
        # Portfolio state
        self._total_deployed: float = 0.0
        self._total_cash: float = self.config.total_capital
        self._positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        
        # Risk tracking
        self._daily_risk: float = 0.0
        self._sector_exposure: Dict[str, float] = {}
        
        # Rebalancing
        self._last_rebalance: Optional[datetime] = None
        
        logger.info(f"CapitalAllocator initialized with ₩{self.config.total_capital:,.0f}")
    
    def update_capital(self, total_capital: float) -> None:
        """Update total capital (e.g., from deposit or P&L)."""
        self.config.total_capital = total_capital
        logger.info(f"Total capital updated to ₩{total_capital:,.0f}")
    
    def set_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """Update current positions."""
        self._positions = positions
        
        # Recalculate deployed capital
        self._total_deployed = sum(
            pos.get('market_value', 0) 
            for pos in positions.values()
        )
        self._total_cash = self.config.total_capital - self._total_deployed
    
    # ===== Strategy Allocation =====
    
    def allocate_to_strategies(self) -> Dict[str, StrategyAllocation]:
        """
        Allocate capital to all live strategies.
        
        Uses configured method (Kelly, equal weight, etc.)
        """
        live_strategies = self._registry.get_live_strategies()
        
        if not live_strategies:
            logger.info("No live strategies to allocate to")
            return {}
        
        # Limit to max strategies
        if len(live_strategies) > self.config.max_strategies:
            # Sort by Sharpe ratio and take top N
            live_strategies.sort(
                key=lambda h: (h.live_metrics.sharpe_ratio if h.live_metrics else 0),
                reverse=True
            )
            live_strategies = live_strategies[:self.config.max_strategies]
        
        # Calculate allocations based on method
        if self.config.method == AllocationMethod.EQUAL_WEIGHT:
            allocations = self._allocate_equal_weight(live_strategies)
        elif self.config.method in [AllocationMethod.KELLY, AllocationMethod.HALF_KELLY, AllocationMethod.QUARTER_KELLY]:
            allocations = self._allocate_kelly(live_strategies)
        elif self.config.method == AllocationMethod.PERFORMANCE_WEIGHTED:
            allocations = self._allocate_performance_weighted(live_strategies)
        else:
            allocations = self._allocate_equal_weight(live_strategies)
        
        self._allocations = allocations
        self._last_rebalance = datetime.now()
        
        # Update hypothesis objects
        for hyp_id, alloc in allocations.items():
            hypothesis = self._registry.get(hyp_id)
            if hypothesis:
                hypothesis.allocated_capital = alloc.allocated_capital
                hypothesis.capital_pct = alloc.allocation_pct
        
        logger.info(f"Allocated capital to {len(allocations)} strategies")
        
        return allocations
    
    def _allocate_equal_weight(
        self,
        strategies: List[Hypothesis]
    ) -> Dict[str, StrategyAllocation]:
        """Equal weight allocation across strategies."""
        if not strategies:
            return {}
        
        # Available capital (after cash reserve)
        available = self.config.total_capital * (1 - self.config.min_cash_pct)
        
        # Equal split
        per_strategy = available / len(strategies)
        
        # Apply per-strategy limits
        max_per_strategy = self.config.total_capital * self.config.max_per_strategy_pct
        min_per_strategy = self.config.total_capital * self.config.min_per_strategy_pct
        
        per_strategy = max(min_per_strategy, min(per_strategy, max_per_strategy))
        
        allocations = {}
        for hypothesis in strategies:
            allocations[hypothesis.hypothesis_id] = StrategyAllocation(
                hypothesis_id=hypothesis.hypothesis_id,
                strategy_name=hypothesis.name,
                allocated_capital=per_strategy,
                allocation_pct=per_strategy / self.config.total_capital,
                cash_available=per_strategy
            )
        
        return allocations
    
    def _allocate_kelly(
        self,
        strategies: List[Hypothesis]
    ) -> Dict[str, StrategyAllocation]:
        """Kelly criterion based allocation."""
        if not strategies:
            return {}
        
        # Determine Kelly fraction based on method
        if self.config.method == AllocationMethod.KELLY:
            kelly_mult = 1.0
        elif self.config.method == AllocationMethod.HALF_KELLY:
            kelly_mult = 0.5
        else:  # QUARTER_KELLY
            kelly_mult = 0.25
        
        # Calculate Kelly for each strategy
        kelly_fractions = {}
        for hypothesis in strategies:
            metrics = hypothesis.live_metrics or hypothesis.paper_metrics or hypothesis.backtest_metrics
            
            if metrics and metrics.total_trades >= 10:
                raw_kelly = KellyCalculator.calculate_kelly_from_metrics(metrics)
                kelly = KellyCalculator.apply_kelly_fraction(
                    raw_kelly,
                    fraction=kelly_mult,
                    min_kelly=self.config.min_kelly_fraction,
                    max_kelly=self.config.max_kelly_fraction
                )
            else:
                # Default to minimum allocation for new strategies
                kelly = self.config.min_kelly_fraction
            
            kelly_fractions[hypothesis.hypothesis_id] = kelly
        
        # Normalize if total exceeds available
        total_kelly = sum(kelly_fractions.values())
        available_pct = 1 - self.config.min_cash_pct
        
        if total_kelly > available_pct:
            scale = available_pct / total_kelly
            kelly_fractions = {k: v * scale for k, v in kelly_fractions.items()}
        
        # Apply per-strategy limits
        allocations = {}
        for hypothesis in strategies:
            hyp_id = hypothesis.hypothesis_id
            kelly = kelly_fractions[hyp_id]
            
            # Apply limits
            allocation_pct = max(
                self.config.min_per_strategy_pct,
                min(kelly, self.config.max_per_strategy_pct)
            )
            
            allocated = self.config.total_capital * allocation_pct
            
            metrics = hypothesis.live_metrics or hypothesis.paper_metrics
            
            allocations[hyp_id] = StrategyAllocation(
                hypothesis_id=hyp_id,
                strategy_name=hypothesis.name,
                allocated_capital=allocated,
                allocation_pct=allocation_pct,
                kelly_fraction=kelly,
                win_rate=metrics.win_rate if metrics else 0,
                win_loss_ratio=(metrics.avg_win / metrics.avg_loss) if metrics and metrics.avg_loss else 0,
                cash_available=allocated
            )
        
        return allocations
    
    def _allocate_performance_weighted(
        self,
        strategies: List[Hypothesis]
    ) -> Dict[str, StrategyAllocation]:
        """Weight by Sharpe ratio."""
        if not strategies:
            return {}
        
        # Get Sharpe ratios (use 0.5 as default for strategies without metrics)
        sharpes = {}
        for hypothesis in strategies:
            metrics = hypothesis.live_metrics or hypothesis.paper_metrics
            sharpe = metrics.sharpe_ratio if metrics else 0.5
            sharpes[hypothesis.hypothesis_id] = max(0.1, sharpe)  # Floor at 0.1
        
        # Normalize to weights
        total_sharpe = sum(sharpes.values())
        weights = {k: v / total_sharpe for k, v in sharpes.items()}
        
        # Available capital
        available = self.config.total_capital * (1 - self.config.min_cash_pct)
        
        allocations = {}
        for hypothesis in strategies:
            hyp_id = hypothesis.hypothesis_id
            weight = weights[hyp_id]
            
            # Apply limits
            allocation_pct = max(
                self.config.min_per_strategy_pct,
                min(weight, self.config.max_per_strategy_pct)
            )
            
            allocated = self.config.total_capital * allocation_pct
            
            allocations[hyp_id] = StrategyAllocation(
                hypothesis_id=hyp_id,
                strategy_name=hypothesis.name,
                allocated_capital=allocated,
                allocation_pct=allocation_pct,
                cash_available=allocated
            )
        
        return allocations
    
    # ===== Position Sizing =====
    
    def size_position(
        self,
        signal: TradingSignal,
        current_price: float,
        atr: Optional[float] = None
    ) -> PositionSizing:
        """
        Calculate position size for a trading signal.
        
        Args:
            signal: Trading signal to size
            current_price: Current market price
            atr: Average True Range (for risk calculation)
        
        Returns:
            PositionSizing with calculated size and risk
        """
        # Get strategy allocation
        allocation = self._allocations.get(signal.hypothesis_id)
        
        if not allocation:
            return PositionSizing(
                signal_id=signal.signal_id,
                hypothesis_id=signal.hypothesis_id,
                symbol=signal.symbol,
                position_size_krw=0,
                position_size_shares=0,
                position_pct=0,
                risk_per_share=0,
                total_risk=0,
                risk_pct=0,
                is_valid=False,
                rejection_reason="No allocation for strategy"
            )
        
        # Check available capital
        if allocation.cash_available <= 0:
            return PositionSizing(
                signal_id=signal.signal_id,
                hypothesis_id=signal.hypothesis_id,
                symbol=signal.symbol,
                position_size_krw=0,
                position_size_shares=0,
                position_pct=0,
                risk_per_share=0,
                total_risk=0,
                risk_pct=0,
                is_valid=False,
                rejection_reason="No available capital in strategy allocation"
            )
        
        # Calculate risk per share
        if signal.stop_loss:
            risk_per_share = abs(current_price - signal.stop_loss)
        elif atr:
            risk_per_share = atr * 2  # Default 2 ATR stop
        else:
            risk_per_share = current_price * 0.02  # Default 2% risk
        
        # ===== SIZE CALCULATION =====
        
        # Method 1: Risk-based sizing
        # Size = (Capital * Risk%) / Risk per share
        max_risk_pct = 0.02  # Risk 2% of strategy capital per trade
        risk_based_size = (allocation.allocated_capital * max_risk_pct) / risk_per_share
        
        # Method 2: Percentage of allocation
        # Use signal confidence to scale
        confidence_factor = signal.confidence
        pct_based_size = allocation.cash_available * confidence_factor * 0.5  # Max 50% of available
        
        # Take the smaller of the two
        position_size_krw = min(risk_based_size * current_price, pct_based_size)
        original_size = position_size_krw
        
        # ===== APPLY LIMITS =====
        
        limited_by = None
        
        # Limit 1: Max per position (portfolio level)
        max_position = self.config.total_capital * self.config.max_per_position_pct
        if position_size_krw > max_position:
            position_size_krw = max_position
            limited_by = f"max_per_position ({self.config.max_per_position_pct:.0%})"
        
        # Limit 2: Strategy allocation
        if position_size_krw > allocation.cash_available:
            position_size_krw = allocation.cash_available
            limited_by = "strategy_cash_available"
        
        # Limit 3: Invariant max position
        invariant_max = self.config.total_capital * self._invariants.MAX_SINGLE_POSITION
        if position_size_krw > invariant_max:
            position_size_krw = invariant_max
            limited_by = f"invariant_max_position ({self._invariants.MAX_SINGLE_POSITION:.0%})"
        
        # Limit 4: Daily risk limit
        trade_risk = (position_size_krw / current_price) * risk_per_share
        trade_risk_pct = trade_risk / self.config.total_capital
        
        if self._daily_risk + trade_risk_pct > self.config.max_portfolio_risk_pct:
            # Scale down to fit within risk budget
            available_risk = self.config.max_portfolio_risk_pct - self._daily_risk
            if available_risk <= 0:
                return PositionSizing(
                    signal_id=signal.signal_id,
                    hypothesis_id=signal.hypothesis_id,
                    symbol=signal.symbol,
                    position_size_krw=0,
                    position_size_shares=0,
                    position_pct=0,
                    risk_per_share=risk_per_share,
                    total_risk=0,
                    risk_pct=0,
                    is_valid=False,
                    rejection_reason="Daily risk limit reached"
                )
            
            scale = available_risk / trade_risk_pct
            position_size_krw *= scale
            limited_by = "daily_risk_limit"
        
        # Calculate shares
        position_size_shares = int(position_size_krw / current_price)
        
        # Minimum size check
        if position_size_shares < 1:
            return PositionSizing(
                signal_id=signal.signal_id,
                hypothesis_id=signal.hypothesis_id,
                symbol=signal.symbol,
                position_size_krw=0,
                position_size_shares=0,
                position_pct=0,
                risk_per_share=risk_per_share,
                total_risk=0,
                risk_pct=0,
                is_valid=False,
                rejection_reason="Position size below minimum (1 share)"
            )
        
        # Recalculate with whole shares
        position_size_krw = position_size_shares * current_price
        total_risk = position_size_shares * risk_per_share
        risk_pct = total_risk / self.config.total_capital
        
        return PositionSizing(
            signal_id=signal.signal_id,
            hypothesis_id=signal.hypothesis_id,
            symbol=signal.symbol,
            position_size_krw=position_size_krw,
            position_size_shares=position_size_shares,
            position_pct=position_size_krw / self.config.total_capital,
            risk_per_share=risk_per_share,
            total_risk=total_risk,
            risk_pct=risk_pct,
            limited_by=limited_by,
            original_size=original_size,
            is_valid=True
        )
    
    def reserve_capital(
        self,
        hypothesis_id: str,
        amount: float
    ) -> bool:
        """Reserve capital when entering a position."""
        allocation = self._allocations.get(hypothesis_id)
        if not allocation:
            return False
        
        if amount > allocation.cash_available:
            return False
        
        allocation.cash_available -= amount
        allocation.deployed_capital += amount
        allocation.num_positions += 1
        allocation.last_updated = datetime.now()
        
        self._total_deployed += amount
        self._total_cash -= amount
        
        return True
    
    def release_capital(
        self,
        hypothesis_id: str,
        amount: float,
        pnl: float = 0
    ) -> bool:
        """Release capital when exiting a position."""
        allocation = self._allocations.get(hypothesis_id)
        if not allocation:
            return False
        
        # Return capital + P&L
        returned = amount + pnl
        
        allocation.deployed_capital -= amount
        allocation.cash_available += returned
        allocation.num_positions = max(0, allocation.num_positions - 1)
        allocation.last_updated = datetime.now()
        
        self._total_deployed -= amount
        self._total_cash += returned
        
        # Update total capital with P&L
        self.config.total_capital += pnl
        
        return True
    
    def add_daily_risk(self, risk_pct: float) -> None:
        """Add to daily risk tracking."""
        self._daily_risk += risk_pct
    
    def reset_daily_risk(self) -> None:
        """Reset daily risk (call at market open)."""
        self._daily_risk = 0.0
    
    # ===== Rebalancing =====
    
    def needs_rebalance(self) -> Tuple[bool, List[str]]:
        """Check if portfolio needs rebalancing."""
        reasons = []
        
        # Check time since last rebalance
        if self._last_rebalance:
            days_since = (datetime.now() - self._last_rebalance).days
            if days_since >= self.config.rebalance_frequency_days:
                reasons.append(f"Time-based ({days_since} days since last)")
        
        # Check drift from target allocations
        for hyp_id, allocation in self._allocations.items():
            target_pct = allocation.allocation_pct
            
            # Current actual allocation
            if allocation.allocated_capital > 0:
                current_pct = (allocation.deployed_capital + allocation.cash_available) / self.config.total_capital
                drift = abs(current_pct - target_pct)
                
                if drift > self.config.rebalance_threshold_pct:
                    reasons.append(f"{allocation.strategy_name} drift: {drift:.1%}")
        
        # Check if strategies changed
        live_strategies = self._registry.get_live_strategies()
        current_ids = set(self._allocations.keys())
        live_ids = set(h.hypothesis_id for h in live_strategies)
        
        if current_ids != live_ids:
            added = live_ids - current_ids
            removed = current_ids - live_ids
            if added:
                reasons.append(f"New strategies: {len(added)}")
            if removed:
                reasons.append(f"Removed strategies: {len(removed)}")
        
        return len(reasons) > 0, reasons
    
    def rebalance(self) -> Dict[str, Any]:
        """Perform portfolio rebalancing."""
        logger.info("Starting portfolio rebalance...")
        
        # Store old allocations for comparison
        old_allocations = {k: v.allocation_pct for k, v in self._allocations.items()}
        
        # Recalculate allocations
        new_allocations = self.allocate_to_strategies()
        
        # Calculate changes
        changes = []
        for hyp_id, new_alloc in new_allocations.items():
            old_pct = old_allocations.get(hyp_id, 0)
            new_pct = new_alloc.allocation_pct
            
            if abs(new_pct - old_pct) > 0.01:  # > 1% change
                changes.append({
                    "strategy": new_alloc.strategy_name,
                    "old_pct": old_pct,
                    "new_pct": new_pct,
                    "change": new_pct - old_pct
                })
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "total_capital": self.config.total_capital,
            "num_strategies": len(new_allocations),
            "changes": changes
        }
        
        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.POSITION_UPDATED,
            source="capital_allocator",
            payload={"action": "rebalance", **result}
        ))
        
        logger.info(f"Rebalance complete: {len(changes)} allocation changes")
        
        return result
    
    # ===== Queries =====
    
    def get_allocation(self, hypothesis_id: str) -> Optional[StrategyAllocation]:
        """Get allocation for a strategy."""
        return self._allocations.get(hypothesis_id)
    
    def get_all_allocations(self) -> Dict[str, StrategyAllocation]:
        """Get all strategy allocations."""
        return self._allocations.copy()
    
    def get_available_capital(self) -> float:
        """Get total available (undeployed) capital."""
        return self._total_cash
    
    def get_deployed_capital(self) -> float:
        """Get total deployed capital."""
        return self._total_deployed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get allocator statistics."""
        return {
            "total_capital": self.config.total_capital,
            "deployed_capital": self._total_deployed,
            "available_capital": self._total_cash,
            "deployment_pct": self._total_deployed / self.config.total_capital if self.config.total_capital > 0 else 0,
            "daily_risk_used": self._daily_risk,
            "daily_risk_limit": self.config.max_portfolio_risk_pct,
            "num_strategies": len(self._allocations),
            "method": self.config.method.value,
            "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
            "allocations": {
                k: v.to_dict() for k, v in self._allocations.items()
            }
        }


# Singleton
_allocator: Optional[CapitalAllocator] = None


def get_capital_allocator(config: Optional[AllocationConfig] = None) -> CapitalAllocator:
    """Get the singleton CapitalAllocator instance."""
    global _allocator
    if _allocator is None:
        _allocator = CapitalAllocator(config)
    return _allocator
