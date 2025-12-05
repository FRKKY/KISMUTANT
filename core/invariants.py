"""
INVARIANTS - The Immutable Laws of the Living Trading System

These rules are HARDCODED and cannot be modified by any learning process.
They exist to ensure the system's survival regardless of what it learns.

Modification of this file should be treated as a critical system change
requiring extensive review and testing.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Final
from enum import Enum, auto


class SystemState(Enum):
    """Possible states of the trading system."""
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()  # Temporary halt, can auto-resume
    STOPPED = auto()  # Manual intervention required
    KILLED = auto()   # Emergency shutdown, full manual restart required


@dataclass(frozen=True)  # frozen=True makes this immutable
class Invariants:
    """
    The immutable rules that govern system behavior.
    
    These values are FINAL and cannot be changed by any optimization,
    learning process, or runtime configuration.
    """
    
    # === POSITION LIMITS ===
    MAX_SINGLE_POSITION_PCT: Final[Decimal] = Decimal("0.25")  # 25% max per position
    MAX_CORRELATED_EXPOSURE_PCT: Final[Decimal] = Decimal("0.50")  # 50% max in correlated assets
    MIN_CASH_BUFFER_PCT: Final[Decimal] = Decimal("0.05")  # Always keep 5% cash
    
    # === DRAWDOWN CIRCUIT BREAKERS ===
    DRAWDOWN_LEVEL_1_PCT: Final[Decimal] = Decimal("0.10")  # 10% - Warning, reduce new positions
    DRAWDOWN_LEVEL_2_PCT: Final[Decimal] = Decimal("0.15")  # 15% - Halt new entries
    DRAWDOWN_LEVEL_3_PCT: Final[Decimal] = Decimal("0.20")  # 20% - Begin position reduction
    DRAWDOWN_LEVEL_4_PCT: Final[Decimal] = Decimal("0.25")  # 25% - Aggressive reduction
    DRAWDOWN_KILL_PCT: Final[Decimal] = Decimal("0.30")  # 30% - Full stop, manual restart
    
    # === DAILY LIMITS ===
    MAX_DAILY_LOSS_PCT: Final[Decimal] = Decimal("0.05")  # 5% - Halt trading for day
    MAX_DAILY_TRADES: Final[int] = 50  # Prevent runaway trading
    
    # === LEVERAGE ===
    MAX_LEVERAGE: Final[Decimal] = Decimal("1.0")  # No leverage allowed
    
    # === HYPOTHESIS VALIDATION ===
    MIN_TRADES_FOR_VALIDATION: Final[int] = 30  # Minimum trades before hypothesis can graduate
    MIN_BACKTEST_PERIODS: Final[int] = 252  # ~1 year of daily data minimum
    MAX_HYPOTHESIS_AGE_DAYS: Final[int] = 365  # Force re-validation after 1 year
    
    # === LEARNING BOUNDARIES ===
    MAX_PARAMETER_CHANGE_PCT: Final[Decimal] = Decimal("0.30")  # 30% max change per optimization
    MIN_SAMPLES_FOR_LEARNING: Final[int] = 50  # Minimum samples before adjusting
    
    # === OPERATIONAL ===
    HEARTBEAT_INTERVAL_SECONDS: Final[int] = 60  # System health check frequency
    MAX_ORDER_RETRY_ATTEMPTS: Final[int] = 3
    ORDER_TIMEOUT_SECONDS: Final[int] = 30


# Singleton instance - this is THE invariants object used everywhere
INVARIANTS: Final[Invariants] = Invariants()


class InvariantViolation(Exception):
    """Raised when any code attempts to violate an invariant."""
    
    def __init__(self, invariant_name: str, attempted_value: any, limit_value: any):
        self.invariant_name = invariant_name
        self.attempted_value = attempted_value
        self.limit_value = limit_value
        super().__init__(
            f"INVARIANT VIOLATION: {invariant_name} "
            f"(attempted: {attempted_value}, limit: {limit_value})"
        )


def enforce_position_limit(position_value: Decimal, portfolio_value: Decimal) -> None:
    """Enforce maximum single position size."""
    if portfolio_value <= 0:
        return
    position_pct = position_value / portfolio_value
    if position_pct > INVARIANTS.MAX_SINGLE_POSITION_PCT:
        raise InvariantViolation(
            "MAX_SINGLE_POSITION_PCT",
            float(position_pct),
            float(INVARIANTS.MAX_SINGLE_POSITION_PCT)
        )


def enforce_drawdown_limit(current_drawdown: Decimal) -> SystemState:
    """
    Check drawdown and return appropriate system state.
    
    This function ONLY returns state - it doesn't modify anything.
    The caller is responsible for acting on the returned state.
    """
    if current_drawdown >= INVARIANTS.DRAWDOWN_KILL_PCT:
        return SystemState.KILLED
    elif current_drawdown >= INVARIANTS.DRAWDOWN_LEVEL_3_PCT:
        return SystemState.PAUSED
    else:
        return SystemState.RUNNING


def enforce_leverage_limit(exposure: Decimal, capital: Decimal) -> None:
    """Ensure no leverage is being used."""
    if capital <= 0:
        return
    leverage = exposure / capital
    if leverage > INVARIANTS.MAX_LEVERAGE:
        raise InvariantViolation(
            "MAX_LEVERAGE",
            float(leverage),
            float(INVARIANTS.MAX_LEVERAGE)
        )


def enforce_daily_loss_limit(daily_pnl: Decimal, portfolio_value: Decimal) -> bool:
    """
    Check if daily loss limit is breached.
    Returns True if trading should halt for the day.
    """
    if portfolio_value <= 0:
        return True
    daily_loss_pct = abs(min(daily_pnl, Decimal("0"))) / portfolio_value
    return daily_loss_pct >= INVARIANTS.MAX_DAILY_LOSS_PCT


def enforce_daily_trade_limit(trade_count: int) -> bool:
    """
    Check if daily trade limit is reached.
    Returns True if no more trades should be placed today.
    """
    return trade_count >= INVARIANTS.MAX_DAILY_TRADES


# === VALIDATION ON MODULE LOAD ===
# Ensure invariants are sensible at import time

def _validate_invariants() -> None:
    """Validate that invariants are internally consistent."""
    inv = INVARIANTS
    
    # Drawdown levels must be in ascending order
    assert inv.DRAWDOWN_LEVEL_1_PCT < inv.DRAWDOWN_LEVEL_2_PCT < inv.DRAWDOWN_LEVEL_3_PCT
    assert inv.DRAWDOWN_LEVEL_3_PCT < inv.DRAWDOWN_LEVEL_4_PCT < inv.DRAWDOWN_KILL_PCT
    
    # Position limit must be positive and less than 100%
    assert Decimal("0") < inv.MAX_SINGLE_POSITION_PCT <= Decimal("1")
    
    # Cash buffer must be positive
    assert inv.MIN_CASH_BUFFER_PCT > Decimal("0")
    
    # No leverage means max leverage = 1
    assert inv.MAX_LEVERAGE == Decimal("1")
    
    print("✓ Invariants validated successfully")


_validate_invariants()
