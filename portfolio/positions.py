"""
POSITION MANAGER - Tracks Open Positions and P&L

Manages the state of all open positions:
- Entry tracking
- P&L calculation (realized and unrealized)
- Position updates from broker
- Exit tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from decimal import Decimal

from loguru import logger

from core.events import get_event_bus, Event, EventType


class PositionSide(str, Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionStatus(str, Enum):
    """Position lifecycle status."""
    PENDING = "pending"      # Order submitted, not filled
    OPEN = "open"           # Position is active
    CLOSING = "closing"     # Exit order submitted
    CLOSED = "closed"       # Position fully closed


@dataclass
class Position:
    """Represents an open position."""
    
    # Identity
    position_id: str
    symbol: str
    hypothesis_id: str
    
    # Entry
    side: PositionSide
    entry_price: float
    entry_time: datetime
    quantity: int
    
    # Current state
    status: PositionStatus = PositionStatus.OPEN
    current_price: float = 0.0
    
    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Cost basis
    total_cost: float = 0.0
    commission: float = 0.0
    
    # Exit
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    
    # Metadata
    signal_id: Optional[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.total_cost == 0:
            self.total_cost = self.entry_price * self.quantity
        if self.current_price == 0:
            self.current_price = self.entry_price
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.current_price * self.quantity
    
    @property
    def holding_period(self) -> timedelta:
        """Time since entry."""
        end_time = self.exit_time or datetime.now()
        return end_time - self.entry_time
    
    @property
    def holding_days(self) -> float:
        """Holding period in days."""
        return self.holding_period.total_seconds() / 86400
    
    def update_price(self, price: float) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = price
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
        
        self.unrealized_pnl -= self.commission
        self.unrealized_pnl_pct = self.unrealized_pnl / self.total_cost if self.total_cost > 0 else 0
    
    def close(
        self,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        reason: str = ""
    ) -> float:
        """
        Close the position and calculate realized P&L.
        
        Returns:
            Realized P&L
        """
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.exit_reason = reason
        self.status = PositionStatus.CLOSED
        
        if self.side == PositionSide.LONG:
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
        
        self.realized_pnl -= self.commission
        self.unrealized_pnl = 0
        self.unrealized_pnl_pct = 0
        
        return self.realized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "hypothesis_id": self.hypothesis_id,
            "side": self.side.value,
            "status": self.status.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "realized_pnl": self.realized_pnl,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "holding_days": self.holding_days,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason
        }


@dataclass
class PortfolioSummary:
    """Summary of portfolio state."""
    
    # Positions
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    
    # Value
    total_market_value: float = 0.0
    total_cost_basis: float = 0.0
    
    # P&L
    total_unrealized_pnl: float = 0.0
    total_unrealized_pnl_pct: float = 0.0
    total_realized_pnl: float = 0.0
    
    # By strategy
    pnl_by_strategy: Dict[str, float] = field(default_factory=dict)
    
    # Risk
    largest_position_pct: float = 0.0
    largest_loss_pct: float = 0.0
    
    # Timestamp
    as_of: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_positions": self.total_positions,
            "long_positions": self.long_positions,
            "short_positions": self.short_positions,
            "total_market_value": self.total_market_value,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_unrealized_pnl_pct": self.total_unrealized_pnl_pct,
            "total_realized_pnl": self.total_realized_pnl,
            "largest_position_pct": self.largest_position_pct,
            "as_of": self.as_of.isoformat()
        }


class PositionManager:
    """
    Manages all open and closed positions.
    
    Responsibilities:
    1. Track open positions
    2. Update positions with market prices
    3. Calculate P&L
    4. Manage position lifecycle
    5. Generate position reports
    """
    
    def __init__(self):
        # Open positions by symbol
        self._positions: Dict[str, Position] = {}
        
        # Closed positions (for history)
        self._closed_positions: List[Position] = []
        
        # Aggregates
        self._total_realized_pnl: float = 0.0
        
        # Position counter for IDs
        self._position_counter: int = 0
        
        logger.info("PositionManager initialized")
    
    def _generate_position_id(self) -> str:
        """Generate unique position ID."""
        self._position_counter += 1
        return f"pos_{datetime.now().strftime('%Y%m%d')}_{self._position_counter:05d}"
    
    # ===== Position Lifecycle =====
    
    def open_position(
        self,
        symbol: str,
        hypothesis_id: str,
        side: PositionSide,
        quantity: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        signal_id: Optional[str] = None,
        commission: float = 0.0
    ) -> Position:
        """
        Open a new position.
        
        Returns:
            The created Position object
        """
        # Check if position already exists for symbol
        if symbol in self._positions:
            existing = self._positions[symbol]
            logger.warning(
                f"Position already exists for {symbol}, "
                f"consider adding to existing position"
            )
            # For now, we'll allow multiple logical positions
            # but track by symbol for simplicity
        
        position = Position(
            position_id=self._generate_position_id(),
            symbol=symbol,
            hypothesis_id=hypothesis_id,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=quantity,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_id=signal_id,
            commission=commission
        )
        
        self._positions[symbol] = position
        
        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.POSITION_OPENED,
            source="position_manager",
            payload=position.to_dict()
        ))
        
        logger.info(
            f"Opened {side.value} position: {symbol} "
            f"{quantity} @ {entry_price:.2f}"
        )
        
        return position
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = ""
    ) -> Optional[Position]:
        """
        Close an existing position.
        
        Returns:
            The closed Position object, or None if not found
        """
        if symbol not in self._positions:
            logger.warning(f"No position found for {symbol}")
            return None
        
        position = self._positions[symbol]
        realized_pnl = position.close(exit_price, reason=reason)
        
        # Move to closed positions
        self._closed_positions.append(position)
        del self._positions[symbol]
        
        # Update total realized P&L
        self._total_realized_pnl += realized_pnl
        
        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.POSITION_CLOSED,
            source="position_manager",
            payload={
                **position.to_dict(),
                "realized_pnl": realized_pnl
            }
        ))
        
        logger.info(
            f"Closed position: {symbol} @ {exit_price:.2f}, "
            f"P&L: ₩{realized_pnl:,.0f} ({reason})"
        )
        
        return position
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update all positions with current prices.
        
        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, position in self._positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
    
    def update_position_price(self, symbol: str, price: float) -> None:
        """Update a single position's price."""
        if symbol in self._positions:
            self._positions[symbol].update_price(price)
    
    # ===== Queries =====
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return self._positions.copy()
    
    def get_positions_by_strategy(self, hypothesis_id: str) -> List[Position]:
        """Get all positions for a strategy."""
        return [
            p for p in self._positions.values()
            if p.hypothesis_id == hypothesis_id
        ]
    
    def get_closed_positions(
        self,
        since: Optional[datetime] = None,
        hypothesis_id: Optional[str] = None
    ) -> List[Position]:
        """Get closed positions with optional filters."""
        positions = self._closed_positions
        
        if since:
            positions = [p for p in positions if p.exit_time and p.exit_time >= since]
        
        if hypothesis_id:
            positions = [p for p in positions if p.hypothesis_id == hypothesis_id]
        
        return positions
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position in a symbol."""
        return symbol in self._positions
    
    # ===== Risk Checks =====
    
    def check_stop_losses(self, prices: Dict[str, float]) -> List[Position]:
        """
        Check which positions have hit their stop loss.
        
        Returns:
            List of positions that should be closed
        """
        triggered = []
        
        for symbol, position in self._positions.items():
            if symbol not in prices or not position.stop_loss:
                continue
            
            price = prices[symbol]
            
            if position.side == PositionSide.LONG:
                if price <= position.stop_loss:
                    triggered.append(position)
            else:  # SHORT
                if price >= position.stop_loss:
                    triggered.append(position)
        
        return triggered
    
    def check_take_profits(self, prices: Dict[str, float]) -> List[Position]:
        """
        Check which positions have hit their take profit.
        
        Returns:
            List of positions that should be closed
        """
        triggered = []
        
        for symbol, position in self._positions.items():
            if symbol not in prices or not position.take_profit:
                continue
            
            price = prices[symbol]
            
            if position.side == PositionSide.LONG:
                if price >= position.take_profit:
                    triggered.append(position)
            else:  # SHORT
                if price <= position.take_profit:
                    triggered.append(position)
        
        return triggered
    
    def update_trailing_stops(self, prices: Dict[str, float], atr_multiple: float = 1.5) -> None:
        """Update trailing stops based on current prices."""
        for symbol, position in self._positions.items():
            if symbol not in prices or not position.trailing_stop:
                continue
            
            price = prices[symbol]
            
            if position.side == PositionSide.LONG:
                # Trail up only
                new_stop = price * (1 - atr_multiple * 0.01)  # Simplified
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
            else:
                # Trail down only
                new_stop = price * (1 + atr_multiple * 0.01)
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
    
    # ===== Summaries =====
    
    def get_summary(self, total_capital: float = 0) -> PortfolioSummary:
        """Get portfolio summary."""
        summary = PortfolioSummary()
        
        summary.total_positions = len(self._positions)
        summary.total_realized_pnl = self._total_realized_pnl
        
        pnl_by_strategy = {}
        
        for position in self._positions.values():
            # Count by side
            if position.side == PositionSide.LONG:
                summary.long_positions += 1
            else:
                summary.short_positions += 1
            
            # Aggregate values
            summary.total_market_value += position.market_value
            summary.total_cost_basis += position.total_cost
            summary.total_unrealized_pnl += position.unrealized_pnl
            
            # By strategy
            hyp_id = position.hypothesis_id
            pnl_by_strategy[hyp_id] = pnl_by_strategy.get(hyp_id, 0) + position.unrealized_pnl
            
            # Track largest position
            if total_capital > 0:
                pos_pct = position.market_value / total_capital
                if pos_pct > summary.largest_position_pct:
                    summary.largest_position_pct = pos_pct
            
            # Track largest loss
            if position.unrealized_pnl_pct < summary.largest_loss_pct:
                summary.largest_loss_pct = position.unrealized_pnl_pct
        
        # Calculate percentages
        if summary.total_cost_basis > 0:
            summary.total_unrealized_pnl_pct = summary.total_unrealized_pnl / summary.total_cost_basis
        
        summary.pnl_by_strategy = pnl_by_strategy
        summary.as_of = datetime.now()
        
        return summary
    
    def get_positions_for_broker_sync(self) -> Dict[str, Dict[str, Any]]:
        """
        Get positions in format for syncing with allocator/broker.
        
        Returns dict of symbol -> position info
        """
        return {
            symbol: {
                "symbol": symbol,
                "quantity": pos.quantity,
                "side": pos.side.value,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "hypothesis_id": pos.hypothesis_id
            }
            for symbol, pos in self._positions.items()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get position manager statistics."""
        summary = self.get_summary()

        return {
            "open_positions": len(self._positions),
            "closed_positions": len(self._closed_positions),
            "total_market_value": summary.total_market_value,
            "total_unrealized_pnl": summary.total_unrealized_pnl,
            "total_realized_pnl": self._total_realized_pnl,
            "positions": [p.to_dict() for p in self._positions.values()]
        }

    # ===== Database Persistence =====

    def load_from_database(self) -> int:
        """
        Load open positions from database.

        Called on startup to restore state from persistent storage.

        Returns:
            Number of positions loaded
        """
        try:
            from memory.state_persistence import get_state_persistence

            persistence = get_state_persistence()
            pos_dicts = persistence.load_positions()

            if not pos_dicts:
                logger.info("No open positions found in database")
                return 0

            loaded = 0
            for pdata in pos_dicts:
                try:
                    # Create position from database data
                    position = Position(
                        position_id=self._generate_position_id(),
                        symbol=pdata['symbol'],
                        hypothesis_id=pdata.get('hypothesis_id', 'unknown'),
                        side=PositionSide.LONG,  # Default to long for ISA
                        entry_price=pdata['entry_price'],
                        entry_time=pdata.get('entry_time', datetime.now()),
                        quantity=pdata['quantity'],
                        current_price=pdata.get('current_price', pdata['entry_price']),
                        unrealized_pnl=pdata.get('unrealized_pnl', 0.0),
                        unrealized_pnl_pct=pdata.get('unrealized_pnl_pct', 0.0),
                    )

                    self._positions[position.symbol] = position
                    loaded += 1

                except Exception as e:
                    logger.error(f"Failed to restore position {pdata.get('symbol')}: {e}")

            logger.info(f"Restored {loaded} open positions from database")
            return loaded

        except Exception as e:
            logger.error(f"Failed to load positions from database: {e}")
            return 0

    def save_to_database(self) -> int:
        """
        Save all open positions to database.

        Called periodically and on shutdown to ensure state is persisted.

        Returns:
            Number of positions saved
        """
        try:
            from memory.state_persistence import get_state_persistence

            persistence = get_state_persistence()
            return persistence.save_all_positions(self)

        except Exception as e:
            logger.error(f"Failed to save positions to database: {e}")
            return 0

    def get_all_open(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())


# Singleton
_position_manager: Optional[PositionManager] = None


def get_position_manager() -> PositionManager:
    """Get the singleton PositionManager instance."""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
    return _position_manager
