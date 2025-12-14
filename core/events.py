"""
EVENT SYSTEM - The nervous system of the Living Trading System

All modules communicate through events, creating a loosely coupled architecture
that allows the system to evolve without tight dependencies.

Events are the primary way information flows through the system:
- Market data updates
- Hypothesis signals
- Order status changes
- Risk alerts
- Learning discoveries
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic, Awaitable
from enum import Enum, auto
from collections import defaultdict
import asyncio
from loguru import logger
import uuid


class EventType(str, Enum):
    """All system event types."""
    
    # === SYSTEM ===
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_ERROR = "system_error"
    SYSTEM_ALERT = "system_alert"
    HEARTBEAT = "heartbeat"
    
    # === MARKET ===
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    MARKET_DATA = "market_data"
    
    # === ORDERS ===
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_ERROR = "order_error"
    
    # === POSITIONS ===
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    
    # === PERCEPTION ===
    PATTERN_DISCOVERED = "pattern_discovered"
    UNIVERSE_UPDATED = "universe_updated"
    
    # === HYPOTHESIS ===
    STRATEGY_REGISTERED = "strategy_registered"
    STRATEGY_PROMOTED = "strategy_promoted"
    STRATEGY_DEMOTED = "strategy_demoted"
    STRATEGY_RETIRED = "strategy_retired"
    
    # === SIGNALS ===
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_EXPIRED = "signal_expired"
    
    # === VALIDATION ===
    BACKTEST_COMPLETED = "backtest_completed"
    BACKTEST_FAILED = "backtest_failed"
    
    # === INVARIANTS ===
    INVARIANT_VIOLATED = "invariant_violated"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    
    # === JOURNAL ===
    DECISION_LOGGED = "decision_logged"

@dataclass
class Event:
    """Base event structure."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.HEARTBEAT
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"  # Module that generated this event
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1 = highest, 10 = lowest
    
    def __post_init__(self):
        """Ensure payload is always a dict."""
        if self.payload is None:
            self.payload = {}


@dataclass
class MarketEvent(Event):
    """Event containing market data."""
    symbol: str = ""
    price: float = 0.0
    volume: int = 0


@dataclass
class SignalEvent(Event):
    """Event containing a trading signal from a hypothesis."""
    hypothesis_id: str = ""
    symbol: str = ""
    direction: str = ""  # "long", "short", "close"
    strength: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0


@dataclass
class OrderEvent(Event):
    """Event related to order lifecycle."""
    order_id: str = ""
    symbol: str = ""
    side: str = ""  # "buy", "sell"
    quantity: int = 0
    price: Optional[float] = None
    filled_quantity: int = 0
    filled_price: Optional[float] = None


@dataclass
class RiskEvent(Event):
    """Event related to risk alerts."""
    risk_type: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    action_required: str = ""


# Type alias for event handlers
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Central event bus for the entire system.
    
    Implements pub/sub pattern for loose coupling between modules.
    Supports both sync and async handlers.
    """
    
    _instance: Optional['EventBus'] = None
    
    def __new__(cls):
        """Singleton pattern - only one event bus exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._async_handlers: Dict[EventType, List[AsyncEventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._event_history: List[Event] = []
        self._max_history_size: int = 10000
        self._initialized = True
        
        logger.info("EventBus initialized")
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to a specific event type."""
        self._handlers[event_type].append(handler)
        logger.debug(f"Handler subscribed to {event_type.name}")
    
    def subscribe_async(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """Subscribe an async handler to a specific event type."""
        self._async_handlers[event_type].append(handler)
        logger.debug(f"Async handler subscribed to {event_type.name}")
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe a handler to ALL events (useful for logging)."""
        self._global_handlers.append(handler)
        logger.debug("Global handler subscribed")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler from an event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"Handler unsubscribed from {event_type.name}")
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribed handlers.
        
        Handles exceptions gracefully - one failing handler won't affect others.
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
        
        # Notify global handlers first
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Global handler error: {e}")
        
        # Notify type-specific handlers
        for handler in self._handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event.event_type.name}: {e}")
    
    async def publish_async(self, event: Event) -> None:
        """Publish an event and await async handlers."""
        self._event_history.append(event)
        
        # Notify sync handlers
        self.publish(event)
        
        # Notify async handlers
        tasks = []
        for handler in self._async_handlers[event.event_type]:
            tasks.append(asyncio.create_task(handler(event)))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Async handler error: {result}")
    
    def get_recent_events(
        self, 
        event_type: Optional[EventType] = None, 
        limit: int = 100
    ) -> List[Event]:
        """Retrieve recent events, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history (useful for testing)."""
        self._event_history = []
        logger.warning("Event history cleared")


# Convenience function to get the global event bus
def get_event_bus() -> EventBus:
    """Get the singleton EventBus instance."""
    return EventBus()


# === Helper functions for creating common events ===

def emit_price_update(symbol: str, price: float, volume: int, source: str = "market") -> Event:
    """Create and publish a price update event."""
    event = MarketEvent(
        event_type=EventType.PRICE_UPDATE,
        source=source,
        symbol=symbol,
        price=price,
        volume=volume,
        payload={"symbol": symbol, "price": price, "volume": volume}
    )
    get_event_bus().publish(event)
    return event


def emit_signal(
    hypothesis_id: str,
    symbol: str,
    direction: str,
    strength: float,
    confidence: float,
    source: str = "hypothesis"
) -> Event:
    """Create and publish a trading signal event."""
    event = SignalEvent(
        event_type=EventType.HYPOTHESIS_SIGNAL,
        source=source,
        hypothesis_id=hypothesis_id,
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=confidence,
        payload={
            "hypothesis_id": hypothesis_id,
            "symbol": symbol,
            "direction": direction,
            "strength": strength,
            "confidence": confidence
        }
    )
    get_event_bus().publish(event)
    return event


def emit_risk_alert(
    risk_type: str,
    current_value: float,
    threshold: float,
    action_required: str,
    source: str = "risk"
) -> Event:
    """Create and publish a risk alert event."""
    event = RiskEvent(
        event_type=EventType.DRAWDOWN_WARNING,
        source=source,
        risk_type=risk_type,
        current_value=current_value,
        threshold=threshold,
        action_required=action_required,
        priority=1,  # High priority
        payload={
            "risk_type": risk_type,
            "current_value": current_value,
            "threshold": threshold,
            "action_required": action_required
        }
    )
    get_event_bus().publish(event)
    return event


def emit_system_event(event_type: EventType, message: str, **kwargs) -> Event:
    """Create and publish a system event."""
    event = Event(
        event_type=event_type,
        source="system",
        payload={"message": message, **kwargs}
    )
    get_event_bus().publish(event)
    return event
