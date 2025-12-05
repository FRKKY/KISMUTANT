"""
CLOCK - Market Time Awareness

The system needs to understand:
- When the market is open/closed
- What trading session we're in
- How to handle holidays and half-days
- Time-based triggers for various operations
"""

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Optional, List, Tuple
from enum import Enum, auto
from zoneinfo import ZoneInfo
import asyncio
from loguru import logger

from core.events import get_event_bus, emit_system_event, EventType


# Korean timezone
KST = ZoneInfo("Asia/Seoul")


class MarketSession(Enum):
    """Market session states."""
    PRE_MARKET = auto()      # Before market open
    MARKET_OPEN = auto()     # Regular trading hours
    LUNCH_BREAK = auto()     # KRX doesn't have this, but keeping for flexibility
    MARKET_CLOSE = auto()    # After regular hours
    AFTER_HOURS = auto()     # Extended hours (if applicable)
    WEEKEND = auto()         # Saturday/Sunday
    HOLIDAY = auto()         # Market holiday


@dataclass
class TradingHours:
    """Definition of trading hours for a market."""
    market_name: str
    timezone: ZoneInfo
    open_time: time
    close_time: time
    pre_market_start: Optional[time] = None
    after_hours_end: Optional[time] = None


# KRX (Korea Exchange) trading hours
KRX_HOURS = TradingHours(
    market_name="KRX",
    timezone=KST,
    open_time=time(9, 0),      # 09:00 KST
    close_time=time(15, 30),   # 15:30 KST
    pre_market_start=time(8, 0),
    after_hours_end=time(18, 0)
)


class MarketClock:
    """
    Centralized market time awareness.
    
    Knows when markets are open, handles holidays, and provides
    time-based coordination for the entire system.
    """
    
    _instance: Optional['MarketClock'] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.trading_hours = KRX_HOURS
        self._holidays: List[date] = []
        self._half_days: List[date] = []
        self._callbacks: List[Tuple[time, callable]] = []
        self._running = False
        self._initialized = True
        
        # Load Korean market holidays
        self._load_holidays()
        
        logger.info(f"MarketClock initialized for {self.trading_hours.market_name}")
    
    def _load_holidays(self) -> None:
        """
        Load KRX holidays.
        
        In production, this would fetch from an API or config file.
        For now, includes 2024-2025 major Korean holidays.
        """
        # 2024-2025 KRX holidays (sample - should be updated from official source)
        self._holidays = [
            # 2024
            date(2024, 1, 1),   # New Year
            date(2024, 2, 9),   # Lunar New Year
            date(2024, 2, 10),
            date(2024, 2, 11),
            date(2024, 2, 12),
            date(2024, 3, 1),   # Independence Movement Day
            date(2024, 4, 10),  # Election Day
            date(2024, 5, 5),   # Children's Day
            date(2024, 5, 6),   # Substitute holiday
            date(2024, 5, 15),  # Buddha's Birthday
            date(2024, 6, 6),   # Memorial Day
            date(2024, 8, 15),  # Liberation Day
            date(2024, 9, 16),  # Chuseok
            date(2024, 9, 17),
            date(2024, 9, 18),
            date(2024, 10, 3),  # National Foundation Day
            date(2024, 10, 9),  # Hangul Day
            date(2024, 12, 25), # Christmas
            date(2024, 12, 31), # Year end
            # 2025
            date(2025, 1, 1),   # New Year
            date(2025, 1, 28),  # Lunar New Year
            date(2025, 1, 29),
            date(2025, 1, 30),
            date(2025, 3, 1),   # Independence Movement Day
            date(2025, 5, 5),   # Children's Day
            date(2025, 5, 6),   # Buddha's Birthday
            date(2025, 6, 6),   # Memorial Day
            date(2025, 8, 15),  # Liberation Day
            date(2025, 10, 3),  # National Foundation Day
            date(2025, 10, 5),  # Chuseok
            date(2025, 10, 6),
            date(2025, 10, 7),
            date(2025, 10, 8),  # Substitute
            date(2025, 10, 9),  # Hangul Day
            date(2025, 12, 25), # Christmas
        ]
        
        logger.info(f"Loaded {len(self._holidays)} market holidays")
    
    def now(self) -> datetime:
        """Get current time in market timezone."""
        return datetime.now(self.trading_hours.timezone)
    
    def today(self) -> date:
        """Get current date in market timezone."""
        return self.now().date()
    
    def is_holiday(self, d: Optional[date] = None) -> bool:
        """Check if a date is a market holiday."""
        d = d or self.today()
        return d in self._holidays
    
    def is_weekend(self, d: Optional[date] = None) -> bool:
        """Check if a date is a weekend."""
        d = d or self.today()
        return d.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    def is_trading_day(self, d: Optional[date] = None) -> bool:
        """Check if a date is a trading day."""
        d = d or self.today()
        return not self.is_weekend(d) and not self.is_holiday(d)
    
    def get_session(self, dt: Optional[datetime] = None) -> MarketSession:
        """Determine the current market session."""
        dt = dt or self.now()
        d = dt.date()
        t = dt.time()
        
        if self.is_weekend(d):
            return MarketSession.WEEKEND
        
        if self.is_holiday(d):
            return MarketSession.HOLIDAY
        
        hours = self.trading_hours
        
        if hours.pre_market_start and t < hours.pre_market_start:
            return MarketSession.MARKET_CLOSE
        elif hours.pre_market_start and t < hours.open_time:
            return MarketSession.PRE_MARKET
        elif t < hours.close_time:
            return MarketSession.MARKET_OPEN
        elif hours.after_hours_end and t < hours.after_hours_end:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.MARKET_CLOSE
    
    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if the market is currently open for trading."""
        return self.get_session(dt) == MarketSession.MARKET_OPEN
    
    def time_to_open(self, dt: Optional[datetime] = None) -> Optional[timedelta]:
        """Get time until market opens. Returns None if market is open or it's not a trading day."""
        dt = dt or self.now()
        
        if not self.is_trading_day(dt.date()):
            # Find next trading day
            next_day = self.next_trading_day(dt.date())
            next_open = datetime.combine(next_day, self.trading_hours.open_time, tzinfo=KST)
            return next_open - dt
        
        if self.is_market_open(dt):
            return None
        
        if dt.time() < self.trading_hours.open_time:
            open_dt = datetime.combine(dt.date(), self.trading_hours.open_time, tzinfo=KST)
            return open_dt - dt
        
        # Market closed for today, find next trading day
        next_day = self.next_trading_day(dt.date())
        next_open = datetime.combine(next_day, self.trading_hours.open_time, tzinfo=KST)
        return next_open - dt
    
    def time_to_close(self, dt: Optional[datetime] = None) -> Optional[timedelta]:
        """Get time until market closes. Returns None if market is closed."""
        dt = dt or self.now()
        
        if not self.is_market_open(dt):
            return None
        
        close_dt = datetime.combine(dt.date(), self.trading_hours.close_time, tzinfo=KST)
        return close_dt - dt
    
    def next_trading_day(self, d: Optional[date] = None) -> date:
        """Get the next trading day after the given date."""
        d = d or self.today()
        next_d = d + timedelta(days=1)
        
        while not self.is_trading_day(next_d):
            next_d += timedelta(days=1)
            # Safety limit
            if (next_d - d).days > 30:
                raise RuntimeError("Could not find trading day within 30 days")
        
        return next_d
    
    def previous_trading_day(self, d: Optional[date] = None) -> date:
        """Get the previous trading day before the given date."""
        d = d or self.today()
        prev_d = d - timedelta(days=1)
        
        while not self.is_trading_day(prev_d):
            prev_d -= timedelta(days=1)
            if (d - prev_d).days > 30:
                raise RuntimeError("Could not find trading day within 30 days")
        
        return prev_d
    
    def trading_days_between(self, start: date, end: date) -> int:
        """Count trading days between two dates."""
        count = 0
        current = start
        while current <= end:
            if self.is_trading_day(current):
                count += 1
            current += timedelta(days=1)
        return count
    
    def schedule_at(self, target_time: time, callback: callable) -> None:
        """Schedule a callback to run at a specific time each trading day."""
        self._callbacks.append((target_time, callback))
        logger.info(f"Scheduled callback at {target_time}")
    
    async def run_scheduler(self) -> None:
        """
        Run the time-based scheduler.
        
        This should be started as a background task.
        """
        self._running = True
        logger.info("Clock scheduler started")
        
        last_session = None
        
        while self._running:
            now = self.now()
            current_session = self.get_session(now)
            
            # Emit events on session change
            if last_session != current_session:
                if current_session == MarketSession.MARKET_OPEN:
                    emit_system_event(EventType.MARKET_OPEN, "Market is now open")
                elif current_session == MarketSession.MARKET_CLOSE and last_session == MarketSession.MARKET_OPEN:
                    emit_system_event(EventType.MARKET_CLOSE, "Market is now closed")
                last_session = current_session
            
            # Check scheduled callbacks
            for target_time, callback in self._callbacks:
                # Check if we're within 1 second of target time
                target_dt = datetime.combine(now.date(), target_time, tzinfo=KST)
                if abs((now - target_dt).total_seconds()) < 1:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback()
                        else:
                            callback()
                    except Exception as e:
                        logger.error(f"Scheduled callback error: {e}")
            
            # Sleep for a short interval
            await asyncio.sleep(0.5)
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        logger.info("Clock scheduler stopped")


def get_clock() -> MarketClock:
    """Get the singleton MarketClock instance."""
    return MarketClock()


# === Convenience functions ===

def is_market_open() -> bool:
    """Check if market is currently open."""
    return get_clock().is_market_open()


def now_kst() -> datetime:
    """Get current time in KST."""
    return get_clock().now()


def today_kst() -> date:
    """Get current date in KST."""
    return get_clock().today()
