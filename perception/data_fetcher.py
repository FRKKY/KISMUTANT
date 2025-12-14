"""
DATA FETCHER - Orchestrates Market Data Retrieval

Handles:
1. Daily OHLCV data for all universe ETFs
2. Intraday data (1m, 5m, 15m, 30m, 60m)
3. Real-time price updates during market hours
4. Data storage and caching
5. Rate limiting and API management

Data flows:
KIS API → DataFetcher → Database (PriceBar table)
                      → Feature computation
                      → Pattern detection
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from collections import deque, OrderedDict
import statistics
from threading import Lock

from loguru import logger


class BoundedCache:
    """
    Thread-safe LRU cache with maximum size.
    Automatically evicts oldest entries when full.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = Lock()

    def get(self, key):
        """Get item, moving it to end (most recently used)."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key, value):
        """Set item, evicting oldest if at capacity."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Remove oldest

    def __contains__(self, key):
        return key in self._cache

    def __len__(self):
        return len(self._cache)

    def items(self):
        return list(self._cache.items())

    def clear(self):
        with self._lock:
            self._cache.clear()

from memory.models import get_database, Instrument, PriceBar
from core.events import get_event_bus, emit_price_update, Event, EventType
from core.clock import get_clock, is_market_open, KST


class Timeframe(str, Enum):
    """Supported timeframes for data."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    MINUTE_60 = "60m"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


@dataclass
class OHLCVBar:
    """A single OHLCV bar."""
    symbol: str
    timestamp: datetime
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe.value,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap
        }


@dataclass
class MarketSnapshot:
    """Real-time market snapshot for a symbol."""
    symbol: str
    timestamp: datetime
    price: float
    change: float
    change_pct: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None


@dataclass
class DataFetcherConfig:
    """Configuration for data fetcher."""

    # Rate limiting - Conservative limits to avoid KIS API rate limits (EGW00201)
    requests_per_second: float = 1.0  # 1 request per second (conservative)
    batch_size: int = 10              # Smaller batches for safety
    batch_delay_seconds: float = 2.0  # Longer delay between batches
    
    # Data retention
    daily_history_days: int = 756     # ~3 years of daily data
    intraday_history_days: int = 30   # 30 days of intraday data
    
    # Refresh intervals (during market hours)
    price_refresh_seconds: int = 60   # Real-time price refresh
    intraday_refresh_minutes: int = 5 # Intraday bar refresh
    
    # Timeframes to fetch
    intraday_timeframes: List[Timeframe] = field(default_factory=lambda: [
        Timeframe.MINUTE_5,
        Timeframe.MINUTE_15,
        Timeframe.MINUTE_60
    ])


class DataFetcher:
    """
    Orchestrates all market data retrieval.
    
    Responsibilities:
    1. Fetch historical daily data for backtesting
    2. Fetch intraday data for pattern detection
    3. Stream real-time prices during market hours
    4. Store all data in database
    5. Manage API rate limits
    """
    
    def __init__(self, broker=None, config: Optional[DataFetcherConfig] = None):
        """
        Initialize DataFetcher.
        
        Args:
            broker: KIS broker instance
            config: Fetcher configuration
        """
        self._broker = broker
        self._config = config or DataFetcherConfig()
        self._db = get_database()
        self._clock = get_clock()
        
        # Rate limiting
        self._request_times: deque = deque(maxlen=100)
        self._last_request_time = 0

        # Bounded caches to prevent memory growth
        self._price_cache = BoundedCache(max_size=200)      # Max 200 symbols
        self._daily_cache = BoundedCache(max_size=100)      # Max 100 symbols
        self._intraday_cache = BoundedCache(max_size=300)   # Max 300 symbol-timeframe pairs
        
        # Streaming state
        self._streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        
        logger.info("DataFetcher initialized")
    
    async def _rate_limit(self) -> None:
        """Enforce API rate limiting."""
        now = asyncio.get_event_loop().time()
        
        # Calculate time since last request
        if self._request_times:
            # Clean old requests (older than 1 second)
            while self._request_times and now - self._request_times[0] > 1.0:
                self._request_times.popleft()
            
            # If we've made too many requests, wait
            if len(self._request_times) >= self._config.requests_per_second:
                wait_time = 1.0 - (now - self._request_times[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
        
        self._request_times.append(now)
    
    # ===== DAILY DATA =====
    
    async def fetch_daily_history(
        self,
        symbol: str,
        days: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[OHLCVBar]:
        """
        Fetch daily OHLCV history for a symbol.
        
        Args:
            symbol: ETF symbol
            days: Number of days to fetch (alternative to date range)
            start_date: Start date
            end_date: End date (defaults to today)
        """
        if not self._broker:
            logger.warning("No broker available for data fetch")
            return []
        
        # Calculate date range
        end_date = end_date or date.today()
        if days:
            start_date = end_date - timedelta(days=days)
        elif not start_date:
            start_date = end_date - timedelta(days=self._config.daily_history_days)
        
        logger.debug(f"Fetching daily data for {symbol}: {start_date} to {end_date}")
        
        await self._rate_limit()
        
        try:
            raw_data = self._broker.get_daily_ohlcv(
                symbol=symbol,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d")
            )

            if not raw_data:
                logger.warning(f"API returned no data for {symbol}")
                return []

            bars = []
            for item in raw_data:
                bar = OHLCVBar(
                    symbol=symbol,
                    timestamp=datetime.strptime(item["date"], "%Y%m%d"),
                    timeframe=Timeframe.DAILY,
                    open=item["open"],
                    high=item["high"],
                    low=item["low"],
                    close=item["close"],
                    volume=item["volume"]
                )
                bars.append(bar)

            logger.info(f"API FETCH: Got {len(bars)} bars for {symbol}")

            # Cache
            self._daily_cache.set(symbol, bars)

            # Store in database
            await self._store_bars(bars)

            return bars

        except Exception as e:
            logger.error(f"Failed to fetch daily data for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def fetch_daily_history_batch(
        self,
        symbols: List[str],
        days: int = 365
    ) -> Dict[str, List[OHLCVBar]]:
        """
        Fetch daily history for multiple symbols.
        
        Handles batching and rate limiting.
        """
        results = {}
        total = len(symbols)
        
        for i in range(0, total, self._config.batch_size):
            batch = symbols[i:i + self._config.batch_size]
            
            logger.info(f"Fetching daily data batch {i//self._config.batch_size + 1}/{(total + self._config.batch_size - 1)//self._config.batch_size}")
            
            for symbol in batch:
                bars = await self.fetch_daily_history(symbol, days=days)
                if bars:
                    results[symbol] = bars
            
            # Batch delay
            if i + self._config.batch_size < total:
                await asyncio.sleep(self._config.batch_delay_seconds)
        
        logger.info(f"Fetched daily data for {len(results)} symbols")
        return results
    
    # ===== INTRADAY DATA =====
    
    async def fetch_intraday(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.MINUTE_5,
        days: int = 1
    ) -> List[OHLCVBar]:
        """
        Fetch intraday OHLCV data for a symbol.
        
        Note: KIS API has specific endpoints for intraday data.
        """
        if not self._broker:
            logger.warning("No broker available for intraday fetch")
            return []
        
        if timeframe == Timeframe.DAILY:
            return await self.fetch_daily_history(symbol, days=days)
        
        await self._rate_limit()
        
        try:
            # KIS API intraday endpoint
            # TR_ID for intraday: FHKST03010100 (minute data)
            
            # Map timeframe to KIS period code
            period_map = {
                Timeframe.MINUTE_1: "1",
                Timeframe.MINUTE_5: "5",
                Timeframe.MINUTE_15: "15",
                Timeframe.MINUTE_30: "30",
                Timeframe.MINUTE_60: "60"
            }
            
            period = period_map.get(timeframe, "5")
            
            # Note: This is a simplified implementation
            # Full implementation would use the actual KIS intraday endpoint
            
            bars = await self._fetch_intraday_from_api(symbol, period, days)
            
            # Cache
            cache_key = (symbol, timeframe)
            self._intraday_cache.set(cache_key, bars)
            
            # Store in database
            await self._store_bars(bars)
            
            return bars
            
        except Exception as e:
            logger.error(f"Failed to fetch intraday data for {symbol}: {e}")
            return []
    
    async def _fetch_intraday_from_api(
        self,
        symbol: str,
        period: str,
        days: int
    ) -> List[OHLCVBar]:
        """
        Fetch intraday data from KIS API.
        
        KIS API endpoint: /uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice
        """
        if not self._broker:
            return []
        
        bars = []
        
        try:
            # KIS intraday chart endpoint
            tr_id = "FHKST03010100"
            
            # Time format for KIS: HHMMSS
            end_time = "153000"  # Market close
            
            params = {
                "FID_ETC_CLS_CODE": "",
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_HOUR_1": end_time,
                "FID_PW_DATA_INCU_YN": "Y"  # Include previous data
            }
            
            # Make request through broker's internal method
            data = self._broker._request(
                method="GET",
                endpoint="/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                tr_id=tr_id,
                params=params
            )
            
            # Parse response
            for item in data.get("output2", []):
                if not item.get("stck_cntg_hour"):
                    continue
                
                # Parse time (HHMMSS format)
                time_str = item["stck_cntg_hour"]
                today = date.today()
                
                try:
                    bar_time = datetime.combine(
                        today,
                        time(
                            int(time_str[:2]),
                            int(time_str[2:4]),
                            int(time_str[4:6]) if len(time_str) >= 6 else 0
                        )
                    )
                except:
                    continue
                
                # Determine timeframe from period
                timeframe_map = {
                    "1": Timeframe.MINUTE_1,
                    "5": Timeframe.MINUTE_5,
                    "15": Timeframe.MINUTE_15,
                    "30": Timeframe.MINUTE_30,
                    "60": Timeframe.MINUTE_60
                }
                
                bar = OHLCVBar(
                    symbol=symbol,
                    timestamp=bar_time,
                    timeframe=timeframe_map.get(period, Timeframe.MINUTE_5),
                    open=float(item.get("stck_oprc", 0)),
                    high=float(item.get("stck_hgpr", 0)),
                    low=float(item.get("stck_lwpr", 0)),
                    close=float(item.get("stck_prpr", 0)),
                    volume=int(item.get("cntg_vol", 0))
                )
                bars.append(bar)
            
            # Sort by time
            bars.sort(key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.debug(f"Intraday API call failed for {symbol}: {e}")
        
        return bars
    
    async def fetch_intraday_batch(
        self,
        symbols: List[str],
        timeframe: Timeframe = Timeframe.MINUTE_5
    ) -> Dict[str, List[OHLCVBar]]:
        """Fetch intraday data for multiple symbols."""
        results = {}
        
        for i, symbol in enumerate(symbols):
            bars = await self.fetch_intraday(symbol, timeframe)
            if bars:
                results[symbol] = bars
            
            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Intraday fetch progress: {i + 1}/{len(symbols)}")
        
        return results
    
    # ===== REAL-TIME PRICES =====
    
    async def fetch_current_price(self, symbol: str) -> Optional[MarketSnapshot]:
        """Fetch current price for a symbol."""
        if not self._broker:
            return None
        
        await self._rate_limit()
        
        try:
            data = self._broker.get_price(symbol)
            
            if not data or data.get("current_price", 0) == 0:
                return None
            
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=datetime.now(KST),
                price=data["current_price"],
                change=data.get("change", 0),
                change_pct=data.get("change_pct", 0),
                volume=data.get("volume", 0)
            )
            
            # Cache
            self._price_cache.set(symbol, snapshot)
            
            # Emit event
            emit_price_update(
                symbol=symbol,
                price=snapshot.price,
                volume=snapshot.volume,
                source="data_fetcher"
            )
            
            return snapshot
            
        except Exception as e:
            logger.debug(f"Failed to fetch price for {symbol}: {e}")
            return None
    
    async def fetch_prices_batch(
        self,
        symbols: List[str]
    ) -> Dict[str, MarketSnapshot]:
        """Fetch current prices for multiple symbols."""
        results = {}
        
        for symbol in symbols:
            snapshot = await self.fetch_current_price(symbol)
            if snapshot:
                results[symbol] = snapshot
        
        return results
    
    # ===== STREAMING =====
    
    async def start_price_stream(
        self,
        symbols: List[str],
        callback=None
    ) -> None:
        """
        Start streaming prices for symbols during market hours.
        
        Args:
            symbols: Symbols to stream
            callback: Optional callback for price updates
        """
        if self._streaming:
            logger.warning("Price stream already running")
            return
        
        self._streaming = True
        logger.info(f"Starting price stream for {len(symbols)} symbols")
        
        while self._streaming:
            # Only stream during market hours
            if not is_market_open():
                logger.debug("Market closed, pausing price stream")
                await asyncio.sleep(60)
                continue
            
            # Fetch all prices
            snapshots = await self.fetch_prices_batch(symbols)
            
            # Call callback if provided
            if callback:
                for symbol, snapshot in snapshots.items():
                    try:
                        await callback(snapshot)
                    except Exception as e:
                        logger.error(f"Price callback error: {e}")
            
            # Wait for next update
            await asyncio.sleep(self._config.price_refresh_seconds)
        
        logger.info("Price stream stopped")
    
    def stop_price_stream(self) -> None:
        """Stop the price stream."""
        self._streaming = False
    
    # ===== DATA STORAGE =====
    
    async def _store_bars(self, bars: List[OHLCVBar]) -> None:
        """Store OHLCV bars in database."""
        if not bars:
            logger.debug("_store_bars called with empty bars list")
            return

        symbol = bars[0].symbol if bars else "unknown"
        logger.info(f"DB STORE: Starting to store {len(bars)} bars for {symbol}")

        session = self._db.get_session()
        stored_count = 0
        updated_count = 0

        try:
            for bar in bars:
                # Find instrument
                instrument = session.query(Instrument).filter(
                    Instrument.symbol == bar.symbol
                ).first()

                if not instrument:
                    # Create instrument if not exists
                    instrument = Instrument(
                        symbol=bar.symbol,
                        name=bar.symbol,
                        instrument_type="etf",
                        market="KRX",
                        is_tradeable=True  # Mark as tradeable for universe caching
                    )
                    session.add(instrument)
                    session.flush()

                # Check if bar exists
                existing = session.query(PriceBar).filter(
                    PriceBar.instrument_id == instrument.id,
                    PriceBar.date == bar.timestamp.date(),
                    PriceBar.timeframe == bar.timeframe.value
                ).first()

                if existing:
                    # Update
                    existing.open = bar.open
                    existing.high = bar.high
                    existing.low = bar.low
                    existing.close = bar.close
                    existing.volume = bar.volume
                    existing.vwap = bar.vwap
                    updated_count += 1
                else:
                    # Insert
                    price_bar = PriceBar(
                        instrument_id=instrument.id,
                        date=bar.timestamp.date(),
                        timeframe=bar.timeframe.value,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        vwap=bar.vwap
                    )
                    session.add(price_bar)
                    stored_count += 1

            session.commit()
            logger.info(f"DB STORE: Committed {stored_count} new, {updated_count} updated for {symbol}")

            # Verify the data was actually stored
            from sqlalchemy import func
            verify_count = session.query(func.count(PriceBar.id)).filter(
                PriceBar.instrument_id == instrument.id if instrument else -1
            ).scalar()
            logger.info(f"DB VERIFY: {symbol} now has {verify_count} bars in database")

        except Exception as e:
            session.rollback()
            logger.error(f"DB STORE FAILED for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            session.close()

    def _load_bars_from_db(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.DAILY,
        days: int = 365
    ) -> List[OHLCVBar]:
        """
        Load price bars from database.

        Returns bars sorted by date ascending.
        """
        session = self._db.get_session()

        try:
            # Find instrument
            instrument = session.query(Instrument).filter(
                Instrument.symbol == symbol
            ).first()

            if not instrument:
                logger.debug(f"No instrument found for {symbol}")
                return []

            # Calculate date range
            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            # Query price bars
            db_bars = session.query(PriceBar).filter(
                PriceBar.instrument_id == instrument.id,
                PriceBar.timeframe == timeframe.value,
                PriceBar.date >= start_date,
                PriceBar.date <= end_date
            ).order_by(PriceBar.date.asc()).all()

            if not db_bars:
                logger.debug(f"No bars in DB for {symbol} (instrument_id={instrument.id})")
                return []

            # Convert to OHLCVBar objects
            bars = []
            for db_bar in db_bars:
                bar = OHLCVBar(
                    symbol=symbol,
                    timestamp=datetime.combine(db_bar.date, time(0, 0)),
                    timeframe=timeframe,
                    open=db_bar.open,
                    high=db_bar.high,
                    low=db_bar.low,
                    close=db_bar.close,
                    volume=db_bar.volume,
                    vwap=db_bar.vwap
                )
                bars.append(bar)

            logger.debug(f"Loaded {len(bars)} bars from DB for {symbol}")
            return bars

        except Exception as e:
            logger.error(f"Failed to load bars from DB for {symbol}: {e}")
            return []
        finally:
            session.close()

    def get_db_bar_count(self, symbol: str, timeframe: Timeframe = Timeframe.DAILY) -> int:
        """Get count of bars in database for a symbol."""
        session = self._db.get_session()
        try:
            instrument = session.query(Instrument).filter(
                Instrument.symbol == symbol
            ).first()

            if not instrument:
                return 0

            count = session.query(PriceBar).filter(
                PriceBar.instrument_id == instrument.id,
                PriceBar.timeframe == timeframe.value
            ).count()

            return count
        except Exception as e:
            logger.error(f"Failed to count bars: {e}")
            return 0
        finally:
            session.close()

    def get_db_latest_date(self, symbol: str, timeframe: Timeframe = Timeframe.DAILY) -> Optional[date]:
        """Get the most recent date for a symbol in the database."""
        session = self._db.get_session()
        try:
            instrument = session.query(Instrument).filter(
                Instrument.symbol == symbol
            ).first()

            if not instrument:
                return None

            latest = session.query(PriceBar.date).filter(
                PriceBar.instrument_id == instrument.id,
                PriceBar.timeframe == timeframe.value
            ).order_by(PriceBar.date.desc()).first()

            return latest[0] if latest else None
        except Exception as e:
            logger.error(f"Failed to get latest date: {e}")
            return None
        finally:
            session.close()

    # ===== DATA ACCESS =====

    def get_cached_daily(self, symbol: str) -> List[OHLCVBar]:
        """Get cached daily data for a symbol."""
        return self._daily_cache.get(symbol) or []

    def get_cached_intraday(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.MINUTE_5
    ) -> List[OHLCVBar]:
        """Get cached intraday data for a symbol."""
        return self._intraday_cache.get((symbol, timeframe)) or []

    def get_cached_price(self, symbol: str) -> Optional[MarketSnapshot]:
        """Get cached current price for a symbol."""
        return self._price_cache.get(symbol)
    
    async def get_daily_dataframe(
        self,
        symbol: str,
        days: int = 252
    ):
        """
        Get daily data as a pandas DataFrame.

        Data source priority: memory cache → database → API
        Returns DataFrame with columns: date, open, high, low, close, volume
        """
        import pandas as pd

        # 1. Check memory cache first
        bars = self._daily_cache.get(symbol) or []

        if bars and len(bars) >= days * 0.8:
            logger.debug(f"Using cached data for {symbol}: {len(bars)} bars")
        else:
            # 2. Try loading from database
            db_bars = self._load_bars_from_db(symbol, Timeframe.DAILY, days)

            if db_bars and len(db_bars) >= days * 0.5:
                bars = db_bars
                # Update memory cache
                self._daily_cache.set(symbol, bars)
                logger.debug(f"Loaded {symbol} from DB: {len(bars)} bars")

                # Check if we need to fetch recent data (data might be stale)
                latest_date = self.get_db_latest_date(symbol, Timeframe.DAILY)
                if latest_date and (date.today() - latest_date).days > 1:
                    # Fetch only missing recent data
                    logger.debug(f"Fetching recent data for {symbol} since {latest_date}")
                    recent_bars = await self.fetch_daily_history(
                        symbol,
                        start_date=latest_date + timedelta(days=1),
                        end_date=date.today()
                    )
                    if recent_bars:
                        bars.extend(recent_bars)
                        self._daily_cache.set(symbol, bars)
            else:
                # 3. Fetch from API (will also store to DB)
                logger.debug(f"Fetching {symbol} from API")
                bars = await self.fetch_daily_history(symbol, days=days)

        if not bars:
            return pd.DataFrame()

        data = [
            {
                "date": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            }
            for bar in bars
        ]

        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        return df
    
    async def get_intraday_dataframe(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.MINUTE_5
    ):
        """Get intraday data as a pandas DataFrame."""
        import pandas as pd
        
        bars = self._intraday_cache.get((symbol, timeframe)) or []
        
        if not bars:
            bars = await self.fetch_intraday(symbol, timeframe)
        
        if not bars:
            return pd.DataFrame()
        
        data = [
            {
                "datetime": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            }
            for bar in bars
        ]
        
        df = pd.DataFrame(data)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    # ===== BULK OPERATIONS =====
    
    async def initialize_universe_data(
        self,
        symbols: List[str],
        daily_days: int = 365,
        fetch_intraday: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize all data for the universe.
        
        This is called at system startup to populate the database.
        """
        logger.info(f"Initializing data for {len(symbols)} symbols")
        
        results = {
            "daily": {},
            "intraday": {},
            "errors": []
        }
        
        # Fetch daily data
        logger.info("Fetching daily history...")
        results["daily"] = await self.fetch_daily_history_batch(symbols, days=daily_days)
        
        # Fetch intraday data
        if fetch_intraday:
            logger.info("Fetching intraday data...")
            for tf in self._config.intraday_timeframes:
                logger.info(f"Fetching {tf.value} data...")
                tf_data = await self.fetch_intraday_batch(symbols, tf)
                results["intraday"][tf.value] = tf_data
        
        logger.info(
            f"Data initialization complete: "
            f"{len(results['daily'])} daily, "
            f"{sum(len(v) for v in results['intraday'].values())} intraday records"
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data fetcher statistics."""
        return {
            "daily_cache_size": len(self._daily_cache),
            "intraday_cache_size": len(self._intraday_cache),
            "price_cache_size": len(self._price_cache),
            "streaming": self._streaming,
            "config": {
                "requests_per_second": self._config.requests_per_second,
                "batch_size": self._config.batch_size,
                "daily_history_days": self._config.daily_history_days,
                "intraday_history_days": self._config.intraday_history_days
            }
        }

    def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics for price data."""
        from sqlalchemy import func

        session = self._db.get_session()
        try:
            # Count instruments
            instrument_count = session.query(Instrument).count()

            # Count total price bars
            total_bars = session.query(PriceBar).count()

            # Count by timeframe
            daily_bars = session.query(PriceBar).filter(
                PriceBar.timeframe == Timeframe.DAILY.value
            ).count()

            intraday_bars = total_bars - daily_bars

            # Get date range
            oldest = session.query(func.min(PriceBar.date)).scalar()
            newest = session.query(func.max(PriceBar.date)).scalar()

            # Symbols with data
            symbols_with_data = session.query(
                func.count(func.distinct(PriceBar.instrument_id))
            ).scalar() or 0

            return {
                "instruments": instrument_count,
                "total_bars": total_bars,
                "daily_bars": daily_bars,
                "intraday_bars": intraday_bars,
                "symbols_with_data": symbols_with_data,
                "date_range": {
                    "oldest": oldest.isoformat() if oldest else None,
                    "newest": newest.isoformat() if newest else None
                }
            }
        except Exception as e:
            logger.error(f"Failed to get DB stats: {e}")
            return {"error": str(e)}
        finally:
            session.close()


# Singleton instance
_data_fetcher: Optional[DataFetcher] = None


def get_data_fetcher(broker=None, config=None) -> DataFetcher:
    """Get the singleton DataFetcher instance."""
    global _data_fetcher
    if _data_fetcher is None:
        _data_fetcher = DataFetcher(broker, config)
    elif broker and not _data_fetcher._broker:
        _data_fetcher._broker = broker
    return _data_fetcher
