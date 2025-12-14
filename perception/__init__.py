"""
PERCEPTION LAYER - Market Data Ingestion and Analysis

The Perception Layer is responsible for:
1. Universe Management - Auto-discover and filter tradeable ETFs
2. Data Fetching - Retrieve daily and intraday OHLCV data
3. Feature Engineering - Compute comprehensive technical features
4. Pattern Detection - Identify tradeable patterns for hypothesis generation

Components:
- UniverseManager: Discovers and manages ETF universe
- DataFetcher: Fetches and stores market data
- FeatureEngine: Computes technical indicators and features
- PatternDetector: Identifies tradeable patterns

Usage:
    from perception import PerceptionLayer
    
    perception = PerceptionLayer(broker)
    await perception.initialize()
    await perception.run_daily_update()
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import asyncio

from loguru import logger

from perception.universe import (
    UniverseManager, 
    UniverseFilter, 
    ETFInfo, 
    ETFCategory,
    get_universe_manager
)
from perception.data_fetcher import (
    DataFetcher, 
    DataFetcherConfig, 
    OHLCVBar, 
    MarketSnapshot,
    Timeframe,
    get_data_fetcher
)
from perception.features import (
    FeatureEngine, 
    FeatureConfig, 
    FeatureCategory,
    get_feature_engine
)
from perception.patterns import (
    PatternDetector, 
    PatternConfig, 
    DetectedPattern, 
    PatternType,
    PatternDirection,
    PatternStrength,
    get_pattern_detector
)

from core.events import get_event_bus, Event, EventType


class PerceptionLayer:
    """
    Unified interface to the Perception Layer.
    
    Orchestrates all perception components:
    - Universe discovery and filtering
    - Data fetching and storage
    - Feature computation
    - Pattern detection
    
    The Perception Layer produces:
    - Clean OHLCV data for all universe symbols
    - Rich feature sets for analysis
    - Detected patterns ready for hypothesis generation
    """
    
    def __init__(
        self,
        broker=None,
        universe_filter: Optional[UniverseFilter] = None,
        data_config: Optional[DataFetcherConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
        pattern_config: Optional[PatternConfig] = None
    ):
        """
        Initialize Perception Layer.
        
        Args:
            broker: KIS broker instance for data fetching
            universe_filter: Configuration for universe filtering
            data_config: Configuration for data fetcher
            feature_config: Configuration for feature computation
            pattern_config: Configuration for pattern detection
        """
        # Initialize components
        self.universe = get_universe_manager(broker)
        self.data_fetcher = get_data_fetcher(broker, data_config)
        self.features = get_feature_engine(feature_config)
        self.patterns = get_pattern_detector(pattern_config)
        
        # Store configs
        self._universe_filter = universe_filter or UniverseFilter()
        self._broker = broker
        
        # State
        self._initialized = False
        self._last_update: Optional[datetime] = None
        
        # Cached data
        self._symbol_data: Dict[str, Tuple[Any, Any]] = {}  # symbol -> (ohlcv_df, features_df)
        self._benchmark_data: Optional[Any] = None
        
        logger.info("PerceptionLayer initialized")
    
    async def initialize(
        self,
        daily_history_days: int = 365,
        use_cached_universe: bool = True,
        parallel_fetch: bool = True,
        skip_data_fetch: bool = False,
    ) -> Dict[str, Any]:
        """
        Initialize the perception layer.

        1. Discover ETF universe (or use cached)
        2. Apply filters
        3. Fetch historical data (parallel or from cache)
        4. Compute initial features

        Args:
            daily_history_days: Days of historical data to fetch
            use_cached_universe: Use cached universe from DB instead of discovering
            parallel_fetch: Fetch data in parallel (faster but more API calls)
            skip_data_fetch: Skip fetching, assume data is in database

        Returns:
            Initialization summary
        """
        logger.info("Initializing Perception Layer...")
        summary = {
            "universe_discovery": {},
            "data_fetch": {},
            "features": {},
            "errors": [],
            "optimizations": {
                "use_cached_universe": use_cached_universe,
                "parallel_fetch": parallel_fetch,
                "skip_data_fetch": skip_data_fetch,
            }
        }

        try:
            # Step 1: Get universe (cached or discover)
            if use_cached_universe:
                logger.info("Step 1: Loading cached universe...")
                filtered_symbols = await self._load_cached_universe()
                summary["universe_discovery"]["source"] = "cache"
            else:
                logger.info("Step 1: Discovering ETF universe...")
                etfs = await self.universe.discover_etfs()
                summary["universe_discovery"]["total_discovered"] = len(etfs)
                summary["universe_discovery"]["source"] = "api"

            if not filtered_symbols if use_cached_universe else True:
                # Fall back to discovery if cache empty
                if use_cached_universe:
                    logger.info("Cache empty, falling back to discovery...")
                    etfs = await self.universe.discover_etfs()
                    summary["universe_discovery"]["total_discovered"] = len(etfs)
                    summary["universe_discovery"]["source"] = "api_fallback"

                # Step 2: Apply filter
                logger.info("Step 2: Applying universe filter...")
                self.universe.set_filter(self._universe_filter)
                filtered_symbols = self.universe.apply_filter()
                await self._cache_universe(filtered_symbols)

            summary["universe_discovery"]["filtered_count"] = len(filtered_symbols)
            summary["universe_discovery"]["by_category"] = self.universe.get_universe_by_category()

            if not filtered_symbols:
                logger.warning("No symbols passed filter!")
                return summary

            # Step 3: Fetch benchmark data first
            logger.info("Step 3: Fetching benchmark data...")
            benchmark_symbol = self.features.config.benchmark_symbol
            self._benchmark_data = await self.data_fetcher.get_daily_dataframe(
                benchmark_symbol,
                days=daily_history_days
            )
            if self._benchmark_data is not None and not self._benchmark_data.empty:
                self.features.set_benchmark_data(self._benchmark_data)
                logger.info(f"Benchmark data loaded: {len(self._benchmark_data)} bars")

            # Step 4: Fetch historical data for universe
            if skip_data_fetch:
                logger.info("Step 4: Skipping data fetch (using cached data)...")
                summary["data_fetch"]["source"] = "cache"
                summary["data_fetch"]["daily_symbols"] = len(filtered_symbols)
            elif parallel_fetch:
                logger.info(f"Step 4: Fetching daily data for {len(filtered_symbols)} symbols (parallel)...")
                daily_data = await self._fetch_data_parallel(
                    filtered_symbols,
                    days=daily_history_days
                )
                summary["data_fetch"]["source"] = "api_parallel"
                summary["data_fetch"]["daily_symbols"] = len(daily_data)
                summary["data_fetch"]["daily_bars"] = sum(len(bars) for bars in daily_data.values())
            else:
                logger.info(f"Step 4: Fetching daily data for {len(filtered_symbols)} symbols...")
                daily_data = await self.data_fetcher.fetch_daily_history_batch(
                    filtered_symbols,
                    days=daily_history_days
                )
                summary["data_fetch"]["source"] = "api_sequential"
                summary["data_fetch"]["daily_symbols"] = len(daily_data)
                summary["data_fetch"]["daily_bars"] = sum(len(bars) for bars in daily_data.values())
            
            # Step 5: Compute features
            logger.info("Step 5: Computing features...")
            feature_count = 0
            for symbol in filtered_symbols:
                try:
                    ohlcv_df = await self.data_fetcher.get_daily_dataframe(symbol)
                    if ohlcv_df.empty:
                        continue
                    
                    features_df = self.features.compute_all(
                        ohlcv_df,
                        benchmark_df=self._benchmark_data
                    )
                    
                    self._symbol_data[symbol] = (ohlcv_df, features_df)
                    feature_count += 1
                    
                except Exception as e:
                    logger.debug(f"Feature computation failed for {symbol}: {e}")
                    summary["errors"].append(f"{symbol}: {str(e)}")
            
            summary["features"]["symbols_processed"] = feature_count
            summary["features"]["feature_names"] = self.features.get_feature_names()[:20]  # Sample
            
            self._initialized = True
            self._last_update = datetime.now()
            
            # Emit initialization complete event
            get_event_bus().publish(Event(
                event_type=EventType.SYSTEM_STARTUP,
                source="perception_layer",
                payload={
                    "message": "Perception Layer initialized",
                    "summary": summary
                }
            ))
            
            logger.info(
                f"Perception Layer initialized: "
                f"{len(filtered_symbols)} symbols, "
                f"{feature_count} with features"
            )
            
        except Exception as e:
            logger.error(f"Perception initialization failed: {e}")
            summary["errors"].append(str(e))
        
        return summary
    
    async def run_daily_update(self) -> Dict[str, Any]:
        """
        Run daily data update and pattern detection.
        
        Call this at market close to update all data.
        """
        if not self._initialized:
            logger.warning("Perception Layer not initialized")
            return {"error": "not_initialized"}
        
        logger.info("Running daily perception update...")
        summary = {
            "data_updated": 0,
            "features_updated": 0,
            "patterns_detected": 0,
            "patterns": []
        }
        
        universe = self.universe.get_universe()
        
        for symbol in universe:
            try:
                # Fetch latest data
                ohlcv_df = await self.data_fetcher.get_daily_dataframe(symbol, days=60)
                if ohlcv_df.empty:
                    continue
                
                summary["data_updated"] += 1
                
                # Compute features
                features_df = self.features.compute_all(
                    ohlcv_df,
                    benchmark_df=self._benchmark_data
                )
                
                self._symbol_data[symbol] = (ohlcv_df, features_df)
                summary["features_updated"] += 1
                
                # Detect patterns
                patterns = self.patterns.scan(symbol, ohlcv_df, features_df)
                summary["patterns_detected"] += len(patterns)
                
                for p in patterns:
                    summary["patterns"].append({
                        "symbol": p.symbol,
                        "type": p.pattern_type.value,
                        "direction": p.direction.value,
                        "confidence": p.confidence
                    })
                
            except Exception as e:
                logger.debug(f"Update failed for {symbol}: {e}")
        
        # Clean up expired patterns
        expired = self.patterns.cleanup_expired()
        summary["patterns_expired"] = expired
        
        self._last_update = datetime.now()
        
        logger.info(
            f"Daily update complete: "
            f"{summary['data_updated']} updated, "
            f"{summary['patterns_detected']} patterns"
        )
        
        return summary
    
    async def run_intraday_scan(
        self,
        timeframe: Timeframe = Timeframe.MINUTE_5
    ) -> Dict[str, Any]:
        """
        Run intraday data update and pattern detection.
        
        Call this during market hours for intraday analysis.
        """
        if not self._initialized:
            return {"error": "not_initialized"}
        
        logger.info(f"Running intraday scan ({timeframe.value})...")
        summary = {
            "symbols_scanned": 0,
            "patterns_detected": 0,
            "patterns": []
        }
        
        universe = self.universe.get_universe()
        
        for symbol in universe:
            try:
                # Fetch intraday data
                intraday_df = await self.data_fetcher.get_intraday_dataframe(
                    symbol, 
                    timeframe
                )
                
                if intraday_df.empty or len(intraday_df) < 20:
                    continue
                
                summary["symbols_scanned"] += 1
                
                # Compute features on intraday data
                features_df = self.features.compute_all(intraday_df)
                
                # Detect patterns
                patterns = self.patterns.scan(symbol, intraday_df, features_df)
                summary["patterns_detected"] += len(patterns)
                
                for p in patterns:
                    summary["patterns"].append({
                        "symbol": p.symbol,
                        "type": p.pattern_type.value,
                        "direction": p.direction.value,
                        "confidence": p.confidence
                    })
                
            except Exception as e:
                logger.debug(f"Intraday scan failed for {symbol}: {e}")
        
        return summary
    
    def get_symbol_data(
        self, 
        symbol: str
    ) -> Optional[Tuple[Any, Any]]:
        """
        Get cached data for a symbol.
        
        Returns:
            Tuple of (ohlcv_dataframe, features_dataframe) or None
        """
        return self._symbol_data.get(symbol)
    
    def get_all_active_patterns(
        self,
        min_confidence: float = 0.5
    ) -> List[DetectedPattern]:
        """Get all active patterns across universe."""
        return self.patterns.get_active_patterns(min_confidence=min_confidence)
    
    def get_patterns_by_direction(
        self,
        direction: PatternDirection
    ) -> List[DetectedPattern]:
        """Get patterns filtered by direction."""
        return self.patterns.get_active_patterns(direction=direction)
    
    def get_universe(self) -> List[str]:
        """Get current universe symbols."""
        return self.universe.get_universe()
    
    def get_universe_by_category(self) -> Dict[ETFCategory, List[str]]:
        """Get universe grouped by category."""
        return self.universe.get_universe_by_category()
    
    def get_etf_info(self, symbol: str) -> Optional[ETFInfo]:
        """Get ETF information."""
        return self.universe.get_etf_info(symbol)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get perception layer statistics."""
        return {
            "initialized": self._initialized,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "universe": self.universe.get_stats(),
            "data_fetcher": self.data_fetcher.get_stats(),
            "patterns": self.patterns.get_stats(),
            "cached_symbols": len(self._symbol_data)
        }

    # ===== CACHING AND PERFORMANCE HELPERS =====

    async def _load_cached_universe(self) -> List[str]:
        """
        Load universe from database cache.

        Returns cached symbols if available, empty list otherwise.
        """
        try:
            from memory.models import get_database, Instrument

            db = get_database()
            session = db.get_session()

            # Get tradeable instruments from database
            instruments = session.query(Instrument).filter(
                Instrument.is_tradeable == True,
                Instrument.instrument_type.in_(["etf", "stock"])
            ).all()

            symbols = [inst.symbol for inst in instruments]
            session.close()

            if symbols:
                logger.info(f"Loaded {len(symbols)} symbols from cache")
                # Also update universe manager
                for symbol in symbols:
                    self.universe._universe.add(symbol)
            return symbols

        except Exception as e:
            logger.warning(f"Failed to load cached universe: {e}")
            return []

    async def _cache_universe(self, symbols: List[str]) -> None:
        """
        Cache universe symbols to database.

        Stores symbols for faster startup on next run.
        """
        try:
            from memory.models import get_database, Instrument

            db = get_database()
            session = db.get_session()

            for symbol in symbols:
                # Check if exists
                existing = session.query(Instrument).filter_by(symbol=symbol).first()
                if not existing:
                    instrument = Instrument(
                        symbol=symbol,
                        name=f"Symbol {symbol}",
                        instrument_type="etf",
                        is_tradeable=True,
                    )
                    session.add(instrument)

            session.commit()
            session.close()
            logger.info(f"Cached {len(symbols)} symbols to database")

        except Exception as e:
            logger.warning(f"Failed to cache universe: {e}")

    async def _fetch_data_parallel(
        self,
        symbols: List[str],
        days: int = 365,
        max_concurrent: int = 5,
    ) -> Dict[str, List]:
        """
        Fetch data for multiple symbols in parallel.

        Uses a semaphore to limit concurrent requests and respect rate limits.

        Args:
            symbols: Symbols to fetch
            days: Days of history
            max_concurrent: Max concurrent fetches

        Returns:
            Dict of symbol -> bars
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_one(symbol: str) -> tuple:
            async with semaphore:
                try:
                    bars = await self.data_fetcher.fetch_daily_history(symbol, days)
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.5)
                    return symbol, bars
                except Exception as e:
                    logger.debug(f"Parallel fetch failed for {symbol}: {e}")
                    return symbol, []

        # Create tasks for all symbols
        tasks = [fetch_one(symbol) for symbol in symbols]

        # Run in parallel
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in completed:
            if isinstance(result, tuple):
                symbol, bars = result
                if bars:
                    results[symbol] = bars

        logger.info(f"Parallel fetch complete: {len(results)}/{len(symbols)} successful")
        return results


# Module-level exports
__all__ = [
    # Main class
    'PerceptionLayer',
    
    # Universe
    'UniverseManager',
    'UniverseFilter',
    'ETFInfo',
    'ETFCategory',
    'get_universe_manager',
    
    # Data
    'DataFetcher',
    'DataFetcherConfig',
    'OHLCVBar',
    'MarketSnapshot',
    'Timeframe',
    'get_data_fetcher',
    
    # Features
    'FeatureEngine',
    'FeatureConfig',
    'FeatureCategory',
    'get_feature_engine',
    
    # Patterns
    'PatternDetector',
    'PatternConfig',
    'DetectedPattern',
    'PatternType',
    'PatternDirection',
    'PatternStrength',
    'get_pattern_detector',
]
