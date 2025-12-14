"""
ALTERNATIVE DATA SOURCES

Provides access to additional data sources beyond the KIS API:
- FinanceDataReader: Years of Korean market historical data
- pykrx: Direct KRX data access

These sources are useful for:
1. Bulk historical data loading (backtesting)
2. Filling gaps when KIS API returns limited data
3. Cross-validation of data accuracy
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import asyncio

from loguru import logger


class FinanceDataReaderSource:
    """
    Data source using FinanceDataReader library.

    FinanceDataReader provides Korean market data from:
    - KRX (Korea Exchange)
    - Naver Finance
    - Yahoo Finance

    Supports years of historical data.
    """

    def __init__(self):
        self._fdr = None
        self._available = False
        self._check_availability()

    def _check_availability(self):
        """Check if FinanceDataReader is installed."""
        try:
            import FinanceDataReader as fdr
            self._fdr = fdr
            self._available = True
            logger.info("FinanceDataReader available for historical data")
        except ImportError:
            logger.warning(
                "FinanceDataReader not installed. "
                "Install with: pip install finance-datareader"
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def get_daily_ohlcv(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        years: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get historical OHLCV data for a Korean stock/ETF.

        Args:
            symbol: Stock/ETF symbol (e.g., '069500' for KODEX 200)
            start_date: Start date (YYYY-MM-DD or YYYYMMDD)
            end_date: End date (YYYY-MM-DD or YYYYMMDD)
            years: Number of years to fetch if start_date not specified

        Returns:
            List of OHLCV bars with keys: date, open, high, low, close, volume
        """
        if not self._available:
            logger.error("FinanceDataReader not available")
            return []

        try:
            # Parse dates
            if end_date:
                if len(end_date) == 8:  # YYYYMMDD format
                    end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            else:
                end_date = date.today().strftime("%Y-%m-%d")

            if start_date:
                if len(start_date) == 8:  # YYYYMMDD format
                    start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            else:
                start_dt = date.today() - timedelta(days=365 * years)
                start_date = start_dt.strftime("%Y-%m-%d")

            logger.info(f"FDR: Fetching {symbol} from {start_date} to {end_date}")

            # Fetch data
            df = self._fdr.DataReader(symbol, start_date, end_date)

            if df is None or df.empty:
                logger.warning(f"FDR: No data returned for {symbol}")
                return []

            # Convert to list of dicts
            bars = []
            for idx, row in df.iterrows():
                bar = {
                    "date": idx.strftime("%Y%m%d"),
                    "open": float(row.get("Open", 0)),
                    "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)),
                    "close": float(row.get("Close", 0)),
                    "volume": int(row.get("Volume", 0)),
                }
                bars.append(bar)

            logger.info(f"FDR: Got {len(bars)} bars for {symbol}")
            return sorted(bars, key=lambda x: x["date"])

        except Exception as e:
            logger.error(f"FDR fetch failed for {symbol}: {e}")
            return []

    def get_etf_list(self) -> List[Dict[str, str]]:
        """Get list of Korean ETFs."""
        if not self._available:
            return []

        try:
            df = self._fdr.StockListing("ETF/KR")
            if df is None or df.empty:
                return []

            etfs = []
            for _, row in df.iterrows():
                etfs.append({
                    "symbol": row.get("Symbol", ""),
                    "name": row.get("Name", ""),
                })
            return etfs
        except Exception as e:
            logger.error(f"Failed to get ETF list: {e}")
            return []


class AlternativeDataManager:
    """
    Manages alternative data sources.

    Provides a unified interface to fetch historical data from
    multiple sources with automatic fallback.
    """

    def __init__(self):
        self.fdr = FinanceDataReaderSource()
        logger.info("AlternativeDataManager initialized")

    async def fetch_historical_data(
        self,
        symbol: str,
        years: int = 3,
        source: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical data, trying multiple sources.

        Args:
            symbol: Stock/ETF symbol
            years: Years of history to fetch
            source: Data source ("fdr", "auto")

        Returns:
            List of OHLCV bars
        """
        if source == "auto" or source == "fdr":
            if self.fdr.is_available:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                bars = await loop.run_in_executor(
                    None,
                    lambda: self.fdr.get_daily_ohlcv(symbol, years=years)
                )
                if bars:
                    return bars

        logger.warning(f"No alternative data source available for {symbol}")
        return []

    async def bulk_fetch_historical(
        self,
        symbols: List[str],
        years: int = 3,
        delay: float = 0.5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of symbols
            years: Years of history
            delay: Delay between requests (rate limiting)

        Returns:
            Dict mapping symbol to bars
        """
        results = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            logger.info(f"Bulk fetch {i+1}/{total}: {symbol}")
            bars = await self.fetch_historical_data(symbol, years=years)
            if bars:
                results[symbol] = bars
            await asyncio.sleep(delay)

        logger.info(f"Bulk fetch complete: {len(results)}/{total} symbols")
        return results


# Singleton instance
_alt_data_manager: Optional[AlternativeDataManager] = None


def get_alternative_data_manager() -> AlternativeDataManager:
    """Get the singleton AlternativeDataManager instance."""
    global _alt_data_manager
    if _alt_data_manager is None:
        _alt_data_manager = AlternativeDataManager()
    return _alt_data_manager
