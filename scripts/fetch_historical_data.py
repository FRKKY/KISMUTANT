#!/usr/bin/env python3
"""
HISTORICAL DATA FETCHER - Fetch and store all available historical data

This script fetches maximum available historical data from KIS API and stores it
in the database for backtesting and analysis.

KIS API supports:
- Daily data: Up to 756 days (~3 years)
- Intraday data: Up to 30 days

Usage:
    python scripts/fetch_historical_data.py [--symbols SYMBOLS] [--days DAYS] [--intraday]

Options:
    --symbols   Comma-separated list of symbols, or "all" for universe
    --days      Number of days to fetch (max 756 for daily)
    --intraday  Also fetch intraday data (5m, 15m, 60m)
    --backup    Backup to cloud storage after fetching
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


async def fetch_historical_data(
    symbols: Optional[List[str]] = None,
    days: int = 756,
    include_intraday: bool = False,
    backup_to_cloud: bool = False,
    batch_size: int = 5,
    delay_between_batches: float = 3.0,
) -> Dict[str, Any]:
    """
    Fetch historical data for symbols.

    Args:
        symbols: List of symbols to fetch. If None, fetches entire universe.
        days: Number of days of daily data to fetch (max 756)
        include_intraday: Also fetch intraday data
        backup_to_cloud: Backup data to cloud storage after fetching
        batch_size: Number of symbols to fetch in parallel
        delay_between_batches: Delay between batches in seconds

    Returns:
        Summary of fetched data
    """
    from config.loader import get_config
    from execution.broker import KISBroker
    from perception import UniverseFilter, PerceptionLayer
    from memory.models import get_database, Instrument, PriceBar
    from memory.cloud_storage import get_cloud_storage

    # Initialize components
    logger.info("Initializing data fetcher...")

    config = get_config()
    kis_config = config.get_kis_config()

    if not kis_config:
        logger.error("KIS API not configured. Check your credentials.")
        return {"error": "KIS API not configured"}

    broker = KISBroker(
        app_key=kis_config.app_key,
        app_secret=kis_config.app_secret,
        account_number=kis_config.account_number,
        account_product_code=kis_config.account_product_code,
        paper_trading=True,
    )

    db = get_database()

    # Get symbols to fetch
    if not symbols:
        logger.info("Discovering ETF universe...")
        universe_filter = UniverseFilter(
            min_aum=10_000_000_000,
            min_avg_daily_volume=10_000,
            include_leverage=False,
            include_inverse=False,
        )
        # This would need to be implemented or we use a predefined list
        symbols = await discover_symbols(broker, universe_filter)

    logger.info(f"Fetching data for {len(symbols)} symbols, {days} days of history")

    summary = {
        "symbols_requested": len(symbols),
        "symbols_fetched": 0,
        "symbols_failed": 0,
        "total_bars": 0,
        "daily_bars": 0,
        "intraday_bars": 0,
        "start_time": datetime.now().isoformat(),
        "errors": [],
    }

    # Process in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(symbols) + batch_size - 1) // batch_size}: {batch}")

        # Fetch each symbol in batch (sequentially due to rate limits)
        for symbol in batch:
            try:
                # Fetch daily data
                daily_result = await fetch_symbol_daily(broker, db, symbol, days)
                summary["daily_bars"] += daily_result.get("bars", 0)

                # Fetch intraday if requested
                if include_intraday:
                    intraday_result = await fetch_symbol_intraday(broker, db, symbol)
                    summary["intraday_bars"] += intraday_result.get("bars", 0)

                summary["symbols_fetched"] += 1
                summary["total_bars"] = summary["daily_bars"] + summary["intraday_bars"]

                logger.info(f"  {symbol}: {daily_result.get('bars', 0)} daily bars")

            except Exception as e:
                logger.error(f"  {symbol}: Failed - {e}")
                summary["symbols_failed"] += 1
                summary["errors"].append({"symbol": symbol, "error": str(e)})

            # Rate limiting delay between symbols
            await asyncio.sleep(1.0)

        # Delay between batches
        if i + batch_size < len(symbols):
            logger.info(f"Waiting {delay_between_batches}s before next batch...")
            await asyncio.sleep(delay_between_batches)

    summary["end_time"] = datetime.now().isoformat()

    # Backup to cloud if requested
    if backup_to_cloud:
        logger.info("Backing up to cloud storage...")
        try:
            storage = get_cloud_storage()
            backup_key = storage.backup_database()
            summary["backup_key"] = backup_key
        except Exception as e:
            logger.error(f"Cloud backup failed: {e}")
            summary["backup_error"] = str(e)

    # Print summary
    logger.info("=" * 50)
    logger.info("FETCH COMPLETE")
    logger.info(f"  Symbols: {summary['symbols_fetched']}/{summary['symbols_requested']} successful")
    logger.info(f"  Daily bars: {summary['daily_bars']:,}")
    logger.info(f"  Intraday bars: {summary['intraday_bars']:,}")
    logger.info(f"  Total bars: {summary['total_bars']:,}")
    logger.info("=" * 50)

    return summary


async def discover_symbols(broker, universe_filter) -> List[str]:
    """Discover symbols from the ETF universe."""
    # Use predefined seed ETFs as a starting point
    seed_etfs = [
        # Domestic Index ETFs
        "069500",  # KODEX 200
        "102110",  # TIGER 200
        "229200",  # KODEX KOSDAQ150
        "278530",  # KODEX KOSDAQ150 레버리지 -> excluded
        # Sector ETFs
        "091160",  # KODEX 반도체
        "091170",  # KODEX 은행
        "117700",  # KODEX 건설
        "139260",  # TIGER 200 IT
        # Thematic ETFs
        "379810",  # KODEX 2차전지
        "371160",  # TIGER 차이나전기차SOLACTIVE
        "091180",  # KODEX 자동차
        # International ETFs
        "143850",  # TIGER 미국S&P500
        "195930",  # TIGER 미국나스닥100
        "192090",  # TIGER 차이나CSI300
        # Fixed Income
        "152380",  # KODEX 국고채3년
        "148070",  # KOSEF 국고채10년
        # Commodity
        "132030",  # KODEX 골드선물(H)
    ]

    # Filter out leverage/inverse
    filtered = [
        s for s in seed_etfs
        if "레버리지" not in s and "인버스" not in s
    ]

    return filtered[:20]  # Limit for initial testing


async def fetch_symbol_daily(broker, db, symbol: str, days: int) -> Dict[str, Any]:
    """Fetch daily OHLCV data for a symbol."""
    from memory.models import Instrument, PriceBar

    session = db.get_session()

    try:
        # Get or create instrument
        instrument = session.query(Instrument).filter_by(symbol=symbol).first()
        if not instrument:
            instrument = Instrument(
                symbol=symbol,
                name=f"Symbol {symbol}",
                instrument_type="etf",
                market="KOSPI",
            )
            session.add(instrument)
            session.commit()

        # Fetch data from broker
        bars = await broker.get_daily_ohlcv(symbol, days=days)

        if not bars:
            return {"bars": 0, "error": "No data returned"}

        # Store bars
        bars_added = 0
        for bar in bars:
            existing = session.query(PriceBar).filter_by(
                instrument_id=instrument.id,
                date=bar["date"],
                timeframe="1d",
            ).first()

            if not existing:
                price_bar = PriceBar(
                    instrument_id=instrument.id,
                    date=bar["date"],
                    timeframe="1d",
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                )
                session.add(price_bar)
                bars_added += 1

        session.commit()
        return {"bars": bars_added}

    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


async def fetch_symbol_intraday(broker, db, symbol: str) -> Dict[str, Any]:
    """Fetch intraday OHLCV data for a symbol."""
    from memory.models import Instrument, PriceBar

    session = db.get_session()
    total_bars = 0

    try:
        instrument = session.query(Instrument).filter_by(symbol=symbol).first()
        if not instrument:
            return {"bars": 0, "error": "Instrument not found"}

        # Fetch 5-minute, 15-minute, and 60-minute data
        timeframes = [("5m", 5), ("15m", 15), ("60m", 60)]

        for tf_name, tf_minutes in timeframes:
            try:
                bars = await broker.get_intraday_ohlcv(symbol, minutes=tf_minutes, days=30)

                if not bars:
                    continue

                for bar in bars:
                    existing = session.query(PriceBar).filter_by(
                        instrument_id=instrument.id,
                        date=bar["date"],
                        timeframe=tf_name,
                    ).first()

                    if not existing:
                        price_bar = PriceBar(
                            instrument_id=instrument.id,
                            date=bar["date"],
                            timeframe=tf_name,
                            open=bar["open"],
                            high=bar["high"],
                            low=bar["low"],
                            close=bar["close"],
                            volume=bar["volume"],
                        )
                        session.add(price_bar)
                        total_bars += 1

                # Rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"  {symbol} {tf_name}: {e}")

        session.commit()
        return {"bars": total_bars}

    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fetch historical market data")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols, or 'all' for universe",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=756,
        help="Number of days of daily data to fetch (max 756)",
    )
    parser.add_argument(
        "--intraday",
        action="store_true",
        help="Also fetch intraday data",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup to cloud storage after fetching",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of symbols per batch",
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = None
    if args.symbols and args.symbols != "all":
        symbols = [s.strip() for s in args.symbols.split(",")]

    # Run async fetch
    result = asyncio.run(
        fetch_historical_data(
            symbols=symbols,
            days=min(args.days, 756),  # Cap at 756
            include_intraday=args.intraday,
            backup_to_cloud=args.backup,
            batch_size=args.batch_size,
        )
    )

    # Exit with error if failed
    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
