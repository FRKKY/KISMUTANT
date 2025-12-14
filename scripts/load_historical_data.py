#!/usr/bin/env python3
"""
HISTORICAL DATA LOADER

Loads years of historical data using FinanceDataReader and stores it in the database.

Usage:
    python scripts/load_historical_data.py [--years 3] [--symbols SYMBOL1,SYMBOL2]

This script:
1. Fetches historical data from FinanceDataReader (KRX/Naver sources)
2. Stores it in the PostgreSQL/SQLite database
3. Can be run once to bootstrap the database with historical data

Requirements:
    pip install finance-datareader
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level: <8} | {message}", level="INFO")


async def load_historical_data(
    symbols: list = None,
    years: int = 3,
    use_kis: bool = False
):
    """
    Load historical data for symbols.

    Args:
        symbols: List of symbols (or None for all universe)
        years: Years of history to load
        use_kis: Try KIS API first (with new pagination)
    """
    from memory.models import get_database, PriceBar, Instrument
    from perception.alternative_data import get_alternative_data_manager
    from perception.data_fetcher import OHLCVBar, Timeframe

    db = get_database()
    alt_data = get_alternative_data_manager()

    # Get symbols from database if not specified
    if not symbols:
        session = db.get_session()
        instruments = session.query(Instrument).filter(
            Instrument.is_tradeable == True
        ).all()
        symbols = [i.symbol for i in instruments]
        session.close()

        if not symbols:
            logger.warning("No symbols found. Using default ETF list.")
            # Default popular Korean ETFs
            symbols = [
                "069500",  # KODEX 200
                "102110",  # TIGER 200
                "148020",  # KINDEX 레버리지
                "252670",  # KODEX 200선물인버스2X
                "091160",  # KODEX 반도체
                "091170",  # KODEX 은행
                "091180",  # KODEX 자동차
                "139260",  # TIGER 200 IT
                "139270",  # TIGER 200 금융
            ]

    logger.info(f"Loading {years} years of history for {len(symbols)} symbols")

    # Check if FinanceDataReader is available
    if not alt_data.fdr.is_available:
        logger.error(
            "FinanceDataReader not installed!\n"
            "Install with: pip install finance-datareader"
        )
        return

    session = db.get_session()
    total_bars_stored = 0
    successful_symbols = 0

    try:
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Fetching {symbol}...")

            # Fetch from FinanceDataReader
            bars = alt_data.fdr.get_daily_ohlcv(symbol, years=years)

            if not bars:
                logger.warning(f"No data for {symbol}")
                continue

            # Get or create instrument
            instrument = session.query(Instrument).filter_by(symbol=symbol).first()
            if not instrument:
                instrument = Instrument(
                    symbol=symbol,
                    name=f"Symbol {symbol}",
                    instrument_type="etf",
                    is_tradeable=True,
                    market="KRX"
                )
                session.add(instrument)
                session.flush()

            # Store bars
            stored = 0
            updated = 0
            for bar in bars:
                bar_date = datetime.strptime(bar["date"], "%Y%m%d").date()

                existing = session.query(PriceBar).filter_by(
                    instrument_id=instrument.id,
                    date=bar_date,
                    timeframe="1d"
                ).first()

                if existing:
                    # Update existing
                    existing.open = bar["open"]
                    existing.high = bar["high"]
                    existing.low = bar["low"]
                    existing.close = bar["close"]
                    existing.volume = bar["volume"]
                    updated += 1
                else:
                    # Insert new
                    price_bar = PriceBar(
                        instrument_id=instrument.id,
                        date=bar_date,
                        timeframe="1d",
                        open=bar["open"],
                        high=bar["high"],
                        low=bar["low"],
                        close=bar["close"],
                        volume=bar["volume"]
                    )
                    session.add(price_bar)
                    stored += 1

            session.commit()
            total_bars_stored += stored
            successful_symbols += 1
            logger.info(f"  {symbol}: {stored} new, {updated} updated (total: {len(bars)} bars)")

            # Rate limiting
            await asyncio.sleep(0.3)

    except Exception as e:
        session.rollback()
        logger.error(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

    logger.info(f"\n=== COMPLETE ===")
    logger.info(f"Symbols processed: {successful_symbols}/{len(symbols)}")
    logger.info(f"New bars stored: {total_bars_stored}")

    # Show database stats
    session = db.get_session()
    from sqlalchemy import func
    bar_count = session.query(func.count(PriceBar.id)).scalar()
    min_date = session.query(func.min(PriceBar.date)).scalar()
    max_date = session.query(func.max(PriceBar.date)).scalar()
    session.close()

    logger.info(f"Total bars in database: {bar_count}")
    logger.info(f"Date range: {min_date} to {max_date}")


async def main():
    parser = argparse.ArgumentParser(description="Load historical data into database")
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Years of history to load (default: 3)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: all in database)"
    )
    parser.add_argument(
        "--use-kis",
        action="store_true",
        help="Try KIS API first with pagination"
    )

    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None

    await load_historical_data(
        symbols=symbols,
        years=args.years,
        use_kis=args.use_kis
    )


if __name__ == "__main__":
    asyncio.run(main())
