#!/usr/bin/env python3
"""
KISMUTANT - Living Trading System
Main Entry Point
"""

import sys
import os
import signal
import asyncio
import argparse
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

# Setup logging
Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/system_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="DEBUG"
)


class LivingTradingSystem:
    """Main system controller."""
    
    def __init__(self, mode: str = "paper"):
        self.mode = mode
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Components
        self.broker = None
        self.orchestrator = None
        self.telegram_bot = None
        self.web_server = None
        
        logger.info(f"Initializing KISMUTANT in {mode} mode")
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            # 1. Initialize broker
            logger.info("Initializing broker connection...")
            from execution.broker import KISBroker
            
            self.broker = KISBroker(mode=self.mode)

            # Test connection (will authenticate if needed)
            if not self.broker.test_connection():
                logger.warning("Broker connection test failed - continuing anyway")

            logger.info(f"Broker initialized: {self.mode} mode")
            
            # 2. Initialize database
            logger.info("Initializing database...")
            from memory.models import get_database, get_database_url

            db_url = get_database_url()
            if "postgresql" in db_url:
                logger.info(f"Connecting to PostgreSQL: {db_url.split('@')[-1] if '@' in db_url else 'configured'}")
            else:
                logger.info("Using SQLite database (local)")

            db = get_database()

            # Verify connection
            if db.health_check():
                logger.info(f"Database connection verified: {'PostgreSQL' if db.is_postgres else 'SQLite'}")

                # Log database contents summary
                from memory.models import PriceBar, Instrument, Hypothesis
                from sqlalchemy import func
                session = db.get_session()
                try:
                    bar_count = session.query(func.count(PriceBar.id)).scalar()
                    inst_count = session.query(func.count(Instrument.id)).scalar()
                    hyp_count = session.query(func.count(Hypothesis.id)).scalar()
                    logger.info(f"DB contents: {inst_count} instruments, {bar_count} bars, {hyp_count} hypotheses")
                    if bar_count > 0:
                        latest = session.query(func.max(PriceBar.date)).scalar()
                        earliest = session.query(func.min(PriceBar.date)).scalar()
                        logger.info(f"DB date range: {earliest} to {latest}")
                finally:
                    session.close()
            else:
                logger.error("Database health check failed!")

            # 3. Initialize orchestrator
            logger.info("Initializing orchestrator...")
            from orchestrator import get_orchestrator, OrchestratorConfig
            
            # Get capital from broker or use default
            try:
                balance = self.broker.get_balance()
                initial_capital = balance.get("total_equity", 10_000_000)
            except:
                initial_capital = 10_000_000
            
            config = OrchestratorConfig(
                initial_capital=initial_capital,
                max_live_strategies=3,
                max_paper_strategies=10,
                auto_discover_patterns=True,
                auto_generate_hypotheses=True,
                auto_backtest=True,
                auto_promote=(self.mode == "paper"),  # Auto-promote only in paper mode
                daily_loss_limit_pct=0.02,
                send_telegram_alerts=True
            )
            
            self.orchestrator = get_orchestrator(self.broker, config)
            
            # Initialize orchestrator (loads data, computes features)
            init_summary = await self.orchestrator.initialize()
            logger.info(f"Orchestrator initialized: {init_summary}")
            
            # 4. Initialize Telegram bot
            await self._init_telegram()
            
            # 5. Initialize web dashboard
            self._init_web_dashboard()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _init_telegram(self):
        """Initialize Telegram bot."""
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        if telegram_token and telegram_chat_id:
            try:
                from interface.telegram.bot import TelegramInterface
                self.telegram_bot = TelegramInterface(
                    bot_token=telegram_token,
                    authorized_chat_ids=[int(telegram_chat_id)],
                    system_controller=self
                )
                await self.telegram_bot.start()
                logger.info("Telegram bot started")
            except Exception as e:
                logger.warning(f"Telegram bot failed to start: {e}")
        else:
            logger.info("Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
    
    def _init_web_dashboard(self):
        """Initialize web dashboard in background thread."""
        try:
            from interface.web.dashboard import run_dashboard

            port = int(os.environ.get("PORT", 8080))

            self.web_server = threading.Thread(
                target=run_dashboard,
                kwargs={"host": "0.0.0.0", "port": port},
                daemon=True
            )
            self.web_server.start()
            logger.info(f"Web dashboard started on port {port}")

        except Exception as e:
            logger.warning(f"Web dashboard failed to start: {e}")
    
    async def run(self):
        """Run the main trading loop."""
        self.running = True
        
        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)
        
        logger.info("Starting main trading loop...")
        
        try:
            # Start orchestrator
            orchestrator_task = asyncio.create_task(self.orchestrator.start())
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            # Stop orchestrator
            await self.orchestrator.stop()
            orchestrator_task.cancel()
            
            try:
                await orchestrator_task
            except asyncio.CancelledError:
                pass
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.shutdown()
    
    def _handle_shutdown(self):
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self._shutdown_event.set()
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        
        # Stop Telegram bot
        if self.telegram_bot:
            try:
                await self.telegram_bot.stop()
            except:
                pass
        
        # Disconnect broker
        if self.broker:
            try:
                self.broker.disconnect()
            except:
                pass
        
        logger.info("Shutdown complete")


async def run_web_only():
    """Run only the web dashboard (for testing)."""
    from interface.web.dashboard import app
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def db_check():
    """Check database status and contents."""
    from memory.models import get_database, get_database_url, PriceBar, Instrument, Hypothesis
    from sqlalchemy import func, text

    print("\n=== DATABASE DIAGNOSTIC ===\n")

    # Show which database we're connecting to
    db_url = get_database_url()
    if "postgresql" in db_url:
        safe_url = db_url.split('@')[-1] if '@' in db_url else 'configured'
        print(f"Database type: PostgreSQL")
        print(f"Host/DB: {safe_url}")
    else:
        print(f"Database type: SQLite (local)")
        print(f"Path: {db_url}")

    # Check environment variables
    print("\n--- Environment Variables ---")
    for var in ["DATABASE_URL", "DATABASE_PRIVATE_URL", "POSTGRES_URL", "PGHOST", "PGDATABASE"]:
        value = os.environ.get(var)
        if value:
            # Mask password in URL
            if "@" in str(value):
                safe = value.split("@")[-1]
                print(f"{var}: ***@{safe}")
            else:
                print(f"{var}: {value[:20]}..." if len(str(value)) > 20 else f"{var}: {value}")
        else:
            print(f"{var}: (not set)")

    # Connect and check contents
    print("\n--- Database Contents ---")
    try:
        db = get_database()
        session = db.get_session()

        # First, list all tables in the database using raw SQL
        print("\n--- Tables in Database ---")
        if db.is_postgres:
            result = session.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ))
            tables = [row[0] for row in result]
        else:
            result = session.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ))
            tables = [row[0] for row in result]

        if tables:
            print(f"Found {len(tables)} tables: {', '.join(tables)}")
        else:
            print("NO TABLES FOUND!")

        # Count records in each table using raw SQL
        print("\n--- Row Counts (Raw SQL) ---")
        for table in ['instruments', 'price_bars', 'hypotheses', 'signals', 'trades', 'orders']:
            try:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"{table}: {count} rows")
            except Exception as e:
                print(f"{table}: ERROR - {e}")

        # Count records using ORM
        print("\n--- Row Counts (ORM) ---")
        instrument_count = session.query(func.count(Instrument.id)).scalar()
        bar_count = session.query(func.count(PriceBar.id)).scalar()
        hypothesis_count = session.query(func.count(Hypothesis.id)).scalar()

        print(f"Instruments: {instrument_count}")
        print(f"Price Bars: {bar_count}")
        print(f"Hypotheses: {hypothesis_count}")

        # Show date range if we have bars
        if bar_count > 0:
            earliest = session.query(func.min(PriceBar.date)).scalar()
            latest = session.query(func.max(PriceBar.date)).scalar()
            print(f"Date range: {earliest} to {latest}")

            # Sample some symbols
            sample_instruments = session.query(Instrument).limit(5).all()
            print(f"\nSample instruments: {[i.symbol for i in sample_instruments]}")

        session.close()
        print("\n✓ Database connection successful")

    except Exception as e:
        print(f"\n✗ Database error: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== END DIAGNOSTIC ===\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="KISMUTANT - Living Trading System")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Run only the web dashboard"
    )
    parser.add_argument(
        "--db-check",
        action="store_true",
        help="Check database connection and contents"
    )

    args = parser.parse_args()

    if args.db_check:
        await db_check()
        return

    if args.web_only:
        logger.info("Starting web-only mode...")
        await run_web_only()
        return
    
    # Safety check for live mode
    if args.mode == "live":
        confirm = input("⚠️  LIVE TRADING MODE - Are you sure? Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            logger.info("Live trading cancelled")
            return
    
    # Create and run system
    system = LivingTradingSystem(mode=args.mode)
    
    if await system.initialize():
        await system.run()
    else:
        logger.error("System initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())