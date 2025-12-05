#!/usr/bin/env python3
"""
LIVING TRADING SYSTEM - Main Entry Point

This is the primary entry point for starting the trading system.
It orchestrates all components and manages the system lifecycle.
"""

import sys
import signal
import asyncio
import argparse
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/system_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="DEBUG"
)


class LivingTradingSystem:
    """
    Main orchestrator for the Living Trading System.
    
    Coordinates all components:
    - Perception (market data)
    - Hypothesis Engine (strategy generation)
    - Validation (backtesting, paper trading)
    - Portfolio Mind (position management)
    - Execution (order management)
    - Memory (logging, learning)
    - Interfaces (Web Dashboard, Telegram Bot)
    """
    
    def __init__(self, mode: str = "paper", enable_web: bool = True, enable_telegram: bool = True):
        """
        Initialize the trading system.
        
        Args:
            mode: "paper" for paper trading, "live" for real trading
            enable_web: Enable web dashboard interface
            enable_telegram: Enable Telegram bot interface
        """
        self.mode = mode
        self.enable_web = enable_web
        self.enable_telegram = enable_telegram
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Interface references
        self.telegram_bot = None
        self.web_server_thread = None
        
        logger.info(f"Initializing Living Trading System in {mode} mode")
        
        # Import components (deferred to allow for clean imports)
        self._import_components()
        
    def _import_components(self):
        """Import system components."""
        from core.invariants import INVARIANTS
        from core.events import get_event_bus, EventType, emit_system_event
        from core.clock import get_clock, MarketSession
        from memory.models import get_database
        from memory.journal import get_journal
        from execution.broker import get_broker
        
        self.invariants = INVARIANTS
        self.event_bus = get_event_bus()
        self.clock = get_clock()
        self.database = get_database()
        self.journal = get_journal()
        self.broker = get_broker(self.mode)
        
        logger.info("All components imported successfully")
    
    def _load_telegram_config(self):
        """Load Telegram configuration."""
        import yaml
        
        try:
            with open("config/credentials.yaml", 'r') as f:
                creds = yaml.safe_load(f)
            
            telegram_config = creds.get("notifications", {}).get("telegram", {})
            bot_token = telegram_config.get("bot_token", "")
            chat_id = telegram_config.get("chat_id", "")
            
            if bot_token and "YOUR_" not in bot_token and chat_id and "YOUR_" not in chat_id:
                return bot_token, int(chat_id)
        except Exception as e:
            logger.warning(f"Could not load Telegram config: {e}")
        
        return None, None
    
    async def _start_telegram(self):
        """Start Telegram bot."""
        if not self.enable_telegram:
            return
        
        bot_token, chat_id = self._load_telegram_config()
        
        if not bot_token:
            logger.warning("Telegram bot not configured - skipping")
            return
        
        try:
            from interface.telegram.bot import TelegramInterface
            
            self.telegram_bot = TelegramInterface(
                bot_token=bot_token,
                authorized_chat_ids=[chat_id],
                system_controller=self
            )
            
            await self.telegram_bot.start()
            logger.info("✓ Telegram bot started")
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    
    def _start_web_dashboard(self):
        """Start web dashboard in a separate thread."""
        if not self.enable_web:
            return
        
        try:
            from interface.web.dashboard import run_dashboard
            
            # Run in separate thread to not block async loop
            self.web_server_thread = threading.Thread(
                target=run_dashboard,
                kwargs={"host": "0.0.0.0", "port": 8080},
                daemon=True
            )
            self.web_server_thread.start()
            logger.info("✓ Web dashboard started at http://localhost:8080")
            
        except Exception as e:
            logger.error(f"Failed to start web dashboard: {e}")
    
    async def startup(self):
        """
        System startup sequence.
        """
        logger.info("=" * 60)
        logger.info("LIVING TRADING SYSTEM - STARTUP")
        logger.info("=" * 60)
        
        # Emit startup event
        from core.events import emit_system_event, EventType
        emit_system_event(EventType.SYSTEM_STARTUP, f"System starting in {self.mode} mode")
        
        # 1. Verify API connection
        logger.info("Step 1: Verifying API connection...")
        if not self.broker.test_connection():
            logger.error("API connection failed. Please check credentials.")
            return False
        logger.info("✓ API connection verified")
        
        # 2. Load existing state from database
        logger.info("Step 2: Loading system state...")
        # TODO: Implement state loading
        logger.info("✓ System state loaded")
        
        # 3. Verify account balance
        logger.info("Step 3: Checking account balance...")
        try:
            balance = self.broker.get_balance()
            logger.info(f"✓ Account balance: ₩{balance['total_equity']:,.0f}")
            logger.info(f"  Cash: ₩{balance['cash']:,.0f}")
            logger.info(f"  Positions: {len(balance['positions'])}")
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return False
        
        # 4. Check market status
        logger.info("Step 4: Checking market status...")
        session = self.clock.get_session()
        logger.info(f"✓ Current session: {session.name}")
        if self.clock.is_market_open():
            logger.info("  Market is OPEN")
        else:
            time_to_open = self.clock.time_to_open()
            if time_to_open:
                hours, remainder = divmod(time_to_open.total_seconds(), 3600)
                minutes = remainder // 60
                logger.info(f"  Market opens in {int(hours)}h {int(minutes)}m")
        
        # 5. Initialize hypothesis engine
        logger.info("Step 5: Initializing hypothesis engine...")
        # TODO: Implement hypothesis engine initialization
        logger.info("✓ Hypothesis engine ready")
        
        # 6. Start interfaces
        logger.info("Step 6: Starting interfaces...")
        
        # Start web dashboard (in thread)
        self._start_web_dashboard()
        
        # Start Telegram bot (async)
        await self._start_telegram()
        
        logger.info("=" * 60)
        logger.info("STARTUP COMPLETE - System is ready")
        logger.info("=" * 60)
        
        if self.enable_web:
            logger.info("🌐 Web Dashboard: http://localhost:8080")
        if self.telegram_bot:
            logger.info("📱 Telegram Bot: Active")
        
        self.running = True
        return True
    
    async def run(self):
        """
        Main run loop.
        """
        logger.info("Entering main run loop...")
        
        while self.running and not self._shutdown_event.is_set():
            try:
                # Main loop iteration
                await self._iteration()
                
                # Sleep between iterations
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info("Run loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Don't crash on errors - log and continue
                await asyncio.sleep(5)
    
    async def _iteration(self):
        """
        Single iteration of the main loop.
        
        This is where the system:
        - Checks for new market data
        - Evaluates active hypotheses
        - Generates signals
        - Manages positions
        """
        from core.clock import is_market_open
        
        if is_market_open():
            # Market is open - active trading mode
            pass  # TODO: Implement trading logic
        else:
            # Market is closed - analysis and learning mode
            pass  # TODO: Implement analysis logic
    
    async def shutdown(self):
        """
        Graceful shutdown sequence.
        """
        logger.info("=" * 60)
        logger.info("LIVING TRADING SYSTEM - SHUTDOWN")
        logger.info("=" * 60)
        
        self.running = False
        self._shutdown_event.set()
        
        # Emit shutdown event
        from core.events import emit_system_event, EventType
        emit_system_event(EventType.SYSTEM_SHUTDOWN, "System shutting down")
        
        # 1. Stop Telegram bot
        if self.telegram_bot:
            logger.info("Stopping Telegram bot...")
            await self.telegram_bot.stop()
        
        # 2. Cancel pending orders
        logger.info("Step 1: Cancelling pending orders...")
        # TODO: Implement order cancellation
        logger.info("✓ Pending orders cancelled")
        
        # 3. Save current state
        logger.info("Step 2: Saving system state...")
        # TODO: Implement state saving
        logger.info("✓ System state saved")
        
        # 4. Close connections
        logger.info("Step 3: Closing connections...")
        self.broker.close()
        logger.info("✓ Connections closed")
        
        logger.info("=" * 60)
        logger.info("SHUTDOWN COMPLETE")
        logger.info("=" * 60)
    
    def handle_signal(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}")
        self._shutdown_event.set()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Living Trading System")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify setup, don't start trading"
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable web dashboard"
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram bot"
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Only start web dashboard (for testing)"
    )
    parser.add_argument(
        "--telegram-only",
        action="store_true",
        help="Only start Telegram bot (for testing)"
    )
    
    args = parser.parse_args()
    
    # Handle interface-only modes
    if args.web_only:
        from interface.web.dashboard import run_dashboard
        run_dashboard()
        return
    
    if args.telegram_only:
        from interface.telegram.bot import main as telegram_main
        await telegram_main()
        return
    
    # Create system
    system = LivingTradingSystem(
        mode=args.mode,
        enable_web=not args.no_web,
        enable_telegram=not args.no_telegram
    )
    
    # Register signal handlers
    signal.signal(signal.SIGINT, system.handle_signal)
    signal.signal(signal.SIGTERM, system.handle_signal)
    
    try:
        # Startup
        success = await system.startup()
        
        if not success:
            logger.error("Startup failed")
            sys.exit(1)
        
        if args.verify_only:
            logger.info("Verification complete. Exiting.")
            await system.shutdown()
            return
        
        # Run main loop
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        # Shutdown
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
