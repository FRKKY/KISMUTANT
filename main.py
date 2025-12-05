#!/usr/bin/env python3
"""
LIVING TRADING SYSTEM - Main Entry Point
"""

import sys
import os
import signal
import asyncio
import argparse
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

Path("logs").mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
logger.add("logs/system_{time:YYYY-MM-DD}.log", rotation="00:00", retention="7 days", level="DEBUG")


class LivingTradingSystem:
    def __init__(self, mode: str = "paper"):
        self.mode = mode
        self.running = False
        self._shutdown_event = asyncio.Event()
        self.telegram_bot = None
        logger.info(f"Initializing Living Trading System in {mode} mode")
    
    def _get_telegram_config(self):
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if bot_token and chat_id:
            try:
                return bot_token, int(chat_id)
            except ValueError:
                pass
        try:
            import yaml
            with open("config/credentials.yaml", 'r') as f:
                creds = yaml.safe_load(f)
            tg = creds.get("notifications", {}).get("telegram", {})
            token = tg.get("bot_token", "")
            cid = tg.get("chat_id", "")
            if token and "YOUR_" not in token:
                return token, int(cid)
        except:
            pass
        return None, None
    
    async def start_telegram(self):
        bot_token, chat_id = self._get_telegram_config()
        if not bot_token:
            logger.info("Telegram not configured - skipping")
            return
        try:
            from interface.telegram.bot import TelegramInterface
            self.telegram_bot = TelegramInterface(bot_token=bot_token, authorized_chat_ids=[chat_id], system_controller=self)
            await self.telegram_bot.start()
            logger.info("✓ Telegram bot started")
        except Exception as e:
            logger.error(f"Telegram bot failed: {e}")
    
    def start_web(self):
        try:
            from interface.web.dashboard import run_dashboard
            port = int(os.environ.get("PORT", 8080))
            thread = threading.Thread(target=run_dashboard, kwargs={"host": "0.0.0.0", "port": port}, daemon=True)
            thread.start()
            logger.info(f"✓ Web dashboard started on port {port}")
            return thread
        except Exception as e:
            logger.error(f"Web dashboard failed: {e}")
            return None
    
    async def run(self):
        self.running = True
        logger.info("System running. Press Ctrl+C to stop.")
        while self.running and not self._shutdown_event.is_set():
            await asyncio.sleep(10)
    
    async def shutdown(self):
        logger.info("Shutting down...")
        self.running = False
        self._shutdown_event.set()
        if self.telegram_bot:
            try:
                await self.telegram_bot.stop()
            except:
                pass
        logger.info("Shutdown complete")
    
    def handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self._shutdown_event.set()


async def main():
    parser = argparse.ArgumentParser(description="Living Trading System")
    parser.add_argument("--mode", choices=["paper", "live"], default=os.environ.get("TRADING_MODE", "paper"))
    parser.add_argument("--web-only", action="store_true")
    args = parser.parse_args()
    
    if args.web_only:
        from interface.web.dashboard import run_dashboard
        run_dashboard()
        return
    
    system = LivingTradingSystem(mode=args.mode)
    signal.signal(signal.SIGINT, system.handle_signal)
    signal.signal(signal.SIGTERM, system.handle_signal)
    
    try:
        logger.info("=" * 50)
        logger.info("LIVING TRADING SYSTEM - STARTING")
        logger.info("=" * 50)
        system.start_web()
        await asyncio.sleep(2)
        await system.start_telegram()
        logger.info("=" * 50)
        logger.info("SYSTEM READY")
        logger.info("=" * 50)
        await system.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
