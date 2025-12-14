"""
TELEGRAM BOT - Mobile Interface for Living Trading System

Provides:
- Real-time alerts (trades, risk events, system status)
- Status queries (portfolio, positions, hypotheses)
- Basic controls (pause, resume, emergency stop)

Setup:
1. Message @BotFather on Telegram
2. Send /newbot and follow instructions
3. Copy the bot token to config/credentials.yaml
4. Get your chat_id by messaging your bot and checking /get_chat_id
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import json

from loguru import logger

try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application, 
        CommandHandler, 
        ContextTypes,
        MessageHandler,
        filters
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    CRITICAL = "🚨"


@dataclass
class Alert:
    """An alert to send to the user."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def format(self) -> str:
        """Format alert for Telegram."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"{self.level.value} *{self.title}*\n{self.message}\n_{time_str}_"


class TelegramInterface:
    """
    Telegram bot interface for the Living Trading System.
    
    Commands:
    - /start - Welcome message and help
    - /status - Current system status
    - /portfolio - Portfolio summary
    - /positions - List current positions
    - /hypotheses - Active hypotheses
    - /trades - Recent trades
    - /decisions - Recent decisions
    - /pause - Pause trading
    - /resume - Resume trading
    - /stop - Emergency stop (requires confirmation)
    - /help - List all commands
    """
    
    def __init__(
        self,
        bot_token: str,
        authorized_chat_ids: List[int],
        system_controller = None
    ):
        """
        Initialize Telegram interface.
        
        Args:
            bot_token: Telegram bot token from @BotFather
            authorized_chat_ids: List of chat IDs allowed to control the bot
            system_controller: Reference to main system controller
        """
        if not TELEGRAM_AVAILABLE:
            raise RuntimeError("python-telegram-bot not installed")
        
        self.bot_token = bot_token
        self.authorized_chat_ids = set(authorized_chat_ids)
        self.system = system_controller
        
        self._app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._running = False
        
        # Pending confirmations (for dangerous commands)
        self._pending_confirmations: Dict[int, Dict[str, Any]] = {}
        
        logger.info(f"TelegramInterface initialized with {len(authorized_chat_ids)} authorized users")
    
    def _is_authorized(self, chat_id: int) -> bool:
        """Check if a chat ID is authorized."""
        return chat_id in self.authorized_chat_ids
    
    async def _check_auth(self, update: Update) -> bool:
        """Check authorization and respond if not authorized."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(
                "⛔ Unauthorized. Your chat ID is not registered.\n"
                f"Your chat ID: `{chat_id}`\n"
                "Add this to your config to authorize.",
                parse_mode="Markdown"
            )
            logger.warning(f"Unauthorized access attempt from chat_id: {chat_id}")
            return False
        return True
    
    # === COMMAND HANDLERS ===
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        if not await self._check_auth(update):
            return
        
        welcome_msg = """
🤖 *Living Trading System*

Welcome! I'm your trading system interface.

*Quick Commands:*
/status - System overview
/portfolio - Portfolio details
/positions - Current holdings
/trades - Recent trades

*Controls:*
/pause - Pause trading
/resume - Resume trading
/stop - Emergency stop

Type /help for full command list.
        """
        await update.message.reply_text(welcome_msg.strip(), parse_mode="Markdown")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not await self._check_auth(update):
            return
        
        help_msg = """
📚 *Available Commands*

*Information:*
/status - System status overview
/portfolio - Portfolio summary
/positions - Current positions
/hypotheses - Active hypotheses
/trades [n] - Recent trades (default: 5)
/decisions [n] - Recent decisions
/performance - Performance metrics

*Controls:*
/pause - Pause new trades
/resume - Resume trading
/stop - Emergency stop (requires confirm)
/refresh - Force data refresh

*Settings:*
/alerts on|off - Toggle alerts
/mode - Show current mode (paper/live)

*Help:*
/help - This message
/about - System information
        """
        await update.message.reply_text(help_msg.strip(), parse_mode="Markdown")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not await self._check_auth(update):
            return
        
        # Get system status
        status = await self._get_system_status()
        
        status_msg = f"""
📊 *System Status*

*Mode:* {status['mode']}
*State:* {status['state']}
*Uptime:* {status['uptime']}

💰 *Portfolio*
Total: ₩{status['total_equity']:,.0f}
Cash: ₩{status['cash']:,.0f}
Positions: {status['position_count']}

📈 *Performance*
Today: {status['daily_pnl']:+.2f}%
Drawdown: {status['drawdown']:.2f}%

🎯 *Hypotheses*
Active: {status['active_hypotheses']}
Incubating: {status['incubating_hypotheses']}

⏰ _Updated: {status['timestamp']}_
        """
        await update.message.reply_text(status_msg.strip(), parse_mode="Markdown")
    
    async def cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command."""
        if not await self._check_auth(update):
            return
        
        portfolio = await self._get_portfolio()
        
        msg = f"""
💼 *Portfolio Details*

*Value:*
Total Equity: ₩{portfolio['total_equity']:,.0f}
Cash: ₩{portfolio['cash']:,.0f}
Invested: ₩{portfolio['invested']:,.0f}

*Performance:*
Today: {portfolio['daily_pnl']:+.2f}% (₩{portfolio['daily_pnl_krw']:+,.0f})
Week: {portfolio['weekly_pnl']:+.2f}%
Month: {portfolio['monthly_pnl']:+.2f}%

*Risk:*
Current Drawdown: {portfolio['drawdown']:.2f}%
Max Drawdown: {portfolio['max_drawdown']:.2f}%

*Allocation:*
{portfolio['allocation_summary']}
        """
        await update.message.reply_text(msg.strip(), parse_mode="Markdown")
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        if not await self._check_auth(update):
            return
        
        positions = await self._get_positions()
        
        if not positions:
            await update.message.reply_text("📋 No open positions.")
            return
        
        msg = "📋 *Current Positions*\n\n"
        
        for i, pos in enumerate(positions, 1):
            pnl_emoji = "🟢" if pos['pnl_pct'] >= 0 else "🔴"
            msg += f"{i}. *{pos['name']}* ({pos['symbol']})\n"
            msg += f"   {pos['quantity']}주 @ ₩{pos['avg_cost']:,.0f}\n"
            msg += f"   {pnl_emoji} {pos['pnl_pct']:+.2f}% (₩{pos['pnl_krw']:+,.0f})\n\n"
        
        await update.message.reply_text(msg.strip(), parse_mode="Markdown")
    
    async def cmd_hypotheses(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /hypotheses command."""
        if not await self._check_auth(update):
            return
        
        hypotheses = await self._get_hypotheses()
        
        if not hypotheses:
            await update.message.reply_text("🎯 No active hypotheses.")
            return
        
        msg = "🎯 *Active Hypotheses*\n\n"
        
        for hyp in hypotheses:
            status_emoji = {
                "active": "🟢",
                "incubating": "🟡",
                "paper": "⚪",
                "paused": "⏸️"
            }.get(hyp['status'], "❓")
            
            msg += f"{status_emoji} *{hyp['id']}*\n"
            msg += f"   Status: {hyp['status']}\n"
            msg += f"   Allocation: {hyp['allocation']:.1f}%\n"
            msg += f"   Win Rate: {hyp['win_rate']:.1f}%\n"
            msg += f"   P&L: ₩{hyp['pnl']:+,.0f}\n\n"
        
        await update.message.reply_text(msg.strip(), parse_mode="Markdown")
    
    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command."""
        if not await self._check_auth(update):
            return
        
        # Parse optional count argument
        try:
            count = int(context.args[0]) if context.args else 5
            count = min(count, 20)  # Cap at 20
        except (ValueError, IndexError):
            count = 5
        
        trades = await self._get_recent_trades(count)
        
        if not trades:
            await update.message.reply_text("📜 No recent trades.")
            return
        
        msg = f"📜 *Recent Trades* (last {len(trades)})\n\n"
        
        for trade in trades:
            side_emoji = "🟢" if trade['side'] == 'buy' else "🔴"
            msg += f"{side_emoji} *{trade['side'].upper()}* {trade['symbol']}\n"
            msg += f"   {trade['quantity']}주 @ ₩{trade['price']:,.0f}\n"
            msg += f"   _{trade['time']}_\n\n"
        
        await update.message.reply_text(msg.strip(), parse_mode="Markdown")
    
    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command."""
        if not await self._check_auth(update):
            return
        
        success = await self._pause_system()
        
        if success:
            await update.message.reply_text(
                "⏸️ *System Paused*\n\n"
                "No new trades will be placed.\n"
                "Existing positions are maintained.\n\n"
                "Use /resume to continue trading.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("❌ Failed to pause system.")
    
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command."""
        if not await self._check_auth(update):
            return
        
        success = await self._resume_system()
        
        if success:
            await update.message.reply_text(
                "▶️ *System Resumed*\n\n"
                "Trading is now active.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("❌ Failed to resume system.")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command - requires confirmation."""
        if not await self._check_auth(update):
            return
        
        chat_id = update.effective_chat.id
        
        # Check if confirming
        if context.args and context.args[0].upper() == "CONFIRM":
            success = await self._emergency_stop()
            if success:
                await update.message.reply_text(
                    "🚨 *EMERGENCY STOP EXECUTED*\n\n"
                    "All trading halted.\n"
                    "Manual restart required.",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("❌ Failed to execute emergency stop.")
            return
        
        # Request confirmation
        await update.message.reply_text(
            "🚨 *Emergency Stop*\n\n"
            "This will:\n"
            "• Halt all trading immediately\n"
            "• Cancel all pending orders\n"
            "• Require manual restart\n\n"
            "To confirm, type:\n"
            "`/stop CONFIRM`",
            parse_mode="Markdown"
        )
    
    async def cmd_decisions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /decisions command."""
        if not await self._check_auth(update):
            return
        
        try:
            count = int(context.args[0]) if context.args else 5
            count = min(count, 10)
        except (ValueError, IndexError):
            count = 5
        
        decisions = await self._get_recent_decisions(count)
        
        if not decisions:
            await update.message.reply_text("🧠 No recent decisions.")
            return
        
        msg = f"🧠 *Recent Decisions*\n\n"
        
        for dec in decisions:
            msg += f"*{dec['type']}*\n"
            msg += f"{dec['description']}\n"
            msg += f"Confidence: {dec['confidence']:.0%}\n"
            msg += f"_{dec['time']}_\n\n"
        
        await update.message.reply_text(msg.strip(), parse_mode="Markdown")
    
    # === DATA FETCHING (connects to system) ===

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status from orchestrator."""
        try:
            from orchestrator import get_orchestrator
            from core.clock import is_market_open

            orch = get_orchestrator()
            status = orch.get_status()

            # Calculate uptime
            uptime_seconds = status.get("uptime_seconds", 0)
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)

            return {
                "mode": status.get("mode", "Unknown"),
                "state": status.get("state", "Unknown"),
                "uptime": f"{hours}h {minutes}m",
                "total_equity": status.get("total_equity", 0),
                "cash": status.get("cash", 0),
                "position_count": status.get("position_count", 0),
                "daily_pnl": status.get("daily_pnl_pct", 0),
                "drawdown": status.get("drawdown_pct", 0),
                "active_hypotheses": status.get("live_strategies", 0),
                "incubating_hypotheses": status.get("paper_strategies", 0),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        except Exception as e:
            logger.debug(f"Could not get system status: {e}")
            return {
                "mode": "Unknown",
                "state": "Not Connected",
                "uptime": "N/A",
                "total_equity": 0,
                "cash": 0,
                "position_count": 0,
                "daily_pnl": 0,
                "drawdown": 0,
                "active_hypotheses": 0,
                "incubating_hypotheses": 0,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }

    async def _get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio details from orchestrator."""
        try:
            from orchestrator import get_orchestrator
            orch = get_orchestrator()
            status = orch.get_status()

            total_equity = status.get("total_equity", 0)
            cash = status.get("cash", 0)
            invested = total_equity - cash

            return {
                "total_equity": total_equity,
                "cash": cash,
                "invested": invested,
                "daily_pnl": status.get("daily_pnl_pct", 0),
                "daily_pnl_krw": status.get("daily_pnl", 0),
                "weekly_pnl": 0,  # Would need historical tracking
                "monthly_pnl": 0,
                "drawdown": status.get("drawdown_pct", 0),
                "max_drawdown": status.get("max_drawdown_pct", 0),
                "allocation_summary": f"Invested: {invested/total_equity*100:.0f}% | Cash: {cash/total_equity*100:.0f}%" if total_equity > 0 else "N/A"
            }
        except Exception as e:
            logger.debug(f"Could not get portfolio: {e}")
            return {
                "total_equity": 0, "cash": 0, "invested": 0,
                "daily_pnl": 0, "daily_pnl_krw": 0, "weekly_pnl": 0, "monthly_pnl": 0,
                "drawdown": 0, "max_drawdown": 0, "allocation_summary": "N/A"
            }

    async def _get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions from broker."""
        try:
            from execution.broker import KISBroker
            broker = KISBroker()
            positions = await broker.get_positions()

            result = []
            for pos in positions:
                result.append({
                    "symbol": pos.get("symbol", ""),
                    "name": pos.get("name", pos.get("symbol", "")),
                    "quantity": pos.get("quantity", 0),
                    "avg_cost": pos.get("avg_cost", 0),
                    "pnl_pct": pos.get("pnl_pct", 0),
                    "pnl_krw": pos.get("pnl", 0),
                })
            return result
        except Exception as e:
            logger.debug(f"Could not get positions: {e}")
            return []

    async def _get_hypotheses(self) -> List[Dict[str, Any]]:
        """Get active hypotheses from registry."""
        try:
            from hypothesis import get_registry

            registry = get_registry()
            hypotheses = registry.get_all()

            result = []
            for hyp in hypotheses:
                status = hyp.state.value if hasattr(hyp.state, 'value') else str(hyp.state)
                metrics = hyp.backtest_metrics or {}

                result.append({
                    "id": hyp.hypothesis_id[:12],
                    "status": status,
                    "allocation": 0,  # Would need allocator
                    "win_rate": metrics.get("win_rate", 0) * 100,
                    "pnl": metrics.get("total_pnl", 0),
                })
            return result
        except Exception as e:
            logger.debug(f"Could not get hypotheses: {e}")
            return []

    async def _get_recent_trades(self, count: int) -> List[Dict[str, Any]]:
        """Get recent trades from database."""
        try:
            from memory.models import get_database, Trade

            db = get_database()
            session = db.get_session()

            trades = session.query(Trade).order_by(Trade.executed_at.desc()).limit(count).all()

            result = []
            for trade in trades:
                result.append({
                    "side": trade.side.value if hasattr(trade.side, 'value') else str(trade.side),
                    "symbol": trade.symbol,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "time": trade.executed_at.strftime("%H:%M") if trade.executed_at else "N/A"
                })

            session.close()
            return result
        except Exception as e:
            logger.debug(f"Could not get trades: {e}")
            return []

    async def _get_recent_decisions(self, count: int) -> List[Dict[str, Any]]:
        """Get recent decisions from database."""
        try:
            from memory.models import get_database, Decision

            db = get_database()
            session = db.get_session()

            decisions = session.query(Decision).order_by(Decision.timestamp.desc()).limit(count).all()

            result = []
            for dec in decisions:
                result.append({
                    "type": dec.decision_type,
                    "description": dec.description[:100] if dec.description else "N/A",
                    "confidence": dec.confidence or 0,
                    "time": dec.timestamp.strftime("%H:%M") if dec.timestamp else "N/A"
                })

            session.close()
            return result
        except Exception as e:
            logger.debug(f"Could not get decisions: {e}")
            return []

    async def _pause_system(self) -> bool:
        """Pause the trading system."""
        try:
            from orchestrator import get_orchestrator
            orch = get_orchestrator()
            await orch.pause()
            logger.info("System paused via Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to pause system: {e}")
            return False

    async def _resume_system(self) -> bool:
        """Resume the trading system."""
        try:
            from orchestrator import get_orchestrator
            orch = get_orchestrator()
            await orch.resume()
            logger.info("System resumed via Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to resume system: {e}")
            return False

    async def _emergency_stop(self) -> bool:
        """Execute emergency stop."""
        try:
            from orchestrator import get_orchestrator
            orch = get_orchestrator()
            await orch.stop()
            logger.warning("EMERGENCY STOP executed via Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to execute emergency stop: {e}")
            return False
    
    # === ALERT SENDING ===
    
    async def send_alert(self, alert: Alert):
        """Send an alert to all authorized users."""
        if not self._bot:
            logger.warning("Bot not initialized, cannot send alert")
            return
        
        message = alert.format()
        
        for chat_id in self.authorized_chat_ids:
            try:
                await self._bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Failed to send alert to {chat_id}: {e}")
    
    async def send_trade_alert(
        self,
        side: str,
        symbol: str,
        name: str,
        quantity: int,
        price: float,
        hypothesis_id: str = None
    ):
        """Send a trade execution alert."""
        emoji = "🟢" if side == "buy" else "🔴"
        
        alert = Alert(
            level=AlertLevel.INFO,
            title=f"{emoji} Trade Executed",
            message=(
                f"*{side.upper()}* {name} ({symbol})\n"
                f"Quantity: {quantity}주\n"
                f"Price: ₩{price:,.0f}\n"
                f"Hypothesis: {hypothesis_id or 'N/A'}"
            )
        )
        await self.send_alert(alert)
    
    async def send_risk_alert(
        self,
        risk_type: str,
        current_value: float,
        threshold: float,
        action: str
    ):
        """Send a risk alert."""
        alert = Alert(
            level=AlertLevel.WARNING,
            title=f"Risk Alert: {risk_type}",
            message=(
                f"Current: {current_value:.2f}%\n"
                f"Threshold: {threshold:.2f}%\n"
                f"Action: {action}"
            )
        )
        await self.send_alert(alert)
    
    async def send_system_alert(self, title: str, message: str, level: AlertLevel = AlertLevel.INFO):
        """Send a general system alert."""
        alert = Alert(level=level, title=title, message=message)
        await self.send_alert(alert)
    
    # === LIFECYCLE ===
    
    async def start(self):
        """Start the Telegram bot."""
        logger.info("Starting Telegram bot...")
        
        self._app = Application.builder().token(self.bot_token).build()
        self._bot = self._app.bot
        
        # Register command handlers
        self._app.add_handler(CommandHandler("start", self.cmd_start))
        self._app.add_handler(CommandHandler("help", self.cmd_help))
        self._app.add_handler(CommandHandler("status", self.cmd_status))
        self._app.add_handler(CommandHandler("portfolio", self.cmd_portfolio))
        self._app.add_handler(CommandHandler("positions", self.cmd_positions))
        self._app.add_handler(CommandHandler("hypotheses", self.cmd_hypotheses))
        self._app.add_handler(CommandHandler("trades", self.cmd_trades))
        self._app.add_handler(CommandHandler("decisions", self.cmd_decisions))
        self._app.add_handler(CommandHandler("pause", self.cmd_pause))
        self._app.add_handler(CommandHandler("resume", self.cmd_resume))
        self._app.add_handler(CommandHandler("stop", self.cmd_stop))
        
        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        
        self._running = True
        logger.info("Telegram bot started successfully")
        
        # Send startup notification
        await self.send_system_alert(
            "System Started",
            "Living Trading System is now online.",
            AlertLevel.SUCCESS
        )
    
    async def stop(self):
        """Stop the Telegram bot."""
        logger.info("Stopping Telegram bot...")
        
        if self._app:
            # Send shutdown notification
            await self.send_system_alert(
                "System Stopping",
                "Living Trading System is shutting down.",
                AlertLevel.WARNING
            )
            
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        
        self._running = False
        logger.info("Telegram bot stopped")


# === STANDALONE RUNNER ===

async def main():
    """Run Telegram bot standalone for testing."""
    import yaml
    
    # Load credentials
    try:
        with open("config/credentials.yaml", 'r') as f:
            creds = yaml.safe_load(f)
        
        telegram_config = creds.get("notifications", {}).get("telegram", {})
        bot_token = telegram_config.get("bot_token")
        chat_id = telegram_config.get("chat_id")
        
        if not bot_token or not chat_id:
            print("Telegram credentials not configured.")
            print("Please add bot_token and chat_id to config/credentials.yaml")
            return
        
        bot = TelegramInterface(
            bot_token=bot_token,
            authorized_chat_ids=[int(chat_id)]
        )
        
        await bot.start()
        
        # Keep running
        print("Bot is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        await bot.stop()
    except FileNotFoundError:
        print("config/credentials.yaml not found")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
