"""
ORCHESTRATOR - Main Trading System Controller

The Orchestrator ties all components together and runs the main loop:

Daily Flow:
1. Market Open
   - Refresh universe data
   - Compute features
   - Scan for patterns
   
2. Signal Generation
   - Evaluate hypotheses against current data
   - Generate trading signals
   
3. Position Sizing
   - Apply Kelly criterion
   - Check risk limits
   
4. Execution
   - Submit orders via broker
   - Track fills
   
5. Market Close
   - Update positions
   - Calculate P&L
   - Evaluate strategy performance
   - Run lifecycle checks (promote/demote)
   
6. End of Day
   - Generate reports
   - Send notifications
   - Persist state
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import signal
import sys

from loguru import logger

# Core
from core.events import get_event_bus, Event, EventType
from core.clock import get_clock, is_market_open, KST
from core.invariants import get_invariants, SystemState

# Perception
from perception import (
    PerceptionLayer,
    UniverseFilter,
    Timeframe,
    get_pattern_detector
)

# Hypothesis
from hypothesis import (
    get_registry,
    get_signal_generator,
    get_promoter,
    HypothesisFactory,
    StrategyState
)

# Portfolio
from portfolio import (
    get_capital_allocator,
    get_position_manager,
    AllocationConfig,
    AllocationMethod
)

# Validation
from validation import get_backtester, BacktestConfig

# Memory
from memory.journal import get_journal

# Learning - Self-improvement components
from learning import (
    get_analyzer,
    get_optimizer,
    get_evolver,
    get_regime_detector,
    PerformanceAnalyzer,
    ParameterOptimizer,
    StrategyEvolver,
    RegimeDetector
)

# Research - External knowledge acquisition
from research import (
    get_fetcher as get_paper_fetcher,
    get_extractor,
    get_research_generator,
    get_knowledge_base,
    PaperFetcher,
    IdeaExtractor,
    ResearchHypothesisGenerator,
    KnowledgeBase
)


class OrchestratorState(str, Enum):
    """States of the orchestrator."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # Capital
    initial_capital: float = 10_000_000  # ₩10M

    # Timing
    market_open: time = time(9, 0)
    market_close: time = time(15, 30)

    # Update frequencies (seconds)
    price_update_interval: int = 60      # Real-time price updates
    intraday_scan_interval: int = 300    # Pattern scan (5 min)

    # Strategy limits
    max_live_strategies: int = 3
    max_paper_strategies: int = 10

    # Auto-management
    auto_discover_patterns: bool = True
    auto_generate_hypotheses: bool = True
    auto_backtest: bool = True
    auto_promote: bool = True

    # Risk
    daily_loss_limit_pct: float = 0.02   # Stop trading if down 2%

    # Notifications
    send_telegram_alerts: bool = True

    # Persistence
    save_state_interval: int = 300       # Save state every 5 minutes

    # === Learning Module ===
    learning_enabled: bool = True
    performance_analysis_interval: int = 3600  # Analyze every hour
    regime_detection_interval: int = 1800      # Check regime every 30 min
    strategy_evolution_interval: int = 86400   # Evolve strategies daily
    min_trades_for_analysis: int = 10          # Min trades before analyzing

    # === Research Module ===
    research_enabled: bool = True
    paper_fetch_interval: int = 21600          # Fetch papers every 6 hours
    max_papers_per_fetch: int = 20             # Papers per fetch
    auto_generate_from_research: bool = True   # Auto-create strategies from papers


class Orchestrator:
    """
    Main controller for the Living Trading System.
    
    Coordinates all components and runs the trading loop.
    """
    
    def __init__(
        self,
        broker=None,
        config: Optional[OrchestratorConfig] = None
    ):
        self.config = config or OrchestratorConfig()
        self._broker = broker
        
        # State
        self._state = OrchestratorState.INITIALIZING
        self._start_time: Optional[datetime] = None
        self._last_daily_update: Optional[datetime] = None
        
        # Components (lazy initialization)
        self._perception: Optional[PerceptionLayer] = None
        self._registry = get_registry()
        self._signal_generator = get_signal_generator()
        self._allocator = get_capital_allocator(AllocationConfig(
            total_capital=self.config.initial_capital,
            method=AllocationMethod.HALF_KELLY,
            max_strategies=self.config.max_live_strategies
        ))
        self._positions = get_position_manager()
        self._backtester = get_backtester()
        self._journal = get_journal()
        self._invariants = get_invariants()
        self._clock = get_clock()

        # Learning components (self-improvement)
        self._performance_analyzer = get_analyzer()
        self._parameter_optimizer = get_optimizer()
        self._strategy_evolver = get_evolver()
        self._regime_detector = get_regime_detector()

        # Research components (knowledge acquisition)
        self._paper_fetcher = get_paper_fetcher()
        self._idea_extractor = get_extractor()
        self._research_generator = get_research_generator()
        self._knowledge_base = get_knowledge_base()

        # Strategy promotion
        self._promoter = get_promoter()

        # Tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Daily tracking
        self._daily_starting_equity: float = self.config.initial_capital
        self._daily_pnl: float = 0.0

        # Research tracking
        self._last_research_fetch: Optional[datetime] = None

        logger.info("Orchestrator created with learning and research modules")
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize all components.
        
        Call this before starting the main loop.
        """
        logger.info("Initializing orchestrator...")
        summary = {"components": {}, "errors": []}
        
        try:
            # Initialize perception layer
            logger.info("Initializing perception layer...")
            self._perception = PerceptionLayer(
                broker=self._broker,
                universe_filter=UniverseFilter(
                    min_aum=10_000_000_000,
                    min_avg_daily_volume=10_000,
                    include_leverage=False,
                    include_inverse=False
                )
            )
            
            perception_summary = await self._perception.initialize(
                daily_history_days=365
            )
            summary["components"]["perception"] = {
                "status": "ok",
                "universe_size": perception_summary.get("universe_discovery", {}).get("filtered_count", 0)
            }
            
            # Allocate capital to live strategies
            allocations = self._allocator.allocate_to_strategies()
            summary["components"]["allocator"] = {
                "status": "ok",
                "allocations": len(allocations)
            }
            
            # Register event handlers
            self._register_event_handlers()
            summary["components"]["events"] = {"status": "ok"}

            # Initialize learning components
            if self.config.learning_enabled:
                summary["components"]["learning"] = {
                    "status": "ok",
                    "modules": ["analyzer", "optimizer", "evolver", "regime_detector"]
                }
                logger.info("Learning modules initialized")

            # Initialize research components
            if self.config.research_enabled:
                summary["components"]["research"] = {
                    "status": "ok",
                    "modules": ["paper_fetcher", "idea_extractor", "hypothesis_generator", "knowledge_base"]
                }
                logger.info("Research modules initialized")

            self._state = OrchestratorState.STOPPED
            self._start_time = datetime.now()

            logger.info("Orchestrator initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            summary["errors"].append(str(e))
        
        return summary
    
    def _register_event_handlers(self) -> None:
        """Register handlers for system events."""
        bus = get_event_bus()

        # Pattern discovered -> potentially create hypothesis (async handler)
        bus.subscribe_async(
            EventType.PATTERN_DISCOVERED,
            self._on_pattern_discovered
        )

        # Signal generated -> execute trade (async handler)
        bus.subscribe_async(
            EventType.SIGNAL_GENERATED,
            self._on_signal_generated
        )
        
        # Position events (async handlers)
        bus.subscribe_async(EventType.POSITION_OPENED, self._on_position_opened)
        bus.subscribe_async(EventType.POSITION_CLOSED, self._on_position_closed)

        # Strategy lifecycle (async handlers)
        bus.subscribe_async(EventType.STRATEGY_PROMOTED, self._on_strategy_promoted)
        bus.subscribe_async(EventType.STRATEGY_DEMOTED, self._on_strategy_demoted)
    
    # ===== Event Handlers =====
    
    async def _on_pattern_discovered(self, event: Event) -> None:
        """Handle pattern discovery."""
        if not self.config.auto_generate_hypotheses:
            return
        
        pattern_data = event.payload
        logger.debug(f"Pattern discovered: {pattern_data.get('pattern_type')}")
        
        # Create hypothesis from pattern
        try:
            from perception.patterns import DetectedPattern
            
            # Reconstruct pattern (simplified)
            pattern = DetectedPattern(**pattern_data)
            
            hypothesis = HypothesisFactory.from_pattern(pattern)
            
            # Auto-backtest
            if self.config.auto_backtest and self._perception:
                data = {}
                for symbol in hypothesis.symbols:
                    symbol_data = self._perception.get_symbol_data(symbol)
                    if symbol_data:
                        data[symbol] = symbol_data
                
                if data:
                    result = self._backtester.run(hypothesis, data)
                    hypothesis.backtest_metrics = result.to_performance_metrics()
                    
                    if result.passed_criteria:
                        self._registry.register(hypothesis)
                        logger.info(f"Auto-registered hypothesis: {hypothesis.name}")
        
        except Exception as e:
            logger.debug(f"Failed to create hypothesis from pattern: {e}")
    
    async def _on_signal_generated(self, event: Event) -> None:
        """Handle trading signal."""
        signal_data = event.payload
        logger.info(f"Signal: {signal_data.get('signal_type')} {signal_data.get('symbol')}")
        
        # Get signal details
        symbol = signal_data.get("symbol")
        hypothesis_id = signal_data.get("hypothesis_id")
        
        # Get hypothesis
        hypothesis = self._registry.get(hypothesis_id)
        if not hypothesis:
            return
        
        # Only execute for live strategies
        if hypothesis.state != StrategyState.LIVE:
            logger.debug(f"Skipping signal for non-live strategy: {hypothesis.name}")
            return
        
        # Size position
        from hypothesis.models import TradingSignal, SignalType
        
        signal = TradingSignal(**signal_data)
        
        # Get current price and ATR
        if self._perception:
            symbol_data = self._perception.get_symbol_data(symbol)
            if symbol_data:
                ohlcv, features = symbol_data
                current_price = ohlcv['close'].iloc[-1]
                atr = features['atr'].iloc[-1] if 'atr' in features.columns else None
                
                sizing = self._allocator.size_position(signal, current_price, atr)
                
                if sizing.is_valid and sizing.position_size_shares > 0:
                    # Execute via broker
                    await self._execute_signal(signal, sizing)
    
    async def _execute_signal(self, signal, sizing) -> None:
        """Execute a trading signal."""
        from hypothesis.models import SignalType
        from portfolio.positions import PositionSide
        
        symbol = signal.symbol
        
        if signal.signal_type in [SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY]:
            # Entry signal
            side = PositionSide.LONG if signal.direction == 1 else PositionSide.SHORT
            
            # Submit order via broker
            if self._broker:
                try:
                    order_result = self._broker.place_order(
                        symbol=symbol,
                        side="buy" if side == PositionSide.LONG else "sell",
                        quantity=sizing.position_size_shares,
                        order_type="market"
                    )
                    
                    if order_result.get("success"):
                        # Track position
                        self._positions.open_position(
                            symbol=symbol,
                            hypothesis_id=signal.hypothesis_id,
                            side=side,
                            quantity=sizing.position_size_shares,
                            entry_price=signal.price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            signal_id=signal.signal_id
                        )
                        
                        # Reserve capital
                        self._allocator.reserve_capital(
                            signal.hypothesis_id,
                            sizing.position_size_krw
                        )
                        
                        # Add to daily risk
                        self._allocator.add_daily_risk(sizing.risk_pct)
                        
                        logger.info(f"Executed {side.value} {symbol} x{sizing.position_size_shares}")
                
                except Exception as e:
                    logger.error(f"Order execution failed: {e}")
        
        elif signal.signal_type in [SignalType.LONG_EXIT, SignalType.SHORT_EXIT, 
                                     SignalType.STOP_LOSS, SignalType.TAKE_PROFIT]:
            # Exit signal
            position = self._positions.get_position(symbol)
            
            if position and self._broker:
                try:
                    side = "sell" if position.side == PositionSide.LONG else "buy"
                    
                    order_result = self._broker.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=position.quantity,
                        order_type="market"
                    )
                    
                    if order_result.get("success"):
                        closed = self._positions.close_position(
                            symbol,
                            signal.price,
                            signal.signal_type.value
                        )
                        
                        if closed:
                            # Release capital
                            self._allocator.release_capital(
                                signal.hypothesis_id,
                                closed.total_cost,
                                closed.realized_pnl
                            )
                            
                            logger.info(
                                f"Closed {symbol}: P&L ₩{closed.realized_pnl:,.0f}"
                            )
                
                except Exception as e:
                    logger.error(f"Exit execution failed: {e}")
    
    async def _on_position_opened(self, event: Event) -> None:
        """Handle position opened."""
        self._journal.log_trade(event.payload)
    
    async def _on_position_closed(self, event: Event) -> None:
        """Handle position closed."""
        self._journal.log_trade(event.payload)

        pnl = event.payload.get("realized_pnl", 0)
        self._daily_pnl += pnl

        # Feed trade data to learning module
        if self.config.learning_enabled:
            trade_data = event.payload
            hypothesis_id = trade_data.get("hypothesis_id")
            if hypothesis_id:
                self._performance_analyzer.record_trade(
                    strategy_id=hypothesis_id,
                    trade={
                        "symbol": trade_data.get("symbol"),
                        "pnl": pnl,
                        "return_pct": trade_data.get("return_pct", 0),
                        "holding_period": trade_data.get("holding_period_days", 1),
                        "exit_reason": trade_data.get("exit_reason", "unknown"),
                        "timestamp": datetime.now()
                    }
                )

        # Check daily loss limit
        daily_return = self._daily_pnl / self._daily_starting_equity
        if daily_return < -self.config.daily_loss_limit_pct:
            logger.warning(f"Daily loss limit hit: {daily_return:.2%}")
            await self.pause("daily_loss_limit")
    
    async def _on_strategy_promoted(self, event: Event) -> None:
        """Handle strategy promotion."""
        logger.info(f"Strategy promoted: {event.payload.get('name')}")
        
        # Reallocate capital
        self._allocator.allocate_to_strategies()
    
    async def _on_strategy_demoted(self, event: Event) -> None:
        """Handle strategy demotion."""
        logger.warning(f"Strategy demoted: {event.payload.get('name')}")
        
        # Reallocate capital
        self._allocator.allocate_to_strategies()
    
    # ===== Main Loop =====
    
    async def start(self) -> None:
        """Start the main trading loop."""
        if self._state == OrchestratorState.RUNNING:
            logger.warning("Orchestrator already running")
            return
        
        self._running = True
        self._state = OrchestratorState.RUNNING
        
        logger.info("Starting orchestrator...")
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._daily_update_loop()),
            asyncio.create_task(self._signal_generation_loop()),
            asyncio.create_task(self._position_monitor_loop()),
            asyncio.create_task(self._lifecycle_check_loop()),
        ]

        # Add learning tasks if enabled
        if self.config.learning_enabled:
            self._tasks.extend([
                asyncio.create_task(self._learning_loop()),
                asyncio.create_task(self._regime_detection_loop()),
                asyncio.create_task(self._strategy_evolution_loop()),
            ])
            logger.info("Learning background tasks started")

        # Add research tasks if enabled
        if self.config.research_enabled:
            self._tasks.append(
                asyncio.create_task(self._research_loop())
            )
            logger.info("Research background task started")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator tasks cancelled")
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        logger.info("Stopping orchestrator...")
        self._running = False
        self._state = OrchestratorState.STOPPING
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._state = OrchestratorState.STOPPED
        logger.info("Orchestrator stopped")
    
    async def pause(self, reason: str = "") -> None:
        """Pause trading."""
        logger.warning(f"Pausing orchestrator: {reason}")
        self._state = OrchestratorState.PAUSED
        
        # Emit alert
        get_event_bus().publish(Event(
            event_type=EventType.SYSTEM_ALERT,
            source="orchestrator",
            payload={"action": "paused", "reason": reason}
        ))
    
    async def resume(self) -> None:
        """Resume trading."""
        if self._state != OrchestratorState.PAUSED:
            return
        
        logger.info("Resuming orchestrator")
        self._state = OrchestratorState.RUNNING
    
    # ===== Background Loops =====
    
    async def _daily_update_loop(self) -> None:
        """Run daily data updates."""
        while self._running:
            try:
                now = datetime.now(KST)
                
                # Run at market open
                if (now.time() >= self.config.market_open and 
                    (self._last_daily_update is None or 
                     self._last_daily_update.date() < now.date())):
                    
                    logger.info("Running daily update...")
                    
                    # Reset daily tracking
                    self._daily_starting_equity = self._allocator.get_stats()["total_capital"]
                    self._daily_pnl = 0
                    self._allocator.reset_daily_risk()
                    
                    # Update perception data
                    if self._perception:
                        await self._perception.run_daily_update()
                    
                    self._last_daily_update = now
                    
                    logger.info("Daily update complete")
                
                # Sleep until next check
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Daily update error: {e}")
                await asyncio.sleep(60)
    
    async def _signal_generation_loop(self) -> None:
        """Generate signals during market hours."""
        while self._running:
            try:
                if self._state != OrchestratorState.RUNNING:
                    await asyncio.sleep(10)
                    continue
                
                if not is_market_open():
                    await asyncio.sleep(60)
                    continue
                
                # Get current market data
                if self._perception:
                    # Run intraday scan for patterns
                    await self._perception.run_intraday_scan()
                    
                    # Generate signals from active strategies
                    data = {}
                    for symbol in self._perception.get_universe():
                        symbol_data = self._perception.get_symbol_data(symbol)
                        if symbol_data:
                            data[symbol] = symbol_data
                    
                    # Update position manager for signal generator
                    self._signal_generator.set_positions(
                        self._positions.get_positions_for_broker_sync()
                    )
                    
                    # Generate signals
                    signals = self._signal_generator.generate_signals(data)
                    
                    # Signals are handled via event bus
                
                await asyncio.sleep(self.config.intraday_scan_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal generation error: {e}")
                await asyncio.sleep(60)
    
    async def _position_monitor_loop(self) -> None:
        """Monitor open positions for stop loss / take profit."""
        while self._running:
            try:
                if self._state != OrchestratorState.RUNNING:
                    await asyncio.sleep(10)
                    continue
                
                if not is_market_open():
                    await asyncio.sleep(60)
                    continue
                
                # Update position prices
                if self._broker and self._positions.get_all_positions():
                    prices = {}
                    
                    for symbol in self._positions.get_all_positions().keys():
                        try:
                            price_data = self._broker.get_price(symbol)
                            if price_data:
                                prices[symbol] = price_data.get("current_price", 0)
                        except Exception as e:
                            logger.debug(f"Price fetch error for {symbol}: {e}")
                    
                    if prices:
                        self._positions.update_prices(prices)
                        
                        # Check stop losses
                        stopped = self._positions.check_stop_losses(prices)
                        for position in stopped:
                            # Generate exit signal
                            from hypothesis.models import TradingSignal, SignalType
                            
                            exit_signal = TradingSignal(
                                signal_id=f"sl_{position.position_id}",
                                hypothesis_id=position.hypothesis_id,
                                symbol=position.symbol,
                                timestamp=datetime.now(),
                                signal_type=SignalType.STOP_LOSS,
                                direction=0,
                                price=prices[position.symbol],
                                confidence=1.0,
                                reason="Stop loss triggered"
                            )
                            
                            sizing = type('obj', (object,), {
                                'is_valid': True,
                                'position_size_shares': position.quantity
                            })()
                            
                            await self._execute_signal(exit_signal, sizing)
                        
                        # Check take profits
                        profited = self._positions.check_take_profits(prices)
                        for position in profited:
                            exit_signal = TradingSignal(
                                signal_id=f"tp_{position.position_id}",
                                hypothesis_id=position.hypothesis_id,
                                symbol=position.symbol,
                                timestamp=datetime.now(),
                                signal_type=SignalType.TAKE_PROFIT,
                                direction=0,
                                price=prices[position.symbol],
                                confidence=1.0,
                                reason="Take profit triggered"
                            )
                            
                            sizing = type('obj', (object,), {
                                'is_valid': True,
                                'position_size_shares': position.quantity
                            })()
                            
                            await self._execute_signal(exit_signal, sizing)
                
                await asyncio.sleep(self.config.price_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _lifecycle_check_loop(self) -> None:
        """Check strategy lifecycle (promote/demote) using intelligent promoter."""
        while self._running:
            try:
                now = datetime.now(KST)

                # Run after market close
                if now.time() >= time(16, 0):
                    # Use intelligent promoter for evaluations
                    results = self._promoter.evaluate_all()

                    if results.get("promotions") or results.get("demotions") or results.get("retirements"):
                        logger.info(
                            f"Lifecycle check: "
                            f"{len(results.get('promotions', []))} promoted, "
                            f"{len(results.get('demotions', []))} demoted, "
                            f"{len(results.get('retirements', []))} retired"
                        )

                        # Log promotion candidates for transparency
                        candidates = self._promoter.get_promotion_candidates()
                        if candidates:
                            logger.debug(f"Promotion candidates: {len(candidates)}")
                            for c in candidates[:5]:  # Top 5
                                logger.debug(f"  - {c['hypothesis_id']}: score={c['priority_score']:.3f}")

                    # Process any pending backtests
                    next_backtest = self._promoter.get_next_backtest()
                    if next_backtest and self._perception:
                        await self._run_backtest(next_backtest)

                    # Wait until next day
                    await asyncio.sleep(3600 * 12)
                else:
                    await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lifecycle check error: {e}")
                await asyncio.sleep(3600)

    async def _run_backtest(self, hypothesis_id: str) -> None:
        """Run backtest for a queued hypothesis."""
        hypothesis = self._registry.get(hypothesis_id)
        if not hypothesis:
            return

        try:
            # Gather data for symbols
            data = {}
            for symbol in hypothesis.symbols:
                if self._perception:
                    symbol_data = self._perception.get_symbol_data(symbol)
                    if symbol_data:
                        data[symbol] = symbol_data

            if data:
                result = self._backtester.run(hypothesis, data)
                metrics = result.to_performance_metrics()

                # Notify promoter of completed backtest
                self._promoter.complete_backtest(hypothesis_id, metrics)

                logger.info(
                    f"Backtest complete for {hypothesis.name}: "
                    f"Sharpe={metrics.sharpe_ratio:.2f}, "
                    f"Return={metrics.total_return:.1%}"
                )

        except Exception as e:
            logger.error(f"Backtest failed for {hypothesis_id}: {e}")
    
    # ===== Learning Background Loops =====

    async def _learning_loop(self) -> None:
        """Analyze strategy performance and learn from results."""
        while self._running:
            try:
                # Get all strategies with enough trades
                strategies = self._registry.get_all()

                for hypothesis in strategies:
                    strategy_id = hypothesis.hypothesis_id

                    # Check if we have enough trades to analyze
                    analyzer_stats = self._performance_analyzer.get_stats()
                    trade_count = analyzer_stats.get("by_strategy", {}).get(strategy_id, {}).get("trade_count", 0)

                    if trade_count >= self.config.min_trades_for_analysis:
                        # Analyze performance
                        analysis = self._performance_analyzer.analyze_strategy(strategy_id)

                        if analysis:
                            logger.info(
                                f"Strategy {hypothesis.name}: "
                                f"Win rate={analysis.get('win_rate', 0):.1%}, "
                                f"Sharpe={analysis.get('sharpe_ratio', 0):.2f}"
                            )

                            # Store in knowledge base
                            self._knowledge_base.add_hypothesis(hypothesis)

                            # Check if optimization is needed
                            if analysis.get("sharpe_ratio", 0) < 0.5:
                                logger.info(f"Strategy {hypothesis.name} underperforming, suggesting optimization")
                                suggestions = self._performance_analyzer.get_improvement_suggestions(strategy_id)
                                for suggestion in suggestions[:3]:
                                    logger.debug(f"  - {suggestion}")

                await asyncio.sleep(self.config.performance_analysis_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                await asyncio.sleep(300)

    async def _regime_detection_loop(self) -> None:
        """Detect market regime and adjust strategy parameters."""
        while self._running:
            try:
                if not self._perception:
                    await asyncio.sleep(60)
                    continue

                # Get market data for regime detection (use a broad ETF like KODEX 200)
                market_symbols = ["069500"]  # KODEX 200 as market proxy
                for symbol in market_symbols:
                    symbol_data = self._perception.get_symbol_data(symbol)
                    if symbol_data:
                        ohlcv, features = symbol_data
                        regime = self._regime_detector.detect_regime(ohlcv)

                        logger.info(
                            f"Market regime: {regime.primary_regime.value} "
                            f"(confidence={regime.confidence:.1%}, duration={regime.duration_days}d)"
                        )

                        # Get strategy adjustments based on regime
                        for hypothesis in self._registry.get_live_strategies():
                            adjustments = self._regime_detector.get_strategy_adjustment(
                                hypothesis.strategy_type.value if hasattr(hypothesis, 'strategy_type') else 'momentum'
                            )

                            if adjustments.get("position_size_mult", 1.0) != 1.0:
                                logger.debug(
                                    f"Strategy {hypothesis.name}: position size adjustment "
                                    f"= {adjustments['position_size_mult']:.1%}"
                                )
                        break

                await asyncio.sleep(self.config.regime_detection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Regime detection error: {e}")
                await asyncio.sleep(300)

    async def _strategy_evolution_loop(self) -> None:
        """Evolve strategies using genetic algorithms."""
        while self._running:
            try:
                now = datetime.now(KST)

                # Run evolution after market close
                if now.time() >= time(17, 0):
                    # Get underperforming strategies
                    strategies = self._registry.get_all()
                    underperforming = []

                    for hypothesis in strategies:
                        if hypothesis.backtest_metrics:
                            sharpe = hypothesis.backtest_metrics.get("sharpe_ratio", 0)
                            if sharpe < 0.5:
                                underperforming.append(hypothesis)

                    if underperforming:
                        logger.info(f"Found {len(underperforming)} underperforming strategies for evolution")

                        # Initialize population from underperforming strategies
                        for hypothesis in underperforming[:3]:  # Limit to 3
                            strategy_type = getattr(hypothesis, 'strategy_type', 'momentum')
                            base_params = hypothesis.parameters if hasattr(hypothesis, 'parameters') else {}

                            self._strategy_evolver.initialize_population(
                                strategy_type=strategy_type,
                                base_params=base_params
                            )

                            # Run a few generations (lightweight)
                            def mock_fitness(gene):
                                # Simple fitness based on parameter quality
                                score = 0.5
                                if gene.parameters.get("stop_loss", 0.1) < 0.05:
                                    score += 0.1
                                if len(gene.indicators) >= 2:
                                    score += 0.1
                                return score

                            best_gene = self._strategy_evolver.evolve(
                                fitness_fn=mock_fitness,
                                n_generations=5
                            )

                            logger.info(
                                f"Evolution complete: best fitness = {best_gene.fitness:.3f}"
                            )

                    # Wait until next day
                    await asyncio.sleep(self.config.strategy_evolution_interval)
                else:
                    await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Strategy evolution error: {e}")
                await asyncio.sleep(3600)

    # ===== Research Background Loop =====

    async def _research_loop(self) -> None:
        """Fetch academic papers and generate hypotheses from research."""
        while self._running:
            try:
                now = datetime.now()

                # Check if we should fetch (once per day)
                should_fetch = (
                    self._last_research_fetch is None or
                    (now - self._last_research_fetch).total_seconds() >= self.config.paper_fetch_interval
                )

                if should_fetch:
                    logger.info("Starting research paper fetch from all sources...")

                    # Fetch papers from all sources (arXiv, Semantic Scholar, SSRN)
                    papers = await self._paper_fetcher.fetch_all_sources(
                        max_per_source=self.config.max_papers_per_fetch
                    )

                    if papers:
                        logger.info(f"Fetched {len(papers)} research papers from multiple sources")

                        # Store papers in knowledge base
                        for paper in papers:
                            self._knowledge_base.add_paper(paper)

                        # Extract trading ideas
                        total_ideas = 0
                        total_hypotheses = 0

                        for paper in papers:
                            ideas = self._idea_extractor.extract_ideas(paper)
                            total_ideas += len(ideas)

                            for idea in ideas:
                                # Store idea
                                self._knowledge_base.add_idea(idea)

                                # Generate hypothesis if auto-generate enabled
                                if self.config.auto_generate_from_research:
                                    hypothesis = self._research_generator.generate_hypothesis(idea)

                                    # Store hypothesis
                                    self._knowledge_base.add_hypothesis(hypothesis)
                                    total_hypotheses += 1

                                    # Auto-backtest if enabled and data available
                                    if self.config.auto_backtest and self._perception:
                                        logger.debug(
                                            f"Generated research hypothesis: {hypothesis.name}"
                                        )

                        logger.info(
                            f"Research processing complete: "
                            f"{len(papers)} papers -> {total_ideas} ideas -> {total_hypotheses} hypotheses"
                        )

                        # Save knowledge base
                        self._knowledge_base.save()

                    self._last_research_fetch = now

                await asyncio.sleep(3600)  # Check every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Research loop error: {e}")
                await asyncio.sleep(3600)

    # ===== Status =====

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        status = {
            "state": self._state.value,
            "running_since": self._start_time.isoformat() if self._start_time else None,
            "market_open": is_market_open(),
            "daily_pnl": self._daily_pnl,
            "daily_pnl_pct": self._daily_pnl / self._daily_starting_equity if self._daily_starting_equity else 0,
            "positions": self._positions.get_stats(),
            "allocator": self._allocator.get_stats(),
            "registry": self._registry.get_stats(),
            "perception": self._perception.get_stats() if self._perception else None
        }

        # Add learning stats if enabled
        if self.config.learning_enabled:
            status["learning"] = {
                "performance_analyzer": self._performance_analyzer.get_stats(),
                "regime_detector": self._regime_detector.export_state(),
                "strategy_evolver": {
                    "best_genes": len(self._strategy_evolver.get_best_genes()),
                    "evolution_history": len(self._strategy_evolver.get_evolution_history())
                }
            }

        # Add research stats if enabled
        if self.config.research_enabled:
            status["research"] = {
                "knowledge_base": self._knowledge_base.get_stats(),
                "paper_fetcher": self._paper_fetcher.get_stats(),
                "idea_extractor": self._idea_extractor.get_stats(),
                "hypothesis_generator": self._research_generator.get_stats(),
                "last_fetch": self._last_research_fetch.isoformat() if self._last_research_fetch else None
            }

        # Add promoter stats
        status["promoter"] = self._promoter.get_stats()

        return status


# Singleton
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator(
    broker=None,
    config: Optional[OrchestratorConfig] = None
) -> Orchestrator:
    """Get the singleton Orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(broker, config)
    return _orchestrator
