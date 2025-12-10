"""
BACKTESTER - Historical Strategy Validation

Simulates trading strategies on historical data to validate
hypotheses before paper trading.

Backtest Flow:
1. Load historical data for strategy symbols
2. Compute features at each point in time
3. Evaluate entry/exit rules
4. Simulate trades with realistic assumptions
5. Calculate performance metrics
6. Return results for promotion decision

Realistic Assumptions:
- Slippage modeling
- Commission costs
- No look-ahead bias (point-in-time features)
- Fill delays
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
from copy import deepcopy

from loguru import logger

from hypothesis.models import (
    Hypothesis,
    StrategyState,
    PerformanceMetrics,
    TradingSignal,
    SignalType
)
from hypothesis.signals import RuleEvaluator
from validation.metrics import (
    MetricsCalculator,
    TradeRecord,
    get_metrics_calculator
)
from core.events import get_event_bus, Event, EventType


class BacktestMode(str, Enum):
    """Backtesting modes."""
    VECTORIZED = "vectorized"    # Fast, less realistic
    EVENT_DRIVEN = "event_driven"  # Slower, more realistic


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Capital
    initial_capital: float = 10_000_000  # ₩10M
    
    # Costs
    commission_pct: float = 0.00015      # 0.015% (typical Korean broker)
    slippage_pct: float = 0.001          # 0.1% slippage estimate
    
    # Position sizing
    position_size_pct: float = 0.10      # 10% per position
    max_positions: int = 5
    
    # Risk management
    use_stop_loss: bool = True
    use_take_profit: bool = True
    
    # Execution
    fill_delay_bars: int = 1             # Enter on next bar's open
    
    # Data
    warmup_periods: int = 50             # Periods for indicator calculation


@dataclass
class BacktestPosition:
    """Position during backtest."""
    
    symbol: str
    side: str  # "long" or "short"
    entry_time: datetime
    entry_price: float
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    high_water_mark: float = 0.0
    
    def update(self, price: float) -> None:
        """Update position with new price."""
        self.current_price = price
        
        if self.side == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
        
        # Track high water mark for trailing stops
        if self.side == "long":
            self.high_water_mark = max(self.high_water_mark, price)
        else:
            if self.high_water_mark == 0:
                self.high_water_mark = price
            self.high_water_mark = min(self.high_water_mark, price)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    hypothesis_id: str
    hypothesis_name: str
    
    # Performance
    metrics: Dict[str, Any]
    
    # Equity curve
    equity_curve: pd.Series
    
    # Trades
    trades: List[TradeRecord]
    
    # Summary
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    
    # Validation
    passed_criteria: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    
    # Meta
    config: Optional[BacktestConfig] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    run_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_name": self.hypothesis_name,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "passed_criteria": self.passed_criteria,
            "failure_reasons": self.failure_reasons,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "metrics": self.metrics
        }
    
    def to_performance_metrics(self) -> PerformanceMetrics:
        """Convert to PerformanceMetrics for hypothesis update."""
        return PerformanceMetrics(
            total_return=self.total_return,
            annualized_return=self.metrics.get("annualized_return", 0),
            sharpe_ratio=self.sharpe_ratio,
            sortino_ratio=self.metrics.get("sortino_ratio", 0),
            max_drawdown=self.max_drawdown,
            volatility=self.metrics.get("volatility", 0),
            total_trades=self.total_trades,
            winning_trades=self.metrics.get("winning_trades", 0),
            losing_trades=self.metrics.get("losing_trades", 0),
            win_rate=self.win_rate,
            avg_win=self.metrics.get("avg_win", 0),
            avg_loss=self.metrics.get("avg_loss", 0),
            profit_factor=self.metrics.get("profit_factor", 0),
            avg_holding_period_hours=self.metrics.get("avg_holding_hours", 0),
            max_consecutive_wins=self.metrics.get("max_consecutive_wins", 0),
            max_consecutive_losses=self.metrics.get("max_consecutive_losses", 0),
            t_statistic=self.metrics.get("t_statistic"),
            p_value=self.metrics.get("p_value"),
            start_date=self.start_date,
            end_date=self.end_date,
            trading_days=self.metrics.get("trading_days", 0)
        )


class Backtester:
    """
    Event-driven backtesting engine.
    
    Simulates strategy execution on historical data
    with realistic trading assumptions.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._evaluator = RuleEvaluator()
        self._metrics = get_metrics_calculator()
        
        # State during backtest
        self._capital: float = 0.0
        self._positions: Dict[str, BacktestPosition] = {}
        self._pending_entries: List[Dict] = []
        self._pending_exits: List[Dict] = []
        self._trades: List[TradeRecord] = []
        self._equity_history: List[Tuple[datetime, float]] = []
        
        logger.info("Backtester initialized")
    
    def _reset(self) -> None:
        """Reset state for new backtest."""
        self._capital = self.config.initial_capital
        self._positions.clear()
        self._pending_entries.clear()
        self._pending_exits.clear()
        self._trades.clear()
        self._equity_history.clear()
    
    def _get_position_size(
        self,
        price: float,
        available_capital: float
    ) -> int:
        """Calculate position size in shares."""
        target_value = available_capital * self.config.position_size_pct
        
        # Limit by max positions
        if len(self._positions) >= self.config.max_positions:
            return 0
        
        shares = int(target_value / price)
        return max(0, shares)
    
    def _apply_slippage(
        self,
        price: float,
        is_buy: bool
    ) -> float:
        """Apply slippage to price."""
        slippage = price * self.config.slippage_pct
        
        if is_buy:
            return price + slippage
        else:
            return price - slippage
    
    def _calculate_commission(self, value: float) -> float:
        """Calculate commission for a trade."""
        return value * self.config.commission_pct
    
    def _get_equity(self) -> float:
        """Calculate current equity."""
        equity = self._capital
        
        for position in self._positions.values():
            equity += position.unrealized_pnl
        
        return equity
    
    def _check_entry_conditions(
        self,
        hypothesis: Hypothesis,
        symbol: str,
        bar_idx: int,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame
    ) -> Optional[Dict]:
        """Check if entry conditions are met."""
        
        # Skip if already have position
        if symbol in self._positions:
            return None
        
        # Get data up to current bar (no look-ahead)
        current_ohlcv = ohlcv.iloc[:bar_idx + 1]
        current_features = features.iloc[:bar_idx + 1]
        
        if len(current_features) < 2:
            return None
        
        # Determine direction
        is_long = hypothesis.entry_rules.get('direction', 'long') == 'long'
        
        # Evaluate entry rules
        entry_rules = hypothesis.entry_rules.copy()
        
        # Check specific direction rules first
        if is_long and 'long' in entry_rules:
            rules_to_check = entry_rules['long']
            if isinstance(rules_to_check, str):
                rules_to_check = {'condition': rules_to_check}
        elif not is_long and 'short' in entry_rules:
            rules_to_check = entry_rules['short']
            if isinstance(rules_to_check, str):
                rules_to_check = {'condition': rules_to_check}
        else:
            rules_to_check = entry_rules
        
        passed, evidence = self._evaluator.evaluate_rules(
            rules_to_check,
            current_features,
            current_ohlcv
        )
        
        if passed:
            return {
                "symbol": symbol,
                "side": "long" if is_long else "short",
                "bar_idx": bar_idx,
                "evidence": evidence
            }
        
        return None
    
    def _check_exit_conditions(
        self,
        hypothesis: Hypothesis,
        position: BacktestPosition,
        bar_idx: int,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame
    ) -> Optional[str]:
        """
        Check if exit conditions are met.
        
        Returns exit reason or None.
        """
        current_bar = ohlcv.iloc[bar_idx]
        current_features = features.iloc[:bar_idx + 1]
        price = current_bar['close']
        
        # Check stop loss
        if self.config.use_stop_loss and position.stop_loss:
            if position.side == "long" and price <= position.stop_loss:
                return "stop_loss"
            elif position.side == "short" and price >= position.stop_loss:
                return "stop_loss"
        
        # Check take profit
        if self.config.use_take_profit and position.take_profit:
            if position.side == "long" and price >= position.take_profit:
                return "take_profit"
            elif position.side == "short" and price <= position.take_profit:
                return "take_profit"
        
        # Check max holding period
        exit_rules = hypothesis.exit_rules
        max_days = exit_rules.get('max_holding_days', hypothesis.max_holding_days)
        
        if max_days:
            entry_idx = ohlcv.index.get_loc(position.entry_time) if position.entry_time in ohlcv.index else 0
            holding_bars = bar_idx - entry_idx
            if holding_bars >= max_days:
                return "max_holding_period"
        
        # Check rule-based exit
        if 'exit_condition' in exit_rules:
            passed, _ = self._evaluator.evaluate_rules(
                {'condition': exit_rules['exit_condition']},
                current_features,
                ohlcv.iloc[:bar_idx + 1]
            )
            if passed:
                return "exit_condition"
        
        return None
    
    def _execute_entry(
        self,
        entry: Dict,
        bar: pd.Series,
        hypothesis: Hypothesis
    ) -> Optional[BacktestPosition]:
        """Execute a pending entry."""
        symbol = entry["symbol"]
        side = entry["side"]
        
        # Entry price with slippage (use open of execution bar)
        is_buy = side == "long"
        entry_price = self._apply_slippage(bar['open'], is_buy)
        
        # Calculate position size
        available = self._capital
        quantity = self._get_position_size(entry_price, available)
        
        if quantity <= 0:
            return None
        
        # Calculate cost and commission
        cost = entry_price * quantity
        commission = self._calculate_commission(cost)
        
        if cost + commission > self._capital:
            return None
        
        # Deduct capital
        self._capital -= (cost + commission)
        
        # Calculate stop loss and take profit
        atr = bar.get('atr', entry_price * 0.02)
        if isinstance(atr, pd.Series):
            atr = atr.iloc[-1] if len(atr) > 0 else entry_price * 0.02
        
        exit_rules = hypothesis.exit_rules
        
        if side == "long":
            stop_loss = entry_price - (atr * exit_rules.get('stop_loss_atr_multiple', 2.0))
            take_profit = entry_price + (atr * exit_rules.get('take_profit_atr_multiple', 3.0))
        else:
            stop_loss = entry_price + (atr * exit_rules.get('stop_loss_atr_multiple', 2.0))
            take_profit = entry_price - (atr * exit_rules.get('take_profit_atr_multiple', 3.0))
        
        # Apply percentage-based stops if defined
        if hypothesis.stop_loss_pct:
            if side == "long":
                stop_loss = max(stop_loss, entry_price * (1 - hypothesis.stop_loss_pct))
            else:
                stop_loss = min(stop_loss, entry_price * (1 + hypothesis.stop_loss_pct))
        
        if hypothesis.take_profit_pct:
            if side == "long":
                take_profit = min(take_profit, entry_price * (1 + hypothesis.take_profit_pct))
            else:
                take_profit = max(take_profit, entry_price * (1 - hypothesis.take_profit_pct))
        
        position = BacktestPosition(
            symbol=symbol,
            side=side,
            entry_time=bar.name,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss if self.config.use_stop_loss else None,
            take_profit=take_profit if self.config.use_take_profit else None,
            current_price=entry_price,
            high_water_mark=entry_price
        )
        
        self._positions[symbol] = position
        
        return position
    
    def _execute_exit(
        self,
        position: BacktestPosition,
        bar: pd.Series,
        reason: str
    ) -> TradeRecord:
        """Execute an exit and record the trade."""
        # Exit price with slippage
        is_sell = position.side == "long"
        exit_price = self._apply_slippage(bar['open'], not is_sell)
        
        # Calculate P&L
        if position.side == "long":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        # Deduct exit commission
        commission = self._calculate_commission(exit_price * position.quantity)
        pnl -= commission
        
        # Return capital
        self._capital += (exit_price * position.quantity) - commission + pnl
        
        # Calculate holding period
        entry_time = position.entry_time
        exit_time = bar.name
        
        if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
            holding_hours = (exit_time - entry_time).total_seconds() / 3600
        else:
            holding_hours = 0
        
        # Create trade record
        trade = TradeRecord(
            symbol=position.symbol,
            side=position.side,
            entry_time=entry_time,
            entry_price=position.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl / (position.entry_price * position.quantity),
            holding_period_hours=holding_hours,
            exit_reason=reason
        )
        
        self._trades.append(trade)
        
        # Remove position
        del self._positions[position.symbol]
        
        return trade
    
    def run(
        self,
        hypothesis: Hypothesis,
        data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest for a hypothesis.
        
        Args:
            hypothesis: Strategy to test
            data: Dict of symbol -> (ohlcv_df, features_df)
            start_date: Optional start date
            end_date: Optional end date
        
        Returns:
            BacktestResult with performance metrics
        """
        import time
        start_time = time.time()
        
        self._reset()
        
        # Get symbols to trade
        symbols = hypothesis.symbols
        if not symbols:
            symbols = list(data.keys())
        
        # Filter to available data
        symbols = [s for s in symbols if s in data]
        
        if not symbols:
            logger.warning(f"No data available for hypothesis symbols")
            return BacktestResult(
                hypothesis_id=hypothesis.hypothesis_id,
                hypothesis_name=hypothesis.name,
                metrics={},
                equity_curve=pd.Series(),
                trades=[],
                passed_criteria=False,
                failure_reasons=["No data available"]
            )
        
        # Use first symbol for date alignment
        primary_symbol = symbols[0]
        ohlcv, features = data[primary_symbol]
        
        # Apply date filters
        if start_date:
            ohlcv = ohlcv[ohlcv.index >= start_date]
            features = features[features.index >= start_date]
        if end_date:
            ohlcv = ohlcv[ohlcv.index <= end_date]
            features = features[features.index <= end_date]
        
        # Skip warmup period
        warmup = self.config.warmup_periods
        if len(ohlcv) <= warmup:
            return BacktestResult(
                hypothesis_id=hypothesis.hypothesis_id,
                hypothesis_name=hypothesis.name,
                metrics={},
                equity_curve=pd.Series(),
                trades=[],
                passed_criteria=False,
                failure_reasons=["Insufficient data after warmup"]
            )
        
        # Main simulation loop
        logger.info(f"Running backtest for {hypothesis.name} on {len(symbols)} symbols")
        
        for bar_idx in range(warmup, len(ohlcv)):
            current_bar = ohlcv.iloc[bar_idx]
            current_time = ohlcv.index[bar_idx]
            
            # Process pending entries from previous bar
            entries_to_process = self._pending_entries.copy()
            self._pending_entries.clear()
            
            for entry in entries_to_process:
                self._execute_entry(entry, current_bar, hypothesis)
            
            # Process pending exits
            exits_to_process = self._pending_exits.copy()
            self._pending_exits.clear()
            
            for exit_info in exits_to_process:
                position = self._positions.get(exit_info["symbol"])
                if position:
                    self._execute_exit(position, current_bar, exit_info["reason"])
            
            # Update position prices
            for symbol in list(self._positions.keys()):
                if symbol in data:
                    sym_ohlcv, sym_features = data[symbol]
                    if current_time in sym_ohlcv.index:
                        price = sym_ohlcv.loc[current_time, 'close']
                        self._positions[symbol].update(price)
            
            # Check exits
            for symbol, position in list(self._positions.items()):
                if symbol in data:
                    sym_ohlcv, sym_features = data[symbol]
                    if current_time in sym_ohlcv.index:
                        sym_bar_idx = sym_ohlcv.index.get_loc(current_time)
                        exit_reason = self._check_exit_conditions(
                            hypothesis, position, sym_bar_idx, sym_ohlcv, sym_features
                        )
                        if exit_reason:
                            self._pending_exits.append({
                                "symbol": symbol,
                                "reason": exit_reason
                            })
            
            # Check entries
            for symbol in symbols:
                if symbol in data and symbol not in self._positions:
                    sym_ohlcv, sym_features = data[symbol]
                    if current_time in sym_ohlcv.index:
                        sym_bar_idx = sym_ohlcv.index.get_loc(current_time)
                        entry = self._check_entry_conditions(
                            hypothesis, symbol, sym_bar_idx, sym_ohlcv, sym_features
                        )
                        if entry:
                            self._pending_entries.append(entry)
            
            # Record equity
            equity = self._get_equity()
            self._equity_history.append((current_time, equity))
        
        # Close any remaining positions at end
        final_bar = ohlcv.iloc[-1]
        for position in list(self._positions.values()):
            self._execute_exit(position, final_bar, "backtest_end")
        
        # Build equity curve
        equity_curve = pd.Series(
            [e[1] for e in self._equity_history],
            index=[e[0] for e in self._equity_history]
        )
        
        # Calculate metrics
        metrics = self._metrics.calculate_all(equity_curve, self._trades)
        
        # Check promotion criteria
        passed, failures = self._check_promotion_criteria(metrics)
        
        run_time = time.time() - start_time
        
        result = BacktestResult(
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis_name=hypothesis.name,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=self._trades.copy(),
            total_return=metrics.get("total_return", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            win_rate=metrics.get("win_rate", 0),
            total_trades=metrics.get("total_trades", 0),
            passed_criteria=passed,
            failure_reasons=failures,
            config=self.config,
            start_date=ohlcv.index[warmup],
            end_date=ohlcv.index[-1],
            run_time_seconds=run_time
        )
        
        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.BACKTEST_COMPLETED,
            source="backtester",
            payload=result.to_dict()
        ))
        
        logger.info(
            f"Backtest complete: {hypothesis.name} - "
            f"Return: {result.total_return:.1%}, "
            f"Sharpe: {result.sharpe_ratio:.2f}, "
            f"Trades: {result.total_trades}, "
            f"Passed: {passed}"
        )
        
        return result
    
    def _check_promotion_criteria(
        self,
        metrics: Dict[str, Any],
        min_sharpe: float = 1.0,
        max_drawdown: float = 0.15,
        min_trades: int = 30,
        min_profit_factor: float = 1.5
    ) -> Tuple[bool, List[str]]:
        """Check if backtest passes promotion criteria."""
        failures = []
        
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe < min_sharpe:
            failures.append(f"Sharpe {sharpe:.2f} < {min_sharpe}")
        
        max_dd = metrics.get("max_drawdown", 1)
        if max_dd > max_drawdown:
            failures.append(f"Max DD {max_dd:.1%} > {max_drawdown:.0%}")
        
        trades = metrics.get("total_trades", 0)
        if trades < min_trades:
            failures.append(f"Trades {trades} < {min_trades}")
        
        pf = metrics.get("profit_factor", 0)
        if pf < min_profit_factor:
            failures.append(f"Profit factor {pf:.2f} < {min_profit_factor}")
        
        return len(failures) == 0, failures
    
    def run_batch(
        self,
        hypotheses: List[Hypothesis],
        data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> List[BacktestResult]:
        """Run backtests for multiple hypotheses."""
        results = []
        
        for hypothesis in hypotheses:
            result = self.run(hypothesis, data)
            results.append(result)
        
        # Sort by Sharpe ratio
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        
        return results


# Singleton
_backtester: Optional[Backtester] = None


def get_backtester(config: Optional[BacktestConfig] = None) -> Backtester:
    """Get the singleton Backtester instance."""
    global _backtester
    if _backtester is None:
        _backtester = Backtester(config)
    return _backtester
