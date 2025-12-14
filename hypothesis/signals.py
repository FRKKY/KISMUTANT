"""
SIGNAL GENERATOR - Generates Trading Signals from Hypotheses

The Signal Generator takes hypotheses and evaluates them against
current market data to produce actionable trading signals.

Signal Flow:
1. Hypothesis defines entry/exit rules
2. Signal Generator evaluates rules against current features
3. If conditions met, generate TradingSignal
4. Signals are passed to Portfolio Mind for sizing
5. Sized signals go to Execution for order placement
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple
from collections import defaultdict
import uuid

import pandas as pd
import numpy as np
from loguru import logger

from hypothesis.models import (
    Hypothesis,
    StrategyState,
    StrategyType,
    TradingSignal,
    SignalType
)
from hypothesis.registry import get_registry
from core.events import get_event_bus, Event, EventType
from memory.models import get_database, Signal as DBSignal, SignalDirection


@dataclass
class SignalGeneratorConfig:
    """Configuration for signal generation."""
    
    # Throttling
    min_signal_interval_minutes: int = 60  # Min time between signals per symbol
    max_signals_per_day: int = 10          # Max signals per strategy per day
    
    # Filtering
    min_confidence: float = 0.5
    require_volume_confirmation: bool = True
    
    # Position awareness
    check_existing_positions: bool = True


class RuleEvaluator:
    """
    Evaluates hypothesis rules against market data.
    
    Rules are stored as dictionaries with conditions.
    This class interprets and evaluates those conditions.
    """
    
    def __init__(self):
        self._operators = {
            '>': lambda a, b: a > b,
            '<': lambda a, b: a < b,
            '>=': lambda a, b: a >= b,
            '<=': lambda a, b: a <= b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            'crosses_above': self._crosses_above,
            'crosses_below': self._crosses_below,
        }
    
    def _crosses_above(
        self, 
        series1: pd.Series, 
        series2: pd.Series
    ) -> bool:
        """Check if series1 crossed above series2."""
        if len(series1) < 2 or len(series2) < 2:
            return False
        return (series1.iloc[-2] <= series2.iloc[-2] and 
                series1.iloc[-1] > series2.iloc[-1])
    
    def _crosses_below(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> bool:
        """Check if series1 crossed below series2."""
        if len(series1) < 2 or len(series2) < 2:
            return False
        return (series1.iloc[-2] >= series2.iloc[-2] and 
                series1.iloc[-1] < series2.iloc[-1])
    
    def evaluate_condition(
        self,
        condition: str,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame
    ) -> bool:
        """
        Evaluate a single condition string.
        
        Examples:
            "rsi_14 < 30"
            "close > sma_50"
            "ema_10 crosses_above ema_30"
        """
        try:
            # Parse condition
            parts = condition.split()
            
            if len(parts) == 3:
                left, operator, right = parts
                
                # Get left value
                if left in features.columns:
                    left_val = features[left].iloc[-1]
                elif left in ohlcv.columns:
                    left_val = ohlcv[left].iloc[-1]
                else:
                    try:
                        left_val = float(left)
                    except:
                        return False
                
                # Get right value
                if right in features.columns:
                    right_val = features[right].iloc[-1]
                elif right in ohlcv.columns:
                    right_val = ohlcv[right].iloc[-1]
                else:
                    try:
                        right_val = float(right)
                    except:
                        return False
                
                # Handle cross conditions specially
                if operator in ['crosses_above', 'crosses_below']:
                    if left in features.columns and right in features.columns:
                        return self._operators[operator](
                            features[left],
                            features[right]
                        )
                    return False
                
                # Regular comparison
                if operator in self._operators:
                    return self._operators[operator](left_val, right_val)
            
            return False
            
        except Exception as e:
            logger.debug(f"Condition evaluation error: {condition} - {e}")
            return False
    
    def evaluate_rules(
        self,
        rules: Dict[str, Any],
        features: pd.DataFrame,
        ohlcv: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate a set of rules.
        
        Returns:
            Tuple of (all_passed, list of passed conditions)
        """
        passed = []
        failed = []
        
        for key, value in rules.items():
            # Skip metadata keys
            if key.startswith('_'):
                continue
            
            # Handle simple condition strings
            if isinstance(value, str) and any(op in value for op in ['>', '<', '=']):
                if self.evaluate_condition(value, features, ohlcv):
                    passed.append(f"{key}: {value}")
                else:
                    failed.append(f"{key}: {value}")
            
            # Handle boolean flags
            elif isinstance(value, bool):
                # Check if feature exists and has expected value
                if key in features.columns:
                    actual = features[key].iloc[-1]
                    if bool(actual) == value:
                        passed.append(f"{key}={value}")
                    else:
                        failed.append(f"{key}={value} (got {actual})")
            
            # Handle threshold checks
            elif isinstance(value, (int, float)):
                if key in features.columns:
                    actual = features[key].iloc[-1]
                    # Assume we're checking if feature >= threshold
                    if actual >= value:
                        passed.append(f"{key} >= {value}")
                    else:
                        failed.append(f"{key} >= {value} (got {actual:.2f})")
        
        all_passed = len(failed) == 0 and len(passed) > 0
        return all_passed, passed


class SignalGenerator:
    """
    Generates trading signals from active hypotheses.
    
    Evaluates hypothesis rules against current market data
    and produces signals when conditions are met.
    """
    
    def __init__(self, config: Optional[SignalGeneratorConfig] = None):
        self.config = config or SignalGeneratorConfig()
        self._evaluator = RuleEvaluator()
        self._registry = get_registry()
        
        # Signal tracking
        self._signal_times: Dict[str, Dict[str, datetime]] = defaultdict(dict)  # hypothesis_id -> symbol -> last signal time
        self._daily_counts: Dict[str, int] = defaultdict(int)  # hypothesis_id -> signal count today
        self._last_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0)
        
        # Active positions (updated externally)
        self._positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        
        logger.info("SignalGenerator initialized")
    
    def set_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """Update current position information."""
        self._positions = positions
    
    def _reset_daily_counts(self) -> None:
        """Reset daily signal counts at market open."""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0)
        
        if self._last_reset < today_start:
            self._daily_counts.clear()
            self._last_reset = today_start
    
    def _can_generate_signal(
        self,
        hypothesis: Hypothesis,
        symbol: str
    ) -> Tuple[bool, str]:
        """Check if we can generate a signal for this hypothesis/symbol."""
        
        # Check daily limit
        if self._daily_counts[hypothesis.hypothesis_id] >= self.config.max_signals_per_day:
            return False, "Daily signal limit reached"
        
        # Check time since last signal
        last_signal = self._signal_times.get(hypothesis.hypothesis_id, {}).get(symbol)
        if last_signal:
            elapsed = (datetime.now() - last_signal).total_seconds() / 60
            if elapsed < self.config.min_signal_interval_minutes:
                return False, f"Too soon since last signal ({elapsed:.0f}m < {self.config.min_signal_interval_minutes}m)"
        
        # Check existing position
        if self.config.check_existing_positions and symbol in self._positions:
            position = self._positions[symbol]
            # Don't generate entry signals if already have position
            if position.get('quantity', 0) != 0:
                return False, "Already have position"
        
        return True, ""
    
    def _record_signal(
        self,
        hypothesis: Hypothesis,
        symbol: str,
        signal: Optional[TradingSignal] = None
    ) -> None:
        """Record that a signal was generated."""
        self._signal_times[hypothesis.hypothesis_id][symbol] = datetime.now()
        self._daily_counts[hypothesis.hypothesis_id] += 1

        # Persist to database if signal provided
        if signal:
            self._persist_signal(signal)

    def _persist_signal(self, signal: TradingSignal) -> None:
        """Persist a trading signal to the database."""
        try:
            db = get_database()
            session = db.get_session()

            # Map signal type to direction
            direction_map = {
                SignalType.LONG_ENTRY: SignalDirection.LONG,
                SignalType.SHORT_ENTRY: SignalDirection.SHORT,
                SignalType.LONG_EXIT: SignalDirection.CLOSE,
                SignalType.SHORT_EXIT: SignalDirection.CLOSE,
            }
            direction = direction_map.get(signal.signal_type, SignalDirection.HOLD)

            db_signal = DBSignal(
                signal_id=signal.signal_id,
                hypothesis_id=signal.hypothesis_id,
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                direction=direction,
                strength=signal.confidence,
                confidence=signal.confidence,
                price_at_signal=signal.price,
                features_snapshot=signal.metadata,
                was_executed=False,
                execution_reason=signal.reason
            )

            session.add(db_signal)
            session.commit()
            logger.debug(f"Persisted signal {signal.signal_id} to database")

        except Exception as e:
            logger.error(f"Failed to persist signal: {e}")
            if session:
                session.rollback()
        finally:
            if session:
                session.close()
    
    def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        symbol: str,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame
    ) -> Optional[TradingSignal]:
        """
        Evaluate a single hypothesis for a symbol.
        
        Returns:
            TradingSignal if conditions are met, None otherwise
        """
        if not hypothesis.entry_rules:
            return None
        
        # Check if we can generate signal
        can_signal, reason = self._can_generate_signal(hypothesis, symbol)
        if not can_signal:
            logger.debug(f"{hypothesis.name}/{symbol}: {reason}")
            return None
        
        # Get current price
        current_price = ohlcv['close'].iloc[-1]
        
        # Determine direction from hypothesis rules
        is_long = hypothesis.entry_rules.get('direction', 'long') == 'long'
        
        # Check for long entry
        if is_long and 'long' in hypothesis.entry_rules:
            condition = hypothesis.entry_rules['long']
            passed, evidence = self._evaluator.evaluate_rules(
                {'condition': condition} if isinstance(condition, str) else condition,
                features,
                ohlcv
            )
        elif is_long and 'condition' in hypothesis.entry_rules:
            passed, evidence = self._evaluator.evaluate_rules(
                hypothesis.entry_rules,
                features,
                ohlcv
            )
        elif not is_long and 'short' in hypothesis.entry_rules:
            condition = hypothesis.entry_rules['short']
            passed, evidence = self._evaluator.evaluate_rules(
                {'condition': condition} if isinstance(condition, str) else condition,
                features,
                ohlcv
            )
        else:
            # Try evaluating all entry rules
            passed, evidence = self._evaluator.evaluate_rules(
                hypothesis.entry_rules,
                features,
                ohlcv
            )
        
        if not passed:
            return None
        
        # Calculate stop loss and take profit
        atr = features['atr'].iloc[-1] if 'atr' in features.columns else current_price * 0.02
        
        if is_long:
            stop_loss = current_price - (atr * hypothesis.exit_rules.get('stop_loss_atr_multiple', 2.0))
            take_profit = current_price + (atr * hypothesis.exit_rules.get('take_profit_atr_multiple', 3.0))
        else:
            stop_loss = current_price + (atr * hypothesis.exit_rules.get('stop_loss_atr_multiple', 2.0))
            take_profit = current_price - (atr * hypothesis.exit_rules.get('take_profit_atr_multiple', 3.0))
        
        # Apply percentage-based stops if defined
        if hypothesis.stop_loss_pct:
            stop_loss_pct = current_price * (1 - hypothesis.stop_loss_pct) if is_long else current_price * (1 + hypothesis.stop_loss_pct)
            stop_loss = max(stop_loss, stop_loss_pct) if is_long else min(stop_loss, stop_loss_pct)
        
        if hypothesis.take_profit_pct:
            take_profit_pct = current_price * (1 + hypothesis.take_profit_pct) if is_long else current_price * (1 - hypothesis.take_profit_pct)
            take_profit = min(take_profit, take_profit_pct) if is_long else max(take_profit, take_profit_pct)
        
        # Calculate confidence based on evidence strength
        base_confidence = hypothesis.parameters.get('source_confidence', 0.5)
        evidence_boost = len(evidence) * 0.05
        confidence = min(0.95, base_confidence + evidence_boost)
        
        if confidence < self.config.min_confidence:
            return None
        
        # Create signal
        signal = TradingSignal(
            signal_id=f"sig_{uuid.uuid4().hex[:12]}",
            hypothesis_id=hypothesis.hypothesis_id,
            symbol=symbol,
            timestamp=datetime.now(),
            signal_type=SignalType.LONG_ENTRY if is_long else SignalType.SHORT_ENTRY,
            direction=1 if is_long else -1,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=f"Entry conditions met: {', '.join(evidence)}",
            metadata={
                "hypothesis_name": hypothesis.name,
                "strategy_type": hypothesis.strategy_type.value,
                "atr": atr,
                "evidence": evidence
            }
        )
        
        # Record signal and persist to database
        self._record_signal(hypothesis, symbol, signal)

        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.SIGNAL_GENERATED,
            source="signal_generator",
            payload=signal.to_dict()
        ))
        
        logger.info(
            f"Signal generated: {hypothesis.name} {signal.signal_type.value} "
            f"{symbol} @ {current_price:.2f}"
        )
        
        return signal
    
    def check_exit_conditions(
        self,
        hypothesis: Hypothesis,
        symbol: str,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        entry_price: float,
        entry_time: datetime,
        is_long: bool
    ) -> Optional[TradingSignal]:
        """
        Check if exit conditions are met for an open position.
        
        Returns:
            Exit signal if conditions met, None otherwise
        """
        current_price = ohlcv['close'].iloc[-1]
        
        exit_rules = hypothesis.exit_rules
        reasons = []
        
        # Check stop loss
        atr = features['atr'].iloc[-1] if 'atr' in features.columns else entry_price * 0.02
        stop_loss_atr = exit_rules.get('stop_loss_atr', 2.0)
        
        if is_long:
            stop_price = entry_price - (atr * stop_loss_atr)
            if current_price <= stop_price:
                reasons.append(f"Stop loss hit ({current_price:.2f} <= {stop_price:.2f})")
        else:
            stop_price = entry_price + (atr * stop_loss_atr)
            if current_price >= stop_price:
                reasons.append(f"Stop loss hit ({current_price:.2f} >= {stop_price:.2f})")
        
        # Check take profit
        take_profit_atr = exit_rules.get('take_profit_atr_multiple', 3.0)
        
        if is_long:
            target = entry_price + (atr * take_profit_atr)
            if current_price >= target:
                reasons.append(f"Take profit hit ({current_price:.2f} >= {target:.2f})")
        else:
            target = entry_price - (atr * take_profit_atr)
            if current_price <= target:
                reasons.append(f"Take profit hit ({current_price:.2f} <= {target:.2f})")
        
        # Check max holding period
        max_days = exit_rules.get('max_holding_days', hypothesis.max_holding_days)
        if max_days:
            holding_days = (datetime.now() - entry_time).days
            if holding_days >= max_days:
                reasons.append(f"Max holding period reached ({holding_days} days)")
        
        # Check exit condition rules
        if 'exit_condition' in exit_rules:
            passed, _ = self._evaluator.evaluate_rules(
                {'condition': exit_rules['exit_condition']},
                features,
                ohlcv
            )
            if passed:
                reasons.append(f"Exit condition met: {exit_rules['exit_condition']}")
        
        # Check reverse cross for MA strategies
        if exit_rules.get('exit_on_reverse_cross', False):
            # Check for reverse cross
            if is_long and 'macd_cross' in features.columns:
                if features['macd_cross'].iloc[-1] == 0 and features['macd_cross'].iloc[-2] == 1:
                    reasons.append("MACD bearish cross")
            elif not is_long and 'macd_cross' in features.columns:
                if features['macd_cross'].iloc[-1] == 1 and features['macd_cross'].iloc[-2] == 0:
                    reasons.append("MACD bullish cross")
        
        if reasons:
            signal_type = SignalType.LONG_EXIT if is_long else SignalType.SHORT_EXIT
            
            # Determine if it's a stop loss or take profit
            if 'Stop loss' in reasons[0]:
                signal_type = SignalType.STOP_LOSS
            elif 'Take profit' in reasons[0]:
                signal_type = SignalType.TAKE_PROFIT
            
            signal = TradingSignal(
                signal_id=f"sig_{uuid.uuid4().hex[:12]}",
                hypothesis_id=hypothesis.hypothesis_id,
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=signal_type,
                direction=0,  # Exit
                price=current_price,
                confidence=0.9,
                reason="; ".join(reasons),
                metadata={
                    "entry_price": entry_price,
                    "pnl_pct": ((current_price / entry_price) - 1) * (1 if is_long else -1) * 100
                }
            )

            # Persist exit signal to database
            self._persist_signal(signal)

            get_event_bus().publish(Event(
                event_type=EventType.SIGNAL_GENERATED,
                source="signal_generator",
                payload=signal.to_dict()
            ))

            return signal
        
        return None
    
    def generate_signals(
        self,
        market_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> List[TradingSignal]:
        """
        Generate signals for all active strategies.
        
        Args:
            market_data: Dict of symbol -> (ohlcv_df, features_df)
        
        Returns:
            List of generated signals
        """
        self._reset_daily_counts()
        
        signals = []
        
        # Get active strategies (paper trading and live)
        active_states = [StrategyState.PAPER_TRADING, StrategyState.LIVE]
        
        for state in active_states:
            for hypothesis in self._registry.get_all(state):
                for symbol in hypothesis.symbols:
                    if symbol not in market_data:
                        continue
                    
                    ohlcv, features = market_data[symbol]
                    
                    if ohlcv.empty or features.empty:
                        continue
                    
                    try:
                        signal = self.evaluate_hypothesis(
                            hypothesis,
                            symbol,
                            ohlcv,
                            features
                        )
                        
                        if signal:
                            signals.append(signal)
                    
                    except Exception as e:
                        logger.error(
                            f"Error evaluating {hypothesis.name}/{symbol}: {e}"
                        )
        
        logger.info(f"Generated {len(signals)} signals from {len(self._registry.get_all())} strategies")
        
        return signals
    
    def get_stats(self) -> Dict[str, Any]:
        """Get signal generator statistics."""
        return {
            "daily_counts": dict(self._daily_counts),
            "total_signals_today": sum(self._daily_counts.values()),
            "positions_tracked": len(self._positions),
            "config": {
                "min_interval_minutes": self.config.min_signal_interval_minutes,
                "max_signals_per_day": self.config.max_signals_per_day,
                "min_confidence": self.config.min_confidence
            }
        }


# Singleton
_signal_generator: Optional[SignalGenerator] = None


def get_signal_generator(config: Optional[SignalGeneratorConfig] = None) -> SignalGenerator:
    """Get the singleton SignalGenerator instance."""
    global _signal_generator
    if _signal_generator is None:
        _signal_generator = SignalGenerator(config)
    return _signal_generator
