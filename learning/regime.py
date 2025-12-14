"""
REGIME DETECTOR - Identify market regimes and adapt strategies

Detects market conditions:
- Trending vs ranging
- High vs low volatility
- Bull vs bear markets
- Risk-on vs risk-off
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(str, Enum):
    """Market regime classifications."""
    # Trend regimes
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"

    # Volatility regimes
    HIGH_VOLATILITY = "high_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    LOW_VOLATILITY = "low_volatility"

    # Risk regimes
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"

    # Special conditions
    CRISIS = "crisis"
    EUPHORIA = "euphoria"
    UNCERTAINTY = "uncertainty"


@dataclass
class RegimeState:
    """Current regime state."""
    primary_regime: MarketRegime
    secondary_regime: Optional[MarketRegime]
    confidence: float
    duration_days: int
    regime_strength: float  # -1 to 1
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_regime": self.primary_regime.value,
            "secondary_regime": self.secondary_regime.value if self.secondary_regime else None,
            "confidence": self.confidence,
            "duration_days": self.duration_days,
            "regime_strength": self.regime_strength,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class RegimeTransition:
    """A regime change event."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_date: datetime
    confidence: float
    indicators_triggered: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_regime.value,
            "to": self.to_regime.value,
            "date": self.transition_date.isoformat(),
            "confidence": self.confidence,
            "indicators": self.indicators_triggered,
        }


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Trend detection
    trend_lookback: int = 60         # Days for trend analysis
    trend_threshold: float = 0.05    # 5% move for trend confirmation

    # Volatility detection
    vol_lookback: int = 20           # Days for volatility
    vol_high_percentile: float = 0.8  # Above 80th percentile = high vol
    vol_low_percentile: float = 0.2   # Below 20th percentile = low vol

    # Regime change detection
    min_regime_duration: int = 5     # Min days before regime can change
    transition_threshold: float = 0.7 # Confidence threshold for transition


class RegimeDetector:
    """
    Detects and tracks market regimes.

    Uses multiple indicators:
    - Price trends (SMA crossovers, momentum)
    - Volatility (ATR, realized vol)
    - Breadth (advance/decline)
    - Sentiment proxies (VIX-like)
    """

    _instance: Optional['RegimeDetector'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[RegimeConfig] = None):
        if self._initialized:
            return

        self.config = config or RegimeConfig()
        self._current_regime: Optional[RegimeState] = None
        self._regime_history: List[RegimeState] = []
        self._transitions: List[RegimeTransition] = []
        self._last_analysis: Optional[datetime] = None

        self._initialized = True
        logger.info("RegimeDetector initialized")

    def detect_regime(
        self,
        price_data: pd.DataFrame,
        volatility_data: Optional[pd.Series] = None
    ) -> RegimeState:
        """
        Detect current market regime from price data.

        Args:
            price_data: DataFrame with 'close' column (and optionally 'high', 'low')
            volatility_data: Optional external volatility measure

        Returns:
            Current RegimeState
        """
        if len(price_data) < self.config.trend_lookback:
            return RegimeState(
                primary_regime=MarketRegime.UNCERTAINTY,
                secondary_regime=None,
                confidence=0.3,
                duration_days=0,
                regime_strength=0.0,
            )

        # Detect trend regime
        trend_regime, trend_strength, trend_conf = self._detect_trend(price_data)

        # Detect volatility regime
        vol_regime, vol_strength, vol_conf = self._detect_volatility(price_data, volatility_data)

        # Combine into overall regime
        primary, secondary, confidence = self._combine_regimes(
            trend_regime, trend_conf,
            vol_regime, vol_conf
        )

        # Calculate duration
        duration = self._calculate_duration(primary)

        # Create new state
        new_state = RegimeState(
            primary_regime=primary,
            secondary_regime=secondary,
            confidence=confidence,
            duration_days=duration,
            regime_strength=(trend_strength + vol_strength) / 2,
        )

        # Check for regime transition
        if self._current_regime and self._current_regime.primary_regime != primary:
            if confidence >= self.config.transition_threshold:
                self._record_transition(self._current_regime.primary_regime, primary, confidence)

        # Update current state
        self._current_regime = new_state
        self._regime_history.append(new_state)
        self._last_analysis = datetime.utcnow()

        return new_state

    def _detect_trend(
        self,
        data: pd.DataFrame
    ) -> Tuple[MarketRegime, float, float]:
        """Detect trend regime from price data."""
        close = data['close']

        # Calculate SMAs
        sma_short = close.rolling(20).mean()
        sma_long = close.rolling(50).mean()

        # Current values
        current_price = close.iloc[-1]
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]

        # Trend indicators
        indicators = []

        # 1. Price vs SMAs
        price_vs_sma = (current_price - current_sma_long) / current_sma_long
        indicators.append(np.clip(price_vs_sma * 10, -1, 1))

        # 2. SMA crossover
        sma_diff = (current_sma_short - current_sma_long) / current_sma_long
        indicators.append(np.clip(sma_diff * 20, -1, 1))

        # 3. Price momentum (rate of change)
        if len(close) >= self.config.trend_lookback:
            roc = (current_price - close.iloc[-self.config.trend_lookback]) / close.iloc[-self.config.trend_lookback]
            indicators.append(np.clip(roc * 5, -1, 1))

        # 4. Higher highs / lower lows
        recent = close.iloc[-20:]
        highs_trend = (recent.rolling(5).max().iloc[-1] > recent.rolling(5).max().iloc[-10])
        lows_trend = (recent.rolling(5).min().iloc[-1] > recent.rolling(5).min().iloc[-10])
        if highs_trend and lows_trend:
            indicators.append(1.0)  # Uptrend
        elif not highs_trend and not lows_trend:
            indicators.append(-1.0)  # Downtrend
        else:
            indicators.append(0.0)  # Mixed

        # Aggregate
        avg_indicator = np.mean(indicators)
        strength = abs(avg_indicator)
        confidence = min(strength + 0.3, 1.0)

        if avg_indicator > self.config.trend_threshold:
            regime = MarketRegime.BULL_TREND
        elif avg_indicator < -self.config.trend_threshold:
            regime = MarketRegime.BEAR_TREND
        else:
            regime = MarketRegime.SIDEWAYS

        return regime, avg_indicator, confidence

    def _detect_volatility(
        self,
        data: pd.DataFrame,
        external_vol: Optional[pd.Series] = None
    ) -> Tuple[MarketRegime, float, float]:
        """Detect volatility regime."""
        close = data['close']

        # Calculate realized volatility
        returns = close.pct_change().dropna()
        current_vol = returns.iloc[-self.config.vol_lookback:].std() * np.sqrt(252)

        # Historical volatility distribution
        rolling_vol = returns.rolling(self.config.vol_lookback).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 50:
            return MarketRegime.NORMAL_VOLATILITY, 0.0, 0.5

        # Percentile rank
        vol_percentile = (rolling_vol < current_vol).mean()

        # ATR if available
        if 'high' in data.columns and 'low' in data.columns:
            tr = pd.concat([
                data['high'] - data['low'],
                abs(data['high'] - close.shift(1)),
                abs(data['low'] - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_percentile = (atr < atr.iloc[-1]).mean()
            vol_percentile = (vol_percentile + atr_percentile) / 2

        # Determine regime
        if vol_percentile >= self.config.vol_high_percentile:
            regime = MarketRegime.HIGH_VOLATILITY
            strength = (vol_percentile - 0.5) * 2
        elif vol_percentile <= self.config.vol_low_percentile:
            regime = MarketRegime.LOW_VOLATILITY
            strength = (0.5 - vol_percentile) * 2
        else:
            regime = MarketRegime.NORMAL_VOLATILITY
            strength = 0.0

        confidence = 0.6 + abs(vol_percentile - 0.5) * 0.8

        return regime, strength, confidence

    def _combine_regimes(
        self,
        trend_regime: MarketRegime,
        trend_conf: float,
        vol_regime: MarketRegime,
        vol_conf: float
    ) -> Tuple[MarketRegime, Optional[MarketRegime], float]:
        """Combine trend and volatility regimes."""
        # Primary is the higher confidence one
        if trend_conf >= vol_conf:
            primary = trend_regime
            secondary = vol_regime
            confidence = trend_conf
        else:
            primary = vol_regime
            secondary = trend_regime
            confidence = vol_conf

        # Special cases
        if trend_regime == MarketRegime.BEAR_TREND and vol_regime == MarketRegime.HIGH_VOLATILITY:
            if trend_conf > 0.7 and vol_conf > 0.7:
                primary = MarketRegime.CRISIS
                confidence = max(trend_conf, vol_conf)

        if trend_regime == MarketRegime.BULL_TREND and vol_regime == MarketRegime.LOW_VOLATILITY:
            if trend_conf > 0.8:
                primary = MarketRegime.EUPHORIA
                confidence = max(trend_conf, vol_conf)

        return primary, secondary, confidence

    def _calculate_duration(self, current: MarketRegime) -> int:
        """Calculate how long we've been in current regime."""
        duration = 1

        for state in reversed(self._regime_history):
            if state.primary_regime == current:
                duration += 1
            else:
                break

        return duration

    def _record_transition(
        self,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
        confidence: float
    ) -> None:
        """Record a regime transition."""
        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_date=datetime.utcnow(),
            confidence=confidence,
            indicators_triggered=["trend", "volatility"],
        )
        self._transitions.append(transition)

        logger.info(f"Regime transition: {from_regime.value} -> {to_regime.value}")

    def get_current_regime(self) -> Optional[RegimeState]:
        """Get current regime state."""
        return self._current_regime

    def get_regime_history(self, days: int = 30) -> List[RegimeState]:
        """Get regime history for last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [r for r in self._regime_history if r.detected_at >= cutoff]

    def get_transitions(self) -> List[RegimeTransition]:
        """Get all regime transitions."""
        return self._transitions.copy()

    def get_strategy_adjustment(
        self,
        strategy_type: str
    ) -> Dict[str, Any]:
        """
        Get recommended adjustments for a strategy type based on current regime.

        Returns dict with:
        - position_size_mult: Multiplier for position sizing
        - trade_frequency_mult: Multiplier for trade frequency
        - stop_loss_mult: Multiplier for stop loss distance
        - take_profit_mult: Multiplier for take profit
        - confidence: How confident we are in recommendations
        """
        if not self._current_regime:
            return {
                "position_size_mult": 1.0,
                "trade_frequency_mult": 1.0,
                "stop_loss_mult": 1.0,
                "take_profit_mult": 1.0,
                "confidence": 0.0,
            }

        regime = self._current_regime.primary_regime
        adjustments = {
            "position_size_mult": 1.0,
            "trade_frequency_mult": 1.0,
            "stop_loss_mult": 1.0,
            "take_profit_mult": 1.0,
            "confidence": self._current_regime.confidence,
        }

        # Trend-following strategies
        if strategy_type in ["momentum", "trend_following"]:
            if regime == MarketRegime.BULL_TREND:
                adjustments["position_size_mult"] = 1.3
                adjustments["trade_frequency_mult"] = 1.2
            elif regime == MarketRegime.BEAR_TREND:
                adjustments["position_size_mult"] = 0.5
            elif regime == MarketRegime.SIDEWAYS:
                adjustments["trade_frequency_mult"] = 0.5
            elif regime == MarketRegime.HIGH_VOLATILITY:
                adjustments["stop_loss_mult"] = 1.5
                adjustments["position_size_mult"] = 0.7

        # Mean reversion strategies
        elif strategy_type == "mean_reversion":
            if regime == MarketRegime.SIDEWAYS:
                adjustments["position_size_mult"] = 1.3
                adjustments["trade_frequency_mult"] = 1.5
            elif regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
                adjustments["position_size_mult"] = 0.5
                adjustments["trade_frequency_mult"] = 0.5
            elif regime == MarketRegime.HIGH_VOLATILITY:
                adjustments["take_profit_mult"] = 1.5

        # Breakout strategies
        elif strategy_type == "breakout":
            if regime == MarketRegime.LOW_VOLATILITY:
                adjustments["trade_frequency_mult"] = 1.5  # Compression before expansion
            elif regime == MarketRegime.HIGH_VOLATILITY:
                adjustments["stop_loss_mult"] = 1.5
            elif regime == MarketRegime.SIDEWAYS:
                adjustments["trade_frequency_mult"] = 0.5

        # Crisis/special regimes
        if regime == MarketRegime.CRISIS:
            adjustments["position_size_mult"] = 0.25
            adjustments["trade_frequency_mult"] = 0.25
        elif regime == MarketRegime.EUPHORIA:
            adjustments["take_profit_mult"] = 0.8  # Take profits earlier

        return adjustments

    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence."""
        return {
            "current_regime": self._current_regime.to_dict() if self._current_regime else None,
            "history_length": len(self._regime_history),
            "transitions": [t.to_dict() for t in self._transitions[-20:]],
            "last_analysis": self._last_analysis.isoformat() if self._last_analysis else None,
        }


# Singleton accessor
_detector_instance: Optional[RegimeDetector] = None

def get_regime_detector() -> RegimeDetector:
    """Get the singleton RegimeDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = RegimeDetector()
    return _detector_instance
