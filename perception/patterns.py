"""
PATTERN DETECTION - Identifies Tradeable Patterns for Hypothesis Generation

This module scans feature data to identify potential trading patterns:

1. STATISTICAL ANOMALIES
   - Z-score extremes
   - Unusual volume/volatility
   - Mean reversion setups
   
2. TECHNICAL SETUPS
   - Breakout setups
   - Support/Resistance tests
   - Moving average configurations
   - Indicator divergences
   
3. MOMENTUM PATTERNS
   - Trend continuation
   - Trend reversal signals
   - Momentum exhaustion
   
4. CROSS-ASSET PATTERNS
   - Relative strength breakouts
   - Correlation breakdowns
   - Sector rotation signals

Detected patterns are passed to the Hypothesis Engine for strategy creation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
from collections import defaultdict

from loguru import logger

from core.events import get_event_bus, Event, EventType


class PatternType(str, Enum):
    """Types of patterns the system can detect."""
    
    # Price Breakouts
    BREAKOUT_RESISTANCE = "breakout_resistance"
    BREAKOUT_SUPPORT = "breakdown_support"
    CHANNEL_BREAKOUT = "channel_breakout"
    BOLLINGER_BREAKOUT = "bollinger_breakout"
    
    # Mean Reversion
    MEAN_REVERSION_OVERSOLD = "mean_reversion_oversold"
    MEAN_REVERSION_OVERBOUGHT = "mean_reversion_overbought"
    ZSCORE_EXTREME = "zscore_extreme"
    
    # Momentum
    MOMENTUM_SURGE = "momentum_surge"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    
    # Technical
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    MA_BOUNCE = "ma_bounce"
    INDICATOR_DIVERGENCE = "indicator_divergence"
    
    # Volume
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DRY_UP = "volume_dry_up"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    
    # Volatility
    VOLATILITY_SQUEEZE = "volatility_squeeze"
    VOLATILITY_EXPANSION = "volatility_expansion"
    
    # Cross-Asset
    RELATIVE_STRENGTH_BREAKOUT = "relative_strength_breakout"
    SECTOR_ROTATION = "sector_rotation"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    
    # Candlestick
    REVERSAL_PATTERN = "reversal_pattern"
    CONTINUATION_PATTERN = "continuation_pattern"


class PatternStrength(str, Enum):
    """Strength/reliability of detected pattern."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class PatternDirection(str, Enum):
    """Expected direction from pattern."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class DetectedPattern:
    """A detected pattern with metadata."""
    
    pattern_id: str
    pattern_type: PatternType
    symbol: str
    timestamp: datetime
    
    # Pattern characteristics
    direction: PatternDirection
    strength: PatternStrength
    confidence: float  # 0.0 to 1.0
    
    # Context
    price_at_detection: float
    features_snapshot: Dict[str, float] = field(default_factory=dict)
    
    # Trade setup
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward: Optional[float] = None
    
    # Description
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    
    # Expiry
    valid_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "price_at_detection": self.price_at_detection,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "risk_reward": self.risk_reward,
            "description": self.description,
            "evidence": self.evidence
        }


@dataclass
class PatternConfig:
    """Configuration for pattern detection."""
    
    # Z-score thresholds
    zscore_extreme_threshold: float = 2.5
    zscore_reversion_threshold: float = 2.0
    
    # RSI thresholds
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    rsi_extreme_oversold: float = 20
    rsi_extreme_overbought: float = 80
    
    # Momentum thresholds
    momentum_surge_threshold: float = 3.0  # % move
    volume_spike_threshold: float = 2.5    # multiple of average
    
    # Trend thresholds
    adx_trending: float = 25
    adx_strong_trend: float = 40
    
    # Bollinger Band thresholds
    bb_squeeze_percentile: float = 20  # Width in bottom 20%
    
    # Minimum confidence to report
    min_confidence: float = 0.5
    
    # Pattern validity period (hours)
    pattern_validity_hours: int = 24


class PatternDetector:
    """
    Scans market data to detect tradeable patterns.
    
    Detected patterns are candidates for hypothesis generation.
    """
    
    def __init__(self, config: Optional[PatternConfig] = None):
        self.config = config or PatternConfig()
        
        # Track detected patterns
        self._active_patterns: Dict[str, DetectedPattern] = {}
        self._pattern_history: List[DetectedPattern] = []
        self._pattern_counter = 0
        
        logger.info("PatternDetector initialized")
    
    def _generate_pattern_id(self) -> str:
        """Generate unique pattern ID."""
        self._pattern_counter += 1
        return f"pat_{datetime.now().strftime('%Y%m%d')}_{self._pattern_counter:05d}"
    
    def _calculate_strength(
        self,
        confidence: float,
        evidence_count: int
    ) -> PatternStrength:
        """Calculate pattern strength from confidence and evidence."""
        score = confidence * (1 + min(evidence_count, 5) * 0.1)
        
        if score >= 0.85:
            return PatternStrength.VERY_STRONG
        elif score >= 0.70:
            return PatternStrength.STRONG
        elif score >= 0.55:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK
    
    def _add_pattern(self, pattern: DetectedPattern) -> None:
        """Add a detected pattern."""
        if pattern.confidence >= self.config.min_confidence:
            self._active_patterns[pattern.pattern_id] = pattern
            self._pattern_history.append(pattern)
            
            # Emit event
            get_event_bus().publish(Event(
                event_type=EventType.PATTERN_DISCOVERED,
                source="pattern_detector",
                payload=pattern.to_dict()
            ))
            
            logger.info(
                f"Pattern detected: {pattern.pattern_type.value} on {pattern.symbol} "
                f"({pattern.direction.value}, {pattern.strength.value})"
            )
    
    # ===== BREAKOUT PATTERNS =====
    
    def detect_breakouts(
        self,
        symbol: str,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect breakout patterns."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        latest = df.iloc[-1]
        price = latest['close']
        timestamp = df.index[-1]
        
        # Resistance breakout (20-day high)
        if 'resistance_20' in features.columns:
            resistance = features['resistance_20'].iloc[-2]  # Previous resistance
            if price > resistance * 1.001:  # Small buffer
                evidence = [
                    f"Price {price:.2f} broke above 20-day resistance {resistance:.2f}",
                    f"Breakout magnitude: {((price/resistance)-1)*100:.2f}%"
                ]
                
                # Higher confidence if volume confirms
                confidence = 0.6
                if 'volume_ratio_20' in features.columns:
                    vol_ratio = features['volume_ratio_20'].iloc[-1]
                    if vol_ratio > 1.5:
                        confidence += 0.15
                        evidence.append(f"Volume confirmation: {vol_ratio:.2f}x average")
                
                # Higher confidence if trend is up
                if 'adx' in features.columns and 'plus_di' in features.columns:
                    if features['adx'].iloc[-1] > 20 and features['plus_di'].iloc[-1] > features['minus_di'].iloc[-1]:
                        confidence += 0.1
                        evidence.append("Trending market with bullish ADX")
                
                # Calculate stop and target
                atr = features['atr'].iloc[-1] if 'atr' in features.columns else price * 0.02
                stop_loss = resistance - atr
                target = price + 2 * (price - stop_loss)
                
                pattern = DetectedPattern(
                    pattern_id=self._generate_pattern_id(),
                    pattern_type=PatternType.BREAKOUT_RESISTANCE,
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=PatternDirection.BULLISH,
                    strength=self._calculate_strength(confidence, len(evidence)),
                    confidence=confidence,
                    price_at_detection=price,
                    features_snapshot=features.iloc[-1].to_dict() if not features.empty else {},
                    entry_price=price,
                    stop_loss=stop_loss,
                    target_price=target,
                    risk_reward=(target - price) / (price - stop_loss) if price > stop_loss else 0,
                    description="Resistance breakout with momentum",
                    evidence=evidence,
                    valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                )
                patterns.append(pattern)
        
        # Support breakdown
        if 'support_20' in features.columns:
            support = features['support_20'].iloc[-2]
            if price < support * 0.999:
                evidence = [
                    f"Price {price:.2f} broke below 20-day support {support:.2f}",
                    f"Breakdown magnitude: {((support/price)-1)*100:.2f}%"
                ]
                
                confidence = 0.6
                if 'volume_ratio_20' in features.columns:
                    vol_ratio = features['volume_ratio_20'].iloc[-1]
                    if vol_ratio > 1.5:
                        confidence += 0.15
                        evidence.append(f"Volume confirmation: {vol_ratio:.2f}x average")
                
                atr = features['atr'].iloc[-1] if 'atr' in features.columns else price * 0.02
                stop_loss = support + atr
                target = price - 2 * (stop_loss - price)
                
                pattern = DetectedPattern(
                    pattern_id=self._generate_pattern_id(),
                    pattern_type=PatternType.BREAKOUT_SUPPORT,
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=PatternDirection.BEARISH,
                    strength=self._calculate_strength(confidence, len(evidence)),
                    confidence=confidence,
                    price_at_detection=price,
                    entry_price=price,
                    stop_loss=stop_loss,
                    target_price=target,
                    risk_reward=(price - target) / (stop_loss - price) if stop_loss > price else 0,
                    description="Support breakdown",
                    evidence=evidence,
                    valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                )
                patterns.append(pattern)
        
        # Bollinger Band breakout (squeeze release)
        if 'bb_squeeze' in features.columns and 'bb_percent_b' in features.columns:
            # Was in squeeze, now breaking out
            if len(features) >= 3:
                was_squeeze = features['bb_squeeze'].iloc[-3:-1].sum() >= 2
                percent_b = features['bb_percent_b'].iloc[-1]
                
                if was_squeeze:
                    if percent_b > 1.0:  # Breaking above upper band
                        pattern = DetectedPattern(
                            pattern_id=self._generate_pattern_id(),
                            pattern_type=PatternType.BOLLINGER_BREAKOUT,
                            symbol=symbol,
                            timestamp=timestamp,
                            direction=PatternDirection.BULLISH,
                            strength=PatternStrength.STRONG,
                            confidence=0.7,
                            price_at_detection=price,
                            description="Bollinger squeeze breakout (bullish)",
                            evidence=["BB squeeze detected", "Price breaking above upper band"],
                            valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                        )
                        patterns.append(pattern)
                    elif percent_b < 0.0:  # Breaking below lower band
                        pattern = DetectedPattern(
                            pattern_id=self._generate_pattern_id(),
                            pattern_type=PatternType.BOLLINGER_BREAKOUT,
                            symbol=symbol,
                            timestamp=timestamp,
                            direction=PatternDirection.BEARISH,
                            strength=PatternStrength.STRONG,
                            confidence=0.7,
                            price_at_detection=price,
                            description="Bollinger squeeze breakout (bearish)",
                            evidence=["BB squeeze detected", "Price breaking below lower band"],
                            valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                        )
                        patterns.append(pattern)
        
        return patterns
    
    # ===== MEAN REVERSION PATTERNS =====
    
    def detect_mean_reversion(
        self,
        symbol: str,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect mean reversion setups."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        latest = df.iloc[-1]
        price = latest['close']
        timestamp = df.index[-1]
        
        evidence = []
        confidence = 0.5
        direction = None
        
        # RSI extreme
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14'].iloc[-1]
            
            if rsi < self.config.rsi_extreme_oversold:
                direction = PatternDirection.BULLISH
                evidence.append(f"RSI extremely oversold: {rsi:.1f}")
                confidence += 0.2
            elif rsi < self.config.rsi_oversold:
                direction = PatternDirection.BULLISH
                evidence.append(f"RSI oversold: {rsi:.1f}")
                confidence += 0.1
            elif rsi > self.config.rsi_extreme_overbought:
                direction = PatternDirection.BEARISH
                evidence.append(f"RSI extremely overbought: {rsi:.1f}")
                confidence += 0.2
            elif rsi > self.config.rsi_overbought:
                direction = PatternDirection.BEARISH
                evidence.append(f"RSI overbought: {rsi:.1f}")
                confidence += 0.1
        
        # Z-score extreme
        if 'price_zscore' in features.columns:
            zscore = features['price_zscore'].iloc[-1]
            
            if zscore < -self.config.zscore_extreme_threshold:
                direction = PatternDirection.BULLISH
                evidence.append(f"Price Z-score extreme: {zscore:.2f}")
                confidence += 0.15
            elif zscore > self.config.zscore_extreme_threshold:
                direction = PatternDirection.BEARISH
                evidence.append(f"Price Z-score extreme: {zscore:.2f}")
                confidence += 0.15
        
        # Bollinger Band position
        if 'bb_percent_b' in features.columns:
            bb_pct = features['bb_percent_b'].iloc[-1]
            
            if bb_pct < 0:
                if direction == PatternDirection.BULLISH:
                    evidence.append(f"Below lower Bollinger Band (%B: {bb_pct:.2f})")
                    confidence += 0.1
            elif bb_pct > 1:
                if direction == PatternDirection.BEARISH:
                    evidence.append(f"Above upper Bollinger Band (%B: {bb_pct:.2f})")
                    confidence += 0.1
        
        # Stochastic
        if 'stoch_k' in features.columns:
            stoch = features['stoch_k'].iloc[-1]
            
            if stoch < 20 and direction == PatternDirection.BULLISH:
                evidence.append(f"Stochastic oversold: {stoch:.1f}")
                confidence += 0.1
            elif stoch > 80 and direction == PatternDirection.BEARISH:
                evidence.append(f"Stochastic overbought: {stoch:.1f}")
                confidence += 0.1
        
        # Create pattern if we have enough evidence
        if direction and len(evidence) >= 2 and confidence >= self.config.min_confidence:
            pattern_type = (
                PatternType.MEAN_REVERSION_OVERSOLD 
                if direction == PatternDirection.BULLISH 
                else PatternType.MEAN_REVERSION_OVERBOUGHT
            )
            
            # Calculate stop and target
            atr = features['atr'].iloc[-1] if 'atr' in features.columns else price * 0.02
            
            if direction == PatternDirection.BULLISH:
                stop_loss = price - 2 * atr
                # Target: mean (SMA 20)
                target = features['sma_20'].iloc[-1] if 'sma_20' in features.columns else price * 1.03
            else:
                stop_loss = price + 2 * atr
                target = features['sma_20'].iloc[-1] if 'sma_20' in features.columns else price * 0.97
            
            pattern = DetectedPattern(
                pattern_id=self._generate_pattern_id(),
                pattern_type=pattern_type,
                symbol=symbol,
                timestamp=timestamp,
                direction=direction,
                strength=self._calculate_strength(confidence, len(evidence)),
                confidence=min(confidence, 0.9),
                price_at_detection=price,
                features_snapshot=features.iloc[-1].to_dict() if not features.empty else {},
                entry_price=price,
                stop_loss=stop_loss,
                target_price=target,
                risk_reward=abs(target - price) / abs(price - stop_loss) if price != stop_loss else 0,
                description=f"Mean reversion setup ({direction.value})",
                evidence=evidence,
                valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
            )
            patterns.append(pattern)
        
        return patterns
    
    # ===== MOMENTUM PATTERNS =====
    
    def detect_momentum(
        self,
        symbol: str,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect momentum patterns."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        latest = df.iloc[-1]
        price = latest['close']
        timestamp = df.index[-1]
        
        # Momentum surge
        if 'return_1d' in features.columns:
            daily_return = features['return_1d'].iloc[-1]
            
            if abs(daily_return) > self.config.momentum_surge_threshold:
                direction = PatternDirection.BULLISH if daily_return > 0 else PatternDirection.BEARISH
                
                evidence = [f"Strong daily move: {daily_return:.2f}%"]
                confidence = 0.55
                
                # Volume confirmation
                if 'volume_ratio_20' in features.columns:
                    vol_ratio = features['volume_ratio_20'].iloc[-1]
                    if vol_ratio > self.config.volume_spike_threshold:
                        confidence += 0.2
                        evidence.append(f"Volume spike: {vol_ratio:.2f}x average")
                
                # Trend alignment
                if 'adx' in features.columns and features['adx'].iloc[-1] > self.config.adx_trending:
                    confidence += 0.1
                    evidence.append(f"Strong trend (ADX: {features['adx'].iloc[-1]:.1f})")
                
                if confidence >= self.config.min_confidence:
                    pattern = DetectedPattern(
                        pattern_id=self._generate_pattern_id(),
                        pattern_type=PatternType.MOMENTUM_SURGE,
                        symbol=symbol,
                        timestamp=timestamp,
                        direction=direction,
                        strength=self._calculate_strength(confidence, len(evidence)),
                        confidence=confidence,
                        price_at_detection=price,
                        description=f"Momentum surge ({direction.value})",
                        evidence=evidence,
                        valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                    )
                    patterns.append(pattern)
        
        # Trend continuation (price above MAs with rising ADX)
        if all(col in features.columns for col in ['close_above_sma_20', 'close_above_sma_50', 'adx']):
            above_sma20 = features['close_above_sma_20'].iloc[-1]
            above_sma50 = features['close_above_sma_50'].iloc[-1]
            adx = features['adx'].iloc[-1]
            adx_rising = features['adx'].iloc[-1] > features['adx'].iloc[-5] if len(features) >= 5 else False
            
            if above_sma20 and above_sma50 and adx > self.config.adx_trending and adx_rising:
                pattern = DetectedPattern(
                    pattern_id=self._generate_pattern_id(),
                    pattern_type=PatternType.TREND_CONTINUATION,
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=PatternDirection.BULLISH,
                    strength=PatternStrength.MODERATE,
                    confidence=0.65,
                    price_at_detection=price,
                    description="Uptrend continuation setup",
                    evidence=[
                        "Price above SMA 20 and SMA 50",
                        f"ADX rising: {adx:.1f}",
                        "Trend strength increasing"
                    ],
                    valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                )
                patterns.append(pattern)
        
        # Golden Cross
        if 'golden_cross' in features.columns and len(features) >= 2:
            current_cross = features['golden_cross'].iloc[-1]
            previous_cross = features['golden_cross'].iloc[-2]
            
            if current_cross == 1 and previous_cross == 0:
                pattern = DetectedPattern(
                    pattern_id=self._generate_pattern_id(),
                    pattern_type=PatternType.GOLDEN_CROSS,
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=PatternDirection.BULLISH,
                    strength=PatternStrength.STRONG,
                    confidence=0.7,
                    price_at_detection=price,
                    description="Golden Cross (SMA 50 crossed above SMA 200)",
                    evidence=["SMA 50 crossed above SMA 200"],
                    valid_until=timestamp + timedelta(hours=48)
                )
                patterns.append(pattern)
            
            elif current_cross == 0 and previous_cross == 1:
                pattern = DetectedPattern(
                    pattern_id=self._generate_pattern_id(),
                    pattern_type=PatternType.DEATH_CROSS,
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=PatternDirection.BEARISH,
                    strength=PatternStrength.STRONG,
                    confidence=0.7,
                    price_at_detection=price,
                    description="Death Cross (SMA 50 crossed below SMA 200)",
                    evidence=["SMA 50 crossed below SMA 200"],
                    valid_until=timestamp + timedelta(hours=48)
                )
                patterns.append(pattern)
        
        return patterns
    
    # ===== VOLUME PATTERNS =====
    
    def detect_volume_patterns(
        self,
        symbol: str,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect volume-based patterns."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        latest = df.iloc[-1]
        price = latest['close']
        timestamp = df.index[-1]
        
        # Volume spike
        if 'volume_ratio_20' in features.columns:
            vol_ratio = features['volume_ratio_20'].iloc[-1]
            
            if vol_ratio > self.config.volume_spike_threshold:
                # Determine direction from price action
                if 'return_1d' in features.columns:
                    daily_return = features['return_1d'].iloc[-1]
                    direction = PatternDirection.BULLISH if daily_return > 0 else PatternDirection.BEARISH
                else:
                    direction = PatternDirection.NEUTRAL
                
                evidence = [
                    f"Volume {vol_ratio:.2f}x average",
                    f"Price move: {features['return_1d'].iloc[-1]:.2f}%" if 'return_1d' in features.columns else ""
                ]
                
                pattern = DetectedPattern(
                    pattern_id=self._generate_pattern_id(),
                    pattern_type=PatternType.VOLUME_SPIKE,
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=direction,
                    strength=PatternStrength.MODERATE,
                    confidence=0.6,
                    price_at_detection=price,
                    description=f"Volume spike ({vol_ratio:.1f}x average)",
                    evidence=[e for e in evidence if e],
                    valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                )
                patterns.append(pattern)
        
        # Accumulation (rising OBV with price consolidation)
        if 'obv_trend' in features.columns and 'volatility_10d' in features.columns:
            obv_bullish = features['obv_trend'].iloc[-5:].sum() >= 4 if len(features) >= 5 else False
            low_vol = features['volatility_10d'].iloc[-1] < features['volatility_10d'].rolling(60).quantile(0.3).iloc[-1] if len(features) >= 60 else False
            
            if obv_bullish and low_vol:
                pattern = DetectedPattern(
                    pattern_id=self._generate_pattern_id(),
                    pattern_type=PatternType.ACCUMULATION,
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=PatternDirection.BULLISH,
                    strength=PatternStrength.MODERATE,
                    confidence=0.6,
                    price_at_detection=price,
                    description="Accumulation pattern (rising OBV, low volatility)",
                    evidence=["OBV trending up", "Low price volatility", "Possible accumulation phase"],
                    valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                )
                patterns.append(pattern)
        
        return patterns
    
    # ===== CANDLESTICK PATTERNS =====
    
    def detect_candlestick_patterns(
        self,
        symbol: str,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect significant candlestick patterns."""
        patterns = []
        
        if len(df) < 5:
            return patterns
        
        latest = df.iloc[-1]
        price = latest['close']
        timestamp = df.index[-1]
        
        # Reversal patterns
        reversal_patterns = [
            ('hammer', PatternDirection.BULLISH),
            ('inverted_hammer', PatternDirection.BULLISH),
            ('bullish_engulfing', PatternDirection.BULLISH),
            ('morning_star', PatternDirection.BULLISH),
            ('bearish_engulfing', PatternDirection.BEARISH),
            ('evening_star', PatternDirection.BEARISH),
        ]
        
        for pattern_name, direction in reversal_patterns:
            if pattern_name in features.columns and features[pattern_name].iloc[-1] == 1:
                # Higher confidence if at support/resistance
                confidence = 0.55
                evidence = [f"{pattern_name.replace('_', ' ').title()} pattern detected"]
                
                # Check if at key level
                if direction == PatternDirection.BULLISH and 'rsi_14' in features.columns:
                    if features['rsi_14'].iloc[-1] < 40:
                        confidence += 0.15
                        evidence.append("Near oversold conditions")
                elif direction == PatternDirection.BEARISH and 'rsi_14' in features.columns:
                    if features['rsi_14'].iloc[-1] > 60:
                        confidence += 0.15
                        evidence.append("Near overbought conditions")
                
                if confidence >= self.config.min_confidence:
                    pattern = DetectedPattern(
                        pattern_id=self._generate_pattern_id(),
                        pattern_type=PatternType.REVERSAL_PATTERN,
                        symbol=symbol,
                        timestamp=timestamp,
                        direction=direction,
                        strength=self._calculate_strength(confidence, len(evidence)),
                        confidence=confidence,
                        price_at_detection=price,
                        description=f"Candlestick reversal: {pattern_name}",
                        evidence=evidence,
                        valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                    )
                    patterns.append(pattern)
        
        # Continuation patterns
        continuation_patterns = [
            ('three_white_soldiers', PatternDirection.BULLISH),
            ('three_black_crows', PatternDirection.BEARISH),
        ]
        
        for pattern_name, direction in continuation_patterns:
            if pattern_name in features.columns and features[pattern_name].iloc[-1] == 1:
                pattern = DetectedPattern(
                    pattern_id=self._generate_pattern_id(),
                    pattern_type=PatternType.CONTINUATION_PATTERN,
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=direction,
                    strength=PatternStrength.STRONG,
                    confidence=0.65,
                    price_at_detection=price,
                    description=f"Candlestick continuation: {pattern_name}",
                    evidence=[f"{pattern_name.replace('_', ' ').title()} pattern detected"],
                    valid_until=timestamp + timedelta(hours=self.config.pattern_validity_hours)
                )
                patterns.append(pattern)
        
        return patterns
    
    # ===== MAIN DETECTION =====
    
    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[DetectedPattern]:
        """
        Scan a symbol for all pattern types.
        
        Args:
            symbol: Symbol to scan
            df: OHLCV DataFrame
            features: Computed features DataFrame
        
        Returns:
            List of detected patterns
        """
        all_patterns = []
        
        try:
            # Breakouts
            all_patterns.extend(self.detect_breakouts(symbol, df, features))
            
            # Mean Reversion
            all_patterns.extend(self.detect_mean_reversion(symbol, df, features))
            
            # Momentum
            all_patterns.extend(self.detect_momentum(symbol, df, features))
            
            # Volume
            all_patterns.extend(self.detect_volume_patterns(symbol, df, features))
            
            # Candlesticks
            all_patterns.extend(self.detect_candlestick_patterns(symbol, df, features))
            
            # Add all detected patterns
            for pattern in all_patterns:
                self._add_pattern(pattern)
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
        
        return all_patterns
    
    def scan_universe(
        self,
        data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Dict[str, List[DetectedPattern]]:
        """
        Scan all symbols in the universe.
        
        Args:
            data: Dict of symbol -> (ohlcv_df, features_df)
        
        Returns:
            Dict of symbol -> detected patterns
        """
        results = {}
        total_patterns = 0
        
        for symbol, (df, features) in data.items():
            patterns = self.scan(symbol, df, features)
            if patterns:
                results[symbol] = patterns
                total_patterns += len(patterns)
        
        logger.info(f"Universe scan complete: {total_patterns} patterns across {len(results)} symbols")
        
        return results
    
    def get_active_patterns(
        self,
        symbol: Optional[str] = None,
        pattern_type: Optional[PatternType] = None,
        direction: Optional[PatternDirection] = None,
        min_confidence: Optional[float] = None
    ) -> List[DetectedPattern]:
        """Get active patterns with optional filters."""
        now = datetime.now()
        
        patterns = [
            p for p in self._active_patterns.values()
            if p.valid_until is None or p.valid_until > now
        ]
        
        if symbol:
            patterns = [p for p in patterns if p.symbol == symbol]
        
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        if direction:
            patterns = [p for p in patterns if p.direction == direction]
        
        if min_confidence:
            patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        return sorted(patterns, key=lambda p: p.confidence, reverse=True)
    
    def cleanup_expired(self) -> int:
        """Remove expired patterns. Returns count of removed patterns."""
        now = datetime.now()
        expired = [
            pid for pid, p in self._active_patterns.items()
            if p.valid_until and p.valid_until < now
        ]
        
        for pid in expired:
            del self._active_patterns[pid]
        
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pattern detection statistics."""
        by_type = defaultdict(int)
        by_direction = defaultdict(int)
        by_strength = defaultdict(int)
        
        for p in self._active_patterns.values():
            by_type[p.pattern_type.value] += 1
            by_direction[p.direction.value] += 1
            by_strength[p.strength.value] += 1
        
        return {
            "active_patterns": len(self._active_patterns),
            "total_detected": len(self._pattern_history),
            "by_type": dict(by_type),
            "by_direction": dict(by_direction),
            "by_strength": dict(by_strength)
        }


# Singleton instance
_pattern_detector: Optional[PatternDetector] = None


def get_pattern_detector(config: Optional[PatternConfig] = None) -> PatternDetector:
    """Get the singleton PatternDetector instance."""
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = PatternDetector(config)
    return _pattern_detector
