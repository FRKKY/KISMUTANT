"""
FEATURES - Comprehensive Technical Feature Computation

Computes a full suite of technical features for hypothesis generation:

1. MOMENTUM INDICATORS
   - RSI, Stochastic, MACD, ROC, Williams %R, CCI, Ultimate Oscillator
   
2. TREND INDICATORS
   - SMA, EMA, WMA, DEMA, TEMA, ADX, Parabolic SAR, Ichimoku Cloud, Aroon
   
3. VOLATILITY INDICATORS
   - ATR, Bollinger Bands, Keltner Channels, Donchian Channels, Standard Deviation
   
4. VOLUME INDICATORS
   - OBV, VWAP, MFI, A/D Line, CMF, Volume SMA, Force Index, VWMA
   
5. PRICE ACTION
   - Support/Resistance, Pivot Points, Price Channels, Gap Analysis
   
6. CANDLESTICK PATTERNS
   - Doji, Hammer, Engulfing, Morning/Evening Star, etc.
   
7. CROSS-ASSET FEATURES
   - Relative Strength vs Benchmark, Correlation, Beta, Sector Rotation

All features are computed for multiple timeframes (daily + intraday).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
import warnings

from loguru import logger

warnings.filterwarnings('ignore')


class FeatureCategory(str, Enum):
    """Categories of technical features."""
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PRICE_ACTION = "price_action"
    CANDLESTICK = "candlestick"
    CROSS_ASSET = "cross_asset"
    STATISTICAL = "statistical"


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    
    # Momentum parameters
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    roc_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    cci_period: int = 20
    williams_period: int = 14
    
    # Trend parameters
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    adx_period: int = 14
    
    # Volatility parameters
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    keltner_period: int = 20
    keltner_atr_mult: float = 2.0
    
    # Volume parameters
    volume_sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    mfi_period: int = 14
    cmf_period: int = 20
    
    # Cross-asset
    correlation_period: int = 60
    beta_period: int = 252
    
    # Benchmark for relative strength
    benchmark_symbol: str = "069500"  # KODEX 200


class FeatureEngine:
    """
    Computes comprehensive technical features from OHLCV data.
    
    Usage:
        engine = FeatureEngine()
        features = engine.compute_all(df)  # df has OHLCV columns
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._benchmark_data: Optional[pd.DataFrame] = None
        
        logger.info("FeatureEngine initialized")
    
    def set_benchmark_data(self, df: pd.DataFrame) -> None:
        """Set benchmark data for relative strength calculations."""
        self._benchmark_data = df
    
    # ===== MOMENTUM INDICATORS =====
    
    def compute_rsi(
        self,
        df: pd.DataFrame,
        period: int = 14,
        column: str = 'close'
    ) -> pd.Series:
        """
        Relative Strength Index.
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def compute_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA(%K, d_period)
        """
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    def compute_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.
        
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal)
        Histogram = MACD - Signal
        """
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def compute_roc(
        self,
        df: pd.DataFrame,
        period: int = 10,
        column: str = 'close'
    ) -> pd.Series:
        """
        Rate of Change.
        
        ROC = ((Close - Close[n]) / Close[n]) * 100
        """
        return ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
    
    def compute_williams_r(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Williams %R.
        
        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        """
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        return ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
    
    def compute_cci(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> pd.Series:
        """
        Commodity Channel Index.
        
        CCI = (Typical Price - SMA(TP)) / (0.015 * Mean Deviation)
        """
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mean_dev = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        return (tp - sma_tp) / (0.015 * mean_dev)
    
    def compute_ultimate_oscillator(
        self,
        df: pd.DataFrame,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28
    ) -> pd.Series:
        """
        Ultimate Oscillator.
        
        Combines short, medium, and long-term price action.
        """
        prev_close = df['close'].shift(1)

        # True Low/High for Ultimate Oscillator
        true_low = np.minimum(df['low'], prev_close)
        true_high = np.maximum(df['high'], prev_close)

        bp = df['close'] - true_low  # Buying Pressure
        tr = true_high - true_low  # True Range
        
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        
        return 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
    
    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all momentum features."""
        features = pd.DataFrame(index=df.index)
        
        # RSI for multiple periods
        for period in self.config.rsi_periods:
            features[f'rsi_{period}'] = self.compute_rsi(df, period)
        
        # Stochastic
        stoch_k, stoch_d = self.compute_stochastic(
            df, 
            self.config.stoch_k_period, 
            self.config.stoch_d_period
        )
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        features['stoch_cross'] = (stoch_k > stoch_d).astype(int)
        
        # MACD
        macd, signal, histogram = self.compute_macd(
            df,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        features['macd_cross'] = (macd > signal).astype(int)
        
        # ROC for multiple periods
        for period in self.config.roc_periods:
            features[f'roc_{period}'] = self.compute_roc(df, period)
        
        # Williams %R
        features['williams_r'] = self.compute_williams_r(df, self.config.williams_period)
        
        # CCI
        features['cci'] = self.compute_cci(df, self.config.cci_period)
        
        # Ultimate Oscillator
        features['ultimate_osc'] = self.compute_ultimate_oscillator(df)
        
        return features
    
    # ===== TREND INDICATORS =====
    
    def compute_sma(
        self,
        df: pd.DataFrame,
        period: int,
        column: str = 'close'
    ) -> pd.Series:
        """Simple Moving Average."""
        return df[column].rolling(window=period).mean()
    
    def compute_ema(
        self,
        df: pd.DataFrame,
        period: int,
        column: str = 'close'
    ) -> pd.Series:
        """Exponential Moving Average."""
        return df[column].ewm(span=period, adjust=False).mean()
    
    def compute_wma(
        self,
        df: pd.DataFrame,
        period: int,
        column: str = 'close'
    ) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return df[column].rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(),
            raw=True
        )
    
    def compute_dema(
        self,
        df: pd.DataFrame,
        period: int,
        column: str = 'close'
    ) -> pd.Series:
        """Double Exponential Moving Average."""
        ema1 = df[column].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2
    
    def compute_tema(
        self,
        df: pd.DataFrame,
        period: int,
        column: str = 'close'
    ) -> pd.Series:
        """Triple Exponential Moving Average."""
        ema1 = df[column].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3
    
    def compute_adx(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index.
        
        Returns: ADX, +DI, -DI
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        # Smoothed DM
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def compute_parabolic_sar(
        self,
        df: pd.DataFrame,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.20
    ) -> pd.Series:
        """
        Parabolic SAR.
        
        Trend-following indicator.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        length = len(df)
        psar = np.zeros(length)
        psarbull = np.zeros(length)
        psarbear = np.zeros(length)
        bull = True
        iaf = af_start
        af = iaf
        ep = low[0]
        hp = high[0]
        lp = low[0]
        
        for i in range(2, length):
            if bull:
                psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            else:
                psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            
            reverse = False
            
            if bull:
                if low[i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = low[i]
                    af = iaf
            else:
                if high[i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = high[i]
                    af = iaf
            
            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + af_increment, af_max)
                    if low[i - 1] < psar[i]:
                        psar[i] = low[i - 1]
                    if low[i - 2] < psar[i]:
                        psar[i] = low[i - 2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + af_increment, af_max)
                    if high[i - 1] > psar[i]:
                        psar[i] = high[i - 1]
                    if high[i - 2] > psar[i]:
                        psar[i] = high[i - 2]
            
            if bull:
                psarbull[i] = psar[i]
            else:
                psarbear[i] = psar[i]
        
        return pd.Series(psar, index=df.index)
    
    def compute_ichimoku(
        self,
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        senkou: int = 52
    ) -> Dict[str, pd.Series]:
        """
        Ichimoku Cloud.
        
        Returns: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=senkou).max() + low.rolling(window=senkou).min()) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def compute_aroon(
        self,
        df: pd.DataFrame,
        period: int = 25
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Aroon Indicator.
        
        Returns: Aroon Up, Aroon Down, Aroon Oscillator
        """
        aroon_up = 100 * df['high'].rolling(window=period + 1).apply(
            lambda x: (period - x.argmax()) / period * 100 if len(x) > 0 else 50,
            raw=True
        )
        
        aroon_down = 100 * df['low'].rolling(window=period + 1).apply(
            lambda x: (period - x.argmin()) / period * 100 if len(x) > 0 else 50,
            raw=True
        )
        
        aroon_osc = aroon_up - aroon_down
        
        return aroon_up, aroon_down, aroon_osc
    
    def compute_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all trend features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # Moving Averages
        for period in self.config.sma_periods:
            sma = self.compute_sma(df, period)
            features[f'sma_{period}'] = sma
            features[f'close_to_sma_{period}'] = (close - sma) / sma * 100
            features[f'close_above_sma_{period}'] = (close > sma).astype(int)
        
        for period in self.config.ema_periods:
            ema = self.compute_ema(df, period)
            features[f'ema_{period}'] = ema
            features[f'close_to_ema_{period}'] = (close - ema) / ema * 100
        
        # DEMA and TEMA
        features['dema_20'] = self.compute_dema(df, 20)
        features['tema_20'] = self.compute_tema(df, 20)
        
        # Moving Average Crossovers
        if 50 in self.config.sma_periods and 200 in self.config.sma_periods:
            features['golden_cross'] = (features['sma_50'] > features['sma_200']).astype(int)
        
        # ADX
        adx, plus_di, minus_di = self.compute_adx(df, self.config.adx_period)
        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        features['adx_trend_strength'] = np.where(adx > 25, 1, 0)
        
        # Parabolic SAR
        psar = self.compute_parabolic_sar(df)
        features['psar'] = psar
        features['psar_bullish'] = (close > psar).astype(int)
        
        # Ichimoku
        ichimoku = self.compute_ichimoku(df)
        features['tenkan_sen'] = ichimoku['tenkan_sen']
        features['kijun_sen'] = ichimoku['kijun_sen']
        features['senkou_span_a'] = ichimoku['senkou_span_a']
        features['senkou_span_b'] = ichimoku['senkou_span_b']
        features['ichimoku_bullish'] = (
            (close > ichimoku['senkou_span_a']) & 
            (close > ichimoku['senkou_span_b'])
        ).astype(int)
        
        # Aroon
        aroon_up, aroon_down, aroon_osc = self.compute_aroon(df)
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_osc'] = aroon_osc
        
        return features
    
    # ===== VOLATILITY INDICATORS =====
    
    def compute_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def compute_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Returns: middle, upper, lower, %B
        """
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        # %B - where price is relative to bands
        percent_b = (df['close'] - lower) / (upper - lower)
        
        return middle, upper, lower, percent_b
    
    def compute_keltner_channels(
        self,
        df: pd.DataFrame,
        period: int = 20,
        atr_mult: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.
        
        Returns: middle, upper, lower
        """
        middle = df['close'].ewm(span=period, adjust=False).mean()
        atr = self.compute_atr(df, period)
        
        upper = middle + (atr_mult * atr)
        lower = middle - (atr_mult * atr)
        
        return middle, upper, lower
    
    def compute_donchian_channels(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels.
        
        Returns: upper, lower, middle
        """
        upper = df['high'].rolling(window=period).max()
        lower = df['low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper, lower, middle
    
    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all volatility features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # ATR
        atr = self.compute_atr(df, self.config.atr_period)
        features['atr'] = atr
        features['atr_percent'] = (atr / close) * 100
        
        # Bollinger Bands
        bb_mid, bb_upper, bb_lower, bb_pct = self.compute_bollinger_bands(
            df, self.config.bb_period, self.config.bb_std
        )
        features['bb_middle'] = bb_mid
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_percent_b'] = bb_pct
        features['bb_width'] = (bb_upper - bb_lower) / bb_mid * 100
        features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(50).mean()).astype(int)
        
        # Keltner Channels
        kc_mid, kc_upper, kc_lower = self.compute_keltner_channels(
            df, self.config.keltner_period, self.config.keltner_atr_mult
        )
        features['kc_middle'] = kc_mid
        features['kc_upper'] = kc_upper
        features['kc_lower'] = kc_lower
        
        # Squeeze (BB inside KC)
        features['squeeze'] = (
            (bb_lower > kc_lower) & (bb_upper < kc_upper)
        ).astype(int)
        
        # Donchian Channels
        dc_upper, dc_lower, dc_middle = self.compute_donchian_channels(df, 20)
        features['dc_upper'] = dc_upper
        features['dc_lower'] = dc_lower
        features['dc_middle'] = dc_middle
        features['dc_position'] = (close - dc_lower) / (dc_upper - dc_lower)
        
        # Historical Volatility
        returns = close.pct_change()
        for period in [10, 20, 60]:
            features[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252) * 100
        
        # Volatility ratio
        features['volatility_ratio'] = features['volatility_10d'] / features['volatility_60d']
        
        return features
    
    # ===== VOLUME INDICATORS =====
    
    def compute_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def compute_mfi(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Money Flow Index.
        
        Volume-weighted RSI.
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.inf)))
        
        return mfi
    
    def compute_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution Line."""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfm * df['volume']
        ad = mfv.cumsum()
        return ad
    
    def compute_cmf(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> pd.Series:
        """Chaikin Money Flow."""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfm * df['volume']
        cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf
    
    def compute_force_index(
        self,
        df: pd.DataFrame,
        period: int = 13
    ) -> pd.Series:
        """Force Index."""
        force = df['close'].diff() * df['volume']
        force_smoothed = force.ewm(span=period, adjust=False).mean()
        return force_smoothed
    
    def compute_vwma(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> pd.Series:
        """Volume Weighted Moving Average."""
        vwma = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
        return vwma
    
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all volume features."""
        features = pd.DataFrame(index=df.index)
        
        # OBV
        features['obv'] = self.compute_obv(df)
        features['obv_sma_20'] = features['obv'].rolling(20).mean()
        features['obv_trend'] = (features['obv'] > features['obv_sma_20']).astype(int)
        
        # VWAP
        features['vwap'] = self.compute_vwap(df)
        features['close_to_vwap'] = (df['close'] - features['vwap']) / features['vwap'] * 100
        
        # MFI
        features['mfi'] = self.compute_mfi(df, self.config.mfi_period)
        
        # A/D Line
        features['ad_line'] = self.compute_ad_line(df)
        
        # CMF
        features['cmf'] = self.compute_cmf(df, self.config.cmf_period)
        
        # Force Index
        features['force_index'] = self.compute_force_index(df)
        
        # VWMA
        features['vwma_20'] = self.compute_vwma(df, 20)
        features['close_to_vwma'] = (df['close'] - features['vwma_20']) / features['vwma_20'] * 100
        
        # Volume SMAs and ratios
        for period in self.config.volume_sma_periods:
            vol_sma = df['volume'].rolling(period).mean()
            features[f'volume_sma_{period}'] = vol_sma
            features[f'volume_ratio_{period}'] = df['volume'] / vol_sma
        
        # Volume trend
        features['volume_trend'] = (df['volume'] > df['volume'].rolling(20).mean()).astype(int)
        
        return features
    
    # ===== PRICE ACTION =====
    
    def compute_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price action features."""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        # Price changes
        features['return_1d'] = close.pct_change() * 100
        features['return_5d'] = close.pct_change(5) * 100
        features['return_10d'] = close.pct_change(10) * 100
        features['return_20d'] = close.pct_change(20) * 100
        
        # High-Low range
        features['daily_range'] = (high - low) / close * 100
        features['daily_range_avg'] = features['daily_range'].rolling(20).mean()
        
        # Gap analysis
        features['gap'] = (open_price - close.shift(1)) / close.shift(1) * 100
        features['gap_filled'] = ((high >= close.shift(1)) & (open_price < close.shift(1)) | 
                                   (low <= close.shift(1)) & (open_price > close.shift(1))).astype(int)
        
        # Body and wick analysis
        body = abs(close - open_price)
        upper_wick = high - pd.concat([close, open_price], axis=1).max(axis=1)
        lower_wick = pd.concat([close, open_price], axis=1).min(axis=1) - low
        
        features['body_size'] = body / close * 100
        features['upper_wick'] = upper_wick / close * 100
        features['lower_wick'] = lower_wick / close * 100
        features['wick_ratio'] = upper_wick / (lower_wick + 0.0001)
        
        # Close position in day's range
        features['close_position'] = (close - low) / (high - low + 0.0001)
        
        # Support and Resistance (simplified - rolling highs/lows)
        for period in [10, 20, 50]:
            features[f'resistance_{period}'] = high.rolling(period).max()
            features[f'support_{period}'] = low.rolling(period).min()
            features[f'to_resistance_{period}'] = (features[f'resistance_{period}'] - close) / close * 100
            features[f'to_support_{period}'] = (close - features[f'support_{period}']) / close * 100
        
        # Pivot Points (Standard)
        pp = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        features['pivot_point'] = pp
        features['r1'] = 2 * pp - low.shift(1)
        features['r2'] = pp + (high.shift(1) - low.shift(1))
        features['s1'] = 2 * pp - high.shift(1)
        features['s2'] = pp - (high.shift(1) - low.shift(1))
        
        # Higher highs / Lower lows
        features['higher_high'] = (high > high.shift(1)).astype(int)
        features['lower_low'] = (low < low.shift(1)).astype(int)
        features['higher_high_streak'] = features['higher_high'].groupby(
            (features['higher_high'] != features['higher_high'].shift()).cumsum()
        ).cumsum()
        
        return features
    
    # ===== CANDLESTICK PATTERNS =====
    
    def compute_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect candlestick patterns."""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        body = close - open_price
        body_abs = abs(body)
        upper_shadow = high - pd.concat([close, open_price], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_price], axis=1).min(axis=1) - low
        avg_body = body_abs.rolling(10).mean()
        
        # Doji (small body)
        features['doji'] = (body_abs < avg_body * 0.1).astype(int)
        
        # Hammer (small body at top, long lower shadow)
        features['hammer'] = (
            (lower_shadow > 2 * body_abs) & 
            (upper_shadow < body_abs * 0.5) &
            (body_abs < avg_body)
        ).astype(int)
        
        # Inverted Hammer
        features['inverted_hammer'] = (
            (upper_shadow > 2 * body_abs) & 
            (lower_shadow < body_abs * 0.5) &
            (body_abs < avg_body)
        ).astype(int)
        
        # Bullish Engulfing
        prev_body = body.shift(1)
        features['bullish_engulfing'] = (
            (prev_body < 0) & (body > 0) &
            (open_price < close.shift(1)) &
            (close > open_price.shift(1))
        ).astype(int)
        
        # Bearish Engulfing
        features['bearish_engulfing'] = (
            (prev_body > 0) & (body < 0) &
            (open_price > close.shift(1)) &
            (close < open_price.shift(1))
        ).astype(int)
        
        # Morning Star (3-bar pattern)
        features['morning_star'] = (
            (body.shift(2) < 0) &  # First bar bearish
            (body_abs.shift(1) < avg_body * 0.3) &  # Second bar small
            (body > 0) &  # Third bar bullish
            (close > (open_price.shift(2) + close.shift(2)) / 2)  # Close above midpoint of first
        ).astype(int)
        
        # Evening Star (opposite of morning star)
        features['evening_star'] = (
            (body.shift(2) > 0) &
            (body_abs.shift(1) < avg_body * 0.3) &
            (body < 0) &
            (close < (open_price.shift(2) + close.shift(2)) / 2)
        ).astype(int)
        
        # Three White Soldiers
        features['three_white_soldiers'] = (
            (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0) &
            (close > close.shift(1)) & (close.shift(1) > close.shift(2)) &
            (open_price > open_price.shift(1)) & (open_price.shift(1) > open_price.shift(2))
        ).astype(int)
        
        # Three Black Crows
        features['three_black_crows'] = (
            (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0) &
            (close < close.shift(1)) & (close.shift(1) < close.shift(2)) &
            (open_price < open_price.shift(1)) & (open_price.shift(1) < open_price.shift(2))
        ).astype(int)
        
        # Spinning Top (small body, shadows on both sides)
        features['spinning_top'] = (
            (body_abs < avg_body * 0.3) &
            (upper_shadow > body_abs) &
            (lower_shadow > body_abs)
        ).astype(int)
        
        return features
    
    # ===== CROSS-ASSET FEATURES =====
    
    def compute_cross_asset_features(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Compute cross-asset relative strength features."""
        features = pd.DataFrame(index=df.index)
        
        if benchmark_df is None:
            benchmark_df = self._benchmark_data
        
        if benchmark_df is None or len(benchmark_df) == 0:
            logger.debug("No benchmark data available for cross-asset features")
            return features
        
        # Align indices
        common_idx = df.index.intersection(benchmark_df.index)
        if len(common_idx) < 20:
            logger.debug("Insufficient common dates for cross-asset features")
            return features
        
        stock_returns = df.loc[common_idx, 'close'].pct_change()
        bench_returns = benchmark_df.loc[common_idx, 'close'].pct_change()
        
        # Relative Strength
        features['relative_strength'] = (
            df['close'] / df['close'].iloc[0]
        ) / (
            benchmark_df['close'] / benchmark_df['close'].iloc[0]
        )
        
        # Rolling correlation
        features['correlation_60d'] = stock_returns.rolling(
            self.config.correlation_period
        ).corr(bench_returns)
        
        # Beta (rolling)
        def calc_beta(stock, bench):
            if len(stock) < 2 or stock.std() == 0 or bench.std() == 0:
                return np.nan
            return stock.cov(bench) / bench.var()
        
        features['beta'] = stock_returns.rolling(
            self.config.beta_period
        ).apply(lambda x: calc_beta(x, bench_returns.loc[x.index]), raw=False)
        
        # Alpha (Jensen's Alpha approximation)
        risk_free_rate = 0.03 / 252  # Approximate daily risk-free rate
        
        # Relative performance periods
        for period in [5, 10, 20, 60]:
            stock_ret = df['close'].pct_change(period)
            bench_ret = benchmark_df['close'].pct_change(period)
            features[f'relative_return_{period}d'] = stock_ret - bench_ret
        
        return features
    
    # ===== STATISTICAL FEATURES =====
    
    def compute_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute statistical features."""
        features = pd.DataFrame(index=df.index)
        
        returns = df['close'].pct_change()
        
        # Rolling statistics
        for period in [20, 60]:
            features[f'return_mean_{period}d'] = returns.rolling(period).mean() * 252
            features[f'return_std_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
            features[f'return_skew_{period}d'] = returns.rolling(period).skew()
            features[f'return_kurt_{period}d'] = returns.rolling(period).kurt()
            
            # Sharpe ratio (assuming 3% risk-free)
            risk_free = 0.03 / 252
            excess_ret = returns - risk_free
            features[f'sharpe_{period}d'] = (
                excess_ret.rolling(period).mean() / returns.rolling(period).std()
            ) * np.sqrt(252)
        
        # Z-score
        features['price_zscore'] = (
            df['close'] - df['close'].rolling(50).mean()
        ) / df['close'].rolling(50).std()
        
        # Percentile rank
        features['price_percentile'] = df['close'].rolling(252).apply(
            lambda x: (x.iloc[-1] > x[:-1]).sum() / len(x) * 100 if len(x) > 1 else 50
        )
        
        # Maximum drawdown (rolling)
        rolling_max = df['close'].rolling(60).max()
        drawdown = (df['close'] - rolling_max) / rolling_max
        features['drawdown_60d'] = drawdown * 100
        features['max_drawdown_60d'] = drawdown.rolling(60).min() * 100
        
        return features
    
    # ===== MAIN COMPUTATION =====
    
    def compute_all(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        categories: Optional[List[FeatureCategory]] = None
    ) -> pd.DataFrame:
        """
        Compute all features for a DataFrame.
        
        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            benchmark_df: Optional benchmark data for cross-asset features
            categories: Optional list of categories to compute (default: all)
        
        Returns:
            DataFrame with all computed features
        """
        if len(df) < 30:
            logger.warning("Insufficient data for feature computation (need at least 30 bars)")
            return pd.DataFrame(index=df.index)
        
        all_features = pd.DataFrame(index=df.index)
        
        categories = categories or list(FeatureCategory)
        
        if FeatureCategory.MOMENTUM in categories:
            logger.debug("Computing momentum features...")
            all_features = pd.concat([all_features, self.compute_momentum_features(df)], axis=1)
        
        if FeatureCategory.TREND in categories:
            logger.debug("Computing trend features...")
            all_features = pd.concat([all_features, self.compute_trend_features(df)], axis=1)
        
        if FeatureCategory.VOLATILITY in categories:
            logger.debug("Computing volatility features...")
            all_features = pd.concat([all_features, self.compute_volatility_features(df)], axis=1)
        
        if FeatureCategory.VOLUME in categories:
            logger.debug("Computing volume features...")
            all_features = pd.concat([all_features, self.compute_volume_features(df)], axis=1)
        
        if FeatureCategory.PRICE_ACTION in categories:
            logger.debug("Computing price action features...")
            all_features = pd.concat([all_features, self.compute_price_action_features(df)], axis=1)
        
        if FeatureCategory.CANDLESTICK in categories:
            logger.debug("Computing candlestick features...")
            all_features = pd.concat([all_features, self.compute_candlestick_features(df)], axis=1)
        
        if FeatureCategory.CROSS_ASSET in categories and benchmark_df is not None:
            logger.debug("Computing cross-asset features...")
            all_features = pd.concat([
                all_features, 
                self.compute_cross_asset_features(df, benchmark_df)
            ], axis=1)
        
        if FeatureCategory.STATISTICAL in categories:
            logger.debug("Computing statistical features...")
            all_features = pd.concat([all_features, self.compute_statistical_features(df)], axis=1)
        
        # Handle infinities and extreme values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Computed {len(all_features.columns)} features for {len(df)} bars")
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be computed."""
        # Create a dummy DataFrame to get column names
        dummy_data = {
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'volume': np.abs(np.random.randn(100)) * 1000000
        }
        dummy_df = pd.DataFrame(dummy_data)
        features = self.compute_all(dummy_df)
        return list(features.columns)


# Singleton instance
_feature_engine: Optional[FeatureEngine] = None


def get_feature_engine(config: Optional[FeatureConfig] = None) -> FeatureEngine:
    """Get the singleton FeatureEngine instance."""
    global _feature_engine
    if _feature_engine is None:
        _feature_engine = FeatureEngine(config)
    return _feature_engine
