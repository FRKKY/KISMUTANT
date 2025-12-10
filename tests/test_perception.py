"""
Test Perception Layer - Verify all modules load and work correctly.

Run with: python -m pytest tests/test_perception.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestPerceptionImports:
    """Test that all perception modules import correctly."""
    
    def test_import_universe(self):
        from perception.universe import (
            UniverseManager,
            UniverseFilter,
            ETFInfo,
            ETFCategory
        )
        assert UniverseManager is not None
        assert UniverseFilter is not None
    
    def test_import_data_fetcher(self):
        from perception.data_fetcher import (
            DataFetcher,
            DataFetcherConfig,
            OHLCVBar,
            Timeframe
        )
        assert DataFetcher is not None
        assert Timeframe.DAILY.value == "1d"
    
    def test_import_features(self):
        from perception.features import (
            FeatureEngine,
            FeatureConfig,
            FeatureCategory
        )
        assert FeatureEngine is not None
        assert FeatureCategory.MOMENTUM.value == "momentum"
    
    def test_import_patterns(self):
        from perception.patterns import (
            PatternDetector,
            PatternType,
            PatternDirection,
            PatternStrength
        )
        assert PatternDetector is not None
        assert PatternDirection.BULLISH.value == "bullish"
    
    def test_import_perception_layer(self):
        from perception import PerceptionLayer
        assert PerceptionLayer is not None


class TestFeatureEngine:
    """Test feature computation with synthetic data."""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        
        # Generate realistic price series
        returns = np.random.randn(n) * 0.02
        close = 10000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(n) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(n) * 0.01)),
            'low': close * (1 - np.abs(np.random.randn(n) * 0.01)),
            'close': close,
            'volume': np.abs(np.random.randn(n) * 1000000 + 5000000).astype(int)
        })
        
        df.index = pd.date_range(end=datetime.now(), periods=n, freq='D')
        return df
    
    def test_feature_engine_init(self):
        from perception.features import FeatureEngine
        engine = FeatureEngine()
        assert engine is not None
    
    def test_compute_rsi(self, sample_ohlcv):
        from perception.features import FeatureEngine
        engine = FeatureEngine()
        
        rsi = engine.compute_rsi(sample_ohlcv, period=14)
        
        assert len(rsi) == len(sample_ohlcv)
        assert rsi.dropna().min() >= 0
        assert rsi.dropna().max() <= 100
    
    def test_compute_macd(self, sample_ohlcv):
        from perception.features import FeatureEngine
        engine = FeatureEngine()
        
        macd, signal, histogram = engine.compute_macd(sample_ohlcv)
        
        assert len(macd) == len(sample_ohlcv)
        assert len(signal) == len(sample_ohlcv)
        assert len(histogram) == len(sample_ohlcv)
    
    def test_compute_bollinger_bands(self, sample_ohlcv):
        from perception.features import FeatureEngine
        engine = FeatureEngine()
        
        middle, upper, lower, pct_b = engine.compute_bollinger_bands(sample_ohlcv)
        
        # Upper should be above middle, middle above lower
        valid_idx = middle.dropna().index
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
    
    def test_compute_all_features(self, sample_ohlcv):
        from perception.features import FeatureEngine
        engine = FeatureEngine()
        
        features = engine.compute_all(sample_ohlcv)
        
        assert len(features) == len(sample_ohlcv)
        assert len(features.columns) > 50  # Should have many features
        
        # Check some expected columns exist
        assert 'rsi_14' in features.columns
        assert 'macd' in features.columns
        assert 'bb_upper' in features.columns
        assert 'atr' in features.columns


class TestPatternDetector:
    """Test pattern detection."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV and features data."""
        from perception.features import FeatureEngine
        
        np.random.seed(42)
        n = 100
        
        returns = np.random.randn(n) * 0.02
        close = 10000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(n) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(n) * 0.01)),
            'low': close * (1 - np.abs(np.random.randn(n) * 0.01)),
            'close': close,
            'volume': np.abs(np.random.randn(n) * 1000000 + 5000000).astype(int)
        })
        df.index = pd.date_range(end=datetime.now(), periods=n, freq='D')
        
        engine = FeatureEngine()
        features = engine.compute_all(df)
        
        return df, features
    
    def test_pattern_detector_init(self):
        from perception.patterns import PatternDetector
        detector = PatternDetector()
        assert detector is not None
    
    def test_scan_symbol(self, sample_data):
        from perception.patterns import PatternDetector
        
        df, features = sample_data
        detector = PatternDetector()
        
        patterns = detector.scan("TEST", df, features)
        
        # May or may not find patterns, but should not error
        assert isinstance(patterns, list)
    
    def test_get_stats(self, sample_data):
        from perception.patterns import PatternDetector
        
        df, features = sample_data
        detector = PatternDetector()
        detector.scan("TEST", df, features)
        
        stats = detector.get_stats()
        
        assert 'active_patterns' in stats
        assert 'total_detected' in stats


class TestUniverseManager:
    """Test universe management."""
    
    def test_universe_filter(self):
        from perception.universe import UniverseFilter, ETFInfo, ETFCategory
        
        filter_config = UniverseFilter(
            min_aum=10_000_000_000,
            min_avg_daily_volume=10_000,
            include_leverage=False,
            include_inverse=False
        )
        
        # Create a test ETF that passes
        etf_pass = ETFInfo(
            symbol="069500",
            name="KODEX 200",
            category=ETFCategory.INDEX_KOSPI,
            aum=50_000_000_000_000,
            avg_daily_volume=1_000_000,
            avg_daily_value=50_000_000_000,
            current_price=35000,
            is_leverage=False,
            is_inverse=False,
            is_tradeable=True
        )
        
        # Create a test ETF that fails (leverage)
        etf_fail = ETFInfo(
            symbol="122630",
            name="KODEX 레버리지",
            category=ETFCategory.LEVERAGE_2X,
            aum=50_000_000_000_000,
            avg_daily_volume=1_000_000,
            avg_daily_value=50_000_000_000,
            current_price=20000,
            is_leverage=True,
            is_inverse=False,
            is_tradeable=True
        )
        
        assert filter_config.passes(etf_pass) == True
        assert filter_config.passes(etf_fail) == False
    
    def test_etf_classification(self):
        from perception.universe import UniverseManager
        
        manager = UniverseManager()
        
        # Test classification
        from perception.universe import ETFCategory
        
        assert manager._classify_etf("KODEX 200") == ETFCategory.INDEX_KOSPI
        assert manager._classify_etf("TIGER 미국S&P500") == ETFCategory.INTL_US
        assert manager._classify_etf("KODEX 레버리지") == ETFCategory.LEVERAGE_2X
        assert manager._classify_etf("KODEX 인버스") == ETFCategory.INVERSE_1X


class TestDataFetcher:
    """Test data fetcher initialization."""
    
    def test_data_fetcher_init(self):
        from perception.data_fetcher import DataFetcher, DataFetcherConfig
        
        config = DataFetcherConfig(
            requests_per_second=5.0,
            batch_size=20
        )
        
        fetcher = DataFetcher(config=config)
        assert fetcher is not None
    
    def test_timeframes(self):
        from perception.data_fetcher import Timeframe
        
        assert Timeframe.MINUTE_1.value == "1m"
        assert Timeframe.MINUTE_5.value == "5m"
        assert Timeframe.DAILY.value == "1d"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
