"""
METRICS COLLECTION - System performance tracking

Provides:
- Counter, Gauge, Histogram metrics
- Prometheus-compatible export
- In-memory metrics storage
- Periodic metrics logging
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from threading import Lock
from loguru import logger


@dataclass
class MetricValue:
    """A single metric value with timestamp."""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """A monotonically increasing counter."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = Lock()

    def inc(self, value: float = 1.0) -> None:
        """Increment the counter."""
        with self._lock:
            self._value += value

    def get(self) -> float:
        """Get current value."""
        return self._value

    def reset(self) -> None:
        """Reset to zero."""
        with self._lock:
            self._value = 0.0


class Gauge:
    """A value that can go up and down."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = Lock()

    def set(self, value: float) -> None:
        """Set the gauge value."""
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment the gauge."""
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement the gauge."""
        with self._lock:
            self._value -= value

    def get(self) -> float:
        """Get current value."""
        return self._value


class Histogram:
    """A histogram for tracking value distributions."""

    # Default buckets for latency measurements (in seconds)
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf'))

    def __init__(self, name: str, description: str = "", buckets: tuple = None):
        self.name = name
        self.description = description
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._values: deque = deque(maxlen=10000)  # Keep last 10k observations
        self._sum = 0.0
        self._count = 0
        self._lock = Lock()

    def observe(self, value: float) -> None:
        """Observe a value."""
        with self._lock:
            self._values.append(value)
            self._sum += value
            self._count += 1

    def get_count(self) -> int:
        """Get observation count."""
        return self._count

    def get_sum(self) -> float:
        """Get sum of observations."""
        return self._sum

    def get_mean(self) -> float:
        """Get mean of observations."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def get_percentile(self, p: float) -> float:
        """Get a percentile (e.g., 0.95 for p95)."""
        if not self._values:
            return 0.0
        sorted_values = sorted(self._values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_bucket_counts(self) -> Dict[float, int]:
        """Get counts per bucket."""
        counts = {b: 0 for b in self.buckets}
        for value in self._values:
            for bucket in self.buckets:
                if value <= bucket:
                    counts[bucket] += 1
                    break
        return counts


class Timer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram):
        self.histogram = histogram
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        self.histogram.observe(elapsed)


class MetricsRegistry:
    """Central registry for all metrics."""

    _instance: Optional['MetricsRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = Lock()

        # Create default metrics
        self._create_default_metrics()

        self._initialized = True
        logger.info("MetricsRegistry initialized")

    def _create_default_metrics(self) -> None:
        """Create default system metrics."""
        # Trading metrics
        self.counter("trades_total", "Total number of trades executed")
        self.counter("trades_buy", "Total buy trades")
        self.counter("trades_sell", "Total sell trades")
        self.counter("signals_generated", "Total signals generated")
        self.counter("signals_executed", "Total signals executed")
        self.counter("signals_rejected", "Total signals rejected")

        # API metrics
        self.counter("api_requests_total", "Total API requests")
        self.counter("api_errors_total", "Total API errors")
        self.histogram("api_latency_seconds", "API request latency")

        # Portfolio metrics
        self.gauge("portfolio_equity", "Total portfolio equity in KRW")
        self.gauge("portfolio_cash", "Cash available in KRW")
        self.gauge("portfolio_positions", "Number of open positions")
        self.gauge("portfolio_daily_pnl", "Daily P&L in KRW")
        self.gauge("portfolio_drawdown_pct", "Current drawdown percentage")

        # System metrics
        self.gauge("hypotheses_live", "Number of live hypotheses")
        self.gauge("hypotheses_paper", "Number of paper trading hypotheses")
        self.gauge("hypotheses_total", "Total hypotheses in registry")
        self.gauge("research_papers_fetched", "Total papers fetched")
        self.gauge("research_ideas_extracted", "Total ideas extracted")

        # Performance metrics
        self.histogram("signal_generation_seconds", "Signal generation latency")
        self.histogram("trade_execution_seconds", "Trade execution latency")
        self.histogram("data_fetch_seconds", "Data fetch latency")

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]

    def histogram(self, name: str, description: str = "", buckets: tuple = None) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets)
            return self._histograms[name]

    def time(self, histogram_name: str) -> Timer:
        """Get a timer context manager for a histogram."""
        return Timer(self.histogram(histogram_name))

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        result = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        for name, counter in self._counters.items():
            result["counters"][name] = counter.get()

        for name, gauge in self._gauges.items():
            result["gauges"][name] = gauge.get()

        for name, histogram in self._histograms.items():
            result["histograms"][name] = {
                "count": histogram.get_count(),
                "sum": histogram.get_sum(),
                "mean": histogram.get_mean(),
                "p50": histogram.get_percentile(0.5),
                "p95": histogram.get_percentile(0.95),
                "p99": histogram.get_percentile(0.99),
            }

        return result

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, counter in self._counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.get()}")

        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {gauge.get()}")

        for name, histogram in self._histograms.items():
            lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"{name}_count {histogram.get_count()}")
            lines.append(f"{name}_sum {histogram.get_sum()}")
            for bucket, count in histogram.get_bucket_counts().items():
                bucket_str = "+Inf" if bucket == float('inf') else str(bucket)
                lines.append(f'{name}_bucket{{le="{bucket_str}"}} {count}')

        return "\n".join(lines)

    def log_summary(self) -> None:
        """Log a summary of key metrics."""
        metrics = self.get_all_metrics()

        logger.info("=== Metrics Summary ===")

        # Key counters
        logger.info(f"Trades: {metrics['counters'].get('trades_total', 0)}")
        logger.info(f"Signals: {metrics['counters'].get('signals_generated', 0)} generated, "
                   f"{metrics['counters'].get('signals_executed', 0)} executed")

        # Key gauges
        logger.info(f"Portfolio: ₩{metrics['gauges'].get('portfolio_equity', 0):,.0f}")
        logger.info(f"Positions: {metrics['gauges'].get('portfolio_positions', 0)}")
        logger.info(f"Daily P&L: ₩{metrics['gauges'].get('portfolio_daily_pnl', 0):,.0f}")

        # Key latencies
        api_latency = metrics['histograms'].get('api_latency_seconds', {})
        if api_latency.get('count', 0) > 0:
            logger.info(f"API Latency: p50={api_latency.get('p50', 0)*1000:.0f}ms, "
                       f"p95={api_latency.get('p95', 0)*1000:.0f}ms")


# Singleton accessor
_registry_instance: Optional[MetricsRegistry] = None


def get_metrics() -> MetricsRegistry:
    """Get the singleton MetricsRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = MetricsRegistry()
    return _registry_instance


# Convenience functions
def inc_counter(name: str, value: float = 1.0) -> None:
    """Increment a counter."""
    get_metrics().counter(name).inc(value)


def set_gauge(name: str, value: float) -> None:
    """Set a gauge value."""
    get_metrics().gauge(name).set(value)


def observe_histogram(name: str, value: float) -> None:
    """Observe a histogram value."""
    get_metrics().histogram(name).observe(value)


def time_operation(histogram_name: str) -> Timer:
    """Get a timer for an operation."""
    return get_metrics().time(histogram_name)
