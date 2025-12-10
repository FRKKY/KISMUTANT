# KISMUTANT - Living Trading System Architecture

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                              │
│                   (Main Control Loop)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  PERCEPTION  │───►│  HYPOTHESIS  │───►│  PORTFOLIO   │       │
│  │              │    │              │    │              │       │
│  │ • Universe   │    │ • Registry   │    │ • Allocator  │       │
│  │ • DataFetch  │    │ • Signals    │    │ • Positions  │       │
│  │ • Features   │    │ • Factory    │    │ • Kelly      │       │
│  │ • Patterns   │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  VALIDATION  │    │   MEMORY     │    │  EXECUTION   │       │
│  │              │    │              │    │              │       │
│  │ • Backtester │    │ • Journal    │    │ • Broker     │       │
│  │ • Metrics    │    │ • Database   │    │ • Orders     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                          CORE                                    │
│  • Events (pub/sub) • Clock (market time) • Invariants (safety) │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
KISMUTANT/
├── core/                   # Phase 1 - Foundation
│   ├── __init__.py
│   ├── events.py           # Event bus (pub/sub)
│   ├── clock.py            # Market time awareness
│   └── invariants.py       # Safety rules (immutable)
│
├── memory/                 # Phase 1 - Data persistence
│   ├── __init__.py
│   ├── models.py           # Database models
│   └── journal.py          # Decision journal
│
├── execution/              # Phase 1 - Order execution
│   ├── __init__.py
│   └── broker.py           # KIS API integration
│
├── interface/              # Phase 1 - User interfaces
│   ├── web/
│   │   └── dashboard.py    # Web dashboard
│   └── telegram/
│       └── bot.py          # Telegram bot
│
├── perception/             # Phase 2 - Market data & analysis
│   ├── __init__.py
│   ├── universe.py         # ETF universe management
│   ├── data_fetcher.py     # OHLCV data retrieval
│   ├── features.py         # Technical indicators (170+)
│   └── patterns.py         # Pattern detection
│
├── hypothesis/             # Phase 3 - Strategy management
│   ├── __init__.py
│   ├── models.py           # Hypothesis, Signal, Metrics
│   ├── registry.py         # Strategy lifecycle
│   └── signals.py          # Signal generation
│
├── portfolio/              # Phase 4 - Capital management
│   ├── __init__.py
│   ├── allocator.py        # Half-Kelly position sizing
│   └── positions.py        # Position tracking & P&L
│
├── validation/             # Phase 5 - Backtesting
│   ├── __init__.py
│   ├── metrics.py          # Performance calculations
│   └── backtester.py       # Historical simulation
│
├── tests/                  # Test suite
│   └── test_perception.py
│
├── orchestrator.py         # Phase 6 - Main control loop
├── main.py                 # Entry point
└── requirements.txt
```

## Data Flow

### Pattern to Trade Flow

```
1. PERCEPTION
   KIS API → DataFetcher → Features → PatternDetector
                                           │
                                           ▼
2. HYPOTHESIS                    DetectedPattern
   HypothesisFactory.from_pattern() → Hypothesis
                                           │
                                           ▼
3. VALIDATION                    Hypothesis (INCUBATING)
   Backtester.run() → BacktestResult
                                           │
                      (if passes criteria) ▼
4. REGISTRY                      Hypothesis (PAPER_TRADING)
   30 days + metrics pass
                                           │
                                           ▼
5. LIVE TRADING                  Hypothesis (LIVE)
   SignalGenerator → TradingSignal
                                           │
                                           ▼
6. PORTFOLIO
   CapitalAllocator.size_position() → PositionSizing
                                           │
                                           ▼
7. EXECUTION
   Broker.place_order() → Order → Fill → Position
```

## Strategy Lifecycle

```
┌─────────────┐     Backtest      ┌───────────────┐
│ INCUBATING  │─────passes────────►│ PAPER_TRADING │
│             │                    │               │
│ • Backtest  │                    │ • 30 days min │
│ • Sharpe≥1  │                    │ • Sharpe≥0.75 │
│ • DD≤15%    │                    │ • DD≤10%      │
│ • 30 trades │                    │ • 20 trades   │
└─────────────┘                    └───────┬───────┘
                                          │
                                   passes │
                                          ▼
                                   ┌──────────┐
                                   │   LIVE   │
                                   │          │
                                   │ • Real $ │
                                   │ • Kelly  │
                                   └────┬─────┘
                                        │
                          underperforms │
                                        ▼
                                   ┌──────────┐
                                   │ RETIRED  │
                                   └──────────┘
```

## Key Configuration

### Capital Allocation (portfolio/allocator.py)
```python
AllocationConfig(
    total_capital=10_000_000,      # ₩10M
    method=AllocationMethod.HALF_KELLY,
    max_strategies=3,              # Max 3 live strategies
    max_per_strategy_pct=0.40,     # Max 40% per strategy
    max_per_position_pct=0.15,     # Max 15% per position
)
```

### Promotion Criteria (hypothesis/models.py)
```python
# Backtest → Paper
min_sharpe=1.0
max_drawdown=0.15
min_trades=30
min_profit_factor=1.5

# Paper → Live
min_days=30
min_sharpe=0.75
max_drawdown=0.10
min_trades=20
```

### Demotion Criteria
```python
rolling_sharpe < 0.5       # 30-day rolling
max_drawdown > 0.10        # 10% drawdown
consecutive_losses >= 5    # 5 losses in a row
```

## Event Types

Add these to `core/events.py`:

```python
class EventType(str, Enum):
    # ... existing ...
    
    # Perception
    PATTERN_DISCOVERED = "pattern_discovered"
    UNIVERSE_UPDATED = "universe_updated"
    
    # Hypothesis
    STRATEGY_REGISTERED = "strategy_registered"
    STRATEGY_PROMOTED = "strategy_promoted"
    STRATEGY_DEMOTED = "strategy_demoted"
    STRATEGY_RETIRED = "strategy_retired"
    
    # Signals
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_EXPIRED = "signal_expired"
    
    # Positions
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    
    # Validation
    BACKTEST_COMPLETED = "backtest_completed"
    BACKTEST_FAILED = "backtest_failed"
    
    # Orchestrator
    SYSTEM_ALERT = "system_alert"
```

## Running the System

```python
import asyncio
from orchestrator import get_orchestrator, OrchestratorConfig
from execution.broker import KISBroker

# Initialize broker
broker = KISBroker()

# Configure orchestrator
config = OrchestratorConfig(
    initial_capital=10_000_000,
    max_live_strategies=3,
    auto_generate_hypotheses=True,
    auto_backtest=True
)

# Create and initialize
orchestrator = get_orchestrator(broker, config)
await orchestrator.initialize()

# Start trading loop
await orchestrator.start()
```

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
sqlalchemy>=2.0.0
httpx>=0.25.0
loguru>=0.7.2
fastapi>=0.104.0
uvicorn>=0.24.0
python-telegram-bot>=20.7
```
