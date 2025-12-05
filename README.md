# Living Trading System

A self-evolving algorithmic trading system for the Korean stock market (KIS).

## Philosophy

This system operates as a living algorithm—it doesn't just optimize parameters within fixed rules. 
It discovers patterns, generates hypotheses, tests them, and evolves its own structure over time.

## Immutable Invariants (These NEVER change)

1. **Max 25% single position** - No single holding can exceed 25% of portfolio
2. **30% drawdown = full stop** - System halts and requires manual restart
3. **No leverage** - Only trade with available capital
4. **KIS-listed only** - Only instruments tradeable via Korea Investment & Securities
5. **All decisions logged** - Complete audit trail of every decision
6. **Human override** - Owner can halt system at any time

## Architecture

```
living-trading-system/
├── core/                 # Fundamental building blocks
│   ├── invariants.py     # Immutable rules
│   ├── clock.py          # Market time awareness
│   └── events.py         # Event bus for module communication
├── perception/           # Market data ingestion and feature computation
│   ├── data_feed.py      # KIS API data retrieval
│   ├── universe.py       # Dynamic instrument discovery
│   └── features.py       # Automatic feature generation
├── hypothesis/           # Strategy generation and management
│   ├── generator.py      # Creates new hypotheses
│   ├── hypothesis.py     # Hypothesis data structure
│   └── registry.py       # Tracks all hypotheses (active, testing, retired)
├── validation/           # Hypothesis testing pipeline
│   ├── backtester.py     # Historical testing
│   ├── paper_trader.py   # Forward testing without capital
│   └── promotion.py      # Rules for graduating hypotheses
├── portfolio/            # Position management
│   ├── mind.py           # Portfolio construction logic
│   ├── optimizer.py      # Capital allocation
│   └── risk.py           # Risk calculations
├── execution/            # Order management
│   ├── broker.py         # KIS API order interface
│   ├── order_manager.py  # Order lifecycle management
│   └── execution_algo.py # Smart order routing
├── memory/               # Persistent state
│   ├── database.py       # SQLite interface
│   ├── models.py         # Data models
│   └── journal.py        # Decision logging
├── config/               # Configuration files
│   ├── settings.yaml     # System settings
│   └── credentials.yaml  # API keys (gitignored)
├── tests/                # Test suite
└── logs/                 # Runtime logs
```

## Setup Instructions

### Prerequisites

1. Python 3.11 or higher
2. KIS Developers API access (실전투자 + 모의투자)
3. Cloud server or always-on local machine

### Installation

```bash
# Clone or download this directory
cd living-trading-system

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit credentials
cp config/credentials.example.yaml config/credentials.yaml
# Edit credentials.yaml with your KIS API keys
```

### First Run

```bash
# Verify installation
python -m core.verify_setup

# Start in paper trading mode
python main.py --mode paper

# Start live trading (after validation period)
python main.py --mode live
```

## Development Timeline

- **Phase 1 (Week 1-2):** Foundation - Data pipeline, database, core infrastructure
- **Phase 2 (Week 3-4):** Perception - Market data, feature generation
- **Phase 3 (Week 5-7):** Hypothesis Engine - Pattern discovery, hypothesis generation
- **Phase 4 (Week 8-9):** Validation - Backtesting, paper trading pipeline
- **Phase 5 (Week 10-11):** Portfolio Mind - Position sizing, allocation
- **Phase 6 (Week 12):** Execution - KIS API integration, order management
- **Phase 7 (Month 4-6):** Incubation - Paper trading, validation, gradual capital deployment

## Current Status

🚧 **Phase 1: Foundation** - In Progress
