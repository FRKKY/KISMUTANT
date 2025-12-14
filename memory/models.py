"""
DATABASE MODELS - The Memory Structure of the Living System

This defines how the system remembers:
- All market data it has ever seen
- Every hypothesis it has ever generated
- Every trade it has ever made
- Every decision and the context in which it was made
- Its own evolution over time

Supports both SQLite (local development) and PostgreSQL (cloud deployment).
Configure via DATABASE_URL environment variable.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import os

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Date, Text, ForeignKey, Enum as SQLEnum,
    UniqueConstraint, Index, JSON, Numeric, event
)
from sqlalchemy.orm import (
    declarative_base, relationship, sessionmaker, Session
)
from sqlalchemy.pool import StaticPool, QueuePool
from loguru import logger

Base = declarative_base()


# === ENUMS ===

class HypothesisStatus(str, Enum):
    """Status of a hypothesis in its lifecycle."""
    DRAFT = "draft"              # Just created, not yet tested
    BACKTESTING = "backtesting"  # Being tested on historical data
    PAPER_TRADING = "paper"      # Forward testing without real capital
    INCUBATING = "incubating"    # Live with small capital
    ACTIVE = "active"            # Fully deployed
    PAUSED = "paused"            # Temporarily stopped
    RETIRED = "retired"          # No longer used but kept for reference
    FAILED = "failed"            # Failed validation


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"  # Not used in ISA, but keeping for completeness
    CLOSE = "close"
    HOLD = "hold"


# === MARKET DATA ===

class Instrument(Base):
    """
    A tradeable instrument.
    
    The system discovers instruments dynamically - this is not a fixed list.
    """
    __tablename__ = "instruments"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200))
    instrument_type = Column(String(50))  # stock, etf, etn, bond, etc.
    sector = Column(String(100))
    market = Column(String(20))  # KOSPI, KOSDAQ, etc.
    
    # Dynamic attributes discovered by the system
    is_tradeable = Column(Boolean, default=True)
    avg_daily_volume = Column(Float)
    avg_spread = Column(Float)
    volatility_30d = Column(Float)
    
    # Metadata
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(JSON, default=dict)  # Flexible storage for discovered attributes
    
    # Relationships
    prices = relationship("PriceBar", back_populates="instrument")
    
    def __repr__(self):
        return f"<Instrument(symbol={self.symbol}, type={self.instrument_type})>"


class PriceBar(Base):
    """
    OHLCV price data.
    
    Stored for all instruments the system monitors.
    """
    __tablename__ = "price_bars"
    
    id = Column(Integer, primary_key=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    
    date = Column(Date, nullable=False, index=True)
    timeframe = Column(String(10), default="1d")  # 1d, 1h, 15m, etc.
    
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    # Additional data
    vwap = Column(Float)
    trade_count = Column(Integer)
    
    # Relationships
    instrument = relationship("Instrument", back_populates="prices")
    
    __table_args__ = (
        UniqueConstraint("instrument_id", "date", "timeframe", name="unique_price_bar"),
        Index("idx_price_lookup", "instrument_id", "date"),
        Index("idx_price_timeframe", "instrument_id", "date", "timeframe"),  # Composite for queries
        Index("idx_recent_prices", "date", "timeframe"),  # For scanning recent data
    )


class Feature(Base):
    """
    Computed features for instruments.
    
    Features are discovered by the system, not predefined.
    """
    __tablename__ = "features"
    
    id = Column(Integer, primary_key=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(Float)
    
    # How was this feature computed?
    computation_method = Column(Text)  # Store the formula or method
    
    __table_args__ = (
        UniqueConstraint("instrument_id", "date", "feature_name", name="unique_feature"),
        Index("idx_feature_lookup", "instrument_id", "date", "feature_name"),
    )


# === HYPOTHESIS SYSTEM ===

class Hypothesis(Base):
    """
    A hypothesis is a belief about the market that generates trading signals.
    
    Hypotheses are born, tested, promoted, and eventually retired.
    This is the core of the "living" nature of the system.
    """
    __tablename__ = "hypotheses"
    
    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(String(50), unique=True, nullable=False)  # Human-readable ID
    
    # What does this hypothesis believe?
    name = Column(String(200))
    description = Column(Text)
    
    # The actual logic - stored as interpretable code/rules
    logic_type = Column(String(50))  # "rule_based", "ml_model", "genetic", etc.
    logic_definition = Column(JSON)  # The actual rules or model parameters
    
    # Status in lifecycle
    status = Column(SQLEnum(HypothesisStatus), default=HypothesisStatus.DRAFT)
    
    # Where did this hypothesis come from?
    parent_id = Column(String(50))  # If evolved from another hypothesis
    generation = Column(Integer, default=0)  # Evolution generation
    creation_method = Column(String(50))  # "generated", "evolved", "user_defined"
    
    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    
    # Confidence and allocation
    confidence_score = Column(Float, default=0.5)  # System's confidence in this hypothesis
    capital_allocation_pct = Column(Float, default=0.0)  # Current allocation
    
    # Lifecycle timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    backtested_at = Column(DateTime)
    paper_started_at = Column(DateTime)
    live_started_at = Column(DateTime)
    retired_at = Column(DateTime)
    last_signal_at = Column(DateTime)
    
    # Relationships
    signals = relationship("Signal", back_populates="hypothesis")
    performance_snapshots = relationship("HypothesisPerformance", back_populates="hypothesis")
    
    def __repr__(self):
        return f"<Hypothesis(id={self.hypothesis_id}, status={self.status.value})>"


class HypothesisPerformance(Base):
    """
    Time-series of hypothesis performance.
    
    Tracks how well each hypothesis performs over time.
    """
    __tablename__ = "hypothesis_performance"
    
    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"), nullable=False)
    
    date = Column(Date, nullable=False)
    
    # Performance metrics for this period
    trades = Column(Integer, default=0)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    
    # Running statistics
    cumulative_pnl = Column(Float, default=0.0)
    rolling_sharpe_30d = Column(Float)
    rolling_win_rate_30d = Column(Float)
    
    hypothesis = relationship("Hypothesis", back_populates="performance_snapshots")
    
    __table_args__ = (
        UniqueConstraint("hypothesis_id", "date", name="unique_hypothesis_perf"),
    )


class Signal(Base):
    """
    A trading signal generated by a hypothesis.
    
    Every signal is logged, whether acted upon or not.
    """
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"), nullable=False)
    
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False, index=True)
    
    direction = Column(SQLEnum(SignalDirection), nullable=False)
    strength = Column(Float, default=1.0)  # 0.0 to 1.0
    confidence = Column(Float, default=0.5)  # 0.0 to 1.0
    
    # Context at signal time
    price_at_signal = Column(Float)
    features_snapshot = Column(JSON)  # Capture features when signal was generated
    
    # Was this signal acted upon?
    was_executed = Column(Boolean, default=False)
    execution_reason = Column(Text)  # Why it was or wasn't executed
    
    # Outcome tracking
    outcome_pnl = Column(Float)
    outcome_recorded = Column(Boolean, default=False)
    
    hypothesis = relationship("Hypothesis", back_populates="signals")
    
    __table_args__ = (
        Index("idx_signal_lookup", "symbol", "timestamp"),
        Index("idx_signal_hypothesis", "hypothesis_id", "timestamp"),  # For strategy analysis
        Index("idx_signal_executed", "was_executed", "timestamp"),  # For execution analysis
    )


# === PORTFOLIO & TRADING ===

class Position(Base):
    """
    Current and historical positions.
    """
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Position details
    quantity = Column(Integer, nullable=False)
    avg_cost = Column(Float, nullable=False)
    current_price = Column(Float)
    
    # Attribution - which hypothesis/signals led to this position?
    primary_hypothesis_id = Column(String(50))
    entry_signals = Column(JSON)  # List of signal IDs that contributed
    
    # Risk metrics
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_pct = Column(Float, default=0.0)
    max_unrealized_gain = Column(Float, default=0.0)
    max_unrealized_loss = Column(Float, default=0.0)
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    is_open = Column(Boolean, default=True)
    
    trades = relationship("Trade", back_populates="position")


class Trade(Base):
    """
    Every executed trade.
    """
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    
    position_id = Column(Integer, ForeignKey("positions.id"))
    order_id = Column(String(50), index=True)
    
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLEnum(OrderSide), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    
    # Execution details
    executed_at = Column(DateTime, nullable=False)
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)  # Difference from expected price
    
    # Attribution
    hypothesis_id = Column(String(50))
    signal_id = Column(Integer)
    
    # Context
    market_conditions = Column(JSON)  # Market state at execution
    execution_notes = Column(Text)

    position = relationship("Position", back_populates="trades")

    __table_args__ = (
        Index("idx_trade_time", "executed_at"),  # For time-based queries
        Index("idx_trade_hypothesis", "hypothesis_id", "executed_at"),  # For strategy P&L
        Index("idx_trade_symbol_time", "symbol", "executed_at"),  # For symbol history
    )


class Order(Base):
    """
    Order lifecycle tracking.
    """
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, nullable=False)
    
    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum(OrderSide), nullable=False)
    
    order_type = Column(String(20), default="market")  # market, limit, etc.
    quantity = Column(Integer, nullable=False)
    limit_price = Column(Float)
    
    status = Column(SQLEnum(OrderStatus), default=OrderStatus.PENDING)
    
    filled_quantity = Column(Integer, default=0)
    filled_avg_price = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    
    # Attribution
    hypothesis_id = Column(String(50))
    signal_id = Column(Integer)
    
    # Broker response
    broker_order_id = Column(String(100))
    broker_response = Column(JSON)
    error_message = Column(Text)


# === PORTFOLIO STATE ===

class PortfolioSnapshot(Base):
    """
    Daily snapshot of portfolio state.
    
    Used for tracking equity curve, drawdown, etc.
    """
    __tablename__ = "portfolio_snapshots"
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True)
    
    # Values
    total_equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    
    # Performance
    daily_pnl = Column(Float, default=0.0)
    daily_pnl_pct = Column(Float, default=0.0)
    
    # Risk metrics
    drawdown = Column(Float, default=0.0)
    drawdown_pct = Column(Float, default=0.0)
    high_water_mark = Column(Float)
    
    # Activity
    trades_today = Column(Integer, default=0)
    signals_today = Column(Integer, default=0)
    
    # Breakdown by hypothesis
    hypothesis_allocations = Column(JSON)  # {hypothesis_id: allocation_pct}
    hypothesis_pnl = Column(JSON)  # {hypothesis_id: pnl}


# === LEARNING & MEMORY ===

class Decision(Base):
    """
    Log of every significant decision the system makes.
    
    This is crucial for the system to learn from its own behavior.
    """
    __tablename__ = "decisions"
    
    id = Column(Integer, primary_key=True)
    decision_id = Column(String(50), unique=True, nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    decision_type = Column(String(50), nullable=False)  # trade, allocation, hypothesis_promotion, etc.
    
    # What was decided?
    description = Column(Text)
    decision_outcome = Column(JSON)  # The actual decision made
    
    # Context at decision time
    context = Column(JSON)  # All relevant state when decision was made
    
    # What alternatives were considered?
    alternatives_considered = Column(JSON)
    
    # Why was this decision made?
    reasoning = Column(Text)
    confidence = Column(Float)
    
    # Was it good?
    outcome_evaluated = Column(Boolean, default=False)
    outcome_quality = Column(Float)  # -1 to 1, evaluated later
    outcome_notes = Column(Text)


class LearningEvent(Base):
    """
    Records when the system learns something.
    
    Could be: pattern discovered, parameter updated, hypothesis evolved, etc.
    """
    __tablename__ = "learning_events"
    
    id = Column(Integer, primary_key=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String(50), nullable=False)
    
    # What was learned?
    description = Column(Text)
    
    # What changed?
    before_state = Column(JSON)
    after_state = Column(JSON)
    
    # Supporting evidence
    evidence = Column(JSON)  # Data that supported this learning
    confidence = Column(Float)
    
    # Impact tracking
    related_hypotheses = Column(JSON)  # Which hypotheses were affected
    expected_impact = Column(Text)


class SystemState(Base):
    """
    Periodic snapshot of the entire system state.
    
    Used for recovery and analysis of system evolution.
    """
    __tablename__ = "system_states"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Serialized state
    active_hypotheses = Column(JSON)
    portfolio_state = Column(JSON)
    risk_state = Column(JSON)
    
    # System health
    is_healthy = Column(Boolean, default=True)
    health_notes = Column(Text)
    
    # Configuration at this time
    config_snapshot = Column(JSON)


# === DATABASE SETUP ===

def get_database_url() -> str:
    """
    Get database URL from environment or default to SQLite.

    Priority:
    1. DATABASE_URL environment variable (for Railway/Render/Heroku)
    2. POSTGRES_URL environment variable (alternate name)
    3. Default SQLite for local development
    """
    # Check for cloud database URL
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")

    if db_url:
        # Railway/Heroku use postgres:// but SQLAlchemy needs postgresql://
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        logger.info(f"Using cloud database: {db_url.split('@')[-1] if '@' in db_url else 'configured'}")
        return db_url

    # Default to SQLite for local development
    return "sqlite:///memory/trading_system.db"


class Database:
    """
    Database manager for the living trading system.

    Supports:
    - SQLite for local development
    - PostgreSQL for cloud deployment (Railway, Render, Heroku)

    Configure via DATABASE_URL environment variable.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: SQLAlchemy database URL. If None, auto-detect from environment.
        """
        self.db_path = db_path or get_database_url()
        self.is_sqlite = "sqlite" in self.db_path
        self.is_postgres = "postgresql" in self.db_path

        # Configure engine based on database type
        if self.is_sqlite:
            self.engine = create_engine(
                self.db_path,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False
            )
        else:
            # PostgreSQL with connection pooling
            self.engine = create_engine(
                self.db_path,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,  # Recycle connections every 30 minutes
                echo=False
            )

            # Handle SSL for cloud PostgreSQL
            if "sslmode" not in self.db_path:
                # Most cloud providers require SSL
                self.engine = create_engine(
                    self.db_path + ("&" if "?" in self.db_path else "?") + "sslmode=require",
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_recycle=1800,
                    echo=False
                )

        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False)
        logger.info(f"Database initialized: {'SQLite' if self.is_sqlite else 'PostgreSQL'}")

    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def close(self):
        """Close database connections."""
        self.engine.dispose()
        logger.info("Database connections closed")

    def health_check(self) -> bool:
        """Check if database is accessible."""
        try:
            from sqlalchemy import text
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Convenience function for getting database
_db_instance: Optional[Database] = None

def get_database(db_path: Optional[str] = None) -> Database:
    """Get or create the database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
        _db_instance.create_tables()
    return _db_instance


def reset_database_instance():
    """Reset the database instance (for testing or reconnection)."""
    global _db_instance
    if _db_instance:
        _db_instance.close()
    _db_instance = None
