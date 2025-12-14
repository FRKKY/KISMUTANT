"""
STATE PERSISTENCE - Ensures app state survives restarts

This module handles:
1. Saving in-memory state to the database
2. Restoring state from database on startup
3. Periodic state snapshots for recovery

Critical state that must persist:
- Hypotheses (strategies) and their lifecycle state
- Open positions
- Capital allocations
- Risk state (daily loss counters, exposure)
"""

import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger

from memory.models import (
    get_database,
    Hypothesis as HypothesisDB,
    Position as PositionDB,
    SystemState,
    HypothesisStatus,
    OrderSide,
)


class StatePersistence:
    """
    Manages persistence of critical system state.

    Ensures the system can recover its state after restarts,
    deployments, or crashes.
    """

    def __init__(self):
        self._db = get_database()

    # ===== HYPOTHESIS PERSISTENCE =====

    def save_hypothesis(self, hypothesis) -> bool:
        """
        Save a single hypothesis to the database.

        Args:
            hypothesis: Hypothesis object from hypothesis.models

        Returns:
            True if saved successfully
        """
        try:
            session = self._db.get_session()

            # Check if hypothesis exists
            db_hyp = session.query(HypothesisDB).filter_by(
                hypothesis_id=hypothesis.hypothesis_id
            ).first()

            if db_hyp:
                # Update existing
                db_hyp.name = hypothesis.name
                db_hyp.description = hypothesis.description
                db_hyp.logic_type = hypothesis.strategy_type.value
                db_hyp.logic_definition = json.dumps({
                    "parameters": hypothesis.parameters,
                    "entry_rules": hypothesis.entry_rules,
                    "exit_rules": hypothesis.exit_rules,
                    "symbols": hypothesis.symbols,
                    "max_position_pct": hypothesis.max_position_pct,
                    "stop_loss_pct": hypothesis.stop_loss_pct,
                    "take_profit_pct": hypothesis.take_profit_pct,
                    "max_holding_days": hypothesis.max_holding_days,
                    "tags": hypothesis.tags,
                    "notes": hypothesis.notes,
                    "source_pattern_id": hypothesis.source_pattern_id,
                    "source_research": hypothesis.source_research,
                    "parent_id": hypothesis.parent_id,
                    "version": hypothesis.version,
                })
                db_hyp.status = self._map_state_to_status(hypothesis.state)
                db_hyp.capital_allocation_pct = hypothesis.capital_pct
                db_hyp.confidence_score = hypothesis.allocated_capital

                # Update metrics
                metrics = hypothesis.get_current_metrics()
                if metrics:
                    db_hyp.total_trades = metrics.total_trades
                    db_hyp.winning_trades = metrics.winning_trades
                    db_hyp.total_pnl = metrics.total_return
                    db_hyp.sharpe_ratio = metrics.sharpe_ratio
                    db_hyp.max_drawdown = metrics.max_drawdown
                    db_hyp.win_rate = metrics.win_rate
                    db_hyp.profit_factor = metrics.profit_factor

                # Update timestamps
                db_hyp.backtested_at = hypothesis.incubation_start
                db_hyp.paper_started_at = hypothesis.paper_start
                db_hyp.live_started_at = hypothesis.live_start
                db_hyp.retired_at = hypothesis.retired_at

            else:
                # Create new
                db_hyp = HypothesisDB(
                    hypothesis_id=hypothesis.hypothesis_id,
                    name=hypothesis.name,
                    description=hypothesis.description,
                    logic_type=hypothesis.strategy_type.value,
                    logic_definition={
                        "parameters": hypothesis.parameters,
                        "entry_rules": hypothesis.entry_rules,
                        "exit_rules": hypothesis.exit_rules,
                        "symbols": hypothesis.symbols,
                        "max_position_pct": hypothesis.max_position_pct,
                        "stop_loss_pct": hypothesis.stop_loss_pct,
                        "take_profit_pct": hypothesis.take_profit_pct,
                        "max_holding_days": hypothesis.max_holding_days,
                        "tags": hypothesis.tags,
                        "notes": hypothesis.notes,
                        "source_pattern_id": hypothesis.source_pattern_id,
                        "source_research": hypothesis.source_research,
                        "parent_id": hypothesis.parent_id,
                        "version": hypothesis.version,
                    },
                    status=self._map_state_to_status(hypothesis.state),
                    capital_allocation_pct=hypothesis.capital_pct,
                    created_at=hypothesis.created_at,
                    backtested_at=hypothesis.incubation_start,
                    paper_started_at=hypothesis.paper_start,
                    live_started_at=hypothesis.live_start,
                )
                session.add(db_hyp)

            session.commit()
            session.close()
            return True

        except Exception as e:
            logger.error(f"Failed to save hypothesis {hypothesis.hypothesis_id}: {e}")
            return False

    def save_all_hypotheses(self, registry) -> int:
        """
        Save all hypotheses from registry to database.

        Args:
            registry: StrategyRegistry instance

        Returns:
            Number of hypotheses saved
        """
        saved = 0
        for hypothesis in registry.get_all():
            if self.save_hypothesis(hypothesis):
                saved += 1

        logger.info(f"Saved {saved} hypotheses to database")
        return saved

    def load_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Load all active hypotheses from database.

        Returns:
            List of hypothesis dictionaries ready to be converted to Hypothesis objects
        """
        try:
            session = self._db.get_session()

            # Load non-retired hypotheses
            db_hypotheses = session.query(HypothesisDB).filter(
                HypothesisDB.status != HypothesisStatus.RETIRED,
                HypothesisDB.status != HypothesisStatus.FAILED
            ).all()

            hypotheses = []
            for db_hyp in db_hypotheses:
                logic = db_hyp.logic_definition or {}
                if isinstance(logic, str):
                    logic = json.loads(logic)

                hyp_dict = {
                    "hypothesis_id": db_hyp.hypothesis_id,
                    "name": db_hyp.name,
                    "description": db_hyp.description,
                    "strategy_type": db_hyp.logic_type or "momentum",
                    "state": self._map_status_to_state(db_hyp.status),
                    "created_at": db_hyp.created_at,
                    "symbols": logic.get("symbols", []),
                    "parameters": logic.get("parameters", {}),
                    "entry_rules": logic.get("entry_rules", {}),
                    "exit_rules": logic.get("exit_rules", {}),
                    "max_position_pct": logic.get("max_position_pct", 0.1),
                    "stop_loss_pct": logic.get("stop_loss_pct", 0.02),
                    "take_profit_pct": logic.get("take_profit_pct"),
                    "max_holding_days": logic.get("max_holding_days"),
                    "capital_pct": db_hyp.capital_allocation_pct or 0.0,
                    "tags": logic.get("tags", []),
                    "notes": logic.get("notes", ""),
                    "source_pattern_id": logic.get("source_pattern_id"),
                    "source_research": logic.get("source_research"),
                    "parent_id": logic.get("parent_id"),
                    "version": logic.get("version", 1),
                    "incubation_start": db_hyp.backtested_at,
                    "paper_start": db_hyp.paper_started_at,
                    "live_start": db_hyp.live_started_at,
                    # Metrics
                    "total_trades": db_hyp.total_trades,
                    "win_rate": db_hyp.win_rate,
                    "sharpe_ratio": db_hyp.sharpe_ratio,
                    "max_drawdown": db_hyp.max_drawdown,
                    "profit_factor": db_hyp.profit_factor,
                    "total_pnl": db_hyp.total_pnl,
                }
                hypotheses.append(hyp_dict)

            session.close()
            logger.info(f"Loaded {len(hypotheses)} hypotheses from database")
            return hypotheses

        except Exception as e:
            logger.error(f"Failed to load hypotheses: {e}")
            return []

    # ===== POSITION PERSISTENCE =====

    def save_position(self, position) -> bool:
        """
        Save a position to the database.

        Args:
            position: Position object from portfolio.positions

        Returns:
            True if saved successfully
        """
        try:
            session = self._db.get_session()

            # Check if position exists
            db_pos = session.query(PositionDB).filter_by(
                symbol=position.symbol,
                is_open=True
            ).first()

            if db_pos:
                # Update existing
                db_pos.quantity = position.quantity
                db_pos.avg_cost = position.entry_price
                db_pos.current_price = position.current_price
                db_pos.primary_hypothesis_id = position.hypothesis_id
                db_pos.unrealized_pnl = position.unrealized_pnl
                db_pos.unrealized_pnl_pct = position.unrealized_pnl_pct
                db_pos.last_updated = datetime.now()

                if position.status.value == "closed":
                    db_pos.is_open = False
                    db_pos.closed_at = position.exit_time
            else:
                # Create new
                db_pos = PositionDB(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_cost=position.entry_price,
                    current_price=position.current_price,
                    primary_hypothesis_id=position.hypothesis_id,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_pct=position.unrealized_pnl_pct,
                    opened_at=position.entry_time,
                    is_open=True,
                )
                session.add(db_pos)

            session.commit()
            session.close()
            return True

        except Exception as e:
            logger.error(f"Failed to save position {position.symbol}: {e}")
            return False

    def save_all_positions(self, position_manager) -> int:
        """
        Save all open positions to database.

        Args:
            position_manager: PositionManager instance

        Returns:
            Number of positions saved
        """
        saved = 0
        for position in position_manager.get_all_open():
            if self.save_position(position):
                saved += 1

        logger.info(f"Saved {saved} positions to database")
        return saved

    def load_positions(self) -> List[Dict[str, Any]]:
        """
        Load all open positions from database.

        Returns:
            List of position dictionaries
        """
        try:
            session = self._db.get_session()

            db_positions = session.query(PositionDB).filter_by(is_open=True).all()

            positions = []
            for db_pos in db_positions:
                pos_dict = {
                    "symbol": db_pos.symbol,
                    "quantity": db_pos.quantity,
                    "entry_price": db_pos.avg_cost,
                    "current_price": db_pos.current_price or db_pos.avg_cost,
                    "hypothesis_id": db_pos.primary_hypothesis_id,
                    "unrealized_pnl": db_pos.unrealized_pnl or 0.0,
                    "unrealized_pnl_pct": db_pos.unrealized_pnl_pct or 0.0,
                    "entry_time": db_pos.opened_at,
                }
                positions.append(pos_dict)

            session.close()
            logger.info(f"Loaded {len(positions)} open positions from database")
            return positions

        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
            return []

    # ===== SYSTEM STATE SNAPSHOTS =====

    def save_system_state(
        self,
        registry,
        position_manager,
        allocator,
        config: Optional[Dict] = None
    ) -> bool:
        """
        Save complete system state snapshot.

        This is called periodically and on shutdown for recovery.
        """
        try:
            session = self._db.get_session()

            # Gather state
            hypotheses_state = {}
            for hyp in registry.get_all():
                hypotheses_state[hyp.hypothesis_id] = {
                    "state": hyp.state.value,
                    "capital_pct": hyp.capital_pct,
                    "allocated_capital": hyp.allocated_capital,
                }

            positions_state = {}
            for pos in position_manager.get_all_open():
                positions_state[pos.symbol] = {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "hypothesis_id": pos.hypothesis_id,
                }

            # Get allocator state if available
            risk_state = {}
            if hasattr(allocator, 'get_state'):
                risk_state = allocator.get_state()
            else:
                risk_state = {
                    "total_capital": getattr(allocator, 'total_capital', 0),
                    "available_capital": getattr(allocator, 'available_capital', 0),
                }

            # Create snapshot
            snapshot = SystemState(
                timestamp=datetime.now(),
                active_hypotheses=hypotheses_state,
                portfolio_state=positions_state,
                risk_state=risk_state,
                config_snapshot=config or {},
                is_healthy=True,
            )

            session.add(snapshot)
            session.commit()
            session.close()

            logger.info("System state snapshot saved")
            return True

        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            return False

    def load_latest_system_state(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent system state snapshot.

        Returns:
            Dictionary with hypotheses, positions, and risk state
        """
        try:
            session = self._db.get_session()

            snapshot = session.query(SystemState).order_by(
                SystemState.timestamp.desc()
            ).first()

            if not snapshot:
                session.close()
                return None

            state = {
                "timestamp": snapshot.timestamp,
                "hypotheses": snapshot.active_hypotheses or {},
                "positions": snapshot.portfolio_state or {},
                "risk": snapshot.risk_state or {},
                "config": snapshot.config_snapshot or {},
                "is_healthy": snapshot.is_healthy,
            }

            session.close()
            logger.info(f"Loaded system state from {snapshot.timestamp}")
            return state

        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
            return None

    # ===== HELPER METHODS =====

    def _map_state_to_status(self, state) -> HypothesisStatus:
        """Map hypothesis.models.StrategyState to memory.models.HypothesisStatus."""
        mapping = {
            "incubating": HypothesisStatus.BACKTESTING,
            "paper_trading": HypothesisStatus.PAPER_TRADING,
            "live": HypothesisStatus.ACTIVE,
            "paused": HypothesisStatus.PAUSED,
            "retired": HypothesisStatus.RETIRED,
        }
        return mapping.get(state.value, HypothesisStatus.DRAFT)

    def _map_status_to_state(self, status: HypothesisStatus) -> str:
        """Map memory.models.HypothesisStatus to hypothesis.models.StrategyState string."""
        mapping = {
            HypothesisStatus.DRAFT: "incubating",
            HypothesisStatus.BACKTESTING: "incubating",
            HypothesisStatus.PAPER_TRADING: "paper_trading",
            HypothesisStatus.INCUBATING: "incubating",
            HypothesisStatus.ACTIVE: "live",
            HypothesisStatus.PAUSED: "paused",
            HypothesisStatus.RETIRED: "retired",
            HypothesisStatus.FAILED: "retired",
        }
        return mapping.get(status, "incubating")


# Singleton instance
_persistence: Optional[StatePersistence] = None


def get_state_persistence() -> StatePersistence:
    """Get the singleton StatePersistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = StatePersistence()
    return _persistence


# Convenience functions
def save_state(registry, position_manager, allocator, config=None) -> bool:
    """Save all critical state to database."""
    persistence = get_state_persistence()

    # Save individual components
    persistence.save_all_hypotheses(registry)
    persistence.save_all_positions(position_manager)

    # Save combined snapshot
    return persistence.save_system_state(registry, position_manager, allocator, config)


def restore_state() -> Dict[str, Any]:
    """
    Restore state from database.

    Returns:
        Dictionary with 'hypotheses', 'positions', and 'system_state' keys
    """
    persistence = get_state_persistence()

    return {
        "hypotheses": persistence.load_hypotheses(),
        "positions": persistence.load_positions(),
        "system_state": persistence.load_latest_system_state(),
    }
