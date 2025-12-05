"""
DECISION JOURNAL - The Conscience of the Living System

Every significant decision the system makes is logged here with:
- What was decided
- What context existed at the time
- What alternatives were considered
- Why this decision was made
- Later: How good was this decision?

This creates a complete audit trail and enables the system to
learn from its own decision-making patterns.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum, auto
import json
import uuid
from loguru import logger

from memory.models import get_database, Decision, LearningEvent
from core.events import get_event_bus, Event, EventType


class DecisionType(Enum):
    """Categories of decisions the system makes."""
    
    # Trading decisions
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    POSITION_SIZE = "position_size"
    ORDER_TYPE = "order_type"
    
    # Hypothesis decisions
    HYPOTHESIS_CREATE = "hypothesis_create"
    HYPOTHESIS_PROMOTE = "hypothesis_promote"
    HYPOTHESIS_RETIRE = "hypothesis_retire"
    HYPOTHESIS_EVOLVE = "hypothesis_evolve"
    
    # Allocation decisions
    CAPITAL_ALLOCATION = "capital_allocation"
    REBALANCE = "rebalance"
    
    # Risk decisions
    RISK_REDUCTION = "risk_reduction"
    CIRCUIT_BREAKER = "circuit_breaker"
    
    # Learning decisions
    PARAMETER_UPDATE = "parameter_update"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    REGIME_CLASSIFICATION = "regime_classification"
    
    # Meta decisions
    SYSTEM_MODE_CHANGE = "system_mode_change"


@dataclass
class Alternative:
    """An alternative that was considered but not chosen."""
    description: str
    expected_outcome: Dict[str, Any]
    reason_rejected: str
    score: Optional[float] = None


@dataclass
class DecisionRecord:
    """
    A complete record of a decision made by the system.
    """
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    decision_type: DecisionType = DecisionType.TRADE_ENTRY
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # The decision itself
    description: str = ""
    outcome: Dict[str, Any] = field(default_factory=dict)
    
    # Context when decision was made
    market_context: Dict[str, Any] = field(default_factory=dict)
    portfolio_context: Dict[str, Any] = field(default_factory=dict)
    system_context: Dict[str, Any] = field(default_factory=dict)
    
    # Reasoning
    reasoning: str = ""
    confidence: float = 0.5  # 0.0 to 1.0
    
    # Alternatives
    alternatives: List[Alternative] = field(default_factory=list)
    
    # Outcome evaluation (filled in later)
    outcome_evaluated: bool = False
    outcome_quality: Optional[float] = None  # -1.0 to 1.0
    outcome_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["decision_type"] = self.decision_type.value
        data["timestamp"] = self.timestamp.isoformat()
        data["alternatives"] = [asdict(a) for a in self.alternatives]
        return data


class DecisionJournal:
    """
    The system's decision journal.
    
    Records all decisions, enables retrospective analysis,
    and supports learning from past decisions.
    """
    
    _instance: Optional['DecisionJournal'] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._db = get_database()
        self._pending_decisions: List[DecisionRecord] = []
        self._initialized = True
        
        logger.info("DecisionJournal initialized")
    
    def record(self, decision: DecisionRecord) -> str:
        """
        Record a decision.
        
        Returns the decision ID for later reference.
        """
        # Store in database
        session = self._db.get_session()
        try:
            db_decision = Decision(
                decision_id=decision.decision_id,
                timestamp=decision.timestamp,
                decision_type=decision.decision_type.value,
                description=decision.description,
                decision_outcome=decision.outcome,
                context={
                    "market": decision.market_context,
                    "portfolio": decision.portfolio_context,
                    "system": decision.system_context
                },
                alternatives_considered=[asdict(a) for a in decision.alternatives],
                reasoning=decision.reasoning,
                confidence=decision.confidence
            )
            session.add(db_decision)
            session.commit()
            
            logger.info(
                f"Decision recorded: {decision.decision_type.value} - {decision.description[:50]}"
            )
            
            # Emit event
            get_event_bus().publish(Event(
                event_type=EventType.DECISION_LOGGED,
                source="decision_journal",
                payload=decision.to_dict()
            ))
            
            return decision.decision_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record decision: {e}")
            raise
        finally:
            session.close()
    
    def evaluate_outcome(
        self,
        decision_id: str,
        quality: float,
        notes: str = ""
    ) -> None:
        """
        Evaluate the outcome of a past decision.
        
        Args:
            decision_id: ID of the decision to evaluate
            quality: -1.0 (terrible) to 1.0 (excellent)
            notes: Additional notes about the outcome
        """
        session = self._db.get_session()
        try:
            decision = session.query(Decision).filter(
                Decision.decision_id == decision_id
            ).first()
            
            if decision:
                decision.outcome_evaluated = True
                decision.outcome_quality = quality
                decision.outcome_notes = notes
                session.commit()
                
                logger.info(f"Decision {decision_id} evaluated: quality={quality}")
            else:
                logger.warning(f"Decision {decision_id} not found for evaluation")
                
        finally:
            session.close()
    
    def get_decisions(
        self,
        decision_type: Optional[DecisionType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        evaluated_only: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve decisions matching criteria.
        """
        session = self._db.get_session()
        try:
            query = session.query(Decision)
            
            if decision_type:
                query = query.filter(Decision.decision_type == decision_type.value)
            if start_date:
                query = query.filter(Decision.timestamp >= start_date)
            if end_date:
                query = query.filter(Decision.timestamp <= end_date)
            if evaluated_only:
                query = query.filter(Decision.outcome_evaluated == True)
            
            query = query.order_by(Decision.timestamp.desc()).limit(limit)
            
            return [
                {
                    "decision_id": d.decision_id,
                    "decision_type": d.decision_type,
                    "timestamp": d.timestamp.isoformat(),
                    "description": d.description,
                    "reasoning": d.reasoning,
                    "confidence": d.confidence,
                    "outcome_quality": d.outcome_quality
                }
                for d in query.all()
            ]
            
        finally:
            session.close()
    
    def analyze_decision_quality(
        self,
        decision_type: Optional[DecisionType] = None,
        min_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze the quality of past decisions.
        
        Returns statistics about decision-making performance.
        """
        decisions = self.get_decisions(
            decision_type=decision_type,
            evaluated_only=True,
            limit=1000
        )
        
        if len(decisions) < min_samples:
            return {
                "status": "insufficient_data",
                "samples": len(decisions),
                "required": min_samples
            }
        
        qualities = [d["outcome_quality"] for d in decisions if d["outcome_quality"] is not None]
        confidences = [d["confidence"] for d in decisions]
        
        # Calculate correlation between confidence and actual quality
        import numpy as np
        if len(qualities) > 10:
            correlation = np.corrcoef(confidences[:len(qualities)], qualities)[0, 1]
        else:
            correlation = None
        
        return {
            "status": "analyzed",
            "total_decisions": len(decisions),
            "evaluated": len(qualities),
            "avg_quality": np.mean(qualities) if qualities else None,
            "std_quality": np.std(qualities) if qualities else None,
            "avg_confidence": np.mean(confidences) if confidences else None,
            "confidence_quality_correlation": correlation,
            "good_decisions_pct": sum(1 for q in qualities if q > 0) / len(qualities) if qualities else None
        }
    
    def get_lessons_learned(self, decision_type: Optional[DecisionType] = None) -> List[Dict[str, Any]]:
        """
        Extract lessons from past decisions.
        
        Identifies patterns in good vs bad decisions.
        """
        decisions = self.get_decisions(
            decision_type=decision_type,
            evaluated_only=True,
            limit=500
        )
        
        good_decisions = [d for d in decisions if d.get("outcome_quality", 0) > 0.3]
        bad_decisions = [d for d in decisions if d.get("outcome_quality", 0) < -0.3]
        
        lessons = []
        
        # Analyze confidence calibration
        if good_decisions and bad_decisions:
            avg_good_confidence = sum(d["confidence"] for d in good_decisions) / len(good_decisions)
            avg_bad_confidence = sum(d["confidence"] for d in bad_decisions) / len(bad_decisions)
            
            if avg_bad_confidence > avg_good_confidence:
                lessons.append({
                    "lesson": "Overconfidence in bad decisions",
                    "detail": f"Bad decisions had avg confidence {avg_bad_confidence:.2f} vs good decisions {avg_good_confidence:.2f}",
                    "recommendation": "Reduce position sizes when confidence is high"
                })
        
        return lessons


def get_journal() -> DecisionJournal:
    """Get the singleton DecisionJournal instance."""
    return DecisionJournal()


# === Convenience functions for common decision types ===

def log_trade_decision(
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    hypothesis_id: str,
    reasoning: str,
    confidence: float,
    market_context: Dict[str, Any],
    alternatives: List[Dict[str, Any]] = None
) -> str:
    """Log a trade entry or exit decision."""
    decision = DecisionRecord(
        decision_type=DecisionType.TRADE_ENTRY if action == "buy" else DecisionType.TRADE_EXIT,
        description=f"{action.upper()} {quantity} {symbol} @ {price}",
        outcome={
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "hypothesis_id": hypothesis_id
        },
        reasoning=reasoning,
        confidence=confidence,
        market_context=market_context,
        alternatives=[Alternative(**a) for a in (alternatives or [])]
    )
    return get_journal().record(decision)


def log_hypothesis_decision(
    hypothesis_id: str,
    action: str,  # create, promote, retire, evolve
    reasoning: str,
    evidence: Dict[str, Any],
    confidence: float
) -> str:
    """Log a hypothesis lifecycle decision."""
    type_map = {
        "create": DecisionType.HYPOTHESIS_CREATE,
        "promote": DecisionType.HYPOTHESIS_PROMOTE,
        "retire": DecisionType.HYPOTHESIS_RETIRE,
        "evolve": DecisionType.HYPOTHESIS_EVOLVE
    }
    
    decision = DecisionRecord(
        decision_type=type_map.get(action, DecisionType.HYPOTHESIS_CREATE),
        description=f"{action.upper()} hypothesis {hypothesis_id}",
        outcome={
            "hypothesis_id": hypothesis_id,
            "action": action,
            "evidence": evidence
        },
        reasoning=reasoning,
        confidence=confidence,
        system_context={"evidence": evidence}
    )
    return get_journal().record(decision)


def log_risk_decision(
    risk_type: str,
    action_taken: str,
    trigger_value: float,
    threshold: float,
    reasoning: str
) -> str:
    """Log a risk management decision."""
    decision = DecisionRecord(
        decision_type=DecisionType.RISK_REDUCTION,
        description=f"Risk action: {action_taken} due to {risk_type}",
        outcome={
            "risk_type": risk_type,
            "action": action_taken,
            "trigger_value": trigger_value,
            "threshold": threshold
        },
        reasoning=reasoning,
        confidence=1.0  # Risk decisions are high confidence by design
    )
    return get_journal().record(decision)
