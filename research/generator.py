"""
RESEARCH HYPOTHESIS GENERATOR - Create tradeable strategies from research

Converts academic ideas into testable hypotheses:
- Maps research concepts to strategy templates
- Generates parameter ranges from paper claims
- Creates backtest-ready hypothesis objects
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
from loguru import logger

from research.parser import ExtractedIdea, IdeaType


@dataclass
class ResearchHypothesis:
    """A hypothesis generated from research."""
    hypothesis_id: str
    source_idea_id: str
    source_paper_id: str

    # Strategy definition
    name: str
    description: str
    strategy_type: str

    # Trading rules
    entry_logic: str
    exit_logic: str
    position_sizing: str

    # Parameters
    parameters: Dict[str, Any]
    parameter_ranges: Dict[str, tuple]  # For optimization

    # Expected performance (from paper)
    expected_sharpe: Optional[float] = None
    expected_annual_return: Optional[float] = None

    # Status
    backtest_ready: bool = False
    validation_status: str = "pending"

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "source_idea_id": self.source_idea_id,
            "source_paper_id": self.source_paper_id,
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type,
            "entry_logic": self.entry_logic,
            "exit_logic": self.exit_logic,
            "parameters": self.parameters,
            "expected_sharpe": self.expected_sharpe,
            "confidence_score": self.confidence_score,
        }


class ResearchHypothesisGenerator:
    """
    Generates trading hypotheses from research ideas.

    Mapping process:
    1. Identify strategy type from idea
    2. Map indicators to implementation
    3. Generate entry/exit logic
    4. Set parameter ranges for optimization
    """

    # Strategy templates
    STRATEGY_TEMPLATES = {
        "momentum": {
            "entry_logic": "Buy when momentum_score > {threshold} and trend_confirmed",
            "exit_logic": "Sell when momentum_score < 0 or stop_loss hit",
            "position_sizing": "kelly_fraction * confidence",
            "default_params": {
                "lookback_period": 20,
                "threshold": 0.5,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.15,
            },
            "param_ranges": {
                "lookback_period": (5, 60),
                "threshold": (0.3, 0.8),
                "stop_loss_pct": (0.02, 0.10),
                "take_profit_pct": (0.05, 0.30),
            },
        },
        "mean_reversion": {
            "entry_logic": "Buy when z_score < -{entry_z} and not in downtrend",
            "exit_logic": "Sell when z_score > {exit_z} or stop_loss hit",
            "position_sizing": "inverse_volatility * base_size",
            "default_params": {
                "lookback_period": 20,
                "entry_z": 2.0,
                "exit_z": 0.5,
                "stop_loss_pct": 0.05,
            },
            "param_ranges": {
                "lookback_period": (10, 50),
                "entry_z": (1.5, 3.0),
                "exit_z": (0.0, 1.0),
                "stop_loss_pct": (0.03, 0.10),
            },
        },
        "breakout": {
            "entry_logic": "Buy when price breaks above {lookback}-day high with volume confirmation",
            "exit_logic": "Sell when price breaks below trailing stop or {holding_period} days",
            "position_sizing": "atr_based_sizing",
            "default_params": {
                "lookback_period": 20,
                "volume_threshold": 1.5,
                "holding_period": 10,
                "atr_multiplier": 2.0,
            },
            "param_ranges": {
                "lookback_period": (10, 50),
                "volume_threshold": (1.2, 3.0),
                "holding_period": (5, 20),
                "atr_multiplier": (1.5, 4.0),
            },
        },
        "factor": {
            "entry_logic": "Long top {percentile}% by factor score, rebalance {frequency}",
            "exit_logic": "Exit when no longer in top quintile or rebalance date",
            "position_sizing": "equal_weight or factor_weighted",
            "default_params": {
                "percentile": 20,
                "rebalance_frequency": "monthly",
                "min_holding_period": 20,
            },
            "param_ranges": {
                "percentile": (10, 30),
                "min_holding_period": (10, 60),
            },
        },
        "trend_following": {
            "entry_logic": "Buy when fast_ma > slow_ma and ADX > {adx_threshold}",
            "exit_logic": "Sell when fast_ma < slow_ma or trailing stop hit",
            "position_sizing": "volatility_adjusted",
            "default_params": {
                "fast_period": 10,
                "slow_period": 30,
                "adx_threshold": 25,
                "trailing_stop_atr": 3.0,
            },
            "param_ranges": {
                "fast_period": (5, 20),
                "slow_period": (20, 100),
                "adx_threshold": (20, 40),
                "trailing_stop_atr": (2.0, 5.0),
            },
        },
        "statistical_arbitrage": {
            "entry_logic": "Trade spread when z_score exceeds {threshold}",
            "exit_logic": "Exit when spread reverts to mean or stop_loss",
            "position_sizing": "dollar_neutral",
            "default_params": {
                "lookback_period": 60,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "max_holding_days": 20,
            },
            "param_ranges": {
                "lookback_period": (30, 120),
                "entry_threshold": (1.5, 3.0),
                "exit_threshold": (0.0, 1.0),
                "max_holding_days": (10, 40),
            },
        },
        "machine_learning": {
            "entry_logic": "Trade when ML model prediction confidence > {threshold}",
            "exit_logic": "Exit based on model signal or time-based exit",
            "position_sizing": "prediction_confidence_weighted",
            "default_params": {
                "prediction_threshold": 0.6,
                "retrain_frequency": 30,
                "max_holding_period": 5,
            },
            "param_ranges": {
                "prediction_threshold": (0.5, 0.8),
                "retrain_frequency": (20, 60),
                "max_holding_period": (1, 10),
            },
        },
    }

    # Idea type to strategy type mapping
    IDEA_TO_STRATEGY = {
        IdeaType.MOMENTUM: "momentum",
        IdeaType.MEAN_REVERSION: "mean_reversion",
        IdeaType.TREND_FOLLOWING: "trend_following",
        IdeaType.BREAKOUT: "breakout",
        IdeaType.FACTOR: "factor",
        IdeaType.SENTIMENT: "momentum",  # Sentiment often used for momentum
        IdeaType.STATISTICAL_ARBITRAGE: "statistical_arbitrage",
        IdeaType.MACHINE_LEARNING: "machine_learning",
        IdeaType.VOLATILITY: "mean_reversion",  # Vol strategies often mean-reverting
        IdeaType.OTHER: "momentum",  # Default
    }

    _instance: Optional['ResearchHypothesisGenerator'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._generated_hypotheses: Dict[str, ResearchHypothesis] = {}
        self._generation_count = 0

        self._initialized = True
        logger.info("ResearchHypothesisGenerator initialized")

    def generate_hypothesis(self, idea: ExtractedIdea) -> ResearchHypothesis:
        """
        Generate a tradeable hypothesis from a research idea.
        """
        self._generation_count += 1

        # Map idea type to strategy type
        strategy_type = self.IDEA_TO_STRATEGY.get(idea.idea_type, "momentum")

        # Get template
        template = self.STRATEGY_TEMPLATES.get(strategy_type, self.STRATEGY_TEMPLATES["momentum"])

        # Merge parameters from idea with defaults
        params = template["default_params"].copy()
        for key, value in idea.parameters.items():
            if key in params:
                params[key] = value

        # Build entry logic
        entry_logic = template["entry_logic"]
        if idea.entry_conditions:
            entry_logic += f"\nAdditional conditions: {'; '.join(idea.entry_conditions[:2])}"

        # Build exit logic
        exit_logic = template["exit_logic"]
        if idea.exit_conditions:
            exit_logic += f"\nAdditional conditions: {'; '.join(idea.exit_conditions[:2])}"

        # Generate name
        name = f"Research_{idea.idea_type.value}_{self._generation_count}"

        # Create hypothesis
        hypothesis = ResearchHypothesis(
            hypothesis_id=f"research_hyp_{uuid.uuid4().hex[:8]}",
            source_idea_id=idea.idea_id,
            source_paper_id=idea.source_paper_id,
            name=name,
            description=idea.description,
            strategy_type=strategy_type,
            entry_logic=entry_logic,
            exit_logic=exit_logic,
            position_sizing=template["position_sizing"],
            parameters=params,
            parameter_ranges=template["param_ranges"],
            expected_sharpe=idea.claimed_sharpe,
            expected_annual_return=idea.claimed_returns,
            backtest_ready=True,
            confidence_score=self._calculate_confidence(idea),
        )

        self._generated_hypotheses[hypothesis.hypothesis_id] = hypothesis

        logger.info(f"Generated hypothesis: {hypothesis.name} from {idea.idea_type.value}")

        return hypothesis

    def generate_from_ideas(self, ideas: List[ExtractedIdea]) -> List[ResearchHypothesis]:
        """Generate hypotheses from multiple ideas."""
        hypotheses = []

        for idea in ideas:
            try:
                hypothesis = self.generate_hypothesis(idea)
                hypotheses.append(hypothesis)
            except Exception as e:
                logger.warning(f"Failed to generate hypothesis from idea {idea.idea_id}: {e}")

        return hypotheses

    def _calculate_confidence(self, idea: ExtractedIdea) -> float:
        """Calculate confidence score for generated hypothesis."""
        score = 0.3  # Base score

        # Higher confidence from idea
        if idea.confidence.value == "high":
            score += 0.3
        elif idea.confidence.value == "medium":
            score += 0.15

        # Has claimed performance metrics
        if idea.claimed_sharpe is not None:
            score += 0.1
            if idea.claimed_sharpe > 1.5:
                score += 0.1

        # Has specific conditions
        if idea.entry_conditions:
            score += 0.1
        if idea.exit_conditions:
            score += 0.1

        # Has indicators
        if len(idea.indicators) >= 2:
            score += 0.1

        return min(score, 1.0)

    def get_hypothesis(self, hypothesis_id: str) -> Optional[ResearchHypothesis]:
        """Get a specific hypothesis."""
        return self._generated_hypotheses.get(hypothesis_id)

    def get_all_hypotheses(self) -> List[ResearchHypothesis]:
        """Get all generated hypotheses."""
        return list(self._generated_hypotheses.values())

    def get_ready_for_backtest(self) -> List[ResearchHypothesis]:
        """Get hypotheses ready for backtesting."""
        return [h for h in self._generated_hypotheses.values() if h.backtest_ready]

    def get_by_strategy_type(self, strategy_type: str) -> List[ResearchHypothesis]:
        """Get hypotheses of a specific strategy type."""
        return [h for h in self._generated_hypotheses.values()
                if h.strategy_type == strategy_type]

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        hypotheses = list(self._generated_hypotheses.values())
        return {
            "total_generated": len(hypotheses),
            "by_strategy_type": {
                st: len([h for h in hypotheses if h.strategy_type == st])
                for st in set(h.strategy_type for h in hypotheses)
            } if hypotheses else {},
            "avg_confidence": sum(h.confidence_score for h in hypotheses) / len(hypotheses)
                             if hypotheses else 0,
            "ready_for_backtest": len([h for h in hypotheses if h.backtest_ready]),
        }


# Singleton accessor
_generator_instance: Optional[ResearchHypothesisGenerator] = None

def get_research_generator() -> ResearchHypothesisGenerator:
    """Get the singleton ResearchHypothesisGenerator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ResearchHypothesisGenerator()
    return _generator_instance
