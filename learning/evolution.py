"""
STRATEGY EVOLUTION - Evolve and improve strategies over time

Uses genetic algorithms and reinforcement concepts to:
- Mutate strategy parameters
- Crossover successful strategies
- Retire underperforming strategies
- Discover new strategy variants
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
import uuid
from loguru import logger


class MutationType(str, Enum):
    """Types of strategy mutations."""
    PARAMETER_TWEAK = "parameter_tweak"      # Small parameter changes
    PARAMETER_RESET = "parameter_reset"      # Reset to new random values
    INDICATOR_SWAP = "indicator_swap"        # Change indicators used
    TIMEFRAME_CHANGE = "timeframe_change"    # Change trading timeframe
    ENTRY_MODIFY = "entry_modify"            # Modify entry conditions
    EXIT_MODIFY = "exit_modify"              # Modify exit conditions


@dataclass
class StrategyGene:
    """Genetic representation of a strategy."""
    gene_id: str
    strategy_type: str
    parameters: Dict[str, float]
    indicators: List[str]
    entry_rules: List[str]
    exit_rules: List[str]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations_applied: List[MutationType] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_id": self.gene_id,
            "strategy_type": self.strategy_type,
            "parameters": self.parameters,
            "indicators": self.indicators,
            "entry_rules": self.entry_rules,
            "exit_rules": self.exit_rules,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }


@dataclass
class EvolutionConfig:
    """Configuration for strategy evolution."""
    population_size: int = 20
    elite_size: int = 4              # Top performers kept unchanged
    mutation_rate: float = 0.3       # Probability of mutation
    crossover_rate: float = 0.7      # Probability of crossover
    tournament_size: int = 3         # Selection tournament size
    max_generations: int = 50
    fitness_threshold: float = 1.0   # Sharpe ratio threshold


class StrategyEvolver:
    """
    Evolves trading strategies using genetic algorithms.

    Key concepts:
    - Fitness: Strategy performance (Sharpe ratio)
    - Selection: Tournament selection of fit strategies
    - Crossover: Combine parameters from two strategies
    - Mutation: Random modifications to parameters
    """

    # Available indicators for evolution
    AVAILABLE_INDICATORS = [
        "RSI", "MACD", "SMA", "EMA", "Bollinger", "ATR",
        "Stochastic", "ADX", "VWAP", "OBV", "MFI", "CCI"
    ]

    # Entry rule templates
    ENTRY_TEMPLATES = [
        "price_above_sma_{period}",
        "rsi_below_{threshold}",
        "rsi_above_{threshold}",
        "macd_crossover",
        "bollinger_breakout",
        "volume_spike",
        "momentum_positive",
        "trend_confirmed",
    ]

    # Exit rule templates
    EXIT_TEMPLATES = [
        "price_below_sma_{period}",
        "rsi_above_{threshold}",
        "rsi_below_{threshold}",
        "stop_loss_{pct}",
        "take_profit_{pct}",
        "trailing_stop_{pct}",
        "time_exit_{bars}",
        "macd_crossunder",
    ]

    _instance: Optional['StrategyEvolver'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[EvolutionConfig] = None):
        if self._initialized:
            return

        self.config = config or EvolutionConfig()
        self._population: List[StrategyGene] = []
        self._generation = 0
        self._best_genes: List[StrategyGene] = []
        self._evolution_history: List[Dict[str, Any]] = []

        self._initialized = True
        logger.info("StrategyEvolver initialized")

    def initialize_population(
        self,
        strategy_type: str,
        base_params: Optional[Dict[str, float]] = None
    ) -> List[StrategyGene]:
        """Initialize a random population of strategies."""
        self._population = []
        self._generation = 0

        for i in range(self.config.population_size):
            gene = self._create_random_gene(strategy_type, base_params)
            self._population.append(gene)

        logger.info(f"Initialized population of {len(self._population)} strategies")
        return self._population

    def _create_random_gene(
        self,
        strategy_type: str,
        base_params: Optional[Dict[str, float]] = None
    ) -> StrategyGene:
        """Create a random strategy gene."""
        # Random parameters
        params = base_params.copy() if base_params else {}

        # Add some randomization
        if "lookback_period" not in params:
            params["lookback_period"] = np.random.choice([5, 10, 20, 30, 50])
        else:
            params["lookback_period"] = params["lookback_period"] * np.random.uniform(0.7, 1.3)

        if "threshold" not in params:
            params["threshold"] = np.random.uniform(0.01, 0.1)

        if "stop_loss" not in params:
            params["stop_loss"] = np.random.uniform(0.02, 0.1)

        if "take_profit" not in params:
            params["take_profit"] = np.random.uniform(0.03, 0.15)

        # Random indicators (2-4)
        n_indicators = np.random.randint(2, 5)
        indicators = list(np.random.choice(self.AVAILABLE_INDICATORS, n_indicators, replace=False))

        # Random entry rules (1-3)
        n_entry = np.random.randint(1, 4)
        entry_rules = list(np.random.choice(self.ENTRY_TEMPLATES, n_entry, replace=False))

        # Random exit rules (1-3)
        n_exit = np.random.randint(1, 4)
        exit_rules = list(np.random.choice(self.EXIT_TEMPLATES, n_exit, replace=False))

        return StrategyGene(
            gene_id=f"gene_{uuid.uuid4().hex[:8]}",
            strategy_type=strategy_type,
            parameters=params,
            indicators=indicators,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            generation=self._generation,
        )

    def evolve(
        self,
        fitness_fn: Callable[[StrategyGene], float],
        n_generations: Optional[int] = None
    ) -> StrategyGene:
        """
        Run evolution for specified generations.

        Args:
            fitness_fn: Function that evaluates a gene and returns fitness score
            n_generations: Number of generations (default from config)

        Returns:
            Best performing gene
        """
        n_generations = n_generations or self.config.max_generations

        for gen in range(n_generations):
            self._generation = gen

            # Evaluate fitness
            for gene in self._population:
                try:
                    gene.fitness = fitness_fn(gene)
                except Exception as e:
                    logger.debug(f"Fitness evaluation failed: {e}")
                    gene.fitness = 0.0

            # Sort by fitness
            self._population.sort(key=lambda g: g.fitness, reverse=True)

            # Record history
            best_fitness = self._population[0].fitness
            avg_fitness = np.mean([g.fitness for g in self._population])

            self._evolution_history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_gene_id": self._population[0].gene_id,
            })

            logger.debug(f"Gen {gen}: best={best_fitness:.4f}, avg={avg_fitness:.4f}")

            # Check termination
            if best_fitness >= self.config.fitness_threshold:
                logger.info(f"Fitness threshold reached at generation {gen}")
                break

            # Create next generation
            next_population = []

            # Elitism: keep top performers
            for elite in self._population[:self.config.elite_size]:
                next_population.append(elite)

            # Fill rest with offspring
            while len(next_population) < self.config.population_size:
                # Selection
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()

                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = self._clone(parent1)

                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    offspring = self._mutate(offspring)

                offspring.generation = gen + 1
                next_population.append(offspring)

            self._population = next_population

        # Store best genes
        self._best_genes = sorted(self._population, key=lambda g: g.fitness, reverse=True)[:5]

        return self._population[0]

    def _tournament_select(self) -> StrategyGene:
        """Select a gene using tournament selection."""
        tournament = np.random.choice(
            self._population,
            size=min(self.config.tournament_size, len(self._population)),
            replace=False
        )
        return max(tournament, key=lambda g: g.fitness)

    def _crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> StrategyGene:
        """Create offspring by combining two parents."""
        # Mix parameters
        child_params = {}
        for key in set(parent1.parameters.keys()) | set(parent2.parameters.keys()):
            if key in parent1.parameters and key in parent2.parameters:
                # Blend
                alpha = np.random.random()
                child_params[key] = alpha * parent1.parameters[key] + \
                                   (1 - alpha) * parent2.parameters[key]
            elif key in parent1.parameters:
                child_params[key] = parent1.parameters[key]
            else:
                child_params[key] = parent2.parameters[key]

        # Mix indicators (union with random sampling)
        all_indicators = list(set(parent1.indicators) | set(parent2.indicators))
        n_indicators = (len(parent1.indicators) + len(parent2.indicators)) // 2
        child_indicators = list(np.random.choice(
            all_indicators,
            size=min(n_indicators, len(all_indicators)),
            replace=False
        ))

        # Mix entry rules
        all_entry = list(set(parent1.entry_rules) | set(parent2.entry_rules))
        n_entry = (len(parent1.entry_rules) + len(parent2.entry_rules)) // 2
        child_entry = list(np.random.choice(
            all_entry,
            size=min(n_entry, len(all_entry)),
            replace=False
        ))

        # Mix exit rules
        all_exit = list(set(parent1.exit_rules) | set(parent2.exit_rules))
        n_exit = (len(parent1.exit_rules) + len(parent2.exit_rules)) // 2
        child_exit = list(np.random.choice(
            all_exit,
            size=min(n_exit, len(all_exit)),
            replace=False
        ))

        return StrategyGene(
            gene_id=f"gene_{uuid.uuid4().hex[:8]}",
            strategy_type=parent1.strategy_type,
            parameters=child_params,
            indicators=child_indicators,
            entry_rules=child_entry,
            exit_rules=child_exit,
            parent_ids=[parent1.gene_id, parent2.gene_id],
        )

    def _clone(self, gene: StrategyGene) -> StrategyGene:
        """Create a copy of a gene."""
        return StrategyGene(
            gene_id=f"gene_{uuid.uuid4().hex[:8]}",
            strategy_type=gene.strategy_type,
            parameters=gene.parameters.copy(),
            indicators=gene.indicators.copy(),
            entry_rules=gene.entry_rules.copy(),
            exit_rules=gene.exit_rules.copy(),
            parent_ids=[gene.gene_id],
        )

    def _mutate(self, gene: StrategyGene) -> StrategyGene:
        """Apply random mutation to a gene."""
        mutation_type = np.random.choice(list(MutationType))

        if mutation_type == MutationType.PARAMETER_TWEAK:
            # Tweak a random parameter by ±20%
            if gene.parameters:
                key = np.random.choice(list(gene.parameters.keys()))
                gene.parameters[key] *= np.random.uniform(0.8, 1.2)

        elif mutation_type == MutationType.PARAMETER_RESET:
            # Reset a random parameter
            if gene.parameters:
                key = np.random.choice(list(gene.parameters.keys()))
                if "period" in key.lower():
                    gene.parameters[key] = np.random.choice([5, 10, 20, 30, 50])
                else:
                    gene.parameters[key] = np.random.uniform(0.01, 0.2)

        elif mutation_type == MutationType.INDICATOR_SWAP:
            # Swap one indicator
            if gene.indicators:
                idx = np.random.randint(len(gene.indicators))
                available = [i for i in self.AVAILABLE_INDICATORS if i not in gene.indicators]
                if available:
                    gene.indicators[idx] = np.random.choice(available)

        elif mutation_type == MutationType.ENTRY_MODIFY:
            # Modify entry rules
            if np.random.random() < 0.5 and len(gene.entry_rules) > 1:
                gene.entry_rules.pop(np.random.randint(len(gene.entry_rules)))
            else:
                available = [r for r in self.ENTRY_TEMPLATES if r not in gene.entry_rules]
                if available:
                    gene.entry_rules.append(np.random.choice(available))

        elif mutation_type == MutationType.EXIT_MODIFY:
            # Modify exit rules
            if np.random.random() < 0.5 and len(gene.exit_rules) > 1:
                gene.exit_rules.pop(np.random.randint(len(gene.exit_rules)))
            else:
                available = [r for r in self.EXIT_TEMPLATES if r not in gene.exit_rules]
                if available:
                    gene.exit_rules.append(np.random.choice(available))

        gene.mutations_applied.append(mutation_type)
        return gene

    def get_best_genes(self, n: int = 5) -> List[StrategyGene]:
        """Get top N performing genes."""
        return self._best_genes[:n]

    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get evolution history."""
        return self._evolution_history.copy()

    def suggest_improvements(self, gene: StrategyGene) -> List[str]:
        """Suggest potential improvements for a gene."""
        suggestions = []

        # Based on parameter values
        if gene.parameters.get("stop_loss", 0) > 0.08:
            suggestions.append("Consider tighter stop loss for better risk management")

        if len(gene.indicators) > 4:
            suggestions.append("Too many indicators may cause overfitting - consider simplifying")

        if len(gene.entry_rules) > 3:
            suggestions.append("Complex entry conditions may reduce trade frequency")

        if gene.fitness < 0.5:
            suggestions.append("Low fitness - consider more aggressive mutations")

        return suggestions


# Singleton accessor
_evolver_instance: Optional[StrategyEvolver] = None

def get_evolver() -> StrategyEvolver:
    """Get the singleton StrategyEvolver instance."""
    global _evolver_instance
    if _evolver_instance is None:
        _evolver_instance = StrategyEvolver()
    return _evolver_instance
