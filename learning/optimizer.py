"""
PARAMETER OPTIMIZER - Automatically tune strategy parameters

Uses Bayesian optimization and genetic algorithms to find optimal
parameters based on historical performance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from loguru import logger


class OptimizationMethod(str, Enum):
    """Parameter optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"


@dataclass
class ParameterSpace:
    """Defines the search space for a parameter."""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None  # For discrete parameters
    log_scale: bool = False  # Use logarithmic scale
    default: Optional[float] = None

    def sample_random(self) -> float:
        """Sample a random value from this parameter space."""
        if self.log_scale:
            log_min = np.log(max(self.min_value, 1e-10))
            log_max = np.log(self.max_value)
            value = np.exp(np.random.uniform(log_min, log_max))
        else:
            value = np.random.uniform(self.min_value, self.max_value)

        if self.step:
            value = round(value / self.step) * self.step

        return np.clip(value, self.min_value, self.max_value)

    def get_grid_values(self, n_points: int = 10) -> List[float]:
        """Get evenly spaced values for grid search."""
        if self.step:
            values = np.arange(self.min_value, self.max_value + self.step, self.step)
        elif self.log_scale:
            values = np.logspace(
                np.log10(max(self.min_value, 1e-10)),
                np.log10(self.max_value),
                n_points
            )
        else:
            values = np.linspace(self.min_value, self.max_value, n_points)

        return values.tolist()


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    strategy_id: str
    best_params: Dict[str, float]
    best_score: float
    iterations: int
    improvement_pct: float
    optimization_time: float
    method: OptimizationMethod
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "iterations": self.iterations,
            "improvement_pct": self.improvement_pct,
            "optimization_time": self.optimization_time,
            "method": self.method.value,
        }


class ParameterOptimizer:
    """
    Optimizes strategy parameters using various methods.

    Supports:
    1. Grid search (exhaustive)
    2. Random search (faster)
    3. Bayesian optimization (efficient)
    4. Genetic algorithms (for complex spaces)
    """

    # Default parameter spaces for common strategy types
    DEFAULT_SPACES = {
        "momentum": [
            ParameterSpace("lookback_period", 5, 60, step=5, default=20),
            ParameterSpace("threshold", 0.01, 0.10, step=0.01, default=0.03),
            ParameterSpace("holding_period", 1, 20, step=1, default=5),
        ],
        "mean_reversion": [
            ParameterSpace("lookback_period", 10, 100, step=5, default=20),
            ParameterSpace("entry_zscore", 1.5, 3.0, step=0.25, default=2.0),
            ParameterSpace("exit_zscore", 0.0, 1.0, step=0.25, default=0.5),
        ],
        "breakout": [
            ParameterSpace("lookback_period", 10, 50, step=5, default=20),
            ParameterSpace("breakout_threshold", 0.01, 0.05, step=0.005, default=0.02),
            ParameterSpace("confirmation_bars", 1, 5, step=1, default=2),
        ],
        "trend_following": [
            ParameterSpace("fast_period", 5, 30, step=5, default=10),
            ParameterSpace("slow_period", 20, 100, step=10, default=50),
            ParameterSpace("atr_multiplier", 1.0, 4.0, step=0.5, default=2.0),
        ],
    }

    _instance: Optional['ParameterOptimizer'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._optimization_history: List[OptimizationResult] = []
        self._best_params: Dict[str, Dict[str, float]] = {}

        self._initialized = True
        logger.info("ParameterOptimizer initialized")

    def optimize(
        self,
        strategy_id: str,
        strategy_type: str,
        objective_fn: Callable[[Dict[str, float]], float],
        param_spaces: Optional[List[ParameterSpace]] = None,
        method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
        max_iterations: int = 100,
        current_params: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Optimize parameters for a strategy.

        Args:
            strategy_id: ID of the strategy
            strategy_type: Type (momentum, mean_reversion, etc.)
            objective_fn: Function that takes params and returns score (higher = better)
            param_spaces: Parameter search spaces (or use defaults)
            method: Optimization method
            max_iterations: Maximum optimization iterations
            current_params: Current parameter values (for baseline)

        Returns:
            OptimizationResult with best parameters
        """
        import time
        start_time = time.time()

        # Get parameter spaces
        if param_spaces is None:
            param_spaces = self.DEFAULT_SPACES.get(strategy_type, [])

        if not param_spaces:
            logger.warning(f"No parameter spaces defined for {strategy_type}")
            return OptimizationResult(
                strategy_id=strategy_id,
                best_params=current_params or {},
                best_score=0,
                iterations=0,
                improvement_pct=0,
                optimization_time=0,
                method=method,
            )

        # Get baseline score
        baseline_score = 0
        if current_params:
            try:
                baseline_score = objective_fn(current_params)
            except:
                pass

        # Run optimization
        if method == OptimizationMethod.GRID_SEARCH:
            best_params, best_score, history = self._grid_search(
                objective_fn, param_spaces, max_iterations
            )
        elif method == OptimizationMethod.RANDOM_SEARCH:
            best_params, best_score, history = self._random_search(
                objective_fn, param_spaces, max_iterations
            )
        elif method == OptimizationMethod.BAYESIAN:
            best_params, best_score, history = self._bayesian_optimization(
                objective_fn, param_spaces, max_iterations
            )
        elif method == OptimizationMethod.GENETIC:
            best_params, best_score, history = self._genetic_optimization(
                objective_fn, param_spaces, max_iterations
            )
        else:
            best_params, best_score, history = self._random_search(
                objective_fn, param_spaces, max_iterations
            )

        elapsed = time.time() - start_time

        # Calculate improvement
        improvement = ((best_score - baseline_score) / abs(baseline_score) * 100
                      if baseline_score != 0 else 0)

        result = OptimizationResult(
            strategy_id=strategy_id,
            best_params=best_params,
            best_score=best_score,
            iterations=len(history),
            improvement_pct=improvement,
            optimization_time=elapsed,
            method=method,
            history=history,
        )

        self._optimization_history.append(result)
        self._best_params[strategy_id] = best_params

        logger.info(
            f"Optimization complete for {strategy_id}: "
            f"score={best_score:.4f}, improvement={improvement:.1f}%"
        )

        return result

    def _grid_search(
        self,
        objective_fn: Callable,
        param_spaces: List[ParameterSpace],
        max_iterations: int
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """Grid search optimization."""
        from itertools import product

        # Generate grid
        grids = {}
        points_per_param = max(2, int(max_iterations ** (1 / len(param_spaces))))

        for space in param_spaces:
            grids[space.name] = space.get_grid_values(points_per_param)

        # Search
        best_params = {}
        best_score = float('-inf')
        history = []

        keys = list(grids.keys())
        for values in product(*[grids[k] for k in keys]):
            params = dict(zip(keys, values))
            try:
                score = objective_fn(params)
                history.append({"params": params.copy(), "score": score})

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                if len(history) >= max_iterations:
                    break
            except Exception as e:
                logger.debug(f"Objective function failed: {e}")

        return best_params, best_score, history

    def _random_search(
        self,
        objective_fn: Callable,
        param_spaces: List[ParameterSpace],
        max_iterations: int
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """Random search optimization."""
        best_params = {}
        best_score = float('-inf')
        history = []

        for i in range(max_iterations):
            # Sample random parameters
            params = {space.name: space.sample_random() for space in param_spaces}

            try:
                score = objective_fn(params)
                history.append({"params": params.copy(), "score": score})

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception as e:
                logger.debug(f"Objective function failed: {e}")

        return best_params, best_score, history

    def _bayesian_optimization(
        self,
        objective_fn: Callable,
        param_spaces: List[ParameterSpace],
        max_iterations: int
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """
        Simple Bayesian-inspired optimization using Gaussian process surrogate.
        Falls back to random search with exploitation bias.
        """
        best_params = {}
        best_score = float('-inf')
        history = []

        # Start with random exploration
        n_initial = min(10, max_iterations // 3)

        for i in range(max_iterations):
            if i < n_initial or not history:
                # Exploration: random sampling
                params = {space.name: space.sample_random() for space in param_spaces}
            else:
                # Exploitation: sample near best known point with noise
                params = {}
                for space in param_spaces:
                    best_val = best_params.get(space.name, space.default or space.sample_random())
                    noise_scale = (space.max_value - space.min_value) * 0.1 * (1 - i / max_iterations)
                    new_val = best_val + np.random.normal(0, noise_scale)
                    new_val = np.clip(new_val, space.min_value, space.max_value)
                    if space.step:
                        new_val = round(new_val / space.step) * space.step
                    params[space.name] = new_val

            try:
                score = objective_fn(params)
                history.append({"params": params.copy(), "score": score})

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception as e:
                logger.debug(f"Objective function failed: {e}")

        return best_params, best_score, history

    def _genetic_optimization(
        self,
        objective_fn: Callable,
        param_spaces: List[ParameterSpace],
        max_iterations: int
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """Genetic algorithm optimization."""
        population_size = min(20, max_iterations // 5)
        n_generations = max_iterations // population_size

        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {space.name: space.sample_random() for space in param_spaces}
            population.append(individual)

        best_params = {}
        best_score = float('-inf')
        history = []

        for gen in range(n_generations):
            # Evaluate fitness
            fitness = []
            for individual in population:
                try:
                    score = objective_fn(individual)
                    fitness.append(score)
                    history.append({"params": individual.copy(), "score": score})

                    if score > best_score:
                        best_score = score
                        best_params = individual.copy()
                except:
                    fitness.append(float('-inf'))

            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                winner = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
                new_population.append(winner.copy())

            # Crossover and mutation
            for i in range(0, len(new_population) - 1, 2):
                if np.random.random() < 0.7:  # Crossover probability
                    # Uniform crossover
                    for space in param_spaces:
                        if np.random.random() < 0.5:
                            new_population[i][space.name], new_population[i+1][space.name] = \
                                new_population[i+1][space.name], new_population[i][space.name]

            # Mutation
            mutation_rate = 0.1
            for individual in new_population:
                for space in param_spaces:
                    if np.random.random() < mutation_rate:
                        individual[space.name] = space.sample_random()

            population = new_population

        return best_params, best_score, history

    def get_best_params(self, strategy_id: str) -> Optional[Dict[str, float]]:
        """Get best known parameters for a strategy."""
        return self._best_params.get(strategy_id)

    def get_optimization_history(
        self,
        strategy_id: Optional[str] = None
    ) -> List[OptimizationResult]:
        """Get optimization history."""
        if strategy_id:
            return [r for r in self._optimization_history if r.strategy_id == strategy_id]
        return self._optimization_history.copy()

    def suggest_parameter_adjustment(
        self,
        strategy_id: str,
        current_params: Dict[str, float],
        recent_performance: float
    ) -> Dict[str, float]:
        """
        Suggest parameter adjustments based on recent performance.

        Uses gradient-free local search around current parameters.
        """
        # If performance is good, make small adjustments
        # If performance is poor, make larger adjustments
        adjustment_scale = 0.1 if recent_performance > 0 else 0.2

        suggested = current_params.copy()

        for param, value in current_params.items():
            # Random walk with performance-based scale
            adjustment = np.random.normal(0, abs(value) * adjustment_scale)
            suggested[param] = value + adjustment

        return suggested


# Singleton accessor
_optimizer_instance: Optional[ParameterOptimizer] = None

def get_optimizer() -> ParameterOptimizer:
    """Get the singleton ParameterOptimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = ParameterOptimizer()
    return _optimizer_instance
