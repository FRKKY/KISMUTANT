"""
METRICS - Performance Metric Calculations

Calculates all performance metrics for strategy evaluation:
- Returns (total, annualized, risk-adjusted)
- Risk metrics (volatility, drawdown, VaR)
- Trade statistics (win rate, profit factor)
- Statistical significance tests

These metrics are used to:
1. Evaluate backtest results
2. Monitor paper trading performance
3. Make promotion/demotion decisions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from scipy import stats

from loguru import logger


@dataclass
class TradeRecord:
    """Record of a single trade."""
    
    symbol: str
    side: str  # "long" or "short"
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    holding_period_hours: float
    exit_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat(),
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "holding_period_hours": self.holding_period_hours,
            "exit_reason": self.exit_reason
        }


class MetricsCalculator:
    """
    Calculates comprehensive performance metrics.
    
    Can calculate metrics from:
    - A series of returns
    - A list of trade records
    - An equity curve
    """
    
    def __init__(self, risk_free_rate: float = 0.035):
        """
        Initialize calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 3.5% for Korea)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252
    
    # ===== Return Metrics =====
    
    def total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total return."""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    def annualized_return(
        self,
        equity_curve: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate annualized return."""
        total_ret = self.total_return(equity_curve)
        n_periods = len(equity_curve)
        
        if n_periods <= 1 or total_ret <= -1:
            return 0.0
        
        years = n_periods / periods_per_year
        return (1 + total_ret) ** (1 / years) - 1
    
    def cagr(self, equity_curve: pd.Series) -> float:
        """Compound Annual Growth Rate."""
        return self.annualized_return(equity_curve)
    
    # ===== Risk Metrics =====
    
    def volatility(
        self,
        returns: pd.Series,
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(returns) < 2:
            return 0.0
        
        vol = returns.std()
        
        if annualize:
            vol *= np.sqrt(periods_per_year)
        
        return vol
    
    def downside_volatility(
        self,
        returns: pd.Series,
        threshold: float = 0.0,
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """Calculate downside volatility (semi-deviation)."""
        if len(returns) < 2:
            return 0.0
        
        downside_returns = returns[returns < threshold]
        
        if len(downside_returns) < 2:
            return 0.0
        
        vol = downside_returns.std()
        
        if annualize:
            vol *= np.sqrt(periods_per_year)
        
        return vol
    
    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        
        return abs(drawdowns.min())
    
    def drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        rolling_max = equity_curve.expanding().max()
        return (equity_curve - rolling_max) / rolling_max
    
    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(equity_curve) < 2:
            return 0
        
        rolling_max = equity_curve.expanding().max()
        underwater = equity_curve < rolling_max
        
        # Find consecutive underwater periods
        max_duration = 0
        current_duration = 0
        
        for is_underwater in underwater:
            if is_underwater:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 10:
            return 0.0
        
        return abs(np.percentile(returns, (1 - confidence) * 100))
    
    def cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(returns) < 10:
            return 0.0
        
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return abs(var_threshold)
        
        return abs(tail_returns.mean())
    
    # ===== Risk-Adjusted Returns =====
    
    def sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 10:
            return 0.0
        
        excess_returns = returns - self.daily_rf
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std()
        
        # Annualize
        return sharpe * np.sqrt(periods_per_year)
    
    def sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sortino ratio (uses downside volatility)."""
        if len(returns) < 10:
            return 0.0
        
        excess_returns = returns - self.daily_rf
        downside_vol = self.downside_volatility(
            returns, 
            threshold=self.daily_rf,
            annualize=False
        )
        
        if downside_vol == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_vol
        
        return sortino * np.sqrt(periods_per_year)
    
    def calmar_ratio(self, equity_curve: pd.Series) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        cagr = self.cagr(equity_curve)
        max_dd = self.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0.0
        
        return cagr / max_dd
    
    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Information Ratio."""
        if len(returns) < 10 or len(benchmark_returns) < 10:
            return 0.0
        
        # Align series
        aligned = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 10:
            return 0.0
        
        active_returns = aligned['strategy'] - aligned['benchmark']
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        ir = active_returns.mean() / tracking_error
        
        return ir * np.sqrt(periods_per_year)
    
    # ===== Trade Statistics =====
    
    def calculate_trade_stats(
        self,
        trades: List[TradeRecord]
    ) -> Dict[str, Any]:
        """Calculate statistics from trade records."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "avg_trade": 0.0,
                "avg_holding_hours": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0
            }
        
        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]
        
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        # Win rate
        win_rate = len(winners) / len(trades) if trades else 0
        
        # Average win/loss
        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 0
        
        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        # Holding period
        holding_hours = [t.holding_period_hours for t in trades]
        
        return {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_pct": np.mean([p for p in pnl_pcts if p > 0]) if winners else 0,
            "avg_loss_pct": abs(np.mean([p for p in pnl_pcts if p < 0])) if losers else 0,
            "profit_factor": profit_factor,
            "avg_trade": np.mean(pnls),
            "avg_trade_pct": np.mean(pnl_pcts),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "avg_holding_hours": np.mean(holding_hours),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "expectancy": win_rate * avg_win - (1 - win_rate) * avg_loss
        }
    
    # ===== Statistical Tests =====
    
    def t_test(self, returns: pd.Series) -> Tuple[float, float]:
        """
        Perform t-test to check if mean return is significantly different from zero.
        
        Returns:
            Tuple of (t-statistic, p-value)
        """
        if len(returns) < 10:
            return 0.0, 1.0
        
        t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
        
        return t_stat, p_value
    
    def is_statistically_significant(
        self,
        returns: pd.Series,
        alpha: float = 0.05
    ) -> bool:
        """Check if returns are statistically significant."""
        _, p_value = self.t_test(returns)
        return p_value < alpha
    
    def minimum_track_record(
        self,
        sharpe: float,
        target_sharpe: float = 0.0,
        confidence: float = 0.95
    ) -> int:
        """
        Calculate minimum track record length for statistical significance.
        
        Based on Bailey & Lopez de Prado (2012).
        
        Args:
            sharpe: Observed Sharpe ratio
            target_sharpe: Sharpe ratio to test against (usually 0)
            confidence: Confidence level
        
        Returns:
            Minimum number of periods needed
        """
        if sharpe <= target_sharpe:
            return float('inf')
        
        z = stats.norm.ppf(confidence)
        
        # MTL = (z / (sharpe - target_sharpe))^2
        mtl = (z / (sharpe - target_sharpe)) ** 2
        
        return int(np.ceil(mtl))
    
    # ===== Comprehensive Metrics =====
    
    def calculate_all(
        self,
        equity_curve: pd.Series,
        trades: Optional[List[TradeRecord]] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate all metrics.
        
        Args:
            equity_curve: Series of portfolio values over time
            trades: Optional list of trade records
            benchmark_returns: Optional benchmark returns for comparison
        
        Returns:
            Dictionary with all metrics
        """
        # Calculate returns from equity curve
        returns = equity_curve.pct_change().dropna()
        
        # Return metrics
        metrics = {
            "total_return": self.total_return(equity_curve),
            "annualized_return": self.annualized_return(equity_curve),
            "cagr": self.cagr(equity_curve),
        }
        
        # Risk metrics
        metrics.update({
            "volatility": self.volatility(returns),
            "downside_volatility": self.downside_volatility(returns),
            "max_drawdown": self.max_drawdown(equity_curve),
            "max_drawdown_duration": self.max_drawdown_duration(equity_curve),
            "var_95": self.var(returns, 0.95),
            "cvar_95": self.cvar(returns, 0.95),
        })
        
        # Risk-adjusted metrics
        metrics.update({
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "calmar_ratio": self.calmar_ratio(equity_curve),
        })
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics["information_ratio"] = self.information_ratio(
                returns, benchmark_returns
            )
        
        # Statistical significance
        t_stat, p_value = self.t_test(returns)
        metrics.update({
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
        })
        
        # Trade statistics
        if trades:
            trade_stats = self.calculate_trade_stats(trades)
            metrics.update(trade_stats)
        
        # Time info
        metrics.update({
            "start_date": equity_curve.index[0].isoformat() if hasattr(equity_curve.index[0], 'isoformat') else str(equity_curve.index[0]),
            "end_date": equity_curve.index[-1].isoformat() if hasattr(equity_curve.index[-1], 'isoformat') else str(equity_curve.index[-1]),
            "trading_days": len(equity_curve),
        })
        
        return metrics


# Singleton instance
_calculator: Optional[MetricsCalculator] = None


def get_metrics_calculator(risk_free_rate: float = 0.035) -> MetricsCalculator:
    """Get the singleton MetricsCalculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = MetricsCalculator(risk_free_rate)
    return _calculator
