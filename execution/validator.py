"""
SIGNAL VALIDATOR - Validate trading signals before execution

Provides comprehensive signal validation:
- Position limit checks
- Correlation checks with existing positions
- Liquidity checks
- Market condition checks
- Risk limit checks
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from loguru import logger


class ValidationResult(str, Enum):
    """Result of signal validation."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationCheck:
    """A single validation check result."""
    check_name: str
    result: ValidationResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalValidation:
    """Complete validation result for a signal."""
    signal_id: str
    is_valid: bool
    checks: List[ValidationCheck]
    original_confidence: float
    adjusted_confidence: float
    rejection_reason: Optional[str] = None
    validated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "is_valid": self.is_valid,
            "checks": [
                {"name": c.check_name, "result": c.result.value, "message": c.message}
                for c in self.checks
            ],
            "original_confidence": self.original_confidence,
            "adjusted_confidence": self.adjusted_confidence,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class ValidatorConfig:
    """Configuration for signal validator."""
    # Position limits
    max_position_pct: float = 0.15  # Max 15% in one position
    max_total_positions: int = 10  # Max open positions
    min_position_size_krw: float = 100_000  # Min ₩100K per position

    # Correlation limits
    max_correlation: float = 0.7  # Max correlation with existing positions
    correlation_lookback_days: int = 60

    # Liquidity requirements
    min_avg_daily_volume: int = 10_000  # Min 10K shares/day
    max_position_vs_adv: float = 0.1  # Max 10% of ADV

    # Market conditions
    require_market_open: bool = True
    max_spread_pct: float = 0.02  # Max 2% bid-ask spread

    # Risk limits
    max_daily_trades: int = 20
    max_trades_per_symbol: int = 3
    min_time_between_trades_minutes: int = 5


class SignalValidator:
    """
    Validates trading signals before execution.

    Performs checks on:
    1. Position limits
    2. Correlation with existing positions
    3. Liquidity
    4. Market conditions
    5. Risk limits
    """

    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()
        self._trade_history: List[Dict[str, Any]] = []
        self._correlation_cache: Dict[str, Dict[str, float]] = {}
        logger.info("SignalValidator initialized")

    def validate(
        self,
        signal: Any,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> SignalValidation:
        """
        Validate a trading signal.

        Args:
            signal: The trading signal to validate
            portfolio_value: Current portfolio value
            current_positions: Dict of current positions {symbol: position}
            market_data: Optional market data for the signal symbol

        Returns:
            SignalValidation with all check results
        """
        checks = []
        original_confidence = getattr(signal, 'confidence', 0.5)
        adjusted_confidence = original_confidence

        # Get signal attributes
        symbol = getattr(signal, 'symbol', '')
        signal_id = getattr(signal, 'signal_id', str(id(signal)))

        # 1. Position limit check
        position_check = self._check_position_limits(
            signal, portfolio_value, current_positions
        )
        checks.append(position_check)
        if position_check.result == ValidationResult.FAILED:
            return self._create_failed_validation(
                signal_id, checks, original_confidence, position_check.message
            )

        # 2. Correlation check
        correlation_check = self._check_correlation(symbol, current_positions)
        checks.append(correlation_check)
        if correlation_check.result == ValidationResult.WARNING:
            adjusted_confidence *= 0.8  # Reduce confidence for correlated positions

        # 3. Liquidity check
        if market_data:
            liquidity_check = self._check_liquidity(signal, market_data)
            checks.append(liquidity_check)
            if liquidity_check.result == ValidationResult.FAILED:
                return self._create_failed_validation(
                    signal_id, checks, original_confidence, liquidity_check.message
                )

        # 4. Market condition check
        market_check = self._check_market_conditions(market_data)
        checks.append(market_check)
        if market_check.result == ValidationResult.FAILED:
            return self._create_failed_validation(
                signal_id, checks, original_confidence, market_check.message
            )

        # 5. Risk limit check
        risk_check = self._check_risk_limits(symbol)
        checks.append(risk_check)
        if risk_check.result == ValidationResult.FAILED:
            return self._create_failed_validation(
                signal_id, checks, original_confidence, risk_check.message
            )

        # 6. Confidence adjustment based on checks
        warning_count = sum(1 for c in checks if c.result == ValidationResult.WARNING)
        adjusted_confidence *= (1 - 0.1 * warning_count)
        adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))

        # All checks passed
        return SignalValidation(
            signal_id=signal_id,
            is_valid=True,
            checks=checks,
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
        )

    def _check_position_limits(
        self,
        signal: Any,
        portfolio_value: float,
        current_positions: Dict[str, Any],
    ) -> ValidationCheck:
        """Check position size limits."""
        symbol = getattr(signal, 'symbol', '')
        size_pct = getattr(signal, 'size_pct', 0.1)

        # Check max position size
        if size_pct > self.config.max_position_pct:
            return ValidationCheck(
                check_name="position_size",
                result=ValidationResult.FAILED,
                message=f"Position size {size_pct:.1%} exceeds max {self.config.max_position_pct:.1%}",
                details={"requested": size_pct, "max": self.config.max_position_pct}
            )

        # Check total positions
        if len(current_positions) >= self.config.max_total_positions:
            if symbol not in current_positions:
                return ValidationCheck(
                    check_name="total_positions",
                    result=ValidationResult.FAILED,
                    message=f"Max positions ({self.config.max_total_positions}) reached",
                    details={"current": len(current_positions)}
                )

        # Check minimum size
        position_value = portfolio_value * size_pct
        if position_value < self.config.min_position_size_krw:
            return ValidationCheck(
                check_name="min_position_size",
                result=ValidationResult.FAILED,
                message=f"Position value ₩{position_value:,.0f} below minimum ₩{self.config.min_position_size_krw:,.0f}",
            )

        return ValidationCheck(
            check_name="position_limits",
            result=ValidationResult.PASSED,
            message="Position limits OK",
        )

    def _check_correlation(
        self,
        symbol: str,
        current_positions: Dict[str, Any],
    ) -> ValidationCheck:
        """Check correlation with existing positions."""
        if not current_positions:
            return ValidationCheck(
                check_name="correlation",
                result=ValidationResult.PASSED,
                message="No existing positions to check correlation",
            )

        # In a real implementation, we'd calculate actual correlations
        # For now, we'll use a simplified sector-based check
        high_correlation_symbols = []

        for pos_symbol in current_positions.keys():
            # Simple heuristic: same prefix often means same sector
            if symbol[:3] == pos_symbol[:3] and symbol != pos_symbol:
                high_correlation_symbols.append(pos_symbol)

        if high_correlation_symbols:
            return ValidationCheck(
                check_name="correlation",
                result=ValidationResult.WARNING,
                message=f"Potentially correlated with: {high_correlation_symbols[:3]}",
                details={"correlated_symbols": high_correlation_symbols}
            )

        return ValidationCheck(
            check_name="correlation",
            result=ValidationResult.PASSED,
            message="No high correlations detected",
        )

    def _check_liquidity(
        self,
        signal: Any,
        market_data: Dict[str, Any],
    ) -> ValidationCheck:
        """Check liquidity requirements."""
        avg_volume = market_data.get('avg_daily_volume', 0)
        signal_quantity = getattr(signal, 'quantity', 0)

        if avg_volume < self.config.min_avg_daily_volume:
            return ValidationCheck(
                check_name="liquidity",
                result=ValidationResult.FAILED,
                message=f"ADV {avg_volume:,} below minimum {self.config.min_avg_daily_volume:,}",
            )

        if avg_volume > 0 and signal_quantity / avg_volume > self.config.max_position_vs_adv:
            return ValidationCheck(
                check_name="liquidity",
                result=ValidationResult.WARNING,
                message=f"Order size is {signal_quantity/avg_volume:.1%} of ADV",
            )

        return ValidationCheck(
            check_name="liquidity",
            result=ValidationResult.PASSED,
            message="Liquidity OK",
        )

    def _check_market_conditions(
        self,
        market_data: Optional[Dict[str, Any]],
    ) -> ValidationCheck:
        """Check market conditions."""
        if self.config.require_market_open:
            try:
                from core.clock import is_market_open
                if not is_market_open():
                    return ValidationCheck(
                        check_name="market_hours",
                        result=ValidationResult.FAILED,
                        message="Market is closed",
                    )
            except ImportError:
                pass

        if market_data:
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / bid
                if spread_pct > self.config.max_spread_pct:
                    return ValidationCheck(
                        check_name="spread",
                        result=ValidationResult.WARNING,
                        message=f"Spread {spread_pct:.2%} exceeds threshold {self.config.max_spread_pct:.2%}",
                    )

        return ValidationCheck(
            check_name="market_conditions",
            result=ValidationResult.PASSED,
            message="Market conditions OK",
        )

    def _check_risk_limits(self, symbol: str) -> ValidationCheck:
        """Check risk limits like trade frequency."""
        now = datetime.utcnow()

        # Count recent trades
        recent_trades = [
            t for t in self._trade_history
            if (now - t['timestamp']).total_seconds() < 86400  # Last 24 hours
        ]

        if len(recent_trades) >= self.config.max_daily_trades:
            return ValidationCheck(
                check_name="daily_trade_limit",
                result=ValidationResult.FAILED,
                message=f"Daily trade limit ({self.config.max_daily_trades}) reached",
            )

        # Count trades for this symbol
        symbol_trades = [t for t in recent_trades if t.get('symbol') == symbol]
        if len(symbol_trades) >= self.config.max_trades_per_symbol:
            return ValidationCheck(
                check_name="symbol_trade_limit",
                result=ValidationResult.WARNING,
                message=f"Many trades today for {symbol} ({len(symbol_trades)})",
            )

        # Check time since last trade for this symbol
        if symbol_trades:
            last_trade = max(symbol_trades, key=lambda t: t['timestamp'])
            minutes_since = (now - last_trade['timestamp']).total_seconds() / 60
            if minutes_since < self.config.min_time_between_trades_minutes:
                return ValidationCheck(
                    check_name="trade_frequency",
                    result=ValidationResult.WARNING,
                    message=f"Only {minutes_since:.0f}m since last trade for {symbol}",
                )

        return ValidationCheck(
            check_name="risk_limits",
            result=ValidationResult.PASSED,
            message="Risk limits OK",
        )

    def _create_failed_validation(
        self,
        signal_id: str,
        checks: List[ValidationCheck],
        original_confidence: float,
        reason: str,
    ) -> SignalValidation:
        """Create a failed validation result."""
        return SignalValidation(
            signal_id=signal_id,
            is_valid=False,
            checks=checks,
            original_confidence=original_confidence,
            adjusted_confidence=0.0,
            rejection_reason=reason,
        )

    def record_trade(self, symbol: str, side: str) -> None:
        """Record a trade for risk limit tracking."""
        self._trade_history.append({
            'symbol': symbol,
            'side': side,
            'timestamp': datetime.utcnow(),
        })
        # Keep only last 100 trades
        if len(self._trade_history) > 100:
            self._trade_history = self._trade_history[-100:]

    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics."""
        now = datetime.utcnow()
        recent = [
            t for t in self._trade_history
            if (now - t['timestamp']).total_seconds() < 86400
        ]
        return {
            "trades_today": len(recent),
            "max_daily_trades": self.config.max_daily_trades,
            "total_tracked_trades": len(self._trade_history),
        }


# Singleton accessor
_validator_instance: Optional[SignalValidator] = None


def get_validator() -> SignalValidator:
    """Get the singleton SignalValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SignalValidator()
    return _validator_instance
