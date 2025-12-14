"""
EXECUTION MODULE - Order execution and signal validation

Components:
- KISBroker: Interface to Korea Investment & Securities API
- SignalValidator: Validates signals before execution
- CorrelationManager: Calculates correlations for position limits
"""

from execution.broker import KISBroker
from execution.validator import (
    SignalValidator,
    SignalValidation,
    ValidationCheck,
    ValidationResult,
    ValidatorConfig,
    CorrelationManager,
    get_validator,
)

__all__ = [
    "KISBroker",
    "SignalValidator",
    "SignalValidation",
    "ValidationCheck",
    "ValidationResult",
    "ValidatorConfig",
    "CorrelationManager",
    "get_validator",
]
