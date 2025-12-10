"""
PORTFOLIO - Capital Allocation and Position Management

The Portfolio module handles:
1. Capital allocation across strategies (Kelly criterion)
2. Position sizing for individual trades
3. Position tracking and P&L calculation
4. Risk management at portfolio level

Components:
- CapitalAllocator: Allocates capital to strategies using Half-Kelly
- PositionManager: Tracks open positions and calculates P&L

Usage:
    from portfolio import (
        get_capital_allocator,
        get_position_manager,
        AllocationConfig
    )
    
    # Configure allocator
    config = AllocationConfig(
        total_capital=10_000_000,  # ₩10M
        method=AllocationMethod.HALF_KELLY,
        max_strategies=3
    )
    
    allocator = get_capital_allocator(config)
    allocator.allocate_to_strategies()
    
    # Size a position
    sizing = allocator.size_position(signal, current_price, atr)
    
    # Track positions
    positions = get_position_manager()
    positions.open_position(...)
"""

from portfolio.allocator import (
    # Main class
    CapitalAllocator,
    
    # Config
    AllocationConfig,
    AllocationMethod,
    
    # Data types
    StrategyAllocation,
    PositionSizing,
    
    # Kelly
    KellyCalculator,
    
    # Singleton
    get_capital_allocator,
)

from portfolio.positions import (
    # Main class
    PositionManager,
    
    # Data types
    Position,
    PositionSide,
    PositionStatus,
    PortfolioSummary,
    
    # Singleton
    get_position_manager,
)


__all__ = [
    # Allocator
    'CapitalAllocator',
    'AllocationConfig',
    'AllocationMethod',
    'StrategyAllocation',
    'PositionSizing',
    'KellyCalculator',
    'get_capital_allocator',
    
    # Positions
    'PositionManager',
    'Position',
    'PositionSide',
    'PositionStatus',
    'PortfolioSummary',
    'get_position_manager',
]
