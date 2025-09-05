"""Strategy framework for the Alpaca Trading Pipeline.

This module provides the core strategy framework including:
- BaseStrategy: Abstract base class for all trading strategies
- StrategyManager: Manager for running multiple strategies
- Example strategies: Moving average, mean reversion, momentum strategies
- Strategy utilities: Signal generation, position sizing, risk management
"""

from .base import BaseStrategy, StrategySignal, StrategyState, SignalType
from .manager import StrategyManager
from .examples import (
    MovingAverageStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    BuyAndHoldStrategy
)
from ..utils import (
    calculate_position_size,
    validate_strategy_params
)

__all__ = [
    # Base classes
    'BaseStrategy',
    'StrategySignal',
    'StrategyState',
    'SignalType',
    
    # Manager
    'StrategyManager',
    
    # Example strategies
    'MovingAverageStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'BuyAndHoldStrategy',
    
    # Utilities
    'calculate_position_size',
    'validate_strategy_params'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Alpaca Trading Pipeline'
__description__ = 'Strategy framework for algorithmic trading'