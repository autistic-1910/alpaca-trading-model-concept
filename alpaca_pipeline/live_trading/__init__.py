"""Live trading module for the Alpaca Trading Pipeline.

This module provides components for live trading execution, including:
- Live trading engine for real-time strategy execution
- Order execution and management
- Real-time risk monitoring and controls
- Position management and tracking
- Market data integration
- Performance monitoring
"""

from .market_interface import AlpacaMarketInterface

# Placeholder classes for live trading components
class LiveTradingEngine:
    """Placeholder for live trading engine."""
    def __init__(self, *args, **kwargs):
        pass

class OrderExecutor:
    """Placeholder for order executor."""
    def __init__(self, *args, **kwargs):
        pass

__all__ = [
    # Core engine
    'LiveTradingEngine',
    
    # Order execution
    'OrderExecutor',
    
    # Market interface
    'AlpacaMarketInterface'
]

__version__ = '1.0.0'