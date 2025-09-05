"""Backtesting module for the Alpaca Trading Pipeline.

This module provides comprehensive backtesting capabilities including:
- BacktestEngine: Core backtesting engine for strategy evaluation
- PerformanceAnalyzer: Performance metrics calculation and analysis
- Visualizer: Charts and plots for backtest results
- Portfolio: Portfolio simulation and tracking
- OrderManager: Order execution simulation
- RiskManager: Risk management and position sizing
"""

from .engine import BacktestEngine
from .portfolio import Portfolio, Position
from .performance import PerformanceAnalyzer, PerformanceMetrics
from .visualizer import BacktestVisualizer
from .order_manager import OrderManager, Order, OrderStatus, OrderType
from .risk_manager import RiskManager, RiskMetrics

__all__ = [
    'BacktestEngine',
    'Portfolio',
    'Position',
    'PerformanceAnalyzer',
    'PerformanceMetrics',
    'BacktestVisualizer',
    'OrderManager',
    'Order',
    'OrderStatus',
    'OrderType',
    'RiskManager',
    'RiskMetrics'
]

__version__ = '1.0.0'