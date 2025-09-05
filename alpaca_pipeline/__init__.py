"""Alpaca Trading Pipeline - A comprehensive end-to-end trading system."""

__version__ = "1.0.0"
__author__ = "Alpaca Trading Pipeline"
__description__ = "A robust, scalable algorithmic trading pipeline for Alpaca Markets"

from .config import config, get_config
from .data import DataCollector, DataPreprocessor
from .strategies import BaseStrategy, StrategyManager
from .backtesting import BacktestEngine
from .live_trading import LiveTradingEngine as TradingEngine
from .utils import setup_logging, get_logger

__all__ = [
    "config",
    "get_config",
    "DataCollector",
    "DataPreprocessor",
    "BaseStrategy",
    "StrategyManager",
    "BacktestEngine",
    "TradingEngine",
    "setup_logging",
    "get_logger",
]