"""Configuration management for Alpaca Trading Pipeline."""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from enum import Enum


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class TradingMode(str, Enum):
    """Trading modes."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class Config(BaseSettings):
    """Application configuration."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=True)
    
    # Alpaca API Configuration
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        env="ALPACA_BASE_URL"
    )
    
    # Trading Configuration
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)
    max_position_size: float = Field(default=10000.0)
    max_daily_loss: float = Field(default=1000.0)
    risk_per_trade: float = Field(default=0.02)  # 2% risk per trade
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./trading_data.db",
        env="DATABASE_URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL"
    )
    
    # Data Configuration
    data_storage_path: str = Field(default="./data")
    max_bars_per_request: int = Field(default=10000)
    data_update_interval: int = Field(default=60)  # seconds
    
    # Backtesting Configuration
    initial_capital: float = Field(default=100000.0)
    commission_per_share: float = Field(default=0.005)
    slippage_bps: float = Field(default=5.0)  # basis points
    
    # Monitoring Configuration
    log_level: str = Field(default="INFO")
    metrics_port: int = Field(default=8000)
    enable_notifications: bool = Field(default=False)
    notification_webhook: Optional[str] = Field(default=None)
    
    # Performance Configuration
    max_workers: int = Field(default=4)
    batch_size: int = Field(default=1000)
    cache_ttl: int = Field(default=300)  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")


def is_live_trading() -> bool:
    """Check if we're in live trading mode."""
    return config.trading_mode == TradingMode.LIVE


def is_paper_trading() -> bool:
    """Check if we're in paper trading mode."""
    return config.trading_mode == TradingMode.PAPER


def is_backtesting() -> bool:
    """Check if we're in backtesting mode."""
    return config.trading_mode == TradingMode.BACKTEST