"""Configuration management for Alpaca Trading Pipeline."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str = ""
    secret_key: str = ""
    paper_trading: bool = True
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    
    def __post_init__(self):
        """Load from environment variables if not provided."""
        if not self.api_key:
            self.api_key = os.getenv('ALPACA_API_KEY', '')
        if not self.secret_key:
            self.secret_key = os.getenv('ALPACA_SECRET_KEY', '')
        
        # Set URLs based on paper trading mode
        if self.paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return bool(self.api_key and self.secret_key)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    database_type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "alpaca_trading"
    username: str = ""
    password: str = ""
    sqlite_path: str = "data/trading.db"
    
    def __post_init__(self):
        """Load from environment variables if not provided."""
        self.database_type = os.getenv('DATABASE_TYPE', self.database_type)
        self.host = os.getenv('DATABASE_HOST', self.host)
        self.port = int(os.getenv('DATABASE_PORT', str(self.port)))
        self.database = os.getenv('DATABASE_NAME', self.database)
        self.username = os.getenv('DATABASE_USERNAME', self.username)
        self.password = os.getenv('DATABASE_PASSWORD', self.password)
        self.sqlite_path = os.getenv('SQLITE_PATH', self.sqlite_path)
    
    def get_connection_string(self) -> str:
        """Get database connection string."""
        if self.database_type == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        elif self.database_type == "postgresql":
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.database_type == "mysql":
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")


@dataclass
class DataConfig:
    """Data configuration."""
    default_provider: str = "alpaca"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
    data_directory: str = "data"
    
    def __post_init__(self):
        """Load from environment variables if not provided."""
        self.default_provider = os.getenv('DATA_PROVIDER', self.default_provider)
        self.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.cache_ttl_seconds = int(os.getenv('CACHE_TTL_SECONDS', str(self.cache_ttl_seconds)))
        self.max_cache_size = int(os.getenv('MAX_CACHE_SIZE', str(self.max_cache_size)))
        self.data_directory = os.getenv('DATA_DIRECTORY', self.data_directory)
        
        # Create data directory if it doesn't exist
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/trading.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        """Load from environment variables if not provided."""
        self.level = os.getenv('LOG_LEVEL', self.level)
        self.file_enabled = os.getenv('LOG_FILE_ENABLED', 'true').lower() == 'true'
        self.file_path = os.getenv('LOG_FILE_PATH', self.file_path)
        
        # Create logs directory if it doesn't exist
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class TradingConfig:
    """Trading configuration."""
    max_positions: int = 10
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.15  # 15%
    risk_per_trade: float = 0.02  # 2% of portfolio
    
    def __post_init__(self):
        """Load from environment variables if not provided."""
        self.max_positions = int(os.getenv('MAX_POSITIONS', str(self.max_positions)))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', str(self.max_position_size)))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', str(self.stop_loss_pct)))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', str(self.take_profit_pct)))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', str(self.risk_per_trade)))


@dataclass
class Config:
    """Main configuration class."""
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    
    def __post_init__(self):
        """Initialize sub-configurations."""
        if not isinstance(self.alpaca, AlpacaConfig):
            self.alpaca = AlpacaConfig()
        if not isinstance(self.database, DatabaseConfig):
            self.database = DatabaseConfig()
        if not isinstance(self.data, DataConfig):
            self.data = DataConfig()
        if not isinstance(self.logging, LoggingConfig):
            self.logging = LoggingConfig()
        if not isinstance(self.trading, TradingConfig):
            self.trading = TradingConfig()
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.alpaca.is_valid():
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            'alpaca': {
                'paper_trading': self.alpaca.paper_trading,
                'base_url': self.alpaca.base_url,
                'data_url': self.alpaca.data_url,
                'api_key_set': bool(self.alpaca.api_key),
                'secret_key_set': bool(self.alpaca.secret_key)
            },
            'database': {
                'database_type': self.database.database_type,
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database
            },
            'data': {
                'default_provider': self.data.default_provider,
                'cache_enabled': self.data.cache_enabled,
                'cache_ttl_seconds': self.data.cache_ttl_seconds,
                'data_directory': self.data.data_directory
            },
            'logging': {
                'level': self.logging.level,
                'file_enabled': self.logging.file_enabled,
                'file_path': self.logging.file_path
            },
            'trading': {
                'max_positions': self.trading.max_positions,
                'max_position_size': self.trading.max_position_size,
                'stop_loss_pct': self.trading.stop_loss_pct,
                'take_profit_pct': self.trading.take_profit_pct,
                'risk_per_trade': self.trading.risk_per_trade
            }
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global config
    config = Config()
    return config