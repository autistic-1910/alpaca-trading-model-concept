"""Logging utilities for the Alpaca Trading Pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from rich.logging import RichHandler
from rich.console import Console

from ..config import get_config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json: bool = False,
) -> None:
    """Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_json: Whether to enable JSON formatting
    """
    config = get_config()
    
    # Determine log level
    if log_level is None:
        log_level = config.logging.level
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    handlers = []
    
    # Console handler with Rich formatting
    console = Console()
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(getattr(logging, log_level.upper()))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if enable_json:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format="%(message)s",
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("alpaca_trade_api").setLevel(logging.INFO)
    
    # Log startup message
    logger = get_logger("setup")
    logger.info(
        "Logging configured",
        level=log_level,
        file=log_file,
        json_format=enable_json,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class TradingLogger:
    """Specialized logger for trading operations."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_order(self, order_type: str, symbol: str, quantity: float, price: float, **kwargs):
        """Log order information."""
        self.logger.info(
            "Order placed",
            order_type=order_type,
            symbol=symbol,
            quantity=quantity,
            price=price,
            **kwargs
        )
    
    def log_fill(self, symbol: str, quantity: float, price: float, **kwargs):
        """Log order fill information."""
        self.logger.info(
            "Order filled",
            symbol=symbol,
            quantity=quantity,
            price=price,
            **kwargs
        )
    
    def log_pnl(self, symbol: str, realized_pnl: float, unrealized_pnl: float, **kwargs):
        """Log P&L information."""
        self.logger.info(
            "P&L update",
            symbol=symbol,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            **kwargs
        )
    
    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log error with context."""
        self.logger.error(
            "Trading error occurred",
            error=str(error),
            error_type=type(error).__name__,
            context=context,
            **kwargs
        )
    
    def log_strategy_signal(self, strategy: str, symbol: str, signal: str, **kwargs):
        """Log strategy signal."""
        self.logger.info(
            "Strategy signal",
            strategy=strategy,
            symbol=symbol,
            signal=signal,
            **kwargs
        )