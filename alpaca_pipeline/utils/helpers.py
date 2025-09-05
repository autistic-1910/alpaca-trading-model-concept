"""Helper functions for the Alpaca Trading Pipeline."""

import re
import math
from typing import Union, Optional
from decimal import Decimal, ROUND_HALF_UP

from ..config import get_config


def validate_symbol(symbol: str) -> bool:
    """Validate if a symbol is properly formatted.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic symbol validation (1-5 uppercase letters)
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, symbol.upper()))


def calculate_position_size(
    account_value: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_value: Optional[float] = None
) -> int:
    """Calculate position size based on risk management rules.
    
    Args:
        account_value: Total account value
        risk_per_trade: Risk percentage per trade (e.g., 0.02 for 2%)
        entry_price: Entry price per share
        stop_loss_price: Stop loss price per share
        max_position_value: Maximum position value limit
        
    Returns:
        Number of shares to trade
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0
    
    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_loss_price)
    
    if risk_per_share == 0:
        return 0
    
    # Calculate maximum risk amount
    max_risk_amount = account_value * risk_per_trade
    
    # Calculate position size based on risk
    position_size = int(max_risk_amount / risk_per_share)
    
    # Apply maximum position value limit if specified
    if max_position_value:
        max_shares_by_value = int(max_position_value / entry_price)
        position_size = min(position_size, max_shares_by_value)
    
    # Apply global maximum position size from config
    config = get_config()
    max_shares_by_config = int(config.max_position_size / entry_price)
    position_size = min(position_size, max_shares_by_config)
    
    return max(0, position_size)


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency.upper() == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def round_to_tick_size(price: float, tick_size: float = 0.01) -> float:
    """Round price to the nearest tick size.
    
    Args:
        price: Price to round
        tick_size: Minimum price increment
        
    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price
    
    # Use Decimal for precise rounding
    decimal_price = Decimal(str(price))
    decimal_tick = Decimal(str(tick_size))
    
    # Round to nearest tick
    rounded = (decimal_price / decimal_tick).quantize(
        Decimal('1'), rounding=ROUND_HALF_UP
    ) * decimal_tick
    
    return float(rounded)


def calculate_sharpe_ratio(
    returns: list,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """Calculate Sharpe ratio for a series of returns.
    
    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    import numpy as np
    
    returns_array = np.array(returns)
    
    # Calculate excess returns
    period_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = returns_array - period_risk_free_rate
    
    # Calculate Sharpe ratio
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_max_drawdown(equity_curve: list) -> tuple:
    """Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: List of equity values
        
    Returns:
        Tuple of (max_drawdown_pct, max_drawdown_duration)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0
    
    import numpy as np
    
    equity_array = np.array(equity_curve)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)
    
    # Calculate drawdown
    drawdown = (equity_array - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.min(drawdown)
    
    # Calculate drawdown duration
    drawdown_duration = 0
    current_duration = 0
    max_duration = 0
    
    for dd in drawdown:
        if dd < 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return float(abs(max_drawdown)), max_duration


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format.
    
    Args:
        symbol: Raw symbol
        
    Returns:
        Normalized symbol
    """
    if not symbol:
        return ""
    
    return symbol.upper().strip()


def calculate_volatility(prices: list, periods: int = 20) -> float:
    """Calculate rolling volatility.
    
    Args:
        prices: List of prices
        periods: Number of periods for calculation
        
    Returns:
        Volatility value
    """
    if len(prices) < periods + 1:
        return 0.0
    
    import numpy as np
    
    # Calculate returns
    prices_array = np.array(prices)
    returns = np.diff(np.log(prices_array))
    
    # Calculate rolling volatility
    if len(returns) < periods:
        return float(np.std(returns))
    
    recent_returns = returns[-periods:]
    return float(np.std(recent_returns))


def is_market_hours() -> bool:
    """Check if current time is within market hours.
    
    Returns:
        True if market is open, False otherwise
    """
    from datetime import datetime, time
    import pytz
    
    # Get current time in Eastern timezone
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    current_time = now.time()
    return market_open <= current_time <= market_close


def get_trading_days_between(start_date, end_date) -> int:
    """Calculate number of trading days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of trading days
    """
    import pandas as pd
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Filter out holidays (basic implementation)
    # In production, you might want to use a more comprehensive holiday calendar
    trading_days = len(date_range)
    
    return trading_days