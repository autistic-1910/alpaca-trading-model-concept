"""Validation functions for the Alpaca Trading Pipeline."""

from typing import Dict, Any, Union, List
from decimal import Decimal
import re
from datetime import datetime

from .helpers import validate_symbol


def validate_order(order_data: Dict[str, Any]) -> bool:
    """Validate order data structure.
    
    Args:
        order_data: Dictionary containing order information
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['symbol', 'qty', 'side', 'type']
    
    # Check required fields
    for field in required_fields:
        if field not in order_data:
            return False
    
    # Validate symbol
    if not validate_symbol(order_data['symbol']):
        return False
    
    # Validate quantity
    try:
        qty = float(order_data['qty'])
        if qty <= 0:
            return False
    except (ValueError, TypeError):
        return False
    
    # Validate side
    valid_sides = ['buy', 'sell']
    if order_data['side'].lower() not in valid_sides:
        return False
    
    # Validate order type
    valid_types = ['market', 'limit', 'stop', 'stop_limit']
    if order_data['type'].lower() not in valid_types:
        return False
    
    # Validate limit price if limit order
    if order_data['type'].lower() in ['limit', 'stop_limit']:
        if 'limit_price' not in order_data:
            return False
        try:
            limit_price = float(order_data['limit_price'])
            if limit_price <= 0:
                return False
        except (ValueError, TypeError):
            return False
    
    # Validate stop price if stop order
    if order_data['type'].lower() in ['stop', 'stop_limit']:
        if 'stop_price' not in order_data:
            return False
        try:
            stop_price = float(order_data['stop_price'])
            if stop_price <= 0:
                return False
        except (ValueError, TypeError):
            return False
    
    # Validate time in force
    if 'time_in_force' in order_data:
        valid_tif = ['day', 'gtc', 'ioc', 'fok']
        if order_data['time_in_force'].lower() not in valid_tif:
            return False
    
    return True


def validate_strategy_params(params: Dict[str, Any], schema: Dict[str, Dict]) -> bool:
    """Validate strategy parameters against a schema.
    
    Args:
        params: Strategy parameters to validate
        schema: Parameter schema with validation rules
        
    Returns:
        True if valid, False otherwise
    """
    for param_name, rules in schema.items():
        # Check required parameters
        if rules.get('required', False) and param_name not in params:
            return False
        
        if param_name not in params:
            continue
        
        value = params[param_name]
        
        # Check type
        expected_type = rules.get('type')
        if expected_type and not isinstance(value, expected_type):
            # Try to convert if possible
            try:
                if expected_type == float:
                    value = float(value)
                elif expected_type == int:
                    value = int(value)
                elif expected_type == str:
                    value = str(value)
                else:
                    return False
                params[param_name] = value
            except (ValueError, TypeError):
                return False
        
        # Check range
        if 'min' in rules and value < rules['min']:
            return False
        if 'max' in rules and value > rules['max']:
            return False
        
        # Check choices
        if 'choices' in rules and value not in rules['choices']:
            return False
        
        # Custom validator
        if 'validator' in rules and not rules['validator'](value):
            return False
    
    return True


def validate_price(price: Union[float, str, Decimal]) -> bool:
    """Validate price value.
    
    Args:
        price: Price to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        price_float = float(price)
        return price_float > 0 and price_float < 1000000  # Reasonable price range
    except (ValueError, TypeError):
        return False


def validate_quantity(quantity: Union[float, str, int]) -> bool:
    """Validate quantity value.
    
    Args:
        quantity: Quantity to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        qty_float = float(quantity)
        return qty_float > 0 and qty_float <= 1000000  # Reasonable quantity range
    except (ValueError, TypeError):
        return False


def validate_percentage(percentage: Union[float, str]) -> bool:
    """Validate percentage value (0-100).
    
    Args:
        percentage: Percentage to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        pct_float = float(percentage)
        return 0 <= pct_float <= 100
    except (ValueError, TypeError):
        return False


def validate_ratio(ratio: Union[float, str]) -> bool:
    """Validate ratio value (0-1).
    
    Args:
        ratio: Ratio to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        ratio_float = float(ratio)
        return 0 <= ratio_float <= 1
    except (ValueError, TypeError):
        return False


def validate_date_string(date_string: str, format_string: str = "%Y-%m-%d") -> bool:
    """Validate date string format.
    
    Args:
        date_string: Date string to validate
        format_string: Expected date format
        
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_string, format_string)
        return True
    except ValueError:
        return False


def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe string.
    
    Args:
        timeframe: Timeframe to validate (e.g., '1Min', '5Min', '1Hour', '1Day')
        
    Returns:
        True if valid, False otherwise
    """
    valid_timeframes = [
        '1Min', '5Min', '15Min', '30Min',
        '1Hour', '2Hour', '4Hour',
        '1Day', '1Week', '1Month'
    ]
    return timeframe in valid_timeframes


def validate_email(email: str) -> bool:
    """Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_api_key(api_key: str) -> bool:
    """Validate API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic validation - should be alphanumeric and reasonable length
    if len(api_key) < 10 or len(api_key) > 100:
        return False
    
    # Should contain only alphanumeric characters and common special chars
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, api_key))


def validate_portfolio_weights(weights: Dict[str, float]) -> bool:
    """Validate portfolio weights sum to 1.0.
    
    Args:
        weights: Dictionary of symbol -> weight
        
    Returns:
        True if valid, False otherwise
    """
    if not weights:
        return False
    
    try:
        total_weight = sum(weights.values())
        # Allow small floating point errors
        return abs(total_weight - 1.0) < 0.001
    except (TypeError, ValueError):
        return False


def validate_risk_parameters(params: Dict[str, Any]) -> bool:
    """Validate risk management parameters.
    
    Args:
        params: Risk parameters to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_params = ['max_position_size', 'max_daily_loss', 'risk_per_trade']
    
    for param in required_params:
        if param not in params:
            return False
        
        try:
            value = float(params[param])
            if value <= 0:
                return False
        except (ValueError, TypeError):
            return False
    
    # Risk per trade should be reasonable (0.1% to 10%)
    risk_per_trade = float(params['risk_per_trade'])
    if not (0.001 <= risk_per_trade <= 0.1):
        return False
    
    return True


def validate_backtest_parameters(params: Dict[str, Any]) -> bool:
    """Validate backtesting parameters.
    
    Args:
        params: Backtest parameters to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_params = ['start_date', 'end_date', 'initial_capital']
    
    for param in required_params:
        if param not in params:
            return False
    
    # Validate dates
    if not validate_date_string(params['start_date']):
        return False
    if not validate_date_string(params['end_date']):
        return False
    
    # Validate date order
    start_date = datetime.strptime(params['start_date'], "%Y-%m-%d")
    end_date = datetime.strptime(params['end_date'], "%Y-%m-%d")
    if start_date >= end_date:
        return False
    
    # Validate initial capital
    try:
        capital = float(params['initial_capital'])
        if capital <= 0:
            return False
    except (ValueError, TypeError):
        return False
    
    return True


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_or_raise(validator_func, value, error_message: str = None):
    """Validate value or raise ValidationError.
    
    Args:
        validator_func: Validation function
        value: Value to validate
        error_message: Custom error message
        
    Raises:
        ValidationError: If validation fails
    """
    if not validator_func(value):
        if error_message:
            raise ValidationError(error_message)
        else:
            raise ValidationError(f"Validation failed for value: {value}")