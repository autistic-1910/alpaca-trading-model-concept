"""Data models for market data structures."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal


@dataclass
class BarData:
    """OHLCV bar data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: Optional[int] = None
    vwap: Optional[float] = None
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("High price cannot be less than open, close, or low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("Low price cannot be greater than open, close, or high")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def price_range(self) -> float:
        """Calculate price range (high - low)."""
        return self.high - self.low
    
    @property
    def body_size(self) -> float:
        """Calculate candle body size."""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> float:
        """Calculate upper shadow length."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Calculate lower shadow length."""
        return min(self.open, self.close) - self.low
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'trade_count': self.trade_count,
            'vwap': self.vwap,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BarData':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class TradeData:
    """Individual trade data structure."""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    conditions: Optional[list] = None
    exchange: Optional[str] = None
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.size <= 0:
            raise ValueError("Size must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'size': self.size,
            'conditions': self.conditions,
            'exchange': self.exchange,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeData':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class QuoteData:
    """Bid/ask quote data structure."""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    bid_exchange: Optional[str] = None
    ask_exchange: Optional[str] = None
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.bid_price <= 0 or self.ask_price <= 0:
            raise ValueError("Bid and ask prices must be positive")
        if self.bid_size <= 0 or self.ask_size <= 0:
            raise ValueError("Bid and ask sizes must be positive")
        if self.bid_price >= self.ask_price:
            raise ValueError("Bid price cannot be greater than or equal to ask price")
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        return (self.spread / self.mid_price) * 10000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bid_price': self.bid_price,
            'bid_size': self.bid_size,
            'ask_price': self.ask_price,
            'ask_size': self.ask_size,
            'bid_exchange': self.bid_exchange,
            'ask_exchange': self.ask_exchange,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuoteData':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class MarketStatus:
    """Market status information."""
    is_open: bool
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_open': self.is_open,
            'next_open': self.next_open.isoformat() if self.next_open else None,
            'next_close': self.next_close.isoformat() if self.next_close else None,
        }


@dataclass
class AssetInfo:
    """Asset information structure."""
    symbol: str
    name: str
    asset_class: str
    exchange: str
    tradable: bool
    marginable: bool
    shortable: bool
    easy_to_borrow: bool
    fractionable: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'asset_class': self.asset_class,
            'exchange': self.exchange,
            'tradable': self.tradable,
            'marginable': self.marginable,
            'shortable': self.shortable,
            'easy_to_borrow': self.easy_to_borrow,
            'fractionable': self.fractionable,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssetInfo':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot data."""
    timestamp: datetime
    equity: float
    buying_power: float
    cash: float
    portfolio_value: float
    long_market_value: float
    short_market_value: float
    day_trade_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'equity': self.equity,
            'buying_power': self.buying_power,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'long_market_value': self.long_market_value,
            'short_market_value': self.short_market_value,
            'day_trade_count': self.day_trade_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioSnapshot':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)