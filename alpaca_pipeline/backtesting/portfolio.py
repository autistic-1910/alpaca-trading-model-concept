"""Portfolio management for backtesting."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..utils import get_logger
from ..data.models import BarData
from .order_manager import Order, OrderFill, OrderSide

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents a position in a security."""
    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    
    # Position tracking
    total_bought: int = 0
    total_sold: int = 0
    total_cost: float = 0.0
    total_proceeds: float = 0.0
    
    # Timestamps
    first_trade_date: Optional[datetime] = None
    last_trade_date: Optional[datetime] = None
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.quantity == 0
    
    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def cost_basis(self) -> float:
        """Get total cost basis."""
        return abs(self.quantity) * self.avg_cost
    
    def update_market_value(self, current_price: float):
        """Update market value and unrealized P&L.
        
        Args:
            current_price: Current market price
        """
        self.last_price = current_price
        self.market_value = self.quantity * current_price
        
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity
        else:
            self.unrealized_pnl = 0.0
    
    def add_trade(self, side: OrderSide, quantity: int, price: float, timestamp: datetime):
        """Add a trade to the position.
        
        Args:
            side: Buy or sell
            quantity: Number of shares
            price: Trade price
            timestamp: Trade timestamp
        """
        # Update timestamps
        if self.first_trade_date is None:
            self.first_trade_date = timestamp
        self.last_trade_date = timestamp
        
        if side == OrderSide.BUY:
            self._add_buy_trade(quantity, price)
        else:
            self._add_sell_trade(quantity, price)
    
    def _add_buy_trade(self, quantity: int, price: float):
        """Add a buy trade."""
        trade_cost = quantity * price
        
        if self.quantity >= 0:
            # Adding to long position or opening long position
            total_cost = (self.quantity * self.avg_cost) + trade_cost
            self.quantity += quantity
            self.avg_cost = total_cost / self.quantity if self.quantity > 0 else 0.0
        else:
            # Covering short position
            if quantity >= abs(self.quantity):
                # Fully covering short and potentially going long
                shares_to_cover = abs(self.quantity)
                cover_proceeds = shares_to_cover * price
                cover_cost = shares_to_cover * self.avg_cost
                
                # Realize P&L from covering short
                self.realized_pnl += cover_cost - cover_proceeds
                
                # Remaining shares go long
                remaining_shares = quantity - shares_to_cover
                self.quantity = remaining_shares
                self.avg_cost = price if remaining_shares > 0 else 0.0
            else:
                # Partially covering short
                cover_proceeds = quantity * price
                cover_cost = quantity * self.avg_cost
                
                # Realize P&L from partial cover
                self.realized_pnl += cover_cost - cover_proceeds
                
                # Update position
                self.quantity += quantity  # quantity is positive, position is negative
        
        # Update totals
        self.total_bought += quantity
        self.total_cost += trade_cost
    
    def _add_sell_trade(self, quantity: int, price: float):
        """Add a sell trade."""
        trade_proceeds = quantity * price
        
        if self.quantity <= 0:
            # Adding to short position or opening short position
            total_cost = (abs(self.quantity) * self.avg_cost) + trade_proceeds
            self.quantity -= quantity
            self.avg_cost = total_cost / abs(self.quantity) if self.quantity != 0 else 0.0
        else:
            # Selling long position
            if quantity >= self.quantity:
                # Fully selling long and potentially going short
                shares_to_sell = self.quantity
                sell_proceeds = shares_to_sell * price
                sell_cost = shares_to_sell * self.avg_cost
                
                # Realize P&L from selling long
                self.realized_pnl += sell_proceeds - sell_cost
                
                # Remaining shares go short
                remaining_shares = quantity - shares_to_sell
                self.quantity = -remaining_shares
                self.avg_cost = price if remaining_shares > 0 else 0.0
            else:
                # Partially selling long
                sell_proceeds = quantity * price
                sell_cost = quantity * self.avg_cost
                
                # Realize P&L from partial sale
                self.realized_pnl += sell_proceeds - sell_cost
                
                # Update position
                self.quantity -= quantity
        
        # Update totals
        self.total_sold += quantity
        self.total_proceeds += trade_proceeds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'last_price': self.last_price,
            'cost_basis': self.cost_basis,
            'total_bought': self.total_bought,
            'total_sold': self.total_sold,
            'total_cost': self.total_cost,
            'total_proceeds': self.total_proceeds,
            'first_trade_date': self.first_trade_date.isoformat() if self.first_trade_date else None,
            'last_trade_date': self.last_trade_date.isoformat() if self.last_trade_date else None
        }


class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        margin_multiplier: float = 1.0,
        interest_rate: float = 0.0
    ):
        """Initialize portfolio.
        
        Args:
            initial_cash: Starting cash amount
            margin_multiplier: Margin multiplier (1.0 = no margin)
            interest_rate: Annual interest rate for margin
        """
        self.initial_cash = initial_cash
        self.margin_multiplier = margin_multiplier
        self.interest_rate = interest_rate
        
        # Portfolio state
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        
        # Portfolio tracking
        self.total_equity_history: List[Tuple[datetime, float]] = []
        self.cash_history: List[Tuple[datetime, float]] = []
        self.position_history: List[Tuple[datetime, Dict[str, Position]]] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        self.max_equity = initial_cash
        self.min_equity = initial_cash
        
        # Risk tracking
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = initial_cash
        
        logger.info(
            "Portfolio initialized",
            initial_cash=initial_cash,
            margin_multiplier=margin_multiplier
        )
    
    @property
    def total_market_value(self) -> float:
        """Get total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_equity(self) -> float:
        """Get total portfolio equity."""
        return self.cash + self.total_market_value
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        """Get total realized P&L."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Get total P&L."""
        return self.total_equity - self.initial_cash
    
    @property
    def total_return(self) -> float:
        """Get total return percentage."""
        return (self.total_equity - self.initial_cash) / self.initial_cash
    
    @property
    def buying_power(self) -> float:
        """Get available buying power."""
        return self.cash * self.margin_multiplier
    
    @property
    def win_rate(self) -> float:
        """Get win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position object
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_market_data(self, symbol: str, bar_data: BarData):
        """Update portfolio with new market data.
        
        Args:
            symbol: Symbol being updated
            bar_data: Current bar data
        """
        # Update position market value if we have a position
        if symbol in self.positions:
            self.positions[symbol].update_market_value(bar_data.close)
        
        # Update portfolio metrics
        self._update_portfolio_metrics(bar_data.timestamp)
    
    def process_fill(self, fill: OrderFill):
        """Process an order fill.
        
        Args:
            fill: Order fill to process
        """
        # Get the original order to determine side
        # For now, we'll infer from the fill data
        # In a real implementation, you'd pass the order or side
        
        # Update cash
        trade_value = fill.quantity * fill.price + fill.commission
        
        # Get position
        position = self.get_position(fill.symbol)
        
        # Determine if this is a buy or sell based on current position
        # This is a simplified approach - in reality you'd track the order side
        if fill.quantity > 0:
            # Assume buy if quantity is positive
            self.cash -= trade_value
            position.add_trade(OrderSide.BUY, fill.quantity, fill.price, fill.timestamp)
        else:
            # Assume sell if quantity is negative
            self.cash += trade_value
            position.add_trade(OrderSide.SELL, abs(fill.quantity), fill.price, fill.timestamp)
        
        # Update commission tracking
        self.total_commission += fill.commission
        
        # Update trade statistics
        self.total_trades += 1
        
        # Update position market value
        position.update_market_value(fill.price)
        
        logger.debug(
            "Fill processed",
            symbol=fill.symbol,
            quantity=fill.quantity,
            price=fill.price,
            commission=fill.commission,
            new_cash=self.cash
        )
    
    def process_order_fill(self, order: Order, fill: OrderFill):
        """Process an order fill with order context.
        
        Args:
            order: Original order
            fill: Order fill
        """
        # Update cash based on order side
        trade_value = fill.quantity * fill.price
        
        if order.side == OrderSide.BUY:
            self.cash -= (trade_value + fill.commission)
        else:
            self.cash += (trade_value - fill.commission)
        
        # Get position and add trade
        position = self.get_position(fill.symbol)
        position.add_trade(order.side, fill.quantity, fill.price, fill.timestamp)
        
        # Update commission tracking
        self.total_commission += fill.commission
        
        # Update trade statistics
        self.total_trades += 1
        
        # Check if this was a winning or losing trade
        if order.side == OrderSide.SELL and position.realized_pnl > 0:
            self.winning_trades += 1
        elif order.side == OrderSide.SELL and position.realized_pnl < 0:
            self.losing_trades += 1
        
        # Update position market value
        position.update_market_value(fill.price)
        
        logger.debug(
            "Order fill processed",
            order_id=order.order_id,
            symbol=fill.symbol,
            side=order.side.value,
            quantity=fill.quantity,
            price=fill.price,
            commission=fill.commission,
            new_cash=self.cash,
            total_equity=self.total_equity
        )
    
    def _update_portfolio_metrics(self, timestamp: datetime):
        """Update portfolio performance metrics.
        
        Args:
            timestamp: Current timestamp
        """
        current_equity = self.total_equity
        
        # Update equity tracking
        self.total_equity_history.append((timestamp, current_equity))
        self.cash_history.append((timestamp, self.cash))
        
        # Update min/max equity
        self.max_equity = max(self.max_equity, current_equity)
        self.min_equity = min(self.min_equity, current_equity)
        
        # Update drawdown tracking
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Store position snapshot (every 100 updates to save memory)
        if len(self.total_equity_history) % 100 == 0:
            position_snapshot = {symbol: pos for symbol, pos in self.positions.items()}
            self.position_history.append((timestamp, position_snapshot))
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary.
        
        Returns:
            Portfolio summary dictionary
        """
        return {
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'total_market_value': self.total_market_value,
            'total_equity': self.total_equity,
            'total_pnl': self.total_pnl,
            'total_return': self.total_return,
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_commission': self.total_commission,
            'buying_power': self.buying_power,
            'max_equity': self.max_equity,
            'min_equity': self.min_equity,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'number_of_positions': len([pos for pos in self.positions.values() if not pos.is_flat])
        }
    
    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all positions.
        
        Returns:
            List of position summaries
        """
        return [
            pos.to_dict() for pos in self.positions.values()
            if not pos.is_flat or pos.realized_pnl != 0
        ]
    
    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Get equity curve data.
        
        Returns:
            List of (timestamp, equity) tuples
        """
        return self.total_equity_history.copy()
    
    def get_cash_curve(self) -> List[Tuple[datetime, float]]:
        """Get cash curve data.
        
        Returns:
            List of (timestamp, cash) tuples
        """
        return self.cash_history.copy()
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(self.total_equity_history) < 2:
            return 0.0
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(self.total_equity_history)):
            prev_equity = self.total_equity_history[i-1][1]
            curr_equity = self.total_equity_history[i][1]
            daily_return = (curr_equity - prev_equity) / prev_equity
            returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        # Calculate excess returns
        daily_risk_free = risk_free_rate / 252  # Assuming 252 trading days
        excess_returns = [r - daily_risk_free for r in returns]
        
        # Calculate Sharpe ratio
        if len(excess_returns) == 0:
            return 0.0
        
        mean_excess_return = sum(excess_returns) / len(excess_returns)
        
        if len(excess_returns) < 2:
            return 0.0
        
        # Calculate standard deviation
        variance = sum((r - mean_excess_return) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        # Annualize
        sharpe_ratio = (mean_excess_return / std_dev) * (252 ** 0.5)
        
        return sharpe_ratio
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        
        # Clear history
        self.total_equity_history.clear()
        self.cash_history.clear()
        self.position_history.clear()
        
        # Reset performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        self.max_equity = self.initial_cash
        self.min_equity = self.initial_cash
        
        # Reset risk tracking
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = self.initial_cash
        
        logger.info("Portfolio reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary.
        
        Returns:
            Portfolio dictionary
        """
        return {
            'summary': self.get_portfolio_summary(),
            'positions': self.get_positions_summary(),
            'equity_curve': [(ts.isoformat(), equity) for ts, equity in self.total_equity_history],
            'cash_curve': [(ts.isoformat(), cash) for ts, cash in self.cash_history]
        }