"""Order management system for backtesting."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..utils import get_logger
from ..data.models import BarData

logger = get_logger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class OrderFill:
    """Represents an order fill."""
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    symbol: str = ""
    quantity: int = 0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission
        }


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    commission: float = 0.0
    fills: List[OrderFill] = field(default_factory=list)
    
    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity
    
    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return 0 < self.filled_quantity < self.quantity
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity
    
    def add_fill(self, fill: OrderFill):
        """Add a fill to the order."""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.commission += fill.commission
        
        # Update average fill price
        total_value = sum(f.quantity * f.price for f in self.fills)
        total_quantity = sum(f.quantity for f in self.fills)
        self.avg_fill_price = total_value / total_quantity if total_quantity > 0 else 0.0
        
        # Update status
        if self.is_filled:
            self.status = OrderStatus.FILLED
            self.filled_at = fill.timestamp
        elif self.is_partially_filled:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = fill.timestamp
    
    def cancel(self):
        """Cancel the order."""
        if self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
            self.status = OrderStatus.CANCELLED
            self.updated_at = datetime.now()
    
    def reject(self, reason: str = ""):
        """Reject the order."""
        self.status = OrderStatus.REJECTED
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'strategy_id': self.strategy_id,
            'commission': self.commission,
            'fills': [fill.to_dict() for fill in self.fills]
        }


class OrderManager:
    """Manages order execution during backtesting."""
    
    def __init__(
        self,
        commission_per_share: float = 0.0,
        commission_percentage: float = 0.0,
        min_commission: float = 0.0,
        slippage_percentage: float = 0.001,  # 0.1% default slippage
        market_impact_model: Optional[str] = None
    ):
        """Initialize order manager.
        
        Args:
            commission_per_share: Commission per share
            commission_percentage: Commission as percentage of trade value
            min_commission: Minimum commission per trade
            slippage_percentage: Market slippage as percentage
            market_impact_model: Market impact model (linear, sqrt, etc.)
        """
        self.commission_per_share = commission_per_share
        self.commission_percentage = commission_percentage
        self.min_commission = min_commission
        self.slippage_percentage = slippage_percentage
        self.market_impact_model = market_impact_model
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.cancelled_orders: Dict[str, Order] = {}
        
        # Statistics
        self.total_orders = 0
        self.total_fills = 0
        self.total_commission = 0.0
        
        logger.info(
            "OrderManager initialized",
            commission_per_share=commission_per_share,
            commission_percentage=commission_percentage,
            slippage_percentage=slippage_percentage
        )
    
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID
        """
        # Validate order
        if not self._validate_order(order):
            order.reject("Order validation failed")
            return order.order_id
        
        # Add to tracking
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        self.total_orders += 1
        
        logger.debug(
            "Order submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value
        )
        
        return order.order_id
    
    def process_market_data(self, symbol: str, bar_data: BarData):
        """Process market data and execute pending orders.
        
        Args:
            symbol: Symbol being updated
            bar_data: Current bar data
        """
        # Get pending orders for this symbol
        symbol_orders = [
            order for order in self.pending_orders.values()
            if order.symbol == symbol
        ]
        
        for order in symbol_orders:
            self._try_execute_order(order, bar_data)
    
    def _try_execute_order(self, order: Order, bar_data: BarData):
        """Try to execute an order against market data.
        
        Args:
            order: Order to execute
            bar_data: Current market data
        """
        execution_price = None
        
        # Determine execution price based on order type
        if order.order_type == OrderType.MARKET:
            # Market orders execute at current price with slippage
            if order.side == OrderSide.BUY:
                execution_price = bar_data.close * (1 + self.slippage_percentage)
            else:
                execution_price = bar_data.close * (1 - self.slippage_percentage)
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders execute if price is favorable
            if order.price is None:
                logger.error("Limit order missing price", order_id=order.order_id)
                return
            
            if order.side == OrderSide.BUY:
                # Buy limit: execute if market price <= limit price
                if bar_data.low <= order.price:
                    execution_price = min(order.price, bar_data.open)
            else:
                # Sell limit: execute if market price >= limit price
                if bar_data.high >= order.price:
                    execution_price = max(order.price, bar_data.open)
        
        elif order.order_type == OrderType.STOP:
            # Stop orders become market orders when triggered
            if order.stop_price is None:
                logger.error("Stop order missing stop price", order_id=order.order_id)
                return
            
            triggered = False
            if order.side == OrderSide.BUY:
                # Buy stop: trigger if price >= stop price
                triggered = bar_data.high >= order.stop_price
            else:
                # Sell stop: trigger if price <= stop price
                triggered = bar_data.low <= order.stop_price
            
            if triggered:
                # Execute as market order with slippage
                if order.side == OrderSide.BUY:
                    execution_price = bar_data.close * (1 + self.slippage_percentage)
                else:
                    execution_price = bar_data.close * (1 - self.slippage_percentage)
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop-limit orders become limit orders when triggered
            if order.stop_price is None or order.price is None:
                logger.error(
                    "Stop-limit order missing prices",
                    order_id=order.order_id
                )
                return
            
            triggered = False
            if order.side == OrderSide.BUY:
                triggered = bar_data.high >= order.stop_price
                if triggered and bar_data.low <= order.price:
                    execution_price = min(order.price, bar_data.open)
            else:
                triggered = bar_data.low <= order.stop_price
                if triggered and bar_data.high >= order.price:
                    execution_price = max(order.price, bar_data.open)
        
        # Execute the order if we have an execution price
        if execution_price is not None:
            self._execute_order(order, execution_price, bar_data.timestamp)
    
    def _execute_order(self, order: Order, price: float, timestamp: datetime):
        """Execute an order at the given price.
        
        Args:
            order: Order to execute
            price: Execution price
            timestamp: Execution timestamp
        """
        # Calculate commission
        commission = self._calculate_commission(order.quantity, price)
        
        # Create fill
        fill = OrderFill(
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=order.quantity,
            price=price,
            timestamp=timestamp,
            commission=commission
        )
        
        # Add fill to order
        order.add_fill(fill)
        
        # Update tracking
        if order.order_id in self.pending_orders:
            del self.pending_orders[order.order_id]
        
        self.filled_orders[order.order_id] = order
        self.total_fills += 1
        self.total_commission += commission
        
        logger.debug(
            "Order executed",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=order.quantity,
            price=price,
            commission=commission
        )
    
    def _calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate commission for a trade.
        
        Args:
            quantity: Number of shares
            price: Price per share
            
        Returns:
            Commission amount
        """
        trade_value = quantity * price
        
        # Calculate commission components
        per_share_commission = quantity * self.commission_per_share
        percentage_commission = trade_value * self.commission_percentage
        
        # Total commission is the sum of both
        total_commission = per_share_commission + percentage_commission
        
        # Apply minimum commission
        return max(total_commission, self.min_commission)
    
    def _validate_order(self, order: Order) -> bool:
        """Validate an order before submission.
        
        Args:
            order: Order to validate
            
        Returns:
            True if order is valid
        """
        # Basic validation
        if not order.symbol:
            logger.error("Order missing symbol", order_id=order.order_id)
            return False
        
        if order.quantity <= 0:
            logger.error("Invalid quantity", order_id=order.order_id, quantity=order.quantity)
            return False
        
        # Order type specific validation
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                logger.error("Invalid limit price", order_id=order.order_id)
                return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                logger.error("Invalid stop price", order_id=order.order_id)
                return False
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was cancelled
        """
        if order_id not in self.pending_orders:
            logger.warning("Order not found or not pending", order_id=order_id)
            return False
        
        order = self.pending_orders[order_id]
        order.cancel()
        
        # Move to cancelled orders
        del self.pending_orders[order_id]
        self.cancelled_orders[order_id] = order
        
        logger.debug("Order cancelled", order_id=order_id)
        return True
    
    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all pending orders, optionally for a specific symbol.
        
        Args:
            symbol: Optional symbol to filter by
        """
        orders_to_cancel = list(self.pending_orders.keys())
        
        for order_id in orders_to_cancel:
            order = self.pending_orders[order_id]
            if symbol is None or order.symbol == symbol:
                self.cancel_order(order_id)
        
        logger.info(
            "Cancelled orders",
            symbol=symbol,
            count=len(orders_to_cancel)
        )
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order if found, None otherwise
        """
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol.
        
        Args:
            symbol: Symbol to filter by
            
        Returns:
            List of orders
        """
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get pending orders.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of pending orders
        """
        orders = list(self.pending_orders.values())
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        return orders
    
    def get_filled_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get filled orders.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of filled orders
        """
        orders = list(self.filled_orders.values())
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        return orders
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_orders': self.total_orders,
            'pending_orders': len(self.pending_orders),
            'filled_orders': len(self.filled_orders),
            'cancelled_orders': len(self.cancelled_orders),
            'total_fills': self.total_fills,
            'total_commission': self.total_commission,
            'fill_rate': self.total_fills / self.total_orders if self.total_orders > 0 else 0.0
        }
    
    def reset(self):
        """Reset the order manager."""
        self.orders.clear()
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.cancelled_orders.clear()
        
        self.total_orders = 0
        self.total_fills = 0
        self.total_commission = 0.0
        
        logger.info("OrderManager reset")