"""Market interface for live trading with Alpaca API."""

import asyncio
import websockets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
import threading
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce, OrderType as AlpacaOrderType
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar, Trade, Quote

from ..utils import get_logger, retry, rate_limit
from ..data.models import BarData, TradeData, QuoteData
from ..backtesting.order_manager import Order, OrderSide, OrderType, OrderStatus

logger = get_logger(__name__)


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class MarketInterfaceConfig:
    """Configuration for market interface."""
    # API credentials
    api_key: str
    secret_key: str
    paper_trading: bool = True
    
    # Connection settings
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 5.0
    heartbeat_interval: float = 30.0
    
    # Rate limiting
    orders_per_minute: int = 200
    requests_per_minute: int = 200
    
    # Data settings
    enable_live_data: bool = True
    data_buffer_size: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            'paper_trading': self.paper_trading,
            'max_reconnect_attempts': self.max_reconnect_attempts,
            'reconnect_delay': self.reconnect_delay,
            'heartbeat_interval': self.heartbeat_interval,
            'orders_per_minute': self.orders_per_minute,
            'requests_per_minute': self.requests_per_minute,
            'enable_live_data': self.enable_live_data,
            'data_buffer_size': self.data_buffer_size
        }


class AlpacaMarketInterface:
    """Interface for live trading with Alpaca API."""
    
    def __init__(self, config: MarketInterfaceConfig):
        """Initialize market interface.
        
        Args:
            config: Market interface configuration
        """
        self.config = config
        
        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper_trading
        )
        
        self.data_stream = None
        if config.enable_live_data:
            self.data_stream = StockDataStream(
                api_key=config.api_key,
                secret_key=config.secret_key
            )
        
        # Connection state
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.last_heartbeat = datetime.now()
        self.reconnect_attempts = 0
        
        # Subscriptions and callbacks
        self.subscribed_symbols: Set[str] = set()
        self.bar_callbacks: List[Callable[[BarData], None]] = []
        self.trade_callbacks: List[Callable[[TradeData], None]] = []
        self.quote_callbacks: List[Callable[[QuoteData], None]] = []
        self.order_callbacks: List[Callable[[Dict], None]] = []
        
        # Data buffers
        self.bar_buffer: Dict[str, List[BarData]] = {}
        self.trade_buffer: Dict[str, List[TradeData]] = {}
        self.quote_buffer: Dict[str, List[QuoteData]] = {}
        
        # Threading
        self._stop_event = threading.Event()
        self._monitor_thread = None
        
        # Rate limiting decorators
        self._submit_order = rate_limit(
            calls_per_minute=config.orders_per_minute
        )(self._submit_order_impl)
        
        self._make_request = rate_limit(
            calls_per_minute=config.requests_per_minute
        )(self._make_request_impl)
        
        logger.info(
            "AlpacaMarketInterface initialized",
            paper_trading=config.paper_trading,
            enable_live_data=config.enable_live_data
        )
    
    async def connect(self) -> bool:
        """Connect to Alpaca API.
        
        Returns:
            True if connection successful
        """
        try:
            self.connection_status = ConnectionStatus.CONNECTING
            logger.info("Connecting to Alpaca API")
            
            # Test trading client connection
            account = await self._get_account()
            if not account:
                raise Exception("Failed to get account information")
            
            # Connect data stream if enabled
            if self.data_stream:
                await self._connect_data_stream()
            
            self.connection_status = ConnectionStatus.CONNECTED
            self.reconnect_attempts = 0
            self.last_heartbeat = datetime.now()
            
            # Start monitoring thread
            self._start_monitoring()
            
            logger.info("Successfully connected to Alpaca API")
            return True
            
        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            logger.error(f"Failed to connect to Alpaca API: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca API."""
        logger.info("Disconnecting from Alpaca API")
        
        self._stop_event.set()
        
        # Stop data stream
        if self.data_stream:
            try:
                await self.data_stream.stop_ws()
            except Exception as e:
                logger.error(f"Error stopping data stream: {e}")
        
        # Wait for monitoring thread to stop
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.connection_status = ConnectionStatus.DISCONNECTED
        logger.info("Disconnected from Alpaca API")
    
    @retry(max_attempts=3, delay=1.0)
    async def submit_order(self, order: Order) -> Optional[str]:
        """Submit order to Alpaca.
        
        Args:
            order: Order to submit
            
        Returns:
            Alpaca order ID if successful
        """
        try:
            # Convert to Alpaca order request
            alpaca_request = self._convert_to_alpaca_order(order)
            
            # Submit order
            alpaca_order = await self._submit_order(alpaca_request)
            
            if alpaca_order:
                logger.info(
                    "Order submitted successfully",
                    order_id=order.order_id,
                    alpaca_id=alpaca_order.id,
                    symbol=order.symbol,
                    side=order.side.value,
                    quantity=order.quantity
                )
                return str(alpaca_order.id)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to submit order {order.order_id}: {e}")
            return None
    
    @retry(max_attempts=3, delay=1.0)
    async def cancel_order(self, alpaca_order_id: str) -> bool:
        """Cancel order by Alpaca order ID.
        
        Args:
            alpaca_order_id: Alpaca order ID
            
        Returns:
            True if cancellation successful
        """
        try:
            await self._make_request(
                lambda: self.trading_client.cancel_order_by_id(alpaca_order_id)
            )
            
            logger.info(f"Order cancelled successfully: {alpaca_order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {alpaca_order_id}: {e}")
            return False
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            positions = await self._make_request(
                lambda: self.trading_client.get_all_positions()
            )
            
            return [
                {
                    'symbol': pos.symbol,
                    'quantity': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pnl': float(pos.unrealized_pnl),
                    'side': 'long' if float(pos.qty) > 0 else 'short'
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_account(self) -> Optional[Dict[str, Any]]:
        """Get account information.
        
        Returns:
            Account information dictionary
        """
        try:
            account = await self._get_account()
            
            if account:
                return {
                    'equity': float(account.equity),
                    'cash': float(account.cash),
                    'buying_power': float(account.buying_power),
                    'portfolio_value': float(account.portfolio_value),
                    'day_trade_count': int(account.daytrade_count),
                    'pattern_day_trader': account.pattern_day_trader
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None
    
    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get orders.
        
        Args:
            status: Order status filter
            
        Returns:
            List of order dictionaries
        """
        try:
            orders = await self._make_request(
                lambda: self.trading_client.get_orders()
            )
            
            result = []
            for order in orders:
                if status and order.status != status:
                    continue
                
                result.append({
                    'id': str(order.id),
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'order_type': order.order_type.value,
                    'quantity': float(order.qty),
                    'filled_quantity': float(order.filled_qty or 0),
                    'status': order.status.value,
                    'created_at': order.created_at,
                    'filled_at': order.filled_at,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def subscribe_bars(self, symbols: List[str]):
        """Subscribe to bar data.
        
        Args:
            symbols: List of symbols to subscribe to
        """
        if not self.data_stream:
            logger.warning("Data stream not enabled")
            return
        
        try:
            for symbol in symbols:
                self.data_stream.subscribe_bars(self._on_bar, symbol)
                self.subscribed_symbols.add(symbol)
                
                # Initialize buffer
                if symbol not in self.bar_buffer:
                    self.bar_buffer[symbol] = []
            
            logger.info(f"Subscribed to bars for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to bars: {e}")
    
    def subscribe_trades(self, symbols: List[str]):
        """Subscribe to trade data.
        
        Args:
            symbols: List of symbols to subscribe to
        """
        if not self.data_stream:
            logger.warning("Data stream not enabled")
            return
        
        try:
            for symbol in symbols:
                self.data_stream.subscribe_trades(self._on_trade, symbol)
                self.subscribed_symbols.add(symbol)
                
                # Initialize buffer
                if symbol not in self.trade_buffer:
                    self.trade_buffer[symbol] = []
            
            logger.info(f"Subscribed to trades for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to trades: {e}")
    
    def subscribe_quotes(self, symbols: List[str]):
        """Subscribe to quote data.
        
        Args:
            symbols: List of symbols to subscribe to
        """
        if not self.data_stream:
            logger.warning("Data stream not enabled")
            return
        
        try:
            for symbol in symbols:
                self.data_stream.subscribe_quotes(self._on_quote, symbol)
                self.subscribed_symbols.add(symbol)
                
                # Initialize buffer
                if symbol not in self.quote_buffer:
                    self.quote_buffer[symbol] = []
            
            logger.info(f"Subscribed to quotes for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to quotes: {e}")
    
    def add_bar_callback(self, callback: Callable[[BarData], None]):
        """Add callback for bar data.
        
        Args:
            callback: Function to call with bar data
        """
        self.bar_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[TradeData], None]):
        """Add callback for trade data.
        
        Args:
            callback: Function to call with trade data
        """
        self.trade_callbacks.append(callback)
    
    def add_quote_callback(self, callback: Callable[[QuoteData], None]):
        """Add callback for quote data.
        
        Args:
            callback: Function to call with quote data
        """
        self.quote_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable[[Dict], None]):
        """Add callback for order updates.
        
        Args:
            callback: Function to call with order update
        """
        self.order_callbacks.append(callback)
    
    def get_latest_bars(self, symbol: str, count: int = 1) -> List[BarData]:
        """Get latest bars from buffer.
        
        Args:
            symbol: Trading symbol
            count: Number of bars to return
            
        Returns:
            List of latest bars
        """
        if symbol not in self.bar_buffer:
            return []
        
        return self.bar_buffer[symbol][-count:]
    
    def get_latest_trades(self, symbol: str, count: int = 1) -> List[TradeData]:
        """Get latest trades from buffer.
        
        Args:
            symbol: Trading symbol
            count: Number of trades to return
            
        Returns:
            List of latest trades
        """
        if symbol not in self.trade_buffer:
            return []
        
        return self.trade_buffer[symbol][-count:]
    
    def get_latest_quotes(self, symbol: str, count: int = 1) -> List[QuoteData]:
        """Get latest quotes from buffer.
        
        Args:
            symbol: Trading symbol
            count: Number of quotes to return
            
        Returns:
            List of latest quotes
        """
        if symbol not in self.quote_buffer:
            return []
        
        return self.quote_buffer[symbol][-count:]
    
    def is_connected(self) -> bool:
        """Check if connected to Alpaca API.
        
        Returns:
            True if connected
        """
        return self.connection_status == ConnectionStatus.CONNECTED
    
    def get_connection_status(self) -> ConnectionStatus:
        """Get current connection status.
        
        Returns:
            Connection status
        """
        return self.connection_status
    
    async def _get_account(self):
        """Get account information."""
        return await self._make_request(
            lambda: self.trading_client.get_account()
        )
    
    async def _connect_data_stream(self):
        """Connect to data stream."""
        try:
            await self.data_stream.run()
            logger.info("Data stream connected")
        except Exception as e:
            logger.error(f"Failed to connect data stream: {e}")
            raise
    
    def _convert_to_alpaca_order(self, order: Order):
        """Convert internal order to Alpaca order request."""
        # Convert side
        if order.side == OrderSide.BUY:
            alpaca_side = AlpacaOrderSide.BUY
        else:
            alpaca_side = AlpacaOrderSide.SELL
        
        # Convert order type and create request
        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY
            )
        elif order.order_type == OrderType.LIMIT:
            return LimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price
            )
        elif order.order_type == OrderType.STOP:
            return StopOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                stop_price=order.stop_price
            )
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")
    
    async def _submit_order_impl(self, order_request):
        """Submit order implementation."""
        return await self._make_request(
            lambda: self.trading_client.submit_order(order_request)
        )
    
    async def _make_request_impl(self, request_func):
        """Make API request implementation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, request_func)
    
    def _on_bar(self, bar: Bar):
        """Handle incoming bar data."""
        try:
            bar_data = BarData(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume)
            )
            
            # Add to buffer
            if bar.symbol in self.bar_buffer:
                self.bar_buffer[bar.symbol].append(bar_data)
                
                # Limit buffer size
                if len(self.bar_buffer[bar.symbol]) > self.config.data_buffer_size:
                    self.bar_buffer[bar.symbol] = self.bar_buffer[bar.symbol][-self.config.data_buffer_size:]
            
            # Call callbacks
            for callback in self.bar_callbacks:
                try:
                    callback(bar_data)
                except Exception as e:
                    logger.error(f"Bar callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing bar data: {e}")
    
    def _on_trade(self, trade: Trade):
        """Handle incoming trade data."""
        try:
            trade_data = TradeData(
                symbol=trade.symbol,
                timestamp=trade.timestamp,
                price=float(trade.price),
                size=int(trade.size)
            )
            
            # Add to buffer
            if trade.symbol in self.trade_buffer:
                self.trade_buffer[trade.symbol].append(trade_data)
                
                # Limit buffer size
                if len(self.trade_buffer[trade.symbol]) > self.config.data_buffer_size:
                    self.trade_buffer[trade.symbol] = self.trade_buffer[trade.symbol][-self.config.data_buffer_size:]
            
            # Call callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(trade_data)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    def _on_quote(self, quote: Quote):
        """Handle incoming quote data."""
        try:
            quote_data = QuoteData(
                symbol=quote.symbol,
                timestamp=quote.timestamp,
                bid_price=float(quote.bid_price),
                bid_size=int(quote.bid_size),
                ask_price=float(quote.ask_price),
                ask_size=int(quote.ask_size)
            )
            
            # Add to buffer
            if quote.symbol in self.quote_buffer:
                self.quote_buffer[quote.symbol].append(quote_data)
                
                # Limit buffer size
                if len(self.quote_buffer[quote.symbol]) > self.config.data_buffer_size:
                    self.quote_buffer[quote.symbol] = self.quote_buffer[quote.symbol][-self.config.data_buffer_size:]
            
            # Call callbacks
            for callback in self.quote_callbacks:
                try:
                    callback(quote_data)
                except Exception as e:
                    logger.error(f"Quote callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing quote data: {e}")
    
    def _start_monitoring(self):
        """Start connection monitoring thread."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_connection,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _monitor_connection(self):
        """Monitor connection and handle reconnection."""
        while not self._stop_event.is_set():
            try:
                # Check heartbeat
                if (datetime.now() - self.last_heartbeat).total_seconds() > self.config.heartbeat_interval * 2:
                    logger.warning("Heartbeat timeout detected")
                    asyncio.run(self._handle_reconnection())
                
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
                time.sleep(5.0)
    
    async def _handle_reconnection(self):
        """Handle reconnection logic."""
        if self.reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self.connection_status = ConnectionStatus.ERROR
            return
        
        self.connection_status = ConnectionStatus.RECONNECTING
        self.reconnect_attempts += 1
        
        logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.config.max_reconnect_attempts}")
        
        try:
            await asyncio.sleep(self.config.reconnect_delay)
            
            # Test connection
            account = await self._get_account()
            if account:
                self.connection_status = ConnectionStatus.CONNECTED
                self.last_heartbeat = datetime.now()
                logger.info("Reconnection successful")
            else:
                raise Exception("Failed to verify connection")
                
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            # Will retry on next monitoring cycle