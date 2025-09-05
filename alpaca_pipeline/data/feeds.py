"""Data feeds module for real-time and historical data streaming."""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any, AsyncGenerator
from queue import Queue, Empty
from threading import Thread, Event
import websocket
import pandas as pd
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL

from ..config import get_config
from ..utils import get_logger, retry, rate_limit
from .models import BarData, TradeData, QuoteData
from .collector import DataCollector

logger = get_logger(__name__)


class BaseDataFeed:
    """Base class for data feeds."""
    
    def __init__(self):
        """Initialize base data feed."""
        self.config = get_config()
        self.is_running = False
        self.subscribers = []
        
    def subscribe(self, callback: Callable):
        """Subscribe to data updates.
        
        Args:
            callback: Function to call when new data arrives
        """
        self.subscribers.append(callback)
        logger.debug("New subscriber added", total_subscribers=len(self.subscribers))
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from data updates.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.debug("Subscriber removed", total_subscribers=len(self.subscribers))
    
    def _notify_subscribers(self, data: Any):
        """Notify all subscribers of new data.
        
        Args:
            data: Data to send to subscribers
        """
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                logger.error(
                    "Error notifying subscriber",
                    error=str(e),
                    callback=callback.__name__ if hasattr(callback, '__name__') else str(callback)
                )
    
    def start(self):
        """Start the data feed."""
        raise NotImplementedError
    
    def stop(self):
        """Stop the data feed."""
        raise NotImplementedError


class HistoricalDataFeed(BaseDataFeed):
    """Historical data feed for backtesting."""
    
    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1Min',
        data_collector: Optional[DataCollector] = None
    ):
        """Initialize historical data feed.
        
        Args:
            symbols: List of symbols to stream
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Data timeframe
            data_collector: DataCollector instance
        """
        super().__init__()
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.data_collector = data_collector or DataCollector()
        
        self.data_queue = Queue()
        self.current_time = start_date
        self.playback_speed = 1.0  # 1.0 = real-time, 0 = as fast as possible
        self.thread = None
        self.stop_event = Event()
        
        logger.info(
            "HistoricalDataFeed initialized",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
    
    def set_playback_speed(self, speed: float):
        """Set playback speed.
        
        Args:
            speed: Playback speed (0 = as fast as possible, 1.0 = real-time)
        """
        self.playback_speed = speed
        logger.info("Playback speed set", speed=speed)
    
    def start(self):
        """Start historical data streaming."""
        if self.is_running:
            logger.warning("Historical data feed is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Load historical data
        self._load_historical_data()
        
        # Start streaming thread
        self.thread = Thread(target=self._stream_data, daemon=True)
        self.thread.start()
        
        logger.info("Historical data feed started")
    
    def stop(self):
        """Stop historical data streaming."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        logger.info("Historical data feed stopped")
    
    def _load_historical_data(self):
        """Load historical data for all symbols."""
        logger.info("Loading historical data")
        
        all_data = []
        
        for symbol in self.symbols:
            try:
                bars = self.data_collector.get_historical_bars(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    start=self.start_date,
                    end=self.end_date
                )
                
                for bar in bars:
                    all_data.append((bar.timestamp, symbol, bar))
                
                logger.debug(
                    "Loaded historical data for symbol",
                    symbol=symbol,
                    bars_count=len(bars)
                )
                
            except Exception as e:
                logger.error(
                    "Failed to load historical data",
                    symbol=symbol,
                    error=str(e)
                )
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x[0])
        
        # Add to queue
        for timestamp, symbol, bar in all_data:
            self.data_queue.put((timestamp, symbol, bar))
        
        logger.info(
            "Historical data loaded",
            total_bars=len(all_data),
            symbols=self.symbols
        )
    
    def _stream_data(self):
        """Stream historical data with timing control."""
        last_timestamp = None
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Get next data point
                timestamp, symbol, bar = self.data_queue.get(timeout=1.0)
                
                # Update current time
                self.current_time = timestamp
                
                # Calculate sleep time for playback speed
                if self.playback_speed > 0 and last_timestamp:
                    time_diff = (timestamp - last_timestamp).total_seconds()
                    sleep_time = time_diff / self.playback_speed
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 1.0))  # Cap at 1 second
                
                # Notify subscribers
                self._notify_subscribers({
                    'type': 'bar',
                    'symbol': symbol,
                    'data': bar,
                    'timestamp': timestamp
                })
                
                last_timestamp = timestamp
                
            except Empty:
                # No more data
                logger.info("Historical data streaming completed")
                break
            except Exception as e:
                logger.error("Error in historical data streaming", error=str(e))
                break
        
        self.is_running = False
    
    def get_current_time(self) -> datetime:
        """Get current simulation time.
        
        Returns:
            Current simulation timestamp
        """
        return self.current_time
    
    def seek_to_time(self, target_time: datetime):
        """Seek to specific time in historical data.
        
        Args:
            target_time: Target timestamp to seek to
        """
        if target_time < self.start_date or target_time > self.end_date:
            raise ValueError(f"Target time {target_time} is outside data range")
        
        # Stop current streaming
        was_running = self.is_running
        if was_running:
            self.stop()
        
        # Clear queue and reload data from target time
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except Empty:
                break
        
        # Update start date and reload
        original_start = self.start_date
        self.start_date = target_time
        self.current_time = target_time
        
        self._load_historical_data()
        
        # Restore original start date
        self.start_date = original_start
        
        # Restart if it was running
        if was_running:
            self.start()
        
        logger.info("Seeked to time", target_time=target_time)


class RealTimeDataFeed(BaseDataFeed):
    """Real-time data feed using Alpaca's streaming API."""
    
    def __init__(self, symbols: List[str]):
        """Initialize real-time data feed.
        
        Args:
            symbols: List of symbols to stream
        """
        super().__init__()
        self.symbols = symbols
        self.stream = None
        self.connection_retry_count = 0
        self.max_retries = 5
        
        logger.info("RealTimeDataFeed initialized", symbols=symbols)
    
    def start(self):
        """Start real-time data streaming."""
        if self.is_running:
            logger.warning("Real-time data feed is already running")
            return
        
        self.is_running = True
        self._setup_stream()
        
        logger.info("Real-time data feed started")
    
    def stop(self):
        """Stop real-time data streaming."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.stream:
            try:
                self.stream.stop()
            except Exception as e:
                logger.error("Error stopping stream", error=str(e))
        
        logger.info("Real-time data feed stopped")
    
    def _setup_stream(self):
        """Setup Alpaca stream connection."""
        try:
            self.stream = Stream(
                key_id=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
                base_url=URL(self.config.alpaca_base_url),
                data_feed='iex'  # or 'sip' for paid data
            )
            
            # Subscribe to bars
            self.stream.subscribe_bars(
                self._on_bar,
                *self.symbols
            )
            
            # Subscribe to trades
            self.stream.subscribe_trades(
                self._on_trade,
                *self.symbols
            )
            
            # Subscribe to quotes
            self.stream.subscribe_quotes(
                self._on_quote,
                *self.symbols
            )
            
            # Start the stream
            self.stream.run()
            
        except Exception as e:
            logger.error("Failed to setup stream", error=str(e))
            self._handle_connection_error()
    
    def _on_bar(self, bar):
        """Handle incoming bar data.
        
        Args:
            bar: Bar data from Alpaca
        """
        try:
            bar_data = BarData(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
                trade_count=getattr(bar, 'trade_count', None),
                vwap=getattr(bar, 'vwap', None)
            )
            
            self._notify_subscribers({
                'type': 'bar',
                'symbol': bar.symbol,
                'data': bar_data,
                'timestamp': bar.timestamp
            })
            
            logger.debug("Bar received", symbol=bar.symbol, timestamp=bar.timestamp)
            
        except Exception as e:
            logger.error("Error processing bar data", error=str(e), symbol=bar.symbol)
    
    def _on_trade(self, trade):
        """Handle incoming trade data.
        
        Args:
            trade: Trade data from Alpaca
        """
        try:
            trade_data = TradeData(
                symbol=trade.symbol,
                timestamp=trade.timestamp,
                price=float(trade.price),
                size=int(trade.size),
                conditions=getattr(trade, 'conditions', []),
                exchange=getattr(trade, 'exchange', None)
            )
            
            self._notify_subscribers({
                'type': 'trade',
                'symbol': trade.symbol,
                'data': trade_data,
                'timestamp': trade.timestamp
            })
            
            logger.debug("Trade received", symbol=trade.symbol, timestamp=trade.timestamp)
            
        except Exception as e:
            logger.error("Error processing trade data", error=str(e), symbol=trade.symbol)
    
    def _on_quote(self, quote):
        """Handle incoming quote data.
        
        Args:
            quote: Quote data from Alpaca
        """
        try:
            quote_data = QuoteData(
                symbol=quote.symbol,
                timestamp=quote.timestamp,
                bid_price=float(quote.bid_price),
                bid_size=int(quote.bid_size),
                ask_price=float(quote.ask_price),
                ask_size=int(quote.ask_size),
                bid_exchange=getattr(quote, 'bid_exchange', None),
                ask_exchange=getattr(quote, 'ask_exchange', None)
            )
            
            self._notify_subscribers({
                'type': 'quote',
                'symbol': quote.symbol,
                'data': quote_data,
                'timestamp': quote.timestamp
            })
            
            logger.debug("Quote received", symbol=quote.symbol, timestamp=quote.timestamp)
            
        except Exception as e:
            logger.error("Error processing quote data", error=str(e), symbol=quote.symbol)
    
    def _handle_connection_error(self):
        """Handle connection errors with retry logic."""
        self.connection_retry_count += 1
        
        if self.connection_retry_count <= self.max_retries:
            retry_delay = min(2 ** self.connection_retry_count, 60)  # Exponential backoff
            
            logger.warning(
                "Connection error, retrying",
                retry_count=self.connection_retry_count,
                retry_delay=retry_delay
            )
            
            time.sleep(retry_delay)
            
            if self.is_running:
                self._setup_stream()
        else:
            logger.error(
                "Max retries exceeded, stopping real-time feed",
                max_retries=self.max_retries
            )
            self.stop()
    
    def add_symbols(self, symbols: List[str]):
        """Add symbols to the stream.
        
        Args:
            symbols: List of symbols to add
        """
        new_symbols = [s for s in symbols if s not in self.symbols]
        
        if not new_symbols:
            return
        
        self.symbols.extend(new_symbols)
        
        if self.stream and self.is_running:
            try:
                self.stream.subscribe_bars(self._on_bar, *new_symbols)
                self.stream.subscribe_trades(self._on_trade, *new_symbols)
                self.stream.subscribe_quotes(self._on_quote, *new_symbols)
                
                logger.info("Added symbols to stream", symbols=new_symbols)
                
            except Exception as e:
                logger.error("Failed to add symbols to stream", error=str(e))
    
    def remove_symbols(self, symbols: List[str]):
        """Remove symbols from the stream.
        
        Args:
            symbols: List of symbols to remove
        """
        for symbol in symbols:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
        
        if self.stream and self.is_running:
            try:
                self.stream.unsubscribe_bars(*symbols)
                self.stream.unsubscribe_trades(*symbols)
                self.stream.unsubscribe_quotes(*symbols)
                
                logger.info("Removed symbols from stream", symbols=symbols)
                
            except Exception as e:
                logger.error("Failed to remove symbols from stream", error=str(e))


class DataFeedManager:
    """Manager for coordinating multiple data feeds."""
    
    def __init__(self):
        """Initialize data feed manager."""
        self.feeds = {}
        self.subscribers = []
        
        logger.info("DataFeedManager initialized")
    
    def add_feed(self, name: str, feed: BaseDataFeed):
        """Add a data feed.
        
        Args:
            name: Name of the feed
            feed: Data feed instance
        """
        self.feeds[name] = feed
        feed.subscribe(self._on_data)
        
        logger.info("Data feed added", name=name, type=type(feed).__name__)
    
    def remove_feed(self, name: str):
        """Remove a data feed.
        
        Args:
            name: Name of the feed to remove
        """
        if name in self.feeds:
            feed = self.feeds[name]
            feed.unsubscribe(self._on_data)
            feed.stop()
            del self.feeds[name]
            
            logger.info("Data feed removed", name=name)
    
    def start_all(self):
        """Start all data feeds."""
        for name, feed in self.feeds.items():
            try:
                feed.start()
                logger.info("Started data feed", name=name)
            except Exception as e:
                logger.error("Failed to start data feed", name=name, error=str(e))
    
    def stop_all(self):
        """Stop all data feeds."""
        for name, feed in self.feeds.items():
            try:
                feed.stop()
                logger.info("Stopped data feed", name=name)
            except Exception as e:
                logger.error("Failed to stop data feed", name=name, error=str(e))
    
    def subscribe(self, callback: Callable):
        """Subscribe to data from all feeds.
        
        Args:
            callback: Function to call when new data arrives
        """
        self.subscribers.append(callback)
        logger.debug("New subscriber added to manager", total_subscribers=len(self.subscribers))
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from data feeds.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.debug("Subscriber removed from manager", total_subscribers=len(self.subscribers))
    
    def _on_data(self, data: Dict[str, Any]):
        """Handle data from feeds and forward to subscribers.
        
        Args:
            data: Data from feed
        """
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                logger.error(
                    "Error notifying manager subscriber",
                    error=str(e),
                    callback=callback.__name__ if hasattr(callback, '__name__') else str(callback)
                )
    
    def get_feed(self, name: str) -> Optional[BaseDataFeed]:
        """Get a data feed by name.
        
        Args:
            name: Name of the feed
            
        Returns:
            Data feed instance or None
        """
        return self.feeds.get(name)
    
    def list_feeds(self) -> List[str]:
        """List all data feed names.
        
        Returns:
            List of feed names
        """
        return list(self.feeds.keys())
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all data feeds.
        
        Returns:
            Dictionary mapping feed names to running status
        """
        return {name: feed.is_running for name, feed in self.feeds.items()}