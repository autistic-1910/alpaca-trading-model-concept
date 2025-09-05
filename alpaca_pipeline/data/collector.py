"""Data collection module for Alpaca API integration."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Callable, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

from ..config import get_config
from ..utils import get_logger, retry, rate_limit, timing
from .models import BarData, TradeData, QuoteData, AssetInfo, MarketStatus
from .storage import DataStorage

logger = get_logger(__name__)


class DataCollector:
    """Main data collection class for Alpaca API integration."""
    
    def __init__(self, storage: Optional[DataStorage] = None):
        """Initialize the data collector.
        
        Args:
            storage: Optional data storage instance
        """
        self.config = get_config()
        self.storage = storage or DataStorage()
        
        # Initialize Alpaca API client
        self.api = tradeapi.REST(
            key_id=self.config.alpaca.api_key,
            secret_key=self.config.alpaca.secret_key,
            base_url=self.config.alpaca.base_url,
            api_version='v2'
        )
        
        # Timeframe mapping
        self.timeframe_map = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute),
            '30Min': TimeFrame(30, TimeFrameUnit.Minute),
            '1Hour': TimeFrame.Hour,
            '2Hour': TimeFrame(2, TimeFrameUnit.Hour),
            '4Hour': TimeFrame(4, TimeFrameUnit.Hour),
            '1Day': TimeFrame.Day,
            '1Week': TimeFrame.Week,
            '1Month': TimeFrame.Month,
        }
        
        self._executor = ThreadPoolExecutor(max_workers=4)  # Default max_workers
        
        logger.info("DataCollector initialized", base_url=self.config.alpaca.base_url)
    
    @retry(max_attempts=3, delay=1.0)
    @rate_limit(max_calls=200, time_window=60)
    @timing()
    def get_historical_bars(
        self,
        symbols: Union[str, List[str]],
        timeframe: str = '1Day',
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        limit: Optional[int] = None,
        adjustment: str = 'raw'
    ) -> Dict[str, List[BarData]]:
        """Get historical bar data for symbols.
        
        Args:
            symbols: Symbol or list of symbols
            timeframe: Data timeframe (e.g., '1Min', '1Hour', '1Day')
            start: Start date/datetime
            end: End date/datetime
            limit: Maximum number of bars
            adjustment: Price adjustment type ('raw', 'split', 'dividend', 'all')
            
        Returns:
            Dictionary mapping symbols to lists of BarData
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Convert timeframe
        if timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        tf = self.timeframe_map[timeframe]
        
        # Set default date range if not provided
        if start is None:
            start = datetime.now() - timedelta(days=365)
        if end is None:
            end = datetime.now()
        
        # Convert string dates to datetime
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        
        logger.info(
            "Fetching historical bars",
            symbols=symbols,
            timeframe=timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=limit
        )
        
        result = {}
        
        try:
            # Fetch data from Alpaca API
            bars_response = self.api.get_bars(
                symbols,
                tf,
                start=start,
                end=end,
                limit=limit,
                adjustment=adjustment
            )
            
            # Convert to BarData objects
            for symbol in symbols:
                symbol_bars = []
                
                if symbol in bars_response:
                    for bar in bars_response[symbol]:
                        bar_data = BarData(
                            symbol=symbol,
                            timestamp=bar.timestamp.replace(tzinfo=None),
                            open=float(bar.open),
                            high=float(bar.high),
                            low=float(bar.low),
                            close=float(bar.close),
                            volume=int(bar.volume),
                            trade_count=getattr(bar, 'trade_count', None),
                            vwap=getattr(bar, 'vwap', None)
                        )
                        symbol_bars.append(bar_data)
                
                result[symbol] = symbol_bars
                
                logger.debug(
                    "Fetched bars for symbol",
                    symbol=symbol,
                    count=len(symbol_bars)
                )
            
            # Store data if storage is available
            if self.storage:
                for symbol, bars in result.items():
                    if bars:
                        self.storage.store_bars(bars, timeframe)
            
            logger.info(
                "Historical bars fetched successfully",
                total_symbols=len(symbols),
                total_bars=sum(len(bars) for bars in result.values())
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to fetch historical bars",
                symbols=symbols,
                error=str(e)
            )
            raise
    
    @retry(max_attempts=3, delay=1.0)
    @rate_limit(max_calls=200, time_window=60)
    def get_latest_bars(
        self,
        symbols: Union[str, List[str]],
        timeframe: str = '1Day'
    ) -> Dict[str, BarData]:
        """Get latest bar data for symbols.
        
        Args:
            symbols: Symbol or list of symbols
            timeframe: Data timeframe
            
        Returns:
            Dictionary mapping symbols to latest BarData
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        tf = self.timeframe_map[timeframe]
        
        logger.debug("Fetching latest bars", symbols=symbols, timeframe=timeframe)
        
        try:
            latest_bars = self.api.get_latest_bars(symbols)
            
            result = {}
            for symbol in symbols:
                if symbol in latest_bars:
                    bar = latest_bars[symbol]
                    result[symbol] = BarData(
                        symbol=symbol,
                        timestamp=bar.timestamp.replace(tzinfo=None),
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=int(bar.volume),
                        trade_count=getattr(bar, 'trade_count', None),
                        vwap=getattr(bar, 'vwap', None)
                    )
            
            return result
            
        except Exception as e:
            logger.error("Failed to fetch latest bars", symbols=symbols, error=str(e))
            raise
    
    @retry(max_attempts=3, delay=1.0)
    @rate_limit(max_calls=200, time_window=60)
    def get_latest_trades(
        self,
        symbols: Union[str, List[str]]
    ) -> Dict[str, TradeData]:
        """Get latest trade data for symbols.
        
        Args:
            symbols: Symbol or list of symbols
            
        Returns:
            Dictionary mapping symbols to latest TradeData
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.debug("Fetching latest trades", symbols=symbols)
        
        try:
            latest_trades = self.api.get_latest_trades(symbols)
            
            result = {}
            for symbol in symbols:
                if symbol in latest_trades:
                    trade = latest_trades[symbol]
                    result[symbol] = TradeData(
                        symbol=symbol,
                        timestamp=trade.timestamp.replace(tzinfo=None),
                        price=float(trade.price),
                        size=int(trade.size),
                        conditions=getattr(trade, 'conditions', None),
                        exchange=getattr(trade, 'exchange', None)
                    )
            
            return result
            
        except Exception as e:
            logger.error("Failed to fetch latest trades", symbols=symbols, error=str(e))
            raise
    
    @retry(max_attempts=3, delay=1.0)
    @rate_limit(max_calls=200, time_window=60)
    def get_latest_quotes(
        self,
        symbols: Union[str, List[str]]
    ) -> Dict[str, QuoteData]:
        """Get latest quote data for symbols.
        
        Args:
            symbols: Symbol or list of symbols
            
        Returns:
            Dictionary mapping symbols to latest QuoteData
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.debug("Fetching latest quotes", symbols=symbols)
        
        try:
            latest_quotes = self.api.get_latest_quotes(symbols)
            
            result = {}
            for symbol in symbols:
                if symbol in latest_quotes:
                    quote = latest_quotes[symbol]
                    result[symbol] = QuoteData(
                        symbol=symbol,
                        timestamp=quote.timestamp.replace(tzinfo=None),
                        bid_price=float(quote.bid_price),
                        bid_size=int(quote.bid_size),
                        ask_price=float(quote.ask_price),
                        ask_size=int(quote.ask_size),
                        bid_exchange=getattr(quote, 'bid_exchange', None),
                        ask_exchange=getattr(quote, 'ask_exchange', None)
                    )
            
            return result
            
        except Exception as e:
            logger.error("Failed to fetch latest quotes", symbols=symbols, error=str(e))
            raise
    
    @retry(max_attempts=3, delay=1.0)
    @rate_limit(max_calls=200, time_window=60)
    def get_assets(self, status: str = 'active', asset_class: str = 'us_equity') -> List[AssetInfo]:
        """Get list of available assets.
        
        Args:
            status: Asset status ('active', 'inactive')
            asset_class: Asset class ('us_equity', 'crypto')
            
        Returns:
            List of AssetInfo objects
        """
        logger.debug("Fetching assets", status=status, asset_class=asset_class)
        
        try:
            assets = self.api.list_assets(status=status, asset_class=asset_class)
            
            result = []
            for asset in assets:
                asset_info = AssetInfo(
                    symbol=asset.symbol,
                    name=asset.name,
                    asset_class=asset.asset_class,
                    exchange=asset.exchange,
                    tradable=asset.tradable,
                    marginable=asset.marginable,
                    shortable=asset.shortable,
                    easy_to_borrow=asset.easy_to_borrow,
                    fractionable=asset.fractionable
                )
                result.append(asset_info)
            
            logger.info("Assets fetched successfully", count=len(result))
            return result
            
        except Exception as e:
            logger.error("Failed to fetch assets", error=str(e))
            raise
    
    @retry(max_attempts=3, delay=1.0)
    @rate_limit(max_calls=200, time_window=60)
    def get_market_status(self) -> MarketStatus:
        """Get current market status.
        
        Returns:
            MarketStatus object
        """
        logger.debug("Fetching market status")
        
        try:
            clock = self.api.get_clock()
            
            market_status = MarketStatus(
                is_open=clock.is_open,
                next_open=clock.next_open.replace(tzinfo=None) if clock.next_open else None,
                next_close=clock.next_close.replace(tzinfo=None) if clock.next_close else None
            )
            
            logger.debug(
                "Market status fetched",
                is_open=market_status.is_open,
                next_open=market_status.next_open,
                next_close=market_status.next_close
            )
            
            return market_status
            
        except Exception as e:
            logger.error("Failed to fetch market status", error=str(e))
            raise
    
    def to_dataframe(
        self,
        bars: List[BarData],
        include_indicators: bool = False
    ) -> pd.DataFrame:
        """Convert bar data to pandas DataFrame.
        
        Args:
            bars: List of BarData objects
            include_indicators: Whether to include technical indicators
            
        Returns:
            Pandas DataFrame
        """
        if not bars:
            return pd.DataFrame()
        
        data = []
        for bar in bars:
            row = {
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            }
            
            if bar.trade_count is not None:
                row['trade_count'] = bar.trade_count
            if bar.vwap is not None:
                row['vwap'] = bar.vwap
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        if include_indicators:
            df = self._add_technical_indicators(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
        """
        # Simple moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    async def collect_batch(
        self,
        symbols: List[str],
        timeframe: str = '1Day',
        days_back: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """Collect data for multiple symbols in parallel.
        
        Args:
            symbols: List of symbols to collect
            timeframe: Data timeframe
            days_back: Number of days to look back
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        start_date = datetime.now() - timedelta(days=days_back)
        
        logger.info(
            "Starting batch data collection",
            symbols=symbols,
            timeframe=timeframe,
            days_back=days_back
        )
        
        # Split symbols into chunks to avoid API limits
        chunk_size = 100  # Alpaca API limit
        symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        
        result = {}
        
        for chunk in symbol_chunks:
            try:
                bars_data = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.get_historical_bars,
                    chunk,
                    timeframe,
                    start_date
                )
                
                for symbol, bars in bars_data.items():
                    if bars:
                        result[symbol] = self.to_dataframe(bars, include_indicators=True)
                    else:
                        result[symbol] = pd.DataFrame()
                
            except Exception as e:
                logger.error(
                    "Failed to collect data for chunk",
                    chunk=chunk,
                    error=str(e)
                )
                # Add empty DataFrames for failed symbols
                for symbol in chunk:
                    result[symbol] = pd.DataFrame()
        
        logger.info(
            "Batch data collection completed",
            total_symbols=len(symbols),
            successful_symbols=len([df for df in result.values() if not df.empty])
        )
        
        return result
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        logger.info("DataCollector closed")