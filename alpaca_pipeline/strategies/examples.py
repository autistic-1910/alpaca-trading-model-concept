"""Example trading strategies for demonstration and testing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..data.models import BarData, PortfolioSnapshot
from ..utils import get_logger, calculate_sharpe_ratio, calculate_max_drawdown
from .base import BaseStrategy, StrategySignal, SignalType

logger = get_logger(__name__)


class MovingAverageStrategy(BaseStrategy):
    """Simple moving average crossover strategy."""
    
    def __init__(
        self,
        symbols: List[str],
        short_window: int = 20,
        long_window: int = 50,
        position_size: float = 0.1,
        **kwargs
    ):
        """Initialize moving average strategy.
        
        Args:
            symbols: List of symbols to trade
            short_window: Short moving average period
            long_window: Long moving average period
            position_size: Position size as fraction of portfolio
        """
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        
        super().__init__(
            name="Moving Average Crossover",
            symbols=symbols,
            **kwargs
        )
        
        # Track moving averages
        self.short_ma: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.long_ma: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.last_signal: Dict[str, Optional[SignalType]] = {symbol: None for symbol in symbols}
        
        logger.info(
            "MovingAverageStrategy initialized",
            strategy_id=self.strategy_id,
            short_window=short_window,
            long_window=long_window,
            position_size=position_size
        )
    
    def _validate_parameters(self):
        """Validate strategy-specific parameters."""
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")
        
        if not 0 < self.position_size <= 1:
            raise ValueError("Position size must be between 0 and 1")
    
    def generate_signals(
        self,
        current_data: Dict[str, BarData],
        portfolio: PortfolioSnapshot
    ) -> List[StrategySignal]:
        """Generate trading signals based on moving average crossover."""
        signals = []
        
        for symbol, bar_data in current_data.items():
            if symbol not in self.symbols:
                continue
            
            # Update moving averages
            self._update_moving_averages(symbol, bar_data.close)
            
            # Check if we have enough data
            if len(self.long_ma[symbol]) < self.long_window:
                continue
            
            # Calculate current moving averages
            current_short_ma = self.short_ma[symbol][-1]
            current_long_ma = self.long_ma[symbol][-1]
            
            # Get previous moving averages for crossover detection
            if len(self.short_ma[symbol]) < 2:
                continue
            
            prev_short_ma = self.short_ma[symbol][-2]
            prev_long_ma = self.long_ma[symbol][-2]
            
            # Detect crossovers
            signal_type = None
            
            # Bullish crossover: short MA crosses above long MA
            if (prev_short_ma <= prev_long_ma and 
                current_short_ma > current_long_ma and
                self.last_signal[symbol] != SignalType.BUY):
                signal_type = SignalType.BUY
            
            # Bearish crossover: short MA crosses below long MA
            elif (prev_short_ma >= prev_long_ma and 
                  current_short_ma < current_long_ma and
                  self.last_signal[symbol] != SignalType.SELL):
                signal_type = SignalType.SELL
            
            # Generate signal if crossover detected
            if signal_type:
                # Calculate position size
                portfolio_value = portfolio.total_equity
                target_value = portfolio_value * self.position_size
                quantity = int(target_value / bar_data.close)
                
                if quantity > 0:
                    signal = StrategySignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=signal_type,
                        quantity=quantity,
                        price=bar_data.close,
                        confidence=self._calculate_confidence(symbol, current_short_ma, current_long_ma),
                        metadata={
                            'short_ma': current_short_ma,
                            'long_ma': current_long_ma,
                            'short_window': self.short_window,
                            'long_window': self.long_window
                        }
                    )
                    
                    signals.append(signal)
                    self.last_signal[symbol] = signal_type
                    
                    logger.debug(
                        "MA crossover signal generated",
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=signal_type.value,
                        short_ma=current_short_ma,
                        long_ma=current_long_ma
                    )
        
        return signals
    
    def _update_moving_averages(self, symbol: str, price: float):
        """Update moving averages for a symbol."""
        # Add new price
        if len(self.short_ma[symbol]) >= self.long_window:
            # Remove oldest price if we have enough data
            self.short_ma[symbol].pop(0)
            self.long_ma[symbol].pop(0)
        
        # Calculate new moving averages
        prices = self.short_ma[symbol] + [price]
        
        if len(prices) >= self.short_window:
            short_ma_value = sum(prices[-self.short_window:]) / self.short_window
            self.short_ma[symbol].append(short_ma_value)
        
        if len(prices) >= self.long_window:
            long_ma_value = sum(prices[-self.long_window:]) / self.long_window
            self.long_ma[symbol].append(long_ma_value)
    
    def _calculate_confidence(self, symbol: str, short_ma: float, long_ma: float) -> float:
        """Calculate signal confidence based on MA separation."""
        separation = abs(short_ma - long_ma) / long_ma
        # Higher separation = higher confidence, capped at 1.0
        return min(separation * 10, 1.0)
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'short_window': self.short_window,
            'long_window': self.long_window,
            'position_size': self.position_size
        }


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands."""
    
    def __init__(
        self,
        symbols: List[str],
        lookback_period: int = 20,
        num_std: float = 2.0,
        position_size: float = 0.1,
        **kwargs
    ):
        """Initialize mean reversion strategy.
        
        Args:
            symbols: List of symbols to trade
            lookback_period: Period for calculating moving average and std
            num_std: Number of standard deviations for bands
            position_size: Position size as fraction of portfolio
        """
        super().__init__(
            name="Mean Reversion (Bollinger Bands)",
            symbols=symbols,
            **kwargs
        )
        
        self.lookback_period = lookback_period
        self.num_std = num_std
        self.position_size = position_size
        
        # Price history for calculations
        self.price_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.last_signal: Dict[str, Optional[SignalType]] = {symbol: None for symbol in symbols}
        
        logger.info(
            "MeanReversionStrategy initialized",
            strategy_id=self.strategy_id,
            lookback_period=lookback_period,
            num_std=num_std,
            position_size=position_size
        )
    
    def generate_signals(
        self,
        current_data: Dict[str, BarData],
        portfolio: PortfolioSnapshot
    ) -> List[StrategySignal]:
        """Generate mean reversion signals using Bollinger Bands."""
        signals = []
        
        for symbol, bar_data in current_data.items():
            if symbol not in self.symbols:
                continue
            
            # Update price history
            self._update_price_history(symbol, bar_data.close)
            
            # Check if we have enough data
            if len(self.price_history[symbol]) < self.lookback_period:
                continue
            
            # Calculate Bollinger Bands
            prices = self.price_history[symbol][-self.lookback_period:]
            mean_price = sum(prices) / len(prices)
            std_price = np.std(prices)
            
            upper_band = mean_price + (self.num_std * std_price)
            lower_band = mean_price - (self.num_std * std_price)
            
            current_price = bar_data.close
            signal_type = None
            
            # Generate signals
            # Buy when price touches lower band (oversold)
            if (current_price <= lower_band and 
                self.last_signal[symbol] != SignalType.BUY):
                signal_type = SignalType.BUY
            
            # Sell when price touches upper band (overbought)
            elif (current_price >= upper_band and 
                  self.last_signal[symbol] != SignalType.SELL):
                signal_type = SignalType.SELL
            
            # Generate signal
            if signal_type:
                # Calculate position size
                portfolio_value = portfolio.total_equity
                target_value = portfolio_value * self.position_size
                quantity = int(target_value / current_price)
                
                if quantity > 0:
                    signal = StrategySignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=signal_type,
                        quantity=quantity,
                        price=current_price,
                        confidence=self._calculate_confidence(
                            current_price, mean_price, upper_band, lower_band
                        ),
                        metadata={
                            'mean_price': mean_price,
                            'upper_band': upper_band,
                            'lower_band': lower_band,
                            'std_price': std_price,
                            'lookback_period': self.lookback_period
                        }
                    )
                    
                    signals.append(signal)
                    self.last_signal[symbol] = signal_type
                    
                    logger.debug(
                        "Mean reversion signal generated",
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=signal_type.value,
                        price=current_price,
                        upper_band=upper_band,
                        lower_band=lower_band
                    )
        
        return signals
    
    def _update_price_history(self, symbol: str, price: float):
        """Update price history for a symbol."""
        self.price_history[symbol].append(price)
        
        # Keep only required history
        if len(self.price_history[symbol]) > self.lookback_period:
            self.price_history[symbol].pop(0)
    
    def _calculate_confidence(
        self,
        current_price: float,
        mean_price: float,
        upper_band: float,
        lower_band: float
    ) -> float:
        """Calculate signal confidence based on distance from bands."""
        band_width = upper_band - lower_band
        
        if current_price <= lower_band:
            # Distance below lower band
            distance = lower_band - current_price
            return min(distance / band_width, 1.0)
        elif current_price >= upper_band:
            # Distance above upper band
            distance = current_price - upper_band
            return min(distance / band_width, 1.0)
        else:
            return 0.0
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'lookback_period': self.lookback_period,
            'num_std': self.num_std,
            'position_size': self.position_size
        }


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using rate of change."""
    
    def __init__(
        self,
        symbols: List[str],
        lookback_period: int = 14,
        momentum_threshold: float = 0.05,
        position_size: float = 0.1,
        **kwargs
    ):
        """Initialize momentum strategy.
        
        Args:
            symbols: List of symbols to trade
            lookback_period: Period for calculating momentum
            momentum_threshold: Minimum momentum for signal generation
            position_size: Position size as fraction of portfolio
        """
        super().__init__(
            name="Momentum Strategy",
            symbols=symbols,
            **kwargs
        )
        
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.position_size = position_size
        
        # Price history for momentum calculation
        self.price_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.last_signal: Dict[str, Optional[SignalType]] = {symbol: None for symbol in symbols}
        
        logger.info(
            "MomentumStrategy initialized",
            strategy_id=self.strategy_id,
            lookback_period=lookback_period,
            momentum_threshold=momentum_threshold,
            position_size=position_size
        )
    
    def generate_signals(
        self,
        current_data: Dict[str, BarData],
        portfolio: PortfolioSnapshot
    ) -> List[StrategySignal]:
        """Generate momentum-based trading signals."""
        signals = []
        
        for symbol, bar_data in current_data.items():
            if symbol not in self.symbols:
                continue
            
            # Update price history
            self._update_price_history(symbol, bar_data.close)
            
            # Check if we have enough data
            if len(self.price_history[symbol]) < self.lookback_period + 1:
                continue
            
            # Calculate momentum (rate of change)
            current_price = self.price_history[symbol][-1]
            past_price = self.price_history[symbol][-(self.lookback_period + 1)]
            momentum = (current_price - past_price) / past_price
            
            signal_type = None
            
            # Generate signals based on momentum
            if (momentum > self.momentum_threshold and 
                self.last_signal[symbol] != SignalType.BUY):
                signal_type = SignalType.BUY
            elif (momentum < -self.momentum_threshold and 
                  self.last_signal[symbol] != SignalType.SELL):
                signal_type = SignalType.SELL
            
            # Generate signal
            if signal_type:
                # Calculate position size
                portfolio_value = portfolio.total_equity
                target_value = portfolio_value * self.position_size
                quantity = int(target_value / current_price)
                
                if quantity > 0:
                    signal = StrategySignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=signal_type,
                        quantity=quantity,
                        price=current_price,
                        confidence=self._calculate_confidence(momentum),
                        metadata={
                            'momentum': momentum,
                            'momentum_threshold': self.momentum_threshold,
                            'lookback_period': self.lookback_period,
                            'past_price': past_price
                        }
                    )
                    
                    signals.append(signal)
                    self.last_signal[symbol] = signal_type
                    
                    logger.debug(
                        "Momentum signal generated",
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=signal_type.value,
                        momentum=momentum,
                        threshold=self.momentum_threshold
                    )
        
        return signals
    
    def _update_price_history(self, symbol: str, price: float):
        """Update price history for a symbol."""
        self.price_history[symbol].append(price)
        
        # Keep only required history
        if len(self.price_history[symbol]) > self.lookback_period + 1:
            self.price_history[symbol].pop(0)
    
    def _calculate_confidence(self, momentum: float) -> float:
        """Calculate signal confidence based on momentum strength."""
        # Higher absolute momentum = higher confidence
        abs_momentum = abs(momentum)
        # Normalize confidence to 0-1 range
        return min(abs_momentum / (self.momentum_threshold * 2), 1.0)
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'lookback_period': self.lookback_period,
            'momentum_threshold': self.momentum_threshold,
            'position_size': self.position_size
        }


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy."""
    
    def __init__(
        self,
        symbols: List[str],
        position_size: float = 0.1,
        rebalance_frequency: int = 30,  # days
        **kwargs
    ):
        """Initialize buy and hold strategy.
        
        Args:
            symbols: List of symbols to trade
            position_size: Position size as fraction of portfolio per symbol
            rebalance_frequency: How often to rebalance (in days)
        """
        super().__init__(
            name="Buy and Hold",
            symbols=symbols,
            **kwargs
        )
        
        self.position_size = position_size
        self.rebalance_frequency = rebalance_frequency
        
        # Track when we last bought each symbol
        self.last_purchase: Dict[str, Optional[datetime]] = {symbol: None for symbol in symbols}
        self.last_rebalance: Optional[datetime] = None
        
        logger.info(
            "BuyAndHoldStrategy initialized",
            strategy_id=self.strategy_id,
            position_size=position_size,
            rebalance_frequency=rebalance_frequency
        )
    
    def generate_signals(
        self,
        current_data: Dict[str, BarData],
        portfolio: PortfolioSnapshot
    ) -> List[StrategySignal]:
        """Generate buy and hold signals."""
        signals = []
        current_time = datetime.now()
        
        # Check if it's time to rebalance
        should_rebalance = (
            self.last_rebalance is None or
            (current_time - self.last_rebalance).days >= self.rebalance_frequency
        )
        
        if not should_rebalance:
            return signals
        
        for symbol, bar_data in current_data.items():
            if symbol not in self.symbols:
                continue
            
            # Check if we need to buy this symbol
            should_buy = (
                self.last_purchase[symbol] is None or
                (current_time - self.last_purchase[symbol]).days >= self.rebalance_frequency
            )
            
            if should_buy:
                # Calculate target position size
                portfolio_value = portfolio.total_equity
                target_value = portfolio_value * self.position_size
                
                # Get current position
                current_position = portfolio.positions.get(symbol, 0)
                current_value = current_position * bar_data.close
                
                # Calculate how much more to buy
                additional_value = target_value - current_value
                
                if additional_value > 0:
                    quantity = int(additional_value / bar_data.close)
                    
                    if quantity > 0:
                        signal = StrategySignal(
                            strategy_id=self.strategy_id,
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            quantity=quantity,
                            price=bar_data.close,
                            confidence=1.0,  # Always confident in buy and hold
                            metadata={
                                'target_value': target_value,
                                'current_value': current_value,
                                'additional_value': additional_value,
                                'rebalance': True
                            }
                        )
                        
                        signals.append(signal)
                        self.last_purchase[symbol] = current_time
                        
                        logger.debug(
                            "Buy and hold signal generated",
                            strategy_id=self.strategy_id,
                            symbol=symbol,
                            quantity=quantity,
                            target_value=target_value
                        )
        
        if signals:
            self.last_rebalance = current_time
        
        return signals
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'position_size': self.position_size,
            'rebalance_frequency': self.rebalance_frequency
        }


class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy."""
    
    def __init__(
        self,
        symbols: List[str],
        rsi_period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
        position_size: float = 0.1,
        **kwargs
    ):
        """Initialize RSI strategy.
        
        Args:
            symbols: List of symbols to trade
            rsi_period: Period for RSI calculation
            oversold_threshold: RSI level for buy signals
            overbought_threshold: RSI level for sell signals
            position_size: Position size as fraction of portfolio
        """
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.position_size = position_size
        
        super().__init__(
            name="RSI Strategy",
            symbols=symbols,
            **kwargs
        )
        
        # Price history for RSI calculation
        self.price_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.rsi_values: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.last_signal: Dict[str, Optional[SignalType]] = {symbol: None for symbol in symbols}
        
        logger.info(
            "RSIStrategy initialized",
            strategy_id=self.strategy_id,
            rsi_period=rsi_period,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold
        )
    
    def _validate_parameters(self):
        """Validate strategy-specific parameters."""
        if self.rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        
        if not 0 < self.oversold_threshold < 50:
            raise ValueError("Oversold threshold must be between 0 and 50")
        
        if not 50 < self.overbought_threshold < 100:
            raise ValueError("Overbought threshold must be between 50 and 100")
        
        if self.oversold_threshold >= self.overbought_threshold:
            raise ValueError("Oversold threshold must be less than overbought threshold")
        
        if not 0 < self.position_size <= 1:
            raise ValueError("Position size must be between 0 and 1")
    
    def generate_signals(
        self,
        current_data: Dict[str, BarData],
        portfolio: PortfolioSnapshot
    ) -> List[StrategySignal]:
        """Generate RSI-based trading signals."""
        signals = []
        
        for symbol, bar_data in current_data.items():
            if symbol not in self.symbols:
                continue
            
            # Update price history and calculate RSI
            self._update_rsi(symbol, bar_data.close)
            
            # Check if we have enough RSI data
            if len(self.rsi_values[symbol]) < 2:
                continue
            
            current_rsi = self.rsi_values[symbol][-1]
            signal_type = None
            
            # Generate signals based on RSI levels
            if (current_rsi <= self.oversold_threshold and 
                self.last_signal[symbol] != SignalType.BUY):
                signal_type = SignalType.BUY
            elif (current_rsi >= self.overbought_threshold and 
                  self.last_signal[symbol] != SignalType.SELL):
                signal_type = SignalType.SELL
            
            # Generate signal
            if signal_type:
                # Calculate position size
                portfolio_value = portfolio.total_equity
                target_value = portfolio_value * self.position_size
                quantity = int(target_value / bar_data.close)
                
                if quantity > 0:
                    signal = StrategySignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=signal_type,
                        quantity=quantity,
                        price=bar_data.close,
                        confidence=self._calculate_confidence(current_rsi, signal_type),
                        metadata={
                            'rsi': current_rsi,
                            'oversold_threshold': self.oversold_threshold,
                            'overbought_threshold': self.overbought_threshold,
                            'rsi_period': self.rsi_period
                        }
                    )
                    
                    signals.append(signal)
                    self.last_signal[symbol] = signal_type
                    
                    logger.debug(
                        "RSI signal generated",
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        signal_type=signal_type.value,
                        rsi=current_rsi
                    )
        
        return signals
    
    def _update_rsi(self, symbol: str, price: float):
        """Update RSI calculation for a symbol."""
        self.price_history[symbol].append(price)
        
        # Keep only required history
        if len(self.price_history[symbol]) > self.rsi_period + 1:
            self.price_history[symbol].pop(0)
        
        # Calculate RSI if we have enough data
        if len(self.price_history[symbol]) >= self.rsi_period + 1:
            rsi = self._calculate_rsi(self.price_history[symbol])
            self.rsi_values[symbol].append(rsi)
            
            # Keep only recent RSI values
            if len(self.rsi_values[symbol]) > 100:
                self.rsi_values[symbol].pop(0)
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI for a list of prices."""
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in changes]
        losses = [-change if change < 0 else 0 for change in changes]
        
        # Calculate average gains and losses
        avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
        avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period
        
        # Avoid division by zero
        if avg_loss == 0:
            return 100.0
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_confidence(
        self,
        rsi: float,
        signal_type: SignalType
    ) -> float:
        """Calculate signal confidence based on RSI extremity."""
        if signal_type == SignalType.BUY:
            # More oversold = higher confidence
            distance = self.oversold_threshold - rsi
            return min(distance / self.oversold_threshold, 1.0)
        else:  # SELL
            # More overbought = higher confidence
            distance = rsi - self.overbought_threshold
            return min(distance / (100 - self.overbought_threshold), 1.0)
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'rsi_period': self.rsi_period,
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold,
            'position_size': self.position_size
        }