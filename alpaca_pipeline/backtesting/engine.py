"""Main backtesting engine that orchestrates all components."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path

from ..utils import get_logger
from ..data import DataCollector, BarData
from ..strategies import BaseStrategy, StrategySignal, SignalType
from .portfolio import Portfolio
from .order_manager import OrderManager, Order, OrderSide, OrderType
from .performance import PerformanceAnalyzer, PerformanceMetrics
from .visualizer import BacktestVisualizer
from .risk_manager import RiskManager, RiskLimits

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Time period
    start_date: datetime
    end_date: datetime
    
    # Initial capital
    initial_capital: float = 100000.0
    
    # Commission and slippage
    commission_per_share: float = 0.005
    commission_percentage: float = 0.0
    slippage_percentage: float = 0.001
    
    # Risk management
    risk_limits: Optional[RiskLimits] = None
    
    # Data settings
    benchmark_symbol: str = "SPY"
    data_frequency: str = "1Day"
    
    # Execution settings
    fill_delay: int = 1  # Bars delay for order fills
    partial_fills: bool = False
    
    # Output settings
    save_results: bool = True
    output_directory: str = "backtest_results"
    generate_plots: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'commission_per_share': self.commission_per_share,
            'commission_percentage': self.commission_percentage,
            'slippage_percentage': self.slippage_percentage,
            'benchmark_symbol': self.benchmark_symbol,
            'data_frequency': self.data_frequency,
            'fill_delay': self.fill_delay,
            'partial_fills': self.partial_fills,
            'save_results': self.save_results,
            'output_directory': self.output_directory,
            'generate_plots': self.generate_plots
        }


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(
        self,
        config: BacktestConfig,
        data_collector: Optional[DataCollector] = None
    ):
        """Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
            data_collector: Data collector for market data
        """
        self.config = config
        self.data_collector = data_collector
        
        # Initialize components
        self.portfolio = Portfolio(initial_cash=config.initial_capital)
        self.order_manager = OrderManager(
            commission_per_share=config.commission_per_share,
            commission_percentage=config.commission_percentage,
            slippage_percentage=config.slippage_percentage
        )
        self.risk_manager = RiskManager(config.risk_limits)
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = BacktestVisualizer()
        
        # State tracking
        self.strategies: List[BaseStrategy] = []
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.current_bar_index = 0
        self.current_timestamp: Optional[datetime] = None
        
        # Results
        self.results: Optional[Dict[str, Any]] = None
        self.is_running = False
        
        # Callbacks
        self.on_bar_callbacks: List[Callable] = []
        self.on_order_callbacks: List[Callable] = []
        self.on_fill_callbacks: List[Callable] = []
        
        logger.info(
            "BacktestEngine initialized",
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital
        )
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add a trading strategy.
        
        Args:
            strategy: Strategy to add
        """
        self.strategies.append(strategy)
        logger.info(f"Strategy added: {strategy.__class__.__name__}")
    
    def add_data(self, symbol: str, data: pd.DataFrame):
        """Add market data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
        """
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        # Filter data to backtest period
        mask = (data.index >= self.config.start_date) & (data.index <= self.config.end_date)
        filtered_data = data[mask]
        
        if filtered_data.empty:
            logger.warning(f"No data for {symbol} in backtest period")
            return
        
        self.market_data[symbol] = filtered_data
        logger.info(f"Data added for {symbol}: {len(filtered_data)} bars")
    
    def load_data(self, symbols: List[str]):
        """Load market data for symbols.
        
        Args:
            symbols: List of symbols to load
        """
        if not self.data_collector:
            raise ValueError("Data collector not provided")
        
        logger.info(f"Loading data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                # Get historical data
                bars = self.data_collector.get_historical_bars(
                    symbol=symbol,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    timeframe=self.config.data_frequency
                )
                
                if bars:
                    # Convert to DataFrame
                    df = self.data_collector.bars_to_dataframe(bars)
                    self.add_data(symbol, df)
                else:
                    logger.warning(f"No data retrieved for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
        
        # Load benchmark data
        if self.config.benchmark_symbol and self.config.benchmark_symbol not in symbols:
            try:
                benchmark_bars = self.data_collector.get_historical_bars(
                    symbol=self.config.benchmark_symbol,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    timeframe=self.config.data_frequency
                )
                
                if benchmark_bars:
                    self.benchmark_data = self.data_collector.bars_to_dataframe(benchmark_bars)
                    logger.info(f"Benchmark data loaded: {self.config.benchmark_symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to load benchmark data: {e}")
    
    def run(self) -> Dict[str, Any]:
        """Run the backtest.
        
        Returns:
            Backtest results
        """
        if not self.strategies:
            raise ValueError("No strategies added")
        
        if not self.market_data:
            raise ValueError("No market data loaded")
        
        logger.info("Starting backtest")
        self.is_running = True
        
        try:
            # Initialize strategies
            self._initialize_strategies()
            
            # Get all timestamps
            all_timestamps = self._get_all_timestamps()
            
            if not all_timestamps:
                raise ValueError("No valid timestamps found")
            
            logger.info(f"Processing {len(all_timestamps)} time periods")
            
            # Process each timestamp
            for i, timestamp in enumerate(all_timestamps):
                self.current_bar_index = i
                self.current_timestamp = timestamp
                
                # Update portfolio with current market data
                self._update_portfolio_prices(timestamp)
                
                # Update risk manager
                self.risk_manager.update_daily_stats(self.portfolio)
                
                # Process orders from previous bar
                self._process_pending_orders(timestamp)
                
                # Update strategies with current data
                self._update_strategies(timestamp)
                
                # Generate and process signals
                self._process_signals(timestamp)
                
                # Check risk violations
                violations = self.risk_manager.check_risk_violations(self.portfolio)
                if violations:
                    logger.warning(f"Risk violations at {timestamp}: {violations}")
                
                # Call bar callbacks
                for callback in self.on_bar_callbacks:
                    try:
                        callback(timestamp, self.portfolio, self.market_data)
                    except Exception as e:
                        logger.error(f"Bar callback error: {e}")
                
                # Log progress
                if i % 100 == 0 or i == len(all_timestamps) - 1:
                    progress = (i + 1) / len(all_timestamps) * 100
                    logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(all_timestamps)})")
            
            # Finalize strategies
            self._finalize_strategies()
            
            # Generate results
            self.results = self._generate_results()
            
            # Save results if configured
            if self.config.save_results:
                self._save_results()
            
            logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.is_running = False
    
    def run_async(self) -> Dict[str, Any]:
        """Run backtest asynchronously.
        
        Returns:
            Backtest results
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self._run_async_impl())
        finally:
            loop.close()
    
    async def _run_async_impl(self) -> Dict[str, Any]:
        """Async implementation of backtest."""
        # Run in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.run)
            return await asyncio.wrap_future(future)
    
    def add_bar_callback(self, callback: Callable):
        """Add callback for each bar processed.
        
        Args:
            callback: Function to call with (timestamp, portfolio, market_data)
        """
        self.on_bar_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable):
        """Add callback for order events.
        
        Args:
            callback: Function to call with order
        """
        self.on_order_callbacks.append(callback)
    
    def add_fill_callback(self, callback: Callable):
        """Add callback for fill events.
        
        Args:
            callback: Function to call with order and fill
        """
        self.on_fill_callbacks.append(callback)
    
    def get_current_data(self, symbol: str) -> Optional[pd.Series]:
        """Get current bar data for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current bar data or None
        """
        if symbol not in self.market_data or self.current_timestamp is None:
            return None
        
        data = self.market_data[symbol]
        
        # Find the current bar
        if self.current_timestamp in data.index:
            return data.loc[self.current_timestamp]
        
        # Find the most recent bar before current timestamp
        mask = data.index <= self.current_timestamp
        if mask.any():
            return data[mask].iloc[-1]
        
        return None
    
    def get_historical_data(
        self,
        symbol: str,
        lookback_periods: int
    ) -> Optional[pd.DataFrame]:
        """Get historical data for symbol.
        
        Args:
            symbol: Trading symbol
            lookback_periods: Number of periods to look back
            
        Returns:
            Historical data or None
        """
        if symbol not in self.market_data or self.current_bar_index < lookback_periods:
            return None
        
        data = self.market_data[symbol]
        
        # Get data up to current timestamp
        mask = data.index <= self.current_timestamp
        historical_data = data[mask]
        
        if len(historical_data) < lookback_periods:
            return None
        
        return historical_data.tail(lookback_periods)
    
    def _initialize_strategies(self):
        """Initialize all strategies."""
        for strategy in self.strategies:
            try:
                strategy.start()
                logger.info(f"Strategy initialized: {strategy.__class__.__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize strategy {strategy.__class__.__name__}: {e}")
    
    def _finalize_strategies(self):
        """Finalize all strategies."""
        for strategy in self.strategies:
            try:
                strategy.stop()
                logger.info(f"Strategy finalized: {strategy.__class__.__name__}")
            except Exception as e:
                logger.error(f"Failed to finalize strategy {strategy.__class__.__name__}: {e}")
    
    def _get_all_timestamps(self) -> List[datetime]:
        """Get all unique timestamps from market data."""
        all_timestamps = set()
        
        for data in self.market_data.values():
            all_timestamps.update(data.index)
        
        return sorted(list(all_timestamps))
    
    def _update_portfolio_prices(self, timestamp: datetime):
        """Update portfolio with current market prices."""
        current_prices = {}
        
        for symbol, data in self.market_data.items():
            current_data = self.get_current_data(symbol)
            if current_data is not None:
                current_prices[symbol] = current_data['close']
        
        self.portfolio.update_market_data(current_prices, timestamp)
    
    def _update_strategies(self, timestamp: datetime):
        """Update strategies with current market data."""
        for strategy in self.strategies:
            try:
                # Prepare market data for strategy
                strategy_data = {}
                
                for symbol in strategy.symbols:
                    current_data = self.get_current_data(symbol)
                    if current_data is not None:
                        # Convert to BarData
                        bar_data = BarData(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=current_data['open'],
                            high=current_data['high'],
                            low=current_data['low'],
                            close=current_data['close'],
                            volume=current_data['volume']
                        )
                        strategy_data[symbol] = bar_data
                
                # Update strategy
                if strategy_data:
                    strategy.update_market_data(strategy_data)
                    
            except Exception as e:
                logger.error(f"Failed to update strategy {strategy.__class__.__name__}: {e}")
    
    def _process_signals(self, timestamp: datetime):
        """Process signals from all strategies."""
        for strategy in self.strategies:
            try:
                signals = strategy.get_signals()
                
                for signal in signals:
                    self._process_signal(signal, timestamp)
                    
            except Exception as e:
                logger.error(f"Failed to process signals from {strategy.__class__.__name__}: {e}")
    
    def _process_signal(self, signal: StrategySignal, timestamp: datetime):
        """Process a single trading signal."""
        try:
            # Get current price
            current_data = self.get_current_data(signal.symbol)
            if current_data is None:
                logger.warning(f"No current data for signal: {signal.symbol}")
                return
            
            current_price = current_data['close']
            
            # Determine order side
            if signal.signal_type == SignalType.BUY:
                side = OrderSide.BUY
            elif signal.signal_type == SignalType.SELL:
                side = OrderSide.SELL
            else:
                return  # Skip HOLD signals
            
            # Calculate position size if not provided
            quantity = signal.quantity
            if quantity is None or quantity <= 0:
                # Use risk manager to calculate position size
                stop_loss_price = signal.stop_loss or self.risk_manager.calculate_stop_loss(
                    current_price, side
                )
                
                quantity = self.risk_manager.calculate_position_size(
                    signal.symbol,
                    current_price,
                    stop_loss_price,
                    self.portfolio
                )
            
            if quantity <= 0:
                logger.debug(f"Zero quantity calculated for signal: {signal.symbol}")
                return
            
            # Create order
            order = Order(
                symbol=signal.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                strategy_id=signal.strategy_id,
                created_at=timestamp
            )
            
            # Validate order with risk manager
            is_valid, reason = self.risk_manager.validate_order(
                order, self.portfolio, current_price
            )
            
            if not is_valid:
                logger.warning(f"Order rejected: {reason}")
                return
            
            # Submit order
            self.order_manager.submit_order(order)
            
            # Record trade for risk management
            self.risk_manager.record_trade(order, current_price)
            
            # Call order callbacks
            for callback in self.on_order_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Order callback error: {e}")
            
            logger.info(
                "Signal processed",
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                quantity=quantity,
                price=current_price
            )
            
        except Exception as e:
            logger.error(f"Failed to process signal: {e}")
    
    def _process_pending_orders(self, timestamp: datetime):
        """Process pending orders."""
        # Get current market data
        market_data = {}
        for symbol, data in self.market_data.items():
            current_data = self.get_current_data(symbol)
            if current_data is not None:
                market_data[symbol] = {
                    'open': current_data['open'],
                    'high': current_data['high'],
                    'low': current_data['low'],
                    'close': current_data['close'],
                    'volume': current_data['volume']
                }
        
        # Process orders
        fills = self.order_manager.process_market_data(market_data, timestamp)
        
        # Apply fills to portfolio
        for fill in fills:
            self.portfolio.process_fill(fill)
            
            # Call fill callbacks
            for callback in self.on_fill_callbacks:
                try:
                    callback(fill.order, fill)
                except Exception as e:
                    logger.error(f"Fill callback error: {e}")
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results."""
        logger.info("Generating backtest results")
        
        # Performance analysis
        performance_metrics = self.performance_analyzer.analyze_portfolio(self.portfolio)
        
        # Trade analysis
        all_orders = self.order_manager.get_all_orders()
        trade_analysis = self.performance_analyzer.analyze_trades(all_orders)
        
        # Rolling metrics
        equity_curve = self.portfolio.get_equity_curve()
        rolling_metrics = self.performance_analyzer.calculate_rolling_metrics(equity_curve)
        
        # Benchmark comparison
        benchmark_comparison = {}
        if self.benchmark_data is not None:
            benchmark_curve = [(idx, row['close']) for idx, row in self.benchmark_data.iterrows()]
            benchmark_comparison = self.performance_analyzer.compare_to_benchmark(
                equity_curve, benchmark_curve
            )
        
        # Risk analysis
        risk_summary = self.risk_manager.get_risk_summary(self.portfolio)
        
        # Portfolio summary
        portfolio_summary = self.portfolio.get_portfolio_summary()
        
        # Strategy performance
        strategy_performance = {}
        for strategy in self.strategies:
            strategy_performance[strategy.__class__.__name__] = {
                'total_signals': len(strategy.signals_history),
                'performance': strategy.performance.to_dict() if hasattr(strategy, 'performance') else {}
            }
        
        results = {
            'config': self.config.to_dict(),
            'performance_metrics': performance_metrics.to_dict(),
            'portfolio_summary': portfolio_summary,
            'trade_analysis': trade_analysis,
            'rolling_metrics': {
                key: [(ts.isoformat(), value) for ts, value in values]
                for key, values in rolling_metrics.items()
            },
            'benchmark_comparison': benchmark_comparison,
            'risk_analysis': risk_summary,
            'strategy_performance': strategy_performance,
            'equity_curve': [(ts.isoformat(), equity) for ts, equity in equity_curve],
            'order_statistics': self.order_manager.get_order_statistics(),
            'backtest_metadata': {
                'start_time': self.config.start_date.isoformat(),
                'end_time': self.config.end_date.isoformat(),
                'total_bars': len(self._get_all_timestamps()),
                'symbols_traded': list(self.market_data.keys()),
                'strategies_used': [s.__class__.__name__ for s in self.strategies]
            }
        }
        
        return results
    
    def _save_results(self):
        """Save backtest results to files."""
        if not self.results:
            return
        
        # Create output directory
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save results as JSON
            import json
            results_file = output_dir / f"backtest_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
            # Generate and save plots
            if self.config.generate_plots:
                self._generate_plots(output_dir, timestamp)
            
            # Save portfolio data
            equity_df = pd.DataFrame(
                [(datetime.fromisoformat(ts), equity) for ts, equity in self.results['equity_curve']],
                columns=['timestamp', 'equity']
            ).set_index('timestamp')
            
            equity_file = output_dir / f"equity_curve_{timestamp}.csv"
            equity_df.to_csv(equity_file)
            
            logger.info(f"Equity curve saved to {equity_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _generate_plots(self, output_dir: Path, timestamp: str):
        """Generate and save visualization plots."""
        try:
            base_path = str(output_dir / f"backtest_plots_{timestamp}")
            
            # Prepare benchmark data for plotting
            benchmark_data = None
            if self.benchmark_data is not None:
                benchmark_data = [(idx, row['close']) for idx, row in self.benchmark_data.iterrows()]
            
            # Create dashboard
            figures = self.visualizer.create_dashboard(
                portfolio=self.portfolio,
                metrics=PerformanceMetrics(**self.results['performance_metrics']),
                trade_analysis=self.results['trade_analysis'],
                rolling_metrics={
                    key: [(datetime.fromisoformat(ts), value) for ts, value in values]
                    for key, values in self.results['rolling_metrics'].items()
                },
                benchmark_data=benchmark_data,
                save_path=base_path
            )
            
            logger.info(f"Generated {len(figures)} visualization plots")
            
            # Close figures to free memory
            self.visualizer.close_all_figures()
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")