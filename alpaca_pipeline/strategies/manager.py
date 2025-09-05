"""Strategy manager for coordinating multiple trading strategies."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty

from ..config import get_config
from ..utils import get_logger, timing
from ..data.models import BarData, PortfolioSnapshot
from .base import BaseStrategy, StrategySignal, StrategyState

logger = get_logger(__name__)


class StrategyManager:
    """Manager for coordinating multiple trading strategies."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize strategy manager.
        
        Args:
            max_workers: Maximum number of worker threads for strategy execution
        """
        self.config = get_config()
        self.strategies: Dict[str, BaseStrategy] = {}
        self.signal_queue = Queue()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # State management
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.signal_callbacks: List[Callable[[StrategySignal], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        
        # Performance tracking
        self.total_signals_processed = 0
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            "StrategyManager initialized",
            max_workers=max_workers
        )
    
    def add_strategy(self, strategy: BaseStrategy) -> str:
        """Add a strategy to the manager.
        
        Args:
            strategy: Strategy instance to add
            
        Returns:
            Strategy ID
        """
        if strategy.strategy_id in self.strategies:
            raise ValueError(f"Strategy with ID {strategy.strategy_id} already exists")
        
        self.strategies[strategy.strategy_id] = strategy
        self.strategy_performance[strategy.strategy_id] = {
            'signals_generated': 0,
            'execution_time_ms': [],
            'errors': 0,
            'last_execution': None
        }
        
        logger.info(
            "Strategy added",
            strategy_id=strategy.strategy_id,
            name=strategy.name,
            symbols=strategy.symbols
        )
        
        return strategy.strategy_id
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the manager.
        
        Args:
            strategy_id: ID of strategy to remove
            
        Returns:
            True if strategy was removed, False if not found
        """
        if strategy_id not in self.strategies:
            logger.warning("Strategy not found for removal", strategy_id=strategy_id)
            return False
        
        strategy = self.strategies[strategy_id]
        
        # Stop strategy if running
        if strategy.state == StrategyState.ACTIVE:
            strategy.stop()
        
        # Remove from manager
        del self.strategies[strategy_id]
        del self.strategy_performance[strategy_id]
        
        logger.info(
            "Strategy removed",
            strategy_id=strategy_id,
            name=strategy.name
        )
        
        return True
    
    def get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """Get a strategy by ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Strategy instance or None if not found
        """
        return self.strategies.get(strategy_id)
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all strategies with their status.
        
        Returns:
            List of strategy information dictionaries
        """
        strategies_info = []
        
        for strategy_id, strategy in self.strategies.items():
            perf = self.strategy_performance[strategy_id]
            
            strategies_info.append({
                'strategy_id': strategy_id,
                'name': strategy.name,
                'state': strategy.state.value,
                'symbols': strategy.symbols,
                'signals_generated': perf['signals_generated'],
                'errors': perf['errors'],
                'last_execution': perf['last_execution'],
                'avg_execution_time_ms': (
                    sum(perf['execution_time_ms']) / len(perf['execution_time_ms'])
                    if perf['execution_time_ms'] else 0
                )
            })
        
        return strategies_info
    
    def start_strategy(self, strategy_id: str) -> bool:
        """Start a specific strategy.
        
        Args:
            strategy_id: Strategy ID to start
            
        Returns:
            True if strategy was started, False otherwise
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            logger.error("Strategy not found", strategy_id=strategy_id)
            return False
        
        try:
            strategy.start()
            logger.info(
                "Strategy started",
                strategy_id=strategy_id,
                name=strategy.name
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to start strategy",
                strategy_id=strategy_id,
                error=str(e)
            )
            self._handle_strategy_error(strategy_id, e)
            return False
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a specific strategy.
        
        Args:
            strategy_id: Strategy ID to stop
            
        Returns:
            True if strategy was stopped, False otherwise
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            logger.error("Strategy not found", strategy_id=strategy_id)
            return False
        
        try:
            strategy.stop()
            logger.info(
                "Strategy stopped",
                strategy_id=strategy_id,
                name=strategy.name
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to stop strategy",
                strategy_id=strategy_id,
                error=str(e)
            )
            return False
    
    def start_all_strategies(self):
        """Start all strategies."""
        started_count = 0
        
        for strategy_id, strategy in self.strategies.items():
            if self.start_strategy(strategy_id):
                started_count += 1
        
        logger.info(
            "Started strategies",
            started=started_count,
            total=len(self.strategies)
        )
    
    def stop_all_strategies(self):
        """Stop all strategies."""
        stopped_count = 0
        
        for strategy_id, strategy in self.strategies.items():
            if self.stop_strategy(strategy_id):
                stopped_count += 1
        
        logger.info(
            "Stopped strategies",
            stopped=stopped_count,
            total=len(self.strategies)
        )
    
    def start_manager(self):
        """Start the strategy manager."""
        if self.is_running:
            logger.warning("Strategy manager is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start signal processing thread
        self.processing_thread = threading.Thread(
            target=self._process_signals,
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Strategy manager started")
    
    def stop_manager(self):
        """Stop the strategy manager."""
        if not self.is_running:
            logger.warning("Strategy manager is not running")
            return
        
        # Stop all strategies first
        self.stop_all_strategies()
        
        # Stop manager
        self.is_running = False
        self.stop_event.set()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Strategy manager stopped")
    
    def update_market_data(
        self,
        symbol: str,
        bar_data: BarData,
        portfolio: PortfolioSnapshot
    ):
        """Update market data for all strategies trading the symbol.
        
        Args:
            symbol: Symbol that was updated
            bar_data: New bar data
            portfolio: Current portfolio snapshot
        """
        if not self.is_running:
            return
        
        # Find strategies that trade this symbol
        relevant_strategies = [
            strategy for strategy in self.strategies.values()
            if symbol in strategy.symbols and strategy.state == StrategyState.ACTIVE
        ]
        
        if not relevant_strategies:
            return
        
        # Update market data and generate signals
        for strategy in relevant_strategies:
            try:
                # Update market data
                strategy.update_market_data(symbol, bar_data)
                
                # Submit signal generation task
                future = self.executor.submit(
                    self._generate_strategy_signals,
                    strategy,
                    {symbol: bar_data},
                    portfolio
                )
                
                # Don't wait for completion - signals will be processed asynchronously
                
            except Exception as e:
                logger.error(
                    "Error updating market data for strategy",
                    strategy_id=strategy.strategy_id,
                    symbol=symbol,
                    error=str(e)
                )
                self._handle_strategy_error(strategy.strategy_id, e)
    
    @timing()
    def _generate_strategy_signals(
        self,
        strategy: BaseStrategy,
        current_data: Dict[str, BarData],
        portfolio: PortfolioSnapshot
    ):
        """Generate signals for a strategy.
        
        Args:
            strategy: Strategy to generate signals for
            current_data: Current market data
            portfolio: Portfolio snapshot
        """
        start_time = datetime.now()
        
        try:
            # Generate signals
            signals = strategy.generate_signals(current_data, portfolio)
            
            # Process each signal
            for signal in signals:
                strategy.process_signal(signal)
                self.signal_queue.put(signal)
            
            # Update performance tracking
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            perf = self.strategy_performance[strategy.strategy_id]
            perf['signals_generated'] += len(signals)
            perf['execution_time_ms'].append(execution_time)
            perf['last_execution'] = datetime.now().isoformat()
            
            # Keep only recent execution times
            if len(perf['execution_time_ms']) > 100:
                perf['execution_time_ms'] = perf['execution_time_ms'][-100:]
            
            if signals:
                logger.debug(
                    "Signals generated",
                    strategy_id=strategy.strategy_id,
                    signals_count=len(signals),
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            logger.error(
                "Error generating signals",
                strategy_id=strategy.strategy_id,
                error=str(e)
            )
            self._handle_strategy_error(strategy.strategy_id, e)
    
    def _process_signals(self):
        """Process signals from the queue."""
        logger.info("Signal processing thread started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Get signal from queue with timeout
                signal = self.signal_queue.get(timeout=1.0)
                
                # Notify callbacks
                for callback in self.signal_callbacks:
                    try:
                        callback(signal)
                    except Exception as e:
                        logger.error(
                            "Error in signal callback",
                            callback=callback.__name__ if hasattr(callback, '__name__') else str(callback),
                            error=str(e)
                        )
                
                self.total_signals_processed += 1
                
                logger.debug(
                    "Signal processed",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.value
                )
                
            except Empty:
                # No signals in queue, continue
                continue
            except Exception as e:
                logger.error("Error processing signal", error=str(e))
        
        logger.info("Signal processing thread stopped")
    
    def _handle_strategy_error(self, strategy_id: str, error: Exception):
        """Handle strategy errors.
        
        Args:
            strategy_id: ID of strategy that had an error
            error: Exception that occurred
        """
        # Update error count
        if strategy_id in self.strategy_performance:
            self.strategy_performance[strategy_id]['errors'] += 1
        
        # Set strategy state to error
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.state = StrategyState.ERROR
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(strategy_id, error)
            except Exception as e:
                logger.error(
                    "Error in error callback",
                    callback=callback.__name__ if hasattr(callback, '__name__') else str(callback),
                    error=str(e)
                )
    
    def add_signal_callback(self, callback: Callable[[StrategySignal], None]):
        """Add a callback for signal processing.
        
        Args:
            callback: Function to call when signals are generated
        """
        self.signal_callbacks.append(callback)
        logger.debug("Signal callback added", total_callbacks=len(self.signal_callbacks))
    
    def remove_signal_callback(self, callback: Callable[[StrategySignal], None]):
        """Remove a signal callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
            logger.debug("Signal callback removed", total_callbacks=len(self.signal_callbacks))
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """Add a callback for error handling.
        
        Args:
            callback: Function to call when errors occur
        """
        self.error_callbacks.append(callback)
        logger.debug("Error callback added", total_callbacks=len(self.error_callbacks))
    
    def remove_error_callback(self, callback: Callable[[str, Exception], None]):
        """Remove an error callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
            logger.debug("Error callback removed", total_callbacks=len(self.error_callbacks))
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get manager status information.
        
        Returns:
            Dictionary with manager status
        """
        active_strategies = sum(
            1 for strategy in self.strategies.values()
            if strategy.state == StrategyState.ACTIVE
        )
        
        return {
            'is_running': self.is_running,
            'total_strategies': len(self.strategies),
            'active_strategies': active_strategies,
            'total_signals_processed': self.total_signals_processed,
            'signal_queue_size': self.signal_queue.qsize(),
            'max_workers': self.max_workers,
            'signal_callbacks': len(self.signal_callbacks),
            'error_callbacks': len(self.error_callbacks)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies.
        
        Returns:
            Performance summary dictionary
        """
        summary = {
            'total_strategies': len(self.strategies),
            'total_signals': self.total_signals_processed,
            'strategies': {}
        }
        
        for strategy_id, strategy in self.strategies.items():
            perf = self.strategy_performance[strategy_id]
            
            summary['strategies'][strategy_id] = {
                'name': strategy.name,
                'state': strategy.state.value,
                'signals_generated': perf['signals_generated'],
                'errors': perf['errors'],
                'avg_execution_time_ms': (
                    sum(perf['execution_time_ms']) / len(perf['execution_time_ms'])
                    if perf['execution_time_ms'] else 0
                ),
                'performance_metrics': strategy.performance.to_dict()
            }
        
        return summary
    
    def reset_all_strategies(self):
        """Reset all strategies to initial state."""
        for strategy_id, strategy in self.strategies.items():
            try:
                strategy.reset()
                
                # Reset performance tracking
                self.strategy_performance[strategy_id] = {
                    'signals_generated': 0,
                    'execution_time_ms': [],
                    'errors': 0,
                    'last_execution': None
                }
                
                logger.info("Strategy reset", strategy_id=strategy_id)
                
            except Exception as e:
                logger.error(
                    "Error resetting strategy",
                    strategy_id=strategy_id,
                    error=str(e)
                )
        
        # Reset manager stats
        self.total_signals_processed = 0
        
        # Clear signal queue
        while not self.signal_queue.empty():
            try:
                self.signal_queue.get_nowait()
            except Empty:
                break
        
        logger.info("All strategies reset")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_manager()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_manager()