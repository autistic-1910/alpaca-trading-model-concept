"""Performance analysis for backtesting results."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..utils import get_logger, calculate_sharpe_ratio, calculate_max_drawdown
from .portfolio import Portfolio
from .order_manager import Order, OrderFill

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_return_mean: float = 0.0
    daily_return_std: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Portfolio metrics
    initial_capital: float = 0.0
    final_capital: float = 0.0
    max_capital: float = 0.0
    min_capital: float = 0.0
    
    # Time metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: int = 0
    
    # Commission and costs
    total_commission: float = 0.0
    commission_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'daily_return_mean': self.daily_return_mean,
            'daily_return_std': self.daily_return_std,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'max_capital': self.max_capital,
            'min_capital': self.min_capital,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'duration_days': self.duration_days,
            'total_commission': self.total_commission,
            'commission_percentage': self.commission_percentage
        }


class PerformanceAnalyzer:
    """Analyzes backtesting performance and calculates metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        
        logger.info(
            "PerformanceAnalyzer initialized",
            risk_free_rate=risk_free_rate
        )
    
    def analyze_portfolio(self, portfolio: Portfolio) -> PerformanceMetrics:
        """Analyze portfolio performance.
        
        Args:
            portfolio: Portfolio to analyze
            
        Returns:
            Performance metrics
        """
        metrics = PerformanceMetrics()
        
        # Get equity curve
        equity_curve = portfolio.get_equity_curve()
        
        if len(equity_curve) < 2:
            logger.warning("Insufficient data for performance analysis")
            return metrics
        
        # Basic metrics
        metrics.initial_capital = portfolio.initial_cash
        metrics.final_capital = portfolio.total_equity
        metrics.max_capital = portfolio.max_equity
        metrics.min_capital = portfolio.min_equity
        metrics.total_commission = portfolio.total_commission
        
        # Time metrics
        metrics.start_date = equity_curve[0][0]
        metrics.end_date = equity_curve[-1][0]
        metrics.duration_days = (metrics.end_date - metrics.start_date).days
        
        # Commission percentage
        if metrics.initial_capital > 0:
            metrics.commission_percentage = metrics.total_commission / metrics.initial_capital
        
        # Calculate returns
        self._calculate_return_metrics(equity_curve, metrics)
        
        # Calculate risk metrics
        self._calculate_risk_metrics(equity_curve, metrics)
        
        # Calculate trade metrics
        self._calculate_trade_metrics(portfolio, metrics)
        
        logger.info(
            "Portfolio analysis completed",
            total_return=metrics.total_return,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            win_rate=metrics.win_rate
        )
        
        return metrics
    
    def _calculate_return_metrics(
        self,
        equity_curve: List[Tuple[datetime, float]],
        metrics: PerformanceMetrics
    ):
        """Calculate return-based metrics."""
        if len(equity_curve) < 2:
            return
        
        # Total return
        initial_value = equity_curve[0][1]
        final_value = equity_curve[-1][1]
        
        if initial_value > 0:
            metrics.total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev_value = equity_curve[i-1][1]
            curr_value = equity_curve[i][1]
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
        
        if not daily_returns:
            return
        
        # Daily return statistics
        metrics.daily_return_mean = np.mean(daily_returns)
        metrics.daily_return_std = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
        
        # Annualized return
        if metrics.duration_days > 0:
            years = metrics.duration_days / 365.25
            if years > 0 and initial_value > 0:
                metrics.annualized_return = ((final_value / initial_value) ** (1 / years)) - 1
        
        # Volatility (annualized)
        metrics.volatility = metrics.daily_return_std * np.sqrt(252)
    
    def _calculate_risk_metrics(
        self,
        equity_curve: List[Tuple[datetime, float]],
        metrics: PerformanceMetrics
    ):
        """Calculate risk-based metrics."""
        if len(equity_curve) < 2:
            return
        
        # Calculate drawdowns
        drawdowns = []
        peak = equity_curve[0][1]
        max_dd = 0.0
        dd_duration = 0
        max_dd_duration = 0
        in_drawdown = False
        
        for timestamp, value in equity_curve:
            if value > peak:
                peak = value
                if in_drawdown:
                    max_dd_duration = max(max_dd_duration, dd_duration)
                    dd_duration = 0
                    in_drawdown = False
            else:
                if not in_drawdown:
                    in_drawdown = True
                    dd_duration = 0
                dd_duration += 1
            
            if peak > 0:
                drawdown = (peak - value) / peak
                drawdowns.append(drawdown)
                max_dd = max(max_dd, drawdown)
        
        metrics.max_drawdown = max_dd
        metrics.max_drawdown_duration = max_dd_duration
        
        # Calculate daily returns for risk metrics
        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev_value = equity_curve[i-1][1]
            curr_value = equity_curve[i][1]
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
        
        if not daily_returns:
            return
        
        # Sharpe ratio
        daily_risk_free = self.risk_free_rate / 252
        excess_returns = [r - daily_risk_free for r in daily_returns]
        
        if len(excess_returns) > 1:
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns, ddof=1)
            
            if std_excess > 0:
                metrics.sharpe_ratio = (mean_excess / std_excess) * np.sqrt(252)
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns:
            metrics.downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(252)
            
            if metrics.downside_deviation > 0:
                metrics.sortino_ratio = (metrics.daily_return_mean * 252) / metrics.downside_deviation
        
        # Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
    
    def _calculate_trade_metrics(self, portfolio: Portfolio, metrics: PerformanceMetrics):
        """Calculate trade-based metrics."""
        # Basic trade counts
        metrics.total_trades = portfolio.total_trades
        metrics.winning_trades = portfolio.winning_trades
        metrics.losing_trades = portfolio.losing_trades
        
        # Win rate
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # Calculate win/loss amounts from positions
        wins = []
        losses = []
        
        for position in portfolio.positions.values():
            if position.realized_pnl > 0:
                wins.append(position.realized_pnl)
            elif position.realized_pnl < 0:
                losses.append(abs(position.realized_pnl))
        
        # Average win/loss
        if wins:
            metrics.avg_win = np.mean(wins)
        
        if losses:
            metrics.avg_loss = np.mean(losses)
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        
        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            metrics.profit_factor = float('inf')
    
    def analyze_trades(self, orders: List[Order]) -> Dict[str, Any]:
        """Analyze individual trades.
        
        Args:
            orders: List of filled orders
            
        Returns:
            Trade analysis dictionary
        """
        filled_orders = [order for order in orders if order.status.value == 'filled']
        
        if not filled_orders:
            return {
                'total_trades': 0,
                'trade_details': [],
                'symbol_breakdown': {},
                'monthly_breakdown': {},
                'trade_duration_stats': {}
            }
        
        # Group trades by symbol
        symbol_trades = defaultdict(list)
        monthly_trades = defaultdict(list)
        
        trade_details = []
        
        for order in filled_orders:
            symbol_trades[order.symbol].append(order)
            
            # Monthly grouping
            month_key = order.filled_at.strftime('%Y-%m') if order.filled_at else 'unknown'
            monthly_trades[month_key].append(order)
            
            # Trade details
            trade_detail = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'avg_fill_price': order.avg_fill_price,
                'total_value': order.quantity * order.avg_fill_price,
                'commission': order.commission,
                'created_at': order.created_at.isoformat(),
                'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                'strategy_id': order.strategy_id
            }
            
            # Calculate duration
            if order.filled_at:
                duration = order.filled_at - order.created_at
                trade_detail['duration_seconds'] = duration.total_seconds()
            
            trade_details.append(trade_detail)
        
        # Symbol breakdown
        symbol_breakdown = {}
        for symbol, trades in symbol_trades.items():
            total_value = sum(order.quantity * order.avg_fill_price for order in trades)
            total_commission = sum(order.commission for order in trades)
            
            symbol_breakdown[symbol] = {
                'trade_count': len(trades),
                'total_value': total_value,
                'total_commission': total_commission,
                'avg_trade_size': total_value / len(trades) if trades else 0
            }
        
        # Monthly breakdown
        monthly_breakdown = {}
        for month, trades in monthly_trades.items():
            total_value = sum(order.quantity * order.avg_fill_price for order in trades)
            total_commission = sum(order.commission for order in trades)
            
            monthly_breakdown[month] = {
                'trade_count': len(trades),
                'total_value': total_value,
                'total_commission': total_commission
            }
        
        # Trade duration statistics
        durations = [
            (order.filled_at - order.created_at).total_seconds()
            for order in filled_orders
            if order.filled_at
        ]
        
        duration_stats = {}
        if durations:
            duration_stats = {
                'mean_seconds': np.mean(durations),
                'median_seconds': np.median(durations),
                'min_seconds': min(durations),
                'max_seconds': max(durations),
                'std_seconds': np.std(durations)
            }
        
        return {
            'total_trades': len(filled_orders),
            'trade_details': trade_details,
            'symbol_breakdown': symbol_breakdown,
            'monthly_breakdown': monthly_breakdown,
            'trade_duration_stats': duration_stats
        }
    
    def calculate_rolling_metrics(
        self,
        equity_curve: List[Tuple[datetime, float]],
        window_days: int = 30
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """Calculate rolling performance metrics.
        
        Args:
            equity_curve: List of (timestamp, equity) tuples
            window_days: Rolling window size in days
            
        Returns:
            Dictionary of rolling metrics
        """
        if len(equity_curve) < window_days:
            return {}
        
        rolling_returns = []
        rolling_volatility = []
        rolling_sharpe = []
        rolling_drawdown = []
        
        for i in range(window_days, len(equity_curve)):
            window_data = equity_curve[i-window_days:i+1]
            
            # Calculate returns for this window
            returns = []
            for j in range(1, len(window_data)):
                prev_val = window_data[j-1][1]
                curr_val = window_data[j][1]
                if prev_val > 0:
                    returns.append((curr_val - prev_val) / prev_val)
            
            if not returns:
                continue
            
            timestamp = window_data[-1][0]
            
            # Rolling return (annualized)
            total_return = (window_data[-1][1] / window_data[0][1]) - 1
            days_in_window = (window_data[-1][0] - window_data[0][0]).days
            if days_in_window > 0:
                annualized_return = ((1 + total_return) ** (365.25 / days_in_window)) - 1
                rolling_returns.append((timestamp, annualized_return))
            
            # Rolling volatility
            if len(returns) > 1:
                volatility = np.std(returns, ddof=1) * np.sqrt(252)
                rolling_volatility.append((timestamp, volatility))
                
                # Rolling Sharpe ratio
                daily_rf = self.risk_free_rate / 252
                excess_returns = [r - daily_rf for r in returns]
                mean_excess = np.mean(excess_returns)
                std_excess = np.std(excess_returns, ddof=1)
                
                if std_excess > 0:
                    sharpe = (mean_excess / std_excess) * np.sqrt(252)
                    rolling_sharpe.append((timestamp, sharpe))
            
            # Rolling max drawdown
            peak = window_data[0][1]
            max_dd = 0.0
            
            for _, value in window_data:
                if value > peak:
                    peak = value
                if peak > 0:
                    drawdown = (peak - value) / peak
                    max_dd = max(max_dd, drawdown)
            
            rolling_drawdown.append((timestamp, max_dd))
        
        return {
            'rolling_returns': rolling_returns,
            'rolling_volatility': rolling_volatility,
            'rolling_sharpe': rolling_sharpe,
            'rolling_drawdown': rolling_drawdown
        }
    
    def compare_to_benchmark(
        self,
        portfolio_equity: List[Tuple[datetime, float]],
        benchmark_data: List[Tuple[datetime, float]]
    ) -> Dict[str, Any]:
        """Compare portfolio performance to a benchmark.
        
        Args:
            portfolio_equity: Portfolio equity curve
            benchmark_data: Benchmark price data
            
        Returns:
            Comparison metrics
        """
        if len(portfolio_equity) < 2 or len(benchmark_data) < 2:
            return {}
        
        # Align data by dates
        portfolio_dict = {ts: value for ts, value in portfolio_equity}
        benchmark_dict = {ts: value for ts, value in benchmark_data}
        
        common_dates = sorted(set(portfolio_dict.keys()) & set(benchmark_dict.keys()))
        
        if len(common_dates) < 2:
            return {}
        
        # Calculate returns
        portfolio_returns = []
        benchmark_returns = []
        
        for i in range(1, len(common_dates)):
            date = common_dates[i]
            prev_date = common_dates[i-1]
            
            # Portfolio return
            port_prev = portfolio_dict[prev_date]
            port_curr = portfolio_dict[date]
            if port_prev > 0:
                port_return = (port_curr - port_prev) / port_prev
                portfolio_returns.append(port_return)
            
            # Benchmark return
            bench_prev = benchmark_dict[prev_date]
            bench_curr = benchmark_dict[date]
            if bench_prev > 0:
                bench_return = (bench_curr - bench_prev) / bench_prev
                benchmark_returns.append(bench_return)
        
        if not portfolio_returns or not benchmark_returns:
            return {}
        
        # Calculate metrics
        portfolio_total_return = (portfolio_dict[common_dates[-1]] / portfolio_dict[common_dates[0]]) - 1
        benchmark_total_return = (benchmark_dict[common_dates[-1]] / benchmark_dict[common_dates[0]]) - 1
        
        # Alpha and Beta calculation
        if len(portfolio_returns) == len(benchmark_returns) and len(portfolio_returns) > 1:
            # Calculate beta
            covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns, ddof=1)
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Calculate alpha
            portfolio_mean = np.mean(portfolio_returns)
            benchmark_mean = np.mean(benchmark_returns)
            daily_rf = self.risk_free_rate / 252
            
            alpha = (portfolio_mean - daily_rf) - beta * (benchmark_mean - daily_rf)
            alpha_annualized = alpha * 252
        else:
            beta = 0
            alpha_annualized = 0
        
        # Tracking error
        if len(portfolio_returns) == len(benchmark_returns):
            excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
            tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252) if len(excess_returns) > 1 else 0
        else:
            tracking_error = 0
        
        # Information ratio
        if tracking_error > 0:
            information_ratio = alpha_annualized / tracking_error
        else:
            information_ratio = 0
        
        return {
            'portfolio_total_return': portfolio_total_return,
            'benchmark_total_return': benchmark_total_return,
            'excess_return': portfolio_total_return - benchmark_total_return,
            'alpha_annualized': alpha_annualized,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'correlation': np.corrcoef(portfolio_returns, benchmark_returns)[0][1] if len(portfolio_returns) == len(benchmark_returns) and len(portfolio_returns) > 1 else 0
        }
    
    def generate_performance_report(self, portfolio: Portfolio, orders: List[Order]) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            portfolio: Portfolio to analyze
            orders: List of orders
            
        Returns:
            Comprehensive performance report
        """
        # Main performance metrics
        metrics = self.analyze_portfolio(portfolio)
        
        # Trade analysis
        trade_analysis = self.analyze_trades(orders)
        
        # Rolling metrics
        equity_curve = portfolio.get_equity_curve()
        rolling_metrics = self.calculate_rolling_metrics(equity_curve, window_days=30)
        
        # Portfolio summary
        portfolio_summary = portfolio.get_portfolio_summary()
        
        # Position summary
        positions_summary = portfolio.get_positions_summary()
        
        return {
            'performance_metrics': metrics.to_dict(),
            'portfolio_summary': portfolio_summary,
            'positions_summary': positions_summary,
            'trade_analysis': trade_analysis,
            'rolling_metrics': {
                key: [(ts.isoformat(), value) for ts, value in values]
                for key, values in rolling_metrics.items()
            },
            'equity_curve': [(ts.isoformat(), equity) for ts, equity in equity_curve],
            'cash_curve': [(ts.isoformat(), cash) for ts, cash in portfolio.get_cash_curve()]
        }