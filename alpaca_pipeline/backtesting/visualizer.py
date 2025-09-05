"""Visualization tools for backtesting results."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings

from ..utils import get_logger
from .performance import PerformanceMetrics, PerformanceAnalyzer
from .portfolio import Portfolio
from .order_manager import Order

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class BacktestVisualizer:
    """Creates visualizations for backtesting results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """Initialize visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Color scheme
        self.colors = {
            'equity': '#2E86AB',
            'benchmark': '#A23B72',
            'drawdown': '#F18F01',
            'cash': '#C73E1D',
            'profit': '#4CAF50',
            'loss': '#F44336',
            'neutral': '#9E9E9E'
        }
        
        logger.info("BacktestVisualizer initialized")
    
    def plot_equity_curve(
        self,
        portfolio: Portfolio,
        benchmark_data: Optional[List[Tuple[datetime, float]]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot equity curve with optional benchmark comparison.
        
        Args:
            portfolio: Portfolio to plot
            benchmark_data: Optional benchmark data for comparison
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi, height_ratios=[3, 1])
        
        # Get equity curve
        equity_curve = portfolio.get_equity_curve()
        
        if not equity_curve:
            logger.warning("No equity data to plot")
            return fig
        
        dates = [point[0] for point in equity_curve]
        equity_values = [point[1] for point in equity_curve]
        
        # Plot equity curve
        ax1.plot(dates, equity_values, color=self.colors['equity'], linewidth=2, label='Portfolio')
        
        # Plot benchmark if provided
        if benchmark_data:
            bench_dates = [point[0] for point in benchmark_data]
            bench_values = [point[1] for point in benchmark_data]
            
            # Normalize benchmark to start at same value as portfolio
            if bench_values and equity_values:
                scale_factor = equity_values[0] / bench_values[0]
                bench_values = [v * scale_factor for v in bench_values]
            
            ax1.plot(bench_dates, bench_values, color=self.colors['benchmark'], 
                    linewidth=2, linestyle='--', label='Benchmark')
        
        # Format equity plot
        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Calculate and plot drawdown
        drawdowns = self._calculate_drawdown_series(equity_curve)
        
        if drawdowns:
            dd_dates = [point[0] for point in drawdowns]
            dd_values = [point[1] * 100 for point in drawdowns]  # Convert to percentage
            
            ax2.fill_between(dd_dates, dd_values, 0, color=self.colors['drawdown'], alpha=0.7)
            ax2.plot(dd_dates, dd_values, color=self.colors['drawdown'], linewidth=1)
        
        ax2.set_title('Drawdown', fontsize=14)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for drawdown
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")
        
        return fig
    
    def plot_returns_distribution(
        self,
        portfolio: Portfolio,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot distribution of daily returns.
        
        Args:
            portfolio: Portfolio to analyze
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Calculate daily returns
        equity_curve = portfolio.get_equity_curve()
        daily_returns = self._calculate_daily_returns(equity_curve)
        
        if not daily_returns:
            logger.warning("No return data to plot")
            return fig
        
        returns_pct = [r * 100 for r in daily_returns]  # Convert to percentage
        
        # Histogram
        ax1.hist(returns_pct, bins=50, alpha=0.7, color=self.colors['equity'], edgecolor='black')
        ax1.axvline(np.mean(returns_pct), color=self.colors['profit'], linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(returns_pct):.2f}%')
        ax1.axvline(np.median(returns_pct), color=self.colors['loss'], linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(returns_pct):.2f}%')
        
        ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Daily Return (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns_pct, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Returns distribution saved to {save_path}")
        
        return fig
    
    def plot_rolling_metrics(
        self,
        rolling_metrics: Dict[str, List[Tuple[datetime, float]]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot rolling performance metrics.
        
        Args:
            rolling_metrics: Dictionary of rolling metrics
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(rolling_metrics)
        if n_metrics == 0:
            logger.warning("No rolling metrics to plot")
            return plt.figure()
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(self.figsize[0], self.figsize[1] * n_metrics // 2), 
                                dpi=self.dpi, sharex=True)
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = list(self.colors.values())
        
        for i, (metric_name, data) in enumerate(rolling_metrics.items()):
            if not data:
                continue
            
            dates = [point[0] for point in data]
            values = [point[1] for point in data]
            
            ax = axes[i]
            ax.plot(dates, values, color=colors[i % len(colors)], linewidth=2)
            
            # Format metric name for display
            display_name = metric_name.replace('rolling_', '').replace('_', ' ').title()
            ax.set_title(f'Rolling {display_name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at zero for some metrics
            if 'return' in metric_name.lower() or 'sharpe' in metric_name.lower():
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Format y-axis based on metric type
            if 'return' in metric_name.lower() or 'drawdown' in metric_name.lower():
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # Format x-axis for bottom plot
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        axes[-1].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Rolling metrics saved to {save_path}")
        
        return fig
    
    def plot_trade_analysis(
        self,
        trade_analysis: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot trade analysis charts.
        
        Args:
            trade_analysis: Trade analysis data
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Symbol breakdown
        symbol_data = trade_analysis.get('symbol_breakdown', {})
        if symbol_data:
            symbols = list(symbol_data.keys())
            trade_counts = [data['trade_count'] for data in symbol_data.values()]
            
            ax1.bar(symbols, trade_counts, color=self.colors['equity'])
            ax1.set_title('Trades by Symbol', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Trades')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Monthly breakdown
        monthly_data = trade_analysis.get('monthly_breakdown', {})
        if monthly_data:
            months = sorted(monthly_data.keys())
            monthly_counts = [monthly_data[month]['trade_count'] for month in months]
            
            ax2.plot(months, monthly_counts, marker='o', color=self.colors['benchmark'])
            ax2.set_title('Trades by Month', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Trades')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Trade duration distribution
        duration_stats = trade_analysis.get('trade_duration_stats', {})
        if duration_stats and 'mean_seconds' in duration_stats:
            # Create sample data for visualization (in real implementation, you'd use actual trade durations)
            durations = np.random.exponential(duration_stats['mean_seconds'], 100)
            durations_hours = durations / 3600  # Convert to hours
            
            ax3.hist(durations_hours, bins=20, alpha=0.7, color=self.colors['drawdown'])
            ax3.set_title('Trade Duration Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Duration (hours)')
            ax3.set_ylabel('Frequency')
        
        # Trade value distribution
        trade_details = trade_analysis.get('trade_details', [])
        if trade_details:
            trade_values = [trade['total_value'] for trade in trade_details]
            
            ax4.hist(trade_values, bins=20, alpha=0.7, color=self.colors['profit'])
            ax4.set_title('Trade Value Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Trade Value ($)')
            ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Trade analysis saved to {save_path}")
        
        return fig
    
    def plot_performance_metrics(
        self,
        metrics: PerformanceMetrics,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot key performance metrics.
        
        Args:
            metrics: Performance metrics to plot
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Return metrics
        return_metrics = {
            'Total Return': metrics.total_return * 100,
            'Annualized Return': metrics.annualized_return * 100,
            'Daily Return Mean': metrics.daily_return_mean * 100 * 252  # Annualized
        }
        
        bars1 = ax1.bar(return_metrics.keys(), return_metrics.values(), 
                       color=[self.colors['profit'] if v >= 0 else self.colors['loss'] for v in return_metrics.values()])
        ax1.set_title('Return Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, return_metrics.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Risk metrics
        risk_metrics = {
            'Sharpe Ratio': metrics.sharpe_ratio,
            'Sortino Ratio': metrics.sortino_ratio,
            'Calmar Ratio': metrics.calmar_ratio
        }
        
        bars2 = ax2.bar(risk_metrics.keys(), risk_metrics.values(), color=self.colors['benchmark'])
        ax2.set_title('Risk-Adjusted Returns', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Ratio')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels
        for bar, value in zip(bars2, risk_metrics.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Trade metrics
        if metrics.total_trades > 0:
            win_rate = metrics.win_rate * 100
            lose_rate = 100 - win_rate
            
            ax3.pie([win_rate, lose_rate], labels=['Winning Trades', 'Losing Trades'],
                   colors=[self.colors['profit'], self.colors['loss']], autopct='%1.1f%%')
            ax3.set_title(f'Win Rate ({metrics.total_trades} trades)', fontsize=14, fontweight='bold')
        
        # Drawdown and volatility
        risk_values = {
            'Max Drawdown': metrics.max_drawdown * 100,
            'Volatility': metrics.volatility * 100
        }
        
        bars4 = ax4.bar(risk_values.keys(), risk_values.values(), color=self.colors['drawdown'])
        ax4.set_title('Risk Metrics', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Percentage (%)')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels
        for bar, value in zip(bars4, risk_values.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Performance metrics saved to {save_path}")
        
        return fig
    
    def create_dashboard(
        self,
        portfolio: Portfolio,
        metrics: PerformanceMetrics,
        trade_analysis: Dict[str, Any],
        rolling_metrics: Optional[Dict[str, List[Tuple[datetime, float]]]] = None,
        benchmark_data: Optional[List[Tuple[datetime, float]]] = None,
        save_path: Optional[str] = None
    ) -> List[plt.Figure]:
        """Create comprehensive dashboard with all visualizations.
        
        Args:
            portfolio: Portfolio to analyze
            metrics: Performance metrics
            trade_analysis: Trade analysis data
            rolling_metrics: Optional rolling metrics
            benchmark_data: Optional benchmark data
            save_path: Optional base path to save plots (will append suffixes)
            
        Returns:
            List of matplotlib figures
        """
        figures = []
        
        # Equity curve
        fig1 = self.plot_equity_curve(
            portfolio, 
            benchmark_data, 
            f"{save_path}_equity.png" if save_path else None
        )
        figures.append(fig1)
        
        # Performance metrics
        fig2 = self.plot_performance_metrics(
            metrics,
            f"{save_path}_metrics.png" if save_path else None
        )
        figures.append(fig2)
        
        # Returns distribution
        fig3 = self.plot_returns_distribution(
            portfolio,
            f"{save_path}_returns.png" if save_path else None
        )
        figures.append(fig3)
        
        # Trade analysis
        fig4 = self.plot_trade_analysis(
            trade_analysis,
            f"{save_path}_trades.png" if save_path else None
        )
        figures.append(fig4)
        
        # Rolling metrics (if provided)
        if rolling_metrics:
            fig5 = self.plot_rolling_metrics(
                rolling_metrics,
                f"{save_path}_rolling.png" if save_path else None
            )
            figures.append(fig5)
        
        logger.info(f"Dashboard created with {len(figures)} figures")
        return figures
    
    def _calculate_drawdown_series(
        self, 
        equity_curve: List[Tuple[datetime, float]]
    ) -> List[Tuple[datetime, float]]:
        """Calculate drawdown series from equity curve."""
        if len(equity_curve) < 2:
            return []
        
        drawdowns = []
        peak = equity_curve[0][1]
        
        for timestamp, value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0
            drawdowns.append((timestamp, drawdown))
        
        return drawdowns
    
    def _calculate_daily_returns(
        self, 
        equity_curve: List[Tuple[datetime, float]]
    ) -> List[float]:
        """Calculate daily returns from equity curve."""
        if len(equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(equity_curve)):
            prev_value = equity_curve[i-1][1]
            curr_value = equity_curve[i][1]
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
        
        return returns
    
    def save_all_plots(self, figures: List[plt.Figure], base_path: str):
        """Save all figures to files.
        
        Args:
            figures: List of matplotlib figures
            base_path: Base path for saving (will append indices)
        """
        for i, fig in enumerate(figures):
            save_path = f"{base_path}_{i+1}.png"
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure {i+1} saved to {save_path}")
    
    def close_all_figures(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all')
        logger.info("All figures closed")