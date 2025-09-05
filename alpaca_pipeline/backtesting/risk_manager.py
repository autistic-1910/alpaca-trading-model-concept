"""Risk management for backtesting and live trading."""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils import get_logger, safe_divide
from .portfolio import Portfolio, Position
from .order_manager import Order, OrderSide, OrderType

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    # Portfolio risk
    portfolio_var: float = 0.0  # Value at Risk
    portfolio_cvar: float = 0.0  # Conditional Value at Risk
    portfolio_beta: float = 0.0
    portfolio_volatility: float = 0.0
    
    # Position risk
    max_position_size: float = 0.0
    position_concentration: float = 0.0
    sector_concentration: Dict[str, float] = None
    
    # Drawdown risk
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: int = 0
    
    # Leverage and margin
    leverage_ratio: float = 0.0
    margin_usage: float = 0.0
    buying_power_usage: float = 0.0
    
    # Risk level
    overall_risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0
    
    def __post_init__(self):
        if self.sector_concentration is None:
            self.sector_concentration = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'portfolio_var': self.portfolio_var,
            'portfolio_cvar': self.portfolio_cvar,
            'portfolio_beta': self.portfolio_beta,
            'portfolio_volatility': self.portfolio_volatility,
            'max_position_size': self.max_position_size,
            'position_concentration': self.position_concentration,
            'sector_concentration': self.sector_concentration,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'leverage_ratio': self.leverage_ratio,
            'margin_usage': self.margin_usage,
            'buying_power_usage': self.buying_power_usage,
            'overall_risk_level': self.overall_risk_level.value,
            'risk_score': self.risk_score
        }


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Position limits
    max_position_size: float = 0.1  # 10% of portfolio
    max_sector_exposure: float = 0.3  # 30% per sector
    max_single_loss: float = 0.02  # 2% max loss per trade
    
    # Portfolio limits
    max_portfolio_risk: float = 0.15  # 15% portfolio VaR
    max_drawdown_limit: float = 0.2  # 20% max drawdown
    max_leverage: float = 2.0  # 2x leverage
    max_correlation: float = 0.7  # Max correlation between positions
    
    # Daily limits
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_daily_trades: int = 100
    max_daily_volume: float = 1000000  # $1M daily volume
    
    # Stop loss and take profit
    default_stop_loss: float = 0.05  # 5% stop loss
    default_take_profit: float = 0.15  # 15% take profit
    trailing_stop: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_position_size': self.max_position_size,
            'max_sector_exposure': self.max_sector_exposure,
            'max_single_loss': self.max_single_loss,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_drawdown_limit': self.max_drawdown_limit,
            'max_leverage': self.max_leverage,
            'max_correlation': self.max_correlation,
            'max_daily_loss': self.max_daily_loss,
            'max_daily_trades': self.max_daily_trades,
            'max_daily_volume': self.max_daily_volume,
            'default_stop_loss': self.default_stop_loss,
            'default_take_profit': self.default_take_profit,
            'trailing_stop': self.trailing_stop
        }


class RiskManager:
    """Manages risk for trading operations."""
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        """Initialize risk manager.
        
        Args:
            risk_limits: Risk limits configuration
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.daily_stats = {
            'trades_count': 0,
            'total_volume': 0.0,
            'daily_pnl': 0.0,
            'start_equity': 0.0
        }
        self.risk_violations = []
        
        logger.info(
            "RiskManager initialized",
            max_position_size=self.risk_limits.max_position_size,
            max_drawdown=self.risk_limits.max_drawdown_limit
        )
    
    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: float
    ) -> Tuple[bool, str]:
        """Validate order against risk limits.
        
        Args:
            order: Order to validate
            portfolio: Current portfolio state
            current_price: Current market price
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check daily trade limit
        if self.daily_stats['trades_count'] >= self.risk_limits.max_daily_trades:
            return False, "Daily trade limit exceeded"
        
        # Check daily volume limit
        order_value = order.quantity * current_price
        if self.daily_stats['total_volume'] + order_value > self.risk_limits.max_daily_volume:
            return False, "Daily volume limit exceeded"
        
        # Check position size limit
        if not self._check_position_size_limit(order, portfolio, current_price):
            return False, "Position size limit exceeded"
        
        # Check portfolio drawdown
        if not self._check_drawdown_limit(portfolio):
            return False, "Portfolio drawdown limit exceeded"
        
        # Check daily loss limit
        if not self._check_daily_loss_limit(portfolio):
            return False, "Daily loss limit exceeded"
        
        # Check buying power
        if not self._check_buying_power(order, portfolio, current_price):
            return False, "Insufficient buying power"
        
        # Check leverage limit
        if not self._check_leverage_limit(order, portfolio, current_price):
            return False, "Leverage limit exceeded"
        
        return True, "Order validated"
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        portfolio: Portfolio,
        risk_per_trade: Optional[float] = None
    ) -> int:
        """Calculate optimal position size based on risk management.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            portfolio: Current portfolio
            risk_per_trade: Risk per trade (default from limits)
            
        Returns:
            Position size in shares
        """
        if risk_per_trade is None:
            risk_per_trade = self.risk_limits.max_single_loss
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share <= 0:
            logger.warning(f"Invalid risk per share for {symbol}: {risk_per_share}")
            return 0
        
        # Calculate dollar risk
        portfolio_value = portfolio.total_equity
        dollar_risk = portfolio_value * risk_per_trade
        
        # Calculate position size
        position_size = int(dollar_risk / risk_per_share)
        
        # Apply position size limit
        max_position_value = portfolio_value * self.risk_limits.max_position_size
        max_shares = int(max_position_value / entry_price)
        
        position_size = min(position_size, max_shares)
        
        # Ensure minimum position size
        position_size = max(position_size, 0)
        
        logger.info(
            "Position size calculated",
            symbol=symbol,
            position_size=position_size,
            risk_per_share=risk_per_share,
            dollar_risk=dollar_risk
        )
        
        return position_size
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: OrderSide,
        atr: Optional[float] = None,
        volatility_multiplier: float = 2.0
    ) -> float:
        """Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            side: Order side (buy/sell)
            atr: Average True Range for volatility-based stops
            volatility_multiplier: Multiplier for ATR-based stops
            
        Returns:
            Stop loss price
        """
        if atr and atr > 0:
            # ATR-based stop loss
            stop_distance = atr * volatility_multiplier
        else:
            # Percentage-based stop loss
            stop_distance = entry_price * self.risk_limits.default_stop_loss
        
        if side == OrderSide.BUY:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return max(stop_loss, 0.01)  # Ensure positive price
    
    def calculate_take_profit(
        self,
        entry_price: float,
        side: OrderSide,
        risk_reward_ratio: float = 3.0
    ) -> float:
        """Calculate take profit price.
        
        Args:
            entry_price: Entry price
            side: Order side
            risk_reward_ratio: Risk-reward ratio
            
        Returns:
            Take profit price
        """
        profit_distance = entry_price * self.risk_limits.default_take_profit
        
        if side == OrderSide.BUY:
            take_profit = entry_price + profit_distance
        else:
            take_profit = entry_price - profit_distance
        
        return max(take_profit, 0.01)
    
    def calculate_risk_metrics(self, portfolio: Portfolio) -> RiskMetrics:
        """Calculate comprehensive risk metrics.
        
        Args:
            portfolio: Portfolio to analyze
            
        Returns:
            Risk metrics
        """
        metrics = RiskMetrics()
        
        # Portfolio value
        portfolio_value = portfolio.total_equity
        
        if portfolio_value <= 0:
            return metrics
        
        # Position concentration
        position_values = []
        sector_exposure = {}
        
        for position in portfolio.positions.values():
            if position.quantity != 0:
                position_value = abs(position.market_value)
                position_values.append(position_value)
                
                # Sector concentration (simplified - would need asset metadata)
                sector = "Unknown"  # Would get from asset info
                sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value
        
        if position_values:
            metrics.max_position_size = max(position_values) / portfolio_value
            metrics.position_concentration = sum(v**2 for v in position_values) / (sum(position_values)**2)
        
        # Sector concentration
        total_exposure = sum(sector_exposure.values())
        if total_exposure > 0:
            metrics.sector_concentration = {
                sector: exposure / portfolio_value
                for sector, exposure in sector_exposure.items()
            }
        
        # Drawdown calculation
        equity_curve = portfolio.get_equity_curve()
        if len(equity_curve) >= 2:
            peak = max(point[1] for point in equity_curve)
            current = equity_curve[-1][1]
            
            metrics.current_drawdown = (peak - current) / peak if peak > 0 else 0
            metrics.max_drawdown = portfolio.max_drawdown
        
        # Leverage calculation
        total_position_value = sum(abs(pos.market_value) for pos in portfolio.positions.values())
        metrics.leverage_ratio = total_position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Buying power usage
        metrics.buying_power_usage = (portfolio_value - portfolio.cash) / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate overall risk score
        metrics.risk_score = self._calculate_risk_score(metrics)
        metrics.overall_risk_level = self._determine_risk_level(metrics.risk_score)
        
        logger.info(
            "Risk metrics calculated",
            risk_score=metrics.risk_score,
            risk_level=metrics.overall_risk_level.value,
            current_drawdown=metrics.current_drawdown
        )
        
        return metrics
    
    def update_daily_stats(self, portfolio: Portfolio):
        """Update daily statistics.
        
        Args:
            portfolio: Current portfolio
        """
        current_date = datetime.now().date()
        
        # Reset daily stats if new day
        if not hasattr(self, '_last_update_date') or self._last_update_date != current_date:
            self.daily_stats = {
                'trades_count': 0,
                'total_volume': 0.0,
                'daily_pnl': 0.0,
                'start_equity': portfolio.total_equity
            }
            self._last_update_date = current_date
        
        # Update daily P&L
        self.daily_stats['daily_pnl'] = portfolio.total_equity - self.daily_stats['start_equity']
    
    def record_trade(self, order: Order, fill_price: float):
        """Record a trade for daily statistics.
        
        Args:
            order: Executed order
            fill_price: Fill price
        """
        self.daily_stats['trades_count'] += 1
        self.daily_stats['total_volume'] += order.quantity * fill_price
        
        logger.debug(
            "Trade recorded",
            order_id=order.order_id,
            daily_trades=self.daily_stats['trades_count'],
            daily_volume=self.daily_stats['total_volume']
        )
    
    def check_risk_violations(self, portfolio: Portfolio) -> List[str]:
        """Check for risk violations.
        
        Args:
            portfolio: Portfolio to check
            
        Returns:
            List of violation messages
        """
        violations = []
        metrics = self.calculate_risk_metrics(portfolio)
        
        # Check drawdown limit
        if metrics.current_drawdown > self.risk_limits.max_drawdown_limit:
            violations.append(f"Drawdown limit exceeded: {metrics.current_drawdown:.2%} > {self.risk_limits.max_drawdown_limit:.2%}")
        
        # Check position size limit
        if metrics.max_position_size > self.risk_limits.max_position_size:
            violations.append(f"Position size limit exceeded: {metrics.max_position_size:.2%} > {self.risk_limits.max_position_size:.2%}")
        
        # Check leverage limit
        if metrics.leverage_ratio > self.risk_limits.max_leverage:
            violations.append(f"Leverage limit exceeded: {metrics.leverage_ratio:.2f}x > {self.risk_limits.max_leverage:.2f}x")
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_stats['daily_pnl']) / self.daily_stats['start_equity'] if self.daily_stats['start_equity'] > 0 else 0
        if self.daily_stats['daily_pnl'] < 0 and daily_loss_pct > self.risk_limits.max_daily_loss:
            violations.append(f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.risk_limits.max_daily_loss:.2%}")
        
        # Check sector concentration
        for sector, exposure in metrics.sector_concentration.items():
            if exposure > self.risk_limits.max_sector_exposure:
                violations.append(f"Sector exposure limit exceeded for {sector}: {exposure:.2%} > {self.risk_limits.max_sector_exposure:.2%}")
        
        if violations:
            self.risk_violations.extend(violations)
            logger.warning(f"Risk violations detected: {len(violations)} violations")
        
        return violations
    
    def _check_position_size_limit(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: float
    ) -> bool:
        """Check if order violates position size limit."""
        order_value = order.quantity * current_price
        position_size_pct = order_value / portfolio.total_equity
        
        return position_size_pct <= self.risk_limits.max_position_size
    
    def _check_drawdown_limit(self, portfolio: Portfolio) -> bool:
        """Check if portfolio exceeds drawdown limit."""
        equity_curve = portfolio.get_equity_curve()
        
        if len(equity_curve) < 2:
            return True
        
        peak = max(point[1] for point in equity_curve)
        current = equity_curve[-1][1]
        
        current_drawdown = (peak - current) / peak if peak > 0 else 0
        
        return current_drawdown <= self.risk_limits.max_drawdown_limit
    
    def _check_daily_loss_limit(self, portfolio: Portfolio) -> bool:
        """Check if daily loss exceeds limit."""
        if self.daily_stats['start_equity'] <= 0:
            return True
        
        daily_loss_pct = abs(self.daily_stats['daily_pnl']) / self.daily_stats['start_equity']
        
        return not (self.daily_stats['daily_pnl'] < 0 and daily_loss_pct > self.risk_limits.max_daily_loss)
    
    def _check_buying_power(self, order: Order, portfolio: Portfolio, current_price: float) -> bool:
        """Check if sufficient buying power exists."""
        if order.side == OrderSide.SELL:
            return True  # Selling doesn't require buying power
        
        order_value = order.quantity * current_price
        return portfolio.cash >= order_value
    
    def _check_leverage_limit(self, order: Order, portfolio: Portfolio, current_price: float) -> bool:
        """Check if order would exceed leverage limit."""
        order_value = order.quantity * current_price
        current_exposure = sum(abs(pos.market_value) for pos in portfolio.positions.values())
        
        if order.side == OrderSide.BUY:
            new_exposure = current_exposure + order_value
        else:
            new_exposure = current_exposure  # Selling reduces exposure
        
        new_leverage = new_exposure / portfolio.total_equity if portfolio.total_equity > 0 else 0
        
        return new_leverage <= self.risk_limits.max_leverage
    
    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """Calculate overall risk score (0-100)."""
        score = 0.0
        
        # Drawdown component (0-30 points)
        drawdown_score = min(metrics.current_drawdown / self.risk_limits.max_drawdown_limit, 1.0) * 30
        score += drawdown_score
        
        # Position concentration component (0-25 points)
        concentration_score = min(metrics.max_position_size / self.risk_limits.max_position_size, 1.0) * 25
        score += concentration_score
        
        # Leverage component (0-25 points)
        leverage_score = min(metrics.leverage_ratio / self.risk_limits.max_leverage, 1.0) * 25
        score += leverage_score
        
        # Volatility component (0-20 points)
        volatility_score = min(metrics.portfolio_volatility / 0.3, 1.0) * 20  # Assume 30% as high volatility
        score += volatility_score
        
        return min(score, 100.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score."""
        if risk_score < 25:
            return RiskLevel.LOW
        elif risk_score < 50:
            return RiskLevel.MEDIUM
        elif risk_score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def get_risk_summary(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Get comprehensive risk summary.
        
        Args:
            portfolio: Portfolio to analyze
            
        Returns:
            Risk summary dictionary
        """
        metrics = self.calculate_risk_metrics(portfolio)
        violations = self.check_risk_violations(portfolio)
        
        return {
            'risk_metrics': metrics.to_dict(),
            'risk_limits': self.risk_limits.to_dict(),
            'daily_stats': self.daily_stats.copy(),
            'violations': violations,
            'violation_count': len(violations),
            'last_update': datetime.now().isoformat()
        }