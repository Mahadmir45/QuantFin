"""Event-driven backtesting engine."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Trading order."""
    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: str = field(default_factory=lambda: str(np.random.randint(1000000)))


@dataclass
class Fill:
    """Order fill."""
    order_id: str
    ticker: str
    side: OrderSide
    quantity: float
    fill_price: float
    timestamp: datetime
    commission: float = 0.0


@dataclass
class Position:
    """Portfolio position."""
    ticker: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    
    @property
    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price
    
    @property
    def unrealized_pnl(self, current_price: float) -> float:
        return self.quantity * (current_price - self.avg_cost)


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Parameters:
    -----------
    initial_capital : float
        Starting capital
    commission : float
        Commission per trade (fraction)
    slippage : float
        Slippage per trade (fraction)
    """
    
    def __init__(self, initial_capital: float = 100000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.fills: List[Fill] = []
        
        self.equity_curve: List[Dict] = []
        self.current_time: Optional[datetime] = None
    
    def submit_order(self, order: Order):
        """Submit an order."""
        self.orders.append(order)
    
    def process_orders(self, prices: pd.Series):
        """Process pending orders."""
        for order in self.orders[:]:
            if order.ticker not in prices.index:
                continue
            
            price = prices[order.ticker]
            
            # Apply slippage
            if order.side == OrderSide.BUY:
                fill_price = price * (1 + self.slippage)
            else:
                fill_price = price * (1 - self.slippage)
            
            # Calculate commission
            trade_value = abs(order.quantity) * fill_price
            commission = trade_value * self.commission
            
            # Create fill
            fill = Fill(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                timestamp=self.current_time,
                commission=commission
            )
            self.fills.append(fill)
            
            # Update position
            self._update_position(fill)
            
            # Remove order
            self.orders.remove(order)
    
    def _update_position(self, fill: Fill):
        """Update position after fill."""
        if fill.ticker not in self.positions:
            self.positions[fill.ticker] = Position(fill.ticker)
        
        pos = self.positions[fill.ticker]
        
        if fill.side == OrderSide.BUY:
            # Update average cost
            total_cost = pos.quantity * pos.avg_cost + fill.quantity * fill.fill_price
            pos.quantity += fill.quantity
            if pos.quantity > 0:
                pos.avg_cost = total_cost / pos.quantity
            
            # Deduct cash
            self.cash -= fill.quantity * fill.fill_price + fill.commission
        else:
            # Reduce position
            pos.quantity -= fill.quantity
            
            # Add cash
            self.cash += fill.quantity * fill.fill_price - fill.commission
    
    def update_equity(self, prices: pd.Series):
        """Update equity curve."""
        position_value = sum(
            pos.quantity * prices.get(ticker, 0)
            for ticker, pos in self.positions.items()
        )
        
        total_equity = self.cash + position_value
        
        self.equity_curve.append({
            'timestamp': self.current_time,
            'cash': self.cash,
            'position_value': position_value,
            'total_equity': total_equity
        })
    
    def run(self, 
           prices: pd.DataFrame,
           signal_generator: Callable,
           rebalance_freq: str = 'D') -> pd.DataFrame:
        """
        Run backtest.
        
        Parameters:
        -----------
        prices : DataFrame
            Price data (tickers as columns)
        signal_generator : callable
            Function that takes (current_time, prices_history) and returns signals
        rebalance_freq : str
            Rebalancing frequency
        
        Returns:
        --------
        DataFrame : Backtest results
        """
        # Generate rebalancing dates
        if rebalance_freq == 'D':
            rebalance_dates = prices.index
        else:
            rebalance_dates = prices.resample(rebalance_freq).last().index
        
        for i, timestamp in enumerate(prices.index):
            self.current_time = timestamp
            current_prices = prices.loc[timestamp]
            
            # Process any pending orders
            self.process_orders(current_prices)
            
            # Generate signals on rebalance dates
            if timestamp in rebalance_dates:
                hist_prices = prices.iloc[:i+1]
                signals = signal_generator(timestamp, hist_prices)
                
                # Create orders from signals
                for ticker, target_position in signals.items():
                    if ticker not in current_prices.index:
                        continue
                    
                    current_qty = self.positions.get(ticker, Position(ticker)).quantity
                    delta = target_position - current_qty
                    
                    if abs(delta) > 0:
                        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                        order = Order(
                            ticker=ticker,
                            side=side,
                            quantity=abs(delta),
                            timestamp=timestamp
                        )
                        self.submit_order(order)
                
                # Process orders immediately for simplicity
                self.process_orders(current_prices)
            
            # Update equity
            self.update_equity(current_prices)
        
        return pd.DataFrame(self.equity_curve)
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        returns = equity_df['total_equity'].pct_change().dropna()
        
        from ..core.utils import sharpe_ratio, maximum_drawdown, calmar_ratio
        
        return {
            'total_return': (equity_df['total_equity'].iloc[-1] / self.initial_capital) - 1,
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': sharpe_ratio(returns),
            'max_drawdown': maximum_drawdown(equity_df['total_equity'] / self.initial_capital),
            'calmar_ratio': calmar_ratio(returns),
            'final_equity': equity_df['total_equity'].iloc[-1]
        }