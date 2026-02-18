"""Data providers and caching for market data."""

import os
import time
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProvider:
    """Base class for data providers."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_prices(self, tickers: List[str], 
                   start_date: str, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """Get price data for tickers."""
        raise NotImplementedError
    
    def _get_cache_path(self, ticker: str) -> str:
        """Get cache file path for ticker."""
        return os.path.join(self.cache_dir, f"{ticker}_prices.csv")
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load cached data if exists."""
        cache_path = self._get_cache_path(ticker)
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {ticker} from cache")
            return df
        return None
    
    def _save_to_cache(self, ticker: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_path = self._get_cache_path(ticker)
        data.to_csv(cache_path)
        logger.info(f"Saved {ticker} to cache")


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance."""
    
    def __init__(self, cache_dir: str = "./cache"):
        super().__init__(cache_dir)
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance required. Install with: pip install yfinance")
    
    def get_prices(self, tickers: List[str],
                   start_date: str,
                   end_date: Optional[str] = None,
                   auto_adjust: bool = True,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Get price data from Yahoo Finance.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str, optional
            End date (YYYY-MM-DD)
        auto_adjust : bool
            Adjust for splits and dividends
        use_cache : bool
            Use cached data if available
        
        Returns:
        --------
        DataFrame : Price data with tickers as columns
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        all_data = {}
        
        for ticker in tickers:
            # Try cache first
            if use_cache:
                cached = self._load_from_cache(ticker)
                if cached is not None:
                    cached = cached.loc[start_date:end_date]
                    if len(cached) > 0:
                        all_data[ticker] = cached['Close']
                        continue
            
            # Fetch from Yahoo Finance
            try:
                data = self.yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date,
                    auto_adjust=auto_adjust,
                    progress=False,
                    threads=False
                )
                
                if not data.empty:
                    all_data[ticker] = data['Close']
                    if use_cache:
                        self._save_to_cache(ticker, data)
                    logger.info(f"Fetched {ticker}: {len(data)} rows")
                else:
                    logger.warning(f"No data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data fetched for any ticker")
        
        return pd.DataFrame(all_data)
    
    def get_returns(self, tickers: List[str],
                    start_date: str,
                    end_date: Optional[str] = None,
                    periods_per_year: int = 252) -> pd.DataFrame:
        """
        Get returns data for tickers.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date
        end_date : str, optional
            End date
        periods_per_year : int
            Trading periods per year
        
        Returns:
        --------
        DataFrame : Returns data
        """
        prices = self.get_prices(tickers, start_date, end_date)
        returns = prices.pct_change().dropna()
        return returns


class PolygonProvider(DataProvider):
    """Polygon.io data provider."""
    
    def __init__(self, api_key: str, cache_dir: str = "./cache"):
        super().__init__(cache_dir)
        self.api_key = api_key
        try:
            from polygon import RESTClient
            self.client = RESTClient(api_key)
        except ImportError:
            raise ImportError("polygon-api-client required. Install with: pip install polygon-api-client")
    
    def get_prices(self, tickers: List[str],
                   start_date: str,
                   end_date: Optional[str] = None,
                   use_cache: bool = True,
                   rate_limit_delay: float = 12.0) -> pd.DataFrame:
        """
        Get price data from Polygon.io.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str, optional
            End date (YYYY-MM-DD)
        rate_limit_delay : float
            Delay between API calls (seconds)
        
        Returns:
        --------
        DataFrame : Price data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        all_data = {}
        
        for ticker in tickers:
            # Try cache first
            if use_cache:
                cached = self._load_from_cache(ticker)
                if cached is not None:
                    cached = cached.loc[start_date:end_date]
                    if len(cached) > 0:
                        all_data[ticker] = cached['close']
                        continue
            
            # Fetch from Polygon
            try:
                aggs = []
                for a in self.client.list_aggs(
                    ticker, 1, 'day', start_date, end_date, limit=50000
                ):
                    aggs.append({
                        'timestamp': pd.to_datetime(a.timestamp, unit='ms'),
                        'close': a.close
                    })
                
                if aggs:
                    df = pd.DataFrame(aggs)
                    df.set_index('timestamp', inplace=True)
                    all_data[ticker] = df['close']
                    if use_cache:
                        self._save_to_cache(ticker, df)
                    logger.info(f"Fetched {ticker}: {len(df)} rows")
                
                time.sleep(rate_limit_delay)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data fetched for any ticker")
        
        return pd.DataFrame(all_data)


class SyntheticDataProvider(DataProvider):
    """Generate synthetic price data for testing."""
    
    def get_prices(self, tickers: List[str],
                   start_date: str,
                   end_date: Optional[str] = None,
                   seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic price data using GBM.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols (used for column names)
        start_date : str
            Start date
        end_date : str, optional
            End date
        seed : int
            Random seed
        
        Returns:
        --------
        DataFrame : Synthetic price data
        """
        np.random.seed(seed)
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        
        # Generate business days
        dates = pd.bdate_range(start, end)
        n_days = len(dates)
        
        data = {}
        for ticker in tickers:
            # Random parameters for each ticker
            mu = np.random.uniform(0.05, 0.15) / 252  # Annual drift
            sigma = np.random.uniform(0.15, 0.35) / np.sqrt(252)  # Annual vol
            S0 = np.random.uniform(50, 500)  # Starting price
            
            # Generate GBM paths
            dt = 1 / 252
            returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
            prices = S0 * np.exp(np.cumsum(returns))
            
            data[ticker] = prices
        
        return pd.DataFrame(data, index=dates)
    
    def get_returns(self, tickers: List[str],
                    start_date: str,
                    end_date: Optional[str] = None,
                    seed: int = 42) -> pd.DataFrame:
        """Get synthetic returns."""
        prices = self.get_prices(tickers, start_date, end_date, seed)
        return prices.pct_change().dropna()


class DataManager:
    """High-level data management interface."""
    
    def __init__(self, provider: Optional[DataProvider] = None):
        """
        Initialize data manager.
        
        Parameters:
        -----------
        provider : DataProvider, optional
            Data provider instance. If None, uses YahooFinanceProvider.
        """
        if provider is None:
            self.provider = YahooFinanceProvider()
        else:
            self.provider = provider
    
    def get_universe_data(self, universe: str = 'spy500',
                          start_date: str = '2020-01-01',
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get data for predefined universes.
        
        Parameters:
        -----------
        universe : str
            'spy500', 'nasdaq100', 'dow30', or 'tech_giants'
        start_date : str
            Start date
        end_date : str, optional
            End date
        
        Returns:
        --------
        DataFrame : Price data
        """
        universes = {
            'spy500': ['SPY', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ'],
            'nasdaq100': ['QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AVGO', 'PEP'],
            'dow30': ['DIA', 'AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'MCD', 'CAT', 'V', 'JNJ'],
            'tech_giants': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'NFLX', 'CRM', 'ADBE'],
            'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'USB', 'PNC'],
            'hk_tech': ['^HSI', '0700.HK', '9988.HK', '3690.HK', '9618.HK', '1810.HK']
        }
        
        tickers = universes.get(universe, universes['spy500'])
        return self.provider.get_prices(tickers, start_date, end_date)
    
    def get_factor_data(self, factors: List[str] = None,
                        start_date: str = '2020-01-01',
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get common factor data.
        
        Parameters:
        -----------
        factors : list, optional
            List of factor tickers. Default includes common factors.
        start_date : str
            Start date
        end_date : str, optional
            End date
        
        Returns:
        --------
        DataFrame : Factor returns
        """
        if factors is None:
            factors = ['SPY', 'TLT', 'GLD', 'UUP', 'VIXY', 'DBC']
        
        return self.provider.get_returns(factors, start_date, end_date)