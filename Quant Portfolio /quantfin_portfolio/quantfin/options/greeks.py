"""Greeks calculator for options."""

import numpy as np
from typing import Literal, Union, Optional
from scipy.stats import norm


class GreeksCalculator:
    """
    Calculate option Greeks using finite differences.
    
    Works with any pricing model that has a price() method.
    
    Parameters:
    -----------
    model : object
        Pricing model with price() method
    """
    
    def __init__(self, model):
        self.model = model
        self._epsilon = 1e-4
    
    def delta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate delta numerically.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Delta
        """
        S = self.model.S
        
        # Create models with perturbed spot
        model_up = self._create_model(S=S + self._epsilon)
        model_down = self._create_model(S=S - self._epsilon)
        
        price_up = model_up.price(option_type)
        price_down = model_down.price(option_type)
        
        return (price_up - price_down) / (2 * self._epsilon)
    
    def gamma(self) -> float:
        """
        Calculate gamma numerically.
        
        Returns:
        --------
        float : Gamma
        """
        S = self.model.S
        
        model_up = self._create_model(S=S + self._epsilon)
        model_down = self._create_model(S=S - self._epsilon)
        
        price_up = model_up.price('call')
        price_center = self.model.price('call')
        price_down = model_down.price('call')
        
        return (price_up - 2 * price_center + price_down) / (self._epsilon ** 2)
    
    def vega(self) -> float:
        """
        Calculate vega numerically (for 1% change in vol).
        
        Returns:
        --------
        float : Vega
        """
        sigma = self.model.sigma
        
        model_up = self._create_model(sigma=sigma + 0.01)
        model_down = self._create_model(sigma=sigma - 0.01)
        
        price_up = model_up.price('call')
        price_down = model_down.price('call')
        
        return (price_up - price_down) / 2
    
    def theta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate theta numerically (daily time decay).
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Theta (daily)
        """
        T = self.model.T
        
        # Decrease T by one day
        dt = 1 / 365
        model_decayed = self._create_model(T=max(T - dt, 1e-6))
        
        price_now = self.model.price(option_type)
        price_later = model_decayed.price(option_type)
        
        return (price_later - price_now) / dt / 365
    
    def rho(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate rho numerically (for 1% change in rate).
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Rho
        """
        r = self.model.r
        
        model_up = self._create_model(r=r + 0.01)
        model_down = self._create_model(r=r - 0.01)
        
        price_up = model_up.price(option_type)
        price_down = model_down.price(option_type)
        
        return (price_up - price_down) / 2
    
    def vanna(self) -> float:
        """
        Calculate vanna (d delta / d sigma).
        
        Returns:
        --------
        float : Vanna
        """
        S = self.model.S
        sigma = self.model.sigma
        
        # Perturb both S and sigma
        model_pp = self._create_model(S=S + self._epsilon, sigma=sigma + 0.01)
        model_pm = self._create_model(S=S + self._epsilon, sigma=sigma - 0.01)
        model_mp = self._create_model(S=S - self._epsilon, sigma=sigma + 0.01)
        model_mm = self._create_model(S=S - self._epsilon, sigma=sigma - 0.01)
        
        delta_pp = (model_pp.price('call') - model_pm.price('call')) / 0.02
        delta_mp = (model_mp.price('call') - model_mm.price('call')) / 0.02
        
        return (delta_pp - delta_mp) / (2 * self._epsilon)
    
    def charm(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate charm (d delta / d T).
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Charm
        """
        S = self.model.S
        T = self.model.T
        dt = 1 / 365
        
        # Delta now
        delta_now = self.delta(option_type)
        
        # Delta after time decay
        model_later = self._create_model(T=max(T - dt, 1e-6))
        calc_later = GreeksCalculator(model_later)
        delta_later = calc_later.delta(option_type)
        
        return (delta_later - delta_now) / dt / 365
    
    def vomma(self) -> float:
        """
        Calculate vomma (d vega / d sigma).
        
        Returns:
        --------
        float : Vomma
        """
        sigma = self.model.sigma
        
        vega_now = self.vega()
        
        model_up = self._create_model(sigma=sigma + 0.01)
        calc_up = GreeksCalculator(model_up)
        vega_up = calc_up.vega()
        
        return (vega_up - vega_now) / 0.01
    
    def all_greeks(self, option_type: Literal['call', 'put'] = 'call') -> dict:
        """
        Calculate all Greeks at once.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        dict : Dictionary of all Greeks
        """
        return {
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho': self.rho(option_type),
            'vanna': self.vanna(),
            'charm': self.charm(option_type),
            'vomma': self.vomma()
        }
    
    def _create_model(self, **kwargs):
        """Create a new model with modified parameters."""
        import copy
        new_model = copy.deepcopy(self.model)
        for key, value in kwargs.items():
            setattr(new_model, key, value)
        return new_model


class GreeksSurface:
    """
    Generate Greeks surfaces for visualization.
    
    Parameters:
    -----------
    model_class : class
        Option pricing model class
    base_params : dict
        Base parameters for the model
    """
    
    def __init__(self, model_class, base_params: dict):
        self.model_class = model_class
        self.base_params = base_params
    
    def generate_surface(self, greek: str,
                        spot_range: tuple = (0.5, 1.5),
                        vol_range: tuple = (0.1, 0.5),
                        n_points: int = 50) -> tuple:
        """
        Generate a 2D surface for a Greek.
        
        Parameters:
        -----------
        greek : str
            Name of Greek to calculate
        spot_range : tuple
            (min, max) as multiples of base spot
        vol_range : tuple
            (min, max) volatility range
        n_points : int
            Number of points in each dimension
        
        Returns:
        --------
        tuple : (spot_values, vol_values, greek_surface)
        """
        base_S = self.base_params['S']
        base_sigma = self.base_params['sigma']
        
        spot_mults = np.linspace(spot_range[0], spot_range[1], n_points)
        vols = np.linspace(vol_range[0], vol_range[1], n_points)
        
        S_grid, vol_grid = np.meshgrid(spot_mults * base_S, vols)
        greek_surface = np.zeros_like(S_grid)
        
        for i in range(n_points):
            for j in range(n_points):
                params = self.base_params.copy()
                params['S'] = S_grid[i, j]
                params['sigma'] = vol_grid[i, j]
                
                model = self.model_class(**params)
                calc = GreeksCalculator(model)
                
                greek_surface[i, j] = getattr(calc, greek)()
        
        return S_grid, vol_grid, greek_surface