"""Base strategy class for backtesting."""
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseStrategy(ABC):
    """
    Base class for trading strategies.
    
    Strategies should implement the `generate_signals` method which takes
    a DataFrame with OHLCV data and indicators, and returns a Series of signals:
    - 1: Long (buy)
    - -1: Short (sell/close long)
    - 0: Flat (no position)
    """
    
    def __init__(self, **kwargs):
        """Initialize strategy with optional parameters."""
        self.params = kwargs
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from DataFrame.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Series with signals (1=long, -1=short, 0=flat) indexed by df.index
        """
        pass
    
    def get_name(self) -> str:
        """Return strategy name."""
        return self.__class__.__name__
