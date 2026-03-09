from app.strategies.base import BaseStrategy
import pandas as pd
import numpy as np

class ModifiedBollingerBandsStrategy(BaseStrategy):
    def __init__(self, bb_period=20, bb_std=2, atr_multiplier=2, lr_slope_period=18, **kwargs):
        super().__init__(**kwargs)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_multiplier = atr_multiplier
        self.lr_slope_period = lr_slope_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Check data length
        if len(df) < max(self.bb_period, self.lr_slope_period) + 10:
            return signals
        
        # Calculate Bollinger Bands
        bb_upper = df['close'].rolling(window=self.bb_period).mean() + self.bb_std * df['close'].rolling(window=self.bb_period).std()
        bb_lower = df['close'].rolling(window=self.bb_period).mean() - self.bb_std * df['close'].rolling(window=self.bb_period).std()
        
        # Calculate ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=self.bb_period, adjust=False).mean()
        
        # Calculate ATR Bands
        atr_upper = bb_upper + self.atr_multiplier * df['atr']
        atr_lower = bb_lower - self.atr_multiplier * df['atr']
        
        # Calculate Linear Regression Slope
        lr_slope = (df['close'] - df['close'].shift(self.lr_slope_period)) / self.lr_slope_period
        
        # Calculate TDFI (Trend Direction Force Index)
        tdfi = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(2)
        
        # Generate signals
        long_condition = (
            (df['close'] > bb_upper) & 
            (lr_slope.iloc[-1] > 0) &
            (tdfi.iloc[-1] > 0)
        )
        signals[long_condition] = 1
        
        short_condition = (
            (df['close'] < bb_lower) & 
            (lr_slope.iloc[-1] < 0) &
            (tdfi.iloc[-1] < 0)
        )
        signals[short_condition] = -1
        
        # Handle NaN values
        signals = signals.fillna(0)
        
        return signals