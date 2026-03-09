from app.strategies.base import BaseStrategy
import pandas as pd
import numpy as np

class BTCBreakoutStrategy(BaseStrategy):
    def __init__(self, atr_period=14, volume_multiplier=1.2, **kwargs):
        super().__init__(**kwargs)
        self.atr_period = atr_period
        self.volume_multiplier = volume_multiplier
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Check data length - need at least 20 bars for opening range + breakout
        if len(df) < 20:
            return signals
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Calculate ATR if not present
        atr_col = "atr"
        if atr_col not in df.columns:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[atr_col] = tr.ewm(span=self.atr_period, adjust=False).mean()
        
        # Calculate opening range using rolling window (20 bars for more frequent signals)
        opening_range_period = 20
        opening_range_high_series = df['high'].rolling(window=opening_range_period).max()
        opening_range_low_series = df['low'].rolling(window=opening_range_period).min()
        
        # Calculate 20-bar average volume (rolling Series)
        df['volume_20'] = df['volume'].rolling(window=20).mean()
        
        # Fill NaN values forward (rolling windows produce NaN for first N bars)
        opening_range_high_series = opening_range_high_series.bfill().ffill()
        opening_range_low_series = opening_range_low_series.bfill().ffill()
        df['volume_20'] = df['volume_20'].bfill().ffill()
        
        # Replace any remaining NaN with 0 to avoid comparison issues
        opening_range_high_series = opening_range_high_series.fillna(0)
        opening_range_low_series = opening_range_low_series.fillna(0)
        df['volume_20'] = df['volume_20'].fillna(0)
        
        # Create volume surge boolean Series (comparing each bar's volume to its rolling average)
        # Only check volume surge where volume_20 is valid (> 0)
        volume_threshold = self.volume_multiplier * df['volume_20']
        volume_surge = (df['volume'] >= volume_threshold) & (df['volume_20'] > 0)
        
        # Long Entry Conditions: Breakout above opening range high with volume confirmation
        # Check for crossover: current close above range high AND previous close was below
        long_entry_condition = (
            (df['close'] > opening_range_high_series) & 
            (df['close'].shift(1) <= opening_range_high_series.shift(1)) &
            volume_surge &
            (opening_range_high_series > 0)  # Ensure we have valid range high
        )
        
        # Short Entry Conditions: Breakdown below opening range low with volume confirmation
        # Check for crossover: current close below range low AND previous close was above
        short_entry_condition = (
            (df['close'] < opening_range_low_series) & 
            (df['close'].shift(1) >= opening_range_low_series.shift(1)) &
            volume_surge &
            (opening_range_low_series > 0)  # Ensure we have valid range low
        )
        
        signals[long_entry_condition] = 1
        signals[short_entry_condition] = -1
        
        # Handle NaN values
        signals = signals.fillna(0)
        
        return signals