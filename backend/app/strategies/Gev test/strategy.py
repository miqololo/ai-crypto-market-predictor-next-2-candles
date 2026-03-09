from app.strategies.base import BaseStrategy
import pandas as pd

class EmaCrossStrategy(BaseStrategy):
    def __init__(self, ema12_period=12, ema25_period=25, **kwargs):
        super().__init__(**kwargs)
        self.ema12_period = ema12_period
        self.ema25_period = ema25_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Check data length
        if len(df) < max(self.ema12_period, self.ema25_period):
            return signals
        
        # Calculate EMAs if they don't exist
        ema12_col = f"ema_{self.ema12_period}"
        ema25_col = f"ema_{self.ema25_period}"
        
        if ema12_col not in df.columns:
            df[ema12_col] = df["close"].ewm(span=self.ema12_period, adjust=False).mean()
        if ema25_col not in df.columns:
            df[ema25_col] = df["close"].ewm(span=self.ema25_period, adjust=False).mean()
        
        # Extract scalar values for calculations (CRITICAL!)
        ema12_val = float(df[ema12_col].iloc[-1]) if pd.notna(df[ema12_col].iloc[-1]) else 0.0
        ema25_val = float(df[ema25_col].iloc[-1]) if pd.notna(df[ema25_col].iloc[-1]) else 0.0
        
        # Fill NaN values from rolling calculations (CRITICAL for large datasets!)
        ema12 = df[ema12_col].bfill().ffill().fillna(0)
        ema25 = df[ema25_col].bfill().ffill().fillna(0)
        
        # Generate signals (numeric: 1, -1, 0)
        # ALWAYS use parentheses around boolean conditions!
        buy_condition = (
            (ema12 > ema25) & 
            (ema12.shift(1) <= ema25.shift(1)) &
            (ema12_val > 0) &  # Ensure valid values
            (ema25_val > 0)
        )
        signals[buy_condition] = 1
        
        sell_condition = (
            (ema12 < ema25) & 
            (ema12.shift(1) >= ema25.shift(1)) &
            (ema12_val > 0) &  # Ensure valid values
            (ema25_val > 0)
        )
        signals[sell_condition] = -1
        
        # Handle NaN values
        signals = signals.fillna(0)
        
        return signals