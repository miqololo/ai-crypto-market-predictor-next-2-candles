from app.strategies.base import BaseStrategy
import pandas as pd

class RangeBreakoutStrategy(BaseStrategy):
    def __init__(self, timeframe='5min', lookback=20, atr_factor=1.0, buffer_factor=0.1, volume_surge_factor=1.8, rsi_long_threshold=55, rsi_short_threshold=45, rr_ratio=2.0, trailing_stop_buffer=1.0, close_to_close=False, session_close_offset=15, **kwargs):
        super().__init__(**kwargs)
        self.timeframe = timeframe
        self.lookback = lookback
        self.atr_factor = atr_factor
        self.buffer_factor = buffer_factor
        self.volume_surge_factor = volume_surge_factor
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold
        self.rr_ratio = rr_ratio
        self.trailing_stop_buffer = trailing_stop_buffer
        self.close_to_close = close_to_close
        self.session_close_offset = session_close_offset
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Check data length
        if len(df) < max(self.lookback, 20) + 14:
            return signals
        
        # Calculate ATR correctly (needs high, low, close)
        atr_col = "atr"
        if atr_col not in df.columns:
            # Calculate True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            # Calculate ATR as EMA of TR
            df[atr_col] = tr.ewm(span=14, adjust=False).mean()
        
        # Calculate breakout levels - use scalar values
        high_20 = df['high'].rolling(window=self.lookback).max()
        low_20 = df['low'].rolling(window=self.lookback).min()
        
        # Get scalar values, handling NaN
        atr_value = float(df[atr_col].iloc[-1]) if pd.notna(df[atr_col].iloc[-1]) else 0.0
        if atr_value == 0:
            return signals  # Can't calculate without ATR
        
        buffer = self.buffer_factor * atr_value
        
        # Get scalar values for levels
        high_20_val = float(high_20.iloc[-1]) if pd.notna(high_20.iloc[-1]) else 0.0
        low_20_val = float(low_20.iloc[-1]) if pd.notna(low_20.iloc[-1]) else 0.0
        
        if high_20_val == 0 or low_20_val == 0:
            return signals
        
        long_level = high_20_val + buffer
        short_level = low_20_val - buffer
        
        # Calculate volume surge
        volume_sma_col = f"volume_sma_{self.lookback}"
        if volume_sma_col not in df.columns:
            df[volume_sma_col] = df["volume"].rolling(window=self.lookback).mean()
        
        volume_sma_val = float(df[volume_sma_col].iloc[-1]) if pd.notna(df[volume_sma_col].iloc[-1]) else 0.0
        volume_surge = df['volume'] > (self.volume_surge_factor * volume_sma_val)
        
        # Calculate RSI - use pre-calculated if available, otherwise calculate
        rsi_col = "rsi"
        if rsi_col not in df.columns:
            # Calculate RSI manually
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, pd.NA)
            df[rsi_col] = 100 - (100 / (1 + rs))
        
        rsi_series = df[rsi_col]
        
        # Generate signals (numeric: 1, -1, 0)
        # Fix boolean operator precedence with parentheses
        long_condition = (
            (df['close'] > long_level) &
            (df['close'].shift(1) <= long_level) &
            volume_surge &
            (rsi_series > self.rsi_long_threshold) &
            (rsi_series < 75)  # Avoid overbought
        )
        
        short_condition = (
            (df['close'] < short_level) &
            (df['close'].shift(1) >= short_level) &
            volume_surge &
            (rsi_series < self.rsi_short_threshold) &
            (rsi_series > 25)  # Avoid oversold extremes
        )
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        # Handle NaN values
        signals = signals.fillna(0)
        
        return signals