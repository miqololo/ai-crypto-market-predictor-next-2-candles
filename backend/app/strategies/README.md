# Strategy Backtesting System

This system allows you to create custom trading strategies as Python files and run backtests on them via the API.

## Creating a Strategy

Create a Python file that defines a strategy class inheriting from `BaseStrategy`:

```python
from app.strategies.base import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self, param1: float = 10.0, **kwargs):
        """Initialize with custom parameters."""
        super().__init__(**kwargs)
        self.param1 = param1
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Series with signals: 1=long, -1=short, 0=flat
        """
        signals = pd.Series(0, index=df.index)
        
        # Your strategy logic here
        # Example: Buy when RSI < 30, sell when RSI > 70
        if "rsi" in df.columns:
            signals[df["rsi"] < 30] = 1  # Long
            signals[df["rsi"] > 70] = -1  # Short
        
        return signals
```

## Running a Backtest

Use the `/api/backtest/strategy` endpoint:

```bash
POST /api/backtest/strategy
Content-Type: application/json

{
  "symbol": "BTC/USDT:USDT",
  "timeframe": "1h",
  "limit": 500,
  "engine": "vectorbt",
  "strategy_file": "app/strategies/my_strategy.py",
  "initial_capital": 10000.0,
  "strategy_params": {
    "param1": 15.0
  }
}
```

### Request Parameters

- `symbol`: Trading pair (default: "BTC/USDT:USDT")
- `timeframe`: Timeframe (e.g., "5m", "15m", "1h")
- `limit`: Number of candles to fetch (default: 500)
- `engine`: Backtest engine - "vectorbt" or "backtesting" (default: "vectorbt")
- `strategy_file`: Path to strategy Python file (required)
  - Can be relative to backend directory or absolute path
- `initial_capital`: Starting capital (default: 10000.0)
- `strategy_params`: Optional dict of parameters to pass to strategy constructor

### Response

```json
{
  "engine": "vectorbt",
  "total_return": 0.15,
  "sharpe_ratio": 1.2,
  "max_drawdown": -0.05,
  "total_trades": 42,
  "win_rate": 0.55,
  "strategy_name": "MyStrategy",
  "strategy_file": "/path/to/strategy.py"
}
```

## Available Indicators

The DataFrame passed to `generate_signals()` includes:
- OHLCV: `open`, `high`, `low`, `close`, `volume`
- Technical indicators: `rsi`, `macd`, `ema_*`, `sma_*`, `bb_*`, `atr`, etc.
- See `app/indicators/full_ta.py` for full list

## Example Strategy

See `example_strategy.py` for a complete example using EMA crossover.
