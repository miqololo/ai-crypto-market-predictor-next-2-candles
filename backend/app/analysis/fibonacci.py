"""Fibonacci retracement levels from swing highs/lows."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def add_fibonacci_per_candle(
    df: pd.DataFrame,
    swing_period: int = 40,
) -> pd.DataFrame:
    """
    Add Fibonacci retracement levels to each candle (rolling window).
    Returns df with fib_0, fib_23_6, fib_38_2, fib_50, fib_61_8, fib_78_6, fib_100.
    """
    out = df.copy()
    h, l_ = out["high"], out["low"]
    n = len(df)

    fib_cols = ["fib_0", "fib_23_6", "fib_38_2", "fib_50", "fib_61_8", "fib_78_6", "fib_100"]
    for c in fib_cols:
        out[c] = np.nan

    for i in range(swing_period, n):
        window_high = h.iloc[i - swing_period : i + 1]
        window_low = l_.iloc[i - swing_period : i + 1]
        swing_high = float(window_high.max())
        swing_low = float(window_low.min())
        diff = swing_high - swing_low
        if diff == 0:
            diff = 1e-10
        idx = out.index[i]
        out.at[idx, "fib_0"] = swing_high
        out.at[idx, "fib_23_6"] = swing_high - 0.236 * diff
        out.at[idx, "fib_38_2"] = swing_high - 0.382 * diff
        out.at[idx, "fib_50"] = swing_high - 0.5 * diff
        out.at[idx, "fib_61_8"] = swing_high - 0.618 * diff
        out.at[idx, "fib_78_6"] = swing_high - 0.786 * diff
        out.at[idx, "fib_100"] = swing_low

    return out


def calc_fibonacci(
    df: pd.DataFrame,
    swing_period: int = 40,
    atr_col: str = "atr",
) -> Dict:
    """
    Calculate Fibonacci retracement levels from recent swing high/low.
    Levels: 23.6, 38.2, 50, 61.8 (Golden Pocket), 78.6.
    """
    if len(df) < swing_period:
        swing_period = len(df) - 1
    if swing_period < 2:
        return {"levels": {}, "swing_high": None, "swing_low": None, "atr": None}

    recent = df.tail(swing_period)
    high = recent["high"].max()
    low = recent["low"].min()
    high_idx = recent["high"].idxmax()
    low_idx = recent["low"].idxmin()

    # Ensure we have proper swing direction
    if high_idx > low_idx:
        swing_high, swing_low = high, low
        direction = "down"  # price came down from high
    else:
        swing_high, swing_low = high, low
        direction = "up"

    diff = swing_high - swing_low
    if diff == 0:
        return {"levels": {}, "swing_high": float(swing_high), "swing_low": float(swing_low), "atr": None}

    levels = {
        "0": float(swing_high),
        "23.6": float(swing_high - 0.236 * diff),
        "38.2": float(swing_high - 0.382 * diff),
        "50": float(swing_high - 0.5 * diff),
        "61.8": float(swing_high - 0.618 * diff),
        "78.6": float(swing_high - 0.786 * diff),
        "100": float(swing_low),
    }

    atr = df[atr_col].iloc[-1] if atr_col in df.columns and pd.notna(df[atr_col].iloc[-1]) else None
    atr_units = {k: (v - swing_low) / atr if atr and atr > 0 else None for k, v in levels.items()}

    return {
        "levels": levels,
        "swing_high": float(swing_high),
        "swing_low": float(swing_low),
        "direction": direction,
        "atr": float(atr) if atr else None,
        "levels_in_atr": atr_units,
    }
