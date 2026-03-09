"""Funding rates, market sentiment, and derived metrics from Binance API."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from app.data.binance_api import fetch_all_binance_data


def fetch_funding_sentiment(
    symbol: str = "BTC/USDT:USDT",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    timeframe: str = "5m",
) -> Dict[str, Any]:
    """
    Fetch all Binance Futures sentiment data via public API (no key needed):
    - basis (futures/data/basis)
    - global long/short account ratio
    - top long/short account ratio (top traders)
    - top long/short position ratio
    - taker long/short ratio
    - open interest history
    - funding rate history
    """
    try:
        return fetch_all_binance_data(symbol=symbol, timeframe=timeframe, limit=30)
    except Exception as e:
        return {"error": str(e)}


def compute_volume_metrics(df: pd.DataFrame, atr_col: str = "atr", timeframe: str = "5m") -> Dict[str, Any]:
    """Taker net volume / ATR, CVD / ATR (normalized). Tail volume sum uses TF-appropriate window."""
    if "volume" not in df.columns or len(df) < 5:
        return {}
    v = df["volume"]
    c = df["close"]
    o = df["open"]
    atr = df[atr_col] if atr_col in df.columns else (c - c.shift(1)).abs().rolling(5).mean()
    # Approximate CVD: volume * sign(close - open)
    delta = np.where(c > o, 1, np.where(c < o, -1, 0))
    cvd = (v.values * delta).cumsum()
    cvd_series = pd.Series(cvd, index=df.index)
    atr_last = atr.iloc[-1] if pd.notna(atr.iloc[-1]) and atr.iloc[-1] > 0 else 1
    tf = (timeframe or "").lower()
    tail_n = 8 if "1h" in tf or "60" in tf else (6 if "15" in tf else 5)
    return {
        "cvd_atr_ratio": float(cvd_series.iloc[-1] / atr_last) if atr_last else None,
        "volume_tail_sum": float(v.tail(tail_n).sum()),
    }
