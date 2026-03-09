"""Multi-indicator trend determination logic."""
import pandas as pd
import numpy as np
from typing import Optional


def determine_trend(
    df: pd.DataFrame,
    threshold: float = 0.70,
    ema_fast_col: str = "ema_9",
    rsi_bullish_level: float = 50,
    ema_medium_col: str = "ema_21",
) -> dict:
    """
    Determine trend from last row: Uptrend, Downtrend, or Neutral.
    Bullish if >= threshold of indicators align.
    5m: ema_9, ema_21, RSI>50
    15m: ema_12, ema_21, RSI>52.5
    1h: ema_12, ema_26, RSI>52.5, sma_50
    """
    if df.empty or len(df) < 52:
        return {"trend": "Neutral", "confidence": 0.0, "signals": {}}

    row = df.iloc[-1]
    signals = {}

    # Price vs EMAs/SMAs (ema_fast: ema_9 for 5m, ema_12 for 15m)
    c = row["close"]
    ema_fast = row.get(ema_fast_col, row.get("ema_9", np.nan))
    signals["price_above_ema_fast"] = c > ema_fast if pd.notna(ema_fast) else False
    signals["price_above_sma20"] = c > row.get("sma_20", np.nan) if pd.notna(row.get("sma_20")) else False
    ema_med = row.get(ema_medium_col, row.get("ema_21", np.nan))
    signals["price_above_ema_medium"] = c > ema_med if pd.notna(ema_med) else False
    signals["price_above_sma50"] = c > row.get("sma_50", np.nan) if pd.notna(row.get("sma_50")) else False

    # RSI (15m: >50-55 for bullish)
    rsi = row.get("rsi", 50)
    signals["rsi_bullish"] = rsi > rsi_bullish_level if pd.notna(rsi) else False

    # MACD
    macd_hist = row.get("macd_hist", 0)
    macd = row.get("macd", 0)
    macd_sig = row.get("macd_signal", 0)
    signals["macd_bullish"] = (macd > macd_sig and macd_hist > 0) if pd.notna(macd_hist) else False

    # Stochastic
    stoch_k = row.get("stoch_k", 50)
    stoch_d = row.get("stoch_d", 50)
    signals["stoch_bullish"] = (stoch_k > stoch_d and stoch_k > 50) if pd.notna(stoch_k) else False

    # CCI
    cci = row.get("cci", 0)
    signals["cci_bullish"] = cci > 0 if pd.notna(cci) else False

    # Supertrend (1 = bullish, -1 = bearish)
    st_dir = row.get("supertrend_dir", 1)
    signals["supertrend_bullish"] = st_dir > 0 if pd.notna(st_dir) else False

    # Ichimoku: price above cloud (simplified: above senkou_a and senkou_b)
    senkou_a = row.get("senkou_a", 0)
    senkou_b = row.get("senkou_b", 0)
    cloud_top = np.maximum(senkou_a, senkou_b) if pd.notna(senkou_a) and pd.notna(senkou_b) else 0
    signals["above_ichimoku_cloud"] = c > cloud_top if cloud_top and pd.notna(c) else False

    # Awesome Oscillator
    ao = row.get("ao", 0)
    signals["ao_positive"] = ao > 0 if pd.notna(ao) else False

    # Bollinger: price position (optional - not strictly trend)
    bb_mid = row.get("bb_mid", c)
    signals["above_bb_mid"] = c > bb_mid if pd.notna(bb_mid) else False

    # Count bullish vs bearish
    bullish_checks = [
        signals["price_above_ema_fast"],
        signals["price_above_sma20"],
        signals["price_above_ema_medium"],
        signals["price_above_sma50"],
        signals["rsi_bullish"],
        signals["macd_bullish"],
        signals["stoch_bullish"],
        signals["cci_bullish"],
        signals["supertrend_bullish"],
        signals["above_ichimoku_cloud"],
        signals["ao_positive"],
    ]
    bearish_checks = [not x for x in bullish_checks]

    valid_bull = sum(1 for x in bullish_checks if isinstance(x, bool))
    valid_bear = sum(1 for x in bearish_checks if isinstance(x, bool))
    n_bullish = sum(bullish_checks)
    n_bearish = sum(bearish_checks)
    n_total = max(valid_bull, 1)

    bull_pct = n_bullish / n_total
    bear_pct = n_bearish / n_total

    # BB squeeze (low volatility) or RSI near 50 (15m) -> Neutral
    bb_upper = row.get("bb_upper", c * 1.02)
    bb_lower = row.get("bb_lower", c * 0.98)
    bb_width = (bb_upper - bb_lower) / c if c and pd.notna(bb_upper) else 0.02
    atr = row.get("atr", c * 0.01)
    atr_pct = atr / c if c else 0
    low_volatility = bb_width < 0.02 or atr_pct < 0.005
    rsi_near_neutral = 45 <= rsi <= 55 if pd.notna(rsi) else False

    if bull_pct >= threshold:
        trend = "Uptrend"
        confidence = bull_pct
    elif bear_pct >= threshold:
        trend = "Downtrend"
        confidence = bear_pct
    elif low_volatility or rsi_near_neutral:
        trend = "Neutral"
        confidence = 1 - max(bull_pct, bear_pct)
    else:
        trend = "Neutral"
        confidence = 1 - max(bull_pct, bear_pct)

    return {
        "trend": trend,
        "confidence": round(float(confidence), 4),
        "bullish_signals": int(n_bullish),
        "bearish_signals": int(n_bearish),
        "total_checks": int(n_total),
        "signals": {k: bool(v) for k, v in signals.items()},
    }
