"""Main analysis orchestrator: candles + indicators + trend + fib + sentiment + normalized."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from app.data import DataFetcher


def _to_native(obj):
    """Convert numpy/pandas types to native Python for JSON serialization. NaN/Inf -> None."""
    if hasattr(obj, "item"):
        v = obj.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj
from app.indicators.full_ta import add_full_indicators
from app.analysis.trend import determine_trend
from app.analysis.fibonacci import calc_fibonacci
from app.analysis.sentiment import fetch_funding_sentiment, compute_volume_metrics
from app.analysis.normalize import compute_normalized_features
from app.analysis.timeframe_config import get_config


def _normalize_symbol(symbol: str) -> str:
    """Convert BTCUSDT -> BTC/USDT:USDT for Binance futures."""
    s = symbol.upper().replace("/", "")
    if "USDT" in s and ":" not in s:
        base = s.replace("USDT", "")
        return f"{base}/USDT:USDT"
    return symbol if ":" in symbol else symbol


def run_analysis(
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    limit: int = 500,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    include_funding: bool = True,
) -> Dict[str, Any]:
    """
    Full candle analysis: OHLCV, indicators, trend, Fibonacci, funding, normalized features.
    """
    cfg = get_config(timeframe)
    sym = _normalize_symbol(symbol)
    fetcher = DataFetcher(api_key=api_key, api_secret=api_secret)
    df = fetcher.fetch_ohlcv(symbol=sym, timeframe=timeframe, limit=limit)
    df = add_full_indicators(df, timeframe=timeframe)
    df = df.dropna(how="all", subset=[c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]])

    # Trend (5m: 70%, 15m: 65%, 1h: 70%; ema_fast: ema_9 for 5m, ema_12 for 15m/1h; RSI: 50 vs 52.5)
    is_5m = "5" in timeframe and "15" not in timeframe
    ema_fast_col = "ema_9" if is_5m else "ema_12"
    rsi_level = 52.5 if "15" in timeframe or "1h" in timeframe else 50
    ema_medium_col = "ema_26" if "1h" in timeframe else "ema_21"
    trend_result = determine_trend(
        df,
        threshold=cfg["trend_threshold"],
        ema_fast_col=ema_fast_col,
        rsi_bullish_level=rsi_level,
        ema_medium_col=ema_medium_col,
    )

    # Fibonacci (5m: 40, 15m: 30 candles swing)
    fib_result = calc_fibonacci(df, swing_period=cfg["fib_swing_period"])

    # Funding / sentiment (15m uses period=15m for Binance API)
    funding_result = {}
    if include_funding:
        try:
            funding_result = fetch_funding_sentiment(sym, api_key, api_secret, timeframe=timeframe)
        except Exception as e:
            funding_result = {"error": str(e)}

    # Volume metrics
    vol_metrics = compute_volume_metrics(df, timeframe=timeframe)

    # Normalized features (5m: 20, 15m: 40 window)
    norm_features = compute_normalized_features(
        df,
        window=cfg["norm_window"],
        embedding_window=cfg.get("embedding_window", cfg["norm_window"]),
    )

    # Last candle + indicators (summary)
    last = df.iloc[-1]
    indicators = {
        "ema_9": float(last.get("ema_9")) if pd.notna(last.get("ema_9")) else None,
        "ema_12": float(last.get("ema_12")) if pd.notna(last.get("ema_12")) else None,
        "ema_21": float(last.get("ema_21")) if pd.notna(last.get("ema_21")) else None,
        "sma_20": float(last.get("sma_20")) if pd.notna(last.get("sma_20")) else None,
        "sma_50": float(last.get("sma_50")) if pd.notna(last.get("sma_50")) else None,
        "sma_200": float(last.get("sma_200")) if pd.notna(last.get("sma_200")) else None,
        "ema_26": float(last.get("ema_26")) if pd.notna(last.get("ema_26")) else None,
        "rsi": float(last.get("rsi")) if pd.notna(last.get("rsi")) else None,
        "macd": float(last.get("macd")) if pd.notna(last.get("macd")) else None,
        "macd_signal": float(last.get("macd_signal")) if pd.notna(last.get("macd_signal")) else None,
        "macd_hist": float(last.get("macd_hist")) if pd.notna(last.get("macd_hist")) else None,
        "bb_upper": float(last.get("bb_upper")) if pd.notna(last.get("bb_upper")) else None,
        "bb_mid": float(last.get("bb_mid")) if pd.notna(last.get("bb_mid")) else None,
        "bb_lower": float(last.get("bb_lower")) if pd.notna(last.get("bb_lower")) else None,
        "stoch_k": float(last.get("stoch_k")) if pd.notna(last.get("stoch_k")) else None,
        "stoch_d": float(last.get("stoch_d")) if pd.notna(last.get("stoch_d")) else None,
        "cci": float(last.get("cci")) if pd.notna(last.get("cci")) else None,
        "atr": float(last.get("atr")) if pd.notna(last.get("atr")) else None,
        "psar": float(last.get("psar")) if pd.notna(last.get("psar")) else None,
        "supertrend": float(last.get("supertrend")) if pd.notna(last.get("supertrend")) else None,
        "vwap": float(last.get("vwap")) if pd.notna(last.get("vwap")) else None,
        "ao": float(last.get("ao")) if pd.notna(last.get("ao")) else None,
    }

    # Candles (last 100 for response size)
    candles = df.tail(100).reset_index()
    candles["timestamp"] = candles["timestamp"].astype(str)
    candles_data = _to_native(candles.to_dict(orient="records"))

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_count": len(df),
        "trend": trend_result,
        "indicators": indicators,
        "fibonacci": fib_result,
        "funding_and_sentiment": funding_result,
        "volume_metrics": vol_metrics,
        "normalized_features": norm_features,
        "candles": candles_data,
    }
