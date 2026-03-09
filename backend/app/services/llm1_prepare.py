"""
Per-candle view for LLM: OHLC + canonical features only.
Each candle gets: raw OHLC + features from CANONICAL_FEATURES with metadata.
All timestamps stored as UTC datetime for MongoDB.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.services.llm1_feature_meta import CANONICAL_FEATURES, FEATURE_META


def _to_utc_datetime(ts) -> datetime:
    """Convert timestamp (pd.Timestamp, datetime, or str) to UTC datetime for storage."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _extract_from_doc(doc: Dict, *paths: tuple) -> Optional[float]:
    """Extract scalar from nested doc. paths = (key, subkey) or (key,)."""
    val = doc
    for k in paths:
        if val is None or not isinstance(val, dict):
            return None
        val = val.get(k)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _extract_binance_per_candle(docs: List[Dict]) -> Dict[str, List[Optional[float]]]:
    """Extract Binance scalars per candle from MongoDB docs."""
    out = {
        "long_short_ratio": [],
        "taker_ratio": [],
    }
    for d in docs:
        gls = d.get("global_long_short_account_ratio")
        out["long_short_ratio"].append(_extract_from_doc(gls, "longShortRatio") if isinstance(gls, dict) else None)

        taker = d.get("taker_long_short_ratio")
        out["taker_ratio"].append(_extract_from_doc(taker, "buySellRatio") if isinstance(taker, dict) else None)
    return out


def build_per_candle_view(
    df: pd.DataFrame,
    docs: List[Dict],
    feature_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build per-candle view with OHLC + canonical features only.
    df: DataFrame with indicators (from add_full_indicators).
    docs: MongoDB candle docs (same order as df) with Binance data.
    feature_names: list of features to include (default: CANONICAL_FEATURES).
    """
    if len(df) != len(docs) or len(df) == 0:
        return []

    features = feature_names or CANONICAL_FEATURES
    binance = _extract_binance_per_candle(docs)

    # Price scaler
    price_cols = ["open", "high", "low", "close"]
    price_min = df[price_cols].min().min()
    price_max = df[price_cols].max().max()
    price_range = price_max - price_min if price_max > price_min else 1.0

    def price_norm(x):
        if pd.isna(x):
            return None
        return float((x - price_min) / price_range)

    # ATR scaler
    atr_ser = df["atr"].fillna(0)
    atr_min, atr_max = atr_ser.min(), atr_ser.max()
    atr_range = atr_max - atr_min if atr_max > atr_min else 1.0

    # Momentum scalers (macd, macd_hist, ao, cci) -> [-1, 1]
    mom_cols = ["macd", "macd_hist", "ao", "cci"]
    mom_mins, mom_maxs = {}, {}
    for c in mom_cols:
        if c in df.columns:
            s = df[c].fillna(0)
            mom_mins[c] = s.min()
            mom_maxs[c] = s.max()

    result = []
    for i in range(len(df)):
        row = df.iloc[i]
        ts = row.name
        ts_utc = _to_utc_datetime(ts)

        candle: Dict[str, Any] = {
            "timestamp": ts_utc,
            "open": float(row["open"]) if pd.notna(row["open"]) else None,
            "high": float(row["high"]) if pd.notna(row["high"]) else None,
            "low": float(row["low"]) if pd.notna(row["low"]) else None,
            "close": float(row["close"]) if pd.notna(row["close"]) else None,
        }

        # --- Price ---
        if "open_norm" in features:
            candle["open_norm"] = price_norm(row["open"])
        if "high_norm" in features:
            candle["high_norm"] = price_norm(row["high"])
        if "low_norm" in features:
            candle["low_norm"] = price_norm(row["low"])
        if "close_norm" in features:
            candle["close_norm"] = price_norm(row["close"])
        if "vwap_norm" in features and "vwap" in df.columns:
            candle["vwap_norm"] = price_norm(row.get("vwap"))

        # --- Momentum ---
        if "rsi" in features:
            rsi_val = row.get("rsi")
            if pd.notna(rsi_val):
                if rsi_val > 80:
                    candle["rsi"] = 1.0
                elif rsi_val < 20:
                    candle["rsi"] = -1.0
                else:
                    candle["rsi"] = 0.0
            else:
                candle["rsi"] = None

        if "macd_hist_norm" in features and "macd_hist" in df.columns:
            v = row.get("macd_hist")
            if pd.notna(v) and "macd_hist" in mom_mins:
                r = mom_maxs["macd_hist"] - mom_mins["macd_hist"] or 1.0
                candle["macd_hist_norm"] = float(2 * (v - mom_mins["macd_hist"]) / r - 1)
            else:
                candle["macd_hist_norm"] = None

        if "macd_norm" in features and "macd" in df.columns:
            v = row.get("macd")
            if pd.notna(v) and "macd" in mom_mins:
                r = mom_maxs["macd"] - mom_mins["macd"] or 1.0
                candle["macd_norm"] = float(2 * (v - mom_mins["macd"]) / r - 1)
            else:
                candle["macd_norm"] = None

        if "stoch_k" in features:
            sk_val = row.get("stoch_k")
            candle["stoch_k"] = float(sk_val / 100) if pd.notna(sk_val) else None

        if "cci_norm" in features and "cci" in df.columns:
            v = row.get("cci")
            if pd.notna(v) and "cci" in mom_mins:
                r = mom_maxs["cci"] - mom_mins["cci"] or 1.0
                candle["cci_norm"] = float((v - mom_mins["cci"]) / r) if r else 0.0
                candle["cci_norm"] = max(-1.0, min(1.0, candle["cci_norm"]))
            else:
                candle["cci_norm"] = None

        if "ao_norm" in features and "ao" in df.columns:
            v = row.get("ao")
            if pd.notna(v) and "ao" in mom_mins:
                r = mom_maxs["ao"] - mom_mins["ao"] or 1.0
                candle["ao_norm"] = float(2 * (v - mom_mins["ao"]) / r - 1)
            else:
                candle["ao_norm"] = None

        # --- Volatility ---
        if "atr_norm" in features:
            atr_val = row.get("atr")
            candle["atr_norm"] = float((atr_val - atr_min) / atr_range) if pd.notna(atr_val) and atr_range else None

        if "percent_b" in features and all(c in df.columns for c in ["bb_upper", "bb_lower", "bb_mid"]):
            c, bb_u, bb_l = row.get("close"), row.get("bb_upper"), row.get("bb_lower")
            if pd.notna(c) and pd.notna(bb_u) and pd.notna(bb_l) and (bb_u - bb_l) > 0:
                pb = (c - bb_l) / (bb_u - bb_l)
                candle["percent_b"] = float(max(0.0, min(1.0, pb)))
            else:
                candle["percent_b"] = None

        # --- Trend ---
        if "supertrend_dir" in features:
            st_dir = row.get("supertrend_dir")
            candle["supertrend_dir"] = int(st_dir) if pd.notna(st_dir) else None

        if "supertrend_bullish" in features:
            st_dir = row.get("supertrend_dir")
            candle["supertrend_bullish"] = bool(st_dir == 1) if pd.notna(st_dir) else None

        # --- Crossings (need lookback) ---
        if "ema_cross_bull" in features:
            ema_cross = False
            if i >= 1 and "ema_9" in df.columns and "ema_21" in df.columns:
                prev = df.iloc[i - 1]
                ema_cross = (prev.get("ema_9", 0) <= prev.get("ema_21", 0) and
                             row.get("ema_9", 0) > row.get("ema_21", 0))
            candle["ema_cross_bull"] = 1.0 if ema_cross else 0.0

        if "macd_cross_bull" in features:
            macd_cross = False
            if i >= 1 and "macd" in df.columns and "macd_signal" in df.columns:
                prev = df.iloc[i - 1]
                macd_cross = (prev.get("macd", 0) <= prev.get("macd_signal", 0) and
                              row.get("macd", 0) > row.get("macd_signal", 0))
            candle["macd_cross_bull"] = 1.0 if macd_cross else 0.0

        if "supertrend_flip_up" in features:
            flip = False
            if i >= 1:
                prev_dir = df.iloc[i - 1].get("supertrend_dir")
                curr_dir = row.get("supertrend_dir")
                flip = (prev_dir == -1 and curr_dir == 1) if (pd.notna(prev_dir) and pd.notna(curr_dir)) else False
            candle["supertrend_flip_up"] = 1.0 if flip else 0.0

        # --- Sentiment ---
        if "long_short_ratio" in features:
            candle["long_short_ratio"] = binance["long_short_ratio"][i]

        if "taker_ratio" in features:
            candle["taker_ratio"] = binance["taker_ratio"][i]

        # --- Time (cyclical, UTC) ---
        try:
            dt = ts_utc if ts_utc else _to_utc_datetime(ts)
            hour = dt.hour + dt.minute / 60
            dow = dt.weekday()  # 0=Mon, 6=Sun
        except Exception:
            hour, dow = 0, 0

        if "hour_sin" in features:
            candle["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        if "hour_cos" in features:
            candle["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        if "dayofweek_sin" in features:
            candle["dayofweek_sin"] = float(np.sin(2 * np.pi * dow / 7))
        if "dayofweek_cos" in features:
            candle["dayofweek_cos"] = float(np.cos(2 * np.pi * dow / 7))

        # --- Strategy composites ---
        rsi = candle.get("rsi") or 0  # -1 (oversold), 0 (neutral), 1 (overbought)
        atr_n = candle.get("atr_norm") or 0
        st_bull = 1.0 if candle.get("supertrend_bullish") else 0.0
        macd_bull = 1.0 if (candle.get("macd_hist_norm", 0) or 0) > 0 else 0.0
        close_n = candle.get("close_norm") or 0.5
        taker = candle.get("taker_ratio")
        taker_n = (min(float(taker), 2) / 2) if taker is not None and float(taker or 0) > 0 else 0.5

        if "strategy_multi_confirmation" in features:
            rsi_bull = 1.0 if rsi > 0 else 0.0  # overbought = bullish momentum
            candle["strategy_multi_confirmation"] = float((st_bull + rsi_bull + macd_bull) / 3)

        if "strategy_divergence_reversal" in features:
            # RSI oversold (<20) + price recovering
            rev = close_n if rsi == -1 else 0.0
            candle["strategy_divergence_reversal"] = float(min(1.0, rev))

        if "strategy_sentiment_boosted_trend" in features:
            candle["strategy_sentiment_boosted_trend"] = float(st_bull * (0.7 + 0.3 * taker_n))

        if "strategy_risk_adjusted_momentum" in features:
            mom = rsi  # already [-1,1]
            candle["strategy_risk_adjusted_momentum"] = float(max(0, min(1, 0.5 + mom * (1 - atr_n) * 0.5)))

        if "strategy_supertrend_strength" in features:
            rsi_01 = (rsi + 1) / 2  # map [-1,1] to [0,1]
            candle["strategy_supertrend_strength"] = float(st_bull * (0.5 + rsi_01 * 0.5))

        result.append(candle)

    return result
