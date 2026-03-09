"""
Build per-candle view for LLM: OHLC + normalized indicators, booleans, ratios.
Loads candles from MongoDB, adds indicators, builds per-candle structure, saves to llm1_collection.
All timestamps stored as UTC datetime in MongoDB.
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient

from app.indicators.full_ta import add_full_indicators
from app.services.llm1_prepare import build_per_candle_view
from app.services.llm1_feature_meta import FEATURE_META, CANONICAL_FEATURES


MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "traider"
CANDLES_COLLECTION = "candles"
LLM1_COLLECTION = "llm1_collection"
LLM1_CANDLES_COLLECTION = "llm1_candles"
BATCH_SIZE = 500
SYMBOL = "BTCUSDT"


def _to_native(obj: Any) -> Any:
    """Convert numpy/pandas to native Python for JSON/MongoDB."""
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
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    if isinstance(obj, datetime):
        return obj
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _candles_from_mongo_to_df(docs: List[Dict]) -> pd.DataFrame:
    """Convert MongoDB candle documents to OHLCV DataFrame. Timestamps preserved as datetime."""
    if not docs:
        return pd.DataFrame()
    rows = []
    for d in docs:
        ts = d.get("timestamp")
        rows.append({
            "timestamp": ts,
            "open": d.get("open"),
            "high": d.get("high"),
            "low": d.get("low"),
            "close": d.get("close"),
            "volume": d.get("volume"),
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def build_llm1_document(
    timeframe: str,
    symbol: str = SYMBOL,
    limit: Optional[int] = None,
    mongodb_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch candles from DB, add indicators, build per-candle view with:
    - OHLC + open_norm, high_norm, low_norm, close_norm (min-max [0,1] using price scaler)
    - Price-based indicators (EMA, SMA, BB, PSAR, Supertrend, VWAP, Ichimoku) — same price scaler
    - atr_norm, volume_norm, sum_open_interest_norm, sum_open_interest_value_norm — own scalers
    - macd_norm, ao_norm, cci_norm — own scalers [0,1] or [-1,1]
    - funding_rate_norm, basis_norm — own scalers
    - rsi, stoch_k, stoch_d — /100 → [0,1]
    - Booleans: price_above_ema_fast, rsi_bullish, etc.
    - Ratios: long_short_ratio, top_trader_ratio, etc.

    Saves to llm1_collection. Returns the built document.
    """
    uri = mongodb_uri or MONGODB_URI
    client = MongoClient(uri)
    db = client[DB_NAME]
    candles_coll = db[CANDLES_COLLECTION]
    llm1_coll = db[LLM1_COLLECTION]

    query = {"timeframe": timeframe, "symbol": symbol}
    cursor = candles_coll.find(query).sort("timestamp", 1)
    if limit:
        cursor = cursor.limit(limit)
    docs = list(cursor)

    if not docs:
        return {"error": f"No candles found for timeframe={timeframe}, symbol={symbol}"}

    df = _candles_from_mongo_to_df(docs)
    if df.empty or len(df) < 10:
        return {"error": "Insufficient candle data"}

    df = add_full_indicators(df, timeframe=timeframe)

    candles_view = build_per_candle_view(df, docs, feature_names=CANONICAL_FEATURES)
    if not candles_view:
        return {"error": "Failed to build per-candle view"}

    llm1_candles_coll = db[LLM1_CANDLES_COLLECTION]

    # Delete existing candles for this symbol/timeframe
    llm1_candles_coll.delete_many({"symbol": symbol, "timeframe": timeframe})

    # Insert per-candle docs in batches (avoids 16MB document limit)
    native_candles = _to_native(candles_view)
    for i in range(0, len(native_candles), BATCH_SIZE):
        batch = native_candles[i : i + BATCH_SIZE]
        docs_to_insert = [
            {"symbol": symbol, "timeframe": timeframe, **c}
            for c in batch
        ]
        llm1_candles_coll.insert_many(docs_to_insert)

    # Store metadata with feature_meta in llm1_collection
    meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_count": len(candles_view),
        "feature_meta": FEATURE_META,
        "features": CANONICAL_FEATURES,
    }
    llm1_coll.replace_one(
        {"symbol": symbol, "timeframe": timeframe},
        meta,
        upsert=True,
    )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_count": len(candles_view),
        "collection": LLM1_CANDLES_COLLECTION,
        "candles": native_candles[:100],  # Return first 100 in response (for preview)
    }
