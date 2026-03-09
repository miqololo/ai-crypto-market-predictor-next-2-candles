"""
Similarity search service: load candles from MongoDB, add indicators,
run PatternSimilaritySearch for single-point or window queries.
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from pymongo import MongoClient

from app.indicators.full_ta import add_full_indicators
from app.patterns.pattern_similarity import PatternSimilaritySearch


MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "traider"
CANDLES_COLLECTION = "candles"

DEFAULT_FEATURES = ["rsi", "macd_hist", "close", "volume", "atr"]


def _candles_from_mongo_to_df(docs: List[Dict]) -> pd.DataFrame:
    """Convert MongoDB candle documents to OHLCV DataFrame."""
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


def _to_native(obj: Any) -> Any:
    """Convert numpy/pandas to native Python for JSON."""
    import numpy as np
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
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, float):
        import numpy as np
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def load_candles_with_indicators(
    timeframe: str,
    symbol: str = "BTCUSDT",
    limit: Optional[int] = 2000,
    mongodb_uri: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load candles from MongoDB, add indicators, return DataFrame.
    """
    uri = mongodb_uri or MONGODB_URI
    client = MongoClient(uri)
    db = client[DB_NAME]
    candles_coll = db[CANDLES_COLLECTION]

    query = {"timeframe": timeframe, "symbol": symbol}
    cursor = candles_coll.find(query).sort("timestamp", 1)
    if limit:
        cursor = cursor.limit(limit)
    docs = list(cursor)

    if not docs:
        return pd.DataFrame()

    df = _candles_from_mongo_to_df(docs)
    if df.empty or len(df) < 10:
        return df

    df = add_full_indicators(df, timeframe=timeframe)
    df = df.dropna(how="all", subset=[c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]])
    return df


def find_similar_patterns(
    timeframe: str,
    symbol: str = "BTCUSDT",
    datetime_iso: Optional[str] = None,
    selected_features: Optional[List[str]] = None,
    window_size: Optional[int] = None,
    metric: str = "cosine",
    k: int = 10,
    limit: int = 2000,
    missing_strategy: str = "impute_zero",
) -> Dict[str, Any]:
    """
    Find historical patterns similar to the query point or window.

    Args:
        timeframe: e.g. "1h", "15m", "5m"
        symbol: e.g. "BTCUSDT"
        datetime_iso: ISO datetime string; if set, use candle at that time as query. If omitted, uses latest candle from DB.
        selected_features: Features to use; if None, derived from DEFAULT_FEATURES (caller should pass from use_* flags)
        window_size: If set, use sliding-window search (last N candles before datetime)
        metric: "cosine" or "euclidean"
        k: Number of similar patterns to return
        limit: Max candles to load from DB
        missing_strategy: "skip", "impute_zero", "impute_mean"

    Returns:
        Dict with similar_patterns, query_info, error (if any)
    """
    df = load_candles_with_indicators(timeframe=timeframe, symbol=symbol, limit=limit)
    if df.empty or len(df) < 10:
        return {"error": f"No candles for timeframe={timeframe}, symbol={symbol}", "similar_patterns": []}

    searcher = PatternSimilaritySearch(missing_strategy=missing_strategy)
    searcher.fit(df, timestamp_col=None)

    available = [c for c in df.columns if c != "_timestamp" and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    if selected_features:
        features = [f for f in selected_features if f in available]
    else:
        features = [f for f in DEFAULT_FEATURES if f in available]
    if not features:
        return {"error": f"No valid features. Available: {available}", "similar_patterns": []}

    query_info = {"timeframe": timeframe, "symbol": symbol, "features_used": features}

    if window_size and window_size > 0:
        # Window query: use last N candles before datetime (or end of data)
        if datetime_iso:
            try:
                target = pd.to_datetime(datetime_iso)
                df_before = df[df.index <= target].tail(window_size)
            except Exception:
                df_before = df.tail(window_size)
        else:
            df_before = df.tail(window_size)

        if len(df_before) < window_size:
            return {"error": f"Not enough candles for window_size={window_size}", "similar_patterns": [], "query_info": query_info}

        try:
            result_df = searcher.query_window(
                df_before,
                selected_features_list=features,
                metric=metric,
                k=k,
                aggregate="mean",
            )
        except ValueError as e:
            return {"error": str(e), "similar_patterns": [], "query_info": query_info}
        query_info["mode"] = "window"
        query_info["window_size"] = window_size
    else:
        # Single-point query: always from candle at datetime or latest
        if datetime_iso:
            try:
                target = pd.to_datetime(datetime_iso)
                idx = df.index.get_indexer([target], method="nearest")[0]
                row = df.iloc[idx]
            except Exception as e:
                return {"error": str(e), "similar_patterns": [], "query_info": query_info}
        else:
            row = df.iloc[-1]

        query_vec = {f: float(row[f]) for f in features if f in row.index and pd.notna(row.get(f))}
        if len(query_vec) < len(features):
            return {"error": "Candle has NaN for some selected features", "similar_patterns": [], "query_info": query_info}

        try:
            result_df = searcher.query(
                current_values_dict=query_vec,
                selected_features_list=features,
                metric=metric,
                k=k,
            )
        except ValueError as e:
            return {"error": str(e), "similar_patterns": [], "query_info": query_info}
        query_info["mode"] = "point"
        query_info["query_values"] = {k: _to_native(v) for k, v in query_vec.items()}

    similar = []
    for _, r in result_df.iterrows():
        similar.append({
            "timestamp": _to_native(r["timestamp"]),
            "similarity_score": _to_native(r["similarity_score"]),
            "matched_row_data": _to_native(r["matched_row_data"]),
        })

    return {"similar_patterns": similar, "query_info": query_info}
