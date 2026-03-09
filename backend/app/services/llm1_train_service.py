"""
Train ML index from llm1_candles: build searchable structure over sliding windows.
Uses ALL collection keys (numeric) so you can query with any feature subset.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient

# Default timezone for response timestamps (Binance uses UTC)
DEFAULT_TZ = "Asia/Yerevan"  # UTC+4


def _to_local_ts(ts_str: str, tz: str) -> str:
    """Convert UTC timestamp string to local timezone. tz: 'Asia/Yerevan', 'UTC+4', etc."""
    from datetime import timedelta, timezone
    try:
        ts = pd.to_datetime(ts_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if tz.upper().replace(" ", "").startswith("UTC"):
            rest = tz.replace("UTC", "").replace("utc", "").strip()
            if rest.startswith("+"):
                offset = int(rest[1:].strip() or 0)
            elif rest.startswith("-"):
                offset = -int(rest[1:].strip() or 0)
            else:
                offset = 0
            tz_obj = timezone(timedelta(hours=offset))
        else:
            try:
                from zoneinfo import ZoneInfo
            except ImportError:
                from backports.zoneinfo import ZoneInfo
            tz_obj = ZoneInfo(tz)
        ts_local = ts.tz_convert(tz_obj)
        return ts_local.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_str

from app.services.llm1_service import DB_NAME, LLM1_CANDLES_COLLECTION, LLM1_COLLECTION, MONGODB_URI, CANDLES_COLLECTION
from app.services.llm1_feature_meta import CANONICAL_FEATURES, FEATURE_META

LLM1_INDEX_BASE = Path("data/llm1_index")
SKIP_KEYS = {"symbol", "timeframe", "timestamp", "_id"}
# Raw price excluded from similarity search (use *_norm and indicators instead)
EXCLUDE_FROM_SEARCH = {"open", "high", "low", "close"}


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _get_all_numeric_keys(docs: List[Dict]) -> List[str]:
    """Collect all keys that have numeric/boolean values across docs. Sorted for consistency."""
    keys = set()
    for d in docs:
        for k, v in d.items():
            if k in SKIP_KEYS:
                continue
            if v is None:
                continue
            if isinstance(v, (int, float, bool)):
                keys.add(k)
    return sorted(keys)


def _candles_to_df(docs: List[Dict]) -> pd.DataFrame:
    """Convert llm1_candles docs to DataFrame, sorted by timestamp. Handles datetime or ISO string."""
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    return df


def _extract_window_vector(
    window: pd.DataFrame,
    features: List[str],
    aggregate: str,
) -> np.ndarray:
    """Extract vector from window. aggregate: 'mean' or 'flatten'."""
    available = [f for f in features if f in window.columns]
    sub = window[available].copy()
    sub = sub.fillna(0)
    # Convert bool to int for numeric ops (use is_bool_dtype to avoid AttributeError on duplicate cols)
    for c in list(sub.columns):
        try:
            col = sub[c]
            if isinstance(col, pd.Series) and pd.api.types.is_bool_dtype(col):
                sub[c] = col.astype(int)
        except (AttributeError, TypeError):
            pass
    sub = sub.astype(np.float64)
    if aggregate == "mean":
        return sub.mean(axis=0).values.astype(np.float32)
    return sub.values.flatten().astype(np.float32)


def _project_to_features(X: np.ndarray, feature_names: List[str], query_features: List[str], aggregate: str) -> np.ndarray:
    """Project full embedding matrix to selected features only."""
    indices = [feature_names.index(f) for f in query_features if f in feature_names]
    if not indices:
        return np.array([]).reshape(X.shape[0], 0)
    if aggregate == "mean":
        return X[:, indices]
    # flatten: each window is [c0_f0..c0_fn, c1_f0..c1_fn, ...]; n_feat per candle
    n_feat = len(feature_names)
    n_candles = X.shape[1] // n_feat
    cols = []
    for c in range(n_candles):
        for i in indices:
            cols.append(c * n_feat + i)
    return X[:, cols]


def train_llm1_index(
    timeframe: str,
    symbol: str = "BTCUSDT",
    window_size: int = 8,
    features: Optional[List[str]] = None,
    aggregate: str = "mean",
    mongodb_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load llm1_candles, build searchable structure over sliding windows.
    Uses ALL numeric collection keys by default so you can query with any subset.
    """
    uri = mongodb_uri or MONGODB_URI
    client = MongoClient(uri)
    db = client[DB_NAME]
    coll = db[LLM1_CANDLES_COLLECTION]

    cursor = coll.find({"symbol": symbol, "timeframe": timeframe}).sort("timestamp", 1)
    docs = list(cursor)

    if len(docs) < window_size:
        return {"error": f"Need at least {window_size} candles, got {len(docs)}"}

    df = _candles_to_df(docs)

    # Use features from llm1_collection metadata (from build), or explicit list, or fallback to all numeric keys
    if features:
        all_keys = [f for f in features if f in df.columns and f not in EXCLUDE_FROM_SEARCH]
    else:
        meta_doc = db[LLM1_COLLECTION].find_one({"symbol": symbol, "timeframe": timeframe})
        meta_features = meta_doc.get("features") if meta_doc else None
        if meta_features:
            all_keys = [f for f in meta_features if f in df.columns and f not in EXCLUDE_FROM_SEARCH]
        else:
            all_keys = _get_all_numeric_keys(docs)
            all_keys = [k for k in all_keys if k in df.columns and k not in EXCLUDE_FROM_SEARCH]

    if not all_keys:
        return {"error": f"No numeric keys found. Columns: {list(df.columns)[:30]}..."}

    # Build sliding windows with ALL keys
    embeddings = []
    timestamps = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i : i + window_size]
        vec = _extract_window_vector(window, all_keys, aggregate)
        embeddings.append(vec)
        ts = df.index[i + window_size - 1]
        timestamps.append(str(ts))

    X = np.array(embeddings, dtype=np.float32)

    # Per-feature z-score so no single feature dominates; values stay unbounded (not 0-1)
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0)
    feat_std[feat_std < 1e-10] = 1.0
    X_scaled = (X - feat_mean) / feat_std

    # Save full embeddings (all features) for flexible query-time projection
    base = LLM1_INDEX_BASE if LLM1_INDEX_BASE.is_absolute() else Path.cwd() / LLM1_INDEX_BASE
    save_dir = base / f"{symbol}_{timeframe}_w{window_size}"
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "windows.npy", X_scaled)
    np.save(save_dir / "feat_mean.npy", feat_mean)
    np.save(save_dir / "feat_std.npy", feat_std)
    np.save(save_dir / "timestamps.npy", np.array(timestamps, dtype=object), allow_pickle=True)

    meta_doc = db[LLM1_COLLECTION].find_one({"symbol": symbol, "timeframe": timeframe})
    feature_meta = meta_doc.get("feature_meta", FEATURE_META) if meta_doc else FEATURE_META

    meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_size": window_size,
        "features": all_keys,
        "feature_meta": feature_meta,
        "aggregate": aggregate,
        "n_windows": len(embeddings),
        "dim_full": X.shape[1],
        "scaled": True,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "success": True,
        "saved_to": str(save_dir),
        "n_windows": len(embeddings),
        "window_size": window_size,
        "features": all_keys,
        "aggregate": aggregate,
        "dim_full": X.shape[1],
        "next_step": "Call POST /api/llm1/similarity with same timeframe, symbol, window_size",
    }


def search_llm1_similar(
    timeframe: str,
    symbol: str = "BTCUSDT",
    window_size: int = 8,
    datetime_iso: Optional[str] = None,
    features: Optional[List[str]] = None,
    priorities: Optional[Dict[str, float]] = None,
    k: int = 10,
    metric: str = "cosine",
    min_hours_apart: float = 168,
    min_similarity: float = 0.99,
    timezone: str = DEFAULT_TZ,
    mongodb_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Find similar x-candle windows for y timeframe.
    Query with ANY subset of trained features - use features param to choose a,b,c,d,...
    Excludes raw price (open, high, low, close) and results within min_hours_apart of query.
    """
    uri = mongodb_uri or MONGODB_URI
    client = MongoClient(uri)
    db = client[DB_NAME]
    coll = db[LLM1_CANDLES_COLLECTION]

    base = LLM1_INDEX_BASE if LLM1_INDEX_BASE.is_absolute() else Path.cwd() / LLM1_INDEX_BASE
    save_dir = base / f"{symbol}_{timeframe}_w{window_size}"
    meta_path = save_dir / "metadata.json"
    if not meta_path.exists():
        return {
            "error": f"No trained index for {symbol}/{timeframe} w={window_size}. Call POST /api/llm1/train first.",
            "path_checked": str(meta_path),
            "similar_windows": [],
        }

    with open(meta_path) as f:
        meta = json.load(f)

    X = np.load(save_dir / "windows.npy")
    timestamps = np.load(save_dir / "timestamps.npy", allow_pickle=True).tolist()
    trained_features = meta["features"]
    aggregate = meta.get("aggregate", "mean")
    scaled = meta.get("scaled", False)

    feat_mean = np.load(save_dir / "feat_mean.npy") if (save_dir / "feat_mean.npy").exists() else None
    feat_std = np.load(save_dir / "feat_std.npy") if (save_dir / "feat_std.npy").exists() else None
    feat_min = np.load(save_dir / "feat_min.npy") if (save_dir / "feat_min.npy").exists() else None
    feat_max = np.load(save_dir / "feat_max.npy") if (save_dir / "feat_max.npy").exists() else None

    # Get query window from llm1_candles
    llm1_query = {"symbol": symbol, "timeframe": timeframe}
    print(f"[MongoDB] llm1_candles.find({llm1_query}).sort('timestamp', 1)")
    cursor = coll.find(llm1_query).sort("timestamp", 1)
    docs = list(cursor)
    if len(docs) < window_size:
        return {"error": f"Need at least {window_size} candles", "similar_windows": []}

    df = _candles_to_df(docs)

    if datetime_iso:
        try:
            target = pd.to_datetime(datetime_iso, utc=True)
            # Keep target tz-aware — df.index is UTC-aware after _candles_to_df.
            # Stripping tzinfo causes TypeError and silently falls back to last window.
            end_idx = df.index.get_indexer([target], method="nearest")[0]
            end_idx = max(window_size - 1, min(end_idx, len(df) - 1))
            start = end_idx - window_size + 1
        except Exception as e:
            print(f"[similarity] datetime lookup error ({e}), using last window")
            start = len(df) - window_size
    else:
        start = len(df) - window_size

    query_start_ts = str(df.index[start])
    query_end_ts = str(df.index[start + window_size - 1])
    data_start = str(df.index[0])
    data_end = str(df.index[-1])
    print(
        f"[similarity] requested_datetime={datetime_iso} | "
        f"query_window={query_start_ts} -> {query_end_ts} | "
        f"data_range={data_start} -> {data_end}"
    )

    window = df.iloc[start : start + window_size]

    # Use requested features or all trained; exclude raw price from search
    feat = features or trained_features
    available = [
        f for f in feat
        if f in window.columns and f in trained_features and f not in EXCLUDE_FROM_SEARCH
    ]
    if not available:
        available = [f for f in trained_features if f not in EXCLUDE_FROM_SEARCH][:10]

    query_vec = _extract_window_vector(window, available, aggregate)
    query_vec = query_vec.astype(np.float32)

    # Apply same scaling as training (if scaled)
    if scaled:
        if aggregate == "mean":
            indices = [trained_features.index(f) for f in available if f in trained_features]
        else:
            n_feat = len(trained_features)
            n_candles = X.shape[1] // n_feat
            feat_indices = [trained_features.index(f) for f in available if f in trained_features]
            indices = [c * n_feat + i for c in range(n_candles) for i in feat_indices]
        if feat_mean is not None and feat_std is not None:
            q_mean = feat_mean[indices]
            q_std = feat_std[indices]
            q_std = np.where(q_std < 1e-10, 1.0, q_std)
            query_vec = (query_vec - q_mean) / q_std
        elif feat_min is not None and feat_max is not None:
            q_min = feat_min[indices]
            q_max = feat_max[indices]
            q_range = q_max - q_min
            q_range = np.where(q_range < 1e-10, 1.0, q_range)
            query_vec = (query_vec - q_min) / q_range

    # Project stored windows to selected features
    X_proj = _project_to_features(X, trained_features, available, aggregate)
    if X_proj.size == 0:
        return {"error": f"Could not project to features {available}", "similar_windows": []}

    # Apply feature priorities (weights): multiply each dimension by sqrt(weight) for weighted cosine
    if priorities:
        w_vec = np.array([np.sqrt(max(0.01, float(priorities.get(f, 1.0)))) for f in available], dtype=np.float32)
        if aggregate == "flatten":
            n_candles = X_proj.shape[1] // len(available)
            w_vec = np.tile(w_vec, n_candles)
        query_vec = query_vec * w_vec
        X_proj = X_proj * w_vec

    # Normalize for cosine
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    X_norm = X_proj / (np.linalg.norm(X_proj, axis=1, keepdims=True) + 1e-10)

    if metric == "cosine":
        scores = X_norm @ q_norm
        order = np.argsort(-scores)
    else:
        dists = np.linalg.norm(X_norm - q_norm, axis=1)
        order = np.argsort(dists)
        scores = -dists

    query_ts = pd.to_datetime(query_end_ts)

    # Columns for price and trend (last candle of each window)
    PRICE_COLS = ["open", "high", "low", "close"]
    TREND_COLS = ["rsi", "rsi_bullish", "macd_bullish", "supertrend_bullish", "above_ichimoku_cloud", "price_above_ema_fast", "price_above_sma20"]

    def _extract_price_trend(window_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract price and trend from last candle of window."""
        out: Dict[str, Any] = {"price": {}, "trend": {}}
        if window_df.empty:
            return out
        last = window_df.iloc[-1]
        for c in PRICE_COLS:
            if c in window_df.columns and pd.notna(last.get(c)):
                out["price"][c] = round(float(last[c]), 4)
        for c in TREND_COLS:
            if c in window_df.columns:
                v = last.get(c)
                if pd.notna(v):
                    out["trend"][c] = bool(v) if isinstance(v, (bool, np.bool_)) else round(float(v), 4)
        return out

    similar = []
    # Collect scaled prices: for each similar, scale its next 3 candles by (query_close / similar_last_close)
    scaled_next_1: List[Dict[str, float]] = []
    scaled_next_2: List[Dict[str, float]] = []
    scaled_next_3: List[Dict[str, float]] = []
    pred_next_ts: List[Dict[str, str]] = []  # timestamps of predicted next candles
    pred_scores_used: List[float] = []  # raw similarity scores for windows contributing to prediction
    weights: List[float] = []
    similar_atrs: List[float] = []

    query_close = float(window.iloc[-1]["close"]) if "close" in window.columns and len(window) else None
    last = window.iloc[-1]
    query_atr = None
    if "atr" in window.columns and pd.notna(last.get("atr")) and float(last.get("atr", 0) or 0) > 0:
        query_atr = float(last["atr"])
    elif "high" in window.columns and "low" in window.columns:
        query_atr = float(last["high"] - last["low"]) if last["high"] > last["low"] else None
    ohlc_cols = ["open", "high", "low", "close"]

    for idx in order:
        if len(similar) >= k:
            break
        if float(scores[idx]) < min_similarity:
            continue
        if 0 <= idx < len(timestamps) and timestamps[idx] != query_end_ts:
            cand_ts = pd.to_datetime(timestamps[idx])
            if abs((cand_ts - query_ts).total_seconds()) < min_hours_apart * 3600:
                continue
            if idx + window_size <= len(df):
                window_df = df.iloc[idx : idx + window_size]
                price_trend = _extract_price_trend(window_df)
                # Scale similar's next 3 candles to query's price level: price * (query_close / similar_last_close)
                if (
                    query_close and query_close > 0
                    and idx + window_size + 3 <= len(df)
                    and all(c in df.columns for c in ohlc_cols)
                ):
                    similar_last = float(df.iloc[idx + window_size - 1]["close"])
                    if similar_last > 0:
                        ratio = query_close / similar_last
                        # Squared similarity gives stronger preference to the best matches
                        raw_score = max(0, float(scores[idx]))
                        w = raw_score ** 2
                        n1 = df.iloc[idx + window_size]
                        n2 = df.iloc[idx + window_size + 1]
                        n3 = df.iloc[idx + window_size + 2]
                        scaled_next_1.append({c: float(n1[c]) * ratio for c in ohlc_cols if pd.notna(n1.get(c))})
                        scaled_next_2.append({c: float(n2[c]) * ratio for c in ohlc_cols if pd.notna(n2.get(c))})
                        scaled_next_3.append({c: float(n3[c]) * ratio for c in ohlc_cols if pd.notna(n3.get(c))})
                        pred_next_ts.append({
                            "next_1": _to_local_ts(str(n1.name), timezone),
                            "next_2": _to_local_ts(str(n2.name), timezone),
                            "next_3": _to_local_ts(str(n3.name), timezone),
                        })
                        weights.append(w)
                        pred_scores_used.append(max(0.0, float(scores[idx])))
                        sim_row = df.iloc[idx + window_size - 1]
                        sim_atr = sim_row.get("atr")
                        if pd.notna(sim_atr) and float(sim_atr) > 0:
                            similar_atrs.append((float(sim_atr), w))
                        elif "high" in df.columns and "low" in df.columns:
                            hl = float(sim_row["high"] - sim_row["low"])
                            if hl > 0:
                                similar_atrs.append((hl, w))
            else:
                price_trend = {"price": {}, "trend": {}}
            similar.append({
                "timestamp": _to_local_ts(timestamps[idx], timezone),
                "similarity_score": round(float(scores[idx]), 4),
                **price_trend,
            })

    # Predict next 3 candles: weighted avg of scaled prices (each similar window scaled by ratio to query)
    prediction: Dict[str, Any] = {}
    if query_close and query_close > 0 and scaled_next_1 and weights:
        total_w = sum(weights)
        if total_w > 0:
            def _wavg(candles: List[Dict[str, float]], col: str) -> float:
                total = 0.0
                w_sum = 0.0
                for c, w in zip(candles, weights):
                    v = c.get(col)
                    if v is not None:
                        total += v * w
                        w_sum += w
                return total / w_sum if w_sum > 0 else 0.0

            pred_1 = {c: round(_wavg(scaled_next_1, c), 4) for c in ohlc_cols}
            pred_2 = {c: round(_wavg(scaled_next_2, c), 4) for c in ohlc_cols}
            pred_3 = {c: round(_wavg(scaled_next_3, c), 4) for c in ohlc_cols}
            # Ensure open = previous close
            pred_1["open"] = round(query_close, 4)
            pred_2["open"] = pred_1["close"]
            pred_3["open"] = pred_2["close"]

            # Apply volatility ratio: scale predicted movements by (current ATR / similar ATR avg)
            vol_ratio = 1.0
            if query_atr and query_atr > 0 and similar_atrs:
                w_sum = sum(w for _, w in similar_atrs)
                if w_sum > 0:
                    similar_atr_avg = sum(a * w for a, w in similar_atrs) / w_sum
                    if similar_atr_avg > 0:
                        vol_ratio = query_atr / similar_atr_avg

            def _apply_vol(p: Dict[str, float]) -> Dict[str, float]:
                o, h, l_, c = p.get("open"), p.get("high"), p.get("low"), p.get("close")
                if o is None or h is None or l_ is None or c is None:
                    return p
                o = float(o)
                h = float(h)
                l_ = float(l_)
                c = float(c)
                out = {"open": round(o, 4)}
                out["high"] = round(o + (h - o) * vol_ratio, 4)
                out["low"] = round(o - (o - l_) * vol_ratio, 4)
                out["close"] = round(o + (c - o) * vol_ratio, 4)
                return out

            pred_1 = _apply_vol(pred_1)
            pred_2["open"] = pred_1["close"]
            pred_2 = _apply_vol(pred_2)
            pred_3["open"] = pred_2["close"]
            pred_3 = _apply_vol(pred_3)

            def _simplify_pred(p: Dict[str, float]) -> Dict[str, Any]:
                o, h, l_, c = p.get("open"), p.get("high"), p.get("low"), p.get("close")
                out: Dict[str, Any] = {"open": round(o, 4) if o is not None else None}
                if c is not None:
                    out["close"] = round(c, 4)
                if h is not None:
                    out["high"] = round(h, 4)
                if l_ is not None:
                    out["low"] = round(l_, 4)
                if o is not None and h is not None and l_ is not None:
                    out["high_low_diff"] = round(h - l_, 4)
                if o is not None and c is not None:
                    out["abs_open_close_diff"] = round(abs(c - o), 4)
                    out["candle"] = "green" if c >= o else "red"
                return out

            avg_sim = round(float(np.mean(pred_scores_used)), 4) if pred_scores_used else 0.0

            prediction = {
                "base_close": round(query_close, 4),
                "next_1": _simplify_pred(pred_1),
                "next_2": _simplify_pred(pred_2),
                "next_3": _simplify_pred(pred_3),
                "n_similar_used": len(scaled_next_1),
                "avg_similarity": avg_sim,
                "vol_ratio": round(vol_ratio, 4),
            }

    def _ts_to_naive_py(ts):
        """Convert index timestamp to naive UTC datetime for MongoDB query."""
        if hasattr(ts, "to_pydatetime"):
            dt = ts.to_pydatetime()
        else:
            dt = pd.to_datetime(ts).to_pydatetime()
        if dt.tzinfo is not None:
            from datetime import timezone
            dt = dt.astimezone(timezone.utc)
        return dt.replace(tzinfo=None)

    ts_last = df.index[start + window_size - 1]
    ts_last_py = _ts_to_naive_py(ts_last)
    candles_coll = db[CANDLES_COLLECTION]

    # Current candle: prefer raw candles (reliable OHLC), fallback to llm1_candles df
    current_candle: Dict[str, Any] = {"timestamp": _to_local_ts(query_end_ts, timezone)}
    raw_current = candles_coll.find_one(
        {"symbol": symbol, "timeframe": timeframe, "timestamp": ts_last_py},
        projection=["open", "high", "low", "close"]
    )
    if not raw_current:
        from datetime import timedelta
        raw_current = candles_coll.find_one(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": {"$gte": ts_last_py, "$lt": ts_last_py + timedelta(minutes=1)},
            },
            projection=["open", "high", "low", "close"],
            sort=[("timestamp", 1)],
        )
    if raw_current and all(raw_current.get(c) is not None for c in ohlc_cols):
        current_candle["price"] = {c: round(float(raw_current[c]), 4) for c in ohlc_cols}
    elif len(window) and all(c in df.columns for c in ohlc_cols):
        last_row = window.iloc[-1]
        if all(pd.notna(last_row.get(c)) for c in ohlc_cols):
            current_candle["price"] = {c: round(float(last_row[c]), 4) for c in ohlc_cols}
        else:
            current_candle["price"] = _extract_price_trend(window).get("price") or {}
    else:
        current_candle["price"] = _extract_price_trend(window).get("price") if len(window) else {}

    # Actual next 3 candles from raw candles collection only — no fallback
    actual_next: Dict[str, Any] = {}
    cursor = candles_coll.find(
        {"symbol": symbol, "timeframe": timeframe, "timestamp": {"$gt": ts_last_py}}
    ).sort("timestamp", 1).limit(3)
    raw_docs = list(cursor)
    for i, label in enumerate(["actual_next_1", "actual_next_2", "actual_next_3"], 1):
        if i <= len(raw_docs):
            d = raw_docs[i - 1]
            if d and all(d.get(c) is not None for c in ohlc_cols):
                actual_next[label] = {
                    "timestamp": _to_local_ts(str(pd.to_datetime(d["timestamp"])), timezone),
                    "open": round(float(d["open"]), 4),
                    "high": round(float(d["high"]), 4),
                    "low": round(float(d["low"]), 4),
                    "close": round(float(d["close"]), 4),
                }

    # Most similar window's next 3 candles (raw OHLC)
    most_similar_next: Dict[str, Any] = {}
    if similar and all(c in df.columns for c in ohlc_cols):
        first_similar_idx = None
        for idx in order:
            if float(scores[idx]) < min_similarity:
                continue
            if 0 <= idx < len(timestamps) and timestamps[idx] != query_end_ts:
                cand_ts = pd.to_datetime(timestamps[idx])
                if abs((cand_ts - query_ts).total_seconds()) >= min_hours_apart * 3600 and idx + window_size + 3 <= len(df):
                    first_similar_idx = idx
                    break
        if first_similar_idx is not None:
            for j, label in enumerate(["next_1", "next_2", "next_3"], 1):
                row = df.iloc[first_similar_idx + window_size - 1 + j]
                most_similar_next[label] = {
                    "timestamp": _to_local_ts(str(row.name), timezone),
                    "open": round(float(row["open"]), 4),
                    "high": round(float(row["high"]), 4),
                    "low": round(float(row["low"]), 4),
                    "close": round(float(row["close"]), 4),
                }

    return _to_python({
        "current_candle": current_candle,
        "similar_windows": similar,
        "most_similar_next": most_similar_next,
        "prediction": prediction,
        "actual_next": actual_next,
        "query_info": {
            "timeframe": timeframe,
            "symbol": symbol,
            "window_size": window_size,
            "min_similarity": min_similarity,
            "timezone": timezone,
            "requested_datetime": datetime_iso,
            "query_start": _to_local_ts(str(df.index[start]), timezone),
            "query_end": _to_local_ts(query_end_ts, timezone),
            "data_end": _to_local_ts(str(df.index[-1]), timezone) if len(df) else None,
            "at_data_end": (start + window_size - 1) >= len(df) - 1,
            "features_used": available,
            "available_features": trained_features,
            "priorities": priorities,
        },
    })


def search_llm1_similar_batch(
    timeframe: str,
    symbol: str = "BTCUSDT",
    window_size: int = 8,
    n_samples: int = 10,
    features: Optional[List[str]] = None,
    priorities: Optional[Dict[str, float]] = None,
    k: int = 10,
    metric: str = "cosine",
    min_hours_apart: float = 168,
    min_similarity: float = 0.999,
    timezone: str = DEFAULT_TZ,
    mongodb_uri: Optional[str] = None,
    weekday: Optional[int] = None,
    hours_start: Optional[str] = None,
    hours_end: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sample n_samples random valid timestamps from llm1_candles, run similarity for each,
    return list of results. Optional weekday (0=Mon..6=Sun) and hours_start/hours_end (HH:MM)
    filter to sample only from matching candles.
    """
    import random

    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    uri = mongodb_uri or MONGODB_URI
    client = MongoClient(uri)
    db = client[DB_NAME]
    coll = db[LLM1_CANDLES_COLLECTION]

    llm1_query = {"symbol": symbol, "timeframe": timeframe}
    cursor = coll.find(llm1_query).sort("timestamp", 1)
    docs = list(cursor)
    if len(docs) < window_size + 1:
        return {"error": f"Need at least {window_size + 1} candles", "results": []}

    df = _candles_to_df(docs)
    # Valid end indices: need window_size before and 1 after for actual_next_1
    min_end = window_size - 1
    max_end = len(df) - 2
    if min_end > max_end:
        return {"error": "Not enough candles for valid windows", "results": []}

    valid_indices = list(range(min_end, max_end + 1))

    # Filter by weekday and hours (in timezone)
    if weekday is not None or (hours_start is not None and hours_end is not None):
        tz_obj = ZoneInfo(timezone) if not timezone.upper().startswith("UTC") else None

        def _parse_hhmm(s: str) -> int:
            parts = s.strip().split(":")
            h = int(parts[0]) if parts else 0
            m = int(parts[1]) if len(parts) > 1 else 0
            return h * 60 + m

        start_min = _parse_hhmm(hours_start) if hours_start else 0
        end_min = _parse_hhmm(hours_end) if hours_end else 24 * 60 - 1

        def _matches_filters(idx: int) -> bool:
            ts = df.index[idx]
            if hasattr(ts, "to_pydatetime"):
                dt = pd.Timestamp(ts)
            else:
                dt = pd.Timestamp(ts)
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            if tz_obj:
                dt_local = dt.tz_convert(tz_obj)
            else:
                dt_local = dt
            if weekday is not None and dt_local.weekday() != weekday:
                return False
            if hours_start is not None and hours_end is not None:
                mins = dt_local.hour * 60 + dt_local.minute
                if start_min <= end_min:
                    if not (start_min <= mins <= end_min):
                        return False
                else:
                    # Range spans midnight (e.g. 22:00–02:00)
                    if mins < start_min and mins > end_min:
                        return False
            return True

        valid_indices = [i for i in valid_indices if _matches_filters(i)]
        if not valid_indices:
            return {"error": "No candles match weekday/hours filter", "results": []}

    n = min(n_samples, len(valid_indices))
    chosen = random.sample(valid_indices, n)

    def _run_one(end_idx: int) -> Dict[str, Any]:
        ts = df.index[end_idx]
        try:
            dt = pd.Timestamp(ts)
            if dt.tzinfo is None:
                datetime_iso = dt.tz_localize("UTC").strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            else:
                datetime_iso = dt.astimezone("UTC").strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        except Exception:
            datetime_iso = str(ts)[:19] + "Z"
        r = search_llm1_similar(
            timeframe=timeframe,
            symbol=symbol,
            window_size=window_size,
            datetime_iso=datetime_iso,
            features=features,
            priorities=priorities,
            k=k,
            metric=metric,
            min_hours_apart=min_hours_apart,
            min_similarity=min_similarity,
            timezone=timezone,
            mongodb_uri=uri,
        )
        if "error" in r and not r.get("similar_windows"):
            r["query_info"] = r.get("query_info") or {}
            r["query_info"]["requested_datetime"] = datetime_iso
        return r

    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = min(5, n)
    results_dict: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_run_one, idx): idx for idx in chosen}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results_dict[idx] = future.result()
            except Exception as e:
                results_dict[idx] = {"error": str(e), "similar_windows": []}
    results = [results_dict[idx] for idx in chosen]

    return {"results": results}


def search_llm1_similar_by_datetimes(
    datetimes: List[str],
    timeframe: str,
    symbol: str = "BTCUSDT",
    window_size: int = 8,
    features: Optional[List[str]] = None,
    priorities: Optional[Dict[str, float]] = None,
    k: int = 10,
    metric: str = "cosine",
    min_hours_apart: float = 168,
    min_similarity: float = 0.99,
    timezone: str = DEFAULT_TZ,
    mongodb_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run similarity for each given datetime. Returns results for your given times only.
    """
    if not datetimes:
        return {"results": []}

    results: List[Dict[str, Any]] = []
    for datetime_iso in datetimes:
        r = search_llm1_similar(
            timeframe=timeframe,
            symbol=symbol,
            window_size=window_size,
            datetime_iso=datetime_iso.strip() if isinstance(datetime_iso, str) else str(datetime_iso),
            features=features,
            priorities=priorities,
            k=k,
            metric=metric,
            min_hours_apart=min_hours_apart,
            min_similarity=min_similarity,
            timezone=timezone,
            mongodb_uri=mongodb_uri,
        )
        results.append(r)

    return {"results": results}
