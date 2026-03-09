"""LLM1 endpoints: build, train ML index, similarity search, reset."""
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from app.services.llm1_service import build_llm1_document
from app.services.llm1_train_service import (
    train_llm1_index,
    search_llm1_similar,
    search_llm1_similar_batch,
    search_llm1_similar_by_datetimes,
)
from app.services.reset_model_service import reset_llm1_collection

router = APIRouter()


@router.post("/build")
@router.get("/build")
def build_llm1(
    timeframe: str = Query(..., description="Timeframe: 5m, 15m, or 1h"),
    symbol: str = Query("BTCUSDT", description="Symbol"),
    limit: int = Query(None, description="Max candles to fetch (default: all)"),
):
    """
    Fetch candles from DB, build per-candle view with normalized features,
    save to llm1_collection + llm1_candles. Run this before /train.
    """
    doc = build_llm1_document(timeframe=timeframe, symbol=symbol, limit=limit)
    if "error" in doc:
        return {"success": False, "error": doc["error"]}
    return {"success": True, "document": doc}


@router.get("/status")
def llm1_status(
    timeframe: str = Query(..., description="Timeframe"),
    symbol: str = Query("BTCUSDT"),
    window_size: int = Query(8, ge=2, le=50),
):
    """Check if trained index exists for given params."""
    from pathlib import Path
    import json
    from app.services.llm1_train_service import LLM1_INDEX_BASE
    base = LLM1_INDEX_BASE if LLM1_INDEX_BASE.is_absolute() else Path.cwd() / LLM1_INDEX_BASE
    save_dir = base / f"{symbol}_{timeframe}_w{window_size}"
    meta_path = save_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return {"trained": True, "path": str(save_dir), "n_windows": meta.get("n_windows"), "features_count": len(meta.get("features", []))}
    return {"trained": False, "path": str(meta_path), "hint": "Call POST /api/llm1/train first"}


@router.post("/reset")
def llm1_reset(
    timeframe: Optional[str] = Query(None, description="Reset only this timeframe (omit = all)"),
    symbol: Optional[str] = Query(None, description="Reset only this symbol (omit = all)"),
    window_size: Optional[int] = Query(None, ge=2, le=50, description="Reset only this window_size (omit = all)"),
    all_data: bool = Query(False, alias="all", description="If true, also reset llm1_candles + llm1_collection (train from 0)"),
):
    """
    Reset trained model. Use to retrain from scratch.
    - Default: resets only trained index (llm1_candles kept, run /train again)
    - all=true: full reset (llm1_candles + index cleared, run /build then /train)
    - Optional timeframe, symbol, window_size: reset only that specific index
    """
    from pathlib import Path
    import shutil
    from app.services.llm1_train_service import LLM1_INDEX_BASE

    if all_data:
        result = reset_llm1_collection()
        return {"success": True, "reset": result, "next_step": "Call /build then /train"}

    base = LLM1_INDEX_BASE if LLM1_INDEX_BASE.is_absolute() else Path.cwd() / LLM1_INDEX_BASE
    removed = []
    if base.exists():
        for item in base.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            if symbol and not name.startswith(f"{symbol}_"):
                continue
            if timeframe and timeframe not in name:
                continue
            if window_size is not None and f"_w{window_size}" not in name:
                continue
            shutil.rmtree(item)
            removed.append(name)

    return {
        "success": True,
        "removed": removed,
        "next_step": "Call POST /api/llm1/train to retrain",
    }


@router.post("/train")
def train_llm1(
    timeframe: str = Query(..., description="Timeframe: 5m, 15m, or 1h"),
    symbol: str = Query("BTCUSDT", description="Symbol"),
    window_size: int = Query(8, ge=2, le=50, description="Number of candles per window (x)"),
    features: Optional[str] = Query(
        None,
        description="Comma-separated features to use. Default: ALL numeric keys from collection.",
    ),
    aggregate: str = Query("mean", description="Window aggregation: mean or flatten"),
):
    """
    Train ML index from llm1_candles. Uses ALL collection keys by default.
    Enables similarity search by x candles for y timeframe; query with any feature subset.
    Call /build first to populate llm1_candles.
    """
    feat_list = [f.strip() for f in features.split(",")] if features else None
    result = train_llm1_index(
        timeframe=timeframe,
        symbol=symbol,
        window_size=window_size,
        features=feat_list,
        aggregate=aggregate,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


def _parse_priorities(s: Optional[str]) -> Optional[dict]:
    """Parse priorities string like 'close_norm:2,rsi:1.5' into {feature: weight}."""
    if not s or not s.strip():
        return None
    out = {}
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            name, val = part.split(":", 1)
            name = name.strip()
            try:
                out[name] = float(val.strip())
            except ValueError:
                pass
    return out if out else None


@router.post("/similarity")
@router.get("/similarity")
def llm1_similarity(
    timeframe: str = Query(..., description="Timeframe (y)"),
    symbol: str = Query("BTCUSDT", description="Symbol"),
    window_size: int = Query(8, ge=2, le=50, description="Window size (x candles)"),
    datetime: Optional[str] = Query(None, description="ISO datetime for query window; omit for latest"),
    features: Optional[str] = Query(None, description="Comma-separated features to use (any subset of trained)"),
    priorities: Optional[str] = Query(None, description="Feature weights: 'close_norm:2,rsi:1.5' — higher = more weight in similarity"),
    k: int = Query(10, ge=1, le=100, description="Number of similar windows to return"),
    metric: str = Query("cosine", description="cosine or euclidean"),
    min_hours_apart: float = Query(168, ge=0, description="Exclude results within this many hours of query (default 168 = 7 days)"),
    min_similarity: float = Query(0.99, ge=0, le=1, description="Minimum similarity score to include (default 0.95 = 95 percent)"),
    timezone: str = Query("Asia/Yerevan", description="Timezone for response timestamps (e.g. Asia/Yerevan, UTC+4)"),
):
    """
    Find similar x-candle windows for y timeframe. Use features param to query with a,b,c,d...
    Use priorities to weight features (e.g. priorities=close_norm:2,rsi:1.5). Call /train first.
    """
    feat_list = [f.strip() for f in features.split(",")] if features else None
    prio_dict = _parse_priorities(priorities)
    result = search_llm1_similar(
        timeframe=timeframe,
        symbol=symbol,
        window_size=window_size,
        datetime_iso=datetime,
        features=feat_list,
        priorities=prio_dict,
        k=k,
        metric=metric,
        min_hours_apart=min_hours_apart,
        min_similarity=min_similarity,
        timezone=timezone,
    )
    if "error" in result and not result.get("similar_windows"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/similarity-batch")
def llm1_similarity_batch(
    timeframe: str = Query(..., description="Timeframe (y)"),
    symbol: str = Query("BTCUSDT", description="Symbol"),
    window_size: int = Query(8, ge=2, le=50, description="Window size (x candles)"),
    n_samples: int = Query(10, ge=1, le=200, description="Number of random candles to sample"),
    features: Optional[str] = Query(None, description="Comma-separated features to use"),
    priorities: Optional[str] = Query(None, description="Feature weights: 'close_norm:2,rsi:1.5'"),
    k: int = Query(10, ge=1, le=100),
    metric: str = Query("cosine"),
    min_hours_apart: float = Query(168, ge=0),
    min_similarity: float = Query(0.99, ge=0, le=1),
    timezone: str = Query("Asia/Yerevan"),
    weekday: Optional[int] = Query(None, ge=0, le=6, description="Filter by weekday (0=Mon..6=Sun)"),
    hours_start: Optional[str] = Query(None, description="Filter hours start HH:MM (local tz)"),
    hours_end: Optional[str] = Query(None, description="Filter hours end HH:MM (local tz)"),
):
    """
    Sample n_samples random valid timestamps from llm1_candles, run similarity for each,
    return all results. Use for batch validation.
    """
    feat_list = [f.strip() for f in features.split(",")] if features else None
    prio_dict = _parse_priorities(priorities)
    result = search_llm1_similar_batch(
        timeframe=timeframe,
        symbol=symbol,
        window_size=window_size,
        n_samples=n_samples,
        features=feat_list,
        priorities=prio_dict,
        k=k,
        metric=metric,
        min_hours_apart=min_hours_apart,
        min_similarity=min_similarity,
        timezone=timezone,
        weekday=weekday,
        hours_start=hours_start,
        hours_end=hours_end,
    )
    if "error" in result and not result.get("results"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/similarity-by-times")
def llm1_similarity_by_times(
    body: dict = Body(
        ...,
        example={
            "datetimes": ["2024-01-15T12:00:00Z", "2024-01-16T08:00:00Z"],
            "timeframe": "1h",
            "symbol": "BTCUSDT",
            "window_size": 8,
            "features": ["rsi", "close_norm"],
        },
    ),
):
    """
    Run similarity for each given datetime. Send datetimes and timeframe in body.
    Returns results for your given times only.
    """
    datetimes = body.get("datetimes") or []
    timeframe = body.get("timeframe") or "1h"
    symbol = body.get("symbol") or "BTCUSDT"
    window_size = body.get("window_size", 8)
    features = body.get("features")
    if isinstance(features, list):
        feat_list = [str(f) for f in features]
    elif isinstance(features, str):
        feat_list = [f.strip() for f in features.split(",")] if features else None
    else:
        feat_list = None
    priorities = body.get("priorities")
    if isinstance(priorities, str):
        prio_dict = _parse_priorities(priorities)
    elif isinstance(priorities, dict):
        prio_dict = {str(k): float(v) for k, v in priorities.items() if isinstance(v, (int, float))}
    else:
        prio_dict = None

    result = search_llm1_similar_by_datetimes(
        datetimes=datetimes,
        timeframe=timeframe,
        symbol=symbol,
        window_size=window_size,
        features=feat_list,
        priorities=prio_dict,
        k=body.get("k", 10),
        metric=body.get("metric", "cosine"),
        min_hours_apart=body.get("min_hours_apart", 168),
        min_similarity=body.get("min_similarity", 0.999),
        timezone=body.get("timezone", "Asia/Yerevan"),
    )
    return result


MLPARAMS_COLLECTION = "mlParams"


@router.post("/save-params")
def llm1_save_params(
    body: dict = Body(
        ...,
        example={
            "name": "my_config_v1",
            "window_size": 5,
            "timeframe": "1h",
            "symbol": "BTCUSDT",
            "params": {"features": ["open_norm", "close_norm"], "priorities": {"close_norm": 2}},
            "results": [],
            "summary": {"same_colors": 8, "comparable": 10, "accuracy_pct": 80.0},
        },
    ),
):
    """
    Save ML params and test results to mlParams collection.
    """
    from datetime import datetime, timezone
    from pymongo import MongoClient
    from app.services.llm1_service import MONGODB_URI, DB_NAME

    name = body.get("name")
    if not name or not str(name).strip():
        raise HTTPException(status_code=400, detail="name is required")

    doc = {
        "name": str(name).strip(),
        "window_size": int(body.get("window_size", 5)),
        "timeframe": str(body.get("timeframe", "1h")),
        "symbol": str(body.get("symbol", "BTCUSDT")),
        "params": body.get("params") or {},
        "results": body.get("results") or [],
        "summary": body.get("summary") or {},
        "created_at": datetime.now(timezone.utc),
    }

    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    r = db[MLPARAMS_COLLECTION].insert_one(doc)
    return {"success": True, "id": str(r.inserted_id), "name": doc["name"]}


@router.get("/params")
def llm1_list_params(
    limit: int = Query(50, ge=1, le=200),
):
    """List saved ML params from mlParams collection."""
    from pymongo import MongoClient
    from app.services.llm1_service import MONGODB_URI, DB_NAME

    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    cursor = db[MLPARAMS_COLLECTION].find(
        {},
        projection={"name": 1, "window_size": 1, "timeframe": 1, "params": 1, "summary": 1, "created_at": 1},
    ).sort("created_at", -1).limit(limit)
    docs = list(cursor)
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return {"params": docs}


@router.get("/features")
def llm1_features(
    timeframe: Optional[str] = Query(None),
    symbol: str = Query("BTCUSDT"),
    window_size: int = Query(8, ge=2, le=50),
    from_db: bool = Query(True, description="Fetch features from llm1_collection metadata"),
):
    """
    List all features you can use for query. Uses feature_meta from llm1_collection (build).
    If trained, also returns trained_features.
    """
    from pymongo import MongoClient
    from app.services.llm1_service import MONGODB_URI, DB_NAME, LLM1_CANDLES_COLLECTION, LLM1_COLLECTION
    from app.services.llm1_train_service import _get_all_numeric_keys, LLM1_INDEX_BASE
    from app.services.llm1_feature_meta import CANONICAL_FEATURES, FEATURE_META

    result = {
        "training": "Uses canonical features from build metadata",
        "query": "Pass features param to use any subset, e.g. features=open_norm,close_norm,rsi",
    }

    if from_db:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        meta_doc = db[LLM1_COLLECTION].find_one({"symbol": symbol, "timeframe": timeframe or "1h"})
        if meta_doc and meta_doc.get("features"):
            result["all_queryable_features"] = meta_doc["features"]
            result["feature_meta"] = meta_doc.get("feature_meta", FEATURE_META)
        else:
            docs = list(db[LLM1_CANDLES_COLLECTION].find(
                {"symbol": symbol, "timeframe": timeframe or "1h"}
            ).limit(100))
            if docs:
                all_keys = _get_all_numeric_keys(docs)
                result["all_queryable_features"] = all_keys
                result["feature_meta"] = FEATURE_META
            else:
                result["all_queryable_features"] = CANONICAL_FEATURES
                result["feature_meta"] = FEATURE_META
                result["note"] = "No llm1_candles found. Run /build first."

    if timeframe:
        import json
        from pathlib import Path
        base = LLM1_INDEX_BASE if LLM1_INDEX_BASE.is_absolute() else Path.cwd() / LLM1_INDEX_BASE
        save_dir = base / f"{symbol}_{timeframe}_w{window_size}"
        meta_path = save_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            result["trained_features"] = meta.get("features", [])

    return result
