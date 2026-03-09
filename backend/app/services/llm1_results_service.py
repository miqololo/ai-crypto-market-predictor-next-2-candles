"""
Save LLM1 similarity results to MongoDB in LLM-understandable text format.
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo import MongoClient

from app.services.llm1_service import DB_NAME, MONGODB_URI

LLM1_RESULTS_COLLECTION = "llm1_similarity_results"


def _result_to_llm_text(result: Dict[str, Any]) -> str:
    """Convert similarity result to human-readable text for LLM consumption."""
    lines = []
    qi = result.get("query_info") or {}
    symbol = qi.get("symbol", "BTCUSDT")
    tf = qi.get("timeframe", "1h")
    query_end = qi.get("query_end", "")
    features = qi.get("features_used", [])

    lines.append(f"## LLM1 Similarity Result")
    lines.append(f"Symbol: {symbol} | Timeframe: {tf} | Query: {query_end}")
    if features:
        lines.append(f"Features: {', '.join(features)}")
    lines.append("")

    # Current candle
    cc = result.get("current_candle") or {}
    price = cc.get("price") or {}
    if price:
        o, h, l_, c = price.get("open"), price.get("high"), price.get("low"), price.get("close")
        color = "green" if (c is not None and o is not None and c >= o) else "red"
        lines.append(f"**Current candle:** O={o} H={h} L={l_} C={c} ({color})")
        lines.append("")

    # Prediction
    pred = result.get("prediction") or {}
    if pred:
        n1 = pred.get("next_1") or {}
        pred_color = n1.get("candle", "?")
        pred_close = n1.get("close", "?")
        lines.append(f"**Predicted next 1:** {pred_color}, close {pred_close}")

        # Actual
        actual = result.get("actual_next") or {}
        a1 = actual.get("actual_next_1")
        if a1:
            ac = a1.get("close")
            ao = a1.get("open")
            actual_color = "green" if (ac is not None and ao is not None and ac >= ao) else "red"
            lines.append(f"**Actual next 1:** {actual_color}, close {ac}")
            match = "Yes" if str(pred_color).lower() == actual_color else "No"
            lines.append(f"**Match:** {match}")
        else:
            lines.append("**Actual next 1:** N/A (no historical data)")
        lines.append("")

    # Most similar
    similar = result.get("similar_windows") or []
    if similar:
        top = similar[0]
        ts = top.get("timestamp", "?")
        score = top.get("similarity_score", 0)
        pct = round(score * 100, 1) if isinstance(score, (int, float)) else score
        lines.append(f"**Most similar:** {ts} ({pct}% similar)")

    return "\n".join(lines)


def save_similarity_result(
    result: Dict[str, Any],
    mongodb_uri: Optional[str] = None,
) -> Optional[str]:
    """
    Save similarity result to MongoDB in LLM-understandable format.
    Returns inserted document _id as string, or None on error.
    """
    if "error" in result and not result.get("similar_windows"):
        return None

    try:
        client = MongoClient(mongodb_uri or MONGODB_URI)
        db = client[DB_NAME]
        coll = db[LLM1_RESULTS_COLLECTION]

        qi = result.get("query_info") or {}
        doc = {
            "symbol": qi.get("symbol", "BTCUSDT"),
            "timeframe": qi.get("timeframe", "1h"),
            "window_size": qi.get("window_size", 8),
            "query_datetime": qi.get("requested_datetime"),
            "query_end": qi.get("query_end"),
            "features_used": qi.get("features_used", []),
            "llm_text": _result_to_llm_text(result),
            "result": result,
            "created_at": datetime.now(timezone.utc),
        }
        ins = coll.insert_one(doc)
        return str(ins.inserted_id)
    except Exception:
        return None


def save_similarity_results_batch(
    results: list,
    mongodb_uri: Optional[str] = None,
) -> int:
    """Save multiple similarity results. Returns count of inserted docs."""
    if not results:
        return 0
    try:
        client = MongoClient(mongodb_uri or MONGODB_URI)
        db = client[DB_NAME]
        coll = db[LLM1_RESULTS_COLLECTION]
        docs = []
        for r in results:
            if "error" in r and not r.get("similar_windows"):
                continue
            qi = r.get("query_info") or {}
            docs.append({
                "symbol": qi.get("symbol", "BTCUSDT"),
                "timeframe": qi.get("timeframe", "1h"),
                "window_size": qi.get("window_size", 8),
                "query_datetime": qi.get("requested_datetime"),
                "query_end": qi.get("query_end"),
                "features_used": qi.get("features_used", []),
                "llm_text": _result_to_llm_text(r),
                "result": r,
                "created_at": datetime.now(timezone.utc),
            })
        if docs:
            coll.insert_many(docs)
            return len(docs)
    except Exception:
        pass
    return 0
