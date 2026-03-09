"""AI similarity search endpoint: find similar patterns by timeframe, datetime, and feature flags."""
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.similarity_service import find_similar_patterns
from app.services.reset_model_service import reset_all_models


router = APIRouter(prefix="/ai", tags=["ai"])

# Map boolean flags to feature names for selected_features derivation
FEATURE_FLAGS = [
    "use_open", "use_high", "use_low", "use_close", "use_volume",
    "use_rsi", "use_macd", "use_macd_signal", "use_macd_hist",
    "use_ema_9", "use_ema_12", "use_ema_21", "use_ema_26",
    "use_sma_20", "use_sma_50", "use_sma_200",
    "use_bb_upper", "use_bb_mid", "use_bb_lower",
    "use_stoch_k", "use_stoch_d", "use_cci", "use_atr", "use_psar",
    "use_supertrend", "use_supertrend_dir", "use_vwap", "use_ao",
    "use_tenkan", "use_kijun", "use_senkou_a", "use_senkou_b",
]
FLAG_TO_FEATURE = {
    "use_open": "open", "use_high": "high", "use_low": "low",
    "use_close": "close", "use_volume": "volume",
    "use_rsi": "rsi", "use_macd": "macd", "use_macd_signal": "macd_signal",
    "use_macd_hist": "macd_hist",
    "use_ema_9": "ema_9", "use_ema_12": "ema_12", "use_ema_21": "ema_21",
    "use_ema_26": "ema_26",
    "use_sma_20": "sma_20", "use_sma_50": "sma_50", "use_sma_200": "sma_200",
    "use_bb_upper": "bb_upper", "use_bb_mid": "bb_mid", "use_bb_lower": "bb_lower",
    "use_stoch_k": "stoch_k", "use_stoch_d": "stoch_d",
    "use_cci": "cci", "use_atr": "atr", "use_psar": "psar",
    "use_supertrend": "supertrend", "use_supertrend_dir": "supertrend_dir",
    "use_vwap": "vwap", "use_ao": "ao",
    "use_tenkan": "tenkan", "use_kijun": "kijun",
    "use_senkou_a": "senkou_a", "use_senkou_b": "senkou_b",
}


def _features_from_flags(flags: Dict[str, bool]) -> List[str]:
    """Derive selected_features from use_* boolean flags."""
    return [FLAG_TO_FEATURE[k] for k in FEATURE_FLAGS if flags.get(k) is True]


class SimilarityRequest(BaseModel):
    """Request body for similarity search. All fields optional."""

    timeframe: Optional[str] = Field(
        default="1h",
        description="Candle timeframe: 1h, 15m, 5m",
    )
    symbol: Optional[str] = Field(
        default="BTCUSDT",
        description="Trading symbol",
    )
    datetime: Optional[str] = Field(
        default=None,
        description="ISO datetime (e.g. 2025-02-19T12:00:00) to use that candle as query. If omitted, uses latest candle from DB.",
    )
    selected_features: Optional[List[str]] = Field(
        default=None,
        description="Features to use for similarity. If omitted: uses feature flags (use_*).",
    )
    window_size: Optional[int] = Field(
        default=None,
        ge=2,
        le=50,
        description="If set, use sliding-window search over last N candles (instead of single-point).",
    )
    metric: Optional[str] = Field(
        default="cosine",
        description="Similarity metric: cosine or euclidean",
    )
    k: Optional[int] = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of similar patterns to return",
    )
    limit: Optional[int] = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Max candles to load from DB for search",
    )
    missing_strategy: Optional[str] = Field(
        default="impute_zero",
        description="How to handle NaN: skip, impute_zero, impute_mean",
    )
    # Feature flags: include in similarity when true
    use_open: Optional[bool] = Field(default=False, description="Include open price")
    use_high: Optional[bool] = Field(default=False, description="Include high price")
    use_low: Optional[bool] = Field(default=False, description="Include low price")
    use_close: Optional[bool] = Field(default=True, description="Include close price")
    use_volume: Optional[bool] = Field(default=True, description="Include volume")
    use_rsi: Optional[bool] = Field(default=True, description="Include RSI")
    use_macd: Optional[bool] = Field(default=False, description="Include MACD")
    use_macd_signal: Optional[bool] = Field(default=False, description="Include MACD signal")
    use_macd_hist: Optional[bool] = Field(default=True, description="Include MACD histogram")
    use_ema_9: Optional[bool] = Field(default=False, description="Include EMA 9")
    use_ema_12: Optional[bool] = Field(default=False, description="Include EMA 12")
    use_ema_21: Optional[bool] = Field(default=False, description="Include EMA 21")
    use_ema_26: Optional[bool] = Field(default=False, description="Include EMA 26")
    use_sma_20: Optional[bool] = Field(default=False, description="Include SMA 20")
    use_sma_50: Optional[bool] = Field(default=False, description="Include SMA 50")
    use_sma_200: Optional[bool] = Field(default=False, description="Include SMA 200")
    use_bb_upper: Optional[bool] = Field(default=False, description="Include Bollinger upper")
    use_bb_mid: Optional[bool] = Field(default=False, description="Include Bollinger mid")
    use_bb_lower: Optional[bool] = Field(default=False, description="Include Bollinger lower")
    use_stoch_k: Optional[bool] = Field(default=False, description="Include Stochastic %K")
    use_stoch_d: Optional[bool] = Field(default=False, description="Include Stochastic %D")
    use_cci: Optional[bool] = Field(default=False, description="Include CCI")
    use_atr: Optional[bool] = Field(default=True, description="Include ATR")
    use_psar: Optional[bool] = Field(default=False, description="Include Parabolic SAR")
    use_supertrend: Optional[bool] = Field(default=False, description="Include Supertrend")
    use_supertrend_dir: Optional[bool] = Field(default=False, description="Include Supertrend direction")
    use_vwap: Optional[bool] = Field(default=False, description="Include VWAP")
    use_ao: Optional[bool] = Field(default=False, description="Include Awesome Oscillator")
    use_tenkan: Optional[bool] = Field(default=False, description="Include Ichimoku Tenkan")
    use_kijun: Optional[bool] = Field(default=False, description="Include Ichimoku Kijun")
    use_senkou_a: Optional[bool] = Field(default=False, description="Include Ichimoku Senkou A")
    use_senkou_b: Optional[bool] = Field(default=False, description="Include Ichimoku Senkou B")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "timeframe": "1h",
                    "symbol": "BTCUSDT",
                    "datetime": "2025-02-19T12:00:00",
                    "selected_features": None,
                    "window_size": None,
                    "metric": "cosine",
                    "k": 10,
                    "limit": 2000,
                    "missing_strategy": "impute_zero",
                    "use_open": False,
                    "use_high": False,
                    "use_low": False,
                    "use_close": True,
                    "use_volume": True,
                    "use_rsi": True,
                    "use_macd": False,
                    "use_macd_signal": False,
                    "use_macd_hist": True,
                    "use_ema_9": False,
                    "use_ema_12": False,
                    "use_ema_21": False,
                    "use_ema_26": False,
                    "use_sma_20": False,
                    "use_sma_50": False,
                    "use_sma_200": False,
                    "use_bb_upper": False,
                    "use_bb_mid": False,
                    "use_bb_lower": False,
                    "use_stoch_k": False,
                    "use_stoch_d": False,
                    "use_cci": False,
                    "use_atr": True,
                    "use_psar": False,
                    "use_supertrend": False,
                    "use_supertrend_dir": False,
                    "use_vwap": False,
                    "use_ao": False,
                    "use_tenkan": False,
                    "use_kijun": False,
                    "use_senkou_a": False,
                    "use_senkou_b": False,
                },
                {
                    "timeframe": "15m",
                    "symbol": "BTCUSDT",
                    "datetime": None,
                    "selected_features": None,
                    "window_size": 8,
                    "metric": "euclidean",
                    "k": 5,
                    "limit": 1000,
                    "missing_strategy": "impute_zero",
                    "use_open": True,
                    "use_high": True,
                    "use_low": True,
                    "use_close": True,
                    "use_volume": True,
                    "use_rsi": True,
                    "use_macd": False,
                    "use_macd_signal": False,
                    "use_macd_hist": True,
                    "use_ema_9": False,
                    "use_ema_12": True,
                    "use_ema_21": False,
                    "use_ema_26": False,
                    "use_sma_20": False,
                    "use_sma_50": False,
                    "use_sma_200": False,
                    "use_bb_upper": False,
                    "use_bb_mid": False,
                    "use_bb_lower": False,
                    "use_stoch_k": False,
                    "use_stoch_d": False,
                    "use_cci": False,
                    "use_atr": True,
                    "use_psar": False,
                    "use_supertrend": False,
                    "use_supertrend_dir": False,
                    "use_vwap": False,
                    "use_ao": False,
                    "use_tenkan": False,
                    "use_kijun": False,
                    "use_senkou_a": False,
                    "use_senkou_b": False,
                },
            ]
        }
    }


READY_TO_RUN_EXAMPLE = {
    "timeframe": "1h",
    "symbol": "BTCUSDT",
    "datetime": "2025-02-19T12:00:00",
    "window_size": None,
    "metric": "cosine",
    "k": 10,
    "limit": 2000,
    "missing_strategy": "impute_zero",
    "use_open": False,
    "use_high": False,
    "use_low": False,
    "use_close": True,
    "use_volume": True,
    "use_rsi": True,
    "use_macd": False,
    "use_macd_signal": False,
    "use_macd_hist": True,
    "use_ema_9": False,
    "use_ema_12": False,
    "use_ema_21": False,
    "use_ema_26": False,
    "use_sma_20": False,
    "use_sma_50": False,
    "use_sma_200": False,
    "use_bb_upper": False,
    "use_bb_mid": False,
    "use_bb_lower": False,
    "use_stoch_k": False,
    "use_stoch_d": False,
    "use_cci": False,
    "use_atr": True,
    "use_psar": False,
    "use_supertrend": False,
    "use_supertrend_dir": False,
    "use_vwap": False,
    "use_ao": False,
    "use_tenkan": False,
    "use_kijun": False,
    "use_senkou_a": False,
    "use_senkou_b": False,
}


@router.post(
    "/similarity",
    summary="Find similar patterns",
    response_description="Similar patterns with timestamps and scores",
)
def similarity_search(
    req: SimilarityRequest = Body(
        ...,
        openapi_examples={
            "ready_to_run": {
                "summary": "Ready-to-run (datetime + feature flags)",
                "value": READY_TO_RUN_EXAMPLE,
            },
            "latest_candle": {
                "summary": "Latest candle (no datetime)",
                "value": {
                    **READY_TO_RUN_EXAMPLE,
                    "datetime": None,
                },
            },
            "window_search": {
                "summary": "Window search (last 8 candles)",
                "value": {
                    **READY_TO_RUN_EXAMPLE,
                    "datetime": None,
                    "window_size": 8,
                    "metric": "euclidean",
                    "k": 5,
                    "use_open": True,
                    "use_high": True,
                    "use_low": True,
                    "use_ema_12": True,
                },
            },
            "minimal": {
                "summary": "Minimal (defaults only)",
                "value": {"timeframe": "1h", "symbol": "BTCUSDT"},
            },
        },
    ),
):
    """
    Find historical patterns similar to the query candle.

    - **timeframe**: Candle timeframe (1h, 15m, 5m)
    - **datetime**: ISO datetime to use that candle as query; if omitted, uses latest candle from DB
    - **use_***: Boolean flags to include each feature in similarity (e.g. use_rsi, use_close)
    - **window_size**: If set, search by sliding window (last N candles) instead of single point
    - **metric**: cosine (default) or euclidean
    - **k**: Number of similar patterns to return
    """
    if req.metric and req.metric not in ("cosine", "euclidean"):
        raise HTTPException(status_code=400, detail="metric must be 'cosine' or 'euclidean'")
    if req.missing_strategy and req.missing_strategy not in ("skip", "impute_zero", "impute_mean"):
        raise HTTPException(status_code=400, detail="missing_strategy must be skip, impute_zero, or impute_mean")

    # Derive selected_features from use_* flags when selected_features not provided
    flags = {k: getattr(req, k) for k in FEATURE_FLAGS if hasattr(req, k)}
    features_from_flags = _features_from_flags(flags)
    selected_features = req.selected_features
    if selected_features is None and features_from_flags:
        selected_features = features_from_flags

    result = find_similar_patterns(
        timeframe=req.timeframe or "1h",
        symbol=req.symbol or "BTCUSDT",
        datetime_iso=req.datetime,
        selected_features=selected_features,
        window_size=req.window_size,
        metric=req.metric or "cosine",
        k=req.k or 10,
        limit=req.limit or 2000,
        missing_strategy=req.missing_strategy or "impute_zero",
    )

    if "error" in result and result["error"] and not result.get("similar_patterns"):
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/model/reset", summary="Reset AI trained models")
def reset_ai_model(
    llm1: bool = Query(True, description="Reset llm1_collection (MongoDB)"),
    chroma: bool = Query(True, description="Reset ChromaDB persist directory"),
    faiss: bool = Query(True, description="Reset FAISS pattern index (data/patterns/)"),
):
    """
    Reset AI trained models. Clears:
    - **llm1**: MongoDB llm1_collection (pre-computed LLM documents)
    - **chroma**: ChromaDB persist directory (RAG embeddings)
    - **faiss**: FAISS pattern index files in data/patterns/
    """
    result = reset_all_models(llm1=llm1, chroma=chroma, faiss=faiss)
    if not result.get("success", True):
        raise HTTPException(status_code=500, detail=result.get("error", "Reset failed"))
    return result
