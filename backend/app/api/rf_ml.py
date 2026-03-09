"""API routes for RF price direction prediction."""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()

# Model path relative to backend root
RF_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "rf_ml" / "models" / "rf_model.pkl"


class TrainRequest(BaseModel):
    timeframe: str = "5m"
    symbol: str = "BTCUSDT"
    limit: Optional[int] = None
    train_start: Optional[str] = None  # ISO date e.g. "2024-01-01"
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    target_symmetric: bool = True  # up if >t, down if <-t, drop flat
    use_calibration: bool = True
    tune_threshold: bool = True
    model_type: str = "rf"  # rf, xgb, lgb (rf most stable under uvicorn)


class EvalRequest(BaseModel):
    timeframe: str = "5m"
    symbol: str = "BTCUSDT"
    limit: Optional[int] = None


class FeaturesPreviewRequest(BaseModel):
    timeframe: str = "5m"
    symbol: str = "BTCUSDT"
    limit: Optional[int] = None
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    target_symmetric: bool = True


@router.post("/features-preview")
def rf_features_preview(req: FeaturesPreviewRequest):
    """Preview features and latest values for given params — no training required."""
    try:
        from rf_ml.config import RFConfig
        from rf_ml.data_loader import prepare_dataset

        config = RFConfig(
            candle_timeframe=req.timeframe,
            symbol=req.symbol,
            mongodb_uri=os.getenv("MONGODB_URI", ""),
            target_symmetric=req.target_symmetric,
        )
        use_dates = all(x is not None for x in (req.train_start, req.train_end, req.test_start, req.test_end))
        if use_dates:
            result = prepare_dataset(
                config, limit=req.limit,
                train_start=req.train_start, train_end=req.train_end,
                test_start=req.test_start, test_end=req.test_end,
            )
            X_train, y_train, X_test, y_test, feature_names = result[:5]
            X = pd.concat([X_train, X_test])
        else:
            result = prepare_dataset(config, limit=req.limit)
            X, y, feature_names = result[:3]
        last_row = X.iloc[-1]
        sample_values = {name: _safe_value(last_row.get(name)) for name in feature_names}
        return {
            "ok": True,
            "feature_names": feature_names,
            "sample_values": sample_values,
            "n_rows": len(X),
            "all_valid": not any(v is None for v in sample_values.values()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _safe_value(v) -> Optional[float]:
    """Convert to JSON-serializable value."""
    if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@router.post("/reset")
def rf_reset():
    """Delete trained model. Call /train to retrain from scratch."""
    if not RF_MODEL_PATH.exists():
        return {"success": True, "removed": False, "message": "No model to reset", "next_step": "Call POST /api/rf/train to train"}
    try:
        RF_MODEL_PATH.unlink()
        return {"success": True, "removed": True, "next_step": "Call POST /api/rf/train to retrain"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def rf_status():
    """Check if RF model is trained and available."""
    exists = RF_MODEL_PATH.exists()
    return {
        "trained": exists,
        "path": str(RF_MODEL_PATH),
        "hint": "Call POST /api/rf/train to train the model" if not exists else None,
    }


@router.get("/features")
def rf_features():
    """Return feature names used in the trained model."""
    if not RF_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not trained. Call POST /api/rf/train first.")
    try:
        from rf_ml.model import RFPricePredictor
        predictor = RFPricePredictor.load(str(RF_MODEL_PATH))
        return {"ok": True, "feature_names": predictor.feature_names or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
def rf_train(req: TrainRequest):
    """Train RF model on MongoDB candle data."""
    try:
        from rf_ml.config import RFConfig
        from rf_ml.training import train_model

        config = RFConfig(
            candle_timeframe=req.timeframe,
            symbol=req.symbol,
            mongodb_uri=os.getenv("MONGODB_URI", ""),
            target_symmetric=req.target_symmetric,
            use_calibration=req.use_calibration,
            tune_threshold=req.tune_threshold,
            model_type=req.model_type,
        )
        predictor, metrics = train_model(
            config=config,
            limit=req.limit,
            save_path=str(RF_MODEL_PATH),
            train_start=req.train_start,
            train_end=req.train_end,
            test_start=req.test_start,
            test_end=req.test_end,
        )
        importance = predictor.get_feature_importance()
        top_features = list(importance.items())[:15]
        return {
            "ok": True,
            "metrics": metrics,
            "top_features": [{"name": k, "importance": float(v)} for k, v in top_features],
            "feature_names": predictor.feature_names or [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/eval")
def rf_eval(req: EvalRequest):
    """Evaluate trained RF model."""
    if not RF_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not trained. Call POST /api/rf/train first.")
    try:
        from rf_ml.config import RFConfig
        from rf_ml.model import RFPricePredictor
        from rf_ml.evaluation import evaluate_model

        config = RFConfig(
            candle_timeframe=req.timeframe,
            symbol=req.symbol,
            mongodb_uri=os.getenv("MONGODB_URI", ""),
        )
        predictor = RFPricePredictor.load(str(RF_MODEL_PATH))
        metrics = evaluate_model(predictor, config=config, limit=req.limit)
        return {"ok": True, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
def rf_backtest(
    timeframe: str = Query("5m"),
    symbol: str = Query("BTCUSDT"),
    limit: Optional[int] = Query(None),
    threshold: float = Query(0.6, ge=0.5, le=0.95),
    test_start: Optional[str] = Query(None),
    test_end: Optional[str] = Query(None),
):
    """Run simple backtest with trained model. Uses test period when test_start/test_end provided."""
    if not RF_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not trained. Call POST /api/rf/train first.")
    try:
        from rf_ml.config import RFConfig
        from rf_ml.model import RFPricePredictor
        from rf_ml.evaluation import backtest_simple

        config = RFConfig(
            candle_timeframe=timeframe,
            symbol=symbol,
            mongodb_uri=os.getenv("MONGODB_URI", ""),
        )
        predictor = RFPricePredictor.load(str(RF_MODEL_PATH))
        result = backtest_simple(
            predictor,
            config=config,
            limit=limit,
            proba_threshold=threshold,
            test_start=test_start,
            test_end=test_end,
        )
        return {"ok": True, "backtest": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WalkForwardBacktestRequest(BaseModel):
    date_from: str  # e.g. "2024-01-01"
    date_to: str
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    threshold: float = 0.60
    risk_percent: float = 1.0
    commission_percent: float = 0.1
    slippage_percent: float = 0.03
    initial_capital: float = 50000.0


@router.post("/walk-forward-backtest")
def rf_walk_forward_backtest(req: WalkForwardBacktestRequest):
    """Walk-forward backtest: simulates real trading with no lookahead."""
    if not RF_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not trained. Call POST /api/rf/train first.")
    try:
        from rf_ml.walk_forward_backtest import run_walk_forward_backtest, BacktestParams

        params = BacktestParams(
            date_from=req.date_from,
            date_to=req.date_to,
            symbol=req.symbol,
            timeframe=req.timeframe,
            threshold=req.threshold,
            risk_percent=req.risk_percent,
            commission_percent=req.commission_percent,
            slippage_percent=req.slippage_percent,
            initial_capital=req.initial_capital,
            model_path=str(RF_MODEL_PATH),
            mongodb_uri=os.getenv("MONGODB_URI"),
        )
        result = run_walk_forward_backtest(params)
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    timeframe: str = "1h"
    symbol: str = "BTCUSDT"
    limit: Optional[int] = 100


class PredictionsRangeRequest(BaseModel):
    timeframe: str = "1h"
    symbol: str = "BTCUSDT"
    start_date: str  # e.g. "2024-01-01"
    end_date: str
    limit: Optional[int] = None


@router.post("/predictions-range")
def rf_predictions_range(req: PredictionsRangeRequest):
    """Return all predictions in the given date range (for train page)."""
    if not RF_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not trained. Call POST /api/rf/train first.")
    try:
        from rf_ml.config import RFConfig
        from rf_ml.model import RFPricePredictor
        from rf_ml.data_loader import load_candles_from_mongo
        from rf_ml.features import build_features, get_feature_columns, normalize_features

        predictor = RFPricePredictor.load(str(RF_MODEL_PATH))
        cfg = predictor.config or RFConfig()
        cfg.candle_timeframe = req.timeframe
        cfg.symbol = req.symbol
        cfg.mongodb_uri = os.getenv("MONGODB_URI", "")

        df, docs = load_candles_from_mongo(
            symbol=req.symbol,
            timeframe=req.timeframe,
            mongodb_uri=cfg.mongodb_uri,
            start_date=req.start_date,
            end_date=req.end_date,
            limit=req.limit,
        )
        if df.empty and req.timeframe in ("15m", "1h"):
            df, docs = load_candles_from_mongo(
                symbol=req.symbol,
                timeframe=req.timeframe,
                mongodb_uri=cfg.mongodb_uri,
                start_date=req.start_date,
                end_date=req.end_date,
                limit=req.limit,
                resample_from="5m",
            )
        if df.empty or len(df) < 50:
            raise HTTPException(status_code=400, detail="Insufficient candle data in range")

        df = build_features(df, docs, cfg)
        df = normalize_features(df, cfg)
        available = get_feature_columns(cfg, df=df)
        feat = [f for f in predictor.feature_names if f in available]
        if len(feat) != len(predictor.feature_names):
            raise HTTPException(status_code=400, detail="Feature mismatch")

        X = df[feat].fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)
        closes = df["close"].values
        lows = df["low"].values
        highs = df["high"].values

        n_fwd = cfg.candles_per_prediction or {"1h": 2, "15m": 8, "5m": 24}.get(req.timeframe, 2)
        dir_threshold = cfg.direction_threshold_pct / 100  # e.g. 0.005 for 0.5%
        # Actual low/high of next n_fwd candles (min low, max high)
        fwd_low = np.full(len(df), np.nan)
        fwd_high = np.full(len(df), np.nan)
        for i in range(len(df) - n_fwd):
            fwd_low[i] = np.min(lows[i + 1 : i + n_fwd + 1])
            fwd_high[i] = np.max(highs[i + 1 : i + n_fwd + 1])

        proba_arr = predictor.predict_proba(X.values)[:, 1]
        pred_low_arr, pred_high_arr = predictor.predict_range(X.values, closes)
        th = predictor.optimal_threshold

        rows = []
        for i in range(len(df)):
            ts = str(df.index[i])
            proba = float(proba_arr[i])
            direction = 1 if proba >= th else 0
            pl = float(pred_low_arr[i]) if pred_low_arr is not None else None
            ph = float(pred_high_arr[i]) if pred_high_arr is not None else None
            al = float(fwd_low[i]) if not np.isnan(fwd_low[i]) else None
            ah = float(fwd_high[i]) if not np.isnan(fwd_high[i]) else None

            # Confidence: distance from 0.5 (0–100%)
            confidence_pct = round(abs(proba - 0.5) * 200, 1)

            # Accuracy relative to previous candle (% move from prev close)
            low_acc = None
            high_acc = None
            if i > 0 and pl is not None and al is not None:
                close_prev = closes[i - 1]
                if close_prev > 0:
                    pred_low_pct = (pl - close_prev) / close_prev * 100
                    actual_low_pct = (al - close_prev) / close_prev * 100
                    err = abs(pred_low_pct - actual_low_pct)
                    low_acc = round(100 - min(100, err), 1)
            if i > 0 and ph is not None and ah is not None:
                close_prev = closes[i - 1]
                if close_prev > 0:
                    pred_high_pct = (ph - close_prev) / close_prev * 100
                    actual_high_pct = (ah - close_prev) / close_prev * 100
                    err = abs(pred_high_pct - actual_high_pct)
                    high_acc = round(100 - min(100, err), 1)

            # Actual direction: forward return vs threshold (symmetric: drop flat)
            actual_direction = None
            correct = None
            if i + n_fwd < len(df) and closes[i] > 0:
                fwd_return = (closes[i + n_fwd] - closes[i]) / closes[i]
                if fwd_return > dir_threshold:
                    actual_direction = 1
                elif fwd_return < -dir_threshold:
                    actual_direction = 0
                if actual_direction is not None:
                    correct = direction == actual_direction

            rows.append({
                "timestamp": ts,
                "close": round(float(closes[i]), 2),
                "direction": direction,
                "proba_up": round(proba, 4),
                "confidence_pct": confidence_pct,
                "predicted_low": round(pl, 2) if pl is not None else None,
                "predicted_high": round(ph, 2) if ph is not None else None,
                "actual_low": round(al, 2) if al is not None else None,
                "actual_high": round(ah, 2) if ah is not None else None,
                "low_accuracy_pct": low_acc,
                "high_accuracy_pct": high_acc,
                "actual_direction": actual_direction,
                "correct": correct,
            })
        n_correct = sum(1 for r in rows if r.get("correct") is True)
        n_incorrect = sum(1 for r in rows if r.get("correct") is False)
        n_evaluable = n_correct + n_incorrect
        total_accuracy = round(n_correct / n_evaluable * 100, 1) if n_evaluable > 0 else None
        return {
            "ok": True,
            "predictions": rows,
            "n": len(rows),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "total_accuracy": total_accuracy,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
def rf_predict(req: PredictRequest):
    """Predict direction and low/high of next N candles for latest bar."""
    if not RF_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not trained. Call POST /api/rf/train first.")
    try:
        from rf_ml.config import RFConfig
        from rf_ml.model import RFPricePredictor
        from rf_ml.data_loader import load_candles_from_mongo
        from rf_ml.features import build_features, get_feature_columns, normalize_features

        predictor = RFPricePredictor.load(str(RF_MODEL_PATH))
        cfg = predictor.config or RFConfig()
        cfg.candle_timeframe = req.timeframe
        cfg.symbol = req.symbol
        cfg.mongodb_uri = os.getenv("MONGODB_URI", "")
        df, docs = load_candles_from_mongo(
            symbol=req.symbol,
            timeframe=req.timeframe,
            limit=req.limit or 500,
            mongodb_uri=cfg.mongodb_uri,
        )
        if df.empty and req.timeframe in ("15m", "1h"):
            df, docs = load_candles_from_mongo(
                symbol=req.symbol,
                timeframe=req.timeframe,
                limit=req.limit or 500,
                mongodb_uri=cfg.mongodb_uri,
                resample_from="5m",
            )
        if df.empty or len(df) < 50:
            raise HTTPException(status_code=400, detail="Insufficient candle data")
        df = build_features(df, docs, cfg)
        df = normalize_features(df, cfg)
        available = get_feature_columns(cfg, df=df)
        feat = [f for f in predictor.feature_names if f in available]
        if len(feat) != len(predictor.feature_names):
            raise HTTPException(status_code=400, detail="Feature mismatch")
        X = df[feat].fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_last = X.iloc[-1:].values
        close_last = float(df["close"].iloc[-1])
        proba = predictor.predict_proba(X_last)[0, 1]
        direction = 1 if proba >= predictor.optimal_threshold else 0
        pred_low, pred_high = predictor.predict_range(X_last, [close_last])
        return {
            "ok": True,
            "timestamp": str(df.index[-1]),
            "close": close_last,
            "direction": direction,
            "proba_up": round(float(proba), 4),
            "predicted_low": round(float(pred_low[0]), 2) if pred_low is not None else None,
            "predicted_high": round(float(pred_high[0]), 2) if pred_high is not None else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv")
def rf_cv(
    timeframe: str = Query("5m"),
    symbol: str = Query("BTCUSDT"),
    limit: Optional[int] = Query(None),
    n_splits: int = Query(5, ge=2, le=10),
):
    """Time-series cross-validation."""
    try:
        from rf_ml.config import RFConfig
        from rf_ml.evaluation import time_series_cv

        config = RFConfig(
            candle_timeframe=timeframe,
            symbol=symbol,
            mongodb_uri=os.getenv("MONGODB_URI", ""),
        )
        results = time_series_cv(config, n_splits=n_splits, limit=limit)
        return {"ok": True, "folds": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
