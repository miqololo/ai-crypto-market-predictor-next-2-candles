"""API routes: data, patterns, forecast, backtest (frontend-compatible)."""
import os
import logging
from typing import Optional, Dict, Any, Tuple

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd
import numpy as np

from app.data import DataFetcher
from app.analysis.analyzer import run_analysis
from app.analysis.trend import determine_trend
from app.analysis.timeframe_config import get_config
from app.indicators.full_ta import add_full_indicators
from app.patterns.pattern_search import extract_patterns, PatternSearch
from app.backtest.runner import run_backtest, BacktestEngine
from app.strategies.loader import load_strategy_from_file


router = APIRouter()


def validate_dataframe_for_backtest(df: pd.DataFrame, min_rows: int = 20) -> Tuple[bool, Optional[str], Dict[str, Any], pd.DataFrame]:
    """
    Comprehensive validation of DataFrame before backtesting.
    
    Returns:
        (is_valid, error_message, diagnostics_dict, cleaned_dataframe)
    """
    diagnostics = {
        "original_length": len(df),
        "issues": []
    }
    
    # Work on a copy to avoid modifying original
    df_clean = df.copy()
    
    # 1. Check if DataFrame is empty
    if df_clean is None:
        return False, "DataFrame is None", diagnostics, df_clean
    if len(df_clean) == 0:
        return False, "DataFrame is empty", diagnostics, df_clean
    
    # 2. Check required OHLCV columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    if missing_cols:
        diagnostics["issues"].append(f"Missing required columns: {missing_cols}")
        return False, f"Missing required columns: {', '.join(missing_cols)}", diagnostics, df_clean
    
    # 3. Check index type (should be DatetimeIndex)
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        diagnostics["issues"].append("Index is not DatetimeIndex")
        try:
            df_clean.index = pd.to_datetime(df_clean.index)
            diagnostics["issues"].append("Converted index to DatetimeIndex")
        except Exception as e:
            return False, f"Index cannot be converted to DatetimeIndex: {str(e)}", diagnostics, df_clean
    
    # 4. Check for duplicate timestamps
    duplicate_timestamps = df_clean.index.duplicated().sum()
    if duplicate_timestamps > 0:
        diagnostics["issues"].append(f"Found {duplicate_timestamps} duplicate timestamps")
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        diagnostics["duplicates_removed"] = duplicate_timestamps
    
    # 5. Check for NaN values in OHLCV columns
    ohlcv_nan_counts = {}
    for col in required_cols:
        nan_count = df_clean[col].isna().sum()
        if nan_count > 0:
            ohlcv_nan_counts[col] = nan_count
    
    if ohlcv_nan_counts:
        diagnostics["issues"].append(f"NaN values in OHLCV: {ohlcv_nan_counts}")
        # Remove rows with NaN in critical OHLCV columns
        before_nan_removal = len(df_clean)
        df_clean = df_clean.dropna(subset=["open", "high", "low", "close"])
        diagnostics["rows_removed_due_to_nan"] = before_nan_removal - len(df_clean)
    
    # 6. Check data length after cleaning
    if len(df_clean) < min_rows:
        diagnostics["issues"].append(f"Insufficient rows after cleaning: {len(df_clean)} < {min_rows}")
        return False, f"Insufficient data: only {len(df_clean)} candles available after cleaning (need at least {min_rows})", diagnostics, df_clean
    
    # 7. Validate price data quality
    price_issues = []
    before_price_validation = len(df_clean)
    
    # Check for invalid price relationships
    invalid_high_low = (df_clean['high'] < df_clean['low']).sum()
    if invalid_high_low > 0:
        price_issues.append(f"{invalid_high_low} rows where high < low")
        df_clean = df_clean[df_clean['high'] >= df_clean['low']]
    
    invalid_open = ((df_clean['open'] > df_clean['high']) | (df_clean['open'] < df_clean['low'])).sum()
    if invalid_open > 0:
        price_issues.append(f"{invalid_open} rows where open outside [low, high] range")
        df_clean = df_clean[(df_clean['open'] >= df_clean['low']) & (df_clean['open'] <= df_clean['high'])]
    
    invalid_close = ((df_clean['close'] > df_clean['high']) | (df_clean['close'] < df_clean['low'])).sum()
    if invalid_close > 0:
        price_issues.append(f"{invalid_close} rows where close outside [low, high] range")
        df_clean = df_clean[(df_clean['close'] >= df_clean['low']) & (df_clean['close'] <= df_clean['high'])]
    
    if price_issues:
        diagnostics["issues"].extend(price_issues)
        diagnostics["rows_removed_due_to_price_issues"] = before_price_validation - len(df_clean)
    
    # 8. Check for zero or negative prices
    zero_or_negative = ((df_clean[['open', 'high', 'low', 'close']] <= 0).any(axis=1)).sum()
    if zero_or_negative > 0:
        diagnostics["issues"].append(f"{zero_or_negative} rows with zero or negative prices")
        df_clean = df_clean[(df_clean[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    
    # 9. Check for negative volume (should be >= 0)
    negative_volume = (df_clean['volume'] < 0).sum()
    if negative_volume > 0:
        diagnostics["issues"].append(f"{negative_volume} rows with negative volume")
        df_clean['volume'] = df_clean['volume'].clip(lower=0)
    
    # 10. Check for extremely large price changes (potential data errors) - just warn, don't remove
    if len(df_clean) > 1:
        price_changes = df_clean['close'].pct_change().abs()
        extreme_changes = (price_changes > 0.5).sum()  # More than 50% change
        if extreme_changes > 0:
            diagnostics["issues"].append(f"{extreme_changes} rows with >50% price change (potential data errors)")
    
    # 11. Check for large gaps in timestamps (missing data) - just warn, don't remove
    if len(df_clean) > 1:
        time_diffs = df_clean.index.to_series().diff()
        # Get expected timeframe interval (approximate)
        if len(time_diffs) > 1:
            median_diff = time_diffs.median()
            if pd.notna(median_diff) and median_diff.total_seconds() > 0:
                large_gaps = (time_diffs > median_diff * 3).sum()
                if large_gaps > 0:
                    diagnostics["issues"].append(f"{large_gaps} large time gaps detected (potential missing candles)")
    
    # 12. Final length check
    if len(df_clean) < min_rows:
        diagnostics["issues"].append(f"Final row count {len(df_clean)} < minimum {min_rows}")
        return False, f"Insufficient data after validation: {len(df_clean)} candles (need at least {min_rows})", diagnostics, df_clean
    
    diagnostics["final_length"] = len(df_clean)
    diagnostics["validation_passed"] = True
    
    return True, None, diagnostics, df_clean


class FetchDataRequest(BaseModel):
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    limit: int = 500


class PatternsBuildRequest(BaseModel):
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    limit: int = 500


class ForecastRequest(BaseModel):
    query: str
    n_context: int = 5


class BacktestRequest(BaseModel):
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    limit: int = 500
    engine: str = "vectorbt"
    stop_loss: Optional[float] = 0.01  # Stop loss as fraction (default 1% = 0.01, for 1:3 ratio)
    take_profit: Optional[float] = 0.03  # Take profit as fraction (default 3% = 0.03, for 1:3 ratio)


class BacktestStrategyRequest(BaseModel):
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    limit: int = 500
    engine: str = "vectorbt"
    strategy_file: str  # Path to strategy Python file
    initial_capital: float = 10000.0
    strategy_params: Optional[dict] = None  # Optional parameters to pass to strategy
    use_database: bool = True  # If True, fetch from MongoDB; if False, fetch from Binance API
    stop_loss: Optional[float] = 0.01  # Stop loss as fraction (default 1% = 0.01, for 1:3 ratio)
    take_profit: Optional[float] = 0.03  # Take profit as fraction (default 3% = 0.03, for 1:3 ratio)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validate and ensure stop_loss and take_profit are within reasonable bounds
        if self.stop_loss is not None:
            if self.stop_loss < 0 or self.stop_loss > 1:
                raise ValueError(f"stop_loss must be between 0 and 1 (as fraction), got {self.stop_loss}")
        if self.take_profit is not None:
            if self.take_profit < 0 or self.take_profit > 1:
                raise ValueError(f"take_profit must be between 0 and 1 (as fraction), got {self.take_profit}")


@router.post("/data/fetch")
def data_fetch(req: FetchDataRequest):
    """
    Fetch OHLCV from MongoDB database and run full analysis (indicators, trend, fib, funding, normalized).
    
    **Data Source**: Always uses MongoDB database. Make sure data is loaded using scripts/fetch_binance_to_mongo.py
    """
    try:
        # Fetch from MongoDB
        from pymongo import MongoClient
        from app.services.llm1_service import MONGODB_URI, DB_NAME, CANDLES_COLLECTION, _candles_from_mongo_to_df
        from app.analysis.analyzer import (
            add_full_indicators, determine_trend, calc_fibonacci,
            fetch_funding_sentiment, compute_volume_metrics, compute_normalized_features,
            get_config, _normalize_symbol, _to_native
        )
        
        # Convert symbol format: BTC/USDT:USDT -> BTCUSDT
        mongo_symbol = req.symbol.replace("/", "").replace(":USDT", "")
        
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        candles_coll = db[CANDLES_COLLECTION]
        
        query = {"symbol": mongo_symbol, "timeframe": req.timeframe}
        cursor = candles_coll.find(query).sort("timestamp", -1).limit(req.limit)
        docs = list(cursor)
        
        if not docs:
            raise HTTPException(
                status_code=400,
                detail=f"No data found in database for symbol={mongo_symbol}, timeframe={req.timeframe}. "
                       f"Make sure data is loaded using scripts/fetch_binance_to_mongo.py"
            )
        
        docs.reverse()
        df = _candles_from_mongo_to_df(docs)
        df = add_full_indicators(df, timeframe=req.timeframe)
        df = df.dropna(how="all", subset=[c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]])
        
        cfg = get_config(req.timeframe)
        sym = _normalize_symbol(req.symbol)
        
        # Trend analysis
        is_5m = "5" in req.timeframe and "15" not in req.timeframe
        ema_fast_col = "ema_9" if is_5m else "ema_12"
        rsi_level = 52.5 if "15" in req.timeframe or "1h" in req.timeframe else 50
        ema_medium_col = "ema_26" if "1h" in req.timeframe else "ema_21"
        trend_result = determine_trend(
            df,
            threshold=cfg["trend_threshold"],
            ema_fast_col=ema_fast_col,
            rsi_bullish_level=rsi_level,
            ema_medium_col=ema_medium_col,
        )
        
        # Fibonacci
        fib_result = calc_fibonacci(df, swing_period=cfg["fib_swing_period"])
        
        # Funding (try to fetch, but don't fail if unavailable)
        funding_result = {}
        try:
            funding_result = fetch_funding_sentiment(
                sym, 
                os.getenv("BINANCE_API_KEY"), 
                os.getenv("BINANCE_API_SECRET"), 
                timeframe=req.timeframe
            )
        except Exception as e:
            funding_result = {"error": str(e)}
        
        # Volume metrics
        vol_metrics = compute_volume_metrics(df, timeframe=req.timeframe)
        
        # Normalized features
        norm_features = compute_normalized_features(
            df,
            window=cfg["norm_window"],
            embedding_window=cfg.get("embedding_window", cfg["norm_window"]),
        )
        
        # Last candle summary
        last = df.iloc[-1]
        indicators = {}
        for col in df.columns:
            if col not in ["open", "high", "low", "close", "volume"]:
                val = last[col]
                indicators[col] = _to_native(val)
        
        # Candles (last 100)
        candles = df.tail(100).reset_index()
        candles["timestamp"] = candles["timestamp"].astype(str)
        candles_data = _to_native(candles.to_dict(orient="records"))
        
        return {
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "candle_count": len(df),
            "trend": trend_result,
            "indicators": indicators,
            "fibonacci": fib_result,
            "funding_and_sentiment": funding_result,
            "volume_metrics": vol_metrics,
            "normalized_features": norm_features,
            "candles": candles_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in data_fetch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/build")
def patterns_build(req: PatternsBuildRequest):
    """
    Build FAISS pattern index from OHLCV.
    
    **Data Source**: Fetches data from MongoDB database.
    """
    try:
        # Fetch from MongoDB
        from pymongo import MongoClient
        from app.services.llm1_service import MONGODB_URI, DB_NAME, CANDLES_COLLECTION, _candles_from_mongo_to_df
        
        # Convert symbol format: BTC/USDT:USDT -> BTCUSDT
        mongo_symbol = req.symbol.replace("/", "").replace(":USDT", "")
        
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        candles_coll = db[CANDLES_COLLECTION]
        
        query = {"symbol": mongo_symbol, "timeframe": req.timeframe}
        cursor = candles_coll.find(query).sort("timestamp", -1).limit(req.limit)
        docs = list(cursor)
        
        if not docs:
            raise HTTPException(
                status_code=400,
                detail=f"No data found in database for symbol={mongo_symbol}, timeframe={req.timeframe}"
            )
        
        docs.reverse()
        df = _candles_from_mongo_to_df(docs)
        df = add_full_indicators(df, timeframe=req.timeframe)
        # Drop rows where all indicator columns are NaN (keep OHLCV columns)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        indicator_cols = [c for c in df.columns if c not in ohlcv_cols]
        if indicator_cols:
            df = df.dropna(how="all", subset=indicator_cols)

        patterns, timestamps = extract_patterns(df, pattern_len=20)
        if len(patterns) == 0:
            return {"error": "No patterns extracted", "candle_count": len(df)}

        dim = patterns.shape[1]
        searcher = PatternSearch(dim=dim)
        searcher.build(patterns=patterns, timestamps=timestamps)

        from pathlib import Path
        path = Path("data/patterns") / f"{req.symbol.replace('/', '').replace(':', '')}_{req.timeframe}"
        searcher.save(str(path))

        return {
            "pattern_count": len(patterns),
            "dim": dim,
            "saved_to": str(path),
            "candle_count": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast")
def forecast(req: ForecastRequest):
    """RAG forecast via Chroma + LLM (stub if ChromaRAG unavailable)."""
    try:
        from app.rag import ChromaRAG
        rag = ChromaRAG()
        result = rag.forecast(query=req.query, n_context=req.n_context)
        return {"forecast": result}
    except ImportError:
        return {"forecast": f"[ChromaRAG not available] Query: {req.query}. Add chroma_rag.py to enable RAG forecasting."}
    except Exception as e:
        return {"forecast": f"[Forecast error] {str(e)}"}


@router.post("/backtest")
def backtest(req: BacktestRequest):
    """
    Run backtest with trend-based signals.
    
    **Data Source**: Fetches data from MongoDB database.
    """
    try:
        # Fetch data from MongoDB
        from pymongo import MongoClient
        from app.services.llm1_service import MONGODB_URI, DB_NAME, CANDLES_COLLECTION, _candles_from_mongo_to_df
        
        # Convert symbol format: BTC/USDT:USDT -> BTCUSDT
        mongo_symbol = req.symbol.replace("/", "").replace(":USDT", "")
        
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        candles_coll = db[CANDLES_COLLECTION]
        
        # Query MongoDB for candles
        query = {"symbol": mongo_symbol, "timeframe": req.timeframe}
        cursor = candles_coll.find(query).sort("timestamp", -1).limit(req.limit)
        docs = list(cursor)
        
        if not docs:
            raise HTTPException(
                status_code=400,
                detail=f"No data found in database for symbol={mongo_symbol}, timeframe={req.timeframe}. "
                       f"Make sure data is loaded using scripts/fetch_binance_to_mongo.py"
            )
        
        # Convert to DataFrame (reverse to get chronological order)
        docs.reverse()
        df = _candles_from_mongo_to_df(docs)
        df = add_full_indicators(df, timeframe=req.timeframe)
        # Drop rows where all indicator columns are NaN (keep OHLCV columns)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        indicator_cols = [c for c in df.columns if c not in ohlcv_cols]
        if indicator_cols:
            df = df.dropna(how="all", subset=indicator_cols)

        if len(df) < 52:
            raise HTTPException(status_code=400, detail="Insufficient data for backtest")

        cfg = get_config(req.timeframe)
        is_5m = "5" in req.timeframe and "15" not in req.timeframe
        ema_fast_col = "ema_9" if is_5m else "ema_12"
        rsi_level = 52.5 if "15" in req.timeframe or "1h" in req.timeframe else 50
        ema_medium_col = "ema_26" if "1h" in req.timeframe else "ema_21"

        signals_list = []
        for i in range(52, len(df)):
            window = df.iloc[: i + 1]
            tr = determine_trend(
                window,
                threshold=cfg["trend_threshold"],
                ema_fast_col=ema_fast_col,
                rsi_bullish_level=rsi_level,
                ema_medium_col=ema_medium_col,
            )
            if tr["trend"] == "Uptrend":
                signals_list.append(1)
            elif tr["trend"] == "Downtrend":
                signals_list.append(-1)
            else:
                signals_list.append(0)

        import pandas as pd
        signals = pd.Series(signals_list, index=df.index[52:])

        engine = BacktestEngine.VECTORBT if req.engine == "vectorbt" else BacktestEngine.BACKTESTING_PY
        # Use default 1:3 ratio (1% stop loss, 3% take profit) if not specified
        stop_loss = req.stop_loss if req.stop_loss is not None else 0.01
        take_profit = req.take_profit if req.take_profit is not None else 0.03
        result = run_backtest(df, signals, engine=engine, stop_loss=stop_loss, take_profit=take_profit)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest/strategy")
def backtest_strategy(req: BacktestStrategyRequest):
    """
    Run backtest with a custom strategy from a Python file.
    
    **Data Source**: By default (use_database=True), fetches data from MongoDB database.
    Set use_database=False to fetch from Binance API instead.
    
    The strategy file should define a class that inherits from BaseStrategy
    and implements the generate_signals method.
    
    Example strategy file:
        from app.strategies.base import BaseStrategy
        import pandas as pd
        
        class MyStrategy(BaseStrategy):
            def generate_signals(self, df: pd.DataFrame) -> pd.Series:
                # Your signal logic here
                # Return Series with 1=long, -1=short, 0=flat
                signals = pd.Series(0, index=df.index)
                # ... your logic ...
                return signals
    """
    try:
        from pathlib import Path
        import pandas as pd
        
        # Load strategy class from file
        strategy_path = Path(req.strategy_file)
        if not strategy_path.is_absolute():
            # If relative, assume it's relative to backend directory
            backend_dir = Path(__file__).parent.parent.parent
            strategy_path = backend_dir / strategy_path
        
        strategy_class = load_strategy_from_file(str(strategy_path))
        
        # Initialize strategy with optional parameters
        strategy_params = req.strategy_params or {}
        strategy = strategy_class(**strategy_params)
        
        # Fetch data from MongoDB (default) or API (if use_database=False)
        if req.use_database:
            # Fetch from MongoDB
            from pymongo import MongoClient
            from app.services.llm1_service import MONGODB_URI, DB_NAME, CANDLES_COLLECTION, _candles_from_mongo_to_df
            
            # Convert symbol format: BTC/USDT:USDT -> BTCUSDT
            mongo_symbol = req.symbol.replace("/", "").replace(":USDT", "")
            
            client = MongoClient(MONGODB_URI)
            db = client[DB_NAME]
            candles_coll = db[CANDLES_COLLECTION]
            
            # Query MongoDB for candles (get latest N candles, then reverse for chronological order)
            query = {"symbol": mongo_symbol, "timeframe": req.timeframe}
            cursor = candles_coll.find(query).sort("timestamp", -1).limit(req.limit)
            docs = list(cursor)
            
            if not docs:
                raise HTTPException(
                    status_code=400, 
                    detail=f"No data found in database for symbol={mongo_symbol}, timeframe={req.timeframe}. "
                           f"Make sure data is loaded using scripts/fetch_binance_to_mongo.py"
                )
            
            # Convert MongoDB documents to DataFrame
            # Reverse to get chronological order (oldest first)
            docs.reverse()
            df = _candles_from_mongo_to_df(docs)
            
            if df is None or len(df) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to convert MongoDB data to DataFrame for {mongo_symbol} {req.timeframe}"
                )
            
            logging.info(f"Loaded {len(df)} candles from MongoDB for {mongo_symbol} {req.timeframe}")
        else:
            # Fallback to API fetch (original behavior)
            fetcher = DataFetcher(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET"),
            )
            df = fetcher.fetch_ohlcv(symbol=req.symbol, timeframe=req.timeframe, limit=req.limit)
            logging.info(f"Fetched {len(df)} candles from Binance API for {req.symbol} {req.timeframe}")
        
        # Validate data was fetched
        if df is None or len(df) == 0:
            raise HTTPException(status_code=400, detail=f"No data fetched for {req.symbol} {req.timeframe} with limit {req.limit}")
        
        # Log actual data received for debugging
        logging.info(f"Fetched {len(df)} candles (requested {req.limit}) for {req.symbol} {req.timeframe}")
        
        # Validate DataFrame before adding indicators
        is_valid, error_msg, diagnostics, df_cleaned = validate_dataframe_for_backtest(df, min_rows=20)
        if not is_valid:
            logging.warning(f"Data validation failed: {error_msg}. Diagnostics: {diagnostics}")
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation failed: {error_msg}. Issues: {diagnostics.get('issues', [])}"
            )
        
        # Use cleaned DataFrame from validation
        df = df_cleaned
        
        logging.info(f"After initial validation: {len(df)} candles remaining. Issues found: {len(diagnostics.get('issues', []))}")
        
        # Add indicators
        df = add_full_indicators(df, timeframe=req.timeframe)
        
        # Drop rows where all indicator columns are NaN (keep OHLCV columns)
        # But don't drop rows that have valid OHLCV data - strategies can handle NaN indicators
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        indicator_cols = [c for c in df.columns if c not in ohlcv_cols]
        if indicator_cols:
            # Only drop rows where ALL indicators are NaN AND we have indicator columns
            # This preserves rows with valid OHLCV even if indicators are NaN
            df = df.dropna(how="all", subset=indicator_cols)
        
        # Final validation after indicator addition
        is_valid, error_msg, final_diagnostics, df_final = validate_dataframe_for_backtest(df, min_rows=20)
        if not is_valid:
            logging.warning(f"Final data validation failed: {error_msg}. Diagnostics: {final_diagnostics}")
            raise HTTPException(
                status_code=400, 
                detail=f"Final data validation failed: {error_msg}. Issues: {final_diagnostics.get('issues', [])}"
            )
        
        # Use final cleaned DataFrame
        df = df_final
        
        logging.info(f"Final validation passed: {len(df)} candles ready for backtest")
        
        # Generate signals using strategy
        try:
            signals = strategy.generate_signals(df)
        except Exception as e:
            logging.error(f"Error generating signals: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Strategy signal generation failed: {str(e)}"
            )
        
        # Validate signals
        if not isinstance(signals, pd.Series):
            raise HTTPException(
                status_code=500,
                detail=f"Strategy.generate_signals() must return a pandas Series, got {type(signals)}"
            )
        
        if len(signals) == 0:
            raise HTTPException(
                status_code=400,
                detail="Strategy generated empty signals Series"
            )
        
        # Check signal values are valid (should be 1, -1, or 0)
        invalid_signals = ~signals.isin([-1, 0, 1])
        if invalid_signals.any():
            invalid_count = invalid_signals.sum()
            invalid_values = signals[invalid_signals].unique()[:5].tolist()
            logging.warning(f"Found {invalid_count} invalid signal values: {invalid_values}")
            # Replace invalid values with 0
            signals = signals.where(signals.isin([-1, 0, 1]), 0)
        
        # Ensure signals are aligned with dataframe index
        try:
            signals = signals.reindex(df.index).fillna(0)
        except Exception as e:
            logging.error(f"Error aligning signals with dataframe: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to align signals with dataframe index: {str(e)}"
            )
        
        # Validate signal alignment
        if len(signals) != len(df):
            raise HTTPException(
                status_code=500,
                detail=f"Signal length ({len(signals)}) doesn't match dataframe length ({len(df)})"
            )
        
        # Validate that signals contain at least some non-zero values
        non_zero_signals = (signals != 0).sum()
        if non_zero_signals == 0:
            # Return a fallback result indicating no trades
            return {
                "engine": req.engine,
                "total_return": 0.0,
                "total_profit": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "message": "No trading signals generated - strategy returned all zeros",
                "strategy_name": strategy.get_name(),
                "strategy_file": str(strategy_path),
            }
        
        # Run backtest with stop loss and take profit
        engine = BacktestEngine.VECTORBT if req.engine == "vectorbt" else BacktestEngine.BACKTESTING_PY
        stop_loss = req.stop_loss if req.stop_loss is not None else 0.01  # Default 1%
        take_profit = req.take_profit if req.take_profit is not None else 0.03  # Default 3% (1:3 ratio)
        
        # Log the values being used for debugging
        logging.info(f"Running backtest with stop_loss={stop_loss} ({stop_loss*100:.2f}%), take_profit={take_profit} ({take_profit*100:.2f}%)")
        
        result = run_backtest(
            df, 
            signals, 
            engine=engine, 
            initial_capital=req.initial_capital,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Check for errors in result
        if "error" in result:
            # If there's an error, return it with strategy info
            result["strategy_name"] = strategy.get_name()
            result["strategy_file"] = str(strategy_path)
            return result
        
        # Validate result - if all metrics are zero and we had signals, something might be wrong
        if result.get("total_trades", 0) == 0 and non_zero_signals > 0:
            # Check if this is a valid case (signals but no trades due to entry/exit logic)
            # Add a warning but don't fail
            result["warning"] = f"Strategy generated {non_zero_signals} non-zero signals but no trades were executed. Check entry/exit logic."
        
        # Add strategy info to result
        result["strategy_name"] = strategy.get_name()
        result["strategy_file"] = str(strategy_path)
        
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")


STRATEGIES_COLLECTION = "strategies"


class CreateStrategyRequest(BaseModel):
    name: str
    strategy_file: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class UpdateStrategyRequest(BaseModel):
    name: Optional[str] = None
    strategy_file: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


@router.post("/strategies")
def create_strategy(req: CreateStrategyRequest):
    """Create a new strategy with name only."""
    from datetime import datetime, timezone
    from pymongo import MongoClient
    from app.services.llm1_service import MONGODB_URI, DB_NAME
    
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    
    # Check if name already exists
    existing = db[STRATEGIES_COLLECTION].find_one({"name": req.name.strip()})
    if existing:
        raise HTTPException(status_code=400, detail=f"Strategy with name '{req.name}' already exists")
    
    doc = {
        "name": req.name.strip(),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    
    # Add optional fields
    if req.strategy_file:
        doc["strategy_file"] = req.strategy_file
    if req.params:
        doc["params"] = req.params
    if req.description:
        doc["description"] = req.description
    
    r = db[STRATEGIES_COLLECTION].insert_one(doc)
    return {"success": True, "id": str(r.inserted_id), "name": doc["name"]}


@router.get("/strategies")
def list_strategies(
    limit: int = Query(100, ge=1, le=500),
):
    """List all saved strategies."""
    from pymongo import MongoClient
    from app.services.llm1_service import MONGODB_URI, DB_NAME
    
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    cursor = db[STRATEGIES_COLLECTION].find(
        {},
        projection={"name": 1, "created_at": 1, "updated_at": 1},
    ).sort("created_at", -1).limit(limit)
    docs = list(cursor)
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return {"strategies": docs}


@router.get("/strategies/{strategy_id}")
def get_strategy(strategy_id: str):
    """Get a strategy by ID."""
    from pymongo import MongoClient
    from bson import ObjectId
    from app.services.llm1_service import MONGODB_URI, DB_NAME
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        doc = db[STRATEGIES_COLLECTION].find_one({"_id": ObjectId(strategy_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Strategy not found")
        doc["id"] = str(doc.pop("_id"))
        return doc
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/strategies/{strategy_id}")
def update_strategy(strategy_id: str, req: UpdateStrategyRequest):
    """Update an existing strategy by ID."""
    from datetime import datetime, timezone
    from pymongo import MongoClient
    from bson import ObjectId
    from app.services.llm1_service import MONGODB_URI, DB_NAME
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        
        # Check if strategy exists
        existing = db[STRATEGIES_COLLECTION].find_one({"_id": ObjectId(strategy_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Build update document
        update_doc = {"updated_at": datetime.now(timezone.utc)}
        
        if req.name is not None:
            # Check if new name conflicts with another strategy
            name_conflict = db[STRATEGIES_COLLECTION].find_one({
                "name": req.name.strip(),
                "_id": {"$ne": ObjectId(strategy_id)}
            })
            if name_conflict:
                raise HTTPException(status_code=400, detail=f"Strategy with name '{req.name}' already exists")
            update_doc["name"] = req.name.strip()
        
        if req.strategy_file is not None:
            update_doc["strategy_file"] = req.strategy_file
        
        if req.params is not None:
            update_doc["params"] = req.params
        
        if req.description is not None:
            update_doc["description"] = req.description
        
        # Update strategy
        result = db[STRATEGIES_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {"success": True, "id": strategy_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/strategies/{strategy_id}/file")
def download_strategy_file(strategy_id: str):
    """Download the strategy file for a given strategy."""
    from pymongo import MongoClient
    from bson import ObjectId
    from pathlib import Path
    from app.services.llm1_service import MONGODB_URI, DB_NAME
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        doc = db[STRATEGIES_COLLECTION].find_one({"_id": ObjectId(strategy_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy_file = doc.get("strategy_file")
        if not strategy_file:
            raise HTTPException(status_code=404, detail="Strategy file not found")
        
        # Resolve file path
        backend_dir = Path(__file__).parent.parent.parent
        file_path = backend_dir / strategy_file
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Strategy file does not exist on disk")
        
        # Read file content
        file_content = file_path.read_text(encoding="utf-8")
        
        # Return file as download
        return Response(
            content=file_content,
            media_type="text/x-python",
            headers={
                "Content-Disposition": f'attachment; filename="{doc.get("name", "strategy")}.py"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/strategies/{strategy_id}/file")
async def upload_strategy_file(strategy_id: str, file: UploadFile = File(...)):
    """Upload a new strategy file for a given strategy. Automatically reviews and refactors code to convert static values to dynamic parameters."""
    from datetime import datetime, timezone
    from pymongo import MongoClient
    from bson import ObjectId
    from pathlib import Path
    from app.services.llm1_service import MONGODB_URI, DB_NAME
    from app.services.ai_chat_service import get_chat_service
    
    try:
        # Validate file extension
        if not file.filename or not file.filename.endswith('.py'):
            raise HTTPException(status_code=400, detail="File must be a Python file (.py)")
        
        # Read file content
        content = await file.read()
        file_content = content.decode('utf-8')
        
        # Validate that it's valid Python code (basic check)
        try:
            compile(file_content, file.filename, 'exec')
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Invalid Python syntax: {str(e)}")
        
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        
        # Get strategy
        doc = db[STRATEGIES_COLLECTION].find_one({"_id": ObjectId(strategy_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy_name = doc.get("name", f"strategy_{strategy_id}")
        
        # Review and refactor code to convert static values to dynamic parameters
        chat_service = get_chat_service()
        refactored_code = None
        extracted_params = []
        
        try:
            # Collect results from the async iterator
            async for chunk in chat_service.review_and_refactor_code(
                code_content=file_content,
                strategy_id=None  # Don't update yet, we'll do it manually
            ):
                if chunk.get("type") == "complete":
                    refactored_code = chunk.get("code")
                    extracted_params = chunk.get("params", [])
                    break
                elif chunk.get("type") == "error":
                    # If review fails, use original code but log warning
                    import logging
                    logging.warning(f"Code review failed: {chunk.get('message')}. Using original code.")
                    refactored_code = file_content
                    break
            
            # If no refactored code was received, use original
            if not refactored_code:
                refactored_code = file_content
        except Exception as review_error:
            # If review fails completely, use original code
            import logging
            logging.warning(f"Code review error: {review_error}. Using original code.")
            refactored_code = file_content
        
        # Determine file path - use consistent location: app/strategies/{strategy_name}/strategy.py
        backend_dir = Path(__file__).parent.parent.parent
        strategy_dir = backend_dir / "app" / "strategies" / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Save refactored file (overwrite if exists)
        strategy_file_path = strategy_dir / "strategy.py"
        strategy_file_path.write_text(refactored_code, encoding="utf-8")
        
        # Update strategy in database with relative path and extracted parameters
        relative_path = f"app/strategies/{strategy_name}/strategy.py"
        
        # Convert extracted params to dict format
        params_dict = {}
        for param in extracted_params:
            params_dict[param["name"]] = param.get("default", 0)
        
        update_doc = {
            "strategy_file": relative_path,
            "updated_at": datetime.now(timezone.utc),
        }
        
        # Only update params if we extracted them
        if params_dict:
            update_doc["params"] = params_dict
        
        db[STRATEGIES_COLLECTION].update_one(
            {"_id": ObjectId(strategy_id)},
            {"$set": update_doc}
        )
        
        return {
            "success": True,
            "id": strategy_id,
            "strategy_file": relative_path,
            "params": params_dict,
            "message": "Strategy file uploaded and refactored successfully. Static values have been converted to dynamic parameters."
        }
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/strategies/{strategy_id}")
def delete_strategy(strategy_id: str):
    """Delete a strategy by ID and its associated folder if it exists."""
    from pymongo import MongoClient
    from bson import ObjectId
    from pathlib import Path
    import shutil
    from app.services.llm1_service import MONGODB_URI, DB_NAME
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        
        # Get strategy before deleting to access file path
        doc = db[STRATEGIES_COLLECTION].find_one({"_id": ObjectId(strategy_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Delete associated folder if it exists
        strategy_file = doc.get("strategy_file")
        if strategy_file:
            try:
                backend_dir = Path(__file__).parent.parent.parent
                file_path = backend_dir / strategy_file
                
                # Delete the file if it exists
                if file_path.exists() and file_path.is_file():
                    file_path.unlink()
                
                # Delete the entire strategy folder (parent directory of strategy.py)
                # Strategy files are stored as: app/strategies/{strategy_name}/strategy.py
                if "strategies" in strategy_file:
                    strategy_dir = file_path.parent
                    if strategy_dir.exists() and strategy_dir.is_dir():
                        try:
                            # Remove the entire folder and all its contents
                            shutil.rmtree(strategy_dir)
                        except OSError as dir_error:
                            # Log error but don't fail the deletion if folder removal fails
                            import logging
                            logging.warning(f"Failed to delete strategy folder {strategy_dir}: {dir_error}")
            except Exception as file_error:
                # Log error but don't fail the deletion if file removal fails
                import logging
                logging.warning(f"Failed to delete strategy file/folder {strategy_file}: {file_error}")
        
        # Delete strategy from database
        result = db[STRATEGIES_COLLECTION].delete_one({"_id": ObjectId(strategy_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {"success": True, "id": strategy_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
