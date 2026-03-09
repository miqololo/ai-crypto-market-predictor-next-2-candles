"""OHLCV data fetching via CCXT (Binance Futures)."""
import time
import ccxt
import pandas as pd
import numpy as np
from typing import Optional, List, Callable
from datetime import datetime, timedelta


def add_candle_embeddings(df: pd.DataFrame, window: int = 20) -> List[List[float]]:
    """
    Add min-max normalized embedding vector to each candle for pattern matching.
    Each candle gets a 5-dim vector [o_norm, h_norm, l_norm, c_norm, v_norm] in 0-1.
    """
    o, h, l_, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    embeds = []
    for i in range(len(df)):
        start = max(0, i - window + 1)
        w_o, w_h, w_l, w_c, w_v = o.iloc[start : i + 1], h.iloc[start : i + 1], l_.iloc[start : i + 1], c.iloc[start : i + 1], v.iloc[start : i + 1]

        def _norm(s, val):
            mn, mx = s.min(), s.max()
            if mx - mn == 0:
                return 0.5
            return float((val - mn) / (mx - mn))

        embeds.append([
            _norm(w_o, o.iloc[i]),
            _norm(w_h, h.iloc[i]),
            _norm(w_l, l_.iloc[i]),
            _norm(w_c, c.iloc[i]),
            _norm(w_v, v.iloc[i]),
        ])
    return embeds


class DataFetcher:
    """Fetch crypto futures OHLCV from Binance via CCXT."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = False,
    ):
        config = {}
        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret
        self.exchange = ccxt.binanceusdm(config)
        self.exchange.options["defaultType"] = "future"
        if sandbox:
            self.exchange.set_sandbox_mode(True)

    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        limit: int = 1000,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV and return DataFrame. Handles limits > 1000 by chunking."""
        # Binance API max limit is 1000 per request, so chunk if needed
        max_chunk_size = 1000
        if limit <= max_chunk_size:
            # Single request
            if since is None:
                # Approximate: 5m=5, 15m=15, 1h=60, 4h=240, 1d=1440 min per candle
                tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
                mins_back = limit * tf_min
                since = int((datetime.utcnow() - timedelta(minutes=mins_back)).timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        else:
            # Chunk requests for limits > 1000
            tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
            period_ms = tf_min * 60 * 1000
            
            if since is None:
                mins_back = limit * tf_min
                since = int((datetime.utcnow() - timedelta(minutes=mins_back)).timestamp() * 1000)
            
            all_dfs = []
            current_since = since
            remaining = limit
            
            while remaining > 0:
                chunk_limit = min(max_chunk_size, remaining)
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_since, chunk_limit)
                
                if not ohlcv or len(ohlcv) == 0:
                    break
                
                df_chunk = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], unit="ms")
                df_chunk.set_index("timestamp", inplace=True)
                all_dfs.append(df_chunk)
                
                fetched = len(ohlcv)
                remaining -= fetched
                
                if fetched < chunk_limit:
                    # Got fewer than requested, likely reached available data
                    break
                
                # Move to next chunk: start from the last candle timestamp + 1 period
                current_since = int(ohlcv[-1][0]) + period_ms
                time.sleep(0.1)  # Rate limiting
            
            if not all_dfs:
                return pd.DataFrame()
            
            # Combine all chunks, remove duplicates, sort by timestamp
            result = pd.concat(all_dfs).drop_duplicates().sort_index()
            # Limit to requested number of candles (take most recent)
            if len(result) > limit:
                result = result.tail(limit)
            return result

    def fetch_ohlcv_range(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        days: int = 365,
        chunk_size: int = 500,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV for N days in chunks (Binance max 500/request). Returns oldest-first DataFrame."""
        tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
        total_candles = days * 24 * 60 // tf_min
        end_ts = int(datetime.utcnow().timestamp() * 1000)
        start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        all_dfs = []
        since = start_ts
        fetched = 0
        while since < end_ts and fetched < total_candles:
            limit = min(chunk_size, total_candles - fetched)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            all_dfs.append(df)
            fetched += len(ohlcv)
            if progress_callback:
                progress_callback(fetched, total_candles, timeframe)
            if len(ohlcv) < limit:
                break
            period_ms = tf_min * 60 * 1000
            since = int(ohlcv[-1][0]) + period_ms
            time.sleep(0.15)
        if not all_dfs:
            return pd.DataFrame()
        result = pd.concat(all_dfs).drop_duplicates().sort_index()
        return result

    def fetch_ohlcv_ending_at(
        self,
        end_datetime: datetime,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch limit candles ending at end_datetime (last 200 before/including that time)."""
        tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
        period_ms = tf_min * 60 * 1000
        end_ts = int(end_datetime.timestamp() * 1000)
        since = end_ts - limit * period_ms
        return self.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit, since=since)


def fetch_ohlcv(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "1h",
    limit: int = 1000,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function to fetch OHLCV."""
    fetcher = DataFetcher(api_key=api_key, api_secret=api_secret)
    return fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
