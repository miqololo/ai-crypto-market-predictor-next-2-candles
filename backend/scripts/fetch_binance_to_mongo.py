#!/usr/bin/env python3
"""
Fetch Binance Futures OHLCV (5m) and save to local MongoDB.
Continues from the latest candle in DB (or from DEFAULT_START_UTC if empty) until now.
Uses CCXT for OHLCV and optionally merges Binance metadata (basis, long_short_ratio,
taker_ratio, funding_rate, etc.) via build_candle_binance_lookup endpoints.

Usage:
    python scripts/fetch_binance_to_mongo.py

Requires: MONGODB_URI in .env (default: mongodb://localhost:27017)
"""
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import pandas as pd
from pymongo import MongoClient, UpdateOne

from app.data.fetcher import DataFetcher
from app.data.binance_api import fetch_all_binance_histories


DEFAULT_START_UTC = "2025-09-13T12:15:00.000+00:00"  # user's latest candle; we fetch from next candle when DB empty
TIMEFRAMES = ["5m"]
CHUNK_SIZE = 500
BATCH_INSERT_SIZE = 2000
REQUEST_DELAY = 0.1
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "traider"
COLLECTION_NAME = "candles"
SYMBOL = "BTC/USDT:USDT"


def _to_native(val):
    """Convert numpy/pandas types to native Python for MongoDB."""
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (pd.Timestamp, datetime)):
        return val
    if isinstance(val, float) and (val != val):  # NaN
        return None
    return val


def _serialize_for_mongo(obj):
    """Recursively serialize for MongoDB (lists of dicts, timestamps, etc.)."""
    if isinstance(obj, list):
        return [_serialize_for_mongo(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize_for_mongo(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj
    if hasattr(obj, "item"):
        v = obj.item()
        return None if isinstance(v, float) and (v != v) else v
    if isinstance(obj, float) and (obj != obj):
        return None
    return obj


def _record_ts_ms(record: dict, ts_key: str = "timestamp") -> int:
    """Get timestamp in ms from API record (timestamp or fundingTime)."""
    ts = record.get(ts_key, record.get("fundingTime", 0))
    if ts is None:
        return 0
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp() * 1000)
    return int(ts)


def _find_record_for_candle(
    records: list,
    candle_ts: pd.Timestamp,
    candle_timeframe: str,
    ts_key: str = "timestamp",
) -> Optional[dict]:
    """
    Find the single API record that matches the candle's timestamp/timeframe.
    Returns the record with timestamp closest to candle, or None. The returned
    object has timestamp and timeframe set to match the candle.
    """
    if not records:
        return None
    candle_ts_ms = int(candle_ts.timestamp() * 1000)
    period_ms = {"5m": 300000, "15m": 900000, "1h": 3600000}.get(candle_timeframe, 3600000)
    candle_end_ms = candle_ts_ms + period_ms
    best = None
    best_dist = float("inf")
    for r in records:
        t = _record_ts_ms(r, ts_key)
        if not t or not (candle_ts_ms <= t < candle_end_ms):
            continue
        dist = abs(t - candle_ts_ms)
        if dist < best_dist:
            best_dist = dist
            best = dict(r)
    if best is None:
        return None
    best["timestamp"] = candle_ts.to_pydatetime() if hasattr(candle_ts, "to_pydatetime") else candle_ts
    best["timeframe"] = candle_timeframe
    return best


def get_latest_candle_ts(collection, symbol: str, timeframe: str) -> Optional[int]:
    """Return latest candle timestamp in ms, or None if no candles."""
    sym = symbol.replace("/", "").replace(":USDT", "")
    doc = collection.find_one(
        {"symbol": sym, "timeframe": timeframe},
        sort=[("timestamp", -1)],
        projection={"timestamp": 1},
    )
    if not doc or not doc.get("timestamp"):
        return None
    ts = doc["timestamp"]
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp() * 1000)
    return int(ts) if isinstance(ts, (int, float)) else None


def fetch_ohlcv_chunks(
    fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    start_ts: int,
    end_ts: int,
    chunk_size: int = 500,
    include_binance_histories: bool = True,
):
    """
    Yield chunks of OHLCV as list of dicts. Each chunk is fetched from Binance.
    Fetches from start_ts (inclusive) to end_ts (exclusive).
    When include_binance_histories=True, fetches from all Binance APIs and
    attaches one object per API per candle (timestamp and timeframe match the candle).
    """
    tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
    period_ms = tf_min * 60 * 1000
    total_candles = max(0, (end_ts - start_ts) // period_ms)

    since = start_ts
    fetched = 0

    while since < end_ts and fetched < total_candles:
        limit = min(chunk_size, total_candles - fetched)
        ohlcv = fetcher.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if not ohlcv:
            break

        records = [
            {
                "timestamp": pd.Timestamp(o[0], unit="ms"),
                "open": float(o[1]),
                "high": float(o[2]),
                "low": float(o[3]),
                "close": float(o[4]),
                "volume": float(o[5]),
            }
            for o in ohlcv
        ]

        if include_binance_histories and records:
            chunk_start_ms = int(records[0]["timestamp"].timestamp() * 1000)
            chunk_end_ms = int(records[-1]["timestamp"].timestamp() * 1000) + period_ms
            histories = fetch_all_binance_histories(symbol, timeframe, chunk_start_ms, chunk_end_ms)
            for r in records:
                r["basis"] = _find_record_for_candle(
                    histories["basis"], r["timestamp"], timeframe
                )
                r["global_long_short_account_ratio"] = _find_record_for_candle(
                    histories["global_long_short_account_ratio"], r["timestamp"], timeframe
                )
                r["top_long_short_account_ratio"] = _find_record_for_candle(
                    histories["top_long_short_account_ratio"], r["timestamp"], timeframe
                )
                r["top_long_short_position_ratio"] = _find_record_for_candle(
                    histories["top_long_short_position_ratio"], r["timestamp"], timeframe
                )
                r["taker_long_short_ratio"] = _find_record_for_candle(
                    histories["taker_long_short_ratio"], r["timestamp"], timeframe
                )
                r["open_interest_hist"] = _find_record_for_candle(
                    histories["open_interest_hist"], r["timestamp"], timeframe
                )
                r["funding_rate_history"] = _find_record_for_candle(
                    histories["funding_rate_history"], r["timestamp"], timeframe, ts_key="fundingTime"
                )

        yield records, fetched + len(records), total_candles
        fetched += len(records)

        if len(ohlcv) < limit:
            break
        since = int(ohlcv[-1][0]) + period_ms
        time.sleep(REQUEST_DELAY)


def doc_for_mongo(record: dict, symbol: str, timeframe: str) -> dict:
    """
    Convert record to MongoDB document. Saves OHLCV + one object per API
    (basis, global_long_short_account_ratio, etc.) with timestamp and timeframe
    matching the candle.
    """
    ts = record.get("timestamp")
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    doc = {
        "symbol": symbol.replace("/", "").replace(":USDT", ""),
        "timeframe": timeframe,
        "timestamp": ts,
    }
    for key, val in record.items():
        if key == "timestamp":
            continue
        if val is None:
            continue
        if isinstance(val, dict):
            doc[key] = _serialize_for_mongo(val)
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            doc[key] = _serialize_for_mongo(val)
        else:
            doc[key] = _to_native(val)
    return doc


def ensure_index(collection):
    """Create unique index on (symbol, timeframe, timestamp)."""
    collection.create_index(
        [("symbol", 1), ("timeframe", 1), ("timestamp", 1)],
        unique=True,
        name="symbol_timeframe_timestamp",
    )


def _ts():
    return datetime.utcnow().strftime("%H:%M:%S")


def run():
    start_time = time.time()
    print(f"[{_ts()}] Fetching data for {SYMBOL} — continue from latest candle until now")
    print(f"[{_ts()}] Timeframes: {TIMEFRAMES}")
    print(f"[{_ts()}] MongoDB: {MONGODB_URI} / {DB_NAME}.{COLLECTION_NAME}")
    print("-" * 60)

    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print(f"[{_ts()}] Ensuring index on (symbol, timeframe, timestamp)...")
    ensure_index(collection)

    fetcher = DataFetcher(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
    )

    end_ts = int(datetime.utcnow().timestamp() * 1000)
    tf_min_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    default_start_dt = datetime.fromisoformat(DEFAULT_START_UTC.replace("Z", "+00:00"))
    if default_start_dt.tzinfo is None:
        default_start_dt = default_start_dt.replace(tzinfo=timezone.utc)
    default_start_ts = int(default_start_dt.timestamp() * 1000)

    total_inserted = 0
    for tf_idx, timeframe in enumerate(TIMEFRAMES, 1):
        tf_min = tf_min_map.get(timeframe, 60)
        period_ms = tf_min * 60 * 1000

        latest_ms = get_latest_candle_ts(collection, SYMBOL, timeframe)
        if latest_ms is not None:
            start_ts = latest_ms + period_ms
            start_str = datetime.utcfromtimestamp(start_ts / 1000).strftime("%Y-%m-%d %H:%M UTC")
            print(f"\n[{_ts()}] [{tf_idx}/{len(TIMEFRAMES)}] {timeframe} — Continuing from {start_str} (after latest in DB)")
        else:
            start_ts = default_start_ts + period_ms
            start_str = datetime.utcfromtimestamp(start_ts / 1000).strftime("%Y-%m-%d %H:%M UTC")
            print(f"\n[{_ts()}] [{tf_idx}/{len(TIMEFRAMES)}] {timeframe} — Starting from {start_str} (no data in DB, using default)")

        if start_ts >= end_ts:
            print(f"[{_ts()}] [{timeframe}] Already up to date, skipping")
            continue

        tf_start = time.time()
        batch = []
        last_pct = -1
        chunk_num = 0

        for records, fetched, total in fetch_ohlcv_chunks(
            fetcher, SYMBOL, timeframe, start_ts, end_ts, CHUNK_SIZE, include_binance_histories=True
        ):
            chunk_num += 1
            for r in records:
                doc = doc_for_mongo(r, SYMBOL, timeframe)
                batch.append(doc)

            while len(batch) >= BATCH_INSERT_SIZE:
                to_insert = batch[:BATCH_INSERT_SIZE]
                batch = batch[BATCH_INSERT_SIZE:]
                ops = [
                    UpdateOne(
                        {"symbol": d["symbol"], "timeframe": d["timeframe"], "timestamp": d["timestamp"]},
                        {"$set": d},
                        upsert=True,
                    )
                    for d in to_insert
                ]
                result = collection.bulk_write(ops, ordered=False)
                total_inserted += result.upserted_count + result.modified_count

            pct = int(100 * fetched / total) if total else 0
            if pct != last_pct and (pct % 5 == 0 or pct >= 99 or chunk_num == 1):
                print(f"[{_ts()}]   chunk {chunk_num} | {fetched:,}/{total:,} ({pct}%) | upserted {total_inserted:,}")
                last_pct = pct

        if batch:
            ops = [
                UpdateOne(
                    {"symbol": d["symbol"], "timeframe": d["timeframe"], "timestamp": d["timestamp"]},
                    {"$set": d},
                    upsert=True,
                )
                for d in batch
            ]
            result = collection.bulk_write(ops, ordered=False)
            ups = result.upserted_count + result.modified_count
            total_inserted += ups
            print(f"[{_ts()}]   final batch inserted: {ups:,}")

        tf_elapsed = time.time() - tf_start
        count = collection.count_documents({"timeframe": timeframe})
        print(f"[{_ts()}] [{timeframe}] Done in {tf_elapsed:.1f}s — {count:,} candles in DB")

    elapsed = time.time() - start_time
    print(f"\n[{_ts()}] Completed in {elapsed:.1f}s. Total upserted: {total_inserted:,}")


if __name__ == "__main__":
    run()
