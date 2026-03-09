#!/usr/bin/env python3
"""
Ready-to-run example: call POST /api/ai/similarity with all possible properties.
Run with: python scripts/example_similarity_request.py
Ensure the API is running: python run.py (or uvicorn app.main:app)
"""
import json
import urllib.request

BASE_URL = "http://localhost:8000"

# Example 1: Single-point search with datetime and feature flags (query from candle at datetime)
EXAMPLE_WITH_DATETIME = {
    "timeframe": "1h",
    "symbol": "BTCUSDT",
    "datetime": "2025-02-19T12:00:00",
    "window_size": None,
    "metric": "cosine",
    "k": 10,
    "limit": 2000,
    "missing_strategy": "impute_zero",
    # All feature flags (true = include in similarity)
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

# Example 2: Window search (no datetime = uses latest candle)
EXAMPLE_WINDOW_SEARCH = {
    "timeframe": "15m",
    "symbol": "BTCUSDT",
    "datetime": None,
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
}

# Example 3: Minimal - all flags false, uses defaults
EXAMPLE_MINIMAL = {
    "timeframe": "1h",
    "symbol": "BTCUSDT",
}


def call_similarity(payload: dict) -> dict:
    """POST to /api/ai/similarity and return parsed JSON."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/ai/similarity",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


if __name__ == "__main__":
    import sys

    example = EXAMPLE_WITH_DATETIME
    if len(sys.argv) > 1:
        if sys.argv[1] == "window":
            example = EXAMPLE_WINDOW_SEARCH
        elif sys.argv[1] == "minimal":
            example = EXAMPLE_MINIMAL
        elif sys.argv[1] == "latest":
            example = {"timeframe": "1h", "symbol": "BTCUSDT"}  # no datetime = latest candle

    print("Calling POST /api/ai/similarity with payload:")
    print(json.dumps(example, indent=2))
    print("\n--- Response ---")
    try:
        result = call_similarity(example)
        print(json.dumps(result, indent=2))
    except urllib.error.URLError as e:
        print(f"Error: {e}")
        print("Ensure the API is running: python run.py")
        sys.exit(1)
