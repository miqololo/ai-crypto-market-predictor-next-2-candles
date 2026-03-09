#!/bin/bash
# Ready-to-run curl example for POST /api/ai/similarity
# Ensure API is running: python run.py

curl -X POST "http://localhost:8000/api/ai/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe": "1h",
    "symbol": "BTCUSDT",
    "datetime": "2025-02-19T12:00:00",
    "window_size": null,
    "metric": "cosine",
    "k": 10,
    "limit": 2000,
    "missing_strategy": "impute_zero",
    "use_open": false,
    "use_high": false,
    "use_low": false,
    "use_close": true,
    "use_volume": true,
    "use_rsi": true,
    "use_macd": false,
    "use_macd_signal": false,
    "use_macd_hist": true,
    "use_ema_9": false,
    "use_ema_12": false,
    "use_ema_21": false,
    "use_ema_26": false,
    "use_sma_20": false,
    "use_sma_50": false,
    "use_sma_200": false,
    "use_bb_upper": false,
    "use_bb_mid": false,
    "use_bb_lower": false,
    "use_stoch_k": false,
    "use_stoch_d": false,
    "use_cci": false,
    "use_atr": true,
    "use_psar": false,
    "use_supertrend": false,
    "use_supertrend_dir": false,
    "use_vwap": false,
    "use_ao": false,
    "use_tenkan": false,
    "use_kijun": false,
    "use_senkou_a": false,
    "use_senkou_b": false
  }'
