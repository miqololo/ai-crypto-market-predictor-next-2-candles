"""Timeframe-specific config for 5m, 15m, 1h analysis."""
from typing import Dict, Any

# 5m: sensitive, fast signals
CONFIG_5M: Dict[str, Any] = {
    "ema_fast": 9,
    "ema_medium": 21,
    "sma_short": 20,
    "sma_medium": 50,
    "rsi_period": 9,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "cci_period": 14,
    "atr_period": 14,
    "psar_step": 0.02,
    "psar_max": 0.2,
    "supertrend_period": 10,
    "supertrend_mult": 3,
    "ao_fast": 5,
    "ao_slow": 34,
    "ichimoku_tenkan": 9,
    "ichimoku_kijun": 26,
    "ichimoku_senkou": 52,
    "trend_threshold": 0.70,
    "fib_swing_period": 40,
    "norm_window": 20,
    "embedding_window": 20,
    "funding_delta_candles": [5, 10],
    "basis_delta_candles": 5,
    "ls_ratio_delta_candles": 5,
    "top_trader_delta_candles": 5,
    "liquidations_delta_candles": 3,
}

# 15m: 15m-optimized, smoother defaults
CONFIG_15M: Dict[str, Any] = {
    "ema_fast": 12,
    "ema_medium": 21,
    "sma_short": 20,
    "sma_medium": 50,
    "sma_long": 200,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "cci_period": 20,
    "atr_period": 14,
    "psar_step": 0.02,
    "psar_max": 0.18,
    "supertrend_period": 10,
    "supertrend_mult": 3,
    "ao_fast": 5,
    "ao_slow": 34,
    "ichimoku_tenkan": 9,
    "ichimoku_kijun": 26,
    "ichimoku_senkou": 52,
    "trend_threshold": 0.65,
    "fib_swing_period": 30,
    "norm_window": 40,
    "embedding_window": 35,
    "funding_delta_candles": [8, 12],
    "basis_delta_candles": 7,
    "ls_ratio_delta_candles": 6,
    "top_trader_delta_candles": 6,
    "liquidations_delta_candles": 3,
}

# 1h: larger candles, more conservative, 12/26 EMA combo
CONFIG_1H: Dict[str, Any] = {
    "ema_fast": 12,
    "ema_medium": 26,
    "sma_short": 20,
    "sma_medium": 50,
    "sma_long": 200,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "cci_period": 20,
    "atr_period": 14,
    "psar_step": 0.02,
    "psar_max": 0.18,
    "supertrend_period": 10,
    "supertrend_mult": 3,
    "ao_fast": 5,
    "ao_slow": 34,
    "ichimoku_tenkan": 9,
    "ichimoku_kijun": 26,
    "ichimoku_senkou": 52,
    "trend_threshold": 0.70,
    "fib_swing_period": 65,
    "norm_window": 60,
    "embedding_window": 55,
    "funding_delta_candles": [8, 15],
    "basis_delta_candles": 10,
    "ls_ratio_delta_candles": 8,
    "top_trader_delta_candles": 8,
    "liquidations_delta_candles": 6,
}

# RSI neutral band for 15m/1h (near 50 ±5)
RSI_NEUTRAL_LOW = 45
RSI_NEUTRAL_HIGH = 55


def get_config(timeframe: str) -> Dict[str, Any]:
    """Get config for timeframe. Default to 5m for unknown."""
    if not timeframe:
        return CONFIG_5M.copy()
    tf = timeframe.lower()
    if "1h" in tf or "60" in tf:
        return CONFIG_1H.copy()
    if "15" in tf:
        return CONFIG_15M.copy()
    return CONFIG_5M.copy()
