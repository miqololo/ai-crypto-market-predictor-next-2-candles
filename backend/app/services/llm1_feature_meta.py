"""
Canonical LLM1 feature metadata. Only these features are built and saved.
Train/fit uses this list when metadata is available.
"""
from typing import Dict, List

FEATURE_META: Dict[str, Dict[str, str]] = {
    "open_norm": {"category": "Price", "description": "Open price normalized [0,1] — базовый уровень"},
    "high_norm": {"category": "Price", "description": "High normalized [0,1]"},
    "low_norm": {"category": "Price", "description": "Low normalized [0,1]"},
    "close_norm": {"category": "Price", "description": "Close normalized [0,1] — главная цель предсказания"},
    "vwap_norm": {"category": "Price", "description": "VWAP normalized [0,1] — институциональный уровень"},

    "rsi": {"category": "Momentum", "description": "RSI: 1 if >80, -1 if <20, else 0 — overbought/oversold"},
    "macd_hist_norm": {"category": "Momentum", "description": "MACD histogram normalized — ускорение"},
    "macd_norm": {"category": "Momentum", "description": "MACD line normalized"},
    "stoch_k": {"category": "Momentum", "description": "Stochastic %K [0,1]"},
    "cci_norm": {"category": "Momentum", "description": "CCI normalized [-1,1]"},
    "ao_norm": {"category": "Momentum", "description": "Awesome Oscillator normalized"},

    "atr_norm": {"category": "Volatility", "description": "ATR normalized — ключевой для волатильности"},
    "percent_b": {"category": "Volatility", "description": "Bollinger %B [0,1] — позиция в полосах"},

    "supertrend_dir": {"category": "Trend", "description": "Supertrend direction (-1/0/+1)"},
    "supertrend_bullish": {"category": "Trend", "description": "Supertrend bullish flag"},
    "ema_cross_bull": {"category": "Crossings", "description": "Fast EMA crossed above slow recently"},
    "macd_cross_bull": {"category": "Crossings", "description": "MACD cross bullish recently"},
    "supertrend_flip_up": {"category": "Crossings", "description": "Supertrend flip to bull recently"},

    "long_short_ratio": {"category": "Sentiment", "description": "Long/short ratio"},
    "taker_ratio": {"category": "Sentiment", "description": "Taker buy/sell ratio"},

    "hour_sin": {"category": "Time", "description": "sin(2π·hour/24) — cyclical encoding часа дня"},
    "hour_cos": {"category": "Time", "description": "cos(2π·hour/24) — cyclical encoding часа дня"},
    "dayofweek_sin": {"category": "Time", "description": "sin(2π·dayofweek/7) — cyclical encoding дня недели (Mon=0…Sun=6)"},
    "dayofweek_cos": {"category": "Time", "description": "cos(2π·dayofweek/7) — cyclical encoding дня недели"},

    "strategy_multi_confirmation": {"category": "Strategy", "description": "Ensemble score [0,1] — тренд + моментум + волатильность"},
    "strategy_divergence_reversal": {"category": "Strategy", "description": "Divergence reversal score [0,1] — RSI/MACD vs price divergence"},
    "strategy_sentiment_boosted_trend": {"category": "Strategy", "description": "Trend + sentiment boost [0,1]"},
    "strategy_risk_adjusted_momentum": {"category": "Strategy", "description": "Risk-adjusted momentum score [0,1] — моментум × ATR"},
    "strategy_supertrend_strength": {"category": "Strategy", "description": "Supertrend + strength score [0,1]"},
}

# Ordered list of feature names for consistent build/train
CANONICAL_FEATURES: List[str] = list(FEATURE_META.keys())


def get_feature_meta(name: str) -> Dict[str, str]:
    """Return category and description for a feature. Unknown features get Other."""
    return FEATURE_META.get(name, {"category": "Other", "description": "Custom feature"})
