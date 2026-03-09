"""Technical indicators using pandas-ta (primary) and optional TA-Lib."""
import pandas as pd
import numpy as np
from typing import Optional

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


def add_indicators(
    df: pd.DataFrame,
    use_talib: bool = False,
) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV DataFrame.
    Uses pandas-ta by default; TA-Lib for speed when use_talib=True.
    """
    out = df.copy()
    h, l, c, v = out["high"], out["low"], out["close"], out["volume"]

    if use_talib and HAS_TALIB:
        out["rsi"] = talib.RSI(c, timeperiod=14)
        out["macd"], out["macd_signal"], out["macd_hist"] = talib.MACD(
            c, fastperiod=12, slowperiod=26, signalperiod=9
        )
        out["bb_upper"], out["bb_mid"], out["bb_lower"] = talib.BBANDS(
            c, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        out["atr"] = talib.ATR(h, l, c, timeperiod=14)
        out["ema_9"] = talib.EMA(c, timeperiod=9)
        out["ema_21"] = talib.EMA(c, timeperiod=21)
    elif HAS_PANDAS_TA:
        out["rsi"] = ta.rsi(c, length=14)
        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty and macd.shape[1] >= 3:
            out["macd"] = macd.iloc[:, 0]
            out["macd_signal"] = macd.iloc[:, 1]
            out["macd_hist"] = macd.iloc[:, 2]
        bb = ta.bbands(c, length=20, std=2)
        if bb is not None and not bb.empty and bb.shape[1] >= 3:
            out["bb_upper"] = bb.iloc[:, 0]
            out["bb_mid"] = bb.iloc[:, 1]
            out["bb_lower"] = bb.iloc[:, 2]
        else:
            out["bb_mid"] = c.rolling(20).mean()
            std = c.rolling(20).std().fillna(0)
            out["bb_upper"] = out["bb_mid"] + 2 * std
            out["bb_lower"] = out["bb_mid"] - 2 * std
        out["atr"] = ta.atr(h, l, c, length=14)
        out["ema_9"] = ta.ema(c, length=9)
        out["ema_21"] = ta.ema(c, length=21)
    else:
        # Fallback: minimal numpy-based indicators
        out["rsi"] = _rsi_np(c, 14)
        out["ema_9"] = c.ewm(span=9, adjust=False).mean()
        out["ema_21"] = c.ewm(span=21, adjust=False).mean()
        out["atr"] = _atr_np(h, l, c, 14)

    return out


def _rsi_np(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr_np(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()
