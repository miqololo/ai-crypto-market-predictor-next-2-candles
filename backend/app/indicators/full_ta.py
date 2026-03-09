"""Full technical indicator suite for 5m and 15m candle analysis."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

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


def _get_tf_config(timeframe: str) -> Dict[str, Any]:
    """Import config lazily to avoid circular imports."""
    from app.analysis.timeframe_config import get_config
    return get_config(timeframe)


def add_full_indicators(
    df: pd.DataFrame,
    use_talib: bool = False,
    timeframe: str = "5m",
) -> pd.DataFrame:
    """
    Add all technical indicators. Uses timeframe-specific defaults:
    - 5m: RSI 9, CCI 14, EMA 9/21, PSAR max 0.2
    - 15m: RSI 14, CCI 20, EMA 12/21, PSAR max 0.18, SMA 200
    - 1h: RSI 14, CCI 20, EMA 12/26, PSAR max 0.18, SMA 200
    """
    cfg = _get_tf_config(timeframe)
    out = df.copy()
    h, l, c, o, v = out["high"], out["low"], out["close"], out["open"], out["volume"]

    def _ema(s, period):
        return s.ewm(span=period, adjust=False).mean()

    def _sma(s, period):
        return s.rolling(period).mean()

    def _rsi(s, period=9):
        delta = s.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _atr(period=14):
        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    # EMA: fast + medium (9 or 12, 21 or 26)
    out["ema_9"] = _ema(c, 9)
    out["ema_12"] = _ema(c, 12)
    out["ema_21"] = _ema(c, min(21, cfg["ema_medium"]))
    if cfg["ema_medium"] == 26:
        out["ema_26"] = _ema(c, 26)

    # SMA: 20, 50, 200 (15m only)
    out["sma_20"] = _sma(c, cfg["sma_short"])
    out["sma_50"] = _sma(c, cfg["sma_medium"])
    if cfg.get("sma_long"):
        out["sma_200"] = _sma(c, cfg["sma_long"])

    # RSI: 9 (5m) or 14 (15m)
    out["rsi"] = _rsi(c, cfg["rsi_period"])

    # MACD: 12, 26, 9
    ema12 = _ema(c, cfg["macd_fast"])
    ema26 = _ema(c, cfg["macd_slow"])
    out["macd"] = ema12 - ema26
    out["macd_signal"] = _ema(out["macd"], cfg["macd_signal"])
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Bollinger Bands: 20, 2 std
    bb_period = cfg["bb_period"]
    bb_std_val = cfg["bb_std"]
    out["bb_mid"] = _sma(c, bb_period)
    bb_std = c.rolling(bb_period).std()
    out["bb_upper"] = out["bb_mid"] + bb_std_val * bb_std
    out["bb_lower"] = out["bb_mid"] - bb_std_val * bb_std

    # Stochastic: %K 14, %D 3, Smooth 3
    sk, sd, ss = cfg["stoch_k"], cfg["stoch_d"], cfg["stoch_smooth"]
    if HAS_PANDAS_TA:
        stoch = ta.stoch(h, l, c, k=sk, d=sd, smooth_k=ss)
        if stoch is not None and not stoch.empty:
            cols = stoch.columns.tolist()
            out["stoch_k"] = stoch[cols[0]]
            out["stoch_d"] = stoch[cols[1]] if len(cols) > 1 else stoch[cols[0]].rolling(sd).mean()
    else:
        low_k = l.rolling(sk).min()
        high_k = h.rolling(sk).max()
        out["stoch_k"] = 100 * (c - low_k) / (high_k - low_k + 1e-10)
        out["stoch_d"] = out["stoch_k"].rolling(sd).mean()

    # CCI: period 14 (5m) or 20 (15m)
    cci_period = cfg["cci_period"]
    tp = (h + l + c) / 3
    out["cci"] = (tp - tp.rolling(cci_period).mean()) / (0.015 * tp.rolling(cci_period).std().replace(0, np.nan))

    # ATR: 14
    out["atr"] = _atr(cfg["atr_period"])

    # Parabolic SAR: step 0.02, max 0.2 (5m) or 0.18 (15m)
    psar_step, psar_max = cfg["psar_step"], cfg["psar_max"]
    if HAS_PANDAS_TA:
        psar = ta.psar(h, l, c, af0=psar_step, af=psar_step, max_af=psar_max)
        if psar is not None and not psar.empty:
            out["psar"] = psar.iloc[:, 0]
    else:
        out["psar"] = _psar_np(h, l, c, psar_step, psar_max)

    # Supertrend: 10, 3
    st_period, st_mult = cfg["supertrend_period"], cfg["supertrend_mult"]
    if HAS_PANDAS_TA:
        supertrend = ta.supertrend(h, l, c, length=st_period, multiplier=st_mult)
        if supertrend is not None and not supertrend.empty:
            out["supertrend"] = supertrend.iloc[:, 0]
            out["supertrend_dir"] = supertrend.iloc[:, 1] if supertrend.shape[1] > 1 else 1
    else:
        out["supertrend"], out["supertrend_dir"] = _supertrend_np(h, l, c, st_period, st_mult)

    # VWAP
    typical = (h + l + c) / 3
    out["vwap"] = (typical * v).cumsum() / v.cumsum().replace(0, np.nan)

    # Awesome Oscillator: 5, 34
    ao_fast, ao_slow = cfg["ao_fast"], cfg["ao_slow"]
    out["ao"] = _sma((h + l) / 2, ao_fast) - _sma((h + l) / 2, ao_slow)

    # Ichimoku: 9, 26, 52
    it, ik, isk = cfg["ichimoku_tenkan"], cfg["ichimoku_kijun"], cfg["ichimoku_senkou"]
    if HAS_PANDAS_TA:
        try:
            ichi = ta.ichimoku(h, l, c, tenkan=it, kijun=ik, senkou=isk)
            if ichi is not None and len(ichi) > 0:
                ik_df = ichi[0] if isinstance(ichi, tuple) else ichi
                cols = ik_df.columns.tolist() if hasattr(ik_df, "columns") else []
                out["tenkan"] = ik_df[cols[0]] if len(cols) > 0 else ik_df.iloc[:, 0]
                out["kijun"] = ik_df[cols[1]] if len(cols) > 1 else ik_df.iloc[:, 1]
                out["senkou_a"] = ik_df[cols[2]] if len(cols) > 2 else ik_df.iloc[:, 2]
                out["senkou_b"] = ik_df[cols[3]] if len(cols) > 3 else ik_df.iloc[:, 3]
        except Exception:
            pass
    if "tenkan" not in out.columns:
        out["tenkan"] = (h.rolling(it).max() + l.rolling(it).min()) / 2
        out["kijun"] = (h.rolling(ik).max() + l.rolling(ik).min()) / 2
        out["senkou_a"] = ((out["tenkan"] + out["kijun"]) / 2).shift(ik)
        out["senkou_b"] = ((h.rolling(isk).max() + l.rolling(isk).min()) / 2).shift(ik)

    return out


def _psar_np(high: pd.Series, low: pd.Series, close: pd.Series, step: float, max_af: float) -> pd.Series:
    """Simple Parabolic SAR."""
    psar = close.copy()
    af = step
    ep = high.iloc[0]
    trend = 1
    for i in range(1, len(close)):
        if trend == 1:
            psar.iloc[i] = psar.iloc[i - 1] + af * (ep - psar.iloc[i - 1])
            if low.iloc[i] < psar.iloc[i]:
                trend = -1
                psar.iloc[i] = ep
                ep = low.iloc[i]
                af = step
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_af)
        else:
            psar.iloc[i] = psar.iloc[i - 1] - af * (psar.iloc[i - 1] - ep)
            if high.iloc[i] > psar.iloc[i]:
                trend = 1
                psar.iloc[i] = ep
                ep = high.iloc[i]
                af = step
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_af)
    return psar


def _supertrend_np(high: pd.Series, low: pd.Series, close: pd.Series, period: int, mult: float):
    """Supertrend indicator."""
    atr = (high - low).abs()
    atr = atr.rolling(period).mean()
    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    supertrend = close.copy()
    dir_ = pd.Series(1, index=close.index)
    for i in range(period, len(close)):
        if close.iloc[i] > upper.iloc[i - 1]:
            dir_.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = dir_.iloc[i - 1]
        if dir_.iloc[i] == 1:
            supertrend.iloc[i] = lower.iloc[i] if lower.iloc[i] > supertrend.iloc[i - 1] else supertrend.iloc[i - 1]
        else:
            supertrend.iloc[i] = upper.iloc[i] if upper.iloc[i] < supertrend.iloc[i - 1] else supertrend.iloc[i - 1]
    return supertrend, dir_
