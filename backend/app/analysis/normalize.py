"""Z-scores, min-max embeddings, log-returns for pattern matching / RAG."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def compute_normalized_features(
    df: pd.DataFrame,
    window: int = 20,
    atr_col: str = "atr",
    embedding_window: Optional[int] = None,
) -> Dict:
    """
    Z-score by ATR, min-max 0-1, log-returns, relative stretch.
    For FAISS/Chroma pattern matching. 5m: window 20, 15m: window 40.
    """
    emb_win = embedding_window or window
    if len(df) < window:
        window = len(df) - 1
    recent = df.tail(window)
    c = recent["close"]
    o = recent["open"]
    h = recent["high"]
    l_ = recent["low"]
    atr = recent[atr_col] if atr_col in recent.columns else (c - c.shift(1)).abs().rolling(5).mean()
    atr_val = atr.iloc[-1] if pd.notna(atr.iloc[-1]) and atr.iloc[-1] > 0 else 1e-10
    ema = recent["ema_9"] if "ema_9" in recent.columns else c.rolling(5).mean()
    sma = recent["sma_20"] if "sma_20" in recent.columns else c.rolling(5).mean()

    # Z-score by ATR (preserves shape)
    z_close = (c - c.mean()) / (atr_val + 1e-10)
    z_open = (o - o.mean()) / (atr_val + 1e-10)
    z_high = (h - h.mean()) / (atr_val + 1e-10)
    z_low = (l_ - l_.mean()) / (atr_val + 1e-10)

    # Min-Max 0-1 within window (for geometric similarity)
    def minmax_norm(s):
        mn, mx = s.min(), s.max()
        if mx - mn == 0:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    mm_close = minmax_norm(c)
    mm_open = minmax_norm(o)
    mm_high = minmax_norm(h)
    mm_low = minmax_norm(l_)

    # Log-returns
    log_ret = np.log(c / c.shift(1)).fillna(0)

    # Relative stretch: (Close - EMA) / ATR
    ema_last = ema.iloc[-1] if pd.notna(ema.iloc[-1]) else c.iloc[-1]
    stretch = (c.iloc[-1] - ema_last) / atr_val if atr_val else 0

    # Last row values for API response
    last = df.index.get_loc(recent.index[-1]) if hasattr(df.index, "get_loc") else len(df) - 1

    return {
        "z_score_atr": {
            "close": float(z_close.iloc[-1]) if len(z_close) else None,
            "open": float(z_open.iloc[-1]) if len(z_open) else None,
            "high": float(z_high.iloc[-1]) if len(z_high) else None,
            "low": float(z_low.iloc[-1]) if len(z_low) else None,
        },
        "minmax_0_1": {
            "close": float(mm_close.iloc[-1]) if len(mm_close) else None,
            "open": float(mm_open.iloc[-1]) if len(mm_open) else None,
            "high": float(mm_high.iloc[-1]) if len(mm_high) else None,
            "low": float(mm_low.iloc[-1]) if len(mm_low) else None,
        },
        "log_returns_last_n": [float(x) for x in log_ret.tail(min(emb_win, 40)).tolist()],
        "relative_stretch_atr": float(stretch),
        "embedding_vector": [float(x) for x in mm_close.tail(emb_win).fillna(0.5).values],  # For Chroma/FAISS
    }
