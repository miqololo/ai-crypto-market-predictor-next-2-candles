import pandas as pd
import numpy as np
from app.strategies.base import BaseStrategy


class FibonacciRSIMACDStrategy(BaseStrategy):
    """
    Fibonacci Retracement Strategy with RSI and MACD confirmation.

    Fixes applied vs original:
    - Swing high/low use shift(window//2) to avoid look-ahead bias
    - Fib levels are frozen on confirmed swings (cooldown), not recalculated every bar
    - Entry requires a bullish/bearish candle body filter + volume filter
    - fib_tolerance band replaces exact price cross (more realistic)
    - Short/long resistance levels are symmetric and consistent
    - Signals are cast to int to avoid float dtype issues

    Entry signals:
    - Long:  Price enters Fib support band + RSI > threshold + MACD histogram turns positive
             + bullish candle body + volume above average
    - Short: Price enters Fib resistance band + RSI < threshold + MACD histogram turns negative
             + bearish candle body + volume above average

    Parameters:
    Parameters tuned for M15 crypto (BTC/ETH/SOL):
    - swing_window:      10 bars = 2.5h lookback для внутридневного свинга
    - swing_cooldown:    16 bars = 4h, не перерисовываем Fib чаще раза в сессию
    - fib_levels:        [0.382, 0.5, 0.618]
    - fib_tolerance:     0.004 = ±0.4%, чуть шире из-за шума M15
    - rsi_threshold:     50.0
    - volume_multiplier: 1.15, мягче чем на H1 — M15 объёмы неровные
    - body_ratio:        0.25, M15 свечи мельче чем H1
    """

    def __init__(
        self,
        swing_window: int = 10,
        swing_cooldown: int = 16,
        fib_levels: list = None,
        fib_tolerance: float = 0.004,
        rsi_threshold: float = 50.0,
        volume_multiplier: float = 1.15,
        body_ratio: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.swing_window = swing_window
        self.swing_cooldown = swing_cooldown
        self.fib_levels = fib_levels if fib_levels is not None else [0.382, 0.5, 0.618]
        self.fib_tolerance = fib_tolerance
        self.rsi_threshold = rsi_threshold
        self.volume_multiplier = volume_multiplier
        self.body_ratio = body_ratio

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _confirmed_swings(self, high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Return swing high / swing low series without look-ahead bias.

        A bar is a swing high if it is the rolling max of the preceding
        `swing_window` bars (pure past data, no center=True).
        We then apply a cooldown so Fib anchors are not reset on every bar.
        """
        half = self.swing_window // 2

        # Rolling max/min over past bars only — no future data
        rolling_high = high.rolling(self.swing_window).max()
        rolling_low  = low.rolling(self.swing_window).min()

        is_swing_high = (high == rolling_high)
        is_swing_low  = (low  == rolling_low)

        # Cooldown: keep only the first swing in each cooldown window
        swing_high_confirmed = pd.Series(np.nan, index=high.index)
        swing_low_confirmed  = pd.Series(np.nan, index=low.index)

        last_high_bar = -self.swing_cooldown
        last_low_bar  = -self.swing_cooldown

        for i in range(len(high)):
            if is_swing_high.iloc[i] and (i - last_high_bar) >= self.swing_cooldown:
                swing_high_confirmed.iloc[i] = high.iloc[i]
                last_high_bar = i
            if is_swing_low.iloc[i] and (i - last_low_bar) >= self.swing_cooldown:
                swing_low_confirmed.iloc[i] = low.iloc[i]
                last_low_bar = i

        # Forward-fill so every bar has the most recent confirmed anchor
        swing_high_confirmed = swing_high_confirmed.ffill()
        swing_low_confirmed  = swing_low_confirmed.ffill()

        return swing_high_confirmed, swing_low_confirmed

    def _fib_bands(
        self, swing_high: pd.Series, swing_low: pd.Series
    ) -> dict[str, tuple[pd.Series, pd.Series]]:
        """
        For each Fib level return (lower_band, upper_band) pairs.

        Uptrend levels (support):  swing_high - level * range  ± tolerance
        Downtrend levels (resist.): swing_low  + level * range  ± tolerance
        Both are computed so callers can choose which to use.
        """
        price_range = swing_high - swing_low
        bands = {}
        for level in self.fib_levels:
            support   = swing_high - level * price_range
            resistance = swing_low  + level * price_range
            bands[level] = {
                "support_lo":    support    * (1 - self.fib_tolerance),
                "support_hi":    support    * (1 + self.fib_tolerance),
                "resistance_lo": resistance * (1 - self.fib_tolerance),
                "resistance_hi": resistance * (1 + self.fib_tolerance),
            }
        return bands

    @staticmethod
    def _bullish_candle(df: pd.DataFrame, body_ratio: float) -> pd.Series:
        total_range = (df["high"] - df["low"]).replace(0, np.nan)
        body = df["close"] - df["open"]
        return (body > 0) & (body / total_range >= body_ratio)

    @staticmethod
    def _bearish_candle(df: pd.DataFrame, body_ratio: float) -> pd.Series:
        total_range = (df["high"] - df["low"]).replace(0, np.nan)
        body = df["open"] - df["close"]
        return (body > 0) & (body / total_range >= body_ratio)

    @staticmethod
    def _volume_ok(df: pd.DataFrame, multiplier: float, window: int = 20) -> pd.Series:
        avg_vol = df["volume"].rolling(window).mean()
        return df["volume"] > avg_vol * multiplier

    # ─────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Fibonacci retracement, RSI, and MACD.

        Args:
            df: DataFrame with OHLCV columns and pre-computed indicators:
                'rsi', 'macd', 'macd_signal', 'volume'

        Returns:
            Series with signals: 1 = long, -1 = short, 0 = flat
        """
        signals = pd.Series(0, index=df.index, dtype=int)

        required = {"high", "low", "close", "open", "volume", "rsi", "macd", "macd_signal"}
        if not required.issubset(df.columns):
            return signals

        min_bars = self.swing_window + self.swing_cooldown
        if len(df) < min_bars:
            return signals

        # ── Swing anchors (no look-ahead) ──────────────────────
        swing_high, swing_low = self._confirmed_swings(df["high"], df["low"])

        # ── Fib bands around each level ────────────────────────
        bands = self._fib_bands(swing_high, swing_low)

        # ── Shared filters ─────────────────────────────────────
        rsi         = df["rsi"]
        macd_hist   = df["macd"] - df["macd_signal"]
        macd_cross_up   = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
        macd_cross_down = (macd_hist < 0) & (macd_hist.shift(1) >= 0)

        vol_ok        = self._volume_ok(df, self.volume_multiplier)
        bull_candle   = self._bullish_candle(df, self.body_ratio)
        bear_candle   = self._bearish_candle(df, self.body_ratio)

        price = df["close"]

        # ── Trend direction from swing relationship ─────────────
        uptrend   = swing_high > swing_high.shift(self.swing_cooldown)
        downtrend = swing_low  < swing_low.shift(self.swing_cooldown)

        # ── Build composite signal across all Fib levels ───────
        long_signal  = pd.Series(False, index=df.index)
        short_signal = pd.Series(False, index=df.index)

        for level, b in bands.items():
            # Long: price inside support band in an uptrend
            in_support_band = (price >= b["support_lo"]) & (price <= b["support_hi"])
            long_signal |= (
                uptrend
                & in_support_band
                & (rsi > self.rsi_threshold)
                & macd_cross_up
                & bull_candle
                & vol_ok
            )

            # Short: price inside resistance band in a downtrend
            in_resistance_band = (price >= b["resistance_lo"]) & (price <= b["resistance_hi"])
            short_signal |= (
                downtrend
                & in_resistance_band
                & (rsi < self.rsi_threshold)
                & macd_cross_down
                & bear_candle
                & vol_ok
            )

        # Long takes priority if both fire on the same bar (rare, but explicit)
        signals[long_signal]                      =  1
        signals[short_signal & ~long_signal]      = -1

        return signals.fillna(0).astype(int)
