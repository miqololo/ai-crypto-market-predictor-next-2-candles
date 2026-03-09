"""
Binance Futures public API data fetcher.
Fetches: basis, top-long-short-account-ratio, taker-long-short-ratio,
long-short-position-ratio, open-interest, funding-rate-history.
All endpoints are public (no API key required).
"""
import time
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, List


FAPI_BASE = "https://fapi.binance.com"
BAPI_BASE = "https://www.binance.com/bapi/futures/v1/public/future/common"


def _sym(symbol: str) -> str:
    """Normalize symbol: BTC/USDT:USDT -> BTCUSDT."""
    return symbol.replace("/", "").replace(":USDT", "").replace(":USDT", "")


def _period(timeframe: str) -> str:
    tf = (timeframe or "5m").lower()
    if "1h" in tf or "60" in tf:
        return "1h"
    if "15" in tf:
        return "15m"
    if "30" in tf:
        return "30m"
    return "5m"


def fetch_basis(
    symbol: str = "BTCUSDT",
    period: str = "5m",
    limit: int = 30,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> List[Dict]:
    """
    Basis = futures price - index price.
    GET /futures/data/basis (Binance USDS-M Futures)
    """
    sym = _sym(symbol)
    params = {"pair": sym, "contractType": "PERPETUAL", "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    try:
        r = httpx.get(
            f"{FAPI_BASE}/futures/data/basis",
            params=params,
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_global_long_short_account_ratio(
    symbol: str,
    period: str = "5m",
    limit: int = 30,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> List[Dict]:
    """Global long/short account ratio."""
    params = {"symbol": _sym(symbol), "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    try:
        r = httpx.get(
            f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio",
            params=params,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_top_long_short_account_ratio(
    symbol: str,
    period: str = "5m",
    limit: int = 30,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> List[Dict]:
    """Top trader long/short account ratio."""
    params = {"symbol": _sym(symbol), "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    try:
        r = httpx.get(
            f"{FAPI_BASE}/futures/data/topLongShortAccountRatio",
            params=params,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_top_long_short_position_ratio(
    symbol: str,
    period: str = "5m",
    limit: int = 30,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> List[Dict]:
    """Top trader long/short position ratio."""
    params = {"symbol": _sym(symbol), "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    try:
        r = httpx.get(
            f"{FAPI_BASE}/futures/data/topLongShortPositionRatio",
            params=params,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_taker_long_short_ratio(
    symbol: str,
    period: str = "5m",
    limit: int = 30,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> List[Dict]:
    """Taker buy/sell volume ratio."""
    params = {"symbol": _sym(symbol), "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    try:
        r = httpx.get(
            f"{FAPI_BASE}/futures/data/takerlongshortRatio",
            params=params,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_open_interest_hist(
    symbol: str,
    period: str = "5m",
    limit: int = 30,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> List[Dict]:
    """Open interest history."""
    params = {"symbol": _sym(symbol), "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    try:
        r = httpx.get(
            f"{FAPI_BASE}/futures/data/openInterestHist",
            params=params,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_funding_rate_history(
    symbol: str,
    limit: int = 100,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> List[Dict]:
    """Historical funding rates via FAPI (fapi.binance.com)."""
    params = {"symbol": _sym(symbol), "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    try:
        r = httpx.get(
            f"{FAPI_BASE}/fapi/v1/fundingRate",
            params=params,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_funding_rate_bapi(
    symbol: str,
    limit: int = 100,
) -> List[Dict]:
    """
    Historical funding rates via Binance BAPI (used by Binance web UI).
    Returns data with calcTime, lastFundingRate. Normalized to match FAPI format:
    fundingTime=calcTime, fundingRate=lastFundingRate.
    """
    try:
        r = httpx.post(
            f"{BAPI_BASE}/get-funding-rate-history",
            json={"symbol": _sym(symbol), "limit": limit},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("success") and data.get("data"):
                # Normalize to FAPI format (fundingTime, fundingRate). BAPI returns newest first; reverse to ascending.
                normalized = [
                    {"fundingTime": item["calcTime"], "fundingRate": item["lastFundingRate"], "symbol": item.get("symbol", "")}
                    for item in reversed(data["data"])
                ]
                return normalized
    except Exception:
        pass
    return []


def fetch_all_binance_data(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "5m",
    limit: int = 30,
) -> Dict[str, Any]:
    """
    Fetch all Binance Futures sentiment/market data.
    Returns basis, long/short ratios, taker ratio, open interest, funding history.
    """
    period = _period(timeframe)
    sym = _sym(symbol)

    result = {
        "symbol": sym,
        "period": period,
        "basis": [],
        "basis_current": None,
        "global_long_short_account_ratio": [],
        "long_short_ratio": None,
        "top_long_short_account_ratio": [],
        "top_trader_ratio": None,
        "top_long_short_position_ratio": [],
        "top_position_ratio": None,
        "taker_long_short_ratio": [],
        "taker_ratio": None,
        "open_interest_hist": [],
        "open_interest": None,
        "funding_rate_history": [],
        "funding_rate": None,
    }

    # Basis (futures/data/basis or fallback to premiumIndex)
    basis_data = fetch_basis(symbol, period, limit)
    if basis_data:
        result["basis"] = basis_data[:10]
        b = basis_data[0].get("basis")
        result["basis_current"] = float(b) if b is not None else None
    if not basis_data:
        # Fallback: premiumIndex (mark - index)
        try:
            r = httpx.get(f"{FAPI_BASE}/fapi/v1/premiumIndex", params={"symbol": sym}, timeout=10)
            if r.status_code == 200:
                d = r.json()
                mark = d.get("markPrice")
                index = d.get("indexPrice") or mark
                if mark is not None and index is not None:
                    result["basis_current"] = float(mark) - float(index)
                    result["mark_price"] = float(mark)
                    result["index_price"] = float(index)
        except Exception:
            pass

    # Global long/short account ratio
    gls = fetch_global_long_short_account_ratio(symbol, period, limit)
    if gls:
        result["global_long_short_account_ratio"] = gls[:10]
        ls = gls[0].get("longShortRatio")
        result["long_short_ratio"] = float(ls) if ls is not None else None

    # Top long/short account ratio
    tls = fetch_top_long_short_account_ratio(symbol, period, limit)
    if tls:
        result["top_long_short_account_ratio"] = tls[:10]
        tlr = tls[0].get("longShortRatio")
        result["top_trader_ratio"] = float(tlr) if tlr is not None else None

    # Top long/short position ratio
    tlp = fetch_top_long_short_position_ratio(symbol, period, limit)
    if tlp:
        result["top_long_short_position_ratio"] = tlp[:10]
        plr = tlp[0].get("longShortRatio")
        result["top_position_ratio"] = float(plr) if plr is not None else None

    # Taker long/short ratio
    taker = fetch_taker_long_short_ratio(symbol, period, limit)
    if taker:
        result["taker_long_short_ratio"] = taker[:10]
        bs = taker[0].get("buySellRatio")
        result["taker_ratio"] = float(bs) if bs is not None else None

    # Open interest history
    oi = fetch_open_interest_hist(symbol, period, limit)
    if oi:
        result["open_interest_hist"] = oi[:10]
        soi = oi[0].get("sumOpenInterest")
        result["open_interest"] = float(soi) if soi is not None else None

    # Funding rate history (FAPI, fallback to BAPI)
    fr = fetch_funding_rate_history(symbol, limit=limit)
    if not fr:
        fr = fetch_funding_rate_bapi(symbol, limit=limit)
    if fr:
        result["funding_rate_history"] = fr[:10]
        frv = fr[-1].get("fundingRate")
        result["funding_rate"] = float(frv) if frv is not None else None

    return result


def _ts_ms(ts) -> int:
    if ts is None:
        return 0
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp() * 1000)
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str):
        try:
            return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
        except Exception:
            pass
    return 0


def fetch_all_binance_histories(
    symbol: str,
    timeframe: str,
    start_time_ms: int,
    end_time_ms: int,
    limit: int = 500,
) -> Dict[str, List[Dict]]:
    """
    Fetch raw historical data from all Binance APIs for a time range.
    Returns dict of api_name -> list of raw records (as returned by each endpoint).
    Used to store each candle's own historical data.
    """
    period = _period(timeframe)

    result: Dict[str, List[Dict]] = {}

    result["basis"] = fetch_basis(symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms)
    if not result["basis"]:
        result["basis"] = fetch_basis(symbol, period, limit)
    time.sleep(0.1)

    result["global_long_short_account_ratio"] = fetch_global_long_short_account_ratio(
        symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms
    )
    if not result["global_long_short_account_ratio"]:
        result["global_long_short_account_ratio"] = fetch_global_long_short_account_ratio(symbol, period, limit)
    time.sleep(0.1)

    result["top_long_short_account_ratio"] = fetch_top_long_short_account_ratio(
        symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms
    )
    if not result["top_long_short_account_ratio"]:
        result["top_long_short_account_ratio"] = fetch_top_long_short_account_ratio(symbol, period, limit)
    time.sleep(0.1)

    result["top_long_short_position_ratio"] = fetch_top_long_short_position_ratio(
        symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms
    )
    if not result["top_long_short_position_ratio"]:
        result["top_long_short_position_ratio"] = fetch_top_long_short_position_ratio(symbol, period, limit)

    result["taker_long_short_ratio"] = fetch_taker_long_short_ratio(
        symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms
    )
    if not result["taker_long_short_ratio"]:
        result["taker_long_short_ratio"] = fetch_taker_long_short_ratio(symbol, period, limit)

    result["open_interest_hist"] = fetch_open_interest_hist(
        symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms
    )
    if not result["open_interest_hist"]:
        result["open_interest_hist"] = fetch_open_interest_hist(symbol, period, limit)

    result["funding_rate_history"] = fetch_funding_rate_history(
        symbol, limit, start_time=start_time_ms, end_time=end_time_ms
    )
    if not result["funding_rate_history"]:
        result["funding_rate_history"] = fetch_funding_rate_history(symbol, limit)
    if not result["funding_rate_history"]:
        result["funding_rate_history"] = fetch_funding_rate_bapi(symbol, limit)

    return result


def build_candle_binance_lookup(
    symbol: str,
    timeframe: str,
    start_time_ms: int,
    end_time_ms: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Fetch Binance data for the candle date range. Returns lookup: ts_ms -> {basis, long_short_ratio, ...}.
    basis = futures price - spot price (single value per timestamp).
    Falls back to recent data if range query returns empty.
    """
    period = _period(timeframe)
    limit = 500
    lookup: Dict[int, Dict[str, Any]] = {}

    def _add(records: List[Dict], key: str, value_key: str, ts_key: str = "timestamp"):
        for r in records:
            ts = int(r.get(ts_key, r.get("fundingTime", 0)))
            if ts:
                raw = r.get(value_key)
                val = float(raw) if raw is not None else None
                if ts not in lookup:
                    lookup[ts] = {}
                lookup[ts][key] = val

    basis_data = fetch_basis(symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms)
    if not basis_data:
        basis_data = fetch_basis(symbol, period, limit)
    _add(basis_data, "basis", "basis")

    gls = fetch_global_long_short_account_ratio(symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms)
    if not gls:
        gls = fetch_global_long_short_account_ratio(symbol, period, limit)
    _add(gls, "long_short_ratio", "longShortRatio")

    tls = fetch_top_long_short_account_ratio(symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms)
    if not tls:
        tls = fetch_top_long_short_account_ratio(symbol, period, limit)
    _add(tls, "top_trader_ratio", "longShortRatio")

    tlp = fetch_top_long_short_position_ratio(symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms)
    if not tlp:
        tlp = fetch_top_long_short_position_ratio(symbol, period, limit)
    for r in tlp:
        ts = int(r.get("timestamp", 0))
        if ts:
            raw = r.get("longShortRatio")
            val = float(raw) if raw is not None else None
            if ts not in lookup:
                lookup[ts] = {}
            lookup[ts]["top_position_ratio"] = val

    taker = fetch_taker_long_short_ratio(symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms)
    if not taker:
        taker = fetch_taker_long_short_ratio(symbol, period, limit)
    _add(taker, "taker_ratio", "buySellRatio")

    oi = fetch_open_interest_hist(symbol, period, limit, start_time=start_time_ms, end_time=end_time_ms)
    if not oi:
        oi = fetch_open_interest_hist(symbol, period, limit)
    _add(oi, "sum_open_interest", "sumOpenInterest")
    _add(oi, "sum_open_interest_value", "sumOpenInterestValue")
    _add(oi, "cmc_circulating_supply", "CMCCirculatingSupply")

    fr = fetch_funding_rate_history(symbol, limit, start_time=start_time_ms, end_time=end_time_ms)
    if not fr:
        fr = fetch_funding_rate_history(symbol, limit)
    if not fr:
        fr = fetch_funding_rate_bapi(symbol, limit=limit)
    for r in fr:
        ts = int(r.get("fundingTime", r.get("timestamp", 0)))
        if ts:
            raw = r.get("fundingRate")
            val = float(raw) if raw is not None else None
            if ts not in lookup:
                lookup[ts] = {}
            lookup[ts]["funding_rate"] = val

    return lookup


def merge_binance_into_candles(
    records: List[Dict],
    symbol: str,
    timeframe: str,
) -> None:
    """
    Attach Binance data to each candle. Same count and date range as candles.
    basis = futures - spot (single value). funding_rate = that candle's rate (not array).
    """
    period = _period(timeframe)
    period_ms = {"5m": 300000, "15m": 900000, "30m": 1800000, "1h": 3600000}.get(period, 300000)

    if not records:
        return

    ts_list = [_ts_ms(r.get("timestamp")) for r in records]
    start_ms = min(ts_list)
    end_ms = max(ts_list)
    lookup = build_candle_binance_lookup(symbol, timeframe, start_ms, end_ms)

    # Build aligned arrays: one value per candle (nearest Binance timestamp)
    # Funding rate is every 8h; use 8h tolerance. Others use 2x period.
    funding_tolerance_ms = 8 * 60 * 60 * 1000  # 8 hours

    def _get_val(key: str, ts_ms: int) -> Optional[float]:
        rounded = (ts_ms // period_ms) * period_ms
        if not lookup:
            return None
        # For funding_rate, only consider records that have this key
        candidates = [(k, lookup[k].get(key)) for k in lookup if lookup[k].get(key) is not None]
        if not candidates:
            return None
        tolerance = funding_tolerance_ms if key == "funding_rate" else period_ms * 2
        nearest_ts = min(candidates, key=lambda x: abs(x[0] - rounded))[0]
        if abs(nearest_ts - rounded) <= tolerance:
            return lookup[nearest_ts].get(key)
        return None

    keys = ["basis", "long_short_ratio", "top_trader_ratio", "top_position_ratio", "taker_ratio", "sum_open_interest", "sum_open_interest_value", "cmc_circulating_supply", "funding_rate"]
    aligned: Dict[str, List[Optional[float]]] = {k: [] for k in keys}
    for i, ts_ms in enumerate(ts_list):
        for k in keys:
            aligned[k].append(_get_val(k, ts_ms))

    for i, rec in enumerate(records):
        for k in keys:
            val = aligned[k][i] if i < len(aligned[k]) else None
            rec[k] = val
