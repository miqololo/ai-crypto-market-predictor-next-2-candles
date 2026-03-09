"""Backtesting runner: vectorbt (main) and Backtesting.py (prototype)."""
import logging
import pandas as pd
import numpy as np
from typing import Optional
from enum import Enum


class BacktestEngine(str, Enum):
    VECTORBT = "vectorbt"
    BACKTESTING_PY = "backtesting"


def _safe_get_avg_return(pf, total_trades, _f):
    """Safely get average trade return."""
    if total_trades == 0:
        return 0.0
    try:
        if hasattr(pf.trades, 'returns'):
            returns = pf.trades.returns()
            if hasattr(returns, 'mean'):
                return _f(returns.mean())
        if hasattr(pf.trades, 'avg_return'):
            return _f(pf.trades.avg_return())
    except Exception:
        pass
    return 0.0


def _safe_get_best_trade(pf, total_trades, _f):
    """Safely get best trade return."""
    if total_trades == 0:
        return 0.0
    try:
        # Try to get from stats first (more reliable)
        stats = pf.stats()
        if isinstance(stats, pd.Series):
            stats_dict = stats.to_dict()
            if 'Best Trade [%]' in stats_dict:
                return _f(stats_dict['Best Trade [%]']) / 100.0  # Convert from percentage to decimal
            elif 'Best Trade' in stats_dict:
                val = stats_dict['Best Trade']
                # If it's already a percentage string or number > 1, divide by 100
                if isinstance(val, str) and '%' in val:
                    return _f(val.replace('%', '')) / 100.0
                elif isinstance(val, (int, float)) and val > 1:
                    return _f(val) / 100.0
                return _f(val)
        
        # Fallback to direct trade returns
        if hasattr(pf.trades, 'returns'):
            returns = pf.trades.returns()
            if isinstance(returns, pd.Series) and len(returns) > 0:
                return _f(returns.max())
            elif hasattr(returns, 'max'):
                return _f(returns.max())
        if hasattr(pf.trades, 'best'):
            return _f(pf.trades.best())
    except Exception as e:
        import logging
        logging.warning(f"Error getting best trade: {e}")
        pass
    return 0.0


def _safe_get_worst_trade(pf, total_trades, _f):
    """Safely get worst trade return."""
    if total_trades == 0:
        return 0.0
    try:
        # Try to get from stats first (more reliable)
        stats = pf.stats()
        if isinstance(stats, pd.Series):
            stats_dict = stats.to_dict()
            if 'Worst Trade [%]' in stats_dict:
                return _f(stats_dict['Worst Trade [%]']) / 100.0  # Convert from percentage to decimal
            elif 'Worst Trade' in stats_dict:
                val = stats_dict['Worst Trade']
                # If it's already a percentage string or number > 1, divide by 100
                if isinstance(val, str) and '%' in val:
                    return _f(val.replace('%', '')) / 100.0
                elif isinstance(val, (int, float)) and val > 1:
                    return _f(val) / 100.0
                return _f(val)
        
        # Fallback to direct trade returns
        if hasattr(pf.trades, 'returns'):
            returns = pf.trades.returns()
            if isinstance(returns, pd.Series) and len(returns) > 0:
                return _f(returns.min())
            elif hasattr(returns, 'min'):
                return _f(returns.min())
        if hasattr(pf.trades, 'worst'):
            return _f(pf.trades.worst())
    except Exception as e:
        import logging
        logging.warning(f"Error getting worst trade: {e}")
        pass
    return 0.0


def _safe_get_expectancy(pf, total_trades, _f):
    """Safely get trade expectancy."""
    if total_trades == 0:
        return 0.0
    try:
        if hasattr(pf.trades, 'expectancy'):
            return _f(pf.trades.expectancy())
        # Calculate expectancy manually: avg_win * win_rate - avg_loss * (1 - win_rate)
        if hasattr(pf.trades, 'returns'):
            returns = pf.trades.returns()
            if hasattr(returns, 'mean'):
                return _f(returns.mean())
    except Exception:
        pass
    return 0.0


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    engine: BacktestEngine = BacktestEngine.VECTORBT,
    initial_capital: float = 10000.0,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> dict:
    """
    Run backtest. signals: 1=long, -1=short, 0=flat.
    
    Args:
        df: DataFrame with OHLCV data
        signals: Trading signals Series
        engine: Backtest engine to use
        initial_capital: Starting capital
        stop_loss: Stop loss as fraction (e.g., 0.01 for 1%). Default: None (disabled)
        take_profit: Take profit as fraction (e.g., 0.03 for 3%). Default: None (disabled)
    """
    import logging  # Ensure logging is available in function scope
    # Validate stop_loss and take_profit values
    if stop_loss is not None:
        if stop_loss < 0 or stop_loss > 1:
            logging.warning(f"Invalid stop_loss value: {stop_loss}. Must be between 0 and 1. Using default 0.01")
            stop_loss = 0.01
    if take_profit is not None:
        if take_profit < 0 or take_profit > 1:
            logging.warning(f"Invalid take_profit value: {take_profit}. Must be between 0 and 1. Using default 0.03")
            take_profit = 0.03
    
    logging.info(f"run_backtest called with stop_loss={stop_loss}, take_profit={take_profit}, engine={engine}")
    
    if engine == BacktestEngine.VECTORBT:
        return _backtest_vectorbt(df, signals, initial_capital, stop_loss=stop_loss, take_profit=take_profit)
    return _backtest_backtesting_py(df, signals, initial_capital)


def _backtest_vectorbt(
    df: pd.DataFrame, 
    signals: pd.Series, 
    initial_capital: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> dict:
    """vectorbt backtest with comprehensive metrics."""
    import logging  # Ensure logging is available in function scope
    try:
        import vectorbt as vbt
    except ImportError:
        return {"error": "vectorbt not installed", "engine": "vectorbt"}

    # Validate inputs with detailed diagnostics
    if len(df) == 0:
        return {
            "error": "Empty dataframe",
            "engine": "vectorbt",
            "message": "DataFrame is empty - check data fetching",
        }
    
    if len(signals) == 0:
        return {
            "error": "Empty signals",
            "engine": "vectorbt",
            "message": "Signals Series is empty - check strategy signal generation",
        }
    
    # Validate required columns exist
    required_cols = ["close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {
            "error": f"Missing required columns: {missing_cols}",
            "engine": "vectorbt",
            "message": f"DataFrame missing columns: {', '.join(missing_cols)}",
        }
    
    # Ensure signals are aligned with dataframe
    try:
        signals = signals.reindex(df.index).fillna(0)
    except Exception as reindex_error:
        return {
            "error": f"Failed to align signals with dataframe: {str(reindex_error)}",
            "engine": "vectorbt",
            "message": "Signal index mismatch with dataframe index",
        }
    
    # Check if we have any non-zero signals
    non_zero_count = (signals != 0).sum()
    if non_zero_count == 0:
        # Return zero result for no signals with diagnostic info
        return {
            "engine": "vectorbt",
            "total_return": 0.0,
            "total_profit": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "message": f"No trading signals generated (data length: {len(df)}, signals length: {len(signals)})",
            "diagnostics": {
                "data_length": len(df),
                "signals_length": len(signals),
                "signal_range": f"{signals.min()} to {signals.max()}" if len(signals) > 0 else "N/A",
            },
        }

    # Extract and validate price data
    # For stop loss/take profit, we need OHLC data
    try:
        price = df["close"].reindex(signals.index).ffill().bfill()
        # Also extract high and low for stop loss/take profit calculations
        high = df["high"].reindex(signals.index).ffill().bfill() if "high" in df.columns else None
        low = df["low"].reindex(signals.index).ffill().bfill() if "low" in df.columns else None
    except Exception as price_error:
        return {
            "error": f"Failed to extract price data: {str(price_error)}",
            "engine": "vectorbt",
            "message": "Error processing close prices",
        }
    
    # Validate price data quality
    if price.isna().all():
        return {
            "error": "No valid price data",
            "engine": "vectorbt",
            "message": "All price values are NaN after forward/backward fill",
        }
    
    # Check for sufficient valid price data
    valid_price_count = price.notna().sum()
    if valid_price_count < len(price) * 0.5:  # Less than 50% valid prices
        return {
            "error": "Insufficient valid price data",
            "engine": "vectorbt",
            "message": f"Only {valid_price_count}/{len(price)} price values are valid",
        }
    
    # Convert signals to entry/exit signals
    # Entry: signal changes from 0 or -1 to 1 (long entry)
    # Exit: signal changes from 1 to -1 or 0 (exit long position)
    entries = (signals == 1) & (signals.shift(1) != 1)
    # For exits: we want to exit when signal becomes -1 or goes back to 0
    # But if we have stop loss/take profit, those will handle exits automatically
    exits = (signals == -1) | ((signals == 0) & (signals.shift(1) == 1))
    
    # Check if we have any entries
    if entries.sum() == 0:
        return {
            "engine": "vectorbt",
            "total_return": 0.0,
            "total_profit": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "message": "No entry signals detected - check signal generation logic",
        }
    
    # Prepare stop loss and take profit parameters
    # vectorbt expects sl_stop and tp_stop as fractions (e.g., 0.01 for 1%)
    # These are percentage distances from entry price
    # For stop loss: price falls sl_stop% below entry -> exit
    # For take profit: price rises tp_stop% above entry -> exit
    sl_stop = stop_loss if stop_loss is not None else np.nan
    tp_stop = take_profit if take_profit is not None else np.nan
    
    # Log the values being used
    logging.info(f"VectorBT backtest: stop_loss={sl_stop}, take_profit={tp_stop}, has_high={high is not None}, has_low={low is not None}")
    
    try:
        # Build kwargs for Portfolio.from_signals
        # vectorbt expects 'close' parameter, not 'price'
        # For stop loss/take profit to work properly, we need to provide high and low
        portfolio_kwargs = {
            "close": price,
            "entries": entries,
            "exits": exits,
            "init_cash": initial_capital,
            "freq": "1h",
        }
        
        # Add high and low if available (needed for stop loss/take profit)
        if high is not None:
            portfolio_kwargs["high"] = high
        if low is not None:
            portfolio_kwargs["low"] = low
        
        # Add stop loss and take profit if provided
        # Note: sl_stop and tp_stop are percentage values relative to entry price
        # sl_stop=0.01 means exit if price falls 1% below entry
        # tp_stop=0.03 means exit if price rises 3% above entry
        if not np.isnan(sl_stop) and sl_stop > 0:
            portfolio_kwargs["sl_stop"] = sl_stop
            logging.info(f"Added sl_stop={sl_stop} to portfolio kwargs")
        if not np.isnan(tp_stop) and tp_stop > 0:
            portfolio_kwargs["tp_stop"] = tp_stop
            logging.info(f"Added tp_stop={tp_stop} to portfolio kwargs")
        
        logging.info(f"Creating Portfolio.from_signals with keys: {list(portfolio_kwargs.keys())}")
        pf = vbt.Portfolio.from_signals(**portfolio_kwargs)
    except Exception as e:
        return {
            "error": f"VectorBT portfolio creation failed: {str(e)}",
            "engine": "vectorbt",
            "total_return": 0.0,
            "total_profit": 0.0,
            "total_trades": 0,
        }
    
    def _f(v):
        """Convert value to float, handling numpy/pandas types."""
        if v is None:
            return 0.0
        try:
            # Handle pandas Series
            if isinstance(v, pd.Series):
                if len(v) == 0:
                    return 0.0
                # Try to get scalar value
                scalar_val = v.iloc[-1] if len(v) > 0 else v.iloc[0]
                if pd.isna(scalar_val):
                    return 0.0
                x = float(scalar_val)
            # Handle numpy arrays
            elif isinstance(v, np.ndarray):
                if len(v) == 0:
                    return 0.0
                scalar_val = v[-1] if len(v) > 0 else v[0]
                if np.isnan(scalar_val):
                    return 0.0
                x = float(scalar_val)
            # Handle objects with .item() method (numpy scalars, pandas scalars)
            elif hasattr(v, "item"):
                x = float(v.item())
            # Try direct conversion
            else:
                x = float(v)
            # Check for NaN or Inf - convert to JSON-safe values
            if x != x:  # NaN check
                return 0.0
            if abs(x) == float("inf"):
                # Convert infinity to None (JSON-safe) - frontend can handle as "N/A" or "inf"
                return None
            return x
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            # Log error for debugging
            logging.warning(f"Error converting value to float: {type(v)}, {str(e)}")
            return 0.0

    def _i(v):
        """Convert value to int, handling numpy/pandas types."""
        if v is None:
            return 0
        try:
            if hasattr(v, "item"):
                return int(v.item())
            elif isinstance(v, (pd.Series, np.ndarray)):
                return int(v.iloc[0] if len(v) > 0 else 0)
            else:
                return int(v)
        except (ValueError, TypeError, IndexError):
            return 0
    
    def _safe_series(v):
        """Convert Series to list of values for JSON serialization."""
        try:
            if isinstance(v, pd.Series):
                return v.fillna(0).tolist()
            elif isinstance(v, np.ndarray):
                return v.tolist()
            elif hasattr(v, '__iter__') and not isinstance(v, str):
                return list(v)
            return []
        except Exception:
            return []
    
    # Get comprehensive stats - handle potential errors
    try:
        stats = pf.stats()
        stats_dict = {}
        if isinstance(stats, pd.Series):
            stats_dict = stats.to_dict()
    except Exception as stats_error:
        import logging
        logging.warning(f"Failed to get portfolio stats: {stats_error}")
        stats_dict = {}
    
    # Get trade count - handle potential errors
    try:
        tc = pf.trades.count()
        total_trades = _i(tc)
    except Exception as trade_error:
        import logging
        logging.warning(f"Failed to get trade count: {trade_error}")
        total_trades = 0
    
    # Extract equity curve - handle potential errors
    equity_series = []
    try:
        equity = pf.value()
        if isinstance(equity, pd.Series):
            equity_series = equity.fillna(initial_capital).tolist()
        elif isinstance(equity, np.ndarray):
            equity_series = equity.tolist()
        elif hasattr(equity, 'tolist'):
            equity_series = equity.tolist()
    except Exception as equity_error:
        import logging
        logging.warning(f"Failed to extract equity curve: {equity_error}")
        equity_series = []
    
    # Extract drawdowns
    max_dd = _f(pf.max_drawdown())
    drawdown_series = []
    try:
        drawdowns = pf.drawdowns()
        if hasattr(drawdowns, 'drawdown'):
            dd_vals = drawdowns.drawdown
            if isinstance(dd_vals, pd.Series):
                drawdown_series = dd_vals.fillna(0).tolist()
            elif isinstance(dd_vals, np.ndarray):
                drawdown_series = dd_vals.tolist()
    except Exception:
        pass
    
    # Extract trades data (limit to prevent huge JSON)
    trades_data = []
    try:
        if total_trades > 0:
            trades_records = pf.trades.records
            if hasattr(trades_records, 'read'):
                trades_df = trades_records.read()
                # Limit to 100 trades for JSON size
                for idx in range(min(100, len(trades_df))):
                    trade = trades_df.iloc[idx]
                    trades_data.append({
                        "entry_idx": int(trade.get('entry_idx', 0)),
                        "exit_idx": int(trade.get('exit_idx', 0)),
                        "pnl": _f(trade.get('pnl', 0)),
                        "return": _f(trade.get('return', 0)),
                        "duration": int(trade.get('duration', 0)),
                        "entry_price": _f(trade.get('entry_price', 0)),
                        "exit_price": _f(trade.get('exit_price', 0)),
                    })
    except Exception:
        pass
    
    # Calculate additional metrics safely
    winning_trades = 0
    losing_trades = 0
    avg_win = 0.0
    avg_loss = 0.0
    profit_factor = 0.0
    
    try:
        if total_trades > 0:
            # Get win rate - prefer stats_dict if available (more reliable)
            win_rate_percentage = None
            if 'Win Rate [%]' in stats_dict:
                win_rate_percentage = _f(stats_dict['Win Rate [%]'])
            else:
                # Fallback to direct method
                win_rate_raw = pf.trades.win_rate()
                win_rate_val = _f(win_rate_raw)
                # VectorBT typically returns percentage (0-100), but check format
                if win_rate_val <= 1.0:
                    # Decimal (0-1), convert to percentage
                    win_rate_percentage = win_rate_val * 100.0
                else:
                    # Already percentage (0-100)
                    win_rate_percentage = win_rate_val
            
            # Calculate winning/losing trades from percentage
            if win_rate_percentage is not None:
                winning_trades = _i(win_rate_percentage * total_trades / 100.0)
                losing_trades = total_trades - winning_trades
            else:
                # Fallback: count winning/losing trades directly
                try:
                    winning_count = len(pf.trades.winning) if hasattr(pf.trades, 'winning') else 0
                    losing_count = len(pf.trades.losing) if hasattr(pf.trades, 'losing') else 0
                    if winning_count > 0 or losing_count > 0:
                        winning_trades = winning_count
                        losing_trades = losing_count
                except Exception:
                    pass
            
            # Get average win/loss using returns property (not method!)
            if winning_trades > 0:
                try:
                    # VectorBT: returns is a property, not a method - don't call it!
                    winning_returns = pf.trades.winning.returns
                    # Convert to array/Series if needed
                    if hasattr(winning_returns, 'values'):
                        winning_returns = winning_returns.values
                    elif hasattr(winning_returns, 'to_numpy'):
                        winning_returns = winning_returns.to_numpy()
                    
                    if isinstance(winning_returns, pd.Series) and len(winning_returns) > 0:
                        avg_win = _f(winning_returns.mean())
                    elif hasattr(winning_returns, '__len__') and len(winning_returns) > 0:
                        # Handle numpy array or MappedArray
                        if isinstance(winning_returns, np.ndarray):
                            avg_win = _f(float(np.mean(winning_returns)))
                        else:
                            # Fallback: convert to list and calculate mean
                            try:
                                returns_list = list(winning_returns) if hasattr(winning_returns, '__iter__') else [winning_returns]
                                avg_win = _f(float(sum(returns_list) / len(returns_list)))
                            except Exception:
                                avg_win = 0.0
                except Exception as e:
                    logging.warning(f"Error calculating avg_win: {e}")
                    pass
            
            if losing_trades > 0:
                try:
                    # VectorBT: returns is a property, not a method - don't call it!
                    losing_returns = pf.trades.losing.returns
                    # Convert to array/Series if needed
                    if hasattr(losing_returns, 'values'):
                        losing_returns = losing_returns.values
                    elif hasattr(losing_returns, 'to_numpy'):
                        losing_returns = losing_returns.to_numpy()
                    
                    if isinstance(losing_returns, pd.Series) and len(losing_returns) > 0:
                        avg_loss = _f(losing_returns.mean())
                    elif hasattr(losing_returns, '__len__') and len(losing_returns) > 0:
                        # Handle numpy array or MappedArray
                        if isinstance(losing_returns, np.ndarray):
                            avg_loss = _f(float(np.mean(losing_returns)))
                        else:
                            # Fallback: convert to list and calculate mean
                            try:
                                returns_list = list(losing_returns) if hasattr(losing_returns, '__iter__') else [losing_returns]
                                avg_loss = _f(float(sum(returns_list) / len(returns_list)))
                            except Exception:
                                avg_loss = 0.0
                except Exception as e:
                    logging.warning(f"Error calculating avg_loss: {e}")
                    pass
            
            # Calculate profit factor: total profit from winners / total loss from losers
            try:
                # VectorBT: pnl is a property, not a method - don't call it!
                winning_pnl = pf.trades.winning.pnl
                losing_pnl = pf.trades.losing.pnl
                
                # Convert to array/Series if needed
                if hasattr(winning_pnl, 'values'):
                    winning_pnl = winning_pnl.values
                elif hasattr(winning_pnl, 'to_numpy'):
                    winning_pnl = winning_pnl.to_numpy()
                
                if hasattr(losing_pnl, 'values'):
                    losing_pnl = losing_pnl.values
                elif hasattr(losing_pnl, 'to_numpy'):
                    losing_pnl = losing_pnl.to_numpy()
                
                # Calculate sums - use module-level np
                if isinstance(winning_pnl, pd.Series):
                    winning_pnl_sum = _f(winning_pnl.sum())
                elif isinstance(winning_pnl, np.ndarray):
                    winning_pnl_sum = _f(float(np.sum(winning_pnl)))
                elif hasattr(winning_pnl, 'sum'):
                    winning_pnl_sum = _f(winning_pnl.sum())
                elif hasattr(winning_pnl, '__len__') and len(winning_pnl) > 0:
                    try:
                        pnl_list = list(winning_pnl) if hasattr(winning_pnl, '__iter__') else [winning_pnl]
                        winning_pnl_sum = _f(float(sum(pnl_list)))
                    except Exception:
                        winning_pnl_sum = _f(winning_pnl) if winning_trades > 0 else 0.0
                else:
                    winning_pnl_sum = _f(winning_pnl) if winning_trades > 0 else 0.0
                
                if isinstance(losing_pnl, pd.Series):
                    losing_pnl_sum = abs(_f(losing_pnl.sum()))
                elif isinstance(losing_pnl, np.ndarray):
                    losing_pnl_sum = abs(_f(float(np.sum(losing_pnl))))
                elif hasattr(losing_pnl, 'sum'):
                    losing_pnl_sum = abs(_f(losing_pnl.sum()))
                elif hasattr(losing_pnl, '__len__') and len(losing_pnl) > 0:
                    try:
                        pnl_list = list(losing_pnl) if hasattr(losing_pnl, '__iter__') else [losing_pnl]
                        losing_pnl_sum = abs(_f(float(sum(pnl_list))))
                    except Exception:
                        losing_pnl_sum = abs(_f(losing_pnl)) if losing_trades > 0 else 0.0
                else:
                    losing_pnl_sum = abs(_f(losing_pnl)) if losing_trades > 0 else 0.0
                
                if losing_pnl_sum != 0:
                    profit_factor = winning_pnl_sum / losing_pnl_sum
                elif winning_pnl_sum > 0:
                    # All trades are winners - use a very large number instead of inf for JSON compatibility
                    profit_factor = 999999.0  # JSON-safe representation of "infinite" profit factor
            except Exception as e:
                logging.warning(f"Error calculating profit_factor: {e}")
                pass
    except Exception as e:
        import logging
        logging.warning(f"Error calculating trade metrics: {e}")
        pass
    
    # Extract annualized return and volatility from stats
    annual_return = 0.0
    volatility = 0.0
    try:
        if 'Total Return [%]' in stats_dict:
            annual_return = _f(stats_dict['Total Return [%]'] / 100)
        elif hasattr(pf, 'annualized_return'):
            annual_return = _f(pf.annualized_return())
        
        if 'Volatility (Ann.) [%]' in stats_dict:
            volatility = _f(stats_dict['Volatility (Ann.) [%]'] / 100)
    except Exception:
        pass
    
    # Extract basic metrics with error handling
    try:
        total_return_val = _f(pf.total_return())
        total_profit_val = _f(pf.total_profit())
        sharpe_val = _f(pf.sharpe_ratio())
        if total_trades > 0:
            # Prefer stats_dict for win_rate (more reliable)
            if 'Win Rate [%]' in stats_dict:
                win_rate_val = _f(stats_dict['Win Rate [%]'])
            else:
                # Fallback to direct method
                win_rate_raw = pf.trades.win_rate()
                win_rate_val = _f(win_rate_raw)
                # VectorBT typically returns percentage (0-100), but check format
                if win_rate_val <= 1.0:
                    # It's a decimal, convert to percentage for consistency
                    win_rate_val = win_rate_val * 100.0
        else:
            win_rate_val = 0.0
    except Exception as metrics_error:
        import logging
        logging.warning(f"Failed to extract basic metrics: {metrics_error}")
        total_return_val = 0.0
        total_profit_val = 0.0
        sharpe_val = 0.0
        win_rate_val = 0.0
    
    def _json_safe(v):
        """Convert value to JSON-safe format (handle inf, nan, None)."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            if v != v:  # NaN
                return None
            if abs(v) == float("inf"):
                return None  # JSON-safe representation of infinity
        return v
    
    result = {
        "engine": "vectorbt",
        # Basic metrics
        "total_return": _json_safe(total_return_val),
        "total_profit": _json_safe(total_profit_val),
        "sharpe_ratio": _json_safe(sharpe_val),
        "max_drawdown": _json_safe(max_dd),
        "total_trades": total_trades,
        "win_rate": _json_safe(win_rate_val),
        
        # Additional ratios
        "sortino_ratio": _json_safe(_f(pf.sortino_ratio()) if hasattr(pf, 'sortino_ratio') else 0.0),
        "calmar_ratio": _json_safe(_f(pf.calmar_ratio()) if hasattr(pf, 'calmar_ratio') else 0.0),
        
        # Performance metrics
        "annual_return": _json_safe(annual_return),
        "volatility": _json_safe(volatility),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "avg_trade_return": _json_safe(_safe_get_avg_return(pf, total_trades, _f)),
        "best_trade": _json_safe(_safe_get_best_trade(pf, total_trades, _f)),
        "worst_trade": _json_safe(_safe_get_worst_trade(pf, total_trades, _f)),
        "avg_win": _json_safe(avg_win),
        "avg_loss": _json_safe(avg_loss),
        "profit_factor": _json_safe(profit_factor),
        "expectancy": _json_safe(_safe_get_expectancy(pf, total_trades, _f)),
        
        # Equity and drawdowns
        "equity_curve": equity_series[:1000] if len(equity_series) > 1000 else equity_series,  # Limit to 1000 points
        "drawdowns": drawdown_series[:1000] if len(drawdown_series) > 1000 else drawdown_series,
        "final_value": _json_safe(_f(equity_series[-1]) if len(equity_series) > 0 else initial_capital),
        
        # Trades data (limited to prevent huge JSON)
        "trades": trades_data,
        "trades_count": total_trades,
        
        # Full stats dictionary (converted to JSON-serializable format)
        "stats": {str(k): _json_safe(_f(v) if isinstance(v, (int, float, np.number)) else str(v)) for k, v in stats_dict.items()},
    }
    
    return result


def _backtest_backtesting_py(
    df: pd.DataFrame, signals: pd.Series, initial_capital: float
) -> dict:
    """Backtesting.py prototype - uses Strategy with external signals."""
    try:
        from backtesting import Backtest, Strategy
        from backtesting.lib import crossover
    except ImportError:
        return {"error": "backtesting not installed", "engine": "backtesting"}

    # Align signals with df and convert column names to capitalized format required by backtesting.py
    bt_df = df[["open", "high", "low", "close", "volume"]].dropna().copy()
    
    # Rename columns to capitalized format required by backtesting.py
    bt_df = bt_df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    
    sig = signals.reindex(bt_df.index).ffill().fillna(0)

    class SignalStrategy(Strategy):
        def init(self):
            self.signal = self.I(lambda: sig, name="signal")

        def next(self):
            if self.signal[-1] == 1 and not self.position:
                self.buy()
            elif self.signal[-1] == -1 and self.position:
                self.position.close()

    bt = Backtest(bt_df, SignalStrategy, cash=initial_capital, trade_on_close=True)
    stats = bt.run()

    def _f(v):
        """Convert value to float, handling numpy/pandas types."""
        if v is None:
            return 0.0
        try:
            # Handle numpy/pandas scalar types
            if hasattr(v, "item"):
                x = float(v.item())
            elif isinstance(v, (pd.Series, np.ndarray)):
                x = float(v.iloc[0] if len(v) > 0 else 0)
            else:
                x = float(v)
            # Check for NaN or Inf
            if x != x or abs(x) == float("inf"):
                return 0.0
            return x
        except (ValueError, TypeError, IndexError):
            return 0.0

    def _i(v):
        """Convert value to int, handling numpy/pandas types."""
        if v is None:
            return 0
        try:
            # Handle numpy/pandas scalar types
            if hasattr(v, "item"):
                return int(v.item())
            elif isinstance(v, (pd.Series, np.ndarray)):
                return int(v.iloc[0] if len(v) > 0 else 0)
            else:
                return int(v)
        except (ValueError, TypeError, IndexError):
            return 0

    # Safely extract values from stats, handling missing keys and non-serializable types
    # stats from backtesting.py is a pandas Series, convert to dict for easier access
    try:
        if isinstance(stats, pd.Series):
            stats_dict = stats.to_dict()
        elif hasattr(stats, '__dict__'):
            stats_dict = stats.__dict__
        else:
            stats_dict = dict(stats) if hasattr(stats, '__iter__') and not isinstance(stats, str) else {}
    except Exception:
        stats_dict = {}

    total_return = 0.0
    sharpe_ratio = 0.0
    max_drawdown = 0.0
    total_trades = 0
    win_rate = 0.0

    try:
        if "Return [%]" in stats_dict:
            val = stats_dict["Return [%]"]
            total_return = _f(val / 100) if val is not None else 0.0
    except Exception:
        pass

    try:
        val = stats_dict.get("Sharpe Ratio", 0)
        sharpe_ratio = _f(val) if val is not None else 0.0
    except Exception:
        pass

    try:
        if "Max. Drawdown [%]" in stats_dict:
            val = stats_dict["Max. Drawdown [%]"]
            max_drawdown = _f(val / 100) if val is not None else 0.0
    except Exception:
        pass

    try:
        val = stats_dict.get("# Trades", 0)
        total_trades = _i(val) if val is not None else 0
    except Exception:
        pass

    try:
        if total_trades > 0 and "Win Rate [%]" in stats_dict:
            val = stats_dict["Win Rate [%]"]
            win_rate = _f(val / 100) if val is not None else 0.0
    except Exception:
        pass

    return {
        "engine": "backtesting",
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "win_rate": win_rate,
    }
