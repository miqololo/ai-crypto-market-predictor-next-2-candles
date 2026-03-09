import { useState } from 'react'
import { BarChart3, Loader2, Play, TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react'

const API_BASE = '/api'

interface BacktestResult {
  engine: string
  total_return: number | null
  total_profit?: number | null
  sharpe_ratio: number | null
  sortino_ratio?: number | null
  calmar_ratio?: number | null
  max_drawdown: number | null
  total_trades: number
  win_rate?: number | null
  annual_return?: number | null
  volatility?: number | null
  winning_trades?: number
  losing_trades?: number
  avg_trade_return?: number | null
  best_trade?: number | null
  worst_trade?: number | null
  avg_win?: number | null
  avg_loss?: number | null
  profit_factor?: number | null
  expectancy?: number | null
  equity_curve?: number[]
  drawdowns?: number[]
  final_value?: number | null
  trades?: Array<{
    entry_idx: number
    exit_idx: number
    pnl: number
    return: number
    duration: number
    entry_price: number
    exit_price: number
  }>
  trades_count?: number
  stats?: Record<string, number | string>
  strategy_name?: string
  strategy_file?: string
}

interface StrategyParams {
  lookback_period?: number
  atr_threshold_pct?: number
  volume_multiplier?: number
  stop_loss_pct?: number
  take_profit_multiplier?: number
  rsi_overbought?: number
  rsi_oversold?: number
  avoid_first_minutes?: number
  exit_before_close_minutes?: number
  use_trailing_stop?: boolean
  trailing_stop_pct?: number
}

export default function StrategyBacktest() {
  const [symbol] = useState('BTC/USDT:USDT')
  const [timeframe, setTimeframe] = useState('1h')
  const [limit, setLimit] = useState(500)
  const [initialCapital, setInitialCapital] = useState(10000)
  const [strategyFile, setStrategyFile] = useState('app/strategies/breakout_strategy.py')
  const [stopLoss, setStopLoss] = useState(1.0) // Stop loss in percentage (default 1%)
  const [takeProfit, setTakeProfit] = useState(3.0) // Take profit in percentage (default 3% for 1:3 ratio)
  
  const [strategyParams, setStrategyParams] = useState<StrategyParams>({
    lookback_period: 20,
    atr_threshold_pct: 2.0,
    volume_multiplier: 1.5,
    stop_loss_pct: 0.75,
    take_profit_multiplier: 2.5,
    rsi_overbought: 70.0,
    rsi_oversold: 30.0,
    avoid_first_minutes: 30,
    exit_before_close_minutes: 30,
    use_trailing_stop: false,
    trailing_stop_pct: 1.0,
  })
  
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BacktestResult | null>(null)

  const updateParam = (key: keyof StrategyParams, value: number | boolean) => {
    setStrategyParams(prev => ({ ...prev, [key]: value }))
  }

  const handleRunBacktest = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Validate inputs before sending
      if (!strategyFile || strategyFile.trim() === '') {
        throw new Error('Strategy file is required')
      }
      if (limit < 10 || limit > 10000) {
        throw new Error('Limit must be between 10 and 10000')
      }
      if (initialCapital < 100) {
        throw new Error('Initial capital must be at least 100')
      }
      if (stopLoss <= 0 || stopLoss > 100) {
        throw new Error('Stop loss must be between 0.1% and 100%')
      }
      if (takeProfit <= 0 || takeProfit > 100) {
        throw new Error('Take profit must be between 0.1% and 100%')
      }

      // Prepare request body with all required fields
      const requestBody = {
        symbol: symbol || 'BTC/USDT:USDT',
        timeframe: timeframe || '1h',
        limit: Number(limit),
        engine: 'vectorbt' as const,
        initial_capital: Number(initialCapital),
        strategy_file: strategyFile.trim(),
        strategy_params: strategyParams || {},
        stop_loss: Number((stopLoss / 100).toFixed(4)), // Convert percentage to fraction (1% -> 0.01)
        take_profit: Number((takeProfit / 100).toFixed(4)), // Convert percentage to fraction (3% -> 0.03)
        use_database: true, // Always use MongoDB database
      }

      console.log('Sending backtest request:', requestBody)

      const response = await fetch(`${API_BASE}/backtest/strategy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || `HTTP ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full">
      <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        <div className="flex items-center gap-3 mb-6">
          <BarChart3 className="w-8 h-8 text-emerald-500" />
          <h1 className="text-2xl font-semibold">Strategy Backtest</h1>
        </div>

        {error && (
          <div className="rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Configuration */}
          <div className="lg:col-span-2 space-y-6">
            {/* Basic Settings */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
              <h2 className="text-lg font-medium text-zinc-200 mb-4">Backtest Settings</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Timeframe</label>
                  <select
                    value={timeframe}
                    onChange={(e) => setTimeframe(e.target.value)}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                  >
                    <option value="5m">5m</option>
                    <option value="15m">15m</option>
                    <option value="1h">1h</option>
                    <option value="4h">4h</option>
                    <option value="1d">1d</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Candles Limit</label>
                  <input
                    type="number"
                    value={limit}
                    onChange={(e) => setLimit(Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    min="100"
                    max="5000"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Strategy File</label>
                  <input
                    type="text"
                    value={strategyFile}
                    onChange={(e) => setStrategyFile(e.target.value)}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none font-mono text-xs"
                    placeholder="app/strategies/breakout_strategy.py"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Initial Capital</label>
                  <input
                    type="number"
                    value={initialCapital}
                    onChange={(e) => setInitialCapital(Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    min="1000"
                    step="1000"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    Stop Loss (%) <span className="text-zinc-600">(default: 1%)</span>
                  </label>
                  <input
                    type="number"
                    value={stopLoss}
                    onChange={(e) => setStopLoss(Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    min="0.1"
                    max="50"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    Take Profit (%) <span className="text-zinc-600">(default: 3%, 1:3 ratio)</span>
                  </label>
                  <input
                    type="number"
                    value={takeProfit}
                    onChange={(e) => setTakeProfit(Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    min="0.1"
                    max="100"
                    step="0.1"
                  />
                </div>
              </div>
            </div>

            {/* Strategy Parameters */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
              <h2 className="text-lg font-medium text-zinc-200 mb-4">Strategy Parameters</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    Lookback Period <span className="text-zinc-600">(default: 20)</span>
                  </label>
                  <input
                    type="number"
                    value={strategyParams.lookback_period || ''}
                    onChange={(e) => updateParam('lookback_period', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="20"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    ATR Threshold % <span className="text-zinc-600">(default: 2.0)</span>
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={strategyParams.atr_threshold_pct || ''}
                    onChange={(e) => updateParam('atr_threshold_pct', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="2.0"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    Volume Multiplier <span className="text-zinc-600">(default: 1.5)</span>
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={strategyParams.volume_multiplier || ''}
                    onChange={(e) => updateParam('volume_multiplier', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="1.5"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    Stop Loss % <span className="text-zinc-600">(default: 0.75)</span>
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={strategyParams.stop_loss_pct || ''}
                    onChange={(e) => updateParam('stop_loss_pct', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="0.75"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    Take Profit Multiplier <span className="text-zinc-600">(default: 2.5)</span>
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={strategyParams.take_profit_multiplier || ''}
                    onChange={(e) => updateParam('take_profit_multiplier', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="2.5"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    RSI Overbought <span className="text-zinc-600">(default: 70)</span>
                  </label>
                  <input
                    type="number"
                    step="1"
                    value={strategyParams.rsi_overbought || ''}
                    onChange={(e) => updateParam('rsi_overbought', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="70"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    RSI Oversold <span className="text-zinc-600">(default: 30)</span>
                  </label>
                  <input
                    type="number"
                    step="1"
                    value={strategyParams.rsi_oversold || ''}
                    onChange={(e) => updateParam('rsi_oversold', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="30"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    Avoid First Minutes <span className="text-zinc-600">(default: 30)</span>
                  </label>
                  <input
                    type="number"
                    step="1"
                    value={strategyParams.avoid_first_minutes || ''}
                    onChange={(e) => updateParam('avoid_first_minutes', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="30"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    Exit Before Close (min) <span className="text-zinc-600">(default: 30)</span>
                  </label>
                  <input
                    type="number"
                    step="1"
                    value={strategyParams.exit_before_close_minutes || ''}
                    onChange={(e) => updateParam('exit_before_close_minutes', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="30"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">
                    Trailing Stop % <span className="text-zinc-600">(default: 1.0)</span>
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={strategyParams.trailing_stop_pct || ''}
                    onChange={(e) => updateParam('trailing_stop_pct', Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    placeholder="1.0"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="use_trailing_stop"
                    checked={strategyParams.use_trailing_stop || false}
                    onChange={(e) => updateParam('use_trailing_stop', e.target.checked)}
                    className="w-4 h-4 rounded bg-zinc-800 border-zinc-700 text-emerald-600 focus:ring-emerald-500"
                  />
                  <label htmlFor="use_trailing_stop" className="text-sm text-zinc-400">
                    Use Trailing Stop
                  </label>
                </div>
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunBacktest}
              disabled={loading}
              className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 px-6 py-3 text-sm font-medium text-white disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running Backtest...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Backtest
                </>
              )}
            </button>
          </div>

          {/* Right Column: Results */}
          <div className="space-y-6">
            {result && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
                <h2 className="text-lg font-medium text-zinc-200 mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-emerald-500" />
                  Backtest Results
                </h2>
                {result.strategy_name && (
                  <div className="mb-4 pb-4 border-b border-zinc-800">
                    <div className="text-xs text-zinc-500 mb-1">Strategy</div>
                    <div className="text-sm font-mono text-zinc-300">{result.strategy_name}</div>
                    {result.strategy_file && (
                      <div className="text-xs text-zinc-600 mt-1 font-mono">{result.strategy_file}</div>
                    )}
                  </div>
                )}
                {/* Key Metrics */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp className="w-4 h-4 text-emerald-400" />
                      <div className="text-xs text-zinc-500">Total Return</div>
                    </div>
                    <div className={`text-2xl font-bold ${result.total_return !== null && result.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {result.total_return !== null ? (result.total_return * 100).toFixed(2) + '%' : 'N/A'}
                    </div>
                    {result.total_profit !== undefined && result.total_profit !== null && (
                      <div className="text-xs text-zinc-500 mt-1">
                        ${result.total_profit.toFixed(2)}
                      </div>
                    )}
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <DollarSign className="w-4 h-4 text-blue-400" />
                      <div className="text-xs text-zinc-500">Sharpe Ratio</div>
                    </div>
                    <div className="text-2xl font-bold text-zinc-200">
                      {result.sharpe_ratio !== null ? result.sharpe_ratio.toFixed(2) : 'N/A'}
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingDown className="w-4 h-4 text-red-400" />
                      <div className="text-xs text-zinc-500">Max Drawdown</div>
                    </div>
                    <div className="text-2xl font-bold text-red-400">
                      {result.max_drawdown !== null ? (result.max_drawdown * 100).toFixed(2) + '%' : 'N/A'}
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <BarChart3 className="w-4 h-4 text-violet-400" />
                      <div className="text-xs text-zinc-500">Total Trades</div>
                    </div>
                    <div className="text-2xl font-bold text-zinc-200">
                      {result.total_trades}
                    </div>
                    {result.win_rate !== undefined && result.win_rate !== null && (
                      <div className="text-xs text-zinc-500 mt-1">
                        {(result.win_rate * 100).toFixed(1)}% win rate
                      </div>
                    )}
                  </div>
                </div>

                {/* Additional Performance Metrics */}
                {(result.sortino_ratio !== undefined || result.calmar_ratio !== undefined || result.profit_factor !== undefined) && (
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 mb-4">
                    {result.sortino_ratio !== undefined && result.sortino_ratio !== null && result.sortino_ratio !== 0 && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Sortino Ratio</div>
                        <div className="text-lg font-semibold text-zinc-200">{result.sortino_ratio.toFixed(2)}</div>
                      </div>
                    )}
                    {result.calmar_ratio !== undefined && result.calmar_ratio !== null && result.calmar_ratio !== 0 && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Calmar Ratio</div>
                        <div className="text-lg font-semibold text-zinc-200">{result.calmar_ratio.toFixed(2)}</div>
                      </div>
                    )}
                    {result.profit_factor !== undefined && result.profit_factor !== null && result.profit_factor !== 0 && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Profit Factor</div>
                        <div className="text-lg font-semibold text-zinc-200">
                          {result.profit_factor === 999999.0 ? '∞' : result.profit_factor.toFixed(2)}
                        </div>
                      </div>
                    )}
                    {result.annual_return !== undefined && result.annual_return !== null && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Annual Return</div>
                        <div className={`text-lg font-semibold ${result.annual_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {(result.annual_return * 100).toFixed(2)}%
                        </div>
                      </div>
                    )}
                    {result.volatility !== undefined && result.volatility !== null && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Volatility</div>
                        <div className="text-lg font-semibold text-zinc-200">{(result.volatility * 100).toFixed(2)}%</div>
                      </div>
                    )}
                    {result.expectancy !== undefined && result.expectancy !== null && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Expectancy</div>
                        <div className={`text-lg font-semibold ${result.expectancy >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {result.expectancy.toFixed(4)}
                        </div>
                      </div>
                    )}
                    {result.final_value !== undefined && result.final_value !== null && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Final Value</div>
                        <div className="text-lg font-semibold text-zinc-200">${result.final_value.toFixed(2)}</div>
                      </div>
                    )}
                  </div>
                )}

                {/* Trade Statistics */}
                {(result.winning_trades !== undefined || result.best_trade !== undefined) && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    {result.winning_trades !== undefined && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Winning Trades</div>
                        <div className="text-lg font-semibold text-emerald-400">{result.winning_trades}</div>
                      </div>
                    )}
                    {result.losing_trades !== undefined && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Losing Trades</div>
                        <div className="text-lg font-semibold text-red-400">{result.losing_trades}</div>
                      </div>
                    )}
                    {result.best_trade !== undefined && result.best_trade !== null && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Best Trade</div>
                        <div className={`text-lg font-semibold ${result.best_trade >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {(result.best_trade * 100).toFixed(2)}%
                        </div>
                      </div>
                    )}
                    {result.worst_trade !== undefined && result.worst_trade !== null && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Worst Trade</div>
                        <div className={`text-lg font-semibold ${result.worst_trade >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {(result.worst_trade * 100).toFixed(2)}%
                        </div>
                      </div>
                    )}
                    {result.avg_win !== undefined && result.avg_win !== null && result.avg_win !== 0 && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Avg Win</div>
                        <div className="text-lg font-semibold text-emerald-400">{(result.avg_win * 100).toFixed(2)}%</div>
                      </div>
                    )}
                    {result.avg_loss !== undefined && result.avg_loss !== null && result.avg_loss !== 0 && (
                      <div className="rounded-lg bg-zinc-800/30 p-3">
                        <div className="text-xs text-zinc-500">Avg Loss</div>
                        <div className="text-lg font-semibold text-red-400">{(result.avg_loss * 100).toFixed(2)}%</div>
                      </div>
                    )}
                  </div>
                )}

                {/* Equity Curve Info */}
                {result.equity_curve && result.equity_curve.length > 0 && (
                  <div className="mb-4 rounded-lg bg-zinc-800/30 p-3">
                    <div className="text-xs text-zinc-500 mb-2">Equity Curve</div>
                    <div className="text-xs text-zinc-400 font-mono">
                      {result.equity_curve.length} data points
                      {result.final_value !== undefined && result.final_value !== null && (
                        <span className="ml-2">
                          Final: ${result.final_value.toFixed(2)}
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Trades Table */}
                {result.trades && result.trades.length > 0 && (
                  <div className="mb-4 rounded-lg bg-zinc-800/30 p-3">
                    <div className="text-xs text-zinc-500 mb-2">
                      Recent Trades ({result.trades.length} of {result.total_trades})
                    </div>
                    <div className="max-h-48 overflow-y-auto">
                      <table className="w-full text-xs">
                        <thead className="text-zinc-500 border-b border-zinc-700">
                          <tr>
                            <th className="text-left py-1">Entry</th>
                            <th className="text-left py-1">Exit</th>
                            <th className="text-right py-1">PnL</th>
                            <th className="text-right py-1">Return</th>
                            <th className="text-right py-1">Duration</th>
                          </tr>
                        </thead>
                        <tbody className="text-zinc-300">
                          {result.trades.slice(0, 20).map((trade, idx) => (
                            <tr key={idx} className="border-b border-zinc-800/50">
                              <td className="py-1">${trade.entry_price.toFixed(2)}</td>
                              <td className="py-1">${trade.exit_price.toFixed(2)}</td>
                              <td className={`text-right py-1 ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                ${trade.pnl.toFixed(2)}
                              </td>
                              <td className={`text-right py-1 ${trade.return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {(trade.return * 100).toFixed(2)}%
                              </td>
                              <td className="text-right py-1 text-zinc-400">{trade.duration}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Full Stats */}
                {result.stats && Object.keys(result.stats).length > 0 && (
                  <details className="mb-4 rounded-lg bg-zinc-800/30 p-3">
                    <summary className="text-xs text-zinc-400 cursor-pointer hover:text-zinc-300">
                      View All Statistics ({Object.keys(result.stats).length} metrics)
                    </summary>
                    <div className="mt-2 grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                      {Object.entries(result.stats).map(([key, value]) => (
                        <div key={key} className="flex justify-between border-b border-zinc-800/50 pb-1">
                          <span className="text-zinc-500">{key}</span>
                          <span className="text-zinc-300 font-mono">{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}

                <div className="mt-4 pt-4 border-t border-zinc-800">
                  <div className="text-xs text-zinc-500">Engine</div>
                  <div className="text-sm text-zinc-400 font-mono">{result.engine}</div>
                </div>
              </div>
            )}

            {!result && !loading && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
                <div className="text-center text-zinc-500 text-sm">
                  Configure settings and run backtest to see results here
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
