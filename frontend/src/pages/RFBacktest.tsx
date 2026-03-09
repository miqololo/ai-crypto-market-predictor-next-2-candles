import { useState, useCallback } from 'react'
import { BarChart3, Loader2, Play, TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react'

const API_BASE = '/api'

function formatDate(s: string | undefined): string {
  if (!s) return '—'
  const d = new Date(s)
  if (isNaN(d.getTime())) return s
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })
}

interface WalkForwardResult {
  ok: boolean
  total_pnl?: number
  total_return?: number
  final_capital?: number
  initial_capital?: number
  n_trades?: number
  win_rate?: number
  profit_factor?: number
  max_drawdown_pct?: number
  avg_win?: number
  avg_loss?: number
  expectancy?: number
  expectancy_pct?: number
  winning_trades?: number
  losing_trades?: number
  equity_curve?: number[]
  equity_timestamps?: string[]
  trades?: Array<{
    entry_idx: number
    exit_idx: number
    entry_date?: string
    exit_date?: string
    entry_price: number
    exit_price: number
    position: number
    pnl: number
    return_pct: number
    exit_reason: string
    actual_low?: number
    actual_high?: number
    predicted_low?: number
    predicted_high?: number
    low_accuracy_pct?: number | null
    high_accuracy_pct?: number | null
    confidence_pct?: number
  }>
  params?: {
    date_from: string
    date_to: string
    symbol: string
    timeframe: string
    threshold: number
  }
}

export default function RFBacktest() {
  const [dateFrom, setDateFrom] = useState('2025-03-01')
  const [dateTo, setDateTo] = useState('2025-04-01')
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [timeframe, setTimeframe] = useState('1h')
  const [threshold, setThreshold] = useState('0.6')
  const [initialCapital, setInitialCapital] = useState(50000)
  const [riskPercent, setRiskPercent] = useState(1)
  const [slippagePercent, setSlippagePercent] = useState(0.03)
  const [commissionPercent, setCommissionPercent] = useState(0.1)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<WalkForwardResult | null>(null)

  const handleRun = useCallback(async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/rf/walk-forward-backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          date_from: dateFrom,
          date_to: dateTo,
          symbol,
          timeframe,
          threshold: parseFloat(threshold),
          risk_percent: riskPercent,
          slippage_percent: slippagePercent,
          commission_percent: commissionPercent,
          initial_capital: initialCapital,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [dateFrom, dateTo, symbol, timeframe, threshold, initialCapital, riskPercent, slippagePercent, commissionPercent])

  // SVG equity curve
  const equityCurveSvg = () => {
    if (!result?.equity_curve || result.equity_curve.length < 2) return null
    const curve = result.equity_curve
    const min = Math.min(...curve)
    const max = Math.max(...curve)
    const range = max - min || 1
    const w = 800
    const h = 300
    const pad = { top: 20, right: 20, bottom: 30, left: 60 }
    const xScale = (i: number) => pad.left + (i / (curve.length - 1)) * (w - pad.left - pad.right)
    const yScale = (v: number) => pad.top + h - pad.top - pad.bottom - ((v - min) / range) * (h - pad.top - pad.bottom)
    const pathD = curve.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(v)}`).join(' ')
    return (
      <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="rounded-lg bg-zinc-900/50">
        <path d={pathD} fill="none" stroke="rgb(52, 211, 153)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <line x1={pad.left} y1={h - pad.bottom} x2={w - pad.right} y2={h - pad.bottom} stroke="rgb(82, 82, 91)" strokeWidth="1" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={h - pad.bottom} stroke="rgb(82, 82, 91)" strokeWidth="1" />
        <text x={pad.left - 5} y={pad.top + 4} fill="rgb(161, 161, 170)" fontSize="10" textAnchor="end">{max.toFixed(0)}</text>
        <text x={pad.left - 5} y={h - pad.bottom + 4} fill="rgb(161, 161, 170)" fontSize="10" textAnchor="end">{min.toFixed(0)}</text>
      </svg>
    )
  }

  return (
    <div className="min-h-screen w-full">
      <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        <div className="flex items-center gap-3 mb-6">
          <BarChart3 className="w-8 h-8 text-emerald-500" />
          <h1 className="text-2xl font-semibold">Walk-Forward Backtest</h1>
        </div>
        <p className="text-sm text-zinc-500">
          Simulates real trading with no lookahead. Predictions use only historical data up to each bar.
        </p>

        {error && (
          <div className="rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-1 gap-6">
          <div className="space-y-6">
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
              <h2 className="text-lg font-medium text-zinc-200 mb-4">Backtest Settings</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Date From</label>
                  <input
                    type="date"
                    value={dateFrom}
                    onChange={(e) => setDateFrom(e.target.value)}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Date To</label>
                  <input
                    type="date"
                    value={dateTo}
                    onChange={(e) => setDateTo(e.target.value)}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Symbol</label>
                  <input
                    type="text"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value)}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                  />
                </div>
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
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Threshold</label>
                  <input
                    type="number"
                    value={threshold}
                    onChange={(e) => setThreshold(e.target.value)}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    min="0.5"
                    max="0.95"
                    step="0.05"
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
                  <label className="block text-sm text-zinc-500 mb-2">Risk %</label>
                  <input
                    type="number"
                    value={riskPercent}
                    onChange={(e) => setRiskPercent(Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    min="0.1"
                    max="10"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Slippage %</label>
                  <input
                    type="number"
                    value={slippagePercent}
                    onChange={(e) => setSlippagePercent(Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    min="0"
                    max="1"
                    step="0.01"
                  />
                </div>
                <div>
                  <label className="block text-sm text-zinc-500 mb-2">Commission %</label>
                  <input
                    type="number"
                    value={commissionPercent}
                    onChange={(e) => setCommissionPercent(Number(e.target.value))}
                    className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
                    min="0"
                    max="1"
                    step="0.01"
                  />
                </div>
              </div>
            </div>

            <button
              onClick={handleRun}
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
                  Run Walk-Forward Backtest
                </>
              )}
            </button>
          </div>

          <div className="space-y-6">
            {result && result.ok && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
                <h2 className="text-lg font-medium text-zinc-200 mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-emerald-500" />
                  Results
                </h2>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp className="w-4 h-4 text-emerald-400" />
                      <div className="text-xs text-zinc-500">Total PnL</div>
                    </div>
                    <div className={`text-2xl font-bold ${(result.total_pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      ${result.total_pnl?.toFixed(2) ?? '0'}
                    </div>
                    <div className="text-xs text-zinc-500 mt-1">
                      {((result.total_return ?? 0) * 100).toFixed(2)}% return
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <DollarSign className="w-4 h-4 text-blue-400" />
                      <div className="text-xs text-zinc-500">Final Capital</div>
                    </div>
                    <div className="text-2xl font-bold text-zinc-200">
                      ${result.final_capital?.toFixed(2) ?? '0'}
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <BarChart3 className="w-4 h-4 text-violet-400" />
                      <div className="text-xs text-zinc-500">Trades</div>
                    </div>
                    <div className="text-2xl font-bold text-zinc-200">{result.n_trades ?? 0}</div>
                    <div className="text-xs text-zinc-500 mt-1">
                      {((result.win_rate ?? 0) * 100).toFixed(1)}% win rate
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingDown className="w-4 h-4 text-red-400" />
                      <div className="text-xs text-zinc-500">Max Drawdown</div>
                    </div>
                    <div className="text-2xl font-bold text-red-400">
                      {result.max_drawdown_pct?.toFixed(2) ?? '0'}%
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="text-xs text-zinc-500">Profit Factor</div>
                    <div className="text-lg font-semibold text-zinc-200">
                      {result.profit_factor === 999999 ? '∞' : result.profit_factor?.toFixed(2) ?? '0'}
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="text-xs text-zinc-500">Expectancy</div>
                    <div className={`text-lg font-semibold ${(result.expectancy ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      ${result.expectancy?.toFixed(2) ?? '0'}
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="text-xs text-zinc-500">Avg Win</div>
                    <div className="text-lg font-semibold text-emerald-400">
                      ${result.avg_win?.toFixed(2) ?? '0'}
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 p-4">
                    <div className="text-xs text-zinc-500">Avg Loss</div>
                    <div className="text-lg font-semibold text-red-400">
                      ${result.avg_loss?.toFixed(2) ?? '0'}
                    </div>
                  </div>
                </div>

                {result.equity_curve && result.equity_curve.length > 1 && (
                  <div className="mt-4">
                    <div className="text-sm text-zinc-500 mb-2">Equity Curve</div>
                    <div className="w-full overflow-x-auto">
                      {equityCurveSvg()}
                    </div>
                  </div>
                )}

                {result.trades && result.trades.length > 0 && (
                  <div className="mt-4 rounded-lg bg-zinc-800/30 p-3 max-h-64 overflow-y-auto">
                    <div className="text-sm text-zinc-500 mb-2">Trades ({result.trades.length})</div>
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-zinc-500">
                          <th className="text-left py-1">#</th>
                          <th className="text-left py-1">Dir</th>
                          <th className="text-left py-1">Entry</th>
                          <th className="text-left py-1">Exit</th>
                          <th className="text-right py-1">Entry $</th>
                          <th className="text-right py-1">Exit $</th>
                          <th className="text-right py-1">PnL</th>
                          <th className="text-right py-1" title="Model confidence 0–100%">Conf%</th>
                          <th className="text-right py-1" title="Actual / Pred low">Low</th>
                          <th className="text-right py-1" title="Actual / Pred high">High</th>
                          <th className="text-right py-1" title="Low/High accuracy % vs prev candle">Acc%</th>
                          <th className="text-left py-1">Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.trades.map((t, i) => (
                          <tr key={i} className="border-t border-zinc-700/50">
                            <td className="py-1">{i + 1}</td>
                            <td className={t.position === 1 ? 'text-emerald-400' : 'text-red-400'}>
                              {t.position === 1 ? 'L' : 'S'}
                            </td>
                            <td className="text-zinc-400 text-xs">{formatDate(t.entry_date)}</td>
                            <td className="text-zinc-400 text-xs">{formatDate(t.exit_date)}</td>
                            <td className="text-right">{t.entry_price.toFixed(2)}</td>
                            <td className="text-right">{t.exit_price.toFixed(2)}</td>
                            <td className={`text-right ${t.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              ${t.pnl.toFixed(2)}
                            </td>
                            <td className="text-right text-cyan-400 text-xs">{t.confidence_pct != null ? `${t.confidence_pct}%` : '—'}</td>
                            <td className="text-right text-zinc-400 text-xs" title={`Actual: ${t.actual_low?.toFixed(2) ?? '—'} / Pred: ${t.predicted_low?.toFixed(2) ?? '—'}`}>
                              {t.actual_low != null && t.predicted_low != null
                                ? `${t.actual_low.toFixed(0)} / ${t.predicted_low.toFixed(0)}`
                                : '—'}
                            </td>
                            <td className="text-right text-zinc-400 text-xs" title={`Actual: ${t.actual_high?.toFixed(2) ?? '—'} / Pred: ${t.predicted_high?.toFixed(2) ?? '—'}`}>
                              {t.actual_high != null && t.predicted_high != null
                                ? `${t.actual_high.toFixed(0)} / ${t.predicted_high.toFixed(0)}`
                                : '—'}
                            </td>
                            <td className="text-right text-xs" title={`Low: ${t.low_accuracy_pct ?? '—'}% / High: ${t.high_accuracy_pct ?? '—'}%`}>
                              {t.low_accuracy_pct != null && t.high_accuracy_pct != null
                                ? `${t.low_accuracy_pct} / ${t.high_accuracy_pct}`
                                : '—'}
                            </td>
                            <td className="text-zinc-500">{t.exit_reason}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
