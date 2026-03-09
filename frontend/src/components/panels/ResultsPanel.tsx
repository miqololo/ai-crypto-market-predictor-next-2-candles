"use client"

import { useState, useEffect } from "react"
import { cn } from "../../lib/utils"

interface BacktestMetrics {
  total_return: number | null
  sharpe_ratio: number | null
  max_drawdown: number | null
  win_rate: number | null
  num_trades: number
  total_profit?: number | null
  sortino_ratio?: number | null
  calmar_ratio?: number | null
  profit_factor?: number | null
  final_value?: number | null
}

interface Suggestion {
  title: string
  description: string
  priority: string
  suggested_change?: string
}

interface ResultsPanelProps {
  compact?: boolean
  metrics?: BacktestMetrics | null
}

export function ResultsPanel({ compact = false, metrics: externalMetrics = null }: ResultsPanelProps) {
  const [metrics, setMetrics] = useState<BacktestMetrics | null>(externalMetrics)
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])

  useEffect(() => {
    if (externalMetrics) {
      setMetrics(externalMetrics)
    }
  }, [externalMetrics])

  if (!metrics) {
    return (
      <div className={cn("rounded-lg border border-zinc-800 bg-zinc-900/50", compact ? "p-2" : "p-4")}>
        <h2 className={cn("font-semibold text-zinc-200", compact ? "text-sm" : "text-lg")}>Results</h2>
        <p className="mt-1 text-sm text-zinc-500">
          Run a backtest to see metrics.
        </p>
      </div>
    )
  }

  return (
    <div className={cn("rounded-lg border border-zinc-800 bg-zinc-900/50", compact ? "p-2" : "p-4")}>
      <h2 className={cn("font-semibold text-zinc-200", compact ? "mb-2 text-sm" : "mb-3 text-lg")}>Backtest Results</h2>

      <div className={cn("grid grid-cols-2 gap-1.5", compact ? "mb-2" : "mb-4 sm:grid-cols-3")}>
        <MetricCard label="Total Return" value={metrics.total_return !== null ? `${(metrics.total_return * 100).toFixed(2)}%` : 'N/A'} compact={compact} />
        {metrics.total_profit !== undefined && metrics.total_profit !== null && (
          <MetricCard label="Total Profit" value={`$${metrics.total_profit.toFixed(2)}`} compact={compact} />
        )}
        <MetricCard label="Sharpe" value={metrics.sharpe_ratio !== null ? metrics.sharpe_ratio.toFixed(2) : 'N/A'} compact={compact} />
        {metrics.sortino_ratio !== undefined && metrics.sortino_ratio !== null && metrics.sortino_ratio !== 0 && (
          <MetricCard label="Sortino" value={metrics.sortino_ratio.toFixed(2)} compact={compact} />
        )}
        {metrics.calmar_ratio !== undefined && metrics.calmar_ratio !== null && metrics.calmar_ratio !== 0 && (
          <MetricCard label="Calmar" value={metrics.calmar_ratio.toFixed(2)} compact={compact} />
        )}
        <MetricCard label="Drawdown" value={metrics.max_drawdown !== null ? `${(metrics.max_drawdown * 100).toFixed(2)}%` : 'N/A'} compact={compact} />
        <MetricCard label="Win Rate" value={metrics.win_rate !== null ? `${(metrics.win_rate * 100).toFixed(1)}%` : 'N/A'} compact={compact} />
        <MetricCard label="Trades" value={String(metrics.num_trades)} compact={compact} />
        {metrics.profit_factor !== undefined && metrics.profit_factor !== null && metrics.profit_factor !== 0 && (
          <MetricCard label="Profit Factor" value={metrics.profit_factor === 999999.0 ? '∞' : metrics.profit_factor.toFixed(2)} compact={compact} />
        )}
        {metrics.final_value !== undefined && metrics.final_value !== null && (
          <MetricCard label="Final Value" value={`$${metrics.final_value.toFixed(0)}`} compact={compact} />
        )}
      </div>
    </div>
  )
}

function MetricCard({ label, value, compact }: { label: string; value: string; compact?: boolean }) {
  return (
    <div className={cn("rounded border border-zinc-800 bg-zinc-800/30", compact ? "p-1.5" : "p-2")}>
      <p className={cn("text-zinc-500", compact ? "text-[10px]" : "text-xs")}>{label}</p>
      <p className={cn("font-mono font-semibold text-zinc-200", compact ? "text-xs" : "text-sm")}>{value}</p>
    </div>
  )
}
