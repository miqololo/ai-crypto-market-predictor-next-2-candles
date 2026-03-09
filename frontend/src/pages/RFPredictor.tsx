import { useState, useCallback, useEffect } from 'react'
import { Loader2, Play, BarChart3, Target, GitBranch, RefreshCw, CheckCircle, XCircle, Trash2, Eye } from 'lucide-react'

const API_BASE = '/api'

function formatDate(s: string | undefined): string {
  if (!s) return '—'
  const d = new Date(s)
  if (isNaN(d.getTime())) return s
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })
}

async function fetchApi(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

const TIMEFRAMES = ['5m', '15m', '1h'] as const
type Timeframe = (typeof TIMEFRAMES)[number]

interface TrainResult {
  ok: boolean
  metrics?: {
    accuracy: number
    precision: number
    recall: number
    f1: number
    roc_auc: number
    roc_auc_train?: number
    roc_auc_cv?: number
    roc_auc_cv_std?: number
    cv_mean_accuracy?: number
    cv_n_folds?: number
    auc_gap?: number
    overfitting_warning?: boolean
    threshold_optimize_by?: string
    train_samples: number
    test_samples: number
    n_features: number
    class_balance_pct_up?: number
    class_balance_n_up?: number
    class_balance_n_down?: number
    optimal_threshold?: number
    model_type?: string
    filtered_accuracy?: number
    filtered_precision?: number
    filtered_f1?: number
    filtered_signals?: number
    filtered_pct?: number
    filter_confidence_min?: number
  }
  top_features?: Array<{ name: string; importance: number }>
  feature_names?: string[]
}

interface EvalResult {
  ok: boolean
  metrics?: {
    accuracy: number
    precision: number
    recall: number
    f1: number
    roc_auc: number
    confusion_matrix?: { tn: number; fp: number; fn: number; tp: number }
    test_samples: number
  }
}

interface BacktestResult {
  ok: boolean
  backtest?: {
    total_pnl: number
    n_trades: number
    win_rate: number
    proba_threshold: number
  }
}

interface PredictResult {
  ok: boolean
  timestamp?: string
  close?: number
  direction?: number
  proba_up?: number
  predicted_low?: number | null
  predicted_high?: number | null
}

interface PredictionsRangeResult {
  ok: boolean
  predictions: Array<{
    timestamp: string
    close: number
    direction: number
    proba_up: number
    confidence_pct: number
    predicted_low: number | null
    predicted_high: number | null
    actual_low: number | null
    actual_high: number | null
    low_accuracy_pct: number | null
    high_accuracy_pct: number | null
    actual_direction: number | null
    correct: boolean | null
  }>
  n: number
  n_correct?: number
  n_incorrect?: number
  total_accuracy?: number | null
}

interface CVResult {
  ok: boolean
  folds?: Array<{
    fold: number
    accuracy: number
    roc_auc: number
    train_size: number
    test_size: number
  }>
}

interface FeaturesPreview {
  feature_names: string[]
  sample_values: Record<string, number | null>
  n_rows: number
  all_valid: boolean
}

export default function RFPredictor() {
  const [timeframe, setTimeframe] = useState<Timeframe>('1h')
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [limit, setLimit] = useState<string>('')
  const [trainStart, setTrainStart] = useState<string>('2022-01-01')
  const [trainEnd, setTrainEnd] = useState<string>('2024-01-01')
  const [testStart, setTestStart] = useState<string>('2024-01-01')
  const [testEnd, setTestEnd] = useState<string>('2025-01-01')
  const [threshold, setThreshold] = useState('0.6')
  const [nSplits, setNSplits] = useState('5')
  const [targetSymmetric, setTargetSymmetric] = useState(true)
  const [useCalibration, setUseCalibration] = useState(true)
  const [tuneThreshold, setTuneThreshold] = useState(true)
  const [modelType, setModelType] = useState('lgb')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<{ trained: boolean } | null>(null)
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null)
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null)
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null)
  const [cvResult, setCVResult] = useState<CVResult | null>(null)
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null)
  const [predictionsRange, setPredictionsRange] = useState<PredictionsRangeResult | null>(null)
  const [predictionsSort, setPredictionsSort] = useState<{ key: string; asc: boolean }>({ key: 'timestamp', asc: true })
  const [featuresPreview, setFeaturesPreview] = useState<FeaturesPreview | null>(null)

  const loadStatus = useCallback(async () => {
    try {
      const data = await fetchApi('/rf/status')
      setStatus(data)
    } catch {
      setStatus({ trained: false })
    }
  }, [])

  useEffect(() => {
    loadStatus()
  }, [loadStatus])

  const handleReset = useCallback(async () => {
    if (!confirm('Delete the trained RF model? You will need to train again.')) return
    setLoading(true)
    setError(null)
    try {
      await fetchApi('/rf/reset', { method: 'POST' })
      setTrainResult(null)
      setEvalResult(null)
      setBacktestResult(null)
      setPredictResult(null)
      setPredictionsRange(null)
      setFeaturesPreview(null)
      loadStatus()
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [loadStatus])

  const handleTrain = useCallback(async () => {
    setLoading(true)
    setError(null)
    setTrainResult(null)
    try {
      const data = await fetchApi('/rf/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timeframe,
          symbol,
          limit: limit ? parseInt(limit, 10) : null,
          train_start: trainStart || null,
          train_end: trainEnd || null,
          test_start: testStart || null,
          test_end: testEnd || null,
          target_symmetric: targetSymmetric,
          use_calibration: useCalibration,
          tune_threshold: tuneThreshold,
          model_type: modelType,
        }),
      })
      setTrainResult(data)
      loadStatus()
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [timeframe, symbol, limit, trainStart, trainEnd, testStart, testEnd, targetSymmetric, useCalibration, tuneThreshold, modelType, loadStatus])

  const handleEval = useCallback(async () => {
    setLoading(true)
    setError(null)
    setEvalResult(null)
    try {
      const data = await fetchApi('/rf/eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timeframe,
          symbol,
          limit: limit ? parseInt(limit, 10) : null,
        }),
      })
      setEvalResult(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [timeframe, symbol, limit])

  const handleBacktest = useCallback(async () => {
    setLoading(true)
    setError(null)
    setBacktestResult(null)
    try {
      const params = new URLSearchParams({
        timeframe,
        symbol,
        threshold,
      })
      if (limit) params.set('limit', limit)
      if (testStart) params.set('test_start', testStart)
      if (testEnd) params.set('test_end', testEnd)
      const data = await fetchApi(`/rf/backtest?${params}`, { method: 'POST' })
      setBacktestResult(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [timeframe, symbol, limit, threshold, testStart, testEnd])

  const handleFeaturesPreview = useCallback(async () => {
    setLoading(true)
    setError(null)
    setFeaturesPreview(null)
    try {
      const data = await fetchApi('/rf/features-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timeframe,
          symbol,
          limit: limit ? parseInt(limit, 10) : null,
          train_start: trainStart || null,
          train_end: trainEnd || null,
          test_start: testStart || null,
          test_end: testEnd || null,
          target_symmetric: targetSymmetric,
        }),
      })
      setFeaturesPreview(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [timeframe, symbol, limit, trainStart, trainEnd, testStart, testEnd, targetSymmetric])

  const handlePredict = useCallback(async () => {
    setLoading(true)
    setError(null)
    setPredictResult(null)
    try {
      const data = await fetchApi('/rf/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timeframe, symbol, limit: limit ? parseInt(limit, 10) : undefined }),
      })
      setPredictResult(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [timeframe, symbol, limit])

  const handlePredictionsRange = useCallback(async () => {
    setLoading(true)
    setError(null)
    setPredictionsRange(null)
    try {
      const data = await fetchApi('/rf/predictions-range', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timeframe,
          symbol,
          start_date: testStart || '2024-01-01',
          end_date: testEnd || '2025-01-01',
          limit: limit ? parseInt(limit, 10) : undefined,
        }),
      })
      setPredictionsRange(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [timeframe, symbol, testStart, testEnd, limit])

  const handleCV = useCallback(async () => {
    setLoading(true)
    setError(null)
    setCVResult(null)
    try {
      const params = new URLSearchParams({
        timeframe,
        symbol,
        n_splits: nSplits || '5',
      })
      if (limit) params.set('limit', limit)
      const data = await fetchApi(`/rf/cv?${params}`, { method: 'POST' })
      setCVResult(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [timeframe, symbol, limit, nSplits])

  return (
    <div className="min-h-screen w-full">
      <main className="px-6 py-8 max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-zinc-200">RF Price Direction</h1>
          <p className="text-sm text-zinc-500 mt-1">
            Random Forest model predicting price direction (up/down) 2 hours ahead
          </p>
        </div>

        {/* Params */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 mb-6">
          <h2 className="text-sm font-medium text-zinc-400 mb-4">Parameters</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Timeframe</label>
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value as Timeframe)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
              >
                {TIMEFRAMES.map((tf) => (
                  <option key={tf} value={tf}>{tf}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Symbol</label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Limit (optional)</label>
              <input
                type="text"
                value={limit}
                onChange={(e) => setLimit(e.target.value)}
                placeholder="e.g. 5000"
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none placeholder:text-zinc-600"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Proba threshold</label>
              <input
                type="text"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">CV splits</label>
              <input
                type="text"
                value={nSplits}
                onChange={(e) => setNSplits(e.target.value)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Train start</label>
              <input
                type="date"
                value={trainStart}
                onChange={(e) => setTrainStart(e.target.value)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Train end</label>
              <input
                type="date"
                value={trainEnd}
                onChange={(e) => setTrainEnd(e.target.value)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Test start</label>
              <input
                type="date"
                value={testStart}
                onChange={(e) => setTestStart(e.target.value)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Test end</label>
              <input
                type="date"
                value={testEnd}
                onChange={(e) => setTestEnd(e.target.value)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
          </div>
          <p className="mt-2 text-xs text-zinc-500">If all 4 dates are set, train/test split uses these ranges instead of train_ratio.</p>
          <div className="mt-4 flex flex-wrap gap-4">
            <label className="flex items-center gap-2 text-sm text-zinc-400 cursor-pointer">
              <input type="checkbox" checked={targetSymmetric} onChange={(e) => setTargetSymmetric(e.target.checked)} className="rounded border-zinc-600" />
              Symmetric target (drop flat)
            </label>
            <label className="flex items-center gap-2 text-sm text-zinc-400 cursor-pointer">
              <input type="checkbox" checked={useCalibration} onChange={(e) => setUseCalibration(e.target.checked)} className="rounded border-zinc-600" />
              Probability calibration
            </label>
            <label className="flex items-center gap-2 text-sm text-zinc-400 cursor-pointer">
              <input type="checkbox" checked={tuneThreshold} onChange={(e) => setTuneThreshold(e.target.checked)} className="rounded border-zinc-600" />
              Tune threshold for F1
            </label>
            <div className="flex items-center gap-2">
              <span className="text-sm text-zinc-500">Model:</span>
              <select value={modelType} onChange={(e) => setModelType(e.target.value)} className="rounded bg-zinc-800 border border-zinc-600 px-2 py-1 text-sm text-zinc-200">
                <option value="rf">RF</option>
                <option value="xgb">XGBoost</option>
                <option value="lgb">LightGBM</option>
              </select>
            </div>
          </div>
        </div>

        {/* Status */}
        <div className="flex items-center gap-2 mb-6">
          <button
            type="button"
            onClick={loadStatus}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800 text-zinc-400 hover:text-zinc-200 text-sm"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh status
          </button>
          {status && (
            <span className="flex items-center gap-1.5 text-sm">
              {status.trained ? (
                <>
                  <CheckCircle className="w-4 h-4 text-emerald-500" />
                  <span className="text-emerald-400">Model trained</span>
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-amber-500" />
                  <span className="text-amber-400">Not trained</span>
                </>
              )}
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-3 mb-8">
          <button
            type="button"
            onClick={handleFeaturesPreview}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
            Preview features
          </button>
          <button
            type="button"
            onClick={handleTrain}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Train
          </button>
          <button
            type="button"
            onClick={handleEval}
            disabled={loading || !status?.trained}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Target className="w-4 h-4" />}
            Evaluate
          </button>
          <button
            type="button"
            onClick={handlePredict}
            disabled={loading || !status?.trained}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-700/50 hover:bg-emerald-600/50 text-emerald-200 border border-emerald-500/40 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Target className="w-4 h-4" />}
            Predict
          </button>
          <button
            type="button"
            onClick={handlePredictionsRange}
            disabled={loading || !status?.trained}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-700/50 hover:bg-cyan-600/50 text-cyan-200 border border-cyan-500/40 disabled:opacity-50 disabled:cursor-not-allowed"
            title={`View all predictions in test range (${testStart} → ${testEnd})`}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
            View predictions
          </button>
          <button
            type="button"
            onClick={handleBacktest}
            disabled={loading || !status?.trained}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <BarChart3 className="w-4 h-4" />}
            Backtest
          </button>
          <button
            type="button"
            onClick={handleCV}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <GitBranch className="w-4 h-4" />}
            Cross-validate
          </button>
          <button
            type="button"
            onClick={handleReset}
            disabled={loading || !status?.trained}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-900/50 hover:bg-red-900/70 text-red-400 border border-red-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
            Reset ML
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 mb-6 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Predict result */}
        {/* Predictions in range (train page) */}
        {predictionsRange?.ok && predictionsRange.predictions.length > 0 && (() => {
          const sortKey = predictionsSort.key
          const asc = predictionsSort.asc
          const sorted = [...predictionsRange.predictions].sort((a, b) => {
            let va: string | number | boolean | null = a[sortKey as keyof typeof a]
            let vb: string | number | boolean | null = b[sortKey as keyof typeof b]
            if (sortKey === 'correct') {
              va = va === true ? 1 : va === false ? 0 : -1
              vb = vb === true ? 1 : vb === false ? 0 : -1
            }
            if (va == null && vb == null) return 0
            if (va == null) return asc ? 1 : -1
            if (vb == null) return asc ? -1 : 1
            const cmp = typeof va === 'string' ? va.localeCompare(String(vb)) : (va as number) - (vb as number)
            return asc ? cmp : -cmp
          })
          const toggleSort = (key: string) =>
            setPredictionsSort((s) => ({ key, asc: s.key === key ? !s.asc : true }))
          const SortTh = ({ colKey, label, title }: { colKey: string; label: string; title?: string }) => (
            <th
              className="px-3 py-2 font-medium text-right cursor-pointer hover:text-zinc-300 select-none"
              onClick={() => toggleSort(colKey)}
              title={title}
            >
              {label} {sortKey === colKey ? (asc ? '↑' : '↓') : ''}
            </th>
          )
          return (
            <div className="rounded-xl border border-cyan-500/30 bg-cyan-900/20 p-4 mb-6">
              <h3 className="text-sm font-medium text-cyan-400 mb-3">
                All predictions in range ({testStart} → {testEnd}) — {predictionsRange.n} rows
              </h3>
              <div className="flex flex-wrap gap-4 mb-3 text-sm">
                <span className="text-emerald-400 font-medium">Correct: {predictionsRange.n_correct ?? 0}</span>
                <span className="text-red-400 font-medium">Incorrect: {predictionsRange.n_incorrect ?? 0}</span>
                <span className="text-zinc-200 font-medium">
                  Accuracy: {predictionsRange.total_accuracy != null ? `${predictionsRange.total_accuracy}%` : '—'}
                </span>
              </div>
              <div className="overflow-x-auto max-h-96 overflow-y-auto rounded-lg border border-zinc-700">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-zinc-800/95 text-zinc-500 text-left">
                    <tr>
                      <th className="px-3 py-2 font-medium">#</th>
                      <th
                        className="px-3 py-2 font-medium cursor-pointer hover:text-zinc-300 select-none"
                        onClick={() => toggleSort('timestamp')}
                      >
                        Timestamp {sortKey === 'timestamp' ? (asc ? '↑' : '↓') : ''}
                      </th>
                      <SortTh colKey="close" label="Close" />
                      <SortTh colKey="direction" label="Dir" />
                      <SortTh colKey="proba_up" label="P(up)" />
                      <SortTh colKey="confidence_pct" label="Conf%" title="Model confidence 0–100%" />
                      <SortTh colKey="correct" label="Result" title="Correct (green) / Incorrect (red)" />
                      <th className="px-3 py-2 font-medium text-right" title="Actual / Pred low">Low</th>
                      <th className="px-3 py-2 font-medium text-right" title="Actual / Pred high">High</th>
                      <th className="px-3 py-2 font-medium text-right" title="Low / High accuracy % (vs prev candle)">Acc%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((p, i) => (
                      <tr
                        key={p.timestamp}
                        className={`border-t border-zinc-800 hover:opacity-90 ${
                          p.correct === true ? 'bg-emerald-900/40' : p.correct === false ? 'bg-red-900/40' : 'bg-transparent'
                        }`}
                      >
                        <td className="px-3 py-1.5 text-zinc-500 font-mono">{i + 1}</td>
                        <td className="px-3 py-1.5 text-zinc-400 text-xs">{formatDate(p.timestamp)}</td>
                        <td className="px-3 py-1.5 text-right font-mono">{p.close.toFixed(2)}</td>
                        <td className={`px-3 py-1.5 ${p.direction === 1 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {p.direction === 1 ? 'Up' : 'Down'}
                        </td>
                        <td className="px-3 py-1.5 text-right font-mono">{p.proba_up.toFixed(2)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-cyan-400">{p.confidence_pct}%</td>
                        <td className="px-3 py-1.5 text-right font-medium">
                          {p.correct === true ? (
                            <span className="text-emerald-400">✓</span>
                          ) : p.correct === false ? (
                            <span className="text-red-400">✗</span>
                          ) : (
                            <span className="text-zinc-500">—</span>
                          )}
                        </td>
                        <td className="px-3 py-1.5 text-right text-zinc-400 text-xs">
                          {p.actual_low != null && p.predicted_low != null
                            ? `${p.actual_low.toFixed(0)} / ${p.predicted_low.toFixed(0)}`
                            : '—'}
                        </td>
                        <td className="px-3 py-1.5 text-right text-zinc-400 text-xs">
                          {p.actual_high != null && p.predicted_high != null
                            ? `${p.actual_high.toFixed(0)} / ${p.predicted_high.toFixed(0)}`
                            : '—'}
                        </td>
                        <td className="px-3 py-1.5 text-right text-xs">
                          {p.low_accuracy_pct != null && p.high_accuracy_pct != null
                            ? `${p.low_accuracy_pct} / ${p.high_accuracy_pct}`
                            : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )
        })()}

        {predictResult?.ok && (
          <div className="rounded-xl border border-emerald-500/30 bg-emerald-900/20 p-4 mb-6">
            <h3 className="text-sm font-medium text-emerald-400 mb-3">Latest prediction (next 2 candles)</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
              <div className="rounded-lg bg-zinc-800/50 p-3">
                <div className="text-zinc-500 text-xs">Direction</div>
                <div className={predictResult.direction === 1 ? 'text-emerald-400' : 'text-red-400'}>
                  {predictResult.direction === 1 ? 'Up' : 'Down'}
                </div>
              </div>
              <div className="rounded-lg bg-zinc-800/50 p-3">
                <div className="text-zinc-500 text-xs">P(up)</div>
                <div className="text-zinc-200 font-mono">{(predictResult.proba_up ?? 0).toFixed(2)}</div>
              </div>
              <div className="rounded-lg bg-zinc-800/50 p-3">
                <div className="text-zinc-500 text-xs">Close</div>
                <div className="text-zinc-200 font-mono">{(predictResult.close ?? 0).toFixed(2)}</div>
              </div>
              <div className="rounded-lg bg-zinc-800/50 p-3">
                <div className="text-zinc-500 text-xs">Pred. low</div>
                <div className="text-red-400/90 font-mono">
                  {predictResult.predicted_low != null ? predictResult.predicted_low.toFixed(2) : '—'}
                </div>
              </div>
              <div className="rounded-lg bg-zinc-800/50 p-3">
                <div className="text-zinc-500 text-xs">Pred. high</div>
                <div className="text-emerald-400/90 font-mono">
                  {predictResult.predicted_high != null ? predictResult.predicted_high.toFixed(2) : '—'}
                </div>
              </div>
            </div>
            {predictResult.timestamp && (
              <div className="text-zinc-500 text-xs mt-2">As of {formatDate(predictResult.timestamp)}</div>
            )}
          </div>
        )}

        {/* Results */}
        <div className="space-y-6">
          {trainResult?.metrics && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400 mb-3">Train metrics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">Accuracy</div>
                  <div className="text-emerald-400 font-mono">{(trainResult.metrics.accuracy * 100).toFixed(2)}%</div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">ROC-AUC</div>
                  <div className="text-emerald-400 font-mono">{(trainResult.metrics.roc_auc * 100).toFixed(2)}%</div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">F1</div>
                  <div className="text-zinc-200 font-mono">{trainResult.metrics.f1.toFixed(3)}</div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">Train / Test</div>
                  <div className="text-zinc-200 font-mono">{trainResult.metrics.train_samples} / {trainResult.metrics.test_samples}</div>
                </div>
                {trainResult.metrics.class_balance_pct_up != null && (
                  <div className="rounded-lg bg-zinc-800/50 p-3">
                    <div className="text-zinc-500 text-xs">Class balance (up %)</div>
                    <div className="text-zinc-200 font-mono">{trainResult.metrics.class_balance_pct_up.toFixed(1)}%</div>
                    {trainResult.metrics.class_balance_n_up != null && (
                      <div className="text-zinc-500 text-[10px]">up={trainResult.metrics.class_balance_n_up} down={trainResult.metrics.class_balance_n_down}</div>
                    )}
                  </div>
                )}
                {trainResult.metrics.optimal_threshold != null && (
                  <div className="rounded-lg bg-zinc-800/50 p-3">
                    <div className="text-zinc-500 text-xs">Optimal threshold ({trainResult.metrics.threshold_optimize_by || 'f1'})</div>
                    <div className="text-emerald-400 font-mono">{trainResult.metrics.optimal_threshold.toFixed(2)}</div>
                  </div>
                )}
                {trainResult.metrics.roc_auc_train != null && (
                  <div className="rounded-lg bg-zinc-800/50 p-3">
                    <div className="text-zinc-500 text-xs">Train AUC</div>
                    <div className="text-zinc-200 font-mono">{(trainResult.metrics.roc_auc_train * 100).toFixed(2)}%</div>
                  </div>
                )}
                {trainResult.metrics.roc_auc_cv != null && (
                  <div className="rounded-lg bg-zinc-800/50 p-3">
                    <div className="text-zinc-500 text-xs">Walk-forward CV AUC ({trainResult.metrics.cv_n_folds || '?'} folds)</div>
                    <div className="text-emerald-400 font-mono">
                      {(trainResult.metrics.roc_auc_cv * 100).toFixed(2)}%
                      {trainResult.metrics.roc_auc_cv_std != null && (
                        <span className="text-zinc-500 text-xs ml-1">±{(trainResult.metrics.roc_auc_cv_std * 100).toFixed(2)}%</span>
                      )}
                    </div>
                    {trainResult.metrics.cv_mean_accuracy != null && (
                      <div className="text-zinc-400 text-xs mt-1">CV Accuracy: {(trainResult.metrics.cv_mean_accuracy * 100).toFixed(1)}%</div>
                    )}
                  </div>
                )}
                {trainResult.metrics.auc_gap != null && (
                  <div className={`rounded-lg p-3 ${trainResult.metrics.overfitting_warning ? 'bg-amber-900/30 border border-amber-500/40' : 'bg-zinc-800/50'}`}>
                    <div className="text-zinc-500 text-xs">AUC gap (train−test)</div>
                    <div className={`font-mono ${trainResult.metrics.overfitting_warning ? 'text-amber-400' : 'text-zinc-200'}`}>
                      {trainResult.metrics.auc_gap.toFixed(3)}
                      {trainResult.metrics.overfitting_warning && ' ⚠ overfitting'}
                    </div>
                  </div>
                )}
              </div>
              {trainResult.metrics.filtered_accuracy != null && (
                <div className="mt-4 rounded-lg border border-cyan-500/30 bg-cyan-900/20 p-3">
                  <div className="text-xs font-medium text-cyan-400 mb-2">Filtered predictions (confidence &ge; {trainResult.metrics.filter_confidence_min ?? 0.1})</div>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
                    <div>
                      <div className="text-zinc-500 text-xs">Accuracy</div>
                      <div className="text-cyan-300 font-mono">{(trainResult.metrics.filtered_accuracy * 100).toFixed(2)}%</div>
                    </div>
                    <div>
                      <div className="text-zinc-500 text-xs">Precision</div>
                      <div className="text-cyan-300 font-mono">{(trainResult.metrics.filtered_precision! * 100).toFixed(2)}%</div>
                    </div>
                    <div>
                      <div className="text-zinc-500 text-xs">F1</div>
                      <div className="text-cyan-300 font-mono">{trainResult.metrics.filtered_f1!.toFixed(3)}</div>
                    </div>
                    <div>
                      <div className="text-zinc-500 text-xs">Signals</div>
                      <div className="text-cyan-300 font-mono">{trainResult.metrics.filtered_signals}</div>
                    </div>
                    <div>
                      <div className="text-zinc-500 text-xs">Coverage</div>
                      <div className="text-cyan-300 font-mono">{trainResult.metrics.filtered_pct}%</div>
                    </div>
                  </div>
                </div>
              )}
              {trainResult.top_features && trainResult.top_features.length > 0 && (
                <div className="mt-4">
                  <div className="text-xs text-zinc-500 mb-2">Top features</div>
                  <div className="flex flex-wrap gap-2">
                    {trainResult.top_features.slice(0, 10).map((f) => (
                      <span key={f.name} className="px-2 py-1 rounded bg-zinc-800 text-xs font-mono text-zinc-300">
                        {f.name} {(f.importance * 100).toFixed(1)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {featuresPreview && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400 mb-3">Features (latest row, no train)</h3>
              <div className="flex items-center gap-3 mb-3 text-xs">
                <span className="text-zinc-500">Rows: {featuresPreview.n_rows}</span>
                <span className={featuresPreview.all_valid ? 'text-emerald-400' : 'text-amber-400'}>
                  {featuresPreview.all_valid ? '✓ All values valid' : '⚠ Some values missing'}
                </span>
              </div>
              <div className="overflow-x-auto max-h-72 overflow-y-auto rounded-lg border border-zinc-700">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-zinc-800/95 text-zinc-500 text-left">
                    <tr>
                      <th className="px-3 py-2 font-medium">#</th>
                      <th className="px-3 py-2 font-medium">Feature</th>
                      <th className="px-3 py-2 font-medium">Latest value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {featuresPreview.feature_names.map((name, i) => (
                      <tr key={name} className="border-t border-zinc-800 hover:bg-zinc-800/30">
                        <td className="px-3 py-1.5 text-zinc-500 font-mono">{i + 1}</td>
                        <td className="px-3 py-1.5 font-mono text-zinc-300" title={name}>{name}</td>
                        <td className="px-3 py-1.5 font-mono text-zinc-200">
                          {featuresPreview.sample_values[name] != null
                            ? Number(featuresPreview.sample_values[name]).toLocaleString(undefined, { maximumFractionDigits: 6 })
                            : <span className="text-amber-400">—</span>
                          }
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="text-xs text-zinc-500 mt-2">
                Total: {featuresPreview.feature_names.length} features
              </div>
            </div>
          )}

          {evalResult?.metrics && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400 mb-3">Evaluation</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">Accuracy</div>
                  <div className="text-emerald-400 font-mono">{(evalResult.metrics.accuracy * 100).toFixed(2)}%</div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">ROC-AUC</div>
                  <div className="text-emerald-400 font-mono">{(evalResult.metrics.roc_auc * 100).toFixed(2)}%</div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">Precision / Recall</div>
                  <div className="text-zinc-200 font-mono">{evalResult.metrics.precision.toFixed(2)} / {evalResult.metrics.recall.toFixed(2)}</div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">Test samples</div>
                  <div className="text-zinc-200 font-mono">{evalResult.metrics.test_samples}</div>
                </div>
              </div>
              {evalResult.metrics.confusion_matrix && (
                <div className="mt-3 text-xs text-zinc-500">
                  Confusion: TN={evalResult.metrics.confusion_matrix.tn} FP={evalResult.metrics.confusion_matrix.fp} FN={evalResult.metrics.confusion_matrix.fn} TP={evalResult.metrics.confusion_matrix.tp}
                </div>
              )}
            </div>
          )}

          {backtestResult?.backtest && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400 mb-3">Backtest</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">Total PnL</div>
                  <div className={`font-mono ${backtestResult.backtest.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {backtestResult.backtest.total_pnl}
                  </div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">Trades</div>
                  <div className="text-zinc-200 font-mono">{backtestResult.backtest.n_trades}</div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">Win rate</div>
                  <div className="text-emerald-400 font-mono">{(backtestResult.backtest.win_rate * 100).toFixed(1)}%</div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 p-3">
                  <div className="text-zinc-500 text-xs">Threshold</div>
                  <div className="text-zinc-200 font-mono">{backtestResult.backtest.proba_threshold}</div>
                </div>
              </div>
            </div>
          )}

          {cvResult?.folds && cvResult.folds.length > 0 && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400 mb-3">Time-series CV</h3>
              <div className="space-y-2">
                {cvResult.folds.map((f) => (
                  <div key={f.fold} className="flex items-center justify-between rounded-lg bg-zinc-800/50 px-3 py-2 text-sm">
                    <span className="text-zinc-400">Fold {f.fold}</span>
                    <span className="text-zinc-200">acc={(f.accuracy * 100).toFixed(1)}% auc={(f.roc_auc * 100).toFixed(1)}%</span>
                    <span className="text-zinc-500 text-xs">train={f.train_size} test={f.test_size}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
