import { useState, useEffect, useCallback } from 'react'
import { Loader2, Search, BarChart3, ToggleLeft, ToggleRight, ChevronLeft, ChevronRight, Shuffle, RefreshCw, Save, X, FolderOpen } from 'lucide-react'
import { getFeatureMeta, STORAGE_KEY, PRIORITIES_STORAGE_KEY } from '../config/featureMeta'

const API_BASE = '/api'

async function fetchApi(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

const TIMEFRAMES = ['5m', '15m', '1h'] as const
const DEFAULT_WINDOW_SIZE = 5
type Timeframe = (typeof TIMEFRAMES)[number]

function getTimeStep(timeframe: Timeframe): number {
  if (timeframe === '5m') return 5
  if (timeframe === '15m') return 15
  return 60
}

/** Format Date as YYYY-MM-DDTHH:mm for datetime-local input (local time) */
function toDateTimeLocal(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function shiftDateByTimeframe(dateTime: string, timeframe: Timeframe, direction: 1 | -1): string {
  const d = new Date(dateTime)
  const stepMin = getTimeStep(timeframe)
  d.setMinutes(d.getMinutes() + direction * stepMin)
  return toDateTimeLocal(d)
}

/** Send UTC ISO string to API (backend uses UTC for all queries) */
function formatDateTimeForApi(date: Date, timeframe: Timeframe): string {
  const step = getTimeStep(timeframe)
  const min = Math.floor(date.getUTCMinutes() / step) * step
  const d = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate(), date.getUTCHours(), min, 0))
  return d.toISOString().slice(0, 19) + 'Z'
}

interface CandleCard {
  open?: number
  high?: number
  low?: number
  close?: number
  high_low_diff?: number
  abs_open_close_diff?: number
  candle?: string
  timestamp?: string
}

interface SimilarityResult {
  current_candle: { price: Record<string, number>; trend: Record<string, unknown>; timestamp: string }
  similar_windows: Array<{ timestamp: string; similarity_score: number; price: Record<string, number>; trend: Record<string, unknown> }>
  most_similar_next?: Record<string, { timestamp: string; open: number; high: number; low: number; close: number }>
  prediction: {
    base_close: number
    next_1: CandleCard
    next_2: CandleCard
    next_3: CandleCard
    n_similar_used: number
  }
  actual_next: Record<string, { timestamp: string; open: number; high: number; low: number; close: number }>
  query_info?: { features_used?: string[]; priorities?: Record<string, number>; [k: string]: unknown }
}

function CandleCard({ label, candle }: { label: string; candle: CandleCard }) {
  if (!candle || Object.keys(candle).length === 0) return null
  const isGreen = candle.candle === 'green'
  return (
    <div className={`rounded-lg border ${isGreen ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-red-500/40 bg-red-500/5'} p-3`}>
      <div className="text-xs font-medium text-zinc-500 mb-2">{label}</div>
      <div className="space-y-1 text-sm font-mono">
        {candle.open != null && <div>Open: {candle.open.toFixed(2)}</div>}
        {candle.close != null && <div>Close: {candle.close.toFixed(2)}</div>}
        {candle.high_low_diff != null && <div>H-L: {candle.high_low_diff.toFixed(2)}</div>}
      </div>
    </div>
  )
}

function loadStoredFeatureState(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as Record<string, boolean>
      if (parsed && typeof parsed === 'object') return parsed
    }
  } catch {
    /* ignore */
  }
  return {}
}

function saveFeatureState(state: Record<string, boolean>) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    /* ignore */
  }
}

function loadStoredFeaturePriorities(): Record<string, string> {
  try {
    const raw = localStorage.getItem(PRIORITIES_STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as Record<string, string>
      if (parsed && typeof parsed === 'object') return parsed
    }
  } catch {
    /* ignore */
  }
  return {}
}

function saveFeaturePriorities(state: Record<string, string>) {
  try {
    localStorage.setItem(PRIORITIES_STORAGE_KEY, JSON.stringify(state))
  } catch {
    /* ignore */
  }
}

export default function LLM1Similarity() {
  const [timeframe, setTimeframe] = useState<Timeframe>('1h')
  const [dateTime, setDateTime] = useState(() => {
    const d = new Date()
    d.setMinutes(Math.floor(d.getMinutes() / 60) * 60, 0, 0)
    d.setHours(d.getHours() - 24, 0, 0, 0)
    return toDateTimeLocal(d)
  })
  const [features, setFeatures] = useState<string[]>([])
  const [featureEnabled, setFeatureEnabled] = useState<Record<string, boolean>>(loadStoredFeatureState)
  const [featurePriorities, setFeaturePriorities] = useState<Record<string, string>>(loadStoredFeaturePriorities)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SimilarityResult | null>(null)
  const [batchResult, setBatchResult] = useState<SimilarityResult[] | null>(null)
  const [regeneratingIndex, setRegeneratingIndex] = useState<number | null>(null)
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const [saveName, setSaveName] = useState('')
  const [saving, setSaving] = useState(false)
  const [savedParams, setSavedParams] = useState<Array<{ id: string; name: string; summary?: { accuracy_pct?: number; n_samples?: number }; params?: { features?: string[]; priorities?: Record<string, number> }; created_at?: string }>>([])
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([])
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>([])
  const [selectedWeekDays, setSelectedWeekDays] = useState<string[]>([])
  const [selectedTimes, setSelectedTimes] = useState<string[]>([])
  const [selectedMicrostrategyId, setSelectedMicrostrategyId] = useState<string | null>(null)

  const toggleFeature = useCallback((name: string) => {
    setFeatureEnabled((prev) => {
      const next = { ...prev, [name]: !(prev[name] ?? true) }
      saveFeatureState(next)
      return next
    })
  }, [])

  const enableAll = useCallback(() => {
    setFeatureEnabled((prev) => {
      const next = { ...prev }
      for (const f of features) next[f] = true
      saveFeatureState(next)
      return next
    })
  }, [features])

  const disableAll = useCallback(() => {
    setFeatureEnabled((prev) => {
      const next = { ...prev }
      for (const f of features) next[f] = false
      saveFeatureState(next)
      return next
    })
  }, [features])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const data = await fetchApi(`/llm1/features?timeframe=${timeframe}&symbol=BTCUSDT&window_size=${DEFAULT_WINDOW_SIZE}`)
        const feats = data.trained_features || data.all_queryable_features || []
        if (!cancelled) {
          setFeatures(feats)
          setFeatureEnabled((prev) => {
            const next = { ...prev }
            let changed = false
            for (const f of feats) {
              if (!(f in next)) {
                next[f] = true
                changed = true
              }
            }
            if (changed) saveFeatureState(next)
            return next
          })
        }
      } catch {
        if (!cancelled) setFeatures([])
      }
    }
    load()
    return () => { cancelled = true }
  }, [timeframe])

  const loadSavedParams = useCallback(async () => {
    try {
      const data = await fetchApi('/llm1/params')
      setSavedParams(data.params || [])
    } catch {
      setSavedParams([])
    }
  }, [])

  useEffect(() => {
    loadSavedParams()
  }, [loadSavedParams, saveModalOpen])

  const enabledFeatures = features.filter((f) => featureEnabled[f] !== false)

  const prioritiesStr = Object.entries(featurePriorities)
    .filter(([, w]) => w.trim())
    .map(([f, w]) => `${f}:${w.trim()}`)
    .join(',')

  const handleSearch = useCallback(async (enabledToUse: string[], dateTimeOverride?: string) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setBatchResult(null)
    try {
      const dt = new Date(dateTimeOverride ?? dateTime)
      const datetimeParam = formatDateTimeForApi(dt, timeframe)
      const params = new URLSearchParams({
        timeframe,
        symbol: 'BTCUSDT',
        window_size: String(DEFAULT_WINDOW_SIZE),
        datetime: datetimeParam,
        k: '10',
        min_similarity: '0.99',
      })
      if (enabledToUse.length > 0) {
        params.set('features', enabledToUse.join(','))
      }
      if (prioritiesStr) {
        params.set('priorities', prioritiesStr)
      }
      const data = await fetchApi(`/llm1/similarity?${params}`, { method: 'POST' })
      setResult(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [dateTime, timeframe, prioritiesStr])

  const handleRandomBatch = useCallback(async (n: number) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setBatchResult(null)
    try {
      const params = new URLSearchParams({
        timeframe,
        symbol: 'BTCUSDT',
        window_size: String(DEFAULT_WINDOW_SIZE),
        n_samples: String(n),
        k: '10',
        min_similarity: '0.99',
      })
      if (enabledFeatures.length > 0) {
        params.set('features', enabledFeatures.join(','))
      }
      if (prioritiesStr) {
        params.set('priorities', prioritiesStr)
      }
      const data = await fetchApi(`/llm1/similarity-batch?${params}`, { method: 'POST' })
      setBatchResult(data.results || [])
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [timeframe, enabledFeatures, prioritiesStr])

  const handleRegenerateChain = useCallback(async () => {
    const list = batchResult?.filter((r) => !!r.prediction?.next_1?.candle) ?? []
    const datetimes = list
      .map((r) => r.query_info?.requested_datetime as string)
      .filter(Boolean)
    if (datetimes.length === 0) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchApi('/llm1/similarity-by-times', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datetimes,
          timeframe,
          symbol: 'BTCUSDT',
          window_size: DEFAULT_WINDOW_SIZE,
          features: enabledFeatures.length > 0 ? enabledFeatures : undefined,
          priorities: prioritiesStr || undefined,
          k: 10,
          min_similarity: 0.99,
        }),
      })
      const newResults = data.results || []
      if (newResults.length > 0) setBatchResult(newResults)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [batchResult, timeframe, enabledFeatures, prioritiesStr])

  const handleRegenerate = useCallback(async (index: number) => {
    const list = batchResult?.filter((r) => !!r.prediction?.next_1?.candle) ?? []
    const r = list[index]
    if (!r?.query_info?.requested_datetime) return
    const datetimeIso = r.query_info.requested_datetime as string
    setRegeneratingIndex(index)
    try {
      const data = await fetchApi('/llm1/similarity-by-times', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datetimes: [datetimeIso],
          timeframe,
          symbol: 'BTCUSDT',
          window_size: DEFAULT_WINDOW_SIZE,
          features: enabledFeatures.length > 0 ? enabledFeatures : undefined,
          priorities: prioritiesStr || undefined,
          k: 10,
          min_similarity: 0.99,
        }),
      })
      const newResult = data.results?.[0]
      if (newResult && batchResult) {
        const globalIndex = batchResult.indexOf(list[index])
        const next = [...batchResult]
        next[globalIndex] = newResult
        setBatchResult(next)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setRegeneratingIndex(null)
    }
  }, [batchResult, timeframe, enabledFeatures, prioritiesStr])

  const handleLoadSaved = useCallback((p: { id?: string; params?: { features?: string[]; priorities?: Record<string, number> } }) => {
    if (p.id) {
      setSelectedMicrostrategyId(p.id)
    }
    const params = p.params
    if (!params) return
    if (params.features?.length) {
      setFeatureEnabled((prev) => {
        const next = { ...prev }
        for (const f of features) {
          next[f] = params.features!.includes(f)
        }
        saveFeatureState(next)
        return next
      })
    }
    if (params.priorities && Object.keys(params.priorities).length > 0) {
      setFeaturePriorities(
        Object.fromEntries(Object.entries(params.priorities).map(([k, v]) => [k, String(v)]))
      )
      saveFeaturePriorities(Object.fromEntries(Object.entries(params.priorities).map(([k, v]) => [k, String(v)])))
    }
  }, [features])

  const handleSaveOpen = useCallback(() => {
    setSaveName('')
    setSaveModalOpen(true)
  }, [])

  const handleSaveSubmit = useCallback(async () => {
    if (!batchResult || batchResult.length === 0) return
    const name = saveName.trim()
    if (!name) return
    setSaving(true)
    setError(null)
    try {
      const allWithPred = batchResult.filter((r) => !!r.prediction?.next_1?.candle)
      const comparable = allWithPred.filter((r) => !!r.actual_next?.actual_next_1)
      const sameColors = comparable.filter((r) => {
        const pred = String(r.prediction!.next_1!.candle!).toLowerCase()
        const actual = r.actual_next!.actual_next_1!
        const actualColor = (actual.close - actual.open) >= -1e-8 ? 'green' : 'red'
        return pred === actualColor
      }).length
      const accuracyPct = comparable.length > 0 ? (sameColors / comparable.length) * 100 : 0
      const priceErrors = comparable.map((r) => {
        const predClose = r.prediction!.next_1!.close ?? 0
        const actualClose = r.actual_next!.actual_next_1!.close ?? 0
        return Math.abs(predClose - actualClose)
      })
      const avgAbsError = priceErrors.length > 0 ? priceErrors.reduce((a, b) => a + b, 0) / priceErrors.length : 0
      const avgActualClose = comparable.length > 0
        ? comparable.reduce((s, r) => s + (r.actual_next!.actual_next_1!.close ?? 0), 0) / comparable.length
        : 0
      const avgPredClose = comparable.length > 0
        ? comparable.reduce((s, r) => s + (r.prediction!.next_1!.close ?? 0), 0) / comparable.length
        : 0
      const avgErrorPct = avgActualClose > 0 ? (avgAbsError / avgActualClose) * 100 : 0

      const params: Record<string, unknown> = {
        features: enabledFeatures,
        priorities: Object.fromEntries(
          Object.entries(featurePriorities)
            .filter(([, w]) => w.trim())
            .map(([f, w]) => [f, parseFloat(w.trim()) || 1])
        ),
        k: 10,
        min_similarity: 0.999,
      }

      await fetchApi('/llm1/save-params', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          window_size: DEFAULT_WINDOW_SIZE,
          timeframe,
          symbol: 'BTCUSDT',
          params,
          results: batchResult,
          summary: {
            n_samples: batchResult.length,
            same_colors: sameColors,
            comparable: comparable.length,
            accuracy_pct: accuracyPct,
            avg_abs_error: avgAbsError,
            avg_error_pct: avgErrorPct,
            avg_pred_close: avgPredClose,
            avg_actual_close: avgActualClose,
          },
        }),
      })
      setSaveModalOpen(false)
      setSaveName('')
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }, [batchResult, enabledFeatures, featurePriorities, saveName, timeframe])

  const mostSimilar = result?.similar_windows?.[0]
  const pred = result?.prediction
  const actual = result?.actual_next || {}

  const featureCardsContent = (() => {
    const byCategory = new Map<string, string[]>()
    for (const f of features) {
      const { category } = getFeatureMeta(f)
      if (!byCategory.has(category)) byCategory.set(category, [])
      byCategory.get(category)!.push(f)
    }
    const order = ['Price', 'Trend', 'Momentum', 'Volatility', 'Crossings', 'Sentiment', 'Time', 'Strategy', 'Other']
    const cats = [...new Set([...order, ...byCategory.keys()])].filter((c) => byCategory.has(c))
    return cats.map((cat) => (
      <div key={cat}>
        <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">{cat}</h3>
        <div className="space-y-2">
          {(byCategory.get(cat) || []).map((f) => {
            const { description } = getFeatureMeta(f)
            const enabled = featureEnabled[f] !== false
            return (
              <div key={f} className="rounded-lg border border-zinc-700 overflow-hidden">
                <button
                  type="button"
                  onClick={() => toggleFeature(f)}
                  className={`flex items-start gap-2 p-2.5 w-full text-left transition-colors ${
                    enabled
                      ? 'bg-emerald-500/5 hover:bg-emerald-500/10'
                      : 'bg-zinc-900/50 hover:bg-zinc-800/50 opacity-60'
                  }`}
                >
                  <span className="mt-0.5 shrink-0 text-base">
                    {enabled ? <ToggleRight className="text-emerald-500" /> : <ToggleLeft className="text-zinc-500" />}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-mono text-zinc-200 truncate">{f}</div>
                    <div className="text-[11px] text-zinc-500 mt-0.5 line-clamp-2">{description}</div>
                  </div>
                </button>
                {enabled && (
                  <div className="px-2.5 pb-2 pt-0 flex items-center gap-2 border-t border-zinc-700/50">
                    <span className="text-[10px] text-zinc-500 shrink-0">weight</span>
                    <input
                      type="text"
                      value={featurePriorities[f] ?? ''}
                      onChange={(e) => {
                        e.stopPropagation()
                        setFeaturePriorities((prev) => {
                          const next = { ...prev, [f]: e.target.value }
                          saveFeaturePriorities(next)
                          return next
                        })
                      }}
                      onClick={(e) => e.stopPropagation()}
                      placeholder="1"
                      className="flex-1 min-w-0 rounded bg-zinc-900 border border-zinc-600 px-2 py-1 text-xs font-mono focus:border-violet-500 focus:outline-none"
                    />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    ))
  })()

  return (
    <div className="min-h-screen w-full flex">
      <aside className="left-0 top-0 bottom-0 w-72 border-r border-zinc-800 bg-zinc-950/95 overflow-y-auto z-10 hidden lg:block">
        <div className="p-4 pt-8 space-y-6">
          {/* Strategy Filters */}
          <div>
            <h3 className="text-xs font-medium text-zinc-500 mb-2">Strategy filters</h3>
            <div className="flex flex-wrap gap-1.5">
              {['Trend Following', 'Mean Reversion', 'Breakout', 'Momentum', 'Scalping'].map((strategy) => (
                <button
                  key={strategy}
                  type="button"
                  onClick={() => {
                    setSelectedStrategies((prev) =>
                      prev.includes(strategy)
                        ? prev.filter((s) => s !== strategy)
                        : [...prev, strategy]
                    )
                  }}
                  className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                    selectedStrategies.includes(strategy)
                      ? 'bg-emerald-600/20 border-emerald-500/50 text-emerald-400'
                      : 'bg-zinc-800/50 border-zinc-700 text-zinc-400 hover:border-zinc-600'
                  }`}
                >
                  {strategy}
                </button>
              ))}
            </div>
          </div>

          {/* Week Day Filter */}
          <div>
            <h3 className="text-xs font-medium text-zinc-500 mb-2">Week day filter</h3>
            <div className="flex flex-wrap gap-1.5">
              {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day) => (
                <button
                  key={day}
                  type="button"
                  onClick={() => {
                    setSelectedWeekDays((prev) =>
                      prev.includes(day)
                        ? prev.filter((d) => d !== day)
                        : [...prev, day]
                    )
                  }}
                  className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                    selectedWeekDays.includes(day)
                      ? 'bg-amber-600/20 border-amber-500/50 text-amber-400'
                      : 'bg-zinc-800/50 border-zinc-700 text-zinc-400 hover:border-zinc-600'
                  }`}
                >
                  {day}
                </button>
              ))}
            </div>
          </div>

          {/* Time Filter */}
          <div>
            <h3 className="text-xs font-medium text-zinc-500 mb-2">
              Time filter <span className="text-zinc-600">(UTC+0)</span>
            </h3>
            <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto">
              {Array.from({ length: 24 }, (_, i) => {
                const hour = String(i).padStart(2, '0')
                const time = `${hour}:00`
                return (
                  <button
                    key={time}
                    type="button"
                    onClick={() => {
                      setSelectedTimes((prev) =>
                        prev.includes(time)
                          ? prev.filter((t) => t !== time)
                          : [...prev, time]
                      )
                    }}
                    className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                      selectedTimes.includes(time)
                        ? 'bg-blue-600/20 border-blue-500/50 text-blue-400'
                        : 'bg-zinc-800/50 border-zinc-700 text-zinc-400 hover:border-zinc-600'
                    }`}
                  >
                    {time}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </aside>
      <main className="px-6 w-full min-h-screen">
        <div className="max-w-4xl mx-auto space-y-8">
        {/* Microstrategies Tabs */}
        <div className="border-b border-zinc-800">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-medium text-zinc-400">Microstrategies</h2>
            <button
              type="button"
              onClick={loadSavedParams}
              className="text-xs px-2 py-1 rounded bg-zinc-600 hover:bg-zinc-500 text-zinc-200"
              title="Refresh"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-zinc-900">
            {savedParams.length === 0 ? (
              <div className="text-xs text-zinc-500 py-2">No microstrategies yet</div>
            ) : (
              savedParams.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => handleLoadSaved(p)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-t-lg border-b-2 transition-colors whitespace-nowrap ${
                    selectedMicrostrategyId === p.id
                      ? 'border-cyan-500 bg-zinc-900/50 text-cyan-400'
                      : 'border-transparent bg-zinc-900/30 text-zinc-400 hover:text-zinc-300 hover:bg-zinc-900/40'
                  }`}
                >
                  <FolderOpen className={`w-4 h-4 shrink-0 ${selectedMicrostrategyId === p.id ? 'text-cyan-500' : 'text-zinc-500'}`} />
                  <span className="text-xs font-mono truncate max-w-[120px]">{p.name}</span>
                  {p.summary?.accuracy_pct != null && (
                    <span className="text-[10px] text-zinc-500">
                      {p.summary.accuracy_pct.toFixed(1)}%
                    </span>
                  )}
                </button>
              ))
            )}
          </div>
        </div>
        {/* Mobile: filters as collapsible sections */}
        <div className="lg:hidden space-y-4">
          <details className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden">
            <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-zinc-400">
              Strategy filters
            </summary>
            <div className="px-4 pb-4">
              <div className="flex flex-wrap gap-1.5">
                {['Trend Following', 'Mean Reversion', 'Breakout', 'Momentum', 'Scalping'].map((strategy) => (
                  <button
                    key={strategy}
                    type="button"
                    onClick={() => {
                      setSelectedStrategies((prev) =>
                        prev.includes(strategy)
                          ? prev.filter((s) => s !== strategy)
                          : [...prev, strategy]
                      )
                    }}
                    className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                      selectedStrategies.includes(strategy)
                        ? 'bg-emerald-600/20 border-emerald-500/50 text-emerald-400'
                        : 'bg-zinc-800/50 border-zinc-700 text-zinc-400 hover:border-zinc-600'
                    }`}
                  >
                    {strategy}
                  </button>
                ))}
              </div>
            </div>
          </details>
          <details className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden">
            <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-zinc-400">
              Week day filter
            </summary>
            <div className="px-4 pb-4">
              <div className="flex flex-wrap gap-1.5">
                {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day) => (
                  <button
                    key={day}
                    type="button"
                    onClick={() => {
                      setSelectedWeekDays((prev) =>
                        prev.includes(day)
                          ? prev.filter((d) => d !== day)
                          : [...prev, day]
                      )
                    }}
                    className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                      selectedWeekDays.includes(day)
                        ? 'bg-amber-600/20 border-amber-500/50 text-amber-400'
                        : 'bg-zinc-800/50 border-zinc-700 text-zinc-400 hover:border-zinc-600'
                    }`}
                  >
                    {day}
                  </button>
                ))}
              </div>
            </div>
          </details>
          <details className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden">
            <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-zinc-400">
              Time filter <span className="text-zinc-600">(UTC+0)</span>
            </summary>
            <div className="px-4 pb-4">
              <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto">
                {Array.from({ length: 24 }, (_, i) => {
                  const hour = String(i).padStart(2, '0')
                  const time = `${hour}:00`
                  return (
                    <button
                      key={time}
                      type="button"
                      onClick={() => {
                        setSelectedTimes((prev) =>
                          prev.includes(time)
                            ? prev.filter((t) => t !== time)
                            : [...prev, time]
                        )
                      }}
                      className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                        selectedTimes.includes(time)
                          ? 'bg-blue-600/20 border-blue-500/50 text-blue-400'
                          : 'bg-zinc-800/50 border-zinc-700 text-zinc-400 hover:border-zinc-600'
                      }`}
                    >
                      {time}
                    </button>
                  )
                })}
              </div>
            </div>
          </details>
          <details className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden">
            <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-zinc-400">
              Features — {enabledFeatures.length}/{features.length}
            </summary>
            <div className="px-4 pb-4 space-y-3 max-h-64 overflow-y-auto">
              {featureCardsContent}
            </div>
          </details>
        </div>
        <h1 className="text-2xl font-semibold flex items-center gap-2">
          <BarChart3 className="w-8 h-8 text-emerald-500" />
          LLM1 Similarity
        </h1>

        {error && (
          <div className="rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <section className="flex flex-wrap items-end gap-4">
          <div>
            <label className="block text-sm text-zinc-500 mb-1">Timeframe</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value as Timeframe)}
              className="w-full rounded-lg bg-zinc-900 border border-zinc-700 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none min-w-[6rem]"
            >
              {TIMEFRAMES.map((tf) => (
                <option key={tf} value={tf}>{tf}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-zinc-500 mb-1">Date & Time</label>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => {
                  const prev = shiftDateByTimeframe(dateTime, timeframe, -1)
                  setDateTime(prev)
                  handleSearch(enabledFeatures, prev)
                }}
                disabled={loading}
                className="rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-600 p-2 disabled:opacity-50"
                title="Previous candle"
              >
                <ChevronLeft className="w-5 h-5 text-zinc-400" />
              </button>
              <input
                type="datetime-local"
                value={dateTime}
                onChange={(e) => setDateTime(e.target.value)}
                step={String(getTimeStep(timeframe) * 60)}
                className="rounded-lg bg-zinc-900 border border-zinc-700 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none min-w-[11rem]"
              />
              <button
                type="button"
                onClick={() => {
                  const next = shiftDateByTimeframe(dateTime, timeframe, 1)
                  setDateTime(next)
                  handleSearch(enabledFeatures, next)
                }}
                disabled={loading}
                className="rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-600 p-2 disabled:opacity-50"
                title="Next candle"
              >
                <ChevronRight className="w-5 h-5 text-zinc-400" />
              </button>
            </div>
          </div>
          <button
            onClick={() => handleSearch(enabledFeatures)}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 px-6 py-3 text-sm font-medium text-white disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
            Get Similarity
          </button>
          <div className="flex flex-wrap items-end gap-2 border-l border-zinc-700 pl-4">
            <button
              onClick={() => handleRandomBatch(10)}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg bg-amber-600 hover:bg-amber-500 px-5 py-3 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Shuffle className="w-5 h-5" />}
              Random 10
            </button>
            <button
              onClick={() => handleRandomBatch(20)}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg bg-violet-600 hover:bg-violet-500 px-5 py-3 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Shuffle className="w-5 h-5" />}
              Random 20
            </button>
            <button
              onClick={() => handleRandomBatch(200)}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg bg-rose-600 hover:bg-rose-500 px-5 py-3 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Shuffle className="w-5 h-5" />}
              Random 200
            </button>
            <button
              onClick={() => handleRandomBatch(100)}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg bg-sky-600 hover:bg-sky-500 px-5 py-3 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Shuffle className="w-5 h-5" />}
              Random 100
            </button>
            <button
              onClick={handleSaveOpen}
              disabled={!batchResult || batchResult.length === 0}
              className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 px-5 py-3 text-sm font-medium text-white disabled:opacity-50 disabled:cursor-not-allowed"
              title="Save params and results (run Random test first)"
            >
              <Save className="w-5 h-5" />
              Save
            </button>
          </div>
        </section>

        {batchResult && batchResult.length > 0 && (() => {
          const allWithPred = batchResult.filter((r) => !!r.prediction?.next_1?.candle)
          const comparable = allWithPred.filter((r) => !!r.actual_next?.actual_next_1)
          const sameColors = comparable.filter((r) => {
            const pred = String(r.prediction!.next_1!.candle!).toLowerCase()
            const actual = r.actual_next!.actual_next_1!
            const actualColor = (actual.close - actual.open) >= -1e-8 ? 'green' : 'red'
            return pred === actualColor
          }).length
          const accuracyPct = comparable.length > 0 ? (sameColors / comparable.length) * 100 : 0
          const priceErrors = comparable.map((r) => {
            const predClose = r.prediction!.next_1!.close ?? 0
            const actualClose = r.actual_next!.actual_next_1!.close ?? 0
            return Math.abs(predClose - actualClose)
          })
          const avgAbsError = priceErrors.length > 0
            ? priceErrors.reduce((a, b) => a + b, 0) / priceErrors.length
            : 0
          const avgActualClose = comparable.length > 0
            ? comparable.reduce((s, r) => s + (r.actual_next!.actual_next_1!.close ?? 0), 0) / comparable.length
            : 0
          const avgPredClose = comparable.length > 0
            ? comparable.reduce((s, r) => s + (r.prediction!.next_1!.close ?? 0), 0) / comparable.length
            : 0
          const avgErrorPct = avgActualClose > 0 ? (avgAbsError / avgActualClose) * 100 : 0
          return (
          <section className="space-y-4">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <h2 className="text-lg font-medium text-zinc-300">Batch Results ({batchResult.length} random candles)</h2>
              <div className="flex items-center gap-3 flex-wrap">
                <button
                  type="button"
                  onClick={handleRegenerateChain}
                  disabled={loading}
                  className="inline-flex items-center gap-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 border border-zinc-600 px-4 py-2 text-sm font-medium text-zinc-200 disabled:opacity-50"
                  title="Regenerate same chain with fresh predictions"
                >
                  {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                  Regenerate Same Chain
                </button>
                <div className="text-sm font-mono px-4 py-2 rounded-lg bg-zinc-800 border border-zinc-600">
                  <span className="text-zinc-400">Green/Red accuracy: </span>
                  <span className="text-emerald-400 font-semibold">{sameColors}</span>
                  <span className="text-zinc-500"> / {comparable.length} </span>
                  <span className="text-emerald-400 font-semibold">({accuracyPct.toFixed(1)}%)</span>
                </div>
                <div className="text-sm font-mono px-4 py-2 rounded-lg bg-zinc-800 border border-zinc-600">
                  <span className="text-zinc-400">Avg price error: </span>
                  <span className="text-amber-400 font-semibold">{avgAbsError.toFixed(2)}</span>
                  <span className="text-zinc-500"> ({avgErrorPct.toFixed(2)}% of avg close)</span>
                </div>
                <div className="text-xs font-mono px-3 py-2 rounded-lg bg-zinc-800/70 border border-zinc-600 text-zinc-400">
                  Avg pred close: {avgPredClose.toFixed(2)} vs actual: {avgActualClose.toFixed(2)}
                </div>
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div>
                <h3 className="text-sm font-medium text-emerald-400 mb-3">Predicted</h3>
                <div className="space-y-3">
                  {allWithPred.map((r, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <div className="flex-1 min-w-0">
                        <CandleCard
                          label={`#${i + 1} ${r.actual_next?.actual_next_1?.timestamp ?? r.query_info?.query_end ?? ''}`}
                          candle={r.prediction!.next_1}
                        />
                      </div>
                      <button
                        type="button"
                        onClick={() => handleRegenerate(i)}
                        disabled={regeneratingIndex === i}
                        className="shrink-0 mt-1 p-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 border border-zinc-600 disabled:opacity-50"
                        title="Regenerate"
                      >
                        {regeneratingIndex === i ? (
                          <Loader2 className="w-4 h-4 animate-spin text-zinc-400" />
                        ) : (
                          <RefreshCw className="w-4 h-4 text-zinc-400" />
                        )}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <h3 className="text-sm font-medium text-amber-400 mb-3">Actual</h3>
                <div className="space-y-3">
                  {allWithPred.map((r, i) => {
                    const actual = r.actual_next?.actual_next_1
                    if (!actual) {
                      return (
                        <div key={i} className="flex items-start gap-2">
                          <div className="flex-1 min-w-0 rounded-lg border border-zinc-600 bg-zinc-800/50 p-3">
                            <div className="text-xs font-medium text-zinc-500 mb-2">{`#${i + 1} ${r.query_info?.query_end ?? ''}`}</div>
                            <div className="text-sm text-zinc-400 italic">Cannot predict for this one</div>
                          </div>
                          <div className="w-10 shrink-0" />
                        </div>
                      )
                    }
                    return (
                      <div key={i} className="flex items-start gap-2">
                        <div className="flex-1 min-w-0">
                          <CandleCard
                            label={`#${i + 1} ${actual.timestamp ?? ''}`}
                            candle={{
                              open: actual.open,
                              high: actual.high,
                              low: actual.low,
                              close: actual.close,
                              high_low_diff: actual.high - actual.low,
                              candle: actual.close >= actual.open ? 'green' : 'red',
                              timestamp: actual.timestamp,
                            }}
                          />
                        </div>
                        <div className="w-10 shrink-0" />
                      </div>
                    )
                  })}
                </div>
              </div>
              <div>
                <h3 className="text-sm font-medium text-cyan-400 mb-3">Difference (pred − actual)</h3>
                <div className="space-y-3">
                  {allWithPred.map((r, i) => {
                    const actual = r.actual_next?.actual_next_1
                    const pred = r.prediction?.next_1
                    if (!actual || !pred) {
                      return (
                        <div key={i} className="flex items-start gap-2">
                          <div className="flex-1 min-w-0 rounded-lg border border-zinc-600 bg-zinc-800/50 p-3 min-h-[5.5rem] flex items-center">
                            <div className="text-sm text-zinc-500 italic">—</div>
                          </div>
                          <div className="w-10 shrink-0" />
                        </div>
                      )
                    }
                    const predHigh = pred.high
                    const predLow = pred.low
                    if (predHigh == null || predLow == null) {
                      return (
                        <div key={i} className="flex items-start gap-2">
                          <div className="flex-1 min-w-0 rounded-lg border border-zinc-600 bg-zinc-800/50 p-3 min-h-[5.5rem] flex items-center">
                            <div className="text-sm text-zinc-500 italic">—</div>
                          </div>
                          <div className="w-10 shrink-0" />
                        </div>
                      )
                    }
                    const highDiff = predHigh - actual.high
                    const lowDiff = predLow - actual.low
                    const range = actual.high - actual.low
                    const totalError = Math.abs(highDiff) + Math.abs(lowDiff)
                    const accuracyPct = range > 0 ? Math.max(0, 100 - (totalError / range) * 100) : 0
                    return (
                      <div key={i} className="flex items-start gap-2">
                        <div className="flex-1 min-w-0 rounded-lg border border-zinc-600 bg-zinc-800/50 p-3">
                          <div className="text-xs font-medium text-zinc-500 mb-2">{`#${i + 1} ${actual.timestamp ?? ''}`}</div>
                          <div className="space-y-1 text-sm font-mono">
                            <div className={highDiff >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                              Δ High: {highDiff >= 0 ? '+' : ''}{highDiff.toFixed(2)}
                            </div>
                            <div className={lowDiff >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                              Δ Low: {lowDiff >= 0 ? '+' : ''}{lowDiff.toFixed(2)}
                            </div>
                            <div className={accuracyPct >= 50 ? 'text-emerald-400' : accuracyPct >= 25 ? 'text-amber-400' : 'text-red-400'}>
                              Accuracy: {accuracyPct.toFixed(1)}%
                            </div>
                          </div>
                        </div>
                        <div className="w-10 shrink-0" />
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          </section>
          )
        })()}

        {result && (
          <div className="space-y-6">
          {(result.query_info?.features_used || result.query_info?.priorities) && (
            <div className="text-xs text-zinc-500 space-y-1">
              {result.query_info.features_used && (
                <div>Features used: {result.query_info.features_used.join(', ')}</div>
              )}
              {result.query_info.priorities && Object.keys(result.query_info.priorities).length > 0 && (
                <div>Priorities: {Object.entries(result.query_info.priorities).map(([f, w]) => `${f}:${w}`).join(', ')}</div>
              )}
            </div>
          )}

          {/* Current candle + Predicted/Actual + Most similar */}
          <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Current candle */}
            {result.current_candle?.price && (
              <CandleCard
                label="Current Candle"
                candle={{
                  open: result.current_candle.price.open,
                  high: result.current_candle.price.high,
                  low: result.current_candle.price.low,
                  close: result.current_candle.price.close,
                  high_low_diff: (result.current_candle.price.high ?? 0) - (result.current_candle.price.low ?? 0),
                  abs_open_close_diff: Math.abs((result.current_candle.price.open ?? 0) - (result.current_candle.price.close ?? 0)),
                  candle: (result.current_candle.price.close ?? 0) >= (result.current_candle.price.open ?? 0) ? 'green' : 'red',
                  timestamp: result.current_candle.timestamp,
                }}
              />
            )}

            {/* Predicted + Actual (single candle each) */}
            {pred && (
              <div className="space-y-2">
                <CandleCard label="Predicted" candle={pred.next_1} />
                {actual.actual_next_1 ? (
                  <CandleCard
                    label="Actual"
                    candle={{
                      open: actual.actual_next_1.open,
                      high: actual.actual_next_1.high,
                      low: actual.actual_next_1.low,
                      close: actual.actual_next_1.close,
                      high_low_diff: actual.actual_next_1.high - actual.actual_next_1.low,
                      candle: actual.actual_next_1.close >= actual.actual_next_1.open ? 'green' : 'red',
                      timestamp: actual.actual_next_1.timestamp,
                    }}
                  />
                ) : (
                  <div className="rounded-lg border border-zinc-600 bg-zinc-800/50 p-3">
                    <div className="text-xs font-medium text-zinc-500 mb-2">Actual</div>
                    <div className="text-sm text-zinc-400 italic">
                      {result.query_info?.at_data_end
                        ? `Pick a date at least 3 candles before data end (${result.query_info.data_end ?? '—'}) to see actual`
                        : 'Cannot predict for this one'}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Most similar candle only */}
            {mostSimilar?.price && (
              <CandleCard
                label={`Most Similar (${((mostSimilar.similarity_score || 0) * 100).toFixed(1)}%)`}
                candle={{
                  open: mostSimilar.price.open,
                  high: mostSimilar.price.high,
                  low: mostSimilar.price.low,
                  close: mostSimilar.price.close,
                  high_low_diff: (mostSimilar.price.high ?? 0) - (mostSimilar.price.low ?? 0),
                  candle: (mostSimilar.price.close ?? 0) >= (mostSimilar.price.open ?? 0) ? 'green' : 'red',
                  timestamp: mostSimilar.timestamp,
                }}
              />
            )}
          </section>
          </div>
        )}
        </div>
      </main>

      <aside className="right-0 top-0 bottom-0 w-72 border-l border-zinc-800 bg-zinc-950/95 overflow-y-auto z-10 hidden lg:block">
        <div className="p-4 ">
          <div className="flex items-center justify-between gap-2 mb-2">
            <h2 className="text-sm font-medium text-zinc-400">
              Features — {enabledFeatures.length}/{features.length}
            </h2>
            <div className="flex gap-1">
              <button
                type="button"
                onClick={enableAll}
                className="text-xs px-2 py-1 rounded bg-emerald-600/80 hover:bg-emerald-500 text-white"
              >
                All
              </button>
              <button
                type="button"
                onClick={disableAll}
                className="text-xs px-2 py-1 rounded bg-zinc-600 hover:bg-zinc-500 text-zinc-200"
              >
                None
              </button>
            </div>
          </div>
          <div className="space-y-4 max-h-[calc(100vh-8rem)] overflow-y-auto pr-1">
            {featureCardsContent}
          </div>
        </div>
      </aside>

      {saveModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => !saving && setSaveModalOpen(false)}>
          <div className="rounded-xl bg-zinc-900 border border-zinc-700 p-6 w-full max-w-md shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-zinc-200">Save params & results</h3>
              <button type="button" onClick={() => !saving && setSaveModalOpen(false)} className="p-1 rounded hover:bg-zinc-800 text-zinc-400">
                <X className="w-5 h-5" />
              </button>
            </div>
            <label className="block text-sm text-zinc-500 mb-2">Name</label>
            <input
              type="text"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="e.g. close_norm_priority_v1"
              className="w-full rounded-lg bg-zinc-800 border border-zinc-600 px-4 py-2 text-sm text-zinc-200 placeholder-zinc-500 focus:border-emerald-500 focus:outline-none mb-4"
              autoFocus
            />
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => !saving && setSaveModalOpen(false)}
                disabled={saving}
                className="px-4 py-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-zinc-200 text-sm font-medium disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveSubmit}
                disabled={saving || !saveName.trim()}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium disabled:opacity-50"
              >
                {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
