import { useState, useEffect } from 'react'
import { Layers, Plus, X, Loader2, Trash2, FileCode, ArrowRight } from 'lucide-react'

const API_BASE = '/api'

interface Strategy {
  id: string
  name: string
  strategy_file: string
  description?: string
  params?: Record<string, unknown>
  created_at?: string
  updated_at?: string
}

async function fetchApi(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    const errorText = await res.text()
    throw new Error(errorText || `HTTP ${res.status}`)
  }
  return res.json()
}

interface StrategiesProps {
  onStrategyClick?: (strategyId: string) => void
}

export default function Strategies({ onStrategyClick }: StrategiesProps) {
  const [newStrategyModalOpen, setNewStrategyModalOpen] = useState(false)
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [strategyName, setStrategyName] = useState('')

  const loadStrategies = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchApi('/strategies')
      setStrategies(data.strategies || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadStrategies()
  }, [])

  const handleCreateStrategy = async () => {
    if (!strategyName.trim()) {
      setError('Strategy name is required')
      return
    }

    setCreating(true)
    setError(null)
    try {
      await fetchApi('/strategies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: strategyName.trim(),
        }),
      })
      setStrategyName('')
      setNewStrategyModalOpen(false)
      await loadStrategies()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setCreating(false)
    }
  }

  const handleDeleteStrategy = async (id: string) => {
    if (!confirm('Are you sure you want to delete this strategy?')) {
      return
    }
    try {
      await fetchApi(`/strategies/${id}`, {
        method: 'DELETE',
      })
      await loadStrategies()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="min-h-screen w-full">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Layers className="w-8 h-8 text-emerald-500" />
            <h1 className="text-2xl font-semibold">Strategies</h1>
          </div>
          <button
            onClick={() => setNewStrategyModalOpen(true)}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-emerald-600 hover:bg-emerald-500 text-white"
          >
            <Plus className="w-4 h-4" />
            New Strategy
          </button>
        </div>

        {error && (
          <div className="rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 text-sm mb-4">
            {error}
          </div>
        )}

        {loading ? (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-8">
            <div className="flex items-center justify-center gap-2 text-zinc-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading strategies...
            </div>
          </div>
        ) : strategies.length === 0 ? (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-8">
            <p className="text-zinc-400 text-center">No strategies yet. Create your first strategy!</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {strategies.map((strategy) => (
              <div
                key={strategy.id}
                className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 hover:border-zinc-700 transition-colors cursor-pointer group"
                onClick={() => onStrategyClick?.(strategy.id)}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <FileCode className="w-5 h-5 text-emerald-500 shrink-0" />
                    <h3 className="text-sm font-semibold text-zinc-200 truncate">{strategy.name}</h3>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onStrategyClick?.(strategy.id)
                      }}
                      className="p-1 rounded hover:bg-zinc-800 text-zinc-400 hover:text-emerald-400 transition-colors"
                      title="View details"
                    >
                      <ArrowRight className="w-4 h-4" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDeleteStrategy(strategy.id)
                      }}
                      className="p-1 rounded hover:bg-zinc-800 text-zinc-400 hover:text-red-400 transition-colors"
                      title="Delete strategy"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                {strategy.created_at && (
                  <div className="text-xs text-zinc-600 mt-2">
                    Created: {new Date(strategy.created_at).toLocaleDateString()}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* New Strategy modal */}
      {newStrategyModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => !creating && setNewStrategyModalOpen(false)}>
          <div className="rounded-xl bg-zinc-900 border border-zinc-700 p-6 w-full max-w-md shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-zinc-200">New Strategy</h3>
              <button
                type="button"
                onClick={() => !creating && setNewStrategyModalOpen(false)}
                disabled={creating}
                className="p-1 rounded hover:bg-zinc-800 text-zinc-400 disabled:opacity-50"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div>
              <label className="block text-sm text-zinc-500 mb-2">Strategy Name *</label>
              <input
                type="text"
                value={strategyName}
                onChange={(e) => setStrategyName(e.target.value)}
                placeholder="e.g. Breakout Strategy v1"
                className="w-full rounded-lg bg-zinc-800 border border-zinc-600 px-4 py-2 text-sm text-zinc-200 placeholder-zinc-500 focus:border-emerald-500 focus:outline-none"
                autoFocus
                disabled={creating}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && strategyName.trim() && !creating) {
                    handleCreateStrategy()
                  }
                }}
              />
            </div>
            
            <div className="flex justify-end gap-2 mt-6">
              <button
                type="button"
                onClick={() => !creating && setNewStrategyModalOpen(false)}
                disabled={creating}
                className="px-4 py-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-zinc-200 text-sm font-medium disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleCreateStrategy}
                disabled={creating || !strategyName.trim()}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium disabled:opacity-50"
              >
                {creating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4" />
                    Create
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
