import { useState, useEffect, useRef, useMemo } from 'react'
import { ArrowLeft, FileCode, Loader2, Play, BarChart3, Download, Upload, Save } from 'lucide-react'
import { StudioSidebar } from '../components/StudioSidebar'
import type { ParamSpec } from '../components/panels/ParamPanel'

const API_BASE = '/api'

interface Strategy {
  id: string
  name: string
  strategy_file?: string
  description?: string
  params?: Record<string, unknown>
  created_at?: string
  updated_at?: string
}

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

async function fetchApi(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    const errorText = await res.text()
    throw new Error(errorText || `HTTP ${res.status}`)
  }
  return res.json()
}

interface StrategyDetailsProps {
  strategyId: string
  onBack?: () => void
}

export default function StrategyDetails({ strategyId, onBack }: StrategyDetailsProps) {
  const [strategy, setStrategy] = useState<Strategy | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [paramSpecs, setParamSpecs] = useState<ParamSpec[]>([])
  const [paramValues, setParamValues] = useState<Record<string, number | string | boolean>>({})
  const [strategyCode, setStrategyCode] = useState<string>('')
  const [backtestMetrics, setBacktestMetrics] = useState<any>(null)
  const [backtestLoading, setBacktestLoading] = useState(false)
  const [backtestError, setBacktestError] = useState<string | null>(null)
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null)
  
  // Track original values to detect changes
  const [originalParamValues, setOriginalParamValues] = useState<Record<string, number | string | boolean>>({})
  const [originalStrategyCode, setOriginalStrategyCode] = useState<string>('')
  const [saving, setSaving] = useState(false)
  
  // Backtest settings
  const [symbol] = useState('BTC/USDT:USDT')
  const [timeframe, setTimeframe] = useState('1h')
  const [limit, setLimit] = useState(500)
  const [initialCapital, setInitialCapital] = useState(10000)
  const [stopLoss, setStopLoss] = useState(1.0) // Stop loss in percentage (default 1%)
  const [takeProfit, setTakeProfit] = useState(3.0) // Take profit in percentage (default 3% for 1:3 ratio)
  
  const openAIPanelRef = useRef<(() => void) | null>(null)
  
  // Check if there are unsaved changes
  const hasUnsavedChanges = useMemo(() => {
    // Check if params changed
    const paramsChanged = JSON.stringify(paramValues) !== JSON.stringify(originalParamValues)
    // Check if code changed
    const codeChanged = strategyCode !== originalStrategyCode && strategyCode.length > 0
    return paramsChanged || codeChanged
  }, [paramValues, originalParamValues, strategyCode, originalStrategyCode])

  useEffect(() => {
    const loadStrategy = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await fetchApi(`/strategies/${strategyId}`)
        setStrategy(data)
        
        // If strategy has params, convert them to ParamSpec format
        if (data.params) {
          const specs: ParamSpec[] = Object.entries(data.params).map(([key, value]: [string, any]) => ({
            name: key,
            type: typeof value === 'boolean' ? 'bool' : typeof value === 'number' ? 'float' : 'string',
            default: value,
            default_value: value,
          }))
          setParamSpecs(specs)
          const params = data.params as Record<string, number | string | boolean>
          setParamValues(params)
          setOriginalParamValues({ ...params }) // Store original for comparison
        }
        
        // Load strategy code if file exists
        if (data.strategy_file) {
          try {
            const codeResponse = await fetch(`${API_BASE}/strategies/${strategyId}/file`)
            if (codeResponse.ok) {
              const codeContent = await codeResponse.text()
              setStrategyCode(codeContent)
              setOriginalStrategyCode(codeContent) // Store original for comparison
            }
          } catch (e) {
            console.error('Failed to load strategy code:', e)
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setLoading(false)
      }
    }

    if (strategyId) {
      loadStrategy()
    }
  }, [strategyId])

  const handleParamChange = (name: string, value: number | string | boolean) => {
    setParamValues((prev) => ({ ...prev, [name]: value }))
  }

  const handleParamValuesChange = (values: Record<string, number | string | boolean>) => {
    setParamValues(values)
  }

  const handleParamsReceived = async (params: ParamSpec[], code?: string) => {
    setParamSpecs(params)
    if (code) {
      setStrategyCode(code)
      // Don't update originalStrategyCode here - let user save manually
    }
    
    // Update param values from new params
    if (params.length > 0) {
      const paramsDict: Record<string, number | string | boolean> = {}
      params.forEach(param => {
        paramsDict[param.name] = param.default as number | string | boolean
      })
      setParamValues(paramsDict)
      // Don't update originalParamValues here - let user save manually
    }
  }

  const handleStrategySaved = async (strategyId: string, _strategyName: string, strategyFile: string) => {
    // Reload strategy to get updated info
    try {
      const data = await fetchApi(`/strategies/${strategyId}`)
      setStrategy(data)
      
      // Update param specs if available
      if (data.params) {
        const specs: ParamSpec[] = Object.entries(data.params).map(([key, value]: [string, any]) => ({
          name: key,
          type: typeof value === 'boolean' ? 'bool' : typeof value === 'number' ? 'float' : 'string',
          default: value,
          default_value: value,
        }))
        setParamSpecs(specs)
        const params = data.params as Record<string, number | string | boolean>
        setParamValues(params)
        setOriginalParamValues({ ...params }) // Update original after save
      }
      
      // Reload code if file exists
      if (strategyFile) {
        try {
          const codeResponse = await fetch(`${API_BASE}/strategies/${strategyId}/file`)
          if (codeResponse.ok) {
            const codeContent = await codeResponse.text()
            setStrategyCode(codeContent)
            setOriginalStrategyCode(codeContent) // Update original after save
          }
        } catch (e) {
          console.error('Failed to reload strategy code:', e)
        }
      }
    } catch (e) {
      console.error('Failed to reload strategy:', e)
    }
  }

  const handleSaveStrategy = async () => {
    if (!strategy) return
    
    setSaving(true)
    try {
      // If code changed and we have a file, update the file
      if (strategyCode !== originalStrategyCode && strategyCode.length > 0 && strategy.strategy_file) {
        // Update the file content
        const formData = new FormData()
        const blob = new Blob([strategyCode], { type: 'text/python' })
        formData.append('file', blob, 'strategy.py')
        
        await fetch(`${API_BASE}/strategies/${strategyId}/file`, {
          method: 'POST',
          body: formData,
        })
      }
      
      // Update params
      await fetchApi(`/strategies/${strategyId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          params: paramValues,
        }),
      })
      
      // Reload strategy to get updated info
      const data = await fetchApi(`/strategies/${strategyId}`)
      setStrategy(data)
      
      // Update original values to reflect saved state
      setOriginalParamValues({ ...paramValues })
      if (strategyCode.length > 0) {
        setOriginalStrategyCode(strategyCode)
      }
      
      // Show success feedback (you could add a toast notification here)
      console.log('Strategy saved successfully')
    } catch (e) {
      console.error('Failed to save strategy:', e)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const handleDownloadFile = async () => {
    if (!strategy?.strategy_file) return
    
    try {
      const response = await fetch(`${API_BASE}/strategies/${strategyId}/file`)
      if (!response.ok) {
        throw new Error('Failed to download file')
      }
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${strategy.name}.py`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (e) {
      console.error('Failed to download file:', e)
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const handleUploadFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    
    if (!file.name.endsWith('.py')) {
      setError('File must be a Python file (.py)')
      return
    }
    
    try {
      setError(null)
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await fetch(`${API_BASE}/strategies/${strategyId}/file`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || `HTTP ${response.status}`)
      }
      
      const result = await response.json()
      
      // Reload strategy to get updated file path and params
      const data = await fetchApi(`/strategies/${strategyId}`)
      setStrategy(data)
      
      // Update parameters if they were extracted during refactoring
      if (result.params && Object.keys(result.params).length > 0) {
        const specs: ParamSpec[] = Object.entries(result.params).map(([key, value]: [string, any]) => ({
          name: key,
          type: typeof value === 'boolean' ? 'bool' : typeof value === 'number' ? 'float' : 'string',
          default: value,
          default_value: value,
        }))
        setParamSpecs(specs)
        const params = result.params as Record<string, number | string | boolean>
        setParamValues(params)
        setOriginalParamValues({ ...params }) // Update original after upload
      } else if (data.params && Object.keys(data.params).length > 0) {
        // Fallback to params from reloaded strategy
        const specs: ParamSpec[] = Object.entries(data.params).map(([key, value]: [string, any]) => ({
          name: key,
          type: typeof value === 'boolean' ? 'bool' : typeof value === 'number' ? 'float' : 'string',
          default: value,
          default_value: value,
        }))
        setParamSpecs(specs)
        setParamValues(data.params as Record<string, number | string | boolean>)
      }
      
      // Try to load the refactored code content if available
      if (result.strategy_file) {
        try {
          const fileResponse = await fetch(`${API_BASE}/strategies/${strategyId}/file`)
          if (fileResponse.ok) {
            const codeContent = await fileResponse.text()
            setStrategyCode(codeContent)
            setOriginalStrategyCode(codeContent) // Update original after upload
          }
        } catch (e) {
          console.warn('Could not load refactored code:', e)
        }
      }
      
      console.log('File uploaded and refactored successfully:', result.message || 'Upload complete')
      
      // Clear the input
      event.target.value = ''
    } catch (e) {
      console.error('Failed to upload file:', e)
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const handleRunBacktest = async () => {
    if (!strategy) return
    
    setBacktestLoading(true)
    setBacktestError(null)
    setBacktestResult(null)

    try {
      const strategyFile = strategy.strategy_file || 'app/strategies/breakout_strategy.py'
      
      // Validate inputs before sending
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
        strategy_params: paramValues || {},
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
      setBacktestResult(data)
      
      // Update metrics for ResultsPanel
      setBacktestMetrics({
        total_return: data.total_return,
        sharpe_ratio: data.sharpe_ratio,
        max_drawdown: data.max_drawdown,
        win_rate: data.win_rate || 0,
        num_trades: data.total_trades,
        total_profit: data.total_profit,
        sortino_ratio: data.sortino_ratio,
        calmar_ratio: data.calmar_ratio,
        profit_factor: data.profit_factor,
        final_value: data.final_value,
      })
      
      // Store full result for comprehensive display
      setBacktestResult(data)
    } catch (e) {
      setBacktestError(e instanceof Error ? e.message : String(e))
    } finally {
      setBacktestLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="flex items-center gap-2 text-zinc-400">
          <Loader2 className="w-4 h-4 animate-spin" />
          Loading strategy...
        </div>
      </div>
    )
  }

  if (error || !strategy) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-4">{error || 'Strategy not found'}</p>
          {onBack && (
            <button
              onClick={onBack}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Strategies
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen overflow-y-auto">
      {/* Main content */}
      <div className="flex-1 flex flex-col ">
        {/* Header */}
        <div className="shrink-0 border-b border-zinc-800 px-6 py-4">
          <div className="flex items-center gap-4">
            {onBack && (
              <button
                onClick={onBack}
                className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition-colors"
                aria-label="Back"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
            )}
            <div className="flex items-center gap-3 flex-1">
              <FileCode className="w-6 h-6 text-emerald-500" />
              <div className="flex-1">
                <h1 className="text-xl font-semibold text-zinc-200">{strategy.name}</h1>
                {strategy.created_at && (
                  <p className="text-sm text-zinc-500">
                    Created: {new Date(strategy.created_at).toLocaleDateString()}
                  </p>
                )}
              </div>
              {hasUnsavedChanges && (
                <button
                  onClick={handleSaveStrategy}
                  disabled={saving}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {saving ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4" />
                      Save Changes
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Content area */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {strategy.description && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <h2 className="text-sm font-semibold text-zinc-300 mb-2">Description</h2>
                <p className="text-sm text-zinc-400">{strategy.description}</p>
              </div>
            )}

            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-sm font-semibold text-zinc-300">Strategy File</h2>
                <div className="flex gap-2">
                  {strategy.strategy_file && (
                    <button
                      onClick={handleDownloadFile}
                      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm font-medium transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                  )}
                  <label className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition-colors cursor-pointer">
                    <Upload className="w-4 h-4" />
                    Upload
                    <input
                      type="file"
                      accept=".py"
                      onChange={handleUploadFile}
                      className="hidden"
                    />
                  </label>
                </div>
              </div>
              {strategy.strategy_file ? (
                <code className="text-sm text-zinc-400">{strategy.strategy_file}</code>
              ) : (
                <p className="text-sm text-zinc-500">No strategy file uploaded yet</p>
              )}
            </div>

            {/* Backtest Section */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-6">
              <div className="flex items-center gap-3 mb-6">
                <BarChart3 className="w-6 h-6 text-emerald-500" />
                <h2 className="text-lg font-semibold text-zinc-200">Backtest</h2>
              </div>

              {backtestError && (
                <div className="rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 text-sm mb-4">
                  {backtestError}
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
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

              <button
                onClick={handleRunBacktest}
                disabled={backtestLoading}
                className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 px-6 py-3 text-sm font-medium text-white disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {backtestLoading ? (
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

              {backtestResult && (
                <div className="mt-6 pt-6 border-t border-zinc-800">
                  <h3 className="text-sm font-semibold text-zinc-300 mb-4">Backtest Results</h3>
                  
                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    <div className="rounded-lg bg-zinc-800/50 p-3">
                      <div className="text-xs text-zinc-500 mb-1">Total Return</div>
                      <div className={`text-lg font-bold ${backtestResult.total_return != null && typeof backtestResult.total_return === 'number' && backtestResult.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {backtestResult.total_return != null && typeof backtestResult.total_return === 'number' ? (backtestResult.total_return * 100).toFixed(2) + '%' : 'N/A'}
                      </div>
                      {backtestResult.total_profit != null && typeof backtestResult.total_profit === 'number' && (
                        <div className="text-xs text-zinc-500 mt-1">
                          ${backtestResult.total_profit.toFixed(2)}
                        </div>
                      )}
                    </div>
                    <div className="rounded-lg bg-zinc-800/50 p-3">
                      <div className="text-xs text-zinc-500 mb-1">Sharpe Ratio</div>
                      <div className="text-lg font-bold text-zinc-200">
                        {backtestResult.sharpe_ratio != null && typeof backtestResult.sharpe_ratio === 'number' ? backtestResult.sharpe_ratio.toFixed(2) : 'N/A'}
                      </div>
                    </div>
                    <div className="rounded-lg bg-zinc-800/50 p-3">
                      <div className="text-xs text-zinc-500 mb-1">Max Drawdown</div>
                      <div className="text-lg font-bold text-red-400">
                        {backtestResult.max_drawdown != null && typeof backtestResult.max_drawdown === 'number' ? (backtestResult.max_drawdown * 100).toFixed(2) + '%' : 'N/A'}
                      </div>
                    </div>
                    <div className="rounded-lg bg-zinc-800/50 p-3">
                      <div className="text-xs text-zinc-500 mb-1">Total Trades</div>
                      <div className="text-lg font-bold text-zinc-200">
                        {backtestResult.total_trades}
                      </div>
                      {backtestResult.win_rate != null && typeof backtestResult.win_rate === 'number' && (
                        <div className="text-xs text-zinc-500 mt-1">
                          {(backtestResult.win_rate * 100).toFixed(1)}% win rate
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Additional Performance Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 mb-4">
                    {backtestResult.sortino_ratio != null && typeof backtestResult.sortino_ratio === 'number' && backtestResult.sortino_ratio !== 0 && (
                      <div className="rounded-lg bg-zinc-800/30 p-2">
                        <div className="text-xs text-zinc-500">Sortino Ratio</div>
                        <div className="text-sm font-semibold text-zinc-200">{backtestResult.sortino_ratio.toFixed(2)}</div>
                      </div>
                    )}
                    {backtestResult.calmar_ratio != null && typeof backtestResult.calmar_ratio === 'number' && backtestResult.calmar_ratio !== 0 && (
                      <div className="rounded-lg bg-zinc-800/30 p-2">
                        <div className="text-xs text-zinc-500">Calmar Ratio</div>
                        <div className="text-sm font-semibold text-zinc-200">{backtestResult.calmar_ratio.toFixed(2)}</div>
                      </div>
                    )}
                    {backtestResult.annual_return != null && typeof backtestResult.annual_return === 'number' && (
                      <div className="rounded-lg bg-zinc-800/30 p-2">
                        <div className="text-xs text-zinc-500">Annual Return</div>
                        <div className={`text-sm font-semibold ${backtestResult.annual_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {(backtestResult.annual_return * 100).toFixed(2)}%
                        </div>
                      </div>
                    )}
                    {backtestResult.volatility != null && typeof backtestResult.volatility === 'number' && (
                      <div className="rounded-lg bg-zinc-800/30 p-2">
                        <div className="text-xs text-zinc-500">Volatility</div>
                        <div className="text-sm font-semibold text-zinc-200">{(backtestResult.volatility * 100).toFixed(2)}%</div>
                      </div>
                    )}
                    {backtestResult.profit_factor != null && typeof backtestResult.profit_factor === 'number' && backtestResult.profit_factor !== 0 && (
                      <div className="rounded-lg bg-zinc-800/30 p-2">
                        <div className="text-xs text-zinc-500">Profit Factor</div>
                        <div className="text-sm font-semibold text-zinc-200">
                          {backtestResult.profit_factor === 999999.0 ? '∞' : backtestResult.profit_factor.toFixed(2)}
                        </div>
                      </div>
                    )}
                    {backtestResult.expectancy != null && typeof backtestResult.expectancy === 'number' && (
                      <div className="rounded-lg bg-zinc-800/30 p-2">
                        <div className="text-xs text-zinc-500">Expectancy</div>
                        <div className={`text-sm font-semibold ${backtestResult.expectancy >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {backtestResult.expectancy.toFixed(4)}
                        </div>
                      </div>
                    )}
                    {backtestResult.avg_trade_return != null && typeof backtestResult.avg_trade_return === 'number' && (
                      <div className="rounded-lg bg-zinc-800/30 p-2">
                        <div className="text-xs text-zinc-500">Avg Trade Return</div>
                        <div className={`text-sm font-semibold ${backtestResult.avg_trade_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {(backtestResult.avg_trade_return * 100).toFixed(2)}%
                        </div>
                      </div>
                    )}
                    {backtestResult.final_value != null && typeof backtestResult.final_value === 'number' && (
                      <div className="rounded-lg bg-zinc-800/30 p-2">
                        <div className="text-xs text-zinc-500">Final Value</div>
                        <div className="text-sm font-semibold text-zinc-200">${backtestResult.final_value.toFixed(2)}</div>
                      </div>
                    )}
                  </div>

                  {/* Trade Statistics */}
                  {(backtestResult.winning_trades !== undefined || backtestResult.best_trade !== undefined) && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
                      {backtestResult.winning_trades !== undefined && (
                        <div className="rounded-lg bg-zinc-800/30 p-2">
                          <div className="text-xs text-zinc-500">Winning Trades</div>
                          <div className="text-sm font-semibold text-emerald-400">{backtestResult.winning_trades}</div>
                        </div>
                      )}
                      {backtestResult.losing_trades !== undefined && (
                        <div className="rounded-lg bg-zinc-800/30 p-2">
                          <div className="text-xs text-zinc-500">Losing Trades</div>
                          <div className="text-sm font-semibold text-red-400">{backtestResult.losing_trades}</div>
                        </div>
                      )}
                      {backtestResult.best_trade != null && typeof backtestResult.best_trade === 'number' && (
                        <div className="rounded-lg bg-zinc-800/30 p-2">
                          <div className="text-xs text-zinc-500">Best Trade</div>
                          <div className={`text-sm font-semibold ${backtestResult.best_trade >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {(backtestResult.best_trade * 100).toFixed(2)}%
                          </div>
                        </div>
                      )}
                      {backtestResult.worst_trade != null && typeof backtestResult.worst_trade === 'number' && (
                        <div className="rounded-lg bg-zinc-800/30 p-2">
                          <div className="text-xs text-zinc-500">Worst Trade</div>
                          <div className={`text-sm font-semibold ${backtestResult.worst_trade >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {(backtestResult.worst_trade * 100).toFixed(2)}%
                          </div>
                        </div>
                      )}
                      {backtestResult.avg_win != null && typeof backtestResult.avg_win === 'number' && backtestResult.avg_win !== 0 && (
                        <div className="rounded-lg bg-zinc-800/30 p-2">
                          <div className="text-xs text-zinc-500">Avg Win</div>
                          <div className="text-sm font-semibold text-emerald-400">{(backtestResult.avg_win * 100).toFixed(2)}%</div>
                        </div>
                      )}
                      {backtestResult.avg_loss != null && typeof backtestResult.avg_loss === 'number' && backtestResult.avg_loss !== 0 && (
                        <div className="rounded-lg bg-zinc-800/30 p-2">
                          <div className="text-xs text-zinc-500">Avg Loss</div>
                          <div className="text-sm font-semibold text-red-400">{(backtestResult.avg_loss * 100).toFixed(2)}%</div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Equity Curve and Drawdowns Info */}
                  {backtestResult.equity_curve && backtestResult.equity_curve.length > 0 && (
                    <div className="mb-4 rounded-lg bg-zinc-800/30 p-3">
                      <div className="text-xs text-zinc-500 mb-2">Equity Curve</div>
                      <div className="text-xs text-zinc-400 font-mono">
                        {backtestResult.equity_curve.length} data points
                        {backtestResult.final_value != null && typeof backtestResult.final_value === 'number' && (
                          <span className="ml-2">
                            Final: ${backtestResult.final_value.toFixed(2)}
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Trades Table */}
                  {backtestResult.trades && backtestResult.trades.length > 0 && (
                    <div className="mb-4 rounded-lg bg-zinc-800/30 p-3">
                      <div className="text-xs text-zinc-500 mb-2">
                        Recent Trades ({backtestResult.trades.length} of {backtestResult.total_trades})
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
                            {backtestResult.trades.slice(0, 20).map((trade, idx) => (
                              <tr key={idx} className="border-b border-zinc-800/50">
                                <td className="py-1">${trade.entry_price != null && typeof trade.entry_price === 'number' ? trade.entry_price.toFixed(2) : 'N/A'}</td>
                                <td className="py-1">${trade.exit_price != null && typeof trade.exit_price === 'number' ? trade.exit_price.toFixed(2) : 'N/A'}</td>
                                <td className={`text-right py-1 ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                  ${trade.pnl != null && typeof trade.pnl === 'number' ? trade.pnl.toFixed(2) : 'N/A'}
                                </td>
                                <td className={`text-right py-1 ${trade.return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                  {trade.return != null && typeof trade.return === 'number' ? (trade.return * 100).toFixed(2) + '%' : 'N/A'}
                                </td>
                                <td className="text-right py-1 text-zinc-400">{trade.duration || 'N/A'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Full Stats */}
                  {backtestResult.stats && Object.keys(backtestResult.stats).length > 0 && (
                    <details className="mb-4 rounded-lg bg-zinc-800/30 p-3">
                      <summary className="text-xs text-zinc-400 cursor-pointer hover:text-zinc-300">
                        View All Statistics ({Object.keys(backtestResult.stats).length} metrics)
                      </summary>
                      <div className="mt-2 grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                        {Object.entries(backtestResult.stats).map(([key, value]) => (
                          <div key={key} className="flex justify-between border-b border-zinc-800/50 pb-1">
                            <span className="text-zinc-500">{key}</span>
                            <span className="text-zinc-300 font-mono">{String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}

                  <p className="mt-4 text-xs text-zinc-500">
                    View detailed results in the Results panel on the right sidebar.
                  </p>
                </div>
              )}
            </div>

            {paramSpecs.length === 0 && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-8 text-center">
                <p className="text-zinc-500">No parameters configured yet. Use the AI panel to generate strategy parameters.</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right sidebar */}
      <StudioSidebar
        onParamsReceived={handleParamsReceived}
        onStrategySaved={handleStrategySaved}
        paramSpecs={paramSpecs}
        paramValues={paramValues}
        onParamChange={handleParamChange}
        onParamValuesChange={handleParamValuesChange}
        strategyCode={strategyCode}
        onSaveStrategy={handleSaveStrategy}
        projectName={strategy.name}
        openAIPanelRef={openAIPanelRef}
        backtestMetrics={backtestMetrics}
        strategyId={strategyId}
      />
    </div>
  )
}
