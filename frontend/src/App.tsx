import { useNavigate, useParams, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import {
  TrendingUp,
  Cpu,
  BarChart3,
  Layers,
  Network,
  FileText,
  TreePine,
} from 'lucide-react'
import LLM1Similarity from './pages/LLM1Similarity'
import RFPredictor from './pages/RFPredictor'
import RFBacktest from './pages/RFBacktest'
import StrategyBacktest from './pages/StrategyBacktest'
import Strategies from './pages/Strategies'
import MLs from './pages/MLs'
import Combinations from './pages/Combinations'
import StrategyDetails from './pages/StrategyDetails'
import OCR from './pages/OCR'

function StrategyDetailsRoute() {
  const { strategyId } = useParams<{ strategyId: string }>()
  const navigate = useNavigate()
  
  if (!strategyId) {
    return <div>Strategy ID not found</div>
  }
  
  return (
    <StrategyDetails
      strategyId={strategyId}
      onBack={() => {
        navigate('/strategies')
      }}
    />
  )
}

function App() {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <div className="min-h-screen bg-[#0f0f12] text-zinc-200">
      <header className="border-b border-zinc-800/50 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-8 h-8 text-emerald-500" />
            <h1 className="text-xl font-semibold tracking-tight">
              Candle Pattern Forecasting
            </h1>
          </div>
          <nav className="flex gap-2 flex-wrap">
            <NavLink
              to="/"
              className={({ isActive }) =>
                `inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700'
                }`
              }
            >
              <Cpu className="w-4 h-4" />
              Similarity Models
            </NavLink>
            <NavLink
              to="/backtest"
              className={({ isActive }) =>
                `inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700'
                }`
              }
            >
              <BarChart3 className="w-4 h-4" />
              Strategy Backtest
            </NavLink>
            <NavLink
              to="/strategies"
              className={({ isActive }) => {
                const isStrategiesPage = location.pathname.startsWith('/strategies')
                return `inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive || isStrategiesPage
                    ? 'bg-emerald-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700'
                }`
              }}
            >
              <Layers className="w-4 h-4" />
              Strategies
            </NavLink>
            <NavLink
              to="/combinations"
              className={({ isActive }) =>
                `inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700'
                }`
              }
            >
              <Network className="w-4 h-4" />
              Combinations
            </NavLink>
            <NavLink
              to="/rf"
              className={({ isActive }) =>
                `inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700'
                }`
              }
            >
              <TreePine className="w-4 h-4" />
              RF Predictor
            </NavLink>
            <NavLink
              to="/rf-backtest"
              className={({ isActive }) =>
                `inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700'
                }`
              }
            >
              <BarChart3 className="w-4 h-4" />
              RF Backtest
            </NavLink>
            <NavLink
              to="/ocr"
              className={({ isActive }) =>
                `inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700'
                }`
              }
            >
              <FileText className="w-4 h-4" />
              OCR Scanner
            </NavLink>
          </nav>
        </div>
        <p className="text-sm text-zinc-500 mt-1">
          Similarity-based crypto futures • RAG + Qwen2.5 • FAISS + DTW
        </p>
      </header>

      <main className="w-full mx-auto">
        <Routes>
          <Route path="/" element={<LLM1Similarity />} />
          <Route path="/backtest" element={<StrategyBacktest />} />
          <Route
            path="/strategies"
            element={
              <Strategies
                onStrategyClick={(id) => {
                  navigate(`/strategies/${id}`)
                }}
              />
            }
          />
          <Route
            path="/strategies/:strategyId"
            element={<StrategyDetailsRoute />}
          />
          <Route path="/mls" element={<MLs />} />
          <Route path="/rf" element={<RFPredictor />} />
          <Route path="/rf-backtest" element={<RFBacktest />} />
          <Route path="/combinations" element={<Combinations />} />
          <Route path="/ocr" element={<OCR />} />
        </Routes>
      </main>

      {/* New ML modal */}
      
    </div>
  )
}

export default App
