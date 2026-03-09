import { Cpu, X } from 'lucide-react'
import { useState } from 'react'

export default function MLs() {

  const [newMLModalOpen, setNewMLModalOpen] = useState(false)
  return (
    <div className="min-h-screen w-full">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between gap-3 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <Cpu className="w-8 h-8 text-emerald-500" />
            <h1 className="text-2xl font-semibold">ML-s</h1>
          </div>
            <div>
              <button
                onClick={() => setNewMLModalOpen(true)}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
              >
                <Cpu className="w-4 h-4" />
                New ML
              </button>
            </div>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-8">
          <p className="text-zinc-400 text-center">ML-s page - Coming soon</p>
        </div>
      </div>
      {newMLModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setNewMLModalOpen(false)}>
          <div className="rounded-xl bg-zinc-900 border border-zinc-700 p-6 w-full max-w-md shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-zinc-200">New ML</h3>
              <button type="button" onClick={() => setNewMLModalOpen(false)} className="p-1 rounded hover:bg-zinc-800 text-zinc-400">
                <X className="w-5 h-5" />
              </button>
            </div>
            <p className="text-sm text-zinc-500 mb-4">Create a new ML model.</p>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => setNewMLModalOpen(false)}
                className="px-4 py-2 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-zinc-200 text-sm font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
