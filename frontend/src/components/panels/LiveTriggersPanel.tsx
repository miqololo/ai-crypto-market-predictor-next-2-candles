"use client"

import { cn } from "../../lib/utils"

export function LiveTriggersPanel({ compact = false }: { compact?: boolean }) {
  return (
    <div className={cn("rounded-lg border border-zinc-800 bg-zinc-900/50", compact ? "p-2" : "p-4")}>
      <h2 className={cn("font-semibold text-zinc-200", compact ? "mb-2 text-sm" : "mb-3 text-lg")}>Live Triggers</h2>
      <p className="text-sm text-zinc-500">
        Active triggers will appear here when running in live mode.
      </p>
      <p className="mt-2 text-xs text-zinc-600">
        Enable live mode to start receiving real-time signals.
      </p>
    </div>
  )
}
