"use client"

import { useState } from "react"
import { Wallet, ArrowUpRight, ArrowDownRight, ExternalLink, ChevronRight, Bell, BellRing, Info, Plus, Trash2 } from "lucide-react"
import { cn } from "../../lib/utils"

interface WalletMove {
  id: string
  address: string
  direction: "in" | "out"
  amount: string
  token: string
  valueUsd: string
  time: string
}

const MOCK_WALLETS = [
  {
    id: "1",
    address: "0x742d...8a3f",
    label: "Whale #1",
    balance: "12,450 ETH",
    valueUsd: "$28.2M",
    change24h: 5.2,
  },
]

const MOCK_BIG_MOVES: WalletMove[] = [
  {
    id: "1",
    address: "0x742d...8a3f",
    direction: "in",
    amount: "2,500",
    token: "ETH",
    valueUsd: "$5.7M",
    time: "15m ago",
  },
]

export function ChainPanel({ compact = false }: { compact?: boolean }) {
  return (
    <div className="space-y-4">
      <h2 className={cn("font-semibold text-zinc-200", compact ? "text-sm" : "text-lg")}>Chain Summary</h2>
      <p className="text-xs text-zinc-500">
        Key on-chain metrics and whale activity for traders.
      </p>

      <div className="rounded-lg border border-zinc-800 bg-zinc-800/30 p-3">
        <h3 className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-500">
          <Info className="h-3.5 w-3.5" />
          About Ethereum
        </h3>
        <p className="text-xs text-zinc-300">Ethereum is a decentralized blockchain for smart contracts and dApps.</p>
      </div>

      <div>
        <h3 className="mb-2 flex items-center gap-1.5 text-sm font-medium text-zinc-200">
          <Wallet className="h-4 w-4" />
          Big Money Wallets
        </h3>
        <div className="space-y-2">
          {MOCK_WALLETS.map((w) => (
            <div
              key={w.id}
              className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/50 p-3"
            >
              <div className="min-w-0 flex-1">
                <div className="font-medium text-zinc-200">{w.label ?? w.address}</div>
                <div className="text-xs text-zinc-500">{w.address}</div>
                <div className="mt-1 text-xs text-zinc-400">
                  {w.balance} · {w.valueUsd}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-zinc-200">Big Money Moves</h3>
        <div className="space-y-2">
          {MOCK_BIG_MOVES.map((m) => (
            <div
              key={m.id}
              className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2"
            >
              <div className="flex items-center gap-2">
                {m.direction === "in" ? (
                  <ArrowDownRight className="h-4 w-4 text-emerald-400" />
                ) : (
                  <ArrowUpRight className="h-4 w-4 text-red-400" />
                )}
                <div>
                  <div className="text-sm font-medium text-zinc-200">
                    {m.direction === "in" ? "In" : "Out"} {m.amount} {m.token}
                  </div>
                  <div className="text-[10px] text-zinc-600">
                    {m.address} · {m.time}
                  </div>
                </div>
              </div>
              <span className="text-xs font-medium text-zinc-400">
                {m.valueUsd}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
