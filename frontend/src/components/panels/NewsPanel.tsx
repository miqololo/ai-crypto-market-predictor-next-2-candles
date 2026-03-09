"use client"

import { ExternalLink, TrendingUp, TrendingDown, Minus } from "lucide-react"
import { cn } from "../../lib/utils"

export type NewsSentiment = "bullish" | "bearish" | "neutral"

export interface NewsItem {
  id: string
  title: string
  description: string
  sentiment: NewsSentiment
  url: string
  source?: string
  time?: string
}

const MOCK_NEWS: NewsItem[] = [
  {
    id: "1",
    title: "ETH ETF Approval Expected This Week",
    description: "Analysts predict SEC approval for spot Ethereum ETFs could drive significant inflows.",
    sentiment: "bullish",
    url: "#",
    source: "CoinDesk",
    time: "2h ago",
  },
  {
    id: "2",
    title: "Large Whale Transfers 50K ETH to Exchange",
    description: "On-chain data shows major holder moving tokens, potentially signaling sell pressure.",
    sentiment: "bearish",
    url: "#",
    source: "CryptoQuant",
    time: "4h ago",
  },
]

const SENTIMENT_CONFIG: Record<
  NewsSentiment,
  { color: string; bg: string; icon: typeof TrendingUp; label: string }
> = {
  bullish: {
    color: "text-emerald-400",
    bg: "bg-emerald-500/15",
    icon: TrendingUp,
    label: "Bullish",
  },
  bearish: {
    color: "text-red-400",
    bg: "bg-red-500/15",
    icon: TrendingDown,
    label: "Bearish",
  },
  neutral: {
    color: "text-zinc-500",
    bg: "bg-zinc-500/15",
    icon: Minus,
    label: "Neutral",
  },
}

export function NewsPanel({ compact = false }: { compact?: boolean }) {
  return (
    <div className="space-y-4">
      <h2 className={cn("font-semibold text-zinc-200", compact ? "text-sm" : "text-lg")}>News & Events</h2>
      <p className="text-xs text-zinc-500">
        Market news and events that may impact price action.
      </p>

      <div className="space-y-2">
        {MOCK_NEWS.map((item) => (
          <NewsCard key={item.id} item={item} compact={compact} />
        ))}
      </div>
    </div>
  )
}

function NewsCard({ item, compact }: { item: NewsItem; compact: boolean }) {
  const config = SENTIMENT_CONFIG[item.sentiment]
  const Icon = config.icon

  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 transition-colors hover:bg-zinc-800/50"
    >
      <div className="flex items-start gap-2">
        <span
          className={cn("mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded", config.bg, config.color)}
          title={config.label}
        >
          <Icon className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <h3 className={cn("font-medium text-zinc-200", compact ? "text-sm" : "text-base")}>
            {item.title}
          </h3>
          <p className="mt-0.5 line-clamp-2 text-xs text-zinc-500">
            {item.description}
          </p>
          <div className="mt-2 flex items-center justify-between gap-2">
            <span className="text-[10px] text-zinc-600">
              {item.source}
              {item.time && ` · ${item.time}`}
            </span>
            <ExternalLink className="h-3 w-3 shrink-0 text-zinc-600" />
          </div>
        </div>
      </div>
    </a>
  )
}
