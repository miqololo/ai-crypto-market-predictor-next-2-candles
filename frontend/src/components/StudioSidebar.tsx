"use client"

import { useState, useEffect } from "react"
import { Bot, SlidersHorizontal, BarChart3, Zap, Save } from "lucide-react"
import type { ParamSpec } from "./panels/ParamPanel"
import { ParamPanel } from "./panels/ParamPanel"
import { ResultsPanel } from "./panels/ResultsPanel"
import { LiveTriggersPanel } from "./panels/LiveTriggersPanel"
import { NewsPanel } from "./panels/NewsPanel"
import { ChainPanel } from "./panels/ChainPanel"
import { ChatUI, type ChatMessage } from "./panels/ChatUI"
import { cn } from "../lib/utils"

export type SidebarPanel = "ai" | "params" | "results" | "triggers" | "news" | "chain"

interface StudioSidebarProps {
  onParamsReceived?: (params: ParamSpec[], code?: string) => void
  onStrategySaved?: (strategyId: string, strategyName: string, strategyFile: string) => void
  paramSpecs: ParamSpec[]
  paramValues: Record<string, number | string | boolean>
  onParamChange: (name: string, value: number | string | boolean) => void
  onParamValuesChange: (values: Record<string, number | string | boolean>) => void
  strategyCode?: string
  onSaveStrategy?: () => void
  projectName?: string
  openAIPanelRef?: React.MutableRefObject<(() => void) | null>
  backtestMetrics?: any
  strategyId?: string  // Strategy ID for updating existing strategy
}

const PANELS: { id: SidebarPanel; icon: React.ElementType; label: string }[] = [
  { id: "ai", icon: Bot, label: "AI" },
  { id: "params", icon: SlidersHorizontal, label: "Params" },
  { id: "results", icon: BarChart3, label: "Results" },
  { id: "triggers", icon: Zap, label: "Triggers" },
]

export function StudioSidebar({
  onParamsReceived,
  onStrategySaved,
  paramSpecs,
  paramValues,
  onParamChange,
  onParamValuesChange,
  strategyCode,
  onSaveStrategy,
  projectName,
  openAIPanelRef,
  backtestMetrics,
  strategyId,
}: StudioSidebarProps) {
  const [activePanel, setActivePanel] = useState<SidebarPanel>("params")
  const [isMobile, setIsMobile] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)")
    setIsMobile(mq.matches)
    const handler = () => setIsMobile(mq.matches)
    mq.addEventListener("change", handler)
    return () => mq.removeEventListener("change", handler)
  }, [])

  useEffect(() => {
    if (openAIPanelRef) {
      openAIPanelRef.current = () => setActivePanel("ai")
      return () => {
        openAIPanelRef.current = null
      }
    }
  }, [openAIPanelRef])

  const compact = true

  const renderPanelContent = () => {
    switch (activePanel) {
      case "ai":
        return (
          <div className="flex min-h-0 flex-1 flex-col">
            <ChatUI
              messages={messages}
              onMessagesChange={setMessages}
              onParamsReceived={onParamsReceived}
              onStrategySaved={onStrategySaved}
              placeholder="Describe strategy..."
              sessionId={`strategy-${projectName || 'default'}`}
              strategyContext={strategyCode}
              strategyId={strategyId}
            />
          </div>
        )
      case "params":
        return (
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <div className="min-h-0 flex-1 overflow-y-auto p-3">
              <ParamPanel
                params={paramSpecs}
                values={paramValues}
                onChange={onParamChange}
                onValuesChange={onParamValuesChange}
                compact={compact}
              />
            </div>
            {strategyCode && onSaveStrategy && (
              <div className="shrink-0 border-t border-zinc-800 p-2">
                <button
                  type="button"
                  onClick={onSaveStrategy}
                  className="flex w-full items-center justify-center gap-2 rounded-md bg-emerald-600 px-3 py-2 text-sm font-medium text-white hover:bg-emerald-500"
                >
                  <Save className="h-4 w-4" />
                  Save strategy
                </button>
              </div>
            )}
          </div>
        )
      case "results":
        return (
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            <ResultsPanel compact={compact} metrics={backtestMetrics} />
          </div>
        )
      case "triggers":
        return (
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            <LiveTriggersPanel compact={compact} />
          </div>
        )
      case "news":
        return (
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            <NewsPanel compact={compact} />
          </div>
        )
      case "chain":
        return (
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            <ChainPanel compact={compact} />
          </div>
        )
      default:
        return null
    }
  }

  const activePanelInfo = PANELS.find((p) => p.id === activePanel)
  const ActiveIcon = activePanelInfo?.icon

  if (isMobile) {
    return (
      <>
        <div className="fixed right-6 z-40 flex flex-col gap-2 md:hidden" style={{ bottom: "max(1.5rem, env(safe-area-inset-bottom))" }}>
          {PANELS.map((p) => {
            const Icon = p.icon
            const isActive = activePanel === p.id
            return (
              <button
                key={p.id}
                onClick={() => setActivePanel(p.id)}
                className={cn(
                  "flex h-12 w-12 items-center justify-center rounded-full shadow-lg touch-manipulation transition-all",
                  isActive 
                    ? "bg-emerald-600 text-white scale-110" 
                    : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                )}
                aria-label={`Open ${p.label}`}
              >
                <Icon className={cn("transition-all", isActive ? "h-6 w-6" : "h-5 w-5")} />
              </button>
            )
          })}
        </div>
        {activePanelInfo && (
          <>
            <div
              className="fixed inset-0 z-40 bg-black/50 md:hidden"
              onClick={() => {}}
              aria-hidden="true"
            />
            <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-sm flex-col border-l border-zinc-800 bg-zinc-950 shadow-xl md:hidden">
              <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-3 py-2">
                <span className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                  {ActiveIcon && <ActiveIcon className="h-5 w-5 text-emerald-400" />}
                  {activePanelInfo.label}
                </span>
              </div>
              {renderPanelContent()}
            </div>
          </>
        )}
      </>
    )
  }

  return (
    <div className="flex h-[calc(100vh-98px)] border-l border-zinc-800 bg-zinc-950 shadow-sm">
      {/* Icons sidebar - always visible */}
      <div className="shrink-0 w-16 border-r border-zinc-800 bg-zinc-900/50">
        <div className="flex flex-col gap-1 p-2">
          {PANELS.map((p) => {
            const Icon = p.icon
            const isActive = activePanel === p.id
            return (
              <button
                key={p.id}
                onClick={() => setActivePanel(p.id)}
                className={cn(
                  "flex flex-col items-center gap-0.5 rounded-md p-2 touch-manipulation transition-all",
                  isActive 
                    ? "bg-emerald-600/20 hover:bg-emerald-600/30" 
                    : "hover:bg-zinc-800"
                )}
                aria-label={`Open ${p.label}`}
                title={p.label}
              >
                <Icon className={cn(
                  "transition-all",
                  isActive ? "h-6 w-6 text-emerald-400" : "h-5 w-5 text-zinc-500"
                )} />
                <span className={cn(
                  "text-[10px] font-medium transition-all",
                  isActive ? "text-emerald-400" : "text-zinc-500"
                )}>
                  {p.label}
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Panel content - always visible */}
      <div className="flex-1 flex flex-col min-w-[280px] w-80">
        {activePanelInfo && (
          <>
            <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-3 py-2">
              <span className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                {ActiveIcon && <ActiveIcon className="h-5 w-5 text-emerald-400" />}
                {activePanelInfo.label}
              </span>
            </div>
            {renderPanelContent()}
          </>
        )}
      </div>
    </div>
  )
}
