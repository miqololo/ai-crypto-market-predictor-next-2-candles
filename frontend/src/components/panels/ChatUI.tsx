"use client"

import { useState, useCallback, useRef, useEffect } from "react"
import { Check, Copy, ThumbsUp, ThumbsDown, Loader2, Send } from "lucide-react"
import type { ParamSpec } from "./ParamPanel"
import { cn } from "../../lib/utils"

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  sending?: boolean
  streaming?: boolean
  complete?: boolean
  thumbs?: "up" | "down" | null
}

export interface ChatUIProps {
  messages: ChatMessage[]
  onMessagesChange: (messages: ChatMessage[]) => void
  onParamsReceived?: (params: ParamSpec[], code?: string) => void
  onStrategySaved?: (strategyId: string, strategyName: string, strategyFile: string) => void
  placeholder?: string
  sessionId?: string
  strategyContext?: string
  strategyId?: string  // If provided, update existing strategy instead of creating new
}

const API_BASE = "/api"

export function ChatUI({
  messages,
  onMessagesChange,
  onParamsReceived,
  onStrategySaved,
  placeholder = "Describe strategy or create custom indicator...",
  sessionId,
  strategyContext,
  strategyId,
}: ChatUIProps) {
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [copiedId, setCopiedId] = useState<number | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const currentSessionId = useRef<string>(sessionId || `session-${Date.now()}`)

  const scrollToBottom = useCallback(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    })
  }, [])

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isLoading) return

    const userContent = input.trim()
    setInput("")
    setIsLoading(true)
    setStatusMessage(null)

    const userMsg: ChatMessage = {
      role: "user",
      content: userContent,
      sending: true,
    }
    const newMessages: ChatMessage[] = [...messages, userMsg]
    onMessagesChange(newMessages)
    scrollToBottom()

    // Mark user message as sent
    const updatedMessages = newMessages.map((m) => (m.sending ? { ...m, sending: false } : m))
    onMessagesChange(updatedMessages)

    // Create assistant message for streaming
    let assistantMsg: ChatMessage = {
      role: "assistant",
      content: "",
      streaming: true,
      complete: false,
    }
    const finalMessages = [...updatedMessages, assistantMsg]
    onMessagesChange(finalMessages)
    scrollToBottom()

    try {
      const response = await fetch(`${API_BASE}/ai/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userContent,
          session_id: currentSessionId.current,
          strategy_context: strategyContext,
          strategy_id: strategyId,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      if (!reader) {
        throw new Error("No response body")
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6)
            if (!dataStr.trim()) continue

            try {
              const data = JSON.parse(dataStr)

              if (data.type === "status") {
                setStatusMessage(data.message || "")
              } else if (data.type === "content") {
                // Append content to assistant message
                assistantMsg.content += data.content || ""
                const updated = [...updatedMessages, { ...assistantMsg }]
                onMessagesChange(updated)
                scrollToBottom()
              } else if (data.type === "complete") {
                // Finalize message
                let finalMessage = data.message || assistantMsg.content
                
                // Show the full response if code wasn't extracted
                if (!data.code && assistantMsg.content) {
                  finalMessage = assistantMsg.content + (data.message ? `\n\nNote: ${data.message}` : "")
                }
                
                // Add strategy saved info if available
                if (data.strategy_id && data.strategy_name) {
                  finalMessage += `\n\n✅ Strategy saved: "${data.strategy_name}" (ID: ${data.strategy_id})`
                  if (data.strategy_file) {
                    finalMessage += `\n📁 File: ${data.strategy_file}`
                  }
                  
                  // Notify parent component
                  if (onStrategySaved) {
                    onStrategySaved(data.strategy_id, data.strategy_name, data.strategy_file || "")
                  }
                } else if (!data.code) {
                  // If no code was generated, show helpful message
                  finalMessage += `\n\n⚠️ Could not extract Python code from the response. Please ensure the AI generates complete Python code with class definition.`
                }
                
                assistantMsg = {
                  ...assistantMsg,
                  content: finalMessage,
                  streaming: false,
                  complete: true,
                }
                
                // Extract params and code if available
                if (data.params && data.params.length > 0 && onParamsReceived) {
                  onParamsReceived(data.params, data.code)
                }

                const updated = [...updatedMessages, assistantMsg]
                onMessagesChange(updated)
                setStatusMessage(null)
                scrollToBottom()
              } else if (data.type === "error") {
                throw new Error(data.message || "Unknown error")
              } else if (data.type === "session") {
                if (data.session_id) {
                  currentSessionId.current = data.session_id
                }
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e)
            }
          }
        }
      }
    } catch (e) {
      const errorMsg: ChatMessage = {
        role: "assistant",
        content: `Error: ${e instanceof Error ? e.message : String(e)}`,
        complete: true,
      }
      onMessagesChange([...updatedMessages, errorMsg])
      setStatusMessage(null)
      scrollToBottom()
    } finally {
      setIsLoading(false)
      setStatusMessage(null)
    }
  }, [input, isLoading, messages, onMessagesChange, scrollToBottom, onParamsReceived, strategyContext])

  const handleCopy = useCallback((idx: number, content: string) => {
    navigator.clipboard.writeText(content)
    setCopiedId(idx)
    setTimeout(() => setCopiedId(null), 1500)
  }, [])

  const handleThumbs = useCallback(
    (idx: number, value: "up" | "down") => {
      const updated = [...messages]
      if (updated[idx]?.role === "assistant") {
        updated[idx] = { ...updated[idx], thumbs: value }
        onMessagesChange(updated)
      }
    },
    [messages, onMessagesChange]
  )

  return (
    <div className="flex h-full flex-col">
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-y-auto p-3 space-y-3"
      >
        {messages.length === 0 && (
          <div className="space-y-2 text-sm text-zinc-500">
            <p>Describe your strategy to generate it.</p>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}
          >
            <div
              className={cn(
                "max-w-[90%] rounded-xl px-3 py-2.5 text-sm shadow-sm",
                m.role === "user"
                  ? "bg-emerald-600/20 text-zinc-200"
                  : "bg-zinc-800/80 text-zinc-200"
              )}
            >
              {m.role === "user" ? (
                <div className="flex items-center gap-2">
                  <span>{m.content}</span>
                  {m.sending && (
                    <span className="inline-flex items-center gap-1 text-xs text-zinc-500">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      sending…
                    </span>
                  )}
                </div>
              ) : (
                <>
                  <div className="text-sm whitespace-pre-wrap">{m.content}</div>
                  {m.complete && (
                    <div className="mt-2 flex items-center gap-1.5 opacity-70">
                      <span className="flex items-center gap-1 text-xs text-emerald-400">
                        <Check className="h-3.5 w-3.5" /> Done
                      </span>
                      <button
                        onClick={() => handleCopy(i, m.content)}
                        className="rounded p-1 hover:bg-zinc-700/50"
                        aria-label="Copy"
                      >
                        {copiedId === i ? (
                          <Check className="h-3.5 w-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="h-3.5 w-3.5 text-zinc-400" />
                        )}
                      </button>
                      <button
                        onClick={() => handleThumbs(i, "up")}
                        className={cn("rounded p-1 hover:bg-zinc-700/50", m.thumbs === "up" ? "text-emerald-400" : "text-zinc-400")}
                        aria-label="Good"
                      >
                        <ThumbsUp className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => handleThumbs(i, "down")}
                        className={cn("rounded p-1 hover:bg-zinc-700/50", m.thumbs === "down" ? "text-red-400" : "text-zinc-400")}
                        aria-label="Bad"
                      >
                        <ThumbsDown className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="rounded-xl bg-zinc-800/80 px-3 py-2.5 text-sm text-zinc-400">
              {statusMessage ? (
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>{statusMessage}</span>
                </div>
              ) : (
                <Loader2 className="h-4 w-4 animate-spin" />
              )}
            </div>
          </div>
        )}
      </div>

      <div className="shrink-0 border-t border-zinc-800 p-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                sendMessage()
              }
            }}
            placeholder={placeholder}
            className="flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
