"use client"

import { useState, useCallback, useEffect } from "react"
import { Slider } from "../ui/slider"
import { Input } from "../ui/input"
import { Switch } from "../ui/switch"
import { Button } from "../ui/button"
import { RotateCcw } from "lucide-react"
import { cn } from "../../lib/utils"

export interface ParamSpec {
  name: string
  type: string
  default?: number | string | boolean
  default_value?: number | string | boolean
  min?: number
  max?: number
  step?: number
  description?: string
}

interface ParamPanelProps {
  params: ParamSpec[]
  values?: Record<string, number | string | boolean>
  onChange?: (name: string, value: number | string | boolean) => void
  onValuesChange?: (values: Record<string, number | string | boolean>) => void
  compact?: boolean
}

function getDefault(spec: ParamSpec): number | string | boolean {
  const d = spec.default ?? spec.default_value
  return d ?? 0
}

function getInitialValues(params: ParamSpec[]): Record<string, number | string | boolean> {
  return Object.fromEntries(params.map((p) => [p.name, getDefault(p)]))
}

export function ParamPanel({
  params,
  values: controlledValues,
  onChange,
  onValuesChange,
  compact = false,
}: ParamPanelProps) {
  const [internalValues, setInternalValues] = useState<Record<string, number | string | boolean>>(
    () => getInitialValues(params)
  )

  const values = controlledValues ?? internalValues

  useEffect(() => {
    if (!controlledValues && params.length > 0) {
      setInternalValues(getInitialValues(params))
    }
  }, [params, controlledValues])

  const updateParam = useCallback(
    (name: string, value: number | string | boolean) => {
      const next = { ...values, [name]: value }
      if (!controlledValues) {
        setInternalValues(next)
      }
      onChange?.(name, value)
      onValuesChange?.(next)
    },
    [values, controlledValues, onChange, onValuesChange]
  )

  const resetToDefault = useCallback(
    (name: string) => {
      const spec = params.find((p) => p.name === name)
      if (spec) {
        updateParam(name, getDefault(spec))
      }
    },
    [params, updateParam]
  )

  const resetAll = useCallback(() => {
    const next = getInitialValues(params)
    if (!controlledValues) {
      setInternalValues(next)
    }
    onValuesChange?.(next)
    params.forEach((p) => onChange?.(p.name, getDefault(p)))
  }, [params, controlledValues, onChange, onValuesChange])

  if (params.length === 0) {
    return (
      <div className={cn("rounded-lg border border-zinc-800 bg-zinc-900/50", compact ? "p-2" : "p-4")}>
        <h2 className={cn("font-semibold text-zinc-200", compact ? "mb-1 text-sm" : "mb-3 text-lg")}>Parameters</h2>
        <p className="text-sm text-zinc-500">No editable parameters found.</p>
      </div>
    )
  }

  return (
    <div className={cn("rounded-lg border border-zinc-800 bg-zinc-900/50", compact ? "p-2" : "p-4")}>
      <div className={cn("flex items-center justify-between", compact ? "mb-2" : "mb-3")}>
        <h2 className={cn("font-semibold text-zinc-200", compact ? "text-sm" : "text-lg")}>Parameters</h2>
        <Button variant="ghost" size="sm" onClick={resetAll} className={compact ? "h-6 gap-0.5 px-1.5 text-xs" : "h-8 gap-1 px-2"}>
          <RotateCcw className="h-3.5 w-3.5" />
          Reset all
        </Button>
      </div>
      <div className={compact ? "space-y-2" : "space-y-4"}>
        {params.map((spec) => (
          <div key={spec.name} className={compact ? "space-y-1" : "space-y-2"}>
            <div className="flex items-center justify-between gap-2">
              <label className="text-sm font-medium text-zinc-300">
                {spec.name}
                {spec.description && (
                  <span className="ml-1 font-normal text-zinc-500">
                    ({spec.description})
                  </span>
                )}
              </label>
              <div className="flex items-center gap-1">
                <span className="min-w-[3rem] text-right font-mono text-sm text-zinc-400">
                  {String(values[spec.name] ?? getDefault(spec))}
                </span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={() => resetToDefault(spec.name)}
                  title="Reset to default"
                >
                  <RotateCcw className="h-3 w-3" />
                </Button>
              </div>
            </div>

            {spec.type === "int" || spec.type === "float" ? (
              <div className="flex items-center gap-3">
                <Slider
                  value={[Number(values[spec.name] ?? getDefault(spec))]}
                  onValueChange={([v]) => updateParam(spec.name, spec.type === "int" ? Math.round(v) : v)}
                  min={spec.min ?? 0}
                  max={spec.max ?? 100}
                  step={spec.step ?? (spec.type === "int" ? 1 : 0.01)}
                  className="flex-1"
                />
                <Input
                  type="number"
                  value={String(values[spec.name] ?? getDefault(spec))}
                  onChange={(e) => {
                    const val = e.target.value
                    if (val === '') {
                      updateParam(spec.name, spec.type === "int" ? 0 : 0.0)
                      return
                    }
                    const numVal = spec.type === "int" ? parseInt(val, 10) : parseFloat(val)
                    if (!isNaN(numVal)) {
                      // Apply min/max constraints
                      const min = spec.min ?? (spec.type === "int" ? -Infinity : -Infinity)
                      const max = spec.max ?? (spec.type === "int" ? Infinity : Infinity)
                      const constrainedVal = Math.max(min, Math.min(max, numVal))
                      updateParam(spec.name, spec.type === "int" ? Math.round(constrainedVal) : constrainedVal)
                    }
                  }}
                  onBlur={(e) => {
                    // Ensure value is valid on blur
                    const val = e.target.value
                    if (val === '' || isNaN(Number(val))) {
                      updateParam(spec.name, getDefault(spec))
                    }
                  }}
                  min={spec.min}
                  max={spec.max}
                  step={spec.step ?? (spec.type === "int" ? 1 : 0.01)}
                  className="w-20 font-mono text-sm"
                />
              </div>
            ) : spec.type === "bool" ? (
              <div className="flex items-center gap-2">
                <Switch
                  checked={Boolean(values[spec.name] ?? getDefault(spec))}
                  onCheckedChange={(v) => updateParam(spec.name, v)}
                />
                <span className="text-sm text-zinc-400">
                  {values[spec.name] ? "On" : "Off"}
                </span>
              </div>
            ) : (
              <Input
                type="text"
                value={String(values[spec.name] ?? getDefault(spec))}
                onChange={(e) => updateParam(spec.name, e.target.value)}
                className="font-mono text-sm"
              />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
