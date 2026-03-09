"use client"

import * as React from "react"
import { cn } from "../../lib/utils"

interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  value?: number[]
  onValueChange?: (value: number[]) => void
  min?: number
  max?: number
  step?: number
}

export const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, value = [0], onValueChange, min = 0, max = 100, step = 1, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = [Number(e.target.value)]
      onValueChange?.(newValue)
    }

    return (
      <div className="relative flex w-full items-center">
        <input
          type="range"
          ref={ref}
          min={min}
          max={max}
          step={step}
          value={value[0]}
          onChange={handleChange}
          className={cn(
            "h-2 w-full cursor-pointer appearance-none rounded-lg bg-zinc-800 accent-emerald-600",
            className
          )}
          {...props}
        />
      </div>
    )
  }
)
Slider.displayName = "Slider"
