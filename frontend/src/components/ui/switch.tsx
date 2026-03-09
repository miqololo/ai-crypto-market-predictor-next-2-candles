"use client"

import * as React from "react"
import { cn } from "../../lib/utils"

interface SwitchProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  checked?: boolean
  onCheckedChange?: (checked: boolean) => void
}

export const Switch = React.forwardRef<HTMLInputElement, SwitchProps>(
  ({ className, checked, onCheckedChange, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      onCheckedChange?.(e.target.checked)
    }

    return (
      <label className="relative inline-flex cursor-pointer items-center">
        <input
          type="checkbox"
          ref={ref}
          checked={checked}
          onChange={handleChange}
          className="peer sr-only"
          {...props}
        />
        <div
          className={cn(
            "peer h-6 w-11 rounded-full bg-zinc-700 transition-colors",
            "peer-checked:bg-emerald-600",
            "peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-emerald-500 peer-focus:ring-offset-2",
            "peer-disabled:cursor-not-allowed peer-disabled:opacity-50",
            className
          )}
        >
          <div
            className={cn(
              "absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition-transform",
              "peer-checked:translate-x-5"
            )}
          />
        </div>
      </label>
    )
  }
)
Switch.displayName = "Switch"
