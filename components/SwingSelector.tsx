'use client'

import { useMemo } from 'react'
import { detectSwings } from '@/lib/jointAngles'
import type { SwingSegment } from '@/lib/jointAngles'
import type { PoseFrame } from '@/lib/supabase'

interface SwingSelectorProps {
  allFrames: PoseFrame[]
  selectedIndex: number | null
  onSelect: (segment: SwingSegment) => void
}

export default function SwingSelector({ allFrames, selectedIndex, onSelect }: SwingSelectorProps) {
  const segments = useMemo(() => detectSwings(allFrames), [allFrames])

  if (segments.length <= 1) return null

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4">
      <h3 className="text-sm font-semibold text-white mb-1">
        {segments.length} swings detected
      </h3>
      <p className="text-xs text-white/40 mb-3">Select which swing to analyze</p>
      <div className="flex gap-2 flex-wrap">
        {segments.map((seg) => {
          const durationSec = ((seg.endMs - seg.startMs) / 1000).toFixed(1)
          const startSec = (seg.startMs / 1000).toFixed(1)
          const isSelected = selectedIndex === seg.index
          return (
            <button
              key={seg.index}
              onClick={() => onSelect(seg)}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-all border ${
                isSelected
                  ? 'bg-emerald-500 border-emerald-400 text-white'
                  : 'bg-white/5 border-white/10 text-white/70 hover:bg-white/10 hover:text-white'
              }`}
            >
              <div>Swing {seg.index}</div>
              <div className={`text-xs ${isSelected ? 'text-white/80' : 'text-white/40'}`}>
                {startSec}s &middot; {durationSec}s &middot; {seg.frames.length}f
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
