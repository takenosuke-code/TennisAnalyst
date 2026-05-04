'use client'

import { useEffect, useRef } from 'react'
import type { CoachDebugEntry } from '@/store/live'
import { useLiveStore } from '@/store/live'

function formatSessionMs(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

function outcomeLabel(entry: CoachDebugEntry) {
  if (entry.outcome.kind === 'cue') {
    return { label: entry.outcome.text, tone: 'text-emerald-300' }
  }
  if (entry.outcome.kind === 'silence') {
    return { label: '[silent — no cue spoken]', tone: 'text-white/50' }
  }
  return { label: '[error — request failed]', tone: 'text-red-300' }
}

export default function CoachDebugPanel() {
  const log = useLiveStore((s) => s.coachDebugLog)
  const listRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [log])

  return (
    <section
      data-testid="coach-debug-panel"
      className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden"
    >
      <header className="px-4 py-3 border-b border-white/5">
        <h2 className="text-white font-medium text-sm">LLM thinking (debug)</h2>
        <p className="text-white/50 text-xs">
          What the live coach saw + what came back, per batch. Newest at the
          bottom.
        </p>
      </header>

      <div
        ref={listRef}
        className="max-h-72 overflow-y-auto px-4 py-3 space-y-3 font-mono text-[11px] text-white/70"
      >
        {log.length === 0 ? (
          <p className="text-white/40">
            Waiting for the first batch… (3 swings, or 5 s of idle after 2)
          </p>
        ) : (
          log.map((entry) => {
            const { label, tone } = outcomeLabel(entry)
            return (
              <div
                key={entry.id}
                data-testid="coach-debug-entry"
                className="rounded-md border border-white/5 bg-black/30 px-3 py-2 space-y-1.5"
              >
                <div className="flex items-center justify-between text-white/50 text-[10px] uppercase tracking-wide">
                  <span>
                    t={formatSessionMs(entry.sessionMs)} · {entry.swingCount}{' '}
                    swing{entry.swingCount === 1 ? '' : 's'}
                  </span>
                  <span>{entry.latencyMs} ms</span>
                </div>

                <div>
                  <div className="text-white/40 text-[10px] uppercase tracking-wide mb-0.5">
                    angles in
                  </div>
                  <ul className="space-y-0.5">
                    {entry.angleSummaries.map((s, idx) => (
                      <li key={idx} className="text-white/70 break-words">
                        <span className="text-white/40">
                          {idx + 1}.
                        </span>{' '}
                        {s.length > 240 ? `${s.slice(0, 240)}…` : s}
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <div className="text-white/40 text-[10px] uppercase tracking-wide mb-0.5">
                    cue out
                  </div>
                  <div className={tone}>{label}</div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </section>
  )
}
