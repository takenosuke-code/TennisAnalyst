'use client'

import { useState } from 'react'
import { useAnalysisStore, usePoseStore } from '@/store'
import type { PoseFrame } from '@/lib/supabase'

interface LLMCoachingPanelProps {
  compareMode?: 'solo' | 'custom' | 'baseline'
  frames?: PoseFrame[]
  // Second-take frames. When present in 'custom' or 'baseline' mode,
  // analyze runs as a user-vs-user comparison (consistency / best-day drift).
  compareFrames?: PoseFrame[]
  // Shown when compareMode === 'baseline'. Wired to the prompt so coaching
  // references the specific baseline by name.
  baselineLabel?: string
}

export default function LLMCoachingPanel({ compareMode = 'solo', frames, compareFrames, baselineLabel }: LLMCoachingPanelProps) {
  const [open, setOpen] = useState(false)
  const { feedback, loading, setFeedback, appendFeedback, setLoading, reset } =
    useAnalysisStore()
  const { framesData, sessionId } = usePoseStore()

  const [userFocus, setUserFocus] = useState('')

  const canAnalyze = framesData.length > 0

  const runAnalysis = async () => {
    if (!canAnalyze || loading) return
    reset()
    setOpen(true)
    setLoading(true)

    const effectiveFrames = frames ?? framesData
    const keypointsJson = {
      fps_sampled: 30,
      frame_count: effectiveFrames.length,
      frames: effectiveFrames,
    }

    // Pass the second take only in user-vs-user modes (custom or baseline).
    const compareKeypointsJson =
      (compareMode === 'custom' || compareMode === 'baseline') &&
      compareFrames &&
      compareFrames.length > 0
        ? {
            fps_sampled: 30,
            frame_count: compareFrames.length,
            frames: compareFrames,
          }
        : undefined

    const trimmedFocus = userFocus.trim()

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          keypointsJson,
          ...(compareKeypointsJson ? { compareKeypointsJson } : {}),
          ...(trimmedFocus ? { userFocus: trimmedFocus } : {}),
          ...(compareMode === 'baseline' ? { compareMode: 'baseline', baselineLabel } : {}),
        }),
      })

      if (!res.ok || !res.body) throw new Error('Analysis failed')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        appendFeedback(decoder.decode(value, { stream: true }))
      }
      const remaining = decoder.decode()
      if (remaining) appendFeedback(remaining)

      const currentFeedback = useAnalysisStore.getState().feedback
      if (currentFeedback.includes('\n\n[ERROR] ')) {
        const cleaned = currentFeedback.split('\n\n[ERROR] ')[0]
        setFeedback(cleaned || 'Analysis failed. Please try again.')
      }
    } catch {
      setFeedback('Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-xl border border-white/10 overflow-hidden">
      <div className="bg-white/5 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">🤖</span>
          <h3 className="text-sm font-semibold text-white">AI Coach</h3>
          {compareMode === 'baseline' ? (
            <span className="text-xs text-white/40">
              vs {baselineLabel ?? 'your best day'}
            </span>
          ) : compareMode === 'custom' ? (
            <span className="text-xs text-white/40">Form Analysis</span>
          ) : null}
        </div>

        <div className="flex items-center gap-2">
          {feedback && (
            <button
              onClick={() => setOpen((o) => !o)}
              className="text-xs text-white/50 hover:text-white"
            >
              {open ? 'Collapse' : 'Expand'}
            </button>
          )}
          <button
            onClick={runAnalysis}
            disabled={!canAnalyze || loading}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              canAnalyze && !loading
                ? 'bg-emerald-500 hover:bg-emerald-400 text-white'
                : 'bg-white/10 text-white/30 cursor-not-allowed'
            }`}
          >
            {loading ? (
              <span className="flex items-center gap-1">
                <span className="animate-spin">⚙️</span> Analyzing...
              </span>
            ) : feedback ? (
              'Re-analyze'
            ) : (
              'Analyze Swing'
            )}
          </button>
        </div>
      </div>

      {canAnalyze && !loading && !feedback && (
        <div className="px-4 py-3 bg-white/[0.02] border-t border-white/5">
          <label className="block text-xs text-white/50 mb-1.5">
            Anything specific you want feedback on?{' '}
            <span className="text-white/30">(optional)</span>
          </label>
          <textarea
            value={userFocus}
            onChange={(e) => setUserFocus(e.target.value)}
            placeholder={
              compareMode === 'baseline' && compareFrames?.length
                ? `e.g. "did my hip rotation hold up?" or "focus on follow-through"`
                : compareMode === 'custom' && compareFrames?.length
                  ? `e.g. "why is my second shot flatter?" or "focus on hip rotation"`
                  : `e.g. "working on topspin" or "am I rotating my hips enough?"`
            }
            rows={2}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-white/25 focus:outline-none focus:border-emerald-500/40 resize-none"
          />
        </div>
      )}

      {open && (
        <div className="bg-black/20 p-4 max-h-[500px] overflow-y-auto">
          {!canAnalyze && !loading && !feedback && (
            <p className="text-white/40 text-sm text-center py-4">
              Upload a video first to get coaching feedback.
            </p>
          )}
          {(loading || feedback) && (
            <div className="prose prose-invert prose-sm max-w-none">
              <MarkdownText text={feedback} />
              {loading && (
                <span className="inline-block w-1.5 h-4 bg-emerald-400 animate-pulse ml-0.5 rounded-sm" />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Minimal markdown renderer for bold + headers
function MarkdownText({ text }: { text: string }) {
  const lines = text.split('\n')
  return (
    <div className="space-y-1 text-white/80 text-sm leading-relaxed">
      {lines.map((line, i) => {
        if (line.startsWith('## ')) {
          return (
            <h3 key={i} className="text-white font-bold text-base mt-3 mb-1">
              {line.replace('## ', '')}
            </h3>
          )
        }
        const parts = line.split(/(\*\*[^*]+\*\*)/)
        return (
          <p key={i} className={line.startsWith('-') ? 'pl-3' : ''}>
            {parts.map((part, j) =>
              part.startsWith('**') && part.endsWith('**') ? (
                <strong key={j} className="text-white">
                  {part.replace(/\*\*/g, '')}
                </strong>
              ) : (
                part
              )
            )}
          </p>
        )
      })}
    </div>
  )
}
