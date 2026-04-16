'use client'

import { useState } from 'react'
import { useAnalysisStore, usePoseStore } from '@/store'
import type { ProSwing, PoseFrame } from '@/lib/supabase'

interface LLMCoachingPanelProps {
  proSwing: ProSwing | null
  compareMode?: 'pro' | 'custom'
  frames?: PoseFrame[]
}

export default function LLMCoachingPanel({ proSwing, compareMode = 'pro', frames }: LLMCoachingPanelProps) {
  const [open, setOpen] = useState(false)
  const { feedback, loading, setFeedback, appendFeedback, setLoading, reset } =
    useAnalysisStore()
  const { framesData, sessionId } = usePoseStore()

  const canAnalyze =
    framesData.length > 0 &&
    (compareMode === 'custom' || proSwing !== null)

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

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          ...(proSwing ? { proSwingId: proSwing.id } : {}),
          keypointsJson,
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
      // Flush any remaining bytes held by the streaming decoder (UTF-8 boundary safety)
      const remaining = decoder.decode()
      if (remaining) appendFeedback(remaining)

      // Check if the stream ended with an error sentinel from the server
      const currentFeedback = useAnalysisStore.getState().feedback
      if (currentFeedback.includes('\n\n[ERROR] ')) {
        const cleaned = currentFeedback.split('\n\n[ERROR] ')[0]
        setFeedback(cleaned || 'Analysis failed. Please try again.')
      }
    } catch (err) {
      setFeedback('Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-xl border border-white/10 overflow-hidden">
      {/* Header / trigger */}
      <div className="bg-white/5 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">🤖</span>
          <h3 className="text-sm font-semibold text-white">AI Coach</h3>
          {proSwing ? (
            <span className="text-xs text-white/40">
              vs {proSwing.pros?.name ?? 'Pro'} · {proSwing.shot_type}
            </span>
          ) : compareMode === 'custom' ? (
            <span className="text-xs text-white/40">
              Form Analysis
            </span>
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

      {/* Hint when disabled due to no pro selected */}
      {!canAnalyze && !loading && !feedback && framesData.length > 0 && compareMode !== 'custom' && !proSwing && (
        <div className="px-4 py-2 bg-white/[0.02]">
          <p className="text-white/40 text-xs">Select a pro player above to compare against</p>
        </div>
      )}

      {/* Content */}
      {open && (
        <div className="bg-black/20 p-4 max-h-96 overflow-y-auto">
          {!canAnalyze && !loading && !feedback && (
            <p className="text-white/40 text-sm text-center py-4">
              {framesData.length === 0
                ? 'Upload a video first to get coaching feedback.'
                : compareMode === 'custom'
                  ? 'Click Analyze Swing for AI coaching on your form.'
                  : 'Select a pro player to compare against.'}
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
        // Inline bold - split handles all cases including mixed bold/normal
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
