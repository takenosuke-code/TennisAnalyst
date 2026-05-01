'use client'

import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { PoseFrame } from '@/lib/supabase'

interface ShotCoachingModalProps {
  open: boolean
  onClose: () => void
  // Frames scoped to a single swing (not the whole video). Sent to
  // /api/analyze the same way LLMCoachingPanel does it for the page-
  // level call, just with a smaller frames payload — the backend
  // tailors advice to whatever it gets.
  frames: PoseFrame[]
  // Display title / subtitle. Index is the SwingSegment.index, label
  // is the user's preferred phrasing (e.g. "Swing 3 — forehand").
  title: string
  subtitle?: string
  sessionId?: string | null
}

/*
 * ShotCoachingModal — opens a focused coaching session for a single
 * swing. Mirrors the architecture of LLMCoachingPanel but uses fully
 * local state so it doesn't trample the page-level coach's global
 * Zustand-backed feedback string. Streaming response handling is
 * identical: POST /api/analyze, read the body as a text stream,
 * append chunks to the local feedback buffer.
 *
 * Closes on ESC, backdrop click, and the close button. Body scroll
 * is locked while open. Reuses the SegmentLightbox layout pattern
 * for visual consistency.
 */
export default function ShotCoachingModal({
  open,
  onClose,
  frames,
  title,
  subtitle,
  sessionId,
}: ShotCoachingModalProps) {
  const [userFocus, setUserFocus] = useState('')
  const [feedback, setFeedback] = useState('')
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const focusInputRef = useRef<HTMLInputElement | null>(null)

  // ESC + body scroll lock + cleanup on close.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    // Auto-focus the input on open so the user can immediately type
    // a focus area without an extra click.
    setTimeout(() => focusInputRef.current?.focus(), 50)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
      abortRef.current?.abort()
      abortRef.current = null
    }
  }, [open, onClose])

  // Reset analysis state every time the modal opens — a re-open should
  // be a fresh coaching session, not a stale buffer from the previous
  // swing.
  useEffect(() => {
    if (!open) return
    setFeedback('')
    setErrorMsg(null)
    setLoading(false)
    setUserFocus('')
  }, [open])

  const runAnalysis = async () => {
    if (loading || frames.length === 0) return
    setFeedback('')
    setErrorMsg(null)
    setLoading(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          keypointsJson: {
            fps_sampled: 30,
            frame_count: frames.length,
            frames,
          },
          ...(userFocus.trim() ? { userFocus: userFocus.trim() } : {}),
        }),
        signal: controller.signal,
      })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        setFeedback((prev) => prev + decoder.decode(value, { stream: true }))
      }
      const remaining = decoder.decode()
      if (remaining) setFeedback((prev) => prev + remaining)
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      setErrorMsg(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setLoading(false)
      abortRef.current = null
    }
  }

  if (!open) return null
  if (typeof document === 'undefined') return null

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Coach for ${title}`}
      onClick={onClose}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-ink/85 backdrop-blur-sm px-4 py-8"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-2xl flex flex-col gap-4 bg-cream text-ink rounded-2xl overflow-hidden"
      >
        <div className="px-6 pt-5 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[11px] uppercase tracking-[0.18em] text-ink/55">Coach this shot</p>
            <h2 className="font-display font-bold text-ink text-lg mt-1">{title}</h2>
            {subtitle && <p className="text-xs text-ink/55 mt-1">{subtitle}</p>}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-ink/45 hover:text-ink text-xl leading-none px-2"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="px-6 pb-5 flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label className="text-[10px] uppercase tracking-[0.16em] text-ink/55">
              What do you want feedback on? (optional)
            </label>
            <input
              ref={focusInputRef}
              value={userFocus}
              onChange={(e) => setUserFocus(e.target.value)}
              placeholder="e.g. is my elbow dropping too much?"
              maxLength={200}
              disabled={loading}
              className="bg-cream-soft border border-ink/15 text-ink text-sm px-3 py-2 rounded-lg focus:outline-none focus:border-clay"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !loading) runAnalysis()
              }}
            />
          </div>

          {!feedback && !loading && !errorMsg && (
            <button
              type="button"
              onClick={runAnalysis}
              disabled={frames.length === 0}
              className="self-start px-5 py-2 rounded-full bg-clay hover:bg-[#c4633f] text-cream text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Get coaching
            </button>
          )}

          {loading && !feedback && (
            <p className="text-sm text-ink/60">Analyzing this swing…</p>
          )}

          {feedback && (
            <div className="bg-cream-soft border border-ink/10 rounded-lg p-4 text-sm text-ink leading-relaxed whitespace-pre-wrap max-h-[50vh] overflow-y-auto">
              {feedback}
              {loading && <span className="inline-block w-2 h-4 bg-clay/60 align-middle ml-1 animate-pulse" />}
            </div>
          )}

          {errorMsg && (
            <p className="text-xs text-clay">{errorMsg}</p>
          )}

          {feedback && !loading && (
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={runAnalysis}
                className="text-xs text-ink/55 hover:text-ink underline"
              >
                Re-run analysis
              </button>
              <button
                type="button"
                onClick={onClose}
                className="ml-auto px-4 py-1.5 rounded-full bg-ink hover:bg-ink-soft text-cream text-xs font-semibold transition-colors"
              >
                Done
              </button>
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}
