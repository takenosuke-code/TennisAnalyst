'use client'

import { useEffect, useMemo, useState } from 'react'
import { useAnalysisStore, usePoseStore } from '@/store'
import { useUser } from '@/hooks/useUser'
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

type Correction = 'correct' | 'too_easy' | 'too_hard'
type FeedbackState = 'idle' | 'sending' | 'sent' | 'error'

// Section names emitted by /api/analyze. The route streams markdown with
// these H2 headers in this order; the parser below splits the
// accumulated text into them. Anything outside the canonical set is
// dropped on the floor (the route is the contract; if it adds a section
// later, surface it here).
const QUICK_READ = 'Quick Read'
const PRIMARY = 'Primary cue'
const OTHER = 'Other things I noticed'
const DRILLS = 'Recommended drills'
const SHOW_WORK = 'Show your work'

/**
 * Split streamed markdown into the canonical sections. Operates on
 * the WHOLE accumulated string (not per-chunk) so a chunk that lands
 * mid-header (e.g. "## Pri" then "mary cue\n…") still parses correctly
 * once the next chunk arrives.
 *
 * Returns null for any section that hasn't started yet — the renderer
 * uses this to progressively reveal sections as they stream in.
 */
function parseSections(text: string): {
  quickRead: string | null
  primary: string | null
  other: string | null
  drills: string | null
  showWork: string | null
} {
  if (!text)
    return {
      quickRead: null,
      primary: null,
      other: null,
      drills: null,
      showWork: null,
    }

  // Split on H2 headers. The first chunk before any header is dropped
  // (the API contract starts with "## Quick Read").
  const parts = text.split(/^## /m)
  // parts[0] is the prelude before the first header; ignore.
  const sections: Record<string, string> = {}
  for (let i = 1; i < parts.length; i++) {
    const block = parts[i]
    const newlineIdx = block.indexOf('\n')
    if (newlineIdx === -1) {
      // The header is still being streamed (no body yet). Treat the
      // header alone as a known section with empty body so the panel
      // can render the heading immediately.
      sections[block.trim()] = ''
    } else {
      const header = block.slice(0, newlineIdx).trim()
      const body = block.slice(newlineIdx + 1)
      sections[header] = body
    }
  }
  return {
    quickRead: sections[QUICK_READ] ?? null,
    primary: sections[PRIMARY] ?? null,
    other: sections[OTHER] ?? null,
    drills: sections[DRILLS] ?? null,
    showWork: sections[SHOW_WORK] ?? null,
  }
}

export default function LLMCoachingPanel({ compareMode = 'solo', frames, compareFrames, baselineLabel }: LLMCoachingPanelProps) {
  const { feedback, loading, setFeedback, appendFeedback, setLoading, reset } =
    useAnalysisStore()
  const { framesData, sessionId } = usePoseStore()
  const { user } = useUser()

  const [userFocus, setUserFocus] = useState('')
  const [analysisEventId, setAnalysisEventId] = useState<string | null>(null)
  const [feedbackState, setFeedbackState] = useState<FeedbackState>('idle')
  // True when /api/analyze responded with the X-Analyze-Empty-State
  // header (pose was unreadable). Drives a softer, no-disclosure render
  // path — no Primary cue heading, no thumbs strip.
  // 2026-05 — the route no longer hard-blocks; it always runs coaching
  // and surfaces low-confidence via X-Analyze-Low-Confidence instead.
  // isEmptyState is left in place as a safety net for any older route
  // version still in flight, but it should rarely fire now.
  const [isEmptyState, setIsEmptyState] = useState(false)
  // Soft warning when pose tracking was uneven — the route ran the LLM
  // but flagged that the read may be less precise. Drives a banner
  // above the normal three-section render.
  const [isLowConfidence, setIsLowConfidence] = useState(false)

  // Effective frames: prop-passed `frames` win (baseline/compare path
  // sends them in directly), with the Zustand store as the fallback for
  // the solo /analyze flow. Bug fix: previously canAnalyze checked ONLY
  // framesData, which gated the Analyze button false whenever a user
  // landed on /baseline/compare in a fresh session — even though both
  // their baseline frames and today's frames were loaded via props.
  const effectiveFrames = frames ?? framesData
  const canAnalyze = effectiveFrames.length > 0

  // Re-parse sections on every render. parseSections is cheap and
  // idempotent over accumulated text, so we don't need to memo on
  // chunk boundaries.
  const sections = useMemo(() => parseSections(feedback), [feedback])

  // If the feedback text clears (e.g. reset/re-analyze), drop any lingering
  // rating + empty-state flags so the panel reads as fresh.
  useEffect(() => {
    if (!feedback) {
      setFeedbackState('idle')
      setAnalysisEventId(null)
      setIsEmptyState(false)
      setIsLowConfidence(false)
    }
  }, [feedback])

  const runAnalysis = async () => {
    if (!canAnalyze || loading) return
    reset()
    setLoading(true)
    setAnalysisEventId(null)
    setFeedbackState('idle')
    setIsEmptyState(false)
    setIsLowConfidence(false)

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

      // Backend emits this header at the start of the stream so we can
      // round-trip a thumbs rating back to the same analysis_events row.
      const eventId = res.headers.get('X-Analysis-Event-Id')
      if (eventId) setAnalysisEventId(eventId)

      // Empty-state flag is also a header so we can branch the render
      // path before the body starts arriving.
      if (res.headers.get('X-Analyze-Empty-State') === 'true') {
        setIsEmptyState(true)
      }
      if (res.headers.get('X-Analyze-Low-Confidence') === 'true') {
        setIsLowConfidence(true)
      }

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

  const submitFeedback = async (correction: Correction) => {
    if (!analysisEventId || feedbackState === 'sending' || feedbackState === 'sent') return
    setFeedbackState('sending')
    try {
      const res = await fetch(`/api/analysis-events/${analysisEventId}/feedback`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ correction }),
      })
      if (!res.ok) throw new Error('Feedback failed')
      setFeedbackState('sent')
    } catch {
      setFeedbackState('error')
    }
  }

  // Thumbs strip only for the real-coaching path: gated on a sent event
  // id (so empty-state and sign-in-less paths skip), no loading flicker,
  // and a signed-in user.
  const showFeedbackStrip =
    Boolean(feedback) &&
    !loading &&
    !isEmptyState &&
    Boolean(user) &&
    Boolean(analysisEventId)

  return (
    <section className="bg-cream text-ink border border-ink/10">
      <header className="px-5 py-3 border-b border-ink/10 flex items-center justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <h2 className="font-display font-bold text-base text-ink">AI Coach</h2>
          {compareMode === 'baseline' ? (
            <span className="text-xs text-ink/50">
              vs {baselineLabel ?? 'your best day'}
            </span>
          ) : compareMode === 'custom' ? (
            <span className="text-xs text-ink/50">Form Analysis</span>
          ) : null}
        </div>

        <button
          onClick={runAnalysis}
          disabled={!canAnalyze || loading}
          className={`px-3 py-1.5 text-xs font-semibold tracking-wide transition-colors ${
            canAnalyze && !loading
              ? 'bg-clay hover:bg-[#c4633f] text-cream'
              : 'bg-ink/10 text-ink/40 cursor-not-allowed'
          }`}
        >
          {loading ? 'Analyzing…' : feedback ? 'Re-analyze' : 'Analyze Swing'}
        </button>
      </header>

      {/* Pre-analysis prompt input. Only shown when no feedback yet
          and we're not actively streaming. Keeps the same Enter-to-submit
          UX as the old panel. */}
      {canAnalyze && !loading && !feedback && (
        <div className="px-5 py-4 border-b border-ink/10">
          <label className="block text-xs text-ink/60 mb-1.5">
            Anything specific you want feedback on?{' '}
            <span className="text-ink/40">(optional)</span>
          </label>
          <div className="relative">
            <textarea
              value={userFocus}
              onChange={(e) => setUserFocus(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  if (canAnalyze && !loading) runAnalysis()
                }
              }}
              placeholder={
                compareMode === 'baseline' && compareFrames?.length
                  ? `e.g. "did my hip rotation hold up?" or "focus on follow-through"`
                  : compareMode === 'custom' && compareFrames?.length
                    ? `e.g. "why is my second shot flatter?" or "focus on hip rotation"`
                    : `e.g. "working on topspin" or "am I rotating my hips enough?"`
              }
              rows={2}
              className="w-full bg-cream-soft border border-ink/15 px-3 py-2 pr-20 text-sm text-ink placeholder:text-ink/30 focus:outline-none focus:border-clay/60 resize-none"
            />
            <button
              type="button"
              onClick={runAnalysis}
              disabled={!canAnalyze || loading}
              className={`absolute bottom-2 right-2 px-3 py-1 text-xs font-semibold transition-colors ${
                !canAnalyze || loading
                  ? 'bg-ink/10 text-ink/40 cursor-not-allowed'
                  : 'bg-clay hover:bg-[#c4633f] text-cream'
              }`}
            >
              Send ↵
            </button>
          </div>
          <p className="text-[11px] text-ink/40 mt-1.5">
            Press Enter to analyze. Shift+Enter for a newline.
          </p>
        </div>
      )}

      {/* Empty placeholder when no frames available */}
      {!canAnalyze && !loading && !feedback && (
        <div className="px-5 py-8 text-center">
          <p className="text-ink/50 text-sm">
            Upload a video first to get coaching feedback.
          </p>
        </div>
      )}

      {/* Body. Three render branches:
          - empty-state (legacy hard-block, header X-Analyze-Empty-State)
          - normal coaching with optional low-confidence banner above
          - loading-no-text-yet: thinking indicator only */}
      {(loading || feedback) && (
        <div className="px-5 py-5">
          {isEmptyState ? (
            <EmptyStateMessage text={feedback} />
          ) : (
            <>
              {isLowConfidence && <LowConfidenceBanner />}
              <CoachingSections
                quickRead={sections.quickRead}
                primary={sections.primary}
                other={sections.other}
                drills={sections.drills}
                showWork={sections.showWork}
                loading={loading}
              />
            </>
          )}
        </div>
      )}

      {showFeedbackStrip && (
        <FeedbackStrip state={feedbackState} onSubmit={submitFeedback} />
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
// Section renderers
// ---------------------------------------------------------------------------

/** Soft warning banner shown above the coaching sections when the route
 *  flagged X-Analyze-Low-Confidence. Pose tracking was uneven so the
 *  read may be less precise; we still ran the analysis. Designed to be
 *  unobtrusive — clay accent stripe + small text, no big alarm. */
function LowConfidenceBanner() {
  return (
    <div className="mb-4 flex gap-3 bg-cream-soft border-l-2 border-clay px-3 py-2">
      <span className="text-xs text-ink/75 leading-relaxed">
        Pose tracking on this clip was uneven, so the read may be less precise
        than usual. For sharper analysis next time, shoot from the side at
        chest height with the player filling the frame.
      </span>
    </div>
  )
}

/** Soft, sympathetic message when /api/analyze couldn't read the pose.
 *  No headers, no disclosure, no thumbs strip — just the body text. */
function EmptyStateMessage({ text }: { text: string }) {
  return (
    <div className="bg-cream-soft border border-ink/10 p-4">
      <p className="text-ink/75 text-sm leading-relaxed whitespace-pre-line">
        {text.trim() ||
          "We couldn't read your swing clearly enough to give specific coaching."}
      </p>
    </div>
  )
}

/** Quick Read + In-depth analysis (Primary cue + Other observations) +
 *  Recommended drills + collapsible Show-your-work. Sections render
 *  progressively as they stream in.
 *  Visual hierarchy (per user direction):
 *   - Quick Read: skim-first card at top, clay accent.
 *   - In-depth analysis: single panel grouping the primary cue (quiet
 *     italic lead-in) with the secondary observations (the readable
 *     focus — strong contrast, generous line-height).
 *   - Recommended drills: clear pre-disclosure section, ink-soft tint.
 *   - Show your work: collapsed details element. */
function CoachingSections({
  quickRead,
  primary,
  other,
  drills,
  showWork,
  loading,
}: {
  quickRead: string | null
  primary: string | null
  other: string | null
  drills: string | null
  showWork: string | null
  loading: boolean
}) {
  // If nothing has parsed yet but we're still loading, show a thinking
  // indicator so the panel doesn't look frozen.
  const nothingYet =
    quickRead == null &&
    primary == null &&
    other == null &&
    drills == null &&
    showWork == null

  return (
    <div className="space-y-5">
      {nothingYet && loading && (
        <p className="text-ink/50 text-sm flex items-center gap-2">
          <span className="inline-block w-1.5 h-4 bg-clay animate-pulse" />
          Reading your swing…
        </p>
      )}

      {quickRead != null && (
        <div className="bg-cream-soft border-l-4 border-clay p-5">
          <p className="text-[11px] uppercase tracking-[0.18em] text-clay mb-3 font-semibold">
            Quick Read
          </p>
          <BulletList markdown={quickRead} />
          {loading && quickRead.trim() && (
            <span className="inline-block w-1.5 h-4 bg-clay animate-pulse mt-1" />
          )}
        </div>
      )}

      {(primary != null || other != null) && (
        <div className="bg-cream-soft border border-ink/15 p-5">
          <p className="text-xs uppercase tracking-[0.18em] text-ink font-bold mb-3">
            In-depth analysis
          </p>
          {primary != null && (
            <p className="text-sm text-ink/60 italic leading-relaxed mb-4">
              {primary.trim()}
              {loading && primary.trim() && (
                <span className="inline-block w-1.5 h-4 bg-clay animate-pulse ml-1 align-middle" />
              )}
            </p>
          )}
          {other != null && <BulletList markdown={other} prominent />}
        </div>
      )}

      {drills != null && (
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-ink/50 mb-2 font-semibold">
            Recommended drills
          </p>
          <BulletList markdown={drills} />
        </div>
      )}

      {showWork != null && (
        <details className="group border border-ink/10 bg-cream-soft">
          <summary className="list-none cursor-pointer px-4 py-2.5 flex items-center justify-between text-xs font-semibold tracking-wide text-ink/70 hover:text-ink">
            <span>Show your work</span>
            <Caret />
          </summary>
          <div className="px-4 pb-3 pt-1 border-t border-ink/10">
            <BulletList markdown={showWork} dense />
          </div>
        </details>
      )}
    </div>
  )
}

function Caret() {
  return (
    <svg
      viewBox="0 0 12 12"
      className="w-3 h-3 text-ink/50 transition-transform group-open:rotate-180"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 4.5l3 3 3-3" />
    </svg>
  )
}

/** Render a markdown bullet list. Supports `- ` and `* ` markers, plus
 *  inline **bold**. Anything that isn't a bullet is rendered as a
 *  paragraph (some sections occasionally narrate before listing). */
function BulletList({
  markdown,
  dense = false,
  prominent = false,
}: {
  markdown: string
  dense?: boolean
  prominent?: boolean
}) {
  const lines = markdown.split('\n').map((l) => l.trimEnd())
  const items: { kind: 'bullet' | 'para'; text: string }[] = []
  for (const raw of lines) {
    const line = raw.trim()
    if (!line) continue
    if (line.startsWith('- ') || line.startsWith('* ')) {
      items.push({ kind: 'bullet', text: line.slice(2) })
    } else {
      items.push({ kind: 'para', text: line })
    }
  }
  if (items.length === 0) return null

  const bullets = items.filter((i) => i.kind === 'bullet')
  const paras = items.filter((i) => i.kind === 'para')

  const paraClass = prominent
    ? 'text-base text-ink/95 leading-7'
    : 'text-sm text-ink/75 leading-relaxed'

  const ulClass = dense
    ? 'list-disc pl-5 space-y-0.5 text-xs text-ink/75 leading-relaxed marker:text-ink/30'
    : prominent
      ? 'list-disc pl-5 space-y-2 text-base text-ink/95 leading-7 marker:text-clay'
      : 'list-disc pl-5 space-y-1 text-sm text-ink/75 leading-relaxed marker:text-ink/30'

  return (
    <div className={dense ? 'space-y-1' : 'space-y-1.5'}>
      {paras.map((p, i) => (
        <p key={`p-${i}`} className={paraClass}>
          <InlineMarkdown text={p.text} />
        </p>
      ))}
      {bullets.length > 0 && (
        <ul className={ulClass}>
          {bullets.map((b, i) => (
            <li key={`b-${i}`}>
              <InlineMarkdown text={b.text} />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** Inline-only markdown: **bold** -> <strong>. */
function InlineMarkdown({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/)
  return (
    <>
      {parts.map((part, i) =>
        part.startsWith('**') && part.endsWith('**') ? (
          <strong key={i} className="text-ink font-semibold">
            {part.replace(/\*\*/g, '')}
          </strong>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  )
}

// ---------------------------------------------------------------------------
// Thumbs feedback strip — preserved as-is from the previous version, with
// the cream/ink palette swap. The route-trip lifecycle is unchanged.
// ---------------------------------------------------------------------------

function FeedbackStrip({
  state,
  onSubmit,
}: {
  state: FeedbackState
  onSubmit: (correction: Correction) => void
}) {
  if (state === 'sent') {
    return (
      <div className="border-t border-ink/10 px-5 py-3 text-xs text-ink/60">
        Thanks — logged.
      </div>
    )
  }

  const disabled = state === 'sending'

  return (
    <div className="border-t border-ink/10 px-5 py-3 flex items-center gap-3 flex-wrap">
      <span className="text-xs text-ink/60">Was this coaching right for you?</span>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onSubmit('correct')}
          disabled={disabled}
          className="px-2.5 py-1 text-xs font-medium bg-green-3 text-cream hover:bg-green-4 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          👍 Spot on
        </button>
        <button
          type="button"
          onClick={() => onSubmit('too_hard')}
          disabled={disabled}
          className="px-2.5 py-1 text-xs font-medium bg-ink/10 text-ink/75 hover:bg-ink/15 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ⬇️ Too advanced
        </button>
        <button
          type="button"
          onClick={() => onSubmit('too_easy')}
          disabled={disabled}
          className="px-2.5 py-1 text-xs font-medium bg-ink/10 text-ink/75 hover:bg-ink/15 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ⬆️ Too simple
        </button>
      </div>
      {state === 'error' && (
        <span className="text-xs text-clay">Couldn&apos;t save — try again?</span>
      )}
    </div>
  )
}
