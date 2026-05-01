'use client'

import { useCallback, useId, useMemo, useRef } from 'react'
import type {
  DetectedStroke,
  StrokeRejectReason,
  StrokeQualityResult,
  StrokeComparisonResult,
} from '@/lib/strokeAnalysis'

// Re-export the canonical types so existing test imports
// (`import type { … } from '@/components/StrokeRibbon'`) keep resolving.
// The single source of truth lives in lib/strokeAnalysis.ts.
export type {
  DetectedStroke,
  StrokeRejectReason,
  StrokeQualityResult,
  StrokeComparisonResult,
}

export interface StrokeRibbonProps {
  strokes: DetectedStroke[]
  // Same length as strokes (caller's contract); we look up by strokeId so
  // mismatched ordering is tolerated. Strokes without a matching quality
  // entry render as a neutral chip — preferable to an empty render.
  quality: StrokeQualityResult[]
  comparison?: StrokeComparisonResult
  onStrokeSelect?: (strokeId: string) => void
  selectedStrokeId?: string | null
}

// ---------------------------------------------------------------------------
// Visuals
// ---------------------------------------------------------------------------

type QualityBucket = 'rejected' | 'worst' | 'neutral' | 'best'

/**
 * Branch on `rejected` BEFORE thresholding so a NaN score (which flunks
 * every numeric comparison) doesn't accidentally pass the `|score| ≤ 0.5`
 * neutral check. Rejected chips get their own bucket so the renderer can
 * style them as muted regardless of score.
 */
function bucketFor(quality: StrokeQualityResult | undefined): QualityBucket {
  if (!quality) return 'neutral'
  if (quality.rejected || Number.isNaN(quality.score)) return 'rejected'
  if (quality.score > 0.5) return 'best'
  if (quality.score < -0.5) return 'worst'
  return 'neutral'
}

const STRIPE_BG: Record<QualityBucket, string> = {
  rejected: 'bg-cream-soft',
  worst: 'bg-clay',
  neutral: 'bg-cream-soft',
  best: 'bg-green-3',
}

/**
 * Format a z-score as "+0.8" / "−1.2" (using the typographic minus
 * U+2212 to match the rest of the codebase's typography). Returns
 * an em dash when the score isn't finite — those chips show the
 * reject reason copy instead of the score pill.
 */
function formatScore(score: number): string {
  if (!Number.isFinite(score)) return '—'
  const rounded = Math.round(score * 10) / 10
  if (rounded > 0) return `+${rounded.toFixed(1)}`
  if (rounded < 0) return `\u2212${Math.abs(rounded).toFixed(1)}`
  return '0.0'
}

// `Record<StrokeRejectReason, string>` (rather than `Partial<…>`) makes
// this map exhaustive: if a future StrokeRejectReason is added in
// lib/strokeAnalysis.ts and not mirrored here, tsc will fail on a
// missing key. That's the compile-time guard the cross-cutting audit
// asked for.
const REJECT_LABEL: Record<StrokeRejectReason, string> = {
  low_visibility: 'Low visibility',
  camera_pan: 'Camera panned',
  camera_zoom: 'Camera zoomed',
  too_short: 'Stroke too short',
  missing_data: 'Incomplete tracking',
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Horizontally scrolling ribbon of stroke chips, color-coded by quality
 * z-score. Tap-to-expand is owned by the parent — the ribbon's job is
 * navigation only; it fires onStrokeSelect with the strokeId and
 * reflects the parent's `selectedStrokeId` as the visual selection.
 *
 * SwingVision-style pattern recommended in the user research over the
 * stacked dual-video layout. Sits between the video and the
 * LLMCoachingPanel as an additional surface — does NOT replace the
 * coaching panel.
 *
 * Keyboard: arrow Left/Right moves selection through chips (roving
 * tabindex — only the active chip is in the tab order). Home/End jump
 * to the first/last chip. Each move fires onStrokeSelect.
 *
 * Accessibility: container has aria-label="Stroke timeline"; each chip
 * is a button labelled "Stroke {n}, score {z}". Reject reasons live in
 * an sr-only span referenced via aria-describedby — present in DOM
 * regardless of hover so screen readers and tests can find them.
 */
export default function StrokeRibbon({
  strokes,
  quality,
  comparison,
  onStrokeSelect,
  selectedStrokeId,
}: StrokeRibbonProps) {
  const ribbonRef = useRef<HTMLDivElement>(null)
  // Stable id prefix so each chip's sr-only describedby element gets a
  // unique, queryable id. useId is stable across SSR/CSR.
  const idPrefix = useId()

  // Index quality by strokeId for O(1) lookup. Memoised so re-renders
  // from selection changes don't rebuild the map.
  const qualityById = useMemo(() => {
    const m = new Map<string, StrokeQualityResult>()
    for (const q of quality) m.set(q.strokeId, q)
    return m
  }, [quality])

  // When the session is consistent, the brief says: hide best/worst
  // badges entirely and surface a banner instead. The badge IDs default
  // to null so the chip render below can branch on them.
  const isConsistent = comparison?.isConsistent === true
  const bestId =
    !isConsistent && comparison?.best ? comparison.best.strokeId : null
  const worstId =
    !isConsistent && comparison?.worst ? comparison.worst.strokeId : null

  // Index of the currently-selected chip — used to drive roving tabindex
  // and arrow-key navigation. Falls back to 0 so the ribbon is always
  // reachable via Tab even when the parent hasn't picked a stroke yet.
  const selectedIndex = useMemo(() => {
    if (!selectedStrokeId) return 0
    const idx = strokes.findIndex((s) => s.strokeId === selectedStrokeId)
    return idx >= 0 ? idx : 0
  }, [strokes, selectedStrokeId])

  const focusChip = useCallback((index: number) => {
    const node = ribbonRef.current
    if (!node) return
    const buttons = node.querySelectorAll<HTMLButtonElement>(
      'button[data-stroke-chip="true"]',
    )
    const target = buttons[index]
    if (target) target.focus()
  }, [])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (strokes.length === 0) return
      let next: number | null = null
      if (e.key === 'ArrowRight') next = Math.min(strokes.length - 1, selectedIndex + 1)
      else if (e.key === 'ArrowLeft') next = Math.max(0, selectedIndex - 1)
      else if (e.key === 'Home') next = 0
      else if (e.key === 'End') next = strokes.length - 1
      if (next === null || next === selectedIndex) return
      e.preventDefault()
      const nextStroke = strokes[next]
      if (nextStroke) {
        onStrokeSelect?.(nextStroke.strokeId)
        // Defer focus to the next tick so any parent-driven re-render
        // from onStrokeSelect mounts the new chip with tabIndex=0
        // before we attempt to focus it.
        requestAnimationFrame(() => focusChip(next!))
      }
    },
    [strokes, selectedIndex, onStrokeSelect, focusChip],
  )

  // ---------- Empty state ----------
  if (strokes.length === 0) {
    return (
      <section
        className="bg-cream text-ink border border-ink/10"
        aria-label="Stroke timeline"
      >
        <header className="px-5 py-3 border-b border-ink/10">
          <h2 className="font-display font-bold text-base text-ink">Stroke Timeline</h2>
        </header>
        <div className="px-5 py-6">
          <p className="text-sm text-ink/50">No strokes detected.</p>
        </div>
      </section>
    )
  }

  return (
    <section className="bg-cream text-ink border border-ink/10">
      <header className="px-5 py-3 border-b border-ink/10 flex items-center justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <h2 className="font-display font-bold text-base text-ink">Stroke Timeline</h2>
          <span className="text-xs text-ink/50">
            {strokes.length} {strokes.length === 1 ? 'stroke' : 'strokes'}
          </span>
        </div>
      </header>

      {/* Consistent-session banner. Replaces best/worst badges per the
          spec — when the session reads as consistent there is no "best"
          or "worst" to flag, so we surface the cue once at the top
          instead. */}
      {isConsistent && (
        <div className="mx-5 mt-4 mb-2 bg-cream-soft border-l-4 border-clay px-4 py-2.5">
          <p className="text-[11px] uppercase tracking-[0.18em] text-clay font-semibold mb-0.5">
            Consistent session
          </p>
          {comparison?.consistentCue && (
            <p className="text-sm text-ink/75 leading-relaxed">
              {comparison.consistentCue}
            </p>
          )}
        </div>
      )}

      {/* Scrolling chip strip. snap-x + snap-mandatory keeps chips
          aligned at the left edge as the user pages through with
          horizontal swipes / wheel. The ribbon itself is a button-row
          with role="toolbar" so AT clients announce it as a navigation
          control rather than a generic group. */}
      <div
        ref={ribbonRef}
        role="toolbar"
        aria-label="Stroke timeline"
        aria-orientation="horizontal"
        onKeyDown={handleKeyDown}
        className="flex gap-2 overflow-x-auto px-5 py-4 snap-x snap-mandatory"
      >
        {strokes.map((stroke, i) => {
          const q = qualityById.get(stroke.strokeId)
          const bucket = bucketFor(q)
          const isSelected = selectedStrokeId === stroke.strokeId
          const isBest = bestId === stroke.strokeId
          const isWorst = worstId === stroke.strokeId
          const reasonId = `${idPrefix}-reason-${i}`
          const labelScore = q && Number.isFinite(q.score)
            ? formatScore(q.score)
            : 'unscored'
          const ariaLabel = `Stroke ${i + 1}, score ${labelScore}`
          // Roving tabindex: only the active chip is in the tab order.
          const tabIndex = i === selectedIndex ? 0 : -1
          const rejected = bucket === 'rejected'

          return (
            <button
              key={stroke.strokeId}
              type="button"
              data-stroke-chip="true"
              data-quality={bucket}
              data-selected={isSelected ? 'true' : 'false'}
              tabIndex={tabIndex}
              aria-label={ariaLabel}
              aria-describedby={rejected && q?.rejectReason ? reasonId : undefined}
              aria-pressed={isSelected}
              onClick={() => onStrokeSelect?.(stroke.strokeId)}
              className={[
                'shrink-0 snap-start flex flex-col bg-cream-soft text-left transition-transform',
                'min-w-[88px] w-[88px] h-16 overflow-hidden',
                'focus:outline-none focus-visible:ring-2 focus-visible:ring-clay/60',
                isSelected
                  ? 'border-2 border-clay scale-[1.05]'
                  : 'border border-ink/10 hover:border-ink/25',
                rejected ? 'opacity-50' : '',
              ]
                .filter(Boolean)
                .join(' ')}
            >
              {/* Top half: thin colored stripe — the quality signal. */}
              <span
                aria-hidden="true"
                data-stripe-bucket={bucket}
                className={`block h-2 w-full ${STRIPE_BG[bucket]}`}
              />

              {/* Bottom half: stroke index + score pill OR reject reason. */}
              <span className="flex-1 flex flex-col justify-center px-2 py-1 gap-0.5">
                <span className="flex items-center gap-1">
                  <span
                    className={`text-[11px] font-display font-semibold text-ink/85 ${
                      rejected ? 'line-through text-ink/50' : ''
                    }`}
                  >
                    Stroke {i + 1}
                  </span>
                  {isBest && (
                    <span className="ml-auto font-display text-[9px] uppercase tracking-[0.14em] text-clay font-bold">
                      Best
                    </span>
                  )}
                  {isWorst && (
                    <span className="ml-auto font-display text-[9px] uppercase tracking-[0.14em] text-ink/50 font-bold">
                      Worst
                    </span>
                  )}
                </span>
                {rejected ? (
                  <span className="text-[10px] text-ink/50 truncate">
                    {q?.rejectReason ? REJECT_LABEL[q.rejectReason] : 'Skipped'}
                  </span>
                ) : (
                  <span
                    className={`text-[10px] font-mono ${
                      bucket === 'best'
                        ? 'text-green-4'
                        : bucket === 'worst'
                          ? 'text-clay'
                          : 'text-ink/60'
                    }`}
                  >
                    {q ? formatScore(q.score) : '—'}
                  </span>
                )}
              </span>

              {/* sr-only describedby target — present in the DOM
                  whether or not the user is hovering, so AT and tests
                  can resolve the reason without a hover gesture. */}
              {rejected && q?.rejectReason && (
                <span id={reasonId} className="sr-only">
                  Rejected: {REJECT_LABEL[q.rejectReason]}
                </span>
              )}
            </button>
          )
        })}
      </div>
    </section>
  )
}

