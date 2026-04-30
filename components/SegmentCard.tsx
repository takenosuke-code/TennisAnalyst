'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import type { VideoSegment } from '@/lib/supabase'

// Shot types the baseline table accepts. Matches the CHECK constraint in
// lib/db/004_user_baselines.sql. Classifier output includes 'unknown' and
// 'idle' too -- those aren't selectable as an override because they can't
// be saved to the baselines row.
const BASELINE_SHOTS = ['forehand', 'backhand', 'serve', 'volley', 'slice'] as const
type BaselineShot = (typeof BASELINE_SHOTS)[number]

// Shot-type → stripe color mapping. Five shot types share three stripe
// colors so the grid reads as variegated instead of stripey-rainbow.
// Forehand/volley = clay (warm), backhand/slice = hard-court (cool),
// serve = green (vertical / overhead). Kept inline so we don't reach
// into lib/ — the brief gates lib changes.
const SHOT_STRIPE: Record<BaselineShot, string> = {
  forehand: 'bg-clay',
  backhand: 'bg-hard-court',
  serve: 'bg-green-3',
  volley: 'bg-clay',
  slice: 'bg-hard-court',
}

function isBaselineShot(v: string | undefined | null): v is BaselineShot {
  return typeof v === 'string' && (BASELINE_SHOTS as readonly string[]).includes(v)
}

export interface SegmentCardSaveOverride {
  shotType: BaselineShot
  label?: string
}

interface SegmentCardProps {
  segment: VideoSegment
  blobUrl: string
  onSave: (segmentId: string, override: SegmentCardSaveOverride) => void | Promise<void>
  saving?: boolean
  saved?: boolean
  canSave?: boolean
  errorMessage?: string | null
  // When false, the save button renders as a "sign in" prompt -- the grid
  // passes auth state down so each card reflects the same gate.
  signedIn?: boolean
  // Inline-enlarge: when true the card spans 2 grid columns on lg+ so
  // the video preview is materially bigger. Toggled by the grid; only
  // one card can be enlarged at a time.
  expanded?: boolean
  onToggleExpand?: () => void
}

// One per-segment card on the multi-shot analyze flow. The card is
// deliberately minimal: a seek-to-midpoint video preview, a shot-type
// override dropdown, an optional label input, and a Save button wired
// up to the grid's onSaveAsBaseline handler.
export default function SegmentCard({
  segment,
  blobUrl,
  onSave,
  saving = false,
  saved = false,
  canSave = true,
  errorMessage = null,
  signedIn = true,
  expanded = false,
  onToggleExpand,
}: SegmentCardProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const classifierIsBaselineShot = isBaselineShot(segment.shot_type)
  // Inline isBaselineShot() call gives TS the type narrowing it needs.
  const initialShot: BaselineShot = isBaselineShot(segment.shot_type)
    ? segment.shot_type
    : 'forehand'
  const [overrideShot, setOverrideShot] = useState<BaselineShot>(initialShot)
  // Tracks whether the user has actively picked from the dropdown. When the
  // classifier returns 'unknown' or 'idle' (not a valid baseline shot type),
  // the dropdown silently defaults to 'forehand' — we must not let the user
  // save a rest-period segment as a forehand without at least a deliberate
  // selection.
  const [overrideTouched, setOverrideTouched] = useState(false)
  const [label, setLabel] = useState<string>(
    segment.label?.trim() || `Segment #${segment.segment_index + 1} — ${overrideShot}`,
  )
  const [labelDirty, setLabelDirty] = useState(false)

  const midpointMs = useMemo(
    () => Math.round((segment.start_ms + segment.end_ms) / 2),
    [segment.start_ms, segment.end_ms],
  )
  const startSec = (segment.start_ms / 1000).toFixed(1)
  const endSec = (segment.end_ms / 1000).toFixed(1)
  const durationSec = ((segment.end_ms - segment.start_ms) / 1000).toFixed(1)

  // Seek the preview <video> to the segment midpoint so the card thumbnail
  // reflects something close to the contact frame. We only need to do this
  // once per blob load.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const handleLoaded = () => {
      try {
        video.currentTime = midpointMs / 1000
      } catch {
        // Some browsers throw if duration isn't known yet; the subsequent
        // 'durationchange' handler will retry.
      }
    }
    video.addEventListener('loadedmetadata', handleLoaded)
    return () => {
      video.removeEventListener('loadedmetadata', handleLoaded)
    }
  }, [midpointMs])

  // Keep the default label in sync with the selected shot type until the
  // user explicitly edits the input. After that we respect their input.
  useEffect(() => {
    if (labelDirty) return
    setLabel(
      segment.label?.trim() || `Segment #${segment.segment_index + 1} — ${overrideShot}`,
    )
  }, [overrideShot, labelDirty, segment.label, segment.segment_index])

  const playSegment = () => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = segment.start_ms / 1000
    void video.play()
  }

  // Pause when we pass end_ms during a "Play this segment" run.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const handleTime = () => {
      if (video.currentTime * 1000 >= segment.end_ms) {
        video.pause()
        video.currentTime = midpointMs / 1000
      }
    }
    video.addEventListener('timeupdate', handleTime)
    return () => video.removeEventListener('timeupdate', handleTime)
  }, [segment.end_ms, midpointMs])

  // Release the <video> decoder and range requests on unmount. A multi-shot
  // rally clip can produce 10+ cards, and `preload="metadata"` keeps every
  // decoder warm until GC fires. Without this cleanup, navigating away or
  // unmounting any card leaves stale blobs + decoders pinned.
  useEffect(() => {
    return () => {
      const video = videoRef.current
      if (!video) return
      video.pause()
      video.removeAttribute('src')
      video.load()
    }
  }, [])

  const classifierShot = segment.shot_type
  const overrideDiffers = classifierIsBaselineShot && classifierShot !== overrideShot
  const needsOverrideConfirmation = !classifierIsBaselineShot && !overrideTouched
  const confidencePct = Number.isFinite(segment.confidence)
    ? Math.round((segment.confidence ?? 0) * 100)
    : null

  const handleSave = async () => {
    if (!canSave || saving || needsOverrideConfirmation) return
    await onSave(segment.id, {
      shotType: overrideShot,
      label: label.trim() || undefined,
    })
  }

  const cardSpan = expanded ? 'sm:col-span-2 lg:col-span-2' : ''
  const stripe = SHOT_STRIPE[overrideShot]

  return (
    <div className={`bg-cream text-ink flex flex-col ${cardSpan}`}>
      {/* Shot-type-tinted top stripe — visual identity per card. Updates
          as the user changes the override dropdown so the stripe always
          reflects what they'd actually save. */}
      <div className={`h-2 ${stripe}`} />
      <div className="p-3 flex flex-col gap-3">
        <div className="overflow-hidden bg-ink aspect-video relative">
          <video
            ref={videoRef}
            src={blobUrl}
            preload="metadata"
            muted
            playsInline
            className="w-full h-full object-contain"
          />
          {onToggleExpand && (
            <button
              type="button"
              onClick={onToggleExpand}
              className="absolute top-2 right-2 px-2.5 py-1 rounded-full bg-cream/90 hover:bg-cream text-ink text-[11px] font-semibold tracking-wide"
              aria-label={expanded ? 'Shrink card' : 'Enlarge card'}
              aria-pressed={expanded}
            >
              {expanded ? 'shrink' : 'enlarge'}
            </button>
          )}
          <button
            type="button"
            onClick={playSegment}
            className="absolute bottom-2 right-2 px-2.5 py-1 rounded-full bg-ink/80 hover:bg-ink text-cream text-[11px] font-semibold tracking-wide"
          >
            Play this segment
          </button>
        </div>

        <div className="flex items-center justify-between text-xs text-ink/60">
          <span>
            #{segment.segment_index + 1} &middot; {startSec}s to {endSec}s ({durationSec}s)
          </span>
          {confidencePct !== null && (
            <span className="text-ink/50">{confidencePct}% conf</span>
          )}
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-[10px] uppercase tracking-[0.16em] text-ink/50">
            shot type
          </label>
          <select
            value={overrideShot}
            onChange={(e) => {
              setOverrideTouched(true)
              setOverrideShot(e.target.value as BaselineShot)
            }}
            className="bg-cream-soft border border-ink/15 text-ink text-sm px-2 py-1.5"
            aria-label="Shot type override"
          >
            {BASELINE_SHOTS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          {needsOverrideConfirmation && (
            <p className="text-[11px] text-clay">
              Classified as {classifierShot} — pick a shot type to save this as a baseline.
            </p>
          )}
          {overrideDiffers && (
            <p className="text-[11px] text-clay">
              Classified as {classifierShot}, saving as {overrideShot}
            </p>
          )}
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-[10px] uppercase tracking-[0.16em] text-ink/50">
            label
          </label>
          <input
            value={label}
            onChange={(e) => {
              setLabelDirty(true)
              setLabel(e.target.value)
            }}
            className="bg-cream-soft border border-ink/15 text-ink text-sm px-2 py-1.5"
            placeholder="Label"
            maxLength={120}
          />
        </div>

        {errorMessage && (
          <p className="text-xs text-clay">{errorMessage}</p>
        )}
        {saved && (
          <p className="text-xs text-ink/70">Saved as baseline.</p>
        )}

        {signedIn ? (
          <button
            type="button"
            onClick={handleSave}
            disabled={!canSave || saving || needsOverrideConfirmation}
            className={`px-4 py-2 rounded-full text-xs font-semibold tracking-wide transition-colors ${
              !canSave || saving || needsOverrideConfirmation
                ? 'bg-ink/10 text-ink/40 cursor-not-allowed'
                : 'bg-ink text-cream hover:bg-ink-soft'
            }`}
          >
            {saving ? 'Saving...' : saved ? 'Saved' : 'Save as baseline'}
          </button>
        ) : (
          <p className="text-xs text-ink/55">Sign in to save this as a baseline.</p>
        )}
      </div>
    </div>
  )
}
