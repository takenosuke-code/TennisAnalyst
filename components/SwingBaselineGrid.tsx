'use client'

import { useEffect, useRef, useState } from 'react'
import type { SwingSegment } from '@/lib/jointAngles'
import SegmentLightbox from './SegmentLightbox'

const BASELINE_SHOTS = ['forehand', 'backhand', 'serve', 'volley', 'slice'] as const
type BaselineShot = (typeof BASELINE_SHOTS)[number]

function isBaselineShot(v: string | undefined | null): v is BaselineShot {
  return typeof v === 'string' && (BASELINE_SHOTS as readonly string[]).includes(v)
}

export interface SwingBaselineSaveOverride {
  shotType: BaselineShot
  label?: string
}

interface SwingBaselineCardProps {
  swing: SwingSegment
  blobUrl: string
  defaultShotType: string
  onSave: (swingIndex: number, override: SwingBaselineSaveOverride) => void | Promise<void>
  // Optional handler to ferry this swing to the baseline-comparison
  // page. When provided, the card renders a "Compare to baseline"
  // button alongside Save. The grid owns the navigation; the card
  // just emits the swing index.
  onCompare?: (swingIndex: number) => void
  saving: boolean
  saved: boolean
  errorMessage: string | null
  signedIn: boolean
}

// One card per client-detected swing. Mirrors SegmentCard's visual style
// (video preview seeked to the swing's contact frame, shot-type override,
// label, save button) but works against the in-memory SwingSegment[]
// produced by detectSwings — no DB segment row required. This is what
// lets a user upload a clip with 15 forehands and save each one
// independently as a baseline without any server-side segmentation.
function SwingBaselineCard({
  swing,
  blobUrl,
  defaultShotType,
  onSave,
  onCompare,
  saving,
  saved,
  errorMessage,
  signedIn,
}: SwingBaselineCardProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const initialShot: BaselineShot = isBaselineShot(defaultShotType)
    ? defaultShotType
    : 'forehand'
  const [overrideShot, setOverrideShot] = useState<BaselineShot>(initialShot)
  const [label, setLabel] = useState<string>(`Swing ${swing.index} — ${initialShot}`)
  const [labelDirty, setLabelDirty] = useState(false)
  const [lightboxOpen, setLightboxOpen] = useState(false)

  // Contact-frame timestamp (peak activity) — used as the preview thumbnail
  // so each card shows the most representative moment of the swing.
  const peakFrame = swing.frames[swing.peakFrame - swing.startFrame]
  const peakMs = peakFrame?.timestamp_ms ?? Math.round((swing.startMs + swing.endMs) / 2)
  const startSec = (swing.startMs / 1000).toFixed(1)
  const endSec = (swing.endMs / 1000).toFixed(1)
  const durationSec = ((swing.endMs - swing.startMs) / 1000).toFixed(1)

  // Auto-seek the preview to the contact frame so the thumbnail is
  // visually distinctive per-swing (rather than every card showing the
  // start of the same long clip).
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const seek = () => {
      try {
        video.currentTime = peakMs / 1000
      } catch {
        // Some browsers throw if duration isn't ready yet — durationchange
        // re-fires below.
      }
    }
    video.addEventListener('loadedmetadata', seek)
    video.addEventListener('durationchange', seek)
    return () => {
      video.removeEventListener('loadedmetadata', seek)
      video.removeEventListener('durationchange', seek)
    }
  }, [peakMs])

  // "Play this swing" — start at start_ms, auto-pause at end_ms so the
  // preview doesn't bleed into the next swing.
  const playSwing = () => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = swing.startMs / 1000
    void video.play()
  }
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const onTime = () => {
      if (video.currentTime * 1000 >= swing.endMs) {
        video.pause()
        video.currentTime = peakMs / 1000
      }
    }
    video.addEventListener('timeupdate', onTime)
    return () => video.removeEventListener('timeupdate', onTime)
  }, [swing.endMs, peakMs])

  // Cleanup: release the video decoder when the card unmounts. With 15+
  // cards on a multi-swing rally, leaving decoders pinned drains memory.
  useEffect(() => {
    return () => {
      const video = videoRef.current
      if (!video) return
      video.pause()
      video.removeAttribute('src')
      video.load()
    }
  }, [])

  // Default label keeps shot-type sync until the user types in the input.
  useEffect(() => {
    if (labelDirty) return
    setLabel(`Swing ${swing.index} — ${overrideShot}`)
  }, [overrideShot, labelDirty, swing.index])

  const handleSave = async () => {
    if (saving || saved) return
    await onSave(swing.index, {
      shotType: overrideShot,
      label: label.trim() || undefined,
    })
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 flex flex-col gap-3">
      <div className="rounded-lg overflow-hidden bg-black aspect-video relative group">
        <video
          ref={videoRef}
          src={blobUrl}
          preload="metadata"
          muted
          playsInline
          className="w-full h-full object-contain"
        />
        {/* Transparent click target over the thumbnail; corner buttons
            sit above this layer so they still get their own clicks. */}
        <button
          type="button"
          onClick={() => setLightboxOpen(true)}
          aria-label={`Open swing ${swing.index} fullscreen`}
          className="absolute inset-0 cursor-zoom-in focus:outline-none focus-visible:ring-2 focus-visible:ring-white"
        >
          <span className="absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <span className="absolute left-2 bottom-2 text-white text-[11px] font-semibold tracking-wide opacity-0 group-hover:opacity-100 transition-opacity">
            Click to enlarge
          </span>
        </button>
        <button
          type="button"
          onClick={playSwing}
          className="absolute bottom-2 right-2 px-2 py-1 rounded bg-black/70 hover:bg-black text-white text-xs font-medium"
        >
          Play this swing
        </button>
      </div>
      <SegmentLightbox
        open={lightboxOpen}
        onClose={() => setLightboxOpen(false)}
        videoUrl={blobUrl}
        startMs={swing.startMs}
        endMs={swing.endMs}
        title={`Swing ${swing.index} — ${overrideShot}`}
      />

      <div className="flex items-center justify-between text-xs text-white/60">
        <span>
          Swing {swing.index} &middot; {startSec}s to {endSec}s ({durationSec}s)
        </span>
        <span className="text-white/40">{swing.frames.length}f</span>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-[11px] uppercase tracking-wide text-white/40">
          Shot type
        </label>
        <select
          value={overrideShot}
          onChange={(e) => setOverrideShot(e.target.value as BaselineShot)}
          className="bg-white/5 border border-white/10 text-white text-sm rounded px-2 py-1"
          aria-label="Shot type"
        >
          {BASELINE_SHOTS.map((s) => (
            <option key={s} value={s} className="bg-neutral-900">
              {s}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-[11px] uppercase tracking-wide text-white/40">
          Label
        </label>
        <input
          value={label}
          onChange={(e) => {
            setLabelDirty(true)
            setLabel(e.target.value)
          }}
          className="bg-white/5 border border-white/10 text-white text-sm rounded px-2 py-1"
          placeholder="Label"
          maxLength={120}
        />
      </div>

      {errorMessage && <p className="text-xs text-red-400">{errorMessage}</p>}
      {saved && <p className="text-xs text-emerald-300">Saved as baseline.</p>}

      {signedIn ? (
        <div className="flex flex-col gap-2">
          <button
            type="button"
            onClick={handleSave}
            disabled={saving || saved}
            className={`px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
              saving || saved
                ? 'bg-white/10 text-white/40 cursor-not-allowed'
                : 'bg-emerald-500 hover:bg-emerald-400 text-white'
            }`}
          >
            {saving ? 'Saving...' : saved ? 'Saved' : 'Save as baseline'}
          </button>
          {onCompare && (
            <button
              type="button"
              onClick={() => onCompare(swing.index)}
              className="px-3 py-2 rounded-lg text-sm font-semibold bg-white/10 hover:bg-white/15 text-white transition-colors"
            >
              Compare to baseline →
            </button>
          )}
        </div>
      ) : (
        <p className="text-xs text-white/50">Sign in to save or compare this swing.</p>
      )}
    </div>
  )
}

interface SwingBaselineGridProps {
  swings: SwingSegment[]
  blobUrl: string
  defaultShotType: string
  signedIn: boolean
  onSaveSwing: (swingIndex: number, override: SwingBaselineSaveOverride) => void | Promise<void>
  // Optional. When set, each card gets a "Compare to baseline" button
  // that hands the swing off to /baseline/compare via the
  // useCompareHandoff store.
  onCompareSwing?: (swingIndex: number) => void
  // Keyed by swing.index. Parent owns the state so per-card UI stays
  // consistent across rerenders.
  savingSwingIndex: number | null
  savedSwingIndices: ReadonlySet<number>
  errorBySwingIndex: Readonly<Record<number, string | null>>
}

/**
 * Grid of swing cards built from client-side detectSwings() output.
 * Each card lets the user save one swing as a baseline independently —
 * the whole point: a 15-forehand rally clip becomes 15 saveable
 * baselines instead of a single "save the selected swing" CTA.
 *
 * Renders nothing when there are <= 1 swings. The single-swing case is
 * handled by the existing baseline CTA on the analyze page.
 */
export default function SwingBaselineGrid({
  swings,
  blobUrl,
  defaultShotType,
  signedIn,
  onSaveSwing,
  onCompareSwing,
  savingSwingIndex,
  savedSwingIndices,
  errorBySwingIndex,
}: SwingBaselineGridProps) {
  if (swings.length <= 1) return null

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-4">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-white">
          {swings.length} swings detected
        </h3>
        <p className="text-xs text-white/40">
          Save each one independently as a baseline. Useful for tracking individual reps from a long rally clip.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {swings.map((swing) => (
          <SwingBaselineCard
            key={swing.index}
            swing={swing}
            blobUrl={blobUrl}
            defaultShotType={defaultShotType}
            onSave={onSaveSwing}
            onCompare={onCompareSwing}
            saving={savingSwingIndex === swing.index}
            saved={savedSwingIndices.has(swing.index)}
            errorMessage={errorBySwingIndex[swing.index] ?? null}
            signedIn={signedIn}
          />
        ))}
      </div>
    </div>
  )
}
