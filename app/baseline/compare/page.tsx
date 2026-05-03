'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { upload } from '@vercel/blob/client'
import JointTogglePanel from '@/components/JointTogglePanel'
import SwingSelector from '@/components/SwingSelector'
import { useBaselineStore } from '@/store/baseline'
import { useCompareHandoff } from '@/store'
import { usePoseExtractor } from '@/hooks/usePoseExtractor'
import { useUser } from '@/hooks/useUser'
import { detectSwings } from '@/lib/jointAngles'
import { extractPoseViaRailway, RailwayExtractError } from '@/lib/poseExtractionRailway'
import type { ExtractorBackend } from '@/lib/poseExtraction'
import type { PoseFrame, Baseline } from '@/lib/supabase'
import BackendChip from '@/components/BackendChip'

// Mirrors UploadZone's flag. When true the today's-clip extraction
// goes through Railway (RTMPose + YOLO crop) which produces meaningfully
// better tracing on small-in-frame subjects than browser MediaPipe.
// Browser MediaPipe stays as the fallback if Railway is unreachable
// or returns no result.
const USE_RAILWAY_EXTRACT =
  process.env.NEXT_PUBLIC_USE_RAILWAY_EXTRACT === 'true'

function safeBlobFilename(name: string): string {
  return (
    name
      .split(/[\\/]/)
      .pop()!
      .replace(/[^a-zA-Z0-9._-]/g, '_')
      .slice(0, 100) || 'upload'
  )
}

const ComparisonLayout = dynamic(() => import('@/components/ComparisonLayout'), { ssr: false })
const LLMCoachingPanel = dynamic(() => import('@/components/LLMCoachingPanel'), { ssr: false })

/**
 * Stub for camera-similarity detection. Agent B owns
 * `lib/cameraNormalization.ts` and may not have shipped
 * `computeCameraSimilarity` yet, so this returns null deterministically
 * and the page renders no warning. Once the lib lands, swap the impl
 * here — the call site below is wrapped in try/catch so a future
 * import error never blanks the page. The args are intentionally
 * accepted (and silenced via void) so consumers can wire real frames
 * through without touching the call site.
 */
function computeCameraWarning(
  baseline: PoseFrame[],
  today: PoseFrame[],
): string | null {
  void baseline
  void today
  return null
}

export default function BaselineComparePage() {
  const { baselines, activeBaseline, loading, error, refresh } = useBaselineStore()
  const { extract, progress, isProcessing } = usePoseExtractor()
  const { user, loading: authLoading } = useUser()
  const router = useRouter()
  // One-shot handoff from /analyze: when the user clicked
  // "Compare to baseline" on a swing card, the analyze page parked the
  // candidate (frames + blob URL + which swing index) in this store.
  // We consume it on mount, populate today's-clip state from it, and
  // clear so a back-and-return doesn't re-prefill.
  const pendingHandoff = useCompareHandoff((s) => s.pending)
  const clearHandoff = useCompareHandoff((s) => s.clearHandoff)

  const [todayObjectUrl, setTodayObjectUrl] = useState<string | null>(null)
  const [todayFrames, setTodayFrames] = useState<PoseFrame[]>([])
  // Surfaces which backend produced today's frames so the diagnostic
  // chip can render. Null until the first upload completes.
  const [todayExtractorBackend, setTodayExtractorBackend] =
    useState<ExtractorBackend | null>(null)
  // When backend is the red 'rtmpose-browser-fallback', this carries
  // the failure reason for the chip tooltip + inline label.
  const [todayFallbackReason, setTodayFallbackReason] = useState<string | null>(null)
  // Local processing state covering the WHOLE upload pipeline (blob
  // upload + Railway extraction + browser fallback). The hook's
  // isProcessing only flips while the browser extractor runs — when
  // the Railway fast-path skips browser extract entirely, there's no
  // signal to gate the click or surface progress, so the user just
  // sees "nothing happens" while Railway extracts for 30-60s.
  const [uploadBusy, setUploadBusy] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string>('')
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [selectedBaselineId, setSelectedBaselineId] = useState<string | null>(null)
  // Index (1-based) of the swing within today's clip the user wants to
  // compare against the baseline. Multi-swing clips would otherwise
  // confuse phase detection — detectSwingPhases finds ONE peak across
  // the whole video, so only one of N swings gets clean phase sync.
  // Slicing to one swing at a time fixes that.
  const [selectedTodaySwing, setSelectedTodaySwing] = useState<number | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Gate behind auth; load baselines once signed in.
  useEffect(() => {
    if (authLoading) return
    if (!user) {
      router.replace('/login?next=/baseline/compare')
      return
    }
    refresh()
  }, [authLoading, user, refresh, router])

  // Derive the effective selection without storing a default — the defaulting
  // logic falls back to the active baseline when the user hasn't explicitly picked one.
  const selectedBaseline: Baseline | null = useMemo(() => {
    if (selectedBaselineId) {
      const picked = baselines.find((b) => b.id === selectedBaselineId)
      if (picked) return picked
    }
    return activeBaseline
  }, [baselines, selectedBaselineId, activeBaseline])

  // Memoised so downstream useMemo deps don't recompute on every render
  // (a fresh `[]` literal would otherwise change identity each render).
  const baselineFrames = useMemo<PoseFrame[]>(
    () => selectedBaseline?.keypoints_json?.frames ?? [],
    [selectedBaseline]
  )

  // Detect swings within today's uploaded clip. Multi-swing clips
  // (e.g. "I just hit 12 forehands and uploaded the rally") get
  // segmented so each swing can be compared independently.
  const todaySwings = useMemo(() => detectSwings(todayFrames), [todayFrames])
  const hasMultipleTodaySwings = todaySwings.length > 1

  // The frames actually fed into ComparisonLayout / LLM. When the user
  // picks a specific swing, slice to it. When there's only one swing,
  // use the whole clip.
  const todayFramesForCompare = useMemo<PoseFrame[]>(() => {
    if (selectedTodaySwing != null && todaySwings[selectedTodaySwing - 1]) {
      return todaySwings[selectedTodaySwing - 1].frames
    }
    return todayFrames
  }, [selectedTodaySwing, todaySwings, todayFrames])

  // Camera-similarity warning (graceful future hook). Wrapped in
  // try/catch so a downstream throw never blanks the page; if the
  // helper hasn't been written yet we silently render nothing.
  const cameraWarning = useMemo<string | null>(() => {
    try {
      return computeCameraWarning(baselineFrames, todayFramesForCompare)
    } catch {
      return null
    }
  }, [baselineFrames, todayFramesForCompare])

  // Clean up any object URL on unmount / replacement
  useEffect(() => {
    return () => {
      if (todayObjectUrl) URL.revokeObjectURL(todayObjectUrl)
    }
  }, [todayObjectUrl])

  const processTodayVideo = async (file: File) => {
    if (!file.type.startsWith('video/')) {
      setUploadError('Please pick a video file (MP4, MOV, or WebM).')
      return
    }
    if (uploadBusy) return

    setUploadBusy(true)
    setUploadError(null)
    setUploadStatus('Uploading video…')

    try {
      let frames: PoseFrame[] | null = null
      let railwayBackend: ExtractorBackend | null = null
      let railwayFailReason: string | null = null
      // Track the blob URL across the Railway path so we don't need to
      // re-upload on a fallback to the browser extractor.
      let uploadedBlobUrl: string | null = null

      if (USE_RAILWAY_EXTRACT) {
        try {
          const blobPath = `videos/${Date.now()}-${safeBlobFilename(file.name)}`
          const blob = await upload(blobPath, file, {
            access: 'public',
            handleUploadUrl: '/api/upload',
            contentType: file.type,
          })
          uploadedBlobUrl = blob.url
          setUploadStatus('Creating session…')
          const sessRes = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ blobUrl: blob.url, shotType: selectedBaseline?.shot_type ?? 'unknown' }),
          })
          if (!sessRes.ok) {
            // Surface the underlying error so the user sees why
            // (matches UploadZone's pattern).
            let detail = ''
            try {
              const errBody = await sessRes.json()
              detail = errBody?.detail || errBody?.error || ''
            } catch {
              try { detail = await sessRes.text() } catch { detail = '' }
            }
            throw new Error(
              `create-pending failed: ${sessRes.status}${detail ? ` (${detail})` : ''}`,
            )
          }
          const { sessionId } = await sessRes.json()
          setUploadStatus('Analyzing pose on the server…')
          const result = await extractPoseViaRailway({
            sessionId,
            blobUrl: blob.url,
          })
          frames = result.frames
          railwayBackend = result.extractorBackend
        } catch (err) {
          if (err instanceof RailwayExtractError) {
            console.info(
              '[baseline/compare] Railway path unavailable, falling back to browser:',
              err.reason, err.message,
            )
            railwayFailReason =
              err.message && err.message !== err.reason
                ? `${err.reason}: ${err.message}`
                : err.reason
          } else {
            console.error('[baseline/compare] Railway path errored, falling back to browser:', err)
            railwayFailReason = err instanceof Error ? err.message : 'unknown'
          }
          frames = null
        }
      }

      // Railway-success fast path: skip the browser extractor entirely.
      // The browser extractor (onnxruntime-web RTMPose via
      // lib/browserPose) is heavy to load and can hang on iPhone Safari
      // if WASM init is slow — running it just to get an object URL is
      // wasted work that turns "Railway worked" into "stuck upload."
      // Mirrors the `usedRailway ? createObjectURL(file) : result.objectUrl`
      // pattern in components/UploadZone.tsx.
      if (frames && railwayBackend) {
        if (todayObjectUrl) URL.revokeObjectURL(todayObjectUrl)
        setTodayObjectUrl(URL.createObjectURL(file))
        setTodayFrames(frames)
        setTodayExtractorBackend(railwayBackend)
        setTodayFallbackReason(null)
        setSelectedTodaySwing(null)
        setUploadStatus('')
        return
      }

      // Browser fallback (Railway disabled, errored, or returned empty).
      setUploadStatus('Loading browser pose model…')
      const localResult = await extract(file)
      if (!localResult) {
        setUploadError('Pose extraction failed. Try a clearer angle with your full body in frame.')
        return
      }
      if (todayObjectUrl) URL.revokeObjectURL(todayObjectUrl)
      setTodayObjectUrl(localResult.objectUrl)
      setTodayFrames(localResult.frames)
      if (USE_RAILWAY_EXTRACT) {
        setTodayExtractorBackend('rtmpose-browser-fallback')
        setTodayFallbackReason(railwayFailReason)
      } else {
        setTodayExtractorBackend(localResult.extractorBackend)
        setTodayFallbackReason(null)
      }
      setSelectedTodaySwing(null)
      void uploadedBlobUrl  // referenced via closure; satisfy lint
    } catch (err) {
      console.error('[baseline/compare] upload pipeline failed:', err)
      setUploadError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploadBusy(false)
      setUploadStatus('')
    }
  }

  // When today's frames change (new upload), default to swing 1 if
  // there are multiple swings so phase sync engages immediately
  // instead of waiting for the user to discover the selector.
  useEffect(() => {
    if (todaySwings.length > 1 && selectedTodaySwing === null) {
      setSelectedTodaySwing(1)
    } else if (todaySwings.length <= 1 && selectedTodaySwing !== null) {
      setSelectedTodaySwing(null)
    }
  }, [todaySwings.length, selectedTodaySwing])

  // Consume any pending handoff from /analyze. We only run this once
  // per page load — the clear-after-consume guard handles
  // back-button-then-return without re-prefilling stale state. The
  // handoff video URL is the Vercel blob https URL, so the existing
  // revokeObjectURL cleanup at line ~117 stays a no-op for handoffs.
  useEffect(() => {
    if (!pendingHandoff) return
    if (authLoading || !user) return
    setTodayObjectUrl(pendingHandoff.videoSrc)
    setTodayFrames(pendingHandoff.frames)
    setTodayExtractorBackend(pendingHandoff.extractorBackend)
    setTodayFallbackReason(pendingHandoff.fallbackReason)
    setSelectedTodaySwing(pendingHandoff.preselectedSwingIndex)
    clearHandoff()
  }, [pendingHandoff, authLoading, user, clearHandoff])

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) processTodayVideo(file)
  }

  const hasToday = todayObjectUrl && todayFrames.length > 0
  const canCompare = selectedBaseline && baselineFrames.length > 0 && hasToday

  if (!loading && baselines.length === 0 && !error) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <h1 className="font-display font-extrabold text-3xl text-cream mb-3">
          No baseline saved yet.
        </h1>
        <p className="text-cream/70 mb-6">
          Analyze a swing first, then mark it as your baseline. You&apos;ll then be able to
          see how future swings stack up.
        </p>
        <Link
          href="/analyze"
          className="inline-flex items-center justify-center px-7 py-3.5 rounded-full bg-clay hover:bg-[#c4633f] text-cream text-sm font-semibold tracking-wide transition-colors"
        >
          Analyze a swing
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8 flex items-start justify-between gap-4">
        <div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-cream mb-2 leading-tight">
            Beat Your Last Swing
          </h1>
          <p className="text-cream/70">
            Your best day on the left. Today&apos;s swing on the right.
          </p>
        </div>
        <Link
          href="/baseline"
          className="shrink-0 text-sm text-cream/70 hover:text-cream transition-colors"
        >
          Manage baselines →
        </Link>
      </div>

      {error && (
        <div className="border border-clay/40 bg-clay/10 p-4 mb-6">
          <p className="text-clay-soft text-sm">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Baseline selector if multiple exist */}
          {baselines.length > 1 && (
            <div className="flex gap-2 flex-wrap">
              {baselines.map((b) => (
                <button
                  key={b.id}
                  onClick={() => setSelectedBaselineId(b.id)}
                  className={`px-3 py-1.5 text-xs font-medium tracking-wide transition-colors ${
                    b.id === selectedBaseline?.id
                      ? 'bg-clay text-cream'
                      : 'bg-cream/10 text-cream/70 hover:bg-cream/20 hover:text-cream'
                  }`}
                >
                  {b.label} · {b.shot_type}
                </button>
              ))}
            </div>
          )}

          {/* Upload zone for today's swing. */}
          {!hasToday && (
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => !uploadBusy && fileInputRef.current?.click()}
              className={`border-2 border-dashed p-12 text-center cursor-pointer transition-colors ${
                dragging
                  ? 'border-clay bg-clay/10'
                  : 'border-cream/20 hover:border-cream/40 hover:bg-cream/5'
              } ${uploadBusy ? 'cursor-not-allowed opacity-80' : ''}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) processTodayVideo(file)
                  // Reset so picking the same file twice still fires onChange.
                  if (e.target) e.target.value = ''
                }}
              />
              {uploadBusy ? (
                <div className="space-y-3">
                  <p className="text-cream">
                    {uploadStatus || `Processing today's swing… ${progress}%`}
                  </p>
                  <div className="w-full bg-cream/10 h-1.5 overflow-hidden">
                    <div
                      className="h-full bg-clay transition-all"
                      style={{ width: isProcessing ? `${progress}%` : '40%' }}
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <p className="text-cream font-medium">Upload today&apos;s swing</p>
                  <p className="text-cream/50 text-sm mt-1">MP4, MOV, WebM</p>
                </div>
              )}
            </div>
          )}
          {uploadError && (
            <div className="border border-clay/40 bg-clay/10 p-4">
              <p className="text-clay-soft text-sm">{uploadError}</p>
            </div>
          )}

          {/* Side-by-side comparison */}
          {canCompare && selectedBaseline && (
            <div className="border border-cream/10 overflow-hidden bg-ink">
              <div className="p-3 border-b border-cream/10 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <p className="text-sm text-cream/70">
                    <span className="text-clay-soft">{selectedBaseline.label}</span>
                    <span className="text-cream/40"> vs </span>
                    <span className="text-cream">Today</span>
                  </p>
                  <BackendChip
                    backend={todayExtractorBackend}
                    reason={todayFallbackReason}
                  />
                </div>
                <button
                  onClick={() => {
                    if (todayObjectUrl) URL.revokeObjectURL(todayObjectUrl)
                    setTodayObjectUrl(null)
                    setTodayFrames([])
                    setTodayExtractorBackend(null)
                    setTodayFallbackReason(null)
                  }}
                  className="text-xs text-cream/50 hover:text-cream"
                >
                  Upload different video
                </button>
              </div>
              {hasMultipleTodaySwings && (
                <div className="px-3 pt-3">
                  <SwingSelector
                    allFrames={todayFrames}
                    selectedIndex={selectedTodaySwing}
                    onSelect={(seg) => setSelectedTodaySwing(seg.index)}
                  />
                </div>
              )}
              <div className="p-3">
                <ComparisonLayout
                  userBlobUrl={selectedBaseline.blob_url}
                  userFrames={baselineFrames}
                  proFrames={todayFramesForCompare}
                  proVideoUrl={todayObjectUrl ?? ''}
                  userName="Baseline"
                  proName="Today"
                  // Baseline-compare mode: both clips are user-owned.
                  // Native 1x playback on both sides; longer side
                  // drives the loop; contact frames align via slave
                  // start-delay. Without this flag the layout
                  // compresses today's clip into baseline's duration
                  // and the user sees the 3-second clip cut to 2
                  // seconds and looped (the bug they reported).
                  compareMode="baseline"
                />
              </div>
            </div>
          )}

          {/* Soft camera-angle warning. Stubbed until Agent B's
              computeCameraSimilarity ships — currently always null and
              renders nothing. Wrapping is in the useMemo above so a
              future throw doesn't break the page. */}
          {canCompare && cameraWarning && (
            <div className="border border-hard-court/40 bg-hard-court/10 p-3">
              <p className="text-cream/80 text-sm">{cameraWarning}</p>
            </div>
          )}

          {canCompare && selectedBaseline && (
            // The panel's `frames` prop is the swing we want coached — today
            // — and `compareFrames` is the reference (baseline). Wiring them
            // the other way around silently swapped today/baseline values
            // through extractObservations, so the LLM saw flipped drift rows
            // ("today's hip rotation is X°" was actually the baseline number).
            <LLMCoachingPanel
              compareMode="baseline"
              frames={todayFramesForCompare}
              compareFrames={baselineFrames}
              baselineLabel={selectedBaseline.label}
            />
          )}
        </div>

        <div className="space-y-4">
          <JointTogglePanel />

          {selectedBaseline && (
            <div className="border border-cream/10 bg-cream/5 p-4">
              <h3 className="font-display font-bold text-sm text-cream mb-2">
                Comparing against
              </h3>
              <p className="text-cream font-medium truncate">{selectedBaseline.label}</p>
              <p className="text-cream/60 text-xs capitalize mt-0.5">
                {selectedBaseline.shot_type} ·{' '}
                {new Date(selectedBaseline.created_at).toLocaleDateString()}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
