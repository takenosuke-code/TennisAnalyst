'use client'

import { useState, useMemo, useEffect } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import JointTogglePanel from '@/components/JointTogglePanel'
import SwingSelector from '@/components/SwingSelector'
import SegmentPickerGrid from '@/components/SegmentPickerGrid'
import BestShotPanel from '@/components/BestShotPanel'
import SwingBaselineGrid, { type SwingBaselineSaveOverride } from '@/components/SwingBaselineGrid'
import BackendChip from '@/components/BackendChip'
import type { SegmentCardSaveOverride } from '@/components/SegmentCard'
import { usePoseStore, useJointStore, useCompareHandoff } from '@/store'
import { detectSwings, deriveFps } from '@/lib/jointAngles'
import { scoreStrokes } from '@/lib/strokeQuality'
import type { DetectedStroke } from '@/lib/strokeAnalysis'
import { useUser } from '@/hooks/useUser'
import { useProfile } from '@/hooks/useProfile'
import type { SwingSegment } from '@/lib/jointAngles'
import type { PoseFrame, VideoSegment } from '@/lib/supabase'

// Dynamic import to avoid SSR issues with MediaPipe WASM
const UploadZone = dynamic(() => import('@/components/UploadZone'), { ssr: false })
const VideoCanvas = dynamic(() => import('@/components/VideoCanvas'), { ssr: false })
const LLMCoachingPanel = dynamic(() => import('@/components/LLMCoachingPanel'), { ssr: false })
const StrokeRibbon = dynamic(() => import('@/components/StrokeRibbon'), { ssr: false })


export default function AnalyzePage() {
  const {
    framesData,
    blobUrl,
    localVideoUrl,
    sessionId,
    shotType,
    extractorBackend,
    fallbackReason,
  } = usePoseStore()
  const { visible, showSkeleton, showRacket, showAngles } = useJointStore()
  const [done, setDone] = useState(false)
  const [allFrames, setAllFrames] = useState<PoseFrame[]>([])
  const [selectedSwing, setSelectedSwing] = useState<number | null>(null)
  const { user, loading: authLoading } = useUser()
  const { profile, skipped, loading: profileLoading } = useProfile()
  const router = useRouter()
  const setCompareHandoff = useCompareHandoff((s) => s.setHandoff)
  const isAdvanced = !profileLoading && profile?.skill_tier === 'advanced'
  // Always render both arms. Earlier the forehand view suppressed the
  // off-hand elbow/wrist on the theory that a one-armed shot makes the
  // off-arm visual noise — in practice it reads as "the tracker can
  // only find one arm," which is the opposite of what we want in a
  // coaching demo. Dual-arm plumbing stays in renderPose /
  // renderAngleLabels for comparison views that want it; the analyze
  // page just passes null.
  const dominantHand: 'left' | 'right' | null = null
  // Show the "set your profile" hint only to users who explicitly skipped
  // onboarding — onboarded users already have a tier, anons can't persist one.
  const showProfileHint = !profileLoading && skipped && !profile

  // Per-segment baseline save state for the multi-shot grid. Kept on the
  // analyze page (not inside the grid) so optimistic UI survives grid
  // remounts and so multiple cards can't be mid-save simultaneously.
  const [segments, setSegments] = useState<VideoSegment[]>([])
  const [savingSegmentId, setSavingSegmentId] = useState<string | null>(null)
  const [savedSegmentIds, setSavedSegmentIds] = useState<Set<string>>(new Set())
  const [errorBySegmentId, setErrorBySegmentId] = useState<Record<string, string | null>>({})

  // Prefer persisted live-session swings (from video_segments) over a fresh
  // detectSwings() pass. The two formulas can disagree by ±1–2 swings — live
  // sees them at 15fps as they're emitted, post-hoc detectSwings re-derives
  // them from smoothed frames at the persisted FPS. Showing one count in
  // live and a different count in /analyze breaks trust; the persisted list
  // is the canonical record of what the player just did.
  const swings = useMemo<SwingSegment[]>(() => {
    if (segments.length > 0 && allFrames.length > 0) {
      const fromSegments: SwingSegment[] = segments
        .slice()
        .sort((a, b) => a.segment_index - b.segment_index)
        .map((seg, i) => {
          const startFrame = Math.max(0, Math.min(allFrames.length - 1, seg.start_frame))
          const endFrame = Math.max(startFrame, Math.min(allFrames.length - 1, seg.end_frame))
          const sliced = allFrames.slice(startFrame, endFrame + 1)
          return {
            index: i + 1,
            startFrame,
            endFrame,
            startMs: seg.start_ms,
            endMs: seg.end_ms,
            // No persisted peak frame — fall back to the midpoint of the
            // segment (sufficient for VideoCanvas's contact-frame coloring;
            // it just needs *a* frame inside the swing).
            peakFrame: startFrame + Math.floor(sliced.length / 2),
            frames: sliced,
          }
        })
        .filter((s) => s.frames.length > 0)
      if (fromSegments.length > 0) return fromSegments
    }
    // 2026-05 — temporarily reverted to no-options detectSwings.
    // Both dropRejected: true (scoreStrokes filtering) and
    // verifyShape: true (kinetic-chain verifier) were rejecting too
    // many real swings on amateur-quality pose, leaving the swing
    // grid empty. The flags remain plumbed through
    // detectSwings/detectStrokes so they can be re-enabled (behind a
    // feature flag or a profile setting) once we have user clips to
    // calibrate thresholds against. The wrist-speed prominence +
    // Voronoi-boundary core still runs by default, which already
    // suppresses most over-detection noise without rejecting real
    // swings.
    return detectSwings(allFrames)
  }, [segments, allFrames])
  const hasMultipleSwings = swings.length > 1

  // Contact frame for the currently-displayed swing. When multiple swings
  // exist we use the selected swing's peakFrame; otherwise the single swing's
  // peakFrame. Passed to VideoCanvas so the angle badges color-code against
  // pro benchmarks only on the contact frame — avoids a sea of red badges
  // during backswing/follow-through where the angles legitimately span a
  // wide range.
  const contactFrameIndex = useMemo<number | null>(() => {
    if (swings.length === 0) return null
    const active = selectedSwing != null ? swings[selectedSwing - 1] : swings[0]
    if (!active) return null
    // peakFrame is already an index into allFrames, and VideoCanvas uses
    // frame.frame_index for matching — they line up because extraction
    // assigns frame_index monotonically from 0.
    const peakFrame = active.frames[active.peakFrame - active.startFrame]
    return peakFrame?.frame_index ?? null
  }, [swings, selectedSwing])

  const handleComplete = (_url: string, frames: PoseFrame[]) => {
    setAllFrames(frames)
    const detected = detectSwings(frames)
    if (detected.length > 1) {
      setSelectedSwing(1)
    } else {
      setSelectedSwing(null)
    }
    setDone(true)
  }

  // Fetch video_segments rows for this session so we know whether to
  // render the multi-shot picker grid. The endpoint returns [] for
  // single-shot sessions, so segments.length > 1 is the signal.
  useEffect(() => {
    if (!done || !sessionId) {
      setSegments([])
      return
    }
    let ignore = false
    fetch(`/api/segments/${sessionId}`, { credentials: 'same-origin' })
      .then(async (res) => (res.ok ? ((await res.json()) as VideoSegment[]) : []))
      .then((rows) => {
        if (!ignore) setSegments(Array.isArray(rows) ? rows : [])
      })
      .catch(() => {
        if (!ignore) setSegments([])
      })
    return () => {
      ignore = true
    }
  }, [done, sessionId])

  // Reset per-segment save state when the user uploads a new video so
  // saved/error badges from the previous session don't bleed through.
  useEffect(() => {
    if (!done) {
      setSavingSegmentId(null)
      setSavedSegmentIds(new Set())
      setErrorBySegmentId({})
    }
  }, [done])

  const handleSaveSegmentAsBaseline = async (
    segmentId: string,
    override: SegmentCardSaveOverride,
  ) => {
    if (!sessionId || savingSegmentId) return
    setSavingSegmentId(segmentId)
    setErrorBySegmentId((prev) => ({ ...prev, [segmentId]: null }))
    try {
      const res = await fetch('/api/baselines/from-segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          sessionId,
          segmentId,
          shotTypeOverride: override.shotType,
          label: override.label,
        }),
      })
      if (!res.ok) {
        const msg = (await res.text()) || `HTTP ${res.status}`
        setErrorBySegmentId((prev) => ({ ...prev, [segmentId]: msg }))
        return
      }
      setSavedSegmentIds((prev) => {
        const next = new Set(prev)
        next.add(segmentId)
        return next
      })
    } catch (err) {
      setErrorBySegmentId((prev) => ({
        ...prev,
        [segmentId]: err instanceof Error ? err.message : 'Failed to save baseline',
      }))
    } finally {
      setSavingSegmentId((current) => (current === segmentId ? null : current))
    }
  }

  const handleSwingSelect = (seg: SwingSegment) => {
    setSelectedSwing(seg.index)
  }

  // ---- Stroke-ribbon inputs --------------------------------------------
  // Scores the in-memory `swings` via the real strokeQuality pipeline
  // (peak wrist speed + kinetic-chain timing error + wrist-angle
  // variance, z-scored across the session). No `comparison` object is
  // produced here — best/worst reasoning needs the LLM compare-strokes
  // route, which runs from a baseline context, not the solo upload
  // flow. Without it the ribbon hides Best/Worst badges and just
  // colors chips by score, which is the honest behavior for a
  // first-time analyze.
  const ribbonInputs = useMemo(() => {
    if (swings.length === 0) return null
    const strokes: DetectedStroke[] = swings.map((s) => ({
      strokeId: `swing-${s.index}`,
      startFrame: s.startFrame,
      endFrame: s.endFrame,
      peakFrame: s.peakFrame,
      fps: 30,
    }))
    const quality = scoreStrokes(strokes, allFrames)
    return { strokes, quality }
  }, [swings, allFrames])

  const selectedStrokeId =
    selectedSwing != null ? `swing-${selectedSwing}` : null

  // Page-level coaching scope: all detected swings concatenated. The
  // panel produces general advice across the full rally (e.g. "your
  // elbow drops in the second half of your sets"). Per-shot drill-down
  // happens inside SwingBaselineGrid's "Coach this shot" action, which
  // scopes a coach call to a single swing's frames.
  const allSwingFrames = useMemo<PoseFrame[]>(
    () => swings.flatMap((s) => s.frames),
    [swings],
  )

  // Per-swing baseline save state. The SwingBaselineGrid lets the user
  // save N independent swings from a long rally clip (e.g. 15 forehands
  // in one upload) — each becomes its own baseline row in user_baselines.
  // The grid does not fetch DB segments; it consumes the in-memory
  // detectSwings() output, so no Railway / segmentation pipeline is
  // required for this flow.
  const [savingSwingIndex, setSavingSwingIndex] = useState<number | null>(null)
  const [savedSwingIndices, setSavedSwingIndices] = useState<Set<number>>(new Set())
  const [errorBySwingIndex, setErrorBySwingIndex] = useState<Record<number, string | null>>({})

  // Reset per-swing save state when a new clip is uploaded so a previous
  // session's saved/saving flags don't leak into the new clip's grid.
  useEffect(() => {
    setSavingSwingIndex(null)
    setSavedSwingIndices(new Set())
    setErrorBySwingIndex({})
  }, [blobUrl])

  const handleSaveSwingAsBaseline = async (
    swingIndex: number,
    override: SwingBaselineSaveOverride,
  ) => {
    const swing = swings[swingIndex - 1]
    if (!swing || !blobUrl) return
    if (!sessionId) {
      // sessionId is required for /from-swing because the route looks
      // up the source video by user_sessions.id. Without it we'd have
      // to fall back to the old (broken) untrimmed POST /api/baselines
      // path. Surface clearly so the user knows to wait for upload to
      // finish or re-upload.
      setErrorBySwingIndex((prev) => ({
        ...prev,
        [swingIndex]: 'Save unavailable: missing session id. Re-upload and try again.',
      }))
      return
    }

    setSavingSwingIndex(swingIndex)
    setErrorBySwingIndex((prev) => ({ ...prev, [swingIndex]: null }))
    try {
      // The new /from-swing endpoint trims the source video to the
      // swing window via Railway and re-zeroes pose timestamps so the
      // saved baseline plays as just the swing clip (not the whole
      // 17-second rally video). The route derives start_ms/end_ms
      // server-side from startFrame/endFrame/fps — we send the frame
      // bounds, not the millisecond bounds, so a malicious client can't
      // crop outside the swing window.
      // SwingSegment doesn't carry fps explicitly — derive it from the
      // parent allFrames timestamps (same fps for the whole capture).
      // Falls back to 30 inside deriveFps when timestamps are missing.
      const sessionFps = deriveFps(allFrames)
      const keypointsJson = {
        fps_sampled: sessionFps,
        frame_count: swing.frames.length,
        frames: swing.frames,
      }
      const res = await fetch('/api/baselines/from-swing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          sessionId,
          startFrame: swing.startFrame,
          endFrame: swing.endFrame,
          peakFrame: swing.peakFrame,
          fps: sessionFps,
          shotType: override.shotType,
          label: override.label,
          keypointsJson,
        }),
      })
      if (!res.ok) {
        const msg = (await res.text()) || `HTTP ${res.status}`
        setErrorBySwingIndex((prev) => ({ ...prev, [swingIndex]: msg }))
        return
      }
      setSavedSwingIndices((prev) => {
        const next = new Set(prev)
        next.add(swingIndex)
        return next
      })
    } catch (err) {
      setErrorBySwingIndex((prev) => ({
        ...prev,
        [swingIndex]: err instanceof Error ? err.message : 'Failed to save baseline',
      }))
    } finally {
      setSavingSwingIndex((current) => (current === swingIndex ? null : current))
    }
  }

  // Park the selected swing in the handoff store and route to the
  // compare page. The compare page re-runs detectSwings on `allFrames`
  // and pre-selects swing N — same code path as a fresh upload + manual
  // pick, just with state pre-populated. Gated upstream on `!!blobUrl`
  // (the card hides the button without one) and on `signedIn` (compare
  // requires auth).
  const handleCompareSwingToBaseline = (swingIndex: number) => {
    if (!blobUrl || allFrames.length === 0) return
    setCompareHandoff({
      frames: allFrames,
      videoSrc: blobUrl,
      preselectedSwingIndex: swingIndex,
      extractorBackend,
      fallbackReason,
    })
    router.push('/baseline/compare')
  }

  return (
    <div className="max-w-7xl mx-auto px-5 sm:px-8 py-12">
      <div className="mb-10">
        <p className="text-[11px] uppercase tracking-[0.18em] text-cream/60 mb-3">Analyze</p>
        <h1 className="font-display font-extrabold text-cream text-4xl sm:text-5xl leading-[1.05] tracking-tight mb-3">
          Analyze Your Swing.
        </h1>
        <p className="text-cream/70 text-sm sm:text-base max-w-xl leading-relaxed">
          Upload A Video To Extract Pose Landmarks And Get AI Coaching Feedback.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: video + upload */}
        <div className="lg:col-span-2 space-y-6">
          {!done ? (
            <div className="bg-cream text-ink">
              <div className="h-2 bg-clay" />
              <div className="p-6">
                <h2 className="font-display font-bold text-lg text-ink mb-4">Upload Video</h2>
                <UploadZone onComplete={handleComplete} />
              </div>
            </div>
          ) : (
            blobUrl && (
              <div className="bg-ink overflow-hidden">
                <div className="p-3 border-b border-cream/10 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-cream">Your Swing</span>
                    <BackendChip backend={extractorBackend} reason={fallbackReason} />
                  </div>
                  <button
                    onClick={() => setDone(false)}
                    className="text-xs text-cream/50 hover:text-cream"
                  >
                    Upload Different Video
                  </button>
                </div>
                <VideoCanvas
                  src={localVideoUrl || blobUrl}
                  framesData={framesData}
                  visible={visible}
                  showSkeleton={showSkeleton}
                  showRacket={showRacket}
                  showAngles={showAngles}
                  shotType={shotType}
                  contactFrameIndex={contactFrameIndex}
                  dominantHand={dominantHand}
                  showControls
                  className="p-2"
                />
              </div>
            )
          )}

          {/* Swing selector for longer videos */}
          {done && hasMultipleSwings && (
            <SwingSelector
              allFrames={allFrames}
              selectedIndex={selectedSwing}
              onSelect={handleSwingSelect}
            />
          )}

          {/* Per-shot baseline grid. The ONLY save-as-baseline path —
              renders for any clip with at least one detected swing
              (single-shot uploads still go through this UI; the prior
              "save the whole 3-min video" path is gone because saving
              a multi-shot rally as a single baseline has no anatomical
              anchor). Long rallies produce N independently-saveable
              baselines. Uses the client-side detectSwings() output
              directly — no Railway / video-segments dependency. */}
          {done && swings.length >= 1 && blobUrl && (
            <SwingBaselineGrid
              swings={swings}
              blobUrl={blobUrl}
              defaultShotType={shotType ?? 'forehand'}
              signedIn={!!user}
              onSaveSwing={handleSaveSwingAsBaseline}
              onCompareSwing={handleCompareSwingToBaseline}
              savingSwingIndex={savingSwingIndex}
              savedSwingIndices={savedSwingIndices}
              errorBySwingIndex={errorBySwingIndex}
            />
          )}

          {/* AI pick of the best shot in the multi-segment set. Calls
              Anthropic with a compact angle summary per segment and renders
              "Shot N — short reasoning" above the grid so the user knows
              which one to focus on first. */}
          {done && sessionId && segments.length > 1 && (
            <BestShotPanel sessionId={sessionId} segmentCount={segments.length} />
          )}

          {/* Multi-shot picker. Rendered whenever the server detected >1
              segment for this session. Users can save any single segment
              as a baseline via /api/baselines/from-segment. The existing
              "Save as baseline" CTAs below are left alone per spec. */}
          {done && sessionId && blobUrl && segments.length > 1 && (
            <SegmentPickerGrid
              sessionId={sessionId}
              segments={segments}
              blobUrl={blobUrl}
              onSaveAsBaseline={handleSaveSegmentAsBaseline}
              signedIn={!!user}
              savingSegmentId={savingSegmentId}
              savedSegmentIds={savedSegmentIds}
              errorBySegmentId={errorBySegmentId}
            />
          )}

          {/* Subtle nudge for users who skipped onboarding — one line, no CTA
              button. Only rendered when an analysis is ready so it sits near
              the coaching output where the tailoring would apply. */}
          {done && showProfileHint && (
            <p className="text-xs text-cream/60">
              Not sure your level?{' '}
              <Link href="/profile" className="underline hover:text-cream">
                Tell us for tailored advice →
              </Link>
            </p>
          )}

          {/* Stroke-chip ribbon — SwingVision-style navigator. Sits
              between the video and the coaching panel. Quality scores
              come from lib/strokeQuality.ts (real, z-scored across the
              session); Best/Worst badges only appear when an LLM
              comparison is wired (baseline-compare flow). */}
          {done && ribbonInputs && (
            <StrokeRibbon
              strokes={ribbonInputs.strokes}
              quality={ribbonInputs.quality}
              selectedStrokeId={selectedStrokeId}
              onStrokeSelect={(id) => {
                const idx = Number(id.replace('swing-', ''))
                if (Number.isFinite(idx)) setSelectedSwing(idx)
              }}
            />
          )}

          {/* LLM coaching */}
          {done && (
            <LLMCoachingPanel frames={allSwingFrames} />
          )}

        </div>

        {/* Right: controls */}
        <div className="space-y-4">
          <JointTogglePanel />

          {framesData.length > 0 && (() => {
            // Joint-centric stats. Racket detection was pulled from the
            // demo (it was firing the wrist fallback ~97% of the time
            // and reading as "jittery racket line" rather than a real
            // signal). What we DO have reliably: per-frame joint
            // visibility + computed angles — those are what the overlay
            // and the LLM coach consume.
            const confidentFrames = framesData.filter((f) => {
              const ls = f.landmarks.find((l) => l.id === 11)
              const rs = f.landmarks.find((l) => l.id === 12)
              return (
                (ls?.visibility ?? 0) > 0.5 &&
                (rs?.visibility ?? 0) > 0.5
              )
            }).length
            const anglesPct =
              framesData.length === 0
                ? 0
                : Math.round((confidentFrames / framesData.length) * 100)
            return (
              <div className="bg-cream text-ink">
                <div className="h-2 bg-hard-court" />
                <div className="p-4">
                  <h3 className="font-display font-bold text-sm text-ink mb-3">Analysis stats</h3>
                  <div className="space-y-1.5 text-sm">
                    <div className="flex justify-between text-ink/65">
                      <span>Frames analyzed</span>
                      <span className="text-ink font-mono">{framesData.length}</span>
                    </div>
                    <div className="flex justify-between text-ink/65">
                      <span>Duration</span>
                      <span className="text-ink font-mono">
                        {((framesData[framesData.length - 1]?.timestamp_ms ?? 0) / 1000).toFixed(1)}s
                      </span>
                    </div>
                    <div className="flex justify-between text-ink/65">
                      <span>Joint data</span>
                      <span className="text-ink font-semibold">Ready</span>
                    </div>
                    <div className="flex justify-between text-ink/65">
                      <span>Upper-body tracking</span>
                      <span
                        className={
                          anglesPct >= 70
                            ? 'text-ink font-mono'
                            : 'text-clay font-mono'
                        }
                      >
                        {anglesPct}%
                      </span>
                    </div>
                  </div>
                  {anglesPct < 70 && (
                    <p className="text-[11px] text-clay mt-3 leading-relaxed">
                      Your upper body wasn&apos;t consistently visible in the
                      frame. For best joint-angle analysis, film
                      perpendicular to your baseline with your full body in
                      view.
                    </p>
                  )}
                </div>
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}
