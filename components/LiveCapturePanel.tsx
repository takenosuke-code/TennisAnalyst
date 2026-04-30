'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { upload } from '@vercel/blob/client'
import { useLiveCapture, type LiveSessionResult, type PoseQuality } from '@/hooks/useLiveCapture'
import { useLiveCoach } from '@/hooks/useLiveCoach'
import { useLiveStore, type LiveStatus } from '@/store/live'
import { usePoseStore } from '@/store'
import { renderPose } from './PoseRenderer'
import { getInterpolatedFrameAtTime } from './VideoCanvas'
import { createStreamingLandmarkSmoother, smoothFrames } from '@/lib/poseSmoothing'
import { JOINT_GROUPS, type JointGroup } from '@/lib/jointAngles'
import type { PoseFrame, KeypointsJson } from '@/lib/supabase'
import {
  clearOrphanedSession,
  getOrphanedSession,
  saveOrphanedSession,
  type OrphanedSession,
} from '@/lib/liveSessionRecovery'
import { extractPoseViaRailway, RailwayExtractError } from '@/lib/poseExtractionRailway'
import LiveSwingCounter from './LiveSwingCounter'

const SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley'] as const

// Joint groups are always-on for the live overlay — no toggles, no
// settings (per Phase 3 spec). Built once at module scope so the
// renderPose call doesn't allocate a fresh object every frame.
const ALL_JOINT_GROUPS_VISIBLE: Record<JointGroup, boolean> = Object.keys(JOINT_GROUPS).reduce(
  (acc, key) => {
    acc[key as JointGroup] = true
    return acc
  },
  {} as Record<JointGroup, boolean>,
)

// How long the swing-detected pulse animates the joint dots, dimming
// alpha 1.0 → 0.4 over this window. ~400ms feels like a fast confirm
// without hiding the skeleton through the next swing's prep phase.
const SWING_PULSE_DURATION_MS = 400

// Don't render the cold-start "Loading pose model…" UI for fast cache
// hits — onnxruntime-web's IndexedDB-cached weights make subsequent
// sessions resolve in <500ms, and a sub-300ms flash reads as visual
// noise more than progress. A setTimeout primes the loading visible
// flag at the 300ms mark; createPoseDetector clears it before that
// fires on a cache hit.
const MODEL_LOAD_VISIBLE_DELAY_MS = 300

// One pose-tick of overlay lag to align with the <video> preview's compositor delay (preview-pipeline lag, not jitter).
const OVERLAY_FRAME_DELAY = 1

type ModelLoadState = {
  loaded: number
  total: number
  label: 'yolo' | 'rtmpose'
}

function formatMb(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB'
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function statusLabel(status: LiveStatus): string {
  // Translate the bare-string status from the live store into the
  // human-readable footer copy required by Phase 3.
  switch (status) {
    case 'idle':
      return 'Ready'
    case 'preflight':
      return 'Ready'
    case 'recording':
      return 'Coaching live'
    case 'uploading':
      return 'Saving session'
    case 'complete':
      return 'Done'
    case 'error':
      return 'Error'
    default:
      return 'Ready'
  }
}

// Tracking-quality pill copy + color, driven by useLiveCapture's
// onPoseQuality callback. The 'wrong-angle' branch anticipates Worker
// 2's Phase 1.5 angle-classifier extension to PoseQuality; if that
// type union doesn't include it on this branch yet, the default
// branch keeps the renderer crash-free.
type PillState = {
  text: string
  dotClass: string
  textClass: string
}

function pillForQuality(q: PoseQuality | null): PillState | null {
  if (q == null) return null
  switch (q) {
    case 'good':
      return {
        text: 'Tracking',
        dotClass: 'bg-emerald-400',
        textClass: 'text-emerald-300',
      }
    case 'weak':
      return {
        text: 'Step back so your full body is in frame',
        dotClass: 'bg-amber-400',
        textClass: 'text-amber-300',
      }
    case 'no-body':
      return {
        text: 'Move into frame',
        dotClass: 'bg-red-400',
        textClass: 'text-red-300',
      }
    // Worker 2 (Phase 1.5) may extend the union with this state. Keep
    // the rendering code crash-free if it lands; until then the
    // default branch below covers any new value the union grows.
    case 'wrong-angle' as PoseQuality:
      return {
        text: 'Turn the phone — coaching needs side-on view',
        dotClass: 'bg-amber-400',
        textClass: 'text-amber-300',
      }
    default:
      return {
        text: 'Tracking',
        dotClass: 'bg-emerald-400',
        textClass: 'text-emerald-300',
      }
  }
}

interface LiveCapturePanelProps {
  onSessionComplete?: (result: LiveSessionResult) => void
}

function extForMime(mimeType: string): string {
  if (mimeType.startsWith('video/mp4')) return 'mp4'
  if (mimeType.startsWith('video/webm')) return 'webm'
  if (mimeType.startsWith('video/quicktime')) return 'mov'
  return 'bin'
}

type ReviewPhase = 'review' | 'uploading' | 'error' | 'done'

export default function LiveCapturePanel({ onSessionComplete }: LiveCapturePanelProps) {
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  // The latest body-visible pose frame. Stashed in a ref so 15fps
  // detection ticks don't trigger React re-renders — the rAF loop
  // reads from this on each draw.
  const latestFrameRef = useRef<PoseFrame | null>(null)
  // Wall-clock of the last `onSwing` event. The rAF loop dims joint
  // alpha based on `now - swingPulseAt < SWING_PULSE_DURATION_MS` so
  // the pulse decays without React state churn.
  const swingPulseAtRef = useRef<number>(0)
  const rafHandleRef = useRef<number | null>(null)
  const reviewVideoRef = useRef<HTMLVideoElement | null>(null)
  const reviewCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const [poseQuality, setPoseQuality] = useState<PoseQuality | null>(null)
  // Triggers the swing-counter scale bump. Toggles 0/1 each onSwing so
  // we always re-run the keyframe transition even on back-to-back hits.
  const [counterPulseTick, setCounterPulseTick] = useState<number>(0)
  // Debug pipeline stats — refreshed every ~500ms while recording so
  // the user can see WHY the tracking pill is whatever it is. Lifted
  // out of the rAF loop to avoid 60Hz state churn.
  const [debugStats, setDebugStats] = useState<{
    yoloDetections: number
    yoloTopScore: number
    kpMaxConf: number
    kpAvgConf: number
    inferenceMs: number
    provider: string | null
  } | null>(null)
  // Pre-start camera selection. Default: back camera ('environment') —
  // the canonical "prop your phone side-on" tennis setup. Users testing
  // the feature on a phone often want selfie mode to see themselves;
  // the toggle below the video flips this before Start.
  const [selectedFacingMode, setSelectedFacingMode] = useState<'user' | 'environment'>('environment')
  // Camera + container aspect orientation. Defaults to landscape, but
  // phones held portrait want a 9:16 stream into a 9:16 container —
  // otherwise object-cover crops the top and bottom of the source so
  // walking away from the camera shows only your legs (real bug
  // surfaced in user testing). We detect once at mount via matchMedia;
  // changing orientation mid-session would require re-getting the
  // stream which is out of scope here.
  const [aspect, setAspect] = useState<'portrait' | 'landscape'>('landscape')
  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    const mq = window.matchMedia('(orientation: portrait)')
    setAspect(mq.matches ? 'portrait' : 'landscape')
    const handler = (e: MediaQueryListEvent) => {
      // Only update before recording starts. Mid-session orientation
      // changes are intentionally ignored — the recording was made
      // with one orientation and shouldn't reframe halfway through.
      if (!isRecordingRef.current) {
        setAspect(e.matches ? 'portrait' : 'landscape')
      }
    }
    mq.addEventListener?.('change', handler)
    return () => mq.removeEventListener?.('change', handler)
  }, [])
  // isRecording is referenced inside the matchMedia handler above. We
  // mirror it into a ref so the handler doesn't need to be in the
  // useEffect dep list (which would re-bind on every recording flip
  // and miss orientation changes).
  const isRecordingRef = useRef(false)
  const [summary, setSummary] = useState<{ swings: number; durationMs: number } | null>(null)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)
  // The recorded session, held until the user picks Save / Discard / Re-record.
  const [pendingResult, setPendingResult] = useState<LiveSessionResult | null>(null)
  // Phase 6 — Railway re-extraction progress (0..100) and a flag the
  // review screen reads to decide whether to overlay the live keypoints
  // (initial state) or the server keypoints (post-success). The note
  // below the player ("Using on-device tracking — server extraction
  // unavailable") is shown only on Railway failure.
  const [extractProgress, setExtractProgress] = useState<number | null>(null)
  const [extractedKeypoints, setExtractedKeypoints] = useState<KeypointsJson | null>(null)
  const [usingFallback, setUsingFallback] = useState(false)
  // Phase C: when Modal extraction fails we pause on the review screen
  // and offer Retry / Continue with on-device. `attempts` includes the
  // initial automatic attempt; we cap manual retries at 2 (so attempts
  // can reach 3 total before the Retry button hides).
  const [extractFailure, setExtractFailure] = useState<{
    reason: string | null
    attempts: number
  } | null>(null)
  // Saved-session context held across the retry/continue choice. Without
  // this we'd have to re-run the upload + session-create steps on retry,
  // which would burn another Vercel Blob entry and create a duplicate row.
  const [pendingHandoff, setPendingHandoff] = useState<{
    sessionId: string
    blobUrl: string
    result: LiveSessionResult
    saveShotType: string
  } | null>(null)
  const [retryInFlight, setRetryInFlight] = useState(false)
  const MAX_EXTRACT_RETRIES = 2
  // Phase B: smoothed copy of the server-extracted keypoints, used only
  // for the review-screen overlay. Modal samples >2min clips at 8fps;
  // a One Euro pass here removes per-frame jitter that becomes visible
  // when getInterpolatedFrameAtTime synthesizes positions between those
  // 125ms-spaced samples on 30fps playback. Live fallback frames
  // (pendingResult.keypoints.frames) are already smoothed at stop time
  // by useLiveCapture, so they don't get a second pass. We deliberately
  // do NOT mutate `extractedKeypoints` itself or `extractedFrames` —
  // those flow to /analyze and `joint_angles` would go stale because
  // smoothFrames doesn't recompute angles. The render path
  // (getInterpolatedFrameAtTime) recomputes angles from interpolated
  // landmarks, so smoothed-landmark + recomputed-angle is consistent
  // there but only there.
  const smoothedExtractedFrames = useMemo(() => {
    if (!extractedKeypoints?.frames || extractedKeypoints.frames.length === 0) return null
    return smoothFrames(extractedKeypoints.frames)
  }, [extractedKeypoints])
  // Empirical skeleton-delay knob for the review screen. Positive value =
  // delay the rendered skeleton by N ms (i.e., look up PoseFrames N ms
  // earlier than `video.currentTime`). Surfaced as +/- buttons so the
  // player can dial in the offset when the live-keypoint anchor drifts
  // (createPoseDetector cold-start gap, MediaRecorder IDR-frame lag,
  // VFR-vs-CFR mismatch in the recorded blob — this knob compensates
  // for whichever is the live cause without us first having to identify
  // it). Default 0; persists only for the current review session.
  const [skeletonDelayMs, setSkeletonDelayMs] = useState(0)
  // Object URL for the inline review player. Created when the user enters
  // review, revoked when they leave it (or on unmount).
  const [reviewVideoUrl, setReviewVideoUrl] = useState<string | null>(null)
  // Where in the post-stop flow we are: review (just stopped), uploading,
  // error (retry available), or done (handed off to /analyze).
  const [reviewPhase, setReviewPhase] = useState<ReviewPhase>('review')
  // "Waiting for last cue…" UI when awaitInFlight is blocking on an
  // in-flight coach request at Stop time.
  const [waitingForCue, setWaitingForCue] = useState(false)
  // Mount-time orphan detection. If the previous session never finished
  // uploading, offer to resume.
  const [orphan, setOrphan] = useState<OrphanedSession | null>(null)
  // Phase 5D — first-time pose-model cold-start UX. modelLoadState holds
  // the per-model bytes-loaded snapshot from onModelLoadProgress; we
  // collapse the YOLO + RTMPose ticks into a single user-visible bar.
  // showModelLoadUi is the debounced "actually render the bar" flag —
  // gated behind a 300ms delay so cache-hit sessions don't flash a
  // useless loading state.
  const [modelLoadState, setModelLoadState] = useState<ModelLoadState | null>(null)
  const [showModelLoadUi, setShowModelLoadUi] = useState(false)
  const modelLoadDelayHandleRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const shotType = useLiveStore((s) => s.shotType)
  const setShotType = useLiveStore((s) => s.setShotType)
  const swingCount = useLiveStore((s) => s.swingCount)
  const setSwingCount = useLiveStore((s) => s.setSwingCount)
  const status = useLiveStore((s) => s.status)
  const setStatus = useLiveStore((s) => s.setStatus)
  const setErrorMessage = useLiveStore((s) => s.setErrorMessage)
  const errorMessage = useLiveStore((s) => s.errorMessage)
  const setSessionStartedAtMs = useLiveStore((s) => s.setSessionStartedAtMs)
  const resetSession = useLiveStore((s) => s.resetSession)
  const transcript = useLiveStore((s) => s.transcript)
  const setCoachRequestInFlight = useLiveStore((s) => s.setCoachRequestInFlight)

  const poseSetFramesData = usePoseStore((s) => s.setFramesData)
  const poseSetBlobUrl = usePoseStore((s) => s.setBlobUrl)
  const poseSetLocalVideoUrl = usePoseStore((s) => s.setLocalVideoUrl)
  const poseSetSessionId = usePoseStore((s) => s.setSessionId)
  const poseSetShotType = usePoseStore((s) => s.setShotType)

  const swingCountRef = useRef(0)
  const coach = useLiveCoach()

  // Keep the coach's shot type in sync with the user's selection.
  useEffect(() => {
    coach.setShotType(shotType)
  }, [coach, shotType])

  const { start, stop, abort, status: captureStatus, error: captureError, isRecording, pickedMimeType, facingMode, getLastDetectStats } = useLiveCapture({
    onSwing: (swing) => {
      swingCountRef.current++
      setSwingCount(swingCountRef.current)
      coach.pushSwing(swing)
      // Trigger the visual confirmation that this counter increment
      // came from a real motion: brief joint-dot dim (rAF-driven) +
      // a CSS scale bump on the counter (state-tick-driven).
      swingPulseAtRef.current = (typeof performance !== 'undefined' ? performance.now() : Date.now())
      setCounterPulseTick((n) => n + 1)
    },
    onStatus: (s) => {
      if (s === 'recording') setStatus('recording')
      if (s === 'error') setStatus('error')
      if (s === 'stopping') setStatus('uploading')
      if (s === 'idle') setStatus('idle')
    },
    onPoseFrame: (frame) => {
      // Stash the latest body-visible frame in a ref. The rAF loop
      // below reads it on every paint — never a React re-render at
      // 15fps.
      latestFrameRef.current = frame
    },
    onPoseQuality: (q) => {
      // React state IS okay for the pill: useLiveCapture only emits
      // onPoseQuality on transitions, not per-frame.
      setPoseQuality(q)
    },
    onModelLoadProgress: (loaded, total, label) => {
      // Phase 5D — surface YOLO + RTMPose download progress so the
      // pre-start screen can show "Loading pose model…" before the
      // first session goes live. Subsequent sessions pull from the
      // loader's cache and may fire a single (total, total) tick or
      // none at all — the 300ms debounce below keeps the UI quiet.
      setModelLoadState({ loaded, total, label })
    },
  })

  // Surface capture errors into the store
  useEffect(() => {
    if (captureError) setErrorMessage(captureError)
  }, [captureError, setErrorMessage])

  // Skeleton overlay rAF loop. Runs only while recording.
  //
  // We mirror the off-screen capture canvas (in useLiveCapture) which
  // produces landmarks in raw, un-mirrored coordinate space. The
  // <video> is CSS-mirrored for a selfie effect; the overlay canvas
  // gets the same `transform: scaleX(-1)` so landmarks land on the
  // on-screen body without any coordinate flipping in JS.
  useEffect(() => {
    if (!isRecording) return
    const canvas = overlayCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let cancelled = false
    let sizedKey: string | null = null
    // Streaming One Euro smoother for the rendered skeleton. Tuned for
    // BODY motion (the defaults in poseSmoothing.ts are pen/mouse from
    // Casiez 2012, which over-damp human-scale motion — beta=0.007 caps
    // adaptive cutoff at ~1Hz no matter how fast the wrist moves, so
    // fast swings barely register on the rendered skeleton). Body
    // tracking values per the One Euro paper's tuning section:
    //   minCutoff = 4Hz (lets fast joint motion through)
    //   beta = 0.7 (aggressive speedup on velocity)
    //   dcutoff = 1.0 (default for derivative smoothing)
    // Detector input is left raw — gates and swing detector see what
    // the model produced, only the on-screen skeleton is smoothed.
    const renderSmoother = createStreamingLandmarkSmoother({
      minCutoff: 4.0,
      beta: 0.7,
      dcutoff: 1.0,
    })
    let lastSmoothedFrameIndex = -1
    const frameBuffer: PoseFrame[] = []
    const BUFFER_LIMIT = OVERLAY_FRAME_DELAY + 1

    const draw = () => {
      if (cancelled) return
      const video = videoRef.current
      const dpr =
        typeof window !== 'undefined' && window.devicePixelRatio ? window.devicePixelRatio : 1
      const vw = video?.videoWidth || 0
      const vh = video?.videoHeight || 0

      // DPR-aware sizing pattern from VideoCanvas.tsx:124–146. Re-apply
      // ctx.scale only on (size, dpr) changes — ctx.scale is cumulative.
      if (vw > 0 && vh > 0) {
        const key = `${vw}x${vh}@${dpr}`
        if (sizedKey !== key) {
          canvas.width = vw * dpr
          canvas.height = vh * dpr
          ctx.setTransform(1, 0, 0, 1, 0, 0)
          ctx.scale(dpr, dpr)
          sizedKey = key
        }
      }

      const logicalW = (canvas.width || 0) / dpr
      const logicalH = (canvas.height || 0) / dpr
      ctx.clearRect(0, 0, logicalW, logicalH)

      const incoming = latestFrameRef.current
      // Smooth + buffer only when a NEW frame has arrived; rAF runs at
      // ~60Hz but pose ticks at ~15Hz, so we'd otherwise feed the
      // smoother the same landmarks 4× per pose tick and falsely zero
      // out its derivative.
      if (incoming && incoming.frame_index !== lastSmoothedFrameIndex) {
        lastSmoothedFrameIndex = incoming.frame_index
        const smoothed: PoseFrame = {
          ...incoming,
          landmarks: renderSmoother.smooth(
            incoming.landmarks,
            incoming.timestamp_ms / 1000,
          ),
        }
        frameBuffer.push(smoothed)
        if (frameBuffer.length > BUFFER_LIMIT) frameBuffer.shift()
      }

      // Render the oldest buffered frame (OVERLAY_FRAME_DELAY pose-ticks behind once warm).
      const renderFrame = frameBuffer.length > 0 ? frameBuffer[0] : null
      if (renderFrame && logicalW > 0 && logicalH > 0) {

        // Swing-detected pulse: dim joint dots from alpha 1.0 → 0.4
        // over SWING_PULSE_DURATION_MS, then snap back to 1.0. We
        // drive this through renderPose's `scale` (alpha multiplier)
        // arg per the Phase 3 spec ("Reuse the existing scale arg").
        const now = (typeof performance !== 'undefined' ? performance.now() : Date.now())
        const sinceSwing = now - swingPulseAtRef.current
        let alphaScale = 1
        if (
          swingPulseAtRef.current > 0 &&
          sinceSwing >= 0 &&
          sinceSwing < SWING_PULSE_DURATION_MS
        ) {
          const t = sinceSwing / SWING_PULSE_DURATION_MS
          alphaScale = 1 - t * 0.6 // 1.0 → 0.4
        }

        renderPose(ctx, renderFrame, logicalW, logicalH, {
          visible: ALL_JOINT_GROUPS_VISIBLE,
          showSkeleton: true,
          scale: alphaScale,
        })
      }

      rafHandleRef.current = window.requestAnimationFrame(draw)
    }

    rafHandleRef.current = window.requestAnimationFrame(draw)

    return () => {
      cancelled = true
      if (rafHandleRef.current != null) {
        window.cancelAnimationFrame(rafHandleRef.current)
        rafHandleRef.current = null
      }
      // Drop the latest-frame reference so a stale skeleton doesn't
      // flash if recording restarts before the next session produces
      // its first frame.
      latestFrameRef.current = null
      // Reset the streaming smoother — next session starts with no
      // accumulated state.
      renderSmoother.reset()
      // Reset pose quality so the pill clears between sessions.
      setPoseQuality(null)
    }
  }, [isRecording])

  // Phase 5D — debounced "show loading model" UI. We arm a 300ms timer
  // when a load tick first arrives; if the load completes before that
  // fires (cache-hit path), we never flip showModelLoadUi to true and
  // the user sees no flash. On cold start the timer fires, the bar
  // appears, and the bytes-loaded text drives the rest of the wait.
  useEffect(() => {
    // Clear the loading UI as soon as recording starts. The bar's job is
    // pre-start only — once the detector is ready and frames are flowing
    // there's nothing left to load.
    if (isRecording) {
      if (modelLoadDelayHandleRef.current != null) {
        clearTimeout(modelLoadDelayHandleRef.current)
        modelLoadDelayHandleRef.current = null
      }
      setShowModelLoadUi(false)
      setModelLoadState(null)
      return
    }
    // No load tick yet — nothing to show or schedule.
    if (modelLoadState == null) return
    // Already showing — no further work; the bar updates via state.
    if (showModelLoadUi) return
    // Already armed — let it fire.
    if (modelLoadDelayHandleRef.current != null) return
    modelLoadDelayHandleRef.current = setTimeout(() => {
      setShowModelLoadUi(true)
      modelLoadDelayHandleRef.current = null
    }, MODEL_LOAD_VISIBLE_DELAY_MS)
    return () => {
      if (modelLoadDelayHandleRef.current != null) {
        clearTimeout(modelLoadDelayHandleRef.current)
        modelLoadDelayHandleRef.current = null
      }
    }
  }, [isRecording, modelLoadState, showModelLoadUi])

  // Mirror coach.isRequestInFlight() into the live store so
  // LiveCoachingTranscript (rendered in app/live/page.tsx as a
  // sibling) can pulse its header without sharing a hook instance.
  // Polling at ~5Hz is plenty — the pulse is a coarse UX signal,
  // not a tight latency one. Stop polling between sessions.
  useEffect(() => {
    if (!isRecording) {
      setCoachRequestInFlight(false)
      return
    }
    let stopped = false
    const tick = () => {
      if (stopped) return
      setCoachRequestInFlight(coach.isRequestInFlight())
    }
    tick()
    const handle = window.setInterval(tick, 200)
    return () => {
      stopped = true
      window.clearInterval(handle)
      setCoachRequestInFlight(false)
    }
  }, [isRecording, coach, setCoachRequestInFlight])

  const handleStart = useCallback(async () => {
    const videoEl = videoRef.current
    if (!videoEl) return
    // Must prime TTS from the user-gesture handler so iOS Safari unlocks
    // speechSynthesis before the first coaching utterance.
    coach.primeTts()
    setErrorMessage(null)
    setSummary(null)
    swingCountRef.current = 0
    setSwingCount(0)
    setStatus('preflight')
    const startedAt = Date.now()
    setSessionStartedAtMs(startedAt)
    coach.markSessionStart(startedAt)
    await start(videoEl, { facingMode: selectedFacingMode, aspect })
  }, [coach, aspect, selectedFacingMode, setErrorMessage, setSessionStartedAtMs, setStatus, setSwingCount, start])

  // Mirror the live `isRecording` state into a ref the orientation
  // matchMedia handler reads — keeps that handler effect stable while
  // still seeing the latest recording status.
  useEffect(() => {
    isRecordingRef.current = isRecording
  }, [isRecording])

  // Poll the ONNX pipeline stats while recording. Decouples from the
  // 60Hz rAF render loop — 500ms is fast enough to feel live and slow
  // enough not to thrash React state.
  useEffect(() => {
    if (!isRecording) {
      setDebugStats(null)
      return
    }
    const tick = () => {
      const s = getLastDetectStats()
      if (s) {
        setDebugStats({
          yoloDetections: s.yoloDetections,
          yoloTopScore: s.yoloTopScore,
          kpMaxConf: s.kpMaxConf,
          kpAvgConf: s.kpAvgConf,
          inferenceMs: s.inferenceMs,
          provider: s.provider,
        })
      }
    }
    tick()
    const id = window.setInterval(tick, 500)
    return () => window.clearInterval(id)
  }, [isRecording, getLastDetectStats])

  // Run the upload + extract + save pipeline against an already-finalized
  // recording. Pulled out of handleStop so the Retry button (after a
  // failure) and the Resume-orphan path can re-fire it without
  // re-recording.
  //
  // Phase 6 flow:
  //   1. Upload blob to Vercel Blob.
  //   2. POST /api/sessions/live with mode='server-extract' — server creates
  //      the row in status='extracting' and parks the live keypoints in
  //      fallback_keypoints_json.
  //   3. extractPoseViaRailway({ sessionId, blobUrl, onProgress }) — Modal
  //      runs RTMPose on the GPU and writes the result back into
  //      keypoints_json on the row. Polling is internal to the helper.
  //   4. On success: POST /api/sessions/live/finalize with
  //      outcome='server-ok' to flip status to 'complete', then hydrate
  //      usePoseStore with the server frames.
  //   5. On any RailwayExtractError (not-configured, timeout, error-status,
  //      aborted, queue-failed): POST /api/sessions/live/finalize with
  //      outcome='server-failed' so the server copies the live fallback
  //      into keypoints_json. Hydrate the store with the LIVE frames so
  //      /analyze still works (chunkier tracers, but functional). The
  //      fallback note is surfaced inline in the review screen.
  // Phase C: shared post-extract handoff. Hydrates usePoseStore with
  // whichever frames we settled on (server-extracted or live fallback)
  // and navigates to /analyze. Pulled out of runSaveFlow so the
  // retry-after-failure path and the continue-with-fallback path land
  // on the same final state without re-running upload + session-create.
  const completeHandoff = useCallback((args: {
    result: LiveSessionResult
    sessionId: string
    blobUrl: string
    handoffFrames: PoseFrame[]
    saveShotType: string
    didFallBack: boolean
  }) => {
    if (args.didFallBack) {
      setUsingFallback(true)
    }
    void clearOrphanedSession()
    const objectUrl = URL.createObjectURL(args.result.blob)
    poseSetFramesData(args.handoffFrames)
    poseSetBlobUrl(args.blobUrl)
    poseSetLocalVideoUrl(objectUrl)
    poseSetSessionId(args.sessionId)
    poseSetShotType(args.saveShotType)
    setUploadStatus(null)
    setExtractProgress(null)
    setExtractFailure(null)
    setPendingHandoff(null)
    setStatus('complete')
    setReviewPhase('done')
    router.replace('/analyze')
  }, [
    poseSetBlobUrl,
    poseSetFramesData,
    poseSetLocalVideoUrl,
    poseSetSessionId,
    poseSetShotType,
    router,
    setStatus,
  ])

  const runSaveFlow = useCallback(async (
    result: LiveSessionResult,
    saveShotType: string,
  ) => {
    setReviewPhase('uploading')
    setStatus('uploading')
    setErrorMessage(null)
    setExtractProgress(null)
    setExtractedKeypoints(null)
    setUsingFallback(false)
    setExtractFailure(null)
    setPendingHandoff(null)

    // 1. Upload the blob.
    let blobUrl: string
    try {
      setUploadStatus('Uploading video…')
      const ext = extForMime(result.blobMimeType)
      const blobPath = `live/${Date.now()}-session.${ext}`
      const uploaded = await upload(blobPath, result.blob, {
        access: 'public',
        handleUploadUrl: '/api/upload',
        contentType: result.blobMimeType,
      })
      blobUrl = uploaded.url
    } catch (err) {
      // Persist the recording so the user can retry without losing it.
      void saveOrphanedSession(result.blob, result.keypoints, result.swings, saveShotType)
      setErrorMessage(err instanceof Error ? err.message : 'Upload failed')
      setUploadStatus(null)
      setStatus('error')
      setReviewPhase('error')
      return
    }

    const batchEventIds = transcript
      .map((e) => e.eventId)
      .filter((id): id is string => typeof id === 'string' && id.length > 0)

    // 2. Create the session row in 'server-extract' mode.
    let sessionId: string
    try {
      setUploadStatus('Saving session…')
      const res = await fetch('/api/sessions/live', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          blobUrl,
          shotType: saveShotType,
          mode: 'server-extract',
          keypointsJson: result.keypoints,
          swings: result.swings.map((s) => ({
            startFrame: s.startFrameIndex,
            endFrame: s.endFrameIndex,
            startMs: s.startMs,
            endMs: s.endMs,
          })),
          batchEventIds,
        }),
      })
      if (!res.ok) {
        const msg = await res.text().catch(() => 'Save failed')
        void saveOrphanedSession(result.blob, result.keypoints, result.swings, saveShotType)
        setErrorMessage(`Save failed: ${msg}`)
        setUploadStatus(null)
        setStatus('error')
        setReviewPhase('error')
        return
      }
      const body = (await res.json()) as { sessionId: string }
      sessionId = body.sessionId
    } catch (err) {
      void saveOrphanedSession(result.blob, result.keypoints, result.swings, saveShotType)
      setErrorMessage(err instanceof Error ? err.message : 'Save failed')
      setUploadStatus(null)
      setStatus('error')
      setReviewPhase('error')
      return
    }

    // 3. Re-extract via Railway/Modal. extractPoseViaRailway throws a
    // RailwayExtractError on any terminal failure; we catch and finalize
    // with outcome='server-failed' so the row still ends in
    // status='complete' (with the live keypoints copied into
    // keypoints_json by the server).
    setUploadStatus('Extracting…')
    setExtractProgress(25)
    let extractedFrames: PoseFrame[] | null = null
    let extractFailureReason: string | null = null
    try {
      const extractResult = await extractPoseViaRailway({
        sessionId,
        blobUrl,
        onProgress: (pct) => setExtractProgress(pct),
      })
      extractedFrames = extractResult.frames
      // Hold the server keypoints so the review-screen overlay can
      // swap in once they land. We mirror the live-keypoints shape so
      // the existing render code paths just work.
      setExtractedKeypoints({
        ...result.keypoints,
        frames: extractResult.frames,
        fps_sampled: extractResult.fps,
      })
    } catch (err) {
      if (err instanceof RailwayExtractError) {
        extractFailureReason = err.reason
      } else {
        extractFailureReason = 'unknown'
      }
    }

    // 4/5. Finalize the row. We don't surface a finalize-side network
    // failure as a user-facing error: the row may still settle correctly,
    // and the worst case is /analyze sees status='extracting' on a row
    // we can heal later. Logged for observability.
    const finalizeOutcome = extractedFrames ? 'server-ok' : 'server-failed'
    try {
      const finalizeRes = await fetch('/api/sessions/live/finalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId, outcome: finalizeOutcome }),
      })
      if (!finalizeRes.ok) {
        // Non-fatal — log and continue. The user-facing flow has
        // already succeeded (or has a fallback in hand); the row
        // just hasn't been told yet.
        console.warn(
          'sessions/live/finalize returned',
          finalizeRes.status,
          'outcome=', finalizeOutcome,
        )
      }
    } catch (err) {
      console.warn('sessions/live/finalize threw', err)
    }

    // Phase C: on extract failure we PAUSE on the review screen so the
    // user can retry against the same uploaded blob (no re-record, no
    // duplicate row) or accept the on-device fallback. The Phase 6
    // contract is preserved: finalize was already called with
    // outcome='server-failed' above, so the row is durably settled even
    // if the user closes the tab here. Retry success doesn't need to
    // re-finalize — extractPoseViaRailway writes keypoints_json itself
    // when Modal returns clean data.
    if (!extractedFrames) {
      console.info('Live save: Railway extraction unavailable, paused for retry/continue', extractFailureReason)
      setUsingFallback(true)
      setExtractFailure({ reason: extractFailureReason, attempts: 1 })
      setPendingHandoff({ sessionId, blobUrl, result, saveShotType })
      setUploadStatus(null)
      // reviewPhase stays at 'uploading' so the source pill keeps
      // rendering its diagnostic states; the retry/continue UI inside
      // the review screen reads extractFailure to decide what to show.
      return
    }

    // Happy path — extract succeeded, hand off to /analyze.
    completeHandoff({
      result,
      sessionId,
      blobUrl,
      handoffFrames: extractedFrames,
      saveShotType,
      didFallBack: false,
    })
  }, [
    completeHandoff,
    setErrorMessage,
    setStatus,
    transcript,
  ])

  const handleStop = useCallback(async () => {
    setUploadStatus('Finalizing recording…')
    const result = await stop()
    if (!result) {
      setUploadStatus(null)
      setStatus('idle')
      return
    }

    setSummary({ swings: result.swings.length, durationMs: result.durationMs })
    onSessionComplete?.(result)

    // Wait for any in-flight coach call to settle so its analysis_events row
    // is written before /api/sessions/live backfills session_id. Capped at
    // 5s so a wedged request can't block the player at Stop time.
    if (coach.isRequestInFlight()) {
      setWaitingForCue(true)
      setUploadStatus('Waiting for last cue…')
      await coach.awaitInFlight(5_000)
      setWaitingForCue(false)
    }

    // Hold the recording in review state. The user picks Save / Discard /
    // Re-record next; only Save kicks off the upload + persist flow.
    setPendingResult(result)
    setReviewPhase('review')
    setUploadStatus(null)
    setStatus('idle')
  }, [coach, onSessionComplete, setStatus, stop])

  // Whatever the live overlay was showing, the review screen replays it
  // by syncing each captured PoseFrame to the recorded video's playback
  // time. Lets the user verify the session was actually tracked AFTER
  // they stop, instead of seeing a blank video and wondering.
  //
  // Phase 6 — keep the overlay live during the 'uploading' phase too,
  // so the review video isn't blank while Railway is extracting. Render
  // server keypoints if they've landed (extractedKeypoints), otherwise
  // fall back to the live ones. A single state transition swaps them in
  // (no double-rAF, no canvas remount). The dependency on
  // extractedKeypoints below is what triggers the swap.
  useEffect(() => {
    if (!pendingResult) return
    if (reviewPhase !== 'review' && reviewPhase !== 'uploading') return
    const canvas = reviewCanvasRef.current
    const video = reviewVideoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const frames =
      smoothedExtractedFrames && smoothedExtractedFrames.length > 0
        ? smoothedExtractedFrames
        : pendingResult.keypoints.frames
    if (frames.length === 0) return

    let cancelled = false
    let sizedKey: string | null = null

    const draw = () => {
      if (cancelled) return
      const dpr = typeof window !== 'undefined' && window.devicePixelRatio ? window.devicePixelRatio : 1
      const vw = video.videoWidth || 0
      const vh = video.videoHeight || 0
      if (vw > 0 && vh > 0) {
        const key = `${vw}x${vh}@${dpr}`
        if (sizedKey !== key) {
          canvas.width = vw * dpr
          canvas.height = vh * dpr
          ctx.setTransform(1, 0, 0, 1, 0, 0)
          ctx.scale(dpr, dpr)
          sizedKey = key
        }
      }
      const logicalW = (canvas.width || 0) / dpr
      const logicalH = (canvas.height || 0) / dpr
      ctx.clearRect(0, 0, logicalW, logicalH)
      // Apply the empirical delay knob: positive `skeletonDelayMs`
      // looks up keypoints N ms BEFORE the current playback position,
      // which renders the skeleton N ms LATER (visually delayed).
      // Compensates for any structural offset between PoseFrame
      // timestamps and the recorded blob's PTS axis without us having
      // to identify the exact cause first.
      const lookupTimeSec = video.currentTime - skeletonDelayMs / 1000
      // Phase B: interpolate between bracketing samples so an 8fps server
      // keypoint stream (Modal's adaptive sampling for >2min clips) flows
      // smoothly across 30fps playback instead of snapping every ~125ms.
      const frame = getInterpolatedFrameAtTime(frames, lookupTimeSec)
      if (frame && logicalW > 0 && logicalH > 0) {
        renderPose(ctx, frame, logicalW, logicalH, {
          visible: ALL_JOINT_GROUPS_VISIBLE,
          showSkeleton: true,
        })
      }
    }

    const useRvfc =
      typeof HTMLVideoElement !== 'undefined' &&
      'requestVideoFrameCallback' in HTMLVideoElement.prototype
    let rafHandle = 0
    let rvfcHandle = 0

    if (useRvfc) {
      const tick: VideoFrameRequestCallback = () => {
        if (cancelled) return
        draw()
        rvfcHandle = video.requestVideoFrameCallback(tick)
      }
      // Paint immediately so the still frame at t=0 has skeleton too,
      // before playback even starts.
      draw()
      rvfcHandle = video.requestVideoFrameCallback(tick)
    } else {
      const loop = () => {
        if (cancelled) return
        draw()
        rafHandle = window.requestAnimationFrame(loop)
      }
      rafHandle = window.requestAnimationFrame(loop)
    }

    return () => {
      cancelled = true
      if (rafHandle) window.cancelAnimationFrame(rafHandle)
      if (rvfcHandle && typeof video.cancelVideoFrameCallback === 'function') {
        try { video.cancelVideoFrameCallback(rvfcHandle) } catch { /* ignore */ }
      }
    }
  }, [pendingResult, reviewPhase, reviewVideoUrl, smoothedExtractedFrames, skeletonDelayMs])

  // Aggregate session-quality summary used for the review-screen pill.
  // Computed once per pendingResult — averages the keypoint visibility
  // values across all captured frames (after the average-visibility fix
  // that ignores 0-visibility stub landmarks). Drives a green/amber/red
  // dot mirroring the live tracking pill so the user sees at a glance
  // whether the session was tracked well.
  const reviewQuality = (() => {
    if (!pendingResult) return null
    const frames = pendingResult.keypoints.frames
    if (frames.length === 0) return null
    let total = 0
    let n = 0
    for (const f of frames) {
      for (const lm of f.landmarks) {
        if (lm.visibility > 0) {
          total += lm.visibility
          n++
        }
      }
    }
    const avg = n === 0 ? 0 : total / n
    let label: string
    let dotClass: string
    let textClass: string
    if (avg >= 0.5) {
      label = 'Tracked well'
      dotClass = 'bg-emerald-400'
      textClass = 'text-emerald-300'
    } else if (avg >= 0.3) {
      label = 'Tracking weak'
      dotClass = 'bg-amber-400'
      textClass = 'text-amber-300'
    } else {
      label = 'Tracking failed'
      dotClass = 'bg-red-400'
      textClass = 'text-red-300'
    }
    return { avg, label, dotClass, textClass }
  })()

  // Build (and tear down) the inline review player's object URL. We only
  // create one URL per pendingResult so the <video> element stays stable
  // across re-renders.
  useEffect(() => {
    if (!pendingResult) return
    const url = URL.createObjectURL(pendingResult.blob)
    setReviewVideoUrl(url)
    return () => {
      URL.revokeObjectURL(url)
      setReviewVideoUrl(null)
    }
  }, [pendingResult])

  // On mount, see if a previous session never finished uploading. If so,
  // surface the resume prompt — the recording itself is already in IDB.
  useEffect(() => {
    let cancelled = false
    void getOrphanedSession().then((found) => {
      if (cancelled) return
      if (found) setOrphan(found)
    })
    return () => {
      cancelled = true
    }
  }, [])

  const handleSavePending = useCallback(async () => {
    if (!pendingResult) return
    await runSaveFlow(pendingResult, shotType)
  }, [pendingResult, runSaveFlow, shotType])

  const handleDiscardPending = useCallback(() => {
    setPendingResult(null)
    setSummary(null)
    setUploadStatus(null)
    setErrorMessage(null)
    setReviewPhase('review')
    setStatus('idle')
    setExtractProgress(null)
    setExtractedKeypoints(null)
    setUsingFallback(false)
    setExtractFailure(null)
    setPendingHandoff(null)
    setRetryInFlight(false)
    swingCountRef.current = 0
    setSwingCount(0)
  }, [setErrorMessage, setStatus, setSwingCount])

  const handleRerecordPending = useCallback(() => {
    handleDiscardPending()
    coach.reset()
    resetSession()
  }, [coach, handleDiscardPending, resetSession])

  const handleRetrySave = useCallback(async () => {
    if (!pendingResult) return
    setErrorMessage(null)
    await runSaveFlow(pendingResult, shotType)
  }, [pendingResult, runSaveFlow, setErrorMessage, shotType])

  // Phase C: re-run JUST the Modal extraction step against the already-
  // uploaded blob and existing session row. Unlike handleRetrySave, this
  // does not re-upload, does not create a duplicate row, and does not
  // call finalize (the row is already finalized as 'server-failed';
  // a successful retry will overwrite keypoints_json via Railway's
  // own write path). Capped at MAX_EXTRACT_RETRIES additional attempts.
  const handleExtractRetry = useCallback(async () => {
    if (!pendingHandoff || !extractFailure) return
    if (extractFailure.attempts > MAX_EXTRACT_RETRIES) return
    setRetryInFlight(true)
    setUploadStatus('Retrying server analysis…')
    setExtractProgress(25)
    try {
      const extractResult = await extractPoseViaRailway({
        sessionId: pendingHandoff.sessionId,
        blobUrl: pendingHandoff.blobUrl,
        onProgress: (pct) => setExtractProgress(pct),
      })
      setExtractedKeypoints({
        ...pendingHandoff.result.keypoints,
        frames: extractResult.frames,
        fps_sampled: extractResult.fps,
      })
      setUsingFallback(false)
      setRetryInFlight(false)
      // Hand off with the freshly extracted server frames.
      completeHandoff({
        result: pendingHandoff.result,
        sessionId: pendingHandoff.sessionId,
        blobUrl: pendingHandoff.blobUrl,
        handoffFrames: extractResult.frames,
        saveShotType: pendingHandoff.saveShotType,
        didFallBack: false,
      })
    } catch (err) {
      const reason = err instanceof RailwayExtractError ? err.reason : 'unknown'
      setExtractFailure({
        reason,
        attempts: extractFailure.attempts + 1,
      })
      setExtractProgress(null)
      setUploadStatus(null)
      setRetryInFlight(false)
    }
  }, [
    completeHandoff,
    extractFailure,
    pendingHandoff,
  ])

  const handleContinueWithFallback = useCallback(() => {
    if (!pendingHandoff) return
    completeHandoff({
      result: pendingHandoff.result,
      sessionId: pendingHandoff.sessionId,
      blobUrl: pendingHandoff.blobUrl,
      handoffFrames: pendingHandoff.result.keypoints.frames,
      saveShotType: pendingHandoff.saveShotType,
      didFallBack: true,
    })
  }, [completeHandoff, pendingHandoff])

  const handleResumeOrphan = useCallback(async () => {
    if (!orphan) return
    const result: LiveSessionResult = {
      blob: orphan.blob,
      blobMimeType: orphan.blob.type || 'video/webm',
      keypoints: orphan.keypoints,
      swings: orphan.swings,
      durationMs: 0,
    }
    setSummary({ swings: orphan.swings.length, durationMs: 0 })
    setPendingResult(result)
    setReviewPhase('review')
    setOrphan(null)
  }, [orphan])

  const handleDiscardOrphan = useCallback(() => {
    setOrphan(null)
    void clearOrphanedSession()
  }, [])

  // Kept for future "Hard reset" affordance (e.g. global error). The
  // post-stop flow uses the more targeted Discard / Re-record handlers
  // above instead — those preserve the streaming camera state.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _handleReset = useCallback(() => {
    abort()
    coach.reset()
    resetSession()
    swingCountRef.current = 0
    setSummary(null)
    setPendingResult(null)
    setReviewPhase('review')
  }, [abort, coach, resetSession])

  const busy = captureStatus === 'requesting-permissions' || captureStatus === 'initializing' || captureStatus === 'stopping'
  const displaySeconds = summary ? Math.max(1, Math.round(summary.durationMs / 1000)) : 0

  return (
    <div className="space-y-4">
      {/* Shot type picker + camera flip — both locked once recording starts */}
      <div className="flex items-center gap-2 flex-wrap">
        {SHOT_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => setShotType(type)}
            disabled={isRecording || busy}
            className={`px-4 py-1.5 rounded-full text-sm font-medium capitalize transition-all ${
              shotType === type
                ? 'bg-emerald-500 text-white'
                : 'bg-white/10 text-white/60 hover:bg-white/20 hover:text-white'
            } ${isRecording || busy ? 'opacity-60 cursor-not-allowed' : ''}`}
          >
            {type}
          </button>
        ))}
        {/*
          Camera flip — front (selfie) for testing on a phone you're
          holding, back ('environment') for the canonical "prop the
          phone side-on" tennis setup. Hidden during recording so a
          mid-session flip can't desync the recorder. Defaults to back.
        */}
        <button
          onClick={() =>
            setSelectedFacingMode((m) => (m === 'user' ? 'environment' : 'user'))
          }
          disabled={isRecording || busy}
          aria-pressed={selectedFacingMode === 'user'}
          data-testid="camera-flip-button"
          className={`ml-auto px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
            selectedFacingMode === 'user'
              ? 'bg-white/15 text-white border border-white/20'
              : 'bg-white/5 text-white/60 hover:bg-white/10 hover:text-white border border-white/10'
          } ${isRecording || busy ? 'opacity-40 cursor-not-allowed' : ''}`}
        >
          {selectedFacingMode === 'user' ? 'Selfie' : 'Back camera'}
        </button>
      </div>

      {/* Preview / idle card. Aspect matches the camera stream's
          orientation so object-cover doesn't crop the player vertically:
          landscape (16:9) for laptops + phones held sideways, portrait
          (9:16) for the canonical phone-propped-vertical tennis setup. */}
      <div
        className={`relative rounded-2xl border border-white/10 bg-black overflow-hidden mx-auto ${
          aspect === 'portrait' ? 'aspect-[9/16] max-w-sm' : 'aspect-video'
        }`}
      >
        {/*
          Mirror only the front (selfie) camera. The back camera films
          AWAY from the user — mirroring that flips the world in a
          confusing way (your right hand wave shows on the left side of
          a court layout you're trying to coach off). When recording
          uses the granted facingMode from useLiveCapture; before that,
          mirror to match the user's pre-start selection so the preview
          frame doesn't visibly flip the moment Start is tapped.
        */}
        {(() => {
          const activeFacing = facingMode ?? selectedFacingMode
          const mirrorTransform =
            activeFacing === 'user' ? 'scaleX(-1)' : 'none'
          return (
            <>
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                style={{ transform: mirrorTransform }}
                playsInline
                muted
              />
              {/*
                Skeleton overlay canvas. Mirror flag matches the video
                so RTMPose's un-mirrored coords land on the on-screen
                body. pointer-events:none keeps controls clickable.
              */}
              {isRecording ? (
                <canvas
                  ref={overlayCanvasRef}
                  data-testid="pose-overlay-canvas"
                  aria-hidden="true"
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  style={{ transform: mirrorTransform }}
                />
              ) : null}
            </>
          )
        })()}
        {!isRecording && !busy ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-gradient-to-br from-emerald-500/10 to-transparent">
            <div className="text-5xl">🎾</div>
            <p className="text-white font-medium">Lean your phone on anything side-on. Tap Start and play.</p>
            <p className="text-white/50 text-sm text-center max-w-sm px-4">
              Your bag, a bench, the fence — whatever you've got. We coach you every few swings and save the session for later review.
            </p>
          </div>
        ) : null}

        {isRecording ? (
          <div className="absolute top-3 left-3 space-y-2">
            {/*
              Wrap the counter in a key'd <div> so the scale
              transition re-runs every onSwing tick — React
              remounts the wrapper, retriggering the
              animate-[pulse-once] keyframes via Tailwind's
              animate-* class.
            */}
            <div
              key={`pulse-${counterPulseTick}`}
              data-testid="swing-counter-pulse"
              data-pulse-tick={counterPulseTick}
              className={
                counterPulseTick > 0
                  ? 'origin-top-left transition-transform duration-300 scale-110'
                  : 'origin-top-left'
              }
            >
              <LiveSwingCounter swingCount={swingCount} />
            </div>

            {/* Tracking-quality pill — driven by useLiveCapture's onPoseQuality */}
            {(() => {
              const pill = pillForQuality(poseQuality)
              if (!pill) return null
              return (
                <div
                  data-testid="tracking-quality-pill"
                  data-quality={poseQuality ?? 'unknown'}
                  role="status"
                  aria-live="polite"
                  className="inline-flex items-center gap-1.5 rounded-full bg-black/60 backdrop-blur px-2.5 py-1 text-xs"
                >
                  <span
                    aria-hidden
                    className={`inline-block w-1.5 h-1.5 rounded-full ${pill.dotClass}`}
                  />
                  <span className={pill.textClass}>{pill.text}</span>
                </div>
              )
            })()}

            {/*
              Diagnostic pill — shows ONNX pipeline state in real time.
              Y = YOLO person detections + top score; KP = max keypoint
              confidence from RTMPose; ms = total inference latency.
              Useful when "Move into frame" fires unexpectedly:
                Y:0 → YOLO didn't see a person (model load? threshold?)
                Y:1 0.20 → barely detected, scrunched at edge
                Y:1 0.85 KP:0 → YOLO ok, RTMPose silent (output map)
                Y:1 0.85 KP:0.95 → pipeline fine, gate downstream
            */}
            {debugStats ? (
              <div
                data-testid="onnx-debug-pill"
                className="inline-flex items-center gap-1 rounded-md bg-black/70 backdrop-blur px-2 py-1 text-[10px] font-mono text-white/70"
              >
                <span>Y:{debugStats.yoloDetections}</span>
                <span>{debugStats.yoloTopScore.toFixed(2)}</span>
                <span>·</span>
                <span>KP:{debugStats.kpMaxConf.toFixed(2)}/{debugStats.kpAvgConf.toFixed(2)}</span>
                <span>·</span>
                <span>{debugStats.inferenceMs}ms</span>
                {debugStats.provider ? <span>· {debugStats.provider}</span> : null}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      {/* Resume prompt — appears on mount when an orphaned upload exists. */}
      {orphan && !pendingResult && !isRecording ? (
        <div
          data-testid="resume-orphan-prompt"
          className="rounded-2xl border border-amber-400/30 bg-amber-500/10 px-5 py-4 space-y-3"
        >
          <p className="text-white font-medium">
            We saved a session you didn&apos;t finish uploading. Resume?
          </p>
          <p className="text-white/60 text-sm">
            {orphan.swings.length} {orphan.swings.length === 1 ? 'swing' : 'swings'} detected.
          </p>
          <div className="flex gap-2">
            <button
              onClick={handleResumeOrphan}
              className="bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl px-4 py-2 text-sm transition-all"
            >
              Resume
            </button>
            <button
              onClick={handleDiscardOrphan}
              className="bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl px-4 py-2 text-sm transition-all"
            >
              Discard
            </button>
          </div>
        </div>
      ) : null}

      {/* Phase 5D — first-time pose-model cold-start UI. Shown only on
          the truly-cold path (>300ms of bytes-arriving) and cleared the
          moment recording starts. The bar disables Start so the user
          can't fire a session before the detector is ready. */}
      {showModelLoadUi && modelLoadState && !isRecording && !pendingResult ? (
        <div
          data-testid="model-load-progress"
          className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-4 space-y-3"
        >
          <p className="text-white font-medium">Loading pose model…</p>
          <div
            data-testid="model-load-bar"
            className="h-2 w-full rounded-full bg-white/10 overflow-hidden"
          >
            <div
              className="h-full bg-emerald-500 transition-all"
              style={{
                width:
                  modelLoadState.total > 0
                    ? `${Math.min(100, Math.round((modelLoadState.loaded / modelLoadState.total) * 100))}%`
                    : '0%',
              }}
            />
          </div>
          <p className="text-white/60 text-xs">
            {formatMb(modelLoadState.loaded)} / {formatMb(modelLoadState.total)}
            {' · '}
            First-time only — future sessions start instantly.
          </p>
        </div>
      ) : null}

      {/* Controls — hidden once we're in the post-stop review flow so the
          Stop button can't fire a second time on a recording that's
          already finished. */}
      {!pendingResult ? (
        <div className="flex flex-col sm:flex-row gap-3">
          {!isRecording ? (
            <button
              onClick={handleStart}
              disabled={busy || showModelLoadUi}
              data-testid="start-button"
              className="flex-1 bg-emerald-500 hover:bg-emerald-400 disabled:bg-white/10 disabled:text-white/40 text-white font-bold rounded-2xl px-6 py-4 text-lg transition-all"
            >
              {showModelLoadUi ? 'Loading…' : busy ? 'Starting…' : 'Start'}
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex-1 bg-red-500 hover:bg-red-400 text-white font-bold rounded-2xl px-6 py-4 text-lg transition-all"
            >
              Stop
            </button>
          )}
        </div>
      ) : null}

      {/* Pre-upload review screen. The user just hit Stop; before we burn
          the upload roundtrip + Vercel Blob storage we let them inspect
          the recording and pick Save / Discard / Re-record.
          Phase 6: stays visible during the 'uploading' phase too so the
          tracked-skeleton overlay continues to render while Railway is
          extracting (the review video isn't blank during the wait). */}
      {pendingResult && (reviewPhase === 'review' || reviewPhase === 'uploading') ? (
        <div
          data-testid="review-screen"
          data-review-phase={reviewPhase}
          data-using-server-keypoints={extractedKeypoints ? 'true' : 'false'}
          data-using-fallback={usingFallback ? 'true' : 'false'}
          className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 space-y-4"
        >
          <p className="text-white font-semibold text-base">Review your session</p>
          {reviewVideoUrl ? (
            <div className="relative w-full rounded-xl overflow-hidden bg-black">
              <video
                ref={reviewVideoRef}
                data-testid="review-video"
                src={reviewVideoUrl}
                controls
                playsInline
                className="w-full"
                style={{ transform: facingMode === 'user' ? 'scaleX(-1)' : undefined }}
              />
              {/*
                Skeleton overlay synced to playback time. Mirrors the
                video's CSS transform so landmarks (in raw camera coords)
                land on the on-screen body. pointer-events:none keeps
                the native video controls clickable through the canvas.
              */}
              <canvas
                ref={reviewCanvasRef}
                data-testid="review-overlay-canvas"
                aria-hidden="true"
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ transform: facingMode === 'user' ? 'scaleX(-1)' : undefined }}
              />
              {reviewQuality ? (
                <div
                  data-testid="review-quality-pill"
                  className="absolute top-3 left-3 inline-flex items-center gap-1.5 rounded-full bg-black/60 backdrop-blur px-2.5 py-1 text-xs"
                >
                  <span
                    aria-hidden
                    className={`inline-block w-1.5 h-1.5 rounded-full ${reviewQuality.dotClass}`}
                  />
                  <span className={reviewQuality.textClass}>{reviewQuality.label}</span>
                  <span className="text-white/50 font-mono ml-1">
                    {(reviewQuality.avg * 100).toFixed(0)}%
                  </span>
                </div>
              ) : null}
              {/*
                Phase 6 fallback note. Surfaced only when Railway
                extraction finished as a failure and we're handing off
                to /analyze with the live keypoints. Corner-of-the-review
                placement keeps it ambient — the player doesn't need to
                act on it, just to know why the tracers will look
                chunkier than usual.
              */}
              {usingFallback ? (
                <div
                  data-testid="fallback-note"
                  className="absolute bottom-3 right-3 inline-flex items-center gap-1.5 rounded-full bg-black/60 backdrop-blur px-2.5 py-1 text-[11px] text-white/70"
                >
                  Using on-device tracking — server extraction unavailable
                </div>
              ) : null}
              {/*
                Phase C: source pill widened to four explicit states so
                the user knows (a) whether server analysis succeeded,
                (b) whether they're seeing the chunkier on-device fallback,
                (c) whether a retry is in progress, and (d) whether
                extraction has failed terminally and needs their input.
                  complete    → extractedKeypoints landed (happy path)
                  in-progress → extracting / retrying, nothing yet
                  unavailable → extractFailure latched, awaiting retry/continue
                  live        → fallback path post-continue
                Detail line keeps the diagnostic suffix (frames, fps,
                backend) so we don't lose the existing observability.
              */}
              {(() => {
                const isServer =
                  extractedKeypoints?.frames && extractedKeypoints.frames.length > 0
                const liveFrames = pendingResult?.keypoints?.frames?.length ?? 0
                const serverFrames = extractedKeypoints?.frames?.length ?? 0
                const inFlight =
                  retryInFlight ||
                  (reviewPhase === 'uploading' && !isServer && !usingFallback && !extractFailure)
                const failed = extractFailure != null && !isServer && !inFlight
                const sourceState: 'complete' | 'in-progress' | 'unavailable' | 'live' = isServer
                  ? 'complete'
                  : inFlight
                    ? 'in-progress'
                    : failed
                      ? 'unavailable'
                      : 'live'
                const label =
                  sourceState === 'complete'
                    ? 'Server analysis: complete'
                    : sourceState === 'in-progress'
                      ? 'Server analysis: in progress…'
                      : sourceState === 'unavailable'
                        ? 'Server analysis: unavailable'
                        : 'On-device tracking'
                const dotClass =
                  sourceState === 'complete'
                    ? 'bg-emerald-400'
                    : sourceState === 'in-progress'
                      ? 'bg-sky-400'
                      : sourceState === 'unavailable'
                        ? 'bg-red-400'
                        : 'bg-amber-400'
                const fps =
                  (isServer ? extractedKeypoints?.fps_sampled : pendingResult?.keypoints?.fps_sampled) ?? null
                const backendField = (extractedKeypoints as { pose_backend?: string } | null)
                  ?.pose_backend
                const detail = isServer
                  ? `${serverFrames} fr · ${fps ?? '?'}fps · ${backendField ?? 'rtmpose'}`
                  : `${liveFrames} fr · ${fps ?? '?'}fps · in-browser`
                return (
                  <div
                    data-testid="keypoints-source-pill"
                    data-source={sourceState}
                    className="absolute top-3 right-3 inline-flex items-center gap-1.5 rounded-full bg-black/60 backdrop-blur px-2.5 py-1 text-[11px] text-white/80"
                  >
                    <span aria-hidden className={`inline-block w-1.5 h-1.5 rounded-full ${dotClass}`} />
                    <span>{label}</span>
                    <span className="text-white/40 font-mono">· {detail}</span>
                  </div>
                )
              })()}
            </div>
          ) : null}
          {/*
            Empirical skeleton-delay knob. Positive value = render the
            skeleton N ms LATER than the playback position (i.e. look up
            keypoints N ms earlier in the array). User dials this in
            until the joints land on the body. -3000 / -1000 / -100 / 0 /
            +100 / +1000 / +3000 ms covers the realistic offset range
            from a cold-start async gap (~3s) down to fine compositor
            tweaks (±100ms).
          */}
          <div
            data-testid="skeleton-delay-controls"
            className="flex items-center gap-2 text-xs text-white/60"
          >
            <span className="text-white/50">Skeleton delay</span>
            <div className="inline-flex items-center gap-1 rounded-full bg-white/[0.04] border border-white/10 p-0.5">
              {[-3000, -1000, -100].map((step) => (
                <button
                  key={step}
                  type="button"
                  data-testid={`skeleton-delay-step-${step}`}
                  onClick={() => setSkeletonDelayMs((v) => v + step)}
                  className="px-2 py-1 rounded-full hover:bg-white/10 text-white/70 hover:text-white font-mono"
                >
                  {step > 0 ? `+${step}` : step}
                </button>
              ))}
              <span
                data-testid="skeleton-delay-value"
                className="px-3 py-1 font-mono text-white tabular-nums min-w-[64px] text-center"
              >
                {skeletonDelayMs > 0 ? `+${skeletonDelayMs}` : skeletonDelayMs}ms
              </span>
              {[100, 1000, 3000].map((step) => (
                <button
                  key={step}
                  type="button"
                  data-testid={`skeleton-delay-step-${step}`}
                  onClick={() => setSkeletonDelayMs((v) => v + step)}
                  className="px-2 py-1 rounded-full hover:bg-white/10 text-white/70 hover:text-white font-mono"
                >
                  +{step}
                </button>
              ))}
            </div>
            <button
              type="button"
              data-testid="skeleton-delay-reset"
              onClick={() => setSkeletonDelayMs(0)}
              className="text-white/40 hover:text-white/70 underline-offset-2 hover:underline"
            >
              reset
            </button>
          </div>
          <div className="flex flex-wrap gap-4 text-sm text-white/70">
            <span>
              <span className="text-white font-semibold">{summary?.swings ?? 0}</span>{' '}
              {summary?.swings === 1 ? 'swing' : 'swings'}
            </span>
            <span>
              <span className="text-white font-semibold">{displaySeconds}s</span> duration
            </span>
          </div>
          {reviewPhase === 'review' ? (
            <div className="flex flex-col sm:flex-row gap-2">
              <button
                onClick={handleSavePending}
                className="flex-1 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl px-4 py-3 transition-all"
              >
                Save
              </button>
              <button
                onClick={handleDiscardPending}
                className="flex-1 bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl px-4 py-3 transition-all"
              >
                Discard
              </button>
              <button
                onClick={handleRerecordPending}
                className="flex-1 bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl px-4 py-3 transition-all"
              >
                Re-record
              </button>
            </div>
          ) : extractFailure && !retryInFlight ? (
            // Phase C — Modal extraction has failed and we paused on the
            // review screen. Surface Retry and Continue with on-device.
            // Retry attempts are capped at MAX_EXTRACT_RETRIES; once
            // exhausted, only Continue is offered.
            <div data-testid="extract-failure" className="space-y-3">
              <p className="text-red-200 text-sm">
                Server analysis didn't complete
                {extractFailure.reason && extractFailure.reason !== 'unknown'
                  ? ` (${extractFailure.reason})`
                  : ''}
                . You can retry, or continue with on-device tracking and
                see the chunkier skeleton.
              </p>
              <div className="flex gap-3">
                {extractFailure.attempts <= MAX_EXTRACT_RETRIES ? (
                  <button
                    onClick={handleExtractRetry}
                    data-testid="extract-retry-button"
                    className="flex-1 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl px-4 py-3 transition-all"
                  >
                    Retry server analysis
                    {extractFailure.attempts > 1
                      ? ` (${MAX_EXTRACT_RETRIES + 1 - extractFailure.attempts} left)`
                      : ''}
                  </button>
                ) : null}
                <button
                  onClick={handleContinueWithFallback}
                  data-testid="extract-continue-button"
                  className="flex-1 bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl px-4 py-3 transition-all"
                >
                  Continue with on-device
                </button>
              </div>
            </div>
          ) : (
            // Phase 6 — Extracting… progress bar replaces the action
            // buttons during the uploading/extracting window. The bar
            // is driven by extractPoseViaRailway's onProgress callback
            // (advances through 25 → ~90, then jumps to 100 on
            // completion). On Railway-down or pre-extract failures
            // setExtractProgress is null and the bar stays at 0; that's
            // a visible-but-non-shouty indicator that the extract step
            // never started.
            <div data-testid="extract-progress" className="space-y-2">
              <p className="text-white/80 text-sm">
                {retryInFlight ? 'Retrying server analysis…' : 'Extracting…'}
              </p>
              <div
                data-testid="extract-progress-bar"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={extractProgress ?? 0}
                className="h-2 w-full rounded-full bg-white/10 overflow-hidden"
              >
                <div
                  data-testid="extract-progress-fill"
                  className="h-full bg-emerald-500 transition-all"
                  style={{ width: `${Math.min(100, Math.max(0, extractProgress ?? 0))}%` }}
                />
              </div>
            </div>
          )}
        </div>
      ) : null}

      {errorMessage ? (
        <div
          data-testid="error-box"
          className="rounded-xl border border-red-500/30 bg-red-500/10 text-red-200 text-sm px-4 py-3 space-y-3"
        >
          <p>{errorMessage}</p>
          {pendingResult && reviewPhase === 'error' ? (
            <div className="flex gap-2">
              <button
                onClick={handleRetrySave}
                className="bg-red-500 hover:bg-red-400 text-white font-semibold rounded-lg px-3 py-1.5 text-xs transition-all"
              >
                Retry
              </button>
              <button
                onClick={handleDiscardPending}
                className="bg-white/10 hover:bg-white/20 text-white font-medium rounded-lg px-3 py-1.5 text-xs transition-all"
              >
                Discard
              </button>
            </div>
          ) : null}
        </div>
      ) : null}

      {pickedMimeType && isRecording ? (
        <p className="text-white/40 text-xs">Recording {pickedMimeType}</p>
      ) : null}

      {uploadStatus ? (
        <div
          data-testid="upload-status"
          className="rounded-xl border border-white/10 bg-white/[0.04] text-white text-sm px-4 py-3"
        >
          {waitingForCue ? 'Waiting for last cue…' : uploadStatus}
        </div>
      ) : null}

      {/* Hand-off summary — shown only after the upload+save chain has
          completed and we're navigating to /analyze. */}
      {summary && reviewPhase === 'done' && !uploadStatus ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] px-5 py-4 space-y-1">
          <p className="text-white font-medium">
            {summary.swings} {summary.swings === 1 ? 'swing' : 'swings'} detected in {displaySeconds}s
          </p>
          <p className="text-white/50 text-sm">
            Opening the session in Analyze…
          </p>
        </div>
      ) : null}

      <p className="text-white/40 text-xs leading-relaxed">
        Status: <span data-testid="status-label" className="text-white/60">{statusLabel(status)}</span>
        {' · '}
        Any side-on setup works — prop the phone on your bag, a bench, the fence. Just keep your full body in frame. Front and back-of-court angles are too noisy to coach from.
      </p>
    </div>
  )
}
