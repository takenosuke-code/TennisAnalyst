'use client'

import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { useProLibraryStore, useJointStore } from '@/store'
import { getProVideoUrl } from '@/lib/proVideoUrl'
import { detectSwingPhases, getActivePhase } from '@/lib/syncAlignment'
import { getRecommendedVisibility } from '@/lib/shotTypeConfig'
import type { SwingPhase, PhaseTimestamp } from '@/lib/syncAlignment'
import { renderPose } from '@/components/PoseRenderer'
import VideoCanvas from '@/components/VideoCanvas'
import type { PoseFrame, ProSwing } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'

const PHASE_LABELS: Record<SwingPhase, string> = {
  preparation: 'Prep',
  backswing: 'Load',
  forward_swing: 'Swing',
  contact: 'Contact',
  follow_through: 'Finish',
}

const PHASE_COLORS: Record<SwingPhase, string> = {
  preparation: 'bg-blue-400',
  backswing: 'bg-amber-400',
  forward_swing: 'bg-orange-400',
  contact: 'bg-red-400',
  follow_through: 'bg-emerald-400',
}

// Real-time durations from sports biomechanics research (seconds):
//   Forehand: ~0.7s | Backhand: ~0.7s | Serve (full): ~1.5s | Volley: ~0.4s
const REAL_TIME_DURATIONS: Record<string, number> = {
  forehand: 0.7,
  backhand: 0.7,
  serve: 1.5,
  volley: 0.4,
}

export default function ProSwingViewer() {
  const selectedSwing = useProLibraryStore((s) => s.selectedSwing)
  const { visible, showSkeleton, showTrail, setVisibility } = useJointStore()

  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [speedMultiplier, setSpeedMultiplier] = useState(0) // 0 = original slow-mo, 1 = real-time
  const prevSwingIdRef = useRef<string | null>(null)

  const videoUrl = selectedSwing ? getProVideoUrl(selectedSwing) : null
  const frames = useMemo(
    () => selectedSwing?.keypoints_json?.frames ?? [],
    [selectedSwing]
  )
  const fps = selectedSwing?.fps ?? 30

  // Compute the playback rate needed for real-time speed.
  // Use actual video duration (from <video> element) when available,
  // otherwise fall back to DB duration_ms.
  const clipDurationSec = duration > 0
    ? duration
    : (selectedSwing?.duration_ms ? selectedSwing.duration_ms / 1000 : 0)

  const realTimeRate = useMemo(() => {
    if (clipDurationSec <= 0) return 1
    const shotType = selectedSwing?.shot_type ?? 'forehand'
    const estimatedRealDuration = REAL_TIME_DURATIONS[shotType] ?? 0.7
    return Math.min(16, clipDurationSec / estimatedRealDuration)
  }, [clipDurationSec, selectedSwing?.shot_type])

  // speedMultiplier: 0 = original slow-mo (1x), 1 = real-time, fractional = in between
  const playbackRate = speedMultiplier === 0 ? 1 : realTimeRate * speedMultiplier

  const phases = useMemo(() => detectSwingPhases(frames), [frames])
  const activePhase = useMemo(
    () => getActivePhase(phases, currentTime * 1000),
    [phases, currentTime]
  )

  // Reset playback state when swing changes. Local state resets happen during
  // render (React-supported pattern); the Zustand store update goes in an
  // effect to avoid "cannot update while rendering" errors.
  const swingId = selectedSwing?.id ?? null
  if (swingId !== prevSwingIdRef.current) {
    prevSwingIdRef.current = swingId
    if (selectedSwing) {
      setCurrentTime(0)
      setPlaying(false)
    }
  }

  useEffect(() => {
    if (selectedSwing) {
      setVisibility(getRecommendedVisibility(selectedSwing.shot_type))
    }
  }, [swingId, selectedSwing, setVisibility])

  const handleTimeUpdate = useCallback((t: number) => {
    setCurrentTime(t)
  }, [])

  const handleDurationChange = useCallback((d: number) => {
    setDuration(d)
  }, [])

  const handlePlayPause = useCallback((p: boolean) => {
    setPlaying(p)
  }, [])

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    const ms = Math.floor((s % 1) * 100)
    return `${m}:${sec.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`
  }

  if (!selectedSwing) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-12 text-center">
        <p className="text-white/40 text-sm">Select a swing to view.</p>
      </div>
    )
  }

  // --- Skeleton-only player (no video URL but keypoints exist) ---
  if (!videoUrl && frames.length > 0) {
    return (
      <SkeletonPlayer
        selectedSwing={selectedSwing}
        frames={frames}
        fps={fps}
        visible={visible}
        showSkeleton={showSkeleton}
        phases={phases}
        activePhase={activePhase}
        playing={playing}
        setPlaying={setPlaying}
        currentTime={currentTime}
        setCurrentTime={setCurrentTime}
        speedMultiplier={speedMultiplier}
        setSpeedMultiplier={setSpeedMultiplier}
        formatTime={formatTime}
      />
    )
  }

  if (!videoUrl) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-12 text-center">
        <p className="text-white/50 text-sm font-medium">No video or keypoints available</p>
      </div>
    )
  }

  return (
    <div className="rounded-2xl border border-white/10 overflow-hidden bg-black">
      {/* Header */}
      <div className="p-3 border-b border-white/5 flex items-center justify-between">
        <p className="text-sm text-white/60">
          {selectedSwing.pros?.name ?? 'Pro'} · <span className="capitalize">{selectedSwing.shot_type}</span>
        </p>
        {activePhase && (
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full ${PHASE_COLORS[activePhase]} text-black`}
          >
            {PHASE_LABELS[activePhase]}
          </span>
        )}
      </div>

      {/* Video — no skeleton overlay in Browse mode */}
      <div className="p-3 pb-0">
        <VideoCanvas
          src={videoUrl}
          framesData={[]}
          visible={visible}
          showSkeleton={false}
          showTrail={false}
          showControls={false}
          playbackRate={playbackRate}
          onTimeUpdate={handleTimeUpdate}
          onDurationChange={handleDurationChange}
          onPlayPause={handlePlayPause}
          syncPlaying={playing}
          syncedTime={currentTime}
          isPrimary
        />
      </div>

      {/* Custom controls */}
      <div className="p-3 space-y-2">
        {/* Scrubber with phase markers */}
        <div className="relative">
          <input
            type="range"
            min={0}
            max={duration || 1}
            step={0.001}
            value={currentTime}
            onChange={(e) => {
              const t = parseFloat(e.target.value)
              setCurrentTime(t)
              setPlaying(false)
            }}
            className="w-full accent-emerald-400 h-1 rounded-full"
          />
          {/* Phase marker ticks */}
          {duration > 0 &&
            phases.map((p) => {
              const pct = (p.timestampMs / 1000 / duration) * 100
              return (
                <div
                  key={p.phase}
                  className={`absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full ${PHASE_COLORS[p.phase]} pointer-events-none`}
                  style={{ left: `${pct}%` }}
                  title={PHASE_LABELS[p.phase]}
                />
              )
            })}
        </div>

        {/* Control row */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Play/Pause */}
          <button
            onClick={() => setPlaying(!playing)}
            className="w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors flex-shrink-0"
            aria-label={playing ? 'Pause' : 'Play'}
          >
            {playing ? (
              <svg viewBox="0 0 24 24" className="w-4 h-4 fill-current">
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="w-4 h-4 fill-current">
                <polygon points="5,3 19,12 5,21" />
              </svg>
            )}
          </button>

          {/* Frame step */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => {
                setPlaying(false)
                setCurrentTime(Math.max(0, currentTime - 1000 / fps / 1000))
              }}
              className="px-2 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white/70 text-xs font-medium transition-colors"
            >
              &lt; Prev
            </button>
            <button
              onClick={() => {
                setPlaying(false)
                setCurrentTime(Math.min(duration, currentTime + 1000 / fps / 1000))
              }}
              className="px-2 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white/70 text-xs font-medium transition-colors"
            >
              Next &gt;
            </button>
          </div>

          {/* Speed buttons */}
          <div className="flex items-center gap-1">
            {[
              { label: 'Original', value: 0 },
              { label: 'Real Speed', value: 1 },
              { label: '1/2x', value: 0.5 },
              { label: '1/4x', value: 0.25 },
            ].map(({ label, value }) => (
              <button
                key={value}
                onClick={() => setSpeedMultiplier(value)}
                className={`px-2 py-1 rounded-lg text-xs font-medium transition-all ${
                  speedMultiplier === value
                    ? 'bg-emerald-500 text-white'
                    : 'bg-white/10 text-white/50 hover:bg-white/20 hover:text-white'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Time display */}
          <span className="text-xs text-white/60 font-mono tabular-nums ml-auto flex-shrink-0">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Skeleton-only playback component (used when video URL is unavailable but
// keypoints data exists). Renders the pose skeleton on a canvas and drives
// animation via requestAnimationFrame.
// ---------------------------------------------------------------------------

interface SkeletonPlayerProps {
  selectedSwing: ProSwing
  frames: PoseFrame[]
  fps: number
  visible: Record<JointGroup, boolean>
  showSkeleton: boolean
  phases: PhaseTimestamp[]
  activePhase: SwingPhase | null
  playing: boolean
  setPlaying: (v: boolean) => void
  currentTime: number
  setCurrentTime: (v: number) => void
  speedMultiplier: number
  setSpeedMultiplier: (v: number) => void
  formatTime: (s: number) => string
}

function SkeletonPlayer({
  selectedSwing,
  frames,
  fps,
  visible,
  showSkeleton,
  phases,
  activePhase,
  playing,
  setPlaying,
  currentTime,
  setCurrentTime,
  speedMultiplier,
  setSpeedMultiplier,
  formatTime,
}: SkeletonPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const lastWallRef = useRef<number | null>(null)
  // Use a ref to track playback time inside the rAF loop to avoid
  // restarting the effect on every frame advance.
  const playTimeRef = useRef(currentTime)

  // Keep the ref in sync when the parent changes currentTime (e.g. scrub)
  useEffect(() => {
    playTimeRef.current = currentTime
  }, [currentTime])

  // Derive duration from the last frame's timestamp
  const durationSec = useMemo(() => {
    if (frames.length === 0) return 0
    return frames[frames.length - 1].timestamp_ms / 1000
  }, [frames])

  // Find the closest frame to a given time (in seconds)
  const getFrameAtTime = useCallback(
    (timeSec: number): PoseFrame | null => {
      if (frames.length === 0) return null
      const timeMs = timeSec * 1000
      let closest = frames[0]
      let minDiff = Math.abs(frames[0].timestamp_ms - timeMs)
      for (const frame of frames) {
        const diff = Math.abs(frame.timestamp_ms - timeMs)
        if (diff < minDiff) {
          minDiff = diff
          closest = frame
        }
      }
      return closest
    },
    [frames]
  )

  // Stable refs for values needed inside the rAF loop
  const visibleRef = useRef(visible)
  const showSkeletonRef = useRef(showSkeleton)
  const speedRef = useRef(speedMultiplier)
  const durationRef = useRef(durationSec)
  useEffect(() => { visibleRef.current = visible }, [visible])
  useEffect(() => { showSkeletonRef.current = showSkeleton }, [showSkeleton])
  useEffect(() => { speedRef.current = speedMultiplier }, [speedMultiplier])
  useEffect(() => { durationRef.current = durationSec }, [durationSec])

  // Draw a single frame at a given time
  const drawFrame = useCallback(
    (timeSec: number) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const dpr = window.devicePixelRatio || 1
      const displayW = canvas.width / dpr
      const displayH = canvas.height / dpr

      ctx.clearRect(0, 0, displayW, displayH)

      // Dark background
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, displayW, displayH)

      const frame = getFrameAtTime(timeSec)
      if (!frame) return

      renderPose(ctx, frame, displayW, displayH, {
        visible: visibleRef.current,
        showSkeleton: showSkeletonRef.current,
      })
    },
    [getFrameAtTime]
  )

  // Animation loop driven by requestAnimationFrame
  useEffect(() => {
    if (!playing) {
      lastWallRef.current = null
      drawFrame(playTimeRef.current)
      return
    }

    const tick = (wallTime: number) => {
      if (lastWallRef.current === null) {
        lastWallRef.current = wallTime
      }
      const deltaSec = ((wallTime - lastWallRef.current) / 1000) * speedRef.current
      lastWallRef.current = wallTime

      let next = playTimeRef.current + deltaSec
      if (next >= durationRef.current) {
        next = 0
        lastWallRef.current = null
      }

      playTimeRef.current = next
      setCurrentTime(next)
      drawFrame(next)
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing, drawFrame, setCurrentTime])

  // Redraw when scrubbing or visibility toggles change while paused
  useEffect(() => {
    if (!playing) {
      drawFrame(currentTime)
    }
  }, [currentTime, visible, showSkeleton, drawFrame, playing])

  // Set canvas resolution on mount / resize
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const setSize = () => {
      const rect = canvas.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      canvas.width = rect.width * dpr
      canvas.height = rect.height * dpr
      const ctx = canvas.getContext('2d')
      if (ctx) ctx.scale(dpr, dpr)
      drawFrame(playTimeRef.current)
    }

    const resizeObserver = new ResizeObserver(setSize)
    resizeObserver.observe(canvas)
    setSize()

    return () => resizeObserver.disconnect()
    // Only re-run on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const frameDurationSec = 1 / fps

  return (
    <div className="rounded-2xl border border-white/10 overflow-hidden bg-black">
      {/* Header */}
      <div className="p-3 border-b border-white/5 flex items-center justify-between">
        <p className="text-sm text-white/60">
          {selectedSwing.pros?.name ?? 'Pro'} ·{' '}
          <span className="capitalize">{selectedSwing.shot_type}</span>
          <span className="ml-2 text-white/30 text-xs">(skeleton)</span>
        </p>
        {activePhase && (
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full ${PHASE_COLORS[activePhase]} text-black`}
          >
            {PHASE_LABELS[activePhase]}
          </span>
        )}
      </div>

      {/* Skeleton canvas */}
      <div className="p-3 pb-0">
        <div className="relative rounded-xl overflow-hidden bg-black" style={{ aspectRatio: '16/9' }}>
          <canvas
            ref={canvasRef}
            className="w-full h-full block"
          />
        </div>
      </div>

      {/* Controls */}
      <div className="p-3 space-y-2">
        {/* Scrubber with phase markers */}
        <div className="relative">
          <input
            type="range"
            min={0}
            max={durationSec || 1}
            step={0.001}
            value={currentTime}
            onChange={(e) => {
              const t = parseFloat(e.target.value)
              setCurrentTime(t)
              setPlaying(false)
            }}
            className="w-full accent-emerald-400 h-1 rounded-full"
          />
          {/* Phase marker ticks */}
          {durationSec > 0 &&
            phases.map((p) => {
              const pct = (p.timestampMs / 1000 / durationSec) * 100
              return (
                <div
                  key={p.phase}
                  className={`absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full ${PHASE_COLORS[p.phase]} pointer-events-none`}
                  style={{ left: `${pct}%` }}
                  title={PHASE_LABELS[p.phase]}
                />
              )
            })}
        </div>

        {/* Control row */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Play/Pause */}
          <button
            onClick={() => setPlaying(!playing)}
            className="w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors flex-shrink-0"
            aria-label={playing ? 'Pause' : 'Play'}
          >
            {playing ? (
              <svg viewBox="0 0 24 24" className="w-4 h-4 fill-current">
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="w-4 h-4 fill-current">
                <polygon points="5,3 19,12 5,21" />
              </svg>
            )}
          </button>

          {/* Frame step */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => {
                setPlaying(false)
                setCurrentTime(Math.max(0, currentTime - frameDurationSec))
              }}
              className="px-2 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white/70 text-xs font-medium transition-colors"
            >
              &lt; Prev
            </button>
            <button
              onClick={() => {
                setPlaying(false)
                setCurrentTime(Math.min(durationSec, currentTime + frameDurationSec))
              }}
              className="px-2 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white/70 text-xs font-medium transition-colors"
            >
              Next &gt;
            </button>
          </div>

          {/* Speed buttons */}
          <div className="flex items-center gap-1">
            {[
              { label: 'Original', value: 0 },
              { label: 'Real Speed', value: 1 },
              { label: '1/2x', value: 0.5 },
              { label: '1/4x', value: 0.25 },
            ].map(({ label, value }) => (
              <button
                key={value}
                onClick={() => setSpeedMultiplier(value)}
                className={`px-2 py-1 rounded-lg text-xs font-medium transition-all ${
                  speedMultiplier === value
                    ? 'bg-emerald-500 text-white'
                    : 'bg-white/10 text-white/50 hover:bg-white/20 hover:text-white'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Time display */}
          <span className="text-xs text-white/60 font-mono tabular-nums ml-auto flex-shrink-0">
            {formatTime(currentTime)} / {formatTime(durationSec)}
          </span>
        </div>
      </div>
    </div>
  )
}
