'use client'

import { useRef, useEffect, useCallback, useState } from 'react'
import type { PoseFrame } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'
import { renderPose, normalizeLandmarks } from './PoseRenderer'
import { SwingPathTracer } from './SwingPathTracer'

interface VideoCanvasProps {
  src: string
  framesData: PoseFrame[]
  visible: Record<JointGroup, boolean>
  showSkeleton: boolean
  showTrail: boolean
  overlayColor?: string
  overlaySkeletonColor?: string
  syncedTime?: number
  onTimeUpdate?: (t: number) => void
  onDurationChange?: (d: number) => void
  onPlayPause?: (playing: boolean) => void
  syncPlaying?: boolean
  isPrimary?: boolean
  isSecondary?: boolean
  timeMapping?: ((userTimeMs: number) => number) | null
  showControls?: boolean
  className?: string
  label?: string
  playbackRate?: number
  proStartDelay?: number
}

export default function VideoCanvas({
  src,
  framesData,
  visible,
  showSkeleton,
  showTrail,
  overlayColor,
  overlaySkeletonColor,
  syncedTime,
  onTimeUpdate,
  onDurationChange,
  onPlayPause,
  syncPlaying,
  isPrimary = true,
  isSecondary = false,
  timeMapping = null,
  showControls = true,
  className = '',
  label,
  playbackRate = 1.0,
  proStartDelay = 0,
}: VideoCanvasProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const tracerRef = useRef(new SwingPathTracer())
  const lastFrameIndexRef = useRef(-1)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [videoError, setVideoError] = useState(false)

  // Find the closest frame to a given video timestamp
  const getFrameAtTime = useCallback(
    (time: number): PoseFrame | null => {
      if (!framesData.length) return null
      const timeMs = time * 1000
      let closest = framesData[0]
      let minDiff = Math.abs(framesData[0].timestamp_ms - timeMs)
      for (const frame of framesData) {
        const diff = Math.abs(frame.timestamp_ms - timeMs)
        if (diff < minDiff) {
          minDiff = diff
          closest = frame
        }
      }
      return closest
    },
    [framesData]
  )

  // Main render loop
  const renderFrame = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    if (videoError) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Match canvas size to video display size
    if (
      canvas.width !== video.videoWidth ||
      canvas.height !== video.videoHeight
    ) {
      canvas.width = video.videoWidth || canvas.offsetWidth
      canvas.height = video.videoHeight || canvas.offsetHeight
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const frame = getFrameAtTime(video.currentTime)
    if (!frame) {
      rafRef.current = requestAnimationFrame(renderFrame)
      return
    }

    const displayFrame = overlayColor ? normalizeLandmarks(frame) : frame

    // Update swing trail
    if (showTrail) {
      if (frame.frame_index !== lastFrameIndexRef.current) {
        tracerRef.current.push(frame, canvas.width, canvas.height)
        lastFrameIndexRef.current = frame.frame_index
      }
      tracerRef.current.render(ctx)
    }

    // Draw pose
    renderPose(ctx, displayFrame, canvas.width, canvas.height, {
      visible,
      showSkeleton,
      color: overlayColor,
      skeletonColor: overlaySkeletonColor,
    })

    rafRef.current = requestAnimationFrame(renderFrame)
  }, [
    src,
    framesData,
    visible,
    showSkeleton,
    showTrail,
    overlayColor,
    overlaySkeletonColor,
    getFrameAtTime,
    videoError,
  ])

  useEffect(() => {
    rafRef.current = requestAnimationFrame(renderFrame)
    return () => cancelAnimationFrame(rafRef.current)
  }, [renderFrame])

  // Sync external time control
  useEffect(() => {
    if (syncedTime === undefined) return
    const video = videoRef.current
    if (!video) return

    // For secondary (pro) video: hold at start until the user's swing begins.
    // This prevents the pro video from stuttering during pre-swing dead time.
    if (isSecondary && proStartDelay > 0 && syncedTime < proStartDelay) {
      video.currentTime = 0
      video.pause()
      return
    }

    // For secondary (pro) video, apply phase-based time mapping if available.
    // syncedTime is in seconds (user video time); timeMapping expects/returns ms.
    let targetTime = syncedTime
    if (isSecondary && timeMapping) {
      targetTime = timeMapping(syncedTime * 1000) / 1000
    }

    // When paused, use precise seeking for scrubbing
    if (video.paused) {
      if (Math.abs(video.currentTime - targetTime) < 0.05) return
      video.currentTime = targetTime
      tracerRef.current.reset()
      lastFrameIndexRef.current = -1
      return
    }

    // When playing, adjust playbackRate to stay in sync (no seeking = no stutter)
    if (isSecondary) {
      const drift = targetTime - video.currentTime
      const baseRate = playbackRate

      if (Math.abs(drift) > 2) {
        // Way too far off - hard seek as last resort
        video.currentTime = targetTime
        video.playbackRate = baseRate
        tracerRef.current.reset()
        lastFrameIndexRef.current = -1
      } else if (drift > 0.3) {
        video.playbackRate = baseRate * 1.5
      } else if (drift > 0.1) {
        video.playbackRate = baseRate * 1.2
      } else if (drift < -0.3) {
        video.playbackRate = baseRate * 0.5
      } else if (drift < -0.1) {
        video.playbackRate = baseRate * 0.8
      } else {
        video.playbackRate = baseRate
      }
      return
    }

    // Primary video: standard sync
    if (Math.abs(video.currentTime - targetTime) < 0.05) return
    video.currentTime = targetTime
    tracerRef.current.reset()
    lastFrameIndexRef.current = -1
  }, [syncedTime, isSecondary, timeMapping, playbackRate, proStartDelay])

  // Apply playback rate to the video element (primary only; secondary is managed by sync effect)
  useEffect(() => {
    if (isSecondary) return
    const video = videoRef.current
    if (!video) return
    video.playbackRate = playbackRate
  }, [playbackRate, src, isSecondary])

  // Follow coordinated play/pause from the primary video
  useEffect(() => {
    if (syncPlaying === undefined) return
    const video = videoRef.current
    if (!video) return

    // Don't auto-play the secondary video while the user is still in pre-swing dead time
    if (isSecondary && proStartDelay > 0 && (syncedTime ?? 0) < proStartDelay) {
      video.pause()
      setPlaying(false)
      return
    }

    if (syncPlaying) {
      video.playbackRate = playbackRate
      video.play().catch(() => {})
      setPlaying(true)
    } else {
      video.pause()
      setPlaying(false)
    }
  }, [syncPlaying, playbackRate, isSecondary, proStartDelay, syncedTime])

  // Reset tracer and error state when src changes
  useEffect(() => {
    tracerRef.current.reset()
    lastFrameIndexRef.current = -1
    setVideoError(false)
  }, [src])

  const handleTimeUpdate = () => {
    const t = videoRef.current?.currentTime ?? 0
    setCurrentTime(t)
    // Only the primary video drives syncedTime to avoid a feedback loop
    // where two synced videos ping-pong time updates infinitely.
    if (isPrimary) {
      onTimeUpdate?.(t)
    }
  }

  const handleLoadedMetadata = () => {
    const d = videoRef.current?.duration ?? 0
    setDuration(d)
    onDurationChange?.(d)
  }

  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      video.play()
      setPlaying(true)
      onPlayPause?.(true)
    } else {
      video.pause()
      setPlaying(false)
      onPlayPause?.(false)
    }
  }

  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value)
    if (videoRef.current) videoRef.current.currentTime = t
    tracerRef.current.reset()
    lastFrameIndexRef.current = -1
    if (isPrimary) onTimeUpdate?.(t)
  }

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {label && (
        <div className="text-sm font-semibold text-center text-white/80 bg-black/40 rounded px-2 py-1">
          {label}
        </div>
      )}
      <div className="relative rounded-xl overflow-hidden bg-black">
        <video
          ref={videoRef}
          src={src}
          className="w-full block"
          playsInline
          muted
          loop
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onError={() => setVideoError(true)}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
        {videoError && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80">
            <p className="text-red-400 text-sm">Failed to load video</p>
          </div>
        )}
      </div>

      {showControls && (
        <div className="flex items-center gap-3 px-1">
          <button
            onClick={togglePlay}
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

          <input
            type="range"
            min={0}
            max={duration || 1}
            step={0.01}
            value={currentTime}
            onChange={handleScrub}
            className="flex-1 accent-emerald-400 h-1 rounded-full"
          />

          <span className="text-xs text-white/60 font-mono tabular-nums flex-shrink-0">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>
      )}
    </div>
  )
}
