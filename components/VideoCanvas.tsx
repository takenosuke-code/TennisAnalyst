'use client'

import { useRef, useEffect, useCallback, useState } from 'react'
import type { PoseFrame, Landmark } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'
import { computeJointAngles } from '@/lib/jointAngles'
import { renderPose, renderAngleLabels, normalizeLandmarks } from './PoseRenderer'
import { SwingPathTracer } from './SwingPathTracer'

// Nearest-neighbor lookup: return whichever bracketing pose sample is
// closest in time to the requested moment. At 30fps sampling the worst
// case lag is ±16ms (half a sample interval) instead of strict-
// previous's up-to-33ms one-sided lag. We were on strict-previous
// because of an old "joints moving ahead of the person" complaint, but
// that turned out to trace to a separate predictive racket-extension
// bug, not to nearest-neighbor itself. Switching back here meaningfully
// reduces the visible drift between the player and the skeleton during
// fast motion. Still no interpolation — we never synthesize a position
// the detector never produced.
export function getFrameAtTime(frames: PoseFrame[], timeSec: number): PoseFrame | null {
  if (!frames.length) return null
  const timeMs = timeSec * 1000
  if (timeMs <= frames[0].timestamp_ms) return frames[0]
  const last = frames[frames.length - 1]
  if (timeMs >= last.timestamp_ms) return last
  for (let i = 1; i < frames.length; i++) {
    if (frames[i].timestamp_ms > timeMs) {
      const prev = frames[i - 1]
      const next = frames[i]
      const dPrev = timeMs - prev.timestamp_ms
      const dNext = next.timestamp_ms - timeMs
      return dPrev <= dNext ? prev : next
    }
  }
  return last
}

// Linear-interpolated lookup for the live review screen, where server
// keypoints arrive at 8fps (Modal's adaptive sampling for >2min clips)
// against a 30fps recorded video. Nearest-neighbor on that pairing
// snaps the skeleton every ~125ms; this synthesizes positions between
// bracketing samples so the overlay flows.
//
// /analyze still uses getFrameAtTime — its keypoints are typically at
// the source FPS and snap-free, and an earlier interpolation experiment
// there reintroduced a "skeleton ahead of body" artifact tied to a
// separate predictive racket bug. Keep the two paths separate.
//
// Visibility is carried from the *nearer* frame rather than averaged —
// confidence shouldn't be smeared. Joint angles are recomputed from the
// interpolated landmarks rather than blended numerically (interpolating
// angles directly mishandles the wrap at 0/360).
export function getInterpolatedFrameAtTime(
  frames: PoseFrame[],
  timeSec: number,
): PoseFrame | null {
  if (!frames.length) return null
  const timeMs = timeSec * 1000
  if (timeMs <= frames[0].timestamp_ms) return frames[0]
  const last = frames[frames.length - 1]
  if (timeMs >= last.timestamp_ms) return last
  for (let i = 1; i < frames.length; i++) {
    if (frames[i].timestamp_ms > timeMs) {
      const prev = frames[i - 1]
      const next = frames[i]
      const span = next.timestamp_ms - prev.timestamp_ms
      // Identical timestamps would zero the denominator. The post-clamp
      // path above already handles times at-or-past either bound, so
      // span === 0 here is only possible with degenerate input — fall
      // back to the earlier sample.
      if (span <= 0) return prev
      const t = (timeMs - prev.timestamp_ms) / span
      const nearer = t < 0.5 ? prev : next
      const lerped: Landmark[] = []
      // Iterate the nearer frame's landmark list so we keep a stable id
      // ordering. If prev/next disagree on landmark count (shouldn't
      // happen with our 33-entry BlazePose output, but defensive),
      // missing entries fall through to the nearer frame's value.
      const a = prev.landmarks
      const b = next.landmarks
      for (let j = 0; j < nearer.landmarks.length; j++) {
        const la = a[j]
        const lb = b[j]
        if (!la || !lb) {
          lerped.push(nearer.landmarks[j])
          continue
        }
        lerped.push({
          id: la.id,
          name: la.name,
          x: la.x + (lb.x - la.x) * t,
          y: la.y + (lb.y - la.y) * t,
          z: la.z + (lb.z - la.z) * t,
          visibility: nearer.landmarks[j].visibility,
        })
      }
      return {
        frame_index: nearer.frame_index,
        timestamp_ms: timeMs,
        landmarks: lerped,
        joint_angles: computeJointAngles(lerped),
        // racket_head, when present, is sample-aligned with the nearer
        // frame (we don't synthesize racket positions between samples).
        ...(nearer.racket_head ? { racket_head: nearer.racket_head } : {}),
      }
    }
  }
  return last
}

interface VideoCanvasProps {
  src: string
  framesData: PoseFrame[]
  visible: Record<JointGroup, boolean>
  showSkeleton: boolean
  /** Wrist-trail overlay; off by default and no UI surfaces it anymore. */
  showTrail?: boolean
  /** Render the racket-head trail (schema v2 only; no-op on legacy clips). */
  showRacket?: boolean
  /** Render on-canvas angle labels next to elbow/knee joints. */
  showAngles?: boolean
  /**
   * When set with a contactFrameIndex, angle badges are color-coded against
   * the ideal range for this shot type at the 'contact' phase. Non-contact
   * frames render neutral-white (avoids a sea of red on prep/follow-through).
   */
  shotType?: string | null
  contactFrameIndex?: number | null
  /**
   * Hide the off-hand's elbow/wrist (and its swing-path trail) so the
   * overlay focuses on the racket arm. 'right' keeps right side, 'left'
   * keeps left, null shows both. Shoulders stay visible on both sides.
   */
  dominantHand?: 'left' | 'right' | null
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
  // Optional video window: when provided, the <video> element seeks to
  // windowStartMs on load and loops back to windowStartMs whenever
  // currentTime reaches windowEndMs. The pose overlay continues to
  // match its own video.currentTime against frame.timestamp_ms — both
  // source-relative — so windowing the video without re-zeroing frames
  // keeps overlay alignment correct.
  //
  // Use case: baseline-compare's "today" video. The clip the user
  // uploaded contains a swing in some sub-range (frames carry rally-
  // relative timestamps from pose extraction). Without windowing, the
  // video plays pre-swing footage at currentTime=0 while the pose
  // frames don't cover that range — overlay breaks. Windowing makes
  // the video play exactly the swing range, so pose and video align.
  windowStartMs?: number
  windowEndMs?: number
  // When true, video pauses at windowEnd instead of looping. Parent
  // controls when both videos restart together (used in baseline-
  // compare so the shorter clip waits at end frame for the longer
  // one to finish before both loop in unison).
  holdAtEnd?: boolean
  // Fires when the video reaches windowEnd (or natural end if no
  // window). Used by ComparisonLayout to detect "both clips done"
  // and coordinate a synchronized restart.
  onWindowEnd?: () => void
  // Bumping this number triggers an immediate seek to windowStart
  // and resumes playback (if syncPlaying). Lets the parent restart
  // both videos together after both have hit their respective ends.
  restartTrigger?: number
}

export default function VideoCanvas({
  src,
  framesData,
  visible,
  showSkeleton,
  showTrail = false,
  showRacket = false,
  showAngles = true,
  shotType = null,
  contactFrameIndex = null,
  dominantHand = null,
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
  windowStartMs,
  windowEndMs,
  holdAtEnd = false,
  onWindowEnd,
  restartTrigger,
}: VideoCanvasProps) {
  const hasWindow = typeof windowStartMs === 'number' && typeof windowEndMs === 'number'
  const windowStartSec = hasWindow ? (windowStartMs as number) / 1000 : 0
  const windowEndSec = hasWindow ? (windowEndMs as number) / 1000 : 0
  // Tracks whether onWindowEnd has fired for the current "session"
  // (i.e. since the last restartTrigger / startup). Prevents repeated
  // firing when the video stays paused at windowEnd.
  const hasFiredEndRef = useRef(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // Tracks the most recent syncedTime so the secondary can detect
  // primary-video loops (syncedTime drops sharply backward) and force
  // a hard re-sync rather than relying on the gradual playbackRate
  // correction loop, which can leave several seconds of visible drift.
  const prevSyncedTimeRef = useRef<number | null>(null)
  const rafRef = useRef<number>(0)
  const rvfcHandleRef = useRef<number>(0)
  // Once sized for the current videoWidth/Height, avoid re-scaling the ctx
  // every frame (ctx.scale is cumulative).
  const canvasSizedRef = useRef<{ w: number; h: number; dpr: number } | null>(
    null
  )
  const tracerRef = useRef(new SwingPathTracer())
  const lastFrameIndexRef = useRef(-1)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [videoError, setVideoError] = useState(false)

  // Interpolates between the two bracketing sampled frames (see
  // getFrameAtTime at module scope). Eliminates the ~1 decoded-frame lag that
  // nearest-neighbor lookup produced at 60fps playback against 30fps keypoints.
  const pickFrameAtTime = useCallback(
    (time: number): PoseFrame | null => getFrameAtTime(framesData, time),
    [framesData],
  )

  // Main render loop. `mediaTime` is the video-clock time in seconds reported
  // by requestVideoFrameCallback; when absent (rAF fallback) we fall back to
  // the video element's currentTime.
  const renderFrame = useCallback(
    (mediaTime?: number) => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas) return
      if (videoError) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // DPR-aware sizing (mirrors ProSwingViewer.SkeletonPlayer). The canvas
      // backing store is videoWidth * dpr; drawing code below uses logical
      // pixels (canvas.width / dpr, canvas.height / dpr).
      const dpr = window.devicePixelRatio || 1
      const vw = video.videoWidth
      const vh = video.videoHeight
      if (vw && vh) {
        const sized = canvasSizedRef.current
        if (!sized || sized.w !== vw || sized.h !== vh || sized.dpr !== dpr) {
          canvas.width = vw * dpr
          canvas.height = vh * dpr
          // Apply once per (size, dpr) — ctx.scale is cumulative across calls.
          ctx.setTransform(1, 0, 0, 1, 0, 0)
          ctx.scale(dpr, dpr)
          canvasSizedRef.current = { w: vw, h: vh, dpr }
        }
      } else if (!canvas.width || !canvas.height) {
        canvas.width = canvas.offsetWidth
        canvas.height = canvas.offsetHeight
      }

      const logicalW = canvas.width / dpr
      const logicalH = canvas.height / dpr
      ctx.clearRect(0, 0, logicalW, logicalH)

      const t = mediaTime ?? video.currentTime
      const frame = pickFrameAtTime(t)
      if (!frame) return

      const displayFrame = overlayColor ? normalizeLandmarks(frame) : frame

      // Update swing trail (advance buffer only when the underlying frame
      // changes so rapid re-renders don't stack duplicates).
      if (showTrail || showRacket) {
        if (frame.frame_index !== lastFrameIndexRef.current) {
          tracerRef.current.push(frame, logicalW, logicalH)
          lastFrameIndexRef.current = frame.frame_index
        }
        tracerRef.current.render(ctx, {
          showWristTrails: showTrail,
          showRacketTrail: showRacket,
          dominantHand,
        })
      }

      // Draw pose
      renderPose(ctx, displayFrame, logicalW, logicalH, {
        visible,
        showSkeleton,
        color: overlayColor,
        skeletonColor: overlaySkeletonColor,
        dominantHand,
      })

      // Angle labels go on top of the skeleton so the degree pills
      // aren't hidden under a bone. Skipped on ghost/overlay renders
      // (overlayColor set) to avoid two overlapping sets of numbers in
      // comparison mode. When the current frame matches the detected
      // contact frame, pass phase='contact' so the badges color-code
      // against the ideal range; otherwise phase stays null (neutral).
      if (showAngles && !overlayColor) {
        const isContactFrame =
          contactFrameIndex != null && frame.frame_index === contactFrameIndex
        renderAngleLabels(ctx, displayFrame, logicalW, logicalH, {
          dominantHand,
          shotType,
          phase: isContactFrame ? 'contact' : null,
        })
      }
    },
    [
      visible,
      showSkeleton,
      showTrail,
      showRacket,
      showAngles,
      shotType,
      contactFrameIndex,
      dominantHand,
      overlayColor,
      overlaySkeletonColor,
      pickFrameAtTime,
      videoError,
    ]
  )

  // Prefer requestVideoFrameCallback so we repaint in lockstep with decoded
  // video frames (frame-accurate overlay). Falls back to rAF on Safari/older
  // browsers where rVFC isn't implemented on HTMLVideoElement.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let cancelled = false
    const hasRvfc =
      typeof HTMLVideoElement !== 'undefined' &&
      'requestVideoFrameCallback' in HTMLVideoElement.prototype

    if (hasRvfc) {
      const tick: VideoFrameRequestCallback = (_now, meta) => {
        if (cancelled) return
        renderFrame(meta?.mediaTime)
        rvfcHandleRef.current = video.requestVideoFrameCallback(tick)
      }
      // Paint once immediately so paused/first-load state is correct even
      // before a decoded frame arrives.
      renderFrame()
      rvfcHandleRef.current = video.requestVideoFrameCallback(tick)

      return () => {
        cancelled = true
        if (
          rvfcHandleRef.current &&
          typeof video.cancelVideoFrameCallback === 'function'
        ) {
          video.cancelVideoFrameCallback(rvfcHandleRef.current)
        }
        rvfcHandleRef.current = 0
      }
    }

    // rAF fallback (Safari, etc.)
    const loop = () => {
      if (cancelled) return
      renderFrame()
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      cancelled = true
      cancelAnimationFrame(rafRef.current)
      rafRef.current = 0
    }
  }, [renderFrame, src])

  // Sync external time control
  useEffect(() => {
    if (syncedTime === undefined) return
    const video = videoRef.current
    if (!video) return

    // For secondary (pro) video: hold at start until the user's swing begins.
    // This prevents the pro video from stuttering during pre-swing dead time.
    if (isSecondary && proStartDelay > 0 && syncedTime < proStartDelay) {
      // If a window is set, "frame 0" of the slave is windowStartSec
      // in the source video, not literal 0.
      video.currentTime = hasWindow ? windowStartSec : 0
      video.pause()
      prevSyncedTimeRef.current = syncedTime
      return
    }

    // For secondary (pro) video, apply phase-based time mapping if available.
    // syncedTime is in seconds (user video time); timeMapping expects/returns ms.
    let targetTime = syncedTime
    if (isSecondary && timeMapping) {
      targetTime = timeMapping(syncedTime * 1000) / 1000
    }
    // When the secondary has a window, syncedTime is window-relative
    // (master emits it that way). Translate to source-relative time
    // for the seek. Subtract the start delay so contact alignment
    // still applies before the window offset.
    if (isSecondary && hasWindow) {
      targetTime = Math.max(0, syncedTime - proStartDelay) + windowStartSec
      // Clamp to window end so slave doesn't seek past the swing window.
      if (targetTime > windowEndSec) targetTime = windowEndSec
    }

    // Loop / replay detection: when syncedTime drops sharply backward,
    // the primary video has either looped or been scrubbed back to an
    // earlier point. Snap the secondary to the new mapped position
    // INSTEAD of letting the gradual playbackRate correction try to
    // catch up — that path leaves several seconds of visible drift on
    // every replay, which is the bug the user was seeing get worse on
    // each replay cycle.
    const prev = prevSyncedTimeRef.current
    const isLoopOrSeekBack = isSecondary && prev !== null && syncedTime < prev - 0.5
    prevSyncedTimeRef.current = syncedTime
    if (isLoopOrSeekBack) {
      video.currentTime = targetTime
      tracerRef.current.reset()
      lastFrameIndexRef.current = -1
      // If the primary is still playing, the secondary must resume
      // playback after the seek — when the secondary's `loop` attribute
      // was just turned off it can land in an `ended` state where
      // setting currentTime alone doesn't auto-resume.
      if (!video.paused || syncPlaying) {
        video.playbackRate = playbackRate
        void video.play().catch(() => {})
      }
      return
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
        // Same resume safeguard as the loop-detection branch above.
        if (video.ended && (!video.paused || syncPlaying)) {
          void video.play().catch(() => {})
        }
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
  }, [syncedTime, isSecondary, timeMapping, playbackRate, proStartDelay, syncPlaying])

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

  // Reset tracer, canvas sizing cache, and error state when src changes
  useEffect(() => {
    tracerRef.current.reset()
    lastFrameIndexRef.current = -1
    canvasSizedRef.current = null
    setVideoError(false)
  }, [src])

  const handleTimeUpdate = () => {
    const video = videoRef.current
    if (!video) return
    const t = video.currentTime ?? 0

    // Window/end coordination. Two modes:
    // - holdAtEnd=true: pause at windowEnd, fire onWindowEnd callback
    //   so the parent can coordinate a synchronized restart. Used by
    //   baseline-compare so the shorter clip waits at end frame for
    //   the longer one to finish.
    // - holdAtEnd=false (default): loop back to windowStart on hitting
    //   windowEnd. Browsers' built-in `loop` attribute would jump to
    //   0 — the wrong target when the swing window starts past 0.
    const reachedEnd = hasWindow
      ? t >= windowEndSec - 0.02
      : duration > 0 && t >= duration - 0.02
    if (reachedEnd) {
      if (holdAtEnd) {
        if (!hasFiredEndRef.current) {
          hasFiredEndRef.current = true
          // Pin to windowEnd so the displayed frame is the final
          // frame of the swing, not whatever happens after it in the
          // source clip.
          if (hasWindow) video.currentTime = windowEndSec - 0.001
          video.pause()
          setPlaying(false)
          onWindowEnd?.()
          setCurrentTime(hasWindow ? windowEndSec : duration)
        }
        return
      }
      // Loop mode: seek back to start, keep playing.
      if (hasWindow) {
        video.currentTime = windowStartSec
        tracerRef.current.reset()
        lastFrameIndexRef.current = -1
        if (isPrimary && !video.paused) onTimeUpdate?.(0)
        setCurrentTime(windowStartSec)
        return
      }
    }
    setCurrentTime(t)
    if (isPrimary) {
      onTimeUpdate?.(hasWindow ? Math.max(0, t - windowStartSec) : t)
    }
  }

  // restartTrigger effect: when bumped, seek to windowStart (or 0)
  // and resume playback if syncPlaying. Lets the parent coordinate
  // synchronized restarts in baseline-compare mode.
  useEffect(() => {
    if (restartTrigger === undefined || restartTrigger === 0) return
    const video = videoRef.current
    if (!video) return
    hasFiredEndRef.current = false
    video.currentTime = hasWindow ? windowStartSec : 0
    tracerRef.current.reset()
    lastFrameIndexRef.current = -1
    if (syncPlaying) {
      void video.play().catch(() => {})
    }
  }, [restartTrigger, hasWindow, windowStartSec, syncPlaying])

  // syncPlaying effect: in independent-pair mode (baseline-compare),
  // both videos coordinate via this prop. When parent toggles
  // bothPlaying, each VideoCanvas plays/pauses to match.
  useEffect(() => {
    if (syncPlaying === undefined) return
    const video = videoRef.current
    if (!video) return
    if (syncPlaying) {
      // If we're at windowEnd and the parent wants to play, the
      // restartTrigger effect should have already moved us back. If
      // we're still at end (e.g. parent only toggled syncPlaying
      // without bumping the trigger), seek to start before playing.
      if (hasWindow && video.currentTime >= windowEndSec - 0.05) {
        video.currentTime = windowStartSec
        hasFiredEndRef.current = false
      }
      void video.play().catch(() => {})
    } else if (!video.paused) {
      video.pause()
    }
  }, [syncPlaying, hasWindow, windowStartSec, windowEndSec])

  // videoEl.duration is untrustworthy — broken moov boxes and MediaRecorder
  // uploads report 0, NaN, or Infinity even when the file plays end-to-end.
  // Keypoints_json carries a reliable span since Railway walks every frame
  // during extraction. Displayed duration = max of the two trustworthy
  // signals, falling back to whichever is defined.
  const recomputeDuration = useCallback(() => {
    const video = videoRef.current
    const poseDurSec = framesData.length
      ? framesData[framesData.length - 1].timestamp_ms / 1000
      : 0
    const videoDur = video?.duration
    const videoDurOk =
      typeof videoDur === 'number' && Number.isFinite(videoDur) && videoDur > 0
    const trusted = videoDurOk ? Math.max(videoDur, poseDurSec) : poseDurSec
    setDuration(trusted)
    // Avoid notifying the parent with 0 on mount before either source
    // resolves — the original handleLoadedMetadata only fired once metadata
    // was present, and downstream consumers rely on positive values.
    if (trusted > 0) onDurationChange?.(trusted)
  }, [framesData, onDurationChange])

  const handleLoadedMetadata = () => {
    const video = videoRef.current
    // Chromium MediaRecorder bug: duration stays Infinity until the
    // playhead is seeked past the end, which forces the moov recompute.
    // durationchange fires with the real number, which re-runs recompute.
    if (video && !Number.isFinite(video.duration)) {
      video.currentTime = 1e101
      video.currentTime = 0
    }
    // Seek to windowStartMs on load so playback starts inside the
    // swing window rather than at the source video's t=0 (which would
    // be pre-swing footage with no pose data).
    if (video && hasWindow) {
      video.currentTime = windowStartSec
    }
    recomputeDuration()
  }

  const handleVideoDurationChange = () => {
    recomputeDuration()
  }

  // Pose data may arrive after loadedmetadata (props stream in). Recompute
  // duration when framesData lands so the scrubber max widens to the real
  // clip length even if videoEl.duration was wrong.
  useEffect(() => {
    recomputeDuration()
  }, [recomputeDuration])

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
          // The PRIMARY video auto-loops normally. holdAtEnd
          // explicitly disables this — when set, we want the video to
          // pause at windowEnd so the parent can coordinate a
          // synchronized restart with another video (baseline-compare
          // case). Without disabling the native loop, the browser
          // restarts to t=0 of the source video before our handleTime-
          // Update can intercept at windowEnd, dragging playback into
          // pre-swing footage where there's no pose data — the
          // "tracing is off" symptom.
          loop={!isSecondary && !holdAtEnd}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onDurationChange={handleVideoDurationChange}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => {
            // Backup signal for holdAtEnd. If timeupdate's reachedEnd
            // check missed the windowEnd boundary (timeupdate is
            // throttled to ~250ms and the browser's natural-end event
            // can fire between samples), this catches it so the parent
            // sees onWindowEnd reliably.
            if (holdAtEnd && !hasFiredEndRef.current) {
              hasFiredEndRef.current = true
              onWindowEnd?.()
            }
          }}
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
