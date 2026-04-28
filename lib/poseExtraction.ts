'use client'

import { createPoseDetector, type PoseDetector } from '@/lib/browserPose'
import { computeJointAngles } from '@/lib/jointAngles'
import {
  isFrameConfident,
  smoothFrames,
  filterImplausibleArmJoints,
} from '@/lib/poseSmoothing'
import type { PoseFrame, Landmark } from '@/lib/supabase'

export type ExtractResult = {
  frames: PoseFrame[]
  fps: number
  // The object URL created for the hidden <video>. Caller is responsible for
  // revoking it (or handing it off for playback). Null on abort.
  objectUrl: string | null
}

export type ExtractOptions = {
  fps?: number
  // Called with 0..100 over the course of extraction. Does NOT include upload
  // or model-load time — caller remaps to their own progress bar if needed.
  onProgress?: (pct: number) => void
  abortSignal?: AbortSignal
}

class AbortError extends Error {
  constructor() {
    super('aborted')
    this.name = 'AbortError'
  }
}

// Race a promise against a timeout. The original failure mode (no
// timeout, no user-visible error) was the thing that made iOS Safari
// bugs look like "stuck at 25%" rather than "extraction failed because
// X" — fail fast with a specific message so the caller can surface it.
function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  message: string,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(message)), ms)
    promise.then(
      (v) => {
        clearTimeout(t)
        resolve(v)
      },
      (e) => {
        clearTimeout(t)
        reject(e)
      },
    )
  })
}

/**
 * Extract pose keypoints from every ~1/fps second of a video File.
 *
 * Phase 5: now uses the same browser-side ONNX RTMPose detector as live
 * (via @/lib/browserPose). Output is the same 33-entry BlazePose
 * Landmark[] downstream consumers expect, so smoothing / joint angles /
 * swing detection paths are unchanged.
 */
export async function extractPoseFromVideo(
  source: File,
  opts: ExtractOptions = {}
): Promise<ExtractResult> {
  const fps = opts.fps ?? 30
  const signal = opts.abortSignal
  const check = () => {
    if (signal?.aborted) throw new AbortError()
  }

  const detector: PoseDetector = await createPoseDetector()
  check()

  // Build our own hidden video element. Callers used to manage this ref
  // themselves; now it's fully encapsulated.
  const videoEl = document.createElement('video')
  videoEl.muted = true
  videoEl.playsInline = true
  videoEl.preload = 'auto'

  const objectUrl = URL.createObjectURL(source)
  videoEl.src = objectUrl

  try {
    // Wait for metadata (dimensions/duration). iOS Safari can stall
    // here silently on unsupported codecs — 15s cap + concrete error
    // beats a forever-25% bar.
    await withTimeout(
      new Promise<void>((resolve) => {
        if (videoEl.readyState >= 1) {
          resolve()
          return
        }
        videoEl.onloadedmetadata = () => resolve()
      }),
      15_000,
      'Video metadata never loaded (codec unsupported or file corrupt?). Try re-recording with "Most Compatible" format in iPhone Settings → Camera → Formats.',
    )
    check()

    // iOS Safari needs a play()/pause() to prime the video decoder
    // BEFORE it will advance readyState past 2 or fire canplay. The
    // previous iteration of this fix waited for canplay first and
    // THEN called play() — order was backwards; iOS would sit on
    // readyState 2 indefinitely and the canplay timeout tripped with
    // "Video failed to become playable in time." Correct order: prime
    // first, then check readyState.
    //
    // Muted + playsInline are set above, which satisfies every modern
    // browser's autoplay policy. A raw play() on a just-loaded element
    // can still be rejected on some iOS versions; that's caught and
    // ignored — the seek loop below has a per-frame timeout so even a
    // decoder that never truly primes won't hang the whole extraction.
    try {
      await videoEl.play()
      videoEl.pause()
    } catch {
      // Priming failed. The seek loop's 3s-per-frame timeout still
      // bounds the damage.
    }
    check()

    // readyState 2 = HAVE_CURRENT_DATA, which is enough for drawImage
    // to produce a valid canvas. We used to wait for readyState 3
    // (HAVE_FUTURE_DATA) via canplay, but that's overkill — if
    // individual frames fail to draw the existing per-frame error
    // handling skips them.
    if (videoEl.readyState < 2) {
      await withTimeout(
        new Promise<void>((resolve) => {
          const onReady = () => {
            videoEl.removeEventListener('loadeddata', onReady)
            resolve()
          }
          videoEl.addEventListener('loadeddata', onReady)
        }),
        15_000,
        'Video failed to become readable in time (often HEIF/HEVC on iPhone — try Settings → Camera → Formats → "Most Compatible").',
      )
    }
    check()

    const duration = videoEl.duration

    // Adaptive sampling FPS for long clips. Browser MediaPipe detection
    // runs at ~50-100ms per frame on a phone, so a 2-minute clip at the
    // default 30fps would be 3,600 detections = 3-6 minutes of compute.
    // For analysis purposes (skeleton + joint angles + swing detection)
    // we don't need 30fps — peak swing motion only spans ~10-15 frames
    // even at 15fps, and the visualization stays smooth because the
    // canvas redraws on every video frame regardless of sample density.
    //
    // Tiers tuned so total extraction time stays roughly bounded under
    // ~90 seconds on a typical mobile, regardless of clip length:
    //   <= 30s  → 30 fps (current default; short clips get full fidelity)
    //   30-60s  → 20 fps
    //   60-120s → 15 fps
    //   > 120s  → 10 fps
    const effectiveFps =
      !Number.isFinite(duration) || duration <= 30
        ? fps
        : duration <= 60
          ? Math.min(fps, 20)
          : duration <= 120
            ? Math.min(fps, 15)
            : Math.min(fps, 10)
    if (effectiveFps !== fps) {
      console.info(
        `[poseExtraction] long clip (${duration.toFixed(1)}s) — sampling at ${effectiveFps}fps instead of ${fps}fps to keep extraction under ~90s`,
      )
    }
    const frameInterval = 1 / effectiveFps
    const frames: PoseFrame[] = []
    let frameIndex = 0

    const canvas = document.createElement('canvas')
    canvas.width = videoEl.videoWidth || 640
    canvas.height = videoEl.videoHeight || 360
    const ctx = canvas.getContext('2d')!

    // Seek video to a timestamp and wait for onseeked, with 3s timeout per frame
    const seekToFrame = (time: number): Promise<boolean> =>
      new Promise((resolve) => {
        let settled = false
        const settle = (ok: boolean) => {
          if (settled) return
          settled = true
          videoEl.onseeked = null
          resolve(ok)
        }
        const timeout = setTimeout(() => settle(false), 3000)
        videoEl.onseeked = () => {
          clearTimeout(timeout)
          settle(true)
        }
        videoEl.currentTime = time
      })

    while (frameIndex * frameInterval <= duration) {
      check()

      const currentTime = frameIndex * frameInterval
      const seeked = await seekToFrame(currentTime)
      if (!seeked) {
        frameIndex++
        continue
      }

      // Wait one animation frame so the decoded frame is composited and ready
      // for drawImage.
      await new Promise<void>((r) => requestAnimationFrame(() => r()))
      check()

      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height)

      try {
        // Phase 5: browser ONNX path. The detector internally runs YOLO
        // for person bbox, crops to the player, and runs RTMPose-m on
        // the crop — so the MediaPipe-era two-pass logic (full frame
        // then re-detect on a tight crop for small subjects) is no
        // longer needed; the bbox-crop is already inside the detector.
        const landmarks = await detector.detect(canvas)
        if (!landmarks || landmarks.length === 0) {
          frameIndex++
          continue
        }

        // Skip low-confidence detections (warm-up artifacts, off-camera, etc).
        if (isFrameConfident(landmarks)) {
          const joint_angles = computeJointAngles(landmarks)
          frames.push({
            frame_index: frameIndex,
            timestamp_ms: currentTime * 1000,
            landmarks,
            joint_angles,
          })
        }
      } catch {
        // Skip frames where detection fails
      }

      frameIndex++
      if (opts.onProgress && duration > 0) {
        opts.onProgress(Math.min(100, Math.round((currentTime / duration) * 100)))
      }
    }

    // Smooth first (filtfilt zero-phase), then apply bone-length
    // plausibility so the filter works on the smoothed trajectories.
    // Implausibly-placed elbow/wrist landmarks have their visibility
    // zeroed so the render-side cutoff hides them — partial skeletons
    // through contact instead of wrong-but-visible ones.
    const smoothed = smoothFrames(frames)
    const cleaned = filterImplausibleArmJoints(smoothed)
    // Return the EFFECTIVE fps the loop actually used so downstream
    // smoothing / swing-detection windows compute against real timing.
    // Returning the requested fps would lie about the data density.
    return { frames: cleaned, fps: effectiveFps, objectUrl }
  } catch (err) {
    // On abort or error, release the object URL — nobody's going to use it
    URL.revokeObjectURL(objectUrl)
    if (err instanceof AbortError) {
      return { frames: [], fps, objectUrl: null }
    }
    throw err
  } finally {
    detector.dispose()
    videoEl.src = ''
    videoEl.removeAttribute('src')
    videoEl.load()
  }
}
