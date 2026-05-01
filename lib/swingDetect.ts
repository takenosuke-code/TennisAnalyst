import type { PoseFrame, Landmark } from './supabase'

// MediaPipe Pose landmark indices for the wrists. Inlined rather than
// imported from `./jointAngles` to keep the dependency graph one-way:
// jointAngles.ts delegates to detectStrokes() in this module; importing
// back into here would create a circular initialization order.
const LM_LEFT_WRIST = 15
const LM_RIGHT_WRIST = 16

// ---------------------------------------------------------------------------
// Wrist-velocity peak-pick stroke detector.
//
// Replaces the older energy-hysteresis + 20pre/15post-frame padding scheme
// (still living next door in detectSwings()). The hysteresis approach
// fragmented strokes whose backswing was much quieter than the
// acceleration phase — the loading window from ~800–1200ms before contact
// (Reid/Elliott) was below threshold and got clipped, so coaching
// observations and baselines saw only the contact-onward slice.
//
// New approach:
//   1. Per-frame dominant-wrist linear speed in normalized frame coords
//      (Δposition / Δt). The wrist landmark is whichever side has the
//      higher mean visibility, with the user's `dominant_hand` as an
//      explicit override (default: right wrist when nothing else gives a
//      hint). MediaPipe's normalized coords are [0,1] so a speed of 0.5
//      ≈ "half the frame per second".
//   2. Smooth with a 5-frame centered moving average.
//   3. Find local maxima above an adaptive threshold of `mean + 1.0·std`.
//      Enforce a small min-distance (3 frames) at the peak-finding stage
//      so two genuine peaks 300ms apart can both surface.
//   4. Refractory: greedy-pick by speed; whenever two accepted peaks fall
//      within 500ms of each other, the lower is dropped.
//   5. Pad each peak biomechanically — 1000ms pre (backswing window per
//      tennis-stroke literature) and 500ms post (follow-through tail).
//      Convert ms → frames using the actual fps derived from timestamps,
//      not a hardcoded 30. Clamp to [0, totalFrames-1].
// ---------------------------------------------------------------------------

export interface DetectedStroke {
  /** `stroke_${index}` where index is zero-based in peakFrame ascending order. */
  strokeId: string
  /** Inclusive start frame index after pre-padding (clamped to ≥ 0). */
  startFrame: number
  /** Inclusive end frame index after post-padding (clamped to ≤ totalFrames-1). */
  endFrame: number
  /** Frame index of the wrist-speed peak that anchored this stroke. */
  peakFrame: number
  /** Effective fps used for ms ↔ frame conversion. */
  fps: number
}

export interface DetectStrokesOptions {
  /**
   * Override the input frame rate. When omitted we derive it from the
   * median timestamp delta — robust against a few duplicate or missing
   * frames in long captures.
   */
  fps?: number
  /**
   * `'left' | 'right'` from the user profile, when known. When provided
   * we use the corresponding wrist landmark unconditionally; otherwise we
   * pick whichever wrist had the higher mean visibility, defaulting to
   * right when both are equal or unsampled.
   */
  dominantHand?: 'left' | 'right' | null
  /**
   * Pre-pad window in milliseconds. Default 1000ms (Reid/Elliott — tennis
   * backswing initiates 800–1200ms before contact).
   */
  prePadMs?: number
  /**
   * Post-pad window in milliseconds. Default 500ms (typical
   * follow-through / deceleration phase).
   */
  postPadMs?: number
  /**
   * Minimum gap between accepted peaks. Default 500ms.
   */
  refractoryMs?: number
  /**
   * Adaptive threshold = median + thresholdK · (P90 − median) over the
   * smoothed wrist-speed signal. Default 1.0 (i.e. P90 itself). Lower
   * values surface gentler swings (slices, blocks) at the cost of more
   * false positives during fidgeting.
   */
  thresholdK?: number
  /**
   * Smoothing window in frames for the centered moving average over
   * wrist speed. Default 5 — same as the pose-extraction smoother.
   */
  smoothingFrames?: number
  /**
   * Minimum frame distance between two raw peaks before refractory
   * dedup. Default 3 frames. Refractory does the heavy lifting; this
   * exists so two real peaks 300ms apart (≈9 frames at 30fps) can both
   * make it past peak-finding before refractory chooses one.
   */
  minPeakDistanceFrames?: number
}

const VISIBILITY_FLOOR = 0.5

// Default Stroke-options resolved with applyDefaults below. Exporting
// these keeps the magic numbers documented and lets tests exercise them
// without re-typing.
export const STROKE_DEFAULTS = {
  prePadMs: 1000,
  postPadMs: 500,
  refractoryMs: 500,
  thresholdK: 1.0,
  smoothingFrames: 5,
  minPeakDistanceFrames: 3,
} as const

interface ResolvedOptions {
  fps: number
  dominantHand: 'left' | 'right' | null
  prePadMs: number
  postPadMs: number
  refractoryMs: number
  thresholdK: number
  smoothingFrames: number
  minPeakDistanceFrames: number
}

function applyDefaults(
  frames: PoseFrame[],
  opts: DetectStrokesOptions,
): ResolvedOptions {
  const fps = opts.fps ?? deriveFps(frames)
  return {
    fps,
    dominantHand: opts.dominantHand ?? null,
    prePadMs: opts.prePadMs ?? STROKE_DEFAULTS.prePadMs,
    postPadMs: opts.postPadMs ?? STROKE_DEFAULTS.postPadMs,
    refractoryMs: opts.refractoryMs ?? STROKE_DEFAULTS.refractoryMs,
    thresholdK: opts.thresholdK ?? STROKE_DEFAULTS.thresholdK,
    smoothingFrames: opts.smoothingFrames ?? STROKE_DEFAULTS.smoothingFrames,
    minPeakDistanceFrames:
      opts.minPeakDistanceFrames ?? STROKE_DEFAULTS.minPeakDistanceFrames,
  }
}

/**
 * Derive a sane fps from frame timestamps. Uses the median Δt to
 * tolerate a handful of dropped or duplicated frames; falls back to 30
 * when timestamps are absent or all equal (synthetic fixtures).
 */
export function deriveFps(frames: PoseFrame[]): number {
  if (frames.length < 2) return 30
  const diffs: number[] = []
  for (let i = 1; i < frames.length; i++) {
    const dt = frames[i].timestamp_ms - frames[i - 1].timestamp_ms
    if (Number.isFinite(dt) && dt > 0) diffs.push(dt)
  }
  if (diffs.length === 0) return 30
  diffs.sort((a, b) => a - b)
  const medianDt = diffs[Math.floor(diffs.length / 2)]
  if (!medianDt || medianDt <= 0) return 30
  const fps = 1000 / medianDt
  // Round to the nearest integer when extremely close; keeps frame-count
  // arithmetic identical for canonical 30/60fps captures.
  if (Math.abs(fps - Math.round(fps)) < 0.05) return Math.round(fps)
  return fps
}

function getLandmark(landmarks: Landmark[], id: number): Landmark | undefined {
  return landmarks.find((l) => l.id === id)
}

/**
 * Decide which wrist to track. Profile dominant-hand wins when present;
 * otherwise pick by mean visibility across the clip; otherwise default
 * to right.
 */
function pickDominantWrist(
  frames: PoseFrame[],
  override: 'left' | 'right' | null,
): 'left' | 'right' {
  if (override === 'left' || override === 'right') return override

  let leftSum = 0
  let leftCount = 0
  let rightSum = 0
  let rightCount = 0
  for (const f of frames) {
    const lw = getLandmark(f.landmarks, LM_LEFT_WRIST)
    const rw = getLandmark(f.landmarks, LM_RIGHT_WRIST)
    if (lw) {
      leftSum += lw.visibility
      leftCount++
    }
    if (rw) {
      rightSum += rw.visibility
      rightCount++
    }
  }
  const leftMean = leftCount > 0 ? leftSum / leftCount : 0
  const rightMean = rightCount > 0 ? rightSum / rightCount : 0
  // Right wins ties — matches typical right-handed default.
  return leftMean > rightMean ? 'left' : 'right'
}

/**
 * Per-frame dominant-wrist linear speed in normalized-coords-per-second.
 * Frames where the wrist landmark is missing or below the visibility
 * floor produce a 0 speed (rather than NaN) so the smoother can still
 * roll through the gap without poisoning neighbours.
 */
function computeWristSpeeds(
  frames: PoseFrame[],
  side: 'left' | 'right',
): number[] {
  const id = side === 'left' ? LM_LEFT_WRIST : LM_RIGHT_WRIST
  const speeds: number[] = new Array(frames.length).fill(0)
  if (frames.length < 2) return speeds

  let prev: { x: number; y: number; t: number } | null = null
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]
    const lm = getLandmark(f.landmarks, id)
    if (!lm || lm.visibility < VISIBILITY_FLOOR) {
      // Don't anchor a baseline on an occluded sample — leaves the speed
      // at 0 here and lets the next visible sample's neighbour drive
      // velocity.
      continue
    }
    if (prev) {
      const dx = lm.x - prev.x
      const dy = lm.y - prev.y
      const dt = (f.timestamp_ms - prev.t) / 1000
      if (dt > 0) {
        speeds[i] = Math.sqrt(dx * dx + dy * dy) / dt
      }
    }
    prev = { x: lm.x, y: lm.y, t: f.timestamp_ms }
  }
  return speeds
}

/**
 * Centered moving-average smoother with a half-window of
 * floor(smoothingFrames / 2). Uses whatever neighbours exist at the
 * edges (not zero-padded) so the smoothed signal is the same length as
 * the input.
 */
function smooth(values: number[], windowFrames: number): number[] {
  const half = Math.floor(windowFrames / 2)
  const out: number[] = new Array(values.length).fill(0)
  for (let i = 0; i < values.length; i++) {
    const lo = Math.max(0, i - half)
    const hi = Math.min(values.length - 1, i + half)
    let sum = 0
    let count = 0
    for (let j = lo; j <= hi; j++) {
      sum += values[j]
      count++
    }
    out[i] = count > 0 ? sum / count : 0
  }
  return out
}

/**
 * Find local maxima ≥ threshold respecting a minimum distance between
 * peaks. Returns frame indices in the order they were found (sorted by
 * frame index). When multiple equal-valued peaks sit within the
 * min-distance window, the earliest survives — refractory cleans up
 * later anyway.
 */
function findLocalMaxima(
  values: number[],
  threshold: number,
  minDistanceFrames: number,
): number[] {
  const peaks: number[] = []
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (v < threshold) continue
    // Local max: strictly greater than at least one neighbour and not
    // less than either neighbour. Handles plateaus by picking the first
    // index of the plateau.
    const left = i > 0 ? values[i - 1] : -Infinity
    const right = i < values.length - 1 ? values[i + 1] : -Infinity
    const isLocalMax =
      v >= left && v >= right && (v > left || v > right || values.length === 1)
    if (!isLocalMax) continue

    // Min-distance dedup: if the previous accepted peak is within
    // minDistanceFrames, keep whichever is taller. Done greedy
    // forward-pass style.
    if (peaks.length > 0) {
      const last = peaks[peaks.length - 1]
      if (i - last < minDistanceFrames) {
        if (values[i] > values[last]) peaks[peaks.length - 1] = i
        continue
      }
    }
    peaks.push(i)
  }
  return peaks
}

/**
 * Drop any peak that has a stronger peak within `refractoryFrames` of
 * it. Peaks whose strict neighbours (within window) are all ≤ them
 * survive. Ordered by frame index ascending in output.
 */
function applyRefractory(
  peaks: number[],
  values: number[],
  refractoryFrames: number,
): number[] {
  if (peaks.length === 0) return peaks
  const survivors: number[] = []
  // Sort peaks by speed descending; greedily accept the highest, then
  // exclude any other peak within ±refractory of an accepted one.
  const byStrength = [...peaks].sort((a, b) => values[b] - values[a])
  const accepted: number[] = []
  for (const p of byStrength) {
    let blocked = false
    for (const a of accepted) {
      if (Math.abs(p - a) < refractoryFrames) {
        blocked = true
        break
      }
    }
    if (!blocked) accepted.push(p)
  }
  // Re-sort survivors by frame index for the caller.
  accepted.sort((a, b) => a - b)
  for (const p of accepted) survivors.push(p)
  return survivors
}

/**
 * Detect strokes via dominant-wrist speed peak-pick + biomechanical
 * padding. See the file-level comment for design notes.
 *
 * Returns DetectedStroke[] sorted by peakFrame ascending. Empty array
 * when no peaks survive the threshold and refractory.
 */
export function detectStrokes(
  frames: PoseFrame[],
  opts: DetectStrokesOptions = {},
): DetectedStroke[] {
  const resolved = applyDefaults(frames, opts)
  if (frames.length < 2) return []

  const side = pickDominantWrist(frames, resolved.dominantHand)
  const rawSpeeds = computeWristSpeeds(frames, side)
  const smoothed = smooth(rawSpeeds, resolved.smoothingFrames)

  // Adaptive threshold over smoothed speed. Mean+k·std is the textbook
  // formulation but a single explosive swing in a long rally can
  // inflate the std past the soft swings — the same threshold-
  // inflation failure mode the old hysteresis detector hit with
  // global max. We use a robust percentile-based version: median +
  // thresholdK · (P90 − median). The (P90 − median) "spread" is
  // resistant to outliers because it ignores the top 10% of samples.
  // Anything ≤ 0 spread (perfectly flat clip) bumps the threshold to
  // Infinity so the no-peak path is taken.
  const { median, p90 } = percentileStats(smoothed)
  const spread = Math.max(0, p90 - median)
  const threshold = spread > 0 ? median + resolved.thresholdK * spread : Infinity

  const refractoryFrames = Math.max(
    1,
    Math.round((resolved.refractoryMs / 1000) * resolved.fps),
  )
  const minPeakDistance = Math.min(
    resolved.minPeakDistanceFrames,
    refractoryFrames,
  )

  const rawPeaks = findLocalMaxima(smoothed, threshold, minPeakDistance)
  const survivors = applyRefractory(rawPeaks, smoothed, refractoryFrames)

  const prePadFrames = Math.ceil((resolved.prePadMs / 1000) * resolved.fps)
  const postPadFrames = Math.ceil((resolved.postPadMs / 1000) * resolved.fps)

  const lastIdx = frames.length - 1
  return survivors.map((peakFrame, i) => ({
    strokeId: `stroke_${i}`,
    startFrame: Math.max(0, peakFrame - prePadFrames),
    endFrame: Math.min(lastIdx, peakFrame + postPadFrames),
    peakFrame,
    fps: resolved.fps,
  }))
}

function percentileStats(values: number[]): { median: number; p90: number } {
  if (values.length === 0) return { median: 0, p90: 0 }
  const sorted = [...values].sort((a, b) => a - b)
  const median = sorted[Math.floor(sorted.length / 2)]
  const p90 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))]
  return { median, p90 }
}
