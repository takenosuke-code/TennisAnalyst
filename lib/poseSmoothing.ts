import { computeJointAngles } from './jointAngles'
import type { PoseFrame, Landmark } from './supabase'

/**
 * Minimum average landmark visibility for a frame to be considered valid.
 * Frames below this threshold are likely bad detections from MediaPipe's
 * warm-up period and are discarded.
 */
const DEFAULT_VISIBILITY_THRESHOLD = 0.4

/**
 * Minimum bounding-box area (as fraction of frame area [0,1]x[0,1]) for a
 * detection to be considered valid.  A tiny bbox means MediaPipe only found
 * a sliver of the person -usually an artifact.
 */
const DEFAULT_MIN_BBOX_AREA = 0.01

/**
 * Number of warm-up frames to discard from the start of a detection run.
 * MediaPipe VIDEO mode needs ~2-5 frames for its tracker to converge.
 * Using 5 (upper end) because the cost of discarding a few extra good
 * frames is far lower than letting jittery warm-up data through.
 * Only applied when total frame count exceeds WARMUP_MIN_TOTAL_FRAMES.
 */
const DEFAULT_WARMUP_DISCARD = 5
const WARMUP_MIN_TOTAL_FRAMES = 10

/**
 * Number of initial frames to average for EMA seed initialization.
 * Instead of seeding the EMA with a single (potentially noisy) first
 * frame, we average the first LOOKAHEAD_WINDOW good frames to get a
 * stable starting point.
 */
const DEFAULT_LOOKAHEAD_WINDOW = 3

/**
 * EMA alpha -higher values make the output more responsive to new data,
 * lower values produce smoother (more lagged) output.
 * 0.7 is a good balance for 30fps sports video.
 */
const DEFAULT_ALPHA = 0.7

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Average visibility across all landmarks in a frame. */
export function averageVisibility(landmarks: Landmark[]): number {
  if (landmarks.length === 0) return 0
  let sum = 0
  for (const lm of landmarks) {
    sum += lm.visibility
  }
  return sum / landmarks.length
}

/** Bounding-box area of landmarks in normalized [0,1] coordinate space. */
export function landmarkBboxArea(landmarks: Landmark[]): number {
  if (landmarks.length === 0) return 0
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x
    if (lm.y < minY) minY = lm.y
    if (lm.x > maxX) maxX = lm.x
    if (lm.y > maxY) maxY = lm.y
  }
  return (maxX - minX) * (maxY - minY)
}

// ---------------------------------------------------------------------------
// Confidence gating -filters out bad frames
// ---------------------------------------------------------------------------

export interface GateOptions {
  visibilityThreshold?: number
  minBboxArea?: number
}

/**
 * Returns true if the frame passes confidence gating checks.
 */
export function isFrameConfident(
  landmarks: Landmark[],
  opts: GateOptions = {}
): boolean {
  const { visibilityThreshold = DEFAULT_VISIBILITY_THRESHOLD, minBboxArea = DEFAULT_MIN_BBOX_AREA } = opts
  if (averageVisibility(landmarks) < visibilityThreshold) return false
  if (landmarkBboxArea(landmarks) < minBboxArea) return false
  return true
}

// ---------------------------------------------------------------------------
// EMA smoothing
// ---------------------------------------------------------------------------

/**
 * Compute the average position of each landmark across the first
 * `windowSize` frames. Returns a Map from landmark id to averaged
 * {x, y, z}. This is used to seed the EMA with a stable initial
 * state instead of relying on a single (potentially noisy) frame.
 */
function computeLookaheadSeed(
  frames: PoseFrame[],
  windowSize: number
): Map<number, { x: number; y: number; z: number }> {
  const seed = new Map<number, { x: number; y: number; z: number }>()
  const counts = new Map<number, number>()
  const n = Math.min(windowSize, frames.length)

  for (let i = 0; i < n; i++) {
    for (const lm of frames[i].landmarks) {
      const prev = seed.get(lm.id)
      if (!prev) {
        seed.set(lm.id, { x: lm.x, y: lm.y, z: lm.z })
        counts.set(lm.id, 1)
      } else {
        prev.x += lm.x
        prev.y += lm.y
        prev.z += lm.z
        counts.set(lm.id, (counts.get(lm.id) ?? 0) + 1)
      }
    }
  }

  // Divide by count to get averages
  for (const [id, pos] of seed) {
    const c = counts.get(id) ?? 1
    pos.x /= c
    pos.y /= c
    pos.z /= c
  }

  return seed
}

/**
 * Apply exponential moving average smoothing to an ordered array of
 * PoseFrames. Each landmark's x, y, z coordinates are smoothed
 * independently. Visibility is kept as-is (raw from MediaPipe) so
 * downstream renderers can still gate on it.
 *
 * Joint angles are recomputed from the smoothed landmarks so they stay
 * consistent.
 *
 * The EMA state is seeded by averaging the first `lookaheadWindow`
 * frames (look-ahead initialization). This prevents the common issue
 * where a slightly-off first frame biases the smoothed output for
 * several subsequent frames before the EMA corrects.
 *
 * @param frames          -ordered PoseFrame array (must already be confidence-gated)
 * @param alpha           -EMA weight for new values. 1.0 = no smoothing. 0.0 = ignore new data.
 * @param lookaheadWindow -number of initial frames to average for EMA seed. Default 3.
 */
function emaSmooth(
  frames: PoseFrame[],
  alpha: number,
  lookaheadWindow: number = DEFAULT_LOOKAHEAD_WINDOW
): PoseFrame[] {
  if (frames.length === 0) return []

  const result: PoseFrame[] = []

  // Seed the EMA state with the average of the first N frames
  // instead of blindly trusting the first frame.
  const prev = computeLookaheadSeed(frames, lookaheadWindow)

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i]
    const smoothedLandmarks: Landmark[] = frame.landmarks.map((lm) => {
      const p = prev.get(lm.id)
      if (!p) {
        // Landmark not seen in seed window - use raw values
        prev.set(lm.id, { x: lm.x, y: lm.y, z: lm.z })
        return { ...lm }
      }
      const sx = alpha * lm.x + (1 - alpha) * p.x
      const sy = alpha * lm.y + (1 - alpha) * p.y
      const sz = alpha * lm.z + (1 - alpha) * p.z
      prev.set(lm.id, { x: sx, y: sy, z: sz })
      return { ...lm, x: sx, y: sy, z: sz }
    })

    result.push({
      ...frame,
      landmarks: smoothedLandmarks,
      joint_angles: computeJointAngles(smoothedLandmarks),
    })
  }

  return result
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface SmoothOptions {
  /** EMA alpha (0,1]. Default 0.7. */
  alpha?: number
  /** Min average visibility to keep a frame. Default 0.4. */
  visibilityThreshold?: number
  /** Min bbox area fraction to keep a frame. Default 0.01. */
  minBboxArea?: number
  /** Number of initial warm-up frames to discard. Default 5. */
  warmupDiscard?: number
  /** Number of initial frames to average for EMA seed. Default 3. */
  lookaheadWindow?: number
}

/**
 * Post-process raw PoseFrames extracted from MediaPipe:
 *
 * 1. Discard early warm-up frames (if total > 10)
 * 2. Filter out low-confidence frames
 * 3. Apply EMA smoothing to landmark coordinates
 * 4. Recompute joint angles from smoothed landmarks
 *
 * Returns a new array -the input is not mutated.
 */
export function smoothFrames(
  frames: PoseFrame[],
  opts: SmoothOptions = {}
): PoseFrame[] {
  const {
    alpha = DEFAULT_ALPHA,
    visibilityThreshold = DEFAULT_VISIBILITY_THRESHOLD,
    minBboxArea = DEFAULT_MIN_BBOX_AREA,
    warmupDiscard = DEFAULT_WARMUP_DISCARD,
    lookaheadWindow = DEFAULT_LOOKAHEAD_WINDOW,
  } = opts

  if (frames.length === 0) return []

  // 1. Discard warm-up frames
  let filtered = frames
  if (frames.length > WARMUP_MIN_TOTAL_FRAMES && warmupDiscard > 0) {
    filtered = frames.slice(warmupDiscard)
  }

  // 2. Confidence gating
  filtered = filtered.filter((f) =>
    isFrameConfident(f.landmarks, { visibilityThreshold, minBboxArea })
  )

  if (filtered.length === 0) return []

  // 3. EMA smoothing with look-ahead seed initialization
  return emaSmooth(filtered, alpha, lookaheadWindow)
}
