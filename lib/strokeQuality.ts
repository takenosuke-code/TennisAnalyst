import type { PoseFrame, Landmark } from './supabase'
import { LANDMARK_INDICES } from './jointAngles'

/**
 * Per-stroke quality scoring + rejection gate.
 *
 * Given a list of detected strokes plus the full pose-frame stream, this
 * module slices each stroke's window, applies a hard-rejection gate
 * (visibility / camera pan / camera zoom / window length), and computes a
 * z-scored composite score for the surviving strokes. The score powers
 * "best vs worst stroke" ranking inside a session — it is intentionally
 * relative rather than absolute.
 *
 * Score formula:
 *   score =  0.55 * z(peak_wrist_speed)
 *          + 0.35 * z(-kineticChainTimingError)
 *          + 0.10 * z(-wristAngleVariance)
 *
 * The negation on the latter two flips the sign so that "lower error" and
 * "lower variance" raise the score, like peak speed does.
 *
 * Edge cases:
 *   - Single non-rejected stroke -> score = 0 (degenerate z-score).
 *   - All strokes rejected -> empty array (no rows emitted).
 *   - Mixed: rejected strokes get NaN score and are excluded from
 *     z-score statistics, but a row is still returned for them.
 */

export interface DetectedStroke {
  strokeId: string
  startFrame: number
  endFrame: number
  peakFrame: number
  fps: number
}

export type StrokeRejectReason =
  | 'low_visibility'
  | 'camera_pan'
  | 'camera_zoom'
  | 'too_short'
  | 'missing_data'

export interface StrokeQualityComponents {
  /** Raw peak wrist speed (linear), not z-scored. Units = coord/sec. */
  peakWristSpeed: number
  /** |actual_lag - session_median_lag|, where actual_lag = shoulder_peak_t - hip_peak_t (ms). */
  kineticChainTimingError: number
  /** Variance of dominant-arm elbow joint angle over the contact window. */
  wristAngleVariance: number
}

export interface StrokeQualityResult {
  strokeId: string
  /** z-scored across non-rejected strokes; higher = better; NaN if rejected. */
  score: number
  rejected: boolean
  rejectReason?: StrokeRejectReason
  components: StrokeQualityComponents
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const UPPER_BODY_LANDMARK_IDS = [
  LANDMARK_INDICES.LEFT_SHOULDER,
  LANDMARK_INDICES.RIGHT_SHOULDER,
  LANDMARK_INDICES.LEFT_ELBOW,
  LANDMARK_INDICES.RIGHT_ELBOW,
  LANDMARK_INDICES.LEFT_WRIST,
  LANDMARK_INDICES.RIGHT_WRIST,
  LANDMARK_INDICES.LEFT_HIP,
  LANDMARK_INDICES.RIGHT_HIP,
] as const

const MIN_STROKE_FRAMES = 10
const MIN_VISIBILITY_MEDIAN = 0.6
const PAN_DRIFT_FRACTION = 0.15
const ZOOM_VAR_OVER_MEAN_SQ = 0.05
const CONTACT_HALF_WINDOW = 5 // ±5 frames for wrist-angle variance

// Score component weights.
const W_SPEED = 0.55
const W_TIMING = 0.35
const W_WRIST_VAR = 0.1

// Pixel-vs-normalized detection threshold. Normalized landmarks live in
// [0,1]; anything appreciably above 1 means pixel coordinates.
const PIXEL_DETECT_THRESHOLD = 1.5
const PIXEL_FRAME_WIDTH = 1920
const NORMALIZED_FRAME_WIDTH = 1.0

// ---------------------------------------------------------------------------
// Geometry / window helpers
// ---------------------------------------------------------------------------

function getLandmarkById(
  frame: PoseFrame,
  id: number,
): Landmark | undefined {
  return frame.landmarks.find((l) => l.id === id)
}

/** Slice frames belonging to a stroke window, [start, end] inclusive. */
function sliceStroke(stroke: DetectedStroke, frames: PoseFrame[]): PoseFrame[] {
  const out: PoseFrame[] = []
  for (const f of frames) {
    if (f.frame_index >= stroke.startFrame && f.frame_index <= stroke.endFrame) {
      out.push(f)
    }
  }
  return out
}

/** Detect frame width from the max x value across the entire clip. */
function detectFrameWidth(frames: PoseFrame[]): number {
  let maxX = 0
  for (const f of frames) {
    for (const lm of f.landmarks) {
      if (lm.x > maxX) maxX = lm.x
    }
  }
  return maxX > PIXEL_DETECT_THRESHOLD ? PIXEL_FRAME_WIDTH : NORMALIZED_FRAME_WIDTH
}

function median(values: number[]): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2
}

function variance(values: number[]): number {
  if (values.length === 0) return 0
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  return values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length
}

// ---------------------------------------------------------------------------
// Rejection gate
// ---------------------------------------------------------------------------

interface BBox {
  cx: number
  cy: number
  area: number
}

/** Compute upper-body bbox for one frame; null if no upper-body landmarks present. */
function upperBodyBBox(frame: PoseFrame): BBox | null {
  let xMin = Infinity
  let yMin = Infinity
  let xMax = -Infinity
  let yMax = -Infinity
  let count = 0
  for (const id of UPPER_BODY_LANDMARK_IDS) {
    const lm = getLandmarkById(frame, id)
    if (!lm) continue
    if (lm.x < xMin) xMin = lm.x
    if (lm.x > xMax) xMax = lm.x
    if (lm.y < yMin) yMin = lm.y
    if (lm.y > yMax) yMax = lm.y
    count++
  }
  if (count === 0) return null
  return {
    cx: (xMin + xMax) / 2,
    cy: (yMin + yMax) / 2,
    area: Math.max(0, xMax - xMin) * Math.max(0, yMax - yMin),
  }
}

/** Mean visibility of upper-body landmarks present in the frame. */
function meanUpperBodyVisibility(frame: PoseFrame): number {
  let sum = 0
  let count = 0
  for (const id of UPPER_BODY_LANDMARK_IDS) {
    const lm = getLandmarkById(frame, id)
    if (!lm) continue
    sum += lm.visibility
    count++
  }
  return count === 0 ? 0 : sum / count
}

/**
 * Decide whether a stroke window should be rejected. Order matters only for
 * the reason returned in mixed cases; we evaluate too_short first since it
 * makes the other checks meaningless. Returns null if the stroke passes.
 */
function evaluateRejection(
  windowFrames: PoseFrame[],
  frameWidth: number,
): StrokeRejectReason | null {
  if (windowFrames.length < MIN_STROKE_FRAMES) {
    return 'too_short'
  }

  // Visibility check.
  const visPerFrame = windowFrames.map(meanUpperBodyVisibility)
  if (median(visPerFrame) < MIN_VISIBILITY_MEDIAN) {
    return 'low_visibility'
  }

  // Camera-pan and zoom checks need at least one usable bbox per frame.
  const bboxes: BBox[] = []
  for (const f of windowFrames) {
    const bb = upperBodyBBox(f)
    if (bb) bboxes.push(bb)
  }
  if (bboxes.length < 2) {
    // Not enough geometry to evaluate camera motion -> treat as missing data.
    return 'missing_data'
  }

  let cxMin = Infinity
  let cxMax = -Infinity
  for (const bb of bboxes) {
    if (bb.cx < cxMin) cxMin = bb.cx
    if (bb.cx > cxMax) cxMax = bb.cx
  }
  const drift = cxMax - cxMin
  if (drift / frameWidth > PAN_DRIFT_FRACTION) {
    return 'camera_pan'
  }

  const areas = bboxes.map((b) => b.area)
  const meanArea = areas.reduce((a, b) => a + b, 0) / areas.length
  if (meanArea > 0) {
    const areaVar = variance(areas)
    if (areaVar / (meanArea * meanArea) > ZOOM_VAR_OVER_MEAN_SQ) {
      return 'camera_zoom'
    }
  }

  return null
}

// ---------------------------------------------------------------------------
// Component computations
// ---------------------------------------------------------------------------

/**
 * Compute peak wrist speed (linear, smoothed) across a stroke window.
 * Speed is |Δ(x,y)/Δt| per frame, smoothed with a 3-frame moving average,
 * then we return the max. Returns 0 if the window is too short or the
 * dominant wrist is missing.
 */
function peakWristSpeed(
  windowFrames: PoseFrame[],
  wristLandmarkId: number,
): number {
  if (windowFrames.length < 2) return 0

  const rawSpeeds: number[] = []
  for (let i = 1; i < windowFrames.length; i++) {
    const a = getLandmarkById(windowFrames[i - 1], wristLandmarkId)
    const b = getLandmarkById(windowFrames[i], wristLandmarkId)
    const dt =
      (windowFrames[i].timestamp_ms - windowFrames[i - 1].timestamp_ms) / 1000
    if (!a || !b || dt <= 0) {
      rawSpeeds.push(0)
      continue
    }
    const dx = b.x - a.x
    const dy = b.y - a.y
    rawSpeeds.push(Math.hypot(dx, dy) / dt)
  }

  // 3-frame moving average to reduce single-frame jitter.
  let peak = 0
  for (let i = 0; i < rawSpeeds.length; i++) {
    const lo = Math.max(0, i - 1)
    const hi = Math.min(rawSpeeds.length - 1, i + 1)
    let sum = 0
    let n = 0
    for (let j = lo; j <= hi; j++) {
      sum += rawSpeeds[j]
      n++
    }
    const smoothed = sum / n
    if (smoothed > peak) peak = smoothed
  }
  return peak
}

/**
 * Compute the timestamp (ms) of peak |angular velocity| of the given joint
 * angle within the window. Returns null if not enough samples are present.
 */
function peakAngularVelocityMs(
  windowFrames: PoseFrame[],
  angleKey: 'hip_rotation' | 'trunk_rotation',
): number | null {
  let peakV = 0
  let peakMs: number | null = null
  for (let i = 1; i < windowFrames.length; i++) {
    const prev = windowFrames[i - 1].joint_angles[angleKey]
    const curr = windowFrames[i].joint_angles[angleKey]
    if (prev == null || curr == null) continue
    const dt =
      (windowFrames[i].timestamp_ms - windowFrames[i - 1].timestamp_ms) / 1000
    if (dt <= 0) continue
    const v = Math.abs(curr - prev) / dt
    if (v > peakV) {
      peakV = v
      peakMs = windowFrames[i].timestamp_ms
    }
  }
  return peakMs
}

/**
 * Variance of the dominant-arm elbow joint angle over a ±5-frame window
 * centered on `peakFrame`. Uses values present in `frames` (full clip)
 * since the contact window may extend beyond the stroke boundaries.
 */
function wristAngleVariance(
  frames: PoseFrame[],
  peakFrame: number,
  elbowKey: 'right_elbow' | 'left_elbow',
): number {
  const lo = peakFrame - CONTACT_HALF_WINDOW
  const hi = peakFrame + CONTACT_HALF_WINDOW
  const values: number[] = []
  for (const f of frames) {
    if (f.frame_index < lo || f.frame_index > hi) continue
    const v = f.joint_angles[elbowKey]
    if (typeof v === 'number') values.push(v)
  }
  return variance(values)
}

function countElbowSamplesInContact(
  frames: PoseFrame[],
  peakFrame: number,
  elbowKey: 'right_elbow' | 'left_elbow',
): number {
  const lo = peakFrame - CONTACT_HALF_WINDOW
  const hi = peakFrame + CONTACT_HALF_WINDOW
  let n = 0
  for (const f of frames) {
    if (f.frame_index < lo || f.frame_index > hi) continue
    if (typeof f.joint_angles[elbowKey] === 'number') n++
  }
  return n
}

// ---------------------------------------------------------------------------
// Z-scoring + assembly
// ---------------------------------------------------------------------------

function zScores(values: number[]): number[] {
  if (values.length === 0) return []
  if (values.length === 1) return [0]
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const v = variance(values)
  const sd = Math.sqrt(v)
  if (sd === 0) return values.map(() => 0)
  return values.map((x) => (x - mean) / sd)
}

interface InProgress {
  stroke: DetectedStroke
  rejected: boolean
  rejectReason?: StrokeRejectReason
  peakWristSpeed: number
  hipPeakMs: number | null
  shoulderPeakMs: number | null
  wristAngleVariance: number
}

/**
 * Score each detected stroke. The result preserves input order. Rejected
 * strokes receive NaN scores; if every stroke is rejected, an empty array
 * is returned.
 */
export function scoreStrokes(
  strokes: DetectedStroke[],
  frames: PoseFrame[],
): StrokeQualityResult[] {
  if (strokes.length === 0) return []

  // Default dominant-hand convention: right. The function signature does
  // not carry a profile, so we cannot consult `dominant_hand` here.
  const wristLandmarkId = LANDMARK_INDICES.RIGHT_WRIST
  const elbowKey: 'right_elbow' | 'left_elbow' = 'right_elbow'

  const frameWidth = detectFrameWidth(frames)

  // Pass 1: reject + compute everything except timing error.
  const inProgress: InProgress[] = strokes.map((stroke) => {
    const window = sliceStroke(stroke, frames)
    const rejectReason = evaluateRejection(window, frameWidth)
    if (rejectReason) {
      return {
        stroke,
        rejected: true,
        rejectReason,
        peakWristSpeed: 0,
        hipPeakMs: null,
        shoulderPeakMs: null,
        wristAngleVariance: 0,
      }
    }
    const speed = peakWristSpeed(window, wristLandmarkId)
    const hipPeakMs = peakAngularVelocityMs(window, 'hip_rotation')
    const shoulderPeakMs = peakAngularVelocityMs(window, 'trunk_rotation')
    // Reject when chain-timing peaks aren't detectable: the spec says a
    // missing required angle should fire the rejection gate. Without this
    // a missing-data stroke would silently score 0 timing error and get
    // an unearned bonus relative to peers with real (noisy) data.
    if (hipPeakMs == null || shoulderPeakMs == null) {
      return {
        stroke,
        rejected: true,
        rejectReason: 'missing_data',
        peakWristSpeed: 0,
        hipPeakMs: null,
        shoulderPeakMs: null,
        wristAngleVariance: 0,
      }
    }
    // Same logic for the dominant-elbow contact window: an empty value
    // set produces variance 0, which would unfairly help missing-data
    // strokes. Require at least one elbow sample in the ±5 window.
    const elbowSamples = countElbowSamplesInContact(
      frames,
      stroke.peakFrame,
      elbowKey,
    )
    if (elbowSamples === 0) {
      return {
        stroke,
        rejected: true,
        rejectReason: 'missing_data',
        peakWristSpeed: 0,
        hipPeakMs: null,
        shoulderPeakMs: null,
        wristAngleVariance: 0,
      }
    }
    const wristVar = wristAngleVariance(frames, stroke.peakFrame, elbowKey)
    return {
      stroke,
      rejected: false,
      peakWristSpeed: speed,
      hipPeakMs,
      shoulderPeakMs,
      wristAngleVariance: wristVar,
    }
  })

  const accepted = inProgress.filter((p) => !p.rejected)
  if (accepted.length === 0) return []

  // Pass 2: session_median_lag from non-rejected strokes. Pass 1 already
  // rejected anyone with a missing hip/shoulder peak, so all accepted
  // strokes have both timestamps.
  const lags: number[] = []
  for (const p of accepted) {
    // Non-null asserted: missing-peak strokes were rejected in pass 1.
    lags.push((p.shoulderPeakMs as number) - (p.hipPeakMs as number))
  }
  const sessionMedianLag = median(lags)

  // Pass 3: compute timing error per non-rejected stroke.
  const timingErrors = new Map<string, number>()
  for (const p of accepted) {
    const actualLag = (p.shoulderPeakMs as number) - (p.hipPeakMs as number)
    timingErrors.set(p.stroke.strokeId, Math.abs(actualLag - sessionMedianLag))
  }

  // Z-score components (across accepted only).
  const speeds = accepted.map((p) => p.peakWristSpeed)
  const errors = accepted.map((p) => timingErrors.get(p.stroke.strokeId) ?? 0)
  const wristVars = accepted.map((p) => p.wristAngleVariance)

  const zSpeed = zScores(speeds)
  // For "lower is better" components, negate before z-scoring so that a
  // higher z-score still means "better" in the combined formula.
  const zNegError = zScores(errors.map((e) => -e))
  const zNegVar = zScores(wristVars.map((v) => -v))

  const acceptedScores = new Map<string, number>()
  for (let i = 0; i < accepted.length; i++) {
    const s =
      W_SPEED * zSpeed[i] + W_TIMING * zNegError[i] + W_WRIST_VAR * zNegVar[i]
    acceptedScores.set(accepted[i].stroke.strokeId, s)
  }

  // Assemble output preserving input order.
  return inProgress.map((p) => {
    if (p.rejected) {
      return {
        strokeId: p.stroke.strokeId,
        score: NaN,
        rejected: true,
        rejectReason: p.rejectReason,
        components: {
          peakWristSpeed: 0,
          kineticChainTimingError: 0,
          wristAngleVariance: 0,
        },
      }
    }
    const score = acceptedScores.get(p.stroke.strokeId) ?? 0
    return {
      strokeId: p.stroke.strokeId,
      score,
      rejected: false,
      components: {
        peakWristSpeed: p.peakWristSpeed,
        kineticChainTimingError: timingErrors.get(p.stroke.strokeId) ?? 0,
        wristAngleVariance: p.wristAngleVariance,
      },
    }
  })
}
