/**
 * Per-candidate swing-shape verifier.
 *
 * The wrist-speed peak-pick in lib/swingDetect.ts is a high-recall
 * candidate generator: anything with a sharp wrist acceleration
 * passes. Walking toward the camera, prep bouncing, and hand fidgets
 * also produce wrist-speed peaks. This verifier is the second stage —
 * given a candidate peak frame, it checks whether the surrounding
 * joint trajectories actually look like a tennis swing.
 *
 * Three checks, all derived from biomechanics literature (Landlinger
 * 2010, Elliott review PMC2577481). A real swing has all three;
 * walking and fidgeting fail at least one:
 *
 *   1. HIP ROTATION EXCURSION ≥ 8° across the candidate window. A
 *      tennis swing rotates the hips 30–60°. Walking forward moves
 *      hips translationally with minimal rotation (<5°).
 *   2. TRUNK ROTATION EXCURSION ≥ 10° (when the trunk angle is
 *      available). Same intuition for the shoulder line — a swing
 *      rotates shoulders ≥30°, walking does not.
 *   3. PROXIMAL-TO-DISTAL TIMING. Hip angular-velocity peak must
 *      precede the wrist-speed peak by 25–400ms (loose amateur
 *      bound; elite is 75–100ms). The kinetic chain is the
 *      definitional signature of a swing — walking has hip and
 *      wrist oscillating in phase, no proximal lead.
 *
 * Conservative on missing data: when hip_rotation isn't available,
 * the verifier rejects rather than passes (no signal = can't verify).
 * This is by design — the gate is opt-in, called only at the UI
 * surface where joint_angles are reliably populated.
 */

import type { PoseFrame } from './supabase'

const DEFAULT_HIP_EXCURSION_MIN_DEG = 8
const DEFAULT_TRUNK_EXCURSION_MIN_DEG = 10
const DEFAULT_KINETIC_CHAIN_MIN_DELAY_MS = 25
const DEFAULT_KINETIC_CHAIN_MAX_DELAY_MS = 400
const DEFAULT_PRE_WINDOW_SEC = 0.7
const DEFAULT_POST_WINDOW_SEC = 0.5
const MIN_FRAMES_FOR_VERIFICATION = 6

export interface SwingShapeOptions {
  hipExcursionMinDeg?: number
  trunkExcursionMinDeg?: number
  kineticChainMinDelayMs?: number
  kineticChainMaxDelayMs?: number
  preWindowSec?: number
  postWindowSec?: number
}

export type SwingShapeFailReason =
  | 'no_data'
  | 'no_hip_rotation'
  | 'shallow_hip_excursion'
  | 'shallow_trunk_excursion'
  | 'no_kinetic_chain'

export interface SwingShapeResult {
  passed: boolean
  reason?: SwingShapeFailReason
  metrics: {
    hipExcursion: number | null
    trunkExcursion: number | null
    /** Milliseconds the hip angular-velocity peak preceded the wrist
     *  peak (positive when the chain ordering is correct). */
    hipLeadMs: number | null
  }
}

function shortAngleDelta(a: number, b: number): number {
  let d = b - a
  while (d > 180) d -= 360
  while (d <= -180) d += 360
  return d
}

/**
 * Excursion of an angle channel: max - min after unwrapping ±180°
 * jumps via short-arc deltas. Same algorithm as
 * coachingObservations.angleExcursion so the rule layer and the
 * verifier read the same number for a given window.
 */
function unwrapExcursion(series: number[]): number {
  if (series.length === 0) return 0
  const unwrapped: number[] = [series[0]]
  for (let i = 1; i < series.length; i++) {
    unwrapped.push(unwrapped[i - 1] + shortAngleDelta(unwrapped[i - 1], series[i]))
  }
  let lo = Infinity
  let hi = -Infinity
  for (const v of unwrapped) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return hi - lo
}

function angleSeriesIn(
  frames: PoseFrame[],
  lo: number,
  hi: number,
  key: 'hip_rotation' | 'trunk_rotation',
): number[] {
  const out: number[] = []
  const start = Math.max(0, lo)
  const end = Math.min(frames.length - 1, hi)
  for (let i = start; i <= end; i++) {
    const v = frames[i].joint_angles?.[key]
    if (typeof v === 'number' && Number.isFinite(v)) out.push(v)
  }
  return out
}

/**
 * Frame index of the largest |angular velocity| of `key` within
 * [lo, hi]. Central-difference estimator with short-angle unwrap.
 *
 * When multiple frames are within an epsilon of the maximum (a
 * velocity plateau, common on linear hip ramps or uniform-rotation
 * windows), returns the median plateau index — using the "first one
 * that exceeded peakV" picks an arbitrary boundary frame whose
 * absolute position depends on floating-point noise. The median is
 * stable and biomechanically sensible (mid-rotation is when the hip
 * is "most active").
 *
 * Returns null when no usable angular-velocity samples exist.
 */
function angularVelocityPeakFrame(
  frames: PoseFrame[],
  lo: number,
  hi: number,
  key: 'hip_rotation' | 'trunk_rotation',
): number | null {
  const start = Math.max(1, lo)
  const end = Math.min(frames.length - 2, hi)
  // First pass: find the maximum velocity.
  let peakV = -1
  for (let i = start; i <= end; i++) {
    const prev = frames[i - 1].joint_angles?.[key]
    const next = frames[i + 1].joint_angles?.[key]
    if (
      typeof prev !== 'number' ||
      typeof next !== 'number' ||
      !Number.isFinite(prev) ||
      !Number.isFinite(next)
    ) {
      continue
    }
    const dt = (frames[i + 1].timestamp_ms - frames[i - 1].timestamp_ms) / 1000
    if (dt <= 0) continue
    const v = Math.abs(shortAngleDelta(prev, next)) / dt
    if (v > peakV) peakV = v
  }
  if (peakV <= 0) return null

  // Second pass: collect frames within epsilon of peakV and return
  // their centroid. Epsilon = 1% of peakV — floats inside a true
  // plateau differ by far less than that, while real distinct peaks
  // differ by orders of magnitude more.
  const eps = peakV * 0.01
  const plateau: number[] = []
  for (let i = start; i <= end; i++) {
    const prev = frames[i - 1].joint_angles?.[key]
    const next = frames[i + 1].joint_angles?.[key]
    if (typeof prev !== 'number' || typeof next !== 'number') continue
    const dt = (frames[i + 1].timestamp_ms - frames[i - 1].timestamp_ms) / 1000
    if (dt <= 0) continue
    const v = Math.abs(shortAngleDelta(prev, next)) / dt
    if (v >= peakV - eps) plateau.push(i)
  }
  if (plateau.length === 0) return null
  return plateau[Math.floor(plateau.length / 2)]
}

export function verifySwingShape(
  frames: PoseFrame[],
  peakFrameIdx: number,
  fps: number,
  opts: SwingShapeOptions = {},
): SwingShapeResult {
  const hipMin = opts.hipExcursionMinDeg ?? DEFAULT_HIP_EXCURSION_MIN_DEG
  const trunkMin = opts.trunkExcursionMinDeg ?? DEFAULT_TRUNK_EXCURSION_MIN_DEG
  const minDelay = opts.kineticChainMinDelayMs ?? DEFAULT_KINETIC_CHAIN_MIN_DELAY_MS
  const maxDelay = opts.kineticChainMaxDelayMs ?? DEFAULT_KINETIC_CHAIN_MAX_DELAY_MS
  const preSec = opts.preWindowSec ?? DEFAULT_PRE_WINDOW_SEC
  const postSec = opts.postWindowSec ?? DEFAULT_POST_WINDOW_SEC

  const lo = Math.max(0, peakFrameIdx - Math.round(preSec * fps))
  const hi = Math.min(frames.length - 1, peakFrameIdx + Math.round(postSec * fps))

  const empty: SwingShapeResult['metrics'] = {
    hipExcursion: null,
    trunkExcursion: null,
    hipLeadMs: null,
  }

  if (hi - lo < MIN_FRAMES_FOR_VERIFICATION) {
    return { passed: false, reason: 'no_data', metrics: empty }
  }

  const hipSeries = angleSeriesIn(frames, lo, hi, 'hip_rotation')
  const trunkSeries = angleSeriesIn(frames, lo, hi, 'trunk_rotation')

  if (hipSeries.length < MIN_FRAMES_FOR_VERIFICATION) {
    return { passed: false, reason: 'no_hip_rotation', metrics: empty }
  }

  const hipExcursion = unwrapExcursion(hipSeries)
  const trunkExcursion =
    trunkSeries.length >= MIN_FRAMES_FOR_VERIFICATION
      ? unwrapExcursion(trunkSeries)
      : null

  if (hipExcursion < hipMin) {
    return {
      passed: false,
      reason: 'shallow_hip_excursion',
      metrics: { hipExcursion, trunkExcursion, hipLeadMs: null },
    }
  }
  if (trunkExcursion !== null && trunkExcursion < trunkMin) {
    return {
      passed: false,
      reason: 'shallow_trunk_excursion',
      metrics: { hipExcursion, trunkExcursion, hipLeadMs: null },
    }
  }

  const hipPeakIdx = angularVelocityPeakFrame(frames, lo, hi, 'hip_rotation')
  if (hipPeakIdx === null) {
    return {
      passed: false,
      reason: 'no_kinetic_chain',
      metrics: { hipExcursion, trunkExcursion, hipLeadMs: null },
    }
  }

  const wristMs = frames[peakFrameIdx]?.timestamp_ms ?? 0
  const hipMs = frames[hipPeakIdx]?.timestamp_ms ?? 0
  const hipLeadMs = wristMs - hipMs

  if (hipLeadMs < minDelay || hipLeadMs > maxDelay) {
    return {
      passed: false,
      reason: 'no_kinetic_chain',
      metrics: { hipExcursion, trunkExcursion, hipLeadMs },
    }
  }

  return { passed: true, metrics: { hipExcursion, trunkExcursion, hipLeadMs } }
}
