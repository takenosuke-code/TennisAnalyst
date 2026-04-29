/*
 * Phase 1.5 — Tier inference from observed swing metrics.
 *
 * Replaces the "pick your skill level" form question that the consumer
 * brief flagged as a bounce point ("I'm 3.5, where do I sit between
 * Beginner and Competitive?"). Instead we estimate tier from what the
 * pose pipeline actually saw on the user's first 1–3 uploads, then
 * surface an override chip on the analyze view that lets them confirm
 * or correct without going back to /onboarding.
 *
 * What we measure (with the data we already have — no new inference
 * pass, no new model):
 *
 *   1. Trunk rotation range. Beginners under-rotate (tend to "arm" the
 *      ball with little hip/shoulder turn); refined players show a
 *      40-80° trunk swing across the swing window. Computed from the
 *      `trunk_rotation` joint angle's max-minus-min over the clip.
 *
 *   2. Wrist trajectory smoothness. Scored as the mean magnitude of
 *      the second-order difference of the dominant wrist's normalized
 *      position (jerk proxy). A scattered swing has jerky wrists;
 *      refined kinetic chains have smooth ones. Threshold-tuned, not
 *      load-bearing.
 *
 *   3. Detection confidence. Average `visibility` across the swing's
 *      shoulder/elbow/wrist landmarks. Low average reads as either a
 *      bad camera angle or a swing the model couldn't track cleanly;
 *      we down-weight the inference's confidence in that case rather
 *      than guessing tier from noisy data.
 *
 * We deliberately don't compute racket-head speed: it requires the
 * racket detector, which has its own confidence problems and isn't a
 * tier signal anyway (a beginner can hit hard; an advanced player can
 * hit slow). We deliberately don't compute pronation: a 2D pose lacks
 * the depth info to pin pronation reliably.
 *
 * The output is paired with a `confidence` score in [0,1]. Below a
 * cutoff we return null and the caller defaults to "intermediate"
 * rather than committing to a guess.
 */

import type { PoseFrame, Landmark } from '@/lib/supabase'
import type { DominantHand, SkillTier } from '@/lib/profile'

export interface TierInferenceResult {
  tier: SkillTier
  confidence: number
  reasons: string[]
}

const MIN_FRAMES = 8

// Empirical thresholds. Tuned against the IMG_1098 reference clip
// (intermediate forehand) plus a synthetic beginner / advanced split
// from the existing test fixtures. Treat as starting points, not
// gospel — Phase 0.5 telemetry will let us tune from real data.
const TRUNK_RANGE_BEGINNER_MAX = 25 // < 25° = under-rotated, beginner-shaped
const TRUNK_RANGE_COMPETITIVE_MIN = 50 // > 50° = full kinetic chain rotation

// Mean per-frame wrist jerk (in normalized image units). Beginners >
// 0.04, intermediate ~0.025, refined < 0.018. Heavily-smoothed by the
// per-frame averaging the pipeline already does upstream.
const WRIST_JERK_BEGINNER_MIN = 0.04
const WRIST_JERK_COMPETITIVE_MAX = 0.018

// Low-confidence floor: if the average detection visibility on the
// load-bearing arm joints is below this, our inference is data-bound
// and we down-weight the result so the caller defaults rather than
// commits.
const VIS_FLOOR = 0.5

const LANDMARK = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
} as const

function get(landmarks: Landmark[] | null | undefined, idx: number): Landmark | null {
  if (!landmarks || idx >= landmarks.length) return null
  const lm = landmarks[idx]
  if (!lm || typeof lm.x !== 'number' || typeof lm.y !== 'number') return null
  return lm
}

function trunkRotationRange(frames: PoseFrame[]): number | null {
  let lo = Infinity
  let hi = -Infinity
  let counted = 0
  for (const f of frames) {
    const v = f?.joint_angles?.trunk_rotation
    if (typeof v !== 'number' || !Number.isFinite(v)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
    counted++
  }
  if (counted < MIN_FRAMES / 2) return null
  return hi - lo
}

function meanWristJerk(frames: PoseFrame[], hand: DominantHand): number | null {
  const wristIdx = hand === 'right' ? LANDMARK.RIGHT_WRIST : LANDMARK.LEFT_WRIST
  const positions: { x: number; y: number }[] = []
  for (const f of frames) {
    const lm = get(f.landmarks, wristIdx)
    if (!lm) continue
    positions.push({ x: lm.x, y: lm.y })
  }
  if (positions.length < 4) return null

  // Second-order finite differences as a jerk proxy.
  let sum = 0
  let count = 0
  for (let i = 2; i < positions.length; i++) {
    const ax = positions[i].x - 2 * positions[i - 1].x + positions[i - 2].x
    const ay = positions[i].y - 2 * positions[i - 1].y + positions[i - 2].y
    sum += Math.hypot(ax, ay)
    count++
  }
  if (count === 0) return null
  return sum / count
}

function meanArmVisibility(frames: PoseFrame[], hand: DominantHand): number {
  const wrist = hand === 'right' ? LANDMARK.RIGHT_WRIST : LANDMARK.LEFT_WRIST
  const elbow = hand === 'right' ? LANDMARK.RIGHT_ELBOW : LANDMARK.LEFT_ELBOW
  const shoulder = hand === 'right' ? LANDMARK.RIGHT_SHOULDER : LANDMARK.LEFT_SHOULDER

  let total = 0
  let count = 0
  for (const f of frames) {
    for (const idx of [wrist, elbow, shoulder]) {
      const lm = get(f.landmarks, idx)
      if (!lm) continue
      const vis = typeof lm.visibility === 'number' ? lm.visibility : 0
      total += vis
      count++
    }
  }
  return count === 0 ? 0 : total / count
}

/**
 * Infer skill tier from a single clip's frames. Returns null when the
 * data is too sparse / low-confidence to commit to a guess; the caller
 * should default to "intermediate" in that case.
 */
export function inferTierFromFrames(
  frames: PoseFrame[],
  dominantHand: DominantHand = 'right',
): TierInferenceResult | null {
  if (!frames || frames.length < MIN_FRAMES) return null

  const trunkRange = trunkRotationRange(frames)
  const wristJerk = meanWristJerk(frames, dominantHand)
  const armVis = meanArmVisibility(frames, dominantHand)

  if (armVis < VIS_FLOOR) return null
  if (trunkRange === null && wristJerk === null) return null

  // Score each signal: -1 = beginner-shaped, 0 = intermediate, +1 =
  // competitive-shaped. Then average and bucket.
  const reasons: string[] = []
  let total = 0
  let weight = 0

  if (trunkRange !== null) {
    let s = 0
    if (trunkRange < TRUNK_RANGE_BEGINNER_MAX) {
      s = -1
      reasons.push(`under-rotated trunk (${Math.round(trunkRange)}°)`)
    } else if (trunkRange > TRUNK_RANGE_COMPETITIVE_MIN) {
      s = 1
      reasons.push(`full trunk rotation (${Math.round(trunkRange)}°)`)
    } else {
      reasons.push(`solid trunk rotation (${Math.round(trunkRange)}°)`)
    }
    total += s
    weight += 1
  }

  if (wristJerk !== null) {
    let s = 0
    if (wristJerk > WRIST_JERK_BEGINNER_MIN) {
      s = -1
      reasons.push('jerky wrist trajectory')
    } else if (wristJerk < WRIST_JERK_COMPETITIVE_MAX) {
      s = 1
      reasons.push('smooth wrist path')
    } else {
      reasons.push('controlled wrist path')
    }
    total += s
    weight += 1
  }

  const score = weight > 0 ? total / weight : 0
  let tier: SkillTier
  if (score <= -0.5) tier = 'beginner'
  else if (score >= 0.5) tier = 'competitive'
  else tier = 'intermediate'

  // Confidence: average detection visibility times signal-agreement
  // (1.0 if both signals agree on the bucket, lower otherwise).
  const agreement = weight === 0 ? 0 : Math.abs(total) / weight
  const confidence = Math.min(1, armVis * (0.5 + 0.5 * agreement))

  return { tier, confidence, reasons }
}
