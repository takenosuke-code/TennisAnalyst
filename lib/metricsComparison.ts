import type { JointAngles, PoseFrame } from './supabase'

/**
 * Metrics comparison utilities for comparing user and pro swing angles.
 *
 * Color coding thresholds:
 * - Green:  angle difference < 15 degrees (good match)
 * - Yellow: angle difference 15-30 degrees (needs attention)
 * - Red:    angle difference > 30 degrees (significant deviation)
 */

export type DiffSeverity = 'green' | 'yellow' | 'red'

export interface AngleDiff {
  angleKey: keyof JointAngles
  userValue: number
  proValue: number
  difference: number
  severity: DiffSeverity
}

export interface ComparisonResult {
  /** All angle comparisons with color coding */
  diffs: AngleDiff[]
  /** The body part (angle key) with the largest deviation */
  worstBodyPart: keyof JointAngles | null
  /** The largest angle difference in degrees */
  maxDifference: number
  /** Average angle difference across all compared joints */
  averageDifference: number
}

/**
 * Classify an angle difference into a severity level.
 *
 * - < 15 degrees: green (good match)
 * - 15-30 degrees: yellow (needs attention)
 * - > 30 degrees: red (significant deviation)
 */
export function classifyDifference(angleDiffDegrees: number): DiffSeverity {
  const abs = Math.abs(angleDiffDegrees)
  if (abs < 15) return 'green'
  if (abs <= 30) return 'yellow'
  return 'red'
}

/**
 * Compare two sets of joint angles, returning per-joint color-coded diffs.
 *
 * Only compares angles that are present in BOTH sets. Missing angles in
 * either set are skipped gracefully.
 */
export function compareAngles(
  userAngles: JointAngles,
  proAngles: JointAngles
): ComparisonResult {
  const diffs: AngleDiff[] = []

  const allKeys: (keyof JointAngles)[] = [
    'right_elbow',
    'left_elbow',
    'right_shoulder',
    'left_shoulder',
    'right_knee',
    'left_knee',
    'hip_rotation',
    'trunk_rotation',
  ]

  for (const key of allKeys) {
    const userVal = userAngles[key]
    const proVal = proAngles[key]
    if (userVal == null || proVal == null) continue

    const difference = Math.abs(userVal - proVal)
    diffs.push({
      angleKey: key,
      userValue: userVal,
      proValue: proVal,
      difference,
      severity: classifyDifference(difference),
    })
  }

  let worstBodyPart: keyof JointAngles | null = null
  let maxDifference = 0
  let totalDifference = 0

  for (const diff of diffs) {
    totalDifference += diff.difference
    if (diff.difference > maxDifference) {
      maxDifference = diff.difference
      worstBodyPart = diff.angleKey
    }
  }

  return {
    diffs,
    worstBodyPart,
    maxDifference,
    averageDifference: diffs.length > 0 ? totalDifference / diffs.length : 0,
  }
}

/**
 * Compare angles at a specific frame index (or the closest available)
 * between user and pro frame arrays.
 *
 * Handles empty arrays gracefully by returning an empty result.
 */
export function compareFrameAngles(
  userFrames: PoseFrame[],
  proFrames: PoseFrame[],
  frameIndex: number
): ComparisonResult {
  const emptyResult: ComparisonResult = {
    diffs: [],
    worstBodyPart: null,
    maxDifference: 0,
    averageDifference: 0,
  }

  if (userFrames.length === 0 || proFrames.length === 0) {
    return emptyResult
  }

  const userIdx = Math.min(frameIndex, userFrames.length - 1)
  const proIdx = Math.min(frameIndex, proFrames.length - 1)

  return compareAngles(
    userFrames[userIdx].joint_angles,
    proFrames[proIdx].joint_angles
  )
}

/**
 * Identify the body part needing the most improvement across a range
 * of frames (e.g., the entire swing). Returns the angle key that has
 * the highest average difference.
 */
export function findMostImprovedBodyPart(
  userFrames: PoseFrame[],
  proFrames: PoseFrame[]
): { bodyPart: keyof JointAngles | null; avgDifference: number } {
  if (userFrames.length === 0 || proFrames.length === 0) {
    return { bodyPart: null, avgDifference: 0 }
  }

  const accumulator: Record<string, { total: number; count: number }> = {}
  const numComparisons = Math.min(userFrames.length, proFrames.length)

  for (let i = 0; i < numComparisons; i++) {
    const result = compareAngles(
      userFrames[i].joint_angles,
      proFrames[i].joint_angles
    )
    for (const diff of result.diffs) {
      if (!accumulator[diff.angleKey]) {
        accumulator[diff.angleKey] = { total: 0, count: 0 }
      }
      accumulator[diff.angleKey].total += diff.difference
      accumulator[diff.angleKey].count++
    }
  }

  let worstKey: keyof JointAngles | null = null
  let worstAvg = 0

  for (const [key, { total, count }] of Object.entries(accumulator)) {
    const avg = total / count
    if (avg > worstAvg) {
      worstAvg = avg
      worstKey = key as keyof JointAngles
    }
  }

  return { bodyPart: worstKey, avgDifference: worstAvg }
}
