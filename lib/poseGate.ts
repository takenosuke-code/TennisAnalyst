/**
 * Clip-level confidence gate. Decides whether the pose stream is good
 * enough for the rule layer to grade. When it isn't, the route bypasses
 * the LLM call and surfaces a reshoot banner instead of fabricated
 * observations.
 *
 * Three checks, all framed as "is the underlying signal trustworthy":
 *   1. Median per-frame visibility on RTMPose-filled landmarks. If the
 *      pose model gave up on the body, observations downstream are
 *      noise no matter what we do.
 *   2. Body-presence fraction. The required-joint rules (hip/trunk
 *      excursion, elbow at contact, knee load) need shoulders, hips,
 *      and wrists visible on most frames. If the player is half-out-
 *      of-frame, even a high-confidence pose model can't help.
 *   3. Frame-to-frame hip rotation jump fraction. RTMPose-s on amateur
 *      side-on clips occasionally flips left/right hip keypoints during
 *      rotation — when that happens, hip_rotation jumps >60° between
 *      consecutive frames. A human hip rotates at most ~900°/s during
 *      a forehand (Landlinger 2010); at 30fps that's a 30°/frame
 *      ceiling, so >60° in one frame is a structural failure no
 *      smoother can recover. Run on RAW frames: smoothing distributes
 *      a real flip's delta across multiple frames at the smoother's
 *      cutoff, dropping each individual delta below threshold and
 *      hiding the failure from this check.
 *
 * Thresholds chosen against the IMG_1245 calibration clip (mean vis
 * 0.48, 20.5% jump-fraction). All three checks fail there. The clean
 * Alcaraz fixture (mean vis ~0.78) passes all three.
 */

import type { PoseFrame } from './supabase'
import { RTMPOSE_FILLED_LANDMARK_IDS } from './jointAngles'

export type GateReason =
  | 'low_visibility'
  | 'excessive_jitter'
  | 'body_not_visible'
  | 'too_few_frames'

export interface GateResult {
  passed: boolean
  reason?: GateReason
  metrics: {
    medianVisibility: number
    jumpFraction: number
    bodyVisibleFraction: number
    frameCount: number
  }
}

const RTMPOSE_FILLED_SET = new Set<number>(RTMPOSE_FILLED_LANDMARK_IDS)

const MIN_FRAMES = 12
const MIN_MEDIAN_VISIBILITY = 0.5
const MAX_JUMP_FRACTION = 0.25
const MIN_BODY_VISIBLE_FRACTION = 0.6
const HIP_JUMP_THRESHOLD_DEG = 60
const REQUIRED_JOINT_VIS_FLOOR = 0.4

// MediaPipe / BlazePose-33 ids for the joints required by the rule
// layer. Must stay in sync with extractObservations' rule reads.
const ID_LEFT_SHOULDER = 11
const ID_RIGHT_SHOULDER = 12
const ID_LEFT_WRIST = 15
const ID_RIGHT_WRIST = 16
const ID_LEFT_HIP = 23
const ID_RIGHT_HIP = 24

function shortAngleDelta(a: number, b: number): number {
  let d = b - a
  while (d > 180) d -= 360
  while (d <= -180) d += 360
  return d
}

function frameVisibility(frame: PoseFrame): number {
  // Match coachingObservations.frameVisibility: empty landmarks =>
  // synthetic / test fixture that's bypassing landmark data and
  // trusting the joint_angles directly. Treat as fully visible so
  // the gate doesn't block legacy / hand-crafted inputs.
  if (!frame.landmarks || frame.landmarks.length === 0) return 1.0
  let sum = 0
  let n = 0
  for (const l of frame.landmarks) {
    if (
      typeof l.id === 'number' &&
      RTMPOSE_FILLED_SET.has(l.id) &&
      typeof l.visibility === 'number' &&
      Number.isFinite(l.visibility)
    ) {
      sum += l.visibility
      n += 1
    }
  }
  return n === 0 ? 1.0 : sum / n
}

// True when the frame has any landmark data we can apply per-joint
// visibility checks against. Synthetic fixtures with landmarks=[] or
// only id-less / out-of-whitelist landmarks return false; in that
// case the body-presence check is bypassed.
function hasLandmarkData(frame: PoseFrame): boolean {
  if (!frame.landmarks || frame.landmarks.length === 0) return false
  for (const l of frame.landmarks) {
    if (
      typeof l.id === 'number' &&
      RTMPOSE_FILLED_SET.has(l.id) &&
      typeof l.visibility === 'number' &&
      Number.isFinite(l.visibility)
    ) {
      return true
    }
  }
  return false
}

function median(nums: number[]): number {
  if (nums.length === 0) return 0
  const sorted = [...nums].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid]
}

export function gateClipQuality(frames: PoseFrame[]): GateResult {
  const frameCount = frames.length
  if (frameCount < MIN_FRAMES) {
    return {
      passed: false,
      reason: 'too_few_frames',
      metrics: {
        medianVisibility: 0,
        jumpFraction: 0,
        bodyVisibleFraction: 0,
        frameCount,
      },
    }
  }

  const visPerFrame = frames.map(frameVisibility)
  const medianVisibility = median(visPerFrame)

  // Body-presence check skips frames with no usable landmark data
  // (synthetic test fixtures pass joint_angles directly with
  // landmarks=[]). When no frame has usable landmarks, treat the
  // fraction as 1.0 so the gate doesn't block legacy paths; the
  // visibility / jump checks still apply where they have data.
  let bodyVisibleCount = 0
  let evaluatedCount = 0
  for (const f of frames) {
    if (!hasLandmarkData(f)) continue
    evaluatedCount += 1
    let hipsOk = false
    let shouldersOk = false
    let wristsOk = false
    for (const l of f.landmarks) {
      const v = typeof l.visibility === 'number' ? l.visibility : 0
      if (v < REQUIRED_JOINT_VIS_FLOOR) continue
      if (l.id === ID_LEFT_HIP || l.id === ID_RIGHT_HIP) hipsOk = true
      else if (l.id === ID_LEFT_SHOULDER || l.id === ID_RIGHT_SHOULDER) shouldersOk = true
      else if (l.id === ID_LEFT_WRIST || l.id === ID_RIGHT_WRIST) wristsOk = true
    }
    if (hipsOk && shouldersOk && wristsOk) bodyVisibleCount += 1
  }
  const bodyVisibleFraction =
    evaluatedCount === 0 ? 1.0 : bodyVisibleCount / evaluatedCount

  let jumps = 0
  let total = 0
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i - 1].joint_angles?.hip_rotation
    const b = frames[i].joint_angles?.hip_rotation
    if (
      typeof a === 'number' &&
      typeof b === 'number' &&
      Number.isFinite(a) &&
      Number.isFinite(b)
    ) {
      total += 1
      if (Math.abs(shortAngleDelta(a, b)) > HIP_JUMP_THRESHOLD_DEG) jumps += 1
    }
  }
  const jumpFraction = total === 0 ? 0 : jumps / total

  const metrics = {
    medianVisibility,
    jumpFraction,
    bodyVisibleFraction,
    frameCount,
  }

  if (medianVisibility < MIN_MEDIAN_VISIBILITY) {
    return { passed: false, reason: 'low_visibility', metrics }
  }
  if (bodyVisibleFraction < MIN_BODY_VISIBLE_FRACTION) {
    return { passed: false, reason: 'body_not_visible', metrics }
  }
  if (jumpFraction > MAX_JUMP_FRACTION) {
    return { passed: false, reason: 'excessive_jitter', metrics }
  }

  return { passed: true, metrics }
}

export function gateReasonHumanLabel(reason: GateReason): string {
  switch (reason) {
    case 'low_visibility':
      return 'The pose model could not see your body well on this clip.'
    case 'excessive_jitter':
      return 'The pose tracking was unstable — your body may have been occluded or the clip too low-resolution.'
    case 'body_not_visible':
      return 'Your full body was not consistently in frame — make sure shoulders, hips, and wrists are visible throughout.'
    case 'too_few_frames':
      return 'The clip is too short to grade reliably.'
  }
}
