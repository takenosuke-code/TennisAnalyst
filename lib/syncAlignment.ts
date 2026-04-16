import type { PoseFrame, Landmark } from './supabase'
import { LANDMARK_INDICES } from './jointAngles'

export type SwingPhase =
  | 'preparation'
  | 'backswing'
  | 'forward_swing'
  | 'contact'
  | 'follow_through'

export interface PhaseTimestamp {
  phase: SwingPhase
  timestampMs: number
  frameIndex: number
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function getLandmark(
  landmarks: Landmark[],
  id: number
): Landmark | undefined {
  return landmarks.find((l) => l.id === id)
}

/**
 * Returns the "hip center" as the midpoint of left and right hip landmarks.
 */
function hipCenter(landmarks: Landmark[]): { x: number; y: number } | null {
  const lh = getLandmark(landmarks, LANDMARK_INDICES.LEFT_HIP)
  const rh = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_HIP)
  if (!lh || !rh) return null
  return { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 }
}

/**
 * Determine the dominant hand side from landmark data.
 * The wrist with the greater average x-displacement from hip center over all
 * frames is considered the dominant (racket) hand.
 * Returns 'left' | 'right'.
 */
function detectDominantSide(
  frames: PoseFrame[]
): 'left' | 'right' {
  let leftTotal = 0
  let rightTotal = 0
  let count = 0

  for (const frame of frames) {
    const hip = hipCenter(frame.landmarks)
    if (!hip) continue
    const lw = getLandmark(frame.landmarks, LANDMARK_INDICES.LEFT_WRIST)
    const rw = getLandmark(frame.landmarks, LANDMARK_INDICES.RIGHT_WRIST)
    if (!lw || !rw) continue
    leftTotal += Math.abs(lw.x - hip.x)
    rightTotal += Math.abs(rw.x - hip.x)
    count++
  }

  if (count === 0) return 'right'
  return leftTotal / count > rightTotal / count ? 'left' : 'right'
}

function dominantWristId(side: 'left' | 'right'): number {
  return side === 'left'
    ? LANDMARK_INDICES.LEFT_WRIST
    : LANDMARK_INDICES.RIGHT_WRIST
}

// ---------------------------------------------------------------------------
// Phase detection
// ---------------------------------------------------------------------------

/**
 * Detect swing phases from a sequence of pose frames.
 *
 * Phase detection logic:
 * - Preparation: first stable frame where trunk_rotation < 10 deg from baseline
 * - Backswing: frame where dominant wrist reaches maximum backward x-displacement from hip center
 * - Forward swing: frame with maximum angular velocity of dominant elbow
 * - Contact: frame where dominant wrist has the highest y-velocity change (proxy for ball strike)
 * - Follow through: frame where dominant wrist passes the midline after contact
 *
 * Returns only the phases that could be detected. Callers should handle
 * partial results gracefully.
 */
export function detectSwingPhases(frames: PoseFrame[]): PhaseTimestamp[] {
  if (frames.length < 5) return []

  const side = detectDominantSide(frames)
  const wristId = dominantWristId(side)

  // Pre-compute per-frame wrist position relative to hip center
  const wristDisplacements: { dx: number; dy: number; idx: number }[] = []
  for (let i = 0; i < frames.length; i++) {
    const hip = hipCenter(frames[i].landmarks)
    const wrist = getLandmark(frames[i].landmarks, wristId)
    if (hip && wrist) {
      wristDisplacements.push({
        dx: wrist.x - hip.x,
        dy: wrist.y - hip.y,
        idx: i,
      })
    } else {
      wristDisplacements.push({ dx: 0, dy: 0, idx: i })
    }
  }

  const detected: PhaseTimestamp[] = []

  // --- Preparation ---
  // Baseline trunk rotation = average over first 5 frames
  const baselineTrunk = (() => {
    let sum = 0
    let n = 0
    for (let i = 0; i < Math.min(5, frames.length); i++) {
      const tr = frames[i].joint_angles?.trunk_rotation
      if (tr != null) {
        sum += tr
        n++
      }
    }
    return n > 0 ? sum / n : null
  })()

  if (baselineTrunk != null) {
    for (let i = 0; i < frames.length; i++) {
      const tr = frames[i].joint_angles?.trunk_rotation
      if (tr != null && Math.abs(tr - baselineTrunk) < 10) {
        detected.push({
          phase: 'preparation',
          timestampMs: frames[i].timestamp_ms,
          frameIndex: frames[i].frame_index,
        })
        break
      }
    }
  }

  // --- Backswing ---
  // Peak backward displacement of dominant wrist from hip center.
  // "Backward" depends on side: for right-hander the racket goes to
  // positive-x (right side of frame) during backswing; for left-hander,
  // negative-x. We use the signed value in the dominant direction.
  let peakBackswingIdx = -1
  let peakBackswingVal = -Infinity
  const signMultiplier = side === 'right' ? 1 : -1
  for (const wd of wristDisplacements) {
    const val = signMultiplier * wd.dx
    if (val > peakBackswingVal) {
      peakBackswingVal = val
      peakBackswingIdx = wd.idx
    }
  }
  if (peakBackswingIdx >= 0) {
    detected.push({
      phase: 'backswing',
      timestampMs: frames[peakBackswingIdx].timestamp_ms,
      frameIndex: frames[peakBackswingIdx].frame_index,
    })
  }

  // --- Forward swing ---
  // Frame with maximum angular velocity of the dominant elbow
  const elbowKey =
    side === 'left' ? 'left_elbow' : 'right_elbow'
  let peakAngVelIdx = -1
  let peakAngVel = -Infinity
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1].joint_angles?.[elbowKey]
    const curr = frames[i].joint_angles?.[elbowKey]
    if (prev == null || curr == null) continue
    const dt =
      (frames[i].timestamp_ms - frames[i - 1].timestamp_ms) / 1000
    if (dt <= 0) continue
    const angVel = Math.abs(curr - prev) / dt
    if (angVel > peakAngVel) {
      peakAngVel = angVel
      peakAngVelIdx = i
    }
  }
  if (peakAngVelIdx >= 0) {
    detected.push({
      phase: 'forward_swing',
      timestampMs: frames[peakAngVelIdx].timestamp_ms,
      frameIndex: frames[peakAngVelIdx].frame_index,
    })
  }

  // --- Contact ---
  // Frame with the highest absolute y-velocity change of the dominant wrist
  // (proxy for the abrupt deceleration at ball strike).
  const yVelocities: number[] = [0]
  for (let i = 1; i < wristDisplacements.length; i++) {
    const dt =
      (frames[i].timestamp_ms - frames[i - 1].timestamp_ms) / 1000
    if (dt <= 0) {
      yVelocities.push(0)
      continue
    }
    yVelocities.push(
      (wristDisplacements[i].dy - wristDisplacements[i - 1].dy) / dt
    )
  }

  let peakAccelIdx = -1
  let peakAccel = -Infinity
  for (let i = 1; i < yVelocities.length; i++) {
    const accel = Math.abs(yVelocities[i] - yVelocities[i - 1])
    if (accel > peakAccel) {
      peakAccel = accel
      peakAccelIdx = i
    }
  }
  if (peakAccelIdx >= 0) {
    detected.push({
      phase: 'contact',
      timestampMs: frames[peakAccelIdx].timestamp_ms,
      frameIndex: frames[peakAccelIdx].frame_index,
    })
  }

  // --- Follow through ---
  // First frame after contact where the wrist crosses the midline
  // (i.e., displacement sign flips from the backswing direction).
  const contactIdx = peakAccelIdx >= 0 ? peakAccelIdx : Math.floor(frames.length / 2)
  for (let i = contactIdx + 1; i < wristDisplacements.length; i++) {
    const crossed = signMultiplier * wristDisplacements[i].dx < 0
    if (crossed) {
      detected.push({
        phase: 'follow_through',
        timestampMs: frames[i].timestamp_ms,
        frameIndex: frames[i].frame_index,
      })
      break
    }
  }

  // Sort by timestamp so phases are in chronological order
  detected.sort((a, b) => a.timestampMs - b.timestampMs)

  return detected
}

// ---------------------------------------------------------------------------
// Time mapping
// ---------------------------------------------------------------------------

/**
 * Given user swing phases and pro swing phases, compute a function that maps
 * user video time (ms) to pro video time (ms) so that matching phases align.
 *
 * Uses piecewise linear interpolation between matched phase timestamps.
 * Falls back to identity mapping (scaled by duration ratio) if fewer than 2
 * phases match.
 */
export function computeTimeMapping(
  userPhases: PhaseTimestamp[],
  proPhases: PhaseTimestamp[],
  userDurationMs?: number,
  proDurationMs?: number
): (userTimeMs: number) => number {
  // Build matched anchor points: only phases that appear in both arrays
  const proPhaseMap = new Map<SwingPhase, PhaseTimestamp>()
  for (const p of proPhases) {
    proPhaseMap.set(p.phase, p)
  }

  const anchors: { userMs: number; proMs: number }[] = []
  for (const up of userPhases) {
    const pp = proPhaseMap.get(up.phase)
    if (pp) {
      anchors.push({ userMs: up.timestampMs, proMs: pp.timestampMs })
    }
  }

  // Sort anchors by user time
  anchors.sort((a, b) => a.userMs - b.userMs)

  // Need at least 2 anchors for meaningful piecewise interpolation
  if (anchors.length < 2) {
    // Fallback: linear scale by duration ratio
    const uDur = userDurationMs ?? 1
    const pDur = proDurationMs ?? uDur
    const ratio = uDur > 0 ? pDur / uDur : 1
    return (userTimeMs: number) => userTimeMs * ratio
  }

  return (userTimeMs: number): number => {
    // Before first anchor -hold at the pro's first phase time instead of
    // extrapolating backwards (which produces nonsensical values during the
    // user's pre-swing dead time).
    if (userTimeMs <= anchors[0].userMs) {
      return anchors[0].proMs
    }

    // After last anchor
    const last = anchors[anchors.length - 1]
    if (userTimeMs >= last.userMs) {
      // Extrapolate using the slope of the last segment
      const prev = anchors[anchors.length - 2]
      const dUser = last.userMs - prev.userMs
      const dPro = last.proMs - prev.proMs
      if (dUser === 0) return last.proMs
      const slope = dPro / dUser
      return last.proMs + slope * (userTimeMs - last.userMs)
    }

    // Between anchors: find the segment and linearly interpolate
    for (let i = 0; i < anchors.length - 1; i++) {
      const a = anchors[i]
      const b = anchors[i + 1]
      if (userTimeMs >= a.userMs && userTimeMs <= b.userMs) {
        const t =
          b.userMs === a.userMs
            ? 0
            : (userTimeMs - a.userMs) / (b.userMs - a.userMs)
        return a.proMs + t * (b.proMs - a.proMs)
      }
    }

    // Should not reach here, but return scaled fallback
    return userTimeMs
  }
}

// ---------------------------------------------------------------------------
// Phase label for current time
// ---------------------------------------------------------------------------

/**
 * Return the currently active phase label for a given timestamp.
 */
export function getActivePhase(
  phases: PhaseTimestamp[],
  timeMs: number
): SwingPhase | null {
  if (phases.length === 0) return null

  // Find the latest phase whose timestamp is <= timeMs
  let active: PhaseTimestamp | null = null
  for (const p of phases) {
    if (p.timestampMs <= timeMs) {
      active = p
    }
  }
  return active?.phase ?? null
}

// ---------------------------------------------------------------------------
// Clip selection
// ---------------------------------------------------------------------------

/**
 * Estimate the camera angle from pose landmarks.
 * Returns a number between 0 (pure side view) and 1 (pure front view).
 *
 * Heuristic: in a front-facing view, the shoulder width (in image space)
 * is large relative to the hip-to-shoulder vertical distance. In a side view,
 * the shoulder width is foreshortened and appears small.
 *
 * We compute the ratio: shoulderWidth / (shoulderWidth + hipShoulderDist)
 * averaged over a sample of frames.
 */
function estimateCameraAngle(frames: PoseFrame[]): number {
  const sampleCount = Math.min(20, frames.length)
  const step = Math.max(1, Math.floor(frames.length / sampleCount))
  let totalRatio = 0
  let count = 0

  for (let i = 0; i < frames.length; i += step) {
    const ls = getLandmark(frames[i].landmarks, LANDMARK_INDICES.LEFT_SHOULDER)
    const rs = getLandmark(frames[i].landmarks, LANDMARK_INDICES.RIGHT_SHOULDER)
    const lh = getLandmark(frames[i].landmarks, LANDMARK_INDICES.LEFT_HIP)
    const rh = getLandmark(frames[i].landmarks, LANDMARK_INDICES.RIGHT_HIP)
    if (!ls || !rs || !lh || !rh) continue

    const shoulderWidth = Math.sqrt(
      (rs.x - ls.x) ** 2 + (rs.y - ls.y) ** 2
    )
    const hipMid = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 }
    const shoulderMid = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 }
    const hipShoulderDist = Math.sqrt(
      (shoulderMid.x - hipMid.x) ** 2 + (shoulderMid.y - hipMid.y) ** 2
    )

    const denom = shoulderWidth + hipShoulderDist
    if (denom > 0) {
      totalRatio += shoulderWidth / denom
      count++
    }
  }

  return count > 0 ? totalRatio / count : 0.5
}

/**
 * Given a user's pose frames and shot type, select the best matching pro clip
 * from the available options.
 *
 * Selection criteria:
 * 1. Filter by shot type match.
 * 2. Among matches, pick the clip whose camera angle most closely matches the user's.
 *
 * Returns the index into `availableClips`.
 */
export function selectBestProClip(
  userFrames: PoseFrame[],
  userShotType: string,
  availableClips: { shotType: string; frames: PoseFrame[] }[]
): number {
  if (availableClips.length === 0) return 0

  // Normalize shot type for comparison
  const normalizedUserShot = userShotType.toLowerCase().trim()

  // Filter by shot type
  const shotMatches = availableClips
    .map((clip, idx) => ({ clip, idx }))
    .filter(
      ({ clip }) =>
        clip.shotType.toLowerCase().trim() === normalizedUserShot
    )

  // If no shot type match, fall back to all clips
  const candidates = shotMatches.length > 0 ? shotMatches : availableClips.map((clip, idx) => ({ clip, idx }))

  if (candidates.length === 1) return candidates[0].idx

  // Estimate user camera angle
  const userAngle = estimateCameraAngle(userFrames)

  // Find the clip with the closest camera angle
  let bestIdx = candidates[0].idx
  let bestDiff = Infinity

  for (const { clip, idx } of candidates) {
    const clipAngle = estimateCameraAngle(clip.frames)
    const diff = Math.abs(clipAngle - userAngle)
    if (diff < bestDiff) {
      bestDiff = diff
      bestIdx = idx
    }
  }

  return bestIdx
}
