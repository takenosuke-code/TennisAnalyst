import type { Landmark, JointAngles, PoseFrame } from './supabase'

// MediaPipe Pose landmark indices
export const LANDMARK_INDICES = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
}

// Joint groups for toggle visibility
export const JOINT_GROUPS = {
  shoulders: [LANDMARK_INDICES.LEFT_SHOULDER, LANDMARK_INDICES.RIGHT_SHOULDER],
  elbows: [LANDMARK_INDICES.LEFT_ELBOW, LANDMARK_INDICES.RIGHT_ELBOW],
  wrists: [LANDMARK_INDICES.LEFT_WRIST, LANDMARK_INDICES.RIGHT_WRIST],
  hips: [LANDMARK_INDICES.LEFT_HIP, LANDMARK_INDICES.RIGHT_HIP],
  knees: [LANDMARK_INDICES.LEFT_KNEE, LANDMARK_INDICES.RIGHT_KNEE],
  ankles: [LANDMARK_INDICES.LEFT_ANKLE, LANDMARK_INDICES.RIGHT_ANKLE],
} as const

export type JointGroup = keyof typeof JOINT_GROUPS

// Skeleton connections for drawing lines between joints
export const SKELETON_CONNECTIONS: [number, number][] = [
  // Torso
  [LANDMARK_INDICES.LEFT_SHOULDER, LANDMARK_INDICES.RIGHT_SHOULDER],
  [LANDMARK_INDICES.LEFT_SHOULDER, LANDMARK_INDICES.LEFT_HIP],
  [LANDMARK_INDICES.RIGHT_SHOULDER, LANDMARK_INDICES.RIGHT_HIP],
  [LANDMARK_INDICES.LEFT_HIP, LANDMARK_INDICES.RIGHT_HIP],
  // Left arm
  [LANDMARK_INDICES.LEFT_SHOULDER, LANDMARK_INDICES.LEFT_ELBOW],
  [LANDMARK_INDICES.LEFT_ELBOW, LANDMARK_INDICES.LEFT_WRIST],
  // Right arm
  [LANDMARK_INDICES.RIGHT_SHOULDER, LANDMARK_INDICES.RIGHT_ELBOW],
  [LANDMARK_INDICES.RIGHT_ELBOW, LANDMARK_INDICES.RIGHT_WRIST],
  // Left leg
  [LANDMARK_INDICES.LEFT_HIP, LANDMARK_INDICES.LEFT_KNEE],
  [LANDMARK_INDICES.LEFT_KNEE, LANDMARK_INDICES.LEFT_ANKLE],
  // Right leg
  [LANDMARK_INDICES.RIGHT_HIP, LANDMARK_INDICES.RIGHT_KNEE],
  [LANDMARK_INDICES.RIGHT_KNEE, LANDMARK_INDICES.RIGHT_ANKLE],
]

function vec2(a: Landmark, b: Landmark): [number, number] {
  return [b.x - a.x, b.y - a.y]
}

function angleBetween(v1: [number, number], v2: [number, number]): number {
  const dot = v1[0] * v2[0] + v1[1] * v2[1]
  const mag1 = Math.sqrt(v1[0] ** 2 + v1[1] ** 2)
  const mag2 = Math.sqrt(v2[0] ** 2 + v2[1] ** 2)
  if (mag1 === 0 || mag2 === 0) return 0
  const cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2)))
  return (Math.acos(cosAngle) * 180) / Math.PI
}

function getLandmark(landmarks: Landmark[], id: number): Landmark | undefined {
  return landmarks.find((l) => l.id === id)
}

export function computeJointAngles(landmarks: Landmark[]): JointAngles {
  const lShoulder = getLandmark(landmarks, LANDMARK_INDICES.LEFT_SHOULDER)
  const rShoulder = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_SHOULDER)
  const lElbow = getLandmark(landmarks, LANDMARK_INDICES.LEFT_ELBOW)
  const rElbow = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_ELBOW)
  const lWrist = getLandmark(landmarks, LANDMARK_INDICES.LEFT_WRIST)
  const rWrist = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_WRIST)
  const lIndex = getLandmark(landmarks, LANDMARK_INDICES.LEFT_INDEX)
  const rIndex = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_INDEX)
  const lHip = getLandmark(landmarks, LANDMARK_INDICES.LEFT_HIP)
  const rHip = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_HIP)
  const lKnee = getLandmark(landmarks, LANDMARK_INDICES.LEFT_KNEE)
  const rKnee = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_KNEE)
  const lAnkle = getLandmark(landmarks, LANDMARK_INDICES.LEFT_ANKLE)
  const rAnkle = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_ANKLE)

  const angles: JointAngles = {}

  if (rShoulder && rElbow && rWrist) {
    angles.right_elbow = angleBetween(
      vec2(rElbow, rShoulder),
      vec2(rElbow, rWrist)
    )
  }
  if (lShoulder && lElbow && lWrist) {
    angles.left_elbow = angleBetween(
      vec2(lElbow, lShoulder),
      vec2(lElbow, lWrist)
    )
  }
  if (lShoulder && rShoulder && rElbow) {
    angles.right_shoulder = angleBetween(
      vec2(rShoulder, lShoulder),
      vec2(rShoulder, rElbow)
    )
  }
  if (lShoulder && rShoulder && lElbow) {
    angles.left_shoulder = angleBetween(
      vec2(lShoulder, rShoulder),
      vec2(lShoulder, lElbow)
    )
  }
  // Wrist flexion: elbow → wrist → index. 180° = straight, smaller = flexed.
  if (rElbow && rWrist && rIndex) {
    angles.right_wrist = angleBetween(
      vec2(rWrist, rElbow),
      vec2(rWrist, rIndex)
    )
  }
  if (lElbow && lWrist && lIndex) {
    angles.left_wrist = angleBetween(
      vec2(lWrist, lElbow),
      vec2(lWrist, lIndex)
    )
  }
  if (rHip && rKnee && rAnkle) {
    angles.right_knee = angleBetween(
      vec2(rKnee, rHip),
      vec2(rKnee, rAnkle)
    )
  }
  if (lHip && lKnee && lAnkle) {
    angles.left_knee = angleBetween(
      vec2(lKnee, lHip),
      vec2(lKnee, lAnkle)
    )
  }
  // Hip rotation approximated as angle of hip line relative to vertical
  if (lHip && rHip) {
    const hipVec = vec2(lHip, rHip)
    angles.hip_rotation = Math.abs(
      (Math.atan2(hipVec[1], hipVec[0]) * 180) / Math.PI
    )
  }
  // Trunk rotation: shoulder line angle
  if (lShoulder && rShoulder) {
    const shoulderVec = vec2(lShoulder, rShoulder)
    angles.trunk_rotation = Math.abs(
      (Math.atan2(shoulderVec[1], shoulderVec[0]) * 180) / Math.PI
    )
  }

  return angles
}

// Detected swing segment within a longer video
export type SwingSegment = {
  index: number           // 1-based swing number
  startFrame: number      // index into frames array
  endFrame: number
  startMs: number
  endMs: number
  peakFrame: number       // frame with max activity
  frames: PoseFrame[]
}

/**
 * Detect individual swings in a sequence of frames by tracking
 * wrist movement velocity and joint angle changes.
 * Returns segments of high activity separated by rest periods.
 */
export function detectSwings(
  allFrames: PoseFrame[],
  opts: {
    minSwingFrames?: number
    /**
     * Single-threshold legacy knob. When set, used as the enter
     * threshold and the exit is derived as enter / 4 (matches the
     * 4:1 hysteresis ratio recommended by the EMG-onset literature).
     * Pass `enterThreshold` / `exitThreshold` directly for finer
     * control.
     */
    restThreshold?: number
    enterThreshold?: number
    exitThreshold?: number
    mergeGapFrames?: number
  } = {}
): SwingSegment[] {
  // Hysteresis (Schmitt-trigger) thresholds. Tennis activity profiles
  // dip mid-swing — at the top of the backswing where the racket
  // reverses direction, and through the decelerating follow-through.
  // A single threshold splits a real swing across those valleys; two
  // thresholds (enter HIGH, exit LOW) tolerate them. The 4:1 ratio
  // is the canonical value from EMG onset/offset detection (see
  // PMC10594734) and sits comfortably between the swing's peak and
  // its internal valleys without merging genuinely separate shots.
  //
  // Defaults: enter = 0.40, exit = 0.10 of the (P90 − median) spread.
  // The legacy `restThreshold` knob still works — when provided we
  // use it as the enter and derive exit = enter / 4.
  const enterFrac =
    opts.enterThreshold ?? opts.restThreshold ?? 0.4
  const exitFrac =
    opts.exitThreshold ??
    (opts.restThreshold !== undefined ? opts.restThreshold / 4 : 0.1)
  // mergeGapFrames bumped 5 → 14 so a ~470 ms valley (longer than the
  // backswing-top reversal) closes, but the 1 s+ gap between truly
  // distinct shots in a rally stays open.
  const { minSwingFrames = 12, mergeGapFrames = 14 } = opts

  if (allFrames.length < minSwingFrames) {
    return [{
      index: 1,
      startFrame: 0,
      endFrame: allFrames.length - 1,
      startMs: allFrames[0]?.timestamp_ms ?? 0,
      endMs: allFrames[allFrames.length - 1]?.timestamp_ms ?? 0,
      peakFrame: Math.floor(allFrames.length / 2),
      frames: allFrames,
    }]
  }

  // Compute per-frame "activity" score based on joint angle rate of change
  const activity: number[] = [0]
  for (let i = 1; i < allFrames.length; i++) {
    const prev = allFrames[i - 1].joint_angles
    const curr = allFrames[i].joint_angles
    let delta = 0
    const keys: (keyof typeof curr)[] = [
      'right_elbow', 'left_elbow', 'right_shoulder', 'left_shoulder',
      'hip_rotation', 'trunk_rotation',
    ]
    for (const k of keys) {
      if (prev[k] != null && curr[k] != null) {
        delta += Math.abs((curr[k] as number) - (prev[k] as number))
      }
    }
    activity.push(delta)
  }

  // Smooth activity with a 5-frame moving average
  const smoothed: number[] = []
  const halfWin = 2
  for (let i = 0; i < activity.length; i++) {
    let sum = 0
    let count = 0
    for (let j = Math.max(0, i - halfWin); j <= Math.min(activity.length - 1, i + halfWin); j++) {
      sum += activity[j]
      count++
    }
    smoothed.push(sum / count)
  }

  // Adaptive thresholds = median + frac × (P90 − median), with two fracs:
  // enter (HIGH) and exit (LOW). P90 instead of max so one explosive
  // swing in a long rally doesn't inflate the bar past softer shots —
  // mirrors the same trick LiveSwingDetector.thresholdStats already
  // uses (lib/liveSwingDetector.ts:263-286).
  const sorted = [...smoothed].sort((a, b) => a - b)
  const median = sorted[Math.floor(sorted.length / 2)]
  const p90 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))]
  const spread = p90 - median
  const enterT = median + enterFrac * spread
  const exitT = median + exitFrac * spread

  // Find regions via hysteresis: enter when activity rises above HIGH,
  // exit only when it drops below LOW. Tolerates internal dips smaller
  // than (HIGH − LOW) without splitting the region — exactly the
  // mechanic that stops a single swing from fragmenting at the
  // backswing-top valley.
  const regions: { start: number; end: number; peakIdx: number; peakVal: number }[] = []
  let inRegion = false
  let regionStart = 0
  let peakIdx = 0
  let peakVal = 0

  for (let i = 0; i < smoothed.length; i++) {
    const v = smoothed[i]
    if (!inRegion) {
      if (v >= enterT) {
        inRegion = true
        regionStart = i
        peakIdx = i
        peakVal = v
      }
    } else {
      if (v > peakVal) {
        peakIdx = i
        peakVal = v
      }
      if (v < exitT) {
        regions.push({ start: regionStart, end: i - 1, peakIdx, peakVal })
        inRegion = false
      }
    }
  }
  if (inRegion) {
    regions.push({ start: regionStart, end: smoothed.length - 1, peakIdx, peakVal })
  }

  // Merge regions that are close together
  const merged: typeof regions = []
  for (const r of regions) {
    if (merged.length > 0 && r.start - merged[merged.length - 1].end <= mergeGapFrames) {
      const last = merged[merged.length - 1]
      last.end = r.end
      if (r.peakVal > last.peakVal) {
        last.peakIdx = r.peakIdx
        last.peakVal = r.peakVal
      }
    } else {
      merged.push({ ...r })
    }
  }

  // Asymmetric biomechanical padding. The high-activity span detected
  // above is the *acceleration through follow-through* portion; the
  // backswing/loading runs ~600 ms before that with much lower angle-
  // delta activity (joints are loading, not whipping), so it falls
  // below threshold and gets clipped. Pad more before than after to
  // capture the full motion envelope ~ 670 ms / 500 ms at 30 fps.
  const padBefore = 20
  const padAfter = 15
  const segments: SwingSegment[] = []
  let idx = 1
  for (const r of merged) {
    if (r.end - r.start + 1 < minSwingFrames) continue
    const start = Math.max(0, r.start - padBefore)
    const end = Math.min(allFrames.length - 1, r.end + padAfter)
    segments.push({
      index: idx++,
      startFrame: start,
      endFrame: end,
      startMs: allFrames[start].timestamp_ms,
      endMs: allFrames[end].timestamp_ms,
      peakFrame: r.peakIdx,
      frames: allFrames.slice(start, end + 1),
    })
  }

  // If no swings detected, return the whole video as one segment
  if (segments.length === 0) {
    return [{
      index: 1,
      startFrame: 0,
      endFrame: allFrames.length - 1,
      startMs: allFrames[0]?.timestamp_ms ?? 0,
      endMs: allFrames[allFrames.length - 1]?.timestamp_ms ?? 0,
      peakFrame: Math.floor(allFrames.length / 2),
      frames: allFrames,
    }]
  }

  return segments
}

// Sample N evenly-spaced frames from a keypoints sequence
export function sampleKeyFrames(frames: PoseFrame[], count = 5): PoseFrame[] {
  if (count <= 1) return frames.slice(0, 1)
  if (frames.length <= count) return frames
  const step = (frames.length - 1) / (count - 1)
  return Array.from({ length: count }, (_, i) =>
    frames[Math.round(i * step)]
  )
}

// Build a compact prompt-safe summary of angles (no raw x/y coords)
export function buildAngleSummary(
  frames: PoseFrame[],
  phaseNames: string[] = ['preparation', 'loading', 'contact', 'follow-through', 'finish']
): string {
  const sampled = sampleKeyFrames(frames, 5)
  return sampled
    .map((frame, i) => {
      const a = frame.joint_angles
      const phase = phaseNames[i] ?? `frame_${frame.frame_index}`
      return `${phase}: elbow_R=${a.right_elbow?.toFixed(0) ?? 'N/A'}° elbow_L=${a.left_elbow?.toFixed(0) ?? 'N/A'}° shoulder_R=${a.right_shoulder?.toFixed(0) ?? 'N/A'}° knee_R=${a.right_knee?.toFixed(0) ?? 'N/A'}° hip_rot=${a.hip_rotation?.toFixed(0) ?? 'N/A'}° trunk_rot=${a.trunk_rotation?.toFixed(0) ?? 'N/A'}°`
    })
    .join('\n')
}

// ---------------------------------------------------------------------------
// Camera-robust helpers (re-exported from `lib/cameraNormalization`).
//
// These are siblings of the camera-frame helpers above, factored out so
// callers can opt into camera-invariant comparisons without reaching
// across modules. The implementations live in `lib/cameraNormalization.ts`
// — kept there because they share private constants (visibility floor,
// hip-line minimum) with the other camera-normalization utilities. We
// re-export here so `import { computeRotationExcursion } from
// '@/lib/jointAngles'` is the natural ergonomic path for code already
// pulling other angle helpers from this module.
// ---------------------------------------------------------------------------

export {
  computeBodyFrameAngles,
  computeRotationExcursion,
  computeCameraSimilarity,
} from './cameraNormalization'
export type { CameraClass } from './cameraNormalization'
