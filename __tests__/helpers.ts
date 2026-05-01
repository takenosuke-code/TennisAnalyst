import type { Landmark, JointAngles, PoseFrame } from '@/lib/supabase'
import { LANDMARK_INDICES } from '@/lib/jointAngles'

export function makeLandmark(
  id: number,
  x: number,
  y: number,
  visibility = 1.0,
  z = 0
): Landmark {
  return { id, name: `landmark_${id}`, x, y, z, visibility }
}

export function makeFrame(
  index: number,
  timestampMs: number,
  landmarks: Landmark[],
  angles: JointAngles = {}
): PoseFrame {
  return {
    frame_index: index,
    timestamp_ms: timestampMs,
    landmarks,
    joint_angles: angles,
  }
}

/**
 * Create a standard set of visible landmarks in a natural standing pose.
 * Positions are in normalized [0,1] coordinates.
 */
export function makeStandingPose(): Landmark[] {
  return [
    makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1),
    makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.55, 0.25),
    makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.45, 0.25),
    makeLandmark(LANDMARK_INDICES.LEFT_ELBOW, 0.6, 0.4),
    makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.4, 0.4),
    makeLandmark(LANDMARK_INDICES.LEFT_WRIST, 0.6, 0.55),
    makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.4, 0.55),
    makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.53, 0.55),
    makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.47, 0.55),
    makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.54, 0.72),
    makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.46, 0.72),
    makeLandmark(LANDMARK_INDICES.LEFT_ANKLE, 0.54, 0.9),
    makeLandmark(LANDMARK_INDICES.RIGHT_ANKLE, 0.46, 0.9),
  ]
}

/**
 * Generate a sequence of frames that simulate a forehand swing
 * with realistic joint angle progression through phases:
 * preparation -> loading -> acceleration -> contact -> follow-through
 *
 * The wrist positions and joint angles change significantly during the swing
 * to produce detectable "activity" in detectSwings().
 */
export function makeForehandSwingFrames(
  numFrames: number,
  startMs: number,
  frameDurationMs = 33
): PoseFrame[] {
  const frames: PoseFrame[] = []
  for (let i = 0; i < numFrames; i++) {
    const t = i / Math.max(1, numFrames - 1) // 0..1 progress through swing
    const timestampMs = startMs + i * frameDurationMs

    // Simulate forehand phases via right elbow angle progression:
    //   preparation (t<0.2): elbow ~120 degrees (arm pulled back)
    //   loading (0.2-0.4): elbow compresses to ~80 degrees
    //   acceleration (0.4-0.6): elbow extends rapidly to ~160 degrees
    //   contact (0.6-0.7): elbow nearly straight ~170 degrees
    //   follow-through (0.7-1.0): elbow relaxes to ~130 degrees
    let rightElbow: number
    let hipRotation: number
    let trunkRotation: number
    if (t < 0.2) {
      rightElbow = 120
      hipRotation = 10
      trunkRotation = 15
    } else if (t < 0.4) {
      const phase = (t - 0.2) / 0.2
      rightElbow = 120 - phase * 40 // 120 -> 80
      hipRotation = 10 + phase * 20 // 10 -> 30
      trunkRotation = 15 + phase * 15 // 15 -> 30
    } else if (t < 0.6) {
      const phase = (t - 0.4) / 0.2
      rightElbow = 80 + phase * 80 // 80 -> 160
      hipRotation = 30 + phase * 30 // 30 -> 60
      trunkRotation = 30 + phase * 25 // 30 -> 55
    } else if (t < 0.7) {
      const phase = (t - 0.6) / 0.1
      rightElbow = 160 + phase * 10 // 160 -> 170
      hipRotation = 60 + phase * 5 // 60 -> 65
      trunkRotation = 55 + phase * 5 // 55 -> 60
    } else {
      const phase = (t - 0.7) / 0.3
      rightElbow = 170 - phase * 40 // 170 -> 130
      hipRotation = 65 - phase * 25 // 65 -> 40
      trunkRotation = 60 - phase * 20 // 60 -> 40
    }

    // Drive the right wrist through a forehand arc so the
    // wrist-velocity peak-pick detector (lib/swingDetect.ts) can lock
    // on. Earlier versions of this helper used a static
    // makeStandingPose() and only varied joint angles — that worked
    // when detectSwings was joint-angle-energy based, but the new
    // detector reads dominant-wrist linear speed.
    //
    // Position profile: slow backswing → fast acceleration → contact
    // → fast follow-through → gentle return to rest. Asymmetric on
    // purpose so the SPEED signal has a single dominant peak around
    // contact (t ≈ 0.55), not two peaks like a symmetric sin would
    // produce. Boundary positions match the rest pose (x=0.4) so
    // back-to-back rest/swing fixtures don't generate boundary
    // velocity spikes.
    const landmarks = makeStandingPose().map((lm) => {
      if (lm.id !== LANDMARK_INDICES.RIGHT_WRIST) return lm
      const xArc = forehandWristX(t)
      const yArc = 0.55 - 0.06 * Math.sin(Math.PI * t)
      return { ...lm, x: xArc, y: yArc }
    })

    const angles: JointAngles = {
      right_elbow: rightElbow,
      left_elbow: 160 - t * 20, // slight change
      right_shoulder: 45 + t * 40,
      left_shoulder: 30 + t * 10,
      right_knee: 170 - Math.sin(t * Math.PI) * 20,
      left_knee: 170 - Math.sin(t * Math.PI) * 15,
      hip_rotation: hipRotation,
      trunk_rotation: trunkRotation,
    }

    frames.push(makeFrame(i, timestampMs, landmarks, angles))
  }
  return frames
}

/**
 * Generate frames with no significant joint angle changes (rest/idle).
 */
export function makeRestFrames(
  numFrames: number,
  startMs: number,
  frameDurationMs = 33
): PoseFrame[] {
  const frames: PoseFrame[] = []
  for (let i = 0; i < numFrames; i++) {
    const timestampMs = startMs + i * frameDurationMs
    const landmarks = makeStandingPose()
    const angles: JointAngles = {
      right_elbow: 160,
      left_elbow: 160,
      right_shoulder: 30,
      left_shoulder: 30,
      right_knee: 170,
      left_knee: 170,
      hip_rotation: 10,
      trunk_rotation: 10,
    }
    frames.push(makeFrame(i, timestampMs, landmarks, angles))
  }
  return frames
}

/**
 * Position profile of the right wrist along a forehand swing
 * parameterized by t∈[0,1]. Boundary x equals the rest position (0.4)
 * and the peak SPEED (not position) lands near t≈0.55 — the contact
 * phase. The shape is a piecewise blend of a slow backswing pull-back,
 * a fast forward swing through contact, and a gentle return to rest.
 *
 * Cubic in the [0, 0.6] window then a half-cosine tail. We park this
 * here rather than inline so multiple test helpers can build on the
 * same canonical motion.
 */
export function forehandWristX(t: number): number {
  const REST_X = 0.4
  const PEAK_X = 0.85
  if (t <= 0) return REST_X
  if (t >= 1) return REST_X
  if (t < 0.6) {
    // Cubic ease — slow start, fast finish, peaks ~at t=0.6.
    const u = t / 0.6
    return REST_X + (PEAK_X - REST_X) * (u * u * u)
  }
  // Half-cosine return: from PEAK_X at t=0.6 back to REST_X at t=1.
  const u = (t - 0.6) / 0.4
  return REST_X + (PEAK_X - REST_X) * (1 - u) * (1 - u)
}

/**
 * Make a standing pose with the right wrist driven horizontally by a
 * t∈[0,1] sweep. Useful for inline-fixture swing tests that need the
 * wrist-velocity detector (lib/swingDetect.ts) to fire — a static
 * pose has zero wrist speed and won't trigger detection.
 */
export function makeStandingPoseWithWristAt(t: number): Landmark[] {
  const x = forehandWristX(Math.min(1, Math.max(0, t)))
  const y = 0.55 - 0.06 * Math.sin(Math.PI * t)
  return makeStandingPose().map((lm) =>
    lm.id === LANDMARK_INDICES.RIGHT_WRIST ? { ...lm, x, y } : lm,
  )
}

/**
 * Build a right-angle elbow pose:
 * Shoulder at (0.4, 0.25), Elbow at (0.4, 0.4), Wrist at (0.55, 0.4)
 * The shoulder-elbow vector is (0, 0.15) pointing down,
 * the wrist-elbow vector is (0.15, 0) pointing right.
 * This forms a 90-degree angle at the elbow.
 */
export function makeRightAngleElbowPose(): Landmark[] {
  return [
    makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1),
    makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.55, 0.25),
    makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.4, 0.25),
    makeLandmark(LANDMARK_INDICES.LEFT_ELBOW, 0.6, 0.4),
    makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.4, 0.4),
    makeLandmark(LANDMARK_INDICES.LEFT_WRIST, 0.6, 0.55),
    // Right wrist placed to form a 90-degree angle at the right elbow
    // rShoulder=(0.4,0.25), rElbow=(0.4,0.4), rWrist=(0.55,0.4)
    // vec(elbow->shoulder) = (0, -0.15), vec(elbow->wrist) = (0.15, 0)
    // dot = 0, so angle = 90 degrees
    makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.55, 0.4),
    makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.53, 0.55),
    makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.47, 0.55),
    makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.54, 0.72),
    makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.46, 0.72),
    makeLandmark(LANDMARK_INDICES.LEFT_ANKLE, 0.54, 0.9),
    makeLandmark(LANDMARK_INDICES.RIGHT_ANKLE, 0.46, 0.9),
  ]
}
