import { describe, it, expect } from 'vitest'
import {
  detectSwingPhases,
  computeTimeMapping,
  getActivePhase,
  selectBestProClip,
  type PhaseTimestamp,
} from '@/lib/syncAlignment'
import { LANDMARK_INDICES } from '@/lib/jointAngles'
import type { PoseFrame, JointAngles, Landmark } from '@/lib/supabase'
import { makeLandmark, makeFrame } from '../helpers'

// ---------------------------------------------------------------------------
// Test helpers - builds frames with realistic wrist movement for phase detection
// ---------------------------------------------------------------------------

/**
 * Create a set of landmarks with explicit wrist positions so the phase
 * detector can track wrist displacement from hip center.
 */
function makePoseWithWrist(
  rightWristX: number,
  rightWristY: number,
  leftWristX = 0.6,
  leftWristY = 0.55
): Landmark[] {
  return [
    makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1),
    makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.55, 0.25),
    makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.45, 0.25),
    makeLandmark(LANDMARK_INDICES.LEFT_ELBOW, 0.6, 0.4),
    makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.4, 0.4),
    makeLandmark(LANDMARK_INDICES.LEFT_WRIST, leftWristX, leftWristY),
    makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, rightWristX, rightWristY),
    makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.53, 0.55),
    makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.47, 0.55),
    makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.54, 0.72),
    makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.46, 0.72),
    makeLandmark(LANDMARK_INDICES.LEFT_ANKLE, 0.54, 0.9),
    makeLandmark(LANDMARK_INDICES.RIGHT_ANKLE, 0.46, 0.9),
  ]
}

/**
 * Generate a forehand swing sequence with realistic wrist trajectory.
 * Right-handed swing: wrist starts near hip, goes back (high x),
 * swings forward (low x), and crosses midline.
 */
function makeSwingFrames(
  numFrames: number,
  startMs: number,
  intervalMs: number
): PoseFrame[] {
  const frames: PoseFrame[] = []
  for (let i = 0; i < numFrames; i++) {
    const t = i / Math.max(1, numFrames - 1) // 0..1
    const ts = startMs + i * intervalMs

    // Wrist x: starts at 0.4 (near hip center 0.5), goes to 0.7 (backswing),
    // then drops to 0.2 (forward/contact), then continues to 0.1 (follow-through)
    let wristX: number
    let wristY: number
    let rightElbow: number
    let trunkRot: number

    if (t < 0.15) {
      // Preparation: wrist near body
      wristX = 0.4
      wristY = 0.55
      rightElbow = 120
      trunkRot = 5 // baseline-ish
    } else if (t < 0.35) {
      // Backswing: wrist moves far right (positive x from hip)
      const p = (t - 0.15) / 0.2
      wristX = 0.4 + p * 0.3 // 0.4 -> 0.7
      wristY = 0.45 - p * 0.1 // slight upward
      rightElbow = 120 - p * 40 // compress
      trunkRot = 5 + p * 30
    } else if (t < 0.55) {
      // Forward swing: rapid elbow extension
      const p = (t - 0.35) / 0.2
      wristX = 0.7 - p * 0.4 // 0.7 -> 0.3
      wristY = 0.35 + p * 0.2 // dropping
      rightElbow = 80 + p * 90 // fast extension
      trunkRot = 35 + p * 20
    } else if (t < 0.65) {
      // Contact: sharp y-velocity change
      const p = (t - 0.55) / 0.1
      wristX = 0.3 - p * 0.1 // 0.3 -> 0.2
      wristY = 0.55 - p * 0.3 // sudden upward change
      rightElbow = 170
      trunkRot = 55 + p * 5
    } else {
      // Follow through: wrist crosses midline (below 0.5 for right-hander)
      const p = (t - 0.65) / 0.35
      wristX = 0.2 - p * 0.25 // crosses to negative-side of hip
      wristY = 0.25 + p * 0.2
      rightElbow = 170 - p * 30
      trunkRot = 60 - p * 15
    }

    const landmarks = makePoseWithWrist(wristX, wristY)
    const angles: JointAngles = {
      right_elbow: rightElbow,
      left_elbow: 150,
      right_shoulder: 60,
      left_shoulder: 40,
      right_knee: 170,
      left_knee: 170,
      hip_rotation: 10,
      trunk_rotation: trunkRot,
    }

    frames.push(makeFrame(i, ts, landmarks, angles))
  }
  return frames
}

// ---------------------------------------------------------------------------
// detectSwingPhases
// ---------------------------------------------------------------------------

describe('detectSwingPhases', () => {
  it('returns empty array for fewer than 5 frames', () => {
    const frames = makeSwingFrames(3, 0, 33)
    expect(detectSwingPhases(frames)).toEqual([])
  })

  it('detects at least preparation and backswing for a realistic swing', () => {
    const frames = makeSwingFrames(60, 0, 33)
    const phases = detectSwingPhases(frames)
    const phaseNames = phases.map((p) => p.phase)

    expect(phaseNames).toContain('preparation')
    expect(phaseNames).toContain('backswing')
  })

  it('returns phases in chronological order', () => {
    const frames = makeSwingFrames(60, 0, 33)
    const phases = detectSwingPhases(frames)

    for (let i = 1; i < phases.length; i++) {
      expect(phases[i].timestampMs).toBeGreaterThanOrEqual(
        phases[i - 1].timestampMs
      )
    }
  })

  it('detects contact phase with sharp velocity change', () => {
    const frames = makeSwingFrames(80, 0, 33)
    const phases = detectSwingPhases(frames)
    const phaseNames = phases.map((p) => p.phase)

    expect(phaseNames).toContain('contact')
  })

  it('every phase has a valid frameIndex and timestampMs', () => {
    const frames = makeSwingFrames(60, 100, 33)
    const phases = detectSwingPhases(frames)

    for (const phase of phases) {
      expect(phase.timestampMs).toBeGreaterThanOrEqual(100)
      expect(phase.frameIndex).toBeGreaterThanOrEqual(0)
    }
  })
})

// ---------------------------------------------------------------------------
// computeTimeMapping
// ---------------------------------------------------------------------------

describe('computeTimeMapping', () => {
  const userPhases: PhaseTimestamp[] = [
    { phase: 'preparation', timestampMs: 200, frameIndex: 6 },
    { phase: 'backswing', timestampMs: 600, frameIndex: 18 },
    { phase: 'contact', timestampMs: 1000, frameIndex: 30 },
    { phase: 'follow_through', timestampMs: 1400, frameIndex: 42 },
  ]

  const proPhases: PhaseTimestamp[] = [
    { phase: 'preparation', timestampMs: 0, frameIndex: 0 },
    { phase: 'backswing', timestampMs: 500, frameIndex: 15 },
    { phase: 'contact', timestampMs: 900, frameIndex: 27 },
    { phase: 'follow_through', timestampMs: 1200, frameIndex: 36 },
  ]

  it('maps matched phase timestamps exactly', () => {
    const fn = computeTimeMapping(userPhases, proPhases)
    expect(fn(200)).toBeCloseTo(0, 1)
    expect(fn(600)).toBeCloseTo(500, 1)
    expect(fn(1000)).toBeCloseTo(900, 1)
    expect(fn(1400)).toBeCloseTo(1200, 1)
  })

  it('interpolates linearly between phases', () => {
    const fn = computeTimeMapping(userPhases, proPhases)
    // Midpoint between preparation(200->0) and backswing(600->500)
    const mid = fn(400) // should be ~250
    expect(mid).toBeCloseTo(250, 0)
  })

  it('extrapolates before first anchor', () => {
    const fn = computeTimeMapping(userPhases, proPhases)
    const result = fn(0) // before preparation at 200ms
    expect(result).toBeDefined()
    expect(typeof result).toBe('number')
  })

  it('extrapolates after last anchor', () => {
    const fn = computeTimeMapping(userPhases, proPhases)
    const result = fn(1800) // after follow_through at 1400ms
    expect(result).toBeGreaterThan(1200) // past last pro phase
  })

  it('falls back to duration ratio when fewer than 2 matching phases', () => {
    const singleUser: PhaseTimestamp[] = [
      { phase: 'contact', timestampMs: 1000, frameIndex: 30 },
    ]
    const singlePro: PhaseTimestamp[] = [
      { phase: 'backswing', timestampMs: 500, frameIndex: 15 },
    ]
    // No matching phases -> falls back to linear scale
    const fn = computeTimeMapping(singleUser, singlePro, 2000, 1000)
    expect(fn(1000)).toBeCloseTo(500, 0) // ratio = 0.5
  })
})

// ---------------------------------------------------------------------------
// getActivePhase
// ---------------------------------------------------------------------------

describe('getActivePhase', () => {
  const phases: PhaseTimestamp[] = [
    { phase: 'preparation', timestampMs: 0, frameIndex: 0 },
    { phase: 'backswing', timestampMs: 500, frameIndex: 15 },
    { phase: 'contact', timestampMs: 900, frameIndex: 27 },
  ]

  it('returns null for empty phases', () => {
    expect(getActivePhase([], 500)).toBeNull()
  })

  it('returns the latest phase at or before the given time', () => {
    expect(getActivePhase(phases, 600)).toBe('backswing')
    expect(getActivePhase(phases, 500)).toBe('backswing')
    expect(getActivePhase(phases, 499)).toBe('preparation')
    expect(getActivePhase(phases, 1000)).toBe('contact')
  })

  it('returns the first phase at its exact timestamp', () => {
    expect(getActivePhase(phases, 0)).toBe('preparation')
  })
})

// ---------------------------------------------------------------------------
// selectBestProClip
// ---------------------------------------------------------------------------

describe('selectBestProClip', () => {
  // Side-view pose: narrow shoulders, tall hip-shoulder distance
  function makeSideViewPose(): Landmark[] {
    return [
      makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1),
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.52, 0.25),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.48, 0.25),
      makeLandmark(LANDMARK_INDICES.LEFT_ELBOW, 0.55, 0.4),
      makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.45, 0.4),
      makeLandmark(LANDMARK_INDICES.LEFT_WRIST, 0.55, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.45, 0.55),
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.52, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.48, 0.55),
      makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.52, 0.72),
      makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.48, 0.72),
      makeLandmark(LANDMARK_INDICES.LEFT_ANKLE, 0.52, 0.9),
      makeLandmark(LANDMARK_INDICES.RIGHT_ANKLE, 0.48, 0.9),
    ]
  }

  // Front-view pose: wide shoulders
  function makeFrontViewPose(): Landmark[] {
    return [
      makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1),
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.65, 0.25),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.35, 0.25),
      makeLandmark(LANDMARK_INDICES.LEFT_ELBOW, 0.7, 0.4),
      makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.3, 0.4),
      makeLandmark(LANDMARK_INDICES.LEFT_WRIST, 0.7, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.3, 0.55),
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.55, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.45, 0.55),
      makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.55, 0.72),
      makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.45, 0.72),
      makeLandmark(LANDMARK_INDICES.LEFT_ANKLE, 0.55, 0.9),
      makeLandmark(LANDMARK_INDICES.RIGHT_ANKLE, 0.45, 0.9),
    ]
  }

  it('selects the clip matching the shot type', () => {
    const userFrames = [makeFrame(0, 0, makeFrontViewPose())]
    const clips = [
      { shotType: 'backhand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: 'serve', frames: [makeFrame(0, 0, makeFrontViewPose())] },
    ]

    expect(selectBestProClip(userFrames, 'forehand', clips)).toBe(1)
  })

  it('falls back to camera angle matching when no shot type matches', () => {
    const userFrames = [makeFrame(0, 0, makeSideViewPose())]
    const clips = [
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeSideViewPose())] },
    ]

    // User has 'volley' but no clips match; should pick by camera angle
    const idx = selectBestProClip(userFrames, 'volley', clips)
    // Side-view user should prefer the side-view clip (index 1)
    expect(idx).toBe(1)
  })

  it('among same shot type, picks the closest camera angle', () => {
    const userFrames = [makeFrame(0, 0, makeSideViewPose())]
    const clips = [
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeSideViewPose())] },
    ]

    expect(selectBestProClip(userFrames, 'forehand', clips)).toBe(1)
  })

  it('returns 0 for empty clips array', () => {
    const userFrames = [makeFrame(0, 0, makeFrontViewPose())]
    expect(selectBestProClip(userFrames, 'forehand', [])).toBe(0)
  })

  it('returns the only clip index when a single clip is available', () => {
    const userFrames = [makeFrame(0, 0, makeFrontViewPose())]
    const clips = [
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeSideViewPose())] },
    ]
    expect(selectBestProClip(userFrames, 'forehand', clips)).toBe(0)
  })

  it('matches shot type case-insensitively', () => {
    const userFrames = [makeFrame(0, 0, makeFrontViewPose())]
    const clips = [
      { shotType: 'backhand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: 'Forehand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: 'serve', frames: [makeFrame(0, 0, makeFrontViewPose())] },
    ]
    expect(selectBestProClip(userFrames, 'FOREHAND', clips)).toBe(1)
  })

  it('trims whitespace when matching shot type', () => {
    const userFrames = [makeFrame(0, 0, makeFrontViewPose())]
    const clips = [
      { shotType: 'backhand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: '  forehand  ', frames: [makeFrame(0, 0, makeFrontViewPose())] },
    ]
    expect(selectBestProClip(userFrames, ' forehand ', clips)).toBe(1)
  })

  it('when user shot type is forehand and clips include forehand + backhand, picks forehand', () => {
    const userFrames = [makeFrame(0, 0, makeFrontViewPose())]
    const clips = [
      { shotType: 'backhand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
    ]
    expect(selectBestProClip(userFrames, 'forehand', clips)).toBe(1)
  })

  it('prefers shot type match over camera angle match', () => {
    // User is side-view, forehand
    const userFrames = [makeFrame(0, 0, makeSideViewPose())]
    const clips = [
      // backhand, side view (good angle, wrong shot)
      { shotType: 'backhand', frames: [makeFrame(0, 0, makeSideViewPose())] },
      // forehand, front view (right shot, worse angle)
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
    ]
    // Should pick forehand (index 1) even though backhand has a closer camera angle
    expect(selectBestProClip(userFrames, 'forehand', clips)).toBe(1)
  })

  it('selects front-view clip for front-view user when no shot type matches', () => {
    const userFrames = [makeFrame(0, 0, makeFrontViewPose())]
    const clips = [
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeSideViewPose())] },
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
    ]
    // 'volley' doesn't match, so falls back to camera angle
    const idx = selectBestProClip(userFrames, 'volley', clips)
    expect(idx).toBe(1)
  })

  it('uses camera angle tie-breaking across multiple forehand clips', () => {
    // User is side view
    const userFrames = [makeFrame(0, 0, makeSideViewPose())]

    // Build a "mid-angle" pose: shoulders moderately wide
    const makeMidViewPose = (): Landmark[] => [
      makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1),
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.58, 0.25),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.42, 0.25),
      makeLandmark(LANDMARK_INDICES.LEFT_ELBOW, 0.6, 0.4),
      makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.4, 0.4),
      makeLandmark(LANDMARK_INDICES.LEFT_WRIST, 0.6, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.4, 0.55),
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.53, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.47, 0.55),
      makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.53, 0.72),
      makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.47, 0.72),
      makeLandmark(LANDMARK_INDICES.LEFT_ANKLE, 0.53, 0.9),
      makeLandmark(LANDMARK_INDICES.RIGHT_ANKLE, 0.47, 0.9),
    ]

    const clips = [
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeMidViewPose())] },
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeSideViewPose())] },
    ]
    // Side-view user should pick the side-view forehand clip
    expect(selectBestProClip(userFrames, 'forehand', clips)).toBe(2)
  })

  it('handles multiple frames for camera angle estimation', () => {
    // Use multiple frames so estimateCameraAngle averages over samples
    const userFrames = Array.from({ length: 10 }, (_, i) =>
      makeFrame(i, i * 33, makeSideViewPose())
    )
    const clips = [
      {
        shotType: 'forehand',
        frames: Array.from({ length: 10 }, (_, i) =>
          makeFrame(i, i * 33, makeFrontViewPose())
        ),
      },
      {
        shotType: 'forehand',
        frames: Array.from({ length: 10 }, (_, i) =>
          makeFrame(i, i * 33, makeSideViewPose())
        ),
      },
    ]
    expect(selectBestProClip(userFrames, 'forehand', clips)).toBe(1)
  })

  it('handles frames with missing landmarks gracefully', () => {
    // A frame with no shoulder/hip landmarks: estimateCameraAngle should still work
    const sparseFrame = makeFrame(0, 0, [
      makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1),
      makeLandmark(LANDMARK_INDICES.LEFT_WRIST, 0.6, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.4, 0.55),
    ])
    const userFrames = [sparseFrame]
    const clips = [
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeFrontViewPose())] },
      { shotType: 'forehand', frames: [makeFrame(0, 0, makeSideViewPose())] },
    ]
    // Should not throw; the fallback angle is 0.5 when no landmarks are usable
    const idx = selectBestProClip(userFrames, 'forehand', clips)
    expect(idx).toBeGreaterThanOrEqual(0)
    expect(idx).toBeLessThan(clips.length)
  })
})


// ---------------------------------------------------------------------------
// Serve-specific phase detection (detectSwingPhases with shotType='serve')
// ---------------------------------------------------------------------------

/**
 * Build a synthetic serve frame sequence. Serve kinematics:
 *   - frames 0-2: stance (wrist low near hip, knees straight)
 *   - frames 3-6: knee bend ramps to trophy (knees bend to min at frame 6)
 *   - frames 6-9: wrist accelerates upward (peak -dy/dt around frame 8)
 *   - frame 9-10: contact (wrist at highest point)
 *   - frames 11-14: follow-through (wrist drops back)
 */
function makeServeFrames(): PoseFrame[] {
  const frames: PoseFrame[] = []
  for (let i = 0; i < 15; i++) {
    // Wrist y trajectory: 0.55 → 0.5 → 0.1 (up) → 0.3 (down)
    let wristY: number
    if (i < 3) wristY = 0.55
    else if (i < 6) wristY = 0.5 - (i - 3) * 0.02
    else if (i < 10) wristY = 0.44 - (i - 6) * 0.08  // fast upward to ~0.12
    else wristY = 0.12 + (i - 10) * 0.05 // drop back down

    // Right-knee angle trajectory: 175 → 175 → bends to 135 at trophy
    // (frame 6) → extends back to 165+ by contact
    let rKnee: number
    if (i < 3) rKnee = 175
    else if (i <= 6) rKnee = 175 - (i - 3) * (175 - 135) / 3
    else if (i < 10) rKnee = 135 + (i - 6) * ((165 - 135) / 4)
    else rKnee = 170

    // Right hand is dominant: right_wrist sits far from hip center while
    // left_wrist stays close, so detectDominantSide picks 'right'.
    const lms = [
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.45, 0.3),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.55, 0.3),
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.47, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.53, 0.55),
      makeLandmark(LANDMARK_INDICES.LEFT_WRIST, 0.5, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.82, wristY),
    ]
    const angles: JointAngles = {
      trunk_rotation: 8, // stable across frames (serve rotates trunk less dramatically)
      right_knee: rKnee,
      left_knee: rKnee, // mirror so detection is robust to which side is picked
      right_elbow: 150,
      right_shoulder: 90,
    }
    frames.push(makeFrame(i, i * 33, lms, angles))
  }
  return frames
}

describe('detectSwingPhases (serve branch)', () => {
  it('dispatches to serve heuristics when shotType="serve"', () => {
    const frames = makeServeFrames()
    const phases = detectSwingPhases(frames, 'serve')
    // Expect all 5 phases
    const phaseNames = phases.map((p) => p.phase)
    expect(phaseNames).toContain('preparation')
    expect(phaseNames).toContain('backswing')
    expect(phaseNames).toContain('contact')
    expect(phaseNames).toContain('follow_through')
  })

  it('places backswing (trophy) at the deepest knee-bend frame', () => {
    const frames = makeServeFrames()
    const phases = detectSwingPhases(frames, 'serve')
    const trophy = phases.find((p) => p.phase === 'backswing')
    expect(trophy).toBeDefined()
    // Synthetic trajectory bottoms out at frame 6
    expect(trophy!.frameIndex).toBe(6)
  })

  it('places contact at the highest-wrist frame (min y)', () => {
    const frames = makeServeFrames()
    const phases = detectSwingPhases(frames, 'serve')
    const contact = phases.find((p) => p.phase === 'contact')
    expect(contact).toBeDefined()
    // Wrist y reaches its minimum value (highest on screen, since y=0 is
    // top) at frame 10 in the synthetic trajectory; that's the contact
    // frame per the "peak height" heuristic.
    expect(contact!.frameIndex).toBe(10)
  })

  it('contact frame must come after backswing frame', () => {
    const frames = makeServeFrames()
    const phases = detectSwingPhases(frames, 'serve')
    const backswing = phases.find((p) => p.phase === 'backswing')!
    const contact = phases.find((p) => p.phase === 'contact')!
    expect(contact.frameIndex).toBeGreaterThan(backswing.frameIndex)
  })

  it('forehand detection path still runs when shotType is undefined', () => {
    // Back-compat: legacy callers don't pass a shotType and get the
    // groundstroke branch.
    const frames = Array.from({ length: 10 }, (_, i) =>
      makeFrame(
        i,
        i * 33,
        [
          makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.45, 0.3),
          makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.55, 0.3),
          makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.47, 0.55),
          makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.53, 0.55),
          makeLandmark(LANDMARK_INDICES.LEFT_WRIST, 0.3, 0.5),
          makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.7 - i * 0.04, 0.5),
        ],
        { trunk_rotation: 5 + i * 3, right_elbow: 120 - i * 4 },
      ),
    )
    // No shotType passed — should not crash and should return some phases.
    const phases = detectSwingPhases(frames)
    expect(Array.isArray(phases)).toBe(true)
  })
})
