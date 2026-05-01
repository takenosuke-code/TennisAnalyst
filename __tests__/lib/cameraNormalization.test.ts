import { describe, it, expect } from 'vitest'
import {
  computeBodyFrameAngles,
  computeRotationExcursion,
  computeCameraSimilarity,
} from '@/lib/cameraNormalization'
import { LANDMARK_INDICES } from '@/lib/jointAngles'
import type { Landmark, PoseFrame, JointAngles } from '@/lib/supabase'
import { makeLandmark, makeFrame, makeStandingPose } from '../helpers'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Rotate every landmark's (x, y) around `center` by `angleDeg` degrees.
 * Used to verify body-frame angle invariance under in-plane camera roll.
 */
function rotatePose(
  landmarks: Landmark[],
  angleDeg: number,
  center: [number, number] = [0.5, 0.5],
): Landmark[] {
  const rad = (angleDeg * Math.PI) / 180
  const cos = Math.cos(rad)
  const sin = Math.sin(rad)
  const [cx, cy] = center
  return landmarks.map((lm) => {
    const dx = lm.x - cx
    const dy = lm.y - cy
    return {
      ...lm,
      x: cx + dx * cos - dy * sin,
      y: cy + dx * sin + dy * cos,
    }
  })
}

function makeFrameWithRotations(
  index: number,
  hipRot: number | null,
  trunkRot: number | null,
): PoseFrame {
  const angles: JointAngles = {}
  if (hipRot !== null) angles.hip_rotation = hipRot
  if (trunkRot !== null) angles.trunk_rotation = trunkRot
  return makeFrame(index, index * 33, makeStandingPose(), angles)
}

// ---------------------------------------------------------------------------
// computeBodyFrameAngles — invariance & projection
// ---------------------------------------------------------------------------

describe('computeBodyFrameAngles', () => {
  it('returns elbow / knee angles invariant under 30° in-plane rotation', () => {
    const base = makeStandingPose()
    const rotated = rotatePose(base, 30)

    const a = computeBodyFrameAngles(base)
    const b = computeBodyFrameAngles(rotated)

    // Sanity: the synthetic standing pose has all joints we expect.
    expect(a.right_elbow).toBeDefined()
    expect(a.left_elbow).toBeDefined()
    expect(a.right_knee).toBeDefined()
    expect(a.left_knee).toBeDefined()
    expect(a.right_shoulder).toBeDefined()
    expect(a.left_shoulder).toBeDefined()

    // Each pair must agree within 0.5°.
    const keys: (keyof JointAngles)[] = [
      'right_elbow',
      'left_elbow',
      'right_knee',
      'left_knee',
      'right_shoulder',
      'left_shoulder',
    ]
    for (const k of keys) {
      const va = a[k] as number
      const vb = b[k] as number
      expect(Math.abs(va - vb)).toBeLessThan(0.5)
    }
  })

  it('omits hip_rotation and trunk_rotation (body frame collapses them to ~0)', () => {
    const angles = computeBodyFrameAngles(makeStandingPose())
    expect(angles.hip_rotation).toBeUndefined()
    expect(angles.trunk_rotation).toBeUndefined()
  })

  it('returns empty object when both hips are missing', () => {
    const landmarks = makeStandingPose().filter(
      (l) =>
        l.id !== LANDMARK_INDICES.LEFT_HIP &&
        l.id !== LANDMARK_INDICES.RIGHT_HIP,
    )
    expect(computeBodyFrameAngles(landmarks)).toEqual({})
  })

  it('returns empty object when hips collapse on top of each other (front-on degenerate)', () => {
    // Both hips at the same (x, y) — hip-line length is zero, body
    // frame is undefined, so we refuse to invent a basis.
    const landmarks: Landmark[] = [
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.5, 0.25),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.5, 0.25),
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.5, 0.55),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.5, 0.55),
      makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.5, 0.72),
      makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.5, 0.72),
    ]
    expect(computeBodyFrameAngles(landmarks)).toEqual({})
  })

  it('returns empty object on empty landmarks', () => {
    expect(computeBodyFrameAngles([])).toEqual({})
  })
})

// ---------------------------------------------------------------------------
// computeRotationExcursion — values + camera-offset invariance
// ---------------------------------------------------------------------------

describe('computeRotationExcursion', () => {
  function buildSwing(
    rotations: number[],
    camOffsetDeg = 0,
  ): PoseFrame[] {
    return rotations.map((r, i) =>
      makeFrameWithRotations(i, r + camOffsetDeg, r + camOffsetDeg),
    )
  }

  it('returns 45° excursion for a 10° → 55° → 10° hip-rotation arc', () => {
    const frames = buildSwing([10, 25, 40, 55, 40, 25, 10])
    const out = computeRotationExcursion(frames)
    expect(out.hipExcursion).toBeCloseTo(45, 5)
    expect(out.trunkExcursion).toBeCloseTo(45, 5)
    expect(out.confidence).toBe(1)
  })

  it('is invariant to a uniform +20° camera offset across all frames', () => {
    const frames = buildSwing([10, 25, 40, 55, 40, 25, 10], 20)
    const out = computeRotationExcursion(frames)
    expect(out.hipExcursion).toBeCloseTo(45, 5)
    expect(out.trunkExcursion).toBeCloseTo(45, 5)
    expect(out.confidence).toBe(1)
  })

  it('confidence = 0 when fewer than half the frames have rotation values', () => {
    // 10 frames, only 4 with hip_rotation present.
    const frames: PoseFrame[] = []
    for (let i = 0; i < 10; i++) {
      const hasRot = i < 4
      frames.push(makeFrameWithRotations(i, hasRot ? 10 + i * 5 : null, hasRot ? 10 + i * 5 : null))
    }
    const out = computeRotationExcursion(frames)
    expect(out.confidence).toBe(0)
  })

  it('confidence ≥ 0.5 when at least half the frames carry valid rotation', () => {
    // 10 frames, 6 with rotation (60% valid).
    const frames: PoseFrame[] = []
    for (let i = 0; i < 10; i++) {
      const hasRot = i < 6
      frames.push(makeFrameWithRotations(i, hasRot ? 10 + i * 5 : null, hasRot ? 10 + i * 5 : null))
    }
    const out = computeRotationExcursion(frames)
    expect(out.confidence).toBeGreaterThanOrEqual(0.5)
    expect(out.confidence).toBeLessThanOrEqual(1)
  })

  it('returns zeros for empty frames input', () => {
    const out = computeRotationExcursion([])
    expect(out.hipExcursion).toBe(0)
    expect(out.trunkExcursion).toBe(0)
    expect(out.confidence).toBe(0)
  })
})

// ---------------------------------------------------------------------------
// computeCameraSimilarity — bucket adjacency + slope dampening
// ---------------------------------------------------------------------------

/**
 * Build a clip of N frames where each frame carries the given landmark
 * set unchanged. Useful for contrasting two camera classes against each
 * other.
 */
function clipFromLandmarks(landmarks: Landmark[], n = 10): PoseFrame[] {
  const out: PoseFrame[] = []
  for (let i = 0; i < n; i++) {
    out.push(makeFrame(i, i * 33, landmarks))
  }
  return out
}

/** Side-on landmarks: shoulders almost stacked on x, near shoulder fully
 * visible, far shoulder visibility low. Both hips low-spread too. */
function sideOnLandmarks(): Landmark[] {
  return [
    makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.5, 0.25, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.49, 0.25, 0.3),
    makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.5, 0.55, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.49, 0.55, 0.3),
    makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.5, 0.72, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.49, 0.72, 0.3),
    makeLandmark(LANDMARK_INDICES.LEFT_ANKLE, 0.5, 0.9, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_ANKLE, 0.49, 0.9, 0.3),
  ]
}

/** Front-on landmarks: wide shoulder spread, both hips visible, NOSE
 * visible. */
function frontOnLandmarks(): Landmark[] {
  return [
    makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.65, 0.25, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.35, 0.25, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.6, 0.55, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.4, 0.55, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.6, 0.72, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.4, 0.72, 0.95),
  ]
}

/** Behind-on landmarks: same wide shoulder spread but NOSE not visible
 * (back of head). */
function behindOnLandmarks(): Landmark[] {
  return [
    makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1, 0.05),
    makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.35, 0.25, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.65, 0.25, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.4, 0.55, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.6, 0.55, 0.95),
    makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.4, 0.72, 0.95),
    makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.6, 0.72, 0.95),
  ]
}

/** Mixed (oblique / 3-quarter) landmarks: medium shoulder spread, both
 * shoulders visible. */
function obliqueLandmarks(): Landmark[] {
  return [
    makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1, 0.9),
    makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.55, 0.25, 0.9),
    makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.45, 0.25, 0.7),
    makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.53, 0.55, 0.9),
    makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.47, 0.55, 0.7),
    makeLandmark(LANDMARK_INDICES.LEFT_KNEE, 0.54, 0.72, 0.9),
    makeLandmark(LANDMARK_INDICES.RIGHT_KNEE, 0.46, 0.72, 0.7),
  ]
}

describe('computeCameraSimilarity', () => {
  it('returns score = 1.0 for two identical clips', () => {
    const clip = clipFromLandmarks(sideOnLandmarks(), 8)
    const out = computeCameraSimilarity(clip, clip)
    expect(out.score).toBeCloseTo(1, 5)
    expect(out.classA).toBe(out.classB)
  })

  it('returns score < 0.7 for side-on vs 45°-behind', () => {
    const sideClip = clipFromLandmarks(sideOnLandmarks(), 8)
    // "45° behind" → oblique-ish, three-quarter view from the back.
    // Use the oblique landmark set (mixed bucket) to sit between
    // side-on and behind, which is what 45° behind looks like.
    const obliqueClip = clipFromLandmarks(obliqueLandmarks(), 8)
    const out = computeCameraSimilarity(sideClip, obliqueClip)
    expect(out.score).toBeLessThan(0.7)
  })

  it('returns score < 0.4 for side-on vs front-on (very different visibility patterns)', () => {
    const sideClip = clipFromLandmarks(sideOnLandmarks(), 8)
    const frontClip = clipFromLandmarks(frontOnLandmarks(), 8)
    const out = computeCameraSimilarity(sideClip, frontClip)
    expect(out.score).toBeLessThan(0.4)
    // And the two clips should bucket into recognizably different
    // classes — that's what makes the score drop.
    expect(out.classA).not.toBe(out.classB)
  })

  it('returns classA = classB = "side" for two side-on clips', () => {
    const a = clipFromLandmarks(sideOnLandmarks(), 5)
    const b = clipFromLandmarks(sideOnLandmarks(), 5)
    const out = computeCameraSimilarity(a, b)
    expect(out.classA).toBe('side')
    expect(out.classB).toBe('side')
  })

  it('distinguishes front-on from behind-on via nose visibility', () => {
    const front = clipFromLandmarks(frontOnLandmarks(), 5)
    const behind = clipFromLandmarks(behindOnLandmarks(), 5)
    const out = computeCameraSimilarity(front, behind)
    expect(out.classA).toBe('front')
    expect(out.classB).toBe('behind')
  })

  it('returns score = 0 when one clip has no usable shoulder signal', () => {
    const good = clipFromLandmarks(sideOnLandmarks(), 5)
    const empty: PoseFrame[] = [makeFrame(0, 0, [])]
    const out = computeCameraSimilarity(good, empty)
    expect(out.score).toBe(0)
  })
})
