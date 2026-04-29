import { describe, it, expect } from 'vitest'
import { inferTierFromFrames } from '@/lib/tierInference'
import type { PoseFrame, Landmark } from '@/lib/supabase'
import type { DominantHand } from '@/lib/profile'

/*
 * Phase 1.5 — Tier inference unit tests.
 *
 * `inferTierFromFrames` is a pure function over a sequence of pose frames.
 * Tests synthesize PoseFrame fixtures that exercise the three input
 * signals the inference cares about (trunk_rotation range, dominant-wrist
 * jerk, and arm-landmark visibility) and the early-return guards
 * (too-few-frames, low-visibility).
 */

// MediaPipe / BlazePose-33 ids that the tier inference reads.
const LANDMARK = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
} as const

function makeLandmark(
  id: number,
  x: number,
  y: number,
  visibility = 0.95,
): Landmark {
  return { id, name: `lm${id}`, x, y, z: 0, visibility }
}

interface FrameSpec {
  // Optional explicit positions for the dominant arm landmarks. Caller
  // provides them in normalized [0,1] image coordinates.
  rightWrist?: { x: number; y: number; visibility?: number }
  leftWrist?: { x: number; y: number; visibility?: number }
  rightElbow?: { x: number; y: number; visibility?: number }
  leftElbow?: { x: number; y: number; visibility?: number }
  rightShoulder?: { x: number; y: number; visibility?: number }
  leftShoulder?: { x: number; y: number; visibility?: number }
  trunkRotation?: number
}

function makeFrame(idx: number, spec: FrameSpec): PoseFrame {
  const lms: Landmark[] = []
  // Pad ids 0..16 so the tier inference's index-based lookups work even
  // for landmarks we don't care about. Visibility=0 so they don't skew
  // the visibility average; the inference only averages the load-bearing
  // arm joints anyway, but keeping the array dense matches real fixtures.
  for (let i = 0; i <= 16; i++) {
    lms.push(makeLandmark(i, 0.5, 0.5, 0))
  }

  const set = (
    landmarkId: number,
    pos?: { x: number; y: number; visibility?: number },
  ) => {
    if (!pos) return
    lms[landmarkId] = makeLandmark(landmarkId, pos.x, pos.y, pos.visibility ?? 0.95)
  }

  set(LANDMARK.RIGHT_WRIST, spec.rightWrist)
  set(LANDMARK.LEFT_WRIST, spec.leftWrist)
  set(LANDMARK.RIGHT_ELBOW, spec.rightElbow)
  set(LANDMARK.LEFT_ELBOW, spec.leftElbow)
  set(LANDMARK.RIGHT_SHOULDER, spec.rightShoulder)
  set(LANDMARK.LEFT_SHOULDER, spec.leftShoulder)

  return {
    frame_index: idx,
    timestamp_ms: idx * 33,
    landmarks: lms,
    joint_angles:
      typeof spec.trunkRotation === 'number'
        ? { trunk_rotation: spec.trunkRotation }
        : {},
  }
}

/**
 * Build N frames for a `hand`-handed swing with explicit `wristPath` (a
 * function index → {x, y}) and `trunkRotations` (one number per frame).
 * Defaults arm-joint visibility to 0.95 unless overridden.
 */
function buildFrames(opts: {
  count: number
  hand: DominantHand
  wristPath: (i: number) => { x: number; y: number }
  trunkRotations: number[]
  visibility?: number
}): PoseFrame[] {
  const frames: PoseFrame[] = []
  const armPos = (i: number) => ({
    x: 0.5,
    y: 0.5 + 0.001 * i, // tiny drift; load-bearing visibility average only
    visibility: opts.visibility ?? 0.95,
  })

  for (let i = 0; i < opts.count; i++) {
    const wrist = { ...opts.wristPath(i), visibility: opts.visibility ?? 0.95 }
    frames.push(
      makeFrame(i, {
        rightWrist: opts.hand === 'right' ? wrist : undefined,
        leftWrist: opts.hand === 'left' ? wrist : undefined,
        rightElbow: opts.hand === 'right' ? armPos(i) : undefined,
        leftElbow: opts.hand === 'left' ? armPos(i) : undefined,
        rightShoulder: opts.hand === 'right' ? armPos(i) : undefined,
        leftShoulder: opts.hand === 'left' ? armPos(i) : undefined,
        trunkRotation: opts.trunkRotations[i] ?? opts.trunkRotations[opts.trunkRotations.length - 1],
      }),
    )
  }
  return frames
}

// A "smooth" wrist path: a clean sinusoidal arc through normalized image
// space. Second-order finite differences on a sine of amplitude 0.1 over
// 16 frames give a per-frame jerk well under the competitive threshold
// (≈ 0.018).
const smoothWristPath = (i: number) => ({
  x: 0.4 + 0.1 * Math.sin((i / 16) * Math.PI),
  y: 0.5 + 0.05 * Math.cos((i / 16) * Math.PI),
})

// A "jerky" wrist path: alternates direction every frame with large
// amplitude, producing a per-frame jerk well above the beginner-min
// threshold (> 0.04).
const jerkyWristPath = (i: number) => ({
  x: 0.5 + (i % 2 === 0 ? 0.08 : -0.08),
  y: 0.5 + (i % 2 === 0 ? -0.06 : 0.06),
})

function constantTrunk(value: number, count = 16): number[] {
  return Array(count).fill(value)
}

// Linearly ramp trunk rotation from `from` → `to` so range = |to - from|.
function rampTrunk(from: number, to: number, count = 16): number[] {
  const out: number[] = []
  for (let i = 0; i < count; i++) {
    const t = count <= 1 ? 0 : i / (count - 1)
    out.push(from + (to - from) * t)
  }
  return out
}

describe('inferTierFromFrames', () => {
  it('returns null for an empty array', () => {
    expect(inferTierFromFrames([], 'right')).toBeNull()
  })

  it('returns null when fewer than 8 frames are provided', () => {
    const frames = buildFrames({
      count: 7,
      hand: 'right',
      wristPath: smoothWristPath,
      trunkRotations: rampTrunk(-30, 30, 7),
    })
    expect(inferTierFromFrames(frames, 'right')).toBeNull()
  })

  it('accepts exactly 8 frames as the minimum sample size', () => {
    const frames = buildFrames({
      count: 8,
      hand: 'right',
      wristPath: smoothWristPath,
      trunkRotations: rampTrunk(-30, 30, 8),
    })
    expect(inferTierFromFrames(frames, 'right')).not.toBeNull()
  })

  it('returns null when average arm visibility is below 0.5', () => {
    const frames = buildFrames({
      count: 16,
      hand: 'right',
      wristPath: smoothWristPath,
      trunkRotations: rampTrunk(-30, 30),
      visibility: 0.3,
    })
    expect(inferTierFromFrames(frames, 'right')).toBeNull()
  })

  it('returns "beginner" for low trunk-rotation + jerky wrist data', () => {
    const frames = buildFrames({
      count: 16,
      hand: 'right',
      wristPath: jerkyWristPath,
      trunkRotations: constantTrunk(5), // range = 0, well under 25°
    })
    const result = inferTierFromFrames(frames, 'right')
    expect(result).not.toBeNull()
    expect(result?.tier).toBe('beginner')
    expect(result?.reasons.some((r) => /under-rotated/i.test(r))).toBe(true)
    expect(result?.reasons.some((r) => /jerky/i.test(r))).toBe(true)
  })

  it('returns "competitive" for high trunk-rotation + smooth wrist data', () => {
    const frames = buildFrames({
      count: 16,
      hand: 'right',
      wristPath: smoothWristPath,
      trunkRotations: rampTrunk(-40, 40), // range = 80°, > 50° threshold
    })
    const result = inferTierFromFrames(frames, 'right')
    expect(result).not.toBeNull()
    expect(result?.tier).toBe('competitive')
    expect(result?.reasons.some((r) => /full trunk rotation/i.test(r))).toBe(true)
    expect(result?.reasons.some((r) => /smooth wrist/i.test(r))).toBe(true)
  })

  it('returns "intermediate" for the middle band', () => {
    // Mid-band trunk range (~35°) and mid-band wrist jerk: a path that
    // gives jerk between the competitive ceiling (0.018) and beginner
    // floor (0.04). A small-amplitude sine produces ~0.025 mean jerk.
    const midJerkPath = (i: number) => ({
      x: 0.5 + 0.04 * Math.sin((i / 4) * Math.PI),
      y: 0.5 + 0.02 * Math.cos((i / 4) * Math.PI),
    })
    const frames = buildFrames({
      count: 16,
      hand: 'right',
      wristPath: midJerkPath,
      trunkRotations: rampTrunk(-17.5, 17.5), // range = 35°
    })
    const result = inferTierFromFrames(frames, 'right')
    expect(result).not.toBeNull()
    expect(result?.tier).toBe('intermediate')
  })

  it('honors dominantHand="right": uses the right wrist landmark', () => {
    // Only the right wrist has data; left wrist is at default (zeroed).
    // If the function honors the hand argument it should produce a
    // confident intermediate/competitive read.
    const frames = buildFrames({
      count: 16,
      hand: 'right',
      wristPath: smoothWristPath,
      trunkRotations: rampTrunk(-40, 40),
    })
    const right = inferTierFromFrames(frames, 'right')
    expect(right).not.toBeNull()
    expect(right?.tier).toBe('competitive')
  })

  it('honors dominantHand="left": uses the left wrist landmark', () => {
    // Mirror of the previous test: data is on the LEFT side. If the
    // inference correctly indexes by `dominantHand`, it should return a
    // competitive read; if it incorrectly defaulted to the right wrist
    // it would see no data and either bail or come back as
    // intermediate-with-low-confidence.
    const frames = buildFrames({
      count: 16,
      hand: 'left',
      wristPath: smoothWristPath,
      trunkRotations: rampTrunk(-40, 40),
    })
    const left = inferTierFromFrames(frames, 'left')
    expect(left).not.toBeNull()
    expect(left?.tier).toBe('competitive')

    // Sanity: same data, but the caller asks about the right hand. Arm
    // visibility on the right is zero in this fixture (only left arm
    // landmarks were populated), so the inference must return null —
    // proving it actually swapped landmark ids based on the argument.
    const wrongHand = inferTierFromFrames(frames, 'right')
    expect(wrongHand).toBeNull()
  })

  it('produces a confidence score in [0, 1]', () => {
    const frames = buildFrames({
      count: 16,
      hand: 'right',
      wristPath: smoothWristPath,
      trunkRotations: rampTrunk(-40, 40),
    })
    const result = inferTierFromFrames(frames, 'right')
    expect(result).not.toBeNull()
    expect(result!.confidence).toBeGreaterThanOrEqual(0)
    expect(result!.confidence).toBeLessThanOrEqual(1)
  })

  it('confidence is higher when both signals agree (full kinetic chain) vs mixed signals', () => {
    const agreeingFrames = buildFrames({
      count: 16,
      hand: 'right',
      wristPath: smoothWristPath,
      trunkRotations: rampTrunk(-40, 40), // both signals point "competitive"
    })
    const mixedFrames = buildFrames({
      count: 16,
      hand: 'right',
      wristPath: jerkyWristPath, // wrist says "beginner"
      trunkRotations: rampTrunk(-40, 40), // trunk says "competitive"
    })
    const agreeing = inferTierFromFrames(agreeingFrames, 'right')!
    const mixed = inferTierFromFrames(mixedFrames, 'right')!
    expect(agreeing.confidence).toBeGreaterThan(mixed.confidence)
  })

  it('defaults to the right hand when the dominantHand argument is omitted', () => {
    // No second argument: the function's default should be "right". So a
    // right-handed fixture with smooth+wide swing should still come back
    // competitive without an explicit hand argument.
    const frames = buildFrames({
      count: 16,
      hand: 'right',
      wristPath: smoothWristPath,
      trunkRotations: rampTrunk(-40, 40),
    })
    const result = inferTierFromFrames(frames)
    expect(result).not.toBeNull()
    expect(result?.tier).toBe('competitive')
  })
})
