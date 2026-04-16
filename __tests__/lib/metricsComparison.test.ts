import { describe, it, expect } from 'vitest'
import {
  classifyDifference,
  compareAngles,
  compareFrameAngles,
  findMostImprovedBodyPart,
  type DiffSeverity,
} from '@/lib/metricsComparison'
import type { JointAngles, PoseFrame } from '@/lib/supabase'
import { makeFrame, makeStandingPose } from '../helpers'

// ---------------------------------------------------------------------------
// classifyDifference
// ---------------------------------------------------------------------------

describe('classifyDifference', () => {
  it('returns green for differences < 15 degrees', () => {
    expect(classifyDifference(0)).toBe('green')
    expect(classifyDifference(5)).toBe('green')
    expect(classifyDifference(14)).toBe('green')
    expect(classifyDifference(14.9)).toBe('green')
  })

  it('returns yellow for differences 15-30 degrees', () => {
    expect(classifyDifference(15)).toBe('yellow')
    expect(classifyDifference(20)).toBe('yellow')
    expect(classifyDifference(30)).toBe('yellow')
  })

  it('returns red for differences > 30 degrees', () => {
    expect(classifyDifference(31)).toBe('red')
    expect(classifyDifference(45)).toBe('red')
    expect(classifyDifference(90)).toBe('red')
  })

  it('handles negative values by taking absolute value', () => {
    expect(classifyDifference(-5)).toBe('green')
    expect(classifyDifference(-25)).toBe('yellow')
    expect(classifyDifference(-40)).toBe('red')
  })

  it('handles zero exactly', () => {
    expect(classifyDifference(0)).toBe('green')
  })

  it('handles boundary values at 15 and 30', () => {
    // < 15 is green, exactly 15 is yellow
    expect(classifyDifference(14.999)).toBe('green')
    expect(classifyDifference(15)).toBe('yellow')
    // <= 30 is yellow, > 30 is red
    expect(classifyDifference(30)).toBe('yellow')
    expect(classifyDifference(30.001)).toBe('red')
  })
})

// ---------------------------------------------------------------------------
// compareAngles
// ---------------------------------------------------------------------------

describe('compareAngles', () => {
  it('compares all shared angle keys between user and pro', () => {
    const userAngles: JointAngles = {
      right_elbow: 120,
      left_elbow: 130,
      right_shoulder: 90,
      hip_rotation: 45,
    }
    const proAngles: JointAngles = {
      right_elbow: 110,
      left_elbow: 140,
      right_shoulder: 80,
      hip_rotation: 60,
    }

    const result = compareAngles(userAngles, proAngles)
    expect(result.diffs).toHaveLength(4)

    const elbowDiff = result.diffs.find((d) => d.angleKey === 'right_elbow')
    expect(elbowDiff).toBeDefined()
    expect(elbowDiff!.difference).toBe(10)
    expect(elbowDiff!.severity).toBe('green')
  })

  it('applies correct severity color coding', () => {
    const userAngles: JointAngles = {
      right_elbow: 120,    // diff = 5 => green
      left_elbow: 130,     // diff = 20 => yellow
      right_shoulder: 90,  // diff = 40 => red
    }
    const proAngles: JointAngles = {
      right_elbow: 115,
      left_elbow: 150,
      right_shoulder: 130,
    }

    const result = compareAngles(userAngles, proAngles)

    const findSeverity = (key: keyof JointAngles) =>
      result.diffs.find((d) => d.angleKey === key)?.severity

    expect(findSeverity('right_elbow')).toBe('green')
    expect(findSeverity('left_elbow')).toBe('yellow')
    expect(findSeverity('right_shoulder')).toBe('red')
  })

  it('skips angles missing in user data', () => {
    const userAngles: JointAngles = { right_elbow: 120 }
    const proAngles: JointAngles = {
      right_elbow: 110,
      left_elbow: 140,
      right_shoulder: 80,
    }

    const result = compareAngles(userAngles, proAngles)
    expect(result.diffs).toHaveLength(1)
    expect(result.diffs[0].angleKey).toBe('right_elbow')
  })

  it('skips angles missing in pro data', () => {
    const userAngles: JointAngles = {
      right_elbow: 120,
      left_elbow: 130,
    }
    const proAngles: JointAngles = { right_elbow: 110 }

    const result = compareAngles(userAngles, proAngles)
    expect(result.diffs).toHaveLength(1)
    expect(result.diffs[0].angleKey).toBe('right_elbow')
  })

  it('handles both sets empty gracefully', () => {
    const result = compareAngles({}, {})
    expect(result.diffs).toHaveLength(0)
    expect(result.worstBodyPart).toBeNull()
    expect(result.maxDifference).toBe(0)
    expect(result.averageDifference).toBe(0)
  })

  it('identifies the worst body part', () => {
    const userAngles: JointAngles = {
      right_elbow: 120,    // diff = 10
      hip_rotation: 20,    // diff = 50
      trunk_rotation: 30,  // diff = 25
    }
    const proAngles: JointAngles = {
      right_elbow: 110,
      hip_rotation: 70,
      trunk_rotation: 55,
    }

    const result = compareAngles(userAngles, proAngles)
    expect(result.worstBodyPart).toBe('hip_rotation')
    expect(result.maxDifference).toBe(50)
  })

  it('computes correct average difference', () => {
    const userAngles: JointAngles = {
      right_elbow: 120,   // diff = 10
      left_elbow: 130,    // diff = 20
      right_knee: 170,    // diff = 30
    }
    const proAngles: JointAngles = {
      right_elbow: 110,
      left_elbow: 150,
      right_knee: 140,
    }

    const result = compareAngles(userAngles, proAngles)
    expect(result.averageDifference).toBe(20)
  })

  it('stores correct user and pro values in diffs', () => {
    const userAngles: JointAngles = { right_elbow: 125 }
    const proAngles: JointAngles = { right_elbow: 110 }

    const result = compareAngles(userAngles, proAngles)
    expect(result.diffs[0].userValue).toBe(125)
    expect(result.diffs[0].proValue).toBe(110)
    expect(result.diffs[0].difference).toBe(15)
  })
})

// ---------------------------------------------------------------------------
// compareFrameAngles
// ---------------------------------------------------------------------------

describe('compareFrameAngles', () => {
  it('returns empty result for empty user frames', () => {
    const proFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 120 }),
    ]
    const result = compareFrameAngles([], proFrames, 0)
    expect(result.diffs).toHaveLength(0)
    expect(result.worstBodyPart).toBeNull()
  })

  it('returns empty result for empty pro frames', () => {
    const userFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 120 }),
    ]
    const result = compareFrameAngles(userFrames, [], 0)
    expect(result.diffs).toHaveLength(0)
    expect(result.worstBodyPart).toBeNull()
  })

  it('compares angles at the specified frame index', () => {
    const userFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 100 }),
      makeFrame(1, 33, makeStandingPose(), { right_elbow: 120 }),
      makeFrame(2, 66, makeStandingPose(), { right_elbow: 140 }),
    ]
    const proFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 110 }),
      makeFrame(1, 33, makeStandingPose(), { right_elbow: 115 }),
      makeFrame(2, 66, makeStandingPose(), { right_elbow: 130 }),
    ]

    const result = compareFrameAngles(userFrames, proFrames, 1)
    const elbowDiff = result.diffs.find((d) => d.angleKey === 'right_elbow')
    expect(elbowDiff).toBeDefined()
    expect(elbowDiff!.difference).toBe(5) // |120 - 115|
  })

  it('clamps frame index to array bounds', () => {
    const userFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 120 }),
    ]
    const proFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 130 }),
    ]

    // frameIndex=10 but arrays have length 1 => should clamp to index 0
    const result = compareFrameAngles(userFrames, proFrames, 10)
    expect(result.diffs.length).toBeGreaterThan(0)
    const elbowDiff = result.diffs.find((d) => d.angleKey === 'right_elbow')
    expect(elbowDiff!.difference).toBe(10)
  })
})

// ---------------------------------------------------------------------------
// findMostImprovedBodyPart
// ---------------------------------------------------------------------------

describe('findMostImprovedBodyPart', () => {
  it('returns null body part for empty user frames', () => {
    const proFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 120 }),
    ]
    const result = findMostImprovedBodyPart([], proFrames)
    expect(result.bodyPart).toBeNull()
    expect(result.avgDifference).toBe(0)
  })

  it('returns null body part for empty pro frames', () => {
    const userFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 120 }),
    ]
    const result = findMostImprovedBodyPart(userFrames, [])
    expect(result.bodyPart).toBeNull()
    expect(result.avgDifference).toBe(0)
  })

  it('identifies the body part with the highest average difference', () => {
    // right_elbow: avg diff = 10
    // hip_rotation: avg diff = 40
    const userFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 120, hip_rotation: 20 }),
      makeFrame(1, 33, makeStandingPose(), { right_elbow: 115, hip_rotation: 25 }),
    ]
    const proFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 110, hip_rotation: 60 }),
      makeFrame(1, 33, makeStandingPose(), { right_elbow: 110, hip_rotation: 65 }),
    ]

    const result = findMostImprovedBodyPart(userFrames, proFrames)
    expect(result.bodyPart).toBe('hip_rotation')
    expect(result.avgDifference).toBe(40)
  })

  it('handles mismatched frame array lengths by using the minimum', () => {
    const userFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 120 }),
      makeFrame(1, 33, makeStandingPose(), { right_elbow: 130 }),
      makeFrame(2, 66, makeStandingPose(), { right_elbow: 140 }),
    ]
    const proFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 110 }),
    ]

    // Only 1 comparison should be made (min of 3, 1)
    const result = findMostImprovedBodyPart(userFrames, proFrames)
    expect(result.bodyPart).toBe('right_elbow')
    expect(result.avgDifference).toBe(10)
  })

  it('handles frames with no shared angle keys', () => {
    const userFrames = [
      makeFrame(0, 0, makeStandingPose(), { right_elbow: 120 }),
    ]
    const proFrames = [
      makeFrame(0, 0, makeStandingPose(), { left_elbow: 130 }),
    ]

    const result = findMostImprovedBodyPart(userFrames, proFrames)
    expect(result.bodyPart).toBeNull()
    expect(result.avgDifference).toBe(0)
  })
})
