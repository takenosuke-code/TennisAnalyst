import { describe, it, expect } from 'vitest'
import {
  getFrameAtTime,
  interpolateFrame,
} from '@/components/VideoCanvas'
import { makeLandmark, makeFrame } from '../helpers'
import type { PoseFrame } from '@/lib/supabase'

function frameAt(indexAndMs: number, x: number, angle: number): PoseFrame {
  return makeFrame(
    indexAndMs,
    indexAndMs,
    [makeLandmark(0, x, 0.5, 1.0, 0)],
    { right_elbow: angle },
  )
}

describe('getFrameAtTime', () => {
  it('returns null for an empty framesData array', () => {
    expect(getFrameAtTime([], 1.0)).toBeNull()
  })

  it('clamps to the first frame when time is before the first sample', () => {
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.2, 100)]
    // request 0.05s = 50ms, before first sample at 100ms
    expect(getFrameAtTime(frames, 0.05)).toBe(frames[0])
  })

  it('clamps to the last frame when time is past the last sample', () => {
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.2, 100)]
    expect(getFrameAtTime(frames, 5.0)).toBe(frames[1])
  })

  it('returns an interpolated frame at the midpoint of two samples', () => {
    // samples at 100ms and 200ms; request 150ms -> alpha = 0.5
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    const got = getFrameAtTime(frames, 0.15)
    expect(got).not.toBeNull()
    expect(got!.landmarks[0].x).toBeCloseTo(0.15, 6)
    expect(got!.joint_angles.right_elbow).toBeCloseTo(100, 6)
    // frame_index should track the "next" bracketing sample for tracer dedupe
    expect(got!.frame_index).toBe(200)
  })

  it('does not divide by zero when two samples share a timestamp', () => {
    const frames = [frameAt(100, 0.1, 90), frameAt(100, 0.2, 110)]
    // request exactly that timestamp; must not NaN
    const got = getFrameAtTime(frames, 0.1)
    expect(got).not.toBeNull()
    expect(Number.isFinite(got!.landmarks[0].x)).toBe(true)
  })
})

describe('interpolateFrame', () => {
  it('lerps landmark x/y/z/visibility and joint angles', () => {
    const a = makeFrame(0, 0, [makeLandmark(0, 0, 0, 0.5, 0)], {
      right_elbow: 90,
    })
    const b = makeFrame(
      1,
      100,
      [makeLandmark(0, 1, 1, 1.0, 2)],
      { right_elbow: 110 },
    )
    const got = interpolateFrame(a, b, 0.5)
    expect(got.landmarks[0].x).toBeCloseTo(0.5, 6)
    expect(got.landmarks[0].y).toBeCloseTo(0.5, 6)
    expect(got.landmarks[0].z).toBeCloseTo(1.0, 6)
    expect(got.landmarks[0].visibility).toBeCloseTo(0.75, 6)
    expect(got.joint_angles.right_elbow).toBeCloseTo(100, 6)
  })

  it('falls back to the defined side when one frame is missing a joint angle', () => {
    const a = makeFrame(0, 0, [makeLandmark(0, 0, 0)], { right_elbow: 90 })
    const b = makeFrame(1, 100, [makeLandmark(0, 1, 1)], {})
    const got = interpolateFrame(a, b, 0.5)
    expect(got.joint_angles.right_elbow).toBe(90)
  })

  it('lerps racket_head when both sides have a detection', () => {
    const a: PoseFrame = {
      ...makeFrame(0, 0, [makeLandmark(0, 0, 0)]),
      racket_head: { x: 0.0, y: 0.0, confidence: 0.5 },
    }
    const b: PoseFrame = {
      ...makeFrame(1, 100, [makeLandmark(0, 1, 1)]),
      racket_head: { x: 1.0, y: 0.5, confidence: 0.9 },
    }
    const got = interpolateFrame(a, b, 0.5)
    expect(got.racket_head).not.toBeNull()
    expect(got.racket_head!.x).toBeCloseTo(0.5, 6)
    expect(got.racket_head!.y).toBeCloseTo(0.25, 6)
    expect(got.racket_head!.confidence).toBeCloseTo(0.7, 6)
  })

  it('keeps the racket_head from the detected side when the other is null', () => {
    const a: PoseFrame = {
      ...makeFrame(0, 0, [makeLandmark(0, 0, 0)]),
      racket_head: { x: 0.3, y: 0.4, confidence: 0.8 },
    }
    const b: PoseFrame = {
      ...makeFrame(1, 100, [makeLandmark(0, 1, 1)]),
      racket_head: null,
    }
    const got = interpolateFrame(a, b, 0.5)
    expect(got.racket_head).toEqual({ x: 0.3, y: 0.4, confidence: 0.8 })
  })

  it('returns undefined racket_head when neither frame has the field (schema v1)', () => {
    const a = makeFrame(0, 0, [makeLandmark(0, 0, 0)])
    const b = makeFrame(1, 100, [makeLandmark(0, 1, 1)])
    const got = interpolateFrame(a, b, 0.5)
    expect(got.racket_head).toBeUndefined()
  })

  it('clamps alpha outside [0,1] to the boundary', () => {
    const a = makeFrame(0, 0, [makeLandmark(0, 0, 0)], { right_elbow: 90 })
    const b = makeFrame(1, 100, [makeLandmark(0, 1, 1)], { right_elbow: 110 })
    expect(interpolateFrame(a, b, -5).landmarks[0].x).toBeCloseTo(0, 6)
    expect(interpolateFrame(a, b, 5).landmarks[0].x).toBeCloseTo(1, 6)
  })
})
