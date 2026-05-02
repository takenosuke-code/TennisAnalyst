/**
 * Unit tests for the clip-quality gate. Each scenario constructs a
 * synthetic frame stream targeting one specific failure mode and
 * asserts the gate flags it with the right reason.
 */
import { describe, it, expect } from 'vitest'
import { gateClipQuality } from '@/lib/poseGate'
import type { Landmark, PoseFrame } from '@/lib/supabase'

const FILLED_IDS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

function makeLandmark(id: number, overrides: Partial<Landmark> = {}): Landmark {
  return {
    id,
    name: `landmark_${id}`,
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 1.0,
    ...overrides,
  }
}

function makeBodyVisibleLandmarks(visibility = 0.9): Landmark[] {
  // Hips, shoulders, wrists at the requested visibility, others at 0.
  return Array.from({ length: 33 }, (_, i) => {
    const isFilled = FILLED_IDS.includes(i)
    return makeLandmark(i, {
      visibility: isFilled ? visibility : 0,
      x: 0.5,
      y: 0.5,
    })
  })
}

function makeFrame(
  i: number,
  landmarks: Landmark[],
  hipRotation = 0,
): PoseFrame {
  return {
    frame_index: i,
    timestamp_ms: i * 33.33,
    landmarks,
    joint_angles: { hip_rotation: hipRotation },
  }
}

describe('gateClipQuality', () => {
  it('rejects too-short clips with too_few_frames', () => {
    const frames = Array.from({ length: 5 }, (_, i) =>
      makeFrame(i, makeBodyVisibleLandmarks()),
    )
    const result = gateClipQuality(frames)
    expect(result.passed).toBe(false)
    expect(result.reason).toBe('too_few_frames')
  })

  it('passes a clean stream', () => {
    const frames = Array.from({ length: 30 }, (_, i) =>
      makeFrame(i, makeBodyVisibleLandmarks(0.9), Math.sin(i / 10) * 30),
    )
    const result = gateClipQuality(frames)
    expect(
      result.passed,
      `expected pass, got ${result.reason} (${JSON.stringify(result.metrics)})`,
    ).toBe(true)
  })

  it('rejects with low_visibility when median vis is below floor', () => {
    const frames = Array.from({ length: 30 }, (_, i) =>
      makeFrame(i, makeBodyVisibleLandmarks(0.3)),
    )
    const result = gateClipQuality(frames)
    expect(result.passed).toBe(false)
    expect(result.reason).toBe('low_visibility')
    expect(result.metrics.medianVisibility).toBeLessThan(0.5)
  })

  it('rejects with body_not_visible when hips are missing on most frames', () => {
    const frames = Array.from({ length: 30 }, (_, i) => {
      // Visibility on shoulders/wrists is high, hips visibility is 0.
      const lms = Array.from({ length: 33 }, (_, id) => {
        const isHip = id === 23 || id === 24
        const isFilled = FILLED_IDS.includes(id)
        return makeLandmark(id, {
          visibility: isHip ? 0 : isFilled ? 0.9 : 0,
        })
      })
      return makeFrame(i, lms)
    })
    const result = gateClipQuality(frames)
    expect(result.passed).toBe(false)
    expect(result.reason).toBe('body_not_visible')
  })

  it('rejects with excessive_jitter when >25% of frames have >60° hip jumps', () => {
    // Alternate hip rotation between -90 and +90 so every consecutive
    // delta is 180° (well past the 60° threshold).
    const frames = Array.from({ length: 30 }, (_, i) =>
      makeFrame(
        i,
        makeBodyVisibleLandmarks(0.9),
        i % 2 === 0 ? -90 : 90,
      ),
    )
    const result = gateClipQuality(frames)
    expect(result.passed).toBe(false)
    expect(result.reason).toBe('excessive_jitter')
    expect(result.metrics.jumpFraction).toBeGreaterThan(0.25)
  })

  it('does NOT flag jitter on a slow continuous rotation that wraps ±180°', () => {
    // A real swing can wrap from +175° to -175° as the hip line crosses
    // horizontal — that's a 10° rotation in shortest-arc terms, not
    // 350°. The gate's shortAngleDelta must handle this without
    // counting wraps as jumps.
    const frames = Array.from({ length: 30 }, (_, i) => {
      const t = i / 29
      // sweep from -178 to +178 linearly (max true delta 356 over 29
      // frames = ~12°/frame, well under the 60° threshold)
      const hip = -178 + 356 * t
      return makeFrame(i, makeBodyVisibleLandmarks(0.9), hip)
    })
    const result = gateClipQuality(frames)
    expect(result.metrics.jumpFraction).toBeLessThanOrEqual(0.25)
  })

  it('reports metrics on every result, even when passed', () => {
    const frames = Array.from({ length: 30 }, (_, i) =>
      makeFrame(i, makeBodyVisibleLandmarks(0.9)),
    )
    const result = gateClipQuality(frames)
    expect(result.metrics.frameCount).toBe(30)
    expect(typeof result.metrics.medianVisibility).toBe('number')
    expect(typeof result.metrics.jumpFraction).toBe('number')
    expect(typeof result.metrics.bodyVisibleFraction).toBe('number')
  })

  it('returns jumpFraction = 0 when every frame is missing hip_rotation', () => {
    // No hip_rotation values anywhere → no consecutive deltas to count
    // → division-by-zero guard returns 0. The check then trivially
    // passes; that's intentional (we can't flag jitter we don't see),
    // but the test pins the behavior so a future refactor can't make
    // it return NaN or skew the result by including non-finite deltas.
    const frames: PoseFrame[] = Array.from({ length: 30 }, (_, i) => ({
      frame_index: i,
      timestamp_ms: i * 33.33,
      landmarks: makeBodyVisibleLandmarks(0.9),
      joint_angles: {},
    }))
    const result = gateClipQuality(frames)
    expect(result.metrics.jumpFraction).toBe(0)
    expect(result.passed).toBe(true)
  })

  it('passes synthetic fixtures with landmarks=[] (empty-landmark trust mode)', () => {
    // Hand-crafted joint_angles fixtures (analyze-stream tests) bypass
    // landmark data. Gate must not block them.
    const frames: PoseFrame[] = Array.from({ length: 60 }, (_, i) => ({
      frame_index: i,
      timestamp_ms: i * 33.33,
      landmarks: [],
      joint_angles: { hip_rotation: 30, trunk_rotation: 45 },
    }))
    const result = gateClipQuality(frames)
    expect(result.passed).toBe(true)
    expect(result.metrics.bodyVisibleFraction).toBe(1)
    expect(result.metrics.medianVisibility).toBe(1)
  })
})
