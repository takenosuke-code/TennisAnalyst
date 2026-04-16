import { describe, it, expect } from 'vitest'
import {
  normalizeLandmarks,
  computeLandmarkBounds,
} from '@/components/PoseRenderer'
import { LANDMARK_INDICES } from '@/lib/jointAngles'
import { makeLandmark, makeFrame, makeStandingPose } from '../helpers'
import type { PoseFrame } from '@/lib/supabase'

// ---------------------------------------------------------------------------
// normalizeLandmarks
// ---------------------------------------------------------------------------

describe('normalizeLandmarks', () => {
  it('centers landmarks around 0.5 based on hip midpoint', () => {
    const landmarks = makeStandingPose()
    const frame = makeFrame(0, 0, landmarks)
    const result = normalizeLandmarks(frame)

    // The hip midpoint should be at (0.5, 0.5) after normalization
    const lHip = result.landmarks.find(
      (l) => l.id === LANDMARK_INDICES.LEFT_HIP
    )!
    const rHip = result.landmarks.find(
      (l) => l.id === LANDMARK_INDICES.RIGHT_HIP
    )!
    const hipMidX = (lHip.x + rHip.x) / 2
    const hipMidY = (lHip.y + rHip.y) / 2

    expect(hipMidX).toBeCloseTo(0.5, 5)
    expect(hipMidY).toBeCloseTo(0.5, 5)
  })

  it('returns frame unchanged when hips are missing', () => {
    const landmarks = makeStandingPose().filter(
      (l) =>
        l.id !== LANDMARK_INDICES.LEFT_HIP &&
        l.id !== LANDMARK_INDICES.RIGHT_HIP
    )
    const frame = makeFrame(0, 0, landmarks)
    const result = normalizeLandmarks(frame)

    // Should return the exact same frame object
    expect(result).toBe(frame)
  })

  it('returns frame unchanged when only one hip is present', () => {
    const landmarks = makeStandingPose().filter(
      (l) => l.id !== LANDMARK_INDICES.LEFT_HIP
    )
    const frame = makeFrame(0, 0, landmarks)
    const result = normalizeLandmarks(frame)

    expect(result).toBe(frame)
  })

  it('returns frame unchanged when hips are at the same position (degenerate)', () => {
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.5, 0.5),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.5, 0.5),
      makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.1),
    ]
    const frame = makeFrame(0, 0, landmarks)
    const result = normalizeLandmarks(frame)

    // hipWidth < 0.01 guard should trigger
    expect(result).toBe(frame)
  })

  it('maintains relative positions between landmarks', () => {
    const landmarks = makeStandingPose()
    const frame = makeFrame(0, 0, landmarks)

    // Find the original relative positions before normalization
    const origNose = landmarks.find(
      (l) => l.id === LANDMARK_INDICES.NOSE
    )!
    const origLShoulder = landmarks.find(
      (l) => l.id === LANDMARK_INDICES.LEFT_SHOULDER
    )!
    const origDx = origNose.x - origLShoulder.x
    const origDy = origNose.y - origLShoulder.y

    const result = normalizeLandmarks(frame)

    const normNose = result.landmarks.find(
      (l) => l.id === LANDMARK_INDICES.NOSE
    )!
    const normLShoulder = result.landmarks.find(
      (l) => l.id === LANDMARK_INDICES.LEFT_SHOULDER
    )!

    // After normalization, the direction between landmarks should be preserved
    const normDx = normNose.x - normLShoulder.x
    const normDy = normNose.y - normLShoulder.y

    // The ratio of displacements should be the same (both scaled by the same factor)
    if (origDx !== 0) {
      expect(normDy / normDx).toBeCloseTo(origDy / origDx, 5)
    }
  })

  it('scales coordinates by inverse hip width', () => {
    // lHip at (0.4, 0.5), rHip at (0.6, 0.5) => hipWidth = 0.2
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.4, 0.5),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.6, 0.5),
      makeLandmark(LANDMARK_INDICES.NOSE, 0.5, 0.3),
    ]
    const frame = makeFrame(0, 0, landmarks)
    const result = normalizeLandmarks(frame)

    // hipMidX=0.5, hipMidY=0.5, hipWidth=0.2, scale=1/0.2=5
    // nose: (0.5-0.5)*5 + 0.5 = 0.5, (0.3-0.5)*5 + 0.5 = -0.5
    const nose = result.landmarks.find(
      (l) => l.id === LANDMARK_INDICES.NOSE
    )!
    expect(nose.x).toBeCloseTo(0.5, 5)
    expect(nose.y).toBeCloseTo(-0.5, 5)
  })
})

// ---------------------------------------------------------------------------
// computeLandmarkBounds
// ---------------------------------------------------------------------------

describe('computeLandmarkBounds', () => {
  it('returns correct min/max with padding for known positions', () => {
    const landmarks = [
      makeLandmark(0, 0.2, 0.3, 1.0),
      makeLandmark(1, 0.8, 0.9, 1.0),
    ]
    const frame = makeFrame(0, 0, landmarks)
    const bounds = computeLandmarkBounds(frame, 0.05)

    // Raw: minX=0.2, maxX=0.8, minY=0.3, maxY=0.9
    // w=0.6, h=0.6
    // padded: minX=0.2-0.03=0.17, maxX=0.8+0.03=0.83
    //         minY=0.3-0.03=0.27, maxY=0.9+0.03=0.93
    expect(bounds).not.toBeNull()
    expect(bounds!.minX).toBeCloseTo(0.17, 2)
    expect(bounds!.maxX).toBeCloseTo(0.83, 2)
    expect(bounds!.minY).toBeCloseTo(0.27, 2)
    expect(bounds!.maxY).toBeCloseTo(0.93, 2)
  })

  it('returns null when all landmarks are invisible', () => {
    const landmarks = [
      makeLandmark(0, 0.5, 0.5, 0.1),
      makeLandmark(1, 0.6, 0.6, 0.2),
    ]
    const frame = makeFrame(0, 0, landmarks)
    const bounds = computeLandmarkBounds(frame)

    expect(bounds).toBeNull()
  })

  it('ignores landmarks with visibility < 0.3', () => {
    const landmarks = [
      makeLandmark(0, 0.1, 0.1, 0.2), // invisible - should be ignored
      makeLandmark(1, 0.4, 0.4, 1.0),
      makeLandmark(2, 0.6, 0.6, 1.0),
    ]
    const frame = makeFrame(0, 0, landmarks)
    const bounds = computeLandmarkBounds(frame, 0)

    // Only landmarks 1 and 2 are visible
    expect(bounds).not.toBeNull()
    expect(bounds!.minX).toBeCloseTo(0.4, 5)
    expect(bounds!.maxX).toBeCloseTo(0.6, 5)
    expect(bounds!.minY).toBeCloseTo(0.4, 5)
    expect(bounds!.maxY).toBeCloseTo(0.6, 5)
  })

  it('clamps bounds to [0, 1] after padding', () => {
    const landmarks = [
      makeLandmark(0, 0.0, 0.0, 1.0),
      makeLandmark(1, 1.0, 1.0, 1.0),
    ]
    const frame = makeFrame(0, 0, landmarks)
    const bounds = computeLandmarkBounds(frame, 0.1)

    expect(bounds).not.toBeNull()
    // Padding would push beyond 0 and 1, but should be clamped
    expect(bounds!.minX).toBe(0)
    expect(bounds!.minY).toBe(0)
    expect(bounds!.maxX).toBe(1)
    expect(bounds!.maxY).toBe(1)
  })

  it('works with a single visible landmark', () => {
    const landmarks = [makeLandmark(0, 0.5, 0.5, 1.0)]
    const frame = makeFrame(0, 0, landmarks)
    const bounds = computeLandmarkBounds(frame, 0.05)

    // w=0, h=0, so padding = 0 * 0.05 = 0
    expect(bounds).not.toBeNull()
    expect(bounds!.minX).toBeCloseTo(0.5, 5)
    expect(bounds!.maxX).toBeCloseTo(0.5, 5)
    expect(bounds!.minY).toBeCloseTo(0.5, 5)
    expect(bounds!.maxY).toBeCloseTo(0.5, 5)
  })

  it('uses default padding of 0.05', () => {
    const landmarks = [
      makeLandmark(0, 0.3, 0.3, 1.0),
      makeLandmark(1, 0.7, 0.7, 1.0),
    ]
    const frame = makeFrame(0, 0, landmarks)
    const bounds = computeLandmarkBounds(frame)

    // w=0.4, h=0.4, padding = 0.4*0.05 = 0.02
    expect(bounds).not.toBeNull()
    expect(bounds!.minX).toBeCloseTo(0.28, 2)
    expect(bounds!.maxX).toBeCloseTo(0.72, 2)
    expect(bounds!.minY).toBeCloseTo(0.28, 2)
    expect(bounds!.maxY).toBeCloseTo(0.72, 2)
  })
})
