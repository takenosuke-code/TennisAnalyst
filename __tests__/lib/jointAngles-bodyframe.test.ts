import { describe, it, expect } from 'vitest'
import {
  computeBodyFrameAngles as bodyFrameViaJoint,
  computeRotationExcursion as excursionViaJoint,
  computeCameraSimilarity as similarityViaJoint,
} from '@/lib/jointAngles'
import {
  computeBodyFrameAngles as bodyFrameDirect,
  computeRotationExcursion as excursionDirect,
  computeCameraSimilarity as similarityDirect,
} from '@/lib/cameraNormalization'
import { makeFrame, makeStandingPose } from '../helpers'
import type { PoseFrame } from '@/lib/supabase'

// ---------------------------------------------------------------------------
// Re-export round-trip
// ---------------------------------------------------------------------------
//
// `lib/jointAngles.ts` re-exports the camera-robust helpers from
// `lib/cameraNormalization.ts` for ergonomic import. Because the two
// import paths must wire through to the *same function*, identity
// equality (toBe) is the strongest check we can run — it proves the
// re-export is a literal rebinding rather than a fresh wrapper that
// drifts independently.
// ---------------------------------------------------------------------------

describe('jointAngles bodyframe re-exports', () => {
  it('re-exports computeBodyFrameAngles as the same function reference', () => {
    expect(bodyFrameViaJoint).toBe(bodyFrameDirect)
  })

  it('re-exports computeRotationExcursion as the same function reference', () => {
    expect(excursionViaJoint).toBe(excursionDirect)
  })

  it('re-exports computeCameraSimilarity as the same function reference', () => {
    expect(similarityViaJoint).toBe(similarityDirect)
  })

  it('produces identical body-frame angles via either import path', () => {
    const landmarks = makeStandingPose()
    const a = bodyFrameViaJoint(landmarks)
    const b = bodyFrameDirect(landmarks)
    expect(a).toEqual(b)
  })

  it('produces identical excursion via either import path', () => {
    const frames: PoseFrame[] = [10, 25, 40, 55, 40, 25, 10].map((r, i) =>
      makeFrame(i, i * 33, [], { hip_rotation: r, trunk_rotation: r }),
    )
    const a = excursionViaJoint(frames)
    const b = excursionDirect(frames)
    expect(a).toEqual(b)
    expect(a.hipExcursion).toBeCloseTo(45, 5)
  })

  it('produces identical similarity scores via either import path', () => {
    const clip: PoseFrame[] = Array.from({ length: 4 }, (_, i) =>
      makeFrame(i, i * 33, makeStandingPose()),
    )
    const a = similarityViaJoint(clip, clip)
    const b = similarityDirect(clip, clip)
    expect(a).toEqual(b)
  })
})
