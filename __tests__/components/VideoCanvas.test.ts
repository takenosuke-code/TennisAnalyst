import { describe, it, expect } from 'vitest'
import { getFrameAtTime, getInterpolatedFrameAtTime } from '@/components/VideoCanvas'
import { makeLandmark, makeFrame, makeStandingPose } from '../helpers'
import { LANDMARK_INDICES } from '@/lib/jointAngles'

function frameAt(indexAndMs: number, x: number, angle: number) {
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
    expect(getFrameAtTime(frames, 0.05)).toBe(frames[0])
  })

  it('clamps to the last frame when time is past the last sample', () => {
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.2, 100)]
    expect(getFrameAtTime(frames, 5.0)).toBe(frames[1])
  })

  it('returns the nearer-in-time bracketing sample', () => {
    // Samples at 100ms and 200ms. Nearest-neighbor: 140ms is closer to
    // 100ms (40ms vs 60ms), 170ms is closer to 200ms (30ms vs 70ms).
    // Switching from strict-previous halves the worst-case visible drift
    // between the player and the skeleton overlay during fast motion.
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    expect(getFrameAtTime(frames, 0.14)).toBe(frames[0])
    expect(getFrameAtTime(frames, 0.17)).toBe(frames[1])
    // 200ms exactly resolves to the frame at 200 (the post-loop clamp).
    expect(getFrameAtTime(frames, 0.2)).toBe(frames[1])
  })

  it('breaks ties by returning the earlier sample', () => {
    // 150ms is exactly between 100ms and 200ms (50ms each way). Tie
    // broken in favor of the earlier sample (dPrev <= dNext).
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    expect(getFrameAtTime(frames, 0.15)).toBe(frames[0])
  })

  it('returns the frame whose timestamp exactly matches', () => {
    const frames = [frameAt(100, 0.10, 90), frameAt(200, 0.20, 110)]
    // 200ms exactly -> the second frame via the post-loop clamp
    expect(getFrameAtTime(frames, 0.2)).toBe(frames[1])
  })

  it('does not synthesize a position between samples', () => {
    // Regression guard: even with nearest-neighbor we never interpolate
    // — the returned frame's landmarks must equal one of the two
    // bracketing samples exactly. (Synthesizing positions reintroduces
    // the predictive-racket bug class from earlier.)
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.9, 110)]
    const got = getFrameAtTime(frames, 0.15)
    expect(got).not.toBeNull()
    const x = got!.landmarks[0].x
    expect([0.1, 0.9]).toContain(x)
  })
})

// Phase B (live review only): synthesize landmark positions between
// bracketing 8fps server samples so the overlay flows on a 30fps video
// instead of snapping every ~125ms. /analyze keeps using getFrameAtTime.
describe('getInterpolatedFrameAtTime', () => {
  it('returns null for an empty framesData array', () => {
    expect(getInterpolatedFrameAtTime([], 1.0)).toBeNull()
  })

  it('clamps to the first frame when time is before the first sample', () => {
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.2, 100)]
    expect(getInterpolatedFrameAtTime(frames, 0.05)).toBe(frames[0])
  })

  it('clamps to the last frame when time is past the last sample', () => {
    const frames = [frameAt(100, 0.1, 90), frameAt(200, 0.2, 100)]
    expect(getInterpolatedFrameAtTime(frames, 5.0)).toBe(frames[1])
  })

  it('linear-interpolates landmark x/y/z between bracketing samples', () => {
    // Samples at 100ms (x=0.10) and 200ms (x=0.20). Lookup at 150ms is
    // exactly midway → x should be 0.15. Nearest-neighbor would have
    // returned one of {0.10, 0.20}.
    const a = makeFrame(0, 100, [makeLandmark(0, 0.10, 0.40, 1.0, 0.5)])
    const b = makeFrame(1, 200, [makeLandmark(0, 0.20, 0.60, 1.0, 0.7)])
    const got = getInterpolatedFrameAtTime([a, b], 0.15)
    expect(got).not.toBeNull()
    expect(got!.landmarks[0].x).toBeCloseTo(0.15, 6)
    expect(got!.landmarks[0].y).toBeCloseTo(0.50, 6)
    expect(got!.landmarks[0].z).toBeCloseTo(0.60, 6)
  })

  it('carries visibility from the nearer frame rather than averaging', () => {
    // visibility is a confidence signal, not a position — averaging two
    // confidences would smear them. Lookup at 130ms is closer to 100ms
    // (30ms vs 70ms), so visibility comes from the first frame.
    const a = makeFrame(0, 100, [makeLandmark(0, 0.10, 0.5, 0.20)])
    const b = makeFrame(1, 200, [makeLandmark(0, 0.20, 0.5, 0.90)])
    const earlier = getInterpolatedFrameAtTime([a, b], 0.13)
    expect(earlier!.landmarks[0].visibility).toBe(0.20)
    const later = getInterpolatedFrameAtTime([a, b], 0.17)
    expect(later!.landmarks[0].visibility).toBe(0.90)
  })

  it('tags the returned frame timestamp with the requested time', () => {
    // Downstream renderers compare timestamps, not frame_index, when
    // locating the current overlay against video.currentTime. A returned
    // frame must report the time the consumer asked for.
    const a = makeFrame(0, 100, [makeLandmark(0, 0.1, 0.5, 1.0)])
    const b = makeFrame(1, 200, [makeLandmark(0, 0.2, 0.5, 1.0)])
    const got = getInterpolatedFrameAtTime([a, b], 0.142)
    expect(got!.timestamp_ms).toBe(142)
  })

  it('recomputes joint_angles from the interpolated landmarks', () => {
    // Pre-set joint_angles on both bracketing frames to bogus sentinels.
    // The interpolated frame must NOT carry those forward unchanged —
    // it must geometrically recompute angles from the lerped landmarks.
    // (Direct numeric interpolation of angles mishandles 0/360 wrap.)
    const lmA = makeStandingPose()
    const lmB = makeStandingPose().map((l) => ({ ...l }))
    // Move the right elbow halfway along x so the interpolated pose is
    // a real geometric mid-point.
    lmB.find((l) => l.id === LANDMARK_INDICES.RIGHT_ELBOW)!.x = 0.6
    const a = makeFrame(0, 100, lmA, { right_elbow: 999 })
    const b = makeFrame(1, 200, lmB, { right_elbow: 999 })
    const mid = getInterpolatedFrameAtTime([a, b], 0.15)!
    expect(mid.joint_angles.right_elbow).not.toBe(999)
    expect(typeof mid.joint_angles.right_elbow).toBe('number')
  })

  it('falls back gracefully on degenerate identical timestamps', () => {
    // Two samples with the same timestamp would zero the denominator.
    // The clamp branches above handle the typical at-or-past cases;
    // the in-loop branch needs to not return NaN if a malformed input
    // arrives. We accept either bracketing sample — what matters is no
    // NaN propagates to the renderer.
    const a = makeFrame(0, 100, [makeLandmark(0, 0.1, 0.5, 1.0)])
    const b = makeFrame(1, 100, [makeLandmark(0, 0.9, 0.5, 1.0)])
    const got = getInterpolatedFrameAtTime([a, b], 0.1)
    expect(got).not.toBeNull()
    expect(Number.isFinite(got!.landmarks[0].x)).toBe(true)
  })
})
