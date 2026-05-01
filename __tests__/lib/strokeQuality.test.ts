import { describe, it, expect } from 'vitest'
import { scoreStrokes } from '@/lib/strokeQuality'
import type {
  DetectedStroke,
  StrokeQualityResult,
} from '@/lib/strokeAnalysis'
import type { PoseFrame, JointAngles, Landmark } from '@/lib/supabase'
import { LANDMARK_INDICES } from '@/lib/jointAngles'
import { makeFrame, makeLandmark } from '../helpers'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a frame with full upper-body landmark coverage at controllable
 * locations. The right-wrist position can be overridden frame-by-frame to
 * simulate different swing speeds.
 */
function makeQualityFrame(
  index: number,
  timestampMs: number,
  opts: {
    rightWristX?: number
    rightWristY?: number
    centroidShift?: number // shift entire upper body horizontally
    scale?: number // scale upper-body bbox (zoom)
    visibility?: number
    angles?: JointAngles
  } = {},
): PoseFrame {
  const {
    rightWristX = 0.4,
    rightWristY = 0.55,
    centroidShift = 0,
    scale = 1.0,
    visibility = 1.0,
    angles = {},
  } = opts

  // Anchor the upper body centered on (0.5, 0.4).
  const cx = 0.5 + centroidShift
  const cy = 0.4
  const lms: Landmark[] = [
    makeLandmark(
      LANDMARK_INDICES.LEFT_SHOULDER,
      cx + 0.08 * scale,
      cy - 0.15 * scale,
      visibility,
    ),
    makeLandmark(
      LANDMARK_INDICES.RIGHT_SHOULDER,
      cx - 0.08 * scale,
      cy - 0.15 * scale,
      visibility,
    ),
    makeLandmark(
      LANDMARK_INDICES.LEFT_ELBOW,
      cx + 0.12 * scale,
      cy,
      visibility,
    ),
    makeLandmark(
      LANDMARK_INDICES.RIGHT_ELBOW,
      cx - 0.12 * scale,
      cy,
      visibility,
    ),
    makeLandmark(
      LANDMARK_INDICES.LEFT_WRIST,
      cx + 0.18 * scale,
      cy + 0.15 * scale,
      visibility,
    ),
    // Right wrist is the dominant wrist; allow override.
    makeLandmark(
      LANDMARK_INDICES.RIGHT_WRIST,
      rightWristX,
      rightWristY,
      visibility,
    ),
    makeLandmark(
      LANDMARK_INDICES.LEFT_HIP,
      cx + 0.06 * scale,
      cy + 0.25 * scale,
      visibility,
    ),
    makeLandmark(
      LANDMARK_INDICES.RIGHT_HIP,
      cx - 0.06 * scale,
      cy + 0.25 * scale,
      visibility,
    ),
  ]
  return makeFrame(index, timestampMs, lms, angles)
}

/**
 * Build a clean stroke window of N frames. Right wrist sweeps through the
 * window at a configurable peak speed. Joint angles include hip_rotation
 * and trunk_rotation peaking at controlled timestamps.
 */
function makeCleanStrokeFrames(
  startIndex: number,
  numFrames: number,
  opts: {
    intervalMs?: number
    wristTravel?: number // total horizontal distance of right wrist
    elbowJitter?: number // elbow angle wobble within window
    hipPeakAtFrame?: number
    shoulderPeakAtFrame?: number
  } = {},
): PoseFrame[] {
  const {
    intervalMs = 33,
    wristTravel = 0.3,
    elbowJitter = 0,
    hipPeakAtFrame = startIndex + Math.floor(numFrames / 2) - 2,
    shoulderPeakAtFrame = startIndex + Math.floor(numFrames / 2),
  } = opts

  const out: PoseFrame[] = []
  for (let i = 0; i < numFrames; i++) {
    const t = numFrames > 1 ? i / (numFrames - 1) : 0
    const idx = startIndex + i
    const ts = idx * intervalMs

    // Wrist sweeps from 0.3 to 0.3 + wristTravel.
    const rightWristX = 0.3 + t * wristTravel

    // Build hip/trunk angles with a smooth peak at the configured frame.
    const hipBase = 10
    const trunkBase = 10
    const hipDelta = idx === hipPeakAtFrame ? 25 : 0
    const trunkDelta = idx === shoulderPeakAtFrame ? 30 : 0
    const elbow = 120 + (elbowJitter ? Math.sin(idx * 1.3) * elbowJitter : 0)

    const angles: JointAngles = {
      right_elbow: elbow,
      hip_rotation: hipBase + hipDelta,
      trunk_rotation: trunkBase + trunkDelta,
    }

    out.push(
      makeQualityFrame(idx, ts, {
        rightWristX,
        angles,
      }),
    )
  }
  return out
}

function strokeFromFrames(
  strokeId: string,
  frames: PoseFrame[],
  fps = 30,
): DetectedStroke {
  const startFrame = frames[0].frame_index
  const endFrame = frames[frames.length - 1].frame_index
  const peakFrame = frames[Math.floor(frames.length / 2)].frame_index
  return { strokeId, startFrame, endFrame, peakFrame, fps }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('scoreStrokes — base behavior', () => {
  it('returns deterministic z-scores for identical input', () => {
    const a = makeCleanStrokeFrames(0, 20, { wristTravel: 0.1 })
    const b = makeCleanStrokeFrames(40, 20, { wristTravel: 0.4 })
    const c = makeCleanStrokeFrames(80, 20, { wristTravel: 0.2 })
    const frames = [...a, ...b, ...c]
    const strokes = [
      strokeFromFrames('s1', a),
      strokeFromFrames('s2', b),
      strokeFromFrames('s3', c),
    ]

    const r1 = scoreStrokes(strokes, frames)
    const r2 = scoreStrokes(strokes, frames)
    expect(r1).toEqual(r2)
    expect(r1).toHaveLength(3)
    for (const r of r1) expect(Number.isFinite(r.score)).toBe(true)
  })

  it('orders strokes high-speed clean > medium > slow noisy', () => {
    // High-speed clean stroke: large wrist travel, no elbow jitter.
    const fast = makeCleanStrokeFrames(0, 20, {
      wristTravel: 0.5,
      elbowJitter: 0,
    })
    // Medium speed.
    const mid = makeCleanStrokeFrames(40, 20, {
      wristTravel: 0.25,
      elbowJitter: 0,
    })
    // Slow noisy stroke: small wrist travel, large elbow wobble.
    const slow = makeCleanStrokeFrames(80, 20, {
      wristTravel: 0.05,
      elbowJitter: 30,
    })
    const frames = [...fast, ...mid, ...slow]
    const strokes = [
      strokeFromFrames('fast', fast),
      strokeFromFrames('mid', mid),
      strokeFromFrames('slow', slow),
    ]
    const result = scoreStrokes(strokes, frames)
    expect(result).toHaveLength(3)
    const byId = new Map(result.map((r) => [r.strokeId, r]))
    const sFast = byId.get('fast') as StrokeQualityResult
    const sMid = byId.get('mid') as StrokeQualityResult
    const sSlow = byId.get('slow') as StrokeQualityResult
    expect(sFast.score).toBeGreaterThan(sMid.score)
    expect(sMid.score).toBeGreaterThan(sSlow.score)
    // Sanity: fast has larger raw peak wrist speed than slow.
    expect(sFast.components.peakWristSpeed).toBeGreaterThan(
      sSlow.components.peakWristSpeed,
    )
  })

  it('single non-rejected stroke -> score = 0', () => {
    const a = makeCleanStrokeFrames(0, 20, { wristTravel: 0.3 })
    const result = scoreStrokes([strokeFromFrames('only', a)], a)
    expect(result).toHaveLength(1)
    expect(result[0].rejected).toBe(false)
    expect(result[0].score).toBe(0)
  })

  it('all-rejected session -> empty results', () => {
    // All strokes too short.
    const a = makeCleanStrokeFrames(0, 5, { wristTravel: 0.1 })
    const b = makeCleanStrokeFrames(20, 5, { wristTravel: 0.1 })
    const result = scoreStrokes(
      [strokeFromFrames('a', a), strokeFromFrames('b', b)],
      [...a, ...b],
    )
    expect(result).toEqual([])
  })

  it('empty stroke list -> empty results', () => {
    const result = scoreStrokes([], [])
    expect(result).toEqual([])
  })
})

describe('scoreStrokes — rejection gate', () => {
  it('fires too_short on a 5-frame window', () => {
    // Need a non-rejected partner so we test the rejection-marker path
    // alongside an accepted stroke (mixed output shape).
    const tiny = makeCleanStrokeFrames(0, 5, { wristTravel: 0.1 })
    const ok = makeCleanStrokeFrames(20, 20, { wristTravel: 0.3 })
    const result = scoreStrokes(
      [strokeFromFrames('tiny', tiny), strokeFromFrames('ok', ok)],
      [...tiny, ...ok],
    )
    expect(result).toHaveLength(2)
    const tinyR = result.find((r) => r.strokeId === 'tiny')!
    expect(tinyR.rejected).toBe(true)
    expect(tinyR.rejectReason).toBe('too_short')
    expect(Number.isNaN(tinyR.score)).toBe(true)
  })

  it('fires low_visibility when median upper-body visibility < 0.6', () => {
    // 20 frames with visibility = 0.3 across upper body.
    const out: PoseFrame[] = []
    for (let i = 0; i < 20; i++) {
      out.push(
        makeQualityFrame(i, i * 33, {
          rightWristX: 0.3 + (i / 19) * 0.3,
          visibility: 0.3,
          angles: { right_elbow: 120, hip_rotation: 10, trunk_rotation: 10 },
        }),
      )
    }
    // Pair with an accepted clean stroke so we get a row back.
    const ok = makeCleanStrokeFrames(40, 20, { wristTravel: 0.3 })
    const result = scoreStrokes(
      [strokeFromFrames('blur', out), strokeFromFrames('ok', ok)],
      [...out, ...ok],
    )
    const blur = result.find((r) => r.strokeId === 'blur')!
    expect(blur.rejected).toBe(true)
    expect(blur.rejectReason).toBe('low_visibility')
  })

  it('fires camera_pan when bbox centroid drifts > 15% frame width', () => {
    const out: PoseFrame[] = []
    for (let i = 0; i < 20; i++) {
      // Sweep centroidShift from -0.1 to +0.2 -> drift 0.3 (>0.15).
      const shift = -0.1 + (i / 19) * 0.3
      out.push(
        makeQualityFrame(i, i * 33, {
          rightWristX: 0.3 + (i / 19) * 0.05,
          centroidShift: shift,
          visibility: 1,
          angles: { right_elbow: 120, hip_rotation: 10, trunk_rotation: 10 },
        }),
      )
    }
    const ok = makeCleanStrokeFrames(40, 20, { wristTravel: 0.3 })
    const result = scoreStrokes(
      [strokeFromFrames('pan', out), strokeFromFrames('ok', ok)],
      [...out, ...ok],
    )
    const pan = result.find((r) => r.strokeId === 'pan')!
    expect(pan.rejected).toBe(true)
    expect(pan.rejectReason).toBe('camera_pan')
  })

  it('fires camera_zoom when bbox area variance / mean^2 > 0.05', () => {
    const out: PoseFrame[] = []
    for (let i = 0; i < 20; i++) {
      // Alternate between scale 0.7 and 1.3 -> high variance / mean^2.
      const scale = i % 2 === 0 ? 0.7 : 1.3
      out.push(
        makeQualityFrame(i, i * 33, {
          rightWristX: 0.3 + (i / 19) * 0.05,
          scale,
          visibility: 1,
          angles: { right_elbow: 120, hip_rotation: 10, trunk_rotation: 10 },
        }),
      )
    }
    const ok = makeCleanStrokeFrames(40, 20, { wristTravel: 0.3 })
    const result = scoreStrokes(
      [strokeFromFrames('zoom', out), strokeFromFrames('ok', ok)],
      [...out, ...ok],
    )
    const zoom = result.find((r) => r.strokeId === 'zoom')!
    expect(zoom.rejected).toBe(true)
    expect(zoom.rejectReason).toBe('camera_zoom')
  })

  it('fires missing_data when hip_rotation / trunk_rotation are absent', () => {
    // Build a stroke window with full landmarks but no joint_angles. The
    // rejection gate must catch this so the stroke does not silently get
    // a 0 timing error and an unearned bonus.
    const out: PoseFrame[] = []
    for (let i = 0; i < 20; i++) {
      out.push(
        makeQualityFrame(i, i * 33, {
          rightWristX: 0.4 + (i / 19) * 0.05,
          visibility: 1,
          // No angles whatsoever.
          angles: {},
        }),
      )
    }
    const ok = makeCleanStrokeFrames(40, 20, { wristTravel: 0.3 })
    const result = scoreStrokes(
      [strokeFromFrames('blank', out), strokeFromFrames('ok', ok)],
      [...out, ...ok],
    )
    const blank = result.find((r) => r.strokeId === 'blank')!
    expect(blank.rejected).toBe(true)
    expect(blank.rejectReason).toBe('missing_data')
    expect(Number.isNaN(blank.score)).toBe(true)
  })

  it('handles pixel-space landmarks via frame-width detection', () => {
    // Build pixel-space frames where landmarks live in [0, 1920].
    const out: PoseFrame[] = []
    for (let i = 0; i < 20; i++) {
      const cx = 960 // anchor
      const lms: Landmark[] = [
        makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, cx + 80, 400, 1),
        makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, cx - 80, 400, 1),
        makeLandmark(LANDMARK_INDICES.LEFT_ELBOW, cx + 120, 500, 1),
        makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, cx - 120, 500, 1),
        makeLandmark(LANDMARK_INDICES.LEFT_WRIST, cx + 160, 600, 1),
        // Sweep right wrist a small amount so signal exists.
        makeLandmark(
          LANDMARK_INDICES.RIGHT_WRIST,
          cx - 200 + i * 5,
          600,
          1,
        ),
        makeLandmark(LANDMARK_INDICES.LEFT_HIP, cx + 60, 700, 1),
        makeLandmark(LANDMARK_INDICES.RIGHT_HIP, cx - 60, 700, 1),
      ]
      out.push(
        makeFrame(i, i * 33, lms, {
          right_elbow: 120,
          // Need at least one detectable change in hip and trunk rotation
          // so the chain-timing rejection gate doesn't fire.
          hip_rotation: i === 8 ? 35 : 10,
          trunk_rotation: i === 10 ? 35 : 10,
        }),
      )
    }
    const result = scoreStrokes([strokeFromFrames('px', out)], out)
    expect(result).toHaveLength(1)
    expect(result[0].rejected).toBe(false)
  })
})

describe('scoreStrokes — components', () => {
  it('peakWristSpeed matches the expected magnitude on a known fixture', () => {
    // 11 frames at 1000ms each; right wrist drifts a small amount per
    // frame (Δx=0.005 over Δt=1s -> 0.005/s). Using small motion keeps
    // the upper-body bbox centroid drift below the camera_pan threshold.
    // hip_rotation and trunk_rotation must change at least once each so
    // the chain-timing peaks are detectable (otherwise rejected as
    // missing_data).
    const frames: PoseFrame[] = []
    for (let i = 0; i < 11; i++) {
      frames.push(
        makeQualityFrame(i, i * 1000, {
          rightWristX: 0.4 + i * 0.005,
          rightWristY: 0.5,
          angles: {
            right_elbow: 120,
            hip_rotation: i === 4 ? 30 : 10,
            trunk_rotation: i === 5 ? 35 : 10,
          },
        }),
      )
    }
    const stroke = strokeFromFrames('linear', frames)
    const result = scoreStrokes([stroke], frames)
    expect(result).toHaveLength(1)
    expect(result[0].rejected).toBe(false)
    // 3-frame moving average of constant 0.005 is still 0.005.
    expect(result[0].components.peakWristSpeed).toBeCloseTo(0.005, 4)
  })

  it('wristAngleVariance is nonzero with elbow jitter and zero without', () => {
    const clean = makeCleanStrokeFrames(0, 20, {
      wristTravel: 0.3,
      elbowJitter: 0,
    })
    const noisy = makeCleanStrokeFrames(40, 20, {
      wristTravel: 0.3,
      elbowJitter: 30,
    })
    const result = scoreStrokes(
      [strokeFromFrames('clean', clean), strokeFromFrames('noisy', noisy)],
      [...clean, ...noisy],
    )
    const cleanR = result.find((r) => r.strokeId === 'clean')!
    const noisyR = result.find((r) => r.strokeId === 'noisy')!
    expect(cleanR.components.wristAngleVariance).toBe(0)
    expect(noisyR.components.wristAngleVariance).toBeGreaterThan(0)
  })

  it('kineticChainTimingError = 0 for the median-lag stroke and nonzero for outliers', () => {
    // Three strokes: shoulderPeak - hipPeak lags of 33ms, 99ms (median), 165ms.
    // The middle one should have 0 timing error.
    //
    // peakAngularVelocityMs uses a central-difference estimator
    // v[i] = (angle[i+1] - angle[i-1]) / (t[i+1] - t[i-1]). Each fixture
    // has a single-frame angle spike (delta != 0) at hipPeakAtFrame /
    // shoulderPeakAtFrame, so |angle[i+1] - angle[i-1]| is maximal at both
    // i = peakFrame - 1 (first encountered, wins ties) and i = peakFrame + 1.
    // The first hit wins, so the reported peak timestamp is timestamp at
    // frame (peakFrame - 1). The lag = (shoulderPeakAtFrame - hipPeakAtFrame)
    // * intervalMs because both reported timestamps shift back by one frame
    // and that shift cancels in the difference.
    //
    //   stroke a: shoulderPeak - hipPeak = 6 - 5  = 1 frame  -> lag = 33ms
    //   stroke b: shoulderPeak - hipPeak = 48 - 45 = 3 frames -> lag = 99ms
    //   stroke c: shoulderPeak - hipPeak = 90 - 85 = 5 frames -> lag = 165ms
    //
    //   sessionMedianLag = 99ms
    //   error(a) = |33 - 99|  = 66ms
    //   error(b) = |99 - 99|  = 0ms
    //   error(c) = |165 - 99| = 66ms
    const a = makeCleanStrokeFrames(0, 20, {
      wristTravel: 0.3,
      hipPeakAtFrame: 5,
      shoulderPeakAtFrame: 6, // lag = 33ms
    })
    const b = makeCleanStrokeFrames(40, 20, {
      wristTravel: 0.3,
      hipPeakAtFrame: 45,
      shoulderPeakAtFrame: 48, // lag = 99ms
    })
    const c = makeCleanStrokeFrames(80, 20, {
      wristTravel: 0.3,
      hipPeakAtFrame: 85,
      shoulderPeakAtFrame: 90, // lag = 165ms
    })
    const frames = [...a, ...b, ...c]
    const result = scoreStrokes(
      [
        strokeFromFrames('a', a),
        strokeFromFrames('b', b),
        strokeFromFrames('c', c),
      ],
      frames,
    )
    const byId = new Map(result.map((r) => [r.strokeId, r]))
    expect(byId.get('b')!.components.kineticChainTimingError).toBeCloseTo(0, 6)
    expect(byId.get('a')!.components.kineticChainTimingError).toBeCloseTo(66, 6)
    expect(byId.get('c')!.components.kineticChainTimingError).toBeCloseTo(66, 6)
  })
})
