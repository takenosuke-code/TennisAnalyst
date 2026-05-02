import { describe, it, expect } from 'vitest'
import {
  computeJointAngles,
  detectSwings,
  sampleKeyFrames,
  buildAngleSummary,
  LANDMARK_INDICES,
} from '@/lib/jointAngles'
import type { PoseFrame, JointAngles } from '@/lib/supabase'
import {
  makeLandmark,
  makeFrame,
  makeStandingPose,
  makeRightAngleElbowPose,
  makeForehandSwingFrames,
  makeRestFrames,
} from '../helpers'

// ---------------------------------------------------------------------------
// computeJointAngles
// ---------------------------------------------------------------------------

describe('computeJointAngles', () => {
  it('returns all angle fields with a full set of visible landmarks', () => {
    const landmarks = makeStandingPose()
    const angles = computeJointAngles(landmarks)

    expect(angles.right_elbow).toBeDefined()
    expect(angles.left_elbow).toBeDefined()
    expect(angles.right_shoulder).toBeDefined()
    expect(angles.left_shoulder).toBeDefined()
    expect(angles.right_knee).toBeDefined()
    expect(angles.left_knee).toBeDefined()
    expect(angles.hip_rotation).toBeDefined()
    expect(angles.trunk_rotation).toBeDefined()
  })

  it('returns partial angles when right elbow landmark is missing', () => {
    const landmarks = makeStandingPose().filter(
      (l) => l.id !== LANDMARK_INDICES.RIGHT_ELBOW
    )
    const angles = computeJointAngles(landmarks)

    // Right elbow angle needs rShoulder, rElbow, rWrist - rElbow is missing
    expect(angles.right_elbow).toBeUndefined()
    // Right shoulder angle needs lShoulder, rShoulder, rElbow - rElbow is missing
    expect(angles.right_shoulder).toBeUndefined()
    // Left side should still work
    expect(angles.left_elbow).toBeDefined()
    expect(angles.left_shoulder).toBeDefined()
  })

  it('returns partial angles when wrist landmarks are missing', () => {
    const landmarks = makeStandingPose().filter(
      (l) =>
        l.id !== LANDMARK_INDICES.RIGHT_WRIST &&
        l.id !== LANDMARK_INDICES.LEFT_WRIST
    )
    const angles = computeJointAngles(landmarks)

    expect(angles.right_elbow).toBeUndefined()
    expect(angles.left_elbow).toBeUndefined()
    // Shoulder angles only need lShoulder, rShoulder, and the respective elbow
    expect(angles.right_shoulder).toBeDefined()
    expect(angles.left_shoulder).toBeDefined()
    // Knees, hips, trunk still work
    expect(angles.right_knee).toBeDefined()
    expect(angles.left_knee).toBeDefined()
    expect(angles.hip_rotation).toBeDefined()
    expect(angles.trunk_rotation).toBeDefined()
  })

  it('computes 90-degree angle for a right-angle elbow', () => {
    const landmarks = makeRightAngleElbowPose()
    const angles = computeJointAngles(landmarks)

    expect(angles.right_elbow).toBeCloseTo(90, 0)
  })

  it('computes 180-degree angle for a straight arm (collinear landmarks)', () => {
    // rShoulder, rElbow, rWrist all on the same vertical line
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.6, 0.25),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.4, 0.25),
      makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.4, 0.4),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.4, 0.55),
    ]
    const angles = computeJointAngles(landmarks)

    expect(angles.right_elbow).toBeCloseTo(180, 0)
  })

  it('returns NaN when two landmarks are at the same position (zero-length vector)', () => {
    // rElbow and rShoulder at the same position
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.6, 0.25),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.4, 0.4), // same as elbow
      makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.4, 0.4),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.55, 0.4),
    ]
    const angles = computeJointAngles(landmarks)

    // vec(elbow->shoulder) = (0,0), magnitude = 0 => angleBetween returns
    // NaN so downstream Number.isFinite filters can drop the bad frame
    // rather than treating a degenerate measurement as "arm folded shut".
    expect(angles.right_elbow).toBeNaN()
  })

  it('returns angles in degrees in the 0-180 range', () => {
    const landmarks = makeStandingPose()
    const angles = computeJointAngles(landmarks)

    const angleValues = [
      angles.right_elbow,
      angles.left_elbow,
      angles.right_shoulder,
      angles.left_shoulder,
      angles.right_knee,
      angles.left_knee,
    ].filter((v): v is number => v !== undefined)

    for (const angle of angleValues) {
      expect(angle).toBeGreaterThanOrEqual(0)
      expect(angle).toBeLessThanOrEqual(180)
    }
  })

  it('computes hip_rotation as the SIGNED atan2 angle of the hip line', () => {
    // Hips perfectly horizontal: lHip left of rHip
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.3, 0.5),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.7, 0.5),
    ]
    const angles = computeJointAngles(landmarks)

    // vec from lHip to rHip = (0.4, 0) => atan2(0, 0.4) = 0 degrees
    expect(angles.hip_rotation).toBeCloseTo(0, 1)
  })

  it('computes hip_rotation for tilted hips', () => {
    // rHip is below and to the right of lHip
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.4, 0.5),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.6, 0.7),
    ]
    const angles = computeJointAngles(landmarks)

    // vec from lHip to rHip = (0.2, 0.2) => atan2(0.2, 0.2) = 45 degrees
    expect(angles.hip_rotation).toBeCloseTo(45, 0)
  })

  it('returns SIGNED hip_rotation (negative when rHip is above lHip)', () => {
    // rHip is ABOVE and to the right of lHip — opposite tilt direction.
    // The previous Math.abs implementation returned +45° here (wrong);
    // signed math returns -45° so excursion across square-to-camera is
    // correct.
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_HIP, 0.4, 0.7),
      makeLandmark(LANDMARK_INDICES.RIGHT_HIP, 0.6, 0.5),
    ]
    const angles = computeJointAngles(landmarks)
    expect(angles.hip_rotation).toBeCloseTo(-45, 0)
  })

  it('skips wrist angle when index landmark visibility is below 0.5', () => {
    // RTMPose-2D leaves the LEFT_INDEX/RIGHT_INDEX landmarks at
    // {x:0,y:0,visibility:0}. The angle calc must NOT compute a
    // phantom wrist angle from that — it should return undefined so
    // downstream consumers treat it as missing.
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.6, 0.3),
      makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.65, 0.45),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.7, 0.6),
      // Phantom index — present in the array but visibility=0
      makeLandmark(LANDMARK_INDICES.RIGHT_INDEX, 0, 0, 0),
    ]
    const angles = computeJointAngles(landmarks)
    expect(angles.right_wrist).toBeUndefined()
  })

  it('computes wrist angle when index landmark has real visibility', () => {
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.6, 0.3),
      makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.65, 0.45),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.7, 0.6),
      makeLandmark(LANDMARK_INDICES.RIGHT_INDEX, 0.75, 0.65),
    ]
    const angles = computeJointAngles(landmarks)
    expect(angles.right_wrist).toBeDefined()
    expect(typeof angles.right_wrist).toBe('number')
  })

  it('computes trunk_rotation for horizontal shoulders', () => {
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.3, 0.3),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.7, 0.3),
    ]
    const angles = computeJointAngles(landmarks)

    // vec from lShoulder to rShoulder = (0.4, 0) => atan2(0, 0.4) = 0
    expect(angles.trunk_rotation).toBeCloseTo(0, 1)
  })

  it('returns empty object for empty landmarks', () => {
    const angles = computeJointAngles([])
    expect(angles).toEqual({})
  })
})

// ---------------------------------------------------------------------------
// detectSwings
// ---------------------------------------------------------------------------

describe('detectSwings', () => {
  it('returns single segment covering all frames when given empty array', () => {
    // With 0 frames, allFrames.length < minSwingFrames (15) so it returns a
    // single segment. But the segment has startMs/endMs from allFrames[0] which
    // is undefined, so it falls back to 0.
    const result = detectSwings([])
    expect(result).toHaveLength(1)
    expect(result[0].startMs).toBe(0)
    expect(result[0].endMs).toBe(0)
    expect(result[0].frames).toEqual([])
  })

  it('returns single segment when frames < minSwingFrames', () => {
    const frames: PoseFrame[] = Array.from({ length: 10 }, (_, i) =>
      makeFrame(i, i * 33, makeStandingPose())
    )
    const result = detectSwings(frames)
    expect(result).toHaveLength(1)
    expect(result[0].startFrame).toBe(0)
    expect(result[0].endFrame).toBe(9)
    expect(result[0].frames).toHaveLength(10)
  })

  it('detects two distinct swing activity regions', () => {
    // Build: 40 rest, 30 swing, 50 rest, 30 swing, 40 rest
    // The long rest gap (50 frames) ensures the two swings are distinct
    const rest1 = makeRestFrames(40, 0)
    const swing1 = makeForehandSwingFrames(30, 40 * 33)
    const rest2 = makeRestFrames(50, (40 + 30) * 33)
    const swing2 = makeForehandSwingFrames(30, (40 + 30 + 50) * 33)
    const rest3 = makeRestFrames(40, (40 + 30 + 50 + 30) * 33)

    // Fix frame indices to be sequential
    const allFrames: PoseFrame[] = []
    let idx = 0
    for (const f of [...rest1, ...swing1, ...rest2, ...swing2, ...rest3]) {
      allFrames.push({ ...f, frame_index: idx++ })
    }

    const result = detectSwings(allFrames, { mergeGapFrames: 3, minSwingFrames: 10 })
    expect(result.length).toBe(2)
    expect(result[0].index).toBe(1)
    expect(result[1].index).toBe(2)
  })

  it('merges two close activity regions into one swing', () => {
    // Build frames with two activity regions separated by a small gap
    const frames: PoseFrame[] = []
    for (let i = 0; i < 50; i++) {
      // Activity in [5..20] and [25..40], gap of 4 frames
      const isSwing = (i >= 5 && i <= 20) || (i >= 25 && i <= 40)
      const angles: JointAngles = isSwing
        ? {
            right_elbow: 90 + i * 5,
            left_elbow: 120 + i * 3,
            right_shoulder: 45 + i * 3,
            left_shoulder: 50 + i * 3,
            hip_rotation: 10 + i * 2,
            trunk_rotation: 5 + i * 2,
          }
        : {
            right_elbow: 90,
            left_elbow: 120,
            right_shoulder: 45,
            left_shoulder: 50,
            hip_rotation: 10,
            trunk_rotation: 5,
          }
      frames.push(makeFrame(i, i * 33, makeStandingPose(), angles))
    }

    // mergeGapFrames=10 should merge the two regions (gap of 4 < 10)
    const result = detectSwings(frames, { mergeGapFrames: 10 })
    expect(result.length).toBe(1)
  })

  it('applies padding to swing boundaries', () => {
    // Build frames with a clear activity region in the middle
    const frames: PoseFrame[] = []
    for (let i = 0; i < 50; i++) {
      const isSwing = i >= 15 && i <= 35
      const angles: JointAngles = isSwing
        ? {
            right_elbow: 90 + i * 5,
            left_elbow: 120 + i * 3,
            right_shoulder: 45 + i * 3,
            left_shoulder: 50 + i * 3,
            hip_rotation: 10 + i * 2,
            trunk_rotation: 5 + i * 2,
          }
        : {
            right_elbow: 90,
            left_elbow: 120,
            right_shoulder: 45,
            left_shoulder: 50,
            hip_rotation: 10,
            trunk_rotation: 5,
          }
      frames.push(makeFrame(i, i * 33, makeStandingPose(), angles))
    }

    const result = detectSwings(frames)
    expect(result.length).toBeGreaterThanOrEqual(1)
    // Padding of 5 frames is applied - startFrame should be <= activity start
    // and endFrame should be >= activity end
    const seg = result[0]
    expect(seg.startFrame).toBeLessThanOrEqual(15)
    expect(seg.endFrame).toBeGreaterThanOrEqual(35)
  })

  it('returns single segment with uniform activity (all same angles)', () => {
    const frames: PoseFrame[] = Array.from({ length: 30 }, (_, i) =>
      makeFrame(i, i * 33, makeStandingPose(), {
        right_elbow: 90,
        left_elbow: 120,
        right_shoulder: 45,
        left_shoulder: 50,
        hip_rotation: 10,
        trunk_rotation: 5,
      })
    )

    const result = detectSwings(frames)
    // All frames have identical angles => zero activity => no regions detected
    // => falls back to returning the whole video as one segment
    expect(result).toHaveLength(1)
    expect(result[0].startFrame).toBe(0)
    expect(result[0].endFrame).toBe(29)
  })

  it('detects soft swings even when one explosive swing dominates the clip', () => {
    // Regression for the global-max threshold inflation bug. Before the P90
    // fix, one explosive swing's peak set `max`, which pushed
    // `threshold = median + 0.3 × (max − median)` past every softer swing
    // in the clip — slices, drop volleys, blocked returns would all fall
    // below threshold and be silently dropped. With P90 as the spread
    // denominator, one outlier can no longer inflate the bar.
    //
    // Fixture shape:
    //   - Triangle-profile swings (angle ramps to apex and back). Activity
    //     stays at a single plateau through the swing, so the
    //     above-threshold region is one continuous span — not two short
    //     halves split by a mid-swing zero like a sin profile would
    //     produce.
    //   - 1 hard swing (peakDelta=60°) followed by 3 soft swings
    //     (peakDelta=12°). Hard activity is ~5× soft activity per frame.
    //   - Long rest periods (60 frames each) so the hard swing's
    //     high-activity frames are well under 10% of the clip — the
    //     condition for P90 to land in the soft-swing range rather than
    //     in the hard-swing range.
    const buildSwing = (
      numFrames: number,
      startMs: number,
      peakDelta: number,
    ): PoseFrame[] => {
      const out: PoseFrame[] = []
      // Wrist amplitude scales with peakDelta. The hard swing
      // (peakDelta=60) sweeps the wrist through a wide arc; the soft
      // swings (peakDelta=12) move the wrist a fifth as far. The
      // detector's mean+1·std threshold sits between the two — it
      // catches the soft swings as long as one big swing's high-speed
      // window doesn't dominate the normalization.
      const amplitude = 0.45 * (peakDelta / 60)
      for (let i = 0; i < numFrames; i++) {
        const t = i / Math.max(1, numFrames - 1)
        const ramp = t < 0.5 ? t * 2 : (1 - t) * 2
        const e = peakDelta * ramp
        const angles: JointAngles = {
          right_elbow: 160 - e,
          left_elbow: 160 - e * 0.4,
          right_shoulder: 30 + e * 0.6,
          left_shoulder: 30 + e * 0.3,
          hip_rotation: 10 + e * 0.4,
          trunk_rotation: 10 + e * 0.4,
        }
        // Asymmetric wrist arc returning to the rest x at swing
        // boundaries so back-to-back rest/swing fixtures don't spike
        // velocity at the seams.
        const wristX =
          t < 0.5
            ? 0.4 + amplitude * (t * 2) ** 1.5
            : 0.4 + amplitude * (1 - (t - 0.5) * 2) ** 1.5
        const wristY = 0.55 - 0.05 * Math.sin(Math.PI * t) * (peakDelta / 60)
        const landmarks = makeStandingPose().map((lm) =>
          lm.id === LANDMARK_INDICES.RIGHT_WRIST
            ? { ...lm, x: wristX, y: wristY }
            : lm,
        )
        out.push(makeFrame(i, startMs + i * 33, landmarks, angles))
      }
      return out
    }

    const REST = 60
    const SWING = 30
    const sections: PoseFrame[][] = []
    let cursorMs = 0
    sections.push(makeRestFrames(REST, cursorMs)); cursorMs += REST * 33
    sections.push(buildSwing(SWING, cursorMs, 60));  cursorMs += SWING * 33
    sections.push(makeRestFrames(REST, cursorMs)); cursorMs += REST * 33
    sections.push(buildSwing(SWING, cursorMs, 30));  cursorMs += SWING * 33
    sections.push(makeRestFrames(REST, cursorMs)); cursorMs += REST * 33
    sections.push(buildSwing(SWING, cursorMs, 30));  cursorMs += SWING * 33
    sections.push(makeRestFrames(REST, cursorMs)); cursorMs += REST * 33
    sections.push(buildSwing(SWING, cursorMs, 30));  cursorMs += SWING * 33
    sections.push(makeRestFrames(REST, cursorMs)); cursorMs += REST * 33

    const allFrames: PoseFrame[] = []
    let idx = 0
    for (const sect of sections) {
      for (const f of sect) {
        allFrames.push({ ...f, frame_index: idx++ })
      }
    }

    const result = detectSwings(allFrames)
    expect(result.length).toBe(4)
  })

  it('keeps a swing whole when activity dips at the top of the backswing', () => {
    // Regression: a real forehand has a brief reversal at the top of the
    // backswing (~150-250ms of near-zero joint-angle delta as the racket
    // transitions from back-swing to forward-swing). Before the
    // hysteresis + wider mergeGap fix, that valley split a single swing
    // into two — both halves often passing the minSwingFrames floor and
    // showing up as two distinct cards. Fixture: 25 rest, then 50-frame
    // swing with a 12-frame near-zero valley in the middle, then 25
    // rest. Expect ONE swing.
    const FPS = 30
    const dt = Math.round(1000 / FPS)
    const buildSwingWithValley = (startMs: number): PoseFrame[] => {
      const frames: PoseFrame[] = []
      // Phase plan (50 frames):
      //   0-15: ramp activity up (backswing → loading)
      //   16-27: VALLEY (top-of-backswing reversal, near-zero delta)
      //   28-37: peak (acceleration through contact)
      //   38-49: ramp down (follow-through tail)
      // Each frame's angles step by an amount proportional to the
      // phase's "activity level" so the smoothed signal carries the
      // valley dip.
      let elbow = 160, shoulder = 30, hip = 10, trunk = 10
      for (let i = 0; i < 50; i++) {
        let step = 0
        if (i < 16) step = 4               // ramp up
        else if (i < 28) step = 0.3        // valley (very small but nonzero)
        else if (i < 38) step = 6          // peak
        else step = 2                       // ramp down
        elbow -= step
        shoulder += step * 0.5
        hip += step * 0.3
        trunk += step * 0.3
        const angles: JointAngles = {
          right_elbow: elbow,
          left_elbow: 160 - elbow * 0.2,
          right_shoulder: shoulder,
          left_shoulder: shoulder * 0.4,
          hip_rotation: hip,
          trunk_rotation: trunk,
        }
        frames.push(makeFrame(i, startMs + i * dt, makeStandingPose(), angles))
      }
      return frames
    }

    const REST = 25
    const sections: PoseFrame[][] = []
    let cursorMs = 0
    sections.push(makeRestFrames(REST, cursorMs)); cursorMs += REST * dt
    sections.push(buildSwingWithValley(cursorMs)); cursorMs += 50 * dt
    sections.push(makeRestFrames(REST, cursorMs)); cursorMs += REST * dt

    const allFrames: PoseFrame[] = []
    let idx = 0
    for (const sect of sections) {
      for (const f of sect) {
        allFrames.push({ ...f, frame_index: idx++ })
      }
    }

    const result = detectSwings(allFrames)
    // Hysteresis + wider mergeGap should keep the swing whole.
    expect(result.length).toBe(1)
    // And the segment should span (most of) the 50 swing frames plus
    // padding — proves it didn't truncate at the valley.
    const swing = result[0]
    expect(swing.endFrame - swing.startFrame).toBeGreaterThan(45)
  })

  it('sets peakFrame to the frame with maximum activity', () => {
    const frames: PoseFrame[] = []
    for (let i = 0; i < 30; i++) {
      // Big spike at frame 15
      const multiplier = i === 15 ? 50 : i >= 10 && i <= 20 ? 3 : 0
      const angles: JointAngles = {
        right_elbow: 90 + multiplier * i,
        left_elbow: 120 + multiplier * i,
        right_shoulder: 45 + multiplier * i,
        left_shoulder: 50 + multiplier * i,
        hip_rotation: 10 + multiplier,
        trunk_rotation: 5 + multiplier,
      }
      frames.push(makeFrame(i, i * 33, makeStandingPose(), angles))
    }

    const result = detectSwings(frames)
    expect(result.length).toBeGreaterThanOrEqual(1)
    // peakFrame should be near the spike
    expect(result[0].peakFrame).toBeGreaterThanOrEqual(10)
    expect(result[0].peakFrame).toBeLessThanOrEqual(20)
  })

  // -------------------------------------------------------------------------
  // dropRejected option
  // -------------------------------------------------------------------------

  it('dropRejected: false (default) keeps all detected strokes regardless of quality', () => {
    // Build a clip where the detector finds a swing but scoreStrokes
    // would reject it (low landmark visibility — lands below the
    // 0.6 median floor in evaluateRejection).
    const frames: PoseFrame[] = []
    for (let i = 0; i < 80; i++) {
      const u = (i - 40) * 0.4
      const sigmoid = 1 / (1 + Math.exp(-u))
      const wristX = 0.4 + 0.45 * sigmoid
      const landmarks = makeStandingPose().map((lm) =>
        lm.id === LANDMARK_INDICES.RIGHT_WRIST
          ? { ...lm, x: wristX, y: 0.55, visibility: 0.9 }
          : { ...lm, visibility: 0.2 } // upper body invisible
      )
      frames.push(makeFrame(i, Math.round((i * 1000) / 30), landmarks))
    }
    const result = detectSwings(frames)
    expect(result.length).toBe(1)
  })

  it('dropRejected: true filters out strokes flagged by scoreStrokes', () => {
    // Same low-visibility setup as above. With dropRejected=true the
    // scoreStrokes pass should reject the stroke for low_visibility,
    // and the wrapper should return an empty list (NOT the legacy
    // whole-clip fallback — that would just produce a phantom card).
    const frames: PoseFrame[] = []
    for (let i = 0; i < 80; i++) {
      const u = (i - 40) * 0.4
      const sigmoid = 1 / (1 + Math.exp(-u))
      const wristX = 0.4 + 0.45 * sigmoid
      const landmarks = makeStandingPose().map((lm) =>
        lm.id === LANDMARK_INDICES.RIGHT_WRIST
          ? { ...lm, x: wristX, y: 0.55, visibility: 0.9 }
          : { ...lm, visibility: 0.2 }
      )
      frames.push(makeFrame(i, Math.round((i * 1000) / 30), landmarks))
    }
    const result = detectSwings(frames, { dropRejected: true })
    expect(result.length).toBe(0)
  })

  it('dropRejected: true keeps strokes that pass quality and drops the bad ones', () => {
    // Two swings; both have high upper-body visibility, modest wrist
    // amplitude (so bbox cx drift stays under the 0.15 camera-pan
    // floor), and provide hip/trunk angles so scoreStrokes' kinetic-
    // chain timing peaks resolve. The option must be non-destructive
    // on this clean input.
    const frames: PoseFrame[] = []
    for (let i = 0; i < 180; i++) {
      const swing1 = 1 / (1 + Math.exp(-(i - 60) * 0.6))
      const swing2 = 1 / (1 + Math.exp(-(i - 120) * 0.6))
      const wristX = 0.4 + 0.10 * swing1 + 0.10 * swing2
      const landmarks = makeStandingPose().map((lm) =>
        lm.id === LANDMARK_INDICES.RIGHT_WRIST
          ? { ...lm, x: wristX, y: 0.55, visibility: 0.9 }
          : { ...lm, visibility: 0.9 }
      )
      frames.push(
        makeFrame(i, Math.round((i * 1000) / 30), landmarks, {
          right_elbow: 130,
          left_elbow: 130,
          hip_rotation: 30 + 30 * (swing1 + swing2),
          trunk_rotation: 50 + 30 * (swing1 + swing2),
        }),
      )
    }
    const result = detectSwings(frames, { dropRejected: true })
    expect(result.length).toBe(2)
  })
})

// ---------------------------------------------------------------------------
// sampleKeyFrames
// ---------------------------------------------------------------------------

describe('sampleKeyFrames', () => {
  const frames: PoseFrame[] = Array.from({ length: 100 }, (_, i) =>
    makeFrame(i, i * 33, [])
  )

  it('returns first frame when count=1', () => {
    const result = sampleKeyFrames(frames, 1)
    expect(result).toHaveLength(1)
    expect(result[0].frame_index).toBe(0)
  })

  it('returns all frames when count >= frames.length', () => {
    const shortFrames = frames.slice(0, 5)
    const result = sampleKeyFrames(shortFrames, 10)
    expect(result).toHaveLength(5)
  })

  it('returns evenly spaced frames for 100 frames with count=5', () => {
    const result = sampleKeyFrames(frames, 5)
    expect(result).toHaveLength(5)
    // step = 99/4 = 24.75
    // indices: round(0)=0, round(24.75)=25, round(49.5)=50, round(74.25)=74, round(99)=99
    expect(result[0].frame_index).toBe(0)
    expect(result[1].frame_index).toBe(25)
    expect(result[2].frame_index).toBe(50)
    expect(result[3].frame_index).toBe(74)
    expect(result[4].frame_index).toBe(99)
  })

  it('returns empty array for count=0 (sliced to 0 elements)', () => {
    // count <= 1 path: returns frames.slice(0, 1) which is [frames[0]]
    // Actually count=0 satisfies count <= 1, so it returns slice(0,1)
    const result = sampleKeyFrames(frames, 0)
    expect(result).toHaveLength(1)
    expect(result[0].frame_index).toBe(0)
  })

  it('returns all frames when count equals frames.length', () => {
    const result = sampleKeyFrames(frames, 100)
    expect(result).toHaveLength(100)
  })

  it('handles single-frame input', () => {
    const single = [makeFrame(0, 0, [])]
    const result = sampleKeyFrames(single, 5)
    expect(result).toHaveLength(1)
    expect(result[0].frame_index).toBe(0)
  })
})

// ---------------------------------------------------------------------------
// buildAngleSummary
// ---------------------------------------------------------------------------

describe('buildAngleSummary', () => {
  it('produces the correct string format with all angles present', () => {
    const frames: PoseFrame[] = Array.from({ length: 5 }, (_, i) =>
      makeFrame(i, i * 100, [], {
        right_elbow: 90,
        left_elbow: 120,
        right_shoulder: 45,
        right_knee: 160,
        hip_rotation: 10,
        trunk_rotation: 5,
      })
    )

    const result = buildAngleSummary(frames)
    const lines = result.split('\n')
    expect(lines).toHaveLength(5)
    expect(lines[0]).toContain('preparation:')
    expect(lines[0]).toContain('elbow_R=90')
    expect(lines[0]).toContain('elbow_L=120')
    expect(lines[0]).toContain('shoulder_R=45')
    expect(lines[0]).toContain('knee_R=160')
    expect(lines[0]).toContain('hip_rot=10')
    expect(lines[0]).toContain('trunk_rot=5')
  })

  it('uses custom phase names', () => {
    const frames: PoseFrame[] = Array.from({ length: 5 }, (_, i) =>
      makeFrame(i, i * 100, [], { right_elbow: 90 })
    )
    const customPhases = ['ready', 'backswing', 'forward', 'impact', 'recovery']
    const result = buildAngleSummary(frames, customPhases)
    const lines = result.split('\n')

    expect(lines[0]).toMatch(/^ready:/)
    expect(lines[1]).toMatch(/^backswing:/)
    expect(lines[2]).toMatch(/^forward:/)
    expect(lines[3]).toMatch(/^impact:/)
    expect(lines[4]).toMatch(/^recovery:/)
  })

  it('shows N/A for missing angles', () => {
    const frames: PoseFrame[] = [makeFrame(0, 0, [], {})]
    const result = buildAngleSummary(frames)

    expect(result).toContain('elbow_R=N/A')
    expect(result).toContain('elbow_L=N/A')
    expect(result).toContain('shoulder_R=N/A')
    expect(result).toContain('knee_R=N/A')
    expect(result).toContain('hip_rot=N/A')
    expect(result).toContain('trunk_rot=N/A')
  })

  it('falls back to frame_<index> when phase names run out', () => {
    const frames: PoseFrame[] = Array.from({ length: 10 }, (_, i) =>
      makeFrame(i, i * 100, [], { right_elbow: 90 })
    )
    // Only 2 phase names but sampleKeyFrames will pick 5 frames
    const result = buildAngleSummary(frames, ['phase1', 'phase2'])
    const lines = result.split('\n')

    expect(lines[0]).toMatch(/^phase1:/)
    expect(lines[1]).toMatch(/^phase2:/)
    // Remaining lines should use frame_<index> fallback
    expect(lines[2]).toMatch(/^frame_\d+:/)
  })
})
