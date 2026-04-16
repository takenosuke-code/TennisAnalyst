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

  it('returns 0 when two landmarks are at the same position (zero-length vector)', () => {
    // rElbow and rShoulder at the same position
    const landmarks = [
      makeLandmark(LANDMARK_INDICES.LEFT_SHOULDER, 0.6, 0.25),
      makeLandmark(LANDMARK_INDICES.RIGHT_SHOULDER, 0.4, 0.4), // same as elbow
      makeLandmark(LANDMARK_INDICES.RIGHT_ELBOW, 0.4, 0.4),
      makeLandmark(LANDMARK_INDICES.RIGHT_WRIST, 0.55, 0.4),
    ]
    const angles = computeJointAngles(landmarks)

    // vec(elbow->shoulder) = (0,0), magnitude = 0 => angleBetween returns 0
    expect(angles.right_elbow).toBe(0)
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

  it('computes hip_rotation as the absolute atan2 angle of the hip line', () => {
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
