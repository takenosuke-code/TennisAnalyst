import { describe, it, expect } from 'vitest'
import {
  computeConfidence,
  extractObservations,
  pickPrimary,
  pickSecondary,
  CONFIDENCE_FLOOR,
  SEVERITY_RANK,
  type Observation,
} from '@/lib/coachingObservations'
import type { JointAngles, Landmark, PoseFrame } from '@/lib/supabase'
import { makeStandingPose } from '../helpers'

// ---------------------------------------------------------------------------
// Synthetic-frame builder. Lets each test set the joint_angles it cares about
// and reuse the standing-pose landmark set so visibility stays high.
// ---------------------------------------------------------------------------

function makeFrames(
  angleProgression: JointAngles[],
  options: { visibility?: number } = {},
): PoseFrame[] {
  const vis = options.visibility ?? 1.0
  const landmarks: Landmark[] = makeStandingPose().map((l) => ({
    ...l,
    visibility: vis,
  }))
  return angleProgression.map((angles, i) => ({
    frame_index: i,
    timestamp_ms: i * 33,
    landmarks,
    joint_angles: angles,
  }))
}

/**
 * Build a 60-frame swing with a peak in the middle. The peak frame uses the
 * `peakAngles` overrides; surrounding frames decay toward `restAngles` so the
 * detector can find a clear peak.
 */
function makeSwingAroundPeak(
  peakAngles: JointAngles,
  restAngles: JointAngles,
  numFrames = 60,
): PoseFrame[] {
  const peakIdx = Math.floor(numFrames / 2)
  const angles: JointAngles[] = []
  for (let i = 0; i < numFrames; i++) {
    const dist = Math.abs(i - peakIdx)
    const t = Math.max(0, 1 - dist / (numFrames / 2)) // 1 at peak, 0 at edges
    const blended: JointAngles = {}
    const keys = new Set([
      ...Object.keys(peakAngles),
      ...Object.keys(restAngles),
    ]) as Set<keyof JointAngles>
    for (const k of keys) {
      const peakVal = peakAngles[k]
      const restVal = restAngles[k]
      if (typeof peakVal === 'number' && typeof restVal === 'number') {
        blended[k] = restVal + (peakVal - restVal) * t
      } else if (typeof peakVal === 'number') {
        blended[k] = peakVal
      } else if (typeof restVal === 'number') {
        blended[k] = restVal
      }
    }
    angles.push(blended)
  }
  return makeFrames(angles)
}

// ---------------------------------------------------------------------------
// computeConfidence
// ---------------------------------------------------------------------------

describe('computeConfidence', () => {
  it('multiplies the three signals and clamps to [0,1]', () => {
    expect(
      computeConfidence({
        landmarkVisibility: 1.0,
        thresholdMargin: 1.0,
        ruleApplicability: 1.0,
      }),
    ).toBeCloseTo(1.0)
    expect(
      computeConfidence({
        landmarkVisibility: 0.5,
        thresholdMargin: 0.5,
        ruleApplicability: 1.0,
      }),
    ).toBeCloseTo(0.25)
  })

  it('clamps negative or out-of-range values', () => {
    expect(
      computeConfidence({
        landmarkVisibility: -0.2,
        thresholdMargin: 0.8,
        ruleApplicability: 1.0,
      }),
    ).toBe(0)
    expect(
      computeConfidence({
        landmarkVisibility: 1.5,
        thresholdMargin: 1.5,
        ruleApplicability: 1.5,
      }),
    ).toBe(1)
  })
})

// ---------------------------------------------------------------------------
// extractObservations: detection rules
// ---------------------------------------------------------------------------

describe('extractObservations', () => {
  it('returns empty array when there are no frames', () => {
    expect(
      extractObservations({ todaySummary: [], shotType: 'forehand' }),
    ).toEqual([])
  })

  it('detects cramped_elbow at contact when right elbow < 90°', () => {
    const frames = makeSwingAroundPeak(
      {
        right_elbow: 70, // cramped
        right_knee: 150,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 130,
        right_knee: 165,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const cramped = obs.find((o) => o.pattern === 'cramped_elbow')
    expect(cramped).toBeDefined()
    expect(cramped!.phase).toBe('contact')
    expect(cramped!.joint).toBe('right_elbow')
    expect(cramped!.todayValue).toBeLessThan(90)
    expect(cramped!.confidence).toBeGreaterThanOrEqual(CONFIDENCE_FLOOR)
  })

  it('detects over_extended_elbow when right elbow > 175°', () => {
    const frames = makeSwingAroundPeak(
      {
        right_elbow: 178,
        right_knee: 150,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 130,
        right_knee: 165,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const over = obs.find((o) => o.pattern === 'over_extended_elbow')
    expect(over).toBeDefined()
    expect(over!.todayValue).toBeGreaterThan(175)
  })

  it('detects locked_knees when right knee > 170°', () => {
    const frames = makeSwingAroundPeak(
      {
        right_elbow: 120,
        right_knee: 178,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 130,
        right_knee: 178,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const locked = obs.find((o) => o.pattern === 'locked_knees')
    expect(locked).toBeDefined()
    expect(locked!.phase).toBe('loading')
  })

  it('detects shallow_knee_load in the high-but-not-locked band', () => {
    // 168° is well into the shallow band (160-170 not locked) — gives enough
    // margin to clear the confidence floor with default applicability.
    const frames = makeSwingAroundPeak(
      {
        right_elbow: 120,
        right_knee: 168,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 130,
        right_knee: 168,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const shallow = obs.find((o) => o.pattern === 'shallow_knee_load')
    expect(shallow).toBeDefined()
  })

  it('detects insufficient_hip_excursion when hip rotation barely changes', () => {
    // Hip stays at 30° the entire swing -> excursion ~0 -> < 40° target.
    const angles: JointAngles[] = []
    for (let i = 0; i < 60; i++) {
      angles.push({
        right_elbow: 120,
        right_knee: 150,
        hip_rotation: 30,
        trunk_rotation: 30 + (i < 30 ? i : 60 - i), // gives the swing-detector activity
      })
    }
    const frames = makeFrames(angles)
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const stuck = obs.find((o) => o.pattern === 'insufficient_hip_excursion')
    expect(stuck).toBeDefined()
    expect(stuck!.todayValue).toBeLessThan(40)
  })

  it('detects insufficient_trunk_excursion when trunk rotation barely changes', () => {
    const angles: JointAngles[] = []
    for (let i = 0; i < 60; i++) {
      angles.push({
        right_elbow: 100 + (i < 30 ? i : 60 - i), // arm waves to give activity
        right_knee: 150,
        hip_rotation: 20 + (i < 30 ? i * 2 : (60 - i) * 2),
        trunk_rotation: 25, // flat
      })
    }
    const frames = makeFrames(angles)
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const stuck = obs.find((o) => o.pattern === 'insufficient_trunk_excursion')
    expect(stuck).toBeDefined()
  })

  it('detects insufficient_unit_turn when both excursions are small', () => {
    const angles: JointAngles[] = []
    for (let i = 0; i < 60; i++) {
      angles.push({
        right_elbow: 90 + (i < 30 ? i * 2 : (60 - i) * 2), // big elbow swing for activity
        right_knee: 150,
        hip_rotation: 15,
        trunk_rotation: 25,
      })
    }
    const frames = makeFrames(angles)
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const noTurn = obs.find((o) => o.pattern === 'insufficient_unit_turn')
    expect(noTurn).toBeDefined()
    expect(noTurn!.phase).toBe('preparation')
  })

  it('returns empty when mean landmark visibility is below the floor', () => {
    const frames = makeSwingAroundPeak(
      {
        right_elbow: 70, // would normally be cramped
        right_knee: 150,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 130,
        right_knee: 165,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    ).map((f) => ({
      ...f,
      landmarks: f.landmarks.map((l) => ({ ...l, visibility: 0.1 })),
    }))
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    expect(obs).toEqual([])
  })

  it('drops single-rule observations when their confidence falls below the floor', () => {
    // Visibility 0.55 × max margin 1.0 × applicability 1.0 = 0.55, below floor.
    const frames = makeSwingAroundPeak(
      {
        right_elbow: 70,
        right_knee: 150,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 130,
        right_knee: 165,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    ).map((f) => ({
      ...f,
      landmarks: f.landmarks.map((l) => ({ ...l, visibility: 0.55 })),
    }))
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    // None of the per-rule confidences clear 0.6 with vis=0.55. All dropped.
    for (const o of obs) {
      expect(o.confidence).toBeGreaterThanOrEqual(CONFIDENCE_FLOOR)
    }
  })

  it('halves rule applicability when shotType is null', () => {
    // Cramped elbow + locked knees -> with shotType=null applicability=0.5,
    // so confidence = 1 * margin * 0.5 = 0.5 * margin. Margin must exceed
    // 1.2 to clear floor — impossible after clamp — so we should drop these.
    const frames = makeSwingAroundPeak(
      {
        right_elbow: 70,
        right_knee: 178,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 130,
        right_knee: 178,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    )
    const obs = extractObservations({ todaySummary: frames, shotType: null })
    // With applicability 0.5 every observation's confidence ≤ 0.5, so all dropped.
    expect(obs.find((o) => o.pattern === 'cramped_elbow')).toBeUndefined()
    expect(obs.find((o) => o.pattern === 'locked_knees')).toBeUndefined()
  })

  // -------------------------------------------------------------------------
  // Baseline-compare mode
  // -------------------------------------------------------------------------

  it('emits drift_from_baseline observations when today vs baseline diverges', () => {
    // Today: elbow contact min 95°, knee load min 130°.
    const today = makeSwingAroundPeak(
      {
        right_elbow: 95,
        right_knee: 130,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 130,
        right_knee: 165,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    )
    // Baseline: elbow contact min ~135°, knee load min ~155° — large drift
    // from today on both joints (>30°).
    const baseline = makeSwingAroundPeak(
      {
        right_elbow: 135,
        right_knee: 155,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 145,
        right_knee: 165,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    )
    const obs = extractObservations({
      todaySummary: today,
      baselineSummary: baseline,
      shotType: 'forehand',
    })
    const drifts = obs.filter((o) => o.pattern === 'drift_from_baseline')
    expect(drifts.length).toBeGreaterThan(0)
    for (const d of drifts) {
      expect(d.baselineValue).toBeDefined()
      expect(d.driftMagnitude).toBeDefined()
      expect(d.driftMagnitude!).toBeGreaterThan(0)
    }
  })

  it('does NOT emit drift when today and baseline match', () => {
    const today = makeSwingAroundPeak(
      {
        right_elbow: 120,
        right_knee: 150,
        hip_rotation: 50,
        trunk_rotation: 60,
      },
      {
        right_elbow: 130,
        right_knee: 165,
        hip_rotation: 10,
        trunk_rotation: 10,
      },
    )
    const obs = extractObservations({
      todaySummary: today,
      baselineSummary: today,
      shotType: 'forehand',
    })
    expect(obs.filter((o) => o.pattern === 'drift_from_baseline')).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// pickPrimary / pickSecondary
// ---------------------------------------------------------------------------

describe('pickPrimary / pickSecondary', () => {
  function obs(p: Partial<Observation>): Observation {
    return {
      phase: 'contact',
      joint: 'right_elbow',
      pattern: 'cramped_elbow',
      severity: 'mild',
      confidence: 0.7,
      todayValue: 100,
      ...p,
    }
  }

  it('returns null on empty input', () => {
    expect(pickPrimary([])).toBeNull()
  })

  it('picks highest severity × confidence', () => {
    const list = [
      obs({ pattern: 'cramped_elbow', severity: 'mild', confidence: 0.95 }),
      obs({ pattern: 'locked_knees', severity: 'severe', confidence: 0.7 }),
      obs({ pattern: 'truncated_followthrough', severity: 'moderate', confidence: 0.7 }),
    ]
    const primary = pickPrimary(list)
    // severity rank: severe=3, moderate=2, mild=1.
    // scores: 1*0.95=0.95, 3*0.7=2.1, 2*0.7=1.4 -> locked_knees wins.
    expect(primary?.pattern).toBe('locked_knees')
  })

  it('pickSecondary excludes the primary and returns up to max items in score order', () => {
    const a = obs({ pattern: 'locked_knees', severity: 'severe', confidence: 0.7 }) // 2.1
    const b = obs({ pattern: 'truncated_followthrough', severity: 'moderate', confidence: 0.7 }) // 1.4
    const c = obs({ pattern: 'cramped_elbow', severity: 'mild', confidence: 0.95 }) // 0.95
    const d = obs({ pattern: 'over_extended_elbow', severity: 'moderate', confidence: 0.6 }) // 1.2
    const list = [a, b, c, d]
    const sec = pickSecondary(list, a, 3)
    expect(sec).toHaveLength(3)
    // Score order excluding a: b(1.4), d(1.2), c(0.95).
    expect(sec[0]).toBe(b)
    expect(sec[1]).toBe(d)
    expect(sec[2]).toBe(c)
  })

  it('pickSecondary respects the max parameter', () => {
    const list = [
      obs({ severity: 'severe', confidence: 0.9 }),
      obs({ severity: 'moderate', confidence: 0.9 }),
      obs({ severity: 'mild', confidence: 0.9 }),
    ]
    expect(pickSecondary(list, null, 1)).toHaveLength(1)
    expect(pickSecondary(list, null, 0)).toHaveLength(0)
  })

  it('SEVERITY_RANK is the documented ordering', () => {
    expect(SEVERITY_RANK.mild).toBeLessThan(SEVERITY_RANK.moderate)
    expect(SEVERITY_RANK.moderate).toBeLessThan(SEVERITY_RANK.severe)
  })
})
