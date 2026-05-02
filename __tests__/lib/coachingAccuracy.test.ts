/**
 * Coaching accuracy suite.
 *
 * Goal: prove the analysis pipeline gives SPECIFIC advice on the
 * specific input, not generic boilerplate. For each scenario:
 *   1. Build a synthetic swing with a known fault (or no fault).
 *   2. Run the full extractObservations pipeline.
 *   3. Assert the right pattern fires AND the wrong patterns don't.
 *   4. Assert numeric details (joint side, magnitude, severity)
 *      match the input — i.e. the system can't mistake a cramped-
 *      elbow swing for a hip-rotation issue.
 *
 * Separate from coachingObservations.test.ts because that file tests
 * individual rules in isolation. This one tests the COMPOSITION:
 * does the full extract+pick output reflect the swing it was given?
 */
import { describe, it, expect } from 'vitest'
import {
  extractObservations,
  pickPrimary,
  pickSecondary,
  type Observation,
} from '@/lib/coachingObservations'
import type { JointAngles, Landmark, PoseFrame } from '@/lib/supabase'
import { makeStandingPose } from '../helpers'

// ---------------------------------------------------------------------------
// Helpers — copied from coachingObservations.test.ts so this suite is
// self-contained.
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
 * Build a 60-frame swing with a peak in the middle. The peak uses
 * `peakAngles`; rest frames use `restAngles`; intermediate frames
 * blend linearly between them.
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
    const t = Math.max(0, 1 - dist / (numFrames / 2))
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

/** Reference "clean" angles a real swing should hit. Used to anchor
 *  scenarios so only the channel under test deviates. */
const CLEAN_PEAK: JointAngles = {
  right_elbow: 130,    // within 100-140 ideal contact range
  left_elbow: 130,
  right_knee: 160,     // extending through contact
  left_knee: 160,
  hip_rotation: 30,    // post-rotation reading
  trunk_rotation: 60,  // post-rotation reading
}
const CLEAN_REST: JointAngles = {
  right_elbow: 130,
  left_elbow: 130,
  right_knee: 145,     // loaded
  left_knee: 145,
  hip_rotation: -25,   // pre-rotation: hip line tilted opposite of peak
  trunk_rotation: -50,
}

// ---------------------------------------------------------------------------
// Scenario suite
// ---------------------------------------------------------------------------

describe('Coaching accuracy: scenario fixtures produce specific output', () => {
  it('cramped elbow scenario fires cramped_elbow on the right side, not other patterns', () => {
    const frames = makeSwingAroundPeak(
      { ...CLEAN_PEAK, right_elbow: 70 },
      CLEAN_REST,
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const cramped = obs.find((o) => o.pattern === 'cramped_elbow')
    expect(cramped, 'cramped_elbow should be detected').toBeDefined()
    expect(cramped!.joint).toBe('right_elbow')
    expect(cramped!.todayValue).toBeLessThan(90)
    // No false positives on other channels
    expect(obs.find((o) => o.pattern === 'over_extended_elbow')).toBeUndefined()
    expect(obs.find((o) => o.pattern === 'locked_knees')).toBeUndefined()
    expect(obs.find((o) => o.pattern === 'shallow_knee_load')).toBeUndefined()
    expect(obs.find((o) => o.pattern === 'insufficient_hip_excursion')).toBeUndefined()
    expect(obs.find((o) => o.pattern === 'insufficient_trunk_excursion')).toBeUndefined()
  })

  it('over-extended elbow scenario fires over_extended_elbow, not cramped_elbow', () => {
    const frames = makeSwingAroundPeak(
      { ...CLEAN_PEAK, right_elbow: 178 },
      CLEAN_REST,
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const over = obs.find((o) => o.pattern === 'over_extended_elbow')
    expect(over).toBeDefined()
    expect(over!.joint).toBe('right_elbow')
    expect(over!.todayValue).toBeGreaterThan(175)
    expect(obs.find((o) => o.pattern === 'cramped_elbow')).toBeUndefined()
  })

  it('locked knees scenario fires locked_knees, not insufficient_hip_excursion', () => {
    const frames = makeSwingAroundPeak(
      { ...CLEAN_PEAK, right_knee: 178 },
      { ...CLEAN_REST, right_knee: 178 }, // locked throughout
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const locked = obs.find((o) => o.pattern === 'locked_knees')
    expect(locked).toBeDefined()
    expect(locked!.joint).toBe('right_knee')
    expect(locked!.todayValue).toBeGreaterThan(170)
    expect(obs.find((o) => o.pattern === 'cramped_elbow')).toBeUndefined()
  })

  it('flat hips scenario fires insufficient_hip_excursion, not insufficient_trunk_excursion', () => {
    const frames = makeSwingAroundPeak(
      { ...CLEAN_PEAK, hip_rotation: 8 },   // peak nearly equals rest
      { ...CLEAN_REST, hip_rotation: 5 },   // tiny excursion ~3°
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const stuckHips = obs.find((o) => o.pattern === 'insufficient_hip_excursion')
    expect(stuckHips).toBeDefined()
    expect(stuckHips!.todayValue).toBeLessThan(25)
    // Trunk still rotates 110° here — should NOT fire
    expect(obs.find((o) => o.pattern === 'insufficient_trunk_excursion')).toBeUndefined()
  })

  it('flat trunk scenario fires insufficient_trunk_excursion, not insufficient_hip_excursion', () => {
    const frames = makeSwingAroundPeak(
      { ...CLEAN_PEAK, trunk_rotation: 12 },
      { ...CLEAN_REST, trunk_rotation: 10 },
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const stuckTrunk = obs.find((o) => o.pattern === 'insufficient_trunk_excursion')
    expect(stuckTrunk).toBeDefined()
    expect(stuckTrunk!.todayValue).toBeLessThan(30)
    // Hips still rotate ~55° — should NOT fire
    expect(obs.find((o) => o.pattern === 'insufficient_hip_excursion')).toBeUndefined()
  })

  it('clean swing produces zero rule observations', () => {
    const frames = makeSwingAroundPeak(CLEAN_PEAK, CLEAN_REST)
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    // Exclude drift_from_baseline (only fires when comparing to a baseline)
    const ruleObs = obs.filter((o) => o.pattern !== 'drift_from_baseline')
    expect(
      ruleObs,
      `clean swing should produce no observations, got: ${ruleObs
        .map((o) => o.pattern)
        .join(', ')}`,
    ).toEqual([])
  })

  it('compound fault scenario surfaces both observations; primary picks the worst', () => {
    // Both cramped elbow AND locked knees — primary should be the
    // higher-severity one.
    const frames = makeSwingAroundPeak(
      { ...CLEAN_PEAK, right_elbow: 65, right_knee: 178 },
      { ...CLEAN_REST, right_knee: 178 },
    )
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    expect(obs.find((o) => o.pattern === 'cramped_elbow')).toBeDefined()
    expect(obs.find((o) => o.pattern === 'locked_knees')).toBeDefined()
    const primary = pickPrimary(obs)
    expect(primary).not.toBeNull()
    expect(['cramped_elbow', 'locked_knees']).toContain(primary!.pattern)
    const secondary = pickSecondary(obs, primary, 3)
    // Whatever wasn't primary should appear in secondary
    const allPatterns = [primary!, ...secondary].map((o) => o.pattern)
    expect(allPatterns).toContain('cramped_elbow')
    expect(allPatterns).toContain('locked_knees')
  })

  it('handedness flip: same fault on a left-handed forehand reports left side', () => {
    const frames = makeSwingAroundPeak(
      { ...CLEAN_PEAK, left_elbow: 70, right_elbow: 130 },
      CLEAN_REST,
    )
    const obs = extractObservations({
      todaySummary: frames,
      shotType: 'forehand',
      dominantHand: 'left',
    })
    const cramped = obs.find((o) => o.pattern === 'cramped_elbow')
    expect(cramped).toBeDefined()
    expect(cramped!.joint).toBe('left_elbow')
    expect(cramped!.todayValue).toBeLessThan(90)
  })

  it('one-handed backhand: right-handed player\'s backhand reads from the LEFT side', () => {
    // Right-hander's one-handed backhand uses the left arm. So a
    // cramped LEFT elbow on a backhand should fire (right-elbow data
    // is irrelevant for the dominantElbowKey lookup).
    const frames = makeSwingAroundPeak(
      { ...CLEAN_PEAK, left_elbow: 75, right_elbow: 130 },
      CLEAN_REST,
    )
    const obs = extractObservations({
      todaySummary: frames,
      shotType: 'backhand',
      dominantHand: 'right',
    })
    const cramped = obs.find((o) => o.pattern === 'cramped_elbow')
    expect(cramped).toBeDefined()
    expect(cramped!.joint).toBe('left_elbow')
  })

  it('signed angle math: a hip line that tilts +10 -> 0 -> -10 reports excursion ~20, not 10', () => {
    // Build the rotation as a sweep around 0°. The pre-fix Math.abs
    // would collapse this to 10°; the new signed math reports 20°.
    const angles: JointAngles[] = []
    for (let i = 0; i < 60; i++) {
      const t = i / 30 - 1 // -1 at start, 0 at middle, +1 at end
      angles.push({
        right_elbow: 130,
        right_knee: 160,
        // hip rotates +10 -> 0 -> -10 across the swing (signed).
        hip_rotation: -10 * t,
        // Trunk swings widely so the rule layer doesn't gate on it.
        trunk_rotation: -50 + (i < 30 ? i * 2 : (60 - i) * 2),
      })
    }
    const frames = makeFrames(angles)
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    // Hip excursion ≈ 20°, which is BELOW the new 25° threshold, so
    // insufficient_hip_excursion should fire and report ~20° as
    // todayValue. This single assertion proves the signed-math fix.
    const stuckHips = obs.find((o) => o.pattern === 'insufficient_hip_excursion')
    expect(stuckHips).toBeDefined()
    // 18-22° — accept a small margin for the unwrap path.
    expect(stuckHips!.todayValue).toBeGreaterThan(18)
    expect(stuckHips!.todayValue).toBeLessThan(22)
  })

  it('phantom-landmark immunity: low-fill RTMPose frames still surface real observations', () => {
    // Build a swing whose landmarks include 17 real high-vis IDs +
    // simulated "phantom" zeros at the unfilled BlazePose slots. The
    // pre-fix mean visibility would have been ~0.40 (gating off most
    // observations). With the whitelist, mean visibility on the
    // filled IDs is ~0.95 — observations should fire normally.
    const filledIds = [
      0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
    ]
    const phantomIds = [
      1, 3, 4, 6, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32,
    ]
    function makeRealisticLandmarks(): Landmark[] {
      const lms: Landmark[] = []
      for (const id of filledIds) {
        lms.push({
          id,
          name: `lm_${id}`,
          x: 0.5,
          y: 0.5,
          z: 0,
          visibility: 0.95,
        })
      }
      for (const id of phantomIds) {
        lms.push({
          id,
          name: `lm_${id}`,
          x: 0,
          y: 0,
          z: 0,
          visibility: 0,
        })
      }
      return lms
    }
    const peakAngles: JointAngles = { ...CLEAN_PEAK, right_elbow: 70 }
    const restAngles: JointAngles = CLEAN_REST
    const numFrames = 60
    const peakIdx = Math.floor(numFrames / 2)
    const frames: PoseFrame[] = []
    for (let i = 0; i < numFrames; i++) {
      const dist = Math.abs(i - peakIdx)
      const t = Math.max(0, 1 - dist / (numFrames / 2))
      const blended: JointAngles = {}
      const keys = new Set([
        ...Object.keys(peakAngles),
        ...Object.keys(restAngles),
      ]) as Set<keyof JointAngles>
      for (const k of keys) {
        const pv = peakAngles[k]
        const rv = restAngles[k]
        if (typeof pv === 'number' && typeof rv === 'number') {
          blended[k] = rv + (pv - rv) * t
        } else if (typeof pv === 'number') {
          blended[k] = pv
        } else if (typeof rv === 'number') {
          blended[k] = rv
        }
      }
      frames.push({
        frame_index: i,
        timestamp_ms: i * 33,
        landmarks: makeRealisticLandmarks(),
        joint_angles: blended,
      })
    }
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const cramped = obs.find((o) => o.pattern === 'cramped_elbow')
    expect(cramped, 'cramped_elbow should still fire despite phantom landmarks').toBeDefined()
  })

  it('pickPrimary chooses the most severe observation across patterns', () => {
    // Hand-craft observations and verify pickPrimary's tiebreaker.
    const obs: Observation[] = [
      {
        phase: 'contact',
        joint: 'right_elbow',
        pattern: 'cramped_elbow',
        severity: 'severe',
        confidence: 0.9,
        todayValue: 60,
      },
      {
        phase: 'loading',
        joint: 'right_knee',
        pattern: 'shallow_knee_load',
        severity: 'mild',
        confidence: 0.95,
        todayValue: 168,
      },
    ]
    const primary = pickPrimary(obs)
    expect(primary?.pattern).toBe('cramped_elbow')
  })

  it('pickPrimary returns null for an empty observation list', () => {
    expect(pickPrimary([])).toBeNull()
  })

  it('drift-from-baseline fires when today differs from baseline by >12 deg, not on small drift', () => {
    // Baseline: clean swing. Today: cramped elbow + drift on knee.
    const baseline = makeSwingAroundPeak(CLEAN_PEAK, CLEAN_REST)
    const today = makeSwingAroundPeak(
      { ...CLEAN_PEAK, right_elbow: 100 }, // 30° drift from 130
      CLEAN_REST,
    )
    const obs = extractObservations({
      todaySummary: today,
      baselineSummary: baseline,
      shotType: 'forehand',
    })
    const drifts = obs.filter((o) => o.pattern === 'drift_from_baseline')
    // At least one drift fires (the elbow-at-contact one). Hips/trunk
    // shouldn't drift since both swings have identical excursions.
    expect(drifts.length).toBeGreaterThan(0)
    const elbowDrift = drifts.find((o) => o.joint === 'right_elbow')
    expect(elbowDrift).toBeDefined()
    expect(elbowDrift!.driftMagnitude).toBeGreaterThan(20)
  })

  it('a swing identical to its baseline fires NO drift_from_baseline observations', () => {
    const baseline = makeSwingAroundPeak(CLEAN_PEAK, CLEAN_REST)
    // Build today's swing as a fresh copy of the same angle progression.
    const today = makeSwingAroundPeak(CLEAN_PEAK, CLEAN_REST)
    const obs = extractObservations({
      todaySummary: today,
      baselineSummary: baseline,
      shotType: 'forehand',
    })
    const drifts = obs.filter((o) => o.pattern === 'drift_from_baseline')
    expect(drifts).toEqual([])
  })

  it('observations include numeric todayValue tied to the input — not generic', () => {
    // For the same scenario at two different magnitudes, the
    // observation.todayValue should reflect the SPECIFIC angle,
    // proving the system is reading numbers off the swing rather
    // than emitting a fixed boilerplate.
    // Stay clear of the 90° cramped boundary so confidence clears the
    // CONFIDENCE_FLOOR. The point of the test is per-input numeric
    // accuracy; the cramped threshold itself is verified elsewhere.
    const fixtures: Array<{ elbow: number; expectMin: number; expectMax: number }> = [
      { elbow: 55, expectMin: 50, expectMax: 60 },
      { elbow: 65, expectMin: 60, expectMax: 70 },
      { elbow: 75, expectMin: 70, expectMax: 80 },
    ]
    for (const fixture of fixtures) {
      const frames = makeSwingAroundPeak(
        { ...CLEAN_PEAK, right_elbow: fixture.elbow },
        CLEAN_REST,
      )
      const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
      const cramped = obs.find((o) => o.pattern === 'cramped_elbow')
      expect(cramped, `elbow=${fixture.elbow} should fire cramped_elbow`).toBeDefined()
      expect(
        cramped!.todayValue,
        `elbow=${fixture.elbow} should report todayValue near ${fixture.elbow}`,
      ).toBeGreaterThanOrEqual(fixture.expectMin)
      expect(cramped!.todayValue).toBeLessThanOrEqual(fixture.expectMax)
    }
  })

  it('serve has a relaxed cramped-elbow threshold (140 instead of 90)', () => {
    // A 130° elbow at contact: cramped on a serve (below 140), but
    // FINE on a forehand (above 90).
    const peakAngles = { ...CLEAN_PEAK, right_elbow: 130 }
    const restAngles = CLEAN_REST
    const serveFrames = makeSwingAroundPeak(peakAngles, restAngles)
    const fhFrames = makeSwingAroundPeak(peakAngles, restAngles)
    const serveObs = extractObservations({ todaySummary: serveFrames, shotType: 'serve' })
    const fhObs = extractObservations({ todaySummary: fhFrames, shotType: 'forehand' })
    expect(serveObs.find((o) => o.pattern === 'cramped_elbow')).toBeDefined()
    expect(fhObs.find((o) => o.pattern === 'cramped_elbow')).toBeUndefined()
  })

  it('truncated_followthrough fires when trunk barely rotates past contact', () => {
    // Trunk peaks at 60°, then stays at 65° through the tail (delta
    // 5°, well below the 15° target).
    const angles: JointAngles[] = []
    for (let i = 0; i < 60; i++) {
      const peakIdx = 30
      const isPostContact = i > peakIdx
      angles.push({
        right_elbow: 130,
        right_knee: 160,
        hip_rotation: 30,
        trunk_rotation: isPostContact ? 65 : 10 + (i < peakIdx ? i * 2 : 60 - 2 * (i - peakIdx)),
      })
    }
    const frames = makeFrames(angles)
    const obs = extractObservations({ todaySummary: frames, shotType: 'forehand' })
    const trunc = obs.find((o) => o.pattern === 'truncated_followthrough')
    expect(trunc).toBeDefined()
    expect(trunc!.todayValue).toBeLessThan(15)
  })
})
