/**
 * Real-clip regression test.
 *
 * Loads the cached RTMPose extraction of an Alcaraz forehand
 * (broadcast side-on clip, 82 frames @ 30fps) and runs the FULL
 * client-side analysis pipeline:
 *   1. Recompute joint_angles from landmarks via the fixed
 *      computeJointAngles (so this test exercises the signed-atan2
 *      hip/trunk and the wrist-phantom-landmark guard, not whatever
 *      angles the Python extractor wrote into the JSON).
 *   2. Run extractObservations with the player's known profile
 *      (right-handed forehand).
 *   3. Pick primary + secondary.
 *
 * Verdict: a clean Alcaraz forehand should produce few-to-no rule
 * observations. ANY observation that fires is either real (he's
 * elite, but pose extraction is noisy) or a regression we want to
 * see surfaced.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import {
  computeJointAngles,
  RTMPOSE_FILLED_LANDMARK_IDS,
} from '@/lib/jointAngles'
import {
  extractObservations,
  pickPrimary,
  pickSecondary,
} from '@/lib/coachingObservations'
import type { Landmark, PoseFrame } from '@/lib/supabase'

const FIXTURE_PATH = join(
  process.cwd(),
  '__tests__/fixtures/alcaraz_forehand_pose.json',
)

interface FixtureJSON {
  fps_sampled: number
  frame_count: number
  frames: Array<{
    frame_index: number
    timestamp_ms: number
    landmarks: Landmark[]
    joint_angles?: Record<string, number>
    racket_head?: { x: number; y: number; confidence: number } | null
  }>
}

function loadFixture(): FixtureJSON {
  return JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')) as FixtureJSON
}

function recomputeFrames(fixture: FixtureJSON): PoseFrame[] {
  return fixture.frames.map((f) => ({
    frame_index: f.frame_index,
    timestamp_ms: f.timestamp_ms,
    landmarks: f.landmarks,
    joint_angles: computeJointAngles(f.landmarks),
  }))
}

describe('Real-clip regression: Alcaraz forehand', () => {
  it('fixture loads with 82 frames and the expected schema', () => {
    const fx = loadFixture()
    expect(fx.frame_count).toBe(82)
    expect(fx.frames).toHaveLength(82)
    // Each frame should have 33 BlazePose-33 landmarks (with the 16
    // unfilled slots present at visibility=0).
    expect(fx.frames[0].landmarks).toHaveLength(33)
  })

  it('mean visibility across the 17 RTMPose-filled landmarks is high (>0.7)', () => {
    // Pre-fix code averaged over all 33 slots and got ~0.40 here even
    // on a clean clip. Post-fix uses RTMPOSE_FILLED_LANDMARK_IDS.
    const fx = loadFixture()
    const filled = new Set<number>(RTMPOSE_FILLED_LANDMARK_IDS)
    let sum = 0
    let n = 0
    for (const f of fx.frames) {
      for (const l of f.landmarks) {
        if (filled.has(l.id) && typeof l.visibility === 'number') {
          sum += l.visibility
          n += 1
        }
      }
    }
    const meanVis = sum / n
    expect(meanVis).toBeGreaterThan(0.7)
    expect(meanVis).toBeLessThanOrEqual(1.0)
  })

  it('clean Alcaraz forehand produces few rule observations (<= 2 of {cramped_elbow, locked_knees, insufficient_*})', () => {
    const fx = loadFixture()
    const frames = recomputeFrames(fx)
    const obs = extractObservations({
      todaySummary: frames,
      shotType: 'forehand',
      dominantHand: 'right',
    })

    // The patterns we'd expect a CLEAN swing to NOT fire. Drift
    // doesn't apply (no baseline supplied).
    const faultPatterns = [
      'cramped_elbow',
      'over_extended_elbow',
      'locked_knees',
      'shallow_knee_load',
      'insufficient_hip_excursion',
      'insufficient_trunk_excursion',
      'insufficient_unit_turn',
      'truncated_followthrough',
    ] as const
    const ruleObs = obs.filter((o) =>
      (faultPatterns as readonly string[]).includes(o.pattern),
    )

    // Print to stdout so the test output documents what was found.
    // eslint-disable-next-line no-console
    console.log(
      `[real-clip] Alcaraz forehand fired ${ruleObs.length} rule observation(s):`,
      ruleObs.map((o) => `${o.pattern}@${o.joint}=${Math.round(o.todayValue)}° (conf=${o.confidence.toFixed(2)})`),
    )

    // Hard upper bound: more than 2 fault patterns on a clean elite
    // forehand is a regression signal.
    expect(
      ruleObs.length,
      'Alcaraz forehand should not light up multiple fault patterns',
    ).toBeLessThanOrEqual(2)
  })

  it('does NOT report cramped_elbow or insufficient_hip_excursion (Alcaraz extends fully + rotates fully)', () => {
    // These two patterns are the ones the user has explicitly pushed
    // back on as "the AI is wrong" — Alcaraz's elbow at contact is
    // ~150° (well above 90), and his hip rotation easily clears 25°.
    // If either fires here, that's a clear false positive.
    const fx = loadFixture()
    const frames = recomputeFrames(fx)
    const obs = extractObservations({
      todaySummary: frames,
      shotType: 'forehand',
      dominantHand: 'right',
    })
    const cramped = obs.find((o) => o.pattern === 'cramped_elbow')
    expect(cramped, `unexpected cramped_elbow: ${cramped && JSON.stringify(cramped)}`).toBeUndefined()
    const stuckHips = obs.find((o) => o.pattern === 'insufficient_hip_excursion')
    expect(stuckHips, `unexpected stuck-hips: ${stuckHips && JSON.stringify(stuckHips)}`).toBeUndefined()
  })

  it('any observation that DID fire has confidence above the floor and a real numeric reading', () => {
    const fx = loadFixture()
    const frames = recomputeFrames(fx)
    const obs = extractObservations({
      todaySummary: frames,
      shotType: 'forehand',
      dominantHand: 'right',
    })
    for (const o of obs) {
      // Confidence must clear the runtime floor (CONFIDENCE_FLOOR=0.2).
      expect(o.confidence, `${o.pattern} confidence below floor`).toBeGreaterThanOrEqual(0.2)
      // todayValue must be a real number (not the old phantom-(0,0) noise).
      expect(typeof o.todayValue).toBe('number')
      expect(Number.isFinite(o.todayValue)).toBe(true)
    }
  })

  it('pickPrimary returns null OR a single observation tied to the actual swing', () => {
    const fx = loadFixture()
    const frames = recomputeFrames(fx)
    const obs = extractObservations({
      todaySummary: frames,
      shotType: 'forehand',
      dominantHand: 'right',
    })
    const primary = pickPrimary(obs)
    if (primary !== null) {
      // If something fires, log it so a human can sanity-check.
      // eslint-disable-next-line no-console
      console.log(
        `[real-clip] primary cue:`,
        `${primary.pattern} @ ${primary.joint} = ${Math.round(primary.todayValue)}° (severity=${primary.severity}, conf=${primary.confidence.toFixed(2)})`,
      )
      expect(primary.confidence).toBeGreaterThanOrEqual(0.2)
      expect(['contact', 'loading', 'preparation', 'follow-through']).toContain(primary.phase)
    }
    const secondary = pickSecondary(obs, primary, 3)
    expect(secondary.length).toBeLessThanOrEqual(3)
    if (primary && secondary.length > 0) {
      // Secondary must not duplicate primary.
      expect(secondary.find((s) => s === primary)).toBeUndefined()
    }
  })
})
