import { describe, it, expect } from 'vitest'
import { CUE_EXEMPLARS, findExemplars } from '@/lib/cueExemplars'
import type { DeviationPattern, Observation } from '@/lib/coachingObservations'

// ---------------------------------------------------------------------------
// Vocabulary cleanliness — the LLM-facing voice rules promise to keep these
// out of every cue string. The asserts are aggressive on purpose: a single
// digit or jargon term in any cue would leak through the post-filter and
// degrade the model's voice.
// ---------------------------------------------------------------------------

const FORBIDDEN_SUBSTRINGS = [
  // Biomech jargon
  'kinetic chain',
  'trunk_rotation',
  'hip_rotation',
  'joint angle',
  'pronation',
  'kinematic',
  // Em / en dashes — LLM voice rule.
  '\u2014',
  '\u2013',
]

describe('CUE_EXEMPLARS shape and vocabulary', () => {
  it('contains 15-20 exemplars (closed set)', () => {
    expect(CUE_EXEMPLARS.length).toBeGreaterThanOrEqual(15)
    expect(CUE_EXEMPLARS.length).toBeLessThanOrEqual(20)
  })

  it('every exemplar has plain and technical strings, both non-empty', () => {
    for (const ex of CUE_EXEMPLARS) {
      expect(typeof ex.plain).toBe('string')
      expect(typeof ex.technical).toBe('string')
      expect(ex.plain.trim().length).toBeGreaterThan(0)
      expect(ex.technical.trim().length).toBeGreaterThan(0)
    }
  })

  it('every exemplar has a known DeviationPattern', () => {
    const known: DeviationPattern[] = [
      'cramped_elbow',
      'over_extended_elbow',
      'shallow_knee_load',
      'locked_knees',
      'insufficient_hip_excursion',
      'insufficient_trunk_excursion',
      'insufficient_unit_turn',
      'truncated_followthrough',
      'drift_from_baseline',
    ]
    for (const ex of CUE_EXEMPLARS) {
      expect(known).toContain(ex.pattern)
    }
  })

  it('covers every required pattern at minimum once', () => {
    const required: DeviationPattern[] = [
      'cramped_elbow',
      'over_extended_elbow',
      'shallow_knee_load',
      'locked_knees',
      'insufficient_hip_excursion',
      'insufficient_trunk_excursion',
      'insufficient_unit_turn',
      'truncated_followthrough',
      'drift_from_baseline',
    ]
    for (const p of required) {
      const matching = CUE_EXEMPLARS.filter((ex) => ex.pattern === p)
      expect(matching.length).toBeGreaterThanOrEqual(1)
    }
  })

  it('NO digit characters appear in any cue string', () => {
    const digit = /\d/
    for (const ex of CUE_EXEMPLARS) {
      expect(digit.test(ex.plain)).toBe(false)
      expect(digit.test(ex.technical)).toBe(false)
    }
  })

  it('NO em-dashes, en-dashes, or biomech jargon in any cue string', () => {
    for (const ex of CUE_EXEMPLARS) {
      const fields = [ex.plain.toLowerCase(), ex.technical.toLowerCase()]
      for (const field of fields) {
        for (const banned of FORBIDDEN_SUBSTRINGS) {
          expect(field.includes(banned.toLowerCase())).toBe(false)
        }
      }
    }
  })

  it('mostly external-focus, with 2-3 outcome/holistic alternates', () => {
    const externalCount = CUE_EXEMPLARS.filter((e) => e.externalFocus).length
    const internalCount = CUE_EXEMPLARS.length - externalCount
    expect(externalCount).toBeGreaterThan(internalCount)
    expect(internalCount).toBeGreaterThanOrEqual(2)
  })

  it('all cues are second-person imperative (do not lead with "the player")', () => {
    for (const ex of CUE_EXEMPLARS) {
      const head = ex.plain.toLowerCase().slice(0, 30)
      expect(head).not.toContain('the player')
      expect(head).not.toContain('they should')
    }
  })
})

// ---------------------------------------------------------------------------
// findExemplars
// ---------------------------------------------------------------------------

describe('findExemplars', () => {
  function obs(p: Partial<Observation>): Observation {
    return {
      phase: 'contact',
      joint: 'right_elbow',
      pattern: 'cramped_elbow',
      severity: 'moderate',
      confidence: 0.8,
      todayValue: 80,
      ...p,
    }
  }

  it('returns exemplars matching the observation pattern', () => {
    const result = findExemplars(obs({ pattern: 'cramped_elbow' }), 3)
    expect(result.length).toBeGreaterThan(0)
    for (const r of result) {
      expect(r.pattern).toBe('cramped_elbow')
    }
  })

  it('caps result count at max', () => {
    const result = findExemplars(obs({ pattern: 'cramped_elbow' }), 1)
    expect(result.length).toBe(1)
  })

  it('returns empty when no exemplar matches the pattern', () => {
    // Use a fake pattern that's not in the set — cast through as DeviationPattern.
    const o = obs({ pattern: 'never_real_pattern' as DeviationPattern })
    const result = findExemplars(o, 3)
    expect(result).toEqual([])
  })

  it('phase-matched exemplars appear before phase-agnostic ones', () => {
    const o = obs({ pattern: 'truncated_followthrough', phase: 'follow-through' })
    const result = findExemplars(o, 3)
    expect(result.length).toBeGreaterThan(0)
    expect(result[0].phase).toBe('follow-through')
  })

  it('drift_from_baseline (no phase tag) returns sensible exemplars', () => {
    const o = obs({ pattern: 'drift_from_baseline', phase: 'loading' })
    const result = findExemplars(o, 3)
    expect(result.length).toBeGreaterThan(0)
    for (const r of result) {
      expect(r.pattern).toBe('drift_from_baseline')
    }
  })
})
