/**
 * Cue Exemplars
 *
 * A small, hand-curated set of coaching cues drawn from real on-court
 * vocabulary (Mencinger, Saviano, Mouratoglou registers). The route picks 3-5
 * exemplars matching the chosen Observation and shows them to the LLM as a
 * voice template — the LLM doesn't quote them verbatim, it imitates the
 * register.
 *
 * Voice rules these exemplars all obey:
 *   - imperative, second person ("Sit deeper", "Reach out")
 *   - one issue per cue
 *   - no numbers, no em-dashes, no biomech jargon
 *   - external focus by default (attention on racket, ball, or court target)
 *   - 2-3 outcome/holistic alternates included for variety
 *
 * Phrasings are split into `plain` (beginner-friendly: bodies and racket
 * landmarks the player already knows) and `technical` (skilled-player register:
 * stance/load, contact window, racket lag). The route picks one or the other
 * based on the player's skill_tier.
 */

import type { DeviationPattern, Observation, Phase } from './coachingObservations'
import type { ShotType } from './shotTypeConfig'

export interface CueExemplar {
  pattern: DeviationPattern
  phase?: Phase
  shotType?: ShotType
  plain: string
  technical: string
  // True when the cue directs attention away from the body (Wulf 2013 external
  // focus). False for the rare "outcome / holistic" alternates.
  externalFocus: boolean
}

// ---------------------------------------------------------------------------
// CUE_EXEMPLARS
//
// Coverage requirement (per spec):
//   cramped_elbow, over_extended_elbow, shallow_knee_load, locked_knees,
//   insufficient_hip_excursion, insufficient_trunk_excursion,
//   insufficient_unit_turn, truncated_followthrough, drift_from_baseline.
// ---------------------------------------------------------------------------

export const CUE_EXEMPLARS: CueExemplar[] = [
  // ----- cramped_elbow ----------------------------------------------------
  {
    pattern: 'cramped_elbow',
    phase: 'contact',
    plain: 'Reach out and meet the ball in front of your hip.',
    technical: 'Catch the ball in the contact window forward of the trunk, racket arm long.',
    externalFocus: true,
  },
  {
    pattern: 'cramped_elbow',
    phase: 'contact',
    plain: 'Give yourself room to swing, step away from the ball just a little.',
    technical: 'Take the ball earlier so the contact point clears the front hip.',
    externalFocus: true,
  },

  // ----- over_extended_elbow ---------------------------------------------
  {
    pattern: 'over_extended_elbow',
    phase: 'contact',
    plain: 'Keep the arm relaxed at contact, a soft bend through the ball.',
    technical: 'Hold a soft elbow through the contact window, keep the arm forgiving.',
    externalFocus: true,
  },
  {
    pattern: 'over_extended_elbow',
    phase: 'contact',
    plain: 'Let the racket head do the work, your arm guides, it does not push.',
    technical: 'Relax the racket arm at contact, the racket head whips through on its own.',
    externalFocus: true,
  },

  // ----- shallow_knee_load -----------------------------------------------
  {
    pattern: 'shallow_knee_load',
    phase: 'loading',
    plain: 'Sit deeper into your back leg before the ball arrives.',
    technical: 'Load the trail leg before contact, drive up through it.',
    externalFocus: true,
  },
  {
    pattern: 'shallow_knee_load',
    phase: 'loading',
    plain: 'Bend the knees and feel the ground push you up into the ball.',
    technical: 'Settle into a deeper stance load, then explode up through the shot.',
    externalFocus: false,
  },

  // ----- locked_knees -----------------------------------------------------
  {
    pattern: 'locked_knees',
    phase: 'loading',
    plain: 'Bend your knees, get low and settle into the shot.',
    technical: 'Unlock the knees on the load, drop your hips into a low stance.',
    externalFocus: true,
  },
  {
    pattern: 'locked_knees',
    phase: 'loading',
    plain: 'Soft knees, like you are about to jump.',
    technical: 'Stay athletic in the legs, ready to drive up through contact.',
    externalFocus: false,
  },

  // ----- insufficient_hip_excursion --------------------------------------
  {
    pattern: 'insufficient_hip_excursion',
    phase: 'preparation',
    plain: 'Turn your belt buckle toward the side fence, then drive it at the target.',
    technical: 'Coil the hips early, lead the forward swing with the pelvis.',
    externalFocus: true,
  },
  {
    pattern: 'insufficient_hip_excursion',
    phase: 'preparation',
    plain: 'Get your hips moving first, the arm comes along after.',
    technical: 'Start the swing from the ground up, hips fire before the trunk.',
    externalFocus: false,
  },

  // ----- insufficient_trunk_excursion ------------------------------------
  {
    pattern: 'insufficient_trunk_excursion',
    phase: 'preparation',
    plain: 'Show your back to the net during your turn.',
    technical: 'Coil the upper body further than the hips, store the stretch.',
    externalFocus: true,
  },
  {
    pattern: 'insufficient_trunk_excursion',
    phase: 'preparation',
    plain: 'Turn the shoulders fully, point your chin at your front shoulder.',
    technical: 'Complete the unit turn, chest pointing at the side fence before forward swing.',
    externalFocus: true,
  },

  // ----- insufficient_unit_turn ------------------------------------------
  {
    pattern: 'insufficient_unit_turn',
    phase: 'preparation',
    plain: 'Turn sideways early, racket and shoulders move together as one piece.',
    technical: 'Make the unit turn the first move after the split step, racket stays linked.',
    externalFocus: true,
  },

  // ----- truncated_followthrough -----------------------------------------
  {
    pattern: 'truncated_followthrough',
    phase: 'follow-through',
    plain: 'Finish with the racket wrapped over your other shoulder.',
    technical: 'Carry the swing into a full follow through, racket clears past the trunk.',
    externalFocus: true,
  },
  {
    pattern: 'truncated_followthrough',
    phase: 'follow-through',
    plain: 'Swing through the ball, do not stop at it.',
    technical: 'Accelerate past contact, the deceleration happens after the racket clears.',
    externalFocus: false,
  },

  // ----- drift_from_baseline ---------------------------------------------
  {
    pattern: 'drift_from_baseline',
    plain: 'Get back to the swing that felt easy on your best day.',
    technical: 'Anchor on the baseline shape of this stroke, match its rhythm and load.',
    externalFocus: false,
  },
  {
    pattern: 'drift_from_baseline',
    plain: 'Trust the shape that worked before, do not invent a new one mid point.',
    technical: 'Reset to the baseline timing, let the well grooved version come back.',
    externalFocus: false,
  },
]

// ---------------------------------------------------------------------------
// findExemplars
//
// Finds up to `max` exemplars matching a given Observation. Match priority:
//   1. pattern + phase + shotType
//   2. pattern + phase
//   3. pattern only
// Stable ordering — preserves the curated order in CUE_EXEMPLARS.
// ---------------------------------------------------------------------------

export function findExemplars(
  observation: Observation,
  max = 3,
): CueExemplar[] {
  const exact: CueExemplar[] = []
  const phaseOnly: CueExemplar[] = []
  const patternOnly: CueExemplar[] = []
  for (const ex of CUE_EXEMPLARS) {
    if (ex.pattern !== observation.pattern) continue
    if (ex.phase && ex.phase === observation.phase) {
      exact.push(ex)
    } else if (!ex.phase) {
      patternOnly.push(ex)
    } else {
      phaseOnly.push(ex)
    }
  }
  const merged: CueExemplar[] = []
  for (const list of [exact, patternOnly, phaseOnly]) {
    for (const ex of list) {
      if (!merged.includes(ex)) merged.push(ex)
      if (merged.length >= max) return merged
    }
  }
  return merged.slice(0, max)
}
