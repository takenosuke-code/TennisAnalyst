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
 *   - one issue per cue
 *   - no numbers, no degrees, no em-dashes, no "joint angle"
 *   - external focus where natural (attention on racket, ball, court target)
 *   - 2-3 outcome/holistic alternates included for variety
 *
 * Phrasings are split into two registers:
 *   - `plain`: beginner-friendly imperative coach voice ("Sit deeper",
 *     "Reach out"). Bodies and racket landmarks the player already knows.
 *   - `technical`: ADVISOR voice for skilled players. Strength + gap
 *     framing ("Your X is excellent, however Y shows room for
 *     improvement, which limits Z"). Plain biomech terms like
 *     "unit turn" and "kinetic chain" are welcome here. The LLM in
 *     the analyze route imitates whichever register matches the
 *     player's skill_tier; mixing the two confuses the model and
 *     produces half-implemented voice.
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
    technical: 'Your wrist position at contact is excellent, however your horizontal extension shows room for improvement, which jams the racket against the trunk and shortens your hitting window.',
    externalFocus: true,
  },
  {
    pattern: 'cramped_elbow',
    phase: 'contact',
    plain: 'Give yourself room to swing, step away from the ball just a little.',
    technical: 'Your timing on the take-back is solid, but your contact point is drifting behind the front hip, which crowds the elbow and steals the long arm you usually swing with.',
    externalFocus: true,
  },

  // ----- over_extended_elbow ---------------------------------------------
  {
    pattern: 'over_extended_elbow',
    phase: 'contact',
    plain: 'Keep the arm relaxed at contact, a soft bend through the ball.',
    technical: 'Your reach into the ball is professional and the contact window is forward, however the arm is locking out at contact, which kills the racket-head whip and drops the swing into push territory.',
    externalFocus: true,
  },
  {
    pattern: 'over_extended_elbow',
    phase: 'contact',
    plain: 'Let the racket head do the work, your arm guides, it does not push.',
    technical: 'Your spacing from the ball is good, but the racket arm is fully straightening at contact, which trades the relaxed lag of an expert swing for a stiff arm-driven push.',
    externalFocus: true,
  },

  // ----- shallow_knee_load -----------------------------------------------
  {
    pattern: 'shallow_knee_load',
    phase: 'loading',
    plain: 'Sit deeper into your back leg before the ball arrives.',
    technical: 'Your stance width on the load is professional and gives a strong base, however the legs are staying tall through the backswing, which leaves real room for improvement in using the ground for power.',
    externalFocus: true,
  },
  {
    pattern: 'shallow_knee_load',
    phase: 'loading',
    plain: 'Bend the knees and feel the ground push you up into the ball.',
    technical: 'Your trail leg is set early, but the load itself is shallow, which breaks the kinetic chain at the bottom and forces the arm to make up the missing power.',
    externalFocus: false,
  },

  // ----- locked_knees -----------------------------------------------------
  {
    pattern: 'locked_knees',
    phase: 'loading',
    plain: 'Bend your knees, get low and settle into the shot.',
    technical: 'Your stance is balanced and centered, however your legs are staying locked tall through the load, which removes the spring you would otherwise drive up off the ground.',
    externalFocus: true,
  },
  {
    pattern: 'locked_knees',
    phase: 'loading',
    plain: 'Soft knees, like you are about to jump.',
    technical: 'Your foot positioning is on point, but the knees are reading rigid through the load, which limits the athletic drive that an expert-level forehand relies on.',
    externalFocus: false,
  },

  // ----- insufficient_hip_excursion --------------------------------------
  {
    pattern: 'insufficient_hip_excursion',
    phase: 'preparation',
    plain: 'Turn your belt buckle toward the side fence, then drive it at the target.',
    technical: 'Your shoulder turn shows good intent, however the hips are barely rotating into the load, which shallows your unit turn and forces the upper body to start the swing without the pelvis leading.',
    externalFocus: true,
  },
  {
    pattern: 'insufficient_hip_excursion',
    phase: 'preparation',
    plain: 'Get your hips moving first, the arm comes along after.',
    technical: 'Your preparation timing is solid, but the hips are not coiling deeply enough into the side fence, which breaks the kinetic chain that should fire from the ground up.',
    externalFocus: false,
  },

  // ----- insufficient_trunk_excursion ------------------------------------
  {
    pattern: 'insufficient_trunk_excursion',
    phase: 'preparation',
    plain: 'Show your back to the net during your turn.',
    technical: 'Your hip coil is showing good depth, however the upper body is not rotating past the hips, which removes the stretch separation an expert-level uncoiling relies on.',
    externalFocus: true,
  },
  {
    pattern: 'insufficient_trunk_excursion',
    phase: 'preparation',
    plain: 'Turn the shoulders fully, point your chin at your front shoulder.',
    technical: 'Your unit turn starts on time, but the shoulders are stalling short of the side fence, which leaves the chest pointing forward at a moment when it should still be facing the side.',
    externalFocus: true,
  },

  // ----- insufficient_unit_turn ------------------------------------------
  {
    pattern: 'insufficient_unit_turn',
    phase: 'preparation',
    plain: 'Turn sideways early, racket and shoulders move together as one piece.',
    technical: 'Your reaction off the split step is clean, however the unit turn is shallow with the racket and shoulders moving separately, which places more stress on the arm rather than letting the kinetic chain do the work.',
    externalFocus: true,
  },

  // ----- truncated_followthrough -----------------------------------------
  {
    pattern: 'truncated_followthrough',
    phase: 'follow-through',
    plain: 'Finish with the racket wrapped over your other shoulder.',
    technical: 'Your balance through the follow-through is advanced, however the completion of the stroke is slightly abbreviated, which may be affecting your ability to pull the ball deep into the court.',
    externalFocus: true,
  },
  {
    pattern: 'truncated_followthrough',
    phase: 'follow-through',
    plain: 'Swing through the ball, do not stop at it.',
    technical: 'Your contact is solid and forward, but the racket is decelerating into the ball rather than past it, which abbreviates the follow-through and bleeds depth off the shot.',
    externalFocus: false,
  },

  // ----- weak_leg_drive --------------------------------------------------
  {
    pattern: 'weak_leg_drive',
    phase: 'contact',
    plain: 'Push up through your legs into the ball, finish a little taller than you started.',
    technical: 'Your stance width gives you a great foundation, however you are staying relatively tall through the backswing, which limits your ability to use the ground for power.',
    externalFocus: false,
  },
  {
    pattern: 'weak_leg_drive',
    phase: 'contact',
    plain: 'Sit deeper before you swing, then explode up through the shot.',
    technical: 'Your unit turn loads the upper body well, but the legs are not extending vertically through contact, which removes the kinetic chain push and forces the arm to carry the shot.',
    externalFocus: false,
  },

  // ----- short_pushout ---------------------------------------------------
  {
    pattern: 'short_pushout',
    phase: 'follow-through',
    plain: 'Send the racket out toward where you want the ball to go before it wraps over your shoulder.',
    technical: 'Your follow-through path comes across the body cleanly, however the racket is wrapping early without extending toward the target, which is currently affecting your ability to pull the ball deep through the court.',
    externalFocus: true,
  },
  {
    pattern: 'short_pushout',
    phase: 'follow-through',
    plain: 'Reach forward through the contact, do not stop the racket at the ball.',
    technical: 'Your contact point sits forward of the body, but the racket head is not pushing out along the line of the shot before it wraps, which shortens the hitting zone and softens depth.',
    externalFocus: true,
  },

  // ----- unstable_base ---------------------------------------------------
  {
    pattern: 'unstable_base',
    phase: 'contact',
    plain: 'Keep your head still through contact, like balancing a glass of water on top.',
    technical: 'Your stance load is set up well, however the head is drifting laterally through contact, which trades a stable platform for a moving one and makes the contact point inconsistent.',
    externalFocus: false,
  },
  {
    pattern: 'unstable_base',
    phase: 'contact',
    plain: 'Stay tall and centered, do not fall away as you hit.',
    technical: 'Your weight transfer through the shot reads on time, but the upper body is falling away as you hit, which destabilizes the base just at the moment it should be quiet over the front foot.',
    externalFocus: false,
  },

  // ----- drift_from_baseline ---------------------------------------------
  {
    pattern: 'drift_from_baseline',
    plain: 'Get back to the swing that felt easy on your best day.',
    technical: 'Most of the swing is matching your baseline shape, however the named area has drifted from the version that felt easy on your best day, which is currently the difference between today and your reference clip.',
    externalFocus: false,
  },
  {
    pattern: 'drift_from_baseline',
    plain: 'Trust the shape that worked before, do not invent a new one mid point.',
    technical: 'Your overall rhythm is on-pattern with the baseline, but the named area has shifted off the well-grooved version, which suggests rebuilding from the baseline timing rather than chasing a new shape mid-rally.',
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
