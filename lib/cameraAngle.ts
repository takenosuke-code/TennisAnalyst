import type { Landmark } from './supabase'

/**
 * Heuristic 2D camera-angle classifier.
 *
 * For tennis-stroke analysis we want the player filmed roughly side-on
 * (perpendicular to the swing path) — that's the only angle where the
 * shoulder/hip rotation, elbow extension, and trunk lean we use to grade
 * a swing actually project usefully into 2D. A front-on or back-on shot
 * collapses the swing into the depth axis, where MediaPipe's z-coord is
 * unreliable, so cues based on those frames are guesswork.
 *
 * This classifier exists to gate the live coaching feed: when the camera
 * is wrong, we suppress detector input upstream so no cue ever fires.
 *
 * Inputs are MediaPipe's raw normalized landmarks (x,y in [0,1] of the
 * frame). No model: pure cheap geometry over the shoulders + hips, run
 * every detection tick.
 */

// MediaPipe Pose landmark indices used here. Duplicated rather than
// imported from `lib/jointAngles` to keep this file tiny and free of
// cross-module deps (mirrors what `lib/poseSmoothing` does for the same
// reason).
const LM_LEFT_SHOULDER = 11
const LM_RIGHT_SHOULDER = 12
const LM_LEFT_HIP = 23
const LM_RIGHT_HIP = 24

/**
 * Per-landmark visibility floor. Below this, MediaPipe's coords are
 * essentially noise and the classifier returns `'unknown'` — we can't
 * tell the angle from a frame where the body isn't really detected.
 */
const VIS_FLOOR = 0.5

/**
 * Shoulder x-spread (|leftShoulder.x - rightShoulder.x|) below this is
 * "narrow" — i.e. one shoulder occludes the other, which is what side-on
 * filming looks like (player's torso edge points at the camera). Tuned
 * loose: a fully side-on player can still show ~0.04-0.05 of x-spread
 * because the far shoulder pokes through, so 0.06 catches the typical
 * profile shot without flagging slight rotations.
 */
const SIDE_ON_MAX_SHOULDER_SPREAD = 0.06

/**
 * Visibility asymmetry between the two shoulders above this is also a
 * strong "side-on" signal — when the body blocks the far shoulder, MediaPipe
 * drops its visibility well below the near shoulder's. 0.4 means one
 * shoulder can be at ~0.95 while the other is at ~0.55 (which is roughly
 * what we observe in side-on tennis clips).
 */
const SIDE_ON_MIN_VIS_ASYMMETRY = 0.4

/**
 * Shoulder x-spread above this is "wide" — both shoulders are clearly
 * separated horizontally, which only happens when the player faces (or
 * backs) the camera. Tuned from sample fixtures: a player squarely
 * facing the camera at typical filming distance shows ~0.18-0.25 of
 * x-spread between shoulders.
 */
const FRONT_ON_MIN_SHOULDER_SPREAD = 0.18

/**
 * Both shoulders must be roughly equally visible for `'front-on'` to be
 * the correct classification — otherwise the wide spread is more likely
 * a partial-profile (oblique) view that happens to clear the spread
 * threshold. 0.2 keeps us strict: front-on means both shoulders confidently
 * tracked.
 */
const FRONT_ON_MAX_VIS_ASYMMETRY = 0.2

export type CameraAngle = 'side-on' | 'oblique' | 'front-on' | 'unknown'

/**
 * Classify the camera angle from a single frame's landmarks. Returns:
 *
 * - `'unknown'` only when we have effectively no shoulder signal at all
 *   (neither shoulder above the visibility floor). One missing shoulder
 *   is the *signal* for side-on, not a reason to bail.
 * - `'side-on'` if only one shoulder is confidently tracked (the body
 *   blocks the other), OR if both shoulders project to nearly the same x.
 * - `'front-on'` if both shoulders are confidently tracked, spread wide,
 *   and roughly equal in visibility (square-on to camera).
 * - `'oblique'` otherwise — the three-quarter view. Coaching is allowed
 *   here; the angle is workable.
 *
 * Hips are not required for classification — the shoulder geometry is
 * what reads camera angle, and many side-on tennis frames have one hip
 * occluded by the racket arm or shorts/skirt fabric.
 */
export function classifyCameraAngle(landmarks: Landmark[]): CameraAngle {
  if (landmarks.length === 0) return 'unknown'

  const byId = new Map<number, Landmark>()
  for (const lm of landmarks) byId.set(lm.id, lm)

  const lShoulder = byId.get(LM_LEFT_SHOULDER)
  const rShoulder = byId.get(LM_RIGHT_SHOULDER)

  const lShoulderOk = !!lShoulder && lShoulder.visibility >= VIS_FLOOR
  const rShoulderOk = !!rShoulder && rShoulder.visibility >= VIS_FLOOR

  // No shoulder info at all — can't classify. This is the only path that
  // returns 'unknown'; one occluded shoulder is *expected* in side-on
  // filming, not a failure mode.
  if (!lShoulderOk && !rShoulderOk) return 'unknown'

  // Only one shoulder confidently tracked → the other is occluded by the
  // body, which is exactly what side-on looks like.
  if (lShoulderOk !== rShoulderOk) return 'side-on'

  // Both shoulders confidently tracked — use spread to discriminate.
  const shoulderSpread = Math.abs(lShoulder!.x - rShoulder!.x)
  const visAsymmetry = Math.abs(lShoulder!.visibility - rShoulder!.visibility)

  // Side-on with both shoulders technically visible: shoulders project
  // to nearly the same x (true profile) OR there's still a clear visibility
  // gap between them.
  if (
    shoulderSpread < SIDE_ON_MAX_SHOULDER_SPREAD ||
    visAsymmetry > SIDE_ON_MIN_VIS_ASYMMETRY
  ) {
    return 'side-on'
  }

  // Front-on: wide separation AND symmetric visibility.
  if (
    shoulderSpread > FRONT_ON_MIN_SHOULDER_SPREAD &&
    visAsymmetry < FRONT_ON_MAX_VIS_ASYMMETRY
  ) {
    return 'front-on'
  }

  // Anything in between — partial profile, three-quarter view. Coaching
  // is allowed here; the angle is good enough for some signal.
  return 'oblique'
}
