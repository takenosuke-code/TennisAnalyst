import type { Landmark, JointAngles, PoseFrame } from './supabase'
import { LANDMARK_INDICES } from './jointAngles'

/**
 * Camera-angle normalization helpers.
 *
 * `lib/jointAngles.ts:computeJointAngles` reads angles directly out of the
 * camera frame. That works for interior angles like elbow flexion (which
 * are intrinsically rotation-invariant in 2D — a dot-product angle does
 * not care which way "up" points), but the rotation-style readings
 * `hip_rotation` / `trunk_rotation` it emits are camera-frame absolute
 * angles, which means a clip filmed at a different roll/azimuth produces
 * different numbers for the same body posture.
 *
 * This module is the camera-robust *sibling* of `computeJointAngles` and
 * its swing-level helpers — callers pick which they want. Nothing here
 * replaces the existing functions; both paths must keep working because
 * the original outputs are stored in DB rows that downstream code reads
 * back unchanged.
 *
 * All functions are pure. No I/O, no globals.
 */

// -----------------------------------------------------------------------------
// Body-frame interior angles
// -----------------------------------------------------------------------------

/**
 * Visibility floor below which we treat a landmark as "absent" when
 * deciding whether the body frame is well-defined. Matches the cutoff
 * used in `lib/cameraAngle.ts` for the same reason — sub-0.5 visibility
 * coords are essentially noise in MediaPipe's output.
 */
const VIS_FLOOR = 0.5

/**
 * Minimum hip-line length (in normalized [0,1] coords) before we refuse
 * to construct a body frame. A pose facing nearly straight at the camera
 * collapses both hips onto the same x — the body-frame X-axis is
 * undefined there, so we return an empty angles object rather than
 * silently producing garbage from a near-zero divisor. The threshold is
 * chosen to be a few times larger than typical pose-jitter (~1e-3 in
 * normalized coords) but small enough that any clearly side-on or
 * three-quarter pose clears it.
 */
const MIN_HIP_LINE_LENGTH = 0.01

function getLandmark(landmarks: Landmark[], id: number): Landmark | undefined {
  return landmarks.find((l) => l.id === id)
}

function visible(lm: Landmark | undefined): lm is Landmark {
  return !!lm && lm.visibility >= VIS_FLOOR
}

/**
 * Project a 2D point into the body frame defined by `originX/originY`,
 * the X-axis basis vector `(xx, xy)` and the Y-axis basis vector
 * `(yx, yy)`. The basis is assumed orthonormal — we build it that way in
 * `computeBodyFrameAngles`. Returns the projected (u, v) coords.
 */
function projectToFrame(
  px: number,
  py: number,
  originX: number,
  originY: number,
  xx: number,
  xy: number,
  yx: number,
  yy: number,
): [number, number] {
  const dx = px - originX
  const dy = py - originY
  return [dx * xx + dy * xy, dx * yx + dy * yy]
}

function angleBetween2D(
  a: [number, number],
  b: [number, number],
): number {
  const dot = a[0] * b[0] + a[1] * b[1]
  const ma = Math.sqrt(a[0] ** 2 + a[1] ** 2)
  const mb = Math.sqrt(b[0] ** 2 + b[1] ** 2)
  if (ma === 0 || mb === 0) return 0
  const cos = Math.max(-1, Math.min(1, dot / (ma * mb)))
  return (Math.acos(cos) * 180) / Math.PI
}

/**
 * Compute interior joint angles in a body-relative coordinate system
 * instead of the camera frame.
 *
 * Body frame:
 *   - origin = midpoint of left and right hips
 *   - X-axis = unit vector from left_hip → right_hip
 *   - Y-axis = perpendicular to X, oriented to point toward the
 *             shoulder midpoint (i.e. "up" along the torso)
 *
 * Operating in 2D, so Z is omitted; in this frame the camera roll has
 * been factored out, which makes the angles invariant to in-plane
 * rotation by construction.
 *
 * Returns the same `JointAngles` shape as `computeJointAngles`. The
 * camera-frame rotation readings `hip_rotation` / `trunk_rotation`
 * become trivially zero in body-frame (the hip line *is* the X-axis,
 * the shoulder line projects to be parallel to it), so they are
 * deliberately omitted (left undefined). Callers that need rotation
 * excursion across a swing should use `computeRotationExcursion`,
 * which is camera-invariant via differencing.
 *
 * Sibling, not replacement: `computeJointAngles` is still the canonical
 * source for stored angles. Use this when you want camera-robust
 * comparisons across clips filmed at different angles.
 *
 * Edge cases:
 *   - Either hip missing / below visibility floor → `{}` (frame
 *     undefined).
 *   - Both hips coincident or near-coincident (`|hipVec| <
 *     MIN_HIP_LINE_LENGTH`) → `{}`. Front/back-facing poses collapse
 *     the hip line to ~0 length; we refuse to invent a basis there
 *     instead of dividing by an unstable denominator.
 */
export function computeBodyFrameAngles(landmarks: Landmark[]): JointAngles {
  const lHip = getLandmark(landmarks, LANDMARK_INDICES.LEFT_HIP)
  const rHip = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_HIP)

  // Need both hips visible to define the body frame at all.
  if (!visible(lHip) || !visible(rHip)) return {}

  const hipDx = rHip.x - lHip.x
  const hipDy = rHip.y - lHip.y
  const hipLen = Math.sqrt(hipDx * hipDx + hipDy * hipDy)
  if (hipLen < MIN_HIP_LINE_LENGTH) return {}

  // Body-frame origin = hip midpoint.
  const ox = (lHip.x + rHip.x) / 2
  const oy = (lHip.y + rHip.y) / 2

  // X-axis: unit vector from left_hip to right_hip.
  const xx = hipDx / hipLen
  const xy = hipDy / hipLen

  // Provisional Y-axis: 90° rotation of X. In a y-down image space
  // (which is what BlazePose normalized coords use), the perpendicular
  // (-X.y, X.x) tends to point one way and (X.y, -X.x) the other; we
  // choose the orientation that points *toward* the shoulder midpoint
  // (i.e. up the torso) so the body frame is consistent across poses.
  const lShoulder = getLandmark(landmarks, LANDMARK_INDICES.LEFT_SHOULDER)
  const rShoulder = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_SHOULDER)

  // Pick whichever shoulder(s) we have to find the shoulder midpoint
  // direction. If neither is available we still have to return *some*
  // body frame, so default the Y-axis to (-xy, xx) — but in practice
  // every realistic frame has at least one shoulder.
  let shoulderMidX = ox
  let shoulderMidY = oy - 1 // dummy "up" if no shoulders
  if (lShoulder && rShoulder) {
    shoulderMidX = (lShoulder.x + rShoulder.x) / 2
    shoulderMidY = (lShoulder.y + rShoulder.y) / 2
  } else if (lShoulder) {
    shoulderMidX = lShoulder.x
    shoulderMidY = lShoulder.y
  } else if (rShoulder) {
    shoulderMidX = rShoulder.x
    shoulderMidY = rShoulder.y
  }
  const toShoulderX = shoulderMidX - ox
  const toShoulderY = shoulderMidY - oy

  // Two perpendiculars to (xx, xy): (-xy, xx) and (xy, -xx). Pick the
  // one whose dot with (toShoulderX, toShoulderY) is positive.
  let yx = -xy
  let yy = xx
  const dotPerp = yx * toShoulderX + yy * toShoulderY
  if (dotPerp < 0) {
    yx = xy
    yy = -xx
  }

  // Project all landmarks we need into the body frame, then compute
  // interior angles via the same dot-product method as
  // `computeJointAngles` — but in body-frame coords. Note that 2D
  // interior angles are intrinsically rotation-invariant under any
  // rigid transform, so the values produced here equal those from
  // `computeJointAngles` whenever the source landmarks are simply
  // rotated; the body-frame projection is the *right* abstraction
  // (insulates from camera roll, leaves room for body-relative
  // measurements like shoulder elevation in future), and is also a
  // direct read of the body-relative geometry callers can rely on.
  const proj = (lm: Landmark | undefined): [number, number] | null =>
    lm ? projectToFrame(lm.x, lm.y, ox, oy, xx, xy, yx, yy) : null

  const lShoulderP = proj(lShoulder)
  const rShoulderP = proj(rShoulder)
  const lElbowP = proj(getLandmark(landmarks, LANDMARK_INDICES.LEFT_ELBOW))
  const rElbowP = proj(getLandmark(landmarks, LANDMARK_INDICES.RIGHT_ELBOW))
  const lWristP = proj(getLandmark(landmarks, LANDMARK_INDICES.LEFT_WRIST))
  const rWristP = proj(getLandmark(landmarks, LANDMARK_INDICES.RIGHT_WRIST))
  const lKneeP = proj(getLandmark(landmarks, LANDMARK_INDICES.LEFT_KNEE))
  const rKneeP = proj(getLandmark(landmarks, LANDMARK_INDICES.RIGHT_KNEE))
  const lAnkleP = proj(getLandmark(landmarks, LANDMARK_INDICES.LEFT_ANKLE))
  const rAnkleP = proj(getLandmark(landmarks, LANDMARK_INDICES.RIGHT_ANKLE))
  // hip points in the body frame (after projection); needed for knee
  // angles. By construction lHip projects to (-hipLen/2, 0) and rHip to
  // (+hipLen/2, 0), but proj() through projectToFrame() gives identical
  // numbers and keeps the path uniform.
  const lHipP = proj(lHip)
  const rHipP = proj(rHip)

  const sub = (
    a: [number, number],
    b: [number, number],
  ): [number, number] => [a[0] - b[0], a[1] - b[1]]

  const angles: JointAngles = {}

  if (rShoulderP && rElbowP && rWristP) {
    angles.right_elbow = angleBetween2D(
      sub(rShoulderP, rElbowP),
      sub(rWristP, rElbowP),
    )
  }
  if (lShoulderP && lElbowP && lWristP) {
    angles.left_elbow = angleBetween2D(
      sub(lShoulderP, lElbowP),
      sub(lWristP, lElbowP),
    )
  }
  if (lShoulderP && rShoulderP && rElbowP) {
    angles.right_shoulder = angleBetween2D(
      sub(lShoulderP, rShoulderP),
      sub(rElbowP, rShoulderP),
    )
  }
  if (lShoulderP && rShoulderP && lElbowP) {
    angles.left_shoulder = angleBetween2D(
      sub(rShoulderP, lShoulderP),
      sub(lElbowP, lShoulderP),
    )
  }
  if (rHipP && rKneeP && rAnkleP) {
    angles.right_knee = angleBetween2D(
      sub(rHipP, rKneeP),
      sub(rAnkleP, rKneeP),
    )
  }
  if (lHipP && lKneeP && lAnkleP) {
    angles.left_knee = angleBetween2D(
      sub(lHipP, lKneeP),
      sub(lAnkleP, lKneeP),
    )
  }

  // hip_rotation / trunk_rotation are deliberately omitted: in the
  // body frame the hip line *is* the X-axis (rotation = 0), and the
  // shoulder line is approximately parallel to it for a normal upright
  // posture (also ~0). Leaving them undefined makes it explicit that
  // rotation is not a body-frame concept — callers wanting rotation
  // signal across a swing should use `computeRotationExcursion`.
  return angles
}

// -----------------------------------------------------------------------------
// Rotation excursion across a swing
// -----------------------------------------------------------------------------

/**
 * Excursion (max − min) of camera-frame `hip_rotation` and
 * `trunk_rotation` across all frames in a swing.
 *
 * The per-frame readings are camera-frame absolute angles, so any one
 * of them is biased by camera azimuth. **The difference** between max
 * and min, however, is camera-invariant: an additive offset applied to
 * every frame cancels under subtraction. That makes excursion the
 * cleanest angle-axis-style measurement we can extract from the
 * existing angle stream without re-running pose extraction.
 *
 * `confidence` reflects how many frames carried valid (non-null)
 * rotation values:
 *   - validFraction ≥ 0.5 → confidence = validFraction (∈ [0.5, 1])
 *   - validFraction < 0.5 → confidence = 0 (excursion is unreliable;
 *     the caller should treat the numbers as noise)
 *
 * Empty `frames` → all zeros. No throwing, no NaN — downstream callers
 * compose this into prompts and UIs that don't tolerate either.
 */
export function computeRotationExcursion(
  frames: PoseFrame[],
): { hipExcursion: number; trunkExcursion: number; confidence: number } {
  if (frames.length === 0) {
    return { hipExcursion: 0, trunkExcursion: 0, confidence: 0 }
  }

  const hips: number[] = []
  const trunks: number[] = []
  for (const f of frames) {
    const h = f.joint_angles.hip_rotation
    const t = f.joint_angles.trunk_rotation
    if (typeof h === 'number' && Number.isFinite(h)) hips.push(h)
    if (typeof t === 'number' && Number.isFinite(t)) trunks.push(t)
  }

  // Confidence: take the *worse* of the two streams, since the function
  // emits both numbers and the caller needs to know whether either one
  // is trustworthy. A swing with no trunk readings is just as
  // unreliable as one with no hip readings.
  const total = frames.length
  const hipFrac = hips.length / total
  const trunkFrac = trunks.length / total
  const validFrac = Math.min(hipFrac, trunkFrac)
  const confidence = validFrac >= 0.5 ? validFrac : 0

  const excursion = (xs: number[]): number =>
    xs.length === 0 ? 0 : Math.max(...xs) - Math.min(...xs)

  return {
    hipExcursion: excursion(hips),
    trunkExcursion: excursion(trunks),
    confidence,
  }
}

// -----------------------------------------------------------------------------
// Camera similarity heuristic
// -----------------------------------------------------------------------------

export type CameraClass = 'side' | 'behind' | 'front' | 'mixed'

/**
 * Per-frame visibility floor for the landmarks the similarity classifier
 * inspects. Same constant as the body-frame helper; deliberately re-used
 * to keep one knob.
 */
const SIM_VIS_FLOOR = VIS_FLOOR

/**
 * Threshold below which we consider a shoulder spread "narrow" (and the
 * pose roughly side-on). Mirrors the value used in
 * `lib/cameraAngle.ts:SIDE_ON_MAX_SHOULDER_SPREAD` to keep the two
 * classifiers consistent — anything tuned out of phase between them
 * tends to surface as confusing UX where the gating differs between
 * runtime warnings and post-hoc baseline comparison.
 */
const SIDE_SHOULDER_SPREAD_MAX = 0.06

/**
 * Above this shoulder spread we consider the pose "facing" the camera
 * (front or behind, distinguished by nose visibility). Matches
 * `cameraAngle.FRONT_ON_MIN_SHOULDER_SPREAD`.
 */
const FACING_SHOULDER_SPREAD_MIN = 0.18

/**
 * Slope-difference dampening: a 45° slope difference between two clips
 * drives the similarity multiplier to zero, a 0° difference leaves it at
 * 1. Below 45° the multiplier is `max(0, 1 − |Δslope|/45)`. This is the
 * "soft" component of similarity — bucket adjacency is the hard component.
 */
const SLOPE_DAMP_DEG = 45

type ClipFeatures = {
  cls: CameraClass
  /** Mean slope of the shoulder line vs vertical, in degrees, ∈ [0,90]. */
  meanSlopeDeg: number
  /** Mean shoulder-line / hip-line ratio, when both visible. */
  meanForeshortening: number
  /** How many frames produced a usable feature reading. */
  validCount: number
}

function classifyClip(frames: PoseFrame[]): ClipFeatures {
  let nNoseVisible = 0
  let nNoseSampled = 0
  let totalShoulderSpread = 0
  let nShoulderSpread = 0
  let totalForeshortening = 0
  let nForeshortening = 0
  let totalSlope = 0
  let nSlope = 0
  let nSideHints = 0

  for (const frame of frames) {
    const lms = frame.landmarks
    const nose = lms.find((l) => l.id === LANDMARK_INDICES.NOSE)
    const lSh = lms.find((l) => l.id === LANDMARK_INDICES.LEFT_SHOULDER)
    const rSh = lms.find((l) => l.id === LANDMARK_INDICES.RIGHT_SHOULDER)
    const lHip = lms.find((l) => l.id === LANDMARK_INDICES.LEFT_HIP)
    const rHip = lms.find((l) => l.id === LANDMARK_INDICES.RIGHT_HIP)

    const lShOk = !!lSh && lSh.visibility >= SIM_VIS_FLOOR
    const rShOk = !!rSh && rSh.visibility >= SIM_VIS_FLOOR

    if (nose) {
      nNoseSampled += 1
      if (nose.visibility >= SIM_VIS_FLOOR) nNoseVisible += 1
    }

    // One-shoulder-only frames are a strong side hint regardless of
    // foreshortening — record it so we can short-circuit borderline
    // bucketing later.
    if (lShOk !== rShOk) nSideHints += 1

    if (lShOk && rShOk) {
      const dx = lSh.x - rSh.x
      const dy = lSh.y - rSh.y
      const shoulderLen = Math.sqrt(dx * dx + dy * dy)
      totalShoulderSpread += Math.abs(dx)
      nShoulderSpread += 1

      // Slope vs vertical: 0° = perfectly vertical shoulder line (true
      // profile, body edge faces camera), 90° = perfectly horizontal
      // (square-on). Using `atan2(|dx|, |dy|)` so the result lives in
      // [0, 90] regardless of which shoulder is left.
      const slope = (Math.atan2(Math.abs(dx), Math.abs(dy)) * 180) / Math.PI
      totalSlope += slope
      nSlope += 1

      if (
        lHip &&
        rHip &&
        lHip.visibility >= SIM_VIS_FLOOR &&
        rHip.visibility >= SIM_VIS_FLOOR
      ) {
        const hdx = lHip.x - rHip.x
        const hdy = lHip.y - rHip.y
        const hipLen = Math.sqrt(hdx * hdx + hdy * hdy)
        if (hipLen >= MIN_HIP_LINE_LENGTH) {
          totalForeshortening += shoulderLen / hipLen
          nForeshortening += 1
        }
      }
    } else if (lShOk || rShOk) {
      // One-shoulder frame has no usable spread but still informs the
      // class (handled via nSideHints above).
    }
  }

  const meanShoulderSpread =
    nShoulderSpread > 0 ? totalShoulderSpread / nShoulderSpread : 0
  const meanForeshortening =
    nForeshortening > 0 ? totalForeshortening / nForeshortening : 0
  const meanSlopeDeg = nSlope > 0 ? totalSlope / nSlope : 0
  const noseVisFrac =
    nNoseSampled > 0 ? nNoseVisible / nNoseSampled : 0

  // Bucket. The order matters — side-on is the dominant signal we need
  // to identify first because it's where coaching actually works.
  let cls: CameraClass
  const sideHintFrac = frames.length > 0 ? nSideHints / frames.length : 0
  if (
    sideHintFrac > 0.4 ||
    (nShoulderSpread > 0 && meanShoulderSpread < SIDE_SHOULDER_SPREAD_MAX)
  ) {
    cls = 'side'
  } else if (
    nShoulderSpread > 0 &&
    meanShoulderSpread > FACING_SHOULDER_SPREAD_MIN
  ) {
    // Wide spread → facing camera. Distinguish front vs behind by nose
    // visibility: face is visible from the front, hidden from behind.
    cls = noseVisFrac > 0.5 ? 'front' : 'behind'
  } else {
    cls = 'mixed'
  }

  return {
    cls,
    meanSlopeDeg,
    meanForeshortening,
    validCount: nShoulderSpread + nSideHints,
  }
}

/**
 * Adjacency table: `mixed` is adjacent to every directional class
 * (it's the in-between three-quarter view); the directional classes
 * are *not* adjacent to each other (side and front look totally
 * different to MediaPipe and to a human). Same-class similarity is
 * 1.0; adjacent is 0.6; non-adjacent is 0.2.
 */
function adjacencyScore(a: CameraClass, b: CameraClass): number {
  if (a === b) return 1
  if (a === 'mixed' || b === 'mixed') return 0.6
  // side, front, behind are mutually non-adjacent.
  return 0.2
}

/**
 * Estimate visual camera-angle similarity between two pose-frame
 * sequences. Output:
 *   - `score` ∈ [0, 1], where 1 means visually identical camera
 *     framing (same azimuth class + same shoulder slope).
 *   - `classA` / `classB` are the bucket each clip lands in
 *     (`'side'`, `'behind'`, `'front'`, `'mixed'`), reported back so
 *     callers can show a clearer "this clip is X, baseline is Y"
 *     warning instead of an opaque score.
 *
 * Heuristic, not a model. Uses the *averaged* shoulder-line vs hip-line
 * foreshortening ratio and the averaged shoulder-line slope vs vertical
 * across each clip. Bucketing is via shoulder spread + nose visibility:
 *   - narrow shoulder spread or many one-shoulder frames → `'side'`
 *   - wide shoulder spread + nose visible → `'front'`
 *   - wide shoulder spread + nose hidden → `'behind'`
 *   - everything else → `'mixed'`
 *
 * Score combines a hard adjacency component (1.0 / 0.6 / 0.2 by class
 * pair) with a soft slope dampener that pulls the score down by the
 * shoulder-slope difference (0° diff = 1.0×, 45°+ diff = 0.0×).
 *
 * Caller-side gating: the spec says ~0.55 is the right "warn the user"
 * threshold — adjacent classes (0.6) survive small slope differences,
 * non-adjacent classes (0.2) cannot.
 */
export function computeCameraSimilarity(
  a: PoseFrame[],
  b: PoseFrame[],
): { score: number; classA: CameraClass; classB: CameraClass } {
  const fa = classifyClip(a)
  const fb = classifyClip(b)

  // Empty / no-signal clips: cannot compare. Return a low score with
  // the buckets we did produce (each will be the bucket fallback,
  // typically 'mixed') so the caller can show a coherent warning.
  if (fa.validCount === 0 || fb.validCount === 0) {
    return { score: 0, classA: fa.cls, classB: fb.cls }
  }

  const baseScore = adjacencyScore(fa.cls, fb.cls)
  const slopeDiff = Math.abs(fa.meanSlopeDeg - fb.meanSlopeDeg)
  const slopeDamp = Math.max(0, 1 - slopeDiff / SLOPE_DAMP_DEG)
  const score = Math.max(0, Math.min(1, baseScore * slopeDamp))

  return { score, classA: fa.cls, classB: fb.cls }
}
