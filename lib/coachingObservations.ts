/**
 * Coaching Observations
 *
 * Pure functions that turn pose data + (optional) baseline pose data into a
 * typed list of `Observation` rows. The route then picks one primary +
 * up to 3 secondary observations and feeds them to the LLM.
 *
 * Detection rules mirror lib/biomechanics-reference.ts AMATEUR_MISTAKES_REFERENCE
 * and lib/shotTypeConfig.ts:mistakeChecks, but we emit typed rows rather than
 * coaching strings — the LLM phrases the cue, we just say what we saw.
 *
 * IMPORTANT: hip_rotation and trunk_rotation are computed in jointAngles.ts as
 * the absolute angle of the hip/shoulder line versus horizontal. That number
 * is camera-noisy: a small camera tilt swings it 10–20 degrees on its own, so
 * a single reading is not actionable. Instead we compute EXCURSION across the
 * detected swing — max minus min — which is a much better proxy for how
 * much the player actually rotated. This is why "insufficient_hip_excursion"
 * and "insufficient_trunk_excursion" are the patterns we emit, not raw-angle
 * thresholds.
 *
 * Elbow / knee / shoulder use the standard interior-angle rules at the peak
 * frame (proxy for contact). Body-frame normalization is jointAngles' job;
 * we just consume what's there.
 */

import type { JointAngles, PoseFrame } from './supabase'
import { detectSwings, RTMPOSE_FILLED_LANDMARK_IDS } from './jointAngles'
import { detectStrokes } from './swingDetect'
import { scoreStrokes } from './strokeQuality'
import type { ShotType } from './shotTypeConfig'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type DeviationPattern =
  | 'cramped_elbow'
  | 'over_extended_elbow'
  | 'shallow_knee_load'
  | 'locked_knees'
  | 'insufficient_hip_excursion'
  | 'insufficient_trunk_excursion'
  | 'insufficient_unit_turn'
  | 'truncated_followthrough'
  // Vertical hip-midpoint rise from loading frame to contact frame,
  // normalized by torso length. Proxy for whether the legs drove the
  // shot. Below ~3% torso = no leg push.
  | 'weak_leg_drive'
  // Horizontal wrist displacement from contact to T+150ms (5 frames at
  // 30fps), normalized by shoulder width. Proxy for "racket extends out
  // toward target" — the coaching cue the user explicitly called out
  // when their follow-through ended at the shoulder without pushing
  // the ball forward. Below ~0.3 shoulder-widths = short pushout.
  | 'short_pushout'
  // Horizontal nose drift across an 8-frame window centered on contact,
  // normalized by shoulder width. Proxy for postural stability — head
  // sway through contact is the universal coaching cue for "still
  // platform". Above ~0.15 shoulder-widths = unstable.
  | 'unstable_base'
  | 'drift_from_baseline'
  // Baseline-compare only. Wrist (proxy for racket head) at the contact
  // frame is significantly higher / lower today vs baseline, normalized
  // by torso length and referenced to the shoulder midpoint so camera-
  // height differences cancel. The LLM uses this as CONTEXT for body-
  // mechanic differences: a higher contact point can EXPLAIN a lighter
  // knee load, because the player didn't have to drive up as much. Not
  // a fault on its own — the player rarely controls bounce height.
  | 'contact_height_higher'
  | 'contact_height_lower'
  // Baseline-compare only. Distance from wrist to body center at the
  // contact frame, normalized by shoulder width. "Jammed" = today's
  // contact is much closer to the body than baseline (rushed or late
  // swing); "extended" = much further out (reaching). Like contact
  // height, this is a context signal that explains body-mechanic
  // differences (e.g. a jammed contact can explain a cramped elbow).
  | 'contact_position_jammed'
  | 'contact_position_extended'

export type Phase = 'preparation' | 'loading' | 'contact' | 'follow-through' | 'finish'

export type Severity = 'mild' | 'moderate' | 'severe'

export interface Observation {
  phase: Phase
  joint: string
  pattern: DeviationPattern
  severity: Severity
  confidence: number
  todayValue: number
  baselineValue?: number
  driftMagnitude?: number
}

export interface ExtractionInput {
  todaySummary: PoseFrame[]
  baselineSummary?: PoseFrame[]
  shotType: ShotType | null
  // Player handedness from the auth profile. Drives which side of the
  // body the observation rules read (a left-hander's "dominant elbow"
  // is left_elbow on a forehand, right_elbow on a one-handed backhand).
  // Defaults to 'right' when null/missing — the historical assumption
  // baked into the original rule code.
  dominantHand?: 'right' | 'left' | null
  // When true, todaySummary and baselineSummary are ALREADY scoped to a
  // single swing (e.g. baseline-compare mode pre-slices via
  // detectSwings on the page). The rules skip the internal
  // detectSwings re-detection in primarySwingFrames and treat the
  // entire input as the primary swing slice.
  //
  // 2026-05 — added to fix the "1° today excursion" bug: the server
  // was re-detecting a peak inside an already-sliced swing and
  // re-slicing around it, often cutting prep frames entirely. The
  // rotation channels read flat across the post-coil portion, so
  // hip / trunk / shoulder excursion all reported ~1° simultaneously.
  // The flag is opt-in (default false) — solo /analyze still relies on
  // server detection to pick the primary among multiple swings.
  isPreSliced?: boolean
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Severity score used for ranking. Higher = worse.
export const SEVERITY_RANK: Record<Severity, number> = {
  mild: 1,
  moderate: 2,
  severe: 3,
}

// Confidence floor. Below this, the observation is dropped (we don't trust it).
//
// 2026-05 — lowered 0.6 → 0.2. The previous value rejected legitimate
// borderline observations on visibly clear footage: confidence is the
// product of visibility × threshold-margin × applicability, and even a
// "moderate" deviation rarely produced a margin > 0.7. Result: even a
// well-tracked, modestly-cramped elbow at 80° (threshold 90°) failed
// the floor and got silently dropped, which cascaded into the empty-
// state branch firing on clear shots from the side. A 0.2 floor now
// only rejects truly poor measurements (vis < 0.2 or basically-no-
// margin observations), which is what "we can't read this" actually
// means. The route also no longer hard-blocks on empty observations —
// it warns and runs a generic coaching call, so this floor is a soft
// gate rather than a kill switch.
export const CONFIDENCE_FLOOR = 0.2

// ---------------------------------------------------------------------------
// computeConfidence
// ---------------------------------------------------------------------------

/**
 * Combine three signals into a single 0..1 confidence score:
 *   - landmarkVisibility: how well the camera can see the joints (0..1).
 *   - thresholdMargin: how far past the rule's threshold the value sits,
 *     normalized to 0..1. A value barely past threshold = low margin = low
 *     confidence; one well past = high margin.
 *   - ruleApplicability: 1.0 for clean cases, lower for edge conditions
 *     (unknown shot type → 0.5, etc).
 *
 * Multiplicative because all three must hold for us to trust the obs.
 */
export function computeConfidence(args: {
  landmarkVisibility: number
  thresholdMargin: number
  ruleApplicability: number
}): number {
  const v = clamp01(args.landmarkVisibility)
  const m = clamp01(args.thresholdMargin)
  const r = clamp01(args.ruleApplicability)
  return v * m * r
}

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0
  if (x < 0) return 0
  if (x > 1) return 1
  return x
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Mean visibility across the landmarks RTMPose actually fills (see
 * lib/jointAngles.ts:RTMPOSE_FILLED_LANDMARK_IDS). The other 16 of 33
 * BlazePose-33 slots ride along at visibility=0 on every Railway clip;
 * averaging across all 33 deflated a clean clip's mean to ~0.40, which
 * tripped the meanVis<0.4 short-circuit AND collapsed the multiplicative
 * confidence gate downstream so most legitimate observations were
 * silently dropped. When the frame has no landmarks (e.g. test fixtures
 * that only set joint_angles), default to 1.0.
 */
const RTMPOSE_FILLED_SET = new Set<number>(RTMPOSE_FILLED_LANDMARK_IDS)

function frameVisibility(frame: PoseFrame): number {
  if (!frame.landmarks || frame.landmarks.length === 0) return 1.0
  let sum = 0
  let n = 0
  for (const l of frame.landmarks) {
    // Only count landmarks the extractor is supposed to fill. Accept
    // landmarks whose id matches the whitelist; if a frame omits the
    // id field (older fixtures), fall back to non-zero (x,y) as the
    // proxy for "real landmark".
    const idOk =
      typeof l.id === 'number'
        ? RTMPOSE_FILLED_SET.has(l.id)
        : !(l.x === 0 && l.y === 0)
    if (!idOk) continue
    if (typeof l.visibility === 'number' && Number.isFinite(l.visibility)) {
      sum += l.visibility
      n += 1
    }
  }
  if (n === 0) return 1.0
  return sum / n
}

function meanVisibility(frames: PoseFrame[]): number {
  if (frames.length === 0) return 0
  let sum = 0
  for (const f of frames) sum += frameVisibility(f)
  return sum / frames.length
}

/**
 * Pick the peak frame from the primary detected swing. The peak is the frame
 * with the most joint-angle activity, which is a reasonable proxy for the
 * contact phase.
 */
/**
 * Of the SwingSegments in `swings`, pick the index of the best stroke
 * to coach. "Best" = highest non-NaN score from scoreStrokes (peak
 * wrist speed + kinetic-chain timing + wrist-angle variance), which
 * filters out strokes flagged as low-visibility / camera-pan / too-
 * short. If every stroke scores NaN (all rejected) or scoring fails,
 * fall back to swings[0] for backwards compatibility.
 *
 * Was: always picked swings[0]. On a rally upload that meant the user
 * got coaching on whatever the first detected swing was — often a
 * warm-up stroke at the start of the clip.
 */
function pickPrimarySwingIndex(
  swings: ReturnType<typeof detectSwings>,
  allFrames: PoseFrame[],
): number {
  if (swings.length <= 1) return 0
  try {
    const detectedStrokes = detectStrokes(allFrames)
    if (detectedStrokes.length !== swings.length) return 0
    const scores = scoreStrokes(detectedStrokes, allFrames)
    let bestIdx = -1
    let bestScore = -Infinity
    for (let i = 0; i < scores.length; i++) {
      const s = scores[i]
      if (!s || s.rejected || !Number.isFinite(s.score)) continue
      if (s.score > bestScore) {
        bestScore = s.score
        bestIdx = i
      }
    }
    return bestIdx >= 0 ? bestIdx : 0
  } catch {
    return 0
  }
}

function pickPeakFrame(frames: PoseFrame[], isPreSliced = false): PoseFrame | null {
  if (frames.length === 0) return null
  // When the input is already a single swing slice, the peak is in the
  // middle of the array (Voronoi-bounded slices are roughly centered on
  // the peak by construction). Skip the redundant detectSwings call
  // that would otherwise re-find a peak inside the already-sliced data.
  if (isPreSliced) return frames[Math.floor(frames.length / 2)] ?? null
  const swings = detectSwings(frames)
  if (swings.length === 0) return frames[Math.floor(frames.length / 2)] ?? null
  const primary = swings[pickPrimarySwingIndex(swings, frames)]
  if (!primary) return frames[Math.floor(frames.length / 2)] ?? null
  const peakIdxInSlice = primary.peakFrame - primary.startFrame
  const idx = Math.max(0, Math.min(primary.frames.length - 1, peakIdxInSlice))
  return primary.frames[idx] ?? null
}

/**
 * Excursion of an angle channel: total arc length traveled across the
 * primary swing window. Signed atan2 angles can wrap at ±180°, and a
 * symmetric rotation that crosses square-to-camera (+10° → 0° → -10°)
 * has true excursion 20° but a naive max-min reports 10° if the line
 * doesn't actually cross the wrap (or 360° if it does). Summing
 * |shortDelta| across consecutive frames is correct in both cases.
 */
function shortAngleDelta(a: number, b: number): number {
  let d = b - a
  while (d > 180) d -= 360
  while (d <= -180) d += 360
  return d
}

function angleExcursion(frames: PoseFrame[], key: keyof JointAngles): number | null {
  // Build a continuous (unwrapped) series via shortDelta, then take
  // max - min. Unwrapping handles the ±180° wrap; max-min on the
  // unwrapped series stays robust to per-frame jitter.
  const unwrapped: number[] = []
  let prev: number | null = null
  for (const f of frames) {
    const v = f.joint_angles?.[key]
    if (typeof v !== 'number' || !Number.isFinite(v)) continue
    const u: number = prev === null ? v : prev + shortAngleDelta(prev, v)
    unwrapped.push(u)
    prev = u
  }
  if (unwrapped.length === 0) return null
  let lo = Infinity
  let hi = -Infinity
  for (const v of unwrapped) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return hi - lo
}

function angleMin(frames: PoseFrame[], key: keyof JointAngles): number | null {
  let lo = Infinity
  for (const f of frames) {
    const v = f.joint_angles?.[key]
    if (typeof v === 'number' && Number.isFinite(v) && v < lo) lo = v
  }
  return Number.isFinite(lo) ? lo : null
}

function angleMax(frames: PoseFrame[], key: keyof JointAngles): number | null {
  let hi = -Infinity
  for (const f of frames) {
    const v = f.joint_angles?.[key]
    if (typeof v === 'number' && Number.isFinite(v) && v > hi) hi = v
  }
  return Number.isFinite(hi) ? hi : null
}

/**
 * Get the swing-window of frames the rules should evaluate against. We pull
 * the primary detected swing slice; if no swing is detected (very short clip)
 * we fall back to the whole frames array.
 *
 * When `isPreSliced` is true the caller has already scoped `frames` to a
 * single swing (e.g. baseline-compare mode pre-slices today via
 * detectSwings on the page, and saved baselines are stored as a single
 * swing-trim). Skipping the internal re-detection avoids the "swing
 * inside a swing" trap where the second pass finds a peak in the middle
 * of the already-sliced data and re-slices around it, often cutting prep
 * frames entirely and producing a flat hip/trunk excursion.
 */
function primarySwingFrames(frames: PoseFrame[], isPreSliced = false): PoseFrame[] {
  if (frames.length === 0) return frames
  if (isPreSliced) return frames
  const swings = detectSwings(frames)
  if (swings.length === 0) return frames
  const primary = swings[pickPrimarySwingIndex(swings, frames)]
  if (!primary || primary.frames.length === 0) return frames
  return primary.frames
}

function severityFromMargin(margin: number): Severity {
  // margin is "how far past threshold" normalized to 0..1.
  if (margin >= 0.7) return 'severe'
  if (margin >= 0.35) return 'moderate'
  return 'mild'
}

// Margin normalizer for excursion-style rules: target = the minimum desired
// excursion, actual = the value seen. Returns 0..1, larger when actual is
// much smaller than target.
function shortfallMargin(actual: number, target: number): number {
  if (target <= 0) return 0
  const shortfall = Math.max(0, target - actual)
  return clamp01(shortfall / target)
}

// Margin normalizer for over-thresholds: how far past threshold the value
// sits, normalized by a "fully past" anchor. e.g. threshold=170, anchor=180
// -> a reading of 180 is margin=1.0; 175 is 0.5.
function overshootMargin(actual: number, threshold: number, anchor: number): number {
  const span = anchor - threshold
  if (span <= 0) return 0
  return clamp01((actual - threshold) / span)
}

// Margin normalizer for under-thresholds: how far below threshold, normalized
// by an anchor below it. e.g. threshold=90 (cramped_elbow), anchor=60 -> a
// reading of 60 is margin=1.0; 75 is 0.5.
function undershootMargin(actual: number, threshold: number, anchor: number): number {
  const span = threshold - anchor
  if (span <= 0) return 0
  return clamp01((threshold - actual) / span)
}

// ---------------------------------------------------------------------------
// Detection rules
// ---------------------------------------------------------------------------

/**
 * Returns the dominant elbow + knee keys for a shot, taking handedness
 * into account. A right-handed forehand uses the right side; a left-
 * handed forehand uses the left. Backhand flips: a right-hander's
 * backhand uses the left elbow (one-handed) / both arms (two-handed),
 * a left-hander's backhand uses the right.
 */
function dominantSideForShot(
  shot: ShotType | null,
  handedness: 'right' | 'left' | null | undefined,
): 'right' | 'left' {
  const hand = handedness ?? 'right'
  if (shot === 'backhand') return hand === 'right' ? 'left' : 'right'
  return hand
}

// Take a resolved dominant side (already accounts for handedness +
// shot type via dominantSideForShot) and return the matching joint
// key. Callers pull side from ctx.dominantSide so handedness threads
// through every rule consistently.
function dominantElbowKey(side: 'right' | 'left'): keyof JointAngles {
  return side === 'right' ? 'right_elbow' : 'left_elbow'
}

function dominantKneeKey(side: 'right' | 'left'): keyof JointAngles {
  return side === 'right' ? 'right_knee' : 'left_knee'
}

interface RuleContext {
  shotFrames: PoseFrame[]
  peakFrame: PoseFrame
  meanVis: number
  applicability: number
  dominantSide: 'right' | 'left'
}

function buildRuleContext(input: ExtractionInput): RuleContext | null {
  const frames = input.todaySummary
  if (!frames || frames.length === 0) return null
  const shotFrames = primarySwingFrames(frames, input.isPreSliced ?? false)
  const peakFrame = pickPeakFrame(frames, input.isPreSliced ?? false)
  if (!peakFrame) return null
  const meanVis = meanVisibility(shotFrames)
  // Unknown shot type -> halve applicability (we don't know which rules really
  // apply). Volley/slice are rules-applicable but with reduced weight.
  let applicability = 1.0
  if (input.shotType === null) applicability = 0.5
  else if (input.shotType === 'volley' || input.shotType === 'slice') applicability = 0.8
  const dominantSide: 'right' | 'left' = dominantSideForShot(
    input.shotType,
    input.dominantHand,
  )
  return { shotFrames, peakFrame, meanVis, applicability, dominantSide }
}

/**
 * Elbow rules detect cramped (<90°) and over-extended (>175°) at contact.
 * We read the elbow angle at the peak (contact-proxy) frame, not the
 * extremum across the entire slice — on a normal forehand the elbow is
 * most-flexed during prep (~70-90°) and most-extended through contact
 * (~140-170°), so slice-min would surface the prep angle and falsely
 * flag every clean swing as cramped. peakFrame is the swing detector's
 * peak-activity frame, which lines up with the contact moment closely
 * enough for this rule.
 *
 * Serve cramped threshold relaxed to 140° because serves expect near-full
 * extension at contact.
 */
function checkElbowAtContact(
  ctx: RuleContext,
  shot: ShotType | null,
): Observation[] {
  const out: Observation[] = []
  const key = dominantElbowKey(ctx.dominantSide)

  const elbow = ctx.peakFrame.joint_angles?.[key]
  if (typeof elbow !== 'number' || !Number.isFinite(elbow)) return out

  // Cramped elbow: under 90° on groundstrokes, under 140° on the serve.
  const crampedThreshold = shot === 'serve' ? 140 : 90
  const crampedAnchor = shot === 'serve' ? 100 : 60
  if (elbow < crampedThreshold) {
    const margin = undershootMargin(elbow, crampedThreshold, crampedAnchor)
    const confidence = computeConfidence({
      landmarkVisibility: ctx.meanVis,
      thresholdMargin: margin,
      ruleApplicability: ctx.applicability,
    })
    if (confidence >= CONFIDENCE_FLOOR) {
      out.push({
        phase: 'contact',
        joint: `${ctx.dominantSide}_elbow`,
        pattern: 'cramped_elbow',
        severity: severityFromMargin(margin),
        confidence,
        todayValue: elbow,
      })
    }
  }

  // Over-extended elbow: 175° at contact = locked straight.
  if (elbow > 175) {
    const margin = overshootMargin(elbow, 175, 180)
    const confidence = computeConfidence({
      landmarkVisibility: ctx.meanVis,
      thresholdMargin: margin,
      ruleApplicability: ctx.applicability,
    })
    if (confidence >= CONFIDENCE_FLOOR) {
      out.push({
        phase: 'contact',
        joint: `${ctx.dominantSide}_elbow`,
        pattern: 'over_extended_elbow',
        severity: severityFromMargin(margin),
        confidence,
        todayValue: elbow,
      })
    }
  }

  return out
}

/**
 * Knee rules at the peak frame:
 *   - locked_knees: angle > 170° (groundstrokes) or > 165° (serve)
 *   - shallow_knee_load: angle is in the high-but-not-locked band, suggesting
 *     the player is not loading enough.
 *
 * Shallow load uses (threshold - anchor) margins where threshold is
 * "ideal-max" and anchor is "fully unloaded" (180°).
 */
/**
 * Knee load rules. Use the MINIMUM knee angle over the swing slice — the
 * deepest the player loaded. If even at the deepest point they're standing
 * tall, that's the loading fault.
 */
function checkKneeLoad(
  ctx: RuleContext,
  shot: ShotType | null,
): Observation[] {
  const out: Observation[] = []
  const key = dominantKneeKey(ctx.dominantSide)
  const v = angleMin(ctx.shotFrames, key)
  if (v === null) return out

  const lockedThreshold = shot === 'serve' ? 165 : 170
  // Shallow-load band sits between ideal-max (~160) and locked (170).
  const shallowFloor = shot === 'serve' ? 155 : 160

  if (v > lockedThreshold) {
    const margin = overshootMargin(v, lockedThreshold, 180)
    const confidence = computeConfidence({
      landmarkVisibility: ctx.meanVis,
      thresholdMargin: margin,
      ruleApplicability: ctx.applicability,
    })
    if (confidence >= CONFIDENCE_FLOOR) {
      out.push({
        phase: 'loading',
        joint: `${ctx.dominantSide}_knee`,
        pattern: 'locked_knees',
        severity: severityFromMargin(margin),
        confidence,
        todayValue: v,
      })
    }
    return out
  }

  if (v > shallowFloor && v <= lockedThreshold) {
    const margin = overshootMargin(v, shallowFloor, lockedThreshold)
    const confidence = computeConfidence({
      landmarkVisibility: ctx.meanVis,
      thresholdMargin: margin,
      ruleApplicability: ctx.applicability,
    })
    if (confidence >= CONFIDENCE_FLOOR) {
      out.push({
        phase: 'loading',
        joint: `${ctx.dominantSide}_knee`,
        pattern: 'shallow_knee_load',
        severity: severityFromMargin(margin),
        confidence,
        todayValue: v,
      })
    }
  }

  return out
}

/**
 * Hip / trunk excursion: max - min across the full primary-swing slice.
 * A small excursion = the player barely turned, which is the "arming the
 * ball" / "no unit turn" pattern.
 *
 * Targets per AMATEUR_MISTAKES_REFERENCE: hip excursion ≥ 40°, trunk excursion
 * ≥ 60°. Below those = insufficient.
 */
function checkRotationExcursion(ctx: RuleContext): Observation[] {
  const out: Observation[] = []

  const hipExcursion = angleExcursion(ctx.shotFrames, 'hip_rotation')
  if (hipExcursion !== null) {
    // Target relaxed 40 -> 25 to align with biomech-reference.ts:
    // "Insufficient hip rotation: hip_rot stays flat (<20° change)".
    // Old 40° punished clean amateur swings that rotated 30-35°.
    const target = 25
    if (hipExcursion < target) {
      const margin = shortfallMargin(hipExcursion, target)
      const confidence = computeConfidence({
        landmarkVisibility: ctx.meanVis,
        thresholdMargin: margin,
        ruleApplicability: ctx.applicability,
      })
      if (confidence >= CONFIDENCE_FLOOR) {
        out.push({
          phase: 'loading',
          joint: 'hips',
          pattern: 'insufficient_hip_excursion',
          severity: severityFromMargin(margin),
          confidence,
          todayValue: hipExcursion,
        })
      }
    }
  }

  const trunkExcursion = angleExcursion(ctx.shotFrames, 'trunk_rotation')
  if (trunkExcursion !== null) {
    // Target relaxed 60 -> 30 to align with biomech-reference.ts:
    // "Insufficient trunk rotation: trunk_rot changes <15° between
    // loading and contact". Old 60° punished any amateur whose
    // shoulders rotated less than an elite-grade 90° turn.
    const target = 30
    if (trunkExcursion < target) {
      const margin = shortfallMargin(trunkExcursion, target)
      const confidence = computeConfidence({
        landmarkVisibility: ctx.meanVis,
        thresholdMargin: margin,
        ruleApplicability: ctx.applicability,
      })
      if (confidence >= CONFIDENCE_FLOOR) {
        out.push({
          phase: 'loading',
          joint: 'shoulders',
          pattern: 'insufficient_trunk_excursion',
          severity: severityFromMargin(margin),
          confidence,
          todayValue: trunkExcursion,
        })
      }
    }
  }

  // Insufficient unit turn: BOTH excursions are small AND close together
  // (no separation). Treated as a separate, more emphatic pattern.
  // Tightened (25/35 -> 18/22) to stay below the relaxed individual
  // thresholds — fires only when both rotations are at or below the
  // bare biomech minima (hip <20°, trunk <15°).
  if (hipExcursion !== null && trunkExcursion !== null) {
    if (hipExcursion < 18 && trunkExcursion < 22) {
      const margin = clamp01(
        Math.max(shortfallMargin(hipExcursion, 18), shortfallMargin(trunkExcursion, 22)),
      )
      const confidence = computeConfidence({
        landmarkVisibility: ctx.meanVis,
        thresholdMargin: margin,
        ruleApplicability: ctx.applicability,
      })
      if (confidence >= CONFIDENCE_FLOOR) {
        out.push({
          phase: 'preparation',
          joint: 'torso',
          pattern: 'insufficient_unit_turn',
          severity: severityFromMargin(margin),
          confidence,
          todayValue: Math.max(hipExcursion, trunkExcursion),
        })
      }
    }
  }

  return out
}

/**
 * Truncated follow-through: the trunk rotation barely changes from peak frame
 * to the final frame. We use the LAST frame's trunk rotation versus the peak
 * to estimate "did the body keep rotating past contact?".
 *
 * Threshold per AMATEUR_MISTAKES_REFERENCE: trunk should keep rotating ≥ 15°
 * past contact.
 */
function checkFollowThrough(ctx: RuleContext): Observation[] {
  const out: Observation[] = []
  const frames = ctx.shotFrames
  if (frames.length < 3) return out
  const peakIdx = frames.indexOf(ctx.peakFrame)
  // If we couldn't locate the peak frame in the swing slice, fall back to
  // the midpoint.
  const peakAt = peakIdx >= 0 ? peakIdx : Math.floor(frames.length / 2)
  const tailFrames = frames.slice(peakAt)
  if (tailFrames.length < 2) return out

  const peakTrunk = ctx.peakFrame.joint_angles?.trunk_rotation
  const finalTrunk = tailFrames[tailFrames.length - 1].joint_angles?.trunk_rotation
  if (typeof peakTrunk !== 'number' || typeof finalTrunk !== 'number') return out

  // Use shortAngleDelta on signed atan2 angles — the previous
  // Math.abs(finalTrunk - peakTrunk) had the same wrap bug as the old
  // Math.abs in jointAngles. A trunk that rotates +10° -> 0° -> -10°
  // would report follow-through delta 0° and falsely flag truncated.
  const followThroughDelta = Math.abs(shortAngleDelta(peakTrunk, finalTrunk))
  const target = 15
  if (followThroughDelta < target) {
    const margin = shortfallMargin(followThroughDelta, target)
    const confidence = computeConfidence({
      landmarkVisibility: ctx.meanVis,
      thresholdMargin: margin,
      ruleApplicability: ctx.applicability,
    })
    if (confidence >= CONFIDENCE_FLOOR) {
      out.push({
        phase: 'follow-through',
        joint: 'shoulders',
        pattern: 'truncated_followthrough',
        severity: severityFromMargin(margin),
        confidence,
        todayValue: followThroughDelta,
      })
    }
  }

  return out
}

// ---------------------------------------------------------------------------
// Spatial-trajectory rules — all read landmark (x, y) directly rather
// than joint_angles, so they need real landmark data and abstain on
// synthetic / zero-variation fixtures (where every frame's landmarks
// are identical, the metric is trivially zero and the rule would fire
// every clean test).
// ---------------------------------------------------------------------------

const ID_NOSE = 0
const ID_LEFT_SHOULDER = 11
const ID_RIGHT_SHOULDER = 12
const ID_LEFT_WRIST = 15
const ID_RIGHT_WRIST = 16
const ID_LEFT_HIP = 23
const ID_RIGHT_HIP = 24
const SPATIAL_VIS_FLOOR = 0.4

function inferFps(frames: PoseFrame[]): number {
  if (frames.length < 2) return 30
  const deltas: number[] = []
  for (let i = 1; i < Math.min(frames.length, 11); i++) {
    const dt = frames[i].timestamp_ms - frames[i - 1].timestamp_ms
    if (dt > 0 && Number.isFinite(dt)) deltas.push(dt)
  }
  if (deltas.length === 0) return 30
  deltas.sort((a, b) => a - b)
  const medianDt = deltas[Math.floor(deltas.length / 2)]
  return medianDt > 0 ? 1000 / medianDt : 30
}

function landmarkAt(frame: PoseFrame, id: number) {
  if (!frame.landmarks) return undefined
  const lm = frame.landmarks.find((l) => l.id === id)
  if (!lm) return undefined
  if (typeof lm.visibility !== 'number' || lm.visibility < SPATIAL_VIS_FLOOR) return undefined
  return lm
}

function midpoint(
  a: { x: number; y: number } | undefined,
  b: { x: number; y: number } | undefined,
): { x: number; y: number } | null {
  if (!a || !b) return null
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
}

function torsoLength(frame: PoseFrame): number | null {
  const ls = landmarkAt(frame, ID_LEFT_SHOULDER)
  const rs = landmarkAt(frame, ID_RIGHT_SHOULDER)
  const lh = landmarkAt(frame, ID_LEFT_HIP)
  const rh = landmarkAt(frame, ID_RIGHT_HIP)
  const sm = midpoint(ls, rs)
  const hm = midpoint(lh, rh)
  if (!sm || !hm) return null
  const dy = hm.y - sm.y
  const dx = hm.x - sm.x
  const len = Math.sqrt(dx * dx + dy * dy)
  return len > 0 ? len : null
}

function shoulderWidth(frame: PoseFrame): number | null {
  const ls = landmarkAt(frame, ID_LEFT_SHOULDER)
  const rs = landmarkAt(frame, ID_RIGHT_SHOULDER)
  if (!ls || !rs) return null
  const dx = ls.x - rs.x
  const dy = ls.y - rs.y
  const w = Math.sqrt(dx * dx + dy * dy)
  return w > 0 ? w : null
}

/**
 * Vertical hip-midpoint rise from a "loading" frame (~150ms before
 * contact) to the contact frame, normalized by torso length. In image
 * coords y increases downward, so a rising body has loadingY > contactY.
 *
 * Backed by Reid 2023 / Landlinger 2010: clean forehands show 3-8% torso
 * rise; flat or negative = no leg push. Threshold here at 3% (`target`),
 * anchor at 0 (`shortfallMargin` clamps).
 */
function checkLegDrive(ctx: RuleContext, fps: number): Observation[] {
  const out: Observation[] = []
  const frames = ctx.shotFrames
  if (frames.length < 6) return out
  const peakIdx = frames.indexOf(ctx.peakFrame)
  if (peakIdx < 0) return out
  // Loading frame ~150ms before contact. At 30fps that's 5 frames.
  const offset = Math.max(2, Math.round((fps * 150) / 1000))
  const loadingIdx = Math.max(0, peakIdx - offset)
  if (loadingIdx === peakIdx) return out

  const contact = frames[peakIdx]
  const loading = frames[loadingIdx]
  const cMid = midpoint(landmarkAt(contact, ID_LEFT_HIP), landmarkAt(contact, ID_RIGHT_HIP))
  const lMid = midpoint(landmarkAt(loading, ID_LEFT_HIP), landmarkAt(loading, ID_RIGHT_HIP))
  if (!cMid || !lMid) return out

  // Use the longer of the two torso-length readings as the normalizer
  // — single-frame torso can collapse on perspective foreshortening.
  const tL = torsoLength(loading)
  const tC = torsoLength(contact)
  const torso = tL && tC ? Math.max(tL, tC) : tL ?? tC
  if (!torso || torso <= 0) return out

  // y decreases upward → rise = loadingY - contactY.
  const rise = lMid.y - cMid.y
  const riseFraction = rise / torso

  // Only fire when there's actual landmark variation. A static fixture
  // produces rise=0 exactly; a real swing where the player happened
  // not to drive will produce a small rise (or a tiny negative). The
  // 0.005 floor (~0.5% torso) bypasses synthetic noise without
  // missing real flat-leg cases.
  if (Math.abs(riseFraction) < 0.005) return out

  const target = 0.03
  if (riseFraction >= target) return out

  // shortfall margin: 0 if rise just matches target, 1 if rise is at
  // or below 0 (no push). Anchored at 0 so a negative rise (player
  // dropped) reads as severe.
  const shortfall = Math.max(0, target - riseFraction)
  const margin = clamp01(shortfall / target)
  const confidence = computeConfidence({
    landmarkVisibility: ctx.meanVis,
    thresholdMargin: margin,
    ruleApplicability: ctx.applicability,
  })
  if (confidence < CONFIDENCE_FLOOR) return out
  out.push({
    phase: 'contact',
    joint: 'hips',
    pattern: 'weak_leg_drive',
    severity: severityFromMargin(margin),
    confidence,
    todayValue: riseFraction * 100,
  })
  return out
}

/**
 * Horizontal wrist displacement from the contact frame to T+150ms,
 * normalized by shoulder width. Direction is signed: a positive value
 * means the wrist moved in the direction it was already moving at
 * contact (push-out). A small forward extension means the racket
 * stopped at contact instead of driving through.
 *
 * Backed by tennis-coaching standard "out, up, and through" — a clean
 * follow-through extends the wrist forward >0.6 shoulder-widths before
 * crossing the body. Below 0.3 = short.
 */
function checkPushout(ctx: RuleContext, fps: number): Observation[] {
  const out: Observation[] = []
  const frames = ctx.shotFrames
  if (frames.length < 6) return out
  const peakIdx = frames.indexOf(ctx.peakFrame)
  if (peakIdx < 0) return out

  // T+150ms post contact = ~5 frames at 30fps.
  const offset = Math.max(2, Math.round((fps * 150) / 1000))
  const postIdx = Math.min(frames.length - 1, peakIdx + offset)
  if (postIdx === peakIdx) return out

  const wristId = ctx.dominantSide === 'right' ? ID_RIGHT_WRIST : ID_LEFT_WRIST
  const cWrist = landmarkAt(frames[peakIdx], wristId)
  const pWrist = landmarkAt(frames[postIdx], wristId)
  if (!cWrist || !pWrist) return out

  const sw =
    shoulderWidth(frames[peakIdx]) ?? shoulderWidth(frames[postIdx])
  if (!sw || sw <= 0) return out

  // Signed wrist motion in the direction of motion at contact: take the
  // wrist velocity vector at contact and project the wrist displacement
  // (contact→T+150) onto it.
  const prevIdx = Math.max(0, peakIdx - 1)
  const prevWrist = landmarkAt(frames[prevIdx], wristId)
  if (!prevWrist) return out
  const vx = cWrist.x - prevWrist.x
  const vy = cWrist.y - prevWrist.y
  const vmag = Math.sqrt(vx * vx + vy * vy)
  if (vmag < 1e-6) return out

  const dispX = pWrist.x - cWrist.x
  const dispY = pWrist.y - cWrist.y
  const projection = (dispX * vx + dispY * vy) / vmag
  const fraction = projection / sw

  if (Math.abs(fraction) < 0.02) return out

  const target = 0.3
  if (fraction >= target) return out

  const shortfall = Math.max(0, target - fraction)
  const margin = clamp01(shortfall / target)
  const confidence = computeConfidence({
    landmarkVisibility: ctx.meanVis,
    thresholdMargin: margin,
    ruleApplicability: ctx.applicability,
  })
  if (confidence < CONFIDENCE_FLOOR) return out
  out.push({
    phase: 'follow-through',
    joint: `${ctx.dominantSide}_wrist`,
    pattern: 'short_pushout',
    severity: severityFromMargin(margin),
    confidence,
    todayValue: fraction * 100,
  })
  return out
}

/**
 * 2D nose drift across a tight window around contact (±2 frames =
 * ~150ms at 30fps), measured RELATIVE to the hip midpoint (body-frame
 * to cancel broadcast camera motion) and normalized by torso length.
 * Combines X and Y drift via Euclidean magnitude — captures both head
 * bob (Y) and side-shifts (X) without missing one or the other.
 *
 * Threshold deliberately conservative (0.20 torso) so a clean elite
 * forehand doesn't trigger from normal swing rotation; biomech "still
 * head at contact" coaches against gross instability, not micro-motion.
 * Anchor at 0.40 so margin reaches 1.0 at 2x threshold.
 */
function checkStability(ctx: RuleContext): Observation[] {
  const out: Observation[] = []
  const frames = ctx.shotFrames
  if (frames.length < 4) return out
  const peakIdx = frames.indexOf(ctx.peakFrame)
  if (peakIdx < 0) return out

  // Tight window — head should be quiet in the immediate contact zone.
  const lo = Math.max(0, peakIdx - 2)
  const hi = Math.min(frames.length - 1, peakIdx + 2)
  if (hi - lo < 2) return out

  // Nose position relative to hip-midpoint cancels camera pan.
  const relX: number[] = []
  const relY: number[] = []
  for (let i = lo; i <= hi; i++) {
    const n = landmarkAt(frames[i], ID_NOSE)
    const hMid = midpoint(landmarkAt(frames[i], ID_LEFT_HIP), landmarkAt(frames[i], ID_RIGHT_HIP))
    if (n && hMid) {
      relX.push(n.x - hMid.x)
      relY.push(n.y - hMid.y)
    }
  }
  if (relX.length < 3) return out

  // Median torso length across the window dampens single-frame pose
  // noise on the normalizer.
  const torsoSamples: number[] = []
  for (let i = lo; i <= hi; i++) {
    const t = torsoLength(frames[i])
    if (t) torsoSamples.push(t)
  }
  if (torsoSamples.length === 0) return out
  torsoSamples.sort((a, b) => a - b)
  const torso = torsoSamples[Math.floor(torsoSamples.length / 2)]
  if (!torso || torso <= 0) return out

  // 2D drift radius: how far did the nose-relative-to-hips move at all
  // through the contact window?
  const dx = Math.max(...relX) - Math.min(...relX)
  const dy = Math.max(...relY) - Math.min(...relY)
  const drift = Math.sqrt(dx * dx + dy * dy)
  const fraction = drift / torso

  if (fraction < 0.005) return out

  const threshold = 0.20
  const anchor = 0.40
  if (fraction <= threshold) return out

  const margin = overshootMargin(fraction, threshold, anchor)
  const confidence = computeConfidence({
    landmarkVisibility: ctx.meanVis,
    thresholdMargin: margin,
    ruleApplicability: ctx.applicability,
  })
  if (confidence < CONFIDENCE_FLOOR) return out
  out.push({
    phase: 'contact',
    joint: 'head',
    pattern: 'unstable_base',
    severity: severityFromMargin(margin),
    confidence,
    todayValue: fraction * 100,
  })
  return out
}

/**
 * Drift-from-baseline observations. For each rule output we already produced,
 * compute the same value on the baseline and flag it if the |today - baseline|
 * delta exceeds a per-joint threshold. We also independently scan elbow-at-
 * contact and knee-at-loading for drift even when neither side trips a rule
 * (the baseline-comparison is itself the signal).
 *
 * driftMagnitude is reported in degrees (the same unit the joint angles
 * carry), so the route's deterministic "Show your work" block can render it.
 */
function compareDrift(
  todayCtx: RuleContext,
  baselineCtx: RuleContext,
  shot: ShotType | null,
): Observation[] {
  const out: Observation[] = []

  const elbowKey = dominantElbowKey(todayCtx.dominantSide)
  const kneeKey = dominantKneeKey(todayCtx.dominantSide)

  /**
   * Compare a single-frame extremum across the two swing slices. `pick`
   * selects which extremum to read (min for elbow contact-flexion, min for
   * knee deepest-load).
   */
  const compareExtremum = (
    key: keyof JointAngles,
    pick: 'min' | 'max',
    jointLabel: string,
    phase: Phase,
    threshold: number,
  ) => {
    const today =
      pick === 'min'
        ? angleMin(todayCtx.shotFrames, key)
        : angleMax(todayCtx.shotFrames, key)
    const base =
      pick === 'min'
        ? angleMin(baselineCtx.shotFrames, key)
        : angleMax(baselineCtx.shotFrames, key)
    if (today === null || base === null) return
    const drift = Math.abs(today - base)
    if (drift < threshold) return
    // Margin: how far past the threshold, anchored to 2x threshold (= margin 1).
    const margin = clamp01((drift - threshold) / threshold)
    const meanVis = (todayCtx.meanVis + baselineCtx.meanVis) / 2
    const confidence = computeConfidence({
      landmarkVisibility: meanVis,
      thresholdMargin: margin,
      ruleApplicability: Math.min(todayCtx.applicability, baselineCtx.applicability),
    })
    if (confidence >= CONFIDENCE_FLOOR) {
      out.push({
        phase,
        joint: jointLabel,
        pattern: 'drift_from_baseline',
        severity: severityFromMargin(margin),
        confidence,
        todayValue: today,
        baselineValue: base,
        driftMagnitude: drift,
      })
    }
  }

  // Elbow at contact = the elbow angle at the peak (contact-proxy) frame,
  // matching checkElbowAtContact above. Knee at loading remains the slice
  // min — knee deepest-load IS the most-flexed knee and that lines up with
  // the slice min naturally. 12° threshold sits comfortably outside
  // single-camera measurement noise.
  const todayElbow = todayCtx.peakFrame.joint_angles?.[elbowKey]
  const baseElbow = baselineCtx.peakFrame.joint_angles?.[elbowKey]
  if (
    typeof todayElbow === 'number' && Number.isFinite(todayElbow) &&
    typeof baseElbow === 'number' && Number.isFinite(baseElbow)
  ) {
    const drift = Math.abs(todayElbow - baseElbow)
    if (drift >= 12) {
      const margin = clamp01((drift - 12) / 12)
      const meanVis = (todayCtx.meanVis + baselineCtx.meanVis) / 2
      const confidence = computeConfidence({
        landmarkVisibility: meanVis,
        thresholdMargin: margin,
        ruleApplicability: Math.min(todayCtx.applicability, baselineCtx.applicability),
      })
      if (confidence >= CONFIDENCE_FLOOR) {
        out.push({
          phase: 'contact',
          joint: `${todayCtx.dominantSide}_elbow`,
          pattern: 'drift_from_baseline',
          severity: severityFromMargin(margin),
          confidence,
          todayValue: todayElbow,
          baselineValue: baseElbow,
          driftMagnitude: drift,
        })
      }
    }
  }
  compareExtremum(kneeKey, 'min', `${todayCtx.dominantSide}_knee`, 'loading', 12)

  // Excursion-level drift. Hips and trunk move enough across a swing that a
  // 15° change in excursion is meaningful.
  const compareExcursion = (
    key: keyof JointAngles,
    jointLabel: string,
    phase: Phase,
    threshold: number,
  ) => {
    const today = angleExcursion(todayCtx.shotFrames, key)
    const base = angleExcursion(baselineCtx.shotFrames, key)
    if (today === null || base === null) return
    const drift = Math.abs(today - base)
    if (drift < threshold) return
    const margin = clamp01((drift - threshold) / threshold)
    const meanVis = (todayCtx.meanVis + baselineCtx.meanVis) / 2
    const confidence = computeConfidence({
      landmarkVisibility: meanVis,
      thresholdMargin: margin,
      ruleApplicability: Math.min(todayCtx.applicability, baselineCtx.applicability),
    })
    if (confidence >= CONFIDENCE_FLOOR) {
      out.push({
        phase,
        joint: jointLabel,
        pattern: 'drift_from_baseline',
        severity: severityFromMargin(margin),
        confidence,
        todayValue: today,
        baselineValue: base,
        driftMagnitude: drift,
      })
    }
  }

  compareExcursion('hip_rotation', 'hips', 'loading', 15)
  compareExcursion('trunk_rotation', 'shoulders', 'loading', 15)

  // Contact-context observations: where the player met the ball
  // relative to their body at the moment of contact. These act as
  // CONTEXT for body-mechanic differences — a higher contact point can
  // explain a lighter knee load (the player didn't have to drive up
  // as much); a jammed contact can explain a cramped elbow. The LLM
  // uses them to write coaching like "your mechanics matched baseline
  // but you contacted the ball higher today, which is why it felt
  // different" instead of either silently matching ("trust your
  // foundation") or wrongly flagging ("your knees aren't loading").
  //
  // Single-frame measurement at peakFrame. Pose extraction's wrist
  // landmark can be noisy at peak (fastest motion of the swing) — we
  // gate on visibility and abstain on low confidence.
  const contactHeightDelta = compareContactHeightAtPeak(todayCtx, baselineCtx)
  if (contactHeightDelta) out.push(contactHeightDelta)
  const contactPositionDelta = compareContactPositionAtPeak(todayCtx, baselineCtx)
  if (contactPositionDelta) out.push(contactPositionDelta)

  return out
}

// ---------------------------------------------------------------------------
// Contact-context comparison helpers (baseline-compare only)
// ---------------------------------------------------------------------------

function dominantWristId(side: 'right' | 'left'): number {
  return side === 'right' ? ID_RIGHT_WRIST : ID_LEFT_WRIST
}

/**
 * Compare wrist height at peakFrame, normalized by torso length and
 * referenced to the shoulder midpoint (so camera-height shifts cancel).
 * Image-coords y increases downward, so "wrist above shoulder" = positive.
 */
function compareContactHeightAtPeak(
  todayCtx: RuleContext,
  baselineCtx: RuleContext,
): Observation | null {
  const wristId = dominantWristId(todayCtx.dominantSide)
  const todayWrist = landmarkAt(todayCtx.peakFrame, wristId)
  const baseWrist = landmarkAt(baselineCtx.peakFrame, wristId)
  if (!todayWrist || !baseWrist) return null
  // Abstain when peak-frame wrist visibility is low — pose tracking is
  // hardest at the fastest moment of the swing, so a confident-wrong
  // height delta is the failure mode to avoid.
  if (todayWrist.visibility < 0.5 || baseWrist.visibility < 0.5) return null

  const todayShoulder = midpoint(
    landmarkAt(todayCtx.peakFrame, ID_LEFT_SHOULDER),
    landmarkAt(todayCtx.peakFrame, ID_RIGHT_SHOULDER),
  )
  const baseShoulder = midpoint(
    landmarkAt(baselineCtx.peakFrame, ID_LEFT_SHOULDER),
    landmarkAt(baselineCtx.peakFrame, ID_RIGHT_SHOULDER),
  )
  if (!todayShoulder || !baseShoulder) return null

  const todayTorso = torsoLength(todayCtx.peakFrame)
  const baseTorso = torsoLength(baselineCtx.peakFrame)
  if (!todayTorso || !baseTorso) return null

  // Wrist Y above the shoulder midpoint, normalized by torso length.
  // Positive = wrist above shoulder. Negative = below.
  const todayHeight = (todayShoulder.y - todayWrist.y) / todayTorso
  const baseHeight = (baseShoulder.y - baseWrist.y) / baseTorso
  const delta = todayHeight - baseHeight
  const threshold = 0.08 // 8% of torso length
  if (Math.abs(delta) < threshold) return null

  const margin = clamp01((Math.abs(delta) - threshold) / threshold)
  const meanVis = (todayCtx.meanVis + baselineCtx.meanVis) / 2
  const confidence = computeConfidence({
    landmarkVisibility: meanVis,
    thresholdMargin: margin,
    ruleApplicability: Math.min(todayCtx.applicability, baselineCtx.applicability),
  })
  if (confidence < CONFIDENCE_FLOOR) return null

  return {
    phase: 'contact',
    joint: `${todayCtx.dominantSide}_wrist`,
    pattern: delta > 0 ? 'contact_height_higher' : 'contact_height_lower',
    severity: severityFromMargin(margin),
    confidence,
    todayValue: todayHeight,
    baselineValue: baseHeight,
    driftMagnitude: Math.abs(delta),
  }
}

/**
 * Compare horizontal wrist distance from body center at peakFrame,
 * normalized by shoulder width. "Jammed" = today's distance much
 * smaller than baseline (rushed contact close to body). "Extended" =
 * today's distance much larger (reaching).
 */
function compareContactPositionAtPeak(
  todayCtx: RuleContext,
  baselineCtx: RuleContext,
): Observation | null {
  const wristId = dominantWristId(todayCtx.dominantSide)
  const todayWrist = landmarkAt(todayCtx.peakFrame, wristId)
  const baseWrist = landmarkAt(baselineCtx.peakFrame, wristId)
  if (!todayWrist || !baseWrist) return null
  if (todayWrist.visibility < 0.5 || baseWrist.visibility < 0.5) return null

  const todayHip = midpoint(
    landmarkAt(todayCtx.peakFrame, ID_LEFT_HIP),
    landmarkAt(todayCtx.peakFrame, ID_RIGHT_HIP),
  )
  const baseHip = midpoint(
    landmarkAt(baselineCtx.peakFrame, ID_LEFT_HIP),
    landmarkAt(baselineCtx.peakFrame, ID_RIGHT_HIP),
  )
  if (!todayHip || !baseHip) return null

  const todayShoulderW = shoulderWidth(todayCtx.peakFrame)
  const baseShoulderW = shoulderWidth(baselineCtx.peakFrame)
  if (!todayShoulderW || !baseShoulderW) return null

  // Distance from wrist to body center, normalized.
  const todayDistance = Math.abs(todayWrist.x - todayHip.x) / todayShoulderW
  const baseDistance = Math.abs(baseWrist.x - baseHip.x) / baseShoulderW
  const delta = todayDistance - baseDistance
  const threshold = 0.15 // 15% of shoulder width
  if (Math.abs(delta) < threshold) return null

  const margin = clamp01((Math.abs(delta) - threshold) / threshold)
  const meanVis = (todayCtx.meanVis + baselineCtx.meanVis) / 2
  const confidence = computeConfidence({
    landmarkVisibility: meanVis,
    thresholdMargin: margin,
    ruleApplicability: Math.min(todayCtx.applicability, baselineCtx.applicability),
  })
  if (confidence < CONFIDENCE_FLOOR) return null

  return {
    phase: 'contact',
    joint: `${todayCtx.dominantSide}_wrist`,
    pattern: delta > 0 ? 'contact_position_extended' : 'contact_position_jammed',
    severity: severityFromMargin(margin),
    confidence,
    todayValue: todayDistance,
    baselineValue: baseDistance,
    driftMagnitude: Math.abs(delta),
  }
}

// ---------------------------------------------------------------------------
// Observation IDs
//
// We need a stable, content-addressable id for each Observation row so the
// /api/compare-strokes route can force the LLM to cite which row each
// sentence is talking about. Format:
//   `${strokeId}_${joint}_${phase}_${pattern}`
//
// (joint, pattern) alone collides across phases — e.g. a `drift_from_baseline`
// row can appear at both 'contact' (elbow drift) and 'loading' (knee drift).
// Adding phase to the tuple uniques them. strokeId scopes the id so two
// strokes that happen to expose the same fault don't collide either.
//
// Pure derivation, no hashing — keeps the id human-readable in logs and
// avoids dragging a crypto dep into a route that only needs disambiguation,
// not opacity.
// ---------------------------------------------------------------------------

export function observationId(
  strokeId: string,
  obs: Pick<Observation, 'joint' | 'phase' | 'pattern'>,
): string {
  return `${strokeId}_${obs.joint}_${obs.phase}_${obs.pattern}`
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function extractObservations(input: ExtractionInput): Observation[] {
  const todayCtx = buildRuleContext(input)
  if (!todayCtx) return []

  // If visibility is too low to trust ANY observation, short-circuit so the
  // route can render the "couldn't read your swing" empty state. The bar is
  // intentionally lenient (mean visibility under 0.4) — we want the floor to
  // bite only on truly garbage frames, not a single occluded landmark.
  if (todayCtx.meanVis < 0.4) return []

  // Spatial-trajectory rules need fps to time the loading frame
  // (contact - 150ms) and post-contact frame (contact + 150ms). Derive
  // from the median frame timestamp delta; default to 30 if timestamps
  // are unusable.
  const fps = inferFps(input.todaySummary)

  const todayAbsolute: Observation[] = []
  todayAbsolute.push(...checkElbowAtContact(todayCtx, input.shotType))
  todayAbsolute.push(...checkKneeLoad(todayCtx, input.shotType))
  todayAbsolute.push(...checkRotationExcursion(todayCtx))
  todayAbsolute.push(...checkFollowThrough(todayCtx))
  todayAbsolute.push(...checkLegDrive(todayCtx, fps))
  todayAbsolute.push(...checkPushout(todayCtx, fps))
  todayAbsolute.push(...checkStability(todayCtx))

  // Baseline-compare mode is for CONSISTENCY, not coaching. If a fault
  // shows up in today's clip AND in the baseline (e.g. neither shot
  // bends the knees deeply), the player isn't doing anything different
  // today — they swing this way every day. Calling it out reads as
  // generic coaching, which the user explicitly rejected ("the coach
  // shouldnt say youre not bending your knees in todays shot if youre
  // not doing it the baseline shot either"). So we run the same rules
  // on the baseline and SUPPRESS today-side absolute observations whose
  // (joint, phase, pattern) also fires on the baseline. Drift rows
  // stay — those ARE the differences.
  if (input.baselineSummary && input.baselineSummary.length > 0) {
    // Propagate isPreSliced to the baseline rule-context build too —
    // saved baselines are stored as a single swing trim, so the flag
    // should match today's pre-sliced status when the caller set it.
    const baselineCtx = buildRuleContext({
      todaySummary: input.baselineSummary,
      shotType: input.shotType,
      isPreSliced: input.isPreSliced,
    })
    if (baselineCtx) {
      const baselineAbsolute: Observation[] = []
      baselineAbsolute.push(...checkElbowAtContact(baselineCtx, input.shotType))
      baselineAbsolute.push(...checkKneeLoad(baselineCtx, input.shotType))
      baselineAbsolute.push(...checkRotationExcursion(baselineCtx))
      baselineAbsolute.push(...checkFollowThrough(baselineCtx))
      const baselineFps = inferFps(input.baselineSummary)
      baselineAbsolute.push(...checkLegDrive(baselineCtx, baselineFps))
      baselineAbsolute.push(...checkPushout(baselineCtx, baselineFps))
      baselineAbsolute.push(...checkStability(baselineCtx))

      const baselineFingerprints = new Set(
        baselineAbsolute.map((o) => `${o.joint}|${o.phase}|${o.pattern}`),
      )
      const filteredToday = todayAbsolute.filter(
        (o) => !baselineFingerprints.has(`${o.joint}|${o.phase}|${o.pattern}`),
      )
      const drift = compareDrift(todayCtx, baselineCtx, input.shotType)
      return [...filteredToday, ...drift]
    }
  }

  return todayAbsolute
}

/**
 * Pick the single primary observation: highest severity-rank × confidence.
 * Stable: ties broken by SEVERITY_RANK descending then by confidence.
 * Returns null when there are no observations.
 */
export function pickPrimary(observations: Observation[]): Observation | null {
  if (observations.length === 0) return null
  let best: Observation | null = null
  let bestScore = -Infinity
  for (const o of observations) {
    const score = SEVERITY_RANK[o.severity] * o.confidence
    if (score > bestScore) {
      bestScore = score
      best = o
    }
  }
  return best
}

/**
 * Pick up to `max` secondary observations, excluding the primary. Same
 * scoring as pickPrimary, descending. Stable on ties.
 */
export function pickSecondary(
  observations: Observation[],
  excluding: Observation | null,
  max = 3,
): Observation[] {
  const remaining = excluding
    ? observations.filter((o) => o !== excluding)
    : observations.slice()
  remaining.sort((a, b) => {
    const sa = SEVERITY_RANK[a.severity] * a.confidence
    const sb = SEVERITY_RANK[b.severity] * b.confidence
    return sb - sa
  })
  return remaining.slice(0, Math.max(0, max))
}
