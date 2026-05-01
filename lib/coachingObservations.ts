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
import { detectSwings } from './jointAngles'
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
  | 'drift_from_baseline'

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
 * Mean visibility across the landmarks of a frame. When the frame has no
 * landmarks (e.g. test fixtures that only set joint_angles), default to 1.0
 * so the confidence floor isn't tripped by the absence of data we never had.
 */
function frameVisibility(frame: PoseFrame): number {
  if (!frame.landmarks || frame.landmarks.length === 0) return 1.0
  let sum = 0
  let n = 0
  for (const l of frame.landmarks) {
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
function pickPeakFrame(frames: PoseFrame[]): PoseFrame | null {
  if (frames.length === 0) return null
  const swings = detectSwings(frames)
  const primary = swings[0]
  if (!primary) return frames[Math.floor(frames.length / 2)] ?? null
  const peakIdxInSlice = primary.peakFrame - primary.startFrame
  const idx = Math.max(0, Math.min(primary.frames.length - 1, peakIdxInSlice))
  return primary.frames[idx] ?? null
}

/**
 * Excursion = max - min of an angle key across all frames in the primary
 * swing. Used for hip_rotation / trunk_rotation where the absolute reading is
 * camera-noisy but the change through the swing is robust.
 */
function angleExcursion(frames: PoseFrame[], key: keyof JointAngles): number | null {
  let lo = Infinity
  let hi = -Infinity
  for (const f of frames) {
    const v = f.joint_angles?.[key]
    if (typeof v === 'number' && Number.isFinite(v)) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null
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
 */
function primarySwingFrames(frames: PoseFrame[]): PoseFrame[] {
  if (frames.length === 0) return frames
  const swings = detectSwings(frames)
  const primary = swings[0]
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
 * Returns the dominant elbow + knee keys for a shot. We default to right side
 * when shot type is unknown (the existing pipeline assumes right-handed unless
 * told otherwise). This is intentionally simple — handedness is encoded in
 * the profile, not here, and the prompt layer adapts the cue language.
 */
function dominantElbowKey(shot: ShotType | null): keyof JointAngles {
  return shot === 'backhand' ? 'left_elbow' : 'right_elbow'
}

function dominantKneeKey(shot: ShotType | null): keyof JointAngles {
  return shot === 'backhand' ? 'left_knee' : 'right_knee'
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
  const shotFrames = primarySwingFrames(frames)
  const peakFrame = pickPeakFrame(frames)
  if (!peakFrame) return null
  const meanVis = meanVisibility(shotFrames)
  // Unknown shot type -> halve applicability (we don't know which rules really
  // apply). Volley/slice are rules-applicable but with reduced weight.
  let applicability = 1.0
  if (input.shotType === null) applicability = 0.5
  else if (input.shotType === 'volley' || input.shotType === 'slice') applicability = 0.8
  const dominantSide: 'right' | 'left' = input.shotType === 'backhand' ? 'left' : 'right'
  return { shotFrames, peakFrame, meanVis, applicability, dominantSide }
}

/**
 * Elbow rules detect cramped (<90°) and over-extended (>175°) at contact.
 * We scan the FULL swing slice for the extremum (min for cramped, max for
 * over-extended) rather than reading a single peak-activity frame. The peak
 * activity frame is the moment of fastest angle change, not the contact
 * extremum, so on a clean swing the extremum and the peak don't coincide.
 * Extremum-over-window is robust to that and to single-frame jitter.
 *
 * Serve cramped threshold relaxed to 140° because serves expect near-full
 * extension at contact.
 */
function checkElbowAtContact(
  ctx: RuleContext,
  shot: ShotType | null,
): Observation[] {
  const out: Observation[] = []
  const key = dominantElbowKey(shot)

  const minElbow = angleMin(ctx.shotFrames, key)
  const maxElbow = angleMax(ctx.shotFrames, key)
  if (minElbow === null || maxElbow === null) return out

  // Cramped elbow: under 90° on groundstrokes, under 140° on the serve.
  const crampedThreshold = shot === 'serve' ? 140 : 90
  const crampedAnchor = shot === 'serve' ? 100 : 60
  if (minElbow < crampedThreshold) {
    const margin = undershootMargin(minElbow, crampedThreshold, crampedAnchor)
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
        todayValue: minElbow,
      })
    }
  }

  // Over-extended elbow: 175° at contact = locked straight.
  if (maxElbow > 175) {
    const margin = overshootMargin(maxElbow, 175, 180)
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
        todayValue: maxElbow,
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
  const key = dominantKneeKey(shot)
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
    const target = 40
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
    const target = 60
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
  if (hipExcursion !== null && trunkExcursion !== null) {
    if (hipExcursion < 25 && trunkExcursion < 35) {
      const margin = clamp01(
        Math.max(shortfallMargin(hipExcursion, 25), shortfallMargin(trunkExcursion, 35)),
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

  const followThroughDelta = Math.abs(finalTrunk - peakTrunk)
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

  const elbowKey = dominantElbowKey(shot)
  const kneeKey = dominantKneeKey(shot)

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

  // Elbow at contact = the minimum elbow angle in the swing window (most flexed).
  // Knee at loading = the minimum knee angle in the swing window (deepest load).
  // 12° threshold sits comfortably outside single-camera measurement noise.
  compareExtremum(elbowKey, 'min', `${todayCtx.dominantSide}_elbow`, 'contact', 12)
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

  return out
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

  const observations: Observation[] = []
  observations.push(...checkElbowAtContact(todayCtx, input.shotType))
  observations.push(...checkKneeLoad(todayCtx, input.shotType))
  observations.push(...checkRotationExcursion(todayCtx))
  observations.push(...checkFollowThrough(todayCtx))

  // Baseline-compare layer adds drift_from_baseline rows on top of the
  // standard rules. The standard rules still run on `todaySummary` so we can
  // still surface absolute issues even when baseline matches today's clip.
  if (input.baselineSummary && input.baselineSummary.length > 0) {
    const baselineCtx = buildRuleContext({
      todaySummary: input.baselineSummary,
      shotType: input.shotType,
    })
    if (baselineCtx) {
      observations.push(...compareDrift(todayCtx, baselineCtx, input.shotType))
    }
  }

  return observations
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
