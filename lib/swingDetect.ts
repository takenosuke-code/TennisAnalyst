import type { PoseFrame, Landmark } from './supabase'
import type { DetectedStroke } from './strokeAnalysis'
import { verifySwingShape } from './swingShape'

// Re-export so consumers that have always imported DetectedStroke from
// '@/lib/swingDetect' (or via the '@/lib/jointAngles' re-export) keep
// working. The canonical declaration lives in lib/strokeAnalysis.ts.
export type { DetectedStroke } from './strokeAnalysis'

// MediaPipe Pose landmark indices for the wrists. Inlined rather than
// imported from `./jointAngles` to keep the dependency graph one-way:
// jointAngles.ts delegates to detectStrokes() in this module; importing
// back into here would create a circular initialization order.
const LM_LEFT_WRIST = 15
const LM_RIGHT_WRIST = 16

// ---------------------------------------------------------------------------
// Wrist-velocity peak-pick stroke detector.
//
// 2026-05 rewrite — addresses the "too many overlapping clips on long
// videos" failure mode of the prior height-threshold + refractory-only
// pipeline. Three layered changes:
//
//   1. Topographic PROMINENCE replaces the height threshold. For each
//      candidate peak, prominence is its height above the lowest point
//      reachable on either side within `wlen` frames before hitting a
//      taller peak. A jittery shoulder on a real swing reaches the
//      taller true peak immediately, so its prominence is tiny — the
//      scipy `find_peaks(prominence=…, wlen=…)` standard.
//   2. MAD-based prominence floor + an absolute floor. MAD (median
//      absolute deviation) is non-parametric and uncontaminated by the
//      events being thresholded; the absolute floor stops the floor
//      from collapsing on a quiet clip with no real swings.
//   3. Width filter at half-prominence. Real swings are 150–700ms wide;
//      sub-100ms hand-twitches and >800ms slow drifts can't masquerade.
//   4. VORONOI boundaries replace fixed pre/post pad when neighbors are
//      close. Two real strokes 800ms apart used to produce 60% overlap
//      between their padded clips — Voronoi splits at the midpoint, so
//      every frame belongs to exactly one stroke. Outside neighbors,
//      the default 1000ms / 500ms pad still applies.
//
// Order of operations:
//   a. Per-frame dominant-wrist linear speed (normalized [0,1] / sec).
//   b. 5-frame centered moving-average smoother.
//   c. Find ALL local maxima (no threshold).
//   d. Filter by prominence ≥ floor AND width ∈ [widthMinMs, widthMaxMs].
//   e. Refractory dedup as a guardrail (highest-prominence wins).
//   f. Voronoi-bounded windows with the original pre/post pad as an
//      outer cap. Clamp to [0, totalFrames-1].
// ---------------------------------------------------------------------------

export interface DetectStrokesOptions {
  /**
   * Override the input frame rate. When omitted we derive it from the
   * median timestamp delta — robust against a few duplicate or missing
   * frames in long captures.
   */
  fps?: number
  /**
   * `'left' | 'right'` from the user profile, when known. When provided
   * we use the corresponding wrist landmark unconditionally; otherwise we
   * pick whichever wrist had the higher mean visibility, defaulting to
   * right when both are equal or unsampled.
   */
  dominantHand?: 'left' | 'right' | null
  /**
   * Pre-pad window in milliseconds. Default 1000ms (Reid/Elliott — tennis
   * backswing initiates 800–1200ms before contact).
   */
  prePadMs?: number
  /**
   * Post-pad window in milliseconds. Default 900ms — amateur
   * follow-through often runs 600–800ms past wrist-speed peak. Voronoi
   * caps the pad to the midpoint of the next peak, so this only
   * stretches isolated swings where there's room.
   */
  postPadMs?: number
  /**
   * Minimum gap between accepted peaks. Default 500ms.
   */
  refractoryMs?: number
  /**
   * @deprecated 2026-05 — height-threshold logic was replaced by
   * topographic prominence. Accepted for backwards compatibility, has
   * no effect.
   */
  thresholdK?: number
  /**
   * Smoothing window in frames for the centered moving average over
   * wrist speed. Default 5 — same as the pose-extraction smoother.
   */
  smoothingFrames?: number
  /**
   * Minimum frame distance between two raw peaks before refractory
   * dedup. Default 3 frames. Refractory does the heavy lifting; this
   * exists so two real peaks 300ms apart (≈9 frames at 30fps) can both
   * make it past peak-finding before refractory chooses one.
   */
  minPeakDistanceFrames?: number
  /**
   * scipy-style `wlen` — bound on how far the prominence walk extends
   * before cutting off. Default 2000ms (2·fps frames). Larger values
   * make prominence more global (good when peaks are far apart);
   * smaller values keep it local (good for fast rallies).
   */
  wlenMs?: number
  /**
   * Absolute prominence floor in normalized-coords-per-second units.
   * Anything below is rejected outright regardless of MAD. Default 0.2
   * — sits above typical walking / fidget prominence (0.05–0.15) and
   * below soft amateur swings that were previously being dropped at
   * 0.3.
   */
  prominenceFloorAbs?: number
  /**
   * Multiplier on `1.4826 · MAD(speeds)` for the data-driven prominence
   * floor. Effective floor = max(prominenceFloorAbs, k · 1.4826 · MAD).
   * Default 3. Higher values reject more candidates; 5 is strict,
   * 3 is permissive enough to keep softer amateur swings.
   */
  prominenceFloorMadK?: number
  /**
   * Lower bound on peak width (at half-prominence) in milliseconds.
   * Default 80ms — covers fast amateur swings (sharp acceleration
   * burst) while rejecting sub-50ms jitter. Real swings typically
   * carry 100–300ms half-prominence width.
   */
  widthMinMs?: number
  /**
   * Upper bound on peak width (at half-prominence) in milliseconds.
   * Default 1000ms. Above this the burst is so slow it's almost
   * certainly walking / prep positioning, not a stroke.
   */
  widthMaxMs?: number
  /**
   * When true, runs the swing-shape verifier (lib/swingShape.ts) on
   * every refractory survivor and drops candidates whose surrounding
   * joint trajectories don't look like a real swing. Specifically:
   * hip rotation excursion ≥8°, trunk rotation excursion ≥10°, and
   * hip angular-velocity peak preceding the wrist peak by 25–400ms.
   *
   * Opt-in (default false) because synthetic test fixtures don't
   * populate joint_angles. The analyze-page UI surface enables it.
   */
  verifyShape?: boolean
}

const VISIBILITY_FLOOR = 0.5

// Default Stroke-options resolved with applyDefaults below.
//
// 2026-05 — refractory dropped 500 → 350ms now that Voronoi handles
// inter-stroke window-overlap independently. 350ms still kills tap-back
// double-peaks (~300ms apart) but lets through fast volleys >350ms
// apart that the old 500ms refractory was over-suppressing.
//
// 2026-05 (round 2) — three leniency tweaks driven by amateur-clip
// feedback:
//   - postPadMs 500 → 900: amateur follow-through often runs ~700ms
//     past wrist-speed peak (decel + recovery to neutral). Voronoi
//     still caps the pad when a neighbor is close, so dense rallies
//     don't get over-padded clips.
//   - prominenceFloorAbs 0.3 → 0.2 and prominenceFloorMadK 4 → 3:
//     real swings on sub-tier amateurs (slower wrist speeds) were
//     dipping under the prior absolute floor, so the user reported
//     "still missing some shots." The pair stays comfortably above
//     walking / fidget prominence (0.05–0.15) while accepting softer
//     swings that we were previously dropping.
export const STROKE_DEFAULTS = {
  prePadMs: 1000,
  postPadMs: 900,
  refractoryMs: 350,
  thresholdK: 1.0,
  smoothingFrames: 5,
  minPeakDistanceFrames: 3,
  wlenMs: 2000,
  prominenceFloorAbs: 0.2,
  prominenceFloorMadK: 3,
  widthMinMs: 60,
  widthMaxMs: 1000,
} as const

interface ResolvedOptions {
  fps: number
  dominantHand: 'left' | 'right' | null
  prePadMs: number
  postPadMs: number
  refractoryMs: number
  smoothingFrames: number
  minPeakDistanceFrames: number
  wlenMs: number
  prominenceFloorAbs: number
  prominenceFloorMadK: number
  widthMinMs: number
  widthMaxMs: number
  verifyShape: boolean
}

function applyDefaults(
  frames: PoseFrame[],
  opts: DetectStrokesOptions,
): ResolvedOptions {
  const fps = opts.fps ?? deriveFps(frames)
  return {
    fps,
    dominantHand: opts.dominantHand ?? null,
    prePadMs: opts.prePadMs ?? STROKE_DEFAULTS.prePadMs,
    postPadMs: opts.postPadMs ?? STROKE_DEFAULTS.postPadMs,
    refractoryMs: opts.refractoryMs ?? STROKE_DEFAULTS.refractoryMs,
    smoothingFrames: opts.smoothingFrames ?? STROKE_DEFAULTS.smoothingFrames,
    minPeakDistanceFrames:
      opts.minPeakDistanceFrames ?? STROKE_DEFAULTS.minPeakDistanceFrames,
    wlenMs: opts.wlenMs ?? STROKE_DEFAULTS.wlenMs,
    prominenceFloorAbs:
      opts.prominenceFloorAbs ?? STROKE_DEFAULTS.prominenceFloorAbs,
    prominenceFloorMadK:
      opts.prominenceFloorMadK ?? STROKE_DEFAULTS.prominenceFloorMadK,
    widthMinMs: opts.widthMinMs ?? STROKE_DEFAULTS.widthMinMs,
    widthMaxMs: opts.widthMaxMs ?? STROKE_DEFAULTS.widthMaxMs,
    verifyShape: opts.verifyShape ?? false,
  }
}

/**
 * Derive a sane fps from frame timestamps. Uses the median Δt to
 * tolerate a handful of dropped or duplicated frames; falls back to 30
 * when timestamps are absent or all equal (synthetic fixtures).
 */
export function deriveFps(frames: PoseFrame[]): number {
  if (frames.length < 2) return 30
  const diffs: number[] = []
  for (let i = 1; i < frames.length; i++) {
    const dt = frames[i].timestamp_ms - frames[i - 1].timestamp_ms
    if (Number.isFinite(dt) && dt > 0) diffs.push(dt)
  }
  if (diffs.length === 0) return 30
  diffs.sort((a, b) => a - b)
  const medianDt = diffs[Math.floor(diffs.length / 2)]
  if (!medianDt || medianDt <= 0) return 30
  const fps = 1000 / medianDt
  // Round to the nearest integer when extremely close; keeps frame-count
  // arithmetic identical for canonical 30/60fps captures.
  if (Math.abs(fps - Math.round(fps)) < 0.05) return Math.round(fps)
  return fps
}

function getLandmark(landmarks: Landmark[], id: number): Landmark | undefined {
  return landmarks.find((l) => l.id === id)
}

/**
 * Decide which wrist to track. Profile dominant-hand wins when present;
 * otherwise pick by mean visibility across the clip; otherwise default
 * to right.
 */
function pickDominantWrist(
  frames: PoseFrame[],
  override: 'left' | 'right' | null,
): 'left' | 'right' {
  if (override === 'left' || override === 'right') return override

  let leftSum = 0
  let leftCount = 0
  let rightSum = 0
  let rightCount = 0
  for (const f of frames) {
    const lw = getLandmark(f.landmarks, LM_LEFT_WRIST)
    const rw = getLandmark(f.landmarks, LM_RIGHT_WRIST)
    if (lw) {
      leftSum += lw.visibility
      leftCount++
    }
    if (rw) {
      rightSum += rw.visibility
      rightCount++
    }
  }
  const leftMean = leftCount > 0 ? leftSum / leftCount : 0
  const rightMean = rightCount > 0 ? rightSum / rightCount : 0
  // Right wins ties — matches typical right-handed default.
  return leftMean > rightMean ? 'left' : 'right'
}

/**
 * Per-frame dominant-wrist linear speed in normalized-coords-per-second.
 * Frames where the wrist landmark is missing or below the visibility
 * floor produce a 0 speed (rather than NaN) so the smoother can still
 * roll through the gap without poisoning neighbours.
 */
function computeWristSpeeds(
  frames: PoseFrame[],
  side: 'left' | 'right',
): number[] {
  const id = side === 'left' ? LM_LEFT_WRIST : LM_RIGHT_WRIST
  const speeds: number[] = new Array(frames.length).fill(0)
  if (frames.length < 2) return speeds

  let prev: { x: number; y: number; t: number } | null = null
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]
    const lm = getLandmark(f.landmarks, id)
    if (!lm || lm.visibility < VISIBILITY_FLOOR) {
      // Don't anchor a baseline on an occluded sample — leaves the speed
      // at 0 here and lets the next visible sample's neighbour drive
      // velocity.
      continue
    }
    if (prev) {
      const dx = lm.x - prev.x
      const dy = lm.y - prev.y
      const dt = (f.timestamp_ms - prev.t) / 1000
      if (dt > 0) {
        speeds[i] = Math.sqrt(dx * dx + dy * dy) / dt
      }
    }
    prev = { x: lm.x, y: lm.y, t: f.timestamp_ms }
  }
  return speeds
}

/**
 * Centered moving-average smoother with a half-window of
 * floor(smoothingFrames / 2). Uses whatever neighbours exist at the
 * edges (not zero-padded) so the smoothed signal is the same length as
 * the input.
 */
function smooth(values: number[], windowFrames: number): number[] {
  const half = Math.floor(windowFrames / 2)
  const out: number[] = new Array(values.length).fill(0)
  for (let i = 0; i < values.length; i++) {
    const lo = Math.max(0, i - half)
    const hi = Math.min(values.length - 1, i + half)
    let sum = 0
    let count = 0
    for (let j = lo; j <= hi; j++) {
      sum += values[j]
      count++
    }
    out[i] = count > 0 ? sum / count : 0
  }
  return out
}

/**
 * Find ALL local maxima in `values`. A local max is `v[i] > v[i-1] &&
 * v[i] >= v[i+1]` — strict on the left, allow plateau on the right so
 * a flat-top peak picks its leftmost index exactly once. Out-of-bounds
 * neighbors treated as -Infinity so edge frames can still be peaks.
 *
 * No threshold and no min-distance dedup at this stage — that's the
 * job of the prominence filter and refractory layers downstream.
 */
function findAllLocalMaxima(values: number[]): number[] {
  const peaks: number[] = []
  for (let i = 0; i < values.length; i++) {
    const left = i > 0 ? values[i - 1] : -Infinity
    const right = i < values.length - 1 ? values[i + 1] : -Infinity
    if (values[i] > left && values[i] >= right) {
      peaks.push(i)
    }
  }
  return peaks
}

/**
 * Topographic prominence (scipy.signal.peak_prominences-equivalent).
 *
 * For peak at index `peakIdx` in `values`, walk left within `wlen/2`
 * frames, stopping at either the array edge or the first index whose
 * value strictly exceeds the peak. The lowest value encountered along
 * that walk is the left base. Same on the right. Prominence =
 * peak_height − max(leftBase, rightBase).
 *
 * Bounded by `wlen` so prominence stays local to the peak's
 * neighborhood — without this, a single huge peak in the clip would
 * make every other peak's prominence look small relative to the
 * dominant one.
 */
function computeProminence(
  values: number[],
  peakIdx: number,
  wlen: number,
): number {
  const peakValue = values[peakIdx]
  const halfWlen = Math.max(1, Math.floor(wlen / 2))

  // Left walk. Track whether the walk hit a strictly higher peak
  // (proper terminator) vs just the array/wlen edge — the latter
  // means the peak is cut off by the clip boundary and the base on
  // that side is unreliable.
  let leftBase = peakValue
  let leftHitHigher = false
  const leftBound = Math.max(0, peakIdx - halfWlen)
  for (let i = peakIdx - 1; i >= leftBound; i--) {
    if (values[i] > peakValue) {
      leftHitHigher = true
      break
    }
    if (values[i] < leftBase) leftBase = values[i]
  }

  // Right walk
  let rightBase = peakValue
  let rightHitHigher = false
  const rightBound = Math.min(values.length - 1, peakIdx + halfWlen)
  for (let i = peakIdx + 1; i <= rightBound; i++) {
    if (values[i] > peakValue) {
      rightHitHigher = true
      break
    }
    if (values[i] < rightBase) rightBase = values[i]
  }

  // Boundary handling: if a walk terminated against the array edge
  // (not a higher peak) AND its base equals the peak value (meaning
  // the signal never descended on that side, just got cut off), trust
  // only the other side. A truly truncated swing peak still has a
  // clean descent on its non-cut side; using max(leftBase, rightBase)
  // would compute a near-zero prominence and reject a real swing.
  const leftIsBoundary = !leftHitHigher && leftBound === 0
  const rightIsBoundary = !rightHitHigher && rightBound === values.length - 1
  const leftDescended = leftBase < peakValue
  const rightDescended = rightBase < peakValue

  if (leftIsBoundary && !leftDescended && rightDescended) {
    return peakValue - rightBase
  }
  if (rightIsBoundary && !rightDescended && leftDescended) {
    return peakValue - leftBase
  }
  // Same logic when one side ALSO descended cleanly but the other is
  // truncated by the array edge — take the cleaner-descent side. This
  // handles peaks near the boundary whose other side got a partial
  // descent before being cut off.
  if (rightIsBoundary && leftDescended && rightDescended && rightBase > leftBase) {
    return peakValue - leftBase
  }
  if (leftIsBoundary && leftDescended && rightDescended && leftBase > rightBase) {
    return peakValue - rightBase
  }
  return peakValue - Math.max(leftBase, rightBase)
}

/**
 * Width at half-prominence (scipy.signal.peak_widths with
 * `rel_height=0.5`). Walks outward from `peakIdx` to find where
 * `values` first dips below `peakValue − prominence/2`. Returns the
 * width in frames; 0 when the peak is degenerate (zero prominence).
 */
function computePeakWidth(
  values: number[],
  peakIdx: number,
  prominence: number,
): number {
  if (prominence <= 0) return 0
  const peakValue = values[peakIdx]
  const halfHeight = peakValue - prominence / 2

  let leftEdge = peakIdx
  for (let i = peakIdx - 1; i >= 0; i--) {
    if (values[i] < halfHeight) {
      // Linear interpolation between i and i+1 to refine the edge.
      const lo = values[i]
      const hi = values[i + 1]
      const frac = hi !== lo ? (halfHeight - lo) / (hi - lo) : 0
      leftEdge = i + frac
      break
    }
    leftEdge = i
  }
  let rightEdge = peakIdx
  for (let i = peakIdx + 1; i < values.length; i++) {
    if (values[i] < halfHeight) {
      const lo = values[i]
      const hi = values[i - 1]
      const frac = hi !== lo ? (halfHeight - lo) / (hi - lo) : 0
      rightEdge = i - frac
      break
    }
    rightEdge = i
  }
  return Math.max(0, rightEdge - leftEdge)
}

/**
 * Median absolute deviation (MAD). Robust spread estimator: median of
 * |x − median(x)|. The 1.4826 multiplier converts MAD to a normal-
 * distribution-equivalent standard deviation.
 */
function median(values: number[]): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid]
}

function mad(values: number[]): number {
  if (values.length === 0) return 0
  const m = median(values)
  const dev: number[] = new Array(values.length)
  for (let i = 0; i < values.length; i++) dev[i] = Math.abs(values[i] - m)
  return median(dev)
}

/**
 * Voronoi-style stroke boundaries. For each peak (sorted ascending by
 * frame index), the start frame extends backward from the peak by at
 * most `prePadFrames`; the end frame extends forward by at most
 * `postPadFrames`. When the next peak is closer than that pad, the
 * boundary between the two strokes sits at the midpoint between their
 * peaks — every frame belongs to exactly one stroke, no overlap.
 *
 * Boundaries are clamped to [0, totalFrames-1]. Output preserves input
 * peak order.
 */
function voronoiBoundaries(
  peakFrames: number[],
  prePadFrames: number,
  postPadFrames: number,
  totalFrames: number,
): Array<{ start: number; end: number }> {
  const lastIdx = totalFrames - 1
  return peakFrames.map((peak, i) => {
    const prevPeak = i > 0 ? peakFrames[i - 1] : null
    const nextPeak = i < peakFrames.length - 1 ? peakFrames[i + 1] : null

    // Left bound: default = peak − prePad. If a previous peak exists
    // and the midpoint is closer, use the midpoint + 1 (so adjacent
    // strokes don't share a boundary frame).
    let start = Math.max(0, peak - prePadFrames)
    if (prevPeak !== null) {
      const midpoint = Math.floor((prevPeak + peak) / 2)
      if (midpoint + 1 > start) start = midpoint + 1
    }

    // Right bound: default = peak + postPad. If a next peak exists and
    // its midpoint is closer, use the midpoint.
    let end = Math.min(lastIdx, peak + postPadFrames)
    if (nextPeak !== null) {
      const midpoint = Math.floor((peak + nextPeak) / 2)
      if (midpoint < end) end = midpoint
    }

    // Pathological case where boundaries collapse — keep the peak
    // frame at minimum.
    if (end < start) end = start
    return { start, end }
  })
}

/**
 * Detect strokes via dominant-wrist speed peak-pick + biomechanical
 * padding. See the file-level comment for design notes.
 *
 * Returns DetectedStroke[] sorted by peakFrame ascending. Empty array
 * when no peaks survive the threshold and refractory.
 */
export function detectStrokes(
  frames: PoseFrame[],
  opts: DetectStrokesOptions = {},
): DetectedStroke[] {
  const resolved = applyDefaults(frames, opts)
  if (frames.length < 2) return []

  const side = pickDominantWrist(frames, resolved.dominantHand)
  const rawSpeeds = computeWristSpeeds(frames, side)
  const smoothed = smooth(rawSpeeds, resolved.smoothingFrames)

  // Convert ms knobs to frame counts using the actual fps.
  const wlenFrames = Math.max(
    2,
    Math.round((resolved.wlenMs / 1000) * resolved.fps),
  )
  const widthMinFrames = (resolved.widthMinMs / 1000) * resolved.fps
  const widthMaxFrames = (resolved.widthMaxMs / 1000) * resolved.fps
  const refractoryFrames = Math.max(
    1,
    Math.round((resolved.refractoryMs / 1000) * resolved.fps),
  )

  // Find ALL local maxima — no threshold; prominence + width below
  // are the real gates.
  const allMaxima = findAllLocalMaxima(smoothed)
  if (allMaxima.length === 0) return []

  // Data-driven prominence floor. MAD-based scale + an absolute floor
  // so a quiet clip with no real swings can't have its floor collapse
  // toward zero and leak walking/fidget peaks.
  const madVal = mad(smoothed)
  const madFloor = resolved.prominenceFloorMadK * 1.4826 * madVal
  const promFloor = Math.max(resolved.prominenceFloorAbs, madFloor)

  // Per-candidate prominence + width. Keep peaks that clear both
  // gates simultaneously.
  interface Candidate {
    idx: number
    prominence: number
    width: number
  }
  const candidates: Candidate[] = []
  for (const idx of allMaxima) {
    const prom = computeProminence(smoothed, idx, wlenFrames)
    if (prom < promFloor) continue
    const width = computePeakWidth(smoothed, idx, prom)
    if (width < widthMinFrames || width > widthMaxFrames) continue
    candidates.push({ idx, prominence: prom, width })
  }
  if (candidates.length === 0) return []

  // Refractory dedup as a guardrail: if two surviving candidates sit
  // closer than the refractory window, keep the more prominent one.
  // Most of the work has already been done by prominence — this kills
  // the rare same-real-swing-double-peak that survived width checks.
  let survivors = applyRefractoryByProminence(candidates, refractoryFrames)
  survivors.sort((a, b) => a.idx - b.idx)

  // Optional second-stage shape verifier. Drops candidates whose
  // surrounding joint trajectories don't look like a real swing
  // (insufficient hip rotation, no proximal-to-distal kinetic chain).
  // The wrist-speed pipeline alone has no notion of "swing shape"
  // and fires on walking-toward-camera, prep bouncing, etc. This
  // gate is opt-in because synthetic-fixture tests bypass joint
  // angles; the analyze-page UI surface turns it on.
  if (resolved.verifyShape && survivors.length > 0) {
    survivors = survivors.filter((s) =>
      verifySwingShape(frames, s.idx, resolved.fps).passed,
    )
  }
  if (survivors.length === 0) return []

  const prePadFrames = Math.ceil((resolved.prePadMs / 1000) * resolved.fps)
  const postPadFrames = Math.ceil((resolved.postPadMs / 1000) * resolved.fps)

  // Voronoi-bounded windows: adjacent strokes split the gap at the
  // midpoint, isolated strokes get the full pre/post pad.
  const peakFrames = survivors.map((s) => s.idx)
  const bounds = voronoiBoundaries(
    peakFrames,
    prePadFrames,
    postPadFrames,
    frames.length,
  )

  return survivors.map((s, i) => ({
    strokeId: `stroke_${i}`,
    startFrame: bounds[i].start,
    endFrame: bounds[i].end,
    peakFrame: s.idx,
    fps: resolved.fps,
  }))
}

interface RefractoryCandidate {
  idx: number
  prominence: number
  width: number
}

function applyRefractoryByProminence(
  candidates: RefractoryCandidate[],
  refractoryFrames: number,
): RefractoryCandidate[] {
  if (candidates.length <= 1) return candidates
  // Score-descending greedy keep: highest-prominence wins when two
  // candidates fall within the refractory window of each other.
  const byProminence = [...candidates].sort(
    (a, b) => b.prominence - a.prominence,
  )
  const accepted: RefractoryCandidate[] = []
  for (const c of byProminence) {
    let blocked = false
    for (const a of accepted) {
      if (Math.abs(c.idx - a.idx) < refractoryFrames) {
        blocked = true
        break
      }
    }
    if (!blocked) accepted.push(c)
  }
  return accepted
}
