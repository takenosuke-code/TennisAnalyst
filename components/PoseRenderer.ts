import type { PoseFrame } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'
import {
  JOINT_GROUPS,
  SKELETON_CONNECTIONS,
  LANDMARK_INDICES,
} from '@/lib/jointAngles'
import { SHOT_TYPE_CONFIGS, scoreAngleVsIdeal } from '@/lib/shotTypeConfig'

export type RenderOptions = {
  visible: Record<JointGroup, boolean>
  showSkeleton: boolean
  color?: string           // joint dot color
  skeletonColor?: string   // skeleton line color
  scale?: number           // 0–1 alpha multiplier for ghost/overlay mode
  // Hide the off-hand's elbow and wrist so the overlay focuses on the
  // racket arm. Used for forehand (always single-arm), one-handed
  // backhand, and serve. 'right' = keep right_elbow/right_wrist, drop
  // left_elbow/left_wrist. Null = show both (default). Shoulders stay
  // visible on both sides since they anchor rotation.
  dominantHand?: 'left' | 'right' | null
  // When true, draw live degree labels next to each elbow and knee joint
  // on top of the skeleton. Matches SportAI's on-canvas annotation style
  // and turns computed joint angles from a chart-only signal into a
  // visible coaching cue.
  showAngles?: boolean
}

const JOINT_GROUP_COLORS: Record<JointGroup, string> = {
  shoulders: '#f59e0b',  // amber
  elbows: '#ef4444',     // red
  wrists: '#10b981',     // emerald (racket hand)
  hips: '#8b5cf6',       // violet
  knees: '#3b82f6',      // blue
  ankles: '#ec4899',     // pink
}

// Hard confidence cutoff: below this, the landmark (and any bone touching it)
// is dropped from the render entirely. A faded/ghost joint reads to the user
// as "the tracker thinks the knee is here" even at 20% opacity — we'd rather
// show nothing and let them see what the pipeline is actually confident about.
// Loosened from 0.6 -> 0.5: at 0.6 the overlay was dropping joints on every
// slightly-noisy frame (user: "too fluid, loses the joints easy"). 0.5 still
// filters out genuinely-off-screen / genuinely-hidden joints but keeps the
// skeleton anchored through small detection dips.
const VISIBILITY_CUTOFF = 0.5

// Per-bone width multipliers, applied to the global `boneWidth` scalar.
// `base` is the width at the proximal (declared-first) endpoint, `tip` at
// the distal end — so an upper-arm bone (shoulder→elbow) is fatter at the
// shoulder, a forearm (elbow→wrist) tapers further. Torso bones stay
// uniform. Multipliers are unitless; they shape limb mass relative to
// the size of the player in frame.
//
// Keys are constructed from SKELETON_CONNECTIONS' [from, to] pairs in
// declaration order, so the proximal end is always the `from` landmark.
const BONE_TAPER: Record<string, { base: number; tip: number }> = {
  // Torso — uniform, slightly wider than limbs.
  [`${LANDMARK_INDICES.LEFT_SHOULDER}-${LANDMARK_INDICES.RIGHT_SHOULDER}`]: { base: 1.5, tip: 1.5 },
  [`${LANDMARK_INDICES.LEFT_SHOULDER}-${LANDMARK_INDICES.LEFT_HIP}`]:      { base: 1.7, tip: 1.6 },
  [`${LANDMARK_INDICES.RIGHT_SHOULDER}-${LANDMARK_INDICES.RIGHT_HIP}`]:    { base: 1.7, tip: 1.6 },
  [`${LANDMARK_INDICES.LEFT_HIP}-${LANDMARK_INDICES.RIGHT_HIP}`]:          { base: 1.5, tip: 1.5 },
  // Arms — taper from torso outward.
  [`${LANDMARK_INDICES.LEFT_SHOULDER}-${LANDMARK_INDICES.LEFT_ELBOW}`]:    { base: 1.1, tip: 0.85 },
  [`${LANDMARK_INDICES.LEFT_ELBOW}-${LANDMARK_INDICES.LEFT_WRIST}`]:       { base: 0.85, tip: 0.55 },
  [`${LANDMARK_INDICES.RIGHT_SHOULDER}-${LANDMARK_INDICES.RIGHT_ELBOW}`]:  { base: 1.1, tip: 0.85 },
  [`${LANDMARK_INDICES.RIGHT_ELBOW}-${LANDMARK_INDICES.RIGHT_WRIST}`]:     { base: 0.85, tip: 0.55 },
  // Legs — thigh fatter than shin, both taper outward.
  [`${LANDMARK_INDICES.LEFT_HIP}-${LANDMARK_INDICES.LEFT_KNEE}`]:          { base: 1.3, tip: 1.05 },
  [`${LANDMARK_INDICES.LEFT_KNEE}-${LANDMARK_INDICES.LEFT_ANKLE}`]:        { base: 1.05, tip: 0.75 },
  [`${LANDMARK_INDICES.RIGHT_HIP}-${LANDMARK_INDICES.RIGHT_KNEE}`]:        { base: 1.3, tip: 1.05 },
  [`${LANDMARK_INDICES.RIGHT_KNEE}-${LANDMARK_INDICES.RIGHT_ANKLE}`]:      { base: 1.05, tip: 0.75 },
}

// Draw one bone as a filled, tapered trapezoid plus a soft drop shadow.
// `from` is proximal, `to` distal. `colorRgb` is parsed once by the caller
// so we can apply alpha at both ends without re-parsing per bone. Returns
// nothing — caller manages ctx.save/restore.
function drawTaperedBone(
  ctx: CanvasRenderingContext2D,
  from: { x: number; y: number },
  to: { x: number; y: number },
  baseW: number,
  tipW: number,
  colorRgb: { r: number; g: number; b: number },
  alpha: number,
): void {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const len = Math.hypot(dx, dy)
  if (len < 0.5) return
  // Unit perpendicular for offsetting the trapezoid edges.
  const px = -dy / len
  const py = dx / len
  const fhw = baseW / 2
  const thw = tipW / 2
  const x1 = from.x + px * fhw, y1 = from.y + py * fhw
  const x2 = from.x - px * fhw, y2 = from.y - py * fhw
  const x3 = to.x   - px * thw, y3 = to.y   - py * thw
  const x4 = to.x   + px * thw, y4 = to.y   + py * thw

  // Linear gradient along the bone — full alpha at the proximal end,
  // ~70% alpha at the distal end. Fakes form lighting (limbs fade into
  // their joint dots) without picking a second color.
  const grad = ctx.createLinearGradient(from.x, from.y, to.x, to.y)
  const { r, g, b } = colorRgb
  grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${alpha})`)
  grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, ${alpha * 0.7})`)

  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x2, y2)
  ctx.lineTo(x3, y3)
  ctx.lineTo(x4, y4)
  ctx.closePath()
  ctx.fillStyle = grad
  ctx.fill()
}

// Cheap CSS-color → {r, g, b} parser. Handles the two formats we feed in:
// `#rrggbb` and `rgba(r, g, b, a)` / `rgb(r, g, b)`. Falls back to white
// for anything else (e.g. exotic css var resolutions) so a parse miss
// just means "no tint", not a render crash.
function parseRgb(color: string): { r: number; g: number; b: number } {
  if (color.startsWith('#') && color.length === 7) {
    return {
      r: parseInt(color.slice(1, 3), 16),
      g: parseInt(color.slice(3, 5), 16),
      b: parseInt(color.slice(5, 7), 16),
    }
  }
  const m = color.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/)
  if (m) {
    return { r: parseInt(m[1], 10), g: parseInt(m[2], 10), b: parseInt(m[3], 10) }
  }
  return { r: 255, g: 255, b: 255 }
}

export function renderPose(
  ctx: CanvasRenderingContext2D,
  frame: PoseFrame,
  canvasWidth: number,
  canvasHeight: number,
  options: RenderOptions
): void {
  const {
    visible,
    showSkeleton,
    color,
    skeletonColor,
    scale = 1,
    dominantHand = null,
  } = options

  // Build a map of landmark id -> pixel coords plus z and visibility.
  // Z is needed to sort bones front-to-back so the far-side arm doesn't
  // paint over the near-side torso. Visibility is used to dim bones whose
  // endpoints the tracker is shaky on. Low-confidence landmarks are dropped
  // entirely (consistent with the legacy behavior). When dominantHand is
  // set, the off-hand elbow/wrist are dropped so bones touching them
  // disappear without a separate bone filter.
  const offHandIds: number[] =
    dominantHand === 'right'
      ? [LANDMARK_INDICES.LEFT_ELBOW, LANDMARK_INDICES.LEFT_WRIST]
      : dominantHand === 'left'
        ? [LANDMARK_INDICES.RIGHT_ELBOW, LANDMARK_INDICES.RIGHT_WRIST]
        : []
  const pixelMap = new Map<number, { x: number; y: number; z: number; vis: number }>()
  for (const lm of frame.landmarks) {
    if (lm.visibility < VISIBILITY_CUTOFF) continue
    if (offHandIds.includes(lm.id)) continue
    pixelMap.set(lm.id, {
      x: lm.x * canvasWidth,
      y: lm.y * canvasHeight,
      z: lm.z,
      vis: lm.visibility,
    })
  }

  // Compute a player-size scale factor so joint dots, bone widths, and
  // angle badges shrink proportionally when the subject is small in the
  // frame. Without this, a player who's 25% of the canvas height gets
  // joint dots that visually swamp them — the hardcoded 8/6/2 px radii
  // assume a player filling the canvas.
  //
  // Heuristic: bbox height of visible landmarks / canvas height. Capped
  // at [0.4, 1.0] so very small players still show readable dots and
  // very large players don't get oversized ones.
  const playerScale = computePlayerScale(pixelMap, canvasHeight)
  const dotOuterR = Math.max(3, 8 * playerScale)
  const dotColorR = Math.max(2.5, 6 * playerScale)
  const dotCenterR = Math.max(1, 2 * playerScale)
  const boneWidth = Math.max(1.5, 3 * playerScale)

  // Skeleton render. Three passes, all gated on showSkeleton:
  //   (1) Translucent torso silhouette behind the bones — a four-point
  //       polygon through L_shoulder → R_shoulder → R_hip → L_hip filled
  //       at low alpha. This is the single biggest "wireframe → person"
  //       shift; without it the bones read as data viz, with it they read
  //       as a body being traced. Per the SOTA review (Pose Animator,
  //       SwingVision, OnForm).
  //   (2) Tapered bone trapezoids, sorted far-to-near by mean z so the
  //       far arm doesn't draw over the near torso. Each bone's alpha is
  //       multiplied by min(vis_a, vis_b) clamped to [0.4, 1.0] so the
  //       lower-confidence side of the body recedes naturally. Drop
  //       shadow grounds the figure on the video.
  //   (3) Skipped here — joint dots still draw in the legacy block below.
  if (showSkeleton) {
    const baseWidth = boneWidth
    const strokeColor = skeletonColor ?? 'rgba(244, 241, 234, 0.95)'
    const colorRgb = parseRgb(strokeColor)

    // (1) Torso silhouette. Cream tinted, very low alpha so it implies
    // mass without obscuring the underlying video.
    const lSh = pixelMap.get(LANDMARK_INDICES.LEFT_SHOULDER)
    const rSh = pixelMap.get(LANDMARK_INDICES.RIGHT_SHOULDER)
    const lHip = pixelMap.get(LANDMARK_INDICES.LEFT_HIP)
    const rHip = pixelMap.get(LANDMARK_INDICES.RIGHT_HIP)
    if (lSh && rSh && lHip && rHip) {
      ctx.save()
      ctx.globalAlpha = 0.18 * scale
      ctx.fillStyle = `rgb(${colorRgb.r}, ${colorRgb.g}, ${colorRgb.b})`
      ctx.beginPath()
      ctx.moveTo(lSh.x, lSh.y)
      ctx.lineTo(rSh.x, rSh.y)
      ctx.lineTo(rHip.x, rHip.y)
      ctx.lineTo(lHip.x, lHip.y)
      ctx.closePath()
      ctx.fill()
      ctx.restore()
    }

    // (2) Tapered bones, z-sorted with visibility-driven alpha.
    type BoneDraw = {
      from: { x: number; y: number; z: number; vis: number }
      to:   { x: number; y: number; z: number; vis: number }
      base: number
      tip:  number
      meanZ: number
      alpha: number
    }
    const bones: BoneDraw[] = []
    for (const [fromId, toId] of SKELETON_CONNECTIONS) {
      const from = pixelMap.get(fromId)
      const to = pixelMap.get(toId)
      if (!from || !to) continue
      const taper = BONE_TAPER[`${fromId}-${toId}`] ?? { base: 1.0, tip: 0.8 }
      // Visibility gate: clamp to [0.4, 1.0] so a barely-visible joint
      // ghosts but doesn't disappear (the existing VISIBILITY_CUTOFF=0.5
      // already drops invisible ones). 0.4 is the floor so a bone is
      // never quite imperceptible.
      const conf = Math.max(0.4, Math.min(1, Math.min(from.vis, to.vis)))
      bones.push({
        from, to,
        base: baseWidth * taper.base * 1.6,
        tip:  baseWidth * taper.tip  * 1.6,
        // BlazePose: smaller z = closer to camera. Sorting by z desc
        // (far first) gives correct front-to-back painter ordering.
        meanZ: (from.z + to.z) / 2,
        alpha: 0.92 * conf * scale,
      })
    }
    bones.sort((a, b) => b.meanZ - a.meanZ)

    ctx.save()
    // Subtle drop shadow grounds the limb on the video. shadowBlur
    // fires for every fill that follows until reset, so we set it once.
    ctx.shadowColor = 'rgba(0, 0, 0, 0.55)'
    ctx.shadowBlur = Math.max(4, 6 * playerScale)
    ctx.shadowOffsetX = 0
    ctx.shadowOffsetY = Math.max(1, 2 * playerScale)
    for (const b of bones) {
      drawTaperedBone(ctx, b.from, b.to, b.base, b.tip, colorRgb, b.alpha)
    }
    ctx.restore()
  }

  // Draw joint dots per group with continuous opacity by visibility.
  for (const [group, ids] of Object.entries(JOINT_GROUPS) as unknown as [JointGroup, number[]][]) {
    if (!visible[group]) continue

    const dotColor = color ?? JOINT_GROUP_COLORS[group]
    ctx.save()

    ctx.globalAlpha = 0.9 * scale
    for (const id of ids) {
      const pos = pixelMap.get(id)
      if (!pos) continue

      // Outer ring (dark halo for legibility on bright backgrounds)
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, dotOuterR, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(0,0,0,0.5)'
      ctx.fill()

      // Colored dot
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, dotColorR, 0, Math.PI * 2)
      ctx.fillStyle = dotColor
      ctx.fill()

      // White center for contrast
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, dotCenterR, 0, Math.PI * 2)
      ctx.fillStyle = 'white'
      ctx.fill()
    }
    ctx.globalAlpha = 1
    ctx.restore()
  }
}

/**
 * Derive a 0.4–1.0 scale factor from how much canvas height the player
 * actually occupies. Used to size joint dots, bone strokes, and the
 * angle-label pills so they stay proportional on small-in-frame
 * subjects (e.g., a wide-court shot where the player is ~25% of the
 * frame). The 0.4 floor keeps dots visible on tiny subjects; the 1.0
 * ceiling preserves the default visual size when the player fills the
 * frame.
 *
 * Returns 1.0 when there's no usable bbox (sparse landmarks, unusual
 * pose) — equivalent to the legacy fixed-size behavior.
 */
function computePlayerScale(
  pixelMap: Map<number, { x: number; y: number }>,
  canvasHeight: number,
): number {
  if (pixelMap.size < 4 || canvasHeight <= 0) return 1
  let minY = Infinity
  let maxY = -Infinity
  for (const { y } of pixelMap.values()) {
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  const bboxH = maxY - minY
  if (bboxH <= 0) return 1
  // Linear scale based on player-height-fraction. A player filling the
  // canvas (fraction=1) gets scale=1. A player at 25% height gets
  // scale=0.55, which roughly halves dot area.
  const fraction = Math.max(0, Math.min(1, bboxH / canvasHeight))
  return Math.max(0.4, Math.min(1, 0.4 + fraction * 0.6))
}

// Compute the bounding box of visible landmarks in normalized [0,1] space.
// Returns { minX, minY, maxX, maxY } with padding.
export function computeLandmarkBounds(
  frame: PoseFrame,
  padding = 0.05
): { minX: number; minY: number; maxX: number; maxY: number } | null {
  const visible = frame.landmarks.filter((lm) => lm.visibility >= 0.3)
  if (visible.length === 0) return null

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const lm of visible) {
    if (lm.x < minX) minX = lm.x
    if (lm.y < minY) minY = lm.y
    if (lm.x > maxX) maxX = lm.x
    if (lm.y > maxY) maxY = lm.y
  }

  // Add padding
  const w = maxX - minX
  const h = maxY - minY
  minX = Math.max(0, minX - w * padding)
  minY = Math.max(0, minY - h * padding)
  maxX = Math.min(1, maxX + w * padding)
  maxY = Math.min(1, maxY + h * padding)

  return { minX, minY, maxX, maxY }
}

// Render pose with landmark coordinates zoomed/cropped to a bounding box.
// The bounds define a sub-region of the [0,1] coordinate space that should
// fill the entire canvas.
export function renderPoseZoomed(
  ctx: CanvasRenderingContext2D,
  frame: PoseFrame,
  canvasWidth: number,
  canvasHeight: number,
  bounds: { minX: number; minY: number; maxX: number; maxY: number },
  options: RenderOptions
): void {
  const {
    visible,
    showSkeleton,
    color,
    skeletonColor,
    scale = 1,
    dominantHand = null,
  } = options
  const bw = bounds.maxX - bounds.minX
  const bh = bounds.maxY - bounds.minY
  if (bw <= 0 || bh <= 0) return

  // Map landmark [0,1] coords into the cropped canvas space
  const toPixel = (x: number, y: number) => ({
    x: ((x - bounds.minX) / bw) * canvasWidth,
    y: ((y - bounds.minY) / bh) * canvasHeight,
  })

  // Hard cutoff matches renderPose: uncertain landmarks drop out rather
  // than ghost in at low opacity. Off-hand filter mirrors renderPose too.
  const offHandIds: number[] =
    dominantHand === 'right'
      ? [LANDMARK_INDICES.LEFT_ELBOW, LANDMARK_INDICES.LEFT_WRIST]
      : dominantHand === 'left'
        ? [LANDMARK_INDICES.RIGHT_ELBOW, LANDMARK_INDICES.RIGHT_WRIST]
        : []
  const pixelMap = new Map<number, { x: number; y: number }>()
  for (const lm of frame.landmarks) {
    if (lm.visibility < VISIBILITY_CUTOFF) continue
    if (offHandIds.includes(lm.id)) continue
    const p = toPixel(lm.x, lm.y)
    pixelMap.set(lm.id, { x: p.x, y: p.y })
  }

  // Draw skeleton bones with halo outline.
  if (showSkeleton) {
    ctx.save()
    const baseWidth = 3
    const strokeColor = skeletonColor ?? 'rgba(255,255,255,0.7)'

    for (const [fromId, toId] of SKELETON_CONNECTIONS) {
      const from = pixelMap.get(fromId)
      const to = pixelMap.get(toId)
      if (!from || !to) continue

      ctx.globalAlpha = 0.6 * scale
      ctx.strokeStyle = 'rgba(0,0,0,0.4)'
      ctx.lineWidth = baseWidth + 2
      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)
      ctx.stroke()

      ctx.globalAlpha = 0.6 * scale
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = baseWidth
      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)
      ctx.stroke()
    }
    ctx.restore()
  }

  // Draw joint dots with confidence-modulated opacity.
  for (const [group, ids] of Object.entries(JOINT_GROUPS) as unknown as [JointGroup, number[]][]) {
    if (!visible[group]) continue

    const dotColor = color ?? JOINT_GROUP_COLORS[group]
    ctx.save()

    ctx.globalAlpha = 0.9 * scale
    for (const id of ids) {
      const pos = pixelMap.get(id)
      if (!pos) continue

      // Outer ring
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 10, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(0,0,0,0.5)'
      ctx.fill()

      // Colored dot
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2)
      ctx.fillStyle = dotColor
      ctx.fill()

      // White center
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2)
      ctx.fillStyle = 'white'
      ctx.fill()
    }
    ctx.globalAlpha = 1
    ctx.restore()
  }
}

// Which joint angles get drawn as on-canvas labels, and where. For each
// entry: `anchor` is the landmark id the label is placed next to; `key`
// is the JointAngles field. Labels are skipped when the anchor landmark
// is below the visibility cutoff (same gate as the skeleton) OR the
// angle itself is not computed for this frame.
const ANGLE_LABELS: { key: keyof import('@/lib/supabase').JointAngles; anchor: number; side: 'left' | 'right' }[] = [
  { key: 'right_elbow', anchor: LANDMARK_INDICES.RIGHT_ELBOW, side: 'right' },
  { key: 'left_elbow', anchor: LANDMARK_INDICES.LEFT_ELBOW, side: 'left' },
  { key: 'right_knee', anchor: LANDMARK_INDICES.RIGHT_KNEE, side: 'right' },
  { key: 'left_knee', anchor: LANDMARK_INDICES.LEFT_KNEE, side: 'left' },
]

// Colors keyed to the deviation level that scoreAngleVsIdeal returns.
// `good` = inside the ideal range, `moderate` = within 15° of it,
// `poor` = further off. When no shot-type config is available we keep the
// neutral white fill (the legacy behavior).
const BENCHMARK_COLORS = {
  good: '#10b981',      // emerald
  moderate: '#f59e0b',  // amber
  poor: '#ef4444',      // red
  unknown: '#ffffff',   // white (legacy neutral)
} as const

/**
 * Draw degree labels next to elbow and knee joints (matches SportAI's
 * on-skeleton annotation style). Call AFTER renderPose so the labels
 * layer on top of the bones.
 *
 * When `shotType` + `phase` are supplied, each badge is color-coded
 * against the ideal range for that shot at that phase (from
 * SHOT_TYPE_CONFIGS.keyAngleSpecs via scoreAngleVsIdeal). Without them
 * or when no matching spec exists, badges render in the legacy white.
 *
 * Phase is typically 'contact' (passed in by the caller when the
 * current frame is the detected peak). Other phases are rendered
 * neutral-white to avoid a sea of red during prep/follow-through where
 * the angles legitimately span a wide range.
 */
export function renderAngleLabels(
  ctx: CanvasRenderingContext2D,
  frame: PoseFrame,
  canvasWidth: number,
  canvasHeight: number,
  options: Pick<RenderOptions, 'dominantHand' | 'scale'> & {
    shotType?: string | null
    phase?: string | null
  },
): void {
  const { dominantHand = null, scale = 1, shotType = null, phase = null } = options

  // Off-hand elbow labels are suppressed when we know the racket hand
  // (keeps the hitting-arm number uncrowded). Knees always render —
  // footwork reads from both legs.
  const offHandElbow: 'left' | 'right' | null =
    dominantHand === 'right' ? 'left'
      : dominantHand === 'left' ? 'right'
        : null

  // Shot-type config lookup for benchmark coloring. Skipped when shotType
  // or phase is null so we don't invent a benchmark against the wrong
  // reference (unknown shot type → no color).
  const config = shotType ? SHOT_TYPE_CONFIGS[shotType] : null
  const canBenchmark = !!(config && phase)

  // Player-size scale (matches renderPose). Without this, badges visibly
  // swamp small-in-frame players. Computed once per render call from
  // the same visible-landmark bbox as the joint dots.
  const renderableLandmarks = new Map<number, { x: number; y: number }>()
  for (const lm of frame.landmarks) {
    if (lm.visibility < VISIBILITY_CUTOFF) continue
    renderableLandmarks.set(lm.id, {
      x: lm.x * canvasWidth,
      y: lm.y * canvasHeight,
    })
  }
  const playerScale = computePlayerScale(renderableLandmarks, canvasHeight)
  // Font + pill scale separately from radius scale because text below
  // ~9px gets hard to read on phone screens — clamp at a minimum that
  // stays readable even when the player is tiny.
  const fontPx = Math.max(10, Math.round(14 * playerScale))
  const deltaFontPx = Math.max(9, Math.round(11 * playerScale))
  const pillH = Math.max(14, Math.round(20 * playerScale))
  const pillH2 = Math.max(22, Math.round(32 * playerScale))
  const padX = Math.max(4, Math.round(6 * playerScale))
  const offset = Math.max(10, Math.round(16 * playerScale))

  ctx.save()
  ctx.globalAlpha = 0.95 * scale
  ctx.font = `600 ${fontPx}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
  ctx.textBaseline = 'middle'

  for (const { key, anchor, side } of ANGLE_LABELS) {
    // Elbow labels: skip the off-hand side so the racket arm's angle
    // isn't competing with a non-swinging arm's number.
    if (key.includes('elbow') && offHandElbow === side) continue

    const angle = frame.joint_angles[key]
    if (angle == null) continue

    const lm = frame.landmarks.find((l) => l.id === anchor)
    if (!lm || lm.visibility < VISIBILITY_CUTOFF) continue

    const px = lm.x * canvasWidth
    const py = lm.y * canvasHeight

    // Look up the ideal range for this joint at this phase (if benchmarking).
    let level: keyof typeof BENCHMARK_COLORS = 'unknown'
    let delta: number | null = null
    if (canBenchmark && config) {
      const spec = config.keyAngleSpecs.find(
        (s) => s.angleKey === key && s.phase === phase,
      )
      if (spec) {
        const scored = scoreAngleVsIdeal(angle, spec.idealRange)
        level = scored.level
        delta = scored.delta
      }
    }

    // Badge text: "145°" on top; when off-ideal, a smaller "+8° off" beneath.
    const primary = `${Math.round(angle)}°`
    const deltaText =
      canBenchmark && delta != null && delta > 0 && level !== 'good'
        ? `${Math.round(delta)}° off`
        : null

    // Pill geometry: two-line badges are taller. Width = widest of the
    // two lines + padding. All dimensions scale with playerScale so a
    // far-away player gets smaller pills.
    const primaryW = ctx.measureText(primary).width
    ctx.save()
    ctx.font = `500 ${deltaFontPx}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
    const deltaW = deltaText ? ctx.measureText(deltaText).width : 0
    ctx.restore()
    const currentPillH = deltaText ? pillH2 : pillH
    const pillW = Math.max(primaryW, deltaW) + padX * 2

    const offsetX = side === 'right' ? offset : -offset - pillW
    const pillX = px + offsetX
    const pillY = py - currentPillH / 2

    // Dark pill background for legibility over any video.
    ctx.fillStyle = 'rgba(0,0,0,0.75)'
    roundRect(ctx, pillX, pillY, pillW, currentPillH, 4)
    ctx.fill()

    // Border: neutral faint white when not benchmarking, colored when we
    // have a deviation level. The border is the "at-a-glance" signal —
    // the text stays white for readability even on a red/green pill.
    const borderColor = canBenchmark ? BENCHMARK_COLORS[level] : 'rgba(255,255,255,0.25)'
    ctx.strokeStyle = borderColor
    ctx.lineWidth = canBenchmark && level !== 'unknown' ? 2 : 1
    roundRect(ctx, pillX, pillY, pillW, currentPillH, 4)
    ctx.stroke()

    ctx.fillStyle = 'white'
    ctx.textAlign = 'left'
    // Primary degree text. On the two-line badge the primary sits in
    // the upper third; on the single-line badge it's vertically centered.
    const primaryY = deltaText ? pillY + currentPillH * 0.35 : pillY + currentPillH / 2
    ctx.fillText(primary, pillX + padX, primaryY)

    if (deltaText) {
      // Secondary "+Xº off" text in the benchmark color so the deviation
      // reads at a glance without having to see the border.
      ctx.save()
      ctx.font = `500 ${deltaFontPx}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
      ctx.fillStyle = BENCHMARK_COLORS[level]
      ctx.fillText(deltaText, pillX + padX, pillY + currentPillH * 0.72)
      ctx.restore()
    }
  }

  ctx.restore()
}

// Tiny rounded-rect helper — CanvasRenderingContext2D.roundRect exists
// but shipped late and Safari <17 doesn't have it. This is the defensive
// version.
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const rr = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.lineTo(x + w - rr, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr)
  ctx.lineTo(x + w, y + h - rr)
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h)
  ctx.lineTo(x + rr, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr)
  ctx.lineTo(x, y + rr)
  ctx.quadraticCurveTo(x, y, x + rr, y)
  ctx.closePath()
}

// Normalize landmarks to a common scale for overlay comparison
// Anchors at mid-hip, scales by hip width
export function normalizeLandmarks(frame: PoseFrame): PoseFrame {
  const lHip = frame.landmarks.find((l) => l.id === LANDMARK_INDICES.LEFT_HIP)
  const rHip = frame.landmarks.find((l) => l.id === LANDMARK_INDICES.RIGHT_HIP)

  if (!lHip || !rHip) return frame

  const hipMidX = (lHip.x + rHip.x) / 2
  const hipMidY = (lHip.y + rHip.y) / 2
  const hipWidth = Math.sqrt((rHip.x - lHip.x) ** 2 + (rHip.y - lHip.y) ** 2)
  // Guard against degenerate detections (hips nearly co-located) to prevent Infinity scale
  if (hipWidth < 0.01) return frame
  const scale = 1 / hipWidth

  const normalized = frame.landmarks.map((lm) => ({
    ...lm,
    x: (lm.x - hipMidX) * scale + 0.5,
    y: (lm.y - hipMidY) * scale + 0.5,
  }))

  return { ...frame, landmarks: normalized }
}
