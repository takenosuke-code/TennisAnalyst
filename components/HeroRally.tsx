'use client'

import { useEffect, useRef, useSyncExternalStore } from 'react'

/*
 * HeroRally — two figures rally across a net on a tennis court, with
 * the headline copy centered between them.
 *
 * The right figure (the "you" figure, with angle pills) and a mirrored
 * left figure (the "opponent") trade shots. Each figure has its own
 * state machine, anticipates the incoming ball, swings AT it, and the
 * ball reverses on contact-phase. Court lines + net are static SVG
 * decoration.
 *
 * No collision with the headline copy — the words are visual only,
 * the ball just flies between the figures.
 */

// ---------- Joint structure ----------

type Joints = {
  nose: [number, number]
  left_shoulder: [number, number]
  right_shoulder: [number, number]
  left_elbow: [number, number]
  right_elbow: [number, number]
  left_wrist: [number, number]
  right_wrist: [number, number]
  left_hip: [number, number]
  right_hip: [number, number]
  left_knee: [number, number]
  right_knee: [number, number]
  left_ankle: [number, number]
  right_ankle: [number, number]
}

type Pt = [number, number]

const BONE_LENGTHS = {
  hipHalfWidth: 0.04,
  shoulderHalfWidth: 0.05,
  trunk: 0.25,
  neck: 0.1,
  upperArm: 0.139,
  forearm: 0.133,
  thigh: 0.17,
  shin: 0.2,
} as const

type KeyframeAngles = {
  t: number
  hipCenter: [number, number]
  trunk: number; neck: number
  uArmL: number; fArmL: number
  uArmR: number; fArmR: number
  thighL: number; shinL: number
  thighR: number; shinR: number
  racket: number; racketLen: number
}

// Right-arm follow-through stays below "over-the-shoulder" while the
// racket completes its circular sweep: uArmR / fArmR cap below
// horizontal-left during follow-through, but the racket angle keeps
// rotating in the same direction (no bounce-back). The racket-head
// arc peaks around chest height because the wrist sits at hip level
// at the racket's "up" phase. Off-arm (uArmL / fArmL) stays in
// (85°, 120°) — elbow always below or outside the left shoulder, so
// the non-racket hand never crosses inward across the body.
const KEYFRAME_ANGLES: KeyframeAngles[] = [
  // Forehand modeled on the kinetic-chain phases in
  // lib/biomechanics-reference.ts:
  //  - t=0.4  Loading peak: full coil, racket cocked behind shoulder,
  //           knees flexed (back-leg knee ~140°, biomech ideal range
  //           130-150°), off-arm extended toward the ball.
  //  - t=0.55 Hips fire (kinetic chain begins — hip peak ~75ms before
  //           contact in elite players, ~t=0.55 in this 1.6s cycle).
  //  - t=0.6  Trunk catches up (peak forward lean), arm extending.
  //  - t=0.65 CONTACT: dominant elbow ~120° (within 100-140° biomech
  //           range), arm out front, racket extending toward target.
  //  - t=0.725 Follow-through: arm continues across the body, racket
  //           sweeps out toward the net (no bounce-back).
  //  - t=0.8  Finish: arm relaxes back toward ready, racket curls up
  //           through chest level (capped — never over the shoulder).
  // Off-arm uArmL/fArmL stay ≥ 110/125° throughout so the off-hand
  // never crosses inward across the body.
  { t: 0.0, hipCenter: [0.5, 0.55], trunk: -90.0, neck: -90.0,
    uArmL: 110.0, fArmL: 125.0, uArmR: 85.0, fArmR: 85.0,
    thighL: 92.0, shinL: 90.0, thighR: 88.0, shinR: 90.0,
    racket: -75.0, racketLen: 0.13 },
  // Unit turn — shoulders begin coiling, racket pulls back to right
  { t: 0.15, hipCenter: [0.51, 0.555], trunk: -90.0, neck: -94.0,
    uArmL: 120.0, fArmL: 145.0, uArmR: 70.0, fArmR: 20.0,
    thighL: 92.0, shinL: 90.0, thighR: 88.0, shinR: 90.0,
    racket: -65.0, racketLen: 0.13 },
  // Loading peak — full coil, racket cocked, back-leg loaded ~140°
  { t: 0.4, hipCenter: [0.53, 0.56], trunk: -85.0, neck: -100.0,
    uArmL: 140.0, fArmL: 170.0, uArmR: 20.0, fArmR: -30.0,
    thighL: 105.0, shinL: 80.0, thighR: 72.0, shinR: 105.0,
    racket: -100.0, racketLen: 0.13 },
  // Drop — racket loops below contact zone, weight on back foot
  { t: 0.475, hipCenter: [0.525, 0.56], trunk: -85.0, neck: -98.0,
    uArmL: 125.0, fArmL: 145.0, uArmR: 50.0, fArmR: 70.0,
    thighL: 103.0, shinL: 82.0, thighR: 75.0, shinR: 100.0,
    racket: 40.0, racketLen: 0.13 },
  // Hip rotation peak — kinetic chain firing, forward swing begins
  { t: 0.55, hipCenter: [0.51, 0.555], trunk: -83.0, neck: -96.0,
    uArmL: 115.0, fArmL: 135.0, uArmR: 80.0, fArmR: 110.0,
    thighL: 100.0, shinL: 86.0, thighR: 80.0, shinR: 95.0,
    racket: 110.0, racketLen: 0.13 },
  // Trunk peak — peak forward lean, arm extending toward contact
  { t: 0.6, hipCenter: [0.50, 0.555], trunk: -82.0, neck: -94.0,
    uArmL: 110.0, fArmL: 130.0, uArmR: 105.0, fArmR: 145.0,
    thighL: 98.0, shinL: 88.0, thighR: 82.0, shinR: 93.0,
    racket: 155.0, racketLen: 0.13 },
  // CONTACT — arm extended (elbow ~120°), front knee driving up,
  // off-arm tucks slightly, racket toward target
  { t: 0.65, hipCenter: [0.49, 0.55], trunk: -85.0, neck: -92.0,
    uArmL: 110.0, fArmL: 130.0, uArmR: 120.0, fArmR: 170.0,
    thighL: 96.0, shinL: 90.0, thighR: 85.0, shinR: 92.0,
    racket: -160.0, racketLen: 0.13 },
  // Follow-through — racket continues out toward the net, arm sweeps
  // across the body. uArmR caps at 135° (well below "behind the head"
  // 180°+) so the racket-head arc peaks around chest, not over the
  // off-shoulder.
  { t: 0.725, hipCenter: [0.48, 0.55], trunk: -88.0, neck: -90.0,
    uArmL: 110.0, fArmL: 125.0, uArmR: 135.0, fArmR: 170.0,
    thighL: 94.0, shinL: 90.0, thighR: 87.0, shinR: 91.0,
    racket: -130.0, racketLen: 0.13 },
  // Finish — kinetic chain decelerating, arm relaxes back toward
  // ready. Racket continues circular sweep through "up" (-90 raw =
  // 270° unwrapped), wrist now near hip height so racket stays low.
  { t: 0.8, hipCenter: [0.47, 0.55], trunk: -90.0, neck: -88.0,
    uArmL: 110.0, fArmL: 125.0, uArmR: 110.0, fArmR: 125.0,
    thighL: 92.0, shinL: 90.0, thighR: 89.0, shinR: 90.0,
    racket: -90.0, racketLen: 0.12 },
]
const N = KEYFRAME_ANGLES.length

const BONES: [keyof Joints, keyof Joints][] = [
  ['left_shoulder', 'right_shoulder'],
  ['left_hip', 'right_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_shoulder', 'left_hip'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
]

// Per-bone taper widths in SVG units. Index matches BONES above.
// `base` is the width at the proximal (declared-first) endpoint, `tip`
// at the distal end — so an upper arm fattens at the shoulder and
// tapers toward the elbow. Torso edges stay near-uniform; arm/leg
// extremities taper most. These shape limb mass; without them every
// bone is a uniform-width stick and the figure reads as wireframe.
const BONE_TAPER: { base: number; tip: number }[] = [
  { base: 4.5, tip: 4.5 }, // L_shoulder ↔ R_shoulder
  { base: 4.5, tip: 4.5 }, // L_hip ↔ R_hip
  { base: 5.5, tip: 5.0 }, // R_shoulder → R_hip (torso side)
  { base: 5.5, tip: 5.0 }, // L_shoulder → L_hip (torso side)
  { base: 4.0, tip: 3.2 }, // L upper arm
  { base: 3.2, tip: 2.2 }, // L forearm
  { base: 4.0, tip: 3.2 }, // R upper arm
  { base: 3.2, tip: 2.2 }, // R forearm
  { base: 4.5, tip: 3.6 }, // L thigh
  { base: 3.6, tip: 2.6 }, // L shin
  { base: 4.5, tip: 3.6 }, // R thigh
  { base: 3.6, tip: 2.6 }, // R shin
]

// Build the SVG `points` attribute for a tapered trapezoid bone. Given
// two pixel-space endpoints and base/tip widths, compute the four
// corners of the quad. Used by paintFigure to push trapezoid geometry
// into <polygon> elements each frame.
function trapezoidPoints(
  ax: number, ay: number, bx: number, by: number,
  baseW: number, tipW: number,
): string {
  const dx = bx - ax
  const dy = by - ay
  const len = Math.hypot(dx, dy)
  if (len < 0.5) return ''
  // Unit perpendicular for offsetting the trapezoid edges.
  const px = -dy / len
  const py = dx / len
  const fhw = baseW / 2
  const thw = tipW / 2
  const x1 = ax + px * fhw, y1 = ay + py * fhw
  const x2 = ax - px * fhw, y2 = ay - py * fhw
  const x3 = bx - px * thw, y3 = by - py * thw
  const x4 = bx + px * thw, y4 = by + py * thw
  return `${x1},${y1} ${x2},${y2} ${x3},${y3} ${x4},${y4}`
}

const JOINT_KEYS: (keyof Joints)[] = [
  'nose',
  'left_shoulder', 'right_shoulder',
  'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist',
  'left_hip', 'right_hip',
  'left_knee', 'right_knee',
  'left_ankle', 'right_ankle',
]

// ---------- Animation parameters ----------

const FIGURE_PX_HEIGHT = 380
const FIGURE_PX_WIDTH = FIGURE_PX_HEIGHT * (9 / 16)
const FIGURE_EDGE_PADDING = 28
const FIGURE_Y_TRACK_LERP = 0.07
const FIGURE_Y_TRACK_AMPLITUDE_PX = 70

const BALL_RADIUS = 7
const BALL_SPEED = 540

// SWING_MS = 1600. Real pro forehands run 1.0-1.5s broadcast; this
// is ~0.6x speed, so the swing reads as a deliberate replica of the
// real motion rather than a slow-mo demo or a twitchy hero loop.
// User feedback iteration: 700 -> 1100 -> 1600.
const SWING_MS = 1600
const RACKET_HIT_RADIUS = 50
// At smootherstep5, solving `6u⁵ - 15u⁴ + 10u³ = 0.65` gives u ≈ 0.582,
// so contact lands 0.582 * 1600 ≈ 931 ms after the swing fires.
const SWING_TO_CONTACT_MS = 931
const SWING_CONTACT_PHASE = 0.65
const RACKET_CONTACT_NORM_X = 0.326
const RACKET_CONTACT_NORM_Y = 0.508

// Angle labels — small, analytical, tabular-figure readout near each
// tracked joint. Painted on both figures so the "AI is reading every
// body in frame" intent reads on either side of the rally. Layout per
// joint: the label sits at (offsetX, offsetY) from the joint dot.
// Mirrored figures get offsetX and text-anchor flipped at paint time
// so labels still float outboard of the body. Text is tabular-num so
// digits stay aligned frame-to-frame as the angles tick.
const ANGLE_LABELS: Array<{
  joint: keyof Joints
  parent: keyof Joints
  child: keyof Joints
  offsetX: number
  offsetY: number
  anchor: 'start' | 'end'
}> = [
  { joint: 'right_elbow', parent: 'right_shoulder', child: 'right_wrist', offsetX: 22, offsetY: -2, anchor: 'start' },
  { joint: 'left_elbow', parent: 'left_shoulder', child: 'left_wrist', offsetX: -22, offsetY: -2, anchor: 'end' },
  { joint: 'right_knee', parent: 'right_hip', child: 'right_ankle', offsetX: 22, offsetY: 4, anchor: 'start' },
  { joint: 'left_knee', parent: 'left_hip', child: 'left_ankle', offsetX: -22, offsetY: 4, anchor: 'end' },
]

// 2D angle at vertex b, given three pixel-space points. Returns
// degrees rounded to whole numbers (0-180 for joint angles).
function angleAtPx(a: Pt, b: Pt, c: Pt): number {
  const v1x = a[0] - b[0]
  const v1y = a[1] - b[1]
  const v2x = c[0] - b[0]
  const v2y = c[1] - b[1]
  const dot = v1x * v2x + v1y * v2y
  const m1 = Math.hypot(v1x, v1y)
  const m2 = Math.hypot(v2x, v2y)
  if (m1 === 0 || m2 === 0) return 0
  const cos = Math.max(-1, Math.min(1, dot / (m1 * m2)))
  return Math.round((Math.acos(cos) * 180) / Math.PI)
}

// ---------- Math ----------

function shortDelta(a: number, b: number): number {
  let d = b - a
  while (d > 180) d -= 360
  while (d <= -180) d += 360
  return d
}
function lerpAngle(a: number, b: number, u: number): number {
  return a + shortDelta(a, b) * u
}
function lerp(a: number, b: number, u: number): number {
  return a + (b - a) * u
}
function findSegment(phase: number): { i: number; u: number } {
  const p = phase - Math.floor(phase)
  for (let i = 0; i < N - 1; i++) {
    if (p < KEYFRAME_ANGLES[i + 1].t) {
      const t0 = KEYFRAME_ANGLES[i].t
      const t1 = KEYFRAME_ANGLES[i + 1].t
      return { i, u: (p - t0) / (t1 - t0) }
    }
  }
  const t0 = KEYFRAME_ANGLES[N - 1].t
  return { i: N - 1, u: (p - t0) / (1.0 - t0) }
}
// Perlin's smootherstep5 — `6t⁵ - 15t⁴ + 10t³`. C² continuous (the
// second derivative is zero at both endpoints, so jerk doesn't jump
// at the loop seam the way it would with smoothstep). Mathematically
// matches Flash & Hogan's minimum-jerk reaching profile (1985), which
// human subjects empirically rate as more natural than smoothstep
// (PeerJ 2020). Drop-in replacement for the prior `t*t*(3-2t)`.
function warpPhase(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10)
}

// Precomputed segment widths in phase-space. Used by the Hermite
// interpolation below to build C¹-continuous angle curves across
// non-uniformly-spaced keyframes. The wrap segment (KF[N-1] -> KF[0])
// uses 1.0 as the implicit end time.
const SEGMENT_DT: number[] = (() => {
  const dt: number[] = []
  for (let i = 0; i < N; i++) {
    const t0 = KEYFRAME_ANGLES[i].t
    const t1 = i + 1 < N ? KEYFRAME_ANGLES[i + 1].t : 1.0
    dt.push(t1 - t0)
  }
  return dt
})()

// Cubic Hermite interpolation of a single angle channel across 4
// control keyframes (i-1, i, i+1, i+2) with non-uniform time spacing.
// Tangents at i and i+1 are Catmull-Rom-derived (average of incoming
// and outgoing slopes scaled by segment widths) so the curve is C¹
// at every keyframe boundary. Angles are unwrapped via shortDelta
// chained outward from a1, keeping the math correct across the
// cyclic seam where raw angles can wrap past ±180°.
function lerpAngleHermite(
  rawA: number,   // angle at i-1
  a1: number,    // angle at i (segment start, anchor)
  rawB: number,  // angle at i+1 (segment end)
  rawC: number,  // angle at i+2
  u: number,     // local parameter in [0, 1] within segment i
  dt0: number,   // segment width before i (t[i] - t[i-1])
  dt1: number,   // segment width at i (t[i+1] - t[i])
  dt2: number,   // segment width after i (t[i+2] - t[i+1])
): number {
  // Unwrap into a continuous span around a1.
  const a0 = a1 - shortDelta(rawA, a1)
  const a2 = a1 + shortDelta(a1, rawB)
  const a3 = a2 + shortDelta(rawB, rawC)
  // Tangents in angle/time units (non-uniform Catmull-Rom).
  const m1 = 0.5 * ((a1 - a0) / dt0 + (a2 - a1) / dt1)
  const m2 = 0.5 * ((a2 - a1) / dt1 + (a3 - a2) / dt2)
  // Hermite basis on the local parameter.
  const u2 = u * u
  const u3 = u2 * u
  const h00 = 2 * u3 - 3 * u2 + 1
  const h10 = u3 - 2 * u2 + u
  const h01 = -2 * u3 + 3 * u2
  const h11 = u3 - u2
  // Tangents must be scaled by the segment width to map (angle/time)
  // back into raw angle units along the [0,1] local parameter.
  return h00 * a1 + h10 * dt1 * m1 + h01 * a2 + h11 * dt1 * m2
}

function buildSkeleton(phase: number): { joints: Joints; racketHead: Pt } {
  const { i, u } = findSegment(phase)
  // 4 control keyframes around the active segment. Cyclic indexing
  // for the wrap segment (i = N-1).
  const Aprev = KEYFRAME_ANGLES[(i - 1 + N) % N]
  const A = KEYFRAME_ANGLES[i]
  const B = KEYFRAME_ANGLES[(i + 1) % N]
  const Bnext = KEYFRAME_ANGLES[(i + 2) % N]
  const dt0 = SEGMENT_DT[(i - 1 + N) % N]
  const dt1 = SEGMENT_DT[i]
  const dt2 = SEGMENT_DT[(i + 1) % N]

  // hipCenter (2D position) — keep linear; the figure's translation
  // is small and not visually load-bearing for smoothness.
  const hipCx = lerp(A.hipCenter[0], B.hipCenter[0], u)
  const hipCy = lerp(A.hipCenter[1], B.hipCenter[1], u)
  // racketLen — scalar, near-constant; linear is fine.
  const racketLen = lerp(A.racketLen, B.racketLen, u)

  // Bone angles — Hermite with Catmull-Rom tangents, unwrapped per
  // bone via shortDelta. Eliminates the per-keyframe velocity kinks
  // that linear lerp introduced (which read as the "robotic" feel
  // even at 60fps).
  const H = (rA: number, a1: number, rB: number, rC: number) =>
    lerpAngleHermite(rA, a1, rB, rC, u, dt0, dt1, dt2)
  const trunk  = H(Aprev.trunk,  A.trunk,  B.trunk,  Bnext.trunk)
  const neck   = H(Aprev.neck,   A.neck,   B.neck,   Bnext.neck)
  const uArmL  = H(Aprev.uArmL,  A.uArmL,  B.uArmL,  Bnext.uArmL)
  const fArmL  = H(Aprev.fArmL,  A.fArmL,  B.fArmL,  Bnext.fArmL)
  const uArmR  = H(Aprev.uArmR,  A.uArmR,  B.uArmR,  Bnext.uArmR)
  const fArmR  = H(Aprev.fArmR,  A.fArmR,  B.fArmR,  Bnext.fArmR)
  const thighL = H(Aprev.thighL, A.thighL, B.thighL, Bnext.thighL)
  const shinL  = H(Aprev.shinL,  A.shinL,  B.shinL,  Bnext.shinL)
  const thighR = H(Aprev.thighR, A.thighR, B.thighR, Bnext.thighR)
  const shinR  = H(Aprev.shinR,  A.shinR,  B.shinR,  Bnext.shinR)
  const racket = H(Aprev.racket, A.racket, B.racket, Bnext.racket)

  const polar = (L: number, deg: number): Pt => {
    const r = (deg * Math.PI) / 180
    return [L * Math.cos(r), L * Math.sin(r)]
  }

  const left_hip: Pt = [hipCx - BONE_LENGTHS.hipHalfWidth, hipCy]
  const right_hip: Pt = [hipCx + BONE_LENGTHS.hipHalfWidth, hipCy]
  const [tx, ty] = polar(BONE_LENGTHS.trunk, trunk)
  const midShx = hipCx + tx
  const midShy = hipCy + ty
  const left_shoulder: Pt = [midShx - BONE_LENGTHS.shoulderHalfWidth, midShy]
  const right_shoulder: Pt = [midShx + BONE_LENGTHS.shoulderHalfWidth, midShy]
  const [nx, ny] = polar(BONE_LENGTHS.neck, neck)
  const nose: Pt = [midShx + nx, midShy + ny]
  const [ueLx, ueLy] = polar(BONE_LENGTHS.upperArm, uArmL)
  const left_elbow: Pt = [left_shoulder[0] + ueLx, left_shoulder[1] + ueLy]
  const [ueRx, ueRy] = polar(BONE_LENGTHS.upperArm, uArmR)
  const right_elbow: Pt = [right_shoulder[0] + ueRx, right_shoulder[1] + ueRy]
  const [feLx, feLy] = polar(BONE_LENGTHS.forearm, fArmL)
  const left_wrist: Pt = [left_elbow[0] + feLx, left_elbow[1] + feLy]
  const [feRx, feRy] = polar(BONE_LENGTHS.forearm, fArmR)
  const right_wrist: Pt = [right_elbow[0] + feRx, right_elbow[1] + feRy]
  const [tLx, tLy] = polar(BONE_LENGTHS.thigh, thighL)
  const left_knee: Pt = [left_hip[0] + tLx, left_hip[1] + tLy]
  const [tRx, tRy] = polar(BONE_LENGTHS.thigh, thighR)
  const right_knee: Pt = [right_hip[0] + tRx, right_hip[1] + tRy]
  const [sLx, sLy] = polar(BONE_LENGTHS.shin, shinL)
  const left_ankle: Pt = [left_knee[0] + sLx, left_knee[1] + sLy]
  const [sRx, sRy] = polar(BONE_LENGTHS.shin, shinR)
  const right_ankle: Pt = [right_knee[0] + sRx, right_knee[1] + sRy]
  const [rkx, rky] = polar(racketLen, racket)
  const racketHead: Pt = [right_wrist[0] + rkx, right_wrist[1] + rky]

  return {
    joints: {
      nose,
      left_shoulder, right_shoulder,
      left_elbow, right_elbow,
      left_wrist, right_wrist,
      left_hip, right_hip,
      left_knee, right_knee,
      left_ankle, right_ankle,
    },
    racketHead,
  }
}

// ---------- prefers-reduced-motion ----------

function subscribeReducedMotion(cb: () => void): () => void {
  if (typeof window === 'undefined') return () => {}
  const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
  mq.addEventListener('change', cb)
  return () => mq.removeEventListener('change', cb)
}
function getReducedMotion(): boolean {
  if (typeof window === 'undefined') return false
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}
function getServerReducedMotion(): boolean {
  return false
}

// ---------- Per-figure rendering helpers ----------

interface FigureRefs {
  // Translucent torso polygon drawn behind the bones — implies body
  // mass so the figure reads as a person, not a wireframe.
  silhouette: SVGPolygonElement | null
  // Bones are filled trapezoid polygons (see BONE_TAPER) so each one
  // can taper from base to tip. SVG <line> can't taper along its own
  // length, so polygons are the cheap workaround.
  bones: (SVGPolygonElement | null)[]
  joints: (SVGCircleElement | null)[]
  racketHead: SVGCircleElement | null
  racketGrip: SVGLineElement | null
}

interface LabelRef {
  text: SVGTextElement | null
}

// ---------- Component ----------

export default function HeroRally() {
  const reducedMotion = useSyncExternalStore(
    subscribeReducedMotion,
    getReducedMotion,
    getServerReducedMotion,
  )
  const containerRef = useRef<HTMLDivElement | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)

  // Two sets of figure refs — right (you) and left (opponent).
  const rightRefs = useRef<FigureRefs>({
    silhouette: null,
    bones: [],
    joints: [],
    racketHead: null,
    racketGrip: null,
  })
  const leftRefs = useRef<FigureRefs>({
    silhouette: null,
    bones: [],
    joints: [],
    racketHead: null,
    racketGrip: null,
  })
  const rightLabelRefs = useRef<LabelRef[]>(ANGLE_LABELS.map(() => ({ text: null })))
  const leftLabelRefs = useRef<LabelRef[]>(ANGLE_LABELS.map(() => ({ text: null })))
  const ballRef = useRef<SVGCircleElement | null>(null)
  const courtRef = useRef<SVGGElement | null>(null)

  useEffect(() => {
    const container = containerRef.current
    const svg = svgRef.current
    if (!container || !svg) return

    let containerRect = container.getBoundingClientRect()

    function setSvgSize() {
      svg!.setAttribute('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`)
      svg!.setAttribute('width', String(containerRect.width))
      svg!.setAttribute('height', String(containerRect.height))
    }
    setSvgSize()

    // Figure positions — pinned to the edges of the section.
    let rightBaseX = containerRect.width - FIGURE_PX_WIDTH / 2 - FIGURE_EDGE_PADDING
    let leftBaseX = FIGURE_PX_WIDTH / 2 + FIGURE_EDGE_PADDING
    let figureRestY = containerRect.height / 2
    let rightBaseY = figureRestY
    let leftBaseY = figureRestY

    // toPx converts normalized keyframe coords to pixel coords. The
    // `mirrored` flag flips the X axis so the left figure renders as
    // a mirror-image of the right (its racket sweeps the opposite
    // direction, contact zone is on its right side / net side).
    function toPx(p: Pt, baseX: number, baseY: number, mirrored: boolean): Pt {
      const dx = (p[0] - 0.5) * FIGURE_PX_WIDTH
      return [baseX + (mirrored ? -dx : dx), baseY + (p[1] - 0.5) * FIGURE_PX_HEIGHT]
    }

    function paintFigure(
      phase: number,
      refs: FigureRefs,
      baseX: number,
      baseY: number,
      mirrored: boolean,
      labelSlots: LabelRef[] | null,
    ): Pt {
      const { joints, racketHead } = buildSkeleton(phase)
      const px: Record<keyof Joints, Pt> = {} as Record<keyof Joints, Pt>
      for (const key of JOINT_KEYS) {
        px[key] = toPx(joints[key], baseX, baseY, mirrored)
      }
      JOINT_KEYS.forEach((key, idx) => {
        const dot = refs.joints[idx]
        if (!dot) return
        const [x, y] = px[key]
        dot.setAttribute('cx', String(x))
        dot.setAttribute('cy', String(y))
      })
      // Torso silhouette — quad through L_shoulder → R_shoulder →
      // R_hip → L_hip, drawn behind the bones at low alpha. Implies
      // body mass without obscuring the underlying court color.
      const sil = refs.silhouette
      if (sil) {
        const [lShx, lShy] = px.left_shoulder
        const [rShx, rShy] = px.right_shoulder
        const [lHipx, lHipy] = px.left_hip
        const [rHipx, rHipy] = px.right_hip
        sil.setAttribute(
          'points',
          `${lShx},${lShy} ${rShx},${rShy} ${rHipx},${rHipy} ${lHipx},${lHipy}`,
        )
      }
      BONES.forEach(([from, to], idx) => {
        const poly = refs.bones[idx]
        if (!poly) return
        const [x1, y1] = px[from]
        const [x2, y2] = px[to]
        const taper = BONE_TAPER[idx]
        const pts = trapezoidPoints(x1, y1, x2, y2, taper.base, taper.tip)
        if (pts) poly.setAttribute('points', pts)
      })
      const wristPx = px.right_wrist
      const headPx = toPx(racketHead, baseX, baseY, mirrored)
      const headEl = refs.racketHead
      if (headEl) {
        headEl.setAttribute('cx', String(headPx[0]))
        headEl.setAttribute('cy', String(headPx[1]))
      }
      const grip = refs.racketGrip
      if (grip) {
        grip.setAttribute('x1', String(wristPx[0]))
        grip.setAttribute('y1', String(wristPx[1]))
        grip.setAttribute('x2', String(headPx[0]))
        grip.setAttribute('y2', String(headPx[1]))
      }
      // Angle readouts — painted next to each tracked joint so the
      // analytical "AI is reading your body" intent reads on both
      // figures. For mirrored figures (left/opponent), the offset and
      // anchor flip so labels still float outboard of the body. No
      // leader line; cream-on-green proximity is enough.
      if (labelSlots) {
        ANGLE_LABELS.forEach((label, idx) => {
          const slot = labelSlots[idx]
          if (!slot || !slot.text) return
          const [jx, jy] = px[label.joint]
          const ox = mirrored ? -label.offsetX : label.offsetX
          const anchor = mirrored
            ? (label.anchor === 'start' ? 'end' : 'start')
            : label.anchor
          slot.text.setAttribute('x', String(jx + ox))
          slot.text.setAttribute('y', String(jy + label.offsetY))
          slot.text.setAttribute('text-anchor', anchor)
          const angle = angleAtPx(px[label.parent], px[label.joint], px[label.child])
          slot.text.textContent = `${angle}°`
        })
      }
      return headPx
    }

    // Racket-head pixel position at a phase, without painting.
    function racketHeadAtPhase(phase: number, baseX: number, baseY: number, mirrored: boolean): Pt {
      const { racketHead } = buildSkeleton(phase)
      return toPx(racketHead, baseX, baseY, mirrored)
    }

    // Pick a vy that lands the ball in the opponent's reachable Y
    // band so they can hit it back. No words to clamp against now;
    // only constraint is "land where the other figure can swing."
    function pickReturnVy(ballX: number, ballY: number, opponentX: number): number {
      const dist = Math.abs(opponentX - ballX)
      if (dist <= 8) return 0
      const t = dist / BALL_SPEED
      const figRest = containerRect.height / 2
      const figMin = figRest - FIGURE_Y_TRACK_AMPLITUDE_PX
      const figMax = figRest + FIGURE_Y_TRACK_AMPLITUDE_PX
      const minVy = (figMin - ballY) / t
      const maxVy = (figMax - ballY) / t
      return minVy + Math.random() * (maxVy - minVy)
    }

    // Initial paint.
    paintFigure(0, rightRefs.current, rightBaseX, rightBaseY, false, rightLabelRefs.current)
    paintFigure(0, leftRefs.current, leftBaseX, leftBaseY, true, leftLabelRefs.current)
    const restRacketRight = racketHeadAtPhase(0, rightBaseX, rightBaseY, false)
    if (ballRef.current) {
      ballRef.current.setAttribute('cx', String(restRacketRight[0] - 60))
      ballRef.current.setAttribute('cy', String(restRacketRight[1]))
    }

    if (reducedMotion) return

    // Per-figure rally state.
    type FigState = { mode: 'preswing' | 'swinging'; swingStart: number; hitRegistered: boolean }
    const right: FigState = { mode: 'preswing', swingStart: 0, hitRegistered: false }
    const left: FigState = { mode: 'preswing', swingStart: 0, hitRegistered: false }

    // Ball — start near right figure heading left.
    const racketYOffsetFromHip = (RACKET_CONTACT_NORM_Y - 0.5) * FIGURE_PX_HEIGHT
    const ball = {
      x: restRacketRight[0] - 80,
      y: restRacketRight[1],
      vx: -BALL_SPEED,
      vy: pickReturnVy(restRacketRight[0] - 80, restRacketRight[1], leftBaseX),
    }

    let lastTime = performance.now()
    let rafId = 0
    let cancelled = false

    function step(now: number) {
      if (cancelled) return
      const dt = Math.min(0.05, (now - lastTime) / 1000)
      lastTime = now

      // Resolve each figure's swing phase.
      let rightPhase = 0
      if (right.mode === 'swinging') {
        const u = Math.min(1, (now - right.swingStart) / SWING_MS)
        rightPhase = warpPhase(u)
        if (u >= 1) {
          right.mode = 'preswing'
          right.hitRegistered = false
        }
      }
      let leftPhase = 0
      if (left.mode === 'swinging') {
        const u = Math.min(1, (now - left.swingStart) / SWING_MS)
        leftPhase = warpPhase(u)
        if (u >= 1) {
          left.mode = 'preswing'
          left.hitRegistered = false
        }
      }

      // ─── Per-figure preswing: track Y, fire when ball is close ───────
      // Right figure: ball is incoming when vx > 0 (moving rightward).
      if (right.mode === 'preswing' && ball.vx > 0) {
        const racketContactX = rightBaseX + (RACKET_CONTACT_NORM_X - 0.5) * FIGURE_PX_WIDTH
        const tToContact = (racketContactX - ball.x) / ball.vx
        if (tToContact > 0) {
          const predBallY = ball.y + ball.vy * tToContact
          const targetY = Math.max(
            figureRestY - FIGURE_Y_TRACK_AMPLITUDE_PX,
            Math.min(figureRestY + FIGURE_Y_TRACK_AMPLITUDE_PX, predBallY - racketYOffsetFromHip),
          )
          if (tToContact * 1000 <= SWING_TO_CONTACT_MS) {
            rightBaseY = targetY
            right.mode = 'swinging'
            right.swingStart = now
            right.hitRegistered = false
          } else {
            rightBaseY += (targetY - rightBaseY) * FIGURE_Y_TRACK_LERP
          }
        }
      }
      // Left figure: ball is incoming when vx < 0 (moving leftward).
      // The mirrored figure's racket-contact-X is to the RIGHT of its
      // base (toward the net), so we add instead of subtract the offset.
      if (left.mode === 'preswing' && ball.vx < 0) {
        const racketContactX = leftBaseX - (RACKET_CONTACT_NORM_X - 0.5) * FIGURE_PX_WIDTH
        const tToContact = (racketContactX - ball.x) / ball.vx
        if (tToContact > 0) {
          const predBallY = ball.y + ball.vy * tToContact
          const targetY = Math.max(
            figureRestY - FIGURE_Y_TRACK_AMPLITUDE_PX,
            Math.min(figureRestY + FIGURE_Y_TRACK_AMPLITUDE_PX, predBallY - racketYOffsetFromHip),
          )
          if (tToContact * 1000 <= SWING_TO_CONTACT_MS) {
            leftBaseY = targetY
            left.mode = 'swinging'
            left.swingStart = now
            left.hitRegistered = false
          } else {
            leftBaseY += (targetY - leftBaseY) * FIGURE_Y_TRACK_LERP
          }
        }
      }

      // Ball physics.
      ball.x += ball.vx * dt
      ball.y += ball.vy * dt

      // ─── Right figure contact ────────────────────────────────────────
      if (right.mode === 'swinging' && !right.hitRegistered && rightPhase >= SWING_CONTACT_PHASE) {
        const racketPx = racketHeadAtPhase(SWING_CONTACT_PHASE, rightBaseX, rightBaseY, false)
        const dx = ball.x - racketPx[0]
        const dy = ball.y - racketPx[1]
        if (dx * dx + dy * dy <= (RACKET_HIT_RADIUS * 1.5) ** 2) {
          ball.vx = -Math.abs(ball.vx)  // send leftward
          ball.vy = pickReturnVy(ball.x, ball.y, leftBaseX)
          ball.x = racketPx[0]
          ball.y = racketPx[1]
        }
        right.hitRegistered = true
      }
      // ─── Left figure contact ─────────────────────────────────────────
      if (left.mode === 'swinging' && !left.hitRegistered && leftPhase >= SWING_CONTACT_PHASE) {
        const racketPx = racketHeadAtPhase(SWING_CONTACT_PHASE, leftBaseX, leftBaseY, true)
        const dx = ball.x - racketPx[0]
        const dy = ball.y - racketPx[1]
        if (dx * dx + dy * dy <= (RACKET_HIT_RADIUS * 1.5) ** 2) {
          ball.vx = Math.abs(ball.vx)  // send rightward
          ball.vy = pickReturnVy(ball.x, ball.y, rightBaseX)
          ball.x = racketPx[0]
          ball.y = racketPx[1]
        }
        left.hitRegistered = true
      }

      // Paint everything.
      paintFigure(rightPhase, rightRefs.current, rightBaseX, rightBaseY, false, rightLabelRefs.current)
      paintFigure(leftPhase, leftRefs.current, leftBaseX, leftBaseY, true, leftLabelRefs.current)
      if (ballRef.current) {
        ballRef.current.setAttribute('cx', String(ball.x))
        ballRef.current.setAttribute('cy', String(ball.y))
      }

      rafId = requestAnimationFrame(step)
    }

    // Gate the loop on visibility. Two reasons:
    //   1. Performance — no rAF cost when the hero isn't onscreen.
    //   2. Scroll feel — a continuously-painting absolutely-positioned
    //      element near the top of the page can cause the browser's
    //      scroll-anchoring algorithm to fight the user's scroll
    //      ("pulls them back up"). Stopping rAF when the section is
    //      out of view kills that interaction entirely. The container
    //      also gets `overflow-anchor: none` as a belt-and-suspenders
    //      so anchoring never targets a moving figure.
    let onScreen = true
    let docVisible = typeof document !== 'undefined' ? !document.hidden : true
    function shouldRun() { return onScreen && docVisible }

    function start() {
      if (rafId) return
      rafId = requestAnimationFrame((t) => {
        // Reset lastTime on resume so dt doesn't jump after a pause
        // (which would warp the ball + swing phase forward).
        lastTime = t
        step(t)
      })
    }
    function stop() {
      if (!rafId) return
      cancelAnimationFrame(rafId)
      rafId = 0
    }

    const io = new IntersectionObserver(
      (entries) => {
        onScreen = entries[0]?.isIntersecting ?? false
        if (shouldRun()) start()
        else stop()
      },
      { threshold: 0 },
    )
    io.observe(container)

    function onVisibility() {
      docVisible = !document.hidden
      if (shouldRun()) start()
      else stop()
    }
    document.addEventListener('visibilitychange', onVisibility)

    // Kick off if the section is already onscreen at mount.
    if (shouldRun()) start()

    const ro = new ResizeObserver(() => {
      containerRect = container.getBoundingClientRect()
      rightBaseX = containerRect.width - FIGURE_PX_WIDTH / 2 - FIGURE_EDGE_PADDING
      leftBaseX = FIGURE_PX_WIDTH / 2 + FIGURE_EDGE_PADDING
      figureRestY = containerRect.height / 2
      setSvgSize()
    })
    ro.observe(container)

    return () => {
      cancelled = true
      stop()
      io.disconnect()
      ro.disconnect()
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [reducedMotion])

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 pointer-events-none hidden lg:block"
      aria-hidden="true"
      // Opt out of scroll anchoring. Without this, a moving SVG near
      // the top of the page can become the browser's anchor target,
      // and per-frame motion is read as "content shifted above me" —
      // the browser then nudges scroll to compensate, fighting the
      // user. The IntersectionObserver below already pauses the loop
      // off-screen; this guards the still-onscreen-but-near-edge
      // case where anchoring could engage during scroll.
      style={{ overflowAnchor: 'none' }}
    >
      <svg
        ref={svgRef}
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Court — single-stroke cream lines at low opacity. Drawn as
            percentages of the viewBox so they scale with the section.
            Layout (top-down convention even though the figures are
            side-views — designed inconsistency for a stylized hero):
              - 4 horizontal lines = top/bottom singles + doubles sidelines
              - 2 vertical service lines (1/4 and 3/4 from each baseline)
              - 1 horizontal center service line
              - Net at the exact center vertical */}
        <g ref={courtRef} stroke="var(--cream)" strokeOpacity="0.32" strokeWidth="1.5" fill="none">
          {/* Top doubles + singles sidelines */}
          <line x1="3%" y1="8%" x2="97%" y2="8%" />
          <line x1="3%" y1="18%" x2="97%" y2="18%" />
          {/* Bottom singles + doubles sidelines */}
          <line x1="3%" y1="82%" x2="97%" y2="82%" />
          <line x1="3%" y1="92%" x2="97%" y2="92%" />
          {/* Baselines (left + right ends) */}
          <line x1="3%" y1="8%" x2="3%" y2="92%" />
          <line x1="97%" y1="8%" x2="97%" y2="92%" />
          {/* Service lines */}
          <line x1="28%" y1="18%" x2="28%" y2="82%" />
          <line x1="72%" y1="18%" x2="72%" y2="82%" />
          {/* Center service line — the "T" */}
          <line x1="28%" y1="50%" x2="72%" y2="50%" />
        </g>

        {/* Net — vertical band + tape, centered, posts at top/bottom.
            Spans the full court height now (6%–94%) since the court
            fills the green. */}
        <g pointerEvents="none">
          {/* Net mesh — vertical hatches for the woven look */}
          <g stroke="var(--ink)" strokeOpacity="0.45" strokeWidth="1">
            {Array.from({ length: 12 }).map((_, i) => {
              const yPct = 8 + i * 7
              return <line key={`n${i}`} x1="50%" y1={`${yPct}%`} x2="50%" y2={`${yPct + 5.5}%`} />
            })}
          </g>
          {/* Net top tape — cream bar across the top of the net */}
          <rect x="49.5%" y="6%" width="1%" height="2%" fill="var(--cream)" opacity="0.9" />
          {/* Net body — translucent ink, sits between the two tapes */}
          <rect x="49.5%" y="8%" width="1%" height="84%" fill="var(--ink)" opacity="0.35" />
          {/* Net bottom tape — cream bar mirroring the top */}
          <rect x="49.5%" y="92%" width="1%" height="2%" fill="var(--cream)" opacity="0.9" />
          {/* Net posts (ink dots at the very ends, framing both tapes) */}
          <circle cx="50%" cy="6%" r="4" fill="var(--ink)" />
          <circle cx="50%" cy="94%" r="4" fill="var(--ink)" />
        </g>

        {/* Right figure — silhouette, bones, racket, joint dots.
            Silhouette renders first so it sits behind everything. */}
        <polygon
          ref={(el) => { rightRefs.current.silhouette = el }}
          fill="var(--cream)"
          fillOpacity="0.18"
        />
        {BONES.map((_, idx) => (
          <polygon
            key={`r-bone-${idx}`}
            ref={(el) => { rightRefs.current.bones[idx] = el }}
            fill="var(--cream)"
            fillOpacity="0.95"
          />
        ))}
        <line
          ref={(el) => { rightRefs.current.racketGrip = el }}
          stroke="var(--cream)"
          strokeOpacity="0.95"
          strokeWidth="2.25"
          strokeLinecap="round"
        />
        <circle
          ref={(el) => { rightRefs.current.racketHead = el }}
          r="7"
          fill="var(--clay)"
        />
        {JOINT_KEYS.map((key, idx) => (
          <circle
            key={`r-joint-${idx}`}
            ref={(el) => { rightRefs.current.joints[idx] = el }}
            r={key === 'nose' ? 8 : 5.5}
            fill="var(--cream)"
          />
        ))}

        {/* Left (opponent) figure — same drawing. Slightly lower
            opacity to suggest it's the secondary actor. */}
        <polygon
          ref={(el) => { leftRefs.current.silhouette = el }}
          fill="var(--cream)"
          fillOpacity="0.16"
        />
        {BONES.map((_, idx) => (
          <polygon
            key={`l-bone-${idx}`}
            ref={(el) => { leftRefs.current.bones[idx] = el }}
            fill="var(--cream)"
            fillOpacity="0.85"
          />
        ))}
        <line
          ref={(el) => { leftRefs.current.racketGrip = el }}
          stroke="var(--cream)"
          strokeOpacity="0.85"
          strokeWidth="2.25"
          strokeLinecap="round"
        />
        <circle
          ref={(el) => { leftRefs.current.racketHead = el }}
          r="7"
          fill="var(--clay)"
          opacity="0.95"
        />
        {JOINT_KEYS.map((key, idx) => (
          <circle
            key={`l-joint-${idx}`}
            ref={(el) => { leftRefs.current.joints[idx] = el }}
            r={key === 'nose' ? 8 : 5.5}
            fill="var(--cream)"
            opacity="0.95"
          />
        ))}

        {/* Angle readouts — tabular-figure text per joint on BOTH
            figures. paintFigure rewrites text-anchor per frame to flip
            for the mirrored opponent, so the JSX anchor is just a
            harmless default. Tabular figures via 'tnum' keep digits
            column-aligned frame-to-frame so the readouts read like a
            telemetry HUD instead of jittering as digits change width. */}
        {ANGLE_LABELS.map((label, idx) => (
          <text
            key={`r-label-${idx}`}
            ref={(el) => {
              const slot = rightLabelRefs.current[idx]
              if (slot) slot.text = el
            }}
            fill="var(--cream)"
            fillOpacity="0.95"
            fontFamily="var(--font-sans)"
            fontSize="10"
            fontWeight="600"
            textAnchor={label.anchor}
            dominantBaseline="middle"
            style={{
              fontFeatureSettings: '"tnum" 1, "ss01" 1',
              letterSpacing: '0.02em',
              pointerEvents: 'none',
            }}
          >
            0°
          </text>
        ))}
        {ANGLE_LABELS.map((label, idx) => (
          <text
            key={`l-label-${idx}`}
            ref={(el) => {
              const slot = leftLabelRefs.current[idx]
              if (slot) slot.text = el
            }}
            fill="var(--cream)"
            fillOpacity="0.85"
            fontFamily="var(--font-sans)"
            fontSize="10"
            fontWeight="600"
            textAnchor={label.anchor === 'start' ? 'end' : 'start'}
            dominantBaseline="middle"
            style={{
              fontFeatureSettings: '"tnum" 1, "ss01" 1',
              letterSpacing: '0.02em',
              pointerEvents: 'none',
            }}
          >
            0°
          </text>
        ))}

        {/* Tennis ball */}
        <circle
          ref={ballRef}
          r={BALL_RADIUS}
          fill="var(--clay)"
          stroke="var(--cream)"
          strokeWidth="1.25"
          strokeOpacity="0.7"
        />
      </svg>
    </div>
  )
}
