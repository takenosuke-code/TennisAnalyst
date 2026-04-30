'use client'

import { useEffect, useRef, useSyncExternalStore } from 'react'

/*
 * HeroRally — figure plays a continuous rally; ball bounces off the
 * headline copy.
 *
 * v3 of the hero animation (replaces v2 HeroSwingTracer's 3-cycle-and-
 * settle pattern). The ball is the protagonist: it travels left, hits
 * the H1's bounding box, returns, the figure swings, ball reverses,
 * loop forever. The figure's swing fires *because* the ball arrived
 * — there's no idle / settled state.
 *
 * Architectural shape:
 *   - This component is positioned-absolute over the entire hero
 *     section so figure + ball share one coordinate space.
 *   - The H1's bounding rect is read via a forwarded ref + ResizeObserver
 *     (Plan §3 path a) so the ball collides against the actual rendered
 *     text, not a hard-coded x.
 *   - Figure FK math (skeleton building, angle interpolation) is the
 *     same as v2's HeroSwingTracer — the keyframes, warpPhase, and
 *     buildSkeleton functions are re-derived here so the v2 component
 *     can stay in tree as a fallback during iteration.
 *   - SVG viewBox is the hero section's pixel dimensions in CSS units.
 *     Resize updates the viewBox so the layout stays pixel-aligned.
 *
 * Plan defaults (see /Users/neil/.claude/plans/okay-so-i-put-cuddly-church.md):
 *   - Forehand only (no backhand mirror this round).
 *   - Figure rooted at one spot — only the swing animates.
 *   - Full H1 bounding box as the wall (single rect, both lines).
 *   - No angle pills, no ball trail in v1.
 *   - On scroll-out / scroll-in the rally restarts from rest.
 *   - Reduced-motion: static figure with ball parked at racket.
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

// 9-keyframe forehand sequence — same numbers as HeroSwingTracer v2
// (computed via atan2 from the original position keyframes).
const KEYFRAME_ANGLES: KeyframeAngles[] = [
  { t: 0.0, hipCenter: [0.5, 0.55], trunk: -90.0, neck: -90.0,
    uArmL: 112.9, fArmL: 109.8, uArmR: 67.1, fArmR: 70.2,
    thighL: 92.0, shinL: 90.0, thighR: 88.0, shinR: 90.0,
    racket: -79.8, racketLen: 0.13 },
  { t: 0.15, hipCenter: [0.52, 0.55], trunk: -90.0, neck: -95.7,
    uArmL: 95.0, fArmL: 84.8, uArmR: 69.9, fArmR: 55.1,
    thighL: 92.0, shinL: 90.0, thighR: 88.0, shinR: 90.0,
    racket: -79.8, racketLen: 0.13 },
  { t: 0.4, hipCenter: [0.53, 0.56], trunk: -85.4, neck: -106.7,
    uArmL: 20.1, fArmL: 70.2, uArmR: -15.0, fArmR: -65.2,
    thighL: 109.9, shinL: 79.9, thighR: 70.1, shinR: 100.1,
    racket: -100.2, racketLen: 0.13 },
  { t: 0.475, hipCenter: [0.525, 0.56], trunk: -86.6, neck: -104.0,
    uArmL: 85.0, fArmL: 77.8, uArmR: 2.1, fArmR: 10.0,
    thighL: 108.1, shinL: 82.0, thighR: 71.9, shinR: 98.0,
    racket: 29.9, racketLen: 0.13 },
  { t: 0.55, hipCenter: [0.52, 0.56], trunk: -87.7, neck: -101.3,
    uArmL: 149.7, fArmL: 84.8, uArmR: 20.1, fArmR: 84.8,
    thighL: 105.0, shinL: 85.1, thighR: 75.0, shinR: 94.9,
    racket: 100.2, racketLen: 0.13 },
  { t: 0.6, hipCenter: [0.505, 0.555], trunk: -87.7, neck: -98.5,
    uArmL: 175.0, fArmL: 81.8, uArmR: 64.9, fArmR: 98.2,
    thighL: 101.9, shinL: 86.0, thighR: 78.1, shinR: 94.9,
    racket: 150.1, racketLen: 0.13 },
  { t: 0.65, hipCenter: [0.5, 0.55], trunk: -90.0, neck: -95.7,
    uArmL: -159.9, fArmL: 10.0, uArmR: 112.0, fArmR: 112.1,
    thighL: 100.2, shinL: 88.0, thighR: 79.8, shinR: 94.9,
    racket: -160.2, racketLen: 0.13 },
  { t: 0.725, hipCenter: [0.49, 0.55], trunk: -91.1, neck: -90.0,
    uArmL: -149.7, fArmL: 14.9, uArmR: 155.8, fArmR: -160.2,
    thighL: 97.1, shinL: 89.1, thighR: 81.9, shinR: 92.9,
    racket: -150.1, racketLen: 0.13 },
  { t: 0.8, hipCenter: [0.48, 0.55], trunk: -92.3, neck: -84.3,
    uArmL: -140.0, fArmL: 19.8, uArmR: -159.9, fArmR: -70.2,
    thighL: 95.1, shinL: 90.0, thighR: 84.9, shinR: 90.0,
    racket: -94.8, racketLen: 0.12 },
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

// The figure's bounding rectangle (in normalized 0-1 keyframe space)
// gets scaled into pixels via FIGURE_PX_HEIGHT. The figure renders
// anchored at FIGURE_RIGHT_PX from the right edge of the hero. The
// figure's X is locked; only Y can drift to track the incoming ball
// (FIGURE_Y_TRACK_AMPLITUDE_PX caps the lean range).
const FIGURE_PX_HEIGHT = 480
const FIGURE_PX_WIDTH = FIGURE_PX_HEIGHT * (9 / 16)
const FIGURE_RIGHT_PADDING = 32
const FIGURE_Y_TRACK_LERP = 0.07         // soft tracking — 0=frozen, 1=instant
const FIGURE_Y_TRACK_AMPLITUDE_PX = 100  // max +/- shift from rest Y

// Ball
const BALL_RADIUS = 7
const BALL_SPEED = 540 // px/sec — feels like a rally tempo, not Pong

// Swing
const SWING_MS = 700
const RACKET_HIT_RADIUS = 50

// The keyframes' contact frame is at t=0.65, but warpPhase squishes
// the timing. We solve numerically: warpPhase(u) = u²·(3-2u) = 0.65 →
// u ≈ 0.604. So contact happens at SWING_MS · 0.604 ≈ 423ms after the
// swing starts. We fire the swing this many ms BEFORE the ball reaches
// the racket's contact-frame X — so the racket sweeps through the
// strike zone exactly as the ball arrives. Pong-like timing.
const SWING_TO_CONTACT_MS = 423
const SWING_CONTACT_PHASE = 0.65

// In the keyframes, the racket head at contact is at normalized
// (0.326, 0.508) — across the body, slightly below center. The figure
// must aim this point at the incoming ball, not its hipCenter.
const RACKET_CONTACT_NORM_X = 0.326
const RACKET_CONTACT_NORM_Y = 0.508

// Angle pills — 4 pills surfaced on the figure to match the screenshot
// reference. Each pill labels the joint angle (in degrees) at one of
// the body's main hinges. ANGLE_PILLS[i].joint is where the pill
// anchors; .parent and .child define the angle (parent → joint →
// child). offsetX/offsetY in pixels relative to the joint.
const ANGLE_PILLS: Array<{
  joint: keyof Joints
  parent: keyof Joints
  child: keyof Joints
  offsetX: number
  offsetY: number
}> = [
  // Right (dominant) elbow — pill outboard right
  { joint: 'right_elbow', parent: 'right_shoulder', child: 'right_wrist', offsetX: 30, offsetY: -8 },
  // Left elbow — pill outboard left
  { joint: 'left_elbow', parent: 'left_shoulder', child: 'left_wrist', offsetX: -54, offsetY: -8 },
  // Right knee — pill outboard right
  { joint: 'right_knee', parent: 'right_hip', child: 'right_ankle', offsetX: 22, offsetY: 6 },
  // Left knee — pill outboard left
  { joint: 'left_knee', parent: 'left_hip', child: 'left_ankle', offsetX: -54, offsetY: 6 },
]
const PILL_W = 44
const PILL_H = 18

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
function warpPhase(t: number): number {
  return t * t * (3 - 2 * t)
}

function buildSkeleton(phase: number): { joints: Joints; racketHead: Pt } {
  const { i, u } = findSegment(phase)
  const A = KEYFRAME_ANGLES[i]
  const B = KEYFRAME_ANGLES[(i + 1) % N]

  const hipCx = lerp(A.hipCenter[0], B.hipCenter[0], u)
  const hipCy = lerp(A.hipCenter[1], B.hipCenter[1], u)
  const trunk = lerpAngle(A.trunk, B.trunk, u)
  const neck = lerpAngle(A.neck, B.neck, u)
  const uArmL = lerpAngle(A.uArmL, B.uArmL, u)
  const fArmL = lerpAngle(A.fArmL, B.fArmL, u)
  const uArmR = lerpAngle(A.uArmR, B.uArmR, u)
  const fArmR = lerpAngle(A.fArmR, B.fArmR, u)
  const thighL = lerpAngle(A.thighL, B.thighL, u)
  const shinL = lerpAngle(A.shinL, B.shinL, u)
  const thighR = lerpAngle(A.thighR, B.thighR, u)
  const shinR = lerpAngle(A.shinR, B.shinR, u)
  const racket = lerpAngle(A.racket, B.racket, u)
  const racketLen = lerp(A.racketLen, B.racketLen, u)

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

// ---------- Component ----------

interface Props {
  /** Ref to the H1 the ball should bounce off of. */
  headlineRef: React.RefObject<HTMLElement | null>
}

interface ElRefs {
  bones: (SVGLineElement | null)[]
  joints: (SVGCircleElement | null)[]
  racketHead: SVGCircleElement | null
  racketGrip: SVGLineElement | null
  ball: SVGCircleElement | null
  pills: Array<{ rect: SVGRectElement | null; text: SVGTextElement | null } | null>
}

// Compute the angle in degrees at vertex b given three points a, b, c
// in pixel space. Output is clamped to [0, 180] (we only care about
// magnitude — coaching angles never exceed 180°).
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

export default function HeroRally({ headlineRef }: Props) {
  const reducedMotion = useSyncExternalStore(
    subscribeReducedMotion,
    getReducedMotion,
    getServerReducedMotion,
  )
  const containerRef = useRef<HTMLDivElement | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const refs = useRef<ElRefs>({
    bones: [],
    joints: [],
    racketHead: null,
    racketGrip: null,
    ball: null,
    pills: ANGLE_PILLS.map(() => ({ rect: null, text: null })),
  })

  useEffect(() => {
    const container = containerRef.current
    const svg = svgRef.current
    const headline = headlineRef.current
    if (!container || !svg || !headline) return

    // Geometry — recomputed on resize.
    let containerRect = container.getBoundingClientRect()
    let headlineLocal = computeHeadlineLocal(headline, container)
    let figureBaseX = containerRect.width - FIGURE_PX_WIDTH / 2 - FIGURE_RIGHT_PADDING
    let figureBaseY = containerRect.height / 2

    function paintAtSwingPhase(phase: number) {
      const { joints, racketHead } = buildSkeleton(phase)
      // Convert normalized [0,1] keyframe coords to pixel coords centered on figureBase.
      const toPx = (p: Pt): Pt => [
        figureBaseX + (p[0] - 0.5) * FIGURE_PX_WIDTH,
        figureBaseY + (p[1] - 0.5) * FIGURE_PX_HEIGHT,
      ]
      // Cache joint pixel coords once — pills re-use them.
      const px: Record<keyof Joints, Pt> = {} as Record<keyof Joints, Pt>
      for (const key of JOINT_KEYS) {
        px[key] = toPx(joints[key])
      }
      // Joints
      JOINT_KEYS.forEach((key, idx) => {
        const dot = refs.current.joints[idx]
        if (!dot) return
        const [x, y] = px[key]
        dot.setAttribute('cx', String(x))
        dot.setAttribute('cy', String(y))
      })
      // Bones
      BONES.forEach(([from, to], idx) => {
        const line = refs.current.bones[idx]
        if (!line) return
        const [x1, y1] = px[from]
        const [x2, y2] = px[to]
        line.setAttribute('x1', String(x1))
        line.setAttribute('y1', String(y1))
        line.setAttribute('x2', String(x2))
        line.setAttribute('y2', String(y2))
      })
      // Racket
      const wristPx = px.right_wrist
      const headPx = toPx(racketHead)
      const headEl = refs.current.racketHead
      if (headEl) {
        headEl.setAttribute('cx', String(headPx[0]))
        headEl.setAttribute('cy', String(headPx[1]))
      }
      const grip = refs.current.racketGrip
      if (grip) {
        grip.setAttribute('x1', String(wristPx[0]))
        grip.setAttribute('y1', String(wristPx[1]))
        grip.setAttribute('x2', String(headPx[0]))
        grip.setAttribute('y2', String(headPx[1]))
      }
      // Angle pills — compute angle, position pill near the joint, write text.
      ANGLE_PILLS.forEach((pill, idx) => {
        const slot = refs.current.pills[idx]
        if (!slot || !slot.rect || !slot.text) return
        const angle = angleAtPx(px[pill.parent], px[pill.joint], px[pill.child])
        const cx = px[pill.joint][0] + pill.offsetX
        const cy = px[pill.joint][1] + pill.offsetY
        slot.rect.setAttribute('x', String(cx - PILL_W / 2))
        slot.rect.setAttribute('y', String(cy - PILL_H / 2))
        slot.text.setAttribute('x', String(cx))
        slot.text.setAttribute('y', String(cy + 4))
        slot.text.textContent = `${angle}°`
      })
      return headPx
    }

    // Get the racket head pixel coord at a given phase without painting.
    function racketHeadAtPhase(phase: number): Pt {
      const { racketHead } = buildSkeleton(phase)
      return [
        figureBaseX + (racketHead[0] - 0.5) * FIGURE_PX_WIDTH,
        figureBaseY + (racketHead[1] - 0.5) * FIGURE_PX_HEIGHT,
      ]
    }

    // After a racket hit, pick a vy that guarantees the ball:
    //   1. lands within the headline's vertical band (so it actually
    //      bounces off the words, not over/under them)
    //   2. returns to the racket plane within the figure's reachable
    //      Y range (so the figure can track and hit it on the next cycle)
    // No air bounces — the ball must be captured by the words on the
    // way out and by the racket on the way back; otherwise it would
    // escape, which the user explicitly disallowed.
    function pickReturnVy(ballX: number, ballY: number): number {
      const dist = ballX - headlineLocal.right
      if (dist <= 8) return 0
      const t = dist / BALL_SPEED
      const buffer = 10
      // Outbound — must hit the headline rect on the way out.
      const minOut = (headlineLocal.top + buffer - ballY) / t
      const maxOut = (headlineLocal.bottom - buffer - ballY) / t
      // Inbound — after bouncing off the words, ball returns 2t later.
      // Must land within figure's reachable band centered on rest Y.
      const figRestY = containerRect.height / 2
      const figMinY = figRestY - FIGURE_Y_TRACK_AMPLITUDE_PX
      const figMaxY = figRestY + FIGURE_Y_TRACK_AMPLITUDE_PX
      const minIn = (figMinY - ballY) / (2 * t)
      const maxIn = (figMaxY - ballY) / (2 * t)
      const lo = Math.max(minOut, minIn)
      const hi = Math.min(maxOut, maxIn)
      if (lo > hi) return 0
      return lo + Math.random() * (hi - lo)
    }

    const svgEl = svg
    function setSvgSize() {
      svgEl.setAttribute('viewBox', `0 0 ${containerRect.width} ${containerRect.height}`)
      svgEl.setAttribute('width', String(containerRect.width))
      svgEl.setAttribute('height', String(containerRect.height))
    }

    setSvgSize()

    // Initial paint.
    const restRacket = racketHeadAtPhase(0)
    paintAtSwingPhase(0)
    const ballEl = refs.current.ball
    if (ballEl) {
      ballEl.setAttribute('cx', String(restRacket[0] - 60))
      ballEl.setAttribute('cy', String(restRacket[1]))
    }

    if (reducedMotion) return

    // Rally state.
    type State = 'preswing' | 'swinging'
    let state: State = 'preswing'
    let swingStart = 0
    let hitRegistered = false  // ensures we reverse the ball exactly once per swing
    let figureBaseYRest = containerRect.height / 2
    figureBaseY = figureBaseYRest
    // Ball — start near the racket, traveling toward the headline (left).
    // Initial vy chosen so the very first shot also lands on the words.
    const initialBallX = restRacket[0] - 80
    const initialBallY = restRacket[1]
    let ball = {
      x: initialBallX,
      y: initialBallY,
      vx: -BALL_SPEED,
      vy: 0,
    }
    ball.vy = pickReturnVy(initialBallX, initialBallY)
    let lastTime = performance.now()
    let rafId = 0
    let cancelled = false

    // Pre-derived constants.
    const racketYOffsetFromHip = (RACKET_CONTACT_NORM_Y - 0.5) * FIGURE_PX_HEIGHT

    function step(now: number) {
      if (cancelled) return
      const dt = Math.min(0.05, (now - lastTime) / 1000)
      lastTime = now

      // Determine swing phase.
      let swingPhase = 0
      if (state === 'swinging') {
        const u = Math.min(1, (now - swingStart) / SWING_MS)
        swingPhase = warpPhase(u)
        if (u >= 1) {
          state = 'preswing'
          hitRegistered = false
        }
      }

      // ─── Pre-swing logic: anticipate, track, fire ─────────────────────
      // The figure must START swinging BEFORE the ball arrives so the
      // racket sweeps through the strike zone exactly as the ball gets
      // there — that's the Pong-feel the user asked for. We compute
      // time-to-contact from the ball's current trajectory, then:
      //   (a) while we have time, soft-track figureBaseY toward the
      //       predicted ball Y (the figure leans to meet the shot).
      //   (b) when time-to-contact drops to SWING_TO_CONTACT_MS, snap
      //       figureBaseY to the exact target so the racket lands on
      //       the ball, and fire the swing.
      if (state === 'preswing' && ball.vx > 0) {
        // The racket's contact-frame X is across the body (left of
        // center) — that's where we want the ball at contact time.
        const racketContactX =
          figureBaseX + (RACKET_CONTACT_NORM_X - 0.5) * FIGURE_PX_WIDTH
        const tToContact = (racketContactX - ball.x) / ball.vx

        if (tToContact > 0) {
          const predBallY = ball.y + ball.vy * tToContact
          const targetY = Math.max(
            figureBaseYRest - FIGURE_Y_TRACK_AMPLITUDE_PX,
            Math.min(figureBaseYRest + FIGURE_Y_TRACK_AMPLITUDE_PX,
              predBallY - racketYOffsetFromHip),
          )

          if (tToContact * 1000 <= SWING_TO_CONTACT_MS) {
            // Time to fire. Snap so the contact frame lands on the ball.
            figureBaseY = targetY
            state = 'swinging'
            swingStart = now
            hitRegistered = false
          } else {
            // Soft track — figure leans toward where the ball will be.
            figureBaseY += (targetY - figureBaseY) * FIGURE_Y_TRACK_LERP
          }
        }
      }

      // Ball physics (kinematic — no gravity).
      ball.x += ball.vx * dt
      ball.y += ball.vy * dt

      // ─── Headline collision (only collision surface on the left) ─────
      if (
        ball.vx < 0 &&
        ball.y >= headlineLocal.top - BALL_RADIUS &&
        ball.y <= headlineLocal.bottom + BALL_RADIUS &&
        ball.x - BALL_RADIUS <= headlineLocal.right
      ) {
        ball.vx = -ball.vx
        ball.x = headlineLocal.right + BALL_RADIUS
      }

      // ─── Mid-swing contact: racket strikes the ball ───────────────────
      // The figure committed to a target Y when the swing fired, so the
      // racket head at SWING_CONTACT_PHASE should meet the ball within
      // a small radius. Reverse the ball at the moment the swing crosses
      // contact phase. The proximity guard catches edge cases where the
      // ball trajectory was perturbed (resize, etc.) — if it's still
      // close, we hit; if not, the ball flies past.
      if (state === 'swinging' && !hitRegistered && swingPhase >= SWING_CONTACT_PHASE) {
        const racketPx = racketHeadAtPhase(SWING_CONTACT_PHASE)
        const dx = ball.x - racketPx[0]
        const dy = ball.y - racketPx[1]
        if (dx * dx + dy * dy <= (RACKET_HIT_RADIUS * 1.5) ** 2) {
          ball.vx = -Math.abs(ball.vx)  // ensure leftward
          ball.vy = pickReturnVy(ball.x, ball.y)
          // Snap ball to racket head so the visual is "racket meets
          // ball" rather than "ball reverses near racket."
          ball.x = racketPx[0]
          ball.y = racketPx[1]
        }
        hitRegistered = true
      }

      // No top/bottom air walls. The ball can ONLY bounce off the words
      // (left) and the figure's racket (right). pickReturnVy keeps every
      // trajectory in-bounds; if anything escapes (resize edge case), the
      // ball flies past the figure rather than teleporting back into play.

      // Paint figure + ball.
      paintAtSwingPhase(swingPhase)
      if (refs.current.ball) {
        refs.current.ball.setAttribute('cx', String(ball.x))
        refs.current.ball.setAttribute('cy', String(ball.y))
      }

      rafId = requestAnimationFrame(step)
    }
    rafId = requestAnimationFrame((t) => {
      lastTime = t
      step(t)
    })

    // Resize handling.
    const ro = new ResizeObserver(() => {
      containerRect = container.getBoundingClientRect()
      headlineLocal = computeHeadlineLocal(headline, container)
      figureBaseX = containerRect.width - FIGURE_PX_WIDTH / 2 - FIGURE_RIGHT_PADDING
      figureBaseYRest = containerRect.height / 2
      // Don't snap figureBaseY — let the lerp pull it back toward the
      // new rest line over the next frames so resize doesn't jolt.
      setSvgSize()
    })
    ro.observe(container)
    ro.observe(headline)

    return () => {
      cancelled = true
      cancelAnimationFrame(rafId)
      ro.disconnect()
    }
  }, [headlineRef, reducedMotion])

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 pointer-events-none hidden lg:block"
      aria-hidden="true"
    >
      <svg
        ref={svgRef}
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Bones */}
        {BONES.map((_, idx) => (
          <line
            key={`bone-${idx}`}
            ref={(el) => { refs.current.bones[idx] = el }}
            stroke="var(--cream)"
            strokeOpacity="0.95"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
        ))}

        {/* Racket grip */}
        <line
          ref={(el) => { refs.current.racketGrip = el }}
          stroke="var(--cream)"
          strokeOpacity="0.95"
          strokeWidth="2.25"
          strokeLinecap="round"
        />

        {/* Racket head */}
        <circle
          ref={(el) => { refs.current.racketHead = el }}
          r="7"
          fill="var(--clay)"
        />

        {/* Joint dots */}
        {JOINT_KEYS.map((key, idx) => (
          <circle
            key={`joint-${idx}`}
            ref={(el) => { refs.current.joints[idx] = el }}
            r={key === 'nose' ? 8 : 5.5}
            fill="var(--cream)"
          />
        ))}

        {/* Angle pills — render last so they sit on top of bones/joints.
            Match the screenshot reference: dark rect with cream text,
            the angle dynamically computed each frame. */}
        {ANGLE_PILLS.map((_, idx) => (
          <g key={`pill-${idx}`}>
            <rect
              ref={(el) => {
                const slot = refs.current.pills[idx]
                if (slot) slot.rect = el
              }}
              width={PILL_W}
              height={PILL_H}
              fill="var(--ink)"
              fillOpacity="0.85"
              rx="3"
              ry="3"
            />
            <text
              ref={(el) => {
                const slot = refs.current.pills[idx]
                if (slot) slot.text = el
              }}
              fill="var(--cream)"
              fontFamily="var(--font-sans)"
              fontSize="11"
              fontWeight="600"
              textAnchor="middle"
              style={{ pointerEvents: 'none' }}
            >
              0°
            </text>
          </g>
        ))}

        {/* Tennis ball — clay with a faint cream seam. The seam is a
            thin arc, drawn here as a stroke-only path. */}
        <circle
          ref={(el) => { refs.current.ball = el }}
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

function computeHeadlineLocal(
  headline: HTMLElement,
  container: HTMLElement,
): { left: number; right: number; top: number; bottom: number } {
  const h = headline.getBoundingClientRect()
  const c = container.getBoundingClientRect()
  return {
    left: h.left - c.left,
    right: h.right - c.left,
    top: h.top - c.top,
    bottom: h.bottom - c.top,
  }
}
