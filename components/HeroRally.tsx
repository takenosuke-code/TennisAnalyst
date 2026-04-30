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
// anchored at FIGURE_RIGHT_PX from the right edge of the hero.
const FIGURE_PX_HEIGHT = 480
const FIGURE_PX_WIDTH = FIGURE_PX_HEIGHT * (9 / 16)
const FIGURE_RIGHT_PADDING = 32

// Ball
const BALL_RADIUS = 7
const BALL_SPEED = 540 // px/sec — feels like a rally tempo, not Pong

// Swing
const SWING_MS = 700
const RACKET_HIT_RADIUS = 36

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
      // Joints
      JOINT_KEYS.forEach((key, idx) => {
        const dot = refs.current.joints[idx]
        if (!dot) return
        const [x, y] = toPx(joints[key])
        dot.setAttribute('cx', String(x))
        dot.setAttribute('cy', String(y))
      })
      // Bones
      BONES.forEach(([from, to], idx) => {
        const line = refs.current.bones[idx]
        if (!line) return
        const [x1, y1] = toPx(joints[from])
        const [x2, y2] = toPx(joints[to])
        line.setAttribute('x1', String(x1))
        line.setAttribute('y1', String(y1))
        line.setAttribute('x2', String(x2))
        line.setAttribute('y2', String(y2))
      })
      // Racket
      const wristPx = toPx(joints.right_wrist)
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
    // Ball — start near the racket, traveling toward the headline (left).
    let ball = {
      x: restRacket[0] - 80,
      y: restRacket[1],
      vx: -BALL_SPEED,
      vy: 0,
    }
    let lastTime = performance.now()
    let rafId = 0
    let cancelled = false

    function step(now: number) {
      if (cancelled) return
      const dt = Math.min(0.05, (now - lastTime) / 1000)
      lastTime = now

      // Determine swing phase. preswing → phase 0 (rest pose).
      let swingPhase = 0
      if (state === 'swinging') {
        const u = Math.min(1, (now - swingStart) / SWING_MS)
        swingPhase = warpPhase(u)
        if (u >= 1) {
          state = 'preswing'
        }
      }

      // Ball physics (kinematic — no gravity, simple Euler).
      ball.x += ball.vx * dt
      ball.y += ball.vy * dt

      // Headline collision (single AABB for both lines).
      // The ball is traveling left (vx < 0); we bounce off the right
      // edge of the headline rect when the ball's leading edge crosses it.
      if (
        ball.vx < 0 &&
        ball.y >= headlineLocal.top - BALL_RADIUS &&
        ball.y <= headlineLocal.bottom + BALL_RADIUS &&
        ball.x - BALL_RADIUS <= headlineLocal.right &&
        ball.x - BALL_RADIUS >= headlineLocal.left - 12
      ) {
        ball.vx = -ball.vx
        ball.x = headlineLocal.right + BALL_RADIUS // unstick
      }

      // Racket collision (only when not already mid-swing — one swing per rally).
      if (state === 'preswing' && ball.vx > 0) {
        const racketPx = racketHeadAtPhase(0)
        const dx = ball.x - racketPx[0]
        const dy = ball.y - racketPx[1]
        if (dx * dx + dy * dy <= RACKET_HIT_RADIUS * RACKET_HIT_RADIUS) {
          state = 'swinging'
          swingStart = now
          ball.vx = -ball.vx
          // Add a slight vertical kick so the rally has some y-variation.
          ball.vy = (Math.random() - 0.5) * BALL_SPEED * 0.15
        }
      }

      // Vertical bounds — bounce off top/bottom of the hero section.
      if (ball.y - BALL_RADIUS < 0 && ball.vy < 0) {
        ball.vy = -ball.vy
        ball.y = BALL_RADIUS
      }
      if (ball.y + BALL_RADIUS > containerRect.height && ball.vy > 0) {
        ball.vy = -ball.vy
        ball.y = containerRect.height - BALL_RADIUS
      }

      // Safety — if the ball escapes left of the headline (e.g. resize
      // mid-flight), wrap it back into play.
      if (ball.x < headlineLocal.left - 100) {
        ball.x = headlineLocal.right + BALL_RADIUS
        ball.vx = Math.abs(ball.vx)
      }
      // Same on the right (shouldn't happen but defensive).
      if (ball.x > containerRect.width + 100) {
        ball.x = restRacket[0] - 80
        ball.vx = -BALL_SPEED
      }

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
      figureBaseY = containerRect.height / 2
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
