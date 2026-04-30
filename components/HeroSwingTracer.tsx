'use client'

import { useEffect, useRef, useSyncExternalStore } from 'react'

/*
 * HeroSwingTracer — animated forehand-swing skeleton for the homepage hero.
 *
 * v2: angle-driven forward-kinematics rebuild. The v1 used Catmull-Rom on
 * joint POSITIONS, which collapsed bone lengths up to 66% mid-segment on
 * the high-rotation transitions (the "rubber-banding" the user flagged
 * as not-tennis-like). v2 interpolates BONE ANGLES instead and rebuilds
 * the skeleton each frame as `child = parent + L · unitVec(angle)` — bone
 * lengths are exact at every interpolated frame by construction.
 *
 * Synthesizes four research briefs in /tmp/research-anim-*.md:
 *   - keyframes.md       — 9 hand-validated poses (1.5%-collapse-free at the keyframes)
 *   - tennis-truth.md    — biomechanical tells (X-factor, racket lag, kinetic chain)
 *   - ik-impl.md         — the angle table + buildSkeleton math used here
 *   - polish.md          — breathing micro-motion in the settled pose (alive, not frozen)
 *
 * No animation library — Framer Motion / Motion floor at ~5KB gzipped for our case
 * and we'd still be writing the FK math ourselves. rAF + refs has zero React
 * reconcile cost.
 */

// ---------- Joints + Keyframes ----------

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

// SSR / reduced-motion rest-pose. Computed by feeding KEYFRAME_ANGLES[0] through
// buildSkeleton — duplicated here as literal numbers so SSR doesn't need to run
// any math to render the static fallback.
const REST_POSE: { joints: Joints; racket_head: Pt } = {
  joints: {
    nose: [0.5, 0.2],
    left_shoulder: [0.45, 0.3],
    right_shoulder: [0.55, 0.3],
    left_elbow: [0.396, 0.428],
    right_elbow: [0.604, 0.428],
    left_wrist: [0.351, 0.553],
    right_wrist: [0.649, 0.553],
    left_hip: [0.46, 0.55],
    right_hip: [0.54, 0.55],
    left_knee: [0.454, 0.72],
    right_knee: [0.546, 0.72],
    left_ankle: [0.454, 0.92],
    right_ankle: [0.546, 0.92],
  },
  racket_head: [0.672, 0.425],
}

// Bone lengths in normalized frame-height units. Sourced from the keyframes
// brief's "Locked bone-length invariants" table; verified within 0.7% across
// every keyframe. Single source of truth — no magic numbers in buildSkeleton.
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

// Global-frame bone angles in degrees, signed in (-180, 180]. SVG axes:
// 0=right, 90=down, 180=left, -90=up.
// Computed from the position keyframes via atan2(child - parent), 1-decimal
// precision, max round-off ≤0.001 normalized. See /tmp/research-anim-ik-impl.md
// for the derivation script.
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

const KEYFRAME_ANGLES: KeyframeAngles[] = [
  // ready
  { t: 0.0, hipCenter: [0.5, 0.55], trunk: -90.0, neck: -90.0,
    uArmL: 112.9, fArmL: 109.8, uArmR: 67.1, fArmR: 70.2,
    thighL: 92.0, shinL: 90.0, thighR: 88.0, shinR: 90.0,
    racket: -79.8, racketLen: 0.13 },
  // unit_turn
  { t: 0.15, hipCenter: [0.52, 0.55], trunk: -90.0, neck: -95.7,
    uArmL: 95.0, fArmL: 84.8, uArmR: 69.9, fArmR: 55.1,
    thighL: 92.0, shinL: 90.0, thighR: 88.0, shinR: 90.0,
    racket: -79.8, racketLen: 0.13 },
  // load
  { t: 0.4, hipCenter: [0.53, 0.56], trunk: -85.4, neck: -106.7,
    uArmL: 20.1, fArmL: 70.2, uArmR: -15.0, fArmR: -65.2,
    thighL: 109.9, shinL: 79.9, thighR: 70.1, shinR: 100.1,
    racket: -100.2, racketLen: 0.13 },
  // coil (bridge)
  { t: 0.475, hipCenter: [0.525, 0.56], trunk: -86.6, neck: -104.0,
    uArmL: 85.0, fArmL: 77.8, uArmR: 2.1, fArmR: 10.0,
    thighL: 108.1, shinL: 82.0, thighR: 71.9, shinR: 98.0,
    racket: 29.9, racketLen: 0.13 },
  // drop
  { t: 0.55, hipCenter: [0.52, 0.56], trunk: -87.7, neck: -101.3,
    uArmL: 149.7, fArmL: 84.8, uArmR: 20.1, fArmR: 84.8,
    thighL: 105.0, shinL: 85.1, thighR: 75.0, shinR: 94.9,
    racket: 100.2, racketLen: 0.13 },
  // drive (bridge)
  { t: 0.6, hipCenter: [0.505, 0.555], trunk: -87.7, neck: -98.5,
    uArmL: 175.0, fArmL: 81.8, uArmR: 64.9, fArmR: 98.2,
    thighL: 101.9, shinL: 86.0, thighR: 78.1, shinR: 94.9,
    racket: 150.1, racketLen: 0.13 },
  // contact
  { t: 0.65, hipCenter: [0.5, 0.55], trunk: -90.0, neck: -95.7,
    uArmL: -159.9, fArmL: 10.0, uArmR: 112.0, fArmR: 112.1,
    thighL: 100.2, shinL: 88.0, thighR: 79.8, shinR: 94.9,
    racket: -160.2, racketLen: 0.13 },
  // wrap (bridge)
  { t: 0.725, hipCenter: [0.49, 0.55], trunk: -91.1, neck: -90.0,
    uArmL: -149.7, fArmL: 14.9, uArmR: 155.8, fArmR: -160.2,
    thighL: 97.1, shinL: 89.1, thighR: 81.9, shinR: 92.9,
    racket: -150.1, racketLen: 0.13 },
  // follow_through
  { t: 0.8, hipCenter: [0.48, 0.55], trunk: -92.3, neck: -84.3,
    uArmL: -140.0, fArmL: 19.8, uArmR: -159.9, fArmR: -70.2,
    thighL: 95.1, shinL: 90.0, thighR: 84.9, shinR: 90.0,
    racket: -94.8, racketLen: 0.12 },
]

const N = KEYFRAME_ANGLES.length // 9

const VB_W = 360
const VB_H = 640

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

const SWING_MS = 1600
const REST_MS = 500
const TOTAL_CYCLES = 3
const TRAIL_COUNT = 4
const TRAIL_DT = 0.04
const TRAIL_OPACITIES = [0.25, 0.18, 0.1, 0.05]
const TRAIL_GATE_FROM = 0.55
const TRAIL_GATE_TO = 0.8

// Breathing oscillation in the settled pose (polish brief). 14 bpm = 0.233 Hz.
// Amplitude in normalized [0,1] units — 0.8px on a 640px viewBox = 0.00125.
const BREATH_HZ = 0.233
const BREATH_AMPLITUDE_NORM = 0.8 / VB_H

// ---------- Math helpers ----------

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

// Segment lookup by non-uniform t. Wrap segment is KEYFRAME_ANGLES[N-1] →
// KEYFRAME_ANGLES[0] with end-t treated as 1.0.
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

// Single global phase warp. Smoothstep gives a slow-fast-slow envelope so
// the strike feels punchier than uniform Catmull-Rom traversal would.
function warpPhase(t: number): number {
  return t * t * (3 - 2 * t)
}

// Build the whole skeleton from interpolated angles at a given phase.
// Bone lengths are exact by construction (parent + L · unit_vec(angle)).
// breathOffset applies a tiny y-shift to hipCenter for the breathing
// micro-motion in the settled pose; pass 0 during active swinging.
function buildSkeleton(
  phase: number,
  breathOffset: number = 0,
): { joints: Joints; racketHead: Pt } {
  const { i, u } = findSegment(phase)
  const A = KEYFRAME_ANGLES[i]
  const B = KEYFRAME_ANGLES[(i + 1) % N]

  const hipCx = lerp(A.hipCenter[0], B.hipCenter[0], u)
  const hipCy = lerp(A.hipCenter[1], B.hipCenter[1], u) + breathOffset

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

  // Hips: horizontal bar around hipCenter (NOT rotated by trunk — the
  // keyframes lock hip y globally even when trunk dips; rotating the bar
  // would break that invariant).
  const left_hip: Pt = [hipCx - BONE_LENGTHS.hipHalfWidth, hipCy]
  const right_hip: Pt = [hipCx + BONE_LENGTHS.hipHalfWidth, hipCy]

  // midShoulder = hipCenter + trunk vector
  const [tx, ty] = polar(BONE_LENGTHS.trunk, trunk)
  const midShx = hipCx + tx
  const midShy = hipCy + ty

  // Shoulders: horizontal bar around midShoulder
  const left_shoulder: Pt = [midShx - BONE_LENGTHS.shoulderHalfWidth, midShy]
  const right_shoulder: Pt = [midShx + BONE_LENGTHS.shoulderHalfWidth, midShy]

  // Head
  const [nx, ny] = polar(BONE_LENGTHS.neck, neck)
  const nose: Pt = [midShx + nx, midShy + ny]

  // Arms
  const [ueLx, ueLy] = polar(BONE_LENGTHS.upperArm, uArmL)
  const left_elbow: Pt = [left_shoulder[0] + ueLx, left_shoulder[1] + ueLy]
  const [ueRx, ueRy] = polar(BONE_LENGTHS.upperArm, uArmR)
  const right_elbow: Pt = [right_shoulder[0] + ueRx, right_shoulder[1] + ueRy]
  const [feLx, feLy] = polar(BONE_LENGTHS.forearm, fArmL)
  const left_wrist: Pt = [left_elbow[0] + feLx, left_elbow[1] + feLy]
  const [feRx, feRy] = polar(BONE_LENGTHS.forearm, fArmR)
  const right_wrist: Pt = [right_elbow[0] + feRx, right_elbow[1] + feRy]

  // Legs
  const [tLx, tLy] = polar(BONE_LENGTHS.thigh, thighL)
  const left_knee: Pt = [left_hip[0] + tLx, left_hip[1] + tLy]
  const [tRx, tRy] = polar(BONE_LENGTHS.thigh, thighR)
  const right_knee: Pt = [right_hip[0] + tRx, right_hip[1] + tRy]
  const [sLx, sLy] = polar(BONE_LENGTHS.shin, shinL)
  const left_ankle: Pt = [left_knee[0] + sLx, left_knee[1] + sLy]
  const [sRx, sRy] = polar(BONE_LENGTHS.shin, shinR)
  const right_ankle: Pt = [right_knee[0] + sRx, right_knee[1] + sRy]

  // Racket
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

interface SkeletonRefs {
  bones: (SVGLineElement | null)[]
  joints: (SVGCircleElement | null)[]
  racketHead: SVGCircleElement | null
  racketGrip: SVGLineElement | null
  trail: (SVGCircleElement | null)[]
}

export default function HeroSwingTracer() {
  const reducedMotion = useSyncExternalStore(
    subscribeReducedMotion,
    getReducedMotion,
    getServerReducedMotion,
  )
  const containerRef = useRef<SVGSVGElement | null>(null)
  const refs = useRef<SkeletonRefs>({
    bones: [],
    joints: [],
    racketHead: null,
    racketGrip: null,
    trail: [],
  })
  const hasPlayedRef = useRef(false)

  const paintAtPhase = (phase: number, drawTrail: boolean, breathOffset = 0) => {
    const { joints: positions, racketHead: head } = buildSkeleton(phase, breathOffset)

    JOINT_KEYS.forEach((key, idx) => {
      const dot = refs.current.joints[idx]
      if (!dot) return
      const [x, y] = positions[key]
      dot.setAttribute('cx', String(x * VB_W))
      dot.setAttribute('cy', String(y * VB_H))
    })

    BONES.forEach(([from, to], idx) => {
      const line = refs.current.bones[idx]
      if (!line) return
      const [x1, y1] = positions[from]
      const [x2, y2] = positions[to]
      line.setAttribute('x1', String(x1 * VB_W))
      line.setAttribute('y1', String(y1 * VB_H))
      line.setAttribute('x2', String(x2 * VB_W))
      line.setAttribute('y2', String(y2 * VB_H))
    })

    const wrist = positions.right_wrist
    const headEl = refs.current.racketHead
    if (headEl) {
      headEl.setAttribute('cx', String(head[0] * VB_W))
      headEl.setAttribute('cy', String(head[1] * VB_H))
    }
    const grip = refs.current.racketGrip
    if (grip) {
      grip.setAttribute('x1', String(wrist[0] * VB_W))
      grip.setAttribute('y1', String(wrist[1] * VB_H))
      grip.setAttribute('x2', String(head[0] * VB_W))
      grip.setAttribute('y2', String(head[1] * VB_H))
    }

    const trailVisible = drawTrail && phase >= TRAIL_GATE_FROM && phase <= TRAIL_GATE_TO
    refs.current.trail.forEach((dot, idx) => {
      if (!dot) return
      if (!trailVisible) {
        dot.setAttribute('opacity', '0')
        return
      }
      const offset = (idx + 1) * TRAIL_DT
      const ghostPhase = phase - offset
      if (ghostPhase < TRAIL_GATE_FROM) {
        dot.setAttribute('opacity', '0')
        return
      }
      // Re-build skeleton at ghostPhase to get racket head only — 4 extra
      // builds × ~5µs = ~20µs/frame, well under budget.
      const ghost = buildSkeleton(ghostPhase, 0)
      dot.setAttribute('cx', String(ghost.racketHead[0] * VB_W))
      dot.setAttribute('cy', String(ghost.racketHead[1] * VB_H))
      dot.setAttribute('opacity', String(TRAIL_OPACITIES[idx] ?? 0))
    })
  }

  useEffect(() => {
    paintAtPhase(0, false)
    if (reducedMotion) return

    const svg = containerRef.current
    if (!svg) return

    let rafId = 0
    let cancelled = false
    let started = false
    // settledStart tracks when the swing fully completes; afterwards rAF
    // keeps running purely to drive the breathing oscillation.
    let settledStart = 0

    const start = () => {
      if (started || hasPlayedRef.current) return
      started = true

      let cycle = 0
      let stateStart = performance.now()
      let mode: 'rest' | 'swing' | 'settled' = 'rest'

      const tick = (now: number) => {
        if (cancelled) return
        const elapsed = now - stateStart

        if (mode === 'rest') {
          paintAtPhase(0, false)
          if (elapsed >= REST_MS) {
            mode = 'swing'
            stateStart = now
          }
        } else if (mode === 'swing') {
          const u = Math.min(1, elapsed / SWING_MS)
          const fullPhase = warpPhase(u)
          paintAtPhase(fullPhase, true)

          if (cycle === TOTAL_CYCLES - 1 && fullPhase >= 0.8) {
            mode = 'settled'
            settledStart = now
            hasPlayedRef.current = true
          } else if (u >= 1) {
            cycle++
            if (cycle >= TOTAL_CYCLES) {
              mode = 'settled'
              settledStart = now
              hasPlayedRef.current = true
            } else {
              mode = 'rest'
              stateStart = now
            }
          }
        } else {
          // settled — paint follow_through pose with breath oscillation.
          // The body looks alive (paused person) instead of frozen (paused diagram).
          const sinceSettle = (now - settledStart) / 1000
          const breath = Math.sin(2 * Math.PI * BREATH_HZ * sinceSettle) * BREATH_AMPLITUDE_NORM
          paintAtPhase(0.8, false, breath)
        }

        rafId = requestAnimationFrame(tick)
      }
      rafId = requestAnimationFrame(tick)
    }

    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && !hasPlayedRef.current && !document.hidden) {
            start()
          }
        }
      },
      { threshold: 0.3 },
    )
    io.observe(svg)

    const onVis = () => {
      // No-op — IO will refire if the element becomes visible. rAF is
      // already paused by the browser when the tab is hidden.
    }
    document.addEventListener('visibilitychange', onVis)

    return () => {
      cancelled = true
      cancelAnimationFrame(rafId)
      io.disconnect()
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [reducedMotion])

  const restJoints = REST_POSE.joints
  const restRacket = REST_POSE.racket_head

  return (
    <svg
      ref={containerRef}
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      preserveAspectRatio="xMidYMid meet"
      className="w-full h-full"
      aria-label="Animated forehand-swing skeleton"
      role="img"
    >
      {BONES.map(([from, to], idx) => {
        const [x1, y1] = restJoints[from]
        const [x2, y2] = restJoints[to]
        return (
          <line
            key={idx}
            ref={(el) => { refs.current.bones[idx] = el }}
            x1={x1 * VB_W} y1={y1 * VB_H}
            x2={x2 * VB_W} y2={y2 * VB_H}
            stroke="var(--ink)"
            strokeOpacity="0.85"
            strokeWidth="2.25"
            strokeLinecap="round"
          />
        )
      })}

      <line
        ref={(el) => { refs.current.racketGrip = el }}
        x1={restJoints.right_wrist[0] * VB_W}
        y1={restJoints.right_wrist[1] * VB_H}
        x2={restRacket[0] * VB_W}
        y2={restRacket[1] * VB_H}
        stroke="var(--ink)"
        strokeOpacity="0.85"
        strokeWidth="2"
        strokeLinecap="round"
      />

      {Array.from({ length: TRAIL_COUNT }).map((_, idx) => (
        <circle
          key={`trail-${idx}`}
          ref={(el) => { refs.current.trail[idx] = el }}
          cx={restRacket[0] * VB_W}
          cy={restRacket[1] * VB_H}
          r="3.5"
          fill="var(--clay)"
          opacity="0"
        />
      ))}

      <circle
        ref={(el) => { refs.current.racketHead = el }}
        cx={restRacket[0] * VB_W}
        cy={restRacket[1] * VB_H}
        r="6"
        fill="var(--clay)"
      />

      {JOINT_KEYS.map((key, idx) => {
        const [x, y] = restJoints[key]
        const r = key === 'nose' ? 7 : 5
        return (
          <circle
            key={`joint-${idx}`}
            ref={(el) => { refs.current.joints[idx] = el }}
            cx={x * VB_W}
            cy={y * VB_H}
            r={r}
            fill="var(--hard-court)"
            stroke="var(--cream)"
            strokeWidth="1"
          />
        )
      })}
    </svg>
  )
}
