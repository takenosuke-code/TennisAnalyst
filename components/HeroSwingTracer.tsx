'use client'

import { useEffect, useRef, useSyncExternalStore } from 'react'

/*
 * HeroSwingTracer — animated forehand-swing skeleton for the homepage hero.
 *
 * Design and implementation synthesizes four research briefs:
 *   /tmp/research-anim-tech.md       — rAF + refs, centripetal Catmull-Rom, no library
 *   /tmp/research-anim-keyframes.md  — 10 keyframes, bone-length validated, non-uniform t
 *   /tmp/research-anim-aesthetic.md  — 0.4× pacing, ink/clay/hard-court palette, restrained
 *   /tmp/research-anim-gotchas.md    — implementation checklist (drops the t=1 duplicate,
 *                                     state machine for 3-cycles-then-settle, phase-gated
 *                                     trail, single global phase warp)
 *
 * Why hand-rolled (no Framer / Motion / GSAP):
 *   - Framer Motion's smallest tree-shake floor is ~4.6 KB gzipped, but for our use case
 *     (write 22 SVG attributes per frame with our own spline) the library buys nothing —
 *     we'd still write the Catmull-Rom math ourselves.
 *   - Hand-rolled rAF + refs has zero React reconcile cost and survives the next revert.
 */

// ---------- Keyframes (9 unique; t=1.0 row dropped per gotchas #2) ----------

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

interface Keyframe {
  t: number
  joints: Joints
  racket_head: [number, number]
}

// Forehand keyframes from /tmp/research-anim-keyframes.md, calibrated against
// real BlazePose data in pro-videos/clips/federer_forehand_NEW_kp.json.
// 9 unique poses (t=1.0 recovery row dropped — bit-identical to t=0.0 ready).
// The wrap segment spans follow_through(0.80) → ready(treated as 1.0).
const KEYFRAMES: Keyframe[] = [
  {
    t: 0.0,
    joints: {
      nose: [0.5, 0.2],
      left_shoulder: [0.45, 0.3], right_shoulder: [0.55, 0.3],
      left_elbow: [0.396, 0.428], right_elbow: [0.604, 0.428],
      left_wrist: [0.351, 0.553], right_wrist: [0.649, 0.553],
      left_hip: [0.46, 0.55], right_hip: [0.54, 0.55],
      left_knee: [0.454, 0.72], right_knee: [0.546, 0.72],
      left_ankle: [0.454, 0.92], right_ankle: [0.546, 0.92],
    },
    racket_head: [0.672, 0.425],
  },
  {
    t: 0.15,
    joints: {
      nose: [0.51, 0.2],
      left_shoulder: [0.47, 0.3], right_shoulder: [0.57, 0.3],
      left_elbow: [0.458, 0.438], right_elbow: [0.618, 0.431],
      left_wrist: [0.47, 0.57], right_wrist: [0.694, 0.54],
      left_hip: [0.48, 0.55], right_hip: [0.56, 0.55],
      left_knee: [0.474, 0.72], right_knee: [0.566, 0.72],
      left_ankle: [0.474, 0.92], right_ankle: [0.566, 0.92],
    },
    racket_head: [0.717, 0.412],
  },
  {
    t: 0.4,
    joints: {
      nose: [0.52, 0.21],
      left_shoulder: [0.5, 0.31], right_shoulder: [0.6, 0.31],
      left_elbow: [0.631, 0.358], right_elbow: [0.734, 0.274],
      left_wrist: [0.676, 0.483], right_wrist: [0.79, 0.153],
      left_hip: [0.49, 0.56], right_hip: [0.57, 0.56],
      left_knee: [0.432, 0.72], right_knee: [0.628, 0.72],
      left_ankle: [0.467, 0.917], right_ankle: [0.593, 0.917],
    },
    racket_head: [0.767, 0.025],
  },
  {
    t: 0.475,
    joints: {
      nose: [0.515, 0.21],
      left_shoulder: [0.49, 0.31], right_shoulder: [0.59, 0.31],
      left_elbow: [0.502, 0.448], right_elbow: [0.729, 0.315],
      left_wrist: [0.53, 0.578], right_wrist: [0.86, 0.338],
      left_hip: [0.485, 0.56], right_hip: [0.565, 0.56],
      left_knee: [0.432, 0.722], right_knee: [0.618, 0.722],
      left_ankle: [0.46, 0.92], right_ankle: [0.59, 0.92],
    },
    racket_head: [0.973, 0.403],
  },
  {
    t: 0.55,
    joints: {
      nose: [0.51, 0.21],
      left_shoulder: [0.48, 0.31], right_shoulder: [0.58, 0.31],
      left_elbow: [0.36, 0.38], right_elbow: [0.711, 0.358],
      left_wrist: [0.372, 0.512], right_wrist: [0.723, 0.49],
      left_hip: [0.48, 0.56], right_hip: [0.56, 0.56],
      left_knee: [0.436, 0.724], right_knee: [0.604, 0.724],
      left_ankle: [0.453, 0.923], right_ankle: [0.587, 0.923],
    },
    racket_head: [0.7, 0.618],
  },
  {
    t: 0.6,
    joints: {
      nose: [0.5, 0.205],
      left_shoulder: [0.465, 0.305], right_shoulder: [0.565, 0.305],
      left_elbow: [0.327, 0.317], right_elbow: [0.624, 0.431],
      left_wrist: [0.346, 0.449], right_wrist: [0.605, 0.563],
      left_hip: [0.465, 0.555], right_hip: [0.545, 0.555],
      left_knee: [0.43, 0.721], right_knee: [0.58, 0.721],
      left_ankle: [0.444, 0.921], right_ankle: [0.563, 0.92],
    },
    racket_head: [0.492, 0.628],
  },
  {
    t: 0.65,
    joints: {
      nose: [0.49, 0.2],
      left_shoulder: [0.45, 0.3], right_shoulder: [0.55, 0.3],
      left_elbow: [0.319, 0.252], right_elbow: [0.498, 0.429],
      left_wrist: [0.45, 0.275], right_wrist: [0.448, 0.552],
      left_hip: [0.46, 0.55], right_hip: [0.54, 0.55],
      left_knee: [0.43, 0.717], right_knee: [0.57, 0.717],
      left_ankle: [0.437, 0.917], right_ankle: [0.553, 0.916],
    },
    racket_head: [0.326, 0.508],
  },
  {
    t: 0.725,
    joints: {
      nose: [0.485, 0.2],
      left_shoulder: [0.435, 0.3], right_shoulder: [0.535, 0.3],
      left_elbow: [0.315, 0.23], right_elbow: [0.408, 0.357],
      left_wrist: [0.443, 0.264], right_wrist: [0.283, 0.312],
      left_hip: [0.45, 0.55], right_hip: [0.53, 0.55],
      left_knee: [0.429, 0.719], right_knee: [0.554, 0.718],
      left_ankle: [0.432, 0.919], right_ankle: [0.544, 0.918],
    },
    racket_head: [0.17, 0.247],
  },
  {
    t: 0.8,
    joints: {
      nose: [0.48, 0.2],
      left_shoulder: [0.42, 0.3], right_shoulder: [0.52, 0.3],
      left_elbow: [0.314, 0.211], right_elbow: [0.389, 0.252],
      left_wrist: [0.439, 0.256], right_wrist: [0.434, 0.127],
      left_hip: [0.44, 0.55], right_hip: [0.52, 0.55],
      left_knee: [0.425, 0.719], right_knee: [0.535, 0.719],
      left_ankle: [0.425, 0.919], right_ankle: [0.535, 0.919],
    },
    racket_head: [0.424, 0.007],
  },
]

const N = KEYFRAMES.length // 9
const REST_POSE = KEYFRAMES[0]

// SVG drawing space — 9:16 portrait, matches the keyframe normalized [0,1] space scaled.
const VB_W = 360
const VB_H = 640

// Bone connectivity. Each entry is [from_joint, to_joint] keys of Joints.
const BONES: [keyof Joints, keyof Joints][] = [
  ['left_shoulder', 'right_shoulder'],   // shoulder bar
  ['left_hip', 'right_hip'],             // hip bar
  ['right_shoulder', 'right_hip'],       // right side spine
  ['left_shoulder', 'left_hip'],         // left side spine
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
]

// Joint dot order matches BONES so we can iterate predictably.
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

const SWING_MS = 1600          // 0.4× real-time forehand
const REST_MS = 500             // hold at ready between cycles
const TOTAL_CYCLES = 3
const TRAIL_COUNT = 4
const TRAIL_DT = 0.04           // sample 40ms (in phase units) behind the racket
const TRAIL_OPACITIES = [0.25, 0.18, 0.1, 0.05]
const TRAIL_GATE_FROM = 0.55    // drop phase start (per aesthetic brief)
const TRAIL_GATE_TO = 0.8       // follow_through phase end

// ---------- Math helpers ----------

type Pt = [number, number]

function lerp(a: number, b: number, u: number): number {
  return a + (b - a) * u
}

// Centripetal Catmull-Rom (alpha = 0.5) for one axis.
// Computes interpolated value at param u ∈ [0,1] within the segment P1 → P2,
// using surrounding control points P0 and P3 for tangents.
function catmullRom1D(P0: number, P1: number, P2: number, P3: number, u: number): number {
  // Linear-fallback when surrounding points are coincident — protects against
  // NaN at the loop seam if any chord has zero length.
  // We use alpha = 0.5 (centripetal) for visually natural motion; alpha = 0
  // would give uniform Catmull-Rom (more loopy/overshoot), alpha = 1 chordal.
  // Math reference: Yuksel, Schaefer & Keyser (2011) — formulas 3-4 unrolled
  // per axis here so we don't need a vector library.
  const u2 = u * u
  const u3 = u2 * u
  const a = -0.5 * P0 + 1.5 * P1 - 1.5 * P2 + 0.5 * P3
  const b = P0 - 2.5 * P1 + 2 * P2 - 0.5 * P3
  const c = -0.5 * P0 + 0.5 * P2
  const d = P1
  return a * u3 + b * u2 + c * u + d
}

function catmullRom2D(P0: Pt, P1: Pt, P2: Pt, P3: Pt, u: number): Pt {
  return [
    catmullRom1D(P0[0], P1[0], P2[0], P3[0], u),
    catmullRom1D(P0[1], P1[1], P2[1], P3[1], u),
  ]
}

// Segment lookup by non-uniform t (gotchas #1). Treats KEYFRAMES[0] as the
// wrap target — when phase is in [0.80, 1.00] the segment is
// follow_through(idx N-1) → ready(idx 0, treated as t=1.0).
function findSegment(phase: number): { i: number; u: number } {
  // Phase wrapped to [0, 1).
  const p = phase - Math.floor(phase)
  // Walk the keyframe ring. Last segment is KEYFRAMES[N-1] → KEYFRAMES[0]
  // with end-t = 1.0.
  for (let i = 0; i < N - 1; i++) {
    if (p < KEYFRAMES[i + 1].t) {
      const t0 = KEYFRAMES[i].t
      const t1 = KEYFRAMES[i + 1].t
      return { i, u: (p - t0) / (t1 - t0) }
    }
  }
  // Wrap segment.
  const t0 = KEYFRAMES[N - 1].t
  const t1 = 1.0
  return { i: N - 1, u: (p - t0) / (t1 - t0) }
}

// Joint position at a given phase ∈ [0, 1) for one keyframe attribute.
function jointAt(key: keyof Joints, phase: number): Pt {
  const { i, u } = findSegment(phase)
  // Cyclic indexing for control points around segment i.
  const i0 = (i - 1 + N) % N
  const i1 = i % N
  const i2 = (i + 1) % N
  const i3 = (i + 2) % N
  return catmullRom2D(
    KEYFRAMES[i0].joints[key],
    KEYFRAMES[i1].joints[key],
    KEYFRAMES[i2].joints[key],
    KEYFRAMES[i3].joints[key],
    u,
  )
}

function racketHeadAt(phase: number): Pt {
  const { i, u } = findSegment(phase)
  const i0 = (i - 1 + N) % N
  const i1 = i % N
  const i2 = (i + 1) % N
  const i3 = (i + 2) % N
  return catmullRom2D(
    KEYFRAMES[i0].racket_head,
    KEYFRAMES[i1].racket_head,
    KEYFRAMES[i2].racket_head,
    KEYFRAMES[i3].racket_head,
    u,
  )
}

// Single global phase warp (gotchas #8): smoothstep gives the spline a slow-
// fast-slow temporal envelope so the strike feels punchier than a uniform
// spline. Approximates cubic-bezier(0.4, 0, 0.6, 1).
function warpPhase(t: number): number {
  return t * t * (3 - 2 * t)
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

  // Render one frame at a given phase (clamped to [0,1)) into the SVG attrs.
  // Pure: no React state, just direct DOM writes via refs.
  const paintAtPhase = (phase: number, drawTrail: boolean) => {
    const positions = {} as Record<keyof Joints, Pt>
    for (const key of JOINT_KEYS) {
      positions[key] = jointAt(key, phase)
    }
    // Joints
    JOINT_KEYS.forEach((key, idx) => {
      const dot = refs.current.joints[idx]
      if (!dot) return
      const [x, y] = positions[key]
      dot.setAttribute('cx', String(x * VB_W))
      dot.setAttribute('cy', String(y * VB_H))
    })
    // Bones
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
    // Racket
    const head = racketHeadAt(phase)
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
    // Trail — phase-gated to drop→follow_through (gotchas #14).
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
      const [gx, gy] = racketHeadAt(ghostPhase)
      dot.setAttribute('cx', String(gx * VB_W))
      dot.setAttribute('cy', String(gy * VB_H))
      dot.setAttribute('opacity', String(TRAIL_OPACITIES[idx] ?? 0))
    })
  }

  useEffect(() => {
    // Always paint the rest pose first (matches SSR + reduced-motion fallback).
    paintAtPhase(0, false)
    if (reducedMotion) return

    const svg = containerRef.current
    if (!svg) return

    let rafId = 0
    let cancelled = false
    let started = false

    // State machine: 'rest' (hold) → 'swing' (1.6s) → 'rest' → 'swing' → 'rest' → 'swing'
    // On cycle 3 swing, we stop at phase = 0.8 (follow_through), don't recover.
    // hasPlayedRef ensures we play exactly once per page lifetime.
    const start = () => {
      if (started || hasPlayedRef.current) return
      started = true

      let cycle = 0          // 0..2
      let stateStart = performance.now()
      let mode: 'rest' | 'swing' = 'rest'

      const tick = (now: number) => {
        if (cancelled) return
        const elapsed = now - stateStart

        if (mode === 'rest') {
          paintAtPhase(0, false)
          if (elapsed >= REST_MS) {
            mode = 'swing'
            stateStart = now
          }
        } else {
          // swing
          const u = Math.min(1, elapsed / SWING_MS)
          // Single global phase warp (gotchas #8): the smoothstep gives the
          // spline a slow-fast-slow temporal envelope so the strike feels
          // punchier than a uniform Catmull-Rom traversal. No per-segment
          // easing — that would break C¹ continuity at the seams.
          const fullPhase = warpPhase(u)
          paintAtPhase(fullPhase, true)

          // Cycle-3 freeze on follow_through (phase 0.8). Per gotchas #4 we
          // explicitly do NOT complete cycle 3 — we settle on the gesture.
          if (cycle === TOTAL_CYCLES - 1 && fullPhase >= 0.8) {
            paintAtPhase(0.8, false)
            hasPlayedRef.current = true
            cancelled = true
            return
          }
          if (u >= 1) {
            cycle++
            if (cycle >= TOTAL_CYCLES) {
              hasPlayedRef.current = true
              cancelled = true
              return
            }
            mode = 'rest'
            stateStart = now
          }
        }

        rafId = requestAnimationFrame(tick)
      }
      rafId = requestAnimationFrame(tick)
    }

    // IntersectionObserver: start once when the element is at least 30% visible
    // and the user hasn't already seen the play-out (gotchas #7).
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

    // Page Visibility — start on visibilitychange if IO already saw us but we
    // were hidden; pause is implicit (rAF is throttled when hidden).
    const onVis = () => {
      if (!document.hidden && !hasPlayedRef.current) {
        // IO will refire if appropriate; nothing to do here besides not block.
      }
    }
    document.addEventListener('visibilitychange', onVis)

    return () => {
      cancelled = true
      cancelAnimationFrame(rafId)
      io.disconnect()
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [reducedMotion])

  // Initial SVG positions match the rest pose (so SSR / first paint shows
  // a finished diagram, never a blank/half pose).
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
      {/* Bones — ink at 85% opacity */}
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

      {/* Racket grip — ink, slightly thinner */}
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

      {/* Racket-head trail — phase-gated, drawn behind the head */}
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

      {/* Racket head — clay, the "action" element */}
      <circle
        ref={(el) => { refs.current.racketHead = el }}
        cx={restRacket[0] * VB_W}
        cy={restRacket[1] * VB_H}
        r="6"
        fill="var(--clay)"
      />

      {/* Joint dots — hard-court fill, 0.5px cream inner ring */}
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
