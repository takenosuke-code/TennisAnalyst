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

// Keyframes are RTMPose-extracted from a Carlos Alcaraz forehand
// (railway-service/extract_clip_keypoints.py). 40 frames at 30fps
// span prep through mid follow-through, mirrored around the y-axis
// so the right rally figure swings toward its opponent on the left.
// See the leading comment inside KEYFRAME_ANGLES for full pipeline.
const KEYFRAME_ANGLES: KeyframeAngles[] = [
  // 40 keyframes derived from RTMPose extraction of a Carlos Alcaraz
  // forehand (side-on broadcast clip). Pipeline:
  //   1. railway-service/extract_clip_keypoints.py → 82-frame pose JSON
  //   2. Per-frame polar angles computed in HeroRally's convention
  //      (0°=+x, 90°=+y/down, -90°=-y/up).
  //   3. Window: 1.1s pre-contact (full prep) + 0.2s post (mid
  //      follow-through; cropped before high-wrap to keep the loop seam
  //      sane). Contact aligned to t=0.65 (matches SWING_CONTACT_PHASE).
  //   4. Mirrored θ → 180°-θ around the y-axis: Alcaraz's swing in the
  //      clip goes left→right; the right rally figure hits toward its
  //      opponent on the left, so the swing direction needs flipping.
  //   5. hipCenter scaled by 0.4 of the player's actual hip motion so
  //      the figure doesn't walk across the screen.
  //   6. Racket angle = forearm extension (RTMPose doesn't track racket;
  //      racketLen from the wrist→racket-head distance, clamped 0.08-0.16).
  { t: 0.0, hipCenter: [0.5332, 0.5613], trunk: -93.5, neck: -110.2,
    uArmL: 90.0, fArmL: 121.0, uArmR: 81.6, fArmR: 116.7,
    thighL: 101.0, shinL: 76.5, thighR: 95.0, shinR: 79.6,
    racket: -156.0, racketLen: 0.086 },
  { t: 0.0197, hipCenter: [0.5327, 0.5638], trunk: -93.8, neck: -109.1,
    uArmL: 89.4, fArmL: 120.1, uArmR: 82.0, fArmR: 116.0,
    thighL: 100.1, shinL: 78.5, thighR: 95.8, shinR: 78.2,
    racket: -156.9, racketLen: 0.089 },
  { t: 0.0394, hipCenter: [0.5325, 0.5659], trunk: -94.8, neck: -107.1,
    uArmL: 89.4, fArmL: 120.2, uArmR: 81.4, fArmR: 112.5,
    thighL: 100.9, shinL: 78.5, thighR: 97.1, shinR: 77.2,
    racket: -154.0, racketLen: 0.099 },
  { t: 0.0591, hipCenter: [0.5317, 0.5691], trunk: -94.9, neck: -107.0,
    uArmL: 89.4, fArmL: 117.7, uArmR: 80.6, fArmR: 112.5,
    thighL: 102.0, shinL: 75.7, thighR: 97.5, shinR: 75.9,
    racket: -150.9, racketLen: 0.104 },
  { t: 0.0788, hipCenter: [0.531, 0.5705], trunk: -95.7, neck: -104.4,
    uArmL: 89.4, fArmL: 111.3, uArmR: 79.5, fArmR: 113.7,
    thighL: 103.2, shinL: 72.9, thighR: 97.8, shinR: 75.2,
    racket: -146.8, racketLen: 0.101 },
  { t: 0.0985, hipCenter: [0.5298, 0.5699], trunk: -95.9, neck: -108.5,
    uArmL: 90.7, fArmL: 105.7, uArmR: 78.3, fArmR: 115.6,
    thighL: 102.5, shinL: 71.6, thighR: 96.8, shinR: 75.9,
    racket: -143.8, racketLen: 0.107 },
  { t: 0.1182, hipCenter: [0.5277, 0.5687], trunk: -96.1, neck: -108.7,
    uArmL: 91.3, fArmL: 104.3, uArmR: 76.2, fArmR: 115.4,
    thighL: 102.2, shinL: 69.0, thighR: 94.0, shinR: 77.5,
    racket: -146.3, racketLen: 0.102 },
  { t: 0.1379, hipCenter: [0.5261, 0.5653], trunk: -95.4, neck: -110.1,
    uArmL: 92.0, fArmL: 103.8, uArmR: 76.7, fArmR: 119.4,
    thighL: 99.8, shinL: 69.1, thighR: 92.4, shinR: 78.6,
    racket: -142.9, racketLen: 0.101 },
  { t: 0.1576, hipCenter: [0.523, 0.5623], trunk: -93.8, neck: -107.4,
    uArmL: 92.6, fArmL: 104.6, uArmR: 75.9, fArmR: 123.0,
    thighL: 97.6, shinL: 68.1, thighR: 89.6, shinR: 79.9,
    racket: -137.2, racketLen: 0.118 },
  { t: 0.1773, hipCenter: [0.5207, 0.5582], trunk: -93.1, neck: -108.1,
    uArmL: 94.7, fArmL: 110.4, uArmR: 75.2, fArmR: 118.6,
    thighL: 93.9, shinL: 70.2, thighR: 87.5, shinR: 83.3,
    racket: -134.7, racketLen: 0.113 },
  { t: 0.197, hipCenter: [0.5189, 0.555], trunk: -92.5, neck: -108.4,
    uArmL: 94.3, fArmL: 83.1, uArmR: 73.7, fArmR: 120.3,
    thighL: 91.3, shinL: 70.9, thighR: 87.6, shinR: 85.4,
    racket: -128.7, racketLen: 0.11 },
  { t: 0.2167, hipCenter: [0.5167, 0.5546], trunk: -91.8, neck: -107.4,
    uArmL: 94.7, fArmL: -6.5, uArmR: 71.8, fArmR: 119.4,
    thighL: 89.6, shinL: 70.9, thighR: 87.9, shinR: 86.6,
    racket: -124.3, racketLen: 0.136 },
  { t: 0.2364, hipCenter: [0.5149, 0.5507], trunk: -92.2, neck: -107.7,
    uArmL: 90.0, fArmL: -23.3, uArmR: 66.1, fArmR: 112.8,
    thighL: 88.8, shinL: 66.7, thighR: 89.6, shinR: 88.5,
    racket: 112.8, racketLen: 0.1 },
  { t: 0.2561, hipCenter: [0.5123, 0.5479], trunk: -91.7, neck: -106.5,
    uArmL: 82.0, fArmL: -35.0, uArmR: 65.9, fArmR: 119.2,
    thighL: 88.3, shinL: 70.1, thighR: 89.2, shinR: 90.0,
    racket: 119.2, racketLen: 0.1 },
  { t: 0.2758, hipCenter: [0.5101, 0.5466], trunk: -90.5, neck: -108.3,
    uArmL: 77.8, fArmL: -41.6, uArmR: 62.8, fArmR: 119.4,
    thighL: 88.8, shinL: 72.0, thighR: 90.8, shinR: 87.4,
    racket: 119.4, racketLen: 0.1 },
  { t: 0.2955, hipCenter: [0.5085, 0.5461], trunk: -90.6, neck: -109.3,
    uArmL: 76.0, fArmL: -39.8, uArmR: 55.7, fArmR: 60.5,
    thighL: 91.2, shinL: 68.0, thighR: 92.0, shinR: 86.9,
    racket: -115.0, racketLen: 0.088 },
  { t: 0.3152, hipCenter: [0.5065, 0.5485], trunk: -89.5, neck: -86.1,
    uArmL: 71.7, fArmL: -44.8, uArmR: 52.4, fArmR: -117.0,
    thighL: 90.4, shinL: 70.2, thighR: 92.4, shinR: 90.4,
    racket: -105.5, racketLen: 0.081 },
  { t: 0.3348, hipCenter: [0.5045, 0.5551], trunk: -88.7, neck: -81.0,
    uArmL: 70.1, fArmL: -50.5, uArmR: 47.9, fArmR: -109.7,
    thighL: 92.9, shinL: 69.7, thighR: 92.5, shinR: 89.6,
    racket: -99.4, racketLen: 0.093 },
  { t: 0.3545, hipCenter: [0.5021, 0.5596], trunk: -89.1, neck: -103.4,
    uArmL: 56.1, fArmL: -42.3, uArmR: 28.4, fArmR: -105.9,
    thighL: 94.0, shinL: 69.4, thighR: 92.9, shinR: 86.6,
    racket: -115.0, racketLen: 0.099 },
  { t: 0.3742, hipCenter: [0.5, 0.5606], trunk: -89.9, neck: -114.0,
    uArmL: 41.6, fArmL: -35.4, uArmR: 32.7, fArmR: -51.3,
    thighL: 96.5, shinL: 79.8, thighR: 93.0, shinR: 84.3,
    racket: -119.6, racketLen: 0.1 },
  { t: 0.3939, hipCenter: [0.498, 0.5593], trunk: -90.1, neck: -107.7,
    uArmL: 35.1, fArmL: -29.1, uArmR: 35.5, fArmR: -34.9,
    thighL: 98.7, shinL: 86.5, thighR: 92.6, shinR: 82.9,
    racket: -118.0, racketLen: 0.104 },
  { t: 0.4136, hipCenter: [0.4956, 0.558], trunk: -90.1, neck: -100.3,
    uArmL: 40.2, fArmL: -12.6, uArmR: 26.2, fArmR: 2.9,
    thighL: 99.6, shinL: 89.3, thighR: 91.8, shinR: 82.4,
    racket: 2.9, racketLen: 0.1 },
  { t: 0.4333, hipCenter: [0.493, 0.5535], trunk: -89.2, neck: -110.9,
    uArmL: 48.5, fArmL: 0.0, uArmR: 35.8, fArmR: 12.5,
    thighL: 101.3, shinL: 89.0, thighR: 90.5, shinR: 81.2,
    racket: 12.5, racketLen: 0.1 },
  { t: 0.453, hipCenter: [0.4895, 0.5529], trunk: -88.6, neck: -101.0,
    uArmL: 58.9, fArmL: -2.1, uArmR: 40.1, fArmR: 20.0,
    thighL: 101.0, shinL: 86.4, thighR: 88.7, shinR: 80.1,
    racket: 20.0, racketLen: 0.1 },
  { t: 0.4727, hipCenter: [0.4863, 0.551], trunk: -87.7, neck: -99.2,
    uArmL: 63.2, fArmL: -2.7, uArmR: 49.5, fArmR: 26.4,
    thighL: 100.9, shinL: 84.0, thighR: 87.3, shinR: 79.8,
    racket: 26.4, racketLen: 0.1 },
  { t: 0.4924, hipCenter: [0.4837, 0.5501], trunk: -87.9, neck: -104.3,
    uArmL: 85.1, fArmL: 4.5, uArmR: 47.9, fArmR: 35.1,
    thighL: 100.6, shinL: 82.8, thighR: 87.3, shinR: 78.3,
    racket: 35.1, racketLen: 0.1 },
  { t: 0.5121, hipCenter: [0.4809, 0.5517], trunk: -87.2, neck: -106.0,
    uArmL: 96.2, fArmL: -6.5, uArmR: 48.7, fArmR: 45.2,
    thighL: 99.5, shinL: 82.6, thighR: 86.4, shinR: 77.2,
    racket: -43.6, racketLen: 0.08 },
  { t: 0.5318, hipCenter: [0.4786, 0.5514], trunk: -86.4, neck: -101.5,
    uArmL: 111.8, fArmL: -48.9, uArmR: 53.7, fArmR: 46.2,
    thighL: 98.1, shinL: 81.3, thighR: 86.9, shinR: 75.0,
    racket: -7.7, racketLen: 0.08 },
  { t: 0.5515, hipCenter: [0.477, 0.5501], trunk: -86.9, neck: -92.0,
    uArmL: 125.5, fArmL: -19.4, uArmR: 58.0, fArmR: 50.2,
    thighL: 96.3, shinL: 82.2, thighR: 87.7, shinR: 74.7,
    racket: 21.6, racketLen: 0.082 },
  { t: 0.5712, hipCenter: [0.4758, 0.5446], trunk: -87.3, neck: -82.8,
    uArmL: 128.1, fArmL: -106.0, uArmR: 61.9, fArmR: 55.1,
    thighL: 93.6, shinL: 84.4, thighR: 87.4, shinR: 72.9,
    racket: 35.9, racketLen: 0.1 },
  { t: 0.5909, hipCenter: [0.476, 0.5365], trunk: -88.7, neck: -82.9,
    uArmL: 120.9, fArmL: -136.6, uArmR: 69.9, fArmR: 61.4,
    thighL: 93.0, shinL: 85.1, thighR: 87.9, shinR: 72.2,
    racket: 41.9, racketLen: 0.108 },
  { t: 0.6106, hipCenter: [0.476, 0.5302], trunk: -89.8, neck: -81.2,
    uArmL: 124.0, fArmL: -133.1, uArmR: 80.8, fArmR: 73.5,
    thighL: 93.8, shinL: 87.8, thighR: 86.8, shinR: 71.3,
    racket: 27.5, racketLen: 0.08 },
  { t: 0.6303, hipCenter: [0.4758, 0.5253], trunk: -90.3, neck: -89.0,
    uArmL: 135.2, fArmL: -131.4, uArmR: 101.8, fArmR: 94.3,
    thighL: 95.2, shinL: 89.5, thighR: 85.0, shinR: 112.3,
    racket: 94.3, racketLen: 0.1 },
  { t: 0.65, hipCenter: [0.4782, 0.5245], trunk: -89.7, neck: -93.7,
    uArmL: 143.5, fArmL: -112.1, uArmR: 130.2, fArmR: -101.1,
    thighL: 97.4, shinL: 89.6, thighR: 87.1, shinR: 67.6,
    racket: -101.1, racketLen: 0.1 },
  { t: 0.6833, hipCenter: [0.4777, 0.525], trunk: -91.2, neck: -94.5,
    uArmL: 150.0, fArmL: -155.2, uArmR: 158.6, fArmR: -122.3,
    thighL: 98.8, shinL: 88.7, thighR: 86.6, shinR: 65.3,
    racket: 125.6, racketLen: 0.08 },
  { t: 0.7167, hipCenter: [0.4775, 0.5241], trunk: -90.3, neck: -100.9,
    uArmL: 168.6, fArmL: -151.2, uArmR: 174.5, fArmR: -143.8,
    thighL: 98.9, shinL: 88.7, thighR: 86.4, shinR: 61.1,
    racket: -147.6, racketLen: 0.113 },
  { t: 0.75, hipCenter: [0.4766, 0.5257], trunk: -90.3, neck: -102.6,
    uArmL: 177.8, fArmL: -134.9, uArmR: -164.3, fArmR: -135.4,
    thighL: 97.8, shinL: 88.4, thighR: 85.7, shinR: 59.0,
    racket: -136.5, racketLen: 0.141 },
  { t: 0.7833, hipCenter: [0.4752, 0.5301], trunk: -91.6, neck: -99.8,
    uArmL: 148.7, fArmL: -121.2, uArmR: -149.9, fArmR: -122.0,
    thighL: 96.1, shinL: 86.4, thighR: 84.6, shinR: 53.5,
    racket: -133.3, racketLen: 0.16 },
  { t: 0.8167, hipCenter: [0.4735, 0.5333], trunk: -89.7, neck: -101.4,
    uArmL: 159.5, fArmL: -117.7, uArmR: -143.4, fArmR: -113.1,
    thighL: 94.5, shinL: 84.4, thighR: 84.6, shinR: 47.9,
    racket: -131.1, racketLen: 0.16 },
  { t: 0.85, hipCenter: [0.4726, 0.5429], trunk: -87.9, neck: -119.8,
    uArmL: 178.1, fArmL: -116.5, uArmR: -140.4, fArmR: -113.2,
    thighL: 97.9, shinL: 82.6, thighR: 83.7, shinR: 44.2,
    racket: -133.7, racketLen: 0.16 },
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
