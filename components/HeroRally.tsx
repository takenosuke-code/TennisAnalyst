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
// (railway-service/extract_clip_keypoints.py). 47 frames at 30fps
// span prep through follow-through and recovery toward ready,
// mirrored around the y-axis so the right rally figure swings
// toward its opponent on the left. See the leading comment inside
// KEYFRAME_ANGLES for full pipeline.
const KEYFRAME_ANGLES: KeyframeAngles[] = [
  // 56 keyframes from a Carlos Alcaraz forehand. Racket arm is
  // RTMPose-extracted + smoothed; off-arm trajectory is
  // SYNTHESIZED via a half-sine envelope (idle -> peak -> idle)
  // so it never crosses inward across the body.
  // See railway-service/convert_pose_to_keyframes.py.
  //   - 7-frame moving avg on positions, 5-frame on most unwrapped
  //     angles, 9-frame on the racket-arm channels.
  //   - racket angle = smoothed forearm direction; racketLen=0.10.
  //   - mirrored y-axis (right figure hits left toward opponent).
  //   - uArmR cap: horizontal-LEFT reach >120 deg unwrapped scaled
  //     25%% (racket tip stays at shoulder height in follow-through).
  //   - off-arm: uArmL [95, 125], fArmL [100, 165], peaks at 54%%
  //     of the cycle (~just before contact). Strictly outward, no
  //     inward sweep.
  //   - extended post-contact window (0.85s) for smooth loop seam.
  //   - post-contact t-values stretched (0.65 -> 0.92): follow-
  //     through plays back ~35%% slower than the prep.
  { t: 0.0, hipCenter: [0.5488, 0.5805], trunk: -95.5, neck: -107.6,
    uArmL: 95.0, fArmL: 100.0, uArmR: 77.8, fArmR: 116.1,
    thighL: 101.8, shinL: 71.7, thighR: 95.6, shinR: 76.7,
    racket: 116.1, racketLen: 0.1 },
  { t: 0.0217, hipCenter: [0.5478, 0.5794], trunk: -95.4, neck: -107.6,
    uArmL: 96.9, fArmL: 104.1, uArmR: 77.4, fArmR: 116.5,
    thighL: 101.4, shinL: 71.5, thighR: 95.1, shinR: 77.0,
    racket: 116.5, racketLen: 0.1 },
  { t: 0.0433, hipCenter: [0.5464, 0.5781], trunk: -95.2, neck: -107.7,
    uArmL: 98.8, fArmL: 108.2, uArmR: 76.9, fArmR: 116.8,
    thighL: 100.8, shinL: 71.3, thighR: 94.5, shinR: 77.5,
    racket: 116.8, racketLen: 0.1 },
  { t: 0.065, hipCenter: [0.545, 0.5763], trunk: -94.9, neck: -107.9,
    uArmL: 100.6, fArmL: 112.2, uArmR: 76.3, fArmR: 117.1,
    thighL: 99.6, shinL: 70.8, thighR: 93.4, shinR: 78.5,
    racket: 117.1, racketLen: 0.1 },
  { t: 0.0867, hipCenter: [0.5426, 0.5737], trunk: -94.5, neck: -108.0,
    uArmL: 102.5, fArmL: 116.2, uArmR: 75.7, fArmR: 117.4,
    thighL: 98.1, shinL: 70.3, thighR: 92.2, shinR: 79.7,
    racket: 117.4, racketLen: 0.1 },
  { t: 0.1083, hipCenter: [0.54, 0.5707], trunk: -94.0, neck: -108.1,
    uArmL: 104.3, fArmL: 120.1, uArmR: 74.6, fArmR: 117.7,
    thighL: 96.4, shinL: 70.0, thighR: 91.1, shinR: 81.2,
    racket: 117.7, racketLen: 0.1 },
  { t: 0.13, hipCenter: [0.5372, 0.5671], trunk: -93.5, neck: -108.1,
    uArmL: 106.1, fArmL: 124.0, uArmR: 73.1, fArmR: 118.5,
    thighL: 94.6, shinL: 69.8, thighR: 90.2, shinR: 82.7,
    racket: 118.5, racketLen: 0.1 },
  { t: 0.1517, hipCenter: [0.5343, 0.5632], trunk: -92.9, neck: -108.0,
    uArmL: 107.8, fArmL: 127.8, uArmR: 71.3, fArmR: 120.7,
    thighL: 92.9, shinL: 69.7, thighR: 89.6, shinR: 84.2,
    racket: 120.7, racketLen: 0.1 },
  { t: 0.1733, hipCenter: [0.5313, 0.5597], trunk: -92.3, neck: -107.4,
    uArmL: 109.5, fArmL: 131.4, uArmR: 68.8, fArmR: 127.1,
    thighL: 91.6, shinL: 69.7, thighR: 89.4, shinR: 85.6,
    racket: 127.1, racketLen: 0.1 },
  { t: 0.195, hipCenter: [0.5286, 0.5567], trunk: -91.8, neck: -106.1,
    uArmL: 111.1, fArmL: 134.9, uArmR: 65.9, fArmR: 142.0,
    thighL: 90.7, shinL: 69.8, thighR: 89.5, shinR: 86.7,
    racket: 142.0, racketLen: 0.1 },
  { t: 0.2167, hipCenter: [0.526, 0.5549], trunk: -91.2, neck: -104.6,
    uArmL: 112.7, fArmL: 138.3, uArmR: 62.7, fArmR: 160.4,
    thighL: 90.3, shinL: 69.8, thighR: 90.0, shinR: 87.5,
    racket: 160.4, racketLen: 0.1 },
  { t: 0.2383, hipCenter: [0.5233, 0.5549], trunk: -90.8, neck: -103.3,
    uArmL: 114.2, fArmL: 141.5, uArmR: 59.1, fArmR: -179.0,
    thighL: 90.4, shinL: 70.0, thighR: 90.6, shinR: 87.9,
    racket: -179.0, racketLen: 0.1 },
  { t: 0.26, hipCenter: [0.5206, 0.5559], trunk: -90.4, neck: -102.0,
    uArmL: 115.6, fArmL: 144.6, uArmR: 55.3, fArmR: -156.9,
    thighL: 91.0, shinL: 70.6, thighR: 91.2, shinR: 88.0,
    racket: -156.9, racketLen: 0.1 },
  { t: 0.2817, hipCenter: [0.5178, 0.5577], trunk: -90.1, neck: -101.0,
    uArmL: 116.9, fArmL: 147.5, uArmR: 51.4, fArmR: -132.3,
    thighL: 91.9, shinL: 71.8, thighR: 91.7, shinR: 87.6,
    racket: -132.3, racketLen: 0.1 },
  { t: 0.3033, hipCenter: [0.5151, 0.5598], trunk: -89.9, neck: -100.9,
    uArmL: 118.2, fArmL: 150.2, uArmR: 47.8, fArmR: -105.7,
    thighL: 93.1, shinL: 73.8, thighR: 92.0, shinR: 87.0,
    racket: -105.7, racketLen: 0.1 },
  { t: 0.325, hipCenter: [0.5124, 0.562], trunk: -89.7, neck: -101.3,
    uArmL: 119.3, fArmL: 152.7, uArmR: 44.9, fArmR: -78.3,
    thighL: 94.5, shinL: 76.6, thighR: 92.1, shinR: 86.0,
    racket: -78.3, racketLen: 0.1 },
  { t: 0.3467, hipCenter: [0.5095, 0.5633], trunk: -89.5, neck: -102.2,
    uArmL: 120.4, fArmL: 155.0, uArmR: 42.7, fArmR: -51.1,
    thighL: 95.9, shinL: 79.4, thighR: 92.0, shinR: 84.9,
    racket: -51.1, racketLen: 0.1 },
  { t: 0.3683, hipCenter: [0.5063, 0.5642], trunk: -89.4, neck: -103.0,
    uArmL: 121.3, fArmL: 157.1, uArmR: 41.6, fArmR: -27.2,
    thighL: 97.3, shinL: 81.9, thighR: 91.5, shinR: 83.8,
    racket: -27.2, racketLen: 0.1 },
  { t: 0.39, hipCenter: [0.5029, 0.5634], trunk: -89.2, neck: -103.9,
    uArmL: 122.2, fArmL: 158.9, uArmR: 41.4, fArmR: -11.1,
    thighL: 98.4, shinL: 83.8, thighR: 90.9, shinR: 82.6,
    racket: -11.1, racketLen: 0.1 },
  { t: 0.4117, hipCenter: [0.4995, 0.5616], trunk: -88.9, neck: -104.4,
    uArmL: 122.9, fArmL: 160.5, uArmR: 42.0, fArmR: 1.9,
    thighL: 99.3, shinL: 84.8, thighR: 90.1, shinR: 81.3,
    racket: 1.9, racketLen: 0.1 },
  { t: 0.4333, hipCenter: [0.4959, 0.56], trunk: -88.6, neck: -104.2,
    uArmL: 123.6, fArmL: 161.9, uArmR: 43.7, fArmR: 13.4,
    thighL: 99.7, shinL: 84.9, thighR: 89.3, shinR: 80.2,
    racket: 13.4, racketLen: 0.1 },
  { t: 0.455, hipCenter: [0.4923, 0.5585], trunk: -88.2, neck: -102.5,
    uArmL: 124.1, fArmL: 163.0, uArmR: 46.3, fArmR: 24.0,
    thighL: 99.7, shinL: 84.6, thighR: 88.6, shinR: 79.1,
    racket: 24.0, racketLen: 0.1 },
  { t: 0.4767, hipCenter: [0.4888, 0.557], trunk: -87.9, neck: -100.2,
    uArmL: 124.5, fArmL: 163.9, uArmR: 49.9, fArmR: 32.9,
    thighL: 99.3, shinL: 84.2, thighR: 88.0, shinR: 78.0,
    racket: 32.9, racketLen: 0.1 },
  { t: 0.4983, hipCenter: [0.4856, 0.5554], trunk: -87.7, neck: -97.6,
    uArmL: 124.8, fArmL: 164.5, uArmR: 54.8, fArmR: 39.8,
    thighL: 98.5, shinL: 83.8, thighR: 87.6, shinR: 76.9,
    racket: 39.8, racketLen: 0.1 },
  { t: 0.52, hipCenter: [0.4831, 0.5523], trunk: -87.7, neck: -94.8,
    uArmL: 124.9, fArmL: 164.9, uArmR: 60.5, fArmR: 46.0,
    thighL: 97.6, shinL: 83.8, thighR: 87.3, shinR: 76.9,
    racket: 46.0, racketLen: 0.1 },
  { t: 0.5417, hipCenter: [0.4812, 0.5484], trunk: -87.8, neck: -92.1,
    uArmL: 125.0, fArmL: 165.0, uArmR: 67.4, fArmR: 52.5,
    thighL: 96.7, shinL: 84.2, thighR: 87.1, shinR: 76.8,
    racket: 52.5, racketLen: 0.1 },
  { t: 0.5633, hipCenter: [0.4797, 0.5438], trunk: -88.2, neck: -90.3,
    uArmL: 124.9, fArmL: 164.7, uArmR: 75.9, fArmR: 65.5,
    thighL: 96.0, shinL: 84.9, thighR: 87.0, shinR: 76.7,
    racket: 65.5, racketLen: 0.1 },
  { t: 0.585, hipCenter: [0.4792, 0.5387], trunk: -88.6, neck: -89.4,
    uArmL: 124.5, fArmL: 163.9, uArmR: 86.4, fArmR: 85.2,
    thighL: 95.7, shinL: 85.8, thighR: 86.9, shinR: 76.4,
    racket: 85.2, racketLen: 0.1 },
  { t: 0.6067, hipCenter: [0.4791, 0.5338], trunk: -89.1, neck: -89.6,
    uArmL: 123.9, fArmL: 162.5, uArmR: 99.0, fArmR: 106.2,
    thighL: 95.7, shinL: 86.7, thighR: 86.8, shinR: 76.1,
    racket: 106.2, racketLen: 0.1 },
  { t: 0.6283, hipCenter: [0.4791, 0.5289], trunk: -89.5, neck: -90.6,
    uArmL: 123.0, fArmL: 160.7, uArmR: 113.2, fArmR: 127.7,
    thighL: 95.9, shinL: 87.4, thighR: 86.6, shinR: 74.4,
    racket: 127.7, racketLen: 0.1 },
  { t: 0.65, hipCenter: [0.4793, 0.5254], trunk: -89.9, neck: -92.4,
    uArmL: 122.0, fArmL: 158.4, uArmR: 122.1, fArmR: 148.2,
    thighL: 96.3, shinL: 87.8, thighR: 86.4, shinR: 72.3,
    racket: 148.2, racketLen: 0.1 },
  { t: 0.6608, hipCenter: [0.4791, 0.5242], trunk: -90.1, neck: -95.0,
    uArmL: 121.3, fArmL: 157.1, uArmR: 125.8, fArmR: 167.8,
    thighL: 96.6, shinL: 87.8, thighR: 86.1, shinR: 68.5,
    racket: 167.8, racketLen: 0.1 },
  { t: 0.6716, hipCenter: [0.4787, 0.5248], trunk: -90.3, neck: -98.1,
    uArmL: 120.7, fArmL: 155.6, uArmR: 129.2, fArmR: -171.5,
    thighL: 97.4, shinL: 87.5, thighR: 85.8, shinR: 64.4,
    racket: -171.5, racketLen: 0.1 },
  { t: 0.6824, hipCenter: [0.4781, 0.5281], trunk: -90.4, neck: -101.2,
    uArmL: 120.0, fArmL: 154.1, uArmR: 132.0, fArmR: -150.2,
    thighL: 98.4, shinL: 86.7, thighR: 85.6, shinR: 60.0,
    racket: -150.2, racketLen: 0.1 },
  { t: 0.6932, hipCenter: [0.478, 0.5317], trunk: -90.6, neck: -104.4,
    uArmL: 119.2, fArmL: 152.4, uArmR: 134.0, fArmR: -129.8,
    thighL: 99.8, shinL: 85.7, thighR: 85.6, shinR: 55.5,
    racket: -129.8, racketLen: 0.1 },
  { t: 0.704, hipCenter: [0.478, 0.5358], trunk: -90.8, neck: -107.5,
    uArmL: 118.4, fArmL: 150.6, uArmR: 134.6, fArmR: -116.2,
    thighL: 101.5, shinL: 84.6, thighR: 85.8, shinR: 51.3,
    racket: -116.2, racketLen: 0.1 },
  { t: 0.7148, hipCenter: [0.4778, 0.5405], trunk: -91.2, neck: -110.5,
    uArmL: 117.5, fArmL: 148.8, uArmR: 134.1, fArmR: -109.0,
    thighL: 103.4, shinL: 83.7, thighR: 86.6, shinR: 49.0,
    racket: -109.0, racketLen: 0.1 },
  { t: 0.7256, hipCenter: [0.4781, 0.5447], trunk: -91.7, neck: -113.4,
    uArmL: 116.6, fArmL: 146.8, uArmR: 132.5, fArmR: -103.4,
    thighL: 105.1, shinL: 83.2, thighR: 87.8, shinR: 47.5,
    racket: -103.4, racketLen: 0.1 },
  { t: 0.7364, hipCenter: [0.4786, 0.5478], trunk: -92.4, neck: -115.6,
    uArmL: 115.6, fArmL: 144.7, uArmR: 130.3, fArmR: -97.1,
    thighL: 106.5, shinL: 83.1, thighR: 89.5, shinR: 47.2,
    racket: -97.1, racketLen: 0.1 },
  { t: 0.7472, hipCenter: [0.4796, 0.5496], trunk: -93.1, neck: -117.3,
    uArmL: 114.7, fArmL: 142.6, uArmR: 127.6, fArmR: -90.2,
    thighL: 107.2, shinL: 83.6, thighR: 91.7, shinR: 48.2,
    racket: -90.2, racketLen: 0.1 },
  { t: 0.758, hipCenter: [0.481, 0.5492], trunk: -93.9, neck: -118.6,
    uArmL: 113.6, fArmL: 140.3, uArmR: 124.7, fArmR: -81.1,
    thighL: 107.3, shinL: 84.6, thighR: 94.1, shinR: 50.6,
    racket: -81.1, racketLen: 0.1 },
  { t: 0.7688, hipCenter: [0.4814, 0.5484], trunk: -94.6, neck: -119.0,
    uArmL: 112.6, fArmL: 138.0, uArmR: 121.8, fArmR: -62.0,
    thighL: 106.7, shinL: 86.0, thighR: 96.5, shinR: 54.0,
    racket: -62.0, racketLen: 0.1 },
  { t: 0.7796, hipCenter: [0.482, 0.5469], trunk: -95.2, neck: -119.1,
    uArmL: 111.4, fArmL: 135.6, uArmR: 116.2, fArmR: -39.9,
    thighL: 105.6, shinL: 87.5, thighR: 98.7, shinR: 58.1,
    racket: -39.9, racketLen: 0.1 },
  { t: 0.7904, hipCenter: [0.4829, 0.5455], trunk: -95.6, neck: -119.0,
    uArmL: 110.3, fArmL: 133.2, uArmR: 107.6, fArmR: -17.9,
    thighL: 104.1, shinL: 89.0, thighR: 100.4, shinR: 62.7,
    racket: -17.9, racketLen: 0.1 },
  { t: 0.8012, hipCenter: [0.4835, 0.5446], trunk: -95.7, neck: -118.7,
    uArmL: 109.1, fArmL: 130.7, uArmR: 102.3, fArmR: 3.6,
    thighL: 102.8, shinL: 90.3, thighR: 101.5, shinR: 67.1,
    racket: 3.6, racketLen: 0.1 },
  { t: 0.812, hipCenter: [0.4843, 0.5442], trunk: -95.6, neck: -118.5,
    uArmL: 108.0, fArmL: 128.1, uArmR: 99.5, fArmR: 24.5,
    thighL: 101.6, shinL: 91.0, thighR: 102.1, shinR: 71.2,
    racket: 24.5, racketLen: 0.1 },
  { t: 0.8228, hipCenter: [0.4851, 0.5444], trunk: -95.4, neck: -118.3,
    uArmL: 106.7, fArmL: 125.4, uArmR: 97.9, fArmR: 45.2,
    thighL: 100.7, shinL: 91.2, thighR: 102.2, shinR: 74.7,
    racket: 45.2, racketLen: 0.1 },
  { t: 0.8336, hipCenter: [0.4861, 0.5448], trunk: -95.2, neck: -117.9,
    uArmL: 105.5, fArmL: 122.7, uArmR: 96.9, fArmR: 64.3,
    thighL: 100.1, shinL: 90.8, thighR: 102.0, shinR: 77.6,
    racket: 64.3, racketLen: 0.1 },
  { t: 0.8444, hipCenter: [0.4876, 0.5444], trunk: -94.9, neck: -117.6,
    uArmL: 104.2, fArmL: 120.0, uArmR: 96.2, fArmR: 83.3,
    thighL: 99.7, shinL: 89.9, thighR: 101.6, shinR: 80.2,
    racket: 83.3, racketLen: 0.1 },
  { t: 0.8552, hipCenter: [0.4894, 0.5434], trunk: -94.8, neck: -117.3,
    uArmL: 102.9, fArmL: 117.2, uArmR: 95.7, fArmR: 100.3,
    thighL: 99.6, shinL: 88.5, thighR: 101.3, shinR: 82.4,
    racket: 100.3, racketLen: 0.1 },
  { t: 0.866, hipCenter: [0.4915, 0.5414], trunk: -94.8, neck: -117.0,
    uArmL: 101.6, fArmL: 114.4, uArmR: 95.2, fArmR: 106.4,
    thighL: 99.6, shinL: 87.0, thighR: 101.0, shinR: 84.4,
    racket: 106.4, racketLen: 0.1 },
  { t: 0.8768, hipCenter: [0.494, 0.5389], trunk: -94.9, neck: -116.6,
    uArmL: 100.3, fArmL: 111.5, uArmR: 94.5, fArmR: 108.4,
    thighL: 99.7, shinL: 85.3, thighR: 100.8, shinR: 86.3,
    racket: 108.4, racketLen: 0.1 },
  { t: 0.8876, hipCenter: [0.4966, 0.5364], trunk: -95.1, neck: -116.2,
    uArmL: 99.0, fArmL: 108.7, uArmR: 94.3, fArmR: 109.3,
    thighL: 99.9, shinL: 83.7, thighR: 100.7, shinR: 88.1,
    racket: 109.3, racketLen: 0.1 },
  { t: 0.8984, hipCenter: [0.4981, 0.5346], trunk: -95.3, neck: -115.8,
    uArmL: 97.7, fArmL: 105.8, uArmR: 93.9, fArmR: 110.1,
    thighL: 100.0, shinL: 82.4, thighR: 100.6, shinR: 89.8,
    racket: 110.1, racketLen: 0.1 },
  { t: 0.9092, hipCenter: [0.4996, 0.5324], trunk: -95.4, neck: -115.6,
    uArmL: 96.3, fArmL: 102.9, uArmR: 93.3, fArmR: 110.8,
    thighL: 100.0, shinL: 81.7, thighR: 100.6, shinR: 90.6,
    racket: 110.8, racketLen: 0.1 },
  { t: 0.92, hipCenter: [0.501, 0.5307], trunk: -95.5, neck: -115.4,
    uArmL: 95.0, fArmL: 100.0, uArmR: 92.7, fArmR: 111.5,
    thighL: 100.1, shinL: 81.2, thighR: 100.6, shinR: 91.2,
    racket: 111.5, racketLen: 0.1 },
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
const FIGURE_Y_TRACK_LERP = 0.04
const FIGURE_Y_TRACK_AMPLITUDE_PX = 70
// Step-bob: when the figure has y-velocity (it's "walking" toward the
// ball's predicted contact point), modulate baseY by a small sine so
// the translation reads as steps rather than a glide. Tuned by feel —
// 4px peak-to-peak at ~3 steps/sec is barely perceptible but kills
// the "instant teleport" feel even at low velocities.
const STEP_BOB_AMP_PX = 2
const STEP_BOB_HZ = 3
const STEP_BOB_TRIGGER_VELOCITY = 30 // px/sec — below this, no bob

const BALL_RADIUS = 7
const BALL_SPEED = 540

// SWING_MS = 1600. Real pro forehands run 1.0-1.5s broadcast; this
// is ~0.6x speed, so the swing reads as a deliberate replica of the
// real motion rather than a slow-mo demo or a twitchy hero loop.
// User feedback iteration: 700 -> 1100 -> 1600.
const SWING_MS = 2400
const RACKET_HIT_RADIUS = 50
// At smootherstep5, solving `6u⁵ - 15u⁴ + 10u³ = 0.65` gives u ≈ 0.582,
// so contact lands 0.582 * 2400 ≈ 1397 ms after the swing fires.
const SWING_TO_CONTACT_MS = 1397
const SWING_CONTACT_PHASE = 0.65
// Racket-head position (figure-space) at SWING_CONTACT_PHASE. These
// must match the actual racket-head position the keyframes produce
// at t=0.65 — forward-evaluated through the bone chain (dampened
// hipCenter + fixed BONE_LENGTHS + polar angles), NOT the raw
// RTMPose-normalized position (which differs because of dampening
// and fixed bone-length idealization). The conversion script emits
// these values on every regeneration so they don't drift.
const RACKET_CONTACT_NORM_X = 0.2579
const RACKET_CONTACT_NORM_Y = 0.5161

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
    // Per-frame deltas drive the step-bob modulation; track previous
    // baseY so we can compute Y velocity inside step().
    let rightPrevBaseY = figureRestY
    let leftPrevBaseY = figureRestY

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

      // ─── Per-figure Y tracking + swing-fire ──────────────────────────
      // Both figures lerp baseY toward the predicted contact-Y for as
      // long as the ball is incoming, regardless of whether the swing
      // has already fired. This kills the old "instant snap" at the
      // swing-fire moment — the figure walks to position smoothly
      // through the swing's prep phase.
      // Right figure: ball is incoming when vx > 0 (moving rightward).
      if (ball.vx > 0) {
        const racketContactX = rightBaseX + (RACKET_CONTACT_NORM_X - 0.5) * FIGURE_PX_WIDTH
        const tToContact = (racketContactX - ball.x) / ball.vx
        if (tToContact > 0) {
          const predBallY = ball.y + ball.vy * tToContact
          const targetY = Math.max(
            figureRestY - FIGURE_Y_TRACK_AMPLITUDE_PX,
            Math.min(figureRestY + FIGURE_Y_TRACK_AMPLITUDE_PX, predBallY - racketYOffsetFromHip),
          )
          // Continuous lerp — covers both preswing and pre-contact swing.
          rightBaseY += (targetY - rightBaseY) * FIGURE_Y_TRACK_LERP
          // Fire the swing once the prep window opens.
          if (right.mode === 'preswing' && tToContact * 1000 <= SWING_TO_CONTACT_MS) {
            right.mode = 'swinging'
            right.swingStart = now
            right.hitRegistered = false
          }
        }
      }
      // Left figure: ball is incoming when vx < 0 (moving leftward).
      if (ball.vx < 0) {
        const racketContactX = leftBaseX - (RACKET_CONTACT_NORM_X - 0.5) * FIGURE_PX_WIDTH
        const tToContact = (racketContactX - ball.x) / ball.vx
        if (tToContact > 0) {
          const predBallY = ball.y + ball.vy * tToContact
          const targetY = Math.max(
            figureRestY - FIGURE_Y_TRACK_AMPLITUDE_PX,
            Math.min(figureRestY + FIGURE_Y_TRACK_AMPLITUDE_PX, predBallY - racketYOffsetFromHip),
          )
          leftBaseY += (targetY - leftBaseY) * FIGURE_Y_TRACK_LERP
          if (left.mode === 'preswing' && tToContact * 1000 <= SWING_TO_CONTACT_MS) {
            left.mode = 'swinging'
            left.swingStart = now
            left.hitRegistered = false
          }
        }
      }

      // Step-bob: when the figure has y-velocity (walking toward the
      // ball), add a tiny sinusoidal modulation to baseY at render
      // time so the translation reads as steps. Compute velocity from
      // last-frame baseY (per-figure).
      const rightDy = (rightBaseY - rightPrevBaseY) / Math.max(0.001, dt)
      const leftDy = (leftBaseY - leftPrevBaseY) / Math.max(0.001, dt)
      rightPrevBaseY = rightBaseY
      leftPrevBaseY = leftBaseY
      const rightBobAmp =
        Math.abs(rightDy) > STEP_BOB_TRIGGER_VELOCITY ? STEP_BOB_AMP_PX : 0
      const leftBobAmp =
        Math.abs(leftDy) > STEP_BOB_TRIGGER_VELOCITY ? STEP_BOB_AMP_PX : 0
      const bobPhase = (now / 1000) * STEP_BOB_HZ * 2 * Math.PI
      const rightBobOffset = Math.sin(bobPhase) * rightBobAmp
      const leftBobOffset = Math.sin(bobPhase + Math.PI) * leftBobAmp // antiphase

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
      paintFigure(rightPhase, rightRefs.current, rightBaseX, rightBaseY + rightBobOffset, false, rightLabelRefs.current)
      paintFigure(leftPhase, leftRefs.current, leftBaseX, leftBaseY + leftBobOffset, true, leftLabelRefs.current)
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
      // Reset prev-baseY trackers so the bob doesn't trigger a phantom
      // step-bob from the layout shift.
      rightPrevBaseY = rightBaseY
      leftPrevBaseY = leftBaseY
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
          stroke="var(--clay)"
          strokeOpacity="0.95"
          strokeWidth="2.5"
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
          stroke="var(--clay)"
          strokeOpacity="0.85"
          strokeWidth="2.5"
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
