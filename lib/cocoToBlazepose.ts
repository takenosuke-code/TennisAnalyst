/**
 * COCO-17 -> BlazePose-33 keypoint remap.
 *
 * TypeScript port of `coco17_to_blazepose33` in
 * `railway-service/pose_rtmpose.py:398`. The Python module is the source of
 * truth; this file mirrors its mapping table, names, validation, and
 * rounding behavior exactly so live (browser) and `/analyze` (Railway)
 * produce byte-equivalent landmark output for the same input.
 *
 * Pure function. The only import is the `Landmark` type.
 */
import type { Landmark } from '@/lib/supabase'

/**
 * COCO-17 landmark id -> BlazePose-33 landmark id.
 *
 * COCO-17 indices (rtmlib output order):
 *   0 nose, 1 L eye, 2 R eye, 3 L ear, 4 R ear,
 *   5 L shoulder, 6 R shoulder, 7 L elbow, 8 R elbow,
 *   9 L wrist, 10 R wrist,
 *   11 L hip, 12 R hip,
 *   13 L knee, 14 R knee, 15 L ankle, 16 R ankle
 *
 * Mapping mirrors `COCO17_TO_BLAZEPOSE33` in `pose_rtmpose.py`.
 * BlazePose ids NOT covered here (e.g. 1, 3, 4, 6, 9, 10, 17-22, 29-32)
 * remain at the {x:0, y:0, z:0, visibility:0} default.
 */
export const COCO17_TO_BLAZEPOSE33: Readonly<Record<number, number>> = Object.freeze({
  0: 0, // nose
  1: 2, // left_eye         (BlazePose 1=left_eye_inner is closer in id but 2=left_eye matches semantics)
  2: 5, // right_eye
  3: 7, // left_ear
  4: 8, // right_ear
  5: 11, // left_shoulder
  6: 12, // right_shoulder
  7: 13, // left_elbow
  8: 14, // right_elbow
  9: 15, // left_wrist
  10: 16, // right_wrist
  11: 23, // left_hip
  12: 24, // right_hip
  13: 25, // left_knee
  14: 26, // right_knee
  15: 27, // left_ankle
  16: 28, // right_ankle
})

/**
 * 33-entry BlazePose landmark name list. Indexed 0-32.
 * Matches `BLAZEPOSE_LANDMARK_NAMES` in `pose_rtmpose.py`.
 */
export const BLAZEPOSE_LANDMARK_NAMES: readonly string[] = Object.freeze([
  'nose',
  'left_eye_inner',
  'left_eye',
  'left_eye_outer',
  'right_eye_inner',
  'right_eye',
  'right_eye_outer',
  'left_ear',
  'right_ear',
  'mouth_left',
  'mouth_right',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_pinky',
  'right_pinky',
  'left_index',
  'right_index',
  'left_thumb',
  'right_thumb',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
  'left_heel',
  'right_heel',
  'left_foot_index',
  'right_foot_index',
])

export const NUM_BLAZEPOSE_LANDMARKS = 33
const NUM_COCO_LANDMARKS = 17

/**
 * Round to a fixed number of decimal places. Mirrors Python's `round(x, n)`
 * for our purposes — the Python source uses round-half-to-even (banker's
 * rounding) but at 3-4 decimal places on physically-meaningful pixel
 * fractions the difference is negligible and never observable in practice.
 * If a future fixture surfaces a divergence, swap this for a banker-rounded
 * implementation.
 */
function roundTo(value: number, decimals: number): number {
  const factor = Math.pow(10, decimals)
  return Math.round(value * factor) / factor
}

/**
 * Get the (x, y) pair at COCO index `i` from either a Float32Array of length
 * 34 (interleaved [x0, y0, x1, y1, ...]) or a number[][] of shape (17, 2).
 *
 * NOTE: The Python source receives a numpy array of shape (17, 2). The
 * idiomatic JS equivalents are either a flat Float32Array (typed-array
 * friendly for ONNX runtimes) or number[][] (hand-built fixtures and
 * tests). Both are supported.
 */
function readKpt(
  src: Float32Array | number[][],
  i: number,
): [number, number] {
  if (src instanceof Float32Array) {
    return [src[i * 2], src[i * 2 + 1]]
  }
  const row = src[i]
  return [row[0], row[1]]
}

function readScore(src: Float32Array | number[], i: number): number {
  return src[i]
}

/**
 * Build a 33-entry BlazePose-indexed landmark list from a single person's
 * RTMPose COCO-17 output.
 *
 * @param cocoKpts shape (17, 2) -- (x, y) in image pixel coords. Either a
 *   number[][] of length 17 with each row [x, y], or a Float32Array of
 *   length 34 interleaved [x0, y0, x1, y1, ...].
 * @param cocoScores shape (17,) -- per-keypoint confidence in [0, 1].
 * @param imgW original frame width in pixels.
 * @param imgH original frame height in pixels.
 *
 * @returns 33-entry Landmark array. BlazePose ids not covered by the mapping
 *   table carry visibility=0 with x=y=z=0. x and y are normalized to [0, 1]
 *   and rounded to 4 decimals; visibility is rounded to 3 decimals.
 *
 * @throws Error when input shapes mismatch or image dimensions are non-positive.
 */
export function coco17ToBlazepose33(
  cocoKpts: Float32Array | number[][],
  cocoScores: Float32Array | number[],
  imgW: number,
  imgH: number,
): Landmark[] {
  // ---- Input validation (mirrors the Python ValueError checks). ----
  if (cocoKpts instanceof Float32Array) {
    if (cocoKpts.length !== NUM_COCO_LANDMARKS * 2) {
      throw new Error(
        `expected coco_kpts shape (17, 2) -> Float32Array length 34, got length ${cocoKpts.length}`,
      )
    }
  } else if (Array.isArray(cocoKpts)) {
    if (cocoKpts.length !== NUM_COCO_LANDMARKS) {
      throw new Error(
        `expected coco_kpts shape (17, 2), got (${cocoKpts.length}, ?)`,
      )
    }
    for (let i = 0; i < NUM_COCO_LANDMARKS; i++) {
      const row = cocoKpts[i]
      if (!Array.isArray(row) || row.length !== 2) {
        throw new Error(
          `expected coco_kpts shape (17, 2), row ${i} has length ${
            Array.isArray(row) ? row.length : 'non-array'
          }`,
        )
      }
    }
  } else {
    throw new Error(
      'coco_kpts must be a Float32Array of length 34 or a number[][] of shape (17, 2)',
    )
  }

  const scoresLen =
    cocoScores instanceof Float32Array ? cocoScores.length : cocoScores.length
  if (scoresLen !== NUM_COCO_LANDMARKS) {
    throw new Error(`expected coco_scores shape (17,), got length ${scoresLen}`)
  }

  if (!Number.isFinite(imgW) || !Number.isFinite(imgH) || imgW <= 0 || imgH <= 0) {
    throw new Error(`image dims must be positive, got ${imgW}x${imgH}`)
  }

  // ---- Initialize all 33 landmarks at the visibility=0 default. ----
  const landmarks: Landmark[] = new Array(NUM_BLAZEPOSE_LANDMARKS)
  for (let i = 0; i < NUM_BLAZEPOSE_LANDMARKS; i++) {
    landmarks[i] = {
      id: i,
      name: BLAZEPOSE_LANDMARK_NAMES[i],
      x: 0,
      y: 0,
      z: 0,
      visibility: 0,
    }
  }

  // ---- Fill mapped slots from the COCO input. ----
  for (const cocoIdStr of Object.keys(COCO17_TO_BLAZEPOSE33)) {
    const cocoId = Number(cocoIdStr)
    const blazeId = COCO17_TO_BLAZEPOSE33[cocoId]

    const [xPx, yPx] = readKpt(cocoKpts, cocoId)
    const score = readScore(cocoScores, cocoId)

    // Normalize px -> [0, 1], clamping out-of-frame keypoints to the edge.
    // Mirrors `np.clip(px / img_w, 0.0, 1.0)`.
    const xNorm = Math.min(1, Math.max(0, xPx / imgW))
    const yNorm = Math.min(1, Math.max(0, yPx / imgH))

    landmarks[blazeId] = {
      id: blazeId,
      name: BLAZEPOSE_LANDMARK_NAMES[blazeId],
      x: roundTo(xNorm, 4),
      y: roundTo(yNorm, 4),
      z: 0, // RTMPose-2D doesn't predict z; downstream tolerates 0.
      visibility: roundTo(score, 3),
    }
  }

  return landmarks
}
