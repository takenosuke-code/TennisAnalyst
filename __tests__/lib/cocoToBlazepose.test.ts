import { describe, it, expect } from 'vitest'
import {
  coco17ToBlazepose33,
  COCO17_TO_BLAZEPOSE33,
  BLAZEPOSE_LANDMARK_NAMES,
  NUM_BLAZEPOSE_LANDMARKS,
} from '@/lib/cocoToBlazepose'
import type { Landmark } from '@/lib/supabase'

// Small helper: build a synthetic COCO-17 keypoint table with a unique
// known position per id. Each id i sits at (i * 10, i * 10) px.
function makeCocoKpts(): number[][] {
  const out: number[][] = []
  for (let i = 0; i < 17; i++) {
    out.push([i * 10, i * 10])
  }
  return out
}

function makeCocoScores(value = 0.9): number[] {
  return new Array(17).fill(value)
}

describe('coco17ToBlazepose33', () => {
  // -----------------------------------------------------------------------
  // 1) Mapping correctness for each COCO id.
  // -----------------------------------------------------------------------

  describe('mapping correctness', () => {
    const imgW = 1000
    const imgH = 1000
    const kpts = makeCocoKpts()
    const scores = makeCocoScores(0.9)
    const out = coco17ToBlazepose33(kpts, scores, imgW, imgH)

    it('returns exactly 33 landmarks', () => {
      expect(out.length).toBe(NUM_BLAZEPOSE_LANDMARKS)
    })

    // Build a parameterized check across every entry in the mapping table.
    for (const [cocoIdStr, blazeId] of Object.entries(COCO17_TO_BLAZEPOSE33)) {
      const cocoId = Number(cocoIdStr)
      it(`copies COCO id ${cocoId} to BlazePose id ${blazeId} with normalized coords`, () => {
        const lm = out[blazeId]
        // i * 10 / 1000 = i * 0.01, exact at 4 decimals.
        const expected = cocoId * 0.01
        expect(lm.id).toBe(blazeId)
        expect(lm.name).toBe(BLAZEPOSE_LANDMARK_NAMES[blazeId])
        expect(lm.x).toBeCloseTo(expected, 6)
        expect(lm.y).toBeCloseTo(expected, 6)
        expect(lm.z).toBe(0)
        expect(lm.visibility).toBe(0.9)
      })
    }
  })

  // -----------------------------------------------------------------------
  // 2) Unmapped slots default to visibility=0.
  // -----------------------------------------------------------------------

  describe('unmapped slots', () => {
    const out = coco17ToBlazepose33(makeCocoKpts(), makeCocoScores(0.9), 1000, 1000)
    const mappedBlazeIds = new Set<number>(
      Object.values(COCO17_TO_BLAZEPOSE33).map((v) => v as number),
    )

    // Per the Python mapping: 1, 3, 4, 6, 9, 10, 17-22, 29-32 are unmapped.
    const expectedUnmapped = [1, 3, 4, 6, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]

    it('verifies the unmapped set matches the expected list', () => {
      const unmapped = []
      for (let i = 0; i < NUM_BLAZEPOSE_LANDMARKS; i++) {
        if (!mappedBlazeIds.has(i)) unmapped.push(i)
      }
      expect(unmapped).toEqual(expectedUnmapped)
    })

    for (const id of expectedUnmapped) {
      it(`BlazePose id ${id} (${BLAZEPOSE_LANDMARK_NAMES[id]}) is left at the visibility=0 default`, () => {
        const lm = out[id]
        expect(lm).toEqual<Landmark>({
          id,
          name: BLAZEPOSE_LANDMARK_NAMES[id],
          x: 0,
          y: 0,
          z: 0,
          visibility: 0,
        })
      })
    }
  })

  // -----------------------------------------------------------------------
  // 3) Normalization clamps to [0, 1].
  // -----------------------------------------------------------------------

  describe('normalization clamping', () => {
    it('clamps out-of-frame keypoints (positive overflow) to 1', () => {
      const imgW = 100
      const imgH = 200
      const kpts: number[][] = []
      for (let i = 0; i < 17; i++) kpts.push([imgW + 10, imgH + 10])
      const out = coco17ToBlazepose33(kpts, makeCocoScores(0.9), imgW, imgH)
      // BlazePose id 0 (nose) is COCO id 0 -> mapped.
      expect(out[0].x).toBe(1)
      expect(out[0].y).toBe(1)
      // Same for any other mapped id, e.g. left_shoulder (BlazePose 11).
      expect(out[11].x).toBe(1)
      expect(out[11].y).toBe(1)
    })

    it('clamps negative keypoints to 0', () => {
      const imgW = 100
      const imgH = 200
      const kpts: number[][] = []
      for (let i = 0; i < 17; i++) kpts.push([-5, -5])
      const out = coco17ToBlazepose33(kpts, makeCocoScores(0.9), imgW, imgH)
      expect(out[0].x).toBe(0)
      expect(out[0].y).toBe(0)
      expect(out[27].x).toBe(0) // left_ankle (COCO 15 -> BlazePose 27)
      expect(out[27].y).toBe(0)
    })
  })

  // -----------------------------------------------------------------------
  // 4) Rounding behavior.
  // -----------------------------------------------------------------------

  describe('rounding', () => {
    it('rounds x/y to 4 decimal places', () => {
      // Pick an input that produces 1/3 normalization: imgW=300, x=100 -> 0.33333…
      const imgW = 300
      const imgH = 300
      const kpts: number[][] = []
      for (let i = 0; i < 17; i++) kpts.push([100, 100])
      const out = coco17ToBlazepose33(kpts, makeCocoScores(0.9), imgW, imgH)
      // 0.3333333333… -> 0.3333 at 4 decimals.
      expect(out[0].x).toBe(0.3333)
      expect(out[0].y).toBe(0.3333)
      // Verify it's strictly 4-decimal precision (not 5+).
      const xStr = out[0].x.toString()
      const decimalsX = xStr.includes('.') ? xStr.split('.')[1].length : 0
      expect(decimalsX).toBeLessThanOrEqual(4)
    })

    it('rounds visibility to 3 decimal places', () => {
      // 0.123456 -> 0.123 at 3 decimals.
      const out = coco17ToBlazepose33(
        makeCocoKpts(),
        makeCocoScores(0.123456),
        1000,
        1000,
      )
      expect(out[0].visibility).toBe(0.123)
      // Check edge: 0.4567 -> 0.457
      const out2 = coco17ToBlazepose33(
        makeCocoKpts(),
        makeCocoScores(0.4567),
        1000,
        1000,
      )
      expect(out2[0].visibility).toBe(0.457)
    })

    it('rounds 0.5 thousandths up (Math.round half-up)', () => {
      // Python uses banker's rounding so 0.0005 -> 0.0 — but at three
      // decimals on physical confidences this collision is unobservable.
      // We document the JS behavior here so a future port-vs-source diff
      // surfaces immediately rather than being mistaken for a normalization
      // bug.
      const out = coco17ToBlazepose33(
        makeCocoKpts(),
        makeCocoScores(0.0015),
        1000,
        1000,
      )
      // Math.round(0.0015 * 1000) = Math.round(1.5) = 2 -> 0.002
      expect(out[0].visibility).toBe(0.002)
    })
  })

  // -----------------------------------------------------------------------
  // 5) Name field correctness.
  // -----------------------------------------------------------------------

  describe('name field', () => {
    const out = coco17ToBlazepose33(makeCocoKpts(), makeCocoScores(0.9), 1000, 1000)

    it('sets the right name on every landmark', () => {
      for (let i = 0; i < NUM_BLAZEPOSE_LANDMARKS; i++) {
        expect(out[i].name).toBe(BLAZEPOSE_LANDMARK_NAMES[i])
        expect(out[i].id).toBe(i)
      }
    })

    it('matches the spec-called-out names exactly', () => {
      expect(out[0].name).toBe('nose')
      expect(out[11].name).toBe('left_shoulder')
      expect(out[15].name).toBe('left_wrist')
      expect(out[23].name).toBe('left_hip')
    })

    it('exports the expected count', () => {
      expect(BLAZEPOSE_LANDMARK_NAMES.length).toBe(33)
      expect(NUM_BLAZEPOSE_LANDMARKS).toBe(33)
    })
  })

  // -----------------------------------------------------------------------
  // 6) Input validation.
  // -----------------------------------------------------------------------

  describe('input validation', () => {
    it('throws on wrong-length number[][] (16 entries)', () => {
      const bad: number[][] = []
      for (let i = 0; i < 16; i++) bad.push([0, 0])
      expect(() =>
        coco17ToBlazepose33(bad, makeCocoScores(0.9), 100, 100),
      ).toThrow(/shape \(17, 2\)/)
    })

    it('throws on wrong-length number[][] (18 entries)', () => {
      const bad: number[][] = []
      for (let i = 0; i < 18; i++) bad.push([0, 0])
      expect(() =>
        coco17ToBlazepose33(bad, makeCocoScores(0.9), 100, 100),
      ).toThrow(/shape \(17, 2\)/)
    })

    it('throws on wrong inner row length (3 instead of 2)', () => {
      const bad: number[][] = []
      for (let i = 0; i < 17; i++) bad.push([0, 0, 0])
      expect(() =>
        coco17ToBlazepose33(bad, makeCocoScores(0.9), 100, 100),
      ).toThrow(/shape \(17, 2\)/)
    })

    it('throws on wrong-length Float32Array', () => {
      const bad = new Float32Array(32) // expected 34
      expect(() =>
        coco17ToBlazepose33(bad, makeCocoScores(0.9), 100, 100),
      ).toThrow(/length 34/)
    })

    it('throws on wrong-length scores', () => {
      expect(() =>
        coco17ToBlazepose33(makeCocoKpts(), new Array(16).fill(0.9), 100, 100),
      ).toThrow(/coco_scores shape/)
    })

    it('throws on negative imgW', () => {
      expect(() =>
        coco17ToBlazepose33(makeCocoKpts(), makeCocoScores(0.9), -1, 100),
      ).toThrow(/image dims must be positive/)
    })

    it('throws on negative imgH', () => {
      expect(() =>
        coco17ToBlazepose33(makeCocoKpts(), makeCocoScores(0.9), 100, -1),
      ).toThrow(/image dims must be positive/)
    })

    it('throws on zero imgW', () => {
      expect(() =>
        coco17ToBlazepose33(makeCocoKpts(), makeCocoScores(0.9), 0, 100),
      ).toThrow(/image dims must be positive/)
    })

    it('throws on zero imgH', () => {
      expect(() =>
        coco17ToBlazepose33(makeCocoKpts(), makeCocoScores(0.9), 100, 0),
      ).toThrow(/image dims must be positive/)
    })

    it('throws on NaN imgW', () => {
      expect(() =>
        coco17ToBlazepose33(makeCocoKpts(), makeCocoScores(0.9), NaN, 100),
      ).toThrow(/image dims must be positive/)
    })
  })

  // -----------------------------------------------------------------------
  // 7) Float32Array vs number[][] equivalence.
  // -----------------------------------------------------------------------

  describe('Float32Array and number[][] inputs are equivalent', () => {
    const kptsArr = makeCocoKpts()
    const kptsTyped = new Float32Array(34)
    for (let i = 0; i < 17; i++) {
      kptsTyped[i * 2] = kptsArr[i][0]
      kptsTyped[i * 2 + 1] = kptsArr[i][1]
    }
    const scoresArr = makeCocoScores(0.876)
    const scoresTyped = Float32Array.from(scoresArr)

    it('produces identical output for both kpts shapes', () => {
      const a = coco17ToBlazepose33(kptsArr, scoresArr, 1000, 1000)
      const b = coco17ToBlazepose33(kptsTyped, scoresArr, 1000, 1000)
      expect(b).toEqual(a)
    })

    it('produces identical output for both scores shapes', () => {
      const a = coco17ToBlazepose33(kptsArr, scoresArr, 1000, 1000)
      const b = coco17ToBlazepose33(kptsArr, scoresTyped, 1000, 1000)
      expect(b).toEqual(a)
    })

    it('produces identical output when both inputs are typed arrays', () => {
      const a = coco17ToBlazepose33(kptsArr, scoresArr, 1000, 1000)
      const b = coco17ToBlazepose33(kptsTyped, scoresTyped, 1000, 1000)
      expect(b).toEqual(a)
    })
  })

  // -----------------------------------------------------------------------
  // Bonus: exported constants match the Python source.
  // -----------------------------------------------------------------------

  describe('exported constants', () => {
    it('mapping table matches the Python COCO17_TO_BLAZEPOSE33', () => {
      // Lifted verbatim from railway-service/pose_rtmpose.py:95-113
      const expected: Record<number, number> = {
        0: 0,
        1: 2,
        2: 5,
        3: 7,
        4: 8,
        5: 11,
        6: 12,
        7: 13,
        8: 14,
        9: 15,
        10: 16,
        11: 23,
        12: 24,
        13: 25,
        14: 26,
        15: 27,
        16: 28,
      }
      // Same number of entries
      expect(Object.keys(COCO17_TO_BLAZEPOSE33).length).toBe(17)
      for (const [cocoIdStr, blazeId] of Object.entries(expected)) {
        const cocoId = Number(cocoIdStr)
        expect(COCO17_TO_BLAZEPOSE33[cocoId]).toBe(blazeId)
      }
    })

    it('landmark name list matches the Python BLAZEPOSE_LANDMARK_NAMES', () => {
      // Lifted verbatim from railway-service/pose_rtmpose.py:115-124
      const expected = [
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
      ]
      expect(BLAZEPOSE_LANDMARK_NAMES).toEqual(expected)
    })
  })

  // -----------------------------------------------------------------------
  // Bonus: byte-equivalence fixture vs the Python reference.
  //
  // Hand-computed by running the Python `coco17_to_blazepose33` on the
  // input below; we encode the expected output here so a future divergence
  // (e.g. a rounding edge case) shows up as a test failure in CI rather
  // than as a silent client/server drift.
  // -----------------------------------------------------------------------

  describe('byte-equivalence fixture', () => {
    it('matches a hand-computed Python reference', () => {
      // Stick figure on a 1280x720 frame.
      const imgW = 1280
      const imgH = 720
      const kpts: number[][] = [
        [640, 100], // 0  nose
        [630, 95], // 1  L eye
        [650, 95], // 2  R eye
        [620, 100], // 3  L ear
        [660, 100], // 4  R ear
        [600, 200], // 5  L shoulder
        [680, 200], // 6  R shoulder
        [580, 300], // 7  L elbow
        [700, 300], // 8  R elbow
        [560, 400], // 9  L wrist
        [720, 400], // 10 R wrist
        [610, 400], // 11 L hip
        [670, 400], // 12 R hip
        [600, 530], // 13 L knee
        [680, 530], // 14 R knee
        [590, 660], // 15 L ankle
        [690, 660], // 16 R ankle
      ]
      const scores = [
        0.95, 0.91, 0.92, 0.88, 0.89, 0.94, 0.93, 0.9, 0.91, 0.87, 0.86, 0.92,
        0.93, 0.85, 0.84, 0.81, 0.82,
      ]
      const out = coco17ToBlazepose33(kpts, scores, imgW, imgH)

      // Spot-check across a representative sample of mapped ids.
      // Hand-computed: x = round(px / 1280, 4); y = round(py / 720, 4);
      // visibility = round(score, 3).
      expect(out[0]).toEqual({
        id: 0,
        name: 'nose',
        x: 0.5,
        y: 0.1389,
        z: 0,
        visibility: 0.95,
      })
      expect(out[11]).toEqual({
        id: 11,
        name: 'left_shoulder',
        x: 0.4688,
        y: 0.2778,
        z: 0,
        visibility: 0.94,
      })
      expect(out[15]).toEqual({
        id: 15,
        name: 'left_wrist',
        x: 0.4375,
        y: 0.5556,
        z: 0,
        visibility: 0.87,
      })
      expect(out[23]).toEqual({
        id: 23,
        name: 'left_hip',
        x: 0.4766,
        y: 0.5556,
        z: 0,
        visibility: 0.92,
      })
      expect(out[28]).toEqual({
        id: 28,
        name: 'right_ankle',
        x: 0.5391,
        y: 0.9167,
        z: 0,
        visibility: 0.82,
      })

      // Unmapped slot stays defaulted.
      expect(out[1]).toEqual({
        id: 1,
        name: 'left_eye_inner',
        x: 0,
        y: 0,
        z: 0,
        visibility: 0,
      })
      expect(out[31]).toEqual({
        id: 31,
        name: 'left_foot_index',
        x: 0,
        y: 0,
        z: 0,
        visibility: 0,
      })
    })
  })
})
