import { describe, it, expect } from 'vitest'
import { detectStrokes, deriveFps, STROKE_DEFAULTS } from '@/lib/swingDetect'
import type { DetectedStroke } from '@/lib/strokeAnalysis'
import type { PoseFrame, JointAngles, Landmark } from '@/lib/supabase'
import { LANDMARK_INDICES } from '@/lib/jointAngles'
import { makeFrame, makeStandingPose, makeRestFrames } from '../helpers'

// ---------------------------------------------------------------------------
// Test fixtures.
//
// We build synthetic clips by directly placing the right (or left) wrist
// landmark frame-by-frame. The detector reads dominant-wrist linear speed
// in normalized [0,1] coords / second, so positioning the wrist
// explicitly is the cleanest way to control which frame becomes the peak.
// ---------------------------------------------------------------------------

interface WristTrack {
  x: number
  y: number
}

function frameWithRightWrist(
  index: number,
  timestampMs: number,
  wrist: WristTrack,
  angles: JointAngles = {},
): PoseFrame {
  const landmarks: Landmark[] = makeStandingPose().map((lm) =>
    lm.id === LANDMARK_INDICES.RIGHT_WRIST
      ? { ...lm, x: wrist.x, y: wrist.y }
      : lm,
  )
  return makeFrame(index, timestampMs, landmarks, angles)
}

/**
 * Build a clip with the right wrist following an arbitrary trajectory.
 * The trajectory is sampled at `fps` for `numFrames` frames. Useful when
 * we want full control over the speed signal — backswing-then-peak,
 * dual-peak, etc.
 */
function buildWristClip(
  numFrames: number,
  fps: number,
  trajectory: (frameIdx: number) => WristTrack,
  startMs = 0,
): PoseFrame[] {
  const dtMs = 1000 / fps
  const frames: PoseFrame[] = []
  for (let i = 0; i < numFrames; i++) {
    frames.push(frameWithRightWrist(i, startMs + i * dtMs, trajectory(i)))
  }
  return frames
}

/**
 * Single-peak swing: position follows a sigmoid that crosses
 * `peakFrameIdx`. Wrist starts at REST_X, accelerates smoothly through
 * the peak, and lands at (REST_X + amplitude). Because position is
 * monotonic, the SPEED signal has exactly one peak — at peakFrameIdx —
 * which is what the detector locks on to.
 *
 * `slope` controls how sharp the peak is: larger = narrower, taller
 * peak. The default 0.4 gives a peak ~6 frames wide.
 */
function singlePeakSwing(
  numFrames: number,
  fps: number,
  peakFrameIdx: number,
  amplitude = 0.45,
  slope = 0.4,
): PoseFrame[] {
  const REST_X = 0.4
  return buildWristClip(numFrames, fps, (i) => {
    const u = (i - peakFrameIdx) * slope
    const sigmoid = 1 / (1 + Math.exp(-u))
    return { x: REST_X + amplitude * sigmoid, y: 0.55 }
  })
}

/**
 * Chain of sigmoid swings — one acceleration burst per peak frame.
 * Each burst takes the wrist from one rest position to the next, so
 * the wrist's resting position drifts forward between swings (which
 * is fine for the speed-based detector).
 */
function sigmoidChain(
  numFrames: number,
  fps: number,
  peakFrames: number[],
  amplitudePerSwing = 0.4,
  slope = 0.5,
): PoseFrame[] {
  const REST_X = 0.4
  return buildWristClip(numFrames, fps, (i) => {
    let x = REST_X
    for (const center of peakFrames) {
      const u = (i - center) * slope
      const sigmoid = 1 / (1 + Math.exp(-u))
      x += amplitudePerSwing * sigmoid
    }
    return { x, y: 0.55 }
  })
}

// ---------------------------------------------------------------------------
// deriveFps
// ---------------------------------------------------------------------------

describe('deriveFps', () => {
  it('returns 30 for an empty frame list', () => {
    expect(deriveFps([])).toBe(30)
  })

  it('returns 30 for a single frame', () => {
    expect(deriveFps([makeFrame(0, 0, [])])).toBe(30)
  })

  it('derives 30 fps from exact 1/30s intervals', () => {
    const frames: PoseFrame[] = []
    for (let i = 0; i < 10; i++) {
      frames.push(makeFrame(i, (i * 1000) / 30, []))
    }
    expect(deriveFps(frames)).toBe(30)
  })

  it('derives 60 fps from exact 1/60s intervals', () => {
    const frames: PoseFrame[] = []
    for (let i = 0; i < 10; i++) {
      frames.push(makeFrame(i, (i * 1000) / 60, []))
    }
    expect(deriveFps(frames)).toBe(60)
  })

  it('derives ~30 fps from rounded 33 ms intervals', () => {
    // Real captures often round timestamps to whole milliseconds.
    // 33 ms intervals yield fps ≈ 30.30 — close enough for the
    // detector. We just want the rounding to land on a sane integer.
    const frames: PoseFrame[] = []
    for (let i = 0; i < 10; i++) {
      frames.push(makeFrame(i, Math.round((i * 1000) / 30), []))
    }
    const fps = deriveFps(frames)
    expect(fps).toBeGreaterThan(29)
    expect(fps).toBeLessThan(31)
  })

  it('falls back to 30 when timestamps are all zero', () => {
    const frames: PoseFrame[] = Array.from({ length: 5 }, (_, i) =>
      makeFrame(i, 0, []),
    )
    expect(deriveFps(frames)).toBe(30)
  })
})

// ---------------------------------------------------------------------------
// detectStrokes — output contract
// ---------------------------------------------------------------------------

describe('detectStrokes — output contract', () => {
  it('returns DetectedStroke[] with the spec-mandated fields', () => {
    const frames = singlePeakSwing(60, 30, 30)
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBeGreaterThanOrEqual(1)
    const s = strokes[0]
    expect(s.strokeId).toBe('stroke_0')
    expect(typeof s.startFrame).toBe('number')
    expect(typeof s.endFrame).toBe('number')
    expect(typeof s.peakFrame).toBe('number')
    expect(typeof s.fps).toBe('number')
    // No extra fields beyond the contract
    expect(Object.keys(s).sort()).toEqual(
      ['endFrame', 'fps', 'peakFrame', 'startFrame', 'strokeId'].sort(),
    )
  })

  it('returns strokes sorted by peakFrame ascending', () => {
    // Two sigmoid acceleration bursts, peaks at frames 30 and 90.
    // The wrist races forward through frame 30 to a new resting
    // position, sits there for ~30 frames, then races forward again
    // through frame 90 — like back-to-back forehands.
    const frames = sigmoidChain(150, 30, [30, 90])
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(2)
    expect(strokes[0].peakFrame).toBeLessThan(strokes[1].peakFrame)
  })

  it('strokeIds are zero-indexed in peakFrame order', () => {
    const frames = sigmoidChain(200, 30, [40, 100, 160])
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(3)
    expect(strokes[0].strokeId).toBe('stroke_0')
    expect(strokes[1].strokeId).toBe('stroke_1')
    expect(strokes[2].strokeId).toBe('stroke_2')
  })

  it('returns an empty array for a flat (no-motion) clip', () => {
    const frames = makeRestFrames(60, 0)
    const strokes = detectStrokes(frames)
    expect(strokes).toEqual([])
  })

  it('returns an empty array for a clip shorter than 2 frames', () => {
    expect(detectStrokes([])).toEqual([])
    expect(detectStrokes([makeFrame(0, 0, makeStandingPose())])).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// Hysteresis-bug regression
// ---------------------------------------------------------------------------

describe('detectStrokes — hysteresis-bug regression', () => {
  it('includes a 900ms backswing in startFrame (old algorithm clipped it)', () => {
    // Build a clip where the wrist:
    //   - sits at rest for 30 frames (1s)
    //   - drifts very slowly through "backswing" for 27 frames (~900ms),
    //     accumulating only ~10% of the total wrist travel
    //   - explodes to peak velocity at frame 60 (contact)
    //   - decelerates over 15 frames (follow-through)
    //   - returns to rest
    //
    // Under the old hysteresis-energy detector, the slow-drift backswing
    // sat below the enter threshold and got clipped — the segment
    // started at the explosive contact-onset, not at backswing-start.
    // The new pre-pad of ~1s at 30fps is 30 frames, so the start frame
    // should land at peakFrame - 30 = 30 (backswing-start), NOT at the
    // contact onset.
    const FPS = 30
    const PEAK_FRAME = 60
    const frames = buildWristClip(120, FPS, (i) => {
      const REST_X = 0.4
      const PEAK_X = 0.85
      if (i < 30) return { x: REST_X, y: 0.55 } // pre-rest
      if (i < 57) {
        // 27-frame backswing — small linear drift
        const u = (i - 30) / 27
        return { x: REST_X - 0.05 * u, y: 0.55 } // wrist pulls back to 0.35
      }
      if (i <= 60) {
        // 3-frame snap to contact (huge velocity)
        const u = (i - 57) / 3
        return { x: 0.35 + (PEAK_X - 0.35) * u, y: 0.5 }
      }
      if (i <= 75) {
        // Follow-through ramp down
        const u = (i - 60) / 15
        return { x: PEAK_X - (PEAK_X - REST_X) * u, y: 0.55 }
      }
      return { x: REST_X, y: 0.55 }
    })

    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
    const s = strokes[0]
    // Peak must be near the snap (frame 58-60).
    expect(s.peakFrame).toBeGreaterThanOrEqual(57)
    expect(s.peakFrame).toBeLessThanOrEqual(60)
    // Pre-pad of ≈1s @ 30fps = 30 frames. The start frame must include
    // the backswing window — i.e. start at or before frame 30.
    expect(s.startFrame).toBeLessThanOrEqual(30)
  })
})

// ---------------------------------------------------------------------------
// Padding scales with fps
// ---------------------------------------------------------------------------

describe('detectStrokes — biomechanical padding', () => {
  it('produces ~1s pre-pad / ~0.5s post-pad at 30 fps', () => {
    const FPS = 30
    const frames = singlePeakSwing(120, FPS, 60)
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
    const s = strokes[0]
    // 1s @ 30fps = 30 frames
    expect(s.peakFrame - s.startFrame).toBeCloseTo(30, 0)
    // 0.5s @ 30fps = 15 frames
    expect(s.endFrame - s.peakFrame).toBeCloseTo(15, 0)
    expect(s.fps).toBe(30)
  })

  it('produces ~1s pre-pad / ~0.5s post-pad at 60 fps', () => {
    const FPS = 60
    const frames = singlePeakSwing(240, FPS, 120)
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
    const s = strokes[0]
    // 1s @ 60fps = 60 frames
    expect(s.peakFrame - s.startFrame).toBeCloseTo(60, 0)
    // 0.5s @ 60fps = 30 frames
    expect(s.endFrame - s.peakFrame).toBeCloseTo(30, 0)
    expect(s.fps).toBe(60)
  })

  it('honours the fps override option', () => {
    const frames = singlePeakSwing(120, 30, 60)
    // Tell the detector this is a 60fps clip — pad windows should
    // halve in absolute frame count.
    const strokes = detectStrokes(frames, { fps: 60 })
    expect(strokes.length).toBe(1)
    const s = strokes[0]
    expect(s.peakFrame - s.startFrame).toBeCloseTo(60, 0)
    expect(s.endFrame - s.peakFrame).toBeCloseTo(30, 0)
    expect(s.fps).toBe(60)
  })
})

// ---------------------------------------------------------------------------
// Refractory window
// ---------------------------------------------------------------------------

describe('detectStrokes — 500ms refractory window', () => {
  it('keeps the higher of two peaks 300ms apart', () => {
    // Two sigmoid bursts at frames 30 and 39 (300ms apart at 30fps).
    // The second burst is taller (larger amplitude → higher peak
    // speed). Refractory should drop frame 30 and keep frame 39.
    const FPS = 30
    const frames = buildWristClip(120, FPS, (i) => {
      const REST_X = 0.4
      const sig = (center: number, amp: number, slope = 1.0) =>
        amp * (1 / (1 + Math.exp(-(i - center) * slope)))
      // Two acceleration bursts; second is bigger (peak speed
      // proportional to amplitude × slope/4 for a sigmoid).
      return { x: REST_X + sig(30, 0.20) + sig(39, 0.45), y: 0.55 }
    })
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
    // The taller peak survives. Smoothing may shift by ±1 frame.
    expect(strokes[0].peakFrame).toBeGreaterThanOrEqual(38)
    expect(strokes[0].peakFrame).toBeLessThanOrEqual(40)
  })

  it('keeps both peaks 600ms apart', () => {
    // Two equal sigmoid bursts at frames 30 and 48 (600ms apart).
    const FPS = 30
    const frames = sigmoidChain(120, FPS, [30, 48])
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(2)
    expect(strokes[0].peakFrame).toBeGreaterThanOrEqual(29)
    expect(strokes[0].peakFrame).toBeLessThanOrEqual(31)
    expect(strokes[1].peakFrame).toBeGreaterThanOrEqual(47)
    expect(strokes[1].peakFrame).toBeLessThanOrEqual(49)
  })
})

// ---------------------------------------------------------------------------
// Multiple distinct swings
// ---------------------------------------------------------------------------

describe('detectStrokes — multiple swings', () => {
  it('detects three distinct swings spaced by long rests', () => {
    const FPS = 30
    const frames = sigmoidChain(300, FPS, [60, 150, 240])
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(3)
    // Smoothing may shift the reported peak by ±1 frame.
    expect(strokes[0].peakFrame).toBeGreaterThanOrEqual(59)
    expect(strokes[0].peakFrame).toBeLessThanOrEqual(61)
    expect(strokes[1].peakFrame).toBeGreaterThanOrEqual(149)
    expect(strokes[1].peakFrame).toBeLessThanOrEqual(151)
    expect(strokes[2].peakFrame).toBeGreaterThanOrEqual(239)
    expect(strokes[2].peakFrame).toBeLessThanOrEqual(241)
    expect(strokes.map((s) => s.strokeId)).toEqual([
      'stroke_0',
      'stroke_1',
      'stroke_2',
    ])
  })
})

// ---------------------------------------------------------------------------
// Edge cases — boundary clamping
// ---------------------------------------------------------------------------

describe('detectStrokes — edge cases', () => {
  it('clamps startFrame to 0 when the peak lands near frame 0', () => {
    // Sigmoid burst centered at frame 3 — peak speed is at frame 3.
    // Pre-pad of ~30 frames would push startFrame to -27; clamp must
    // hold it at 0.
    const FPS = 30
    const frames = singlePeakSwing(80, FPS, 3)
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
    expect(strokes[0].peakFrame).toBeLessThanOrEqual(5)
    expect(strokes[0].startFrame).toBe(0)
  })

  it('clamps endFrame to totalFrames-1 when the peak lands near the end', () => {
    // Sigmoid burst centered near the last frame.
    const FPS = 30
    const N = 50
    const frames = singlePeakSwing(N, FPS, N - 3)
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
    expect(strokes[0].peakFrame).toBeGreaterThanOrEqual(N - 5)
    expect(strokes[0].endFrame).toBe(N - 1)
  })
})

// ---------------------------------------------------------------------------
// Dominant-hand selection
// ---------------------------------------------------------------------------

describe('detectStrokes — dominant-hand selection', () => {
  it('honours an explicit dominantHand option', () => {
    // Build a clip where ONLY the LEFT wrist moves. With no override,
    // pickDominantWrist falls back to right (default) and finds nothing.
    // With dominantHand: 'left', the detector should locate the swing.
    // We use a sigmoid position curve centered at frame 40 — the
    // derivative is a bell, so speed peaks cleanly at frame 40 even
    // after the 5-frame moving-average smoother.
    const N = 80
    const FPS = 30
    const frames: PoseFrame[] = []
    for (let i = 0; i < N; i++) {
      const u = (i - 40) * 0.4
      const sigmoid = 1 / (1 + Math.exp(-u))
      const leftWristX = 0.4 + 0.45 * sigmoid
      const landmarks = makeStandingPose().map((lm) =>
        lm.id === LANDMARK_INDICES.LEFT_WRIST
          ? { ...lm, x: leftWristX, y: 0.55 }
          : lm,
      )
      frames.push(makeFrame(i, Math.round((i * 1000) / FPS), landmarks))
    }

    const noOverride = detectStrokes(frames)
    expect(noOverride).toEqual([]) // right wrist is static → nothing

    const withOverride = detectStrokes(frames, { dominantHand: 'left' })
    expect(withOverride.length).toBe(1)
    // Smoothing may shift the reported peak by ±1 frame.
    expect(withOverride[0].peakFrame).toBeGreaterThanOrEqual(39)
    expect(withOverride[0].peakFrame).toBeLessThanOrEqual(41)
  })

  it('picks the higher-visibility wrist when no override is provided', () => {
    // Build a clip where both wrists move identically but the LEFT
    // wrist has high visibility throughout while the RIGHT wrist sits
    // below the visibility floor (0.5). Detector should track the LEFT.
    // Same sigmoid trajectory so the speed peak lands cleanly at 40.
    const N = 80
    const FPS = 30
    const frames: PoseFrame[] = []
    for (let i = 0; i < N; i++) {
      const u = (i - 40) * 0.4
      const sigmoid = 1 / (1 + Math.exp(-u))
      const wristX = 0.4 + 0.45 * sigmoid
      const landmarks = makeStandingPose().map((lm) => {
        if (lm.id === LANDMARK_INDICES.RIGHT_WRIST) {
          return { ...lm, x: wristX, y: 0.55, visibility: 0.2 } // below floor
        }
        if (lm.id === LANDMARK_INDICES.LEFT_WRIST) {
          return { ...lm, x: wristX, y: 0.55, visibility: 1.0 }
        }
        return lm
      })
      frames.push(makeFrame(i, Math.round((i * 1000) / FPS), landmarks))
    }

    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
    expect(strokes[0].peakFrame).toBeGreaterThanOrEqual(39)
    expect(strokes[0].peakFrame).toBeLessThanOrEqual(41)
  })
})

// ---------------------------------------------------------------------------
// Default options sanity check
// ---------------------------------------------------------------------------

describe('detectStrokes — defaults', () => {
  it('uses 1000ms / 500ms / 500ms as documented defaults', () => {
    expect(STROKE_DEFAULTS.prePadMs).toBe(1000)
    expect(STROKE_DEFAULTS.postPadMs).toBe(500)
    expect(STROKE_DEFAULTS.refractoryMs).toBe(500)
  })

  it('respects custom prePadMs / postPadMs', () => {
    const frames = singlePeakSwing(120, 30, 60)
    const strokes = detectStrokes(frames, { prePadMs: 500, postPadMs: 200 })
    expect(strokes.length).toBe(1)
    const s = strokes[0]
    // 500ms @ 30fps = 15 frames pre, 200ms = 6 frames post
    expect(s.peakFrame - s.startFrame).toBeCloseTo(15, 0)
    expect(s.endFrame - s.peakFrame).toBeCloseTo(6, 0)
  })
})

// Reference type usage — purely a compile-time check that DetectedStroke
// is exported correctly. Doesn't run any assertions.
const _typecheck: DetectedStroke = {
  strokeId: 'stroke_0',
  startFrame: 0,
  endFrame: 1,
  peakFrame: 1,
  fps: 30,
}
void _typecheck
