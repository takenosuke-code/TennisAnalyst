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
  it('uses 1000ms / 500ms / 350ms as documented defaults', () => {
    expect(STROKE_DEFAULTS.prePadMs).toBe(1000)
    expect(STROKE_DEFAULTS.postPadMs).toBe(500)
    expect(STROKE_DEFAULTS.refractoryMs).toBe(350)
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

// ---------------------------------------------------------------------------
// 2026-05 rewrite — over-detection + overlap regressions on long video
// ---------------------------------------------------------------------------

describe('detectStrokes — prominence collapses jittery shoulders on a real swing', () => {
  it('returns ONE peak when a real swing carries 3 jittery local maxima around its true peak', () => {
    // Build a clip where the wrist trajectory has a clear sigmoid swing
    // centered at frame 60, but with sub-frame-amplitude jitter that
    // creates 2-3 local maxima in the velocity signal around the peak.
    // Old detector: each local max above threshold = a separate stroke.
    // New detector: prominence walks find the trough between candidate
    // peaks; only the true peak survives.
    const FPS = 30
    const frames = buildWristClip(120, FPS, (i) => {
      const u = (i - 60) * 0.5
      const sigmoid = 1 / (1 + Math.exp(-u))
      // Add tiny jitter that perturbs position in a way that creates
      // multiple local maxima in the SPEED signal around the peak.
      const jitter =
        Math.abs(i - 60) < 8 ? 0.003 * Math.sin((i - 60) * 1.7) : 0
      return { x: 0.4 + 0.45 * sigmoid + jitter, y: 0.55 }
    })
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
    expect(strokes[0].peakFrame).toBeGreaterThanOrEqual(58)
    expect(strokes[0].peakFrame).toBeLessThanOrEqual(62)
  })
})

describe('detectStrokes — Voronoi boundaries on close-spaced strokes', () => {
  it('two strokes ~800ms apart produce non-overlapping clips (midpoint boundary)', () => {
    // 800ms at 30fps = 24 frames. Pre-pad of 30 frames + post-pad of 15
    // would have overlapped these clips by ~21 frames each. Voronoi
    // splits at the midpoint between peaks so neither clip overlaps.
    const FPS = 30
    const frames = sigmoidChain(150, FPS, [40, 64], 0.4, 0.6)
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(2)
    const [a, b] = strokes
    // Clips must not overlap (a.endFrame < b.startFrame).
    expect(a.endFrame).toBeLessThan(b.startFrame)
    // Midpoint between peaks ~52: a.endFrame should be at 52 (or 51,52)
    // and b.startFrame should be at 53.
    const midpoint = Math.floor((a.peakFrame + b.peakFrame) / 2)
    expect(a.endFrame).toBeLessThanOrEqual(midpoint)
    expect(b.startFrame).toBeGreaterThanOrEqual(midpoint)
  })

  it('three back-to-back strokes spaced ~1s apart all isolate cleanly', () => {
    // Three sigmoid bursts 30 frames apart. Default pre-pad is 30
    // frames — without Voronoi, each clip would extend back to the
    // previous stroke's peak. With Voronoi, every frame belongs to
    // exactly one stroke.
    const FPS = 30
    const frames = sigmoidChain(180, FPS, [40, 70, 100], 0.4, 0.6)
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(3)
    // Adjacent clips must not overlap.
    expect(strokes[0].endFrame).toBeLessThan(strokes[1].startFrame)
    expect(strokes[1].endFrame).toBeLessThan(strokes[2].startFrame)
  })

  it('isolated stroke uses full pre/post pad (no neighbor → no Voronoi shrink)', () => {
    const FPS = 30
    const frames = singlePeakSwing(120, FPS, 60)
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
    const s = strokes[0]
    // Default 1000ms pre-pad / 500ms post-pad at 30fps.
    expect(s.peakFrame - s.startFrame).toBeCloseTo(30, 0)
    expect(s.endFrame - s.peakFrame).toBeCloseTo(15, 0)
  })
})

describe('detectStrokes — long-video over-detection regression', () => {
  it('30-second clip with 4 real swings + walking-arm sway returns exactly 4 strokes', () => {
    // Realistic long-video failure mode: between strokes the user is
    // adjusting grip / walking back to position. The wrist still moves
    // (low-amplitude, low-frequency), but it shouldn't fire as a stroke.
    // Old detector: P90 threshold → walking sway clears it → many
    // false positives. New detector: prominence floor 0.3 + width
    // filter → only the real swings survive.
    //
    // Use a monotonic sigmoid-chain trajectory so each planted swing
    // produces exactly one speed peak (no symmetric deceleration tail
    // creating a phantom second peak).
    const FPS = 30
    const TOTAL = 900 // 30 seconds
    const swingCenters = [120, 320, 540, 760]

    const frames = buildWristClip(TOTAL, FPS, (i) => {
      // Slow walking sway: low-amplitude sin that should never clear
      // the prominence floor.
      const sway = 0.02 * Math.sin(i * 0.05)
      // Each planted swing shifts the wrist's resting position
      // forward by a fixed amount via a steep sigmoid. Cumulative
      // drift across 4 swings is 0.4 (well under 1.0 normalized).
      let pos = 0.45
      for (const center of swingCenters) {
        pos += 0.1 / (1 + Math.exp(-(i - center) * 0.6))
      }
      return { x: pos + sway, y: 0.55 }
    })

    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(4)
    for (let i = 0; i < 4; i++) {
      expect(strokes[i].peakFrame).toBeGreaterThanOrEqual(swingCenters[i] - 5)
      expect(strokes[i].peakFrame).toBeLessThanOrEqual(swingCenters[i] + 5)
    }
    for (let i = 1; i < strokes.length; i++) {
      expect(strokes[i - 1].endFrame).toBeLessThan(strokes[i].startFrame)
    }
  })

  it('ignores low-amplitude grip-adjustment fidget between two real swings', () => {
    // Two real swings + a narrow but tiny "fidget" between them. The
    // fidget IS a local maximum but its prominence is far below the
    // 0.3 floor.
    const FPS = 30
    const frames = buildWristClip(180, FPS, (i) => {
      const swing1 = 0.4 / (1 + Math.exp(-(i - 40) * 0.6))
      const swing2 = 0.4 / (1 + Math.exp(-(i - 130) * 0.6))
      // Tiny fidget bump at frame 85 (between swings).
      const fidget =
        Math.abs(i - 85) < 4
          ? 0.015 * Math.exp(-((i - 85) ** 2) / 4)
          : 0
      return { x: 0.4 + swing1 + swing2 + fidget, y: 0.55 }
    })
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(2)
  })
})

describe('detectStrokes — width filter rejects long slow drifts', () => {
  it('drops a wide low-prominence drift (player walking with arm sway)', () => {
    // Wide, gentle wrist drift — a slow ramp over 50 frames (~1.7s).
    // Position changes monotonically; speed signal has a low, broad
    // bump that exceeds the absolute prominence floor only briefly
    // and has width > widthMaxMs.
    const FPS = 30
    const frames = buildWristClip(120, FPS, (i) => {
      // Slow drift from x=0.4 to x=0.6 over 50 frames, then hold.
      if (i < 50) return { x: 0.4 + 0.004 * i, y: 0.55 }
      return { x: 0.6, y: 0.55 }
    })
    const strokes = detectStrokes(frames)
    // Either dropped by prominence floor or by width filter — both
    // are valid outcomes; either kills the false positive.
    expect(strokes.length).toBe(0)
  })
})

// ---------------------------------------------------------------------------
// verifyShape integration — kinetic-chain gate kills walking false positives
// ---------------------------------------------------------------------------

describe('detectStrokes — verifyShape kinetic-chain gate', () => {
  it('with verifyShape: true, a pure-walking clip returns ZERO strokes', () => {
    // The user's IMG_1098 failure mode: detector fires on walking
    // toward the camera. Walking has no hip rotation pattern and no
    // proximal-to-distal kinetic chain. The verifier must reject.
    const FPS = 30
    const frames: PoseFrame[] = []
    for (let i = 0; i < 150; i++) {
      // Wrist sways at stride freq with growing amplitude as the
      // player "approaches" — the realistic walking-toward-camera
      // signature.
      const stride = 0.06 * (1 + i * 0.005) * Math.sin(i * 0.5)
      const drift = 0.002 * i
      const wristX = 0.4 + drift + stride
      const landmarks = makeStandingPose().map((lm) =>
        lm.id === LANDMARK_INDICES.RIGHT_WRIST
          ? { ...lm, x: wristX, y: 0.55, visibility: 0.9 }
          : lm,
      )
      frames.push(
        makeFrame(i, (i * 1000) / FPS, landmarks, {
          // Hips and trunk barely rotate — the definitive non-swing
          // signature.
          hip_rotation: 2 * Math.sin(i * 0.4),
          trunk_rotation: 3 * Math.sin(i * 0.4),
        }),
      )
    }
    const strokes = detectStrokes(frames, { verifyShape: true })
    expect(strokes.length).toBe(0)
  })

  it('with verifyShape: true, a real swing with hip rotation pattern is preserved', () => {
    const FPS = 30
    const swingPeak = 60
    const frames: PoseFrame[] = []
    for (let i = 0; i < 120; i++) {
      const u = (i - swingPeak) * 0.5
      const wristX = 0.4 + 0.45 / (1 + Math.exp(-u))
      const hipT = Math.tanh((i - (swingPeak - 5)) * 0.25)
      const hipRot = -20 + 25 * (hipT + 1)
      const trunkT = Math.tanh((i - (swingPeak - 2)) * 0.25)
      const trunkRot = -25 + 30 * (trunkT + 1)
      const landmarks = makeStandingPose().map((lm) =>
        lm.id === LANDMARK_INDICES.RIGHT_WRIST
          ? { ...lm, x: wristX, y: 0.55, visibility: 0.9 }
          : lm,
      )
      frames.push(
        makeFrame(i, (i * 1000) / FPS, landmarks, {
          hip_rotation: hipRot,
          trunk_rotation: trunkRot,
        }),
      )
    }
    // Force right-wrist tracking. Left wrist stays at the standing-
    // pose default (vis=1) so it would otherwise win the dominant-
    // hand auto-pick over the right wrist (vis=0.9).
    const strokes = detectStrokes(frames, { verifyShape: true, dominantHand: 'right' })
    expect(strokes.length).toBe(1)
    expect(strokes[0].peakFrame).toBeGreaterThanOrEqual(swingPeak - 5)
    expect(strokes[0].peakFrame).toBeLessThanOrEqual(swingPeak + 5)
  })

  it('with verifyShape: true, a swing without hip rotation data returns zero strokes', () => {
    // Conservative-on-missing-data: when joint_angles don't carry
    // hip rotation, the verifier can't run, so it rejects rather
    // than passing through.
    const FPS = 30
    const frames = singlePeakSwing(120, FPS, 60) // no joint_angles set
    const strokes = detectStrokes(frames, { verifyShape: true })
    expect(strokes.length).toBe(0)
  })

  it('without verifyShape (default false), the same swing still returns one stroke', () => {
    // Sanity-check that the verifier is genuinely opt-in. Existing
    // tests + synthetic fixtures (which don't set joint_angles)
    // continue to behave the way they did before this change.
    const FPS = 30
    const frames = singlePeakSwing(120, FPS, 60)
    const strokes = detectStrokes(frames)
    expect(strokes.length).toBe(1)
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
