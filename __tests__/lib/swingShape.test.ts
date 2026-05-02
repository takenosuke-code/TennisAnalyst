/**
 * Unit tests for the swing-shape verifier. Each scenario constructs a
 * synthetic frame stream that targets one specific failure mode of
 * pure wrist-speed peak-pick — walking, fidgeting, in-phase
 * oscillation — and asserts the verifier rejects the candidate.
 * A real-swing fixture confirms the verifier accepts genuine kinetic-
 * chain patterns.
 */
import { describe, it, expect } from 'vitest'
import { verifySwingShape } from '@/lib/swingShape'
import type { JointAngles, PoseFrame } from '@/lib/supabase'

function makeFrame(
  index: number,
  timestampMs: number,
  jointAngles: JointAngles,
): PoseFrame {
  return {
    frame_index: index,
    timestamp_ms: timestampMs,
    landmarks: [],
    joint_angles: jointAngles,
  }
}

/**
 * Build a clip whose hip and trunk rotations follow callbacks. Uses a
 * 30fps timestamp grid and `numFrames` total frames.
 */
function buildAngleClip(
  numFrames: number,
  fps: number,
  hip: (i: number) => number,
  trunk: (i: number) => number,
): PoseFrame[] {
  const dt = 1000 / fps
  const out: PoseFrame[] = []
  for (let i = 0; i < numFrames; i++) {
    out.push(
      makeFrame(i, i * dt, {
        hip_rotation: hip(i),
        trunk_rotation: trunk(i),
      }),
    )
  }
  return out
}

describe('verifySwingShape', () => {
  it('accepts a real-swing fixture: hip rotates 40°, peaks before wrist', () => {
    // Hip rotation ramps from -20° at frame 50 up to +20° at frame 65,
    // peaking in angular velocity at ~frame 57. Wrist peak at frame 60.
    // Trunk follows hip with a small lag.
    const FPS = 30
    const wristPeak = 60
    const frames = buildAngleClip(
      120,
      FPS,
      (i) => {
        if (i < 50) return -20
        if (i > 65) return 20
        const t = (i - 50) / 15
        return -20 + 40 * t
      },
      (i) => {
        if (i < 53) return -25
        if (i > 68) return 25
        const t = (i - 53) / 15
        return -25 + 50 * t
      },
    )
    const result = verifySwingShape(frames, wristPeak, FPS)
    expect(
      result.passed,
      `verifier rejected real swing: reason=${result.reason}, metrics=${JSON.stringify(result.metrics)}`,
    ).toBe(true)
    expect(result.metrics.hipExcursion).toBeGreaterThanOrEqual(35)
    expect(result.metrics.trunkExcursion).toBeGreaterThanOrEqual(40)
    expect(result.metrics.hipLeadMs).toBeGreaterThanOrEqual(25)
    expect(result.metrics.hipLeadMs).toBeLessThanOrEqual(400)
  })

  it('rejects walking: hip oscillates ±2° (below excursion floor)', () => {
    // Walking forward — hips translate but barely rotate. Tiny ±2°
    // sinusoid at stride frequency. Real swing's wrist-peak landmark
    // is positioned at frame 60.
    const FPS = 30
    const frames = buildAngleClip(
      120,
      FPS,
      (i) => 2 * Math.sin(i * 0.4),
      (i) => 3 * Math.sin(i * 0.4),
    )
    const result = verifySwingShape(frames, 60, FPS)
    expect(result.passed).toBe(false)
    expect(result.reason).toBe('shallow_hip_excursion')
    expect(result.metrics.hipExcursion).toBeLessThan(8)
  })

  it('rejects walking with hip wiggle but no kinetic chain order', () => {
    // Hip rotation reaches a healthy excursion but its angular-
    // velocity peak is AFTER the wrist peak — no proximal-to-distal
    // ordering. This pattern shows up when the user picks up a ball
    // or rotates after impact.
    const FPS = 30
    const wristPeak = 60
    const frames = buildAngleClip(
      120,
      FPS,
      (i) => {
        // Hip ramp delayed: starts moving at frame 65 (post-wrist).
        if (i < 65) return -10
        if (i > 80) return 30
        return -10 + 40 * ((i - 65) / 15)
      },
      (i) => {
        if (i < 65) return -10
        if (i > 80) return 30
        return -10 + 40 * ((i - 65) / 15)
      },
    )
    const result = verifySwingShape(frames, wristPeak, FPS)
    expect(result.passed).toBe(false)
    expect(result.reason).toBe('no_kinetic_chain')
  })

  it('rejects when hip_rotation is missing from joint_angles', () => {
    const FPS = 30
    const frames: PoseFrame[] = []
    for (let i = 0; i < 120; i++) {
      frames.push(makeFrame(i, i * 33, { trunk_rotation: 30 }))
    }
    const result = verifySwingShape(frames, 60, FPS)
    expect(result.passed).toBe(false)
    expect(result.reason).toBe('no_hip_rotation')
  })

  it('rejects when window has too few frames around the peak', () => {
    // Peak at frame 1 with a 2-frame clip — not enough room for the
    // kinetic-chain check.
    const FPS = 30
    const frames = buildAngleClip(
      3,
      FPS,
      () => 0,
      () => 0,
    )
    const result = verifySwingShape(frames, 1, FPS)
    expect(result.passed).toBe(false)
    expect(result.reason).toBe('no_data')
  })

  it('rejects when trunk excursion is shallow (player only rotates hips, not shoulders)', () => {
    const FPS = 30
    const frames = buildAngleClip(
      120,
      FPS,
      (i) => {
        if (i < 50) return -20
        if (i > 65) return 20
        return -20 + 40 * ((i - 50) / 15)
      },
      () => 5, // trunk static
    )
    const result = verifySwingShape(frames, 60, FPS)
    expect(result.passed).toBe(false)
    expect(result.reason).toBe('shallow_trunk_excursion')
  })

  it('passes even when trunk_rotation is absent (hip-only verification)', () => {
    // Some pipelines/clips may have hip but not trunk rotation. The
    // verifier should still run on hip alone — gate falls through
    // when trunk data is too sparse.
    const FPS = 30
    const wristPeak = 60
    const frames: PoseFrame[] = []
    for (let i = 0; i < 120; i++) {
      let hip = -20
      if (i >= 50 && i <= 65) hip = -20 + 40 * ((i - 50) / 15)
      else if (i > 65) hip = 20
      frames.push(makeFrame(i, i * 33, { hip_rotation: hip }))
    }
    const result = verifySwingShape(frames, wristPeak, FPS)
    expect(
      result.passed,
      `expected pass (hip-only): reason=${result.reason}`,
    ).toBe(true)
    expect(result.metrics.trunkExcursion).toBeNull()
  })
})
