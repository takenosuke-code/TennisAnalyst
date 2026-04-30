/**
 * Phase 3 — Visibility worker tests for useLiveCoach.isRequestInFlight().
 *
 * Worker 3 owns the batching logic; Worker 5 owns awaitInFlight. This
 * file covers ONLY the `isRequestInFlight()` getter, which Worker 4
 * adds to drive the transcript-header pulse.
 *
 * Contract guarded:
 *   1. Returns false at idle (no pending swings, no batch fired).
 *   2. Returns true the moment a batch fires (i.e. while the fetch is
 *      pending).
 *   3. Returns false the moment the batch settles, regardless of
 *      whether it succeeded or failed.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'

// Avoid pulling in a real speech-synthesis layer.
vi.mock('@/lib/liveTts', () => ({
  createBrowserSynth: () => null,
  LiveTtsQueue: class {
    isAvailable() {
      return false
    }
    enqueue() {}
    mute() {}
    unmute() {}
    prime() {}
    reset() {}
  },
}))

// jointAngles.buildAngleSummary is pure — keep it real but stub the
// payload so the test isn't dependent on its formatting.
vi.mock('@/lib/jointAngles', async (orig) => {
  const actual = (await orig()) as Record<string, unknown>
  return {
    ...actual,
    buildAngleSummary: () => 'fake-angle-summary',
  }
})

import { useLiveCoach } from '@/hooks/useLiveCoach'
import type { StreamedSwing } from '@/lib/liveSwingDetector'
import type { PoseFrame } from '@/lib/supabase'

// Minimum-shape frame helper — buildAngleSummary is mocked, so we only need
// frames.length > 0 for the swing to make it past the empty-frame skip
// guard. The on-device fallback path produces frames whose timestamp_ms is
// session-relative (non-zero except for the literal first frame); the
// server path re-anchors timestamp_ms to 0. Phase E uses that as the
// source-detection heuristic in useLiveCoach.
function makeFrame(timestampMs: number, frameIndex = 0): PoseFrame {
  return {
    frame_index: frameIndex,
    timestamp_ms: timestampMs,
    landmarks: [],
    joint_angles: {},
  }
}

// Default to the on-device fallback shape (non-zero timestamp_ms).
// Tests that need server-shaped swings call fakeSwing(i, { source: 'server' }).
function fakeSwing(
  index: number,
  opts: { source?: 'server' | 'fallback'; frames?: PoseFrame[] } = {},
): StreamedSwing {
  const source = opts.source ?? 'fallback'
  const frames =
    opts.frames ??
    (source === 'server'
      ? [makeFrame(0, 0), makeFrame(33, 1)]
      : [makeFrame(index * 1000 + 100), makeFrame(index * 1000 + 133)])
  return {
    swingIndex: index,
    startFrameIndex: index * 10,
    endFrameIndex: index * 10 + 5,
    peakFrameIndex: index * 10 + 3,
    startMs: index * 1000,
    endMs: index * 1000 + 500,
    frames,
  }
}

// Drives the fetch mock with control over when the response resolves
// or rejects. Lets us assert isRequestInFlight() at three distinct
// points: before the batch fires, during the in-flight window, and
// after settle.
function controllableFetch() {
  const calls: { resolve: (v: Response) => void; reject: (e: unknown) => void }[] = []
  const fn = vi.fn(() => {
    return new Promise<Response>((resolve, reject) => {
      calls.push({ resolve, reject })
    })
  })
  return { fn, calls }
}

describe('useLiveCoach.isRequestInFlight()', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    globalThis.fetch = originalFetch
  })

  it('returns false when no batch has fired and no swings are pending', () => {
    const { result } = renderHook(() => useLiveCoach())
    expect(result.current.isRequestInFlight()).toBe(false)
  })

  it('flips to true the moment a batch fires, and back to false when the response settles', async () => {
    const { fn, calls } = controllableFetch()
    globalThis.fetch = fn as unknown as typeof globalThis.fetch

    const { result } = renderHook(() =>
      useLiveCoach({ maxSwingsPerBatch: 2, minBatchIntervalMs: 0 }),
    )

    // Idle.
    expect(result.current.isRequestInFlight()).toBe(false)

    // Push two swings — that's the batch threshold. Phase 2 added a
    // post-swing grace window (default 3s since last swing) to avoid
    // firing mid-rally, so the batch is scheduled, not fired immediately.
    act(() => {
      result.current.pushSwing(fakeSwing(0))
      result.current.pushSwing(fakeSwing(1))
    })

    // Advance past the post-swing grace window so the scheduled fire
    // attempt actually issues the fetch.
    await act(async () => {
      vi.advanceTimersByTime(3_500)
      await Promise.resolve()
      await Promise.resolve()
    })

    // The fetch promise is held open by our controllable mock. While
    // it's pending, isRequestInFlight() must be true.
    expect(fn).toHaveBeenCalledTimes(1)
    expect(result.current.isRequestInFlight()).toBe(true)

    // Now settle the response. Use a successful Response so the
    // success-path code (transcript append + TTS enqueue) runs.
    await act(async () => {
      calls[0].resolve(
        new Response('Stay loose.', {
          status: 200,
          headers: { 'X-Analysis-Event-Id': 'evt-1' },
        }),
      )
      // Drain microtasks so the await chain completes.
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(result.current.isRequestInFlight()).toBe(false)
  })

  it('returns false after the batch errors out (network failure path)', async () => {
    // The hook retries once with a 2s backoff after a fetch failure
    // before giving up. We need to drive the timer through that.
    const fn = vi.fn(() => Promise.reject(new Error('network down')))
    globalThis.fetch = fn as unknown as typeof globalThis.fetch

    const { result } = renderHook(() =>
      useLiveCoach({ maxSwingsPerBatch: 2, minBatchIntervalMs: 0 }),
    )

    act(() => {
      result.current.pushSwing(fakeSwing(0))
      result.current.pushSwing(fakeSwing(1))
    })

    // Phase 2 added a post-swing grace window (default 3s). Advance past
    // it so the first fetch is actually issued, then let it reject and
    // advance through the 2s retry-backoff so the retry also rejects.
    await act(async () => {
      vi.advanceTimersByTime(3_500)
      await Promise.resolve()
      await Promise.resolve()
      vi.advanceTimersByTime(2_000)
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    // After both attempts have rejected, the hook must clear the
    // in-flight flag so the UI doesn't stay stuck on a stale pulse.
    expect(fn).toHaveBeenCalledTimes(2)
    expect(result.current.isRequestInFlight()).toBe(false)
  })
})

/**
 * Phase E — server-keypoint sourcing for angleSummary + sourceMix telemetry.
 *
 * Contract guarded:
 *   1. When swing.frames is populated, useLiveCoach derives angleSummary
 *      from those frames (via the mocked buildAngleSummary) and forwards
 *      the result to /api/live-coach.
 *   2. When ONE swing in a batch has empty frames, that swing is skipped
 *      but the batch still fires for the others (with the survivors).
 *   3. When ALL swings in a batch have empty frames, no fetch is made.
 *   4. The request body's `sourceMix` field correctly counts server vs
 *      fallback swings using the timestamp_ms === 0 heuristic.
 */
describe('useLiveCoach Phase E sourcing + sourceMix', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    globalThis.fetch = originalFetch
  })

  // Helper to drive a batch through to a settled fetch and return the
  // parsed body of the request that was made.
  async function fireAndCapture(
    swings: StreamedSwing[],
    opts: { maxSwingsPerBatch?: number } = {},
  ) {
    const fn = vi.fn<(input: string, init: RequestInit) => Promise<Response>>(
      () => Promise.resolve(new Response('cue', { status: 200 })),
    )
    globalThis.fetch = fn as unknown as typeof globalThis.fetch

    const { result } = renderHook(() =>
      useLiveCoach({
        maxSwingsPerBatch: opts.maxSwingsPerBatch ?? swings.length,
        minBatchIntervalMs: 0,
      }),
    )

    act(() => {
      for (const s of swings) result.current.pushSwing(s)
    })
    // Drain the post-swing grace window + any retries.
    await act(async () => {
      vi.advanceTimersByTime(3_500)
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    return { fn }
  }

  it('forwards angleSummary derived from swing.frames to /api/live-coach', async () => {
    const { fn } = await fireAndCapture([fakeSwing(0), fakeSwing(1)])
    expect(fn).toHaveBeenCalledTimes(1)
    const [, init] = fn.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(init.body as string)
    expect(Array.isArray(body.recentSwings)).toBe(true)
    expect(body.recentSwings.length).toBe(2)
    // buildAngleSummary is mocked to return 'fake-angle-summary' regardless
    // of frame content, but it's only invoked at all when frames.length>0.
    for (const s of body.recentSwings) {
      expect(s.angleSummary).toBe('fake-angle-summary')
    }
  })

  it('skips a swing whose frames are empty but still fires the batch for the rest', async () => {
    const filled = fakeSwing(0)
    const empty: StreamedSwing = { ...fakeSwing(1), frames: [] }
    const filled2 = fakeSwing(2)
    // maxSwingsPerBatch = 2 so a batch fires after 2 KEPT swings (not 2
    // pushed swings) — confirms the empty one was dropped, not just held.
    const { fn } = await fireAndCapture([filled, empty, filled2], {
      maxSwingsPerBatch: 2,
    })
    expect(fn).toHaveBeenCalledTimes(1)
    const [, init] = fn.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(init.body as string)
    expect(body.recentSwings.length).toBe(2)
    // The two surviving swings should be filled (index 0) and filled2
    // (index 2), in that order.
    expect(body.recentSwings[0].startMs).toBe(0)
    expect(body.recentSwings[1].startMs).toBe(2000)
  })

  it('does not fire the batch if every swing in it has empty frames', async () => {
    const fn = vi.fn<(input: string, init: RequestInit) => Promise<Response>>(
      () => Promise.resolve(new Response('cue', { status: 200 })),
    )
    globalThis.fetch = fn as unknown as typeof globalThis.fetch

    const { result } = renderHook(() =>
      useLiveCoach({ maxSwingsPerBatch: 2, minBatchIntervalMs: 0 }),
    )

    act(() => {
      result.current.pushSwing({ ...fakeSwing(0), frames: [] })
      result.current.pushSwing({ ...fakeSwing(1), frames: [] })
    })
    // Drain through the would-be grace window AND past idleTimeoutMs to
    // confirm no idle-fire path catches the empty queue either.
    await act(async () => {
      vi.advanceTimersByTime(15_000)
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(fn).not.toHaveBeenCalled()
  })

  it('counts sourceMix using the timestamp_ms === 0 heuristic', async () => {
    // Two server-shaped swings (frames[0].timestamp_ms === 0) and one
    // fallback-shaped swing (non-zero timestamp_ms).
    const server1 = fakeSwing(0, { source: 'server' })
    const server2 = fakeSwing(1, { source: 'server' })
    const fallback = fakeSwing(2, { source: 'fallback' })

    const { fn } = await fireAndCapture([server1, server2, fallback])

    expect(fn).toHaveBeenCalledTimes(1)
    const [, init] = fn.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(init.body as string)
    expect(body.sourceMix).toEqual({ server: 2, fallback: 1 })
  })

  it('sourceMix reads { server: 0, fallback: N } when every swing is on-device', async () => {
    const a = fakeSwing(0, { source: 'fallback' })
    const b = fakeSwing(1, { source: 'fallback' })
    const { fn } = await fireAndCapture([a, b])
    const [, init] = fn.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(init.body as string)
    expect(body.sourceMix).toEqual({ server: 0, fallback: 2 })
  })

  it('does not leak the per-swing `source` field into the wire payload', async () => {
    const { fn } = await fireAndCapture([
      fakeSwing(0, { source: 'server' }),
      fakeSwing(1, { source: 'fallback' }),
    ])
    const [, init] = fn.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(init.body as string)
    for (const s of body.recentSwings) {
      // The wire shape stays { angleSummary, startMs, endMs }; sourceMix
      // is the only new field on the batch envelope.
      expect(Object.prototype.hasOwnProperty.call(s, 'source')).toBe(false)
    }
  })
})
