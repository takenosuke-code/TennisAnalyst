import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { pingModalWarmup, startWarmupHeartbeat } from '@/lib/liveModalWarmup'

const WARMUP_BLOB_URL = 'https://blob.vercel-storage.com/warmup-asset.mp4'

describe('pingModalWarmup', () => {
  beforeEach(() => {
    // The module logs at console.info on every failure path. Silence
    // those logs so the test output stays clean — but assert via spies
    // that the right path was taken.
    vi.spyOn(console, 'info').mockImplementation(() => {})
    // Most tests want the env var set; the "unset" test re-stubs to ''.
    vi.stubEnv('NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL', WARMUP_BLOB_URL)
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
    vi.restoreAllMocks()
  })

  it('POSTs the configured blob URL to /api/extract on success', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('{"status":"queued"}', {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(pingModalWarmup()).resolves.toBeUndefined()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/extract')
    const init = fetchMock.mock.calls[0]?.[1]
    expect(init?.method).toBe('POST')
    const body = JSON.parse(init?.body as string)
    expect(body.blobUrl).toBe(WARMUP_BLOB_URL)
    // sessionId must be a synthetic warmup id so Railway logs are
    // grep-able and never collide with a real session row.
    expect(body.sessionId).toMatch(/^warmup-/)
  })

  it('is a no-op when NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL is unset', async () => {
    vi.stubEnv('NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL', '')
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    await expect(pingModalWarmup()).resolves.toBeUndefined()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('resolves (does not reject) on a 404 — endpoint not deployed yet', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('not found', { status: 404 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    // Awaiting must not throw. The whole point of the warmup contract
    // is that the caller is fire-and-forget.
    await expect(pingModalWarmup()).resolves.toBeUndefined()
  })

  it('resolves (does not reject) on a network error', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('network broken'))
    vi.stubGlobal('fetch', fetchMock)

    await expect(pingModalWarmup()).resolves.toBeUndefined()
  })
})

describe('startWarmupHeartbeat', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(console, 'info').mockImplementation(() => {})
    vi.stubEnv('NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL', WARMUP_BLOB_URL)
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
    vi.restoreAllMocks()
  })

  it('returns a stop function that cancels future intervals', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('{}', { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const stop = startWarmupHeartbeat(1000)
    // The first ping fires immediately (synchronously kicked off by
    // startWarmupHeartbeat). Flush microtasks so the fetch promise
    // resolves before we count.
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    // Two more intervals — should be 3 total pings now.
    await vi.advanceTimersByTimeAsync(2000)
    expect(fetchMock).toHaveBeenCalledTimes(3)

    // Stop the heartbeat. No further pings should fire.
    stop()
    const countAtStop = fetchMock.mock.calls.length
    await vi.advanceTimersByTimeAsync(5000)
    expect(fetchMock).toHaveBeenCalledTimes(countAtStop)
  })

  it('uses a 25s default interval when none is passed', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('{}', { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const stop = startWarmupHeartbeat()
    // Initial fire.
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    // Just under 25s — no second ping.
    await vi.advanceTimersByTimeAsync(24_999)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    // Cross 25s — second ping.
    await vi.advanceTimersByTimeAsync(2)
    expect(fetchMock).toHaveBeenCalledTimes(2)

    stop()
  })
})
