import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { extractPoseViaRailway, RailwayExtractError } from '@/lib/poseExtractionRailway'

// All the client-side orchestration: kick off /api/extract, poll
// /api/sessions/[id] until status=='complete', then return the keypoints
// in ExtractResult shape. On any failure, throw RailwayExtractError with
// a categorical `reason` so the caller (UploadZone) knows whether to
// fall back, retry, or surface an error.

const FRAMES = [
  {
    frame_index: 0,
    timestamp_ms: 0,
    landmarks: [],
    joint_angles: {},
  },
]

function mockFetchSequence(responses: Array<{ status: number; body: unknown }>) {
  const fn = vi.fn()
  for (const r of responses) {
    fn.mockResolvedValueOnce(
      new Response(JSON.stringify(r.body), {
        status: r.status,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
  }
  vi.stubGlobal('fetch', fn)
  return fn
}

describe('extractPoseViaRailway', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('resolves with frames when polling sees status=complete', async () => {
    const fetchMock = mockFetchSequence([
      { status: 200, body: { status: 'queued' } },          // /api/extract
      { status: 200, body: { status: 'pending' } },         // poll 1
      {
        status: 200,
        body: {
          status: 'complete',
          keypoints_json: { frames: FRAMES, fps_sampled: 30 },
        },
      },                                                    // poll 2
    ])
    const onProgress = vi.fn()
    const p = extractPoseViaRailway({
      sessionId: 's1',
      blobUrl: 'https://blob/x.mp4',
      onProgress,
    })
    await vi.runAllTimersAsync()
    const result = await p
    expect(result.frames).toEqual(FRAMES)
    expect(result.fps).toBe(30)
    expect(onProgress).toHaveBeenCalledWith(100)
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('throws not-configured when /api/extract returns 503', async () => {
    mockFetchSequence([
      { status: 503, body: { error: 'railway-not-configured' } },
    ])
    try {
      await extractPoseViaRailway({ sessionId: 's1', blobUrl: 'https://blob/x.mp4' })
      throw new Error('should have thrown')
    } catch (err) {
      expect(err).toBeInstanceOf(RailwayExtractError)
      expect((err as RailwayExtractError).reason).toBe('not-configured')
    }
  })

  it('throws queue-failed when /api/extract returns non-2xx other than 503', async () => {
    mockFetchSequence([
      { status: 502, body: { error: 'railway-error', status: 500 } },
    ])
    await expect(
      extractPoseViaRailway({ sessionId: 's1', blobUrl: 'https://blob/x.mp4' }),
    ).rejects.toMatchObject({ reason: 'queue-failed' })
  })

  it('throws error-status when poll sees status=error', async () => {
    mockFetchSequence([
      { status: 200, body: { status: 'queued' } },
      {
        status: 200,
        body: { status: 'error', error_message: 'Extraction failed' },
      },
    ])
    const p = extractPoseViaRailway({
      sessionId: 's1',
      blobUrl: 'https://blob/x.mp4',
    })
    // Attach the rejection handler BEFORE draining timers so the
    // promise rejection doesn't surface as an unhandled error.
    const assertion = expect(p).rejects.toMatchObject({ reason: 'error-status' })
    await vi.runAllTimersAsync()
    await assertion
  })

  it('throws aborted when AbortSignal is triggered before start', async () => {
    const controller = new AbortController()
    controller.abort()
    await expect(
      extractPoseViaRailway({
        sessionId: 's1',
        blobUrl: 'https://blob/x.mp4',
        abortSignal: controller.signal,
      }),
    ).rejects.toMatchObject({ reason: 'aborted' })
  })

  it('emits onProgress 25 after queue-accept and 100 on completion', async () => {
    mockFetchSequence([
      { status: 200, body: { status: 'queued' } },
      {
        status: 200,
        body: {
          status: 'complete',
          keypoints_json: { frames: FRAMES, fps_sampled: 30 },
        },
      },
    ])
    const onProgress = vi.fn()
    const p = extractPoseViaRailway({
      sessionId: 's1',
      blobUrl: 'https://blob/x.mp4',
      onProgress,
    })
    await vi.runAllTimersAsync()
    await p
    const pcts = onProgress.mock.calls.map((c) => c[0])
    expect(pcts).toContain(25)
    expect(pcts[pcts.length - 1]).toBe(100)
  })
})
