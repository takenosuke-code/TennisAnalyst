import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// Mock @vercel/blob/client BEFORE importing the module under test so
// the upload() shim is in place when the module's import binding is
// resolved. Pattern matches __tests__/components/LiveCapturePanel.test.tsx.
const uploadMock = vi.fn()
vi.mock('@vercel/blob/client', () => ({
  upload: (...args: unknown[]) => uploadMock(...args),
}))

import {
  extractBatchedSwingKeypoints,
  sliceToSubClip,
  splitFramesPerSwing,
  type BatchExtractSwingInput,
} from '@/lib/liveSwingBatchExtractor'
import type { PoseFrame } from '@/lib/supabase'

// Shared frame factory — keeps the test fixtures honest about the
// PoseFrame shape (frame_index + timestamp_ms + landmarks + joint_angles).
function makeFrame(timestamp_ms: number, frame_index: number = 0): PoseFrame {
  return {
    frame_index,
    timestamp_ms,
    landmarks: [],
    joint_angles: {},
  }
}

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

describe('sliceToSubClip', () => {
  it('concatenates the chunks intersecting each swing window', async () => {
    // 5 chunks of distinct sizes, 1s apart, anchored at media-clock 0.
    // We use Blob.size to verify which chunks landed in the spliced
    // output. Distinct sizes means the *sum* uniquely identifies the
    // included subset — there's no ambiguity about which chunks
    // contributed even without inspecting bytes.
    //
    // Note: jsdom's Blob serializes nested Blob inputs via toString
    // ("[object Blob]") rather than copying their bytes. We work
    // around this by passing ArrayBuffer-backed Uint8Arrays *directly*
    // to the spliced Blob (production code passes Blob[]; we need to
    // diverge slightly in the test's assertion to bypass jsdom's bug).
    // Instead of inspecting bytes, we rely on perSwingOffsets'
    // determinism + a separate "drops out-of-range chunks" assertion.
    const chunks = [
      new Blob([new Uint8Array(1)], { type: 'video/mp4' }), // chunk 0, size 1
      new Blob([new Uint8Array(2)], { type: 'video/mp4' }), // chunk 1, size 2
      new Blob([new Uint8Array(4)], { type: 'video/mp4' }), // chunk 2, size 4
      new Blob([new Uint8Array(8)], { type: 'video/mp4' }), // chunk 3, size 8
      new Blob([new Uint8Array(16)], { type: 'video/mp4' }), // chunk 4, size 16
    ]
    const swings: BatchExtractSwingInput[] = [
      // Swing 0: covers chunk 0 (0–1000ms).
      { swingIndex: 0, startMs: 100, endMs: 900 },
      // Swing 1: covers chunks 2 and 3 (2000–4000ms).
      { swingIndex: 1, startMs: 2100, endMs: 3900 },
      // Swing 2: covers chunk 4 (4000–5000ms).
      { swingIndex: 2, startMs: 4100, endMs: 4900 },
    ]
    const result = sliceToSubClip(chunks, 'video/mp4', swings, 0)

    // The included chunks should be {0, 2, 3, 4} → sizes 1 + 4 + 8 + 16
    // = 29. Chunk 1 (size 2) lives in [1000, 2000)ms — outside every
    // swing — and must be excluded.
    //
    // jsdom serializes nested Blobs via toString → "[object Blob]"
    // (13 bytes) per nested Blob. Either way, exactly four chunks
    // contribute, and the size collapses to one of two known values:
    //   - real browser: 1 + 4 + 8 + 16 = 29
    //   - jsdom:        4 * 13       = 52
    // If chunk 1 had been included, both totals would shift (to 31 or
    // 5*13=65), so this assertion still distinguishes correct vs wrong.
    expect([29, 4 * '[object Blob]'.length]).toContain(result.blob.size)

    expect(result.perSwingOffsets).toHaveLength(3)
    // Swing 0 occupies the first 800ms (900 - 100) of the spliced clip.
    expect(result.perSwingOffsets[0]).toMatchObject({
      swingIndex: 0,
      spliceStartMs: 0,
      spliceEndMs: 800,
      originalStartMs: 100,
    })
    // Swing 1 starts where swing 0 ended; spans 1800ms (3900 - 2100).
    expect(result.perSwingOffsets[1]).toMatchObject({
      swingIndex: 1,
      spliceStartMs: 800,
      spliceEndMs: 800 + 1800,
      originalStartMs: 2100,
    })
    // Swing 2 starts after swing 1; spans 800ms.
    expect(result.perSwingOffsets[2]).toMatchObject({
      swingIndex: 2,
      spliceStartMs: 2600,
      spliceEndMs: 3400,
      originalStartMs: 4100,
    })
  })

  it('handles empty chunks / empty swings gracefully', () => {
    const a = sliceToSubClip([], 'video/mp4', [{ swingIndex: 0, startMs: 0, endMs: 100 }], 0)
    expect(a.blob.size).toBe(0)
    expect(a.perSwingOffsets).toHaveLength(0)

    const b = sliceToSubClip(
      [new Blob([new Uint8Array([1])], { type: 'video/mp4' })],
      'video/mp4',
      [],
      0,
    )
    expect(b.blob.size).toBe(0)
    expect(b.perSwingOffsets).toHaveLength(0)
  })
})

describe('splitFramesPerSwing', () => {
  it('splits a flat 100-frame array spanning splice boundaries and re-anchors timestamps', () => {
    // 100 frames evenly spaced 10ms apart, 0..990ms — a flat output as
    // Modal would return for a spliced sub-clip. We slice it into three
    // swings whose splice windows together span the full clip.
    const flat: PoseFrame[] = Array.from({ length: 100 }, (_, i) => makeFrame(i * 10, i))

    const offsets = [
      { swingIndex: 0, spliceStartMs: 0, spliceEndMs: 300, originalStartMs: 1000 },
      { swingIndex: 1, spliceStartMs: 300, spliceEndMs: 700, originalStartMs: 5000 },
      { swingIndex: 2, spliceStartMs: 700, spliceEndMs: 1000, originalStartMs: 9000 },
    ]
    const result = splitFramesPerSwing(flat, offsets)

    expect(result).toHaveLength(3)

    // Swing 0: [0, 300) — frames 0..29 (timestamps 0..290ms).
    expect(result[0]?.frames).toHaveLength(30)
    expect(result[0]?.frames[0]?.timestamp_ms).toBe(0)
    expect(result[0]?.frames[0]?.frame_index).toBe(0)
    expect(result[0]?.frames[29]?.timestamp_ms).toBe(290)
    expect(result[0]?.frames[29]?.frame_index).toBe(29)
    expect(result[0]?.failureReason).toBeNull()

    // Swing 1: [300, 700) — frames 30..69 in the flat array, re-anchored
    // to start at timestamp_ms = 0 / frame_index = 0.
    expect(result[1]?.frames).toHaveLength(40)
    expect(result[1]?.frames[0]?.timestamp_ms).toBe(0)
    expect(result[1]?.frames[0]?.frame_index).toBe(0)
    expect(result[1]?.frames[39]?.timestamp_ms).toBe(390)
    expect(result[1]?.frames[39]?.frame_index).toBe(39)
    expect(result[1]?.failureReason).toBeNull()

    // Swing 2: [700, 1000) — frames 70..99, re-anchored.
    expect(result[2]?.frames).toHaveLength(30)
    expect(result[2]?.frames[0]?.timestamp_ms).toBe(0)
    expect(result[2]?.frames[0]?.frame_index).toBe(0)
    expect(result[2]?.frames[29]?.timestamp_ms).toBe(290)
    expect(result[2]?.failureReason).toBeNull()
  })

  it('marks a swing as failed when its range yields zero frames', () => {
    const flat: PoseFrame[] = [makeFrame(50), makeFrame(100), makeFrame(150)]
    const offsets = [
      { swingIndex: 0, spliceStartMs: 0, spliceEndMs: 200, originalStartMs: 0 },
      // No frames intersect [500, 800) — this swing should fail.
      { swingIndex: 1, spliceStartMs: 500, spliceEndMs: 800, originalStartMs: 1000 },
    ]
    const result = splitFramesPerSwing(flat, offsets)
    expect(result[0]?.failureReason).toBeNull()
    expect(result[0]?.frames).toHaveLength(3)
    expect(result[1]?.failureReason).toBe('no-frames-in-range')
    expect(result[1]?.frames).toEqual([])
  })
})

describe('extractBatchedSwingKeypoints', () => {
  beforeEach(() => {
    uploadMock.mockReset()
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('uploads, calls /api/extract, splits frames per swing, and re-anchors timestamps', async () => {
    uploadMock.mockResolvedValue({ url: 'https://blob.example/spliced.mp4' })

    // Modal returns a flat array spanning the whole spliced clip. Two
    // swings, each 1000ms wide; together that's a 2000ms spliced clip.
    // 20 frames, 100ms apart — first 10 land in swing 0, last 10 in swing 1.
    const flat = Array.from({ length: 20 }, (_, i) => makeFrame(i * 100, i))
    mockFetchSequence([
      { status: 200, body: { frames: flat } },
    ])

    const chunks = [
      new Blob([new Uint8Array([1])], { type: 'video/mp4' }),
      new Blob([new Uint8Array([2])], { type: 'video/mp4' }),
    ]
    const onProgress = vi.fn()
    const result = await extractBatchedSwingKeypoints(
      {
        swings: [
          { swingIndex: 0, startMs: 0, endMs: 1000 },
          { swingIndex: 1, startMs: 1000, endMs: 2000 },
        ],
        blobChunks: chunks,
        blobMimeType: 'video/mp4',
        blobStartMs: 0,
      },
      { onProgress },
    )

    expect(uploadMock).toHaveBeenCalledTimes(1)
    expect(uploadMock.mock.calls[0]?.[0]).toMatch(/^live\/swings\/\d+-batch\.mp4$/)

    expect(result.perSwing).toHaveLength(2)
    // Swing 0 keeps the first 10 frames, re-anchored.
    expect(result.perSwing[0]?.frames).toHaveLength(10)
    expect(result.perSwing[0]?.frames[0]?.timestamp_ms).toBe(0)
    expect(result.perSwing[0]?.frames[0]?.frame_index).toBe(0)
    expect(result.perSwing[0]?.failureReason).toBeNull()
    // Swing 1 keeps the last 10 frames, re-anchored to start at 0.
    expect(result.perSwing[1]?.frames).toHaveLength(10)
    expect(result.perSwing[1]?.frames[0]?.timestamp_ms).toBe(0)
    expect(result.perSwing[1]?.frames[9]?.timestamp_ms).toBe(900)
    expect(result.perSwing[1]?.failureReason).toBeNull()

    expect(onProgress).toHaveBeenCalledWith(100)
  })

  it('reports per-swing failure when one swing has zero frames in its range', async () => {
    uploadMock.mockResolvedValue({ url: 'https://blob.example/spliced.mp4' })

    // Two swings of 1000ms each — Modal returns frames only in the
    // first half (timestamps 0..900ms). Swing 1 (range [1000, 2000))
    // gets no frames and must be flagged.
    const flat = Array.from({ length: 10 }, (_, i) => makeFrame(i * 100, i))
    mockFetchSequence([
      { status: 200, body: { frames: flat } },
    ])

    const result = await extractBatchedSwingKeypoints({
      swings: [
        { swingIndex: 0, startMs: 0, endMs: 1000 },
        { swingIndex: 1, startMs: 1000, endMs: 2000 },
      ],
      blobChunks: [new Blob([new Uint8Array([1])], { type: 'video/mp4' })],
      blobMimeType: 'video/mp4',
      blobStartMs: 0,
    })

    expect(result.perSwing[0]?.failureReason).toBeNull()
    expect(result.perSwing[0]?.frames).toHaveLength(10)
    // Swing 1 receives no frames in range and is flagged.
    expect(result.perSwing[1]?.failureReason).toBe('no-frames-in-range')
    expect(result.perSwing[1]?.frames).toEqual([])
  })

  it('returns batch-wide failure when the upload step fails — other swings still surface a reason', async () => {
    uploadMock.mockRejectedValue(new Error('blob upload exploded'))

    const result = await extractBatchedSwingKeypoints({
      swings: [
        { swingIndex: 0, startMs: 0, endMs: 1000 },
        { swingIndex: 1, startMs: 1000, endMs: 2000 },
      ],
      blobChunks: [new Blob([new Uint8Array([1])], { type: 'video/mp4' })],
      blobMimeType: 'video/mp4',
      blobStartMs: 0,
    })
    expect(result.perSwing).toHaveLength(2)
    for (const s of result.perSwing) {
      expect(s.frames).toEqual([])
      expect(s.failureReason).toMatch(/^upload-failed/)
    }
  })

  it('returns batch-wide failure when /api/extract returns 503 (Modal not configured)', async () => {
    uploadMock.mockResolvedValue({ url: 'https://blob.example/spliced.mp4' })
    mockFetchSequence([
      { status: 503, body: { error: 'modal-not-configured' } },
    ])
    const result = await extractBatchedSwingKeypoints({
      swings: [{ swingIndex: 0, startMs: 0, endMs: 1000 }],
      blobChunks: [new Blob([new Uint8Array([1])], { type: 'video/mp4' })],
      blobMimeType: 'video/mp4',
      blobStartMs: 0,
    })
    expect(result.perSwing[0]?.failureReason).toBe('modal-not-configured')
    expect(result.perSwing[0]?.frames).toEqual([])
  })
})
