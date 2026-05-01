/**
 * Phase 3 — Visibility worker tests for useLiveCapture. Phase E updated
 * the on-device inference to a gate-only path (~5fps) and moved per-
 * frame keypoint extraction to a Modal batch via
 * lib/liveSwingBatchExtractor. The test surface stays:
 *
 *   1. onPoseFrame fires for every detection-tick that survives the
 *      strict body-presence gate (and NOT for face-only / no-pose ticks).
 *   2. onPoseQuality emits transitions only — good → weak → no-body —
 *      not one event per frame.
 *
 * Plus three Phase E additions:
 *
 *   3. extractBatchedSwingKeypoints is called once a batch of 4 swings
 *      has accumulated, and each emitted swing carries the server-
 *      extracted frames the lib returned.
 *   4. On per-swing extraction failure the on-device frames remain in
 *      place as a fallback so the live coach still has angle data.
 *   5. pingModalWarmup() fires exactly once at session start.
 *
 * The test pump now uses a 200ms tick to match the new gate FPS (5fps
 * = 200ms gap between detect calls). The pre-Phase-E version assumed
 * 67ms (15fps) which is faster than the gate path now runs.
 *
 * To keep the test deterministic without spinning up a real ONNX runtime
 * + getUserMedia + MediaRecorder stack, we mock the pose detector,
 * camera, recorder, batch extractor, and warmup helper layers and drive
 * the loop through a controllable fake `setInterval` (the fallback path
 * the hook uses when requestVideoFrameCallback is unavailable, which is
 * the case in jsdom).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { LANDMARK_INDICES } from '@/lib/jointAngles'
import type { Landmark, PoseFrame as PoseFrameType } from '@/lib/supabase'
import type { StreamedSwing } from '@/lib/liveSwingDetector'
import type {
  BatchExtractRequest,
  BatchExtractResult,
} from '@/lib/liveSwingBatchExtractor'

// --- Mocks ------------------------------------------------------------------

// browserPose.detect returns Landmark[] | null directly. Tests queue up
// per-tick return values via this stack; once empty, falls back to
// returning null (no-body).
const detectQueue: Array<Landmark[] | null> = []
const detectMock = vi.fn(async () => {
  if (detectQueue.length === 0) return null
  return detectQueue.shift() ?? null
})
const disposeMock = vi.fn()
type CreatePoseDetectorOpts = {
  onProgress?: (
    loaded: number,
    total: number,
    label: 'yolo' | 'rtmpose',
  ) => void
}
const createPoseDetectorMock = vi.fn(async (_opts?: CreatePoseDetectorOpts) => ({
  detect: detectMock,
  dispose: disposeMock,
}))

vi.mock('@/lib/browserPose', () => ({
  createPoseDetector: (opts?: CreatePoseDetectorOpts) =>
    createPoseDetectorMock(opts),
}))

// LiveSwingDetector: we don't want a real one churning on synthetic
// frames in this test (it has its own dedicated suite). A no-op feed
// is enough to verify onPoseFrame plumbing — but the Phase E tests
// need to drive swing-emit, so the mock exposes a queue of swings to
// return from feed(). Tests push synthetic StreamedSwings; once the
// queue drains, feed returns null again.
const swingQueue: Array<StreamedSwing | null> = []
let nextSwingIndexCounter = 0
vi.mock('@/lib/liveSwingDetector', () => {
  return {
    LiveSwingDetector: class {
      feed() {
        if (swingQueue.length === 0) return null
        const next = swingQueue.shift()
        return next ?? null
      }
    },
  }
})

// Phase E batch extractor mock. The hook composes with the real
// implementation in lib/liveSwingBatchExtractor.ts, but in tests we
// stub it so we can assert the request shape and control the frames
// it returns per swing. The stub stores the most recent request for
// inspection and replays results from `batchExtractScript`.
const batchExtractMock = vi.fn<(req: BatchExtractRequest) => Promise<BatchExtractResult>>()
let batchExtractScript: BatchExtractResult | null = null
batchExtractMock.mockImplementation(async (req) => {
  if (batchExtractScript) return batchExtractScript
  // Default: every swing succeeds with a single synthetic server frame.
  return {
    perSwing: req.swings.map((s) => ({
      swingIndex: s.swingIndex,
      frames: [
        {
          frame_index: 0,
          timestamp_ms: 0,
          landmarks: [],
          joint_angles: { server_marker: 1 } as unknown as PoseFrameType['joint_angles'],
        },
      ],
      failureReason: null,
    })),
    totalDurationMs: 0,
  }
})
vi.mock('@/lib/liveSwingBatchExtractor', () => ({
  extractBatchedSwingKeypoints: (req: BatchExtractRequest) => batchExtractMock(req),
}))

// Phase E warmup mock — capture the call count and ensure both helpers
// are no-ops in test (they would otherwise try to fetch /api/extract).
const pingModalWarmupMock = vi.fn(async () => {})
const startWarmupHeartbeatMock = vi.fn(() => () => {})
vi.mock('@/lib/liveModalWarmup', () => ({
  pingModalWarmup: () => pingModalWarmupMock(),
  startWarmupHeartbeat: () => startWarmupHeartbeatMock(),
}))

// poseSmoothing: keep the real isFrameConfident / isBodyVisible behavior
// so transitions are honest. smoothFrames isn't exercised in these tests.

import { useLiveCapture, type PoseQuality } from '@/hooks/useLiveCapture'
import type { PoseFrame } from '@/lib/supabase'

// --- Helpers ----------------------------------------------------------------

function withIds(
  raw: Array<{ x: number; y: number; z?: number; visibility?: number }>,
): Landmark[] {
  // Coerce a flat raw landmark list into the 33-entry BlazePose Landmark[]
  // shape browserPose returns. Same x/y/visibility values; just the
  // explicit `id` and `name` fields the rest of the app expects.
  return raw.map((lm, id) => ({
    id,
    name: `landmark_${id}`,
    x: lm.x,
    y: lm.y,
    z: lm.z ?? 0,
    visibility: lm.visibility ?? 1,
  }))
}

function fullBodyLandmarks(): Landmark[] {
  // Constructs a head-to-ankle pose with all required landmarks at
  // visibility 0.95 (passes both isFrameConfident and isBodyVisible).
  // Must include ankles for vertical-extent-check >= 0.35.
  return withIds([
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 0 nose
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 1 left eye inner
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 2 left eye
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 3 left eye outer
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 4 right eye inner
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 5 right eye
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 6 right eye outer
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 7 left ear
    { x: 0.5, y: 0.1, visibility: 0.9 }, // 8 right ear
    { x: 0.5, y: 0.12, visibility: 0.9 }, // 9 mouth left
    { x: 0.5, y: 0.12, visibility: 0.9 }, // 10 mouth right
    { x: 0.55, y: 0.25, visibility: 0.95 }, // 11 left shoulder
    { x: 0.45, y: 0.25, visibility: 0.95 }, // 12 right shoulder
    { x: 0.6, y: 0.4, visibility: 0.95 }, // 13 left elbow
    { x: 0.4, y: 0.4, visibility: 0.95 }, // 14 right elbow
    { x: 0.6, y: 0.55, visibility: 0.95 }, // 15 left wrist
    { x: 0.4, y: 0.55, visibility: 0.95 }, // 16 right wrist
    { x: 0.6, y: 0.55, visibility: 0.5 }, // 17
    { x: 0.4, y: 0.55, visibility: 0.5 }, // 18
    { x: 0.6, y: 0.55, visibility: 0.5 }, // 19
    { x: 0.4, y: 0.55, visibility: 0.5 }, // 20
    { x: 0.6, y: 0.55, visibility: 0.5 }, // 21
    { x: 0.4, y: 0.55, visibility: 0.5 }, // 22
    { x: 0.53, y: 0.6, visibility: 0.95 }, // 23 left hip
    { x: 0.47, y: 0.6, visibility: 0.95 }, // 24 right hip
    { x: 0.54, y: 0.75, visibility: 0.9 }, // 25 left knee
    { x: 0.46, y: 0.75, visibility: 0.9 }, // 26 right knee
    { x: 0.54, y: 0.9, visibility: 0.9 }, // 27 left ankle
    { x: 0.46, y: 0.9, visibility: 0.9 }, // 28 right ankle
    { x: 0.54, y: 0.95, visibility: 0.9 }, // 29 left heel
    { x: 0.46, y: 0.95, visibility: 0.9 }, // 30 right heel
    { x: 0.54, y: 0.95, visibility: 0.9 }, // 31 left foot index
    { x: 0.46, y: 0.95, visibility: 0.9 }, // 32 right foot index
  ])
}

function faceOnlyLandmarks(): Landmark[] {
  // Passes isFrameConfident (avg visibility >= 0.4 + non-degenerate
  // bbox) but FAILS isBodyVisible (wrists at 0.3 < 0.5 cutoff). The
  // upper-body landmarks read confidently but the racket-arm wrists
  // do not — that's the "head and shoulders only" / "wrists below
  // frame" failure mode the gate is designed to reject.
  return withIds([
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 0 nose
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 1
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 2
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 3
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 4
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 5
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 6
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 7
    { x: 0.5, y: 0.1, visibility: 0.95 }, // 8
    { x: 0.5, y: 0.12, visibility: 0.95 }, // 9
    { x: 0.5, y: 0.12, visibility: 0.95 }, // 10
    { x: 0.55, y: 0.25, visibility: 0.95 }, // 11 left shoulder
    { x: 0.45, y: 0.25, visibility: 0.95 }, // 12 right shoulder
    { x: 0.6, y: 0.4, visibility: 0.6 }, // 13 elbows OK
    { x: 0.4, y: 0.4, visibility: 0.6 }, // 14
    { x: 0.6, y: 0.55, visibility: 0.3 }, // 15 wrists below cutoff
    { x: 0.4, y: 0.55, visibility: 0.3 }, // 16
    { x: 0.6, y: 0.55, visibility: 0.3 }, // 17
    { x: 0.4, y: 0.55, visibility: 0.3 }, // 18
    { x: 0.6, y: 0.55, visibility: 0.3 }, // 19
    { x: 0.4, y: 0.55, visibility: 0.3 }, // 20
    { x: 0.6, y: 0.55, visibility: 0.3 }, // 21
    { x: 0.4, y: 0.55, visibility: 0.3 }, // 22
    { x: 0.53, y: 0.6, visibility: 0.95 }, // 23 left hip (high vis but...)
    { x: 0.47, y: 0.6, visibility: 0.95 }, // 24 right hip
    { x: 0.54, y: 0.75, visibility: 0.3 }, // 25 knees low
    { x: 0.46, y: 0.75, visibility: 0.3 }, // 26
    { x: 0.54, y: 0.9, visibility: 0.2 }, // 27 ankles barely visible
    { x: 0.46, y: 0.9, visibility: 0.2 }, // 28
    { x: 0.54, y: 0.95, visibility: 0.2 }, // 29
    { x: 0.46, y: 0.95, visibility: 0.2 }, // 30
    { x: 0.54, y: 0.95, visibility: 0.2 }, // 31
    { x: 0.46, y: 0.95, visibility: 0.2 }, // 32
  ])
}

// --- Browser-API mocks ------------------------------------------------------

// MediaRecorder spy that exposes its onstop hook so the hook's stop()
// path can resolve the recorder.onstop promise. Not exercised in
// these tests beyond start(), but the hook constructs one immediately.
//
// Phase F3 — `lastInstance` is a static slot that captures the most
// recently constructed FakeMediaRecorder so memory-trim tests can drive
// `ondataavailable` from outside (the hook keeps the recorder ref
// private, but the global is enough for tests).
class FakeMediaRecorder {
  static lastInstance: FakeMediaRecorder | null = null
  state: 'inactive' | 'recording' = 'inactive'
  mimeType: string
  ondataavailable: ((ev: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  constructor(_stream: unknown, opts?: { mimeType?: string }) {
    this.mimeType = opts?.mimeType ?? 'video/webm'
    FakeMediaRecorder.lastInstance = this
  }
  start() {
    this.state = 'recording'
  }
  stop() {
    this.state = 'inactive'
    this.onstop?.()
  }
  static isTypeSupported(_mime: string) {
    return true
  }
}

function installBrowserMocks() {
  ;(globalThis as unknown as { MediaRecorder: typeof FakeMediaRecorder }).MediaRecorder =
    FakeMediaRecorder
  ;(navigator.mediaDevices as unknown as {
    getUserMedia: (c: MediaStreamConstraints) => Promise<MediaStream>
  }) = {
    getUserMedia: vi.fn(
      async () =>
        ({
          getTracks: () => [{ stop: vi.fn() }],
        }) as unknown as MediaStream,
    ),
  }

  // jsdom doesn't implement getContext('2d') and the hook needs one
  // for its offscreen sampling canvas. Patch HTMLCanvasElement to
  // return a no-op context — useLiveCapture only calls drawImage and
  // never reads pixels.
  const noopCtx = {
    drawImage: vi.fn(),
    clearRect: vi.fn(),
    setTransform: vi.fn(),
    scale: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    fillRect: vi.fn(),
    fillText: vi.fn(),
    measureText: vi.fn(() => ({ width: 0 })),
    translate: vi.fn(),
    rotate: vi.fn(),
    quadraticCurveTo: vi.fn(),
    closePath: vi.fn(),
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    globalAlpha: 1,
    font: '',
    textAlign: 'left',
    textBaseline: 'alphabetic',
  } as unknown as CanvasRenderingContext2D
  HTMLCanvasElement.prototype.getContext = (function () {
    return noopCtx
  }) as unknown as HTMLCanvasElement['getContext']
}

function makeVideoEl(): HTMLVideoElement {
  // jsdom doesn't implement video playback. Stub the surface useLiveCapture
  // touches so start() can complete: play(), srcObject setter (tolerant),
  // videoWidth/Height (query-time access).
  const el = document.createElement('video')
  Object.defineProperty(el, 'play', {
    value: () => Promise.resolve(),
    writable: true,
  })
  Object.defineProperty(el, 'videoWidth', { value: 640, writable: true, configurable: true })
  Object.defineProperty(el, 'videoHeight', { value: 360, writable: true, configurable: true })
  // requestVideoFrameCallback is intentionally omitted so the hook
  // falls back to setInterval — that's the path we drive in tests
  // because it's pumpable with vi.useFakeTimers().
  return el
}

// detect() is async, so each fake-timer tick schedules a microtask the
// `await` is waiting on. Drain microtasks twice (drawImage → detect →
// continuation → emit) so onPoseFrame/onPoseQuality have run before we
// assert.
async function flushMicrotasks() {
  for (let i = 0; i < 5; i++) {
    await Promise.resolve()
  }
}

// --- Tests ------------------------------------------------------------------

// Phase E — the hook caps detection at 5fps for the gates (200ms gap)
// regardless of `targetDetectionFps`, since gate-only inference is all
// the on-device path is responsible for now. Tick at 220ms so each tick
// reliably crosses the gap and triggers exactly one detection.
const TICK_MS = 220

describe('useLiveCapture — onPoseFrame + onPoseQuality', () => {
  beforeEach(() => {
    detectMock.mockClear()
    disposeMock.mockClear()
    createPoseDetectorMock.mockClear()
    batchExtractMock.mockClear()
    pingModalWarmupMock.mockClear()
    startWarmupHeartbeatMock.mockClear()
    detectQueue.length = 0
    swingQueue.length = 0
    nextSwingIndexCounter = 0
    batchExtractScript = null
    installBrowserMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('fires onPoseFrame for every detection tick that passes isBodyVisible', async () => {
    const onPoseFrame = vi.fn<(frame: PoseFrame) => void>()
    const onPoseQuality = vi.fn<(q: PoseQuality) => void>()

    // Three good frames in a row.
    detectQueue.push(fullBodyLandmarks(), fullBodyLandmarks(), fullBodyLandmarks())

    const { result } = renderHook(() =>
      useLiveCapture({ onPoseFrame, onPoseQuality, targetDetectionFps: 15 }),
    )

    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })

    // Pump the fallback setInterval past three ticks. Phase E caps the
    // gate path at 5fps (1000/5 = 200ms gap) regardless of target FPS,
    // so the test ticks at 220ms — comfortably over the gap so every
    // tick produces a detection. detect() is async so we drain
    // microtasks in between to let the continuation run before the
    // next interval tick.
    for (let i = 0; i < 6; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }

    expect(onPoseFrame).toHaveBeenCalledTimes(3)
    // Frames are well-formed PoseFrame objects with the expected wrist
    // landmark mapped.
    const firstFrame = onPoseFrame.mock.calls[0][0]
    expect(firstFrame.landmarks).toBeDefined()
    expect(
      firstFrame.landmarks.find((lm) => lm.id === LANDMARK_INDICES.RIGHT_WRIST),
    ).toBeTruthy()
    // First quality emission must be 'good'.
    expect(onPoseQuality.mock.calls[0][0]).toBe('good')
    // No 'weak' or 'no-body' transitions across three good frames in a row.
    const qualities = onPoseQuality.mock.calls.map((c) => c[0])
    expect(qualities.filter((q) => q !== 'good')).toHaveLength(0)
  })

  it('does NOT fire onPoseFrame for face-only frames (weak quality)', async () => {
    const onPoseFrame = vi.fn<(frame: PoseFrame) => void>()
    const onPoseQuality = vi.fn<(q: PoseQuality) => void>()

    detectQueue.push(faceOnlyLandmarks(), faceOnlyLandmarks())

    const { result } = renderHook(() =>
      useLiveCapture({ onPoseFrame, onPoseQuality, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()

    await act(async () => {
      await result.current.start(video)
    })
    for (let i = 0; i < 4; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }

    // Face-only frames are gated out before onPoseFrame, but they
    // still trigger a 'weak' quality transition.
    expect(onPoseFrame).not.toHaveBeenCalled()
    expect(onPoseQuality.mock.calls.some((c) => c[0] === 'weak')).toBe(true)
  })

  it('emits onPoseQuality only on transitions, not per frame', async () => {
    const onPoseQuality = vi.fn<(q: PoseQuality) => void>()

    // Five good frames back-to-back. The first should fire 'good'; the
    // remaining four should NOT re-fire 'good' (no transition).
    for (let i = 0; i < 5; i++) {
      detectQueue.push(fullBodyLandmarks())
    }

    const { result } = renderHook(() =>
      useLiveCapture({ onPoseQuality, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()

    await act(async () => {
      await result.current.start(video)
    })
    for (let i = 0; i < 8; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }

    const goodEmissions = onPoseQuality.mock.calls.filter((c) => c[0] === 'good')
    expect(goodEmissions).toHaveLength(1)
  })

  it('transitions good → weak → no-body across the right fixtures', async () => {
    const onPoseQuality = vi.fn<(q: PoseQuality) => void>()

    // 1 good, 1 weak (face-only), then several "no detection" returns
    // long enough that the no-body 1s timeout fires. Once the queue is
    // drained the mock falls back to returning null (no person).
    detectQueue.push(fullBodyLandmarks())
    detectQueue.push(faceOnlyLandmarks())

    const { result } = renderHook(() =>
      useLiveCapture({ onPoseQuality, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })

    // First two ticks land good then weak.
    for (let i = 0; i < 4; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }
    const earlyQualities = onPoseQuality.mock.calls.map((c) => c[0])
    expect(earlyQualities).toContain('good')
    expect(earlyQualities.indexOf('good')).toBeLessThan(earlyQualities.indexOf('weak'))

    // Now sit through >1s of "no landmarks" frames (queue is drained)
    // so the no-body timeout triggers.
    for (let i = 0; i < 25; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }
    const finalQuality = onPoseQuality.mock.calls.at(-1)?.[0]
    expect(finalQuality).toBe('no-body')
  })

  it('forwards onModelLoadProgress to createPoseDetector', async () => {
    const onModelLoadProgress = vi.fn()

    detectQueue.push(fullBodyLandmarks())

    const { result } = renderHook(() =>
      useLiveCapture({ onModelLoadProgress, targetDetectionFps: 15 }),
    )

    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })

    expect(createPoseDetectorMock).toHaveBeenCalledTimes(1)
    const opts = createPoseDetectorMock.mock.calls[0][0]
    expect(opts?.onProgress).toBe(onModelLoadProgress)
  })

  it('fires pingModalWarmup once at session start', async () => {
    detectQueue.push(fullBodyLandmarks())
    const { result } = renderHook(() =>
      useLiveCapture({ targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })
    expect(pingModalWarmupMock).toHaveBeenCalledTimes(1)
    // Heartbeat is also wired so mid-session pauses don't lose the
    // container — assert it's been started exactly once.
    expect(startWarmupHeartbeatMock).toHaveBeenCalledTimes(1)
  })
})

// =====================================================================
// Phase E — server-extraction batching for swings.
// =====================================================================

function makeSyntheticSwing(swingIndex: number): StreamedSwing {
  return {
    swingIndex,
    startFrameIndex: 0,
    endFrameIndex: 5,
    peakFrameIndex: 2,
    startMs: 1000 * swingIndex,
    endMs: 1000 * swingIndex + 500,
    // On-device frames the detector emitted live. The Phase E batch
    // path replaces these with server-extracted frames on success and
    // leaves them in place on per-swing failure.
    frames: [
      {
        frame_index: 0,
        timestamp_ms: 0,
        landmarks: [],
        joint_angles: { ondevice_marker: swingIndex } as unknown as PoseFrameType['joint_angles'],
      },
    ],
  }
}

describe('useLiveCapture — Phase E server extraction', () => {
  beforeEach(() => {
    detectMock.mockClear()
    disposeMock.mockClear()
    createPoseDetectorMock.mockClear()
    batchExtractMock.mockClear()
    pingModalWarmupMock.mockClear()
    startWarmupHeartbeatMock.mockClear()
    detectQueue.length = 0
    swingQueue.length = 0
    nextSwingIndexCounter = 0
    batchExtractScript = null
    installBrowserMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('emits swings carrying server-extracted frames after batch (size 4)', async () => {
    // Queue 4 detector returns. Each one will emit a swing on its
    // mock-feed call because we also queue 4 synthetic swings.
    for (let i = 0; i < 4; i++) {
      detectQueue.push(fullBodyLandmarks())
      swingQueue.push(makeSyntheticSwing(i + 1))
    }

    // Script the batch result so each swing comes back with frames
    // containing a unique server marker we can assert against.
    batchExtractScript = {
      perSwing: [1, 2, 3, 4].map((idx) => ({
        swingIndex: idx,
        frames: [
          {
            frame_index: 0,
            timestamp_ms: 0,
            landmarks: [],
            joint_angles: { server_swing: idx } as unknown as PoseFrameType['joint_angles'],
          },
        ],
        failureReason: null,
      })),
      totalDurationMs: 1234,
    }

    const onSwing = vi.fn<(s: StreamedSwing) => void>()
    const { result } = renderHook(() =>
      useLiveCapture({ onSwing, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })

    // Pump four detection ticks — the swing queue will emit one swing
    // per tick. The 4th swing trips LIVE_EXTRACTION_BATCH_SIZE and
    // fires extractBatchedSwingKeypoints.
    for (let i = 0; i < 4; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }
    // After the batch fire, drain microtasks one more time so the
    // promise chain inside fireBatch runs to completion.
    await act(async () => {
      await flushMicrotasks()
    })

    expect(batchExtractMock).toHaveBeenCalledTimes(1)
    const req = batchExtractMock.mock.calls[0][0]
    expect(req.swings.map((s) => s.swingIndex)).toEqual([1, 2, 3, 4])
    expect(req.blobMimeType.length).toBeGreaterThan(0)

    // All 4 swings forwarded with server-extracted frames.
    expect(onSwing).toHaveBeenCalledTimes(4)
    for (let i = 0; i < 4; i++) {
      const swing = onSwing.mock.calls[i][0]
      expect(swing.swingIndex).toBe(i + 1)
      const angles = swing.frames[0].joint_angles as unknown as { server_swing?: number }
      expect(angles.server_swing).toBe(i + 1)
    }
  })

  it('falls back to on-device frames on per-swing extraction failure', async () => {
    for (let i = 0; i < 4; i++) {
      detectQueue.push(fullBodyLandmarks())
      swingQueue.push(makeSyntheticSwing(i + 1))
    }

    // Swing 2 fails extraction; the others succeed. The hook must
    // forward swing 2 with its original on-device frames intact.
    batchExtractScript = {
      perSwing: [
        {
          swingIndex: 1,
          frames: [
            {
              frame_index: 0,
              timestamp_ms: 0,
              landmarks: [],
              joint_angles: { server_swing: 1 } as unknown as PoseFrameType['joint_angles'],
            },
          ],
          failureReason: null,
        },
        { swingIndex: 2, frames: [], failureReason: 'no-frames-in-range' },
        {
          swingIndex: 3,
          frames: [
            {
              frame_index: 0,
              timestamp_ms: 0,
              landmarks: [],
              joint_angles: { server_swing: 3 } as unknown as PoseFrameType['joint_angles'],
            },
          ],
          failureReason: null,
        },
        {
          swingIndex: 4,
          frames: [
            {
              frame_index: 0,
              timestamp_ms: 0,
              landmarks: [],
              joint_angles: { server_swing: 4 } as unknown as PoseFrameType['joint_angles'],
            },
          ],
          failureReason: null,
        },
      ],
      totalDurationMs: 0,
    }

    const onSwing = vi.fn<(s: StreamedSwing) => void>()
    const { result } = renderHook(() =>
      useLiveCapture({ onSwing, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })
    for (let i = 0; i < 4; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }
    await act(async () => {
      await flushMicrotasks()
    })

    expect(onSwing).toHaveBeenCalledTimes(4)
    // Swing 2 should retain its on-device frames marker.
    const swing2 = onSwing.mock.calls.find((c) => c[0].swingIndex === 2)?.[0]
    expect(swing2).toBeDefined()
    const swing2Angles = swing2!.frames[0].joint_angles as unknown as { ondevice_marker?: number }
    expect(swing2Angles.ondevice_marker).toBe(2)
    // Swing 3 should have server frames.
    const swing3 = onSwing.mock.calls.find((c) => c[0].swingIndex === 3)?.[0]
    const swing3Angles = swing3!.frames[0].joint_angles as unknown as { server_swing?: number }
    expect(swing3Angles.server_swing).toBe(3)
  })

  it('does not call extractBatchedSwingKeypoints before the batch fills', async () => {
    // Only 2 swings — below LIVE_EXTRACTION_BATCH_SIZE (4). No batch
    // fires until the idle timer (10s) elapses. Within a few ticks of
    // detection no batch should have been issued.
    for (let i = 0; i < 2; i++) {
      detectQueue.push(fullBodyLandmarks())
      swingQueue.push(makeSyntheticSwing(i + 1))
    }
    const onSwing = vi.fn<(s: StreamedSwing) => void>()
    const { result } = renderHook(() =>
      useLiveCapture({ onSwing, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })
    for (let i = 0; i < 2; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }
    expect(batchExtractMock).not.toHaveBeenCalled()
    expect(onSwing).not.toHaveBeenCalled()
  })
})

// =====================================================================
// Phase F3 — memory bounding (sliding-window slice in fireBatch).
// =====================================================================
//
// `chunksRef.current` accumulates every MediaRecorder chunk for the
// session — at ~2 Mbps × 30 minutes that's ~450 MB of Blob refs.
// Pre-Phase-F3, fireBatch snapshotted the FULL chunk array on every
// fire and `lib/liveSwingBatchExtractor.sliceToSubClip` materialized
// the entire buffer via `await new Blob(chunks).arrayBuffer()` — that
// peak alone could OOM-kill iOS Safari on long sessions.
//
// The fix: in fireBatch, slice the chunk buffer to ONLY chunks whose
// recording-relative [startMs, endMs] window overlaps any pending
// swing's window (with 1s padding either side). chunk[0] is always
// included for the mp4 moov header. chunksRef itself stays whole so
// the final stop() blob is complete.

// Test-only registry mapping blob refs back to the index we fed them
// at. Lets us reverse-engineer which chunks the hook included in its
// sliding-window slice.
const chunkIndexRegistry = new WeakMap<Blob, number>()

describe('useLiveCapture — Phase F3 fireBatch memory bounding', () => {
  beforeEach(() => {
    detectMock.mockClear()
    disposeMock.mockClear()
    createPoseDetectorMock.mockClear()
    batchExtractMock.mockClear()
    pingModalWarmupMock.mockClear()
    startWarmupHeartbeatMock.mockClear()
    detectQueue.length = 0
    swingQueue.length = 0
    nextSwingIndexCounter = 0
    batchExtractScript = null
    installBrowserMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  /**
   * Drive `recorder.ondataavailable` directly against the captured
   * FakeMediaRecorder instance. Fires `count` chunks of `sizeBytes`
   * each, advancing fake time by `gapMs` between them so the hook's
   * chunkTimestampsRef captures contiguous wall-clock-relative ranges.
   *
   * Each blob is registered in `chunkIndexRegistry` so the test can
   * verify which chunks landed in the slice.
   */
  async function feedChunks(count: number, gapMs: number, sizeBytes = 1024) {
    const recorder = FakeMediaRecorder.lastInstance
    if (!recorder) throw new Error('FakeMediaRecorder.lastInstance is null')
    for (let i = 0; i < count; i++) {
      // Advance time first so performance.now() at the moment of the
      // ondataavailable callback reflects this chunk's end.
      vi.advanceTimersByTime(gapMs)
      const payload = new Uint8Array(sizeBytes)
      payload[0] = (i + 1) & 0xff
      const blob = new Blob([payload], { type: recorder.mimeType })
      chunkIndexRegistry.set(blob, i)
      recorder.ondataavailable?.({ data: blob })
    }
  }

  it('passes only the chunks covering pending swing windows + 1s padding to the extractor', async () => {
    // Build a session where we feed 100 chunks of ~1s each (so the
    // recording is ~100s long) BEFORE any swings fire. Then 4 swings
    // land at known windows. The extractor mock captures the request
    // and we assert blobChunks contains FAR fewer than 100 chunks —
    // specifically only those whose [startMs, endMs] overlaps the
    // padded swing windows + chunk[0].
    //
    // Swing windows (ms, recording-relative):
    //   swing 1: [10_000, 10_500]
    //   swing 2: [20_000, 20_500]
    //   swing 3: [30_000, 30_500]
    //   swing 4: [40_000, 40_500]
    //
    // With 1s padding either side, the overlapping windows are:
    //   [9_000, 11_500], [19_000, 21_500], [29_000, 31_500],
    //   [39_000, 41_500]
    //
    // Each chunk is ~1000ms wide, so we expect roughly 3 chunks per
    // swing (the chunk straddling each swing plus its padding
    // neighbours) + chunk[0] for the moov header. Definitely NOT all
    // 100 chunks.
    const swings: StreamedSwing[] = [
      { swingIndex: 1, startFrameIndex: 0, endFrameIndex: 5, peakFrameIndex: 2,
        startMs: 10_000, endMs: 10_500, frames: [] },
      { swingIndex: 2, startFrameIndex: 0, endFrameIndex: 5, peakFrameIndex: 2,
        startMs: 20_000, endMs: 20_500, frames: [] },
      { swingIndex: 3, startFrameIndex: 0, endFrameIndex: 5, peakFrameIndex: 2,
        startMs: 30_000, endMs: 30_500, frames: [] },
      { swingIndex: 4, startFrameIndex: 0, endFrameIndex: 5, peakFrameIndex: 2,
        startMs: 40_000, endMs: 40_500, frames: [] },
    ]

    for (const s of swings) {
      detectQueue.push(fullBodyLandmarks())
      swingQueue.push(s)
    }

    const onSwing = vi.fn<(s: StreamedSwing) => void>()
    const { result } = renderHook(() =>
      useLiveCapture({ onSwing, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })

    // Feed 100 1-second chunks BEFORE any detection ticks fire. The
    // recorder timestamps will run from 0 to 100_000 ms.
    await act(async () => {
      await feedChunks(100, 1000)
    })

    // Now drive 4 detection ticks which will queue 4 swings and fire
    // a batch.
    for (let i = 0; i < 4; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }
    await act(async () => {
      await flushMicrotasks()
    })

    expect(batchExtractMock).toHaveBeenCalledTimes(1)
    const req = batchExtractMock.mock.calls[0][0]

    // Sanity: all 4 swings still in the request.
    expect(req.swings.map((s) => s.swingIndex).sort()).toEqual([1, 2, 3, 4])

    // Memory bound check: the slice MUST be much smaller than the full
    // 100-chunk buffer. Allow generous headroom (chunk[0] + ~3 chunks
    // per swing × 4 swings = ~13 chunks worst-case) but assert FAR less
    // than 100.
    expect(req.blobChunks.length).toBeLessThan(20)
    expect(req.blobChunks.length).toBeGreaterThan(0)

    // chunk[0] (moov header) MUST be included.
    const passedIndices = req.blobChunks
      .map((b) => chunkIndexRegistry.get(b))
      .filter((i): i is number => typeof i === 'number')
    expect(passedIndices).toContain(0)

    // For each swing window, at least one chunk overlapping it must be
    // included. Swing 1 covers [10_000, 10_500] → chunk index 9 or 10.
    // Padding 1000 each side: [9_000, 11_500] → chunks 8-11 ish.
    const includesNear = (target: number) =>
      passedIndices.some((i) => Math.abs(i - target) <= 2)
    expect(includesNear(10)).toBe(true)
    expect(includesNear(20)).toBe(true)
    expect(includesNear(30)).toBe(true)
    expect(includesNear(40)).toBe(true)

    // Chunks far from any swing window MUST be excluded. Index 50 is
    // halfway between swings 2 (~20s) and 3 (~30s), well outside any
    // padded window.
    expect(passedIndices).not.toContain(50)
    expect(passedIndices).not.toContain(60)
    expect(passedIndices).not.toContain(70)
  })

  it('stop() produces a complete blob containing every chunk fed during the session', async () => {
    // Feed 10 chunks of distinct sizes so we can verify the final blob
    // size sums them all. chunksRef must NOT be trimmed — only the
    // sliding-window slice that goes to extractBatchedSwingKeypoints is.
    const onSwing = vi.fn<(s: StreamedSwing) => void>()
    const { result } = renderHook(() =>
      useLiveCapture({ onSwing, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })

    // Each chunk has a different byte size so the total can be
    // computed deterministically. Sum: 100 + 200 + ... + 1000 = 5500.
    const recorder = FakeMediaRecorder.lastInstance
    if (!recorder) throw new Error('FakeMediaRecorder.lastInstance is null')
    let expectedTotal = 0
    for (let i = 1; i <= 10; i++) {
      vi.advanceTimersByTime(1000)
      const size = i * 100
      expectedTotal += size
      const payload = new Uint8Array(size)
      payload[0] = i
      const blob = new Blob([payload], { type: recorder.mimeType })
      recorder.ondataavailable?.({ data: blob })
    }

    let stopResult: Awaited<ReturnType<typeof result.current.stop>> = null
    await act(async () => {
      stopResult = await result.current.stop()
    })

    expect(stopResult).not.toBeNull()
    // The final blob must hold ALL 10 chunks of bytes — chunksRef stays
    // whole even after the sliding-window slice in fireBatch.
    expect(stopResult!.blob.size).toBe(expectedTotal)
  })

  it('still slices correctly when no chunks have been fed yet (degenerate empty buffer)', async () => {
    // Defensive: a session that fires a batch before any chunks have
    // arrived (very fast detector + very small batch size in tests)
    // must not crash on the empty chunkTimestampsRef. The extractor
    // gets an empty blobChunks and the lib's empty-subclip path takes
    // over.
    for (let i = 0; i < 4; i++) {
      detectQueue.push(fullBodyLandmarks())
      swingQueue.push(makeSyntheticSwing(i + 1))
    }
    const onSwing = vi.fn<(s: StreamedSwing) => void>()
    const { result } = renderHook(() =>
      useLiveCapture({ onSwing, targetDetectionFps: 15 }),
    )
    const video = makeVideoEl()
    await act(async () => {
      await result.current.start(video)
    })
    // No feedChunks — the recorder never fires ondataavailable in this
    // test. Drive the 4 swings.
    for (let i = 0; i < 4; i++) {
      await act(async () => {
        vi.advanceTimersByTime(TICK_MS)
        await flushMicrotasks()
      })
    }
    await act(async () => {
      await flushMicrotasks()
    })

    expect(batchExtractMock).toHaveBeenCalledTimes(1)
    const req = batchExtractMock.mock.calls[0][0]
    expect(req.blobChunks).toEqual([])
    // All 4 swings still emitted (with on-device or server frames per
    // the mock's default success path).
    expect(onSwing).toHaveBeenCalledTimes(4)
  })
})
