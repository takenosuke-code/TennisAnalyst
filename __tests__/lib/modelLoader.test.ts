import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { __resetInFlightForTests, loadModel } from '@/lib/modelLoader'

// Builds a Response whose body is a real ReadableStream that emits
// `chunks` one at a time. We use it in place of fetch() so the
// streaming-progress code path actually runs (the polyfilled jsdom
// fetch is not stream-friendly enough for this).
function streamedResponse(
  chunks: Uint8Array[],
  opts: { contentLength?: number; status?: number; statusText?: string } = {},
): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk)
      controller.close()
    },
  })
  const totalBytes = chunks.reduce((acc, c) => acc + c.byteLength, 0)
  const headers = new Headers()
  if (opts.contentLength !== undefined) {
    headers.set('content-length', String(opts.contentLength))
  } else {
    headers.set('content-length', String(totalBytes))
  }
  return new Response(stream, {
    status: opts.status ?? 200,
    statusText: opts.statusText ?? 'OK',
    headers,
  })
}

// Minimal in-memory Cache Storage stub. cache.match() returns the
// stored Response (cloned so consumers can tee/read it). cache.put()
// stores Response.arrayBuffer() so the underlying stream is fully
// drained — this matches what the browser does internally.
function makeCacheStorageStub() {
  const store = new Map<string, Uint8Array>()
  const cache = {
    match: vi.fn(async (key: string) => {
      const buf = store.get(key)
      if (!buf) return undefined
      // Return a fresh Response each call so callers can read it. Cast
      // because Response's BodyInit typings tightened in newer TS libs and
      // no longer accept Uint8Array directly even though runtimes do.
      return new Response(buf as unknown as BodyInit, {
        headers: { 'content-length': String(buf.byteLength) },
      })
    }),
    put: vi.fn(async (key: string, response: Response) => {
      const buf = new Uint8Array(await response.arrayBuffer())
      store.set(key, buf)
    }),
  }
  const caches = {
    open: vi.fn(async (_name: string) => cache),
  }
  return { caches, cache, store }
}

const URL_A = 'https://example.test/models/yolo11n.onnx'

describe('loadModel', () => {
  beforeEach(() => {
    __resetInFlightForTests()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('fetches and returns the bytes when no cache exists', async () => {
    const body = new Uint8Array([1, 2, 3, 4, 5])
    const fetchMock = vi.fn(async () => streamedResponse([body]))
    const { caches } = makeCacheStorageStub()
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('caches', caches)

    const out = await loadModel(URL_A)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(URL_A)
    expect(out).toBeInstanceOf(Uint8Array)
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5])
  })

  it('returns from cache on the second call (cache hit)', async () => {
    const body = new Uint8Array([10, 20, 30, 40])
    const fetchMock = vi.fn(async () => streamedResponse([body]))
    const { caches, cache } = makeCacheStorageStub()
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('caches', caches)

    const first = await loadModel(URL_A)
    expect(Array.from(first)).toEqual([10, 20, 30, 40])
    expect(fetchMock).toHaveBeenCalledTimes(1)

    // Second call: nothing should hit the network.
    const onProgress = vi.fn()
    const second = await loadModel(URL_A, { onProgress })
    expect(Array.from(second)).toEqual([10, 20, 30, 40])
    expect(fetchMock).toHaveBeenCalledTimes(1) // still 1
    expect(cache.match).toHaveBeenCalled()
    // Cache hits still fire onProgress once with loaded === total so UI
    // can transition out of its loading state.
    expect(onProgress).toHaveBeenCalledTimes(1)
    expect(onProgress).toHaveBeenCalledWith(4, 4)
  })

  it('fires onProgress with cumulative bytes during a streamed download', async () => {
    const c1 = new Uint8Array([1, 2, 3])
    const c2 = new Uint8Array([4, 5])
    const c3 = new Uint8Array([6, 7, 8, 9])
    const total = c1.byteLength + c2.byteLength + c3.byteLength // 9
    const fetchMock = vi.fn(async () =>
      streamedResponse([c1, c2, c3], { contentLength: total }),
    )
    const { caches } = makeCacheStorageStub()
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('caches', caches)

    const onProgress = vi.fn()
    const out = await loadModel(URL_A, { onProgress })

    expect(out.byteLength).toBe(total)
    // Each call's `loaded` value is monotonically non-decreasing. We
    // can't pin to exact chunk counts (the implementation may emit a
    // final synthetic 100% tick) but we can verify the cumulative
    // semantics and the final value.
    const calls = onProgress.mock.calls.map((c) => [c[0], c[1]])
    expect(calls.length).toBeGreaterThanOrEqual(3)
    let prev = 0
    for (const [loaded, totalArg] of calls) {
      expect(loaded).toBeGreaterThanOrEqual(prev)
      expect(totalArg).toBe(total)
      prev = loaded
    }
    expect(calls[calls.length - 1]).toEqual([total, total])
  })

  it('shares a single fetch across concurrent loadModel() calls for the same URL', async () => {
    const body = new Uint8Array([42, 43, 44])
    let resolveFetch: ((r: Response) => void) | null = null
    // Hold the fetch open so both callers register before it resolves.
    const fetchMock = vi.fn(
      () =>
        new Promise<Response>((res) => {
          resolveFetch = res
        }),
    )
    const { caches } = makeCacheStorageStub()
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('caches', caches)

    const p1 = loadModel(URL_A)
    const p2 = loadModel(URL_A)

    // The deduplication happens synchronously (the second call sees the
    // in-flight Promise registered by the first), but the actual fetch
    // call lives behind a couple of awaits inside doLoad (cache.open,
    // cache.match). Drain microtasks before asserting fetch was called.
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    expect(fetchMock).toHaveBeenCalledTimes(1)
    resolveFetch!(streamedResponse([body]))

    const [a, b] = await Promise.all([p1, p2])
    expect(Array.from(a)).toEqual([42, 43, 44])
    expect(Array.from(b)).toEqual([42, 43, 44])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('falls back gracefully when caches is undefined', async () => {
    const body = new Uint8Array([7, 8, 9])
    const fetchMock = vi.fn(async () => streamedResponse([body]))
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('caches', undefined)

    const out = await loadModel(URL_A)

    expect(Array.from(out)).toEqual([7, 8, 9])
    expect(fetchMock).toHaveBeenCalledTimes(1)

    // A second call with caches still missing should re-fetch (no cache
    // available means no cache hit possible).
    const out2 = await loadModel(URL_A)
    expect(Array.from(out2)).toEqual([7, 8, 9])
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('uses cacheKey when provided to override the URL as the cache key', async () => {
    const body = new Uint8Array([100, 101])
    const fetchMock = vi.fn(async () => streamedResponse([body]))
    const { caches, cache } = makeCacheStorageStub()
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('caches', caches)

    await loadModel(URL_A, { cacheKey: 'custom-key-v2' })
    expect(cache.put).toHaveBeenCalled()
    const putKey = cache.put.mock.calls[0][0]
    expect(putKey).toBe('custom-key-v2')

    // Second call with the same cacheKey should hit the cache.
    const out = await loadModel(URL_A, { cacheKey: 'custom-key-v2' })
    expect(Array.from(out)).toEqual([100, 101])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('throws on non-2xx fetch when nothing is cached', async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response('not found', { status: 404, statusText: 'Not Found' }),
    )
    const { caches } = makeCacheStorageStub()
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('caches', caches)

    await expect(loadModel(URL_A)).rejects.toThrow(/HTTP 404/)
  })
})
