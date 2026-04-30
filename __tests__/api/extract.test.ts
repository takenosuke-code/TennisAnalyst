import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { NextRequest } from 'next/server'

// The route reads RAILWAY_SERVICE_URL, EXTRACT_API_KEY, and
// MODAL_INFERENCE_URL at module scope, so every test resets the module
// and stubs env freshly before importing. Mirrors
// __tests__/api/process-youtube.test.ts.

const MOCK_RAILWAY_URL = 'https://railway.test.example.com'
const MOCK_API_KEY = 'test-extract-key'
const MOCK_MODAL_URL = 'https://modal.test.example.com/extract'

function makeReq(body: unknown): NextRequest {
  return new NextRequest('http://localhost:3000/api/extract', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

async function importRoute() {
  const mod = await import('@/app/api/extract/route')
  return mod.POST
}

describe('POST /api/extract', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.stubGlobal('fetch', vi.fn())
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)
    vi.stubEnv('MODAL_INFERENCE_URL', MOCK_MODAL_URL)
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
  })

  it('returns 503 when RAILWAY_SERVICE_URL is unset — signals fallback to browser', async () => {
    vi.resetModules()
    vi.stubEnv('RAILWAY_SERVICE_URL', '')
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)
    const POST = await importRoute()
    const res = await POST(makeReq({ sessionId: 's1', blobUrl: 'https://blob/x.mp4' }))
    expect(res.status).toBe(503)
    const body = await res.json()
    expect(body.error).toBe('railway-not-configured')
  })

  it('returns 503 when EXTRACT_API_KEY is unset', async () => {
    vi.resetModules()
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', '')
    const POST = await importRoute()
    const res = await POST(makeReq({ sessionId: 's1', blobUrl: 'https://blob/x.mp4' }))
    expect(res.status).toBe(503)
  })

  it('returns 400 when sessionId is missing', async () => {
    const POST = await importRoute()
    const res = await POST(makeReq({ blobUrl: 'https://blob/x.mp4' }))
    expect(res.status).toBe(400)
  })

  it('returns 400 when blobUrl is missing', async () => {
    const POST = await importRoute()
    const res = await POST(makeReq({ sessionId: 's1' }))
    expect(res.status).toBe(400)
  })

  it('forwards to Railway with Bearer auth and translated body shape', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ status: 'queued', session_id: 's1' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    const POST = await importRoute()
    const res = await POST(
      makeReq({ sessionId: 's1', blobUrl: 'https://blob/x.mp4' }),
    )
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body.status).toBe('queued')
    expect(mockFetch).toHaveBeenCalledWith(
      `${MOCK_RAILWAY_URL}/extract`,
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: `Bearer ${MOCK_API_KEY}`,
          'Content-Type': 'application/json',
        }),
      }),
    )
    const call = mockFetch.mock.calls[0]
    const init = call[1] as RequestInit
    const sent = JSON.parse(init.body as string)
    // Railway's /extract expects snake_case video_url + session_id.
    expect(sent).toEqual({
      video_url: 'https://blob/x.mp4',
      session_id: 's1',
    })
  })

  it('propagates 502 when Railway returns a non-2xx response', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'Unauthorized' }), {
        status: 401,
      }),
    )
    const POST = await importRoute()
    const res = await POST(
      makeReq({ sessionId: 's1', blobUrl: 'https://blob/x.mp4' }),
    )
    expect(res.status).toBe(502)
    const body = await res.json()
    expect(body.error).toBe('railway-error')
    expect(body.status).toBe(401)
  })

  it('returns 503 (railway-unreachable) when fetch throws', async () => {
    const mockFetch = vi.mocked(fetch)
    mockFetch.mockRejectedValueOnce(new Error('ECONNREFUSED'))
    const POST = await importRoute()
    const res = await POST(
      makeReq({ sessionId: 's1', blobUrl: 'https://blob/x.mp4' }),
    )
    expect(res.status).toBe(503)
    const body = await res.json()
    expect(body.error).toBe('railway-unreachable')
  })

  // ---------------------------------------------------------------------
  // Phase E: mode === 'batch-ephemeral'.
  //
  // The live-coach batch path skips Railway and calls Modal directly
  // for short sub-clips. No sessionId, no DB write, returns frames inline.
  // ---------------------------------------------------------------------

  describe("mode === 'batch-ephemeral'", () => {
    const MODAL_OK_BODY = {
      fps_sampled: 30,
      frame_count: 2,
      frames: [
        { timestamp_ms: 0, frame_index: 0, landmarks: [] },
        { timestamp_ms: 33, frame_index: 1, landmarks: [] },
      ],
      video_fps: 30,
      duration_ms: 67,
      schema_version: 3,
      pose_backend: 'rtmpose-modal-cuda',
      timing: { download_ms: 100, inference_ms: 200 },
    }

    it('returns 503 (modal-not-configured) when MODAL_INFERENCE_URL is unset', async () => {
      vi.resetModules()
      vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
      vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)
      vi.stubEnv('MODAL_INFERENCE_URL', '')
      const POST = await importRoute()
      const res = await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(res.status).toBe(503)
      const body = await res.json()
      expect(body.error).toBe('modal-not-configured')
    })

    it('returns 400 when blobUrl is missing (no sessionId required)', async () => {
      const POST = await importRoute()
      const res = await POST(makeReq({ mode: 'batch-ephemeral' }))
      expect(res.status).toBe(400)
      const body = await res.json()
      expect(body.error).toBe('blobUrl is required')
    })

    it('does NOT require sessionId — accepts blobUrl-only body', async () => {
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify(MODAL_OK_BODY), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      const POST = await importRoute()
      const res = await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      // No 400 about sessionId — Modal got called.
      expect(res.status).toBe(200)
    })

    it('POSTs directly to MODAL_INFERENCE_URL (not Railway) with video_url+sample_fps', async () => {
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify(MODAL_OK_BODY), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      const POST = await importRoute()
      await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(mockFetch).toHaveBeenCalledTimes(1)
      const [calledUrl, init] = mockFetch.mock.calls[0]
      // Distinguish Modal from Railway by URL — Modal URL is configured
      // separately from RAILWAY_SERVICE_URL.
      expect(String(calledUrl)).toBe(MOCK_MODAL_URL)
      expect(String(calledUrl)).not.toContain('railway')
      const reqInit = init as RequestInit
      // No Bearer header — Modal endpoints are public HTTPS, secured
      // by the URL allowlist on the Modal side.
      const headers = reqInit.headers as Record<string, string>
      expect(headers).not.toHaveProperty('Authorization')
      expect(JSON.parse(reqInit.body as string)).toEqual({
        video_url: 'https://blob/x.mp4',
        sample_fps: 15,
      })
    })

    it('returns 200 with { frames, fps_sampled, pose_backend } inline on Modal success', async () => {
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify(MODAL_OK_BODY), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      const POST = await importRoute()
      const res = await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(res.status).toBe(200)
      const body = await res.json()
      expect(Array.isArray(body.frames)).toBe(true)
      expect(body.frames).toHaveLength(2)
      expect(body.fps_sampled).toBe(30)
      expect(body.pose_backend).toBe('rtmpose-modal-cuda')
    })

    it('returns 502 when Modal returns 5xx', async () => {
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'internal' }), { status: 500 }),
      )
      const POST = await importRoute()
      const res = await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(res.status).toBe(502)
      const body = await res.json()
      expect(body.error).toBe('modal-error')
      expect(body.status).toBe(500)
    })

    it('returns 502 when Modal returns 4xx (e.g. URL allowlist rejection)', async () => {
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'video_url not allowed' }), {
          status: 403,
        }),
      )
      const POST = await importRoute()
      const res = await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(res.status).toBe(502)
      const body = await res.json()
      expect(body.error).toBe('modal-error')
      expect(body.status).toBe(403)
    })

    it('returns 503 (modal-unreachable) when fetch to Modal throws', async () => {
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockRejectedValueOnce(new Error('ECONNREFUSED'))
      const POST = await importRoute()
      const res = await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(res.status).toBe(503)
      const body = await res.json()
      expect(body.error).toBe('modal-unreachable')
    })

    it('does not require Railway env to be set for batch-ephemeral', async () => {
      // batch-ephemeral path should work even if Railway is unconfigured.
      vi.resetModules()
      vi.stubEnv('RAILWAY_SERVICE_URL', '')
      vi.stubEnv('EXTRACT_API_KEY', '')
      vi.stubEnv('MODAL_INFERENCE_URL', MOCK_MODAL_URL)
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify(MODAL_OK_BODY), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      const POST = await importRoute()
      const res = await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(res.status).toBe(200)
    })
  })
})
