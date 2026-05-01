import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { NextRequest } from 'next/server'

// The route reads RAILWAY_SERVICE_URL, EXTRACT_API_KEY, and
// MODAL_INFERENCE_URL at module scope, so every test resets the module
// and stubs env freshly before importing. Mirrors
// __tests__/api/process-youtube.test.ts.

const MOCK_RAILWAY_URL = 'https://railway.test.example.com'
const MOCK_API_KEY = 'test-extract-key'
const MOCK_MODAL_URL = 'https://modal.test.example.com/extract'

// Auth client mock: defaults to a signed-in user so existing
// batch-ephemeral happy-path tests keep working. The 401 test flips
// this to null per-call.
const mockGetUser = vi.fn<() => Promise<{ data: { user: { id: string } | null } }>>(
  async () => ({ data: { user: { id: 'user-1' } } }),
)
vi.mock('@/lib/supabase/server', () => ({
  createClient: async () => ({
    auth: { getUser: mockGetUser },
  }),
}))

// pose_cache hooks. Default to "miss" so all the existing tests that
// don't set sha256 (or do but want the live extraction path) keep
// working. Cache-specific tests below override per-call.
//
// Why we mock with vi.fn() directly (not via wrapper closures): the
// real setCachedPose validates its three arguments (sha256, poseJson,
// modelVersion) and throws on missing sha256/modelVersion. An earlier
// version of this mock declared `setCachedPose: () => mockSetCachedPose()`,
// which DROPPED every argument the route actually passed — so a route
// regression that "forgot" to pass modelVersion would still pass tests.
// Spying directly with the real signature lets us assert
// `toHaveBeenCalledWith(sha, pose, modelVersion)`.
const mockGetCachedPose = vi.fn<(sha: string, mv?: string) => Promise<unknown>>(
  async () => null,
)
const mockSetCachedPose = vi.fn<
  (sha256: string, poseJson: unknown, modelVersion: string) => Promise<void>
>(async () => undefined)
vi.mock('@/lib/poseCache', () => ({
  getCachedPose: mockGetCachedPose,
  setCachedPose: mockSetCachedPose,
  // Re-export the helper that route.ts uses to project a stored
  // ModalExtractResponse down to a KeypointsJson when it writes back
  // to user_sessions. The mocked-out version mirrors the real one.
  poseJsonToKeypointsJson: (p: {
    fps_sampled: number
    frame_count: number
    frames: unknown[]
    schema_version?: number
  }) => ({
    fps_sampled: p.fps_sampled,
    frame_count: p.frame_count,
    frames: p.frames,
    schema_version: p.schema_version,
  }),
}))

// supabaseAdmin mock for the cache-hit short-circuit path that writes
// keypoints into the user_sessions row. Chain: from(...).update(...).eq(...).eq(...)
// resolves to { error }. Default error=null so cache-hit tests can
// rely on a successful write without per-test setup; the ownership-
// mismatch case overrides this.
type SessionUpdateResult = { error: { message: string } | null }
const mockSessionEq2 = vi.fn<() => Promise<SessionUpdateResult>>(
  async () => ({ error: null }),
)
const mockSessionEq1 = vi.fn(() => ({ eq: mockSessionEq2 }))
const mockSessionUpdate = vi.fn(() => ({ eq: mockSessionEq1 }))
vi.mock('@/lib/supabase', () => ({
  supabase: {},
  supabaseAdmin: {
    from: vi.fn(() => ({ update: mockSessionUpdate })),
  },
}))

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
    // Reset auth mock to signed-in default each test.
    mockGetUser.mockReset()
    mockGetUser.mockResolvedValue({ data: { user: { id: 'user-1' } } })
    // Reset cache mocks — default to MISS so non-cache tests keep
    // their pre-cache behavior.
    mockGetCachedPose.mockReset()
    mockGetCachedPose.mockResolvedValue(null)
    mockSetCachedPose.mockReset()
    mockSetCachedPose.mockResolvedValue(undefined)
    // Reset session-update chain.
    mockSessionEq2.mockReset()
    mockSessionEq2.mockResolvedValue({ error: null })
    mockSessionEq1.mockClear()
    mockSessionUpdate.mockClear()
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
  // Phase E: SHA-256 content-hash cache (legacy/Railway path).
  //
  // When the client supplies a sha256 of the source bytes and the
  // server has cached keypoints for that hash, the route writes the
  // cached pose into the user_sessions row and short-circuits the
  // Railway hop. The client's existing /api/sessions/[id] poll picks
  // up status='complete' on the next tick — same contract.
  // ---------------------------------------------------------------------

  describe('content-hash cache (legacy path)', () => {
    const VALID_SHA = 'a'.repeat(64)
    const CACHED_POSE = {
      fps_sampled: 30,
      frame_count: 1,
      frames: [
        { frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} },
      ],
      schema_version: 3,
    }

    it('on cache HIT: writes keypoints into user_sessions, returns cache-hit status, skips Railway', async () => {
      mockGetCachedPose.mockResolvedValueOnce(CACHED_POSE)
      const mockFetch = vi.mocked(fetch)
      const POST = await importRoute()
      const res = await POST(
        makeReq({
          sessionId: 's1',
          blobUrl: 'https://blob/x.mp4',
          sha256: VALID_SHA,
        }),
      )
      expect(res.status).toBe(200)
      const body = await res.json()
      expect(body.status).toBe('cache-hit')
      expect(body.sessionId).toBe('s1')
      // Critical: Railway must NOT be called on cache hit.
      expect(mockFetch).not.toHaveBeenCalled()
      // Session row updated with the cached keypoints + status=complete.
      expect(mockSessionUpdate).toHaveBeenCalledWith({
        status: 'complete',
        keypoints_json: CACHED_POSE,
      })
    })

    it('on cache MISS: forwards to Railway as usual', async () => {
      mockGetCachedPose.mockResolvedValueOnce(null)
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ status: 'queued' }), { status: 200 }),
      )
      const POST = await importRoute()
      const res = await POST(
        makeReq({
          sessionId: 's1',
          blobUrl: 'https://blob/x.mp4',
          sha256: VALID_SHA,
        }),
      )
      expect(res.status).toBe(200)
      const body = await res.json()
      expect(body.status).toBe('queued')
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })

    it('without sha256 in body: skips cache lookup entirely', async () => {
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ status: 'queued' }), { status: 200 }),
      )
      const POST = await importRoute()
      const res = await POST(
        makeReq({ sessionId: 's1', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(res.status).toBe(200)
      // getCachedPose must not be called when there's no hash to lookup.
      expect(mockGetCachedPose).not.toHaveBeenCalled()
    })

    it('with malformed sha256 (wrong length): skips cache, falls through to Railway', async () => {
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ status: 'queued' }), { status: 200 }),
      )
      const POST = await importRoute()
      await POST(
        makeReq({
          sessionId: 's1',
          blobUrl: 'https://blob/x.mp4',
          sha256: 'too-short',
        }),
      )
      expect(mockGetCachedPose).not.toHaveBeenCalled()
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })

    it('cache lookup throw is non-fatal: falls through to Railway', async () => {
      mockGetCachedPose.mockRejectedValueOnce(new Error('cache outage'))
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ status: 'queued' }), { status: 200 }),
      )
      const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        const POST = await importRoute()
        const res = await POST(
          makeReq({
            sessionId: 's1',
            blobUrl: 'https://blob/x.mp4',
            sha256: VALID_SHA,
          }),
        )
        expect(res.status).toBe(200)
        const body = await res.json()
        expect(body.status).toBe('queued')
      } finally {
        errSpy.mockRestore()
      }
    })

    it('cache hit + session update fails: still falls through to Railway (no user-facing error)', async () => {
      mockGetCachedPose.mockResolvedValueOnce(CACHED_POSE)
      mockSessionEq2.mockResolvedValueOnce({ error: { message: 'rls denied' } })
      const mockFetch = vi.mocked(fetch)
      mockFetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ status: 'queued' }), { status: 200 }),
      )
      const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        const POST = await importRoute()
        const res = await POST(
          makeReq({
            sessionId: 's1',
            blobUrl: 'https://blob/x.mp4',
            sha256: VALID_SHA,
          }),
        )
        // User still gets a queued response (Railway path took over).
        expect(res.status).toBe(200)
        const body = await res.json()
        expect(body.status).toBe('queued')
      } finally {
        errSpy.mockRestore()
      }
    })
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

    it('returns 401 when no Supabase session is present', async () => {
      // Auth gate: middleware excludes /api/, so without a signed-in
      // user, an unauthenticated caller could otherwise burn T4 GPU
      // credits on Modal indefinitely.
      mockGetUser.mockResolvedValueOnce({ data: { user: null } })
      const mockFetch = vi.mocked(fetch)
      const POST = await importRoute()
      const res = await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(res.status).toBe(401)
      const body = await res.json()
      expect(body.error).toBe('Not signed in')
      // Critical: Modal must NOT be called when auth fails.
      expect(mockFetch).not.toHaveBeenCalled()
    })

    it('returns 401 when auth.getUser throws', async () => {
      // Defense in depth: if Supabase auth itself errors, we treat the
      // user as unauthenticated rather than letting the request through.
      mockGetUser.mockRejectedValueOnce(new Error('supabase down'))
      const mockFetch = vi.mocked(fetch)
      const POST = await importRoute()
      const res = await POST(
        makeReq({ mode: 'batch-ephemeral', blobUrl: 'https://blob/x.mp4' }),
      )
      expect(res.status).toBe(401)
      expect(mockFetch).not.toHaveBeenCalled()
    })

    it('returns 200 when a valid session is present (auth gate passes)', async () => {
      // Mirror image of the 401 test: a signed-in user reaches Modal
      // and gets keypoints back inline.
      mockGetUser.mockResolvedValueOnce({ data: { user: { id: 'user-42' } } })
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
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })

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
        sample_fps: 30,
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
