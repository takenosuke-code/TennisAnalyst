import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { NextRequest } from 'next/server'

// The route reads RAILWAY_SERVICE_URL and EXTRACT_API_KEY at module scope,
// so every test resets the module and stubs env freshly before importing.
// Mirrors __tests__/api/process-youtube.test.ts.

const MOCK_RAILWAY_URL = 'https://railway.test.example.com'
const MOCK_API_KEY = 'test-extract-key'

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
})
