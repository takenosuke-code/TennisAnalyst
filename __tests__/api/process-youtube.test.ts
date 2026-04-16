import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { NextRequest } from 'next/server'

// ---------------------------------------------------------------------------
// Mocking strategy:
// The route files read process.env at module scope (const RAILWAY_SERVICE_URL = ...),
// so we use vi.stubEnv to control env vars before dynamically importing the routes.
// Global fetch is mocked via vi.fn() to intercept calls to the Railway service.
// ---------------------------------------------------------------------------

const MOCK_RAILWAY_URL = 'https://railway.test.example.com'
const MOCK_API_KEY = 'test-extract-key'

// Helper: build a NextRequest for the POST route
function makePostRequest(body: unknown): NextRequest {
  return new NextRequest('http://localhost:3000/api/process-youtube', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

// Helper: dynamically import the POST handler (fresh module per test group)
async function importPostRoute() {
  const mod = await import('@/app/api/process-youtube/route')
  return mod.POST
}

// Helper: dynamically import the GET handler
async function importGetRoute() {
  const mod = await import('@/app/api/process-youtube/[jobId]/route')
  return mod.GET
}

// ---------------------------------------------------------------------------
// POST /api/process-youtube
// ---------------------------------------------------------------------------
describe('POST /api/process-youtube', () => {
  let POST: Awaited<ReturnType<typeof importPostRoute>>

  beforeEach(async () => {
    vi.resetModules()
    vi.stubGlobal('fetch', vi.fn())
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)
    POST = await importPostRoute()
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
  })

  // ---- env var validation ----

  it('returns 503 when RAILWAY_SERVICE_URL is missing', async () => {
    vi.resetModules()
    vi.stubEnv('RAILWAY_SERVICE_URL', '')
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)
    const handler = await importPostRoute()

    const req = makePostRequest({ youtubeUrl: 'https://www.youtube.com/watch?v=abc' })
    const res = await handler(req)

    expect(res.status).toBe(503)
    const json = await res.json()
    expect(json.error).toMatch(/not configured/i)
  })

  it('returns 503 when EXTRACT_API_KEY is missing', async () => {
    vi.resetModules()
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', '')
    const handler = await importPostRoute()

    const req = makePostRequest({ youtubeUrl: 'https://www.youtube.com/watch?v=abc' })
    const res = await handler(req)

    expect(res.status).toBe(503)
    const json = await res.json()
    expect(json.error).toMatch(/not configured/i)
  })

  // ---- request body validation ----

  it('returns 400 when youtubeUrl is missing from the body', async () => {
    const req = makePostRequest({})
    const res = await POST(req)

    expect(res.status).toBe(400)
    const json = await res.json()
    expect(json.error).toMatch(/youtubeUrl/i)
  })

  it('returns 400 when youtubeUrl is empty string', async () => {
    const req = makePostRequest({ youtubeUrl: '' })
    const res = await POST(req)

    expect(res.status).toBe(400)
    const json = await res.json()
    expect(json.error).toMatch(/youtubeUrl/i)
  })

  it('returns 400 when youtubeUrl is not a string', async () => {
    const req = makePostRequest({ youtubeUrl: 12345 })
    const res = await POST(req)

    expect(res.status).toBe(400)
    const json = await res.json()
    expect(json.error).toMatch(/youtubeUrl/i)
  })

  // ---- YouTube URL pattern validation ----

  it('returns 400 for non-YouTube URLs', async () => {
    const req = makePostRequest({ youtubeUrl: 'https://vimeo.com/123456' })
    const res = await POST(req)

    expect(res.status).toBe(400)
    const json = await res.json()
    expect(json.error).toMatch(/invalid youtube url/i)
  })

  it('accepts standard youtube.com watch URLs', async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: 'job-123', status: 'queued' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    vi.stubGlobal('fetch', mockFetch)

    const req = makePostRequest({ youtubeUrl: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' })
    const res = await POST(req)

    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.job_id).toBe('job-123')
  })

  it('accepts youtu.be short URLs', async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: 'job-456', status: 'queued' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    vi.stubGlobal('fetch', mockFetch)

    const req = makePostRequest({ youtubeUrl: 'https://youtu.be/dQw4w9WgXcQ' })
    const res = await POST(req)

    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.job_id).toBe('job-456')
  })

  // ---- proxy forwarding ----

  it('forwards youtubeUrl, targetShotTypes, and maxDuration to Railway service', async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: 'job-789' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    vi.stubGlobal('fetch', mockFetch)

    const req = makePostRequest({
      youtubeUrl: 'https://www.youtube.com/watch?v=abc123',
      targetShotTypes: ['forehand', 'backhand'],
      maxDuration: 300,
    })
    await POST(req)

    expect(mockFetch).toHaveBeenCalledTimes(1)
    const [url, options] = mockFetch.mock.calls[0]
    expect(url).toBe(`${MOCK_RAILWAY_URL}/process-youtube`)
    expect(options.method).toBe('POST')
    expect(options.headers['Content-Type']).toBe('application/json')
    expect(options.headers['Authorization']).toBe(`Bearer ${MOCK_API_KEY}`)

    const sentBody = JSON.parse(options.body)
    expect(sentBody.youtube_url).toBe('https://www.youtube.com/watch?v=abc123')
    expect(sentBody.target_shot_types).toEqual(['forehand', 'backhand'])
    expect(sentBody.max_duration).toBe(300)
  })

  it('defaults maxDuration to 600 when not provided', async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: 'job-default' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    vi.stubGlobal('fetch', mockFetch)

    const req = makePostRequest({ youtubeUrl: 'https://www.youtube.com/watch?v=abc' })
    await POST(req)

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody.max_duration).toBe(600)
    expect(sentBody.target_shot_types).toBeNull()
  })

  // ---- error forwarding ----

  it('forwards Railway service error status and detail message', async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Video too long' }), {
        status: 422,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    vi.stubGlobal('fetch', mockFetch)

    const req = makePostRequest({ youtubeUrl: 'https://www.youtube.com/watch?v=abc' })
    const res = await POST(req)

    expect(res.status).toBe(422)
    const json = await res.json()
    expect(json.error).toBe('Video too long')
  })

  it('returns 502 when fetch to Railway service throws (network error)', async () => {
    const mockFetch = vi.fn().mockRejectedValue(new Error('ECONNREFUSED'))
    vi.stubGlobal('fetch', mockFetch)

    const req = makePostRequest({ youtubeUrl: 'https://www.youtube.com/watch?v=abc' })
    const res = await POST(req)

    expect(res.status).toBe(502)
    const json = await res.json()
    expect(json.error).toMatch(/failed to reach/i)
  })

  it('handles Railway error response with unparseable JSON body', async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response('Internal Server Error', {
        status: 500,
        headers: { 'Content-Type': 'text/plain' },
      })
    )
    vi.stubGlobal('fetch', mockFetch)

    const req = makePostRequest({ youtubeUrl: 'https://www.youtube.com/watch?v=abc' })
    const res = await POST(req)

    // The route catches json() parse failure and falls back to 'Unknown error'
    expect(res.status).toBe(500)
    const json = await res.json()
    expect(json.error).toBeDefined()
  })
})

// ---------------------------------------------------------------------------
// GET /api/process-youtube/[jobId]
// ---------------------------------------------------------------------------
describe('GET /api/process-youtube/[jobId]', () => {
  let GET: Awaited<ReturnType<typeof importGetRoute>>

  beforeEach(async () => {
    vi.resetModules()
    vi.stubGlobal('fetch', vi.fn())
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)
    GET = await importGetRoute()
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
  })

  // Helper: build a NextRequest for the GET route with jobId via params
  function makeGetRequest(jobId: string): [NextRequest, { params: Promise<{ jobId: string }> }] {
    const req = new NextRequest(
      `http://localhost:3000/api/process-youtube/${encodeURIComponent(jobId)}`,
      { method: 'GET' }
    )
    return [req, { params: Promise.resolve({ jobId }) }]
  }

  it('returns 503 when env vars are missing', async () => {
    vi.resetModules()
    vi.stubEnv('RAILWAY_SERVICE_URL', '')
    vi.stubEnv('EXTRACT_API_KEY', '')
    const handler = await importGetRoute()

    const [req, ctx] = makeGetRequest('job-1')
    const res = await handler(req, ctx)

    expect(res.status).toBe(503)
    const json = await res.json()
    expect(json.error).toMatch(/not configured/i)
  })

  it('proxies job status from Railway service', async () => {
    const statusPayload = { job_id: 'job-1', status: 'processing', progress: 45 }
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(statusPayload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    vi.stubGlobal('fetch', mockFetch)

    const [req, ctx] = makeGetRequest('job-1')
    const res = await GET(req, ctx)

    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.job_id).toBe('job-1')
    expect(json.status).toBe('processing')
    expect(json.progress).toBe(45)

    // Verify the URL sent to Railway includes the jobId
    expect(mockFetch).toHaveBeenCalledTimes(1)
    const [calledUrl, calledOpts] = mockFetch.mock.calls[0]
    expect(calledUrl).toBe(`${MOCK_RAILWAY_URL}/process-youtube/job-1`)
    expect(calledOpts.headers['Authorization']).toBe(`Bearer ${MOCK_API_KEY}`)
  })

  it('URL-encodes the jobId when forwarding to Railway', async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'done' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    vi.stubGlobal('fetch', mockFetch)

    const [req, ctx] = makeGetRequest('job/with spaces')
    await GET(req, ctx)

    const calledUrl = mockFetch.mock.calls[0][0]
    expect(calledUrl).toBe(`${MOCK_RAILWAY_URL}/process-youtube/job%2Fwith%20spaces`)
  })

  it('forwards Railway error status for unknown job', async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Job not found' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    vi.stubGlobal('fetch', mockFetch)

    const [req, ctx] = makeGetRequest('nonexistent-job')
    const res = await GET(req, ctx)

    expect(res.status).toBe(404)
    const json = await res.json()
    expect(json.error).toBe('Job not found')
  })

  it('returns 502 when fetch to Railway throws', async () => {
    const mockFetch = vi.fn().mockRejectedValue(new Error('Network timeout'))
    vi.stubGlobal('fetch', mockFetch)

    const [req, ctx] = makeGetRequest('job-timeout')
    const res = await GET(req, ctx)

    expect(res.status).toBe(502)
    const json = await res.json()
    expect(json.error).toMatch(/failed to reach/i)
  })
})

// ---------------------------------------------------------------------------
// Consistency checks: error format matches other routes
// ---------------------------------------------------------------------------
describe('error response format consistency', () => {
  beforeEach(async () => {
    vi.resetModules()
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
  })

  it('all error responses use { error: string } shape (POST route)', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', MOCK_RAILWAY_URL)
    vi.stubEnv('EXTRACT_API_KEY', MOCK_API_KEY)
    const POST = await importPostRoute()

    // Missing youtubeUrl -> 400
    const req = makePostRequest({})
    const res = await POST(req)
    const json = await res.json()

    expect(json).toHaveProperty('error')
    expect(typeof json.error).toBe('string')
    // Should NOT have unexpected keys like 'message' or 'detail' at the top level
    const keys = Object.keys(json)
    expect(keys).toEqual(['error'])
  })

  it('all error responses use { error: string } shape (GET route)', async () => {
    vi.stubEnv('RAILWAY_SERVICE_URL', '')
    vi.stubEnv('EXTRACT_API_KEY', '')
    const GET = await importGetRoute()

    const req = new NextRequest('http://localhost:3000/api/process-youtube/job-1', {
      method: 'GET',
    })
    const res = await GET(req, { params: Promise.resolve({ jobId: 'job-1' }) })
    const json = await res.json()

    expect(json).toHaveProperty('error')
    expect(typeof json.error).toBe('string')
    const keys = Object.keys(json)
    expect(keys).toEqual(['error'])
  })
})
