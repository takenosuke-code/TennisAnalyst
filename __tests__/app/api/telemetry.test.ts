import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { NextRequest } from 'next/server'

/*
 * Phase 0.5 — POST /api/telemetry sanity coverage.
 *
 * The endpoint MUST NEVER throw. Whether the body is malformed JSON, an
 * empty payload, or a missing-`event` payload, the route returns 204 so
 * the calling client's fire-and-forget promise resolves cleanly.
 *
 * We don't need to mock @vercel/blob — the route falls back to console
 * logging when BLOB_READ_WRITE_TOKEN isn't set (which it isn't, in the
 * test env). We just stub a console.log spy so the test output stays
 * clean.
 */

function makeJsonRequest(body: unknown): NextRequest {
  return new NextRequest('http://localhost/api/telemetry', {
    method: 'POST',
    body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json' },
  })
}

function makeRawRequest(raw: string): NextRequest {
  return new NextRequest('http://localhost/api/telemetry', {
    method: 'POST',
    body: raw,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('POST /api/telemetry', () => {
  let logSpy: ReturnType<typeof vi.spyOn>
  let errorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.resetModules()
    vi.unstubAllEnvs()
    // Force the no-blob-token branch so the route just console.logs and
    // returns 204 without trying to hit Vercel Blob.
    vi.stubEnv('BLOB_READ_WRITE_TOKEN', '')
    logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    logSpy.mockRestore()
    errorSpy.mockRestore()
  })

  it('returns 204 when the body is malformed JSON (does not throw)', async () => {
    const { POST } = await import('@/app/api/telemetry/route')
    const req = makeRawRequest('not-actually-json {')
    const res = await POST(req)
    expect(res.status).toBe(204)
  })

  it('returns 204 when no `event` field is present', async () => {
    const { POST } = await import('@/app/api/telemetry/route')
    const req = makeJsonRequest({ sessionId: 'sess-1', cold_start: true })
    const res = await POST(req)
    expect(res.status).toBe(204)
    // The route bails before logging when the payload is invalid.
    expect(logSpy).not.toHaveBeenCalled()
  })

  it('returns 204 with a valid body and logs when blob token is absent', async () => {
    const { POST } = await import('@/app/api/telemetry/route')
    const req = makeJsonRequest({
      event: 'analyze_complete',
      sessionId: 'sess-42',
      cold_start: false,
      inference_ms: 1234,
      provider: 'modal-trt',
    })
    const res = await POST(req)
    expect(res.status).toBe(204)

    // With BLOB_READ_WRITE_TOKEN unset, the route falls back to a single
    // console.log line. Sanity-check the row contains the event name.
    expect(logSpy).toHaveBeenCalledTimes(1)
    const logged = String(logSpy.mock.calls[0][0])
    expect(logged).toContain('analyze_complete')
    expect(logged).toContain('sess-42')
  })

  it('returns 204 for a completely empty body', async () => {
    const { POST } = await import('@/app/api/telemetry/route')
    const req = makeRawRequest('')
    const res = await POST(req)
    expect(res.status).toBe(204)
  })
})
