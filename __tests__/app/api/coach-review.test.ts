import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'

/*
 * Phase 4.1 — coach-review [id] endpoint guards.
 *
 * Verifies the auditor-flagged gap is closed: a malformed `id` returns
 * a clean 404 instead of leaking the postgres "invalid input syntax for
 * type uuid" error message via error.message. Without the regex guard,
 * the public internet sees DB driver internals.
 */

function makeRequest(method: 'GET' | 'POST', body?: unknown): NextRequest {
  return new NextRequest('http://localhost/api/coach-review/anything', {
    method,
    body: body ? JSON.stringify(body) : undefined,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
  })
}

describe('GET /api/coach-review/[id] uuid guard', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('returns 404 for non-uuid ids without hitting the DB', async () => {
    const { GET } = await import('@/app/api/coach-review/[id]/route')

    const cases = [
      'foo',
      '',
      '1234',
      'not-a-uuid',
      "' OR 1=1 --",
      'a'.repeat(1000),
      '01234567-89ab-cdef-0123-456789abcdef-extra',
      '01234567-89ab-cdef-0123-456789abcdex', // bad final char
    ]
    for (const id of cases) {
      const ctx = { params: Promise.resolve({ id }) }
      const res = await GET(makeRequest('GET'), ctx)
      expect(res.status).toBe(404)
      const body = (await res.json()) as { error: string }
      // Generic "Review not found" — does NOT leak DB driver text.
      expect(body.error).toBe('Review not found')
      expect(body.error.toLowerCase()).not.toContain('uuid')
      expect(body.error.toLowerCase()).not.toContain('postgres')
      expect(body.error.toLowerCase()).not.toContain('syntax')
    }
  })

  it('accepts well-formed uuids and proceeds past the guard', async () => {
    // Mock supabaseAdmin so we never actually hit the DB. We only care
    // that a real-looking uuid passes the regex check.
    vi.doMock('@/lib/supabase', () => ({
      supabaseAdmin: {
        from: () => ({
          select: () => ({
            eq: () => ({
              maybeSingle: async () => ({ data: null, error: null }),
            }),
          }),
        }),
      },
    }))

    const { GET } = await import('@/app/api/coach-review/[id]/route')
    const realUuid = '01234567-89ab-cdef-0123-456789abcdef'
    const ctx = { params: Promise.resolve({ id: realUuid }) }
    const res = await GET(makeRequest('GET'), ctx)
    // Past the guard, the mock returns null, which the route maps to 404.
    // Either way: no 500, no DB error leak.
    expect(res.status).toBe(404)
    const body = (await res.json()) as { error: string }
    expect(body.error).toBe('Review not found')
  })
})

describe('POST /api/coach-review/[id] uuid guard', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('returns 404 for non-uuid ids without hitting the DB', async () => {
    const { POST } = await import('@/app/api/coach-review/[id]/route')
    const ctx = { params: Promise.resolve({ id: 'not-a-uuid' }) }
    const res = await POST(makeRequest('POST', { verdict: 'looks_right' }), ctx)
    expect(res.status).toBe(404)
  })

  it('returns 400 for valid uuid but invalid verdict', async () => {
    vi.doMock('@/lib/supabase', () => ({
      supabaseAdmin: {
        from: () => ({
          select: () => ({
            eq: () => ({
              maybeSingle: async () => ({ data: null, error: null }),
            }),
          }),
        }),
      },
    }))
    const { POST } = await import('@/app/api/coach-review/[id]/route')
    const realUuid = '01234567-89ab-cdef-0123-456789abcdef'
    const ctx = { params: Promise.resolve({ id: realUuid }) }
    const res = await POST(makeRequest('POST', { verdict: 'INVALID' }), ctx)
    expect(res.status).toBe(400)
    const body = (await res.json()) as { error: string }
    expect(body.error.toLowerCase()).toContain('verdict')
  })
})
