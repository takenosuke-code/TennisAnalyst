import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'

// Mock supabase
const mockSelect = vi.fn()
const mockSingle = vi.fn()
const mockEq = vi.fn()
const mockGt = vi.fn()
const mockUpdate = vi.fn()
const mockInsert = vi.fn()
const mockUpsert = vi.fn()
const mockFrom = vi.fn()

function chainMock() {
  mockSelect.mockReturnThis()
  mockSingle.mockReturnThis()
  mockEq.mockReturnThis()
  mockGt.mockReturnThis()
  mockUpdate.mockReturnThis()
  mockInsert.mockReturnThis()
  mockUpsert.mockReturnThis()
  mockFrom.mockReturnValue({
    select: mockSelect,
    single: mockSingle,
    eq: mockEq,
    gt: mockGt,
    update: mockUpdate,
    insert: mockInsert,
    upsert: mockUpsert,
  })
  // Chain returns for nested calls
  mockUpdate.mockReturnValue({ eq: mockEq })
  mockEq.mockReturnValue({ eq: mockEq, select: mockSelect, single: mockSingle, gt: mockGt })
  mockSelect.mockReturnValue({ single: mockSingle, eq: mockEq })
  mockGt.mockReturnValue({ single: mockSingle })
  mockUpsert.mockReturnValue({ select: mockSelect })
}

vi.mock('@/lib/supabase', () => {
  // Single proxy reused for both `supabase` (anon, used by /api/sessions/[id]
  // GET) and `supabaseAdmin` (service-role, used by POST /api/sessions for
  // pending-row creation). Tests mock the chain on whichever client the
  // route happens to import.
  const proxy = new Proxy({}, {
    get: (_t: object, prop: string) => {
      if (prop === 'from') return mockFrom
      return undefined
    },
  })
  return { supabase: proxy, supabaseAdmin: proxy }
})

describe('POST /api/sessions', () => {
  beforeEach(() => {
    vi.resetModules()
    chainMock()
  })

  it('returns 400 when blobUrl is missing', async () => {
    const { POST } = await import('@/app/api/sessions/route')
    const req = new NextRequest('http://localhost/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ keypointsJson: {} }),
    })
    const res = await POST(req)
    expect(res.status).toBe(400)
  })

  it('creates a pending session when keypointsJson is omitted (Railway path)', async () => {
    mockSingle.mockResolvedValueOnce({
      data: { id: 'pending-id', status: 'pending' },
      error: null,
    })

    const { POST } = await import('@/app/api/sessions/route')
    const req = new NextRequest('http://localhost/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ blobUrl: 'https://blob.test/v.mp4' }),
    })
    const res = await POST(req)
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body.sessionId).toBe('pending-id')
    // Upsert must carry status='pending' and must NOT carry keypoints_json
    // (Railway will fill that in when extraction completes).
    expect(mockUpsert).toHaveBeenCalledWith(
      expect.objectContaining({
        blob_url: 'https://blob.test/v.mp4',
        status: 'pending',
      }),
      { onConflict: 'blob_url' },
    )
    const upsertCall = mockUpsert.mock.calls[0][0]
    expect(upsertCall.keypoints_json).toBeUndefined()
  })

  it('uses upsert with onConflict blob_url when no sessionId', async () => {
    mockSingle.mockResolvedValueOnce({
      data: { id: 'new-session-id' },
      error: null,
    })

    const { POST } = await import('@/app/api/sessions/route')
    const req = new NextRequest('http://localhost/api/sessions', {
      method: 'POST',
      body: JSON.stringify({
        blobUrl: 'https://blob.test/v.mp4',
        keypointsJson: { fps_sampled: 30, frame_count: 1, frames: [] },
        shotType: 'forehand',
      }),
    })
    const res = await POST(req)
    expect(res.status).toBe(200)
    expect(mockUpsert).toHaveBeenCalledWith(
      expect.objectContaining({ blob_url: 'https://blob.test/v.mp4' }),
      { onConflict: 'blob_url' }
    )
  })
})

describe('GET /api/sessions/[id]', () => {
  beforeEach(() => {
    vi.resetModules()
    chainMock()
  })

  it('filters by expires_at', async () => {
    mockSingle.mockResolvedValueOnce({
      data: { id: 'sess-1', status: 'complete' },
      error: null,
    })

    const { GET } = await import('@/app/api/sessions/[id]/route')
    const req = new NextRequest('http://localhost/api/sessions/sess-1')
    await GET(req, { params: Promise.resolve({ id: 'sess-1' }) })

    expect(mockGt).toHaveBeenCalledWith('expires_at', expect.any(String))
  })

  it('returns 404 for expired or missing sessions', async () => {
    mockSingle.mockResolvedValueOnce({
      data: null,
      error: { message: 'not found' },
    })

    const { GET } = await import('@/app/api/sessions/[id]/route')
    const req = new NextRequest('http://localhost/api/sessions/sess-1')
    const res = await GET(req, { params: Promise.resolve({ id: 'sess-1' }) })
    expect(res.status).toBe(404)
  })
})
