import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'

const mockSingle = vi.fn()
const mockEq = vi.fn()
const mockSelect = vi.fn()
const mockFrom = vi.fn()

vi.mock('@/lib/supabase', () => ({
  supabase: new Proxy({}, {
    get: (_t: object, prop: string) => {
      if (prop === 'from') return mockFrom
      return undefined
    },
  }),
}))

function chainMock() {
  mockFrom.mockReturnValue({ select: mockSelect })
  mockSelect.mockReturnValue({ eq: mockEq })
  mockEq.mockReturnValue({ single: mockSingle })
}

describe('GET /api/pro-swings/[id]', () => {
  beforeEach(() => {
    vi.resetModules()
    mockFrom.mockClear()
    mockSelect.mockClear()
    mockEq.mockClear()
    mockSingle.mockClear()
    chainMock()
  })

  it('excludes keypoints_json by default', async () => {
    mockSingle.mockResolvedValueOnce({
      data: { id: 'sw-1', shot_type: 'forehand' },
      error: null,
    })

    const { GET } = await import('@/app/api/pro-swings/[id]/route')
    const req = new NextRequest('http://localhost/api/pro-swings/sw-1')
    await GET(req, { params: Promise.resolve({ id: 'sw-1' }) })

    const selectArg = mockSelect.mock.calls[0][0] as string
    expect(selectArg).not.toContain('keypoints_json')
    expect(selectArg).toContain('pros(*)')
  })

  it('includes keypoints_json when ?include=keypoints', async () => {
    mockSingle.mockResolvedValueOnce({
      data: { id: 'sw-1', shot_type: 'forehand', keypoints_json: {} },
      error: null,
    })

    const { GET } = await import('@/app/api/pro-swings/[id]/route')
    const req = new NextRequest('http://localhost/api/pro-swings/sw-1?include=keypoints')
    await GET(req, { params: Promise.resolve({ id: 'sw-1' }) })

    const selectArg = mockSelect.mock.calls[0][0] as string
    expect(selectArg).toContain('keypoints_json')
  })

  it('returns 404 when swing not found', async () => {
    mockSingle.mockResolvedValueOnce({
      data: null,
      error: { message: 'not found' },
    })

    const { GET } = await import('@/app/api/pro-swings/[id]/route')
    const req = new NextRequest('http://localhost/api/pro-swings/sw-1')
    const res = await GET(req, { params: Promise.resolve({ id: 'sw-1' }) })
    expect(res.status).toBe(404)
  })
})
