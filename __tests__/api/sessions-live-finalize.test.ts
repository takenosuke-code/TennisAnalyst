import { describe, it, expect, beforeEach, vi } from 'vitest'
import { NextRequest } from 'next/server'

// --- Supabase mock harness ------------------------------------------------
//
// The finalize route uses two Supabase clients:
//   • the user's RLS-aware client (from createClient in lib/supabase/server)
//     to confirm the session row is visible to the user;
//   • supabaseAdmin (service role) to do the actual UPDATE.
//
// We stub both. The auth client surfaces a select->eq->single chain that
// returns the pre-set ownerRow. The admin client exposes update->eq.

const ownerSingle = vi.fn()
const ownerEq = vi.fn()
const ownerSelect = vi.fn()
const ownerFrom = vi.fn()

const adminEq = vi.fn()
const adminUpdate = vi.fn()
const adminFrom = vi.fn()

function resetMocks() {
  for (const m of [ownerSingle, ownerEq, ownerSelect, ownerFrom, adminEq, adminUpdate, adminFrom]) {
    m.mockReset()
  }
  // owner client default: signed-in user, single() returns { data: ownerRow, error: null }
  ownerEq.mockReturnValue({ single: ownerSingle })
  ownerSelect.mockReturnValue({ eq: ownerEq })
  ownerFrom.mockReturnValue({ select: ownerSelect })

  // admin client default: update().eq() resolves with no error.
  adminEq.mockResolvedValue({ data: null, error: null })
  adminUpdate.mockReturnValue({ eq: adminEq })
  adminFrom.mockReturnValue({ update: adminUpdate })
}

vi.mock('@/lib/supabase', () => ({
  supabaseAdmin: new Proxy({}, {
    get: (_t: object, prop: string) => {
      if (prop === 'from') return adminFrom
      return undefined
    },
  }),
}))

const mockGetUser = vi.fn<() => Promise<{ data: { user: { id: string } | null } }>>(
  async () => ({ data: { user: { id: 'user-1' } } }),
)
vi.mock('@/lib/supabase/server', () => ({
  createClient: async () => ({
    auth: { getUser: mockGetUser },
    from: ownerFrom,
  }),
}))

// --- Helpers --------------------------------------------------------------

function makeRequest(body: Record<string, unknown>) {
  return new NextRequest('http://localhost/api/sessions/live/finalize', {
    method: 'POST',
    body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json' },
  })
}

const FALLBACK_KP = { fps_sampled: 15, frame_count: 30, frames: [{ t: 0, l: [] }] }

// --- Tests ----------------------------------------------------------------

describe('POST /api/sessions/live/finalize', () => {
  beforeEach(() => {
    resetMocks()
    mockGetUser.mockReset()
    mockGetUser.mockResolvedValue({ data: { user: { id: 'user-1' } } })
  })

  it('returns 401 when the user is not signed in', async () => {
    mockGetUser.mockResolvedValueOnce({ data: { user: null } })
    const { POST } = await import('@/app/api/sessions/live/finalize/route')
    const res = await POST(makeRequest({ sessionId: 'sess-1', outcome: 'server-ok' }))
    expect(res.status).toBe(401)
  })

  it('returns 400 when sessionId is missing', async () => {
    const { POST } = await import('@/app/api/sessions/live/finalize/route')
    const res = await POST(makeRequest({ outcome: 'server-ok' }))
    expect(res.status).toBe(400)
  })

  it('returns 400 when outcome is invalid', async () => {
    const { POST } = await import('@/app/api/sessions/live/finalize/route')
    const res = await POST(
      makeRequest({ sessionId: 'sess-1', outcome: 'maybe?' }),
    )
    expect(res.status).toBe(400)
  })

  it("returns 401 when the session is not visible to this user (cross-user access blocked)", async () => {
    // RLS hides the row → single() returns null + an error.
    ownerSingle.mockResolvedValueOnce({ data: null, error: { message: 'no rows' } })
    const { POST } = await import('@/app/api/sessions/live/finalize/route')
    const res = await POST(
      makeRequest({ sessionId: 'sess-other-user', outcome: 'server-ok' }),
    )
    expect(res.status).toBe(401)
  })

  it("server-ok: flips status='complete' WITHOUT touching keypoints_json", async () => {
    ownerSingle.mockResolvedValueOnce({
      data: {
        id: 'sess-1',
        status: 'extracting',
        keypoints_json: { frames: [{ server: true }] },
        fallback_keypoints_json: FALLBACK_KP,
      },
      error: null,
    })

    const { POST } = await import('@/app/api/sessions/live/finalize/route')
    const res = await POST(
      makeRequest({ sessionId: 'sess-1', outcome: 'server-ok' }),
    )
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body.status).toBe('complete')
    expect(body.outcome).toBe('server-ok')

    // The admin update must carry only { status: 'complete' } — not
    // keypoints_json, not fallback_keypoints_json.
    expect(adminUpdate).toHaveBeenCalledTimes(1)
    const updatePayload = adminUpdate.mock.calls[0][0]
    expect(updatePayload).toEqual({ status: 'complete' })
    expect(updatePayload.keypoints_json).toBeUndefined()
    expect(updatePayload.fallback_keypoints_json).toBeUndefined()

    // .eq('id', sessionId) called on the update.
    expect(adminEq).toHaveBeenCalledWith('id', 'sess-1')
  })

  it("server-failed: copies fallback_keypoints_json into keypoints_json and flips status='complete'", async () => {
    ownerSingle.mockResolvedValueOnce({
      data: {
        id: 'sess-2',
        status: 'extracting',
        keypoints_json: null,
        fallback_keypoints_json: FALLBACK_KP,
      },
      error: null,
    })

    const { POST } = await import('@/app/api/sessions/live/finalize/route')
    const res = await POST(
      makeRequest({ sessionId: 'sess-2', outcome: 'server-failed' }),
    )
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body.status).toBe('complete')
    expect(body.outcome).toBe('server-failed')

    expect(adminUpdate).toHaveBeenCalledTimes(1)
    const updatePayload = adminUpdate.mock.calls[0][0]
    expect(updatePayload.status).toBe('complete')
    expect(updatePayload.keypoints_json).toEqual(FALLBACK_KP)

    expect(adminEq).toHaveBeenCalledWith('id', 'sess-2')
  })

  it('server-failed with no fallback stored returns 409 and does NOT update', async () => {
    ownerSingle.mockResolvedValueOnce({
      data: {
        id: 'sess-3',
        status: 'extracting',
        keypoints_json: null,
        fallback_keypoints_json: null,
      },
      error: null,
    })

    const { POST } = await import('@/app/api/sessions/live/finalize/route')
    const res = await POST(
      makeRequest({ sessionId: 'sess-3', outcome: 'server-failed' }),
    )
    expect(res.status).toBe(409)
    expect(adminUpdate).not.toHaveBeenCalled()
  })

  it('returns 500 when the admin update fails', async () => {
    ownerSingle.mockResolvedValueOnce({
      data: {
        id: 'sess-4',
        status: 'extracting',
        keypoints_json: { frames: [{ server: true }] },
        fallback_keypoints_json: FALLBACK_KP,
      },
      error: null,
    })
    adminEq.mockResolvedValueOnce({ data: null, error: { message: 'boom' } })

    const { POST } = await import('@/app/api/sessions/live/finalize/route')
    const res = await POST(
      makeRequest({ sessionId: 'sess-4', outcome: 'server-ok' }),
    )
    expect(res.status).toBe(500)
  })
})
