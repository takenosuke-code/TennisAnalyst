import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest'
import { NextRequest } from 'next/server'

// ---------------------------------------------------------------------------
// POST /api/baselines/from-segment
//
// Mocks:
//   - @/lib/supabase/server createClient -> the user-scoped SSR client
//     (auth + user_baselines writes)
//   - @/lib/supabase supabaseAdmin -> the service-role client used to
//     fetch user_sessions + video_segments (user_sessions has no user_id
//     column today, so we intentionally bypass RLS for reads)
//   - global fetch -> Railway /trim-video
//
// Strategy mirrors __tests__/api/analyze-structured.test.ts: swap the
// supabase mocks per test with a small helper, and reset modules so env
// stubs land in the fresh route module.
// ---------------------------------------------------------------------------

const TRIMMED_BLOB_URL = 'https://abc.public.blob.vercel-storage.com/baseline-trims/x.mp4'
const SESSION_ID = '00000000-0000-0000-0000-000000000001'
const SEGMENT_ID = '00000000-0000-0000-0000-000000000002'

type AuthGetUserResult = { data: { user: { id: string } | null }; error: null }

interface AuthClientMock {
  auth: { getUser: () => Promise<AuthGetUserResult> }
  from: (table: string) => unknown
}

// Capture state so tests can assert on the update/insert calls made
// against user_baselines via the SSR client.
type CapturedCall = { table: string; op: string; args: unknown[] }
let captured: CapturedCall[] = []
let insertResult: { data: unknown; error: unknown } = {
  data: { id: 'new-baseline-id', shot_type: 'forehand' },
  error: null,
}
let deactivateError: { message: string } | null = null

let currentUser: { id: string } | null = { id: 'user-1' }
let sessionRow: unknown = null
let segmentRow: unknown = null
let sessionError: unknown = null
let segmentError: unknown = null

function buildAuthClient(): AuthClientMock {
  return {
    auth: {
      getUser: async () => ({ data: { user: currentUser }, error: null }),
    },
    from(table: string) {
      if (table !== 'user_baselines') {
        throw new Error(`Unexpected table on auth client: ${table}`)
      }
      const builder = {
        update(row: Record<string, unknown>) {
          captured.push({ table, op: 'update', args: [row] })
          return {
            eq(col: string, val: unknown) {
              captured.push({ table, op: 'update.eq', args: [col, val] })
              return {
                eq: async (col2: string, val2: unknown) => {
                  captured.push({ table, op: 'update.eq.eq', args: [col2, val2] })
                  return { error: deactivateError }
                },
              }
            },
          }
        },
        insert(row: Record<string, unknown>) {
          captured.push({ table, op: 'insert', args: [row] })
          return {
            select: () => ({
              single: async () => insertResult,
            }),
          }
        },
      }
      return builder
    },
  }
}

function buildAdminClient() {
  return {
    from(table: string) {
      if (table === 'user_sessions') {
        return {
          select: () => ({
            eq: () => ({
              single: async () => ({ data: sessionRow, error: sessionError }),
            }),
          }),
        }
      }
      if (table === 'video_segments') {
        return {
          select: () => ({
            eq: () => ({
              eq: () => ({
                single: async () => ({ data: segmentRow, error: segmentError }),
              }),
            }),
          }),
        }
      }
      throw new Error(`Unexpected table on admin client: ${table}`)
    },
  }
}

vi.mock('@/lib/supabase/server', () => ({
  createClient: vi.fn(async () => buildAuthClient()),
}))

vi.mock('@/lib/supabase', () => ({
  supabase: buildAdminClient(),
  supabaseAdmin: buildAdminClient(),
}))

const originalFetch = globalThis.fetch
let fetchMock: ReturnType<typeof vi.fn>

function makeFrames(startIdx: number, count: number) {
  return Array.from({ length: count }, (_, i) => ({
    frame_index: startIdx + i,
    timestamp_ms: (startIdx + i) * 33,
    landmarks: [],
    joint_angles: { right_elbow: 120 },
  }))
}

beforeEach(() => {
  vi.resetModules()
  captured = []
  currentUser = { id: 'user-1' }
  sessionRow = {
    id: SESSION_ID,
    blob_url: 'https://abc.public.blob.vercel-storage.com/src.mp4',
    keypoints_json: {
      fps_sampled: 30,
      frame_count: 20,
      frames: makeFrames(0, 20),
    },
  }
  segmentRow = {
    id: SEGMENT_ID,
    session_id: SESSION_ID,
    segment_index: 0,
    shot_type: 'backhand',
    start_frame: 5,
    end_frame: 12,
    start_ms: 200,
    end_ms: 1500,
    confidence: 0.82,
    label: null,
    keypoints_json: {
      fps_sampled: 30,
      frame_count: 8,
      frames: makeFrames(5, 8),
    },
  }
  sessionError = null
  segmentError = null
  insertResult = {
    data: { id: 'new-baseline-id', shot_type: 'forehand' },
    error: null,
  }
  deactivateError = null

  // Default: Railway responds with a plausible trimmed URL
  fetchMock = vi.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => ({ blob_url: TRIMMED_BLOB_URL }),
    text: async () => '',
  }))
  globalThis.fetch = fetchMock as unknown as typeof fetch

  // Required for the route to call Railway at all
  process.env.RAILWAY_SERVICE_URL = 'https://railway.test'
  process.env.EXTRACT_API_KEY = 'test-key'
})

async function callRoute(body: Record<string, unknown>) {
  const { POST } = await import('@/app/api/baselines/from-segment/route')
  const req = new NextRequest('http://localhost/api/baselines/from-segment', {
    method: 'POST',
    body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json' },
  })
  return POST(req)
}

describe('POST /api/baselines/from-segment', () => {
  it('returns 401 when not signed in', async () => {
    currentUser = null
    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(401)
  })

  it('returns 400 when sessionId is missing or malformed', async () => {
    const res = await callRoute({ segmentId: SEGMENT_ID })
    expect(res.status).toBe(400)
    const j = await res.json()
    expect(j.error).toMatch(/sessionId/)

    const res2 = await callRoute({ sessionId: 'not-a-uuid', segmentId: SEGMENT_ID })
    expect(res2.status).toBe(400)
  })

  it('returns 400 when segmentId is missing', async () => {
    const res = await callRoute({ sessionId: SESSION_ID })
    expect(res.status).toBe(400)
  })

  it('returns 400 on invalid shotTypeOverride', async () => {
    const res = await callRoute({
      sessionId: SESSION_ID,
      segmentId: SEGMENT_ID,
      shotTypeOverride: 'not-a-shot',
    })
    expect(res.status).toBe(400)
    const j = await res.json()
    expect(j.error).toMatch(/shotTypeOverride/)
  })

  it('returns 404 when session is missing', async () => {
    sessionRow = null
    sessionError = { message: 'not found' }
    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(404)
  })

  it('returns 404 when segment is missing', async () => {
    segmentRow = null
    segmentError = { message: 'not found' }
    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(404)
  })

  it('returns 400 when segment shot_type is unbaselinable and no override', async () => {
    segmentRow = {
      ...(segmentRow as object),
      shot_type: 'unknown',
    }
    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(400)
  })

  it('happy path: runs deactivate update, calls Railway, inserts new baseline', async () => {
    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(200)
    const j = await res.json()
    expect(j.baseline).toEqual({ id: 'new-baseline-id', shot_type: 'forehand' })

    // 1) Railway was called with the segment's ms range and the session's blob_url
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const fetchArgs = fetchMock.mock.calls[0]
    expect(String(fetchArgs[0])).toContain('/trim-video')
    const sentBody = JSON.parse((fetchArgs[1] as { body: string }).body)
    expect(sentBody).toMatchObject({
      video_url: 'https://abc.public.blob.vercel-storage.com/src.mp4',
      start_ms: 200,
      end_ms: 1500,
    })

    // 2) Sibling deactivate UPDATE fired against user_baselines scoped to
    //    the resolved shot_type ('backhand' from the segment) with is_active=true.
    const updateCalls = captured.filter((c) => c.op === 'update')
    expect(updateCalls).toHaveLength(1)
    expect(updateCalls[0].args[0]).toMatchObject({ is_active: false })

    const eqCalls = captured.filter((c) => c.op.startsWith('update.eq'))
    const eqArgs = eqCalls.map((c) => c.args)
    expect(eqArgs).toContainEqual(['shot_type', 'backhand'])
    expect(eqArgs).toContainEqual(['is_active', true])

    // 3) Insert runs with the trimmed blob_url, shot_type from segment,
    //    keypoints_json carried through, and is_active=true.
    const insertCalls = captured.filter((c) => c.op === 'insert')
    expect(insertCalls).toHaveLength(1)
    const row = insertCalls[0].args[0] as Record<string, unknown>
    expect(row).toMatchObject({
      user_id: 'user-1',
      shot_type: 'backhand',
      blob_url: TRIMMED_BLOB_URL,
      is_active: true,
      source_session_id: SESSION_ID,
    })
    const kp = row.keypoints_json as { frames: unknown[] }
    expect(kp.frames.length).toBe(8)
  })

  it('shotTypeOverride replaces the classifier label on the inserted row', async () => {
    const res = await callRoute({
      sessionId: SESSION_ID,
      segmentId: SEGMENT_ID,
      shotTypeOverride: 'forehand',
    })
    expect(res.status).toBe(200)

    const insertRow = (captured.find((c) => c.op === 'insert')!.args[0]) as Record<string, unknown>
    expect(insertRow.shot_type).toBe('forehand')

    // Sibling deactivate runs against the OVERRIDE shot_type, not the classifier's.
    const eqArgs = captured.filter((c) => c.op.startsWith('update.eq')).map((c) => c.args)
    expect(eqArgs).toContainEqual(['shot_type', 'forehand'])
  })

  it('slices the parent session keypoints when the segment has no keypoints_json', async () => {
    segmentRow = {
      ...(segmentRow as object),
      keypoints_json: null,
    }

    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(200)

    const insertRow = (captured.find((c) => c.op === 'insert')!.args[0]) as Record<string, unknown>
    const kp = insertRow.keypoints_json as { frames: Array<{ frame_index: number }> }
    // start_frame=5, end_frame=12 (inclusive) -> 8 frames
    expect(kp.frames.length).toBe(8)
    expect(kp.frames[0].frame_index).toBe(5)
    expect(kp.frames[kp.frames.length - 1].frame_index).toBe(12)
  })

  it('returns 502 when Railway trim fails', async () => {
    fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
      json: async () => ({}),
      text: async () => 'ffmpeg crashed',
    }))
    globalThis.fetch = fetchMock as unknown as typeof fetch

    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(502)
  })

  it('returns 502 when Railway returns a disallowed blob URL host', async () => {
    fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ blob_url: 'https://evil.example.com/foo.mp4' }),
      text: async () => '',
    }))
    globalThis.fetch = fetchMock as unknown as typeof fetch

    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(502)
  })

  it('returns 503 when Railway env vars are missing', async () => {
    delete process.env.RAILWAY_SERVICE_URL
    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(503)
  })

  it('uses provided label when supplied', async () => {
    const res = await callRoute({
      sessionId: SESSION_ID,
      segmentId: SEGMENT_ID,
      label: 'my great serve',
    })
    expect(res.status).toBe(200)

    const insertRow = (captured.find((c) => c.op === 'insert')!.args[0]) as Record<string, unknown>
    expect(insertRow.label).toBe('my great serve')
  })

  it('falls back to a default label like "Segment #N — <shotType>" when none given', async () => {
    segmentRow = {
      ...(segmentRow as object),
      segment_index: 2,
      label: null,
    }
    const res = await callRoute({ sessionId: SESSION_ID, segmentId: SEGMENT_ID })
    expect(res.status).toBe(200)
    const insertRow = (captured.find((c) => c.op === 'insert')!.args[0]) as Record<string, unknown>
    // segment_index 2 -> "Segment #3"
    expect(insertRow.label).toMatch(/^Segment #3/)
    expect(insertRow.label).toMatch(/backhand/)
  })
})

// Restore global fetch so downstream tests aren't polluted
afterAll(() => {
  globalThis.fetch = originalFetch
})
