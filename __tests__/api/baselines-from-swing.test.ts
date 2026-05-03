import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest'
import { NextRequest } from 'next/server'

// ---------------------------------------------------------------------------
// POST /api/baselines/from-swing
//
// Mirror of __tests__/api/baselines-from-segment.test.ts with the
// frame-bounded trim variant. Exercises the bug fix from 2026-05:
// SwingBaselineGrid was POSTing untrimmed full-video URLs as baselines,
// so a 17s rally with a 3s detected swing landed as a 17s baseline.
// This route trims via Railway and re-zeroes pose timestamps so the
// saved baseline plays as just the swing window.
// ---------------------------------------------------------------------------

const TRIMMED_BLOB_URL = 'https://abc.public.blob.vercel-storage.com/baseline-trims/x.mp4'
const SESSION_ID = '00000000-0000-0000-0000-000000000001'

interface AuthClientMock {
  auth: { getUser: () => Promise<{ data: { user: { id: string } | null }; error: null }> }
  from: (table: string) => unknown
}

type CapturedCall = { table: string; op: string; args: unknown[] }
let captured: CapturedCall[] = []
let insertResult: { data: unknown; error: unknown } = {
  data: { id: 'new-baseline-id', shot_type: 'forehand' },
  error: null,
}
let deactivateError: { message: string } | null = null

let currentUser: { id: string } | null = { id: 'user-1' }
let sessionRow: unknown = null
let sessionError: unknown = null

function buildAuthClient(): AuthClientMock {
  return {
    auth: { getUser: async () => ({ data: { user: currentUser }, error: null }) },
    from(table: string) {
      if (table !== 'user_baselines') {
        throw new Error(`Unexpected table on auth client: ${table}`)
      }
      return {
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
            select: () => ({ single: async () => insertResult }),
          }
        },
      }
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

// @vercel/blob's put() is the upload-side primitive we now use to write
// the trimmed mp4 to Vercel Blob (Railway used to do this, but its
// hand-rolled Python upload protocol broke and we routed around it).
// The mock just returns a fake URL on the allowed host so the route's
// validation passes.
let putMock: ReturnType<typeof vi.fn>
vi.mock('@vercel/blob', () => ({
  put: (...args: unknown[]) => putMock(...args),
}))

const originalFetch = globalThis.fetch
let fetchMock: ReturnType<typeof vi.fn>

function makeFrames(startIdx: number, count: number, fps = 30) {
  const dt = 1000 / fps
  return Array.from({ length: count }, (_, i) => ({
    frame_index: startIdx + i,
    timestamp_ms: Math.round((startIdx + i) * dt),
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
      frame_count: 600,
      frames: makeFrames(0, 600),
    },
  }
  sessionError = null
  insertResult = {
    data: { id: 'new-baseline-id', shot_type: 'forehand' },
    error: null,
  }
  deactivateError = null

  // Default Railway response: 200 OK with a small mp4 byte payload.
  // The route streams response.body into put(); we just need a body
  // that satisfies `if (!trimResp.body)` (any ReadableStream works).
  fetchMock = vi.fn(async () => {
    const fakeBytes = new Uint8Array([0, 0, 0, 0])
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(fakeBytes)
        controller.close()
      },
    })
    return {
      ok: true,
      status: 200,
      body: stream,
      json: async () => ({}),
      text: async () => '',
    } as unknown as Response
  })
  globalThis.fetch = fetchMock as unknown as typeof fetch

  // Default put() response: returns the same TRIMMED_BLOB_URL so the
  // route's allowlist check passes.
  putMock = vi.fn(async () => ({
    url: TRIMMED_BLOB_URL,
    pathname: 'baseline-trims/x.mp4',
    contentType: 'video/mp4',
    contentDisposition: 'inline',
    downloadUrl: TRIMMED_BLOB_URL,
  }))

  process.env.RAILWAY_SERVICE_URL = 'https://railway.test'
  process.env.EXTRACT_API_KEY = 'test-key'
})

async function callRoute(body: Record<string, unknown>) {
  const { POST } = await import('@/app/api/baselines/from-swing/route')
  const req = new NextRequest('http://localhost/api/baselines/from-swing', {
    method: 'POST',
    body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json' },
  })
  return POST(req)
}

// startFrame=420, endFrame=510 at 30fps → start_ms ~14000, end_ms ~17000.
// We send the ms values explicitly (the route stopped recomputing them
// from frame indices in 2026-05 — see route comment).
const VALID_BODY = {
  sessionId: SESSION_ID,
  startFrame: 420,
  endFrame: 510,
  peakFrame: 465,
  startMs: 14000,
  endMs: 17000,
  shotType: 'forehand',
  label: 'My swing',
  // Pre-sliced keypoints from the client. Timestamps are still in
  // source-video time — the route is responsible for re-zeroing.
  keypointsJson: {
    fps_sampled: 30,
    frame_count: 91,
    frames: makeFrames(420, 91),
  },
}

describe('POST /api/baselines/from-swing', () => {
  it('returns 401 when not signed in', async () => {
    currentUser = null
    const res = await callRoute(VALID_BODY)
    expect(res.status).toBe(401)
  })

  it('returns 400 when sessionId is missing or malformed', async () => {
    const res = await callRoute({ ...VALID_BODY, sessionId: 'not-a-uuid' })
    expect(res.status).toBe(400)
    const j = await res.json()
    expect(j.error).toMatch(/sessionId/)
  })

  it('returns 400 when startFrame >= endFrame', async () => {
    const res = await callRoute({ ...VALID_BODY, startFrame: 500, endFrame: 500 })
    expect(res.status).toBe(400)
  })

  it('returns 400 when startMs is missing', async () => {
    const noStart = { ...VALID_BODY }
    delete (noStart as Record<string, unknown>).startMs
    const res = await callRoute(noStart)
    expect(res.status).toBe(400)
  })

  it('returns 400 when endMs <= startMs', async () => {
    const res = await callRoute({ ...VALID_BODY, startMs: 5000, endMs: 5000 })
    expect(res.status).toBe(400)
  })

  it('returns 400 when startMs is before the session timeline', async () => {
    // Session frames are makeFrames(0, 600) at 30fps, so timestamps run
    // 0..19967ms. A startMs well before 0 must be rejected so a malicious
    // client can't sneak the trim into someone else's session.
    const res = await callRoute({ ...VALID_BODY, startMs: -100, endMs: 1000 })
    expect(res.status).toBe(400)
  })

  it('returns 400 when endMs is after the session timeline', async () => {
    // Session ends around 19967ms; an endMs way past that (with the
    // 50ms slop already applied) must be rejected.
    const res = await callRoute({ ...VALID_BODY, startMs: 0, endMs: 30000 })
    expect(res.status).toBe(400)
  })

  it('returns 400 when the trim window exceeds 60 seconds', async () => {
    // Mirrors railway-service _TRIM_MAX_DURATION_MS — saves a Railway
    // round-trip on obvious overruns. Session timestamps don't matter
    // for this check, the duration cap fires first.
    const res = await callRoute({ ...VALID_BODY, startMs: 0, endMs: 65_000 })
    expect(res.status).toBe(400)
  })

  it('returns 400 when shotType is invalid', async () => {
    const res = await callRoute({ ...VALID_BODY, shotType: 'wallop' })
    expect(res.status).toBe(400)
  })

  it('returns 404 with a clear message when the source session is gone', async () => {
    sessionRow = null
    sessionError = { message: 'not found' }
    const res = await callRoute(VALID_BODY)
    expect(res.status).toBe(404)
    const j = await res.json()
    expect(j.error).toMatch(/Source session no longer available/i)
  })

  it('returns 404 when Railway reports the source video is gone', async () => {
    // Source session row exists, but the underlying blob has been
    // deleted (e.g. cleanup-blobs ran). Railway reports 404; the route
    // surfaces it as 404 to the client (not 502) so the UI can prompt
    // a re-upload rather than treat it as a transient infra error.
    fetchMock = vi.fn(async () => ({
      ok: false,
      status: 404,
      json: async () => ({}),
      text: async () => 'video missing',
    }))
    globalThis.fetch = fetchMock as unknown as typeof fetch

    const res = await callRoute(VALID_BODY)
    expect(res.status).toBe(404)
    const j = await res.json()
    expect(j.error).toMatch(/Source video no longer available/i)
  })

  it('returns 503 when Railway env vars are missing', async () => {
    delete process.env.RAILWAY_SERVICE_URL
    const res = await callRoute(VALID_BODY)
    expect(res.status).toBe(503)
  })

  it('returns 502 when Railway trim fails with a non-404 error', async () => {
    fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
      json: async () => ({}),
      text: async () => 'ffmpeg crashed',
    }))
    globalThis.fetch = fetchMock as unknown as typeof fetch

    const res = await callRoute(VALID_BODY)
    expect(res.status).toBe(502)
  })

  it('returns 502 when the uploaded blob URL ends up off-allowlist', async () => {
    // Vercel Blob always returns vercel-storage.com URLs, but a future
    // SDK bug or middleware (e.g. proxying through a CDN) could rewrite
    // the host. The route's allowlist is the last line of defense.
    putMock = vi.fn(async () => ({
      url: 'https://evil.example.com/foo.mp4',
      pathname: 'baseline-trims/x.mp4',
      contentType: 'video/mp4',
      contentDisposition: 'inline',
      downloadUrl: 'https://evil.example.com/foo.mp4',
    }))
    const res = await callRoute(VALID_BODY)
    expect(res.status).toBe(502)
  })

  it('returns 502 when Railway returns an empty body', async () => {
    fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      body: null,
      json: async () => ({}),
      text: async () => '',
    } as unknown as Response))
    globalThis.fetch = fetchMock as unknown as typeof fetch

    const res = await callRoute(VALID_BODY)
    expect(res.status).toBe(502)
  })

  it('happy path: derives ms range from frames + fps, re-zeroes pose timestamps', async () => {
    const res = await callRoute(VALID_BODY)
    expect(res.status).toBe(200)
    const j = await res.json()
    expect(j.baseline).toEqual({ id: 'new-baseline-id', shot_type: 'forehand' })

    // 1) Railway called with start_ms / end_ms derived from frame
    //    indices and fps. startFrame=420, fps=30 → start_ms=14000.
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const fetchArgs = fetchMock.mock.calls[0]
    expect(String(fetchArgs[0])).toContain('/trim-video')
    const sentBody = JSON.parse((fetchArgs[1] as { body: string }).body)
    expect(sentBody).toMatchObject({
      video_url: 'https://abc.public.blob.vercel-storage.com/src.mp4',
      start_ms: 14000,
      end_ms: 17000,
    })

    // 2) Insert runs with the trimmed URL and re-zeroed frames. The
    //    first frame's timestamp must be 0 and frame_index must be 0;
    //    if this regresses, the pose overlay on /baseline/[id] dies
    //    (VideoCanvas matches currentTime*1000 against timestamp_ms).
    const insertCalls = captured.filter((c) => c.op === 'insert')
    expect(insertCalls).toHaveLength(1)
    const row = insertCalls[0].args[0] as Record<string, unknown>
    expect(row).toMatchObject({
      user_id: 'user-1',
      shot_type: 'forehand',
      blob_url: TRIMMED_BLOB_URL,
      is_active: true,
      source_session_id: SESSION_ID,
    })
    const kp = row.keypoints_json as {
      frames: Array<{ frame_index: number; timestamp_ms: number }>
    }
    expect(kp.frames[0].frame_index).toBe(0)
    expect(kp.frames[0].timestamp_ms).toBe(0)
    expect(kp.frames[kp.frames.length - 1].frame_index).toBe(kp.frames.length - 1)
    expect(kp.frames[kp.frames.length - 1].timestamp_ms).toBeGreaterThan(0)

    // 3) Sibling deactivate fires against the requested shot_type.
    const eqArgs = captured.filter((c) => c.op.startsWith('update.eq')).map((c) => c.args)
    expect(eqArgs).toContainEqual(['shot_type', 'forehand'])
    expect(eqArgs).toContainEqual(['is_active', true])
  })

  it('falls back to slicing the session keypoints when the client omits keypointsJson', async () => {
    const noKpBody = { ...VALID_BODY, keypointsJson: undefined }
    delete (noKpBody as Record<string, unknown>).keypointsJson

    const res = await callRoute(noKpBody)
    expect(res.status).toBe(200)

    const insertRow = (captured.find((c) => c.op === 'insert')!.args[0]) as Record<string, unknown>
    const kp = insertRow.keypoints_json as {
      frames: Array<{ frame_index: number; timestamp_ms: number }>
    }
    // startFrame=420, endFrame=510 inclusive → 91 frames sliced from
    // the parent session, then re-zeroed.
    expect(kp.frames.length).toBe(91)
    expect(kp.frames[0].frame_index).toBe(0)
    expect(kp.frames[0].timestamp_ms).toBe(0)
  })

  it('uses default label when none provided', async () => {
    const noLabel = { ...VALID_BODY }
    delete (noLabel as Record<string, unknown>).label
    const res = await callRoute(noLabel)
    expect(res.status).toBe(200)
    const row = (captured.find((c) => c.op === 'insert')!.args[0]) as Record<string, unknown>
    expect(row.label).toMatch(/forehand/)
  })
})

afterAll(() => {
  globalThis.fetch = originalFetch
})
