import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'
import { TIER_MAX_TOKENS } from '@/lib/profile'

// vi.hoisted runs at the top of the module before any vi.mock factory
// is evaluated, so factories can safely close over these refs. Without
// it, the factories would reference uninitialized vars and explode at
// import time.
const { lastArgsBox, responseQueue, mockGetCoachingContext } = vi.hoisted(() => ({
  // Mutable single-cell box so tests can read/clear what
  // messages.stream was last called with.
  lastArgsBox: { current: null as null | {
    model: string
    max_tokens: number
    system: { type: string; text: string }[]
    messages: { role: string; content: string }[]
  } },
  responseQueue: [] as string[],
  mockGetCoachingContext: vi.fn(),
}))

vi.mock('@anthropic-ai/sdk', () => ({
  default: class {
    messages = {
      stream(args: {
        model: string
        max_tokens: number
        system: { type: string; text: string }[]
        messages: { role: string; content: string }[]
      }) {
        lastArgsBox.current = args
        const text = responseQueue.shift() ?? 'Default mock response.'
        return (async function* () {
          const chunks = text.match(/.{1,5}/g) ?? []
          for (const c of chunks) {
            yield {
              type: 'content_block_delta',
              delta: { type: 'text_delta', text: c },
            }
          }
          yield { type: 'message_delta', usage: { output_tokens: 42 } }
        })()
      },
    }
  },
}))

// Supabase: no-op writes, no session fetched (we always pass
// keypointsJson inline).
const mockEq = vi.fn(() => ({ error: null }))
const mockUpdate = vi.fn(() => ({ eq: mockEq }))
const mockSelect = vi.fn(() => ({
  single: vi.fn(async () => ({ data: { id: 'evt-1' }, error: null })),
}))
const mockInsert = vi.fn(() => ({ select: mockSelect }))
const mockFrom = vi.fn(() => ({
  insert: mockInsert,
  update: mockUpdate,
}))

vi.mock('@/lib/supabase', () => ({
  supabase: { from: mockFrom },
  supabaseAdmin: { from: mockFrom },
}))

vi.mock('@/lib/supabase/server', () => ({
  createClient: async () => ({
    auth: {
      getUser: async () => ({ data: { user: null } }),
    },
  }),
}))

vi.mock('@/lib/captureQuality', () => ({
  classifyAndTagCaptureQuality: vi.fn(),
}))

// getCoachingContext is the only profile.ts hook the route calls. We
// override per-test to simulate different tiers; everything else
// (TIER_MAX_TOKENS, buildTierCoachingBlock, parseTierAssessmentTrailer,
// matchesBaselineTemplate, etc) imports from the real module so
// behavioral assertions match prod.
vi.mock('@/lib/profile', async () => {
  const actual = await vi.importActual<typeof import('@/lib/profile')>('@/lib/profile')
  return {
    ...actual,
    getCoachingContext: mockGetCoachingContext,
  }
})

// `next/server` `after()` defers work until response close. The
// runtime ignores it; in tests we just want it to no-op.
vi.mock('next/server', async () => {
  const actual = await vi.importActual<typeof import('next/server')>('next/server')
  return { ...actual, after: (cb: () => void) => cb() }
})

// ---------- Helpers ----------

function makeFakeKeypoints(frameCount = 60) {
  return {
    fps_sampled: 30,
    frame_count: frameCount,
    frames: Array.from({ length: frameCount }, (_, i) => ({
      frame_index: i,
      timestamp_ms: (i / 30) * 1000,
      landmarks: [],
      joint_angles: {
        right_elbow: 120,
        left_elbow: 140,
        right_shoulder: 90,
        left_shoulder: 90,
        right_knee: 150,
        left_knee: 150,
        hip_rotation: 50,
        trunk_rotation: 50,
      },
    })),
  }
}

function makeReq(body: object): NextRequest {
  return new NextRequest('http://localhost/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

async function readStream(res: Response): Promise<string> {
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let out = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    out += decoder.decode(value, { stream: true })
  }
  out += decoder.decode()
  return out
}

// ---------- Tests ----------

describe('POST /api/analyze (streaming)', () => {
  beforeEach(() => {
    vi.resetModules()
    lastArgsBox.current = null
    responseQueue.length = 0
    mockEq.mockClear()
    mockUpdate.mockClear()
    mockInsert.mockClear()
    mockSelect.mockClear()
    mockFrom.mockClear()
    mockGetCoachingContext.mockReset()
    // Re-prime the supabase mock chain that beforeEach cleared.
    mockSelect.mockImplementation(() => ({
      single: vi.fn(async () => ({ data: { id: 'evt-1' }, error: null })),
    }))
    mockInsert.mockImplementation(() => ({ select: mockSelect }))
    mockEq.mockImplementation(() => ({ error: null }))
    mockUpdate.mockImplementation(() => ({ eq: mockEq }))
    mockFrom.mockImplementation(() => ({
      insert: mockInsert,
      update: mockUpdate,
    }))
  })

  for (const tier of ['beginner', 'intermediate', 'competitive', 'advanced'] as const) {
    it(`tier=${tier}: passes the right max_tokens + tier rubric block + voice-only system prompt`, async () => {
      mockGetCoachingContext.mockResolvedValue({
        profile: {
          skill_tier: tier,
          dominant_hand: 'right',
          backhand_style: 'two_handed',
          primary_goal: 'consistency',
        },
        skipped: false,
      })
      responseQueue.push('Body of the coach response.')

      const { POST } = await import('@/app/api/analyze/route')
      const res = await POST(
        makeReq({
          keypointsJson: makeFakeKeypoints(60),
          shotType: 'forehand',
        }),
      )
      expect(res.status).toBe(200)
      await readStream(res) // drain so after() runs

      // Single streaming call — no tool_use primary, no fallback.
      expect(lastArgsBox.current).not.toBeNull()
      // max_tokens wired to the per-tier ceiling.
      expect(lastArgsBox.current!.max_tokens).toBe(TIER_MAX_TOKENS[tier])
      // System prompt is the voice-rules block, NOT the tool_use one.
      expect(lastArgsBox.current!.system[0].text).toMatch(
        /You are a veteran tennis coach/i,
      )
      expect(lastArgsBox.current!.system[0].text).not.toMatch(/emit_coaching tool/i)
      // Tier rubric is woven into the user prompt — the rubric block
      // names the tier explicitly so we can grep for it.
      expect(lastArgsBox.current!.messages[0].content.toLowerCase()).toContain(tier)
    })
  }

  it('mode=shot caps max_tokens at 600 even when the tier ceiling is higher', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: {
        skill_tier: 'competitive', // tier ceiling = 1500
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
      },
      skipped: false,
    })
    responseQueue.push('Quick per-shot advice.')

    const { POST } = await import('@/app/api/analyze/route')
    const res = await POST(
      makeReq({
        keypointsJson: makeFakeKeypoints(60),
        shotType: 'forehand',
        mode: 'shot',
      }),
    )
    expect(res.status).toBe(200)
    await readStream(res)

    expect(lastArgsBox.current).not.toBeNull()
    expect(lastArgsBox.current!.max_tokens).toBe(600)
  })

  it('skipped-onboarding users: TIER_ASSESSMENT trailer is suppressed from the streamed response', async () => {
    mockGetCoachingContext.mockResolvedValue({ profile: null, skipped: true })
    // The trailer must be the tail of the response — that's where
    // parseTierAssessmentTrailer extracts it from.
    responseQueue.push(
      'Coach body text here. More body. And more body content.\n\n[TIER_ASSESSMENT: intermediate]',
    )

    const { POST } = await import('@/app/api/analyze/route')
    const res = await POST(
      makeReq({
        keypointsJson: makeFakeKeypoints(60),
        shotType: 'forehand',
      }),
    )
    expect(res.status).toBe(200)
    const body = await readStream(res)
    // Trailer should not leak to the user.
    expect(body).not.toMatch(/\[TIER_ASSESSMENT:/i)
    // But the body content survived the holdback flush.
    expect(body).toContain('Coach body text here.')
  })

  it('signed-in profile users: emits tokens immediately (no holdback) — full body matches the mock', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: {
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
      },
      skipped: false,
    })
    const expected = 'A complete coaching response with no trailer at all.'
    responseQueue.push(expected)

    const { POST } = await import('@/app/api/analyze/route')
    const res = await POST(
      makeReq({
        keypointsJson: makeFakeKeypoints(60),
        shotType: 'forehand',
      }),
    )
    const body = await readStream(res)
    expect(body).toBe(expected)
  })
})
