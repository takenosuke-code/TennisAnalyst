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
// (TIER_MAX_TOKENS, etc) imports from the real module so behavioral
// assertions match prod.
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

/**
 * Standard fixture: 60 frames whose hip/trunk excursion is zero (constant
 * angles), so the new observation extractor reliably emits at least one
 * `insufficient_*_excursion` observation. With landmarks=[] the visibility
 * defaults to 1.0 inside the extractor (per its contract), so the confidence
 * floor doesn't fire.
 */
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

/**
 * Empty-state fixture: explicit zero-visibility landmarks force the
 * confidence floor to kill every observation, exercising the empty-state
 * branch.
 */
function makeBlindKeypoints(frameCount = 60) {
  return {
    fps_sampled: 30,
    frame_count: frameCount,
    frames: Array.from({ length: frameCount }, (_, i) => ({
      frame_index: i,
      timestamp_ms: (i / 30) * 1000,
      landmarks: [
        { id: 0, name: 'nose', x: 0, y: 0, z: 0, visibility: 0.05 },
        { id: 11, name: 'l_shoulder', x: 0, y: 0, z: 0, visibility: 0.05 },
      ],
      joint_angles: {
        right_elbow: 80,
        right_knee: 178,
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

describe('POST /api/analyze (observation-driven pipeline)', () => {
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

  // -------------------------------------------------------------------------
  // Tier wiring + system prompt voice rules
  // -------------------------------------------------------------------------

  for (const tier of ['beginner', 'intermediate', 'competitive', 'advanced'] as const) {
    it(`tier=${tier}: passes the right max_tokens, voice-rules system prompt, and tier hint in user prompt`, async () => {
      mockGetCoachingContext.mockResolvedValue({
        profile: {
          skill_tier: tier,
          dominant_hand: 'right',
          backhand_style: 'two_handed',
          primary_goal: 'consistency',
        },
        skipped: false,
      })
      // Clean coach reply with the new two-section format.
      responseQueue.push(
        '## Primary cue\nTurn your shoulders earlier and let the racket follow.\n\n## Other things I noticed\n- Set the back leg before the bounce.',
      )

      const { POST } = await import('@/app/api/analyze/route')
      const res = await POST(
        makeReq({
          keypointsJson: makeFakeKeypoints(60),
          shotType: 'forehand',
        }),
      )
      expect(res.status).toBe(200)
      await readStream(res) // drain so after() runs

      expect(lastArgsBox.current).not.toBeNull()
      // max_tokens wired to the per-tier ceiling.
      expect(lastArgsBox.current!.max_tokens).toBe(TIER_MAX_TOKENS[tier])
      // System prompt is the new tight voice-rules block.
      expect(lastArgsBox.current!.system[0].text).toMatch(
        /You are a veteran tennis coach/i,
      )
      expect(lastArgsBox.current!.system[0].text).toMatch(/Voice rules/i)
      // User prompt must mention the tier name so the model can calibrate.
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
    responseQueue.push(
      '## Primary cue\nQuick advice.\n\n## Other things I noticed\n- Stay sideways.',
    )

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

  // -------------------------------------------------------------------------
  // Response shape
  // -------------------------------------------------------------------------

  it('emits "## Primary cue", "## Other things I noticed", and "## Show your work" sections', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: {
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
      },
      skipped: false,
    })
    responseQueue.push(
      '## Primary cue\nTurn your shoulders earlier.\n\n## Other things I noticed\n- Bend the knees.',
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
    expect(body).toContain('## Primary cue')
    expect(body).toContain('## Other things I noticed')
    expect(body).toContain('## Show your work')
  })

  it('numbers only appear in the "## Show your work" section', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: {
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
      },
      skipped: false,
    })
    responseQueue.push(
      '## Primary cue\nTurn your shoulders earlier.\n\n## Other things I noticed\n- Bend the knees.',
    )

    const { POST } = await import('@/app/api/analyze/route')
    const res = await POST(
      makeReq({
        keypointsJson: makeFakeKeypoints(60),
        shotType: 'forehand',
      }),
    )
    const body = await readStream(res)
    const showIdx = body.indexOf('## Show your work')
    expect(showIdx).toBeGreaterThan(-1)
    const beforeShow = body.slice(0, showIdx)
    expect(beforeShow).not.toMatch(/\d/)
    // The Show your work section itself must contain at least one digit
    // (the rendered observation values).
    const afterShow = body.slice(showIdx)
    expect(afterShow).toMatch(/\d/)
  })

  // -------------------------------------------------------------------------
  // Empty-state branch
  // -------------------------------------------------------------------------

  it('empty-state: no observations -> friendly message + X-Analyze-Empty-State header + no LLM call', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: {
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
      },
      skipped: false,
    })

    const { POST } = await import('@/app/api/analyze/route')
    const res = await POST(
      makeReq({
        keypointsJson: makeBlindKeypoints(60),
        shotType: 'forehand',
      }),
    )
    expect(res.status).toBe(200)
    expect(res.headers.get('X-Analyze-Empty-State')).toBe('true')
    const body = await readStream(res)
    expect(body).toMatch(/could not read your swing/i)
    expect(body).toMatch(/from the side/i)
    // Anthropic should NOT have been called.
    expect(lastArgsBox.current).toBeNull()
  })

  // -------------------------------------------------------------------------
  // Post-filter behavior
  // -------------------------------------------------------------------------

  it('post-filter rejects digit-leak output and re-rolls; clean retry survives', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: {
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
      },
      skipped: false,
    })
    // First attempt: contains a digit. Second attempt: clean.
    responseQueue.push(
      '## Primary cue\nTurn 30 degrees earlier.\n\n## Other things I noticed\n- Bend.',
    )
    responseQueue.push(
      '## Primary cue\nTurn your shoulders earlier.\n\n## Other things I noticed\n- Bend the knees.',
    )

    const { POST } = await import('@/app/api/analyze/route')
    const res = await POST(
      makeReq({
        keypointsJson: makeFakeKeypoints(60),
        shotType: 'forehand',
      }),
    )
    const body = await readStream(res)
    // Post-filter accepted the second (clean) attempt; digit appears only in
    // the Show your work block.
    const showIdx = body.indexOf('## Show your work')
    const beforeShow = body.slice(0, showIdx)
    expect(beforeShow).not.toMatch(/\d/)
    expect(beforeShow).toContain('Turn your shoulders earlier')
  })

  it('post-filter falls back to a static cue after 3 dirty LLM responses', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: {
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
      },
      skipped: false,
    })
    // All 3 attempts dirty.
    for (let i = 0; i < 3; i++) {
      responseQueue.push('Bad output with the kinetic chain explanation.')
    }

    const { POST } = await import('@/app/api/analyze/route')
    const res = await POST(
      makeReq({
        keypointsJson: makeFakeKeypoints(60),
        shotType: 'forehand',
      }),
    )
    const body = await readStream(res)
    // Static fallback emits a Primary cue + Other things I noticed structure
    // built from CueExemplar text — none of the disallowed strings.
    expect(body).toContain('## Primary cue')
    expect(body).toContain('## Other things I noticed')
    expect(body).toContain('## Show your work')
    const showIdx = body.indexOf('## Show your work')
    const beforeShow = body.slice(0, showIdx)
    expect(beforeShow).not.toMatch(/kinetic chain/i)
    expect(beforeShow).not.toMatch(/\d/)
  })

  it('post-filter rejects em-dashes and falls through to the next attempt', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: {
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
      },
      skipped: false,
    })
    // Em-dash is in the first attempt; second attempt is clean.
    responseQueue.push(
      '## Primary cue\nTurn your shoulders \u2014 then drive forward.\n\n## Other things I noticed\n- Bend the knees.',
    )
    responseQueue.push(
      '## Primary cue\nTurn your shoulders earlier.\n\n## Other things I noticed\n- Bend the knees.',
    )

    const { POST } = await import('@/app/api/analyze/route')
    const res = await POST(
      makeReq({
        keypointsJson: makeFakeKeypoints(60),
        shotType: 'forehand',
      }),
    )
    const body = await readStream(res)
    const showIdx = body.indexOf('## Show your work')
    const beforeShow = body.slice(0, showIdx)
    expect(beforeShow).not.toContain('\u2014')
    expect(beforeShow).toContain('Turn your shoulders earlier')
  })
})
