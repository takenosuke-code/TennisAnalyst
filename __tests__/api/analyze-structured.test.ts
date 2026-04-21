import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { NextRequest } from 'next/server'
import {
  ADVANCED_BASELINE_TEMPLATE,
  TIER_MAX_TOKENS,
  type SkillTier,
} from '@/lib/profile'

// ---------------------------------------------------------------------------
// Integration coverage for the tool_use (structured output) rewrite of
// /api/analyze. Mocks the Anthropic SDK + the two Supabase clients (auth
// client + service-role admin client) so we can assert on what the route
// sends to Anthropic and what it writes to analysis_events.
//
// Strategy mirrors __tests__/app/api/analyze-capture-flag.test.ts:
//   - mock at the module boundary (@anthropic-ai/sdk, @/lib/supabase, etc.)
//   - vi.resetModules() between tests so env stubs (STRUCTURED_OUTPUT_DISABLE)
//     take effect on a fresh import
//   - mock next/server's after() to run synchronously since unit tests have
//     no request-context store
//   - consume the ReadableStream body + flush microtasks before asserting
// ---------------------------------------------------------------------------

// Anthropic SDK mock. The route instantiates `new Anthropic(...)` at module
// scope, so our mock class must expose `messages.{create, stream}`.
const createMock = vi.fn()
const streamMock = vi.fn()
vi.mock('@anthropic-ai/sdk', () => ({
  default: class {
    messages = { create: createMock, stream: streamMock }
  },
}))

// next/server.after() throws outside a request context. In tests we just
// want the callback to run immediately so the backfill UPDATE fires.
// Note: this runs the callbacks eagerly (synchronously for sync cbs, or
// detached-promise for async cbs). In production `after()` defers until
// after the response, but the tests here only care that the callbacks
// ran before assertions — not about their relative ordering.
vi.mock('next/server', async () => {
  const actual = await vi.importActual<typeof import('next/server')>('next/server')
  return {
    ...actual,
    after: (cb: () => unknown) => {
      try {
        const result = cb()
        if (result instanceof Promise) result.catch(() => {})
      } catch {
        /* ignore */
      }
    },
  }
})

// Keypoints summary is not what we're testing — stub it to a predictable
// string so the route's prompt builders don't crash on synthetic data.
vi.mock('@/lib/jointAngles', () => ({
  buildAngleSummary: () => 'stub-summary',
}))

vi.mock('@/lib/biomechanics-reference', () => ({
  getBiomechanicsReference: () => 'stub-ref',
}))

// Capture-quality telemetry makes an external Railway call. No-op in tests.
vi.mock('@/lib/captureQuality', () => ({
  classifyAndTagCaptureQuality: vi.fn(),
}))

// Supabase auth client (createClient from @/lib/supabase/server). Every test
// reassigns `nextAuthClient` so we can swap profiles per scenario.
type ProfileUserMeta = {
  skill_tier?: SkillTier
  dominant_hand?: 'right' | 'left'
  backhand_style?: 'one_handed' | 'two_handed'
  primary_goal?: string
  onboarded_at?: string
}
let nextAuthClient: {
  auth: { getUser: () => Promise<{ data: { user: { id?: string; user_metadata?: ProfileUserMeta } | null }; error: null }> }
}
vi.mock('@/lib/supabase/server', () => ({
  createClient: vi.fn(async () => nextAuthClient),
}))

// supabaseAdmin: records the insert + update calls so tests can assert on
// what got written to analysis_events. Also used for the user_sessions lookup
// when the request supplies inlineKeypoints without shot_type/blob_url (we
// shortcut that by always providing inline keypoints).
const insertedEventId = 'event-fixture-1'
let insertCalls: Array<Record<string, unknown>> = []
let updateCalls: Array<Record<string, unknown>> = []

function makeAdminMock() {
  return {
    from: (table: string) => {
      if (table !== 'analysis_events') {
        // Default: no-op chain for user_sessions fallback lookups.
        return {
          select: () => ({
            eq: () => ({ single: async () => ({ data: null, error: null }) }),
          }),
        }
      }
      return {
        insert: (row: Record<string, unknown>) => {
          insertCalls.push(row)
          return {
            select: () => ({
              single: async () => ({ data: { id: insertedEventId }, error: null }),
            }),
          }
        },
        update: (row: Record<string, unknown>) => {
          updateCalls.push(row)
          return {
            eq: async () => ({ error: null }),
          }
        },
      }
    },
  }
}

vi.mock('@/lib/supabase', () => ({
  supabase: {
    from: () => ({
      select: () => ({
        eq: () => ({ single: async () => ({ data: null, error: null }) }),
      }),
    }),
  },
  supabaseAdmin: makeAdminMock(),
}))

function makeFakeKeypoints(frameCount = 3) {
  return {
    fps_sampled: 30,
    frame_count: frameCount,
    frames: Array.from({ length: frameCount }, (_, i) => ({
      frame_index: i,
      timestamp_ms: i * 33,
      landmarks: [],
      joint_angles: { right_elbow: 120, hip_rotation: 40 },
    })),
  }
}

function makeProfile(tier: SkillTier): ProfileUserMeta {
  return {
    skill_tier: tier,
    dominant_hand: 'right',
    backhand_style: 'two_handed',
    primary_goal: 'consistency',
    onboarded_at: '2025-01-01T00:00:00Z',
  }
}

function makeToolUseResponse(args: {
  input: unknown
  outputTokens?: number | null
}) {
  return {
    content: [
      {
        type: 'tool_use',
        id: 'toolu_1',
        name: 'emit_coaching',
        input: args.input,
      },
    ],
    usage: { output_tokens: args.outputTokens ?? 180 },
    stop_reason: 'tool_use',
  }
}

// Async-iterable mock for anthropic.messages.stream(...). Emits one chunk of
// text, optionally followed by a TIER_ASSESSMENT trailer.
function makeStreamMock(text: string) {
  return (async function* () {
    yield { type: 'content_block_delta', delta: { type: 'text_delta', text } }
  })()
}

async function consumeStream(res: Response): Promise<string> {
  const reader = res.body?.getReader()
  if (!reader) return ''
  const decoder = new TextDecoder()
  let out = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    out += decoder.decode(value)
  }
  return out
}

async function flushMicrotasks(n = 10) {
  for (let i = 0; i < n; i++) await Promise.resolve()
}

function makeRequest(body: Record<string, unknown>): NextRequest {
  return new NextRequest('http://localhost/api/analyze', {
    method: 'POST',
    body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json' },
  })
}

async function importRoute() {
  return import('@/app/api/analyze/route')
}

describe('POST /api/analyze (structured-output + fallback)', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.unstubAllEnvs()
    createMock.mockReset()
    streamMock.mockReset()
    insertCalls = []
    updateCalls = []
    // Default profile: intermediate, right-handed. Tests override as needed.
    nextAuthClient = {
      auth: {
        getUser: async () => ({
          data: { user: { id: 'user-1', user_metadata: makeProfile('intermediate') } },
          error: null,
        }),
      },
    }
  })

  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('[1] structured happy path (intermediate): renders tool_use input to markdown + writes metrics', async () => {
    createMock.mockResolvedValue(
      makeToolUseResponse({
        input: {
          strengths: [{ text: 'Nice loop' }],
          cues: [
            { title: 'Load', body: 'Body 1' },
            { title: 'Finish', body: 'Body 2' },
          ],
          closing: 'Work on this',
        },
        outputTokens: 180,
      }),
    )

    const { POST } = await importRoute()
    const res = await POST(
      makeRequest({ keypointsJson: makeFakeKeypoints() }) as NextRequest,
    )
    const body = await consumeStream(res)
    await flushMicrotasks()

    expect(res.status).toBe(200)
    expect(body).toContain("## What You're Doing Well")
    expect(body).toContain('**1. Load**')
    expect(body).toContain('**2. Finish**')
    expect(streamMock).not.toHaveBeenCalled()

    expect(insertCalls).toHaveLength(1)
    expect(updateCalls).toHaveLength(1)
    expect(updateCalls[0]).toMatchObject({
      response_tip_count: 2,
      response_token_count: 180,
      used_baseline_template: false,
      // tool_use forecloses tier dissent -> llm_assessed_tier == profile tier.
      llm_assessed_tier: 'intermediate',
      llm_tier_downgrade: false,
    })
  })

  it('[2] advanced baseline: cues=[], closing=template -> emits literal template and flags used_baseline_template=true', async () => {
    nextAuthClient = {
      auth: {
        getUser: async () => ({
          data: { user: { id: 'user-2', user_metadata: makeProfile('advanced') } },
          error: null,
        }),
      },
    }
    createMock.mockResolvedValue(
      makeToolUseResponse({
        input: {
          strengths: [{ text: 'Mechanics look refined through contact.' }],
          cues: [],
          closing: ADVANCED_BASELINE_TEMPLATE,
        },
        outputTokens: 42,
      }),
    )

    const { POST } = await importRoute()
    const res = await POST(
      makeRequest({ keypointsJson: makeFakeKeypoints() }) as NextRequest,
    )
    const body = await consumeStream(res)
    await flushMicrotasks()

    expect(body).toBe(ADVANCED_BASELINE_TEMPLATE)
    expect(updateCalls[0]).toMatchObject({
      used_baseline_template: true,
      response_tip_count: 0,
    })
  })

  it('[3] fallback when tool_use block is missing: streams text and still writes metrics', async () => {
    createMock.mockResolvedValue({
      content: [{ type: 'text', text: 'plain prose with no tool call' }],
      usage: { output_tokens: 12 },
      stop_reason: 'end_turn',
    })
    streamMock.mockReturnValue(
      makeStreamMock(
        '## What You\'re Doing Well\n\nGood stance.\n\n## 3 Things to Work On\n\n**1. A**\nbody\n\n**2. B**\nbody\n\n**3. C**\nbody\n\n[TIER_ASSESSMENT: intermediate]',
      ),
    )

    const { POST } = await importRoute()
    const res = await POST(
      makeRequest({ keypointsJson: makeFakeKeypoints() }) as NextRequest,
    )
    const body = await consumeStream(res)
    await flushMicrotasks()

    expect(createMock).toHaveBeenCalled()
    expect(streamMock).toHaveBeenCalled()
    expect(body).not.toContain('[TIER_ASSESSMENT')
    expect(body).toContain('Good stance.')

    expect(updateCalls).toHaveLength(1)
    expect(updateCalls[0]).toMatchObject({
      llm_assessed_tier: 'intermediate',
      response_token_count: null, // fallback path doesn't capture usage
      response_tip_count: 3,
    })
  })

  it('[4] fallback when tool input fails schema validation (buildCoachingToolInput returns null)', async () => {
    createMock.mockResolvedValue(
      makeToolUseResponse({
        // cues missing `body` -> buildCoachingToolInput returns null
        input: { strengths: [], cues: [{ title: 'x' }], closing: 'c' },
        outputTokens: 9,
      }),
    )
    streamMock.mockReturnValue(
      makeStreamMock('fallback body [TIER_ASSESSMENT: intermediate]'),
    )

    const { POST } = await importRoute()
    const res = await POST(
      makeRequest({ keypointsJson: makeFakeKeypoints() }) as NextRequest,
    )
    await consumeStream(res)
    await flushMicrotasks()

    expect(streamMock).toHaveBeenCalledTimes(1)
    expect(updateCalls).toHaveLength(1)
  })

  it('[5] SDK throws on primary path -> falls back to streaming', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    createMock.mockRejectedValue(new Error('anthropic 500'))
    streamMock.mockReturnValue(
      makeStreamMock('fallback body [TIER_ASSESSMENT: intermediate]'),
    )

    const { POST } = await importRoute()
    const res = await POST(
      makeRequest({ keypointsJson: makeFakeKeypoints() }) as NextRequest,
    )
    const body = await consumeStream(res)
    await flushMicrotasks()

    expect(streamMock).toHaveBeenCalledTimes(1)
    expect(body).toContain('fallback body')
    errSpy.mockRestore()
  })

  it.each<SkillTier>(['beginner', 'intermediate', 'competitive', 'advanced'])(
    '[6] per-tier max_tokens wired to messages.create for tier=%s',
    async (tier) => {
      nextAuthClient = {
        auth: {
          getUser: async () => ({
            data: { user: { id: `user-${tier}`, user_metadata: makeProfile(tier) } },
            error: null,
          }),
        },
      }
      // Provide a minimal VALID tool_use response so the primary path
      // completes cleanly — otherwise the route falls through to stream
      // and the max_tokens assertion is moot.
      const minimalInput =
        tier === 'advanced'
          ? {
              strengths: [{ text: 'Looks clean across the clip.' }],
              cues: [],
              closing: ADVANCED_BASELINE_TEMPLATE,
            }
          : tier === 'competitive'
            ? {
                strengths: [{ text: 'Solid foundation across frames.' }],
                cues: [
                  { title: 'A', body: 'ab' },
                  { title: 'B', body: 'bb' },
                  { title: 'C', body: 'cb' },
                ],
                closing: 'Go practice.',
              }
            : {
                strengths: [{ text: 'Clean base.' }],
                cues: [
                  { title: 'A', body: 'ab' },
                  { title: 'B', body: 'bb' },
                ],
                closing: 'Go practice.',
              }
      createMock.mockResolvedValue(
        makeToolUseResponse({ input: minimalInput, outputTokens: 77 }),
      )

      const { POST } = await importRoute()
      const res = await POST(
        makeRequest({ keypointsJson: makeFakeKeypoints() }) as NextRequest,
      )
      await consumeStream(res)
      await flushMicrotasks()

      expect(createMock).toHaveBeenCalledTimes(1)
      const [args] = createMock.mock.calls[0]
      expect(args.max_tokens).toBe(TIER_MAX_TOKENS[tier])
    },
  )

  it('[7] kill-switch STRUCTURED_OUTPUT_DISABLE=1 bypasses messages.create entirely', async () => {
    vi.stubEnv('STRUCTURED_OUTPUT_DISABLE', '1')
    streamMock.mockReturnValue(
      makeStreamMock('ks body [TIER_ASSESSMENT: intermediate]'),
    )

    const { POST } = await importRoute()
    const res = await POST(
      makeRequest({ keypointsJson: makeFakeKeypoints() }) as NextRequest,
    )
    await consumeStream(res)
    await flushMicrotasks()

    expect(createMock).not.toHaveBeenCalled()
    expect(streamMock).toHaveBeenCalledTimes(1)
  })
})
