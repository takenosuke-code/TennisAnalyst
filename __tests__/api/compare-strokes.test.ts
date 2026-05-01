import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'
import { observationId, type Observation } from '@/lib/coachingObservations'
import { CUE_EXEMPLARS } from '@/lib/cueExemplars'

// vi.hoisted runs before any vi.mock factory is evaluated, so factories
// can safely close over these refs. Mirrors the analyze-stream.test.ts
// pattern adapted to messages.create() instead of messages.stream().
const { lastArgsBox, responseQueue, mockGetCoachingContext } = vi.hoisted(() => ({
  lastArgsBox: { current: null as null | {
    model: string
    max_tokens: number
    system: string | { type: string; text: string }[]
    messages: { role: string; content: string }[]
    tools: unknown[]
    tool_choice: { type: string; name: string }
  } },
  // Each entry is the value `messages.create()` should resolve to. Tests
  // push tool_use blocks (or { error: 'msg' } to simulate API failure).
  responseQueue: [] as Array<
    | { kind: 'tool_use'; input: unknown }
    | { kind: 'no_tool' }
    | { kind: 'error'; message: string }
  >,
  mockGetCoachingContext: vi.fn(),
}))

vi.mock('@anthropic-ai/sdk', () => ({
  default: class {
    messages = {
      async create(args: {
        model: string
        max_tokens: number
        system: string | { type: string; text: string }[]
        messages: { role: string; content: string }[]
        tools: unknown[]
        tool_choice: { type: string; name: string }
      }) {
        lastArgsBox.current = args
        const next = responseQueue.shift()
        if (!next) {
          // Default: emit a no-op tool_use that will fail validation. Tests
          // should always queue an explicit response.
          return {
            content: [],
            usage: { output_tokens: 0 },
          }
        }
        if (next.kind === 'error') {
          throw new Error(next.message)
        }
        if (next.kind === 'no_tool') {
          return { content: [{ type: 'text', text: 'no tool call' }], usage: { output_tokens: 5 } }
        }
        return {
          content: [
            {
              type: 'tool_use',
              name: args.tool_choice.name,
              input: next.input,
            },
          ],
          usage: { output_tokens: 50 },
        }
      },
    }
  },
}))

vi.mock('@/lib/supabase/server', () => ({
  createClient: async () => ({
    auth: {
      getUser: async () => ({ data: { user: null } }),
    },
  }),
}))

vi.mock('@/lib/profile', async () => {
  const actual = await vi.importActual<typeof import('@/lib/profile')>('@/lib/profile')
  return {
    ...actual,
    getCoachingContext: mockGetCoachingContext,
  }
})

// ---------- Helpers ----------

function makeReq(body: object): NextRequest {
  return new NextRequest('http://localhost/api/compare-strokes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

function obs(partial: Partial<Observation> & {
  joint: string
  phase: Observation['phase']
  pattern: Observation['pattern']
}): Observation {
  return {
    severity: 'moderate',
    confidence: 0.6,
    todayValue: 100,
    ...partial,
  } as Observation
}

function cueIdFor(pattern: Observation['pattern'], phase: string | undefined, register: 'plain' | 'technical') {
  // Mirrors cueIdFor() in route.ts: pattern|phase|first 32 chars of text.
  const ex = CUE_EXEMPLARS.find((e) => e.pattern === pattern && (phase ? e.phase === phase : !e.phase))
  if (!ex) throw new Error(`no exemplar found for ${pattern} / ${phase}`)
  const text = register === 'technical' ? ex.technical : ex.plain
  return `${ex.pattern}|${ex.phase ?? '*'}|${text.slice(0, 32)}`
}

// Five strokes with tight scores -> tie path.
function tightStrokes() {
  return [0.05, 0.1, -0.05, 0.0, 0.15].map((score, i) => ({
    strokeId: `s${i}`,
    score,
    rejected: false,
    observations: [],
  }))
}

// Five strokes with a clear best/worst spread. Best = s0 (1.2), Worst = s4 (-1.5).
function spreadStrokes() {
  return [
    {
      strokeId: 's0',
      score: 1.2,
      rejected: false,
      observations: [
        obs({ joint: 'right_elbow', phase: 'contact', pattern: 'cramped_elbow' }),
      ],
    },
    {
      strokeId: 's1',
      score: 0.6,
      rejected: false,
      observations: [],
    },
    {
      strokeId: 's2',
      score: 0.0,
      rejected: false,
      observations: [],
    },
    {
      strokeId: 's3',
      score: -0.4,
      rejected: false,
      observations: [],
    },
    {
      strokeId: 's4',
      score: -1.5,
      rejected: false,
      observations: [
        obs({
          joint: 'right_knee',
          phase: 'loading',
          pattern: 'shallow_knee_load',
        }),
        obs({
          joint: 'shoulders',
          phase: 'loading',
          pattern: 'insufficient_trunk_excursion',
        }),
      ],
    },
  ]
}

// ---------- Tests ----------

describe('POST /api/compare-strokes', () => {
  beforeEach(() => {
    vi.resetModules()
    lastArgsBox.current = null
    responseQueue.length = 0
    mockGetCoachingContext.mockReset()
    mockGetCoachingContext.mockResolvedValue({
      profile: {
        skill_tier: 'intermediate',
        dominant_hand: 'right',
        backhand_style: 'two_handed',
        primary_goal: 'consistency',
      },
      skipped: false,
    })
  })

  // -------------------------------------------------------------------
  // Happy path
  // -------------------------------------------------------------------

  it('happy path: returns best+worst with valid citations', async () => {
    const strokes = spreadStrokes()
    // Build observation_ids that the LLM can cite.
    const bestObsId = observationId('s0', strokes[0].observations[0])
    const worstObsId1 = observationId('s4', strokes[4].observations[0])
    const worstObsId2 = observationId('s4', strokes[4].observations[1])

    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's0',
          // Each sentence ends with [obs_id]. Word count (excl. brackets)
          // sits inside 20-55. No digits, no em-dashes, no jargon.
          reasoning:
            `Your best stroke held the contact window in front of the hip and the racket arm stayed long through the ball [${bestObsId}]. Trust this reach and groove the same shape on the next basket so the pattern repeats every ball [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: [],
        },
        worst: {
          strokeId: 's4',
          reasoning:
            `Your worst stroke leaked structure in the load and the shoulders barely turned, so the legs had no spring and the chest stayed open early [${worstObsId1},${worstObsId2}]. Sit deeper into the back leg and finish the unit turn before you swing forward to bring the structure back [${worstObsId1}].`,
          citations: [worstObsId1, worstObsId2],
          citedCues: [
            cueIdFor('shallow_knee_load', 'loading', 'plain'),
          ],
        },
      },
    })

    const { POST } = await import('@/app/api/compare-strokes/route')
    const res = await POST(makeReq({ strokes, shotType: 'forehand' }))
    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.isConsistent).toBe(false)
    expect(json.best?.strokeId).toBe('s0')
    expect(json.worst?.strokeId).toBe('s4')
    expect(json.best.citations).toContain(bestObsId)
    expect(json.worst.citations).toEqual(
      expect.arrayContaining([worstObsId1, worstObsId2]),
    )
    // No retry happened: queue should be empty (we only pushed 1).
    expect(responseQueue.length).toBe(0)
  })

  // -------------------------------------------------------------------
  // Tie suppression
  // -------------------------------------------------------------------

  it('tie path: tight scores -> isConsistent=true, best/worst null, consistencyCue present', async () => {
    const strokes = tightStrokes()
    const { POST } = await import('@/app/api/compare-strokes/route')
    const res = await POST(makeReq({ strokes, shotType: 'forehand' }))
    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.isConsistent).toBe(true)
    expect(json.best).toBeNull()
    expect(json.worst).toBeNull()
    expect(typeof json.consistentCue).toBe('string')
    expect(json.consistentCue.length).toBeGreaterThan(20)
    // Anthropic should NOT have been called on the tie path.
    expect(lastArgsBox.current).toBeNull()
  })

  // -------------------------------------------------------------------
  // Citation rejection -> retry succeeds
  // -------------------------------------------------------------------

  it('citation rejection: uncited sentence triggers retry; second attempt clean -> succeeds', async () => {
    const strokes = spreadStrokes()
    const bestObsId = observationId('s0', strokes[0].observations[0])
    const worstObsId1 = observationId('s4', strokes[4].observations[0])
    const worstObsId2 = observationId('s4', strokes[4].observations[1])

    // First attempt: best.reasoning has a sentence with NO citation tag.
    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's0',
          reasoning:
            `Your best stroke held its shape clean. The arm stayed long and the contact point sat in front [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: [],
        },
        worst: {
          strokeId: 's4',
          reasoning:
            `Your worst stroke leaked structure in the load and the shoulders barely turned, so the legs had no spring and the chest stayed open early [${worstObsId1},${worstObsId2}]. Sit deeper into the back leg and finish the unit turn before you swing forward to bring the structure back [${worstObsId1}].`,
          citations: [worstObsId1, worstObsId2],
          citedCues: [],
        },
      },
    })
    // Retry: clean.
    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's0',
          reasoning:
            `Your best stroke kept the racket arm long and the contact point in front of the hip the whole way through [${bestObsId}]. Trust this reach and groove the same shape on the next basket so the pattern keeps repeating [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: [],
        },
        worst: {
          strokeId: 's4',
          reasoning:
            `Your worst stroke leaked structure in the load and the shoulders barely turned, so the legs had no spring and the chest stayed open early [${worstObsId1},${worstObsId2}]. Sit deeper into the back leg and finish the unit turn before you swing forward to bring the structure back [${worstObsId1}].`,
          citations: [worstObsId1, worstObsId2],
          citedCues: [],
        },
      },
    })

    const { POST } = await import('@/app/api/compare-strokes/route')
    const res = await POST(makeReq({ strokes }))
    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.isConsistent).toBe(false)
    expect(json.best?.strokeId).toBe('s0')
    // Both responses consumed -> retry happened.
    expect(responseQueue.length).toBe(0)
  })

  // -------------------------------------------------------------------
  // Out-of-vocab cue rejection
  // -------------------------------------------------------------------

  it('out-of-vocab cue rejection: cue id not in CUE_EXEMPLARS -> rejected', async () => {
    const strokes = spreadStrokes()
    const bestObsId = observationId('s0', strokes[0].observations[0])
    const worstObsId1 = observationId('s4', strokes[4].observations[0])

    // First attempt: cites a fabricated cue id.
    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's0',
          reasoning:
            `Your best stroke held its shape clean and the arm stayed long through contact, with the ball in front of the hip on every read [${bestObsId}]. Trust this reach and groove the same shape next time [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: ['fabricated_cue|*|nonsense text'],
        },
        worst: {
          strokeId: 's4',
          reasoning:
            `Your worst stroke leaked structure in the load and the legs had no spring at all in the contact window or after [${worstObsId1}]. Sit deeper into the back leg before the next ball [${worstObsId1}].`,
          citations: [worstObsId1],
          citedCues: [],
        },
      },
    })
    // Both retries also bad -> deterministic fallback fires.
    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's0',
          reasoning:
            `Your best stroke held its shape clean and the arm stayed long through contact, with the ball in front of the hip on every read [${bestObsId}]. Trust this reach and groove the same shape next time [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: ['another_invalid|*|still fake'],
        },
        worst: {
          strokeId: 's4',
          reasoning:
            `Your worst stroke leaked structure in the load and the legs had no spring at all in the contact window or after [${worstObsId1}]. Sit deeper into the back leg before the next ball [${worstObsId1}].`,
          citations: [worstObsId1],
          citedCues: [],
        },
      },
    })
    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's0',
          reasoning:
            `Your best stroke held its shape clean and the arm stayed long through contact, with the ball in front of the hip on every read [${bestObsId}]. Trust this reach and groove the same shape next time [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: ['yet_another_bad_id'],
        },
        worst: {
          strokeId: 's4',
          reasoning:
            `Your worst stroke leaked structure in the load and the legs had no spring at all in the contact window or after [${worstObsId1}]. Sit deeper into the back leg before the next ball [${worstObsId1}].`,
          citations: [worstObsId1],
          citedCues: [],
        },
      },
    })

    const { POST } = await import('@/app/api/compare-strokes/route')
    const res = await POST(makeReq({ strokes }))
    expect(res.status).toBe(200)
    const json = await res.json()
    // After 3 bad attempts the route falls back to the deterministic
    // template, which DOES return non-null best/worst (with citations),
    // not the consistency path. The test's contract is just "the bad cue
    // id never made it into the final output".
    expect(JSON.stringify(json)).not.toContain('fabricated_cue')
    expect(JSON.stringify(json)).not.toContain('yet_another_bad_id')
    expect(responseQueue.length).toBe(0)
  })

  // -------------------------------------------------------------------
  // All-rejected strokes
  // -------------------------------------------------------------------

  it('all-rejected strokes -> empty/null result, no error', async () => {
    const strokes = [0, 1, 2, 3, 4].map((i) => ({
      strokeId: `s${i}`,
      score: NaN,
      rejected: true,
      observations: [],
    }))
    const { POST } = await import('@/app/api/compare-strokes/route')
    const res = await POST(makeReq({ strokes }))
    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.best).toBeNull()
    expect(json.worst).toBeNull()
    expect(json.isConsistent).toBe(false)
    // No LLM call.
    expect(lastArgsBox.current).toBeNull()
  })

  // -------------------------------------------------------------------
  // 2 strokes -> tie path (no ranking forced)
  // -------------------------------------------------------------------

  it('2-stroke pool -> tie path (no ranking forced)', async () => {
    const strokes = [
      { strokeId: 's0', score: 1.5, rejected: false, observations: [] },
      { strokeId: 's1', score: -1.5, rejected: false, observations: [] },
    ]
    const { POST } = await import('@/app/api/compare-strokes/route')
    const res = await POST(makeReq({ strokes }))
    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.best).toBeNull()
    expect(json.worst).toBeNull()
    expect(json.isConsistent).toBe(true)
    expect(typeof json.consistentCue).toBe('string')
    expect(lastArgsBox.current).toBeNull()
  })

  // -------------------------------------------------------------------
  // Rejected strokes excluded from best/worst pool
  // -------------------------------------------------------------------

  it('rejected strokes are excluded from best/worst pool but observations still allowed', async () => {
    // s0 is rejected with the highest score; the actual best should be s1.
    // s4 (worst) is non-rejected.
    const strokes = [
      {
        strokeId: 's0',
        score: 99, // would be best but is rejected
        rejected: true,
        observations: [obs({ joint: 'right_elbow', phase: 'contact', pattern: 'cramped_elbow' })],
      },
      {
        strokeId: 's1',
        score: 1.2,
        rejected: false,
        observations: [
          obs({ joint: 'right_elbow', phase: 'contact', pattern: 'cramped_elbow' }),
        ],
      },
      { strokeId: 's2', score: 0.4, rejected: false, observations: [] },
      { strokeId: 's3', score: 0.0, rejected: false, observations: [] },
      {
        strokeId: 's4',
        score: -1.4,
        rejected: false,
        observations: [
          obs({ joint: 'shoulders', phase: 'loading', pattern: 'insufficient_trunk_excursion' }),
        ],
      },
    ]
    const bestObsId = observationId('s1', strokes[1].observations[0])
    const worstObsId = observationId('s4', strokes[4].observations[0])

    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's1',
          reasoning:
            `Your best stroke held the racket arm long and the ball stayed out in front of the hip every time you swung [${bestObsId}]. Trust this reach and groove the same shape across the next basket so it keeps repeating [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: [],
        },
        worst: {
          strokeId: 's4',
          reasoning:
            `Your worst stroke barely turned the shoulders, so the chest opened early and the racket had nowhere to load [${worstObsId}]. Show your back to the net longer before the swing comes forward and the structure returns [${worstObsId}].`,
          citations: [worstObsId],
          citedCues: [],
        },
      },
    })

    const { POST } = await import('@/app/api/compare-strokes/route')
    const res = await POST(makeReq({ strokes }))
    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.isConsistent).toBe(false)
    // Rejected stroke s0 must NOT be picked as best.
    expect(json.best?.strokeId).toBe('s1')
    expect(json.worst?.strokeId).toBe('s4')
  })

  // -------------------------------------------------------------------
  // Word count enforcement
  // -------------------------------------------------------------------

  it('word count out of budget triggers retry', async () => {
    const strokes = spreadStrokes()
    const bestObsId = observationId('s0', strokes[0].observations[0])
    const worstObsId1 = observationId('s4', strokes[4].observations[0])

    // First attempt: too short (<20 words after stripping citations).
    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's0',
          reasoning: `Held shape [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: [],
        },
        worst: {
          strokeId: 's4',
          reasoning: `Lost shape [${worstObsId1}].`,
          citations: [worstObsId1],
          citedCues: [],
        },
      },
    })
    // Retry: clean.
    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's0',
          reasoning:
            `Your best stroke kept the racket arm long and the contact point in front of the hip every single time [${bestObsId}]. Trust this reach and groove the same shape across the next basket so the pattern keeps repeating [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: [],
        },
        worst: {
          strokeId: 's4',
          reasoning:
            `Your worst stroke leaked structure in the load and the legs had no spring at all when contact came around [${worstObsId1}]. Sit deeper into the back leg before the next ball and the structure returns to the swing [${worstObsId1}].`,
          citations: [worstObsId1],
          citedCues: [],
        },
      },
    })

    const { POST } = await import('@/app/api/compare-strokes/route')
    const res = await POST(makeReq({ strokes }))
    expect(res.status).toBe(200)
    const json = await res.json()
    expect(json.best?.strokeId).toBe('s0')
    expect(responseQueue.length).toBe(0)
  })

  // -------------------------------------------------------------------
  // Tool call passes the right schema
  // -------------------------------------------------------------------

  it('passes the emit_comparison tool with tool_choice forced', async () => {
    const strokes = spreadStrokes()
    const bestObsId = observationId('s0', strokes[0].observations[0])
    const worstObsId1 = observationId('s4', strokes[4].observations[0])
    responseQueue.push({
      kind: 'tool_use',
      input: {
        best: {
          strokeId: 's0',
          reasoning:
            `Your best stroke kept the racket arm long and the contact point in front of the hip every single time [${bestObsId}]. Trust this reach and groove the same shape across the next basket so the pattern keeps repeating [${bestObsId}].`,
          citations: [bestObsId],
          citedCues: [],
        },
        worst: {
          strokeId: 's4',
          reasoning:
            `Your worst stroke leaked structure in the load and the legs had no spring at all when contact came around [${worstObsId1}]. Sit deeper into the back leg before the next ball and the structure returns to the swing [${worstObsId1}].`,
          citations: [worstObsId1],
          citedCues: [],
        },
      },
    })
    const { POST } = await import('@/app/api/compare-strokes/route')
    await POST(makeReq({ strokes }))
    expect(lastArgsBox.current).not.toBeNull()
    expect(lastArgsBox.current!.tool_choice).toEqual({
      type: 'tool',
      name: 'emit_comparison',
    })
    // Tool list should include exactly the emit_comparison schema.
    expect(Array.isArray(lastArgsBox.current!.tools)).toBe(true)
    const tool = lastArgsBox.current!.tools[0] as { name: string }
    expect(tool.name).toBe('emit_comparison')
  })
})
