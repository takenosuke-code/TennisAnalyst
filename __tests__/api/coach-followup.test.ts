/**
 * Tests for /api/coach-followup. Verifies that observations passed in
 * via the request body are rendered into the LLM context, so a "what
 * about my legs?" follow-up can be answered concretely instead of
 * deflected as "I didn't focus on that."
 */
import { describe, it, expect, vi } from 'vitest'
import { NextRequest } from 'next/server'

const { lastArgsBox, mockGetCoachingContext } = vi.hoisted(() => ({
  lastArgsBox: { current: null as null | {
    model: string
    max_tokens: number
    system: string
    messages: { role: string; content: string }[]
  } },
  mockGetCoachingContext: vi.fn(),
}))

vi.mock('@anthropic-ai/sdk', () => ({
  default: class {
    messages = {
      stream(args: {
        model: string
        max_tokens: number
        system: string
        messages: { role: string; content: string }[]
      }) {
        lastArgsBox.current = args
        return (async function* () {
          yield {
            type: 'content_block_delta',
            delta: { type: 'text_delta', text: 'OK' },
          }
        })()
      },
    }
  },
}))

vi.mock('@/lib/profile', async () => {
  const actual = await vi.importActual<typeof import('@/lib/profile')>('@/lib/profile')
  return {
    ...actual,
    getCoachingContext: mockGetCoachingContext,
  }
})

vi.mock('@/lib/supabase/server', () => ({
  createClient: vi.fn(async () => ({})),
}))

function makeReq(body: unknown): NextRequest {
  return new NextRequest('http://localhost/api/coach-followup', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

async function readStream(res: Response): Promise<string> {
  if (!res.body) return ''
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
  }
  return buf
}

describe('POST /api/coach-followup', () => {
  it('rejects requests with no question', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: { skill_tier: 'intermediate' },
      skipped: false,
    })
    const { POST } = await import('@/app/api/coach-followup/route')
    const res = await POST(makeReq({ priorAnalysis: 'something' }))
    expect(res.status).toBe(400)
  })

  it('rejects requests with no priorAnalysis', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: { skill_tier: 'intermediate' },
      skipped: false,
    })
    const { POST } = await import('@/app/api/coach-followup/route')
    const res = await POST(makeReq({ question: 'what about legs?' }))
    expect(res.status).toBe(400)
  })

  it('renders observations into the user message when provided', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: { skill_tier: 'intermediate' },
      skipped: false,
    })
    const { POST } = await import('@/app/api/coach-followup/route')
    const res = await POST(
      makeReq({
        question: 'What about my legs?',
        priorAnalysis: '## Primary cue\nFinish your follow-through.',
        observations: [
          {
            phase: 'follow-through',
            joint: 'shoulders',
            pattern: 'truncated_followthrough',
            severity: 'moderate',
            confidence: 0.5,
            todayValue: 8,
          },
          {
            phase: 'contact',
            joint: 'hips',
            pattern: 'weak_leg_drive',
            severity: 'severe',
            confidence: 0.6,
            todayValue: 1,
          },
        ],
      }),
    )
    expect(res.status).toBe(200)
    await readStream(res)
    expect(lastArgsBox.current).not.toBeNull()
    // First message in the seeded thread carries the prior analysis +
    // an OBSERVATIONS block listing the rule firings.
    const firstUser = lastArgsBox.current!.messages[0].content
    expect(firstUser).toContain('OBSERVATIONS')
    // patternHumanLabel turns the raw pattern into coach-voice prose.
    expect(firstUser).toMatch(/follow-through cut short/)
    expect(firstUser).toMatch(/legs not driving up through the shot/)
    // The system prompt instructs the model to ground in the
    // observations rather than deflect.
    expect(lastArgsBox.current!.system).toMatch(/Ground every claim/i)
  })

  it('drops malformed observation rows but keeps valid ones', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: { skill_tier: 'intermediate' },
      skipped: false,
    })
    const { POST } = await import('@/app/api/coach-followup/route')
    const res = await POST(
      makeReq({
        question: 'q',
        priorAnalysis: 'p',
        observations: [
          // Missing required fields — should be ignored.
          { phase: 'contact' },
          'not an object',
          null,
          // Valid row — should pass through.
          {
            phase: 'contact',
            joint: 'hips',
            pattern: 'weak_leg_drive',
            severity: 'severe',
          },
        ],
      }),
    )
    expect(res.status).toBe(200)
    await readStream(res)
    const firstUser = lastArgsBox.current!.messages[0].content
    expect(firstUser).toContain('OBSERVATIONS')
    expect(firstUser).toContain('legs not driving up through the shot')
    // Only one valid observation should be rendered.
    const obsLines = firstUser
      .split('\n')
      .filter((l) => l.startsWith('- '))
    expect(obsLines.length).toBe(1)
  })

  it('omits the OBSERVATIONS block when no observations are provided', async () => {
    mockGetCoachingContext.mockResolvedValue({
      profile: { skill_tier: 'intermediate' },
      skipped: false,
    })
    const { POST } = await import('@/app/api/coach-followup/route')
    const res = await POST(
      makeReq({
        question: 'q',
        priorAnalysis: 'p',
      }),
    )
    expect(res.status).toBe(200)
    await readStream(res)
    const firstUser = lastArgsBox.current!.messages[0].content
    expect(firstUser).not.toContain('OBSERVATIONS')
  })
})
