// Follow-up Q&A endpoint: takes the prior analysis text + a user
// question + optional prior Q&A turns, streams back a coach-voice
// answer. Same voice rules as /api/analyze.
//
// Lighter than /api/analyze — no observation extraction, no exemplar
// retrieval. The prior analysis already has the observations baked in;
// the model just needs to answer a follow-up against that context.

import { NextRequest } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { createClient } from '@/lib/supabase/server'
import { sanitizePromptInput } from '@/lib/sanitize'
import {
  getCoachingContext,
  registerForTier,
  TIER_MAX_TOKENS,
  type SkillTier,
} from '@/lib/profile'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

const FOLLOWUP_SYSTEM = `You are a veteran tennis coach answering a follow-up question about a swing analysis you already gave the player.
Voice rules:
1. Second person, imperative. Address the player directly.
2. Plain coach language, short sentences.
3. NO numbers, digits, percentages, counts.
4. NO em-dashes / en-dashes (commas or full sentences instead).
5. NO biomechanics jargon (no "kinetic chain", no "trunk rotation", etc.).
6. External focus on racket, ball, court targets, not muscles.
7. Stay grounded in the prior analysis — don't invent observations the analysis doesn't support. If the question can't be answered from what you saw, say so honestly and ask what they'd like you to look for next time.
8. Keep the answer to 2 to 4 short paragraphs at most. No section headers.`

const REJECT_DIGIT_RE = /\d/
const REJECT_JARGON_RE = /(kinetic chain|trunk_rotation|hip_rotation|joint angle|—|–)/i

interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export async function POST(request: NextRequest) {
  let body: unknown
  try {
    body = await request.json()
  } catch {
    return new Response(JSON.stringify({ error: 'invalid-json' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
  }
  if (!body || typeof body !== 'object') {
    return new Response(JSON.stringify({ error: 'invalid-body' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
  }
  const b = body as Record<string, unknown>

  const question = typeof b.question === 'string' ? b.question.trim() : ''
  if (!question) {
    return new Response(JSON.stringify({ error: 'missing-question' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
  }
  const sanitizedQuestion = sanitizePromptInput(question, 600)
  if (!sanitizedQuestion) {
    return new Response(JSON.stringify({ error: 'invalid-question' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const priorAnalysis =
    typeof b.priorAnalysis === 'string' ? b.priorAnalysis : ''
  if (!priorAnalysis.trim()) {
    return new Response(JSON.stringify({ error: 'missing-prior-analysis' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  // Optional: previous Q&A turns. Cap to last 6 to keep token count tight.
  const rawHistory = Array.isArray(b.history) ? b.history : []
  const history: ChatTurn[] = []
  for (const turn of rawHistory.slice(-6)) {
    if (!turn || typeof turn !== 'object') continue
    const t = turn as Record<string, unknown>
    if (
      (t.role === 'user' || t.role === 'assistant') &&
      typeof t.content === 'string'
    ) {
      history.push({ role: t.role, content: t.content })
    }
  }

  // Pull the player's skill tier from the auth profile so the register
  // matches /api/analyze. Falls back to 'intermediate' for skipped /
  // unauthenticated.
  const authClient = await createClient()
  const { profile } = await getCoachingContext(authClient)
  const tier: SkillTier = (profile?.skill_tier as SkillTier) ?? 'intermediate'
  const register = registerForTier(tier)
  const maxTokens = Math.min(TIER_MAX_TOKENS[tier] ?? 600, 700)

  const registerNote =
    register === 'technical'
      ? 'Player has a real racket vocabulary. Reference grip, contact point, racket-face, follow-through naturally.'
      : 'Player is newer. Use everyday body words and metaphors instead of tennis jargon.'

  const systemPrompt = `${FOLLOWUP_SYSTEM}\n\n${registerNote}`

  const messages: { role: 'user' | 'assistant'; content: string }[] = [
    {
      role: 'user',
      content: `Here is the swing analysis you already gave me:\n\n${priorAnalysis}`,
    },
    {
      role: 'assistant',
      content:
        'Got it. I have the analysis in mind. Ask me a follow-up about it and I will answer in coach voice.',
    },
    ...history,
    { role: 'user', content: sanitizedQuestion },
  ]

  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    async start(controller) {
      try {
        const apiStream = anthropic.messages.stream({
          model: 'claude-sonnet-4-5',
          max_tokens: maxTokens,
          system: systemPrompt,
          messages,
        })

        let buffered = ''
        for await (const chunk of apiStream) {
          if (
            chunk.type === 'content_block_delta' &&
            chunk.delta.type === 'text_delta'
          ) {
            const text = chunk.delta.text
            buffered += text
            controller.enqueue(encoder.encode(text))
          }
        }

        // Cheap post-validation. Don't block — log a warning if voice
        // rules slipped (digits, em-dashes, jargon).
        if (
          REJECT_DIGIT_RE.test(buffered) ||
          REJECT_JARGON_RE.test(buffered)
        ) {
          console.warn(
            '[coach-followup] response contained rejected pattern; emitting anyway',
          )
        }

        controller.close()
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'unknown'
        controller.enqueue(encoder.encode(`\n\n[ERROR] ${msg}`))
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-cache',
    },
  })
}
