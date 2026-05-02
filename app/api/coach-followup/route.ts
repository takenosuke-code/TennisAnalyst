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
7. Ground every claim in the OBSERVATIONS list (when present) and the prior analysis. If the player asks about something covered by an observation that the prior text did not surface, address it directly with what the observation shows — do not refuse. If a question cannot be answered from either source, say so honestly and offer to look for it next time.
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

  // Optional structured observation rows from the original /api/analyze
  // call. When present the LLM can ground "what about X?" follow-ups in
  // the actual rule firing instead of refusing because the prior text
  // didn't mention it. Defensive: accept only objects with the four
  // required Observation fields (phase, joint, pattern, severity); drop
  // anything else so a malformed client payload can't poison the prompt.
  const rawObservations = Array.isArray(b.observations) ? b.observations : []
  type ObservationLike = {
    phase: string
    joint: string
    pattern: string
    severity: string
    todayValue?: number
  }
  const observations: ObservationLike[] = []
  for (const o of rawObservations.slice(0, 12)) {
    if (!o || typeof o !== 'object') continue
    const r = o as Record<string, unknown>
    if (
      typeof r.phase === 'string' &&
      typeof r.joint === 'string' &&
      typeof r.pattern === 'string' &&
      typeof r.severity === 'string'
    ) {
      observations.push({
        phase: r.phase,
        joint: r.joint,
        pattern: r.pattern,
        severity: r.severity,
        todayValue: typeof r.todayValue === 'number' ? r.todayValue : undefined,
      })
    }
  }

  function patternHumanLabel(pattern: string): string {
    switch (pattern) {
      case 'cramped_elbow': return 'cramped elbow at contact'
      case 'over_extended_elbow': return 'arm locked straight at contact'
      case 'shallow_knee_load': return 'shallow knee load'
      case 'locked_knees': return 'legs staying tall through the load'
      case 'insufficient_hip_excursion': return 'hips barely turning through the swing'
      case 'insufficient_trunk_excursion': return 'shoulders barely turning through the swing'
      case 'insufficient_unit_turn': return 'no full unit turn into preparation'
      case 'truncated_followthrough': return 'follow-through cut short'
      case 'weak_leg_drive': return 'legs not driving up through the shot'
      case 'short_pushout': return 'racket not extending out toward the target after contact'
      case 'unstable_base': return 'head and base shifting through contact'
      case 'drift_from_baseline': return 'drift from your best-day baseline'
      default: return pattern.replace(/_/g, ' ')
    }
  }
  const observationsBlock =
    observations.length > 0
      ? `\n\nOBSERVATIONS the rule layer flagged on this clip:\n${observations
          .map(
            (o) =>
              `- ${o.joint.replace(/_/g, ' ')} at ${o.phase}: ${patternHumanLabel(o.pattern)} (severity: ${o.severity})`,
          )
          .join('\n')}`
      : ''

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
      content: `Here is the swing analysis you already gave me:\n\n${priorAnalysis}${observationsBlock}`,
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
