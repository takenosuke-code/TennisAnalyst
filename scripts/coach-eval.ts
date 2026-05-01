/**
 * coach-eval.ts — side-by-side Haiku 4.5 vs Sonnet 4.6 eval for the
 * per-shot coaching prompt.
 *
 * Run with:
 *   npx tsx scripts/coach-eval.ts > coach-eval-results.md
 *
 * Requires ANTHROPIC_API_KEY in env (the same key the analyze route
 * uses on Vercel — drop it into a local .env.local or export it
 * inline).
 *
 * What this does:
 *   - Picks 3 synthetic-but-realistic angle summaries (a clean
 *     forehand, a forehand with two flagged issues, and a serve with
 *     locked knees).
 *   - Builds the same user prompt the production /api/analyze route
 *     does for the solo / per-shot path, with the same biomechanics
 *     reference + auto-detected mistakes block.
 *   - Calls both models with identical inputs, measures wall-time and
 *     time-to-first-token, captures the full streamed response.
 *   - Writes a markdown report so you can eyeball voice + tone
 *     differences side-by-side.
 *
 * Notes:
 *   - Both calls go through `messages.stream` — no tool_use, since the
 *     proposed migration is to native streaming output. This keeps
 *     the latency comparison apples-to-apples.
 *   - The system prompt + user prompt are reproduced inline rather
 *     than imported from app/api/analyze/route.ts, because that route
 *     is a Next.js handler that pulls Supabase clients etc at module
 *     load. Mirrors the prod prompt verbatim.
 */

import Anthropic from '@anthropic-ai/sdk'
import { getBiomechanicsReference } from '../lib/biomechanics-reference'

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })

if (!process.env.ANTHROPIC_API_KEY) {
  console.error('Missing ANTHROPIC_API_KEY in env. Add it to .env.local and `source .env.local` before running.')
  process.exit(1)
}

// ---------- Test fixtures: realistic angle summaries ----------

interface Fixture {
  name: string
  shotType: 'forehand' | 'backhand' | 'serve'
  angleSummary: string
  detectedMistakes: string[]
  userFocus: string | null
}

const FIXTURES: Fixture[] = [
  {
    name: 'Clean forehand (advanced player)',
    shotType: 'forehand',
    angleSummary: [
      'preparation: elbow_R=115° elbow_L=160° shoulder_R=50° knee_R=145° hip_rot=40° trunk_rot=85°',
      'loading: elbow_R=95° elbow_L=145° shoulder_R=110° knee_R=135° hip_rot=55° trunk_rot=100°',
      'contact: elbow_R=125° elbow_L=145° shoulder_R=95° knee_R=160° hip_rot=58° trunk_rot=50°',
      'follow-through: elbow_R=140° elbow_L=130° shoulder_R=70° knee_R=170° hip_rot=70° trunk_rot=20°',
      'finish: elbow_R=120° elbow_L=140° shoulder_R=45° knee_R=170° hip_rot=75° trunk_rot=-10°',
    ].join('\n'),
    detectedMistakes: [],
    userFocus: null,
  },
  {
    name: 'Forehand with cramped elbow + locked knees (intermediate, mistakes flagged)',
    shotType: 'forehand',
    angleSummary: [
      'preparation: elbow_R=110° elbow_L=160° shoulder_R=45° knee_R=175° hip_rot=10° trunk_rot=30°',
      'loading: elbow_R=85° elbow_L=145° shoulder_R=95° knee_R=178° hip_rot=20° trunk_rot=45°',
      'contact: elbow_R=82° elbow_L=140° shoulder_R=85° knee_R=178° hip_rot=25° trunk_rot=20°',
      'follow-through: elbow_R=130° elbow_L=125° shoulder_R=70° knee_R=175° hip_rot=30° trunk_rot=10°',
      'finish: elbow_R=125° elbow_L=140° shoulder_R=50° knee_R=175° hip_rot=35° trunk_rot=5°',
    ].join('\n'),
    detectedMistakes: [
      "Cramped elbow at contact: You're hitting too close to your body — reach out and meet the ball in front.",
      "Locked knees: Straight legs are dead legs. Bend and feel the ground push you into the shot.",
    ],
    userFocus: null,
  },
  {
    name: 'Serve with focus question on knee bend',
    shotType: 'serve',
    angleSummary: [
      'preparation: elbow_R=160° elbow_L=160° shoulder_R=30° knee_R=165° hip_rot=20° trunk_rot=20°',
      'loading: elbow_R=70° elbow_L=170° shoulder_R=170° knee_R=162° hip_rot=15° trunk_rot=40°',
      'contact: elbow_R=175° elbow_L=80° shoulder_R=175° knee_R=175° hip_rot=10° trunk_rot=10°',
      'follow-through: elbow_R=140° elbow_L=70° shoulder_R=140° knee_R=170° hip_rot=5° trunk_rot=-10°',
      'finish: elbow_R=120° elbow_L=80° shoulder_R=80° knee_R=175° hip_rot=0° trunk_rot=-20°',
    ].join('\n'),
    detectedMistakes: [
      "Insufficient knee bend on serve: You're loading too shallow — drop into the back leg before you go up.",
    ],
    userFocus: 'is my knee bend deep enough on the serve?',
  },
]

// ---------- Prompt builder (mirrors app/api/analyze/route.ts solo path) ----------

function buildSoloPrompt(f: Fixture): string {
  const ref = getBiomechanicsReference(f.shotType)

  const shotIntroBlock = `\nSHOT TYPE: You are analyzing a ${f.shotType} swing. Tailor every cue and reference angle to this specific shot. Do NOT generalize across shot types. Do NOT open by naming the shot type.\n`

  const detectedIssuesBlock = f.detectedMistakes.length > 0
    ? `\nAUTO-DETECTED ISSUES (system-flagged for this swing, weave coaching around these, do not just list them):\n${f.detectedMistakes.map((m, i) => `${i + 1}. ${m}`).join('\n')}\n`
    : ''

  const focusBlock = f.userFocus
    ? `\nTHE PLAYER ASKED: "${f.userFocus}"\nThis is the QUESTION you must answer. Open your response by directly addressing this exact question in 1 to 2 sentences before anything else. Do not generalize. Do not pivot to unrelated cues. After answering their question, you can add the standard coaching sections, but the question comes first and gets a real answer grounded in their swing data.\n`
    : ''

  // Generic tier block (no profile available in the eval, mirroring
  // the anonymous-user path).
  const tierBlock = `\nTIER: This player has not self-reported a skill tier. Calibrate your tone for an intermediate player. Use clear language, give 3 actionable cues, and keep the tone encouraging.\n`

  return `You are a tennis coach talking to a player right after watching their swing on video. Be encouraging and practical.
${shotIntroBlock}
${tierBlock}
${focusBlock}${detectedIssuesBlock}
STRICT RULES:
- NEVER mention degrees, angles, or numbers of any kind. Not even once. Describe everything in feel and body language.
- NEVER rate or score the player (no X/100, no percentages, no grades). Just give advice.
- WRITE LIKE A HUMAN COACH TALKING. Never use em dashes (—), en dashes (–), or hyphens (-) as a way to chain ideas or replace punctuation. Do not use them for emphasis, parenthetical asides, or to glue two clauses together. If you would have used a dash, write a full sentence with a period, OR connect the clauses with a comma, "and", "but", or a colon. Compound-word hyphens inside a single word like "feel-based" or "external-focus" are fine.
- Talk like a real person. Short sentences. "You" and "your" constantly.
- If the swing is already clean, SAY THAT and give fine-tuning cues. Don't fabricate problems.

Use the data below to understand what's happening, but ONLY talk in coaching language. The player never sees these numbers.

REFERENCE: ${ref}
USER SWING DATA: ${f.angleSummary}

Respond in this format:

## What You're Doing Well
Two or three sentences about what's genuinely working in their swing. Be specific, not generic.

## 3 Things to Work On

**1. [Short coaching cue]**
What you see, why it matters for their game, how it compares to solid technique. Give one drill or feel-based tip they can try on the next ball. Use phrases like "load into your legs", "let the racket drop behind you", "turn your hips before your shoulders".

**2. [Short coaching cue]**
Same approach. Observation, why it matters, one actionable tip.

**3. [Short coaching cue]**
Same approach. Observation, why it matters, one actionable tip.

## Your Practice Plan
Three specific things to focus on in their next hitting session. Make each one a single sentence they can remember on court.

Keep it under 350 words. Sound like a coach who believes in this player.`
}

const SYSTEM_PROMPT = `You are a veteran tennis coach who has been on court for 30 years. You talk like a real person having a conversation with a player between points.

VOICE RULES (follow these strictly):
- Write like you TALK. Short sentences. Casual tone. "Your hips are way ahead of your arm here" not "The hip rotation precedes the arm extension."
- WRITE LIKE A HUMAN COACH TALKING. Never use em dashes (—), en dashes (–), or hyphens (-) as a way to chain ideas or replace punctuation. Do not use them for emphasis, parenthetical asides, or to glue two clauses together. If you would have used a dash, write a full sentence with a period, OR connect the clauses with a comma, "and", "but", or a colon. Compound-word hyphens inside a single word like "feel-based" or "external-focus" are fine.
- NEVER list raw degree numbers on their own. Wrong: "elbow: 170°, ideal: 110°". Right: "your arm is almost locked straight when you want a nice relaxed bend."
- You CAN mention a number to back up a point, but always in a natural sentence: "you're only getting about 20 degrees of hip turn when ideal range is around 45."
- Use coaching cues a player can FEEL: "load your weight into your back leg", "let the racket drop behind you like a pendulum", "snap your hips like you're throwing a punch."
- Keep it practical. Every piece of advice should be something they can try on the very next ball.
- Sound encouraging, not critical. You're helping, not grading.
- No bullet points with just numbers. No tables. No clinical language.
- Write "you" and "your" constantly. Talk TO the player.`

// ---------- Run a streamed call against one model, time it ----------

interface CallResult {
  model: string
  ttftMs: number
  totalMs: number
  outputTokens: number | null
  text: string
}

async function runOne(
  model: 'claude-sonnet-4-6' | 'claude-haiku-4-5',
  prompt: string,
  maxTokens: number,
): Promise<CallResult> {
  const t0 = Date.now()
  let firstTokenAt: number | null = null
  let text = ''

  const stream = anthropic.messages.stream({
    model,
    max_tokens: maxTokens,
    system: [
      { type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } },
    ],
    messages: [{ role: 'user', content: prompt }],
  })

  for await (const event of stream) {
    if (
      event.type === 'content_block_delta' &&
      event.delta.type === 'text_delta'
    ) {
      if (firstTokenAt === null) firstTokenAt = Date.now()
      text += event.delta.text
    }
  }

  const final = await stream.finalMessage()
  const totalMs = Date.now() - t0
  const ttftMs = firstTokenAt !== null ? firstTokenAt - t0 : totalMs

  return {
    model,
    ttftMs,
    totalMs,
    outputTokens: final.usage?.output_tokens ?? null,
    text,
  }
}

// ---------- Main: run every fixture × every model, write report ----------

async function main() {
  const lines: string[] = []
  lines.push('# Coach Eval — Haiku 4.5 vs Sonnet 4.6 (per-shot prompt)')
  lines.push('')
  lines.push(`Generated ${new Date().toISOString()}`)
  lines.push('')
  lines.push('Both models run via `messages.stream` (no tool_use). Same system prompt + user prompt. `max_tokens=600`. Identical biomechanics reference + auto-detected mistakes block.')
  lines.push('')

  for (const fixture of FIXTURES) {
    lines.push('---')
    lines.push('')
    lines.push(`## Fixture: ${fixture.name}`)
    lines.push('')
    lines.push('**Angle summary input:**')
    lines.push('```')
    lines.push(fixture.angleSummary)
    lines.push('```')
    lines.push('')
    if (fixture.detectedMistakes.length > 0) {
      lines.push('**Pre-flagged mistakes:**')
      for (const m of fixture.detectedMistakes) lines.push(`- ${m}`)
      lines.push('')
    }
    if (fixture.userFocus) {
      lines.push(`**User focus question:** "${fixture.userFocus}"`)
      lines.push('')
    }

    const prompt = buildSoloPrompt(fixture)

    for (const model of ['claude-sonnet-4-6', 'claude-haiku-4-5'] as const) {
      console.error(`[${fixture.name}] running ${model}...`)
      try {
        const r = await runOne(model, prompt, 600)
        lines.push(`### ${model}`)
        lines.push('')
        lines.push(`*TTFT: ${r.ttftMs} ms · total: ${r.totalMs} ms · output: ${r.outputTokens ?? '?'} tok*`)
        lines.push('')
        lines.push(r.text)
        lines.push('')
      } catch (err) {
        lines.push(`### ${model}`)
        lines.push('')
        lines.push(`**FAILED**: ${err instanceof Error ? err.message : String(err)}`)
        lines.push('')
      }
    }
  }

  process.stdout.write(lines.join('\n') + '\n')
}

main().catch((err) => {
  console.error('Eval failed:', err)
  process.exit(1)
})
