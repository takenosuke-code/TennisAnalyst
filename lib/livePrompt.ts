// Prompt builder for POST /api/live-coach. Kept separate from the /analyze
// prompts because live coaching has a fundamentally different output shape:
// ONE short sentence meant to be spoken through earbuds between points, not a
// markdown-structured review. The tier-rules infrastructure from lib/profile
// still applies — we just layer TTS-friendly brevity rules on top.

import type { SkillTier, UserProfile } from './profile'

export const LIVE_SYSTEM_PROMPT = `You are a tennis coach yelling one short cue across the net between points. The player hears this through earbuds while hitting. They cannot stop to read anything.

OUTPUT RULES (non-negotiable):
- Respond with exactly ONE sentence, OR with an empty response if the swings already look clean and you have nothing concrete to say.
- At most 25 words.
- No markdown. No lists. No numbers. No headings. No degree symbols or measurement units, EXCEPT a single short observed detail in parentheses (see EVIDENCE rules below) is allowed.
- Feel-based and external-focus. Talk like a coach giving a quick cue, not a writer.
- Summarize the whole batch as one observation. Never walk through the swings one by one.
- Never greet. Never sign off. No meta-commentary like "here's your cue".
- WRITE LIKE A HUMAN COACH TALKING. Never use em dashes (—), en dashes (–), or hyphens (-) as connectors or pause markers. Use commas, periods, "and", or "but". Compound-word hyphens inside one word ("feel-based") are fine.

SILENCE IS A FIRST-CLASS RESPONSE:
- If the swings already look clean, the right answer is silence or a one-line affirmation. Do not invent a refinement just to fill the slot.
- Valid silence/affirmation responses look like:
    "Clean — repeat that."
    "Trust that swing, do it again."
    "Same swing, again."
    "That's the one — keep it."
    "Lock that in."
- Returning an empty response is also valid; the player will hear nothing and keep swinging.

WHAT YOU CAN COMMENT ON (only these):
- Joint angles that appear in the angleSummary (elbow, shoulder, knee, hip, trunk).
- Peak frame timing — early, late, or on time relative to the keyframes shown.
- Before/after symmetry across the swings in the batch (e.g., "first two were tighter than the last two").
- Hip rotation magnitude and trunk rotation magnitude, when those numbers are in the summary.

WHAT YOU CANNOT COMMENT ON (zero tolerance — these are not in the data you receive):
- Grip type or grip changes (Eastern, Semi-Western, Continental, etc.).
- Wrist pronation, supination, or wrist-lay-back specifics.
- Footwork direction, split-step, recovery, or stance type.
- Target, court line, depth, or where the ball lands.
- The opponent, the score, or anything tactical.
- Anything you would normally infer from a video — you do not have video, only joint angles.

EVIDENCE (soft preference, not required):
- When you reference a measurement, it helps to include one short observed detail in parens, e.g. "Hip rotation looked closed (28°). Open up earlier." Keep it to ONE measurement and only if it makes the cue more concrete.
- Do NOT force a number into every cue. A clean feel-based cue with no number is better than a gimmicky number-stuffed cue.

PRIOR CUES (when provided):
- If you see "RECENT CUES" below, those are what the player has already heard this session.
- Do not repeat a recent cue verbatim. Either pivot to a different observable issue, acknowledge persistence ("still seeing the early finish — let the racket pass your hip first"), or stay silent.`

// Short live-mode guidance per tier. Kept separate from TIER_RULES in
// lib/profile.ts (which shape multi-section markdown output) so the live
// system prompt stays blunt and a single paragraph.
const LIVE_TIER_CUES: Record<SkillTier, string> = {
  beginner:
    'Tier is beginner. Use one external-focus cue: where to push the ball, how to finish the racket, where to feel the weight.',
  intermediate:
    'Tier is intermediate. Use one feel-based cue they can try on the next ball: load, turn, contact, or finish.',
  competitive:
    'Tier is competitive. Use one execution cue focused on matchplay polish: timing, target, or finish direction.',
  advanced:
    "Tier is advanced. Default to silence or a one-line affirmation. Only give a cue if you see a concrete micro-refinement in the angleSummary.",
}

export interface LivePromptSwing {
  angleSummary: string
  startMs: number
  endMs: number
}

export interface LivePromptContext {
  profile: UserProfile | null
  skipped: boolean
  shotType: string
  swings: LivePromptSwing[]
  baselineSummary?: string | null
  baselineLabel?: string | null
  // Last few cues spoken to the player this session, oldest -> newest.
  // Used so the model can pivot, acknowledge persistence, or stay silent
  // instead of repeating itself.
  recentCues?: string[] | null
}

export interface LivePromptResult {
  // The fully assembled user-message text passed to the LLM.
  prompt: string
  // True when the prompt explicitly steered the model toward the
  // silence/affirmation "baseline" path. Currently set when the player's
  // tier is advanced — they get the same default-to-silence framing the
  // /analyze advanced template provides. This flag is plumbed straight
  // into the analysis_events.used_baseline_template telemetry column so
  // live and post-hoc cohorts can be compared apples-to-apples.
  usedBaselineTemplate: boolean
}

function formatMs(ms: number): string {
  return (Math.round(ms / 100) / 10).toFixed(1) + 's'
}

function buildPlayerLine(profile: UserProfile): string {
  const hand = profile.dominant_hand === 'right' ? 'right-handed' : 'left-handed'
  const backhand =
    profile.backhand_style === 'one_handed' ? 'one-handed backhand' : 'two-handed backhand'
  const goalNote =
    profile.primary_goal === 'other' && profile.primary_goal_note
      ? ` ("${profile.primary_goal_note}")`
      : ''
  return `Player is ${hand}, ${backhand}. Stated focus: ${profile.primary_goal}${goalNote}.`
}

function sanitizeRecentCues(cues: string[] | null | undefined): string[] {
  if (!Array.isArray(cues)) return []
  const out: string[] = []
  for (const raw of cues) {
    if (typeof raw !== 'string') continue
    const trimmed = raw.replace(/\s+/g, ' ').trim()
    if (trimmed.length === 0) continue
    // Keep them short — three cues * ~200 chars is plenty of memory.
    out.push(trimmed.slice(0, 200))
  }
  // We only show the model the last 3 cues so the prompt doesn't bloat as the
  // session goes on. Caller can pass more — we slice here as defense in depth.
  return out.slice(-3)
}

/**
 * Build the user-message body for the live coaching call.
 *
 * Returns both the prompt text and a `usedBaselineTemplate` flag so the
 * caller can record truthful telemetry. The flag is true when the prompt
 * was framed in the silence-by-default ("baseline") shape — currently the
 * advanced-tier path. Mirrors the /analyze route's used_baseline_template
 * field so the two cohorts are directly comparable.
 */
export function buildLiveCoachingPrompt(ctx: LivePromptContext): LivePromptResult {
  const {
    profile,
    skipped,
    shotType,
    swings,
    baselineSummary,
    baselineLabel,
    recentCues,
  } = ctx

  const tier: SkillTier | null = profile?.skill_tier ?? null
  const tierLine = tier
    ? LIVE_TIER_CUES[tier]
    : skipped
      ? 'Tier unknown because the player skipped onboarding. Infer tier from the data and coach to it.'
      : 'Tier unknown. Keep the cue broadly applicable.'

  const baselineBlock = baselineSummary
    ? `BASELINE (${baselineLabel ?? 'your best day'}):\n${baselineSummary}`
    : ''

  const swingLines = swings
    .map(
      (s, i) =>
        `SWING ${i + 1} (${formatMs(s.startMs)}–${formatMs(s.endMs)}):\n${s.angleSummary}`,
    )
    .join('\n\n')

  const countWord = swings.length === 1 ? shotType : `${shotType}s`

  const sanitizedCues = sanitizeRecentCues(recentCues)
  const recentCuesBlock =
    sanitizedCues.length > 0
      ? `RECENT CUES (already spoken to this player this session, oldest first):\n${sanitizedCues
          .map((c, i) => `${i + 1}. ${c}`)
          .join('\n')}`
      : ''

  const usedBaselineTemplate = tier === 'advanced'
  const closingDirective = usedBaselineTemplate
    ? 'If this batch is clean for an advanced player, respond with silence or one short affirmation ("Clean — repeat that.", "Trust that swing, do it again."). Only give a refinement cue if you can name one concrete observable detail from the angleSummary.'
    : 'Give ONE cue for the next ball, OR stay silent if the batch is already clean. One sentence. Max 25 words.'

  const sections = [
    `They just hit ${swings.length} ${countWord} in a row.`,
    tierLine,
    profile ? buildPlayerLine(profile) : null,
    baselineBlock || null,
    recentCuesBlock || null,
    `LAST ${swings.length} ${swings.length === 1 ? 'SWING' : 'SWINGS'}:\n${swingLines}`,
    closingDirective,
  ].filter((s): s is string => typeof s === 'string' && s.length > 0)

  return {
    prompt: sections.join('\n\n'),
    usedBaselineTemplate,
  }
}
