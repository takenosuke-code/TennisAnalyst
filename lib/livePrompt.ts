// Prompt builder for POST /api/live-coach. Kept separate from the /analyze
// prompts because live coaching has a fundamentally different output shape:
// ONE short sentence meant to be spoken through earbuds between points, not a
// markdown-structured review. The tier-rules infrastructure from lib/profile
// still applies — we just layer TTS-friendly brevity rules on top.

import type { SkillTier, UserProfile } from './profile'

export const LIVE_SYSTEM_PROMPT = `You are a tennis coach yelling one short cue across the net between points. The player hears this through earbuds while hitting. They cannot stop to read anything.

The job is to look across a batch of recent swings, find the most COMMON pattern (or the most repeatedly-noticeable one), and shout ONE short cue about it. You are NOT walking through individual swings, you are NOT analyzing one swing in detail. You are calling out the consistent thing.

OUTPUT RULES (non-negotiable):
- AT MOST 6 WORDS. Imperative. No commas. No connectors.
- Examples of the right shape:
    "Bend your legs more."
    "Earlier prep on backswing."
    "Eye on the contact point."
    "Drive up through the legs."
    "Finish over the shoulder."
    "Stay down through the ball."
    "Same swing, again."
    "Lock that in."
- Wrong shape (too long, too analytical, multi-clause):
    "Hip rotation looked closed (28°). Open up earlier."
    "Your follow-through cut short, try extending out toward the target."
    "First two were tighter than the last two."
- No numbers. No digits anywhere. No degree symbols. No parenthetical measurements. Plain words only.
- No em dashes, en dashes, or hyphens-as-connectors. Compound-word hyphens inside one word ("follow-through") are fine.
- No greeting, no sign-off, no meta-commentary.

SILENCE IS A FIRST-CLASS RESPONSE:
- If the batch is clean, return one of: "Same swing, again." / "Lock that in." / "Trust it." / OR an empty response.
- Don't invent a refinement just to fill the slot.

WHAT TO LOOK FOR (in priority order):
1. A single mechanical thing repeating across MULTIPLE swings of the batch (legs not bending across all four, hips not rotating across all four, follow-through cut short across all four).
2. A trend that's worsening across the batch (first swing okay, then progressively tighter).
3. Silence if neither pattern is clear.

WHAT YOU CAN COMMENT ON (only what's in the data):
- Joint angles in the angleSummary (elbow, shoulder, knee, hip, trunk).
- Hip rotation and trunk rotation magnitude when they're in the summary.
- Peak-frame timing relative to the keyframes shown.

WHAT YOU CANNOT COMMENT ON (zero tolerance):
- Grip type, wrist pronation, footwork direction, stance, target, depth, opponent, anything tactical, anything inferred from video.

PRIOR CUES (when provided):
- If RECENT CUES are listed below, those are what the player just heard.
- Do not repeat a recent cue verbatim.
- If the same problem persists across cues, repeat the SAME cue — that's a coach hammering on the most important thing. But change the words slightly each time so it doesn't sound canned.`

// Short live-mode guidance per tier. Kept separate from TIER_RULES in
// lib/profile.ts (which shape multi-section markdown output) so the live
// system prompt stays blunt and a single paragraph.
const LIVE_TIER_CUES: Record<SkillTier, string> = {
  beginner:
    'Tier is beginner. One external-focus cue, six words max.',
  intermediate:
    'Tier is intermediate. One feel-based cue, six words max.',
  competitive:
    'Tier is competitive. One execution cue, six words max.',
  advanced:
    'Tier is advanced. Default to silence or one short affirmation. Cue only if a concrete pattern repeats across the batch.',
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
    ? 'If clean, respond with silence or "Lock that in." / "Trust it." Only cue if one specific thing repeats across the batch.'
    : 'Look across the whole batch. Find the most common repeating issue. Give ONE cue, AT MOST SIX WORDS. Or stay silent.'

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
