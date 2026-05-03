// Telemetry uses Railway's /classify-angle endpoint. Requires:
//   RAILWAY_SERVICE_URL=https://...railway.app
//   EXTRACT_API_KEY=<same key used for /extract>
// Missing either -> capture_quality_flag stays null on every row.
// This is intentional -- telemetry must never block coaching.
//
// Pipeline (rewritten 2026-04 to be observation-driven):
//   1. extractObservations() turns pose data into typed Observations.
//   2. If empty -> stream the friendly "couldn't read your swing" empty-state
//      response and skip Anthropic entirely. X-Analyze-Empty-State signals it.
//   3. Otherwise pickPrimary + pickSecondary, build a tight system prompt
//      (10 lines of voice rules) and a user prompt with chosen exemplars +
//      observations + format spec.
//   4. Buffer the LLM body, validate against the post-filter (no digits, no
//      em-dash, no biomech jargon), re-roll twice with stricter framing on
//      reject; on the third failure emit a CueExemplar-driven static fallback.
//   5. Append a deterministic "## Show your work" markdown block constructed
//      from the Observation rows. Numbers are ALLOWED only in this section.

import { NextRequest, NextResponse } from 'next/server'
import { after } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { supabase, supabaseAdmin } from '@/lib/supabase'
import { createClient } from '@/lib/supabase/server'
import { buildAngleSummary } from '@/lib/jointAngles'
import { classifyAndTagCaptureQuality } from '@/lib/captureQuality'
import {
  DEFAULT_MAX_TOKENS,
  extractResponseMetrics,
  getCoachingContext,
  registerForTier,
  TIER_MAX_TOKENS,
  type Register,
  type SkillTier,
} from '@/lib/profile'
import { sanitizePromptInput } from '@/lib/sanitize'
import {
  extractObservations,
  pickPrimary,
  pickSecondary,
  type Observation,
} from '@/lib/coachingObservations'
import { smoothFramesForRules } from '@/lib/poseSmoothing'
import { gateClipQuality } from '@/lib/poseGate'
import {
  CUE_EXEMPLARS,
  findExemplars,
  type CueExemplar,
} from '@/lib/cueExemplars'
import type { ShotType } from '@/lib/shotTypeConfig'
import { VALID_SHOT_TYPES } from '@/lib/shotTypeConfig'
import type { KeypointsJson, PoseFrame } from '@/lib/supabase'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

// ---------------------------------------------------------------------------
// Voice rules system prompt — kept tight so the model has the contract in
// front of it without being smothered by 30 lines of don't-do-this. Echoes
// the cueExemplars register: imperative, second person, single primary cue,
// then "Other things I noticed" listing secondary observations.
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are a veteran tennis coach who talks like an analyst when explaining what you see and like a coach when telling the player what to do.
Voice rules, no exceptions:
1. NEVER include numbers, digits, percentages, degrees, or counts. No "twelve", no "30%", no "twenty-three degrees" written out either. Describe magnitudes in plain words: "shallow", "deep", "professional-level", "significant room for improvement", "barely turning".
2. NEVER use em-dashes or en-dashes. Use full sentences or commas. Hyphens inside compound words like "follow-through" are fine.
3. Plain biomech terms are FINE: "unit turn", "kinetic chain", "follow-through", "load", "coil". Use them when they make the read tighter.
4. NEVER use raw field names like "trunk_rotation" or "hip_rotation".
5. NEVER use the phrase "joint angle" — no one knows what that means.
6. External focus where natural: attention on the racket, the ball, or a court target. Internal cues are okay when the gap is about the body itself.
7. STRENGTH + GAP framing applies in "## Primary cue" and "## Other things I noticed". Lead with what is working in the area you're about to critique, then name the gap and the consequence on the shot. Example shape: "Your wrist position at contact is excellent, however your horizontal extension shows room for improvement, which jams the racket against the body."
8. Quick Read does NOT need strength + gap framing. Keep each bullet a snappy on-court read, under twelve words. Plain biomech terms like "unit turn shallow" are welcome — pair the term with a brief plain explanation in the same bullet when it helps ("Unit turn shallow, shoulders aren't turning enough").
9. Output exactly four markdown sections in this order: "## Quick Read", "## Primary cue", "## Other things I noticed", "## Recommended drills".
10. "## Primary cue" is one paragraph (three to five sentences) framed as strength + gap. Acknowledge what's working in the area you're naming, then describe the gap and its consequence.
11. "## Other things I noticed" is a short bulleted list of secondary observations. Each bullet is one or two sentences in strength + gap form, leading with the strength.
12. "## Recommended drills" is two or three concrete on-court drills tied to the cues above. One drill per bullet, each a single sentence the player can act on.
13. Do not generate any other section. The server appends "## Show your work" after you.`

// Stricter retry preamble — appended to the user prompt on a failed-output retry.
const STRICT_RETRY_NOTE = `\n\nSTRICT RETRY: Your previous output included disallowed content (digits, em-dashes, raw field names, or the phrase "joint angle"). Rewrite from scratch with NO numbers anywhere, NO dashes other than hyphens inside compound words, NO field names like trunk_rotation, and NO use of "joint angle". Plain biomech terms like "unit turn" and "kinetic chain" are still welcome. Use the exemplar voice exactly.`

// Low-confidence path used to also prepend a markdown banner here, but
// the UI now renders a styled banner above the sections from the
// X-Analyze-Low-Confidence header — so embedding one in the body would
// double up. The header alone is the signal now.

// Patterns we reject in LLM output. Show your work is appended AFTER the
// filter runs, so the digit pattern there does not trip it.
const REJECT_DIGIT_RE = /\d/

// Sentinel emitted to the client when a streamed attempt fails
// validation mid-flight. The client buffers incoming text and, on
// seeing this marker, discards everything before it (i.e. clears the
// rendered feedback so far) and continues building from the next
// chunk. NUL byte prefix keeps it unlikely to collide with real
// model output, which is plain markdown and never contains \x00.
const STREAM_CLEAR_MARKER = '\x00CLEAR\x00'
// 2026-05 — voice loosened to advisor register. "kinetic chain" is now
// allowed (the example output the user liked uses the term explicitly).
// "joint angle" stays banned per user feedback ("no one understands it").
// Underscored field names stay banned (they're SQL-shaped and never
// user-facing). Em/en-dashes stay banned for typographic consistency.
const REJECT_JARGON_RE = /(trunk_rotation|hip_rotation|joint angle|—|–)/i

const MAX_LLM_RETRIES = 2 // primary + 2 retries = 3 total

// ---------------------------------------------------------------------------
// Helpers — pure formatting, kept above POST so the handler stays linear.
// ---------------------------------------------------------------------------

function patternHumanLabel(pattern: Observation['pattern']): string {
  switch (pattern) {
    case 'cramped_elbow':
      return 'cramped elbow at contact'
    case 'over_extended_elbow':
      return 'arm locked straight at contact'
    case 'shallow_knee_load':
      return 'shallow knee load'
    case 'locked_knees':
      return 'legs staying tall through the load'
    case 'insufficient_hip_excursion':
      return 'hips barely turning through the swing'
    case 'insufficient_trunk_excursion':
      return 'shoulders barely turning through the swing'
    case 'insufficient_unit_turn':
      return 'no full unit turn into preparation'
    case 'truncated_followthrough':
      return 'follow-through cut short'
    case 'weak_leg_drive':
      return 'legs not driving up through the shot'
    case 'short_pushout':
      return 'racket not extending out toward the target after contact'
    case 'unstable_base':
      return 'head and base shifting through contact'
    case 'drift_from_baseline':
      return 'drift from your best-day baseline'
    case 'contact_height_higher':
      return 'ball met higher above the body than baseline'
    case 'contact_height_lower':
      return 'ball met lower below the usual contact height'
    case 'contact_position_jammed':
      return 'contact crowded against the body, less extension than baseline'
    case 'contact_position_extended':
      return 'contact further out from the body than baseline'
  }
}

function jointHumanLabel(joint: string): string {
  return joint.replace(/_/g, ' ')
}

function renderObservationLine(o: Observation): string {
  return `${jointHumanLabel(o.joint)} at ${o.phase}: ${patternHumanLabel(o.pattern)} (severity: ${o.severity})`
}

// Contact-context observations carry NORMALIZED FRACTIONS in their
// todayValue / baselineValue fields, not degrees. Height observations
// store wrist-Y above shoulder midpoint as a fraction of torso length;
// position observations store horizontal wrist distance from body
// center as a fraction of shoulder width. Rendering them as "0°" (the
// fraction rounds to 0) was misleading users into thinking the
// measurement had failed. Render as percentages with descriptive units.
const CONTACT_CONTEXT_PATTERNS = new Set<Observation['pattern']>([
  'contact_height_higher',
  'contact_height_lower',
  'contact_position_jammed',
  'contact_position_extended',
])

function isContactContextPattern(p: Observation['pattern']): boolean {
  return CONTACT_CONTEXT_PATTERNS.has(p)
}

function contactContextUnit(p: Observation['pattern']): string {
  if (p === 'contact_height_higher' || p === 'contact_height_lower') {
    return '% of torso above shoulders'
  }
  // Position patterns
  return '% of shoulder-width from body center'
}

/**
 * Render the deterministic Show Your Work block from the chosen observations.
 * Numbers come from the Observation rows. ° is included so the player can
 * tie the human-language coaching back to the underlying angles. This block
 * is concatenated AFTER the LLM body, so post-filter never inspects it.
 *
 * Contact-context patterns get a percentage rendering instead of degrees
 * because their underlying values are 0..1 fractions (torso-normalized
 * height, shoulder-width-normalized horizontal position) — rendering them
 * with the degree treatment produces "0°" / "1°" which looks like a
 * failed measurement.
 */
function renderShowYourWork(
  primary: Observation | null,
  secondary: Observation[],
): string {
  const all = primary ? [primary, ...secondary] : secondary
  if (all.length === 0) return ''
  const lines: string[] = ['', '## Show your work']
  for (const o of all) {
    const phase = o.phase
    const joint = jointHumanLabel(o.joint)
    if (isContactContextPattern(o.pattern)) {
      const unit = contactContextUnit(o.pattern)
      const todayPct = Math.round(o.todayValue * 100)
      if (typeof o.baselineValue === 'number') {
        const basePct = Math.round(o.baselineValue * 100)
        const driftPct =
          typeof o.driftMagnitude === 'number'
            ? Math.round(o.driftMagnitude * 100)
            : Math.abs(todayPct - basePct)
        lines.push(
          `- **${joint} at ${phase}**: baseline ${basePct}${unit}, today ${todayPct}${unit} (drifted ${driftPct} points)`,
        )
      } else {
        lines.push(
          `- **${joint} at ${phase}**: ${todayPct}${unit} today (${o.severity})`,
        )
      }
      continue
    }
    if (
      o.pattern === 'drift_from_baseline' &&
      typeof o.baselineValue === 'number' &&
      typeof o.driftMagnitude === 'number'
    ) {
      const today = Math.round(o.todayValue)
      const base = Math.round(o.baselineValue)
      const drift = Math.round(o.driftMagnitude)
      lines.push(
        `- **${joint} at ${phase}**: ${base}° on baseline, ${today}° today (drifted ${drift}°)`,
      )
    } else {
      const today = Math.round(o.todayValue)
      lines.push(`- **${joint} at ${phase}**: ${today}° today (${o.severity})`)
    }
  }
  return lines.join('\n')
}

/**
 * Format the user prompt. Includes 3-5 exemplars in the player's register,
 * the primary observation rendered as a single line, the list of secondary
 * observations, and the format spec.
 */
function buildUserPrompt(args: {
  primary: Observation
  secondary: Observation[]
  exemplars: CueExemplar[]
  register: Register
  shotType: ShotType | null
  tier: SkillTier | null
  focus: string | null
  handedness: 'right' | 'left' | null
}): string {
  const { primary, secondary, exemplars, register, shotType, tier, focus, handedness } = args

  const exemplarLines = exemplars
    .map((ex) => `- ${register === 'technical' ? ex.technical : ex.plain}`)
    .join('\n')

  const secondaryLines = secondary.length
    ? secondary.map((o) => `- ${renderObservationLine(o)}`).join('\n')
    : '- (none)'

  const tierHint = tier
    ? `Player skill: ${tier}. Calibrate vocabulary to that level.`
    : `Player skill: unknown. Use plain coaching language.`
  const shotHint = shotType ? `Shot: ${shotType}.` : `Shot: not specified.`
  const handednessHint = handedness
    ? `Player handedness: ${handedness}-handed. Cue accordingly (a left-hander's forehand uses their left arm).`
    : `Player handedness: not specified.`
  const focusHint = focus
    ? `Player asked: "${focus}". Address it in your primary cue when relevant.`
    : ''

  return `${tierHint}
${shotHint}
${handednessHint}
${focusHint}

EXEMPLARS (imitate this voice register, do not copy verbatim):
${exemplarLines}

PRIMARY ISSUE: ${renderObservationLine(primary)}

OTHER OBSERVATIONS:
${secondaryLines}

FORMAT — respond with EXACTLY four sections in this order, no other content:

## Quick Read
Two or three snappy bullets summarizing the swing at a glance. Each bullet under twelve words. No numbers, no degrees, no em-dashes, no "joint angle". Plain biomech terms like "unit turn shallow" are welcome — pair the term with a brief plain explanation in the same bullet ("Unit turn shallow, shoulders aren't turning enough"). No strength + gap framing required here; this is the on-court glance read.

## Primary cue
One paragraph (three to five sentences) framed as strength + gap. Acknowledge what's working in the named mechanical area first, then describe the gap and the consequence on the shot. No numbers, no degrees, no "joint angle". Plain biomech terms welcome ("unit turn", "kinetic chain", "follow-through").

## Other things I noticed
One bullet for EVERY secondary observation listed above — do not drop any. Each bullet is one or two sentences in strength + gap form, leading with the strength. No numbers, no degrees, no "joint angle". If there are no secondary observations, write a single bullet acknowledging the rest of the swing is hanging together.

## Recommended drills
Two or three concrete on-court drills tied to the cues above. One drill per bullet, each a single sentence the player can act on. No numbers, no degrees, no em-dashes.`
}

/**
 * Baseline-compare prompt. Used when compareMode === 'baseline' AND the
 * pipeline produced at least one observation (typically a
 * `drift_from_baseline` row, but could also be a today-only absolute
 * fault that the baseline does NOT exhibit).
 *
 * Voice: consistency-focused. The player picked a "best-day" baseline
 * and wants to know how today differs. We frame the coaching around
 * the diff, not absolute coaching. If the only signal is that the swing
 * changed in a specific dimension (e.g. drift in elbow angle at
 * contact), say so plainly. We do NOT invent generic faults.
 */
function buildBaselineUserPrompt(args: {
  primary: Observation
  secondary: Observation[]
  exemplars: CueExemplar[]
  register: Register
  shotType: ShotType | null
  tier: SkillTier | null
  focus: string | null
  handedness: 'right' | 'left' | null
  baselineLabel: string | null
}): string {
  const { primary, secondary, exemplars, register, shotType, tier, focus, handedness, baselineLabel } = args

  const exemplarLines = exemplars
    .map((ex) => `- ${register === 'technical' ? ex.technical : ex.plain}`)
    .join('\n')

  const secondaryLines = secondary.length
    ? secondary.map((o) => `- ${renderObservationLine(o)}`).join('\n')
    : '- (none)'

  const tierHint = tier
    ? `Player skill: ${tier}. Calibrate vocabulary to that level.`
    : `Player skill: unknown. Use plain coaching language.`
  const shotHint = shotType ? `Shot: ${shotType}.` : `Shot: not specified.`
  const handednessHint = handedness
    ? `Player handedness: ${handedness}-handed.`
    : `Player handedness: not specified.`
  const focusHint = focus
    ? `Player asked: "${focus}". Address it in your primary cue when relevant.`
    : ''
  const baselineHint = baselineLabel
    ? `The player picked their baseline labelled "${baselineLabel}" as the reference for "their best day."`
    : `The player picked a saved baseline as their reference for "their best day."`

  return `${tierHint}
${shotHint}
${handednessHint}
${focusHint}

${baselineHint}

CONTEXT — this is a CONSISTENCY check, not coaching. The player wants to know how today's swing differs from their saved baseline. The list below contains ONLY differences between today and baseline (or new faults today that the baseline did not show). Do NOT invent additional generic coaching cues. Do NOT critique anything that wasn't flagged. If the list is light (one mild drift), the swing is mostly matching the baseline — say so.

EXPLANATION CHAINS — some observations explain others. Among the observations you receive, "ball met higher / lower than baseline" and "contact crowded / further out than baseline" are CONTEXT signals, not faults the player controls. They describe where the ball was when the player met it. When a body-mechanic difference (for example shallow knee load, cramped elbow, or shorter follow-through) shows up alongside a contact-context difference, ASK YOURSELF: does the contact context EXPLAIN the body difference?
- A higher contact point can explain lighter knee load: the player did not need to drive up as much.
- A jammed contact can explain a cramped elbow: the player was rushed close to the body.
- A higher / lower contact can explain a different finish line on the follow-through.
When a contact-context observation explains a body-mechanic one, frame the body difference as "expected given how the ball came in" rather than as a fault to fix. Lead the analysis with "your shot is on-pattern with your baseline" and then explain what was different about the contact context. Do NOT coach the contact context itself — the player rarely controls bounce height or ball depth swing-by-swing.

NEVER hedge on ball context that you don't have. If NO contact-context observation appears in the OBSERVATIONS list, the system has determined the contact location was not meaningfully different from baseline. Do NOT say "without knowing where the ball came in" or "if the ball came in differently" or "watch your video and see if the contact point changed" or any equivalent. Do NOT ask the player to verify their contact context. Do NOT speculate about what would have happened if the ball had been different. When contact-context is absent from the observations, treat the body-mechanic differences as REAL drift and coach them confidently with the strength + gap framing. Hedging on missing ball data reads as evasive — coach what the rules saw.

EXEMPLARS (imitate this voice register, do not copy verbatim):
${exemplarLines}

PRIMARY DIFFERENCE: ${renderObservationLine(primary)}

OTHER DIFFERENCES:
${secondaryLines}

FORMAT — respond with EXACTLY four sections in this order, no other content:

## Quick Read
Two or three snappy bullets. Lead with how today compares to the baseline overall (close match, drifted in this one area, etc.). Each bullet under twelve words. No numbers, no degrees, no "joint angle". Plain biomech terms welcome.

## Primary cue
One paragraph (three to five sentences) framed as strength + gap, applied to the diff. Acknowledge what's holding up vs the baseline first, then name the area that drifted and the consequence on the shot. No numbers, no degrees, no "joint angle". If the difference is small or the player asked about something specific that the diff doesn't cover, acknowledge that the swing is largely on-pattern.

## Other things I noticed
One bullet for EVERY other difference listed above — do not drop any. Each bullet is one or two sentences in strength + gap form, leading with what's holding up vs the baseline before naming the drift. If there are no other differences, write a single bullet acknowledging the rest of the swing is matching the baseline.

## Recommended drills
Two or three concrete on-court drills tied to the differences above. One drill per bullet, each a single sentence the player can act on. No numbers, no degrees, no em-dashes.`
}

/**
 * Baseline-compare fallback prompt. Used when compareMode === 'baseline'
 * AND the pipeline produced ZERO observations — meaning today and
 * baseline match across every rule. The user explicitly asked for this
 * case to acknowledge similarity instead of inventing coaching: "the
 * coach should know when to acknowledge that theyre very similar". We
 * tell the LLM exactly that.
 */
function buildBaselineMatchPrompt(args: {
  register: Register
  shotType: ShotType | null
  tier: SkillTier | null
  focus: string | null
  handedness: 'right' | 'left' | null
  baselineLabel: string | null
}): string {
  const { register, shotType, tier, focus, handedness, baselineLabel } = args
  const tierHint = tier
    ? `Player skill: ${tier}. Calibrate vocabulary to that level.`
    : `Player skill: unknown. Use plain coaching language.`
  const shotHint = shotType ? `Shot: ${shotType}.` : `Shot: not specified.`
  const handednessHint = handedness
    ? `Player handedness: ${handedness}-handed.`
    : `Player handedness: not specified.`
  const registerHint =
    register === 'technical'
      ? 'Use the technical register (skilled-player vocabulary).'
      : 'Use the plain register (beginner-friendly vocabulary).'
  const focusHint = focus
    ? `Player asked: "${focus}". Address it in your primary cue when relevant.`
    : ''
  const baselineHint = baselineLabel
    ? `The player is comparing to their baseline labelled "${baselineLabel}".`
    : `The player is comparing to their saved baseline.`

  return `${tierHint}
${shotHint}
${handednessHint}
${registerHint}
${focusHint}

${baselineHint}

CONTEXT — today's swing matches the baseline across every check the system runs. The hips, trunk, elbow at contact, knee load, follow-through, and stability all came in within tolerance of the baseline. This is a CONSISTENCY check, not coaching. Tell the player that today's swing is on-pattern. Do NOT invent specific faults. Do NOT pivot into generic coaching tips. If the player asked a focus question and that area also matched, say it matched.

If you genuinely have to give one piece of feedback because the player asked, you may add a single light fine-tuning thought, but lead with "matches your baseline."

FORMAT — respond with EXACTLY four sections in this order, no other content:

## Quick Read
Two or three snappy bullets. The first bullet says today matches the baseline. Other bullets can call out a specific area that held up well (unit turn, contact, finish). Each under twelve words. No numbers, no degrees, no "joint angle". Plain biomech terms welcome.

## Primary cue
One paragraph (three to five sentences) confirming today's swing is matching the baseline. Lead with the strongest area that held up. No numbers, no degrees, no "joint angle". Plain biomech terms welcome ("unit turn", "kinetic chain", "follow-through"). If the player asked a focus question, address it in plain language.

## Other things I noticed
A short bulleted list (one to three bullets) of areas that specifically held up. Each bullet one or two sentences. Frame as "your X is matching" or "the Y read the same as on your best day". No numbers, no degrees, no "joint angle".

## Recommended drills
Two or three drills focused on REPEATING this swing — grooving rhythm and contact point — not on fixing a fault. One drill per bullet, each a single sentence. No numbers, no degrees, no em-dashes.`
}

/**
 * Generic-coaching prompt used when no observation cleared the confidence
 * floor. The LLM gets the textual angle summary instead of structured
 * observations, plus a directive to coach generally without inventing
 * specific faults. Output format matches the observation path so the UI
 * parser doesn't need a special case.
 */
function buildFallbackUserPrompt(args: {
  userSummary: string
  register: Register
  shotType: ShotType | null
  tier: SkillTier | null
  focus: string | null
  handedness: 'right' | 'left' | null
}): string {
  const { userSummary, register, shotType, tier, focus, handedness } = args

  const tierHint = tier
    ? `Player skill: ${tier}. Calibrate vocabulary to that level.`
    : `Player skill: unknown. Use plain coaching language.`
  const shotHint = shotType ? `Shot: ${shotType}.` : `Shot: not specified.`
  const handednessHint = handedness
    ? `Player handedness: ${handedness}-handed.`
    : `Player handedness: not specified.`
  const registerHint =
    register === 'technical'
      ? 'Use the technical register (skilled-player vocabulary, e.g. "load the trail leg").'
      : 'Use the plain register (beginner-friendly vocabulary, e.g. "sit deeper into your back leg").'
  const focusHint = focus
    ? `Player asked: "${focus}". Address it in your primary cue when relevant.`
    : ''

  return `${tierHint}
${shotHint}
${handednessHint}
${registerHint}
${focusHint}

NO STRONG OBSERVATIONS. The auto-detection rules ran across this swing but no specific deviation cleared the confidence floor. Either the swing is mechanically clean, or the pose tracking was uneven.

Your job: give general feel-based coaching from the angle data below. Do NOT invent specific faults. If the swing reads clean, say so and offer a single fine-tuning thought. If you genuinely can't tell, give one general feel cue any player benefits from.

ANGLE SUMMARY (5 phase snapshots from the swing):
${userSummary}

FORMAT — respond with EXACTLY four sections in this order, no other content:

## Quick Read
Two or three snappy bullets summarizing the swing at a glance. Each bullet under twelve words. No numbers, no degrees, no "joint angle". Plain biomech terms welcome.

## Primary cue
One paragraph (three to five sentences). Lead with what's working in the swing before adding any fine-tuning thought. No numbers, no degrees, no "joint angle". Plain biomech terms welcome ("unit turn", "kinetic chain", "follow-through"). If the swing reads clean, lead with that.

## Other things I noticed
A short bulleted list (one to three bullets) of general feel observations. Each bullet one or two sentences in strength + gap form, leading with what's holding up. If you genuinely have nothing to add, write a single bullet acknowledging the swing has otherwise hung together.

## Recommended drills
Two or three concrete on-court drills any player at this level benefits from. One drill per bullet, each a single sentence the player can act on. No numbers, no degrees, no em-dashes.`
}

/**
 * Removed in 2026-05: streamLlmBody (buffer-and-validate helper) and
 * the senior-coach validator (reviewWithSeniorCoach +
 * SENIOR_COACH_SYSTEM_PROMPT). The route now streams tokens directly
 * to the client per attempt, so a buffered helper is no longer
 * needed; and the senior-coach validator ran AFTER the full junior
 * response, which would cause a jarring "replace what the user just
 * watched stream in" flicker. The strict prompt rules + post-filter
 * remain as the safety net. If we want a senior reviewer back, the
 * cleanest path is a separate route that runs against the saved
 * markdown asynchronously.
 */

/**
 * Static fallback used when all LLM retries fail validation. Constructs a
 * coach-voiced response from a chosen CueExemplar and lists secondary
 * observations as one-liners. No LLM involvement.
 */
function buildStaticFallback(args: {
  primary: Observation | null
  secondary: Observation[]
  exemplar: CueExemplar | null
  register: Register
}): string {
  const { primary, secondary, exemplar, register } = args

  // No primary observation = low-confidence path. Cue is generic, framed
  // around the swing reading clean rather than around a specific fault we
  // couldn't trust enough to call out.
  let cueText: string
  if (exemplar) {
    cueText = register === 'technical' ? exemplar.technical : exemplar.plain
  } else if (primary) {
    cueText = `Focus on ${patternHumanLabel(primary.pattern)} on the next ball.`
  } else {
    cueText =
      'The swing is reading clean. Stay with the same rhythm and trust the contact point you are finding right now.'
  }

  const lines: string[] = []
  lines.push('## Quick Read')
  if (primary) {
    lines.push(`- Main thing to fix: ${patternHumanLabel(primary.pattern)}.`)
  } else {
    lines.push('- The swing reads clean overall.')
  }
  if (secondary.length > 0) {
    lines.push(`- Also keep an eye on ${patternHumanLabel(secondary[0].pattern)}.`)
  } else {
    lines.push('- Rest of the chain is hanging together.')
  }
  lines.push('- Stay with your rhythm and trust the contact point.')
  lines.push('')
  lines.push('## Primary cue')
  lines.push(cueText)
  lines.push('')
  lines.push('## Other things I noticed')
  if (secondary.length === 0) {
    lines.push('- The rest of the swing is hanging together.')
  } else {
    for (const o of secondary) {
      lines.push(`- Watch for ${patternHumanLabel(o.pattern)} on your ${jointHumanLabel(o.joint)}.`)
    }
  }
  lines.push('')
  lines.push('## Recommended drills')
  // Spelled-out counts ("ten reps") were stripped — voice rule says no
  // counts at all, even in word form.
  lines.push('- Shadow swing a few reps focusing on the primary cue before your next rally.')
  lines.push('- Feed yourself slow drop-hits and groove the contact point.')
  lines.push('- Mini-tennis from the service line, prioritizing rhythm over power.')
  return lines.join('\n')
}

function normalizeShotType(raw: unknown): ShotType | null {
  if (typeof raw !== 'string') return null
  const lower = raw.toLowerCase().trim()
  return (VALID_SHOT_TYPES as readonly string[]).includes(lower)
    ? (lower as ShotType)
    : null
}

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------

export async function POST(request: NextRequest) {
  const authClient = await createClient()
  const { profile, skipped } = await getCoachingContext(authClient)
  const { data: userData } = await authClient.auth.getUser().catch(() => ({ data: { user: null } }))
  const userId = userData?.user?.id ?? null

  let body
  try { body = await request.json() }
  catch { return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 }) }
  const {
    sessionId,
    keypointsJson: inlineKeypoints,
    compareKeypointsJson,
    userFocus,
    compareMode,
    baselineLabel,
    shotType: bodyShotType,
    blobUrl: bodyBlobUrl,
    mode: bodyMode,
  } = body
  const isShotMode = bodyMode === 'shot'
  // Baseline-compare path: the player saved a "best day" baseline and is
  // comparing today's swing against it. The point isn't generic coaching —
  // it's consistency. Two changes ride on this flag:
  //   1. extractObservations() suppresses today-side absolute observations
  //      that ALSO fire on the baseline (they're not differences, they're
  //      how the player swings every day).
  //   2. We swap the prompt so the coach focuses on what changed between
  //      baseline and today, and acknowledges similarity when nothing did.
  const isBaselineCompare =
    compareMode === 'baseline' &&
    compareKeypointsJson &&
    typeof compareKeypointsJson === 'object'

  const focus = sanitizePromptInput(userFocus, 240)

  let userKeypoints: KeypointsJson | null = inlineKeypoints ?? null
  if (!userKeypoints && sessionId) {
    const { data: session, error: sessionError } = await supabase
      .from('user_sessions')
      .select('keypoints_json')
      .eq('id', sessionId)
      .single()
    if (sessionError) {
      return NextResponse.json({ error: 'Session not found' }, { status: 404 })
    }
    userKeypoints = session?.keypoints_json ?? null
  }

  if (!userKeypoints || !userKeypoints.frames?.length) {
    return NextResponse.json(
      { error: 'No keypoints data available' },
      { status: 400 }
    )
  }

  const framesValid = userKeypoints.frames.every(
    (f) => f !== null && typeof f === 'object' && typeof f.joint_angles === 'object'
  )
  if (!framesValid) {
    return NextResponse.json({ error: 'Invalid keypoints format' }, { status: 400 })
  }

  // Resolve shot type (request body > session metadata).
  let resolvedShotType: ShotType | null = normalizeShotType(bodyShotType)
  let resolvedBlobUrl: string | null = typeof bodyBlobUrl === 'string' ? bodyBlobUrl : null
  if ((!resolvedShotType || !resolvedBlobUrl) && sessionId) {
    try {
      const { data: sessionMeta } = await supabase
        .from('user_sessions')
        .select('shot_type, blob_url')
        .eq('id', sessionId)
        .single()
      if (sessionMeta) {
        resolvedShotType = resolvedShotType ?? normalizeShotType(sessionMeta.shot_type)
        resolvedBlobUrl = resolvedBlobUrl ?? sessionMeta.blob_url ?? null
      }
    } catch (err) {
      console.error('analyze: shot_type/blob_url lookup failed:', err)
    }
  }

  // Optional baseline frames for compare-mode. Same validation as user frames.
  let baselineFrames: PoseFrame[] | undefined
  if (compareKeypointsJson && typeof compareKeypointsJson === 'object') {
    const cmp = compareKeypointsJson as KeypointsJson
    if (Array.isArray(cmp.frames) && cmp.frames.length > 0) {
      const cmpValid = cmp.frames.every(
        (f) => f !== null && typeof f === 'object' && typeof f.joint_angles === 'object',
      )
      if (!cmpValid) {
        return NextResponse.json(
          { error: 'Invalid compareKeypointsJson format' },
          { status: 400 },
        )
      }
      baselineFrames = cmp.frames
    }
  }

  // Smooth joint angles before any rule fires. Empirically validated
  // on IMG_1245: kills jitter-driven cramped_elbow false positives
  // (peak r_elbow 12° → 158° post-smoothing). Landmarks stay raw so
  // detectSwings's wrist-velocity peak-pick still finds the right
  // window — smoothing landmark coords blunts peak velocities and
  // shifts detection (regression on Alcaraz fixture). beta=0.05 (vs
  // the render-path default 0.007) preserves real angular range
  // through ~200°/s hip rotation at peak.
  const ANALYSIS_SMOOTH_OPTS = {
    minCutoff: 1.0,
    beta: 0.05,
  } as const
  const smoothedFrames = smoothFramesForRules(userKeypoints.frames, ANALYSIS_SMOOTH_OPTS)
  const smoothedBaselineFrames = baselineFrames
    ? smoothFramesForRules(baselineFrames, ANALYSIS_SMOOTH_OPTS)
    : undefined

  // Clip-level confidence gate. Runs on RAW frames so the smoother
  // can't mask structural keypoint failures: angle smoothing tames the
  // frame-to-frame hip rotation jumps that the gate's jumpFraction
  // check uses as a proxy for left/right hip flips. When the gate
  // fails we skip the LLM call entirely and surface the reason via
  // X-Analyze-Gate-Reason for the UI to render a reshoot banner.
  const gateResult = gateClipQuality(userKeypoints.frames)

  // Compact angle summary kept for telemetry — `composite_metrics.user_summary`
  // backs the existing capture-quality / drift dashboards. Not part of the
  // new prompt path.
  const userSummary = buildAngleSummary(smoothedFrames)

  // Run the observation extractor. This is the new core: rule-based detection
  // emits typed Observation rows, and we choose primary + secondary from there.
  // Skip when the gate failed — running rules on garbage input produces
  // garbage output, and the route falls through to the low-confidence
  // path which renders the banner instead.
  const observations = gateResult.passed
    ? extractObservations({
        todaySummary: smoothedFrames,
        baselineSummary: smoothedBaselineFrames,
        shotType: resolvedShotType,
        dominantHand: profile?.dominant_hand ?? null,
        // Baseline-compare mode: today's frames are pre-sliced to a
        // single swing on the page (compare/page.tsx detectSwings) and
        // the baseline frames come from a saved swing trim. Tell the
        // rules to skip server-side re-detection — otherwise the
        // second pass finds a peak inside the already-sliced data and
        // re-slices around it, often cutting prep frames entirely and
        // producing a flat hip / trunk excursion (the "1° today across
        // every joint" bug).
        isPreSliced: isBaselineCompare,
      })
    : []

  const primary = pickPrimary(observations)
  // Cap raised 3 → 6: prior cap dropped real observations off the
  // bottom on swings with multiple genuine faults, e.g. a clip that
  // also fired weak_leg_drive showed up as "I didn't focus on legs"
  // when the player asked. The voice rules still produce one Primary
  // cue + a tight bulleted list, so 6 secondaries doesn't bloat the
  // response — it just stops the rule layer from silently hiding real
  // signal.
  const secondary = pickSecondary(observations, primary, 6)

  // Telemetry insert. Wrapped — must NEVER fail the coaching stream.
  let eventId: string | null = null
  try {
    const { data: inserted, error: insertError } = await supabaseAdmin
      .from('analysis_events')
      .insert({
        user_id: userId,
        session_id: sessionId ?? null,
        segment_id: null,
        self_reported_tier: profile?.skill_tier ?? null,
        was_skipped: skipped,
        handedness: profile?.dominant_hand ?? null,
        backhand_style: profile?.backhand_style ?? null,
        primary_goal: profile?.primary_goal ?? null,
        shot_type: resolvedShotType,
        blob_url: resolvedBlobUrl,
        composite_metrics: { user_summary: userSummary, observation_count: observations.length },
        llm_coached_tier: profile?.skill_tier ?? null,
        llm_assessed_tier: null,
        llm_tier_downgrade: false,
        capture_quality_flag: null,
      })
      .select('id')
      .single()
    if (insertError) {
      console.error('analysis_events insert failed:', insertError)
    } else {
      eventId = inserted?.id ?? null
    }
  } catch (err) {
    console.error('analysis_events insert threw:', err)
  }

  // Camera-angle telemetry — always runs, doesn't block coaching.
  after(() => classifyAndTagCaptureQuality(eventId, resolvedBlobUrl))

  const encoder = new TextEncoder()

  // Whether we're running the soft-fallback path. True when no observations
  // cleared the confidence floor — instead of blocking the user with the
  // old empty-state response, we now ALWAYS run the LLM (with a generic
  // coaching prompt) and surface a low-confidence banner so the UI can
  // hint that the read may be less precise. User feedback: clear side-on
  // shots were getting blocked because borderline-but-real observations
  // didn't clear the strict 0.6 floor; the floor is now 0.2 and this
  // fallback handles the residual "still no observations" case gracefully.
  //
  // Baseline-compare with zero diffs is NOT low confidence — it's
  // "today matches your baseline," which is a positive signal. The
  // banner copy ("pose tracking was uneven") would be wrong, so the
  // header stays off in that branch.
  const isLowConfidence = !primary && !isBaselineCompare

  // -----------------------------------------------------------------------
  // Coaching branch: build prompt, call LLM, post-filter, fall back to static.
  // -----------------------------------------------------------------------
  const tier: SkillTier | null = profile?.skill_tier ?? null
  const register = registerForTier(tier)
  const tierTokens = tier ? TIER_MAX_TOKENS[tier] : DEFAULT_MAX_TOKENS
  const maxTokens = isShotMode ? Math.min(tierTokens, 600) : tierTokens

  // Build the user prompt. Four paths:
  //   - baseline-compare with at least one diff: consistency-framed prompt.
  //   - baseline-compare with zero diffs: "matches your baseline" prompt.
  //   - Normal solo: structured observations + 3-5 exemplars.
  //   - Solo fallback (isLowConfidence): generic coaching from the angle summary.
  let userPrompt: string
  const cleanBaselineLabel: string | null =
    typeof baselineLabel === 'string' && baselineLabel.trim().length > 0
      ? sanitizePromptInput(baselineLabel, 60)
      : null

  if (isBaselineCompare && primary) {
    const exemplarSet: CueExemplar[] = []
    const seen = new Set<CueExemplar>()
    for (const ex of findExemplars(primary, 3)) {
      if (!seen.has(ex)) {
        seen.add(ex)
        exemplarSet.push(ex)
      }
    }
    for (const o of secondary) {
      for (const ex of findExemplars(o, 1)) {
        if (exemplarSet.length >= 5) break
        if (!seen.has(ex)) {
          seen.add(ex)
          exemplarSet.push(ex)
        }
      }
      if (exemplarSet.length >= 5) break
    }
    if (exemplarSet.length < 3) {
      for (const ex of CUE_EXEMPLARS) {
        if (exemplarSet.length >= 3) break
        if (!seen.has(ex)) {
          seen.add(ex)
          exemplarSet.push(ex)
        }
      }
    }
    userPrompt = buildBaselineUserPrompt({
      primary,
      secondary,
      exemplars: exemplarSet,
      register,
      shotType: resolvedShotType,
      tier,
      focus,
      handedness: profile?.dominant_hand ?? null,
      baselineLabel: cleanBaselineLabel,
    })
  } else if (isBaselineCompare && !primary) {
    userPrompt = buildBaselineMatchPrompt({
      register,
      shotType: resolvedShotType,
      tier,
      focus,
      handedness: profile?.dominant_hand ?? null,
      baselineLabel: cleanBaselineLabel,
    })
  } else if (primary) {
    const exemplarSet: CueExemplar[] = []
    const seen = new Set<CueExemplar>()
    for (const ex of findExemplars(primary, 3)) {
      if (!seen.has(ex)) {
        seen.add(ex)
        exemplarSet.push(ex)
      }
    }
    for (const o of secondary) {
      for (const ex of findExemplars(o, 1)) {
        if (exemplarSet.length >= 5) break
        if (!seen.has(ex)) {
          seen.add(ex)
          exemplarSet.push(ex)
        }
      }
      if (exemplarSet.length >= 5) break
    }
    // Top up with generic exemplars from the same register if the matched set
    // is short.
    if (exemplarSet.length < 3) {
      for (const ex of CUE_EXEMPLARS) {
        if (exemplarSet.length >= 3) break
        if (!seen.has(ex)) {
          seen.add(ex)
          exemplarSet.push(ex)
        }
      }
    }

    userPrompt = buildUserPrompt({
      primary,
      secondary,
      exemplars: exemplarSet,
      register,
      shotType: resolvedShotType,
      tier,
      focus,
      handedness: profile?.dominant_hand ?? null,
    })
  } else {
    userPrompt = buildFallbackUserPrompt({
      userSummary,
      register,
      shotType: resolvedShotType,
      tier,
      focus,
      handedness: profile?.dominant_hand ?? null,
    })
  }

  // 2026-05 — switched from buffer-and-validate to live token streaming
  // so the user sees the response build up in real time instead of
  // waiting ~5-10s for the full response. Each LLM attempt streams its
  // tokens directly to the client AS they arrive. If a digit/jargon
  // hits the buffered text, we abort the stream, send a CLEAR_MARKER
  // sentinel that the client uses to discard the bad attempt, then
  // retry. The senior-coach validator was removed from this path —
  // it ran AFTER the full junior response and would have caused a
  // jarring "replace what the user just watched stream in" flicker.
  // The strict prompt rules + post-filter remain as the safety net.

  const stream = new ReadableStream({
    async start(controller) {
      let attempts = 0
      let cleanedBody: string | null = null
      let lastError: string | null = null
      let outputTokens: number | null = null
      let usedFallback = false
      // True once we've streamed any tokens of an attempt to the
      // client. On retry we send CLEAR_MARKER to tell the client to
      // discard what it has so far.
      let hasStreamed = false

      while (attempts <= MAX_LLM_RETRIES && cleanedBody === null) {
        if (hasStreamed) {
          controller.enqueue(encoder.encode(STREAM_CLEAR_MARKER))
          hasStreamed = false
        }
        const userContent =
          attempts === 0 ? userPrompt : userPrompt + STRICT_RETRY_NOTE
        let buffered = ''
        let aborted = false
        try {
          const messageStream = anthropic.messages.stream({
            model: 'claude-sonnet-4-6',
            max_tokens: maxTokens,
            system: [
              { type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } },
            ],
            messages: [{ role: 'user', content: userContent }],
          })
          for await (const chunk of messageStream) {
            if (
              chunk.type === 'content_block_delta' &&
              chunk.delta.type === 'text_delta'
            ) {
              const newText = chunk.delta.text
              buffered += newText
              // Mid-stream abort: if the buffered output trips the
              // post-filter, stop emitting tokens immediately so we
              // don't waste tokens (and CLEAR_MARKER on the next
              // iteration discards what we already streamed).
              if (
                REJECT_DIGIT_RE.test(buffered) ||
                REJECT_JARGON_RE.test(buffered)
              ) {
                aborted = true
                break
              }
              controller.enqueue(encoder.encode(newText))
              hasStreamed = true
            } else if (
              chunk.type === 'message_delta' &&
              chunk.usage?.output_tokens
            ) {
              outputTokens = chunk.usage.output_tokens
            }
          }
        } catch (err) {
          lastError = err instanceof Error ? err.message : 'Anthropic stream failed'
          break
        }
        if (aborted) {
          attempts += 1
          continue
        }
        if (
          REJECT_DIGIT_RE.test(buffered) ||
          REJECT_JARGON_RE.test(buffered)
        ) {
          attempts += 1
          continue
        }
        cleanedBody = buffered.trim()
      }

      if (cleanedBody === null) {
        usedFallback = true
        // Fallback path: build a deterministic response from the chosen
        // observations. If we had streamed bad attempts, clear them
        // first so the static fallback replaces — not appends.
        if (hasStreamed) {
          controller.enqueue(encoder.encode(STREAM_CLEAR_MARKER))
          hasStreamed = false
        }
        const fallbackExemplar = primary
          ? findExemplars(primary, 1)[0] ?? CUE_EXEMPLARS[0] ?? null
          : CUE_EXEMPLARS[0] ?? null
        cleanedBody = buildStaticFallback({
          primary,
          secondary,
          exemplar: fallbackExemplar,
          register,
        })
        controller.enqueue(encoder.encode(cleanedBody))
      }

      const showWork = renderShowYourWork(primary, secondary)
      const fullResponse = showWork
        ? `${cleanedBody}\n${showWork}`
        : cleanedBody
      if (showWork) {
        controller.enqueue(encoder.encode(`\n${showWork}`))
      }
      if (lastError) {
        controller.enqueue(encoder.encode(`\n\n[ERROR] ${lastError}`))
      }
      controller.close()

      // Telemetry backfill. response_tip_count = primary + secondary count
      // (matches the new prompt's structure).
      if (eventId) {
        const metrics = extractResponseMetrics({
          tier,
          toolInput: null,
          markdownText: fullResponse,
          outputTokens,
        })
        const tipCount = 1 + secondary.length
        after(async () => {
          const { error } = await supabaseAdmin
            .from('analysis_events')
            .update({
              response_token_count: metrics.response_token_count,
              response_tip_count: tipCount,
              response_char_count: metrics.response_char_count,
              used_baseline_template: usedFallback,
            })
            .eq('id', eventId)
          if (error) console.error('analysis_events update failed:', error)
        })
      }
    },
  })

  const headers: Record<string, string> = {
    'Content-Type': 'text/plain; charset=utf-8',
    'Cache-Control': 'no-cache',
    'X-Content-Type-Options': 'nosniff',
  }
  if (eventId) headers['X-Analysis-Event-Id'] = eventId
  // Soft signal so the UI can render a hint banner. Replaces the prior
  // hard X-Analyze-Empty-State that blocked the whole response.
  if (isLowConfidence) headers['X-Analyze-Low-Confidence'] = 'true'
  // Specific gate reason so the UI can tailor the reshoot guidance.
  // Only set when the gate explicitly failed; isLowConfidence can also
  // be true on clips that passed the gate but had no observation clear
  // the confidence floor (different failure mode, no specific reshoot
  // advice to give).
  if (!gateResult.passed && gateResult.reason) {
    headers['X-Analyze-Gate-Reason'] = gateResult.reason
  }
  // Forward the chosen primary + secondary observations to the client
  // so /api/coach-followup can ground "what about my legs?"-style
  // questions in the actual rule firings instead of just the prior
  // analysis text. Base64-encoded JSON to dodge header-charset issues.
  const surfacedObservations = primary ? [primary, ...secondary] : secondary
  if (surfacedObservations.length > 0) {
    headers['X-Analyze-Observations'] = Buffer.from(
      JSON.stringify(surfacedObservations),
      'utf8',
    ).toString('base64')
  }
  return new NextResponse(stream, { headers })
}
