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

const SYSTEM_PROMPT = `You are a veteran tennis coach. Talk like one.
Voice rules, no exceptions:
1. Second person, imperative voice. Address the player directly.
2. NEVER include numbers, digits, percentages, or counts.
3. NEVER use em-dashes or en-dashes. Use full sentences or commas.
4. NEVER use biomechanics jargon (no "kinetic chain", no "trunk_rotation", no "joint angle").
5. External focus: attention on the racket, the ball, or a court target. Not on muscles.
6. Plain coach language. Short sentences.
7. Output exactly four markdown sections in this order: "## Quick Read", "## Primary cue", "## Other things I noticed", "## Recommended drills".
8. "## Quick Read" is two or three short bullets summarizing the swing in plain words. The player reads this first and skims; keep each bullet under twelve words.
9. "## Primary cue" is one short paragraph with the single most important fix.
10. "## Other things I noticed" is a short bulleted list of secondary observations.
11. "## Recommended drills" is two or three concrete on-court drills tied to the cues above. One drill per bullet, each one a single sentence the player can act on.
12. Do not generate any other section. The server appends "## Show your work" after you.
13. Do not narrate the observations. Coach them.`

// Stricter retry preamble — appended to the user prompt on a failed-output retry.
const STRICT_RETRY_NOTE = `\n\nSTRICT RETRY: Your previous output included disallowed content (digits, em-dashes, or jargon). Rewrite from scratch with NO numbers, NO dashes other than hyphens inside compound words, and NO biomech terms. Use the exemplar voice exactly.`

// Low-confidence path used to also prepend a markdown banner here, but
// the UI now renders a styled banner above the sections from the
// X-Analyze-Low-Confidence header — so embedding one in the body would
// double up. The header alone is the signal now.

// Patterns we reject in LLM output. Show your work is appended AFTER the
// filter runs, so the digit pattern there does not trip it.
const REJECT_DIGIT_RE = /\d/
const REJECT_JARGON_RE = /(kinetic chain|trunk_rotation|hip_rotation|joint angle|—|–)/i

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
  }
}

function jointHumanLabel(joint: string): string {
  return joint.replace(/_/g, ' ')
}

function renderObservationLine(o: Observation): string {
  return `${jointHumanLabel(o.joint)} at ${o.phase}: ${patternHumanLabel(o.pattern)} (severity: ${o.severity})`
}

/**
 * Render the deterministic Show Your Work block from the chosen observations.
 * Numbers come from the Observation rows. ° is included so the player can
 * tie the human-language coaching back to the underlying angles. This block
 * is concatenated AFTER the LLM body, so post-filter never inspects it.
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
Two or three short bullets summarizing the swing at a glance. Each bullet under twelve words. No numbers, no jargon, no em-dashes.

## Primary cue
One short paragraph addressing the primary issue. Imperative, second person, no numbers, no jargon, no em-dashes.

## Other things I noticed
One bullet for EVERY secondary observation listed above — do not drop any. Each bullet: imperative, single-issue, no numbers, no jargon, no em-dashes. The order can match severity. If there are no secondary observations, write a single bullet acknowledging the swing has otherwise hung together.

## Recommended drills
Two or three concrete on-court drills tied to the cues above. One drill per bullet, each one a single sentence the player can act on. No numbers, no jargon, no em-dashes.`
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
Two or three short bullets summarizing the swing at a glance. Each bullet under twelve words. No numbers, no jargon, no em-dashes.

## Primary cue
One short paragraph. Imperative, second person, no numbers, no jargon, no em-dashes. If the swing reads clean, lead with that.

## Other things I noticed
A short bulleted list (1-3 bullets) of general feel observations. If you genuinely have nothing to add, write a single bullet acknowledging the swing has otherwise hung together.

## Recommended drills
Two or three concrete on-court drills any player at this level benefits from. One drill per bullet, each a single sentence the player can act on. No numbers, no jargon, no em-dashes.`
}

/**
 * Buffer-and-validate one LLM call. Returns the cleaned body or null when
 * the post-filter rejected it.
 */
async function streamLlmBody(args: {
  systemPrompt: string
  userPrompt: string
  maxTokens: number
}): Promise<{ text: string | null; outputTokens: number | null; error: string | null }> {
  let buffered = ''
  let outputTokens: number | null = null
  let error: string | null = null
  try {
    const messageStream = anthropic.messages.stream({
      model: 'claude-sonnet-4-6',
      max_tokens: args.maxTokens,
      system: [
        { type: 'text', text: args.systemPrompt, cache_control: { type: 'ephemeral' } },
      ],
      messages: [{ role: 'user', content: args.userPrompt }],
    })
    for await (const chunk of messageStream) {
      if (
        chunk.type === 'content_block_delta' &&
        chunk.delta.type === 'text_delta'
      ) {
        buffered += chunk.delta.text
      } else if (chunk.type === 'message_delta' && chunk.usage?.output_tokens) {
        outputTokens = chunk.usage.output_tokens
      }
    }
  } catch (err) {
    error = err instanceof Error ? err.message : 'Anthropic stream failed'
  }
  if (error) return { text: null, outputTokens, error }
  if (REJECT_DIGIT_RE.test(buffered) || REJECT_JARGON_RE.test(buffered)) {
    return { text: null, outputTokens, error: null }
  }
  return { text: buffered.trim(), outputTokens, error: null }
}

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
  lines.push('- Shadow swing ten reps focusing on the primary cue before your next rally.')
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
    // compareMode + baselineLabel are still accepted on the wire so the
    // existing UI doesn't have to change, but the new pipeline routes both
    // legacy and baseline compares through extractObservations(), so the
    // labels themselves are no longer interpolated into the prompt.
    shotType: bodyShotType,
    blobUrl: bodyBlobUrl,
    mode: bodyMode,
  } = body
  const isShotMode = bodyMode === 'shot'

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
  const isLowConfidence = !primary

  // -----------------------------------------------------------------------
  // Coaching branch: build prompt, call LLM, post-filter, fall back to static.
  // -----------------------------------------------------------------------
  const tier: SkillTier | null = profile?.skill_tier ?? null
  const register = registerForTier(tier)
  const tierTokens = tier ? TIER_MAX_TOKENS[tier] : DEFAULT_MAX_TOKENS
  const maxTokens = isShotMode ? Math.min(tierTokens, 600) : tierTokens

  // Build the user prompt. Two paths:
  //   - Normal: structured observations + 3-5 exemplars.
  //   - Fallback (isLowConfidence): generic coaching from the angle summary.
  let userPrompt: string
  if (primary) {
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

  // The LLM call is buffered (not streamed straight to the client) so we can
  // run the post-filter and re-roll on rejection. Streaming AND retries are
  // incoherent — we'd have to roll back already-emitted tokens — so the call
  // buffers and we stream the cleaned result + Show Your Work to the client
  // in one shot. Tokens-per-second perception is preserved by keeping the
  // overall response under ~3 seconds.

  const stream = new ReadableStream({
    async start(controller) {
      let attempts = 0
      let cleanedBody: string | null = null
      let lastError: string | null = null
      let outputTokens: number | null = null
      let usedFallback = false

      while (attempts <= MAX_LLM_RETRIES && cleanedBody === null) {
        const userContent =
          attempts === 0 ? userPrompt : userPrompt + STRICT_RETRY_NOTE
        const result = await streamLlmBody({
          systemPrompt: SYSTEM_PROMPT,
          userPrompt: userContent,
          maxTokens,
        })
        if (result.outputTokens != null) outputTokens = result.outputTokens
        if (result.error) {
          lastError = result.error
          break // stream errored; do not retry — fall back below.
        }
        if (result.text) {
          cleanedBody = result.text
          break
        }
        attempts += 1
      }

      if (cleanedBody === null) {
        usedFallback = true
        // exemplarSet is only populated in the structured (primary-present)
        // path; the fallback prompt builder doesn't construct one. Pull a
        // generic exemplar from the global table when we're in low-
        // confidence mode so the static fallback still has a phrasing
        // anchor instead of the generic-pattern text.
        const fallbackExemplar = primary
          ? findExemplars(primary, 1)[0] ?? CUE_EXEMPLARS[0] ?? null
          : CUE_EXEMPLARS[0] ?? null
        cleanedBody = buildStaticFallback({
          primary,
          secondary,
          exemplar: fallbackExemplar,
          register,
        })
      }

      const showWork = renderShowYourWork(primary, secondary)
      // Low-confidence path emits no "Show your work" block (no
      // observations cleared the floor). The warning is purely a header
      // signal; UI renders the banner above the sections.
      const fullResponse = showWork
        ? `${cleanedBody}\n${showWork}`
        : cleanedBody
      controller.enqueue(encoder.encode(fullResponse))
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
