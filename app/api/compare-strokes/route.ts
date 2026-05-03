// POST /api/compare-strokes
//
// Best/worst stroke comparison endpoint. Given a list of detected strokes plus
// their per-stroke quality scores and per-stroke Observation rows, picks the
// best and the worst, hands the LLM a structured payload (observations,
// closed-vocab cue exemplars for the differences, register hint), and forces
// JSON-shape output via Anthropic's tool_use. Every sentence in the returned
// reasoning must cite at least one observation_id from the input set; the
// route validates this post-parse and retries the LLM up to twice on failure.
// After 2 retries it falls back to a deterministic template that quotes the
// cue table directly.
//
// The route deliberately avoids forcing a verdict on tight sessions: when
// the spread between the best and worst non-rejected score sits inside the
// TIE_THRESHOLD (z-score units), it returns isConsistent=true and a single
// register-aware consistencyCue paragraph. The user's research found that
// inventing a winner/loser on a noisy session erodes trust.

import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { createClient } from '@/lib/supabase/server'
import {
  getCoachingContext,
  registerForTier,
  type Register,
  type SkillTier,
} from '@/lib/profile'
import {
  observationId,
  type Observation,
} from '@/lib/coachingObservations'
import { CUE_EXEMPLARS, type CueExemplar } from '@/lib/cueExemplars'
import type { ShotType } from '@/lib/shotTypeConfig'
import { VALID_SHOT_TYPES } from '@/lib/shotTypeConfig'
import type { StrokeComparisonResult } from '@/lib/strokeAnalysis'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Spread (max - min) of non-rejected z-scores below which we suppress the
// best/worst verdict. Tuned from the user's research: 0.7 z-score units is
// roughly "noticeable but not decisive", and pretending to have a winner on
// anything tighter erodes player trust in the system.
const TIE_THRESHOLD = 0.7

// Word-count budget for each per-stroke reasoning paragraph. Both stated in
// the system prompt AND validated post-parse (defense in depth).
const MIN_REASONING_WORDS = 25
const MAX_REASONING_WORDS = 45

// Hard cap on cues quoted per stroke. Echoes the prompt rule.
const MAX_CUES_PER_STROKE = 2

// LLM retry budget. 1 primary + 2 retries = 3 total before deterministic fallback.
const MAX_LLM_RETRIES = 2

// ---------------------------------------------------------------------------
// Input/output types
// ---------------------------------------------------------------------------

interface StrokeInput {
  strokeId: string
  score: number
  rejected: boolean
  observations: Observation[]
}

// StrokeComparisonResult moved to lib/strokeAnalysis.ts (single source
// of truth across the stroke pipeline). Imported + re-exported above.

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

function normalizeShotType(raw: unknown): ShotType | null {
  if (typeof raw !== 'string') return null
  const lower = raw.toLowerCase().trim()
  return (VALID_SHOT_TYPES as readonly string[]).includes(lower)
    ? (lower as ShotType)
    : null
}

/**
 * Lightweight runtime guard for the POST body. Returns the parsed strokes or
 * null when shape doesn't match. We don't deep-validate Observation fields —
 * if the caller fed us garbage, downstream code will simply not find any
 * matching observation_ids and fall through to the deterministic template.
 */
function parseStrokes(raw: unknown): StrokeInput[] | null {
  if (!Array.isArray(raw)) return null
  const out: StrokeInput[] = []
  for (const r of raw) {
    if (!r || typeof r !== 'object') return null
    const strokeId = (r as Record<string, unknown>).strokeId
    const score = (r as Record<string, unknown>).score
    const rejected = (r as Record<string, unknown>).rejected
    const observations = (r as Record<string, unknown>).observations
    if (typeof strokeId !== 'string' || !strokeId) return null
    if (typeof rejected !== 'boolean') return null
    if (!Array.isArray(observations)) return null
    // Score: rejected strokes legitimately carry NaN — and JSON.stringify
    // turns NaN into `null`, so we accept null/non-finite values when the
    // stroke is flagged rejected. For non-rejected strokes the score must
    // be a finite number; otherwise the caller's z-score pipeline produced
    // garbage and we'd rather reject the request than rank on it.
    let scoreNum: number
    if (typeof score === 'number') {
      scoreNum = score
    } else if (score === null && rejected) {
      scoreNum = Number.NaN
    } else {
      return null
    }
    if (!rejected && !Number.isFinite(scoreNum)) return null
    // Tolerate per-observation shape silently — upstream type already enforced
    // it and we don't want to double-validate the union of detection rules.
    out.push({
      strokeId,
      score: scoreNum,
      rejected,
      observations: observations as Observation[],
    })
  }
  return out
}

// ---------------------------------------------------------------------------
// Cue catalog helpers
// ---------------------------------------------------------------------------

/**
 * Stable, content-addressable id for a cue. CUE_EXEMPLARS rows are unique by
 * (pattern, phase) when phase is present, and unique on pattern alone when
 * phase is absent (drift_from_baseline). Including the rendered phrase text
 * disambiguates the rare cases where two rows share (pattern, phase) but
 * carry different phrasings — happens for cramped_elbow, locked_knees, etc.
 */
function cueIdFor(ex: CueExemplar, register: Register): string {
  const text = register === 'technical' ? ex.technical : ex.plain
  return `${ex.pattern}|${ex.phase ?? '*'}|${text.slice(0, 32)}`
}

/**
 * Build the candidate cue set for the LLM. We include every exemplar whose
 * pattern appears in either the best- or worst-stroke's observations, in the
 * chosen register. The LLM will quote (or paraphrase) one of these and cite
 * its id; anything outside this set fails the post-parse cue check.
 */
function buildCueCandidates(
  observations: Observation[],
  register: Register,
): { ex: CueExemplar; id: string; text: string }[] {
  const patterns = new Set<Observation['pattern']>()
  for (const o of observations) patterns.add(o.pattern)
  const candidates: { ex: CueExemplar; id: string; text: string }[] = []
  for (const ex of CUE_EXEMPLARS) {
    if (!patterns.has(ex.pattern)) continue
    const text = register === 'technical' ? ex.technical : ex.plain
    candidates.push({ ex, id: cueIdFor(ex, register), text })
  }
  return candidates
}

// ---------------------------------------------------------------------------
// Tie suppression + best/worst selection
// ---------------------------------------------------------------------------

/**
 * Compute (best, worst, delta) over the eligible (non-rejected, finite-score)
 * stroke pool. Returns null fields when the pool is too thin to rank.
 *
 * 2-stroke pool: we treat as the tie path. The user's research warned that
 * forcing a verdict on a tight session erodes trust, and a 2-stroke pool
 * carries no statistical weight to override that — the consistency framing
 * is the safer UX.
 */
function pickBestWorst(strokes: StrokeInput[]): {
  best: StrokeInput | null
  worst: StrokeInput | null
  delta: number
  eligibleCount: number
} {
  const eligible = strokes.filter(
    (s) => !s.rejected && Number.isFinite(s.score),
  )
  if (eligible.length < 3) {
    return {
      best: null,
      worst: null,
      delta: 0,
      eligibleCount: eligible.length,
    }
  }
  let best = eligible[0]
  let worst = eligible[0]
  for (const s of eligible) {
    if (s.score > best.score) best = s
    if (s.score < worst.score) worst = s
  }
  return {
    best,
    worst,
    delta: best.score - worst.score,
    eligibleCount: eligible.length,
  }
}

// ---------------------------------------------------------------------------
// Consistency cue (used when isConsistent=true)
//
// CUE_EXEMPLARS only carries deviation-cures, not consistency-praise. Rather
// than force a column on cueExemplars.ts that no other consumer needs, we
// keep two register-aware phrasings here. Both obey the cueExemplars voice
// rules (imperative, second-person, no numbers, no jargon, no em-dashes).
// ---------------------------------------------------------------------------

const CONSISTENCY_CUE: Record<Register, string> = {
  plain:
    'Your strokes are all hitting the same spot today. Trust the rhythm you have, and groove the same shape on the next basket.',
  technical:
    'Stroke-to-stroke variance is tight in this session. Lock in this baseline shape and groove the same load and contact window on the next set.',
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are a veteran tennis coach explaining WHY one stroke was better and one was worse.

Voice rules, no exceptions:
1. Second person, imperative voice.
2. NEVER use em-dashes or en-dashes. Use full sentences or commas.
3. NEVER use biomechanics jargon ("kinetic chain", "trunk_rotation", "joint angle").
4. External focus: attention on the racket, the ball, or a court target. Never on muscles.
5. Plain coach language. Short sentences.

Citation rules — these are mechanically validated, not stylistic:
A. Every sentence in your "reasoning" string MUST end with a bracketed list of one or more observation_ids you are citing, e.g. "Your back leg loaded clean here [stroke3_right_knee_loading_shallow_knee_load]."
B. Every cited observation_id MUST be one of the observation_ids you were given for THAT stroke. Inventing ids will fail validation.
C. You may quote at most ${MAX_CUES_PER_STROKE} cues per stroke, and every cue you quote MUST be drawn from the CUES list provided below by id. Reference the cue id in the citedCues array. Do NOT invent new cue phrasings.
D. Keep each per-stroke "reasoning" between ${MIN_REASONING_WORDS} and ${MAX_REASONING_WORDS} words (excluding the bracketed citation tags).

Use the emit_comparison tool to return your verdict. Do not write prose outside the tool call.`

interface PromptInput {
  best: StrokeInput
  worst: StrokeInput
  delta: number
  bestObsLines: { id: string; line: string }[]
  worstObsLines: { id: string; line: string }[]
  deltaPatterns: Observation['pattern'][]
  cues: { id: string; text: string }[]
  register: Register
  shotType: ShotType | null
  tier: SkillTier | null
  retryNote: string | null
}

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

function renderObservationLine(strokeId: string, o: Observation): { id: string; line: string } {
  const id = observationId(strokeId, o)
  const joint = o.joint.replace(/_/g, ' ')
  const line = `- [${id}] ${joint} at ${o.phase}: ${patternHumanLabel(o.pattern)} (severity: ${o.severity})`
  return { id, line }
}

function buildUserPrompt(args: PromptInput): string {
  const {
    best,
    worst,
    delta,
    bestObsLines,
    worstObsLines,
    deltaPatterns,
    cues,
    register,
    shotType,
    tier,
    retryNote,
  } = args

  const tierHint = tier
    ? `Player skill: ${tier}. Calibrate vocabulary to that level.`
    : 'Player skill: unknown. Use plain coaching language.'
  const shotHint = shotType ? `Shot: ${shotType}.` : 'Shot: not specified.'
  const registerHint =
    register === 'technical'
      ? 'Use the technical register (skilled-player vocabulary).'
      : 'Use the plain register (beginner-friendly vocabulary).'

  const bestObsBlock =
    bestObsLines.length > 0
      ? bestObsLines.map((o) => o.line).join('\n')
      : '- (no observations cleared the floor; coach the absence as a strength)'
  const worstObsBlock =
    worstObsLines.length > 0
      ? worstObsLines.map((o) => o.line).join('\n')
      : '- (no observations cleared the floor)'

  const deltaBlock =
    deltaPatterns.length > 0
      ? deltaPatterns.map((p) => `- ${patternHumanLabel(p)}`).join('\n')
      : '- (no clean delta — the worst stroke just stacked more of the same patterns)'

  const cueBlock =
    cues.length > 0
      ? cues.map((c) => `- [${c.id}] ${c.text}`).join('\n')
      : '- (no cues — keep the reasoning observation-only)'

  const retry = retryNote ? `\n\nSTRICT RETRY: ${retryNote}\n` : ''

  return `${tierHint}
${shotHint}
${registerHint}

Score spread between best and worst: ${delta.toFixed(2)} z-score units.

BEST STROKE (id=${best.strokeId}, score=${best.score.toFixed(2)})
Observations:
${bestObsBlock}

WORST STROKE (id=${worst.strokeId}, score=${worst.score.toFixed(2)})
Observations:
${worstObsBlock}

DELTA — observation patterns that differ between the two strokes:
${deltaBlock}

CUES — closed vocabulary you may draw from. Cite each one by id in citedCues.
${cueBlock}${retry}

Now call the emit_comparison tool. Each per-stroke "reasoning" string must cite the relevant observation_ids in brackets at the end of every sentence, must stay between ${MIN_REASONING_WORDS} and ${MAX_REASONING_WORDS} words (excluding the bracketed citation tags), and must quote no more than ${MAX_CUES_PER_STROKE} cues from the CUES list. The "citations" array for each stroke must be the union of observation_ids referenced in that stroke's reasoning. The "citedCues" array for each stroke must be the union of cue ids quoted (or empty if you didn't quote any).`
}

// ---------------------------------------------------------------------------
// Tool schema
// ---------------------------------------------------------------------------

const COMPARE_TOOL_NAME = 'emit_comparison' as const

const compareToolSchema = {
  name: COMPARE_TOOL_NAME,
  description:
    'Emit the best/worst stroke comparison. Each strokeId must match the input id; each citation must be an observation_id from the input set; each citedCue must be a cue id from the candidates list.',
  input_schema: {
    type: 'object',
    additionalProperties: false,
    required: ['best', 'worst'],
    properties: {
      best: {
        type: 'object',
        additionalProperties: false,
        required: ['strokeId', 'reasoning', 'citations', 'citedCues'],
        properties: {
          strokeId: { type: 'string', minLength: 1 },
          reasoning: { type: 'string', minLength: 10, maxLength: 600 },
          citations: { type: 'array', items: { type: 'string' }, minItems: 1 },
          citedCues: { type: 'array', items: { type: 'string' } },
        },
      },
      worst: {
        type: 'object',
        additionalProperties: false,
        required: ['strokeId', 'reasoning', 'citations', 'citedCues'],
        properties: {
          strokeId: { type: 'string', minLength: 1 },
          reasoning: { type: 'string', minLength: 10, maxLength: 600 },
          citations: { type: 'array', items: { type: 'string' }, minItems: 1 },
          citedCues: { type: 'array', items: { type: 'string' } },
        },
      },
    },
  },
} as const

interface CompareToolBest {
  strokeId: string
  reasoning: string
  citations: string[]
  citedCues: string[]
}

interface CompareToolOutput {
  best: CompareToolBest
  worst: CompareToolBest
}

function parseToolOutput(raw: unknown): CompareToolOutput | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const parseOne = (v: unknown): CompareToolBest | null => {
    if (!v || typeof v !== 'object') return null
    const o = v as Record<string, unknown>
    if (typeof o.strokeId !== 'string' || !o.strokeId) return null
    if (typeof o.reasoning !== 'string' || !o.reasoning.trim()) return null
    if (!Array.isArray(o.citations)) return null
    if (!Array.isArray(o.citedCues)) return null
    for (const c of o.citations) if (typeof c !== 'string' || !c) return null
    for (const c of o.citedCues) if (typeof c !== 'string' || !c) return null
    return {
      strokeId: o.strokeId,
      reasoning: o.reasoning,
      citations: o.citations as string[],
      citedCues: o.citedCues as string[],
    }
  }
  const best = parseOne(r.best)
  const worst = parseOne(r.worst)
  if (!best || !worst) return null
  return { best, worst }
}

// ---------------------------------------------------------------------------
// Post-parse validation
// ---------------------------------------------------------------------------

interface ValidationContext {
  expectedBestId: string
  expectedWorstId: string
  validObservationIds: Set<string>
  validCueIds: Set<string>
}

interface ValidationResult {
  ok: boolean
  reason: string | null
}

/**
 * Split reasoning into sentences. We use a permissive splitter rather than
 * Intl.Segmenter (which is jsdom-flaky) — punctuation inside quotes is rare
 * in coach voice, and the validator only needs to see "did each chunk end
 * with [id, ...]?".
 */
function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/g)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
}

const CITATION_TAG_RE = /\[([^\[\]]+)\]\s*\.?$/
const ALL_CITATION_TAGS_RE = /\[[^\[\]]+\]/g

/**
 * Strip bracketed citation tags so the word counter doesn't count them.
 * Also collapses any leftover whitespace.
 */
function stripCitationTags(text: string): string {
  return text.replace(ALL_CITATION_TAGS_RE, '').replace(/\s+/g, ' ').trim()
}

function wordCount(text: string): number {
  const stripped = stripCitationTags(text)
  if (!stripped) return 0
  return stripped.split(/\s+/g).length
}

function validateOne(
  side: 'best' | 'worst',
  emitted: CompareToolBest,
  expectedId: string,
  validObs: Set<string>,
  validCues: Set<string>,
): ValidationResult {
  if (emitted.strokeId !== expectedId) {
    return {
      ok: false,
      reason: `${side}.strokeId was "${emitted.strokeId}" but the eligible ${side} stroke is "${expectedId}". Use the strokeId we passed you.`,
    }
  }
  // Word budget (excluding citation brackets so the LLM isn't penalized for
  // a long citation list).
  const wc = wordCount(emitted.reasoning)
  if (wc < MIN_REASONING_WORDS || wc > MAX_REASONING_WORDS) {
    return {
      ok: false,
      reason: `${side}.reasoning was ${wc} words; the budget is ${MIN_REASONING_WORDS}-${MAX_REASONING_WORDS}.`,
    }
  }
  // Per-sentence citation gate: each sentence must end with [id,...] and
  // every id inside must be a real observation_id from the input set.
  const sentences = splitSentences(emitted.reasoning)
  if (sentences.length === 0) {
    return { ok: false, reason: `${side}.reasoning had no sentences.` }
  }
  for (const sentence of sentences) {
    const m = sentence.match(CITATION_TAG_RE)
    if (!m) {
      return {
        ok: false,
        reason: `${side}.reasoning has a sentence that does not end with a [observation_id] tag: "${sentence}"`,
      }
    }
    const ids = m[1]
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
    if (ids.length === 0) {
      return {
        ok: false,
        reason: `${side}.reasoning has an empty citation tag.`,
      }
    }
    for (const id of ids) {
      if (!validObs.has(id)) {
        return {
          ok: false,
          reason: `${side}.reasoning cites observation_id "${id}" which is not in the input set.`,
        }
      }
    }
    // Mid-sentence bracket guard: the end-anchored CITATION_TAG_RE only
    // covers the trailing tag, so a sentence like
    //   "Your back leg [fabricated] loaded clean here [real_obs]."
    // would otherwise sneak the fabricated id past validation. Walk every
    // bracketed token in the sentence and reject any id outside validObs.
    for (const tagMatch of sentence.matchAll(ALL_CITATION_TAGS_RE)) {
      const inner = tagMatch[0].slice(1, -1)
      const tagIds = inner
        .split(',')
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
      for (const id of tagIds) {
        if (!validObs.has(id)) {
          return {
            ok: false,
            reason: `${side}.reasoning contains bracketed id "${id}" which is not in the input observation set. Every [bracketed] token must be a real observation_id.`,
          }
        }
      }
    }
  }
  // Citations array must be a subset of validObs and must include every id
  // referenced inside the reasoning sentences.
  const allRefIds = new Set<string>()
  for (const sentence of sentences) {
    for (const tagMatch of sentence.matchAll(ALL_CITATION_TAGS_RE)) {
      const inner = tagMatch[0].slice(1, -1)
      for (const id of inner.split(',').map((s) => s.trim()).filter(Boolean)) {
        allRefIds.add(id)
      }
    }
  }
  for (const c of emitted.citations) {
    if (!validObs.has(c)) {
      return {
        ok: false,
        reason: `${side}.citations contains "${c}" which is not in the input observation set.`,
      }
    }
  }
  for (const ref of allRefIds) {
    if (!emitted.citations.includes(ref)) {
      return {
        ok: false,
        reason: `${side}.citations is missing "${ref}" which was cited inside the reasoning.`,
      }
    }
  }
  // Cue gate: every emitted citedCue id must be in the candidate set.
  if (emitted.citedCues.length > MAX_CUES_PER_STROKE) {
    return {
      ok: false,
      reason: `${side}.citedCues quoted ${emitted.citedCues.length} cues; the cap is ${MAX_CUES_PER_STROKE}.`,
    }
  }
  for (const cueId of emitted.citedCues) {
    if (!validCues.has(cueId)) {
      return {
        ok: false,
        reason: `${side}.citedCues contains "${cueId}" which is not in the cue candidate set. Pick one of the listed [id] cue rows.`,
      }
    }
  }
  return { ok: true, reason: null }
}

function validateOutput(
  out: CompareToolOutput,
  ctx: ValidationContext,
): ValidationResult {
  const bestRes = validateOne(
    'best',
    out.best,
    ctx.expectedBestId,
    ctx.validObservationIds,
    ctx.validCueIds,
  )
  if (!bestRes.ok) return bestRes
  return validateOne(
    'worst',
    out.worst,
    ctx.expectedWorstId,
    ctx.validObservationIds,
    ctx.validCueIds,
  )
}

// ---------------------------------------------------------------------------
// LLM driver
// ---------------------------------------------------------------------------

async function callCompareLlm(args: {
  systemPrompt: string
  userPrompt: string
  maxTokens: number
}): Promise<{ output: CompareToolOutput | null; error: string | null }> {
  try {
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: args.maxTokens,
      system: args.systemPrompt,
      messages: [{ role: 'user', content: args.userPrompt }],
      tools: [compareToolSchema as unknown as Anthropic.Tool],
      tool_choice: { type: 'tool', name: COMPARE_TOOL_NAME },
    })
    if (!Array.isArray(response.content)) return { output: null, error: null }
    const toolBlock = response.content.find(
      (b): b is Extract<typeof b, { type: 'tool_use' }> =>
        b.type === 'tool_use' && b.name === COMPARE_TOOL_NAME,
    )
    if (!toolBlock) return { output: null, error: null }
    const parsed = parseToolOutput(toolBlock.input)
    return { output: parsed, error: null }
  } catch (err) {
    const error = err instanceof Error ? err.message : 'Anthropic call failed'
    return { output: null, error }
  }
}

// ---------------------------------------------------------------------------
// Deterministic fallback (used when all LLM retries fail validation)
//
// Quotes the cue table directly, citing every observation we know about for
// each stroke so the citations array is non-empty. Word count tuned to land
// inside the budget (counted post-strip).
// ---------------------------------------------------------------------------

function buildFallbackReasoning(
  side: 'best' | 'worst',
  stroke: StrokeInput,
  cues: { id: string; text: string }[],
): { reasoning: string; citations: string[]; citedCues: string[] } {
  const obsIds = stroke.observations.map((o) => observationId(stroke.strokeId, o))
  const tag = obsIds.length > 0 ? `[${obsIds.join(',')}]` : ''
  if (side === 'best') {
    if (obsIds.length === 0) {
      // Best with no observations is just a clean stroke; we can't cite
      // anything so we leave citations empty and the route caller will fall
      // back the verdict to null. (Caller already handles that.)
      return { reasoning: '', citations: [], citedCues: [] }
    }
    const reasoning = `Your best stroke held its shape and stayed efficient through contact ${tag}. The pieces lined up clean here, so trust this rhythm and keep grooving the same shape on the next basket ${tag}.`
    return {
      reasoning,
      citations: obsIds,
      citedCues: [],
    }
  }
  // Worst path quotes one cue verbatim if we have one, citing the cue id.
  const cue = cues[0] ?? null
  if (obsIds.length === 0) {
    return { reasoning: '', citations: [], citedCues: [] }
  }
  const cueSentence = cue
    ? ` ${cue.text} ${tag}`
    : ` Reset the load and the contact window before the next ball ${tag}`
  const reasoning = `Your worst stroke leaked structure where the best one held it ${tag}.${cueSentence}.`
  return {
    reasoning,
    citations: obsIds,
    citedCues: cue ? [cue.id] : [],
  }
}

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------

export async function POST(request: NextRequest) {
  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  if (!body || typeof body !== 'object') {
    return NextResponse.json({ error: 'Invalid body' }, { status: 400 })
  }
  const b = body as Record<string, unknown>

  const strokes = parseStrokes(b.strokes)
  if (!strokes) {
    return NextResponse.json(
      { error: 'Invalid strokes payload' },
      { status: 400 },
    )
  }
  const shotType = normalizeShotType(b.shotType)

  // Pull profile context the same way /api/analyze does — best-effort, fall
  // through to register='plain' when the user is anonymous.
  const authClient = await createClient()
  const { profile } = await getCoachingContext(authClient)
  const tier: SkillTier | null = profile?.skill_tier ?? null
  const register = registerForTier(tier)

  // ---- Tie suppression --------------------------------------------------
  const { best, worst, delta, eligibleCount } = pickBestWorst(strokes)

  // All-rejected, single-eligible, or 2-eligible strokes: there's no real
  // ranking to make. Per spec: empty/null result with no error for the
  // all-rejected case; tie path for the 2-stroke case.
  if (eligibleCount === 0) {
    const result: StrokeComparisonResult = {
      best: null,
      worst: null,
      isConsistent: false,
    }
    return NextResponse.json(result)
  }
  if (eligibleCount < 3 || !best || !worst || delta < TIE_THRESHOLD) {
    const result: StrokeComparisonResult = {
      best: null,
      worst: null,
      isConsistent: true,
      consistentCue: CONSISTENCY_CUE[register],
    }
    return NextResponse.json(result)
  }

  // ---- Build prompt context --------------------------------------------
  const bestObsLines = best.observations.map((o) =>
    renderObservationLine(best.strokeId, o),
  )
  const worstObsLines = worst.observations.map((o) =>
    renderObservationLine(worst.strokeId, o),
  )
  const validObservationIds = new Set<string>([
    ...bestObsLines.map((x) => x.id),
    ...worstObsLines.map((x) => x.id),
  ])

  // Delta = patterns present in worst but not best. Empty when worst is
  // strictly a "more of the same" version of best.
  const bestPatterns = new Set(best.observations.map((o) => o.pattern))
  const deltaPatterns: Observation['pattern'][] = []
  for (const o of worst.observations) {
    if (!bestPatterns.has(o.pattern) && !deltaPatterns.includes(o.pattern)) {
      deltaPatterns.push(o.pattern)
    }
  }

  // Cue candidates pulled from CUE_EXEMPLARS for every pattern that appears
  // in either stroke. Keeps the LLM phrasing grounded in the closed table.
  const cueCandidates = buildCueCandidates(
    [...best.observations, ...worst.observations],
    register,
  )
  const validCueIds = new Set(cueCandidates.map((c) => c.id))

  // ---- LLM loop --------------------------------------------------------
  let retryNote: string | null = null
  let lastValidation: ValidationResult | null = null
  for (let attempt = 0; attempt <= MAX_LLM_RETRIES; attempt++) {
    const userPrompt = buildUserPrompt({
      best,
      worst,
      delta,
      bestObsLines,
      worstObsLines,
      deltaPatterns,
      cues: cueCandidates.map((c) => ({ id: c.id, text: c.text })),
      register,
      shotType,
      tier,
      retryNote,
    })

    const { output, error } = await callCompareLlm({
      systemPrompt: SYSTEM_PROMPT,
      userPrompt,
      maxTokens: 800,
    })

    if (error) {
      // Network / API failure — don't burn retries on it; fall straight to
      // the deterministic template below. The user gets a verdict, we just
      // log the underlying error.
      console.error('compare-strokes anthropic error:', error)
      break
    }
    if (!output) {
      retryNote =
        'The previous attempt did not return a valid emit_comparison tool call. Call the tool with the required shape.'
      continue
    }
    const ctx: ValidationContext = {
      expectedBestId: best.strokeId,
      expectedWorstId: worst.strokeId,
      validObservationIds,
      validCueIds,
    }
    const v = validateOutput(output, ctx)
    if (v.ok) {
      const result: StrokeComparisonResult = {
        best: {
          strokeId: output.best.strokeId,
          reasoning: output.best.reasoning,
          citations: output.best.citations,
        },
        worst: {
          strokeId: output.worst.strokeId,
          reasoning: output.worst.reasoning,
          citations: output.worst.citations,
        },
        isConsistent: false,
      }
      return NextResponse.json(result)
    }
    lastValidation = v
    retryNote = `${v.reason} Re-emit with the citation rules satisfied: every sentence ends with [observation_id] tags drawn from the input set, only cues from the CUES list quoted, and word count inside ${MIN_REASONING_WORDS}-${MAX_REASONING_WORDS}.`
  }

  // ---- Deterministic fallback ------------------------------------------
  console.warn(
    'compare-strokes: all LLM attempts failed validation, falling back to template.',
    lastValidation?.reason ?? 'no validation context',
  )
  const fbBest = buildFallbackReasoning('best', best, [])
  const fbWorst = buildFallbackReasoning('worst', worst, cueCandidates)
  // If either fallback couldn't synthesize a reasoning (no observations on
  // best, for example), drop into the consistency path so we don't ship an
  // empty string back as a verdict.
  if (!fbBest.reasoning || !fbWorst.reasoning) {
    const result: StrokeComparisonResult = {
      best: null,
      worst: null,
      isConsistent: true,
      consistentCue: CONSISTENCY_CUE[register],
    }
    return NextResponse.json(result)
  }
  const result: StrokeComparisonResult = {
    best: {
      strokeId: best.strokeId,
      reasoning: fbBest.reasoning,
      citations: fbBest.citations,
    },
    worst: {
      strokeId: worst.strokeId,
      reasoning: fbWorst.reasoning,
      citations: fbWorst.citations,
    },
    isConsistent: false,
  }
  return NextResponse.json(result)
}
