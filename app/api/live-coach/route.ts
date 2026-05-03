import { NextRequest, NextResponse } from 'next/server'
import { after } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { supabaseAdmin } from '@/lib/supabase'
import { createClient } from '@/lib/supabase/server'
import { buildAngleSummary } from '@/lib/jointAngles'
import { buildLiveCoachingPrompt, LIVE_SYSTEM_PROMPT } from '@/lib/livePrompt'
import { getCoachingContext, type SkillTier } from '@/lib/profile'
import { sanitizePromptInput } from '@/lib/sanitize'
import type { KeypointsJson } from '@/lib/supabase'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

const VALID_SHOT_TYPES = new Set(['forehand', 'backhand', 'serve', 'volley'] as const)
type LiveShotType = 'forehand' | 'backhand' | 'serve' | 'volley'

const MAX_SWINGS_PER_BATCH = 6
const MAX_SUMMARY_CHARS = 2000
// 2026-05 — dropped 120 → 35 to enforce the new "≤ 6 words" prompt
// rule. At 35 tokens the model physically cannot exceed ~10-12 words.
// Keeps the spoken cue under ~1.5s of TTS audio so it doesn't bleed
// into the next rally.
const LIVE_MAX_TOKENS = 35

// Post-filter for live coach output. Mirrors the analyze-route filter
// but more aggressive: live coaching must be brief and TTS-friendly,
// so we reject digits, em/en-dashes, and any reading of "joint angle"
// or raw field names. The route silently swallows rejections rather
// than retrying because a reject in live mode means "fall through to
// silence" — better to say nothing than to rush a retry that arrives
// even later.
const LIVE_REJECT_RE = /(\d|—|–|joint angle|trunk_rotation|hip_rotation)/i
// Hard cap on how many recent cues we accept from the client. The prompt
// builder slices to the last 3 anyway; this is a defense-in-depth limit so
// a misbehaving client can't blow up the request body.
const MAX_RECENT_CUES = 8
const MAX_RECENT_CUE_CHARS = 200

type IncomingSwing = {
  angleSummary?: unknown
  startMs?: unknown
  endMs?: unknown
}

export async function POST(request: NextRequest) {
  const authClient = await createClient()
  const { profile, skipped } = await getCoachingContext(authClient)
  const { data: userData } = await authClient.auth.getUser().catch(() => ({ data: { user: null } }))
  const userId = userData?.user?.id ?? null
  if (!userId) {
    return NextResponse.json({ error: 'Not signed in' }, { status: 401 })
  }

  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const b = body as Record<string, unknown>

  const shotTypeRaw = b.shotType
  if (
    typeof shotTypeRaw !== 'string' ||
    !(VALID_SHOT_TYPES as Set<string>).has(shotTypeRaw)
  ) {
    return NextResponse.json(
      { error: `shotType must be one of ${Array.from(VALID_SHOT_TYPES).join(', ')}` },
      { status: 400 },
    )
  }
  const shotType = shotTypeRaw as LiveShotType

  const batchIndex =
    typeof b.batchIndex === 'number' && Number.isFinite(b.batchIndex)
      ? Math.max(0, Math.floor(b.batchIndex))
      : 0
  const sessionDurationMs =
    typeof b.sessionDurationMs === 'number' && Number.isFinite(b.sessionDurationMs)
      ? Math.max(0, Math.floor(b.sessionDurationMs))
      : 0

  const rawSwings = Array.isArray(b.recentSwings) ? (b.recentSwings as IncomingSwing[]) : []
  if (rawSwings.length === 0) {
    return NextResponse.json({ error: 'recentSwings must be non-empty' }, { status: 400 })
  }
  if (rawSwings.length > MAX_SWINGS_PER_BATCH) {
    return NextResponse.json(
      { error: `recentSwings cannot exceed ${MAX_SWINGS_PER_BATCH}` },
      { status: 400 },
    )
  }
  const swings = rawSwings
    .map((s) => {
      const angleSummary =
        sanitizePromptInput(typeof s.angleSummary === 'string' ? s.angleSummary : '', MAX_SUMMARY_CHARS) ?? ''
      return {
        angleSummary,
        startMs:
          typeof s.startMs === 'number' && Number.isFinite(s.startMs)
            ? Math.max(0, Math.floor(s.startMs))
            : 0,
        endMs:
          typeof s.endMs === 'number' && Number.isFinite(s.endMs)
            ? Math.max(0, Math.floor(s.endMs))
            : 0,
      }
    })
    .filter((s) => s.angleSummary.length > 0)
  if (swings.length === 0) {
    return NextResponse.json({ error: 'No valid swings after sanitization' }, { status: 400 })
  }

  // Optional recentCues: the last few cues the player has already heard this
  // session. Sanitized + capped so prompt-injection attempts in a returned
  // coach response can't escape the next round-trip.
  const rawRecentCues = Array.isArray(b.recentCues) ? (b.recentCues as unknown[]) : []
  const recentCues: string[] = []
  for (const raw of rawRecentCues.slice(-MAX_RECENT_CUES)) {
    if (typeof raw !== 'string') continue
    const cleaned = sanitizePromptInput(raw, MAX_RECENT_CUE_CHARS)
    if (cleaned && cleaned.length > 0) recentCues.push(cleaned)
  }

  // Best-effort baseline lookup. Failures never block coaching.
  let baselineSummary: string | null = null
  let baselineLabel: string | null = null
  try {
    const { data: baseline } = await supabaseAdmin
      .from('user_baselines')
      .select('label, keypoints_json')
      .eq('user_id', userId)
      .eq('shot_type', shotType)
      .eq('is_active', true)
      .maybeSingle()
    if (baseline?.keypoints_json) {
      const kp = baseline.keypoints_json as KeypointsJson
      if (Array.isArray(kp.frames) && kp.frames.length > 0) {
        baselineSummary = buildAngleSummary(kp.frames)
        baselineLabel = typeof baseline.label === 'string' ? baseline.label : null
      }
    }
  } catch (err) {
    console.error('live-coach baseline lookup failed:', err)
  }

  const { prompt, usedBaselineTemplate } = buildLiveCoachingPrompt({
    profile,
    skipped,
    shotType,
    swings,
    baselineSummary,
    baselineLabel,
    recentCues,
  })

  const coachedTier: SkillTier | null = profile?.skill_tier ?? null

  // Insert telemetry row BEFORE streaming so the X-Analysis-Event-Id header
  // has a value. session_id, segment_id, blob_url stay null until
  // /api/sessions/live backfills them at session stop.
  let eventId: string | null = null
  try {
    const { data: inserted, error: insertError } = await supabaseAdmin
      .from('analysis_events')
      .insert({
        user_id: userId,
        session_id: null,
        segment_id: null,
        self_reported_tier: profile?.skill_tier ?? null,
        was_skipped: skipped,
        handedness: profile?.dominant_hand ?? null,
        backhand_style: profile?.backhand_style ?? null,
        primary_goal: profile?.primary_goal ?? null,
        shot_type: shotType,
        blob_url: null,
        composite_metrics: {
          live_batch: {
            batch_index: batchIndex,
            swing_count: swings.length,
            session_duration_ms: sessionDurationMs,
            baseline_present: baselineSummary !== null,
          },
        },
        llm_coached_tier: coachedTier,
        llm_assessed_tier: null,
        llm_tier_downgrade: false,
        capture_quality_flag: null,
      })
      .select('id')
      .single()
    if (insertError) {
      console.error('live-coach analysis_events insert failed:', insertError)
    } else {
      eventId = inserted?.id ?? null
    }
  } catch (err) {
    console.error('live-coach analysis_events insert threw:', err)
  }

  // Buffer-and-validate. At ≤6 words / ≤35 tokens the whole reply is one
  // ~5-character cue, so chunked SSE streaming buys almost nothing —
  // network round-trip is the bottleneck, not generation. Buffering lets
  // us run the post-filter (digits, em-dashes, jargon) before the TTS
  // queue speaks the cue out loud.
  let text = ''
  let outputTokens: number | null = null
  try {
    const stream = anthropic.messages.stream({
      model: 'claude-sonnet-4-6',
      max_tokens: LIVE_MAX_TOKENS,
      system: LIVE_SYSTEM_PROMPT,
      messages: [{ role: 'user', content: prompt }],
    })
    for await (const chunk of stream) {
      if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
        text += chunk.delta.text
      }
    }
    const final = await stream.finalMessage()
    outputTokens = final.usage?.output_tokens ?? null
  } catch (err) {
    console.error('live-coach LLM call failed:', err)
    return NextResponse.json({ error: 'Coaching call failed' }, { status: 502 })
  }

  let collapsed = text.trim().replace(/\s+/g, ' ')

  // Post-filter: if the cue contains forbidden tokens (digits, em/en
  // dashes, "joint angle", raw field names), fall through to silence.
  // No retry — a retry would push the cue past the next rally. Better
  // to say nothing than to say something wrong with a 4-second delay.
  if (LIVE_REJECT_RE.test(collapsed)) {
    console.warn('[live-coach] post-filter rejected:', collapsed.slice(0, 80))
    collapsed = ''
  }

  if (eventId) {
    const charCount = collapsed.length
    const tokenCount = outputTokens
    // Tip count is 0 when the model deliberately stayed silent (silence is
    // now a first-class response), 1 otherwise. used_baseline_template is
    // sourced from the prompt builder so the telemetry actually reflects
    // whether we routed this batch through the silence-by-default path.
    const tipCount = collapsed.length === 0 ? 0 : 1
    const baselineTemplateFlag = usedBaselineTemplate
    after(async () => {
      const { error } = await supabaseAdmin
        .from('analysis_events')
        .update({
          response_token_count: tokenCount,
          response_char_count: charCount,
          response_tip_count: tipCount,
          used_baseline_template: baselineTemplateFlag,
          llm_assessed_tier: coachedTier,
        })
        .eq('id', eventId)
      if (error) console.error('live-coach analysis_events update failed:', error)
    })
  }

  const headers: Record<string, string> = {
    'Content-Type': 'text/plain; charset=utf-8',
    'Cache-Control': 'no-cache',
    'X-Content-Type-Options': 'nosniff',
  }
  if (eventId) headers['X-Analysis-Event-Id'] = eventId
  // Tell the client the empty body was a deliberate silence, not a dropped
  // response. This is what lets useLiveCoach distinguish "model chose silence"
  // (no error UI) from "request failed" (show inline error).
  if (collapsed.length === 0) headers['X-Live-Coach-Silence'] = '1'

  return new NextResponse(collapsed, { headers })
}
