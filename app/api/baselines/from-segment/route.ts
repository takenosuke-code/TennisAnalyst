import { NextRequest, NextResponse } from 'next/server'
import { put } from '@vercel/blob'
import { randomUUID } from 'node:crypto'
import { createClient } from '@/lib/supabase/server'
import { supabaseAdmin } from '@/lib/supabase'
import type { KeypointsJson } from '@/lib/supabase'

// Valid shot_type values for the baselines table. Matches the CHECK
// constraint in lib/db/004_user_baselines.sql and mirrors the set used
// by POST /api/baselines.
const VALID_SHOTS = ['forehand', 'backhand', 'serve', 'volley', 'slice'] as const
type ShotType = (typeof VALID_SHOTS)[number]

// Mirrors app/api/baselines/route.ts so every baseline row points at a
// Vercel Blob URL -- see the comment there for the threat model.
const ALLOWED_BLOB_HOST_SUFFIX = '.public.blob.vercel-storage.com'

// Rough upper bound on serialized keypoints_json. Chosen to be generous
// for any realistic single-segment clip while preventing a pathologically
// large blob from landing in the baselines table.
const MAX_KEYPOINTS_BYTES = 4 * 1024 * 1024

const RAILWAY_SERVICE_URL = process.env.RAILWAY_SERVICE_URL
const EXTRACT_API_KEY = process.env.EXTRACT_API_KEY

function isShotType(v: unknown): v is ShotType {
  return typeof v === 'string' && (VALID_SHOTS as readonly string[]).includes(v)
}

function isAllowedBlobUrl(raw: string): boolean {
  try {
    const u = new URL(raw)
    return u.protocol === 'https:' && u.hostname.endsWith(ALLOWED_BLOB_HOST_SUFFIX)
  } catch {
    return false
  }
}

// Slice a keypoints json by frame index range. Segments persist their
// own keypoints in the happy path; this fallback is only used when the
// segment row's keypoints_json is empty/missing, in which case we carve
// out the parent session's frames by [start_frame, end_frame].
function sliceKeypointsByFrameRange(
  parent: KeypointsJson,
  startFrame: number,
  endFrame: number,
): KeypointsJson {
  const frames = parent.frames.filter((f) => {
    const idx = f.frame_index
    return typeof idx === 'number' && idx >= startFrame && idx <= endFrame
  })
  return {
    fps_sampled: parent.fps_sampled,
    frame_count: frames.length,
    frames,
    schema_version: parent.schema_version,
  }
}

function isNonEmptyKeypoints(k: unknown): k is KeypointsJson {
  if (!k || typeof k !== 'object') return false
  const frames = (k as { frames?: unknown }).frames
  return Array.isArray(frames) && frames.length > 0
}

type SegmentRow = {
  id: string
  session_id: string
  segment_index: number
  shot_type: string
  start_frame: number
  end_frame: number
  start_ms: number
  end_ms: number
  confidence: number | null
  label: string | null
  keypoints_json: KeypointsJson | null
}

type SessionRow = {
  id: string
  blob_url: string
  keypoints_json: KeypointsJson | null
}

// POST /api/baselines/from-segment
//
// Body: { sessionId, segmentId, label?, shotTypeOverride? }
//
// Trims the parent video to the segment's [start_ms, end_ms] via Railway
// /trim-video, slices the corresponding keypoints, and inserts a new
// baseline row (deactivating any sibling of the same shot_type first).
// Returns the new baseline row.
export async function POST(request: NextRequest) {
  const authClient = await createClient()
  const { data: { user } } = await authClient.auth.getUser()
  if (!user) {
    return NextResponse.json({ error: 'Not signed in' }, { status: 401 })
  }

  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const { sessionId, segmentId, label, shotTypeOverride } = (body ?? {}) as {
    sessionId?: unknown
    segmentId?: unknown
    label?: unknown
    shotTypeOverride?: unknown
  }

  if (typeof sessionId !== 'string' || !/^[0-9a-f-]{36}$/i.test(sessionId)) {
    return NextResponse.json({ error: 'sessionId (uuid) is required' }, { status: 400 })
  }
  if (typeof segmentId !== 'string' || !/^[0-9a-f-]{36}$/i.test(segmentId)) {
    return NextResponse.json({ error: 'segmentId (uuid) is required' }, { status: 400 })
  }
  if (shotTypeOverride !== undefined && !isShotType(shotTypeOverride)) {
    return NextResponse.json(
      { error: `shotTypeOverride must be one of ${VALID_SHOTS.join(', ')}` },
      { status: 400 },
    )
  }

  if (!RAILWAY_SERVICE_URL || !EXTRACT_API_KEY) {
    console.error('[baselines/from-segment] RAILWAY_SERVICE_URL or EXTRACT_API_KEY unset')
    return NextResponse.json(
      { error: 'Trim service not configured on this deployment' },
      { status: 503 },
    )
  }

  // user_sessions has no user_id column today (capability-based via UUID)
  // so we use the service-role client to fetch the blob_url + parent
  // keypoints. RLS already lets anon SELECT, but service-role is more
  // explicit about the intent.
  const { data: sessionRaw, error: sessErr } = await supabaseAdmin
    .from('user_sessions')
    .select('id, blob_url, keypoints_json')
    .eq('id', sessionId)
    .single()

  if (sessErr || !sessionRaw) {
    return NextResponse.json({ error: 'Session not found' }, { status: 404 })
  }
  const session = sessionRaw as SessionRow

  const { data: segmentRaw, error: segErr } = await supabaseAdmin
    .from('video_segments')
    .select(
      'id, session_id, segment_index, shot_type, start_frame, end_frame, start_ms, end_ms, confidence, label, keypoints_json',
    )
    .eq('id', segmentId)
    .eq('session_id', sessionId)
    .single()

  if (segErr || !segmentRaw) {
    return NextResponse.json({ error: 'Segment not found' }, { status: 404 })
  }
  const segment = segmentRaw as SegmentRow

  // Pick the shot_type that ends up on the baseline row. Override beats
  // the classifier. The classifier can also emit 'unknown' / 'idle' which
  // aren't valid baseline shots; fall through to 400 in that case.
  const resolvedShotType: ShotType | null = isShotType(shotTypeOverride)
    ? shotTypeOverride
    : isShotType(segment.shot_type)
      ? segment.shot_type
      : null
  if (!resolvedShotType) {
    return NextResponse.json(
      {
        error: `Segment shot_type is '${segment.shot_type}' which is not baselinable; pass shotTypeOverride to choose one of ${VALID_SHOTS.join(', ')}`,
      },
      { status: 400 },
    )
  }

  // Prefer the segment's own keypoints (already scoped by the Railway
  // classifier). Fall back to slicing the parent session's keypoints by
  // frame range so the user never gets a baseline with an empty frame
  // list.
  let keypointsJson: KeypointsJson | null = isNonEmptyKeypoints(segment.keypoints_json)
    ? segment.keypoints_json
    : null
  if (!keypointsJson) {
    if (!isNonEmptyKeypoints(session.keypoints_json)) {
      return NextResponse.json(
        { error: 'No keypoints available for this segment' },
        { status: 400 },
      )
    }
    keypointsJson = sliceKeypointsByFrameRange(
      session.keypoints_json,
      segment.start_frame,
      segment.end_frame,
    )
  }

  if (!keypointsJson.frames?.length) {
    return NextResponse.json(
      { error: 'No keypoints available for this segment' },
      { status: 400 },
    )
  }

  // Guard against a pathologically huge keypoints payload landing in the
  // row. 4MB serialized is comfortably above any realistic single-segment
  // clip.
  let serializedBytes = 0
  try {
    serializedBytes = JSON.stringify(keypointsJson).length
  } catch {
    return NextResponse.json({ error: 'keypointsJson not serializable' }, { status: 400 })
  }
  if (serializedBytes > MAX_KEYPOINTS_BYTES) {
    return NextResponse.json(
      { error: `keypointsJson exceeds ${MAX_KEYPOINTS_BYTES} byte limit` },
      { status: 400 },
    )
  }

  // Trim via Railway, then upload from THIS function via @vercel/blob.
  // See the matching comment in /api/baselines/from-swing/route.ts for
  // why Railway no longer uploads itself.
  let trimmedUrl: string
  try {
    const trimResp = await fetch(`${RAILWAY_SERVICE_URL}/trim-video`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${EXTRACT_API_KEY}`,
      },
      body: JSON.stringify({
        video_url: session.blob_url,
        start_ms: segment.start_ms,
        end_ms: segment.end_ms,
      }),
    })
    if (!trimResp.ok) {
      const text = await trimResp.text().catch(() => '')
      console.error('[baselines/from-segment] trim service failed:', trimResp.status, text)
      const detail = text ? text.slice(0, 500) : ''
      return NextResponse.json(
        {
          error: `Trim service returned ${trimResp.status}${detail ? `: ${detail}` : ''}`,
          railwayStatus: trimResp.status,
          railwayDetail: detail || null,
        },
        { status: 502 },
      )
    }
    if (!trimResp.body) {
      return NextResponse.json(
        { error: 'Trim service returned an empty body' },
        { status: 502 },
      )
    }
    const blobPath = `baseline-trims/${randomUUID()}.mp4`
    const uploaded = await put(blobPath, trimResp.body, {
      access: 'public',
      contentType: 'video/mp4',
      addRandomSuffix: false,
    })
    trimmedUrl = uploaded.url
  } catch (err) {
    console.error('[baselines/from-segment] trim/upload pipeline error:', err)
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Trim service unreachable' },
      { status: 502 },
    )
  }

  if (!isAllowedBlobUrl(trimmedUrl)) {
    return NextResponse.json(
      { error: `Trim service blob_url must be https on ${ALLOWED_BLOB_HOST_SUFFIX}` },
      { status: 502 },
    )
  }

  // Flip the existing active baseline (if any) of the matching shot_type
  // to inactive. RLS scopes this to the current user's rows. Serial
  // rather than transactional -- single-user race is not a real concern.
  const now = new Date().toISOString()
  const { error: deactivateErr } = await authClient
    .from('user_baselines')
    .update({ is_active: false, replaced_at: now })
    .eq('shot_type', resolvedShotType)
    .eq('is_active', true)

  if (deactivateErr) {
    return NextResponse.json({ error: deactivateErr.message }, { status: 500 })
  }

  const defaultLabel = `Segment #${segment.segment_index + 1} — ${resolvedShotType}`
  const finalLabel =
    typeof label === 'string' && label.trim()
      ? label.trim().slice(0, 120)
      : segment.label?.trim()
        ? segment.label.trim().slice(0, 120)
        : defaultLabel

  const { data, error } = await authClient
    .from('user_baselines')
    .insert({
      user_id: user.id,
      label: finalLabel,
      shot_type: resolvedShotType,
      blob_url: trimmedUrl,
      keypoints_json: keypointsJson,
      source_session_id: sessionId,
      is_active: true,
    })
    .select()
    .single()

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({ baseline: data })
}
