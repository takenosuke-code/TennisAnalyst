import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import { supabaseAdmin } from '@/lib/supabase'
import type { KeypointsJson, PoseFrame } from '@/lib/supabase'

// POST /api/baselines/from-swing
//
// Body: {
//   sessionId: uuid,
//   startFrame: number,
//   endFrame: number,
//   peakFrame?: number,
//   fps: number,
//   shotType: ShotType,
//   label?: string,
//   keypointsJson?: KeypointsJson,  // pre-sliced by the client; server falls back to slicing the parent session if absent
// }
//
// Mirrors /api/baselines/from-segment but for client-detected swings
// (detectSwings output) that don't have a video_segments DB row. The
// long-standing bug fixed here: SwingBaselineGrid was POSTing to plain
// /api/baselines with the FULL session blob_url, so a 17-second rally
// landed as a 17-second "baseline" even when only one 3-second swing
// was selected. This endpoint trims the parent video to the swing
// window via Railway and re-zeroes the pose timestamps before
// inserting.

const VALID_SHOTS = ['forehand', 'backhand', 'serve', 'volley', 'slice'] as const
type ShotType = (typeof VALID_SHOTS)[number]

const ALLOWED_BLOB_HOST_SUFFIX = '.public.blob.vercel-storage.com'

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

function isNonEmptyKeypoints(k: unknown): k is KeypointsJson {
  if (!k || typeof k !== 'object') return false
  const frames = (k as { frames?: unknown }).frames
  return Array.isArray(frames) && frames.length > 0
}

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

// Re-zero frame indices and timestamps so the saved baseline's frames
// align to a video that starts at currentTime=0 instead of the original
// video's offset. Without this the pose overlay on /baseline/[id] looks
// dead — VideoCanvas matches `currentTime * 1000` against
// `frame.timestamp_ms`, and a swing that begins 14s into the source
// has frame[0].timestamp_ms ≈ 14000.
function rezeroFrames(frames: PoseFrame[], startMs: number): PoseFrame[] {
  return frames.map((f, i) => ({
    ...f,
    frame_index: i,
    timestamp_ms: Math.max(0, f.timestamp_ms - startMs),
  }))
}

type SessionRow = {
  id: string
  blob_url: string
  keypoints_json: KeypointsJson | null
}

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

  const {
    sessionId,
    startFrame,
    endFrame,
    fps,
    shotType,
    label,
    keypointsJson: clientKeypoints,
  } = (body ?? {}) as {
    sessionId?: unknown
    startFrame?: unknown
    endFrame?: unknown
    peakFrame?: unknown
    fps?: unknown
    shotType?: unknown
    label?: unknown
    keypointsJson?: unknown
  }

  if (typeof sessionId !== 'string' || !/^[0-9a-f-]{36}$/i.test(sessionId)) {
    return NextResponse.json({ error: 'sessionId (uuid) is required' }, { status: 400 })
  }
  if (typeof startFrame !== 'number' || !Number.isFinite(startFrame) || startFrame < 0) {
    return NextResponse.json({ error: 'startFrame must be a non-negative number' }, { status: 400 })
  }
  if (typeof endFrame !== 'number' || !Number.isFinite(endFrame) || endFrame <= startFrame) {
    return NextResponse.json({ error: 'endFrame must be a number greater than startFrame' }, { status: 400 })
  }
  if (typeof fps !== 'number' || !Number.isFinite(fps) || fps <= 0) {
    return NextResponse.json({ error: 'fps must be a positive number' }, { status: 400 })
  }
  if (!isShotType(shotType)) {
    return NextResponse.json(
      { error: `shotType must be one of ${VALID_SHOTS.join(', ')}` },
      { status: 400 },
    )
  }

  if (!RAILWAY_SERVICE_URL || !EXTRACT_API_KEY) {
    console.error('[baselines/from-swing] RAILWAY_SERVICE_URL or EXTRACT_API_KEY unset')
    return NextResponse.json(
      { error: 'Trim service not configured on this deployment' },
      { status: 503 },
    )
  }

  // Source session lookup. user_sessions has no user_id column today
  // (capability-based via the UUID), so we use service-role to fetch
  // blob_url + parent keypoints. Same pattern as from-segment.
  const { data: sessionRaw, error: sessErr } = await supabaseAdmin
    .from('user_sessions')
    .select('id, blob_url, keypoints_json')
    .eq('id', sessionId)
    .single()

  if (sessErr || !sessionRaw) {
    // Edge case: source session was deleted between the analyze flow
    // and the save click (e.g. cleanup-blobs script ran, or the row
    // expired off the 24h session TTL). Surface this clearly so the
    // client can prompt the user to re-upload.
    return NextResponse.json(
      { error: 'Source session no longer available — re-upload the video and try again' },
      { status: 404 },
    )
  }
  const session = sessionRaw as SessionRow

  // Compute the swing's time window from frame indices + fps. This is
  // the canonical source of truth — relying on a client-supplied
  // start_ms/end_ms would let a malicious client trim outside the
  // swing window or even crop into someone else's session.
  const startMs = (startFrame / fps) * 1000
  const endMs = (endFrame / fps) * 1000
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) {
    return NextResponse.json({ error: 'Computed time window is invalid' }, { status: 400 })
  }

  // Resolve keypoints. Prefer the client-supplied (already-sliced) set;
  // fall back to slicing the parent session's frames by frame range.
  // Either way we re-zero below so the trimmed video and the saved
  // pose overlay agree at currentTime=0.
  let baselineKeypoints: KeypointsJson | null = isNonEmptyKeypoints(clientKeypoints)
    ? (clientKeypoints as KeypointsJson)
    : null
  if (!baselineKeypoints) {
    if (!isNonEmptyKeypoints(session.keypoints_json)) {
      return NextResponse.json(
        { error: 'No keypoints available for this swing' },
        { status: 400 },
      )
    }
    baselineKeypoints = sliceKeypointsByFrameRange(
      session.keypoints_json,
      startFrame,
      endFrame,
    )
  }

  if (!baselineKeypoints.frames?.length) {
    return NextResponse.json(
      { error: 'No keypoints available for this swing' },
      { status: 400 },
    )
  }

  // Re-zero AFTER slicing so the timestamps inside the saved row are
  // 0-relative to the trimmed video. Use the canonical startMs we just
  // computed, not anything from the client's keypoints array.
  baselineKeypoints = {
    ...baselineKeypoints,
    frame_count: baselineKeypoints.frames.length,
    frames: rezeroFrames(baselineKeypoints.frames, startMs),
  }

  // Size guard — same threshold as from-segment.
  let serializedBytes = 0
  try {
    serializedBytes = JSON.stringify(baselineKeypoints).length
  } catch {
    return NextResponse.json({ error: 'keypointsJson not serializable' }, { status: 400 })
  }
  if (serializedBytes > MAX_KEYPOINTS_BYTES) {
    return NextResponse.json(
      { error: `keypointsJson exceeds ${MAX_KEYPOINTS_BYTES} byte limit` },
      { status: 400 },
    )
  }

  // Trim the parent video to [startMs, endMs] via Railway. Railway
  // clamps endMs to the source duration internally (FFmpeg's `-t` won't
  // run past EOF), so a Voronoi-padded swing that overruns the video
  // tail still produces a valid clip. If the source is already trimmed
  // (e.g. the user re-uploaded a short clip), the trim is a near-no-op
  // and works fine.
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
        start_ms: Math.round(startMs),
        end_ms: Math.round(endMs),
      }),
    })
    if (!trimResp.ok) {
      const text = await trimResp.text().catch(() => '')
      console.error('[baselines/from-swing] trim service failed:', trimResp.status, text)
      // 404 from Railway typically means the source video was deleted
      // before the trim ran. Surface that as a 404 to the client so
      // the UI can guide the user to re-upload.
      if (trimResp.status === 404) {
        return NextResponse.json(
          { error: 'Source video no longer available — re-upload the video and try again' },
          { status: 404 },
        )
      }
      return NextResponse.json(
        { error: `Trim service returned ${trimResp.status}` },
        { status: 502 },
      )
    }
    const data = (await trimResp.json()) as { blob_url?: unknown }
    if (typeof data.blob_url !== 'string' || !data.blob_url) {
      return NextResponse.json({ error: 'Trim service returned no blob_url' }, { status: 502 })
    }
    trimmedUrl = data.blob_url
  } catch (err) {
    console.error('[baselines/from-swing] trim service error:', err)
    return NextResponse.json({ error: 'Trim service unreachable' }, { status: 502 })
  }

  if (!isAllowedBlobUrl(trimmedUrl)) {
    return NextResponse.json(
      { error: `Trim service blob_url must be https on ${ALLOWED_BLOB_HOST_SUFFIX}` },
      { status: 502 },
    )
  }

  // Deactivate any existing active baseline of the same shot_type.
  const now = new Date().toISOString()
  const { error: deactivateErr } = await authClient
    .from('user_baselines')
    .update({ is_active: false, replaced_at: now })
    .eq('shot_type', shotType)
    .eq('is_active', true)

  if (deactivateErr) {
    return NextResponse.json({ error: deactivateErr.message }, { status: 500 })
  }

  const finalLabel =
    typeof label === 'string' && label.trim()
      ? label.trim().slice(0, 120)
      : `Swing — ${shotType}`

  const { data, error } = await authClient
    .from('user_baselines')
    .insert({
      user_id: user.id,
      label: finalLabel,
      shot_type: shotType,
      blob_url: trimmedUrl,
      keypoints_json: baselineKeypoints,
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
