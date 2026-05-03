import { NextRequest, NextResponse } from 'next/server'
import { put } from '@vercel/blob'
import { randomUUID } from 'node:crypto'
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
    startMs: clientStartMs,
    endMs: clientEndMs,
    shotType,
    label,
    keypointsJson: clientKeypoints,
  } = (body ?? {}) as {
    sessionId?: unknown
    startFrame?: unknown
    endFrame?: unknown
    peakFrame?: unknown
    startMs?: unknown
    endMs?: unknown
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
  if (typeof clientStartMs !== 'number' || !Number.isFinite(clientStartMs) || clientStartMs < 0) {
    return NextResponse.json({ error: 'startMs must be a non-negative number' }, { status: 400 })
  }
  if (typeof clientEndMs !== 'number' || !Number.isFinite(clientEndMs) || clientEndMs <= clientStartMs) {
    return NextResponse.json({ error: 'endMs must be a number greater than startMs' }, { status: 400 })
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

  // The trim window comes from the client (swing.startMs / swing.endMs
  // off the SwingSegment, which were derived from real frame
  // timestamps via allFrames[startFrame].timestamp_ms). We previously
  // recomputed startMs as (startFrame / fps) * 1000 which drifted on
  // 29.97fps captures and failed entirely when pose extraction's first
  // frame's timestamp wasn't 0 — both bugs desynced the trim AND the
  // re-zeroed pose overlay.
  //
  // Security: the client could lie about ms values to crop outside
  // their session's time window, so we validate the supplied values
  // against the parent session's actual keypoint timestamps. The
  // bounds use min/max defensively (not frames[0]/frames[last]) in
  // case the frames array isn't strictly time-monotonic — pose
  // extraction occasionally drops or reorders early low-confidence
  // frames.
  if (!isNonEmptyKeypoints(session.keypoints_json)) {
    return NextResponse.json(
      {
        error: 'Source session has no pose keypoints — cannot validate trim window. Re-upload the video and try again.',
      },
      { status: 400 },
    )
  }
  let sessionMinTs = Infinity
  let sessionMaxTs = -Infinity
  for (const f of session.keypoints_json.frames) {
    const ts = typeof f.timestamp_ms === 'number' ? f.timestamp_ms : NaN
    if (Number.isFinite(ts)) {
      if (ts < sessionMinTs) sessionMinTs = ts
      if (ts > sessionMaxTs) sessionMaxTs = ts
    }
  }
  if (!Number.isFinite(sessionMinTs) || !Number.isFinite(sessionMaxTs)) {
    return NextResponse.json(
      { error: 'Source session keypoints have no valid timestamps' },
      { status: 400 },
    )
  }

  // Allow a small slop on either side so swings that legitimately end
  // exactly at the last frame don't fail floating-point equality.
  const SLOP_MS = 50
  if (clientStartMs < sessionMinTs - SLOP_MS) {
    return NextResponse.json(
      { error: 'startMs is before the session timeline' },
      { status: 400 },
    )
  }
  if (clientEndMs > sessionMaxTs + SLOP_MS) {
    return NextResponse.json(
      { error: 'endMs is after the session timeline' },
      { status: 400 },
    )
  }
  if (clientEndMs - clientStartMs > 60_000) {
    // Mirrors railway-service/main.py:_TRIM_MAX_DURATION_MS so we
    // surface the cap from the Vercel side too — saves a Railway
    // round-trip on obvious overruns.
    return NextResponse.json(
      { error: 'Trim window must be at most 60 seconds' },
      { status: 400 },
    )
  }
  // Single source of truth for downstream: trim request body, re-zero
  // base, all derived from the same validated values.
  const startMs = clientStartMs
  const endMs = clientEndMs

  // Resolve keypoints. Prefer the client-supplied (already-sliced) set;
  // fall back to slicing the parent session's frames by frame range.
  // Either way we re-zero below so the trimmed video and the saved
  // pose overlay agree at currentTime=0. session.keypoints_json was
  // already validated non-empty by the timestamp-bounds check above.
  let baselineKeypoints: KeypointsJson = isNonEmptyKeypoints(clientKeypoints)
    ? (clientKeypoints as KeypointsJson)
    : sliceKeypointsByFrameRange(
        session.keypoints_json,
        startFrame,
        endFrame,
      )

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

  // Trim via Railway, then upload from THIS function via @vercel/blob.
  //
  // 2026-05 — Railway used to handle the upload itself, but its hand-
  // rolled Python httpx PUT to Vercel's Blob API kept 403'ing despite a
  // valid token. The JS SDK with the same token works fine (proven by
  // scripts/smoke-test-blob-write.ts). So we route around the busted
  // Python upload: Railway returns the trimmed mp4 bytes, this function
  // calls put() with the SDK that we know works.
  //
  // Railway clamps endMs to source duration via ffmpeg's `-t`, so a
  // Voronoi-padded swing overrunning the tail still produces a valid
  // clip.
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
      if (trimResp.status === 404) {
        return NextResponse.json(
          { error: 'Source video no longer available — re-upload the video and try again' },
          { status: 404 },
        )
      }
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
    // Stream the trimmed bytes from Railway directly into put(). Buffering
    // both upstream (fetch -> ArrayBuffer) and downstream (put) would
    // hold the full mp4 in memory twice; passing the ReadableStream
    // through lets @vercel/blob fetch consume it once.
    const blobPath = `baseline-trims/${randomUUID()}.mp4`
    const uploaded = await put(blobPath, trimResp.body, {
      access: 'public',
      contentType: 'video/mp4',
      // The pathname already contains a UUID, so suppress the random
      // suffix the SDK appends by default — keeps URLs predictable.
      addRandomSuffix: false,
    })
    trimmedUrl = uploaded.url
  } catch (err) {
    console.error('[baselines/from-swing] trim/upload pipeline error:', err)
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Trim service unreachable' },
      { status: 502 },
    )
  }

  if (!isAllowedBlobUrl(trimmedUrl)) {
    return NextResponse.json(
      { error: `Uploaded blob_url must be https on ${ALLOWED_BLOB_HOST_SUFFIX}` },
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
