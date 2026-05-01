import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import { supabaseAdmin } from '@/lib/supabase'
import {
  getCachedPose,
  setCachedPose,
  poseJsonToKeypointsJson,
  type ModalExtractResponse,
} from '@/lib/poseCache'

// POST /api/extract — proxy that asks Railway (or Modal directly) to run
// server-side pose extraction on an already-uploaded Vercel Blob.
//
// Two modes:
//
// 1. Legacy (default — `mode` omitted or any value other than
//    `'batch-ephemeral'`):
//    Body:    { sessionId: string, blobUrl: string }
//    Forwards to Railway's `/extract` with Bearer auth. Fire-and-poll —
//    Railway acks with `{ status: 'queued' }` and writes keypoints back
//    to the user_sessions row identified by sessionId. Used by the
//    finished-recording / `/analyze` flow.
//
// 2. `'batch-ephemeral'` (Phase E):
//    Body:    { mode: 'batch-ephemeral', blobUrl: string }
//    Skips Railway entirely. Calls MODAL_INFERENCE_URL directly,
//    synchronously, and returns the Modal keypoints inline. No
//    sessionId is required — the caller (lib/liveSwingBatchExtractor)
//    keeps the keypoints in-memory for the live coach. No DB row is
//    written.
//
// Responses:
//   200 { status: 'queued', ... }            — legacy: Railway accepted
//   200 { frames, fps_sampled, pose_backend, ... }
//                                            — batch-ephemeral: Modal
//                                              returned keypoints inline
//   400 { error: '...' }                     — bad request body
//   502 { error: 'railway-error' | 'modal-error', ... }
//                                            — upstream returned non-2xx
//   503 { error: 'railway-not-configured' }  — legacy env unset; client
//                                              should fall back to browser
//   503 { error: 'modal-not-configured' }    — batch-ephemeral env unset;
//                                              live-batch flow falls back
//                                              gracefully to on-device-only
//                                              frames
//   503 { error: 'railway-unreachable' | 'modal-unreachable' }
//                                            — network error, fall back
//
// EXTRACT_API_KEY only lives on this server route — never sent to the
// browser. MODAL_INFERENCE_URL is the same Modal endpoint Railway already
// uses; for the batch-ephemeral path it must ALSO be set in Vercel env so
// the route can call Modal directly without a Railway hop. If it's unset,
// the live-batch flow falls back gracefully to on-device-only frames.
//
// Pattern mirrors app/api/debug/racket/route.ts and
// app/api/process-youtube/route.ts.

const RAILWAY_SERVICE_URL = process.env.RAILWAY_SERVICE_URL
const EXTRACT_API_KEY = process.env.EXTRACT_API_KEY
const MODAL_INFERENCE_URL = process.env.MODAL_INFERENCE_URL

// Identifies the extractor + weights stamped on the cache entry. Bumped
// when we swap models — old rows whose model_version doesn't match are
// treated as misses on read, so the cache transparently invalidates
// without a manual purge. Falls back to 'rtmpose-modal' when the env
// is unset so dev/staging still produces consistent rows.
//
// Read at request time (not module init) so a Vercel env update flips
// the value on the next cold start without a redeploy.
function currentModelVersion(): string {
  return process.env.POSE_CACHE_MODEL_VERSION || 'rtmpose-modal'
}

// Frame-shape gate before we write Modal output to the pose cache. The
// cache key is sha256(video bytes), so any garbage we memoize would be
// served forever for the same upload — we'd rather pay re-extract on
// the next request than poison the row.
type FrameValidation =
  | { ok: true }
  | { ok: false; reason: string; frameIndex?: number }

function validateFrames(frames: unknown[]): FrameValidation {
  if (frames.length === 0) {
    return { ok: false, reason: 'frames array is empty' }
  }
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]
    if (!f || typeof f !== 'object') {
      return { ok: false, reason: 'frame is not an object', frameIndex: i }
    }
    const obj = f as Record<string, unknown>
    if (typeof obj.frame_index !== 'number') {
      return { ok: false, reason: 'frame_index missing or not a number', frameIndex: i }
    }
    if (typeof obj.timestamp_ms !== 'number') {
      return { ok: false, reason: 'timestamp_ms missing or not a number', frameIndex: i }
    }
    if (!Array.isArray(obj.landmarks)) {
      return { ok: false, reason: 'landmarks missing or not an array', frameIndex: i }
    }
  }
  return { ok: true }
}

// Synchronous Modal call budget for the batch-ephemeral path:
//   ~5-10s cold-start  + ~3-5s blob download (Modal-side)
// + ~3-5s inference    + ~2s overhead
// ≈ ~25s worst case. 30s gives a 5s margin. Vercel free-tier max is 60s.
// Legacy fire-and-poll path doesn't need this — Railway acks in <1s — but
// the same ceiling is harmless.
export const maxDuration = 30

export async function POST(request: NextRequest) {
  let body
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const { mode, sessionId, blobUrl, sha256 } = body ?? {}

  if (mode === 'batch-ephemeral') {
    return handleBatchEphemeral(blobUrl, sha256)
  }

  // Legacy path: sessionId + blobUrl forwarded to Railway. Behavior
  // unchanged from pre-Phase-E.
  if (!RAILWAY_SERVICE_URL || !EXTRACT_API_KEY) {
    return NextResponse.json(
      { error: 'railway-not-configured' },
      { status: 503 }
    )
  }

  if (!sessionId || !blobUrl) {
    return NextResponse.json(
      { error: 'sessionId and blobUrl are required' },
      { status: 400 }
    )
  }

  // Phase E: content-hash cache. When the client supplied a sha256 of
  // the file bytes and we already have keypoints for that hash, write
  // them straight into the user_sessions row and short-circuit the
  // Railway hop. The client polls /api/sessions/[id]?include=keypoints
  // exactly the same way it does for a real Railway run — it sees
  // status='complete' on the next tick and reads keypoints_json. No
  // contract change for the caller.
  //
  // Sha256 is optional: callers that don't pass it (older clients,
  // server-to-server) just bypass the cache and run the full pipeline.
  if (typeof sha256 === 'string' && sha256.length === 64) {
    try {
      const cached = await getCachedPose(sha256, currentModelVersion())
      if (cached) {
        // The cached value is now the full Modal response shape
        // (includes optional video_fps, duration_ms, timing,
        // pose_backend). user_sessions.keypoints_json is typed as the
        // narrower KeypointsJson — project down to that shape so we
        // don't store unexpected keys in a column other consumers
        // might assume is shape-stable.
        const keypointsForSession = poseJsonToKeypointsJson(cached)
        const { error: updErr } = await supabaseAdmin
          .from('user_sessions')
          .update({
            status: 'complete',
            keypoints_json: keypointsForSession,
          })
          .eq('id', sessionId)
          .eq('blob_url', blobUrl) // ownership check; matches /api/sessions
        if (updErr) {
          // Cache hit but session row write failed — log and fall
          // through to the Railway path. Don't block the user on a
          // cache anomaly.
          console.error(
            '[extract] cache-hit session update failed, falling back to Railway:',
            updErr.message,
            { sessionId, sha256 },
          )
        } else {
          return NextResponse.json({ status: 'cache-hit', sessionId })
        }
      }
    } catch (err) {
      // Any cache-lookup error: continue with the live extraction. The
      // user shouldn't see a cache outage as an upload failure.
      console.error('[extract] cache lookup threw, continuing:', err)
    }
  }

  try {
    const resp = await fetch(`${RAILWAY_SERVICE_URL}/extract`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${EXTRACT_API_KEY}`,
      },
      body: JSON.stringify({
        video_url: blobUrl,
        session_id: sessionId,
      }),
    })

    const respBody = await resp.text()
    let parsed: unknown
    try {
      parsed = JSON.parse(respBody)
    } catch {
      parsed = { raw: respBody }
    }

    if (!resp.ok) {
      console.error('[extract] Railway rejected job', resp.status, parsed)
      return NextResponse.json(
        { error: 'railway-error', status: resp.status, body: parsed },
        { status: 502 }
      )
    }

    return NextResponse.json(parsed)
  } catch (err) {
    // Network error reaching Railway. Treat as a 503 so the client
    // gets the same "fall back to browser" signal as when the service
    // isn't configured.
    console.error('[extract] Railway network error', err)
    return NextResponse.json(
      { error: 'railway-unreachable' },
      { status: 503 }
    )
  }
}

// -----------------------------------------------------------------------
// Phase E: batch-ephemeral path.
//
// Live-coach swing-batch extractor sends a short (~10-15s) sub-clip
// blob URL and wants keypoints back inline. We POST directly to Modal's
// `extract_pose` HTTPS endpoint — no Railway hop, no DB write, no
// session row.
//
// Modal's `extract_pose` (railway-service/modal_inference.py:251-386)
// returns a JSON object shaped like:
//   {
//     fps_sampled: number,
//     frame_count: number,
//     frames: PoseFrame[],
//     video_fps: number,
//     duration_ms: number,
//     schema_version: 3,
//     pose_backend: 'rtmpose-modal-cuda',
//     timing: { ... },
//   }
// We pass the whole object through. The client (liveSwingBatchExtractor)
// reads `frames` directly; `fps_sampled` and `pose_backend` are surfaced
// for telemetry / the diagnostic chip.
//
// Auth: Modal endpoints are public HTTPS. Security is the URL allowlist
// inside `extract_pose` (`_is_url_allowed`) which only accepts Vercel
// Blob / Supabase Storage / GCS hosts — i.e. URLs only our own upload
// pipeline mints. No Bearer header is required.
// -----------------------------------------------------------------------
async function handleBatchEphemeral(
  blobUrl: unknown,
  sha256: unknown,
): Promise<NextResponse> {
  // Auth gate: middleware excludes /api/, so this endpoint is otherwise
  // open to anyone with a public Vercel Blob URL — which would let
  // unauthenticated callers burn T4 GPU credits on Modal indefinitely.
  // The legacy path uses Bearer auth via Railway, but batch-ephemeral
  // skips Railway, so we enforce a Supabase session here directly.
  const authClient = await createClient()
  const { data: userData } = await authClient.auth
    .getUser()
    .catch(() => ({ data: { user: null } }))
  if (!userData?.user) {
    return NextResponse.json({ error: 'Not signed in' }, { status: 401 })
  }

  if (!MODAL_INFERENCE_URL) {
    return NextResponse.json(
      { error: 'modal-not-configured' },
      { status: 503 }
    )
  }

  if (!blobUrl || typeof blobUrl !== 'string') {
    return NextResponse.json(
      { error: 'blobUrl is required' },
      { status: 400 }
    )
  }

  // Phase E: cache hit on the batch-ephemeral path returns the stored
  // keypoints inline. The cached object now holds the full Modal
  // response (frames, fps_sampled, frame_count, schema_version,
  // video_fps, duration_ms, pose_backend, timing) so it round-trips
  // byte-for-byte to the client — no field-by-field reshaping that
  // could drop optional metadata the client relies on for telemetry.
  // When sha256 is missing or invalid we just skip the lookup.
  const modelVersion = currentModelVersion()
  if (typeof sha256 === 'string' && sha256.length === 64) {
    try {
      const cached = await getCachedPose(sha256, modelVersion)
      if (cached) {
        // Spread the cached response as-is, then stamp `cached: true`
        // so the client (liveSwingBatchExtractor + similar) can
        // distinguish a hit from a miss for telemetry. Don't override
        // pose_backend on hit — the cached value already records which
        // backend produced these keypoints; rewriting it to
        // 'pose-cache' would lose that signal.
        return NextResponse.json({
          ...cached,
          cached: true,
        })
      }
    } catch (err) {
      console.error('[extract] batch-ephemeral cache lookup threw:', err)
    }
  }

  try {
    const resp = await fetch(MODAL_INFERENCE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_url: blobUrl,
        sample_fps: 30,
      }),
    })

    const respBody = await resp.text()
    let parsed: unknown
    try {
      parsed = JSON.parse(respBody)
    } catch {
      parsed = { raw: respBody }
    }

    if (!resp.ok) {
      console.error('[extract] Modal rejected batch-ephemeral', resp.status, parsed)
      return NextResponse.json(
        { error: 'modal-error', status: resp.status, body: parsed },
        { status: 502 }
      )
    }

    // Coerce to the shape the client expects. Modal already returns the
    // canonical shape (see comment above); this just narrows the type
    // and ensures `frames` is an array even if Modal returned something
    // unexpected.
    const m = (parsed && typeof parsed === 'object' ? parsed : {}) as Record<string, unknown>
    const frames = Array.isArray(m.frames) ? m.frames : []
    const fps_sampled = typeof m.fps_sampled === 'number' ? m.fps_sampled : 30
    const pose_backend = typeof m.pose_backend === 'string' ? m.pose_backend : 'rtmpose-modal'

    // Populate the cache for next time. Race-safe: setCachedPose now
    // uses ON CONFLICT (sha256) DO UPDATE — concurrent extractions
    // of the same hash both succeed; the later write overwrites the
    // earlier with byte-equivalent payload (same model, same bytes).
    // We swallow errors here — a cache write failure shouldn't
    // propagate to the user.
    //
    // Frame validation gate: a future Modal regression that returns
    // malformed frames would otherwise poison the cache with a
    // permanent garbage row (the hash never changes for the same
    // bytes, so we'd serve the bad row forever). validateFrames()
    // checks each frame's required fields; on failure we LOG, SKIP
    // the cache write, but still return the response to the client
    // — the user already has their result; we just don't memoize
    // garbage.
    if (typeof sha256 === 'string' && sha256.length === 64) {
      const validation = validateFrames(frames)
      if (!validation.ok) {
        console.error(
          '[extract] Modal returned malformed frames; skipping cache write:',
          validation.reason,
          { sha256, frameIndex: validation.frameIndex },
        )
      } else {
        // Store the FULL Modal response (not just the KeypointsJson
        // subset). Cache hits round-trip the entire response so the
        // client sees the same shape regardless of hit/miss, including
        // optional video_fps, duration_ms, timing, pose_backend.
        const poseJson: ModalExtractResponse = {
          ...m,
          fps_sampled,
          frame_count:
            typeof m.frame_count === 'number' ? m.frame_count : frames.length,
          // Frames have been validated above; safe to assert the type.
          frames: frames as ModalExtractResponse['frames'],
          schema_version:
            (m.schema_version as ModalExtractResponse['schema_version']) ?? 3,
          pose_backend,
        }
        try {
          await setCachedPose(sha256, poseJson, modelVersion)
        } catch (cacheErr) {
          console.error('[extract] setCachedPose failed (non-fatal):', cacheErr)
        }
      }
    }

    return NextResponse.json({
      ...m,
      frames,
      fps_sampled,
      pose_backend,
    })
  } catch (err) {
    // Network error reaching Modal. Treat as 503 so the client falls
    // back to on-device-only frames the same way it does when the env
    // is unset.
    console.error('[extract] Modal network error', err)
    return NextResponse.json(
      { error: 'modal-unreachable' },
      { status: 503 }
    )
  }
}
