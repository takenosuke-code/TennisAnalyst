import { NextRequest, NextResponse } from 'next/server'

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

  const { mode, sessionId, blobUrl } = body ?? {}

  if (mode === 'batch-ephemeral') {
    return handleBatchEphemeral(blobUrl)
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
async function handleBatchEphemeral(blobUrl: unknown): Promise<NextResponse> {
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

  try {
    const resp = await fetch(MODAL_INFERENCE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_url: blobUrl,
        // 15fps is enough for tennis swing analysis: contact lasts ~5ms
        // but the swing envelope is ~1s, so half-frame timing precision
        // (33ms vs 16ms) doesn't change phase detection. Halving the
        // sample rate cuts server-side extraction roughly in half.
        sample_fps: 15,
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
