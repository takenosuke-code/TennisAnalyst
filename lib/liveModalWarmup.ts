'use client'

// Phase E warmup helper: keeps Modal's container warm so the *first*
// batch of every session doesn't pay the 5–10s cold-start tax. Modal's
// scaledown_window is 30s, so a heartbeat at 25s keeps the container
// alive with a 5s safety margin.
//
// Endpoint shape (E1 recommendation, docs/phase-e-spike-findings.md
// section 3, Option A): we POST a tiny static blob URL to /api/extract
// — same code path the live save flow uses — so the *exact* code path
// the first real swing will hit gets exercised, including provider
// init, model loading, and the cuDNN dlopen chain. The static asset is
// pinned via NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL so we can rotate without
// a code deploy. When the env var is unset, the warmup is a no-op (the
// dev/CI cases) — never throws, never blocks.
//
// The sessionId we send is a synthetic `warmup-<uuid>` value with no
// matching row in user_sessions; Railway's _run_extraction will
// supabase.update() against that id and silently update zero rows. A
// `warmup-` prefix makes the requests grep-able in Railway logs.
//
// Public contract:
//   - `pingModalWarmup()` is fire-and-forget. Resolves whether the ping
//     succeeded or failed — never throws to the caller.
//   - `startWarmupHeartbeat(intervalMs?)` returns a stop function that
//     cancels future intervals. The first ping is fired immediately so
//     the container starts warming the moment the player grants camera
//     permission, not 25s later.

const DEFAULT_HEARTBEAT_INTERVAL_MS = 25_000

const EXTRACT_ENDPOINT = '/api/extract'

function getWarmupBlobUrl(): string | null {
  const url = process.env.NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL
  return typeof url === 'string' && url.length > 0 ? url : null
}

function newWarmupSessionId(): string {
  // crypto.randomUUID is widely available in modern browsers and Node 19+.
  // Fall back to a Math.random tag in the rare environment that lacks it
  // — the warmup is best-effort, so a weaker id is fine.
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `warmup-${crypto.randomUUID()}`
  }
  return `warmup-${Math.random().toString(36).slice(2)}-${Date.now()}`
}

/**
 * Single fire-and-forget ping. Resolves on success or on graceful
 * failure (network error, 4xx, 5xx). The warmup must NEVER throw — the
 * caller is fire-and-forget and can't reasonably handle a rejection.
 *
 * No-op when NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL is unset (dev / CI).
 */
export async function pingModalWarmup(): Promise<void> {
  const blobUrl = getWarmupBlobUrl()
  if (!blobUrl) {
    // Best-effort, no-op when the warmup asset isn't configured. Logged
    // at info-level once per call so it's grep-able if a deploy is
    // missing the env var.
    console.info('[liveModalWarmup] NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL unset — skipping ping')
    return
  }
  const sessionId = newWarmupSessionId()
  try {
    const res = await fetch(EXTRACT_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ blobUrl, sessionId }),
      // keepalive lets the request finish even if the user navigates
      // away mid-warmup. Important on session start where the player
      // may stop and start within the same warmup window.
      keepalive: true,
    })
    if (!res.ok) {
      console.info(`[liveModalWarmup] ping returned HTTP ${res.status} (best-effort, ignored)`)
    }
  } catch (err) {
    console.info(
      '[liveModalWarmup] ping failed (best-effort, ignored):',
      err instanceof Error ? err.message : err,
    )
  }
}

/**
 * Start a periodic heartbeat that pings Modal at `intervalMs`. The
 * first ping fires immediately so the container starts warming up the
 * moment the player grants camera permission. Returns a stop function;
 * call it on session end / unmount.
 *
 * Default interval = 25_000 ms. Modal's scaledown_window is 30s, so
 * 25s gives a 5s safety margin.
 */
export function startWarmupHeartbeat(
  intervalMs: number = DEFAULT_HEARTBEAT_INTERVAL_MS,
): () => void {
  let stopped = false

  // Fire the first ping immediately. void to drop the promise — we
  // don't want to block heartbeat scheduling on the first request.
  void pingModalWarmup()

  const id = setInterval(() => {
    if (stopped) return
    void pingModalWarmup()
  }, intervalMs)

  return () => {
    stopped = true
    clearInterval(id)
  }
}
