// Server-side content-hash cache for pose extraction.
//
// Backed by lib/db/012_pose_cache.sql (table `pose_cache`).
//
// Contract:
//   getCachedPose(sha256, modelVersion?) — returns the cached
//     ModalExtractResponse for the given byte hash, or null on miss.
//     When `modelVersion` is provided, a stored row whose model_version
//     doesn't match is treated as a miss (so model upgrades naturally
//     invalidate stale entries without manual cache busts).
//
//   setCachedPose(sha256, poseJson, modelVersion) — writes the cache
//     entry. Uses INSERT ... ON CONFLICT (sha256) DO UPDATE so a row
//     written by an older `model_version` is *replaced* when a newer
//     extraction lands. Without that, after a model upgrade every
//     prior hash would re-extract on every upload (read-side miss
//     because model_version differs, write-side no-op because PK
//     already exists) — i.e. a permanent cache bypass for legacy
//     content. Concurrent writes for the SAME (sha256, model_version)
//     converge to one winner — both payloads are byte-equivalent so
//     "second writer wins" is fine. Concurrent writes with DIFFERENT
//     model_versions: latest writer wins, which is also fine because
//     either model is acceptable to serve.
//
//   Never throws on conflict; throws on real DB errors (network, RLS,
//   schema mismatch) so the caller decides whether to swallow or
//   surface.
//
// Why this lives in lib/ and not app/api/:
// Two API routes need it (today: /api/extract; tomorrow: any
// re-extraction trigger we add). Keeping the cache logic in a single
// file with one well-tested contract avoids drift between callers.

import type { KeypointsJson, PoseFrame } from '@/lib/supabase'
import { supabaseAdmin } from '@/lib/supabase'

// Full Modal `/extract_pose` response shape. Superset of KeypointsJson
// with optional fields we want to preserve through the cache so that
// cache-hit responses match cache-miss responses byte-for-byte to the
// client (frame_count, schema_version, video_fps, duration_ms, timing,
// pose_backend). Stored as the entire jsonb pose_json column.
//
// `frames` is the canonical PoseFrame[]; the rest are optional because
// older Modal builds don't populate them and the client treats them as
// soft signals (telemetry, diagnostic chip).
export type ModalExtractResponse = {
  fps_sampled: number
  frame_count: number
  frames: PoseFrame[]
  schema_version?: 1 | 2 | 3
  video_fps?: number
  duration_ms?: number
  pose_backend?: string
  timing?: Record<string, number>
  // Allow forward-compat fields without forcing a type bump every time
  // Modal grows the response.
  [key: string]: unknown
}

// Re-exported alias matching the task description's vocabulary. Now
// the canonical Modal-response shape rather than the narrower
// KeypointsJson, so cache hits round-trip the full response.
export type PoseJSON = ModalExtractResponse

// Kept for callers (route.ts) that still need to materialize a
// KeypointsJson for `user_sessions.keypoints_json`. Pulls only the
// canonical fields off a stored ModalExtractResponse.
export function poseJsonToKeypointsJson(p: ModalExtractResponse): KeypointsJson {
  return {
    fps_sampled: p.fps_sampled,
    frame_count: p.frame_count,
    frames: p.frames,
    schema_version: p.schema_version,
  }
}

/**
 * Look up a cached pose-extraction result by content hash.
 *
 * @param sha256 lowercase hex digest of the source video bytes
 * @param expectedModelVersion when provided, rows with a different
 *   `model_version` are treated as misses (so a model upgrade
 *   transparently invalidates the cache for that hash). Omit to
 *   accept any stored model version.
 * @returns the stored ModalExtractResponse, or null if the row is
 *   missing or the model_version doesn't match.
 */
export async function getCachedPose(
  sha256: string,
  expectedModelVersion?: string,
): Promise<PoseJSON | null> {
  if (!sha256 || typeof sha256 !== 'string') {
    return null
  }

  const { data, error } = await supabaseAdmin
    .from('pose_cache')
    .select('pose_json, model_version')
    .eq('sha256', sha256)
    .maybeSingle()

  // PostgREST's PGRST116 ("no rows") shouldn't happen with maybeSingle()
  // — it returns null instead — but other errors (network, RLS, schema
  // out of sync) surface here. Log + treat as miss; the worst case is
  // we re-extract, which is the pre-cache behavior anyway. Silently
  // swallowing would mask a real outage so we still log loudly.
  if (error) {
    console.error('[poseCache] getCachedPose error:', error.message, { sha256 })
    return null
  }

  if (!data) return null

  // Model-version gate. A row whose model_version doesn't match the
  // current server default is stale — treat as miss. The next
  // extraction will overwrite it via the real upsert path in
  // setCachedPose (ON CONFLICT (sha256) DO UPDATE), so legacy rows
  // get replaced transparently rather than turning into a permanent
  // cache bypass for that hash.
  if (expectedModelVersion && data.model_version !== expectedModelVersion) {
    return null
  }

  return data.pose_json as PoseJSON
}

/**
 * Persist a pose-extraction result to the cache. Race-safe: concurrent
 * inserts of the same sha256 from two extractions converge — last
 * writer wins via Postgres ON CONFLICT (sha256) DO UPDATE. Both
 * payloads are byte-equivalent for the same (bytes, model) pair, so
 * "last wins" produces the same row as "first wins" in the steady
 * state.
 *
 * Critical: this is a real upsert, not ON CONFLICT DO NOTHING. After
 * a model upgrade, the read-side gate in getCachedPose treats rows
 * with the old model_version as a miss. Re-extraction lands here, and
 * the new (sha256, model_version, pose_json) tuple OVERWRITES the
 * stale row instead of being silently dropped. Without that, every
 * legacy hash would be a permanent cache bypass after each upgrade.
 *
 * Throws on real database errors. Callers in the extract route should
 * catch and continue: failing to populate the cache shouldn't fail the
 * extraction itself (the user already has their result; the next
 * upload of the same file just won't get the cache hit).
 */
export async function setCachedPose(
  sha256: string,
  poseJson: PoseJSON,
  modelVersion: string,
): Promise<void> {
  if (!sha256 || typeof sha256 !== 'string') {
    throw new Error('setCachedPose: sha256 is required')
  }
  if (!modelVersion || typeof modelVersion !== 'string') {
    throw new Error('setCachedPose: modelVersion is required')
  }

  // upsert(..., { onConflict: 'sha256' }) without `ignoreDuplicates`
  // maps to "INSERT ... ON CONFLICT (sha256) DO UPDATE SET ..." in
  // postgrest-js — i.e. a real upsert that replaces the row on
  // conflict. This is intentional: we want a newer model_version's
  // payload to overwrite an older one, and a newer extraction's
  // payload to overwrite a stale one for the same model_version.
  const { error } = await supabaseAdmin.from('pose_cache').upsert(
    {
      sha256,
      pose_json: poseJson,
      model_version: modelVersion,
    },
    { onConflict: 'sha256' },
  )

  if (error) {
    // Surface real failures (RLS, schema mismatch, network) — never
    // silently swallow. The caller in /api/extract wraps this in a
    // try/catch and downgrades to a console.error so the user's
    // request still succeeds.
    throw new Error(`setCachedPose failed: ${error.message}`)
  }
}
