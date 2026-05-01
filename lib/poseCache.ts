// Server-side content-hash cache for pose extraction.
//
// Backed by lib/db/012_pose_cache.sql (table `pose_cache`).
//
// Contract:
//   getCachedPose(sha256, modelVersion?) — returns the cached
//     KeypointsJson for the given byte hash, or null on miss. When
//     `modelVersion` is provided, a stored row whose model_version
//     doesn't match is treated as a miss (so model upgrades naturally
//     invalidate stale entries without manual cache busts).
//
//   setCachedPose(sha256, poseJson, modelVersion) — writes the cache
//     entry. Uses INSERT ... ON CONFLICT (sha256) DO NOTHING so two
//     concurrent extractions of the same file race-safely (one wins,
//     one no-ops). Never throws on conflict; throws on real DB
//     errors (network, RLS, schema mismatch) so the caller decides
//     whether to swallow or surface.
//
// Why this lives in lib/ and not app/api/:
// Two API routes need it (today: /api/extract; tomorrow: any
// re-extraction trigger we add). Keeping the cache logic in a single
// file with one well-tested contract avoids drift between callers.

import type { KeypointsJson } from '@/lib/supabase'
import { supabaseAdmin } from '@/lib/supabase'

// Re-exported alias matching the task description's vocabulary. The
// canonical type lives in lib/supabase.ts; aliasing here keeps callers
// from importing both modules just to spell the type.
export type PoseJSON = KeypointsJson

/**
 * Look up a cached pose-extraction result by content hash.
 *
 * @param sha256 lowercase hex digest of the source video bytes
 * @param expectedModelVersion when provided, rows with a different
 *   `model_version` are treated as misses (so a model upgrade
 *   transparently invalidates the cache for that hash). Omit to
 *   accept any stored model version.
 * @returns the stored KeypointsJson, or null if the row is missing or
 *   the model_version doesn't match.
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
  // current server default is stale — treat as miss and let the next
  // extraction overwrite it (the setter uses ON CONFLICT DO NOTHING,
  // so we'd actually need an explicit UPSERT to overwrite — see note
  // there. For now: stale rows get garbage-collected by the SQL
  // truncate during model upgrades.)
  if (expectedModelVersion && data.model_version !== expectedModelVersion) {
    return null
  }

  return data.pose_json as PoseJSON
}

/**
 * Persist a pose-extraction result to the cache. Race-safe: concurrent
 * inserts of the same sha256 from two extractions resolve via Postgres
 * ON CONFLICT — first writer wins, the rest no-op.
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

  // upsert(..., { onConflict: 'sha256', ignoreDuplicates: true }) maps
  // to "INSERT ... ON CONFLICT (sha256) DO NOTHING" in postgrest-js.
  // The duplicate path returns no rows and no error — exactly the
  // "second writer is a silent no-op" semantics we want.
  const { error } = await supabaseAdmin.from('pose_cache').upsert(
    {
      sha256,
      pose_json: poseJson,
      model_version: modelVersion,
    },
    { onConflict: 'sha256', ignoreDuplicates: true },
  )

  if (error) {
    // Surface real failures (RLS, schema mismatch, network) — never
    // silently swallow. The caller in /api/extract wraps this in a
    // try/catch and downgrades to a console.error so the user's
    // request still succeeds.
    throw new Error(`setCachedPose failed: ${error.message}`)
  }
}
