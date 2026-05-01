-- Phase E: SHA-256 content-hash cache for pose extraction.
--
-- Re-uploading the exact same video bytes (re-analysis from history,
-- re-encode-with-same-input, two users uploading the same coach demo
-- clip) used to re-run the entire RTMPose pipeline — Modal cold start,
-- video download, GPU inference, the full ~10-30s wallclock cost.
--
-- This table caches `sha256(file_bytes) -> keypoints_json` so re-uploads
-- of identical content skip extraction entirely. The client computes the
-- hash in a Web Worker during/after the file picker step (so the main
-- thread never stalls on a 200 MB digest) and submits it alongside the
-- usual Vercel-Blob URL.
--
-- Why hash bytes, not blob URL: Vercel Blob URLs are session-scoped and
-- get a random suffix on every upload. The bytes are the only thing
-- that's truly content-addressable.
--
-- Cache-key semantics:
--   PK = sha256 only. The `model_version` column is checked on READ,
--   not part of the key. When we swap RTMPose-M for RTMPose-L, the read
--   path treats a row whose model_version doesn't match the current
--   server-side default as a miss, and the next extraction OVERWRITES
--   it via ON CONFLICT (sha256) DO UPDATE (see lib/poseCache.ts:
--   setCachedPose). This keeps the table at one row per file regardless
--   of how many times the model rolls forward — old rows are replaced
--   in place by the first extraction after the upgrade.
--
--   Earlier revisions used ON CONFLICT DO NOTHING here, which
--   silently dropped the rewrite — the read-side miss would re-extract
--   on Modal, but the write-side no-op left the stale row in place, so
--   every subsequent upload of that hash also re-extracted. After a
--   model bump that meant the cache effectively turned itself off for
--   every legacy hash. Real upsert is required for cache freshness;
--   the race-safety argument below still holds because both writers
--   hold byte-equivalent payloads for the same (bytes, model) pair.
--
--   Alternative would be a composite PK (sha256, model_version), but
--   that grows unboundedly.
--
-- Race-safety: setCachedPose() uses INSERT ... ON CONFLICT (sha256) DO
-- UPDATE. Two users uploading the same file simultaneously each
-- trigger their own extraction; whichever PostgREST request lands
-- second overwrites the first. Acceptable: extraction is idempotent
-- (same bytes -> same keypoints, RTMPose is deterministic), and "do
-- the work twice and let the second writer win" is cheaper than
-- coordinating a lock.

CREATE TABLE IF NOT EXISTS pose_cache (
  -- Lowercase hex SHA-256 of the raw video bytes. 64 chars exactly.
  sha256 text PRIMARY KEY,
  -- Canonical KeypointsJson shape (lib/supabase.ts) — fps_sampled,
  -- frame_count, frames[], schema_version. jsonb compresses well; a
  -- typical 2-min clip's keypoints land in the low single-digit MB.
  pose_json jsonb NOT NULL,
  -- Identifies the extractor + weights that produced pose_json. Read-
  -- side compares this to the current server default and treats a
  -- mismatch as a miss. Examples: 'rtmpose-m-modal-cuda',
  -- 'rtmpose-l-modal-cuda', 'mediapipe-heavy'. Free-form text rather
  -- than an enum so swapping models is a one-line code change.
  model_version text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- PK already creates an index on sha256; an extra btree index would be
-- redundant. We never query by model_version alone, so no index there.

-- Telemetry helper: lets ops sanity-check cache hit ratios without
-- needing a separate metric. Pure observability — drop if it ever
-- becomes a write hotspot (it won't; INSERT-only, no triggers).
COMMENT ON TABLE pose_cache IS
  'Content-hash cache: sha256(video_bytes) -> RTMPose keypoints. Hit = skip extraction. See lib/poseCache.ts.';

-- Down migration (run manually):
--   DROP TABLE IF EXISTS pose_cache;
