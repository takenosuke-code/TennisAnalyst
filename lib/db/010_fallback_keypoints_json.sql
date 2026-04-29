-- Phase 6 — server re-extraction on the live → review handoff.
--
-- Adds user_sessions.fallback_keypoints_json so the live save flow can
-- stash the in-browser keypoints (the lower-fidelity ones produced
-- during recording) at the moment the row is created with
-- status='extracting'. Railway then writes its higher-fidelity result
-- into keypoints_json. If Railway fails or times out, the finalize
-- endpoint copies fallback_keypoints_json back into keypoints_json so
-- the row still ends in status='complete' with usable keypoints.
--
-- Why a separate column instead of "just write live keypoints into
-- keypoints_json and let Railway overwrite":
--   1. The /api/sessions/[id]?include=keypoints poll already inspects
--      keypoints_json. Pre-seeding it with live keypoints would race
--      against the poller — clients could pick up the live ones, mark
--      themselves done, and miss the server upgrade.
--   2. Phase 7 wants to retry server extraction after a failed run
--      without re-running the live pipeline. Keeping the fallback as a
--      durable sibling makes that one UPDATE.
--
-- Storage note: each session now carries roughly 2× the JSON it used
-- to. Postgres jsonb compresses well; for typical sessions this is
-- single-digit MB. The finalize endpoint MAY null out
-- fallback_keypoints_json after a successful server extraction
-- (Phase 7 candidate, not in this migration).
--
-- The status enum already allows 'extracting' (see schema.sql), so no
-- CHECK-constraint change is needed here.

ALTER TABLE user_sessions
  ADD COLUMN IF NOT EXISTS fallback_keypoints_json jsonb;

-- Down migration (run manually; we don't ship a migration runner that
-- executes these automatically):
--
--   ALTER TABLE user_sessions DROP COLUMN IF EXISTS fallback_keypoints_json;
