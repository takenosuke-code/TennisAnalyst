-- Drop the pro-comparison feature entirely.
--
-- Pro-vs-user was the original product framing. We've pivoted to
-- self-comparison (baselines) and solo AI coaching; the pros/pro_swings
-- tables are no longer read or written by the app.
--
-- Run order: AFTER 005. Safe to re-run (IF EXISTS on every drop).

-- 1. Drop the foreign-key columns that referenced pro_swings so we don't
--    trip integrity errors when dropping the parent tables.
ALTER TABLE IF EXISTS user_sessions DROP COLUMN IF EXISTS matched_pro_swing_id;
ALTER TABLE IF EXISTS user_sessions DROP COLUMN IF EXISTS similarity_scores;
ALTER TABLE IF EXISTS video_segments DROP COLUMN IF EXISTS matched_pro_swing_id;

-- 2. Drop the tables themselves. CASCADE so any lingering policies, indexes,
--    or views tied to them fall away cleanly.
DROP TABLE IF EXISTS pro_swings CASCADE;
DROP TABLE IF EXISTS pros CASCADE;
