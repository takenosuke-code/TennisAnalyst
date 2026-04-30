-- RLS Policies for Swingframe
-- Run this migration in the Supabase SQL editor after the base schema.

-- =============================================================================
-- Enable RLS on all tables
-- =============================================================================

ALTER TABLE pros ENABLE ROW LEVEL SECURITY;
ALTER TABLE pro_swings ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- pros — public read, admin-only write
-- =============================================================================

CREATE POLICY "pros_select_public"
  ON pros FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "pros_all_service"
  ON pros FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- =============================================================================
-- pro_swings — public read, admin-only write
-- =============================================================================

CREATE POLICY "pro_swings_select_public"
  ON pro_swings FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "pro_swings_all_service"
  ON pro_swings FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- =============================================================================
-- user_sessions — anon can insert, read, and update own sessions.
-- No auth system exists, so sessions are capability-based (UUID as token).
-- Deletion is service-role only (pg_cron cleanup).
-- =============================================================================

CREATE POLICY "user_sessions_select_public"
  ON user_sessions FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "user_sessions_insert_public"
  ON user_sessions FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "user_sessions_update_public"
  ON user_sessions FOR UPDATE
  TO anon, authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "user_sessions_all_service"
  ON user_sessions FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);
