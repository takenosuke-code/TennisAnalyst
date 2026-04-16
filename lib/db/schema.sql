-- Tennis Swing Analyzer — Supabase Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Pro players
CREATE TABLE IF NOT EXISTS pros (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  name text NOT NULL,
  nationality text,
  ranking integer,
  bio text,
  profile_image_url text,
  created_at timestamptz DEFAULT now()
);

-- Pro swing recordings (pre-analyzed)
CREATE TABLE IF NOT EXISTS pro_swings (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  pro_id uuid NOT NULL REFERENCES pros(id) ON DELETE CASCADE,
  shot_type text NOT NULL CHECK (shot_type IN ('forehand','backhand','serve','volley','slice')),
  video_url text NOT NULL,
  thumbnail_url text,
  keypoints_json jsonb NOT NULL DEFAULT '{}',
  fps float4 DEFAULT 30,
  frame_count integer,
  duration_ms integer,
  -- Phase labels: frame indices for each swing phase
  phase_labels jsonb DEFAULT '{}',
  -- e.g. {"preparation": 0, "loading": 12, "contact": 28, "follow_through": 35, "finish": 45}
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pro_swings_pro_id_idx ON pro_swings(pro_id);
CREATE INDEX IF NOT EXISTS pro_swings_shot_type_idx ON pro_swings(shot_type);

-- User sessions (ephemeral, no auth)
CREATE TABLE IF NOT EXISTS user_sessions (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  blob_url text NOT NULL,
  shot_type text CHECK (shot_type IN ('forehand','backhand','serve','volley','slice','unknown')),
  keypoints_json jsonb DEFAULT '{}',
  analysis_result jsonb DEFAULT '{}',
  similarity_scores jsonb DEFAULT '{}',
  -- e.g. {"overall": 0.82, "by_phase": {"loading": 0.75, "contact": 0.88}}
  matched_pro_swing_id uuid REFERENCES pro_swings(id),
  status text NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded','extracting','analyzing','complete','error')),
  error_message text,
  created_at timestamptz DEFAULT now(),
  expires_at timestamptz DEFAULT now() + interval '24 hours'
);

CREATE INDEX IF NOT EXISTS user_sessions_expires_idx ON user_sessions(expires_at);

-- Scheduled cleanup of expired sessions via pg_cron (runs daily at 3am UTC).
-- Enable pg_cron in Supabase: Dashboard → Database → Extensions → pg_cron
-- Run this separately after enabling the extension:
--
-- SELECT cron.schedule(
--   'purge-expired-sessions',
--   '0 3 * * *',
--   $$DELETE FROM user_sessions WHERE expires_at < now()$$
-- );
