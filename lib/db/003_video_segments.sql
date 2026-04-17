-- Video segments: AI-detected shot intervals within a longer practice video.

CREATE TABLE IF NOT EXISTS video_segments (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id uuid NOT NULL REFERENCES user_sessions(id) ON DELETE CASCADE,
  segment_index integer NOT NULL,
  shot_type text NOT NULL CHECK (shot_type IN ('forehand','backhand','serve','volley','slice','unknown','idle')),
  start_frame integer NOT NULL,
  end_frame integer NOT NULL,
  start_ms integer NOT NULL,
  end_ms integer NOT NULL,
  confidence float4 DEFAULT 0,
  label text,
  keypoints_json jsonb DEFAULT '{}',
  analysis_result jsonb DEFAULT '{}',
  matched_pro_swing_id uuid REFERENCES pro_swings(id),
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS video_segments_session_idx ON video_segments(session_id);
CREATE INDEX IF NOT EXISTS video_segments_shot_type_idx ON video_segments(shot_type);

-- RLS: same pattern as user_sessions (capability-based via UUID)
ALTER TABLE video_segments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "video_segments_select_public"
  ON video_segments FOR SELECT TO anon, authenticated USING (true);

CREATE POLICY "video_segments_insert_public"
  ON video_segments FOR INSERT TO anon, authenticated WITH CHECK (true);

CREATE POLICY "video_segments_update_public"
  ON video_segments FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);

CREATE POLICY "video_segments_all_service"
  ON video_segments FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Add multi-shot tracking to user_sessions
ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS is_multi_shot boolean DEFAULT false;
ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS segment_count integer DEFAULT 0;
