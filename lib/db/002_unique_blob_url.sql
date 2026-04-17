-- Prevent duplicate user_sessions for the same uploaded video.
-- Required for the upsert pattern in POST /api/sessions.

ALTER TABLE user_sessions
  ADD CONSTRAINT user_sessions_blob_url_unique UNIQUE (blob_url);
