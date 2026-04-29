-- Phase 4.1 — Async coach review.
--
-- A user with a finished /analyze session can ask their coach to verify
-- one of the LLM cues. The coach gets a public link (no auth required —
-- access is gated by the unguessable review_id token) where they tap
-- "looks right" or "I'd add this:" and write a quick reply. The user's
-- analyze view then shows the cue as pro-reviewed.
--
-- Design notes:
--   * No coach account. The coach is just whoever has the link. This
--     matches the consumer-rebuttal spec ("Marco is 58, takes Venmo,
--     doesn't have a website. He is NOT logging into a SaaS dashboard").
--   * One row per review request. A user can request the same coach
--     review the same session multiple times if they recut the cue.
--   * `responded_at` distinguishes "sent, not yet seen" from
--     "responded with verdict". Until set, the user's UI says
--     "Pending coach review."
--   * No email/SMS layer this round — the user copies the link and
--     sends it to their coach themselves. A follow-up PR can wire up
--     Resend / Twilio without schema changes.

CREATE TABLE IF NOT EXISTS coach_reviews (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL,                      -- the player who asked for review
  session_id uuid REFERENCES user_sessions(id) ON DELETE CASCADE,
  blob_url text NOT NULL,                     -- the clip the coach watches
  cue_title text NOT NULL,                    -- the LLM cue under review
  cue_body text NOT NULL,                     -- full cue body for context
  coach_label text,                           -- "Marco" or "Coach Sarah" — optional self-applied label
  -- Coach response. Both null until the coach actually responds via the
  -- public page. `verdict` is the structured tap; `note` is the optional
  -- typed addendum.
  verdict text CHECK (verdict IN ('looks_right', 'add_this')),
  note text,
  responded_at timestamptz,
  -- Timestamps + soft-delete by user.
  created_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz
);

CREATE INDEX IF NOT EXISTS coach_reviews_user_idx ON coach_reviews(user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS coach_reviews_session_idx ON coach_reviews(session_id) WHERE deleted_at IS NULL;

ALTER TABLE coach_reviews ENABLE ROW LEVEL SECURITY;

-- Permissive RLS, ownership enforced in the API. Same pattern as
-- user_baselines / analysis_events. The PUBLIC coach page reads its own
-- row by id — that's auth-by-unguessable-uuid, identical to the Modal
-- URL allowlist pattern.
CREATE POLICY "coach_reviews_select_public"
  ON coach_reviews FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "coach_reviews_insert_public"
  ON coach_reviews FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "coach_reviews_update_public"
  ON coach_reviews FOR UPDATE
  TO anon, authenticated
  USING (true)
  WITH CHECK (true);
