-- Extends analysis_events with output-shape metrics so we can verify the
-- tier-rule rewrite is producing the intended cue-count distribution.
-- Target invariant (checked via v_tier_output_calibration): median tips
-- should be monotonically NON-DECREASING from advanced -> beginner ->
-- intermediate -> competitive. If beginner shows higher median tips than
-- intermediate, the rules have regressed to the old bug.

ALTER TABLE analysis_events
  ADD COLUMN IF NOT EXISTS response_token_count int,
  ADD COLUMN IF NOT EXISTS response_tip_count int,
  ADD COLUMN IF NOT EXISTS response_char_count int,
  ADD COLUMN IF NOT EXISTS used_baseline_template boolean DEFAULT false;

-- Range check: tip counts never go negative; cap at 10 to catch runaway output.
ALTER TABLE analysis_events
  ADD CONSTRAINT analysis_events_tip_count_range
    CHECK (response_tip_count IS NULL OR (response_tip_count >= 0 AND response_tip_count <= 10));

-- Monotonic invariant check: median tips over the last 7 days, per coached tier.
-- Founder watches this in the Supabase SQL editor. If advanced_median >
-- beginner_median OR beginner_median > intermediate_median OR
-- intermediate_median > competitive_median, the rewrite regressed.
CREATE OR REPLACE VIEW v_tier_output_calibration AS
SELECT
  llm_coached_tier,
  count(*)                                                     AS samples,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY response_tip_count)     AS median_tips,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY response_token_count)   AS median_tokens,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY response_char_count)    AS median_chars,
  avg(CASE WHEN used_baseline_template THEN 1 ELSE 0 END)::float     AS baseline_template_rate
FROM analysis_events
WHERE created_at >= now() - interval '7 days'
  AND llm_coached_tier IS NOT NULL
GROUP BY llm_coached_tier
ORDER BY CASE llm_coached_tier
  WHEN 'advanced'     THEN 0
  WHEN 'beginner'     THEN 1
  WHEN 'intermediate' THEN 2
  WHEN 'competitive'  THEN 3
  ELSE 4 END;
