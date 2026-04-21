// @vitest-environment node
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import path from 'node:path'

// Shape test only — we do not connect to Postgres here. The goal is to
// catch accidental edits that remove the monotonic-invariant view, the
// tip-count range constraint, or any of the four new columns.
const sql = readFileSync(
  path.resolve(__dirname, '../../lib/db/008_analysis_events_output_metrics.sql'),
  'utf8',
)

describe('migration 008_analysis_events_output_metrics.sql', () => {
  it('adds response_token_count int via ADD COLUMN IF NOT EXISTS', () => {
    expect(sql).toMatch(/ADD COLUMN IF NOT EXISTS\s+response_token_count\s+int/)
  })

  it('adds response_tip_count int via ADD COLUMN IF NOT EXISTS', () => {
    expect(sql).toMatch(/ADD COLUMN IF NOT EXISTS\s+response_tip_count\s+int/)
  })

  it('adds response_char_count int via ADD COLUMN IF NOT EXISTS', () => {
    expect(sql).toMatch(/ADD COLUMN IF NOT EXISTS\s+response_char_count\s+int/)
  })

  it('adds used_baseline_template boolean via ADD COLUMN IF NOT EXISTS', () => {
    expect(sql).toMatch(/ADD COLUMN IF NOT EXISTS\s+used_baseline_template\s+boolean/)
  })

  it('creates or replaces the v_tier_output_calibration view', () => {
    expect(sql).toContain('CREATE OR REPLACE VIEW v_tier_output_calibration')
  })

  it('computes the median tip count via percentile_cont over response_tip_count', () => {
    expect(sql).toContain(
      'percentile_cont(0.5) WITHIN GROUP (ORDER BY response_tip_count)',
    )
  })

  it('scopes the calibration view to the last 7 days', () => {
    expect(sql).toContain("interval '7 days'")
  })

  it('documents the monotonic-invariant expectation in a header comment', () => {
    // Either phrasing is acceptable; both live in the header.
    const hasHeader =
      sql.includes('Target invariant') ||
      sql.includes('monotonically NON-DECREASING')
    expect(hasHeader).toBe(true)
  })

  it('orders the view output advanced-first via ORDER BY CASE llm_coached_tier', () => {
    expect(sql).toContain('ORDER BY CASE llm_coached_tier')
  })

  it('adds the analysis_events_tip_count_range CHECK constraint', () => {
    expect(sql).toContain('ADD CONSTRAINT analysis_events_tip_count_range')
  })
})
